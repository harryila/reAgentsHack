"""Additive contextual grounding for exact numeric source claims.

The immutable packet-grounding v2 contract intentionally treats every Unicode dash
as a possible numeric sign.  That is conservative, but it also rejects a reported
confidence interval such as ``18.0 U+2013 1005`` even when the en dash is plainly the range
delimiter.  This module does not modify or relax v2.  It defines an independent,
stricter-context successor contract:

* a model copies an exact support quote, a unique local context, and an exact token;
* trusted code derives character and UTF-8 byte offsets at every nesting level;
* numeric tokens use an ASCII-only closed grammar;
* U+2013 may be accepted only as the exact delimiter between separately grounded
  lower and upper tokens, never as part of either numeric token; and
* all source, candidate, prompt, schema, and pipeline identities are hash-bound.

The three code-owned fixtures are offline feasibility witnesses, not model outputs or
accuracy labels.  Row 16 candidate 1 demonstrates reported odds-ratio/CI grounding
while remaining blocked from a native graph because its candidate surface lacks an
exact study/cohort identity.  Row 17 candidates 2 and 3 additionally demonstrate safe
one-publication native graphs from exact title, registry, endpoint, timepoint, arm,
and event-count evidence.  Every scientific effectiveness, calibration, synthesis,
and claim-release authority remains false.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.effects import EffectFormat, harmonize_effect
from literature_multiverse.evidence_graph import (
    ArmRole,
    OutcomeTimepoint,
    TimepointKind,
    TimeUnit,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    MetaSynPassageCandidateV2,
    validate_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynExtractionRowInputV2,
    MetaSynPacketCandidateInputV2,
    freeze_metasyn_packet_candidate_input_v2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.metasyn_v5_source_surface import (
    MetaSynV5SourceSurfaceV1,
    freeze_metasyn_v5_source_surface,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import (
    NativeArm,
    NativeCohort,
    NativeContrast,
    NativeEffectPayload,
    NativeEvidenceSpan,
    NativeFinding,
    NativePublicationExtraction,
    NativeStudy,
    freeze_native_publication_extraction,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.schemas import assert_closed_object_schema
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    PublicationEvidenceFragment,
)

CONTEXTUAL_GROUNDING_V3_VERSION = "contextual-numeric-grounding-v3"
PROVIDER_CONTEXT_V3_VERSION = "contextual-provider-context-v3"
PROVIDER_BINDING_V3_VERSION = "contextual-provider-binding-v3"
MODEL_OUTCOME_V3_VERSION = "contextual-packet-model-outcome-v3"
GROUNDING_RECEIPT_V3_VERSION = "contextual-grounding-feasibility-receipt-v3"
FEASIBILITY_SUITE_V3_VERSION = "contextual-grounding-offline-feasibility-suite-v3"
PIPELINE_COMPONENT_VERSION = "1"

V2_WORKSPACE = PurePosixPath("data/cache/metasyn/passage-hosted-yield-v2")
V2_EXECUTION_BUNDLE = V2_WORKSPACE / "execution-bundle.json"
EXPECTED_V2_EXECUTION_BUNDLE_SHA256 = (
    "f87eddcbcbafc778f18ff85c92c0f914a763d242311a859d9d979ded229b4972"
)
EXPECTED_V2_EXECUTION_BUNDLE_FILE_SHA256 = (
    "d75f52f2db6e5a826b3cb8b6303cc0b5a0b8dda42cbd3806ac50b0d1fcd2d4da"
)

ROW16_ORDINAL = 16
ROW16_CANDIDATE_INDEX = 1
ROW16_PASSAGE_ID = "p2-1d9e8f0672eea14fa4393c3562b87fdb032147452d954c26ee8a5bd4b940efef"
ROW16_ENDPOINT_QUOTE = (
    "For the primary endpoint, 41.9% of ruxolitinib-treated patients achieved a ≥35% "
    "reduction in spleen volume at week 24 compared with 0.7% of placebo-treated "
    "patients (odds ratio [OR], 134.4; 95% confidence interval [CI], 18.0\u20131005; "
    "P<0.001"
)

ROW17_ORDINAL = 17
ROW17_CANDIDATE_INDEX = 3
ROW17_FALLBACK_CANDIDATE_INDEX = 2
ROW17_FALLBACK_RESULT_PASSAGE_ID = (
    "p2-87b75468511399d328be9b1860ed946ae4dcf6b789a07fbd7122d983870a1df5"
)
ROW17_RESULT_PASSAGE_ID = "p2-e8d17a6c150ced0ab01177f4c4b2329099ff41dd9d5e39bc0a43d186f00b2351"
ROW17_ENDPOINT_PASSAGE_ID = "p2-0e03c6ff5960490a728498da3defc626f4ef67e2cf6efbfacb6e86ba5fd87440"
ROW17_REGISTRY_PASSAGE_ID = "p2-d063b3c76749e9aabfbcabb8471025a02744a4395314fa9f9de07a5db5a0d30f"
ROW17_TITLE_PASSAGE_ID = "p2-77f204edf7b9068968bed930f46ecbdab91d832bba389f2e3bb0f2645d58b769"
ROW17_RESULT_QUOTE = (
    "The primary end point was achieved by 35 of 96 (36% [95% CI, 27%-46%]) and "
    "39 of 97 (40% [95% CI, 30%-50%]) patients in the fedratinib 400-mg and "
    "500-mg groups, vs 1 of 96 (1% [95% CI, 0%-3%]) in the placebo group "
    "(P\u2009<\u2009.001)."
)
ROW17_ENDPOINT_DEFINITION = (
    "The primary end point was spleen response (≥35% reduction in spleen volume "
    "from baseline as determined by magnetic resonance imaging or computed "
    "tomography) at week 24 and confirmed 4 weeks later."
)
ROW17_REGISTRY_QUOTE = "clinicaltrials.gov identifier: NCT01437787."
ROW17_TITLE = (
    "Safety and Efficacy of Fedratinib in Patients With Primary or Secondary "
    "Myelofibrosis: A Randomized Clinical Trial."
)
ROW17_FALLBACK_RESULT_QUOTE = (
    "Symptom response rates at week 24 were 33 of 91 (36% [95% CI, 26%-46%]), "
    "31 of 91 (34% [95% CI, 24%-44%]), and 6 of 85 (7% [95% CI, 2%-13%]) in "
    "the fedratinib 400-mg, 500-mg, and placebo groups, respectively "
    "(P\u2009<\u2009.001)."
)

MAX_QUOTE_CHARACTERS = 4_096
MAX_CONTEXT_CHARACTERS = 1_024
MAX_TOKEN_CHARACTERS = 512
MAX_PASSAGES = 64
UNICODE_RANGE_DELIMITER = "\u2013"

_ASCII_DECIMAL_RE = re.compile(r"^-?(?:(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,12})?|\.[0-9]{1,12})$")
_UNSIGNED_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]{0,9})$")
_NON_ASCII_SIGN_OR_DASH = frozenset("\u2212\u2010\u2011\u2012\u2013\u2014\ufe63\uff0d\u207b\u208b")

NumericFieldPath = Literal[
    "effect.estimate",
    "effect.ci_lower",
    "effect.ci_upper",
    "effect.ci_level",
    "effect.treatment_events",
    "effect.treatment_total",
    "effect.control_events",
    "effect.control_total",
    "finding.timepoint.value",
]
TextFieldPath = Literal[
    "effect.format",
    "study.source_label",
    "study.design",
    "study.registration_id",
    "cohort.registry_id",
    "finding.endpoint_marker",
    "finding.outcome_name",
    "treatment_arm.label",
    "comparator_arm.label",
    "contrast.marker",
    "finding.timepoint.anchor",
]
ClaimFieldPath = NumericFieldPath | TextFieldPath
NormalizationKind = Literal[
    "verbatim_text",
    "decimal_identity",
    "percent_to_proportion",
    "unsigned_integer",
    "timepoint_integer",
]

_NUMERIC_FIELD_PATHS = frozenset(
    {
        "effect.estimate",
        "effect.ci_lower",
        "effect.ci_upper",
        "effect.ci_level",
        "effect.treatment_events",
        "effect.treatment_total",
        "effect.control_events",
        "effect.control_total",
        "finding.timepoint.value",
    }
)
_DIRECT_REQUIRED_PATHS = frozenset(
    {
        "effect.estimate",
        "effect.ci_lower",
        "effect.ci_upper",
        "effect.ci_level",
        "effect.format",
        "finding.endpoint_marker",
        "finding.outcome_name",
        "treatment_arm.label",
        "comparator_arm.label",
        "contrast.marker",
        "finding.timepoint.value",
        "finding.timepoint.anchor",
    }
)
_BINARY_REQUIRED_PATHS = frozenset(
    {
        "effect.treatment_events",
        "effect.treatment_total",
        "effect.control_events",
        "effect.control_total",
        "study.source_label",
        "study.design",
        "study.registration_id",
        "cohort.registry_id",
        "finding.endpoint_marker",
        "finding.outcome_name",
        "treatment_arm.label",
        "comparator_arm.label",
        "contrast.marker",
        "finding.timepoint.value",
        "finding.timepoint.anchor",
    }
)


class ContextualNumericGroundingV3Error(ValueError):
    """A contextual claim, source replay, or projection failed closed."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=MAX_QUOTE_CHARACTERS)]


def _validate_self_hash(model: _FrozenExactModel, field_name: str) -> None:
    observed = getattr(model, field_name)
    expected = hash_canonical(model.model_dump(mode="json", exclude={field_name}))
    if observed != expected:
        raise ValueError(f"contextual_grounding_v3_self_hash_mismatch:{field_name}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode) or not resolved.is_dir():
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_repository_root_invalid")
    return resolved


def _checked_file(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_relative_path_invalid")
    candidate = root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise ContextualNumericGroundingV3Error(
                f"contextual_grounding_v3_file_missing:{relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ContextualNumericGroundingV3Error(
                f"contextual_grounding_v3_symlink_forbidden:{relative.as_posix()}"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_file_invalid:{relative.as_posix()}"
        )
    return resolved


class ContextualProviderPassageV3(_FrozenExactModel):
    passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    text: NonEmptyText
    text_sha256: Sha256
    section_enums: list[str]
    passage_lineage_sha256: Sha256

    @field_validator("section_enums")
    @classmethod
    def validate_sections(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or not value:
            raise ValueError("contextual_grounding_v3_passage_sections_invalid")
        return value

    @model_validator(mode="after")
    def validate_passage(self) -> ContextualProviderPassageV3:
        if self.text_sha256 != _sha256_text(self.text):
            raise ValueError("contextual_grounding_v3_passage_text_hash_mismatch")
        return self


class ContextualProviderContextV3(_FrozenExactModel):
    context_version: Literal["contextual-provider-context-v3"] = PROVIDER_CONTEXT_V3_VERSION
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    row_key: NonEmptyText
    row_input_sha256: Sha256
    projection_v2_sha256: Sha256
    projection_surface_sha256: Sha256
    question_surface_sha256: Sha256
    source_strength_surface_sha256: Sha256
    source_content_scope: Literal["full_text_sections", "title_abstract"]
    release_grade_source_grounding_eligible: bool
    source_strength_blockers: list[str]
    candidate: MetaSynPassageCandidateV2
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    endpoint_passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    passages: Annotated[list[ContextualProviderPassageV3], Field(min_length=1, max_length=64)]
    passage_membership_sha256: Sha256
    allowed_outcome_text: NonEmptyText
    comparison: NonEmptyText
    intervention_or_exposure: NonEmptyText
    contrast_estimand: NonEmptyText
    context_sha256: Sha256

    @field_validator("source_strength_blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("contextual_grounding_v3_source_blockers_not_canonical")
        return value

    @field_validator("passages")
    @classmethod
    def validate_passages(
        cls, value: list[ContextualProviderPassageV3]
    ) -> list[ContextualProviderPassageV3]:
        ids = [item.passage_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("contextual_grounding_v3_passages_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> ContextualProviderContextV3:
        if self.candidate_descriptor_sha256 != self.candidate.descriptor_sha256:
            raise ValueError("contextual_grounding_v3_candidate_descriptor_mismatch")
        passage_ids = {item.passage_id for item in self.passages}
        if self.endpoint_passage_id not in self.candidate.passage_ids:
            raise ValueError("contextual_grounding_v3_endpoint_not_candidate_passage")
        if not set(self.candidate.passage_ids).issubset(passage_ids):
            raise ValueError("contextual_grounding_v3_candidate_passage_missing")
        if self.passage_membership_sha256 != hash_canonical(
            [item.model_dump(mode="json") for item in self.passages]
        ):
            raise ValueError("contextual_grounding_v3_passage_membership_mismatch")
        if self.candidate.outcome_concept_quote not in self.allowed_outcome_text:
            raise ValueError("contextual_grounding_v3_candidate_outcome_protocol_mismatch")
        if self.release_grade_source_grounding_eligible != (
            self.source_content_scope == "full_text_sections" and not self.source_strength_blockers
        ):
            raise ValueError("contextual_grounding_v3_source_strength_alias_mismatch")
        _validate_self_hash(self, "context_sha256")
        return self


_SEMANTIC_TARGET_VALUES: dict[str, dict[str, str]] = {
    "metasyn-row16-candidate1-reported-odds-ratio-ci": {
        "comparator_arm.label": "placebo-treated patients",
        "contrast.marker": "compared with",
        "effect.ci_level": "0.95",
        "effect.ci_lower": "18.0",
        "effect.ci_upper": "1005",
        "effect.estimate": "134.4",
        "effect.format": "odds ratio",
        "finding.endpoint_marker": "primary endpoint",
        "finding.outcome_name": "spleen volume",
        "finding.timepoint.anchor": "week",
        "finding.timepoint.value": "24",
        "treatment_arm.label": "ruxolitinib-treated patients",
    },
    "metasyn-row17-candidate2-binary-symptom-endpoint": {
        "cohort.registry_id": "NCT01437787",
        "comparator_arm.label": "placebo",
        "contrast.marker": "respectively",
        "effect.control_events": "6",
        "effect.control_total": "85",
        "effect.treatment_events": "31",
        "effect.treatment_total": "91",
        "finding.endpoint_marker": "Symptom response rates",
        "finding.outcome_name": "Symptom response",
        "finding.timepoint.anchor": "week",
        "finding.timepoint.value": "24",
        "study.design": "Randomized Clinical Trial",
        "study.registration_id": "NCT01437787",
        "study.source_label": ROW17_TITLE,
        "treatment_arm.label": "500-mg",
    },
    "metasyn-row17-candidate3-binary-primary-endpoint": {
        "cohort.registry_id": "NCT01437787",
        "comparator_arm.label": "placebo group",
        "contrast.marker": "vs",
        "effect.control_events": "1",
        "effect.control_total": "96",
        "effect.treatment_events": "39",
        "effect.treatment_total": "97",
        "finding.endpoint_marker": "primary end point",
        "finding.outcome_name": "spleen response",
        "finding.timepoint.anchor": "week",
        "finding.timepoint.value": "24",
        "study.design": "Randomized Clinical Trial",
        "study.registration_id": "NCT01437787",
        "study.source_label": ROW17_TITLE,
        "treatment_arm.label": "500-mg",
    },
}
_SEMANTIC_TARGET_LOCATION: dict[str, tuple[int, int]] = {
    "metasyn-row16-candidate1-reported-odds-ratio-ci": (
        ROW16_ORDINAL,
        ROW16_CANDIDATE_INDEX,
    ),
    "metasyn-row17-candidate2-binary-symptom-endpoint": (
        ROW17_ORDINAL,
        ROW17_FALLBACK_CANDIDATE_INDEX,
    ),
    "metasyn-row17-candidate3-binary-primary-endpoint": (
        ROW17_ORDINAL,
        ROW17_CANDIDATE_INDEX,
    ),
}


class ContextualSemanticTargetV3(_FrozenExactModel):
    target_version: Literal["contextual-semantic-target-v3"] = "contextual-semantic-target-v3"
    target_id: Literal[
        "metasyn-row16-candidate1-reported-odds-ratio-ci",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
        "metasyn-row17-candidate3-binary-primary-endpoint",
    ]
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1)]
    candidate_descriptor_sha256: Sha256
    expected_normalized_values: dict[str, str]
    source_visible_code_owned_target_not_accuracy_label: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    target_sha256: Sha256

    @model_validator(mode="after")
    def validate_target(self) -> ContextualSemanticTargetV3:
        if (self.row_ordinal, self.candidate_index) != _SEMANTIC_TARGET_LOCATION[self.target_id]:
            raise ValueError("contextual_grounding_v3_semantic_target_location_mismatch")
        if self.expected_normalized_values != _SEMANTIC_TARGET_VALUES[self.target_id]:
            raise ValueError("contextual_grounding_v3_semantic_target_values_mismatch")
        _validate_self_hash(self, "target_sha256")
        return self


def _semantic_target(context: ContextualProviderContextV3) -> ContextualSemanticTargetV3:
    location = (context.row_ordinal, context.candidate.candidate_index)
    matches = [
        target_id
        for target_id, expected_location in _SEMANTIC_TARGET_LOCATION.items()
        if expected_location == location
    ]
    if len(matches) != 1:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_candidate_semantic_target_unsupported"
        )
    target_id = matches[0]
    payload = {
        "target_version": "contextual-semantic-target-v3",
        "target_id": target_id,
        "row_ordinal": context.row_ordinal,
        "candidate_index": context.candidate.candidate_index,
        "candidate_descriptor_sha256": context.candidate_descriptor_sha256,
        "expected_normalized_values": _SEMANTIC_TARGET_VALUES[target_id],
        "source_visible_code_owned_target_not_accuracy_label": True,
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
    }
    return ContextualSemanticTargetV3.model_validate(
        {**payload, "target_sha256": hash_canonical(payload)}
    )


class ContextualClaimV3(_FrozenExactModel):
    field_path: ClaimFieldPath
    passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    support_quote: Annotated[str, Field(min_length=1, max_length=MAX_QUOTE_CHARACTERS)]
    context: Annotated[str, Field(min_length=1, max_length=MAX_CONTEXT_CHARACTERS)]
    token: Annotated[str, Field(min_length=1, max_length=MAX_TOKEN_CHARACTERS)]
    normalization: NormalizationKind

    @model_validator(mode="after")
    def validate_claim_shape(self) -> ContextualClaimV3:
        numeric = self.field_path in _NUMERIC_FIELD_PATHS
        if numeric == (self.normalization == "verbatim_text"):
            raise ValueError("contextual_grounding_v3_claim_normalization_kind_mismatch")
        if self.context.count(self.token) != 1:
            raise ValueError("contextual_grounding_v3_token_not_unique_in_context")
        if self.support_quote.count(self.context) != 1:
            raise ValueError("contextual_grounding_v3_context_not_unique_in_support_quote")
        if numeric and self.context == self.token:
            raise ValueError("contextual_grounding_v3_numeric_context_not_local")
        return self


class ContextualPacketCompletedV3(_FrozenExactModel):
    outcome_version: Literal["contextual-packet-model-outcome-v3"] = MODEL_OUTCOME_V3_VERSION
    packet_status: Literal["completed"] = "completed"
    candidate_binding_sha256: Sha256
    canonical_outcome_id: str
    effect_kind: Literal["direct_confidence_interval", "binary_group_statistics"]
    endpoint_passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    endpoint_quote: NonEmptyText
    effect_format_token: Literal["odds ratio"] | None
    effect_computation: Literal[
        "reported_direct_confidence_interval",
        "binary_group_statistics_to_odds_ratio_via_existing_harmonizer",
    ]
    source_scope_acknowledgement: Literal[
        "full_text_sections",
        "title_abstract_not_release_grade",
    ]
    claims: Annotated[list[ContextualClaimV3], Field(min_length=1, max_length=32)]

    @field_validator("claims")
    @classmethod
    def validate_claim_order(cls, value: list[ContextualClaimV3]) -> list[ContextualClaimV3]:
        keys = [(item.field_path, item.passage_id) for item in value]
        if keys != sorted(set(keys)):
            raise ValueError("contextual_grounding_v3_claims_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> ContextualPacketCompletedV3:
        paths = {item.field_path for item in self.claims}
        if self.effect_kind == "direct_confidence_interval":
            if (
                self.effect_format_token != "odds ratio"
                or self.effect_computation != "reported_direct_confidence_interval"
                or paths != _DIRECT_REQUIRED_PATHS
            ):
                raise ValueError("contextual_grounding_v3_direct_contract_mismatch")
        elif (
            self.effect_format_token is not None
            or self.effect_computation
            != "binary_group_statistics_to_odds_ratio_via_existing_harmonizer"
            or paths != _BINARY_REQUIRED_PATHS
        ):
            raise ValueError("contextual_grounding_v3_binary_contract_mismatch")
        marker = next(item for item in self.claims if item.field_path == "finding.endpoint_marker")
        if (
            marker.passage_id != self.endpoint_passage_id
            or marker.support_quote != self.endpoint_quote
            or self.endpoint_quote.count(marker.token) != 1
        ):
            raise ValueError("contextual_grounding_v3_endpoint_marker_not_exact")
        return self


class ContextualPacketAbstentionV3(_FrozenExactModel):
    outcome_version: Literal["contextual-packet-model-outcome-v3"] = MODEL_OUTCOME_V3_VERSION
    packet_status: Literal["unable_to_complete"] = "unable_to_complete"
    candidate_binding_sha256: Sha256
    reason: Literal[
        "exact_context_not_unique",
        "numeric_token_not_exact",
        "identity_not_exact",
        "endpoint_not_self_contained",
        "unsupported_effect_format",
        "other_grounding_failure",
    ]


ContextualPacketOutcomeV3 = Annotated[
    ContextualPacketCompletedV3 | ContextualPacketAbstentionV3,
    Field(discriminator="packet_status"),
]
_OUTCOME_ADAPTER = TypeAdapter(ContextualPacketOutcomeV3)


def _provider_schema(context: ContextualProviderContextV3) -> dict[str, Any]:
    passage_ids = [item.passage_id for item in context.passages]
    claim_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "field_path": {"enum": sorted(_DIRECT_REQUIRED_PATHS | _BINARY_REQUIRED_PATHS)},
            "passage_id": {"enum": passage_ids, "type": "string"},
            "support_quote": {"type": "string", "minLength": 1, "maxLength": 4096},
            "context": {"type": "string", "minLength": 1, "maxLength": 1024},
            "token": {"type": "string", "minLength": 1, "maxLength": 512},
            "normalization": {
                "enum": [
                    "verbatim_text",
                    "decimal_identity",
                    "percent_to_proportion",
                    "unsigned_integer",
                    "timepoint_integer",
                ]
            },
        },
        "required": [
            "field_path",
            "passage_id",
            "support_quote",
            "context",
            "token",
            "normalization",
        ],
    }
    if context.candidate.effect_kind == "direct_confidence_interval":
        effect_format_schema: dict[str, Any] = {"const": "odds ratio", "type": "string"}
        computation = "reported_direct_confidence_interval"
    else:
        effect_format_schema = {"const": None, "type": "null"}
        computation = "binary_group_statistics_to_odds_ratio_via_existing_harmonizer"
    scope = (
        "full_text_sections"
        if context.source_content_scope == "full_text_sections"
        else "title_abstract_not_release_grade"
    )
    completed = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "outcome_version": {"const": MODEL_OUTCOME_V3_VERSION, "type": "string"},
            "packet_status": {"const": "completed", "type": "string"},
            "candidate_binding_sha256": {
                "const": context.candidate_binding_sha256,
                "type": "string",
            },
            "canonical_outcome_id": {
                "const": context.candidate.canonical_outcome_id,
                "type": "string",
            },
            "effect_kind": {"const": context.candidate.effect_kind, "type": "string"},
            "endpoint_passage_id": {
                "const": context.endpoint_passage_id,
                "type": "string",
            },
            "endpoint_quote": {"type": "string", "minLength": 1, "maxLength": 4096},
            "effect_format_token": effect_format_schema,
            "effect_computation": {"const": computation, "type": "string"},
            "source_scope_acknowledgement": {"const": scope, "type": "string"},
            "claims": {
                "type": "array",
                "items": claim_schema,
                "minItems": len(
                    _DIRECT_REQUIRED_PATHS
                    if context.candidate.effect_kind == "direct_confidence_interval"
                    else _BINARY_REQUIRED_PATHS
                ),
                "maxItems": 32,
            },
        },
        "required": [
            "outcome_version",
            "packet_status",
            "candidate_binding_sha256",
            "canonical_outcome_id",
            "effect_kind",
            "endpoint_passage_id",
            "endpoint_quote",
            "effect_format_token",
            "effect_computation",
            "source_scope_acknowledgement",
            "claims",
        ],
    }
    unable = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "outcome_version": {"const": MODEL_OUTCOME_V3_VERSION, "type": "string"},
            "packet_status": {"const": "unable_to_complete", "type": "string"},
            "candidate_binding_sha256": {
                "const": context.candidate_binding_sha256,
                "type": "string",
            },
            "reason": {
                "enum": [
                    "exact_context_not_unique",
                    "numeric_token_not_exact",
                    "identity_not_exact",
                    "endpoint_not_self_contained",
                    "unsupported_effect_format",
                    "other_grounding_failure",
                ]
            },
        },
        "required": [
            "outcome_version",
            "packet_status",
            "candidate_binding_sha256",
            "reason",
        ],
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:literature-multiverse:contextual-grounding-v3:{context.context_sha256}",
        "oneOf": [completed, unable],
    }
    try:
        validator_for(schema).check_schema(schema)
        assert_closed_object_schema(schema)
    except (SchemaError, ValueError) as exc:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_provider_schema_not_closed"
        ) from exc
    return schema


_PROMPT_INSTRUCTIONS = """Contextual grounding contract v3.
Return exactly one JSON object accepted by the supplied schema. Use only exact text
from PASSAGES. For each claim copy an exact support_quote, a local context that occurs
exactly once in that support_quote, and a token that occurs exactly once in that
context. Never calculate offsets. Never normalize Unicode punctuation. In a reported
confidence interval, U+2013 is a range delimiter and must not be copied into either
numeric token. Ground the primary endpoint, outcome, arms/contrast, timepoint, and all
required identity fields. If any required exact support is absent, abstain. This task
does not ask for an accuracy label or scientific conclusion."""


def _render_provider_prompt(context: ContextualProviderContextV3, schema_sha256: str) -> str:
    surface = {
        "candidate": context.candidate.model_dump(mode="json"),
        "candidate_binding_sha256": context.candidate_binding_sha256,
        "allowed_outcome_text": context.allowed_outcome_text,
        "comparison": context.comparison,
        "intervention_or_exposure": context.intervention_or_exposure,
        "contrast_estimand": context.contrast_estimand,
        "source_content_scope": context.source_content_scope,
        "source_strength_blockers": context.source_strength_blockers,
        "passages": [item.model_dump(mode="json") for item in context.passages],
    }
    return (
        _PROMPT_INSTRUCTIONS.strip()
        + "\nCONTEXT_SHA256="
        + context.context_sha256
        + "\nSCHEMA_SHA256="
        + schema_sha256
        + "\nCONTEXT_JSON="
        + _canonical_json(surface)
    )


class ContextualProviderBindingV3(_FrozenExactModel):
    binding_version: Literal["contextual-provider-binding-v3"] = PROVIDER_BINDING_V3_VERSION
    context: ContextualProviderContextV3
    context_sha256: Sha256
    provider_schema: dict[str, Any]
    provider_schema_sha256: Sha256
    rendered_prompt: Annotated[str, Field(min_length=1, max_length=100_000)]
    rendered_prompt_sha256: Sha256
    deterministic_prompt_binding: Literal[True] = True
    provider_calls_made: Literal[False] = False
    reference_fields_opened: Literal[False] = False
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> ContextualProviderBindingV3:
        if self.context_sha256 != self.context.context_sha256:
            raise ValueError("contextual_grounding_v3_provider_context_alias_mismatch")
        expected_schema = _provider_schema(self.context)
        if self.provider_schema != expected_schema or self.provider_schema_sha256 != hash_canonical(
            expected_schema
        ):
            raise ValueError("contextual_grounding_v3_provider_schema_replay_mismatch")
        expected_prompt = _render_provider_prompt(self.context, self.provider_schema_sha256)
        if self.rendered_prompt != expected_prompt or self.rendered_prompt_sha256 != _sha256_text(
            expected_prompt
        ):
            raise ValueError("contextual_grounding_v3_provider_prompt_replay_mismatch")
        _validate_self_hash(self, "binding_sha256")
        return self


def freeze_contextual_provider_binding_v3(
    *, context: ContextualProviderContextV3
) -> ContextualProviderBindingV3:
    schema = _provider_schema(context)
    schema_sha256 = hash_canonical(schema)
    prompt = _render_provider_prompt(context, schema_sha256)
    payload = {
        "binding_version": PROVIDER_BINDING_V3_VERSION,
        "context": context,
        "context_sha256": context.context_sha256,
        "provider_schema": schema,
        "provider_schema_sha256": schema_sha256,
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": _sha256_text(prompt),
        "deterministic_prompt_binding": True,
        "provider_calls_made": False,
        "reference_fields_opened": False,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
    }
    return ContextualProviderBindingV3.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


class ContextualSourcePassageV3(_FrozenExactModel):
    passage_version: Literal["contextual-source-passage-v3"] = "contextual-source-passage-v3"
    passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    passage_text: NonEmptyText
    passage_text_sha256: Sha256
    passage_lineage_sha256: Sha256
    source_text_sha256: Sha256
    source_locator: NonEmptyText
    section: NonEmptyText
    line_id: NonEmptyText
    source_char_start: Annotated[int, Field(ge=0)]
    source_char_end_exclusive: Annotated[int, Field(gt=0)]
    source_utf8_byte_start: Annotated[int, Field(ge=0)]
    source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    exact_source_occurrence_count: Literal[1] = 1
    single_exact_origin: Literal[True] = True
    passage_sha256: Sha256

    @model_validator(mode="after")
    def validate_passage(self) -> ContextualSourcePassageV3:
        if self.passage_text_sha256 != _sha256_text(self.passage_text):
            raise ValueError("contextual_grounding_v3_source_passage_hash_mismatch")
        if self.source_char_end_exclusive - self.source_char_start != len(self.passage_text):
            raise ValueError("contextual_grounding_v3_source_passage_char_bounds_mismatch")
        if self.source_utf8_byte_end_exclusive - self.source_utf8_byte_start != len(
            self.passage_text.encode("utf-8")
        ):
            raise ValueError("contextual_grounding_v3_source_passage_byte_bounds_mismatch")
        _validate_self_hash(self, "passage_sha256")
        return self


class ContextualGroundedClaimV3(_FrozenExactModel):
    grounding_version: Literal["contextual-grounded-claim-v3"] = "contextual-grounded-claim-v3"
    claim: ContextualClaimV3
    passage_sha256: Sha256
    support_quote_occurrence_count_in_passage: Literal[1] = 1
    context_occurrence_count_in_support_quote: Literal[1] = 1
    token_occurrence_count_in_context: Literal[1] = 1
    support_quote_char_start_in_passage: Annotated[int, Field(ge=0)]
    support_quote_char_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    context_char_start_in_passage: Annotated[int, Field(ge=0)]
    context_char_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    token_char_start_in_passage: Annotated[int, Field(ge=0)]
    token_char_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    support_quote_source_char_start: Annotated[int, Field(ge=0)]
    support_quote_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    context_source_char_start: Annotated[int, Field(ge=0)]
    context_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    token_source_char_start: Annotated[int, Field(ge=0)]
    token_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    support_quote_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    support_quote_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    context_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    context_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    token_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    token_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    normalized_value: NonEmptyText
    model_authored_offsets_permitted: Literal[False] = False
    fuzzy_matching_permitted: Literal[False] = False
    whitespace_normalization_permitted: Literal[False] = False
    unicode_punctuation_normalization_permitted: Literal[False] = False
    grounding_sha256: Sha256

    @model_validator(mode="after")
    def validate_grounding(self) -> ContextualGroundedClaimV3:
        if (
            self.support_quote_char_end_exclusive_in_passage
            - self.support_quote_char_start_in_passage
            != len(self.claim.support_quote)
            or self.context_char_end_exclusive_in_passage - self.context_char_start_in_passage
            != len(self.claim.context)
            or self.token_char_end_exclusive_in_passage - self.token_char_start_in_passage
            != len(self.claim.token)
        ):
            raise ValueError("contextual_grounding_v3_grounded_char_lengths_mismatch")
        if (
            self.support_quote_source_utf8_byte_end_exclusive
            - self.support_quote_source_utf8_byte_start
            != len(self.claim.support_quote.encode("utf-8"))
            or self.context_source_utf8_byte_end_exclusive - self.context_source_utf8_byte_start
            != len(self.claim.context.encode("utf-8"))
            or self.token_source_utf8_byte_end_exclusive - self.token_source_utf8_byte_start
            != len(self.claim.token.encode("utf-8"))
        ):
            raise ValueError("contextual_grounding_v3_grounded_byte_lengths_mismatch")
        _validate_self_hash(self, "grounding_sha256")
        return self


def _normalized_claim_value(claim: ContextualClaimV3) -> str:
    token = claim.token
    if claim.normalization == "verbatim_text":
        return token
    if any(character in _NON_ASCII_SIGN_OR_DASH for character in token):
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_non_ascii_numeric_sign:{claim.field_path}"
        )
    if claim.normalization == "percent_to_proportion":
        if not token.endswith("%") or not _ASCII_DECIMAL_RE.fullmatch(token[:-1]):
            raise ContextualNumericGroundingV3Error(
                f"contextual_grounding_v3_percent_token_invalid:{claim.field_path}"
            )
        try:
            return format(Decimal(token[:-1]) / Decimal(100), "f")
        except InvalidOperation as exc:  # pragma: no cover - grammar is stronger
            raise ContextualNumericGroundingV3Error(
                f"contextual_grounding_v3_percent_token_invalid:{claim.field_path}"
            ) from exc
    if claim.normalization in {"unsigned_integer", "timepoint_integer"}:
        if not _UNSIGNED_INTEGER_RE.fullmatch(token):
            raise ContextualNumericGroundingV3Error(
                f"contextual_grounding_v3_integer_token_invalid:{claim.field_path}"
            )
        return token
    if not _ASCII_DECIMAL_RE.fullmatch(token):
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_decimal_token_invalid:{claim.field_path}"
        )
    try:
        value = Decimal(token)
    except InvalidOperation as exc:  # pragma: no cover - grammar is stronger
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_decimal_token_invalid:{claim.field_path}"
        ) from exc
    return format(value, "f")


def _utf8_prefix_length(value: str, char_offset: int) -> int:
    return len(value[:char_offset].encode("utf-8"))


def ground_contextual_claim_v3(
    *, claim: ContextualClaimV3, passage: ContextualSourcePassageV3
) -> ContextualGroundedClaimV3:
    """Ground one exact token using nested unique contexts and trusted offsets."""

    if claim.passage_id != passage.passage_id:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_claim_passage_identity_mismatch"
        )
    if passage.passage_text.count(claim.support_quote) != 1:
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_support_quote_not_unique:{claim.field_path}"
        )
    if claim.support_quote.count(claim.context) != 1:
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_context_not_unique:{claim.field_path}"
        )
    if claim.context.count(claim.token) != 1:
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_token_not_unique:{claim.field_path}"
        )
    support_start = passage.passage_text.index(claim.support_quote)
    support_end = support_start + len(claim.support_quote)
    context_start = support_start + claim.support_quote.index(claim.context)
    context_end = context_start + len(claim.context)
    token_start = context_start + claim.context.index(claim.token)
    token_end = token_start + len(claim.token)
    support_byte_start = passage.source_utf8_byte_start + _utf8_prefix_length(
        passage.passage_text, support_start
    )
    context_byte_start = passage.source_utf8_byte_start + _utf8_prefix_length(
        passage.passage_text, context_start
    )
    token_byte_start = passage.source_utf8_byte_start + _utf8_prefix_length(
        passage.passage_text, token_start
    )
    payload = {
        "grounding_version": "contextual-grounded-claim-v3",
        "claim": claim,
        "passage_sha256": passage.passage_sha256,
        "support_quote_occurrence_count_in_passage": 1,
        "context_occurrence_count_in_support_quote": 1,
        "token_occurrence_count_in_context": 1,
        "support_quote_char_start_in_passage": support_start,
        "support_quote_char_end_exclusive_in_passage": support_end,
        "context_char_start_in_passage": context_start,
        "context_char_end_exclusive_in_passage": context_end,
        "token_char_start_in_passage": token_start,
        "token_char_end_exclusive_in_passage": token_end,
        "support_quote_source_char_start": passage.source_char_start + support_start,
        "support_quote_source_char_end_exclusive": passage.source_char_start + support_end,
        "context_source_char_start": passage.source_char_start + context_start,
        "context_source_char_end_exclusive": passage.source_char_start + context_end,
        "token_source_char_start": passage.source_char_start + token_start,
        "token_source_char_end_exclusive": passage.source_char_start + token_end,
        "support_quote_source_utf8_byte_start": support_byte_start,
        "support_quote_source_utf8_byte_end_exclusive": support_byte_start
        + len(claim.support_quote.encode("utf-8")),
        "context_source_utf8_byte_start": context_byte_start,
        "context_source_utf8_byte_end_exclusive": context_byte_start
        + len(claim.context.encode("utf-8")),
        "token_source_utf8_byte_start": token_byte_start,
        "token_source_utf8_byte_end_exclusive": token_byte_start + len(claim.token.encode("utf-8")),
        "normalized_value": _normalized_claim_value(claim),
        "model_authored_offsets_permitted": False,
        "fuzzy_matching_permitted": False,
        "whitespace_normalization_permitted": False,
        "unicode_punctuation_normalization_permitted": False,
    }
    return ContextualGroundedClaimV3.model_validate(
        {**payload, "grounding_sha256": hash_canonical(payload)}
    )


class ContextualUnicodeRangeDelimiterV3(_FrozenExactModel):
    range_version: Literal["contextual-unicode-range-delimiter-v3"] = (
        "contextual-unicode-range-delimiter-v3"
    )
    lower_field_path: Literal["effect.ci_lower"] = "effect.ci_lower"
    upper_field_path: Literal["effect.ci_upper"] = "effect.ci_upper"
    delimiter: Literal["\u2013"] = UNICODE_RANGE_DELIMITER
    exact_range_text: NonEmptyText
    delimiter_char_start_in_passage: Annotated[int, Field(ge=0)]
    delimiter_char_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    delimiter_source_char_start: Annotated[int, Field(ge=0)]
    delimiter_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    delimiter_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    delimiter_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    lower_and_upper_separately_grounded: Literal[True] = True
    delimiter_in_lower_numeric_token: Literal[False] = False
    delimiter_in_upper_numeric_token: Literal[False] = False
    delimiter_interpreted_as_numeric_sign: Literal[False] = False
    punctuation_normalized: Literal[False] = False
    range_sha256: Sha256

    @model_validator(mode="after")
    def validate_range(self) -> ContextualUnicodeRangeDelimiterV3:
        if self.exact_range_text.count(self.delimiter) != 1:
            raise ValueError("contextual_grounding_v3_range_delimiter_count_invalid")
        if self.delimiter_char_end_exclusive_in_passage != (
            self.delimiter_char_start_in_passage + 1
        ):
            raise ValueError("contextual_grounding_v3_range_delimiter_char_bounds_invalid")
        if self.delimiter_source_utf8_byte_end_exclusive != (
            self.delimiter_source_utf8_byte_start + len(self.delimiter.encode("utf-8"))
        ):
            raise ValueError("contextual_grounding_v3_range_delimiter_byte_bounds_invalid")
        _validate_self_hash(self, "range_sha256")
        return self


class ContextualGroundedEffectV3(_FrozenExactModel):
    effect_version: Literal["contextual-grounded-effect-v3"] = "contextual-grounded-effect-v3"
    effect_kind: Literal["direct_confidence_interval", "binary_group_statistics"]
    effect_format: Literal["odds_ratio"] = "odds_ratio"
    effect_format_provenance: Literal[
        "exact_reported_token",
        "code_owned_binary_analysis_policy",
    ]
    estimate: str | None = None
    ci_lower: str | None = None
    ci_upper: str | None = None
    ci_level: str | None = None
    treatment_events: int | None = None
    treatment_total: int | None = None
    control_events: int | None = None
    control_total: int | None = None
    outcome_name: NonEmptyText
    treatment_arm_label: NonEmptyText
    comparator_arm_label: NonEmptyText
    contrast_marker: NonEmptyText
    timepoint_value: Annotated[int, Field(ge=0)]
    timepoint_anchor: Literal["week"] = "week"
    study_source_label: str | None = None
    study_design: str | None = None
    study_registration_id: str | None = None
    cohort_registry_id: str | None = None
    unicode_range: ContextualUnicodeRangeDelimiterV3 | None = None
    native_projection_identity_complete: bool
    effect_sha256: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> ContextualGroundedEffectV3:
        direct_fields = (self.estimate, self.ci_lower, self.ci_upper, self.ci_level)
        count_fields = (
            self.treatment_events,
            self.treatment_total,
            self.control_events,
            self.control_total,
        )
        identity = (
            self.study_source_label,
            self.study_design,
            self.study_registration_id,
            self.cohort_registry_id,
        )
        if self.effect_kind == "direct_confidence_interval":
            if (
                any(item is None for item in direct_fields)
                or any(item is not None for item in count_fields)
                or self.effect_format_provenance != "exact_reported_token"
                or self.unicode_range is None
                or self.native_projection_identity_complete
                or any(item is not None for item in identity)
            ):
                raise ValueError("contextual_grounding_v3_direct_effect_shape_invalid")
            assert self.ci_lower is not None and self.estimate is not None
            assert self.ci_upper is not None and self.ci_level is not None
            if not (
                Decimal("0")
                < Decimal(self.ci_lower)
                < Decimal(self.estimate)
                < Decimal(self.ci_upper)
                and Decimal("0") < Decimal(self.ci_level) < Decimal("1")
            ):
                raise ValueError("contextual_grounding_v3_direct_interval_invalid")
        else:
            if (
                any(item is not None for item in direct_fields)
                or any(item is None for item in count_fields)
                or self.effect_format_provenance != "code_owned_binary_analysis_policy"
                or self.unicode_range is not None
                or not self.native_projection_identity_complete
                or any(item is None for item in identity)
            ):
                raise ValueError("contextual_grounding_v3_binary_effect_shape_invalid")
            assert self.treatment_events is not None and self.treatment_total is not None
            assert self.control_events is not None and self.control_total is not None
            if not (
                0 <= self.treatment_events <= self.treatment_total
                and 0 <= self.control_events <= self.control_total
            ):
                raise ValueError("contextual_grounding_v3_binary_counts_invalid")
        _validate_self_hash(self, "effect_sha256")
        return self


class ContextualNativeProjectionV3(_FrozenExactModel):
    projection_version: Literal["contextual-native-projection-v3"] = (
        "contextual-native-projection-v3"
    )
    status: Literal["blocked_missing_exact_identity", "typed_graph_mechanics_completed"]
    outcome_origin: Literal[
        "code_owned_offline_source_visible_fixture",
        "runtime_outcome_supplied_by_caller",
    ]
    runtime_pipeline_sha256: Sha256 | None = None
    provider_execution_binding_sha256: Sha256 | None = None
    contextual_grounding_core_sha256: Sha256 | None = None
    runtime_grounding_binding_sha256: Sha256 | None = None
    blockers: list[str]
    fragment: PublicationEvidenceFragment | None = None
    fragment_sha256: Sha256 | None = None
    harmonization_result: dict[str, Any] | None = None
    harmonization_result_sha256: Sha256 | None = None
    quantitative_mechanics_result: dict[str, Any] | None = None
    quantitative_mechanics_result_sha256: Sha256 | None = None
    title_abstract_only: bool
    title_abstract_only_not_release_grade: bool
    graph_construction_mechanics_authority: bool
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    projection_sha256: Sha256

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("contextual_grounding_v3_projection_blockers_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> ContextualNativeProjectionV3:
        values = (
            self.fragment,
            self.fragment_sha256,
            self.harmonization_result,
            self.harmonization_result_sha256,
            self.quantitative_mechanics_result,
            self.quantitative_mechanics_result_sha256,
        )
        if self.status == "blocked_missing_exact_identity":
            if (
                any(item is not None for item in values)
                or self.graph_construction_mechanics_authority
            ):
                raise ValueError("contextual_grounding_v3_blocked_projection_has_artifact")
            if self.outcome_origin != "code_owned_offline_source_visible_fixture":
                raise ValueError("contextual_grounding_v3_blocked_projection_origin_mismatch")
            if any(
                item is not None
                for item in (
                    self.runtime_pipeline_sha256,
                    self.provider_execution_binding_sha256,
                    self.contextual_grounding_core_sha256,
                    self.runtime_grounding_binding_sha256,
                )
            ):
                raise ValueError("contextual_grounding_v3_blocked_projection_runtime_hash")
            if self.blockers != ["exact_study_or_cohort_identifier_absent_from_candidate_surface"]:
                raise ValueError("contextual_grounding_v3_blocked_projection_reason_mismatch")
        else:
            if (
                any(item is None for item in values)
                or not self.graph_construction_mechanics_authority
            ):
                raise ValueError("contextual_grounding_v3_completed_projection_missing_artifact")
            assert self.fragment is not None
            assert self.fragment_sha256 is not None
            assert self.harmonization_result is not None
            assert self.harmonization_result_sha256 is not None
            assert self.quantitative_mechanics_result is not None
            assert self.quantitative_mechanics_result_sha256 is not None
            if self.fragment_sha256 != self.fragment.fragment_sha256:
                raise ValueError("contextual_grounding_v3_fragment_hash_alias_mismatch")
            estimate = self.fragment.graph.outcome_estimates[0]  # type: ignore[union-attr]
            expected_harmonized = harmonize_effect(estimate.effect).model_dump(mode="json")
            if (
                self.harmonization_result != expected_harmonized
                or self.harmonization_result_sha256 != hash_canonical(expected_harmonized)
            ):
                raise ValueError("contextual_grounding_v3_harmonization_replay_mismatch")
            expected_quantitative = synthesize_evidence_graph(
                self.fragment.graph,  # type: ignore[arg-type]
                outcome_name=estimate.outcome_name,
                require_explicit_timepoint=True,
                confidence_level=0.95,
                assumed_within_cohort_correlation=1.0,
                prespecified_moderators=(),
            )
            if (
                self.quantitative_mechanics_result != expected_quantitative
                or self.quantitative_mechanics_result_sha256
                != hash_canonical(expected_quantitative)
            ):
                raise ValueError("contextual_grounding_v3_quantitative_replay_mismatch")
            if self.outcome_origin == "code_owned_offline_source_visible_fixture":
                origin_blocker = "offline_source_visible_fixture_not_provider_generated"
                origin_warning = "offline_source_visible_fixture_not_provider_generated"
                if any(
                    item is not None
                    for item in (
                        self.runtime_pipeline_sha256,
                        self.provider_execution_binding_sha256,
                        self.contextual_grounding_core_sha256,
                        self.runtime_grounding_binding_sha256,
                    )
                ):
                    raise ValueError("contextual_grounding_v3_offline_projection_runtime_hash")
            else:
                origin_blocker = "provider_execution_not_attested_by_contextual_grounding_v3"
                origin_warning = "runtime_outcome_supplied_by_caller_not_accuracy_evidence"
                runtime_hashes = (
                    self.runtime_pipeline_sha256,
                    self.provider_execution_binding_sha256,
                    self.contextual_grounding_core_sha256,
                    self.runtime_grounding_binding_sha256,
                )
                if any(item is None for item in runtime_hashes):
                    raise ValueError("contextual_grounding_v3_runtime_projection_binding_missing")
                expected_runtime_binding = hash_canonical(
                    {
                        "binding_version": "contextual-runtime-grounding-binding-v3",
                        "runtime_pipeline_sha256": self.runtime_pipeline_sha256,
                        "provider_execution_binding_sha256": (
                            self.provider_execution_binding_sha256
                        ),
                        "contextual_grounding_core_sha256": (self.contextual_grounding_core_sha256),
                    }
                )
                if (
                    self.runtime_grounding_binding_sha256 != expected_runtime_binding
                    or self.fragment.pipeline_fingerprint_sha256 != self.runtime_pipeline_sha256
                    or self.fragment.extraction_context_sha256 != expected_runtime_binding
                    or self.fragment.grounding_receipt_sha256 != expected_runtime_binding
                ):
                    raise ValueError("contextual_grounding_v3_runtime_projection_binding_mismatch")
            expected_blockers = sorted(
                [
                    "calibration_not_performed",
                    "extraction_accuracy_not_evaluated",
                    origin_blocker,
                    "single_publication_mechanics_only",
                    "title_or_abstract_only_not_release_grade",
                ]
            )
            if self.blockers != expected_blockers:
                raise ValueError("contextual_grounding_v3_completed_projection_blockers_mismatch")
            expected_warnings = sorted(
                [
                    origin_warning,
                    "optional_scientific_fields_not_extracted",
                    "title_or_abstract_only_not_release_grade",
                ]
            )
            if self.fragment.extractor_warnings != expected_warnings:
                raise ValueError("contextual_grounding_v3_projection_warnings_mismatch")
        if self.title_abstract_only_not_release_grade != self.title_abstract_only:
            raise ValueError("contextual_grounding_v3_title_abstract_limit_alias_mismatch")
        _validate_self_hash(self, "projection_sha256")
        return self


def _grounded_claim_map(
    groundings: Sequence[ContextualGroundedClaimV3],
) -> dict[str, ContextualGroundedClaimV3]:
    output = {item.claim.field_path: item for item in groundings}
    if len(output) != len(groundings):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_duplicate_grounded_field_path"
        )
    return output


def _freeze_unicode_range(
    *,
    lower: ContextualGroundedClaimV3,
    upper: ContextualGroundedClaimV3,
    passage: ContextualSourcePassageV3,
) -> ContextualUnicodeRangeDelimiterV3:
    exact_range = lower.claim.token + UNICODE_RANGE_DELIMITER + upper.claim.token
    if (
        lower.claim.passage_id != upper.claim.passage_id
        or lower.claim.passage_id != passage.passage_id
        or passage.passage_text.count(exact_range) != 1
        or UNICODE_RANGE_DELIMITER in lower.claim.token
        or UNICODE_RANGE_DELIMITER in upper.claim.token
    ):
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_unicode_range_not_exact")
    range_start = passage.passage_text.index(exact_range)
    delimiter_start = range_start + len(lower.claim.token)
    if delimiter_start != lower.token_char_end_exclusive_in_passage:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_unicode_range_lower_adjacency_mismatch"
        )
    if delimiter_start + 1 != upper.token_char_start_in_passage:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_unicode_range_upper_adjacency_mismatch"
        )
    byte_start = passage.source_utf8_byte_start + _utf8_prefix_length(
        passage.passage_text, delimiter_start
    )
    payload = {
        "range_version": "contextual-unicode-range-delimiter-v3",
        "lower_field_path": "effect.ci_lower",
        "upper_field_path": "effect.ci_upper",
        "delimiter": UNICODE_RANGE_DELIMITER,
        "exact_range_text": exact_range,
        "delimiter_char_start_in_passage": delimiter_start,
        "delimiter_char_end_exclusive_in_passage": delimiter_start + 1,
        "delimiter_source_char_start": passage.source_char_start + delimiter_start,
        "delimiter_source_char_end_exclusive": passage.source_char_start + delimiter_start + 1,
        "delimiter_source_utf8_byte_start": byte_start,
        "delimiter_source_utf8_byte_end_exclusive": byte_start
        + len(UNICODE_RANGE_DELIMITER.encode("utf-8")),
        "lower_and_upper_separately_grounded": True,
        "delimiter_in_lower_numeric_token": False,
        "delimiter_in_upper_numeric_token": False,
        "delimiter_interpreted_as_numeric_sign": False,
        "punctuation_normalized": False,
    }
    return ContextualUnicodeRangeDelimiterV3.model_validate(
        {**payload, "range_sha256": hash_canonical(payload)}
    )


def _require_exact_binary_event_total_pair(
    *,
    events: ContextualGroundedClaimV3,
    total: ContextualGroundedClaimV3,
    passage: ContextualSourcePassageV3,
    pair_label: Literal["treatment", "control"],
) -> None:
    exact_pair = f"{events.claim.token} of {total.claim.token}"
    if (
        events.claim.passage_id != total.claim.passage_id
        or events.claim.passage_id != passage.passage_id
        or passage.passage_text[
            events.token_char_start_in_passage : total.token_char_end_exclusive_in_passage
        ]
        != exact_pair
    ):
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_binary_pair_not_exact:{pair_label}"
        )


def _freeze_grounded_effect(
    *,
    outcome: ContextualPacketCompletedV3,
    groundings: Sequence[ContextualGroundedClaimV3],
    passages: Mapping[str, ContextualSourcePassageV3],
) -> ContextualGroundedEffectV3:
    claims = _grounded_claim_map(groundings)
    timepoint_value = claims["finding.timepoint.value"]
    timepoint_anchor = claims["finding.timepoint.anchor"]
    if timepoint_value.normalized_value != "24" or timepoint_anchor.claim.token != "week":
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_timepoint_semantics_mismatch"
        )
    common = {
        "effect_version": "contextual-grounded-effect-v3",
        "effect_kind": outcome.effect_kind,
        "effect_format": "odds_ratio",
        "outcome_name": claims["finding.outcome_name"].claim.token,
        "treatment_arm_label": claims["treatment_arm.label"].claim.token,
        "comparator_arm_label": claims["comparator_arm.label"].claim.token,
        "contrast_marker": claims["contrast.marker"].claim.token,
        "timepoint_value": int(timepoint_value.normalized_value),
        "timepoint_anchor": "week",
    }
    if outcome.effect_kind == "direct_confidence_interval":
        format_claim = claims["effect.format"]
        if outcome.effect_format_token != "odds ratio" or format_claim.claim.token != "odds ratio":
            raise ContextualNumericGroundingV3Error(
                "contextual_grounding_v3_effect_format_token_mismatch"
            )
        lower = claims["effect.ci_lower"]
        upper = claims["effect.ci_upper"]
        range_receipt = _freeze_unicode_range(
            lower=lower,
            upper=upper,
            passage=passages[lower.claim.passage_id],
        )
        payload = {
            **common,
            "effect_format_provenance": "exact_reported_token",
            "estimate": claims["effect.estimate"].normalized_value,
            "ci_lower": lower.normalized_value,
            "ci_upper": upper.normalized_value,
            "ci_level": claims["effect.ci_level"].normalized_value,
            "treatment_events": None,
            "treatment_total": None,
            "control_events": None,
            "control_total": None,
            "study_source_label": None,
            "study_design": None,
            "study_registration_id": None,
            "cohort_registry_id": None,
            "unicode_range": range_receipt,
            "native_projection_identity_complete": False,
        }
    else:
        treatment_events = claims["effect.treatment_events"]
        treatment_total = claims["effect.treatment_total"]
        control_events = claims["effect.control_events"]
        control_total = claims["effect.control_total"]
        _require_exact_binary_event_total_pair(
            events=treatment_events,
            total=treatment_total,
            passage=passages[treatment_events.claim.passage_id],
            pair_label="treatment",
        )
        _require_exact_binary_event_total_pair(
            events=control_events,
            total=control_total,
            passage=passages[control_events.claim.passage_id],
            pair_label="control",
        )
        payload = {
            **common,
            "effect_format_provenance": "code_owned_binary_analysis_policy",
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_level": None,
            "treatment_events": int(treatment_events.normalized_value),
            "treatment_total": int(treatment_total.normalized_value),
            "control_events": int(control_events.normalized_value),
            "control_total": int(control_total.normalized_value),
            "study_source_label": claims["study.source_label"].claim.token,
            "study_design": claims["study.design"].claim.token,
            "study_registration_id": claims["study.registration_id"].claim.token,
            "cohort_registry_id": claims["cohort.registry_id"].claim.token,
            "unicode_range": None,
            "native_projection_identity_complete": True,
        }
    return ContextualGroundedEffectV3.model_validate(
        {**payload, "effect_sha256": hash_canonical(payload)}
    )


class ContextualGroundingFeasibilityReceiptV3(_FrozenExactModel):
    receipt_version: Literal["contextual-grounding-feasibility-receipt-v3"] = (
        GROUNDING_RECEIPT_V3_VERSION
    )
    witness_id: Literal[
        "metasyn-row16-candidate1-reported-odds-ratio-ci",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
        "metasyn-row17-candidate3-binary-primary-endpoint",
    ]
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: Sha256
    v2_execution_bundle_sha256: Sha256
    v2_extraction_inputs_sha256: Sha256
    v2_extraction_inputs_pipeline_sha256: Sha256
    v5_source_surface_sha256: Sha256
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    row_key: NonEmptyText
    row_input_sha256: Sha256
    source_surface_row_sha256: Sha256
    inventory_receipt_sha256: Sha256
    packet_input_sha256: Sha256
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    provider_binding: ContextualProviderBindingV3
    provider_binding_sha256: Sha256
    semantic_target: ContextualSemanticTargetV3
    semantic_target_sha256: Sha256
    model_outcome: ContextualPacketCompletedV3
    model_outcome_sha256: Sha256
    fixture_provenance: Literal[
        "code_owned_offline_source_visible_feasibility_fixture_not_provider_generated"
    ] = "code_owned_offline_source_visible_feasibility_fixture_not_provider_generated"
    passages: Annotated[list[ContextualSourcePassageV3], Field(min_length=1, max_length=8)]
    passage_membership_sha256: Sha256
    groundings: Annotated[list[ContextualGroundedClaimV3], Field(min_length=1, max_length=32)]
    grounding_membership_sha256: Sha256
    grounded_effect: ContextualGroundedEffectV3
    grounded_effect_sha256: Sha256
    grounding_core_sha256: Sha256
    native_projection: ContextualNativeProjectionV3
    native_projection_sha256: Sha256
    source_content_scope: Literal["full_text_sections", "title_abstract"]
    source_strength_blockers: list[str]
    release_grade_source_grounding_eligible: bool
    source_limitations_explicit: Literal[True] = True
    v2_lineage_external_replayed: Literal[True] = True
    source_bytes_external_rehashed: Literal[True] = True
    provider_calls_made: Literal[False] = False
    reference_fields_opened: Literal[False] = False
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @field_validator("passages")
    @classmethod
    def validate_passages(
        cls, value: list[ContextualSourcePassageV3]
    ) -> list[ContextualSourcePassageV3]:
        ids = [item.passage_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("contextual_grounding_v3_receipt_passages_not_canonical")
        return value

    @field_validator("groundings")
    @classmethod
    def validate_grounding_order(
        cls, value: list[ContextualGroundedClaimV3]
    ) -> list[ContextualGroundedClaimV3]:
        paths = [item.claim.field_path for item in value]
        if paths != sorted(set(paths)):
            raise ValueError("contextual_grounding_v3_receipt_groundings_not_canonical")
        return value

    @field_validator("source_strength_blockers")
    @classmethod
    def validate_source_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("contextual_grounding_v3_receipt_source_blockers_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> ContextualGroundingFeasibilityReceiptV3:
        if self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256:
            raise ValueError("contextual_grounding_v3_pipeline_hash_alias_mismatch")
        if self.v2_execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256:
            raise ValueError("contextual_grounding_v3_v2_bundle_anchor_mismatch")
        if (
            self.provider_binding_sha256 != self.provider_binding.binding_sha256
            or self.candidate_binding_sha256
            != self.provider_binding.context.candidate_binding_sha256
            or self.candidate_descriptor_sha256
            != self.provider_binding.context.candidate_descriptor_sha256
            or self.row_ordinal != self.provider_binding.context.row_ordinal
            or self.row_key != self.provider_binding.context.row_key
            or self.row_input_sha256 != self.provider_binding.context.row_input_sha256
        ):
            raise ValueError("contextual_grounding_v3_provider_alias_mismatch")
        expected_target = _semantic_target(self.provider_binding.context)
        if (
            self.semantic_target != expected_target
            or self.semantic_target_sha256 != expected_target.target_sha256
            or self.semantic_target.candidate_descriptor_sha256 != self.candidate_descriptor_sha256
        ):
            raise ValueError("contextual_grounding_v3_semantic_target_alias_mismatch")
        try:
            validate_json_schema(
                self.model_outcome.model_dump(mode="json"),
                self.provider_binding.provider_schema,
            )
        except ValidationError as exc:
            raise ValueError("contextual_grounding_v3_outcome_schema_mismatch") from exc
        if (
            self.model_outcome_sha256 != hash_canonical(self.model_outcome.model_dump(mode="json"))
            or self.model_outcome.candidate_binding_sha256 != self.candidate_binding_sha256
            or self.model_outcome.canonical_outcome_id
            != self.provider_binding.context.candidate.canonical_outcome_id
            or self.model_outcome.effect_kind != self.provider_binding.context.candidate.effect_kind
        ):
            raise ValueError("contextual_grounding_v3_outcome_alias_mismatch")
        passages = {item.passage_id: item for item in self.passages}
        provider_passages = {
            item.passage_id: item for item in self.provider_binding.context.passages
        }
        if self.passage_membership_sha256 != hash_canonical(
            [item.passage_sha256 for item in self.passages]
        ):
            raise ValueError("contextual_grounding_v3_receipt_passage_membership_mismatch")
        for item in self.passages:
            provider = provider_passages.get(item.passage_id)
            if (
                provider is None
                or provider.text != item.passage_text
                or provider.text_sha256 != item.passage_text_sha256
                or provider.passage_lineage_sha256 != item.passage_lineage_sha256
            ):
                raise ValueError("contextual_grounding_v3_source_provider_drift")
        endpoint = passages.get(self.model_outcome.endpoint_passage_id)
        if endpoint is None or endpoint.passage_text.count(self.model_outcome.endpoint_quote) != 1:
            raise ValueError("contextual_grounding_v3_endpoint_quote_not_exact")
        claims = self.model_outcome.claims
        replayed = [
            ground_contextual_claim_v3(claim=claim, passage=passages[claim.passage_id])
            for claim in claims
        ]
        replayed.sort(key=lambda item: item.claim.field_path)
        if self.groundings != replayed:
            raise ValueError("contextual_grounding_v3_grounding_replay_mismatch")
        if self.grounding_membership_sha256 != hash_canonical(
            [item.grounding_sha256 for item in self.groundings]
        ):
            raise ValueError("contextual_grounding_v3_grounding_membership_mismatch")
        expected_effect = _freeze_grounded_effect(
            outcome=self.model_outcome,
            groundings=self.groundings,
            passages=passages,
        )
        if (
            self.grounded_effect != expected_effect
            or self.grounded_effect_sha256 != expected_effect.effect_sha256
        ):
            raise ValueError("contextual_grounding_v3_effect_replay_mismatch")
        core = {
            "provider_binding_sha256": self.provider_binding_sha256,
            "semantic_target_sha256": self.semantic_target_sha256,
            "model_outcome_sha256": self.model_outcome_sha256,
            "passage_membership_sha256": self.passage_membership_sha256,
            "grounding_membership_sha256": self.grounding_membership_sha256,
            "grounded_effect_sha256": self.grounded_effect_sha256,
        }
        if self.grounding_core_sha256 != hash_canonical(core):
            raise ValueError("contextual_grounding_v3_core_hash_mismatch")
        if self.native_projection_sha256 != self.native_projection.projection_sha256:
            raise ValueError("contextual_grounding_v3_native_projection_alias_mismatch")
        if (
            self.source_content_scope != self.provider_binding.context.source_content_scope
            or self.source_strength_blockers
            != self.provider_binding.context.source_strength_blockers
            or self.release_grade_source_grounding_eligible
            != self.provider_binding.context.release_grade_source_grounding_eligible
        ):
            raise ValueError("contextual_grounding_v3_source_strength_alias_mismatch")
        if self.witness_id.startswith("metasyn-row16"):
            if (
                self.row_ordinal != ROW16_ORDINAL
                or self.grounded_effect.effect_kind != "direct_confidence_interval"
                or self.native_projection.status != "blocked_missing_exact_identity"
            ):
                raise ValueError("contextual_grounding_v3_row16_witness_mismatch")
        elif (
            self.row_ordinal != ROW17_ORDINAL
            or self.grounded_effect.effect_kind != "binary_group_statistics"
            or self.native_projection.status != "typed_graph_mechanics_completed"
            or not self.native_projection.title_abstract_only_not_release_grade
        ):
            raise ValueError("contextual_grounding_v3_row17_witness_mismatch")
        _validate_self_hash(self, "receipt_sha256")
        return self


class ContextualGroundingOfflineFeasibilitySuiteV3(_FrozenExactModel):
    suite_version: Literal["contextual-grounding-offline-feasibility-suite-v3"] = (
        FEASIBILITY_SUITE_V3_VERSION
    )
    status: Literal[
        "three_source_visible_offline_witnesses_two_typed_graph_mechanics_no_empirical_authority"
    ] = "three_source_visible_offline_witnesses_two_typed_graph_mechanics_no_empirical_authority"
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: Sha256
    v2_execution_bundle_sha256: Sha256
    v5_source_surface_sha256: Sha256
    receipts: Annotated[
        list[ContextualGroundingFeasibilityReceiptV3], Field(min_length=3, max_length=3)
    ]
    receipt_membership_sha256: Sha256
    offline_witness_count: Literal[3] = 3
    contextual_grounding_completed_count: Literal[3] = 3
    typed_graph_mechanics_completed_count: Literal[2] = 2
    provider_calls_made: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    suite_sha256: Sha256

    @field_validator("receipts")
    @classmethod
    def validate_receipts(
        cls, value: list[ContextualGroundingFeasibilityReceiptV3]
    ) -> list[ContextualGroundingFeasibilityReceiptV3]:
        ids = [item.witness_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("contextual_grounding_v3_suite_receipts_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_suite(self) -> ContextualGroundingOfflineFeasibilitySuiteV3:
        if (
            self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256
            or self.v2_execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256
            or any(item.pipeline_sha256 != self.pipeline_sha256 for item in self.receipts)
            or any(
                item.v5_source_surface_sha256 != self.v5_source_surface_sha256
                for item in self.receipts
            )
            or self.receipt_membership_sha256
            != hash_canonical([item.receipt_sha256 for item in self.receipts])
        ):
            raise ValueError("contextual_grounding_v3_suite_alias_mismatch")
        _validate_self_hash(self, "suite_sha256")
        return self


_PIPELINE_FILES = (
    "src/literature_multiverse/contextual_numeric_grounding_v3.py",
    "src/literature_multiverse/effects.py",
    "src/literature_multiverse/evidence_graph.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/meta_analysis.py",
    "src/literature_multiverse/metasyn_candidate_inventory_v2.py",
    "src/literature_multiverse/metasyn_extraction_inputs_v2.py",
    "src/literature_multiverse/metasyn_passage_hosted_bundle_v2.py",
    "src/literature_multiverse/metasyn_projection_v2.py",
    "src/literature_multiverse/metasyn_v5_source_surface.py",
    "src/literature_multiverse/native_extraction.py",
    "src/literature_multiverse/pipeline_fingerprint.py",
    "src/literature_multiverse/typed_extraction.py",
    "pyproject.toml",
    "uv.lock",
)


def compute_contextual_numeric_grounding_v3_pipeline_fingerprint(
    *, repository_root: Path
) -> PipelineFingerprint:
    root = _canonical_root(repository_root)
    settings = {
        "contract_version": CONTEXTUAL_GROUNDING_V3_VERSION,
        "v2_execution_bundle_sha256": EXPECTED_V2_EXECUTION_BUNDLE_SHA256,
        "v2_execution_bundle_file_sha256": EXPECTED_V2_EXECUTION_BUNDLE_FILE_SHA256,
        "offline_witnesses": [
            {"row_ordinal": ROW16_ORDINAL, "candidate_index": ROW16_CANDIDATE_INDEX},
            {
                "row_ordinal": ROW17_ORDINAL,
                "candidate_index": ROW17_FALLBACK_CANDIDATE_INDEX,
            },
            {"row_ordinal": ROW17_ORDINAL, "candidate_index": ROW17_CANDIDATE_INDEX},
        ],
        "numeric_grammar": "ascii_decimal_or_unsigned_integer_only",
        "unicode_en_dash_policy": (
            "exact_between_separately_grounded_ci_tokens_never_numeric_sign"
        ),
        "model_authored_offsets": False,
        "provider_calls_permitted": False,
        "source_scope_limitations_explicit": True,
        "all_empirical_and_release_authority": False,
    }
    return compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="contextual-numeric-grounding-v3",
                component_version=PIPELINE_COMPONENT_VERSION,
                file_paths=sorted(_PIPELINE_FILES),
                settings=settings,
            )
        ],
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_json_unreadable:{label}"
        ) from exc
    if not isinstance(value, dict):
        raise ContextualNumericGroundingV3Error(f"contextual_grounding_v3_json_not_object:{label}")
    return value


def _load_replayed_v2_bundle(*, root: Path) -> MetaSynPassageHostedExecutionBundleV2:
    path = _checked_file(root, V2_EXECUTION_BUNDLE)
    if sha256_file(path) != EXPECTED_V2_EXECUTION_BUNDLE_FILE_SHA256:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_v2_bundle_file_hash_mismatch"
        )
    return validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=_load_json(path, label="v2_execution_bundle"),
        repository_root=root,
        external_replay=True,
    )


def _inventory_receipt(
    *,
    root: Path,
    row: MetaSynExtractionRowInputV2,
) -> MetaSynCandidateInventoryReceiptV2:
    relative = V2_WORKSPACE / "inventory-receipts" / f"row-{row.row_ordinal:02d}.json"
    raw = _load_json(_checked_file(root, relative), label=f"inventory_row_{row.row_ordinal}")
    return validate_metasyn_candidate_inventory_receipt_v2(
        raw,
        row_context_sha256=row.upstream_row_context_sha256,
        projection_v2_sha256=row.projection_v2_sha256,
        allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
        passage_text_by_id={item.passage_id: item.text for item in row.projection_surface.passages},
    )


def _provider_context(
    *,
    row: MetaSynExtractionRowInputV2,
    packet: MetaSynPacketCandidateInputV2,
    endpoint_passage_id: str,
) -> ContextualProviderContextV3:
    passages = [
        ContextualProviderPassageV3(
            passage_id=item.passage_id,
            text=item.text,
            text_sha256=item.text_sha256,
            section_enums=sorted(item.section_enums),
            passage_lineage_sha256=item.passage_lineage_sha256,
        )
        for item in sorted(row.projection_surface.passages, key=lambda value: value.passage_id)
    ]
    outcome_text = row.question_surface.allowed_outcome_text_by_id.get(
        packet.candidate.canonical_outcome_id
    )
    if outcome_text is None:
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_allowed_outcome_missing")
    payload = {
        "context_version": PROVIDER_CONTEXT_V3_VERSION,
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "row_input_sha256": row.row_input_sha256,
        "projection_v2_sha256": row.projection_v2_sha256,
        "projection_surface_sha256": row.projection_surface_sha256,
        "question_surface_sha256": row.question_surface_sha256,
        "source_strength_surface_sha256": row.source_strength_surface_sha256,
        "source_content_scope": row.source_strength.source_content_scope,
        "release_grade_source_grounding_eligible": (
            row.source_strength.release_grade_source_grounding_eligible
        ),
        "source_strength_blockers": sorted(row.source_strength.source_strength_blockers),
        "candidate": packet.candidate,
        "candidate_descriptor_sha256": packet.candidate_descriptor_sha256,
        "candidate_binding_sha256": packet.candidate_binding_sha256,
        "endpoint_passage_id": endpoint_passage_id,
        "passages": passages,
        "passage_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in passages]
        ),
        "allowed_outcome_text": outcome_text,
        "comparison": row.question_surface.comparison,
        "intervention_or_exposure": row.question_surface.intervention_or_exposure,
        "contrast_estimand": row.question_surface.contrast_estimand,
    }
    return ContextualProviderContextV3.model_validate(
        {**payload, "context_sha256": hash_canonical(payload)}
    )


def _source_passage(
    *,
    row: MetaSynExtractionRowInputV2,
    passage_id: str,
) -> ContextualSourcePassageV3:
    surfaces = {item.passage_id: item for item in row.projection_surface.passages}
    anchored = {item.passage_anchor: item for item in row.projection_v2.passages}
    surface = surfaces.get(passage_id)
    source = anchored.get(passage_id)
    if surface is None or source is None or surface.text != source.text:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_source_passage_surface_mismatch"
        )
    if (
        surface.exact_source_occurrence_count != 1
        or len(source.origins) != 1
        or source.origin_count != 1
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_source_passage_not_single_exact_origin"
        )
    origin = source.origins[0]
    if (
        origin.parent_char_start != 0
        or origin.parent_char_end_exclusive != len(source.text)
        or origin.segment_text_sha256 != surface.text_sha256
        or origin.segment_characters != len(source.text)
        or origin.segment_utf8_bytes != len(source.text.encode("utf-8"))
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_source_passage_origin_bounds_mismatch"
        )
    payload = {
        "passage_version": "contextual-source-passage-v3",
        "passage_id": passage_id,
        "passage_text": source.text,
        "passage_text_sha256": surface.text_sha256,
        "passage_lineage_sha256": source.passage_lineage_sha256,
        "source_text_sha256": source.source_text_sha256,
        "source_locator": row.projection_v2.source_locator,
        "section": origin.exposed_section,
        "line_id": origin.line_id,
        "source_char_start": origin.source_char_start,
        "source_char_end_exclusive": origin.source_char_end_exclusive,
        "source_utf8_byte_start": origin.source_utf8_byte_start,
        "source_utf8_byte_end_exclusive": origin.source_utf8_byte_end_exclusive,
        "exact_source_occurrence_count": 1,
        "single_exact_origin": True,
    }
    return ContextualSourcePassageV3.model_validate(
        {**payload, "passage_sha256": hash_canonical(payload)}
    )


def _claim(
    field_path: ClaimFieldPath,
    passage_id: str,
    support_quote: str,
    context: str,
    token: str,
    normalization: NormalizationKind,
) -> ContextualClaimV3:
    return ContextualClaimV3(
        field_path=field_path,
        passage_id=passage_id,
        support_quote=support_quote,
        context=context,
        token=token,
        normalization=normalization,
    )


def _row16_outcome(binding_sha256: str) -> ContextualPacketCompletedV3:
    claims = [
        _claim(
            "comparator_arm.label",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "0.7% of placebo-treated patients (odds ratio",
            "placebo-treated patients",
            "verbatim_text",
        ),
        _claim(
            "contrast.marker",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "spleen volume at week 24 compared with 0.7%",
            "compared with",
            "verbatim_text",
        ),
        _claim(
            "effect.ci_level",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "; 95% confidence interval [CI],",
            "95%",
            "percent_to_proportion",
        ),
        _claim(
            "effect.ci_lower",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "confidence interval [CI], 18.0\u20131005;",
            "18.0",
            "decimal_identity",
        ),
        _claim(
            "effect.ci_upper",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "confidence interval [CI], 18.0\u20131005;",
            "1005",
            "decimal_identity",
        ),
        _claim(
            "effect.estimate",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "odds ratio [OR], 134.4; 95% confidence",
            "134.4",
            "decimal_identity",
        ),
        _claim(
            "effect.format",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "patients (odds ratio [OR], 134.4;",
            "odds ratio",
            "verbatim_text",
        ),
        _claim(
            "finding.endpoint_marker",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "For the primary endpoint, 41.9%",
            "primary endpoint",
            "verbatim_text",
        ),
        _claim(
            "finding.outcome_name",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "≥35% reduction in spleen volume at week 24",
            "spleen volume",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.anchor",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "spleen volume at week 24 compared with",
            "week",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.value",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "spleen volume at week 24 compared with",
            "24",
            "timepoint_integer",
        ),
        _claim(
            "treatment_arm.label",
            ROW16_PASSAGE_ID,
            ROW16_ENDPOINT_QUOTE,
            "41.9% of ruxolitinib-treated patients achieved",
            "ruxolitinib-treated patients",
            "verbatim_text",
        ),
    ]
    claims.sort(key=lambda item: (item.field_path, item.passage_id))
    return ContextualPacketCompletedV3(
        candidate_binding_sha256=binding_sha256,
        canonical_outcome_id="outcome-01",
        effect_kind="direct_confidence_interval",
        endpoint_passage_id=ROW16_PASSAGE_ID,
        endpoint_quote=ROW16_ENDPOINT_QUOTE,
        effect_format_token="odds ratio",
        effect_computation="reported_direct_confidence_interval",
        source_scope_acknowledgement="full_text_sections",
        claims=claims,
    )


def _row17_outcome(binding_sha256: str) -> ContextualPacketCompletedV3:
    claims = [
        _claim(
            "cohort.registry_id",
            ROW17_REGISTRY_PASSAGE_ID,
            ROW17_REGISTRY_QUOTE,
            ROW17_REGISTRY_QUOTE,
            "NCT01437787",
            "verbatim_text",
        ),
        _claim(
            "comparator_arm.label",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "in the placebo group (P\u2009<\u2009.001)",
            "placebo group",
            "verbatim_text",
        ),
        _claim(
            "contrast.marker",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "500-mg groups, vs 1 of 96",
            "vs",
            "verbatim_text",
        ),
        _claim(
            "effect.control_events",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "vs 1 of 96 (",
            "1",
            "unsigned_integer",
        ),
        _claim(
            "effect.control_total",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "vs 1 of 96 (",
            "96",
            "unsigned_integer",
        ),
        _claim(
            "effect.treatment_events",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "39 of 97 (40% [95% CI, 30%-50%])",
            "39",
            "unsigned_integer",
        ),
        _claim(
            "effect.treatment_total",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "39 of 97 (40% [95% CI, 30%-50%])",
            "97",
            "unsigned_integer",
        ),
        _claim(
            "finding.endpoint_marker",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "The primary end point was achieved by",
            "primary end point",
            "verbatim_text",
        ),
        _claim(
            "finding.outcome_name",
            ROW17_ENDPOINT_PASSAGE_ID,
            ROW17_ENDPOINT_DEFINITION,
            "The primary end point was spleen response (≥35%",
            "spleen response",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.anchor",
            ROW17_ENDPOINT_PASSAGE_ID,
            ROW17_ENDPOINT_DEFINITION,
            "tomography) at week 24 and confirmed",
            "week",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.value",
            ROW17_ENDPOINT_PASSAGE_ID,
            ROW17_ENDPOINT_DEFINITION,
            "tomography) at week 24 and confirmed",
            "24",
            "timepoint_integer",
        ),
        _claim(
            "study.design",
            ROW17_TITLE_PASSAGE_ID,
            ROW17_TITLE,
            ROW17_TITLE,
            "Randomized Clinical Trial",
            "verbatim_text",
        ),
        _claim(
            "study.registration_id",
            ROW17_REGISTRY_PASSAGE_ID,
            ROW17_REGISTRY_QUOTE,
            ROW17_REGISTRY_QUOTE,
            "NCT01437787",
            "verbatim_text",
        ),
        _claim(
            "study.source_label",
            ROW17_TITLE_PASSAGE_ID,
            ROW17_TITLE,
            ROW17_TITLE,
            ROW17_TITLE,
            "verbatim_text",
        ),
        _claim(
            "treatment_arm.label",
            ROW17_RESULT_PASSAGE_ID,
            ROW17_RESULT_QUOTE,
            "fedratinib 400-mg and 500-mg groups",
            "500-mg",
            "verbatim_text",
        ),
    ]
    claims.sort(key=lambda item: (item.field_path, item.passage_id))
    return ContextualPacketCompletedV3(
        candidate_binding_sha256=binding_sha256,
        canonical_outcome_id="outcome-01",
        effect_kind="binary_group_statistics",
        endpoint_passage_id=ROW17_RESULT_PASSAGE_ID,
        endpoint_quote=ROW17_RESULT_QUOTE,
        effect_format_token=None,
        effect_computation=("binary_group_statistics_to_odds_ratio_via_existing_harmonizer"),
        source_scope_acknowledgement="title_abstract_not_release_grade",
        claims=claims,
    )


def _row17_fallback_outcome(binding_sha256: str) -> ContextualPacketCompletedV3:
    claims = [
        _claim(
            "cohort.registry_id",
            ROW17_REGISTRY_PASSAGE_ID,
            ROW17_REGISTRY_QUOTE,
            ROW17_REGISTRY_QUOTE,
            "NCT01437787",
            "verbatim_text",
        ),
        _claim(
            "comparator_arm.label",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "500-mg, and placebo groups, respectively",
            "placebo",
            "verbatim_text",
        ),
        _claim(
            "contrast.marker",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "and placebo groups, respectively (P\u2009<\u2009.001)",
            "respectively",
            "verbatim_text",
        ),
        _claim(
            "effect.control_events",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "and 6 of 85 (",
            "6",
            "unsigned_integer",
        ),
        _claim(
            "effect.control_total",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "and 6 of 85 (",
            "85",
            "unsigned_integer",
        ),
        _claim(
            "effect.treatment_events",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "31 of 91 (34% [95% CI, 24%-44%])",
            "31",
            "unsigned_integer",
        ),
        _claim(
            "effect.treatment_total",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "31 of 91 (34% [95% CI, 24%-44%])",
            "91",
            "unsigned_integer",
        ),
        _claim(
            "finding.endpoint_marker",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "Symptom response rates at week 24 were",
            "Symptom response rates",
            "verbatim_text",
        ),
        _claim(
            "finding.outcome_name",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "Symptom response rates at week 24",
            "Symptom response",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.anchor",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "Symptom response rates at week 24 were",
            "week",
            "verbatim_text",
        ),
        _claim(
            "finding.timepoint.value",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "Symptom response rates at week 24 were",
            "24",
            "timepoint_integer",
        ),
        _claim(
            "study.design",
            ROW17_TITLE_PASSAGE_ID,
            ROW17_TITLE,
            ROW17_TITLE,
            "Randomized Clinical Trial",
            "verbatim_text",
        ),
        _claim(
            "study.registration_id",
            ROW17_REGISTRY_PASSAGE_ID,
            ROW17_REGISTRY_QUOTE,
            ROW17_REGISTRY_QUOTE,
            "NCT01437787",
            "verbatim_text",
        ),
        _claim(
            "study.source_label",
            ROW17_TITLE_PASSAGE_ID,
            ROW17_TITLE,
            ROW17_TITLE,
            ROW17_TITLE,
            "verbatim_text",
        ),
        _claim(
            "treatment_arm.label",
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            ROW17_FALLBACK_RESULT_QUOTE,
            "fedratinib 400-mg, 500-mg, and placebo groups",
            "500-mg",
            "verbatim_text",
        ),
    ]
    claims.sort(key=lambda item: (item.field_path, item.passage_id))
    return ContextualPacketCompletedV3(
        candidate_binding_sha256=binding_sha256,
        canonical_outcome_id="outcome-01",
        effect_kind="binary_group_statistics",
        endpoint_passage_id=ROW17_FALLBACK_RESULT_PASSAGE_ID,
        endpoint_quote=ROW17_FALLBACK_RESULT_QUOTE,
        effect_format_token=None,
        effect_computation=("binary_group_statistics_to_odds_ratio_via_existing_harmonizer"),
        source_scope_acknowledgement="title_abstract_not_release_grade",
        claims=claims,
    )


def ground_contextual_outcome_v3(
    *,
    provider_binding: ContextualProviderBindingV3,
    raw_outcome: ContextualPacketOutcomeV3 | Mapping[str, Any],
    passages: Sequence[ContextualSourcePassageV3],
) -> tuple[
    ContextualPacketCompletedV3,
    list[ContextualGroundedClaimV3],
    ContextualGroundedEffectV3,
    str,
]:
    """Validate one provider-shaped outcome and replay all contextual groundings."""

    raw = (
        raw_outcome.model_dump(mode="json")
        if isinstance(raw_outcome, (ContextualPacketCompletedV3, ContextualPacketAbstentionV3))
        else deepcopy(dict(raw_outcome))
    )
    try:
        validate_json_schema(raw, provider_binding.provider_schema)
        outcome = _OUTCOME_ADAPTER.validate_python(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_model_outcome_invalid"
        ) from exc
    if isinstance(outcome, ContextualPacketAbstentionV3):
        raise ContextualNumericGroundingV3Error(
            f"contextual_grounding_v3_model_abstained:{outcome.reason}"
        )
    if (
        outcome.candidate_binding_sha256 != provider_binding.context.candidate_binding_sha256
        or outcome.canonical_outcome_id != provider_binding.context.candidate.canonical_outcome_id
        or outcome.effect_kind != provider_binding.context.candidate.effect_kind
        or outcome.endpoint_passage_id != provider_binding.context.endpoint_passage_id
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_model_candidate_binding_mismatch"
        )
    passage_map = {item.passage_id: item for item in passages}
    if len(passage_map) != len(passages):
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_source_passages_duplicate")
    provider_passages = {item.passage_id: item for item in provider_binding.context.passages}
    for passage in passages:
        provider = provider_passages.get(passage.passage_id)
        if (
            provider is None
            or provider.text != passage.passage_text
            or provider.text_sha256 != passage.passage_text_sha256
            or provider.passage_lineage_sha256 != passage.passage_lineage_sha256
        ):
            raise ContextualNumericGroundingV3Error("contextual_grounding_v3_source_passage_drift")
    endpoint = passage_map.get(outcome.endpoint_passage_id)
    if endpoint is None or endpoint.passage_text.count(outcome.endpoint_quote) != 1:
        raise ContextualNumericGroundingV3Error("contextual_grounding_v3_endpoint_quote_not_unique")
    if set(item.passage_id for item in outcome.claims) != set(passage_map):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_source_passage_membership_not_exact"
        )
    groundings = [
        ground_contextual_claim_v3(claim=claim, passage=passage_map[claim.passage_id])
        for claim in outcome.claims
    ]
    groundings.sort(key=lambda item: item.claim.field_path)
    effect = _freeze_grounded_effect(
        outcome=outcome,
        groundings=groundings,
        passages=passage_map,
    )
    semantic_target = _semantic_target(provider_binding.context)
    observed_values = {item.claim.field_path: item.normalized_value for item in groundings}
    if observed_values != semantic_target.expected_normalized_values:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_candidate_semantic_substitution"
        )
    core = {
        "provider_binding_sha256": provider_binding.binding_sha256,
        "semantic_target_sha256": semantic_target.target_sha256,
        "model_outcome_sha256": hash_canonical(outcome.model_dump(mode="json")),
        "passage_membership_sha256": hash_canonical(
            [item.passage_sha256 for item in sorted(passages, key=lambda item: item.passage_id)]
        ),
        "grounding_membership_sha256": hash_canonical(
            [item.grounding_sha256 for item in groundings]
        ),
        "grounded_effect_sha256": effect.effect_sha256,
    }
    return outcome, groundings, effect, hash_canonical(core)


def _blocked_native_projection(*, title_abstract_only: bool) -> ContextualNativeProjectionV3:
    payload = {
        "projection_version": "contextual-native-projection-v3",
        "status": "blocked_missing_exact_identity",
        "outcome_origin": "code_owned_offline_source_visible_fixture",
        "runtime_pipeline_sha256": None,
        "provider_execution_binding_sha256": None,
        "contextual_grounding_core_sha256": None,
        "runtime_grounding_binding_sha256": None,
        "blockers": ["exact_study_or_cohort_identifier_absent_from_candidate_surface"],
        "fragment": None,
        "fragment_sha256": None,
        "harmonization_result": None,
        "harmonization_result_sha256": None,
        "quantitative_mechanics_result": None,
        "quantitative_mechanics_result_sha256": None,
        "title_abstract_only": title_abstract_only,
        "title_abstract_only_not_release_grade": title_abstract_only,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return ContextualNativeProjectionV3.model_validate(
        {**payload, "projection_sha256": hash_canonical(payload)}
    )


def _completed_native_projection(
    *,
    effect: ContextualGroundedEffectV3,
    groundings: Sequence[ContextualGroundedClaimV3],
    row: MetaSynExtractionRowInputV2,
    source_surface: MetaSynV5SourceSurfaceV1,
    pipeline_sha256: str,
    grounding_core_sha256: str,
    outcome_origin: Literal[
        "code_owned_offline_source_visible_fixture",
        "runtime_outcome_supplied_by_caller",
    ] = "code_owned_offline_source_visible_fixture",
) -> ContextualNativeProjectionV3:
    if effect.effect_kind != "binary_group_statistics":
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_native_projection_effect_kind_unsupported"
        )
    source_row = source_surface.rows[row.row_ordinal]
    record = source_row.source_row.source_record
    if (
        source_row.row_key != row.row_key
        or source_row.source_surface_row_sha256 != row.upstream_source_surface_row_sha256
        or record.publication.title != effect.study_source_label
        or effect.study_source_label != ROW17_TITLE
        or effect.study_registration_id != "NCT01437787"
        or effect.cohort_registry_id != "NCT01437787"
        or effect.study_design != "Randomized Clinical Trial"
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_native_identity_external_replay_mismatch"
        )
    grounding_map = _grounded_claim_map(groundings)
    endpoint = grounding_map["finding.endpoint_marker"]
    if endpoint.claim.support_quote not in {
        ROW17_RESULT_QUOTE,
        ROW17_FALLBACK_RESULT_QUOTE,
    }:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_native_endpoint_support_mismatch"
        )
    positive_direction = row.question_surface.raw_positive_direction_meaning_by_outcome_id[
        row.question_surface.allowed_outcome_ids[0]
    ]
    assert effect.treatment_events is not None
    assert effect.treatment_total is not None
    assert effect.control_events is not None
    assert effect.control_total is not None
    assert effect.study_source_label is not None
    assert effect.study_registration_id is not None
    assert effect.cohort_registry_id is not None
    finding_key = (
        "primary-spleen-response-week-24"
        if effect.outcome_name == "spleen response"
        else "secondary-symptom-response-week-24"
    )
    origin_warning = (
        "offline_source_visible_fixture_not_provider_generated"
        if outcome_origin == "code_owned_offline_source_visible_fixture"
        else "runtime_outcome_supplied_by_caller_not_accuracy_evidence"
    )
    origin_blocker = (
        "offline_source_visible_fixture_not_provider_generated"
        if outcome_origin == "code_owned_offline_source_visible_fixture"
        else "provider_execution_not_attested_by_contextual_grounding_v3"
    )
    payload = NativePublicationExtraction(
        status=FragmentStatus.ESTIMABLE,
        studies=[
            NativeStudy(
                key="nct01437787",
                source_label=effect.study_source_label,
                design=effect.study_design,
                registration_ids=[effect.study_registration_id],
                cohorts=[
                    NativeCohort(
                        key="nct01437787-trial-cohort",
                        source_labels=[effect.cohort_registry_id],
                        registry_ids=[effect.cohort_registry_id],
                        arms=[
                            NativeArm(
                                key="fedratinib-500mg",
                                label=effect.treatment_arm_label,
                                role=ArmRole.INTERVENTION,
                                sample_size=effect.treatment_total,
                            ),
                            NativeArm(
                                key="placebo",
                                label=effect.comparator_arm_label,
                                role=ArmRole.COMPARATOR,
                                sample_size=effect.control_total,
                            ),
                        ],
                        contrasts=[
                            NativeContrast(
                                key="fedratinib-500mg-vs-placebo",
                                treatment_arm_key="fedratinib-500mg",
                                comparator_arm_key="placebo",
                                label=effect.contrast_marker,
                                estimand=row.question_surface.contrast_estimand,
                                positive_direction_means=positive_direction,
                            )
                        ],
                        findings=[
                            NativeFinding(
                                key=finding_key,
                                contrast_key="fedratinib-500mg-vs-placebo",
                                outcome_name=effect.outcome_name,
                                timepoint=OutcomeTimepoint(
                                    kind=TimepointKind.EXACT,
                                    value=float(effect.timepoint_value),
                                    unit=TimeUnit.WEEK,
                                    anchor=effect.timepoint_anchor,
                                    raw_label=(
                                        f"{effect.timepoint_anchor} {effect.timepoint_value}"
                                    ),
                                ),
                                effect=NativeEffectPayload(
                                    effect_format=EffectFormat.ODDS_RATIO,
                                    treatment_events=effect.treatment_events,
                                    treatment_total=effect.treatment_total,
                                    control_events=effect.control_events,
                                    control_total=effect.control_total,
                                    extraction_method="computed_from_reported_statistics",
                                ),
                                evidence=NativeEvidenceSpan(
                                    source_locator=record.source_document.source_locator,
                                    quote=endpoint.claim.support_quote,
                                    section="Abstract",
                                    char_start=endpoint.support_quote_source_char_start,
                                    char_end=endpoint.support_quote_source_char_end_exclusive,
                                    line_ids=[
                                        next(
                                            item.line_id
                                            for item in (
                                                _source_passage(
                                                    row=row,
                                                    passage_id=endpoint.claim.passage_id,
                                                ),
                                            )
                                        )
                                    ],
                                ),
                            )
                        ],
                    )
                ],
            )
        ],
        warnings=[
            origin_warning,
            "optional_scientific_fields_not_extracted",
            "title_or_abstract_only_not_release_grade",
        ],
    )
    fragment = freeze_native_publication_extraction(
        payload=payload,
        question_id=row.question_surface.question_id,
        publication=record.publication,
        pipeline_fingerprint_sha256=pipeline_sha256,
        extraction_context_sha256=grounding_core_sha256,
        source_document=record.source_document,
        grounding_receipt_sha256=grounding_core_sha256,
    )
    if fragment.graph is None or len(fragment.graph.outcome_estimates) != 1:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_native_fragment_graph_invalid"
        )
    estimate = fragment.graph.outcome_estimates[0]
    harmonized = harmonize_effect(estimate.effect).model_dump(mode="json")
    if harmonized.get("status") != "estimable":
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_native_effect_not_harmonizable"
        )
    quantitative = synthesize_evidence_graph(
        fragment.graph,
        outcome_name=estimate.outcome_name,
        require_explicit_timepoint=True,
        confidence_level=0.95,
        assumed_within_cohort_correlation=1.0,
        prespecified_moderators=(),
    )
    projection_payload = {
        "projection_version": "contextual-native-projection-v3",
        "status": "typed_graph_mechanics_completed",
        "outcome_origin": outcome_origin,
        "runtime_pipeline_sha256": None,
        "provider_execution_binding_sha256": None,
        "contextual_grounding_core_sha256": None,
        "runtime_grounding_binding_sha256": None,
        "blockers": sorted(
            [
                "calibration_not_performed",
                "extraction_accuracy_not_evaluated",
                origin_blocker,
                "single_publication_mechanics_only",
                "title_or_abstract_only_not_release_grade",
            ]
        ),
        "fragment": fragment,
        "fragment_sha256": fragment.fragment_sha256,
        "harmonization_result": harmonized,
        "harmonization_result_sha256": hash_canonical(harmonized),
        "quantitative_mechanics_result": quantitative,
        "quantitative_mechanics_result_sha256": hash_canonical(quantitative),
        "title_abstract_only": True,
        "title_abstract_only_not_release_grade": True,
        "graph_construction_mechanics_authority": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return ContextualNativeProjectionV3.model_validate(
        {
            **projection_payload,
            "projection_sha256": hash_canonical(projection_payload),
        }
    )


def _runtime_native_projection_from_fixture(
    *,
    fixture_receipt: ContextualGroundingFeasibilityReceiptV3,
    effect: ContextualGroundedEffectV3,
    groundings: Sequence[ContextualGroundedClaimV3],
    grounding_core_sha256: str,
    runtime_pipeline_sha256: str,
    provider_execution_binding_sha256: str,
) -> ContextualNativeProjectionV3:
    """Build fresh graph mechanics from a replayed row-17 fixture identity surface."""

    if (
        SHA256_RE.fullmatch(runtime_pipeline_sha256) is None
        or SHA256_RE.fullmatch(provider_execution_binding_sha256) is None
        or SHA256_RE.fullmatch(grounding_core_sha256) is None
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_sha256_invalid"
        )
    runtime_grounding_binding_sha256 = hash_canonical(
        {
            "binding_version": "contextual-runtime-grounding-binding-v3",
            "runtime_pipeline_sha256": runtime_pipeline_sha256,
            "provider_execution_binding_sha256": provider_execution_binding_sha256,
            "contextual_grounding_core_sha256": grounding_core_sha256,
        }
    )
    template_projection = fixture_receipt.native_projection
    template = template_projection.fragment
    if (
        fixture_receipt.row_ordinal != ROW17_ORDINAL
        or template_projection.status != "typed_graph_mechanics_completed"
        or template is None
        or template.graph is None
        or effect.effect_kind != "binary_group_statistics"
        or effect.study_source_label != ROW17_TITLE
        or effect.study_registration_id != "NCT01437787"
        or effect.cohort_registry_id != "NCT01437787"
        or effect.study_design != "Randomized Clinical Trial"
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_fixture_mismatch"
        )
    if len(template.graph.contrasts) != 1:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_template_contrast_mismatch"
        )
    template_contrast = template.graph.contrasts[0]
    grounding_map = _grounded_claim_map(groundings)
    endpoint = grounding_map["finding.endpoint_marker"]
    if endpoint.claim.support_quote not in {
        ROW17_RESULT_QUOTE,
        ROW17_FALLBACK_RESULT_QUOTE,
    }:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_endpoint_support_mismatch"
        )
    passage_map = {item.passage_id: item for item in fixture_receipt.passages}
    endpoint_passage = passage_map.get(endpoint.claim.passage_id)
    if endpoint_passage is None:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_endpoint_passage_missing"
        )
    assert effect.treatment_events is not None
    assert effect.treatment_total is not None
    assert effect.control_events is not None
    assert effect.control_total is not None
    assert effect.study_source_label is not None
    assert effect.study_registration_id is not None
    assert effect.cohort_registry_id is not None
    finding_key = (
        "primary-spleen-response-week-24"
        if effect.outcome_name == "spleen response"
        else "secondary-symptom-response-week-24"
    )
    native_payload = NativePublicationExtraction(
        status=FragmentStatus.ESTIMABLE,
        studies=[
            NativeStudy(
                key="nct01437787",
                source_label=effect.study_source_label,
                design=effect.study_design,
                registration_ids=[effect.study_registration_id],
                cohorts=[
                    NativeCohort(
                        key="nct01437787-trial-cohort",
                        source_labels=[effect.cohort_registry_id],
                        registry_ids=[effect.cohort_registry_id],
                        arms=[
                            NativeArm(
                                key="fedratinib-500mg",
                                label=effect.treatment_arm_label,
                                role=ArmRole.INTERVENTION,
                                sample_size=effect.treatment_total,
                            ),
                            NativeArm(
                                key="placebo",
                                label=effect.comparator_arm_label,
                                role=ArmRole.COMPARATOR,
                                sample_size=effect.control_total,
                            ),
                        ],
                        contrasts=[
                            NativeContrast(
                                key="fedratinib-500mg-vs-placebo",
                                treatment_arm_key="fedratinib-500mg",
                                comparator_arm_key="placebo",
                                label=effect.contrast_marker,
                                estimand=template_contrast.estimand,
                                positive_direction_means=(
                                    template_contrast.positive_direction_means
                                ),
                            )
                        ],
                        findings=[
                            NativeFinding(
                                key=finding_key,
                                contrast_key="fedratinib-500mg-vs-placebo",
                                outcome_name=effect.outcome_name,
                                timepoint=OutcomeTimepoint(
                                    kind=TimepointKind.EXACT,
                                    value=float(effect.timepoint_value),
                                    unit=TimeUnit.WEEK,
                                    anchor=effect.timepoint_anchor,
                                    raw_label=(
                                        f"{effect.timepoint_anchor} {effect.timepoint_value}"
                                    ),
                                ),
                                effect=NativeEffectPayload(
                                    effect_format=EffectFormat.ODDS_RATIO,
                                    treatment_events=effect.treatment_events,
                                    treatment_total=effect.treatment_total,
                                    control_events=effect.control_events,
                                    control_total=effect.control_total,
                                    extraction_method="computed_from_reported_statistics",
                                ),
                                evidence=NativeEvidenceSpan(
                                    source_locator=template.source_document.source_locator,
                                    quote=endpoint.claim.support_quote,
                                    section="Abstract",
                                    char_start=endpoint.support_quote_source_char_start,
                                    char_end=endpoint.support_quote_source_char_end_exclusive,
                                    line_ids=[endpoint_passage.line_id],
                                ),
                            )
                        ],
                    )
                ],
            )
        ],
        warnings=[
            "optional_scientific_fields_not_extracted",
            "runtime_outcome_supplied_by_caller_not_accuracy_evidence",
            "title_or_abstract_only_not_release_grade",
        ],
    )
    fragment = freeze_native_publication_extraction(
        payload=native_payload,
        question_id=template.question_id,
        publication=template.publication,
        pipeline_fingerprint_sha256=runtime_pipeline_sha256,
        extraction_context_sha256=runtime_grounding_binding_sha256,
        source_document=template.source_document,
        grounding_receipt_sha256=runtime_grounding_binding_sha256,
    )
    if fragment.graph is None or len(fragment.graph.outcome_estimates) != 1:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_native_fragment_graph_invalid"
        )
    estimate = fragment.graph.outcome_estimates[0]
    harmonized = harmonize_effect(estimate.effect).model_dump(mode="json")
    if harmonized.get("status") != "estimable":
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_native_effect_not_harmonizable"
        )
    quantitative = synthesize_evidence_graph(
        fragment.graph,
        outcome_name=estimate.outcome_name,
        require_explicit_timepoint=True,
        confidence_level=0.95,
        assumed_within_cohort_correlation=1.0,
        prespecified_moderators=(),
    )
    projection_payload = {
        "projection_version": "contextual-native-projection-v3",
        "status": "typed_graph_mechanics_completed",
        "outcome_origin": "runtime_outcome_supplied_by_caller",
        "runtime_pipeline_sha256": runtime_pipeline_sha256,
        "provider_execution_binding_sha256": provider_execution_binding_sha256,
        "contextual_grounding_core_sha256": grounding_core_sha256,
        "runtime_grounding_binding_sha256": runtime_grounding_binding_sha256,
        "blockers": sorted(
            [
                "calibration_not_performed",
                "extraction_accuracy_not_evaluated",
                "provider_execution_not_attested_by_contextual_grounding_v3",
                "single_publication_mechanics_only",
                "title_or_abstract_only_not_release_grade",
            ]
        ),
        "fragment": fragment,
        "fragment_sha256": fragment.fragment_sha256,
        "harmonization_result": harmonized,
        "harmonization_result_sha256": hash_canonical(harmonized),
        "quantitative_mechanics_result": quantitative,
        "quantitative_mechanics_result_sha256": hash_canonical(quantitative),
        "title_abstract_only": True,
        "title_abstract_only_not_release_grade": True,
        "graph_construction_mechanics_authority": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return ContextualNativeProjectionV3.model_validate(
        {
            **projection_payload,
            "projection_sha256": hash_canonical(projection_payload),
        }
    )


def project_contextual_grounded_outcome_v3(
    *,
    fixture_receipt: ContextualGroundingFeasibilityReceiptV3,
    raw_outcome: ContextualPacketOutcomeV3 | Mapping[str, Any],
    runtime_pipeline_sha256: str,
    provider_execution_binding_sha256: str,
) -> tuple[
    ContextualPacketCompletedV3,
    list[ContextualGroundedClaimV3],
    ContextualGroundedEffectV3,
    str,
    str,
    ContextualNativeProjectionV3,
]:
    """Ground one fresh row-17 outcome and construct non-authorizing graph mechanics.

    ``fixture_receipt`` must come from the externally replayed offline suite.  This
    function does not attest how ``raw_outcome`` was generated; a provider lifecycle
    must bind that separately.  The returned projection therefore carries an explicit
    caller-supplied origin blocker and grants no empirical or release authority.
    """

    if (
        fixture_receipt.row_ordinal != ROW17_ORDINAL
        or fixture_receipt.witness_id
        not in {
            "metasyn-row17-candidate2-binary-symptom-endpoint",
            "metasyn-row17-candidate3-binary-primary-endpoint",
        }
        or not fixture_receipt.v2_lineage_external_replayed
        or not fixture_receipt.source_bytes_external_rehashed
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_runtime_projection_requires_replayed_row17_fixture"
        )
    outcome, groundings, effect, grounding_core_sha256 = ground_contextual_outcome_v3(
        provider_binding=fixture_receipt.provider_binding,
        raw_outcome=raw_outcome,
        passages=fixture_receipt.passages,
    )
    projection = _runtime_native_projection_from_fixture(
        fixture_receipt=fixture_receipt,
        effect=effect,
        groundings=groundings,
        grounding_core_sha256=grounding_core_sha256,
        runtime_pipeline_sha256=runtime_pipeline_sha256,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
    )
    assert projection.runtime_grounding_binding_sha256 is not None
    return (
        outcome,
        groundings,
        effect,
        grounding_core_sha256,
        projection.runtime_grounding_binding_sha256,
        projection,
    )


def _freeze_feasibility_receipt(
    *,
    witness_id: Literal[
        "metasyn-row16-candidate1-reported-odds-ratio-ci",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
        "metasyn-row17-candidate3-binary-primary-endpoint",
    ],
    pipeline: PipelineFingerprint,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    source_surface: MetaSynV5SourceSurfaceV1,
    row: MetaSynExtractionRowInputV2,
    inventory: MetaSynCandidateInventoryReceiptV2,
    packet: MetaSynPacketCandidateInputV2,
    provider_binding: ContextualProviderBindingV3,
    outcome: ContextualPacketCompletedV3,
) -> ContextualGroundingFeasibilityReceiptV3:
    passage_ids = sorted({item.passage_id for item in outcome.claims})
    passages = [_source_passage(row=row, passage_id=item) for item in passage_ids]
    parsed_outcome, groundings, effect, grounding_core_sha256 = ground_contextual_outcome_v3(
        provider_binding=provider_binding,
        raw_outcome=outcome,
        passages=passages,
    )
    if witness_id.startswith("metasyn-row16"):
        native = _blocked_native_projection(title_abstract_only=False)
    else:
        native = _completed_native_projection(
            effect=effect,
            groundings=groundings,
            row=row,
            source_surface=source_surface,
            pipeline_sha256=pipeline.pipeline_sha256,
            grounding_core_sha256=grounding_core_sha256,
        )
    source_row = source_surface.rows[row.row_ordinal]
    semantic_target = _semantic_target(provider_binding.context)
    payload = {
        "receipt_version": GROUNDING_RECEIPT_V3_VERSION,
        "witness_id": witness_id,
        "pipeline_fingerprint": pipeline,
        "pipeline_sha256": pipeline.pipeline_sha256,
        "v2_execution_bundle_sha256": bundle.execution_bundle_sha256,
        "v2_extraction_inputs_sha256": bundle.extraction_inputs_sha256,
        "v2_extraction_inputs_pipeline_sha256": (bundle.extraction_inputs_pipeline_sha256),
        "v5_source_surface_sha256": source_surface.source_surface_sha256,
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "row_input_sha256": row.row_input_sha256,
        "source_surface_row_sha256": source_row.source_surface_row_sha256,
        "inventory_receipt_sha256": inventory.receipt_sha256,
        "packet_input_sha256": packet.packet_input_sha256,
        "candidate_descriptor_sha256": packet.candidate_descriptor_sha256,
        "candidate_binding_sha256": packet.candidate_binding_sha256,
        "provider_binding": provider_binding,
        "provider_binding_sha256": provider_binding.binding_sha256,
        "semantic_target": semantic_target,
        "semantic_target_sha256": semantic_target.target_sha256,
        "model_outcome": parsed_outcome,
        "model_outcome_sha256": hash_canonical(parsed_outcome.model_dump(mode="json")),
        "fixture_provenance": (
            "code_owned_offline_source_visible_feasibility_fixture_not_provider_generated"
        ),
        "passages": passages,
        "passage_membership_sha256": hash_canonical([item.passage_sha256 for item in passages]),
        "groundings": groundings,
        "grounding_membership_sha256": hash_canonical(
            [item.grounding_sha256 for item in groundings]
        ),
        "grounded_effect": effect,
        "grounded_effect_sha256": effect.effect_sha256,
        "grounding_core_sha256": grounding_core_sha256,
        "native_projection": native,
        "native_projection_sha256": native.projection_sha256,
        "source_content_scope": row.source_strength.source_content_scope,
        "source_strength_blockers": sorted(row.source_strength.source_strength_blockers),
        "release_grade_source_grounding_eligible": (
            row.source_strength.release_grade_source_grounding_eligible
        ),
        "source_limitations_explicit": True,
        "v2_lineage_external_replayed": True,
        "source_bytes_external_rehashed": True,
        "provider_calls_made": False,
        "reference_fields_opened": False,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return ContextualGroundingFeasibilityReceiptV3.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def freeze_contextual_grounding_offline_feasibility_suite_v3(
    *, repository_root: Path
) -> ContextualGroundingOfflineFeasibilitySuiteV3:
    """Externally replay immutable v2 and freeze two zero-call source witnesses."""

    root = _canonical_root(repository_root)
    bundle = _load_replayed_v2_bundle(root=root)
    source_surface = freeze_metasyn_v5_source_surface(repository_root=root)
    if (
        source_surface.source_surface_sha256
        != bundle.extraction_inputs.upstream_source_surface_sha256
    ):
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_source_surface_bundle_mismatch"
        )
    pipeline = compute_contextual_numeric_grounding_v3_pipeline_fingerprint(repository_root=root)
    specifications = (
        (
            "metasyn-row16-candidate1-reported-odds-ratio-ci",
            ROW16_ORDINAL,
            ROW16_CANDIDATE_INDEX,
            ROW16_PASSAGE_ID,
            _row16_outcome,
        ),
        (
            "metasyn-row17-candidate2-binary-symptom-endpoint",
            ROW17_ORDINAL,
            ROW17_FALLBACK_CANDIDATE_INDEX,
            ROW17_FALLBACK_RESULT_PASSAGE_ID,
            _row17_fallback_outcome,
        ),
        (
            "metasyn-row17-candidate3-binary-primary-endpoint",
            ROW17_ORDINAL,
            ROW17_CANDIDATE_INDEX,
            ROW17_RESULT_PASSAGE_ID,
            _row17_outcome,
        ),
    )
    receipts: list[ContextualGroundingFeasibilityReceiptV3] = []
    for witness_id, row_ordinal, candidate_index, endpoint_passage_id, fixture in specifications:
        row = bundle.extraction_inputs.rows[row_ordinal]
        inventory = _inventory_receipt(root=root, row=row)
        packet = freeze_metasyn_packet_candidate_input_v2(
            extraction_inputs=bundle.extraction_inputs,
            row_ordinal=row_ordinal,
            inventory_receipt=inventory,
            candidate_index=candidate_index,
        )
        context = _provider_context(
            row=row,
            packet=packet,
            endpoint_passage_id=endpoint_passage_id,
        )
        provider = freeze_contextual_provider_binding_v3(context=context)
        outcome = fixture(packet.candidate_binding_sha256)
        receipts.append(
            _freeze_feasibility_receipt(
                witness_id=witness_id,  # type: ignore[arg-type]
                pipeline=pipeline,
                bundle=bundle,
                source_surface=source_surface,
                row=row,
                inventory=inventory,
                packet=packet,
                provider_binding=provider,
                outcome=outcome,
            )
        )
    receipts.sort(key=lambda item: item.witness_id)
    payload = {
        "suite_version": FEASIBILITY_SUITE_V3_VERSION,
        "status": (
            "three_source_visible_offline_witnesses_two_typed_graph_mechanics_no_empirical_authority"
        ),
        "pipeline_fingerprint": pipeline,
        "pipeline_sha256": pipeline.pipeline_sha256,
        "v2_execution_bundle_sha256": bundle.execution_bundle_sha256,
        "v5_source_surface_sha256": source_surface.source_surface_sha256,
        "receipts": receipts,
        "receipt_membership_sha256": hash_canonical([item.receipt_sha256 for item in receipts]),
        "offline_witness_count": 3,
        "contextual_grounding_completed_count": 3,
        "typed_graph_mechanics_completed_count": 2,
        "provider_calls_made": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(
        {**payload, "suite_sha256": hash_canonical(payload)}
    )


def validate_contextual_grounding_offline_feasibility_suite_v3(
    *,
    suite: ContextualGroundingOfflineFeasibilitySuiteV3 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> ContextualGroundingOfflineFeasibilitySuiteV3:
    """Validate a suite and, by default, replay immutable v2 and every source byte."""

    try:
        canonical = ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(
            suite.model_dump(mode="json")
            if isinstance(suite, ContextualGroundingOfflineFeasibilitySuiteV3)
            else suite
        )
    except ValueError as exc:
        raise ContextualNumericGroundingV3Error(
            "contextual_grounding_v3_suite_contract_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_contextual_grounding_offline_feasibility_suite_v3(
            repository_root=repository_root
        )
        if replayed != canonical:
            raise ContextualNumericGroundingV3Error(
                "contextual_grounding_v3_suite_external_replay_mismatch"
            )
    return canonical


__all__ = [
    "CONTEXTUAL_GROUNDING_V3_VERSION",
    "EXPECTED_V2_EXECUTION_BUNDLE_SHA256",
    "ContextualClaimV3",
    "ContextualGroundedClaimV3",
    "ContextualGroundedEffectV3",
    "ContextualGroundingFeasibilityReceiptV3",
    "ContextualGroundingOfflineFeasibilitySuiteV3",
    "ContextualNativeProjectionV3",
    "ContextualNumericGroundingV3Error",
    "ContextualPacketAbstentionV3",
    "ContextualPacketCompletedV3",
    "ContextualProviderBindingV3",
    "ContextualProviderContextV3",
    "ContextualProviderPassageV3",
    "ContextualSemanticTargetV3",
    "ContextualSourcePassageV3",
    "ContextualUnicodeRangeDelimiterV3",
    "compute_contextual_numeric_grounding_v3_pipeline_fingerprint",
    "freeze_contextual_grounding_offline_feasibility_suite_v3",
    "freeze_contextual_provider_binding_v3",
    "ground_contextual_claim_v3",
    "ground_contextual_outcome_v3",
    "project_contextual_grounded_outcome_v3",
    "validate_contextual_grounding_offline_feasibility_suite_v3",
]
