"""Strict model-output schema for native numerical evidence extraction.

Model-generated local keys are never trusted as global identifiers.  The conversion
boundary injects authoritative publication/source identity and deterministically derives
all graph identifiers before validating the complete evidence graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.config import QuestionConfig
from literature_multiverse.effects import (
    EffectAvailability,
    EffectEvidence,
    EffectFormat,
    EquivalenceConclusion,
    ReportedSignificance,
)
from literature_multiverse.evidence_graph import (
    ArmNode,
    ArmRole,
    CohortIdentity,
    CohortIdentityBasis,
    CohortNode,
    ContrastNode,
    EvidenceGraph,
    EvidenceSpan,
    EvidenceSpanRole,
    OutcomeEstimateNode,
    OutcomeTimepoint,
    PublicationIdentity,
    RiskOfBiasAssessment,
    StudyNode,
)
from literature_multiverse.models import ContractModel
from literature_multiverse.schemas import assert_closed_object_schema
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    SourceDocumentArtifact,
    freeze_publication_evidence_fragment,
)


class NativeExtractionContractError(ValueError):
    """A native extractor payload cannot safely become a typed evidence graph."""


class NativeSourceRecord(ContractModel):
    """Authoritative identity supplied outside the model extraction boundary."""

    doc_id: Annotated[str, Field(min_length=1)]
    publication: PublicationIdentity
    source_document: SourceDocumentArtifact

    @model_validator(mode="after")
    def validate_document_identity(self) -> NativeSourceRecord:
        if self.publication.doc_id is not None and self.publication.doc_id != self.doc_id:
            raise ValueError("native_source_publication_doc_id_mismatch")
        return self


class NativeSourceManifest(ContractModel):
    source_manifest_version: Literal["native-source-manifest-v1"] = "native-source-manifest-v1"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    records: Annotated[list[NativeSourceRecord], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_records(self) -> NativeSourceManifest:
        doc_ids = [record.doc_id for record in self.records]
        publication_ids = [record.publication.publication_id for record in self.records]
        paper_ids = [record.publication.paper_id for record in self.records]
        if doc_ids != sorted(set(doc_ids)):
            raise ValueError("native_source_doc_ids_not_sorted_unique")
        if len(publication_ids) != len(set(publication_ids)):
            raise ValueError("native_source_publication_ids_not_unique")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("native_source_paper_ids_not_unique")
        return self


class NativeEvidenceSpan(ContractModel):
    source_locator: Annotated[str, Field(min_length=1)]
    quote: Annotated[str, Field(min_length=1)] | None = None
    section: str | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    char_start: Annotated[int, Field(ge=0)] | None = None
    char_end: Annotated[int, Field(gt=0)] | None = None
    line_ids: list[str] = Field(default_factory=list)

    @field_validator("line_ids")
    @classmethod
    def validate_line_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_evidence_line_ids_not_sorted_unique")
        if any(not line.strip() for line in value):
            raise ValueError("native_evidence_line_id_empty")
        return value

    @model_validator(mode="after")
    def validate_grounding(self) -> NativeEvidenceSpan:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("native_evidence_offsets_require_both")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start >= self.char_end
        ):
            raise ValueError("native_evidence_offsets_not_ordered")
        if self.quote is None and self.char_start is None and not self.line_ids:
            raise ValueError("native_evidence_requires_quote_offsets_or_lines")
        return self


class NativeModeratorValue(ContractModel):
    name: Annotated[str, Field(min_length=1)]
    value: str | int | float | bool | None

    @field_validator("value")
    @classmethod
    def validate_value(
        cls, value: str | int | float | bool | None
    ) -> str | int | float | bool | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("native_moderator_value_nonfinite")
        return value


class NativeEffectPayload(ContractModel):
    """Effect fields supplied by the extractor; identity/provenance are injected."""

    effect_format: EffectFormat
    availability: EffectAvailability = EffectAvailability.AVAILABLE
    estimate: float | None = None
    standard_error: Annotated[float, Field(gt=0)] | None = None
    variance: Annotated[float, Field(gt=0)] | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    unit: str | None = None
    treatment_mean: float | None = None
    treatment_sd: Annotated[float, Field(gt=0)] | None = None
    treatment_n: Annotated[int, Field(ge=2)] | None = None
    control_mean: float | None = None
    control_sd: Annotated[float, Field(gt=0)] | None = None
    control_n: Annotated[int, Field(ge=2)] | None = None
    treatment_events: Annotated[int, Field(ge=0)] | None = None
    treatment_total: Annotated[int, Field(ge=1)] | None = None
    control_events: Annotated[int, Field(ge=0)] | None = None
    control_total: Annotated[int, Field(ge=1)] | None = None
    reported_p_value: Annotated[float, Field(ge=0, le=1)] | None = None
    reported_significance: ReportedSignificance = ReportedSignificance.NOT_REPORTED
    equivalence_conclusion: EquivalenceConclusion = EquivalenceConclusion.NOT_TESTED
    equivalence_margin: Annotated[float, Field(gt=0)] | None = None
    moderators: list[NativeModeratorValue] = Field(default_factory=list)
    extraction_method: Literal["reported", "computed_from_reported_statistics"] = "reported"

    @field_validator(
        "estimate",
        "ci_lower",
        "ci_upper",
        "treatment_mean",
        "control_mean",
    )
    @classmethod
    def validate_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("native_effect_value_nonfinite")
        return value

    @field_validator("moderators")
    @classmethod
    def validate_moderators(cls, value: list[NativeModeratorValue]) -> list[NativeModeratorValue]:
        names = [item.name for item in value]
        if names != sorted(set(names)):
            raise ValueError("native_moderator_names_not_sorted_unique")
        return value

    def to_effect(
        self,
        *,
        paper_id: str,
        finding_id: str,
        outcome: str,
        contrast: str,
        evidence: NativeEvidenceSpan,
    ) -> EffectEvidence:
        effect_payload = self.model_dump(mode="python", exclude={"extraction_method", "moderators"})
        return EffectEvidence(
            paper_id=paper_id,
            finding_id=finding_id,
            outcome=outcome,
            contrast=contrast,
            provenance={
                "source_locator": evidence.source_locator,
                "source_quote": evidence.quote,
                "extraction_method": self.extraction_method,
            },
            moderators={item.name: item.value for item in self.moderators},
            **effect_payload,
        )


class NativeArm(ContractModel):
    key: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
    label: Annotated[str, Field(min_length=1)]
    role: ArmRole
    description: str | None = None
    sample_size: Annotated[int, Field(ge=1)] | None = None


class NativeContrast(ContractModel):
    key: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
    treatment_arm_key: Annotated[str, Field(min_length=1)]
    comparator_arm_key: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    estimand: str | None = None
    positive_direction_means: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_distinct_arms(self) -> NativeContrast:
        if self.treatment_arm_key == self.comparator_arm_key:
            raise ValueError("native_contrast_requires_distinct_arms")
        return self


class NativeFinding(ContractModel):
    key: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
    contrast_key: Annotated[str, Field(min_length=1)]
    outcome_name: Annotated[str, Field(min_length=1)]
    timepoint: OutcomeTimepoint
    analysis_population: str | None = None
    effect: NativeEffectPayload
    evidence: NativeEvidenceSpan


class NativeCohort(ContractModel):
    key: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
    source_labels: Annotated[list[str], Field(min_length=1)]
    registry_ids: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    population_description: str | None = None
    recruitment_period: str | None = None
    total_sample_size: Annotated[int, Field(ge=1)] | None = None
    arms: Annotated[list[NativeArm], Field(min_length=2)]
    contrasts: Annotated[list[NativeContrast], Field(min_length=1)]
    findings: list[NativeFinding] = Field(default_factory=list)

    @field_validator("source_labels", "registry_ids", "dataset_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_cohort_identity_values_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> NativeCohort:
        arm_keys = [arm.key for arm in self.arms]
        contrast_keys = [contrast.key for contrast in self.contrasts]
        finding_keys = [finding.key for finding in self.findings]
        if len(arm_keys) != len(set(arm_keys)):
            raise ValueError("native_arm_keys_not_unique")
        if len(contrast_keys) != len(set(contrast_keys)):
            raise ValueError("native_contrast_keys_not_unique")
        if len(finding_keys) != len(set(finding_keys)):
            raise ValueError("native_finding_keys_not_unique")
        arm_set = set(arm_keys)
        for contrast in self.contrasts:
            if contrast.treatment_arm_key not in arm_set:
                raise ValueError("native_contrast_treatment_arm_unknown")
            if contrast.comparator_arm_key not in arm_set:
                raise ValueError("native_contrast_comparator_arm_unknown")
        contrast_set = set(contrast_keys)
        if any(finding.contrast_key not in contrast_set for finding in self.findings):
            raise ValueError("native_finding_contrast_unknown")
        return self


class NativeStudy(ContractModel):
    key: Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
    source_label: Annotated[str, Field(min_length=1)]
    design: str | None = None
    registration_ids: list[str] = Field(default_factory=list)
    cohorts: Annotated[list[NativeCohort], Field(min_length=1)]

    @field_validator("registration_ids")
    @classmethod
    def validate_registration_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_study_registration_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_cohort_keys(self) -> NativeStudy:
        keys = [cohort.key for cohort in self.cohorts]
        if len(keys) != len(set(keys)):
            raise ValueError("native_cohort_keys_not_unique_within_study")
        return self


class NativePublicationExtraction(ContractModel):
    """Identity-free scientific payload returned by a native extraction worker."""

    extraction_schema_version: Literal["native-publication-extraction-v1"] = (
        "native-publication-extraction-v1"
    )
    status: FragmentStatus
    studies: list[NativeStudy] = Field(default_factory=list)
    non_estimability_reason: NonEstimabilityReason | None = None
    non_estimability_detail: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_extraction_warnings_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> NativePublicationExtraction:
        finding_count = sum(
            len(cohort.findings) for study in self.studies for cohort in study.cohorts
        )
        if self.status is FragmentStatus.ESTIMABLE:
            if not self.studies or finding_count == 0:
                raise ValueError("native_estimable_extraction_requires_findings")
            if self.non_estimability_reason is not None or self.non_estimability_detail is not None:
                raise ValueError("native_estimable_extraction_forbids_missing_metadata")
        else:
            if self.studies:
                raise ValueError("native_non_estimable_extraction_forbids_studies")
            if self.non_estimability_reason is None:
                raise ValueError("native_non_estimable_extraction_requires_reason")
            if self.non_estimability_reason is NonEstimabilityReason.OTHER and not (
                self.non_estimability_detail and self.non_estimability_detail.strip()
            ):
                raise ValueError("native_other_non_estimability_requires_detail")
        return self


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _cohort_identity(cohort: NativeCohort, *, cohort_id: str) -> CohortIdentity:
    if cohort.registry_ids:
        return CohortIdentity(
            cohort_id=cohort_id,
            basis=CohortIdentityBasis.REPORTED_REGISTRY_ID,
            source_labels=cohort.source_labels,
            registry_ids=cohort.registry_ids,
            dataset_ids=cohort.dataset_ids,
        )
    if cohort.dataset_ids:
        return CohortIdentity(
            cohort_id=cohort_id,
            basis=CohortIdentityBasis.REPORTED_DATASET_ID,
            source_labels=cohort.source_labels,
            dataset_ids=cohort.dataset_ids,
        )
    return CohortIdentity(
        cohort_id=cohort_id,
        basis=CohortIdentityBasis.SOURCE_REPORTED_LABEL,
        source_labels=cohort.source_labels,
    )


def freeze_native_publication_extraction(
    *,
    payload: NativePublicationExtraction,
    question_id: str,
    publication: PublicationIdentity,
    pipeline_fingerprint_sha256: str,
    extraction_context_sha256: str | None = None,
    source_document: SourceDocumentArtifact,
    grounding_receipt_sha256: str | None,
) -> PublicationEvidenceFragment:
    """Inject authoritative identity and freeze one native extraction fragment."""

    if payload.status is FragmentStatus.NON_ESTIMABLE:
        return freeze_publication_evidence_fragment(
            question_id=question_id,
            publication_id=publication.publication_id,
            paper_id=publication.paper_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_fingerprint_sha256,
            extraction_context_sha256=extraction_context_sha256,
            source_document=source_document,
            grounding_receipt_sha256=grounding_receipt_sha256,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=payload.non_estimability_reason,
            non_estimability_detail=payload.non_estimability_detail,
            extractor_warnings=payload.warnings,
        )

    studies: list[StudyNode] = []
    cohorts: list[CohortNode] = []
    arms: list[ArmNode] = []
    contrasts: list[ContrastNode] = []
    estimates: list[OutcomeEstimateNode] = []
    spans: list[EvidenceSpan] = []
    for study_payload in payload.studies:
        study_id = _id("study", publication.publication_id, study_payload.key)
        studies.append(
            StudyNode(
                study_id=study_id,
                publication_ids=[publication.publication_id],
                primary_publication_id=publication.publication_id,
                design=study_payload.design,
                registration_ids=study_payload.registration_ids,
                risk_of_bias=RiskOfBiasAssessment(),
            )
        )
        for cohort_payload in study_payload.cohorts:
            cohort_id = _id(
                "cohort",
                publication.publication_id,
                study_payload.key,
                cohort_payload.key,
            )
            cohorts.append(
                CohortNode(
                    identity=_cohort_identity(cohort_payload, cohort_id=cohort_id),
                    study_id=study_id,
                    population_description=cohort_payload.population_description,
                    recruitment_period=cohort_payload.recruitment_period,
                    total_sample_size=cohort_payload.total_sample_size,
                )
            )
            arm_ids: dict[str, str] = {}
            for arm_payload in cohort_payload.arms:
                arm_id = _id("arm", cohort_id, arm_payload.key)
                arm_ids[arm_payload.key] = arm_id
                arms.append(
                    ArmNode(
                        arm_id=arm_id,
                        cohort_id=cohort_id,
                        label=arm_payload.label,
                        role=arm_payload.role,
                        description=arm_payload.description,
                        sample_size=arm_payload.sample_size,
                    )
                )
            contrast_ids: dict[str, str] = {}
            contrast_labels: dict[str, str] = {}
            for contrast_payload in cohort_payload.contrasts:
                contrast_id = _id("contrast", cohort_id, contrast_payload.key)
                contrast_ids[contrast_payload.key] = contrast_id
                contrast_labels[contrast_payload.key] = contrast_payload.label
                contrasts.append(
                    ContrastNode(
                        contrast_id=contrast_id,
                        cohort_id=cohort_id,
                        treatment_arm_id=arm_ids[contrast_payload.treatment_arm_key],
                        comparator_arm_id=arm_ids[contrast_payload.comparator_arm_key],
                        label=contrast_payload.label,
                        estimand=contrast_payload.estimand,
                        positive_direction_means=(contrast_payload.positive_direction_means),
                    )
                )
            for finding_payload in cohort_payload.findings:
                finding_id = _id(
                    "finding",
                    publication.publication_id,
                    study_payload.key,
                    cohort_payload.key,
                    finding_payload.key,
                )
                span_id = _id(
                    "span",
                    publication.publication_id,
                    finding_payload.evidence.source_locator,
                    finding_payload.evidence.quote or "",
                )
                spans.append(
                    EvidenceSpan(
                        span_id=span_id,
                        publication_id=publication.publication_id,
                        source_locator=finding_payload.evidence.source_locator,
                        quote=finding_payload.evidence.quote,
                        section=finding_payload.evidence.section,
                        page=finding_payload.evidence.page,
                        char_start=finding_payload.evidence.char_start,
                        char_end=finding_payload.evidence.char_end,
                        line_ids=finding_payload.evidence.line_ids,
                        roles=[EvidenceSpanRole.NUMERICAL_RESULT],
                    )
                )
                contrast_label = contrast_labels[finding_payload.contrast_key]
                effect = finding_payload.effect.to_effect(
                    paper_id=publication.paper_id,
                    finding_id=finding_id,
                    outcome=finding_payload.outcome_name,
                    contrast=contrast_label,
                    evidence=finding_payload.evidence,
                )
                estimates.append(
                    OutcomeEstimateNode(
                        estimate_id=_id("estimate", finding_id),
                        contrast_id=contrast_ids[finding_payload.contrast_key],
                        outcome_name=finding_payload.outcome_name,
                        timepoint=finding_payload.timepoint,
                        analysis_population=finding_payload.analysis_population,
                        effect=effect,
                        evidence_span_ids=[span_id],
                    )
                )
    graph = EvidenceGraph(
        publications=[publication],
        studies=studies,
        cohorts=cohorts,
        arms=arms,
        contrasts=contrasts,
        outcome_estimates=estimates,
        evidence_spans=spans,
    )
    return freeze_publication_evidence_fragment(
        question_id=question_id,
        publication_id=publication.publication_id,
        paper_id=publication.paper_id,
        publication=publication,
        pipeline_fingerprint_sha256=pipeline_fingerprint_sha256,
        extraction_context_sha256=extraction_context_sha256,
        source_document=source_document,
        grounding_receipt_sha256=grounding_receipt_sha256,
        status=FragmentStatus.ESTIMABLE,
        graph=graph,
        extractor_warnings=payload.warnings,
    )


def native_publication_extraction_json_schema() -> dict[str, Any]:
    schema = NativePublicationExtraction.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:literature-multiverse:native-publication-extraction:v1"
    assert_closed_object_schema(schema)
    return schema


def native_extraction_prompt_replacements(config: QuestionConfig) -> dict[str, str]:
    """Render only prespecified scientific fields into the native extraction prompt."""

    return {
        "QUESTION_SPEC_JSON": json.dumps(
            {
                "research_question": config.research_question,
                "target_relation": config.target_relation.model_dump(mode="json"),
                "outcomes": config.outcomes.model_dump(mode="json"),
                "moderators": [
                    moderator.model_dump(mode="json") for moderator in config.moderators
                ],
                "eligibility": config.eligibility.model_dump(mode="json"),
            },
            sort_keys=True,
        )
    }


__all__ = [
    "NativeArm",
    "NativeCohort",
    "NativeContrast",
    "NativeEffectPayload",
    "NativeEvidenceSpan",
    "NativeExtractionContractError",
    "NativeFinding",
    "NativeModeratorValue",
    "NativePublicationExtraction",
    "NativeSourceManifest",
    "NativeSourceRecord",
    "NativeStudy",
    "freeze_native_publication_extraction",
    "native_extraction_prompt_replacements",
    "native_publication_extraction_json_schema",
]
