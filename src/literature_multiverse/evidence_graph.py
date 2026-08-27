"""Closed evidence-graph contracts and conservative legacy adapters.

The graph makes the scientific unit structure explicit::

    publication -> study -> cohort -> arm -> contrast -> outcome estimate -> span

Publications and cohorts are deliberately different identities: several publications can
report one cohort, and one publication can report several cohorts.  Quantitative callers
must not infer either relationship from a paper identifier.

The legacy adapters preserve categorical statements but never turn ``no_effect`` or a
non-significant result into a numerical zero.  A legacy free-text effect remains
non-estimable until a source-grounded numerical extractor supplies its estimand, scale,
and uncertainty.
"""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.effects import (
    EffectAvailability,
    EffectEvidence,
    EffectFormat,
    EffectProvenance,
    EquivalenceConclusion,
    ReportedSignificance,
    harmonize_effect,
)
from literature_multiverse.models import (
    CanonicalDirection,
    ContractModel,
    FindingRow,
    normalize_doi,
)


class EvidenceGraphContractError(ValueError):
    """The graph cannot safely represent or select the requested evidence."""


class CohortIdentityBasis(StrEnum):
    """How the independent participant/sample identity was established."""

    REPORTED_REGISTRY_ID = "reported_registry_id"
    REPORTED_DATASET_ID = "reported_dataset_id"
    SOURCE_REPORTED_LABEL = "source_reported_label"
    REVIEWER_RECONCILED = "reviewer_reconciled"
    LEGACY_PLACEHOLDER = "legacy_placeholder"


class ArmRole(StrEnum):
    INTERVENTION = "intervention"
    COMPARATOR = "comparator"
    EXPOSURE = "exposure"
    CONTROL = "control"
    OTHER = "other"


class TimepointKind(StrEnum):
    EXACT = "exact"
    RANGE = "range"
    REPORTED_TEXT = "reported_text"
    NOT_REPORTED = "not_reported"


class TimeUnit(StrEnum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class RiskOfBiasJudgement(StrEnum):
    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"
    CRITICAL = "critical"
    UNCLEAR = "unclear"
    NOT_ASSESSED = "not_assessed"


class EvidenceSpanRole(StrEnum):
    DESIGN = "design"
    POPULATION = "population"
    INTERVENTION = "intervention"
    COMPARATOR = "comparator"
    OUTCOME = "outcome"
    NUMERICAL_RESULT = "numerical_result"
    RISK_OF_BIAS = "risk_of_bias"
    OTHER = "other"


class AdapterIssueSeverity(StrEnum):
    WARNING = "warning"
    BLOCKING = "blocking"


class PublicationIdentity(ContractModel):
    """One immutable publication identity, distinct from the study/cohort it reports."""

    publication_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    doc_id: str | None = None
    doi: str | None = None
    pmid: str | None = None
    title: str | None = None
    publication_year: Annotated[int, Field(ge=1000, le=3000)] | None = None

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, value: str | None) -> str | None:
        return normalize_doi(value) if value is not None else None

    @field_validator("pmid")
    @classmethod
    def validate_pmid(cls, value: str | None) -> str | None:
        if value is not None and not value.isdigit():
            raise ValueError("invalid_pmid")
        return value


class EvidenceSpan(ContractModel):
    """An auditable source passage with at least one exact representation."""

    span_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]
    source_locator: Annotated[str, Field(min_length=1)]
    quote: str | None = None
    section: str | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    char_start: Annotated[int, Field(ge=0)] | None = None
    char_end: Annotated[int, Field(gt=0)] | None = None
    line_ids: list[str] = Field(default_factory=list)
    roles: list[EvidenceSpanRole] = Field(default_factory=list)

    @field_validator("line_ids", "roles")
    @classmethod
    def validate_sorted_unique(cls, value: list[object]) -> list[object]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_span_list_must_be_unique")
        return value

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceSpan:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("evidence_span_offsets_require_both_bounds")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start >= self.char_end
        ):
            raise ValueError("evidence_span_offsets_not_ordered")
        has_quote = self.quote is not None and bool(self.quote.strip())
        has_offsets = self.char_start is not None and self.char_end is not None
        if not (has_quote or has_offsets or self.line_ids):
            raise ValueError("evidence_span_requires_quote_offsets_or_line_ids")
        return self


class RiskOfBiasDomain(ContractModel):
    """One tool-specific risk-of-bias domain with its supporting spans."""

    domain_id: Annotated[str, Field(min_length=1)]
    judgement: RiskOfBiasJudgement
    rationale: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)

    @field_validator("evidence_span_ids")
    @classmethod
    def validate_evidence_span_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("risk_of_bias_span_ids_must_be_sorted_unique")
        return value


class RiskOfBiasAssessment(ContractModel):
    """Explicit assessment state; absence is represented as ``not_assessed``."""

    tool: str | None = None
    overall: RiskOfBiasJudgement = RiskOfBiasJudgement.NOT_ASSESSED
    domains: list[RiskOfBiasDomain] = Field(default_factory=list)
    assessor: str | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> RiskOfBiasAssessment:
        domain_ids = [domain.domain_id for domain in self.domains]
        if len(domain_ids) != len(set(domain_ids)):
            raise ValueError("risk_of_bias_domain_ids_not_unique")
        if self.overall is RiskOfBiasJudgement.NOT_ASSESSED:
            if self.domains or self.tool is not None or self.assessor is not None:
                raise ValueError("not_assessed_risk_of_bias_cannot_have_assessment_details")
        elif self.tool is None:
            raise ValueError("assessed_risk_of_bias_requires_tool")
        return self


class OutcomeTimepoint(ContractModel):
    """A timepoint without guessing conversions from free text."""

    kind: TimepointKind
    value: Annotated[float, Field(ge=0)] | None = None
    lower: Annotated[float, Field(ge=0)] | None = None
    upper: Annotated[float, Field(ge=0)] | None = None
    unit: TimeUnit | None = None
    anchor: str | None = None
    raw_label: str | None = None

    @field_validator("value", "lower", "upper")
    @classmethod
    def validate_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("timepoint_value_must_be_finite")
        return value

    @model_validator(mode="after")
    def validate_timepoint(self) -> OutcomeTimepoint:
        if self.kind is TimepointKind.EXACT:
            if self.value is None or self.unit is None:
                raise ValueError("exact_timepoint_requires_value_and_unit")
            if self.lower is not None or self.upper is not None:
                raise ValueError("exact_timepoint_forbids_range")
        elif self.kind is TimepointKind.RANGE:
            if self.lower is None or self.upper is None or self.unit is None:
                raise ValueError("range_timepoint_requires_bounds_and_unit")
            if self.lower >= self.upper:
                raise ValueError("timepoint_range_not_ordered")
            if self.value is not None:
                raise ValueError("range_timepoint_forbids_exact_value")
        elif self.kind is TimepointKind.REPORTED_TEXT:
            if self.raw_label is None:
                raise ValueError("reported_text_timepoint_requires_raw_label")
            if any(item is not None for item in (self.value, self.lower, self.upper, self.unit)):
                raise ValueError("reported_text_timepoint_forbids_inferred_numeric_value")
        elif any(
            item is not None
            for item in (
                self.value,
                self.lower,
                self.upper,
                self.unit,
                self.anchor,
                self.raw_label,
            )
        ):
            raise ValueError("not_reported_timepoint_cannot_have_details")
        return self


class StudyNode(ContractModel):
    """A scientific study, which may be reported by several publications."""

    study_id: Annotated[str, Field(min_length=1)]
    publication_ids: Annotated[list[str], Field(min_length=1)]
    primary_publication_id: Annotated[str, Field(min_length=1)]
    design: str | None = None
    registration_ids: list[str] = Field(default_factory=list)
    risk_of_bias: RiskOfBiasAssessment = Field(default_factory=RiskOfBiasAssessment)

    @field_validator("publication_ids", "registration_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("study_identity_lists_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_primary_publication(self) -> StudyNode:
        if self.primary_publication_id not in self.publication_ids:
            raise ValueError("study_primary_publication_not_in_publication_ids")
        return self


class CohortIdentity(ContractModel):
    """Identity of an independent participant/sample cohort across publications."""

    cohort_id: Annotated[str, Field(min_length=1)]
    basis: CohortIdentityBasis
    source_labels: list[str] = Field(default_factory=list)
    registry_ids: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @field_validator("source_labels", "registry_ids", "dataset_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("cohort_identity_lists_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_basis(self) -> CohortIdentity:
        if self.basis is CohortIdentityBasis.REPORTED_REGISTRY_ID and not self.registry_ids:
            raise ValueError("registry_identity_requires_registry_id")
        if self.basis is CohortIdentityBasis.REPORTED_DATASET_ID and not self.dataset_ids:
            raise ValueError("dataset_identity_requires_dataset_id")
        if self.basis is CohortIdentityBasis.SOURCE_REPORTED_LABEL and not self.source_labels:
            raise ValueError("source_label_identity_requires_source_label")
        if self.basis in {
            CohortIdentityBasis.REVIEWER_RECONCILED,
            CohortIdentityBasis.LEGACY_PLACEHOLDER,
        } and self.rationale is None:
            raise ValueError("reviewer_or_placeholder_identity_requires_rationale")
        return self


class CohortNode(ContractModel):
    """A participant/sample cohort nested in one study."""

    identity: CohortIdentity
    study_id: Annotated[str, Field(min_length=1)]
    population_description: str | None = None
    recruitment_period: str | None = None
    total_sample_size: Annotated[int, Field(ge=1)] | None = None
    risk_of_bias: RiskOfBiasAssessment = Field(default_factory=RiskOfBiasAssessment)

    @property
    def cohort_id(self) -> str:
        return self.identity.cohort_id


class ArmNode(ContractModel):
    """One assigned/exposed group within a cohort."""

    arm_id: Annotated[str, Field(min_length=1)]
    cohort_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    role: ArmRole
    description: str | None = None
    sample_size: Annotated[int, Field(ge=1)] | None = None


class ContrastNode(ContractModel):
    """An oriented comparison between two distinct arms from the same cohort."""

    contrast_id: Annotated[str, Field(min_length=1)]
    cohort_id: Annotated[str, Field(min_length=1)]
    treatment_arm_id: Annotated[str, Field(min_length=1)]
    comparator_arm_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    estimand: str | None = None
    positive_direction_means: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_distinct_arms(self) -> ContrastNode:
        if self.treatment_arm_id == self.comparator_arm_id:
            raise ValueError("contrast_requires_two_distinct_arms")
        return self


class OutcomeEstimateNode(ContractModel):
    """One outcome estimate for one oriented contrast at one explicit timepoint."""

    estimate_id: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)]
    outcome_name: Annotated[str, Field(min_length=1)]
    timepoint: OutcomeTimepoint
    analysis_population: str | None = None
    effect: EffectEvidence
    evidence_span_ids: Annotated[list[str], Field(min_length=1)]
    risk_of_bias: RiskOfBiasAssessment = Field(default_factory=RiskOfBiasAssessment)
    # Retained only as a source statement.  It is never a numerical point estimate.
    legacy_reported_direction: CanonicalDirection | None = None
    legacy_effect_size_raw: str | None = None

    @field_validator("evidence_span_ids")
    @classmethod
    def validate_span_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("outcome_estimate_span_ids_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_effect_identity(self) -> OutcomeEstimateNode:
        if self.effect.outcome != self.outcome_name:
            raise ValueError("outcome_estimate_effect_outcome_mismatch")
        return self


class EvidenceGraph(ContractModel):
    """A closed, referentially valid evidence graph."""

    graph_schema_version: Literal["1"] = "1"
    publications: Annotated[list[PublicationIdentity], Field(min_length=1)]
    studies: Annotated[list[StudyNode], Field(min_length=1)]
    cohorts: Annotated[list[CohortNode], Field(min_length=1)]
    arms: Annotated[list[ArmNode], Field(min_length=2)]
    contrasts: Annotated[list[ContrastNode], Field(min_length=1)]
    outcome_estimates: Annotated[list[OutcomeEstimateNode], Field(min_length=1)]
    evidence_spans: Annotated[list[EvidenceSpan], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_graph(self) -> EvidenceGraph:
        publications = _index_unique(self.publications, "publication_id", "publication")
        studies = _index_unique(self.studies, "study_id", "study")
        cohorts = _index_unique_by_property(self.cohorts, "cohort_id", "cohort")
        arms = _index_unique(self.arms, "arm_id", "arm")
        contrasts = _index_unique(self.contrasts, "contrast_id", "contrast")
        estimates = _index_unique(self.outcome_estimates, "estimate_id", "estimate")
        spans = _index_unique(self.evidence_spans, "span_id", "evidence_span")
        del estimates

        global_ids = [
            *publications,
            *studies,
            *cohorts,
            *arms,
            *contrasts,
            *spans,
            *(item.estimate_id for item in self.outcome_estimates),
        ]
        if len(global_ids) != len(set(global_ids)):
            raise ValueError("graph_node_ids_must_be_globally_unique")

        paper_ids = [publication.paper_id for publication in self.publications]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("publication_paper_ids_not_unique")
        for field_name in ("doc_id", "doi", "pmid"):
            values = [
                getattr(publication, field_name)
                for publication in self.publications
                if getattr(publication, field_name) is not None
            ]
            if len(values) != len(set(values)):
                raise ValueError(f"publication_{field_name}_values_not_unique")

        for span in self.evidence_spans:
            _require_ref(span.publication_id, publications, "evidence_span_publication")

        for study in self.studies:
            _require_refs(study.publication_ids, publications, "study_publication")
            _validate_risk_spans(study.risk_of_bias, spans)
        for cohort in self.cohorts:
            _require_ref(cohort.study_id, studies, "cohort_study")
            _validate_risk_spans(cohort.risk_of_bias, spans)
        for arm in self.arms:
            _require_ref(arm.cohort_id, cohorts, "arm_cohort")
        for contrast in self.contrasts:
            _require_ref(contrast.cohort_id, cohorts, "contrast_cohort")
            treatment = _require_ref(contrast.treatment_arm_id, arms, "contrast_treatment_arm")
            comparator = _require_ref(
                contrast.comparator_arm_id, arms, "contrast_comparator_arm"
            )
            if treatment.cohort_id != contrast.cohort_id:
                raise ValueError("contrast_treatment_arm_belongs_to_different_cohort")
            if comparator.cohort_id != contrast.cohort_id:
                raise ValueError("contrast_comparator_arm_belongs_to_different_cohort")

        publication_by_paper = {
            publication.paper_id: publication for publication in self.publications
        }
        for estimate in self.outcome_estimates:
            contrast = _require_ref(estimate.contrast_id, contrasts, "estimate_contrast")
            cohort = cohorts[contrast.cohort_id]
            study = studies[cohort.study_id]
            publication = publication_by_paper.get(estimate.effect.paper_id)
            if publication is None:
                raise ValueError("estimate_effect_paper_not_in_graph")
            if publication.publication_id not in study.publication_ids:
                raise ValueError("estimate_effect_publication_not_linked_to_study")
            if estimate.effect.contrast != contrast.label:
                raise ValueError("estimate_effect_contrast_label_mismatch")
            _require_refs(estimate.evidence_span_ids, spans, "estimate_evidence_span")
            allowed_publications = set(study.publication_ids)
            if any(
                spans[span_id].publication_id not in allowed_publications
                for span_id in estimate.evidence_span_ids
            ):
                raise ValueError("estimate_span_publication_not_linked_to_study")
            if any(
                spans[span_id].publication_id != publication.publication_id
                for span_id in estimate.evidence_span_ids
            ):
                raise ValueError("estimate_span_publication_mismatch_effect_paper")
            if not any(
                spans[span_id].source_locator == estimate.effect.provenance.source_locator
                for span_id in estimate.evidence_span_ids
            ):
                raise ValueError("effect_provenance_locator_missing_from_estimate_spans")
            _validate_risk_spans(estimate.risk_of_bias, spans)
        return self


class GraphAdapterContext(ContractModel):
    """Caller-supplied identities that legacy row formats do not contain."""

    publication: PublicationIdentity
    study_id: Annotated[str, Field(min_length=1)]
    cohort_identity: CohortIdentity
    treatment_arm_id: Annotated[str, Field(min_length=1)]
    comparator_arm_id: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)]
    contrast_label: Annotated[str, Field(min_length=1)]
    positive_direction_means: Annotated[str, Field(min_length=1)]
    treatment_label: Annotated[str, Field(min_length=1)]
    comparator_label: Annotated[str, Field(min_length=1)]
    timepoint: OutcomeTimepoint | None = None
    risk_of_bias: RiskOfBiasAssessment = Field(default_factory=RiskOfBiasAssessment)


class AdapterIssue(ContractModel):
    severity: AdapterIssueSeverity
    code: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]


class EvidenceGraphAdapterResult(ContractModel):
    """Graph conversion plus machine-readable caveats; no caveat is silently dropped."""

    status: Literal["ready", "requires_review"]
    graph: EvidenceGraph
    issues: list[AdapterIssue]

    @model_validator(mode="after")
    def validate_status(self) -> EvidenceGraphAdapterResult:
        blocking = any(issue.severity is AdapterIssueSeverity.BLOCKING for issue in self.issues)
        if (self.status == "requires_review") != blocking:
            raise ValueError("adapter_status_does_not_match_blocking_issues")
        return self


class GraphEffectSelection(ContractModel):
    """Safe extraction result for the existing paper-clustered meta-analysis boundary."""

    status: Literal["ready", "insufficient"]
    reason: str | None = None
    records: list[EffectEvidence] = Field(default_factory=list)
    estimate_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> GraphEffectSelection:
        if self.status == "ready" and (self.reason is not None or not self.records):
            raise ValueError("ready_graph_selection_requires_records_only")
        if self.status == "insufficient" and (self.reason is None or self.records):
            raise ValueError("insufficient_graph_selection_requires_reason_only")
        if len(self.records) != len(self.estimate_ids):
            raise ValueError("graph_selection_record_count_mismatch")
        return self


class EvidenceGraphRiskFeatures(ContractModel):
    """Prospective, label-free graph diagnostics suitable for a release-risk model."""

    feature_schema_version: Literal["1"] = "1"
    n_estimates: Annotated[int, Field(ge=0)]
    n_publications: Annotated[int, Field(ge=0)]
    n_cohorts: Annotated[int, Field(ge=0)]
    fraction_non_estimable: Annotated[float, Field(ge=0, le=1)]
    fraction_missing_source_quote: Annotated[float, Field(ge=0, le=1)]
    fraction_timepoint_not_reported: Annotated[float, Field(ge=0, le=1)]
    fraction_risk_of_bias_not_assessed: Annotated[float, Field(ge=0, le=1)]
    fraction_high_or_critical_risk_of_bias: Annotated[float, Field(ge=0, le=1)]
    fraction_unresolved_cohort_identity: Annotated[float, Field(ge=0, le=1)]

    def as_calibration_features(self) -> dict[str, float]:
        """Return a stable sorted numeric mapping for ``ReleaseCandidate.features``."""

        payload = self.model_dump(mode="python", exclude={"feature_schema_version"})
        return {name: float(payload[name]) for name in sorted(payload)}


def _index_unique(items: list[object], attribute: str, label: str) -> dict[str, object]:
    index: dict[str, object] = {}
    for item in items:
        identifier = getattr(item, attribute)
        if identifier in index:
            raise ValueError(f"duplicate_{label}_id:{identifier}")
        index[identifier] = item
    return index


def _index_unique_by_property(
    items: list[object], attribute: str, label: str
) -> dict[str, object]:
    return _index_unique(items, attribute, label)


def _require_ref(identifier: str, index: dict[str, object], label: str) -> object:
    try:
        return index[identifier]
    except KeyError as exc:
        raise ValueError(f"missing_{label}:{identifier}") from exc


def _require_refs(identifiers: list[str], index: dict[str, object], label: str) -> None:
    for identifier in identifiers:
        _require_ref(identifier, index, label)


def _validate_risk_spans(
    assessment: RiskOfBiasAssessment, spans: dict[str, object]
) -> None:
    for domain in assessment.domains:
        _require_refs(domain.evidence_span_ids, spans, "risk_of_bias_evidence_span")


def _span_id(publication_id: str, locator: str, quote: str | None) -> str:
    digest = hashlib.sha256(
        f"{publication_id}\x1f{locator}\x1f{quote or ''}".encode()
    ).hexdigest()[:16]
    return f"span-{digest}"


def _reported_significance(value: bool | None) -> ReportedSignificance:
    if value is True:
        return ReportedSignificance.SIGNIFICANT
    if value is False:
        return ReportedSignificance.NOT_SIGNIFICANT
    return ReportedSignificance.NOT_REPORTED


def _legacy_timepoint(finding: FindingRow) -> OutcomeTimepoint:
    if finding.timepoint_raw is None:
        return OutcomeTimepoint(kind=TimepointKind.NOT_REPORTED)
    return OutcomeTimepoint(
        kind=TimepointKind.REPORTED_TEXT,
        raw_label=finding.timepoint_raw,
        anchor=finding.timing_context,
    )


def _build_single_estimate_graph(
    *,
    evidence: EffectEvidence,
    context: GraphAdapterContext,
    estimate_id: str,
    timepoint: OutcomeTimepoint,
    study_design: str | None,
    population_description: str | None,
    total_sample_size: int | None,
    treatment_description: str | None,
    comparator_description: str | None,
    evidence_section: str | None,
    evidence_line_ids: list[str],
    legacy_reported_direction: CanonicalDirection | None,
    legacy_effect_size_raw: str | None,
) -> EvidenceGraph:
    if evidence.paper_id != context.publication.paper_id:
        raise EvidenceGraphContractError("adapter_publication_paper_id_mismatch")
    if evidence.contrast != context.contrast_label:
        raise EvidenceGraphContractError("adapter_contrast_label_mismatch")
    locator = evidence.provenance.source_locator
    span_id = _span_id(
        context.publication.publication_id, locator, evidence.provenance.source_quote
    )
    span = EvidenceSpan(
        span_id=span_id,
        publication_id=context.publication.publication_id,
        source_locator=locator,
        quote=evidence.provenance.source_quote,
        section=evidence_section,
        line_ids=evidence_line_ids,
        roles=[EvidenceSpanRole.NUMERICAL_RESULT],
    )
    return EvidenceGraph(
        publications=[context.publication],
        studies=[
            StudyNode(
                study_id=context.study_id,
                publication_ids=[context.publication.publication_id],
                primary_publication_id=context.publication.publication_id,
                design=study_design,
                risk_of_bias=context.risk_of_bias,
            )
        ],
        cohorts=[
            CohortNode(
                identity=context.cohort_identity,
                study_id=context.study_id,
                population_description=population_description,
                total_sample_size=total_sample_size,
            )
        ],
        arms=[
            ArmNode(
                arm_id=context.treatment_arm_id,
                cohort_id=context.cohort_identity.cohort_id,
                label=context.treatment_label,
                role=ArmRole.INTERVENTION,
                description=treatment_description,
            ),
            ArmNode(
                arm_id=context.comparator_arm_id,
                cohort_id=context.cohort_identity.cohort_id,
                label=context.comparator_label,
                role=ArmRole.COMPARATOR,
                description=comparator_description,
            ),
        ],
        contrasts=[
            ContrastNode(
                contrast_id=context.contrast_id,
                cohort_id=context.cohort_identity.cohort_id,
                treatment_arm_id=context.treatment_arm_id,
                comparator_arm_id=context.comparator_arm_id,
                label=context.contrast_label,
                positive_direction_means=context.positive_direction_means,
            )
        ],
        outcome_estimates=[
            OutcomeEstimateNode(
                estimate_id=estimate_id,
                contrast_id=context.contrast_id,
                outcome_name=evidence.outcome,
                timepoint=timepoint,
                effect=evidence,
                evidence_span_ids=[span_id],
                risk_of_bias=context.risk_of_bias,
                legacy_reported_direction=legacy_reported_direction,
                legacy_effect_size_raw=legacy_effect_size_raw,
            )
        ],
        evidence_spans=[span],
    )


def adapt_effect_evidence(
    evidence: EffectEvidence, *, context: GraphAdapterContext
) -> EvidenceGraphAdapterResult:
    """Lift typed numerical evidence into the graph without inventing missing identity/time."""

    issues: list[AdapterIssue] = []
    if context.cohort_identity.basis is CohortIdentityBasis.LEGACY_PLACEHOLDER:
        issues.append(
            AdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="unresolved_cohort_identity",
                detail="Replace the legacy placeholder before independent-unit synthesis.",
            )
        )
    timepoint = context.timepoint or OutcomeTimepoint(kind=TimepointKind.NOT_REPORTED)
    if context.timepoint is None:
        issues.append(
            AdapterIssue(
                severity=AdapterIssueSeverity.WARNING,
                code="timepoint_not_reported",
                detail="No outcome timepoint was supplied; cross-timepoint pooling needs review.",
            )
        )
    graph = _build_single_estimate_graph(
        evidence=evidence,
        context=context,
        estimate_id=f"estimate-{evidence.finding_id}",
        timepoint=timepoint,
        study_design=None,
        population_description=None,
        total_sample_size=None,
        treatment_description=None,
        comparator_description=None,
        evidence_section=None,
        evidence_line_ids=[],
        legacy_reported_direction=None,
        legacy_effect_size_raw=None,
    )
    return EvidenceGraphAdapterResult(
        status="requires_review"
        if any(issue.severity is AdapterIssueSeverity.BLOCKING for issue in issues)
        else "ready",
        graph=graph,
        issues=issues,
    )


def adapt_finding_row(
    finding: FindingRow, *, context: GraphAdapterContext
) -> EvidenceGraphAdapterResult:
    """Lift a legacy row while preserving—but never numerically interpreting—its labels."""

    issues: list[AdapterIssue] = [
        AdapterIssue(
            severity=AdapterIssueSeverity.BLOCKING,
            code="legacy_effect_not_quantitatively_interpretable",
            detail=(
                "FindingRow lacks a typed estimand, effect scale, and uncertainty; its raw text "
                "and categorical direction are retained only as source statements."
            ),
        )
    ]
    if finding.effect_direction is CanonicalDirection.NO_EFFECT:
        issues.append(
            AdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="legacy_no_effect_is_ambiguous",
                detail=(
                    "The label may mean non-significance, imprecision, or equivalence and is not "
                    "converted to an exact-zero estimate."
                ),
            )
        )
    if context.cohort_identity.basis is CohortIdentityBasis.LEGACY_PLACEHOLDER:
        issues.append(
            AdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="unresolved_cohort_identity",
                detail="Resolve the participant/sample identity across publications.",
            )
        )
    provenance = EffectProvenance(
        source_locator=(
            f"{finding.paper_id}#lines={','.join(finding.evidence_lines)}"
            if finding.evidence_lines
            else f"{finding.paper_id}#finding={finding.finding_id}"
        ),
        source_quote=finding.evidence_quote,
    )
    evidence = EffectEvidence(
        paper_id=finding.paper_id,
        finding_id=finding.finding_id,
        outcome=finding.outcome_name,
        contrast=context.contrast_label,
        effect_format=EffectFormat.UNSPECIFIED,
        availability=(
            EffectAvailability.INCONCLUSIVE
            if finding.effect_size_raw is not None
            else EffectAvailability.MISSING
        ),
        estimate=None,
        reported_p_value=finding.p_value,
        reported_significance=_reported_significance(finding.significant),
        equivalence_conclusion=EquivalenceConclusion.NOT_TESTED,
        moderators=finding.moderators,
        provenance=provenance,
    )
    timepoint = context.timepoint or _legacy_timepoint(finding)
    graph = _build_single_estimate_graph(
        evidence=evidence,
        context=context,
        estimate_id=f"estimate-{finding.finding_id}",
        timepoint=timepoint,
        study_design=finding.study_type,
        population_description="; ".join(
            value
            for value in (finding.species, finding.population_state)
            if value is not None
        )
        or None,
        total_sample_size=finding.sample_size,
        treatment_description=finding.intervention,
        comparator_description=finding.comparator,
        evidence_section=finding.evidence_section,
        evidence_line_ids=finding.evidence_lines or [],
        legacy_reported_direction=finding.effect_direction,
        legacy_effect_size_raw=finding.effect_size_raw,
    )
    return EvidenceGraphAdapterResult(status="requires_review", graph=graph, issues=issues)


def select_effect_evidence(
    graph: EvidenceGraph,
    *,
    outcome_name: str | None = None,
    contrast_id: str | None = None,
    require_explicit_timepoint: bool = True,
) -> GraphEffectSelection:
    """Select graph effects for the legacy paper-clustered synthesis safely.

    The current numerical engine clusters by publication.  This adapter therefore
    requires a one-to-one cohort/publication mapping among selected estimates and refuses
    graphs where that fallback would misrepresent independence.  A future cohort-aware
    hierarchical engine can remove this explicit limitation.
    """

    selected = [
        estimate
        for estimate in graph.outcome_estimates
        if (outcome_name is None or estimate.outcome_name == outcome_name)
        and (contrast_id is None or estimate.contrast_id == contrast_id)
    ]
    if not selected:
        return GraphEffectSelection(status="insufficient", reason="no_matching_estimates")
    contrast_index = {contrast.contrast_id: contrast for contrast in graph.contrasts}
    cohort_index = {cohort.cohort_id: cohort for cohort in graph.cohorts}
    if any(
        cohort_index[contrast_index[item.contrast_id].cohort_id].identity.basis
        is CohortIdentityBasis.LEGACY_PLACEHOLDER
        for item in selected
    ):
        return GraphEffectSelection(status="insufficient", reason="unresolved_cohort_identity")
    if require_explicit_timepoint and any(
        item.timepoint.kind is TimepointKind.NOT_REPORTED for item in selected
    ):
        return GraphEffectSelection(status="insufficient", reason="timepoint_not_reported")
    timepoint_signatures = {
        item.timepoint.model_dump_json(exclude_none=True) for item in selected
    }
    if len(timepoint_signatures) > 1:
        return GraphEffectSelection(status="insufficient", reason="incompatible_timepoints")

    contrast_orientation_signatures = {
        (
            contrast_index[item.contrast_id].label,
            contrast_index[item.contrast_id].positive_direction_means,
        )
        for item in selected
    }
    if len(contrast_orientation_signatures) > 1:
        return GraphEffectSelection(
            status="insufficient",
            reason="incompatible_contrast_orientations",
        )

    cohort_to_papers: dict[str, set[str]] = {}
    paper_to_cohorts: dict[str, set[str]] = {}
    for item in selected:
        cohort_id = contrast_index[item.contrast_id].cohort_id
        cohort_to_papers.setdefault(cohort_id, set()).add(item.effect.paper_id)
        paper_to_cohorts.setdefault(item.effect.paper_id, set()).add(cohort_id)
    if any(len(papers) != 1 for papers in cohort_to_papers.values()):
        return GraphEffectSelection(
            status="insufficient",
            reason="cohort_reported_by_multiple_publications_requires_cohort_aware_synthesis",
        )
    if any(len(cohorts) != 1 for cohorts in paper_to_cohorts.values()):
        return GraphEffectSelection(
            status="insufficient",
            reason="publication_reports_multiple_cohorts_requires_cohort_aware_synthesis",
        )

    warnings: list[str] = []
    if any(item.risk_of_bias.overall is RiskOfBiasJudgement.NOT_ASSESSED for item in selected):
        warnings.append("risk_of_bias_not_assessed_for_one_or_more_estimates")
    return GraphEffectSelection(
        status="ready",
        records=[item.effect for item in selected],
        estimate_ids=[item.estimate_id for item in selected],
        warnings=warnings,
    )


def graph_risk_features(
    graph: EvidenceGraph,
    *,
    outcome_name: str | None = None,
    contrast_id: str | None = None,
) -> EvidenceGraphRiskFeatures:
    """Compute deterministic pre-adjudication features without correctness labels.

    These diagnostics are inputs, not a calibrated probability of error.  Calibration
    must learn any mapping from them to claim risk on independent question/corpus units.
    """

    selected = [
        estimate
        for estimate in graph.outcome_estimates
        if (outcome_name is None or estimate.outcome_name == outcome_name)
        and (contrast_id is None or estimate.contrast_id == contrast_id)
    ]
    if not selected:
        return EvidenceGraphRiskFeatures(
            n_estimates=0,
            n_publications=0,
            n_cohorts=0,
            fraction_non_estimable=0,
            fraction_missing_source_quote=0,
            fraction_timepoint_not_reported=0,
            fraction_risk_of_bias_not_assessed=0,
            fraction_high_or_critical_risk_of_bias=0,
            fraction_unresolved_cohort_identity=0,
        )
    contrast_index = {contrast.contrast_id: contrast for contrast in graph.contrasts}
    cohort_index = {cohort.cohort_id: cohort for cohort in graph.cohorts}
    span_index = {span.span_id: span for span in graph.evidence_spans}
    selected_cohort_ids = {
        contrast_index[estimate.contrast_id].cohort_id for estimate in selected
    }
    count = len(selected)
    return EvidenceGraphRiskFeatures(
        n_estimates=count,
        n_publications=len({estimate.effect.paper_id for estimate in selected}),
        n_cohorts=len(selected_cohort_ids),
        fraction_non_estimable=sum(
            harmonize_effect(estimate.effect).status != "estimable" for estimate in selected
        )
        / count,
        fraction_missing_source_quote=sum(
            not any(span_index[span_id].quote for span_id in estimate.evidence_span_ids)
            for estimate in selected
        )
        / count,
        fraction_timepoint_not_reported=sum(
            estimate.timepoint.kind is TimepointKind.NOT_REPORTED for estimate in selected
        )
        / count,
        fraction_risk_of_bias_not_assessed=sum(
            estimate.risk_of_bias.overall is RiskOfBiasJudgement.NOT_ASSESSED
            for estimate in selected
        )
        / count,
        fraction_high_or_critical_risk_of_bias=sum(
            estimate.risk_of_bias.overall
            in {RiskOfBiasJudgement.HIGH, RiskOfBiasJudgement.CRITICAL}
            for estimate in selected
        )
        / count,
        fraction_unresolved_cohort_identity=sum(
            cohort_index[contrast_index[estimate.contrast_id].cohort_id].identity.basis
            is CohortIdentityBasis.LEGACY_PLACEHOLDER
            for estimate in selected
        )
        / count,
    )


def evidence_graph_json_schema() -> dict[str, object]:
    """Return the closed graph extraction/storage schema."""

    schema = EvidenceGraph.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:literature-multiverse:evidence-graph:v1"
    schema["title"] = "Literature Multiverse evidence graph"
    return schema
