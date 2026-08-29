"""Held-out confirmation for prespecified qualitative condition dependence.

The preparation stage consumes only a strict outcome-blind graph roster.  It freezes
an independence-component split before development effects are fitted.  The fit API
accepts only the development graph.  The confirmation API opens the complete graph
only after both plan and model hashes are fixed, reconstructs the held-out partition,
and evaluates a frozen sign-prediction and opposite-polarity replication protocol.

This module confirms a predictive literature association under a frozen protocol.  It
does not establish a causal moderator, scientific truth, or robustness to corpus shift.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from literature_multiverse.effects import (
    HarmonizedEffect,
    HarmonizedMeasure,
    harmonize_effect,
)
from literature_multiverse.evidence_graph import (
    ArmRole,
    CohortIdentityBasis,
    EvidenceGraph,
)
from literature_multiverse.independence_identity import (
    AuthorityIdentityClaimV1,
    AuthorityIdentityLedgerV1,
    IndependenceIdentityError,
    authority_identity_set_sha256,
    canonicalize_authority_identity_claims,
    canonicalize_join_only_identity,
    parse_canonical_authority_identity,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import (
    MetaAnalysisContractError,
    _moderator_level_label,
    _paule_mandel_tau_squared,
    _weighted_fit,
    aggregate_one_effect_per_cohort,
    cohort_categorical_meta_regression,
    cohort_random_effects_meta_analysis,
    prespecified_cohort_condition_analysis,
)
from literature_multiverse.models import SHA256_RE, ContractModel, normalize_doi

type ModeratorScalar = StrictBool | StrictInt | StrictFloat | StrictStr
type ConfirmationSplit = Literal["development", "confirmation"]

SPLIT_SALT = "literature-multiverse-condition-confirmation-v2"
SPLIT_ALGORITHM = "sha256-authority-token-set-first8-mod3-zero-is-confirmation-v2"
BOOTSTRAP_PROTOCOL = "paired-component-bootstrap-v1"
_EMPTY_AUTHORITY_IDENTITY_SET_SHA256 = hash_canonical(
    {"unverified_authority_identity_set": "empty"}
)


class ConditionConfirmationError(ValueError):
    """A confirmation artifact cannot be interpreted without weakening its contract."""


def _sha256(value: str, name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"condition_confirmation_invalid_sha256:{name}")
    return value


def _sorted_unique(values: Sequence[str], name: str, *, allow_empty: bool = True) -> list[str]:
    output = list(values)
    if output != sorted(set(output)) or any(not value.strip() for value in output):
        raise ValueError(f"condition_confirmation_invalid_sorted_identity_list:{name}")
    if not allow_empty and not output:
        raise ValueError(f"condition_confirmation_identity_list_empty:{name}")
    return output


def _finite(value: float, name: str, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"condition_confirmation_invalid_numeric:{name}")
    return result


def _strong_token(kind: str, value: str) -> str:
    try:
        return canonicalize_join_only_identity(kind, value)
    except IndependenceIdentityError as exc:
        raise ValueError(
            f"condition_confirmation_empty_strong_identity:{kind}"
        ) from exc


class ConditionConfirmationTargetV1(ContractModel):
    target_version: Literal["condition-confirmation-target-v1"] = (
        "condition-confirmation-target-v1"
    )
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    claim_spec_sha256: Annotated[
        str,
        Field(
            description=(
                "Exact manifest-v3 GlobalConditionDependenceTargetV1 target identity."
            )
        ),
    ]
    question_config_sha256: Annotated[
        str,
        Field(description="Exact frozen extraction/question configuration identity."),
    ]
    corpus_snapshot_sha256: Annotated[
        str,
        Field(description="Exact CompleteCorpusIdentity.membership_sha256."),
    ]
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    outcome_name: Annotated[str, Field(min_length=1)]
    claim_contrast_id: str | None = None
    contrast_label: Annotated[str, Field(min_length=1)]
    contrast_estimand: Annotated[str, Field(min_length=1)]
    positive_direction_means: Annotated[str, Field(min_length=1)]
    treatment_role: ArmRole
    comparator_role: ArmRole
    measure: HarmonizedMeasure
    unit: str | None = None
    moderator_names: Annotated[list[str], Field(min_length=1)]
    target_sha256: str

    @field_validator(
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "target_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("moderator_names")
    @classmethod
    def validate_moderators(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "moderator_names", allow_empty=False)

    @field_validator(
        "outcome_name",
        "contrast_label",
        "contrast_estimand",
        "positive_direction_means",
        "corpus_cutoff",
    )
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_confirmation_target_name_empty")
        return normalized

    @field_validator("claim_contrast_id")
    @classmethod
    def normalize_claim_contrast_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_confirmation_claim_contrast_id_empty")
        return normalized

    @model_validator(mode="after")
    def validate_target(self) -> ConditionConfirmationTargetV1:
        if self.measure is HarmonizedMeasure.MEAN_DIFFERENCE:
            if self.unit is None or not self.unit.strip():
                raise ValueError("condition_confirmation_mean_difference_requires_unit")
        elif self.unit is not None:
            raise ValueError("condition_confirmation_unitless_measure_forbids_unit")
        payload = self.model_dump(mode="json", exclude={"target_sha256"})
        if hash_canonical(payload) != self.target_sha256:
            raise ValueError("condition_confirmation_target_hash_mismatch")
        return self


def freeze_condition_confirmation_target(
    *,
    question_id: str,
    claim_spec_sha256: str,
    question_config_sha256: str,
    corpus_snapshot_sha256: str,
    corpus_cutoff: str,
    outcome_name: str,
    claim_contrast_id: str | None = None,
    contrast_label: str,
    contrast_estimand: str,
    positive_direction_means: str,
    treatment_role: ArmRole,
    comparator_role: ArmRole,
    measure: HarmonizedMeasure,
    moderator_names: Sequence[str],
    unit: str | None = None,
) -> ConditionConfirmationTargetV1:
    payload: dict[str, Any] = {
        "target_version": "condition-confirmation-target-v1",
        "question_id": question_id,
        "claim_spec_sha256": claim_spec_sha256,
        "question_config_sha256": question_config_sha256,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "corpus_cutoff": corpus_cutoff,
        "outcome_name": outcome_name,
        "claim_contrast_id": claim_contrast_id,
        "contrast_label": contrast_label,
        "contrast_estimand": contrast_estimand,
        "positive_direction_means": positive_direction_means,
        "treatment_role": treatment_role,
        "comparator_role": comparator_role,
        "measure": measure,
        "unit": unit,
        "moderator_names": sorted(moderator_names),
    }
    return ConditionConfirmationTargetV1.model_validate(
        {**payload, "target_sha256": hash_canonical(payload)}
    )


class ConditionConfirmationConfigV1(ContractModel):
    config_version: Literal["condition-confirmation-config-v1"] = (
        "condition-confirmation-config-v1"
    )
    development_familywise_alpha: Literal[0.05] = 0.05
    development_min_components_per_level: Literal[2] = 2
    confirmation_familywise_alpha: Literal[0.05] = 0.05
    confirmation_min_components_total: Literal[20] = 20
    confirmation_min_components_per_polarity: Literal[5] = 5
    assumed_within_cohort_correlation: Literal[1.0] = 1.0
    bootstrap_replicates: Literal[10000] = 10000
    bootstrap_upper_quantile: Literal[0.95] = 0.95
    bootstrap_quantile_method: Literal["higher"] = "higher"
    min_brier_improvement: Annotated[float, Field(ge=0)] = 0.0
    config_sha256: str

    @field_validator("config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "config_sha256")

    @field_validator("min_brier_improvement")
    @classmethod
    def validate_improvement(cls, value: float) -> float:
        return _finite(value, "min_brier_improvement", nonnegative=True)

    @model_validator(mode="after")
    def validate_config(self) -> ConditionConfirmationConfigV1:
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if hash_canonical(payload) != self.config_sha256:
            raise ValueError("condition_confirmation_config_hash_mismatch")
        return self


def freeze_condition_confirmation_config(
    *, min_brier_improvement: float = 0.0
) -> ConditionConfirmationConfigV1:
    payload: dict[str, Any] = {
        "config_version": "condition-confirmation-config-v1",
        "development_familywise_alpha": 0.05,
        "development_min_components_per_level": 2,
        "confirmation_familywise_alpha": 0.05,
        "confirmation_min_components_total": 20,
        "confirmation_min_components_per_polarity": 5,
        "assumed_within_cohort_correlation": 1.0,
        "bootstrap_replicates": 10000,
        "bootstrap_upper_quantile": 0.95,
        "bootstrap_quantile_method": "higher",
        "min_brier_improvement": float(min_brier_improvement),
    }
    return ConditionConfirmationConfigV1.model_validate(
        {**payload, "config_sha256": hash_canonical(payload)}
    )


class RosterPublicationV1(ContractModel):
    publication_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    doi: str | None = None
    pmid: str | None = None
    doc_id: str | None = None

    @field_validator("doi")
    @classmethod
    def normalize_doi_value(cls, value: str | None) -> str | None:
        return None if value is None else normalize_doi(value)

    @field_validator("pmid")
    @classmethod
    def validate_pmid(cls, value: str | None) -> str | None:
        if value is not None and not value.isdigit():
            raise ValueError("condition_confirmation_roster_pmid_invalid")
        return value

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("condition_confirmation_roster_doc_id_empty")
        return value


class RosterStudyV1(ContractModel):
    study_id: Annotated[str, Field(min_length=1)]
    publication_ids: Annotated[list[str], Field(min_length=1)]
    registration_ids: list[str] = Field(default_factory=list)

    @field_validator("publication_ids", "registration_ids")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _sorted_unique(
            value,
            info.field_name,
            allow_empty=info.field_name == "registration_ids",
        )


class RosterCohortV1(ContractModel):
    cohort_id: Annotated[str, Field(min_length=1)]
    study_id: Annotated[str, Field(min_length=1)]
    identity_basis: CohortIdentityBasis
    registry_ids: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)

    @field_validator("registry_ids", "dataset_ids")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _sorted_unique(value, info.field_name)


class RosterArmV1(ContractModel):
    arm_id: Annotated[str, Field(min_length=1)]
    cohort_id: Annotated[str, Field(min_length=1)]
    role: ArmRole


class RosterContrastV1(ContractModel):
    contrast_id: Annotated[str, Field(min_length=1)]
    cohort_id: Annotated[str, Field(min_length=1)]
    treatment_arm_id: Annotated[str, Field(min_length=1)]
    comparator_arm_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    estimand: Annotated[str, Field(min_length=1)] | None
    positive_direction_means: Annotated[str, Field(min_length=1)]

    @field_validator("label", "positive_direction_means")
    @classmethod
    def normalize_semantics(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_confirmation_contrast_semantics_empty")
        return normalized

    @field_validator("estimand")
    @classmethod
    def normalize_optional_estimand(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_confirmation_contrast_estimand_empty")
        return normalized


class RosterEstimateV1(ContractModel):
    estimate_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]
    study_id: Annotated[str, Field(min_length=1)]
    cohort_id: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)]
    target_scope: StrictBool
    moderator_values: dict[str, ModeratorScalar | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scope(self) -> RosterEstimateV1:
        if not self.target_scope and self.moderator_values:
            raise ValueError("condition_confirmation_out_of_scope_moderators_forbidden")
        if list(self.moderator_values) != sorted(self.moderator_values):
            raise ValueError("condition_confirmation_roster_moderators_not_sorted")
        return self


class RosterSpanV1(ContractModel):
    span_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]


class LabelFreeGraphRosterV1(ContractModel):
    """Outcome-blind graph topology and prespecified predictor values.

    The roster deliberately has no availability, estimability, effect format, point
    estimate, uncertainty, direction, or significance field. Those properties are
    first inspected in the development graph after the component split is frozen.
    """

    roster_version: Literal["label-free-condition-roster-v1"] = (
        "label-free-condition-roster-v1"
    )
    outcome_values_absent: Literal[True] = True
    source_graph_sha256: str
    publications: Annotated[list[RosterPublicationV1], Field(min_length=1)]
    studies: list[RosterStudyV1]
    cohorts: list[RosterCohortV1]
    arms: list[RosterArmV1]
    contrasts: list[RosterContrastV1]
    estimates: list[RosterEstimateV1]
    spans: list[RosterSpanV1]
    roster_sha256: str

    @field_validator("source_graph_sha256", "roster_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_roster(self) -> LabelFreeGraphRosterV1:
        indexes: dict[str, set[str]] = {}
        for field_name, id_name in (
            ("publications", "publication_id"),
            ("studies", "study_id"),
            ("cohorts", "cohort_id"),
            ("arms", "arm_id"),
            ("contrasts", "contrast_id"),
            ("estimates", "estimate_id"),
            ("spans", "span_id"),
        ):
            rows = getattr(self, field_name)
            ids = [str(getattr(row, id_name)) for row in rows]
            if ids != sorted(set(ids)):
                raise ValueError(f"condition_confirmation_roster_not_sorted:{field_name}")
            indexes[field_name] = set(ids)
        publications = indexes["publications"]
        studies = indexes["studies"]
        cohorts = indexes["cohorts"]
        arms = indexes["arms"]
        contrasts = indexes["contrasts"]
        publication_to_studies: dict[str, set[str]] = defaultdict(set)
        for study in self.studies:
            if not set(study.publication_ids) <= publications:
                raise ValueError("condition_confirmation_roster_study_publication_unknown")
            for publication_id in study.publication_ids:
                publication_to_studies[publication_id].add(study.study_id)
        cohort_to_study: dict[str, str] = {}
        for cohort in self.cohorts:
            if cohort.study_id not in studies:
                raise ValueError("condition_confirmation_roster_cohort_study_unknown")
            cohort_to_study[cohort.cohort_id] = cohort.study_id
        arm_to_cohort: dict[str, str] = {}
        for arm in self.arms:
            if arm.cohort_id not in cohorts:
                raise ValueError("condition_confirmation_roster_arm_cohort_unknown")
            arm_to_cohort[arm.arm_id] = arm.cohort_id
        contrast_to_cohort: dict[str, str] = {}
        for contrast in self.contrasts:
            if (
                contrast.cohort_id not in cohorts
                or contrast.treatment_arm_id not in arms
                or contrast.comparator_arm_id not in arms
            ):
                raise ValueError("condition_confirmation_roster_contrast_reference_unknown")
            if (
                arm_to_cohort[contrast.treatment_arm_id] != contrast.cohort_id
                or arm_to_cohort[contrast.comparator_arm_id] != contrast.cohort_id
            ):
                raise ValueError("condition_confirmation_roster_contrast_cohort_mismatch")
            contrast_to_cohort[contrast.contrast_id] = contrast.cohort_id
        for estimate in self.estimates:
            if (
                estimate.publication_id not in publications
                or estimate.study_id not in studies
                or estimate.cohort_id not in cohorts
                or estimate.contrast_id not in contrasts
            ):
                raise ValueError("condition_confirmation_roster_estimate_reference_unknown")
            if cohort_to_study[estimate.cohort_id] != estimate.study_id:
                raise ValueError("condition_confirmation_roster_estimate_study_mismatch")
            if contrast_to_cohort[estimate.contrast_id] != estimate.cohort_id:
                raise ValueError("condition_confirmation_roster_estimate_contrast_mismatch")
            if estimate.study_id not in publication_to_studies[estimate.publication_id]:
                raise ValueError("condition_confirmation_roster_estimate_publication_mismatch")
        for span in self.spans:
            if span.publication_id not in publications:
                raise ValueError("condition_confirmation_roster_span_publication_unknown")
        payload = self.model_dump(mode="json", exclude={"roster_sha256"})
        if hash_canonical(payload) != self.roster_sha256:
            raise ValueError("condition_confirmation_roster_hash_mismatch")
        return self


def freeze_label_free_graph_roster(
    *,
    source_graph_sha256: str,
    publications: Sequence[RosterPublicationV1],
    studies: Sequence[RosterStudyV1],
    cohorts: Sequence[RosterCohortV1],
    arms: Sequence[RosterArmV1],
    contrasts: Sequence[RosterContrastV1],
    estimates: Sequence[RosterEstimateV1],
    spans: Sequence[RosterSpanV1],
) -> LabelFreeGraphRosterV1:
    payload: dict[str, Any] = {
        "roster_version": "label-free-condition-roster-v1",
        "outcome_values_absent": True,
        "source_graph_sha256": source_graph_sha256,
        "publications": sorted(publications, key=lambda row: row.publication_id),
        "studies": sorted(studies, key=lambda row: row.study_id),
        "cohorts": sorted(cohorts, key=lambda row: row.cohort_id),
        "arms": sorted(arms, key=lambda row: row.arm_id),
        "contrasts": sorted(contrasts, key=lambda row: row.contrast_id),
        "estimates": sorted(estimates, key=lambda row: row.estimate_id),
        "spans": sorted(spans, key=lambda row: row.span_id),
    }
    return LabelFreeGraphRosterV1.model_validate(
        {**payload, "roster_sha256": hash_canonical(payload)}
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        root = self.parent[value]
        while root != self.parent[root]:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


class ConditionComponentAssignmentV1(ContractModel):
    component_id: str
    split_identity_sha256: str
    assignment_sha256: str
    split: ConfirmationSplit
    publication_ids: list[str]
    paper_ids: list[str]
    study_ids: list[str]
    cohort_ids: list[str]
    strong_identity_tokens: list[str]
    split_identity_tokens: list[str]
    authority_identity_conflict_sha256s: list[str]
    arm_ids: list[str]
    contrast_ids: list[str]
    estimate_ids: list[str]
    span_ids: list[str]

    @field_validator("component_id", "split_identity_sha256", "assignment_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator(
        "publication_ids",
        "paper_ids",
        "study_ids",
        "cohort_ids",
        "strong_identity_tokens",
        "split_identity_tokens",
        "authority_identity_conflict_sha256s",
        "arm_ids",
        "contrast_ids",
        "estimate_ids",
        "span_ids",
    )
    @classmethod
    def validate_identity_lists(cls, value: list[str], info: Any) -> list[str]:
        values = _sorted_unique(value, info.field_name)
        if info.field_name == "authority_identity_conflict_sha256s":
            return [_sha256(item, info.field_name) for item in values]
        return values

    @model_validator(mode="after")
    def validate_split_identity(self) -> ConditionComponentAssignmentV1:
        try:
            expected = (
                authority_identity_set_sha256(self.split_identity_tokens)
                if self.split_identity_tokens
                else _EMPTY_AUTHORITY_IDENTITY_SET_SHA256
            )
        except IndependenceIdentityError as exc:
            raise ValueError(
                "condition_confirmation_split_identity_token_not_authority_scoped"
            ) from exc
        if self.split_identity_sha256 != expected:
            raise ValueError("condition_confirmation_split_identity_hash_mismatch")
        return self


class ConditionGraphPartitionV1(ContractModel):
    split: Literal["full", "development", "confirmation"]
    component_ids: list[str]
    publication_ids: list[str]
    study_ids: list[str]
    cohort_ids: list[str]
    arm_ids: list[str]
    contrast_ids: list[str]
    estimate_ids: list[str]
    span_ids: list[str]
    partition_sha256: str

    @field_validator("partition_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "partition_sha256")

    @field_validator(
        "component_ids",
        "publication_ids",
        "study_ids",
        "cohort_ids",
        "arm_ids",
        "contrast_ids",
        "estimate_ids",
        "span_ids",
    )
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _sorted_unique(value, info.field_name)

    @model_validator(mode="after")
    def validate_partition(self) -> ConditionGraphPartitionV1:
        payload = self.model_dump(mode="json", exclude={"partition_sha256"})
        if hash_canonical(payload) != self.partition_sha256:
            raise ValueError("condition_confirmation_partition_hash_mismatch")
        return self


def _freeze_partition(
    split: Literal["full", "development", "confirmation"],
    components: Sequence[ConditionComponentAssignmentV1],
) -> ConditionGraphPartitionV1:
    payload: dict[str, Any] = {
        "split": split,
        "component_ids": sorted(row.component_id for row in components),
        "publication_ids": sorted(
            {value for row in components for value in row.publication_ids}
        ),
        "study_ids": sorted({value for row in components for value in row.study_ids}),
        "cohort_ids": sorted({value for row in components for value in row.cohort_ids}),
        "arm_ids": sorted({value for row in components for value in row.arm_ids}),
        "contrast_ids": sorted({value for row in components for value in row.contrast_ids}),
        "estimate_ids": sorted({value for row in components for value in row.estimate_ids}),
        "span_ids": sorted({value for row in components for value in row.span_ids}),
    }
    return ConditionGraphPartitionV1.model_validate(
        {**payload, "partition_sha256": hash_canonical(payload)}
    )


def _authority_identity_ledger(
    roster: LabelFreeGraphRosterV1,
) -> AuthorityIdentityLedgerV1:
    claims: list[AuthorityIdentityClaimV1] = []
    for publication in roster.publications:
        node_id = f"publication:{publication.publication_id}"
        if publication.doi is not None:
            claims.append(
                AuthorityIdentityClaimV1(
                    node_id=node_id,
                    kind="doi",
                    raw_value=publication.doi,
                )
            )
        if publication.pmid is not None:
            claims.append(
                AuthorityIdentityClaimV1(
                    node_id=node_id,
                    kind="pmid",
                    raw_value=publication.pmid,
                )
            )
    for study in roster.studies:
        node_id = f"study:{study.study_id}"
        claims.append(
            AuthorityIdentityClaimV1(
                node_id=node_id,
                kind="globally_scoped_study_id",
                raw_value=study.study_id,
            )
        )
        claims.extend(
            AuthorityIdentityClaimV1(
                node_id=node_id,
                kind="registration_id",
                raw_value=value,
            )
            for value in study.registration_ids
        )
    for cohort in roster.cohorts:
        node_id = f"cohort:{cohort.cohort_id}"
        claims.append(
            AuthorityIdentityClaimV1(
                node_id=node_id,
                kind="globally_scoped_cohort_id",
                raw_value=cohort.cohort_id,
            )
        )
        claims.extend(
            AuthorityIdentityClaimV1(
                node_id=node_id,
                kind="registry_id",
                raw_value=value,
            )
            for value in cohort.registry_ids
        )
        claims.extend(
            AuthorityIdentityClaimV1(
                node_id=node_id,
                kind="dataset_id",
                raw_value=value,
            )
            for value in cohort.dataset_ids
        )
    return canonicalize_authority_identity_claims(claims)


def _authority_tokens_by_node(
    ledger: AuthorityIdentityLedgerV1,
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for binding in ledger.bindings:
        for node_id in binding.node_ids:
            output[node_id].add(binding.token)
    return output


def _authority_linkage_tokens_by_publication(
    *,
    roster: LabelFreeGraphRosterV1,
    ledger: AuthorityIdentityLedgerV1,
) -> dict[str, set[str]]:
    tokens_by_node = _authority_tokens_by_node(ledger)
    cohorts_by_study: dict[str, list[RosterCohortV1]] = defaultdict(list)
    for cohort in roster.cohorts:
        cohorts_by_study[cohort.study_id].append(cohort)
    output: dict[str, set[str]] = defaultdict(set)
    linkage_kinds = {
        "trial_registry",
        "dataset",
        "globally_scoped_study",
        "globally_scoped_cohort",
    }
    for study in roster.studies:
        tokens = set(tokens_by_node[f"study:{study.study_id}"])
        for cohort in cohorts_by_study[study.study_id]:
            tokens.update(tokens_by_node[f"cohort:{cohort.cohort_id}"])
        linkage_tokens = {
            token
            for token in tokens
            if parse_canonical_authority_identity(token).kind in linkage_kinds
        }
        for publication_id in study.publication_ids:
            output[publication_id].update(linkage_tokens)
    return output


def derive_condition_components(
    roster: LabelFreeGraphRosterV1,
    *,
    question_id: str,
) -> list[ConditionComponentAssignmentV1]:
    """Compute the immutable graph-connected development/confirmation assignment."""

    roster = LabelFreeGraphRosterV1.model_validate(roster.model_dump(mode="json"))
    authority_ledger = _authority_identity_ledger(roster)
    union = _UnionFind()
    token_members: dict[str, set[str]] = defaultdict(set)
    paper_by_publication: dict[str, str] = {}
    strong_by_node: dict[str, set[str]] = defaultdict(set)
    authority_by_node: dict[str, set[str]] = defaultdict(set)
    authority_conflicts_by_node: dict[str, set[str]] = defaultdict(set)

    for publication in roster.publications:
        publication_node = f"publication:{publication.publication_id}"
        paper_node = f"paper:{publication.paper_id}"
        union.union(publication_node, paper_node)
        paper_by_publication[publication.publication_id] = publication.paper_id
        for token in (
            None if publication.doi is None else _strong_token("doi", publication.doi),
            None if publication.pmid is None else _strong_token("pmid", publication.pmid),
            None if publication.doc_id is None else _strong_token("doc_id", publication.doc_id),
        ):
            if token is not None:
                union.union(publication_node, f"strong:{token}")
                strong_by_node[publication_node].add(token)
                token_members[token].add(publication_node)
    for study in roster.studies:
        study_node = f"study:{study.study_id}"
        union.add(study_node)
        for publication_id in study.publication_ids:
            union.union(study_node, f"publication:{publication_id}")
        for raw in study.registration_ids:
            token = _strong_token("registry", raw)
            union.union(study_node, f"strong:{token}")
            strong_by_node[study_node].add(token)
            token_members[token].add(study_node)
    for cohort in roster.cohorts:
        cohort_node = f"cohort:{cohort.cohort_id}"
        union.union(cohort_node, f"study:{cohort.study_id}")
        for kind, values in (
            ("registry", cohort.registry_ids),
            ("dataset", cohort.dataset_ids),
        ):
            for raw in values:
                token = _strong_token(kind, raw)
                union.union(cohort_node, f"strong:{token}")
                strong_by_node[cohort_node].add(token)
                token_members[token].add(cohort_node)

    for binding in authority_ledger.bindings:
        authority_node = f"authority:{binding.token_sha256}"
        for node_id in binding.node_ids:
            union.union(node_id, authority_node)
            authority_by_node[node_id].add(binding.token)
    for conflict in authority_ledger.conflicts:
        conflict_node = f"authority-conflict:{conflict.conflict_sha256}"
        for node_id in conflict.node_ids:
            union.union(node_id, conflict_node)
            authority_conflicts_by_node[node_id].add(conflict.conflict_sha256)

    graph_nodes = [
        *(f"publication:{row.publication_id}" for row in roster.publications),
        *(f"study:{row.study_id}" for row in roster.studies),
        *(f"cohort:{row.cohort_id}" for row in roster.cohorts),
    ]
    grouped: dict[str, dict[str, set[str]]] = {}
    for node in graph_nodes:
        root = union.find(node)
        group = grouped.setdefault(
            root,
            {
                "publication_ids": set(),
                "paper_ids": set(),
                "study_ids": set(),
                "cohort_ids": set(),
                "strong_identity_tokens": set(),
                "authority_identity_tokens": set(),
                "authority_identity_conflict_sha256s": set(),
            },
        )
        kind, value = node.split(":", 1)
        group[f"{kind}_ids"].add(value)
        if kind == "publication":
            group["paper_ids"].add(paper_by_publication[value])
        group["strong_identity_tokens"].update(strong_by_node[node])
        group["authority_identity_tokens"].update(authority_by_node[node])
        group["authority_identity_conflict_sha256s"].update(
            authority_conflicts_by_node[node]
        )
    for token, nodes in token_members.items():
        if nodes:
            grouped[union.find(next(iter(nodes)))]["strong_identity_tokens"].add(token)

    cohort_component_root = {
        cohort.cohort_id: union.find(f"cohort:{cohort.cohort_id}")
        for cohort in roster.cohorts
    }
    publication_component_root = {
        publication.publication_id: union.find(f"publication:{publication.publication_id}")
        for publication in roster.publications
    }
    arms_by_root: dict[str, set[str]] = defaultdict(set)
    contrasts_by_root: dict[str, set[str]] = defaultdict(set)
    estimates_by_root: dict[str, set[str]] = defaultdict(set)
    spans_by_root: dict[str, set[str]] = defaultdict(set)
    for arm in roster.arms:
        arms_by_root[cohort_component_root[arm.cohort_id]].add(arm.arm_id)
    for contrast in roster.contrasts:
        contrasts_by_root[cohort_component_root[contrast.cohort_id]].add(
            contrast.contrast_id
        )
    for estimate in roster.estimates:
        estimates_by_root[cohort_component_root[estimate.cohort_id]].add(
            estimate.estimate_id
        )
    for span in roster.spans:
        spans_by_root[publication_component_root[span.publication_id]].add(span.span_id)

    output: list[ConditionComponentAssignmentV1] = []
    for root, group in grouped.items():
        identity_payload = {
            name: sorted(group[name])
            for name in (
                "publication_ids",
                "paper_ids",
                "study_ids",
                "cohort_ids",
                "strong_identity_tokens",
            )
        }
        component_id = hash_canonical(identity_payload)
        split_identity_tokens = sorted(group["authority_identity_tokens"])
        split_identity_sha256 = (
            authority_identity_set_sha256(split_identity_tokens)
            if split_identity_tokens
            else _EMPTY_AUTHORITY_IDENTITY_SET_SHA256
        )
        assignment_raw = (
            SPLIT_SALT.encode("utf-8")
            + b"\0"
            + question_id.encode("utf-8")
            + b"\0"
            + split_identity_sha256.encode("ascii")
        )
        assignment_sha256 = hashlib.sha256(assignment_raw).hexdigest()
        split: ConfirmationSplit = (
            "confirmation"
            if int.from_bytes(bytes.fromhex(assignment_sha256)[:8], "big") % 3 == 0
            else "development"
        )
        output.append(
            ConditionComponentAssignmentV1(
                component_id=component_id,
                split_identity_sha256=split_identity_sha256,
                assignment_sha256=assignment_sha256,
                split=split,
                **identity_payload,
                split_identity_tokens=split_identity_tokens,
                authority_identity_conflict_sha256s=sorted(
                    group["authority_identity_conflict_sha256s"]
                ),
                arm_ids=sorted(arms_by_root[root]),
                contrast_ids=sorted(contrasts_by_root[root]),
                estimate_ids=sorted(estimates_by_root[root]),
                span_ids=sorted(spans_by_root[root]),
            )
        )
    return sorted(output, key=lambda row: row.component_id)


def _component_by_cohort(
    assignments: Sequence[ConditionComponentAssignmentV1],
) -> dict[str, ConditionComponentAssignmentV1]:
    return {
        cohort_id: assignment
        for assignment in assignments
        for cohort_id in assignment.cohort_ids
    }


def _target_component_ids(
    *,
    roster: LabelFreeGraphRosterV1,
    assignments: Sequence[ConditionComponentAssignmentV1],
    split: ConfirmationSplit | None = None,
) -> set[str]:
    component_by_cohort = _component_by_cohort(assignments)
    return {
        component_by_cohort[row.cohort_id].component_id
        for row in roster.estimates
        if row.target_scope
        and (split is None or component_by_cohort[row.cohort_id].split == split)
    }


def _roster_cohort_levels(
    roster: LabelFreeGraphRosterV1,
    target: ConditionConfirmationTargetV1,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    reasons: list[str] = []
    target_rows = [row for row in roster.estimates if row.target_scope]
    for row in target_rows:
        if set(row.moderator_values) != set(target.moderator_names):
            reasons.append(f"moderator_schema_mismatch:{row.estimate_id}")
            continue
        for moderator in target.moderator_names:
            value = row.moderator_values[moderator]
            if value is None:
                reasons.append(f"moderator_missing:{row.estimate_id}:{moderator}")
            else:
                values[row.cohort_id][moderator].add(_moderator_level_label(value))
    result: dict[str, dict[str, str]] = {}
    for cohort_id, by_moderator in sorted(values.items()):
        result[cohort_id] = {}
        for moderator in target.moderator_names:
            observed = by_moderator.get(moderator, set())
            if len(observed) != 1:
                reasons.append(f"moderator_conflict:{cohort_id}:{moderator}")
            else:
                result[cohort_id][moderator] = next(iter(observed))
    return result, sorted(set(reasons))


def _roster_component_levels(
    *,
    roster: LabelFreeGraphRosterV1,
    target: ConditionConfirmationTargetV1,
    assignments: Sequence[ConditionComponentAssignmentV1],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Project moderators to independent components without pseudo-replication."""

    cohort_levels, reasons = _roster_cohort_levels(roster, target)
    component_by_cohort = _component_by_cohort(assignments)
    observed: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for cohort_id, levels in cohort_levels.items():
        component_id = component_by_cohort[cohort_id].component_id
        for moderator, level in levels.items():
            observed[component_id][moderator].add(level)
    component_levels: dict[str, dict[str, str]] = {}
    for component_id, by_moderator in sorted(observed.items()):
        component_levels[component_id] = {}
        for moderator in target.moderator_names:
            levels = by_moderator.get(moderator, set())
            if len(levels) > 1:
                reasons.append(
                    "moderator_conflict_within_independence_component:"
                    f"{component_id}:{moderator}:{sorted(levels)}"
                )
            elif len(levels) == 1:
                component_levels[component_id][moderator] = next(iter(levels))
    return component_levels, sorted(set(reasons))


def _plan_reasons(
    *,
    roster: LabelFreeGraphRosterV1,
    target: ConditionConfirmationTargetV1,
    config: ConditionConfirmationConfigV1,
    assignments: Sequence[ConditionComponentAssignmentV1],
) -> list[str]:
    reasons: list[str] = []
    target_rows = [row for row in roster.estimates if row.target_scope]
    if not target_rows:
        reasons.append("target_scope_empty")
    contrast_by_id = {row.contrast_id: row for row in roster.contrasts}
    arm_by_id = {row.arm_id: row for row in roster.arms}
    target_contrast_ids = {row.contrast_id for row in target_rows}
    target_contrast_labels = {
        contrast_by_id[contrast_id].label for contrast_id in target_contrast_ids
    }
    if target_contrast_labels != {target.contrast_label}:
        reasons.append(
            "target_contrast_label_mapping_not_exactly_one:"
            f"expected={target.contrast_label}:observed={sorted(target_contrast_labels)}"
        )
    target_contrast_estimands = {
        contrast_by_id[contrast_id].estimand for contrast_id in target_contrast_ids
    }
    if target_contrast_estimands != {target.contrast_estimand}:
        reasons.append(
            "target_contrast_estimand_mapping_not_exactly_one:"
            f"expected={target.contrast_estimand}:"
            f"observed={sorted(str(value) for value in target_contrast_estimands)}"
        )
    target_positive_direction_semantics = {
        contrast_by_id[contrast_id].positive_direction_means
        for contrast_id in target_contrast_ids
    }
    if target_positive_direction_semantics != {target.positive_direction_means}:
        reasons.append(
            "target_positive_direction_mapping_not_exactly_one:"
            f"expected={target.positive_direction_means}:"
            f"observed={sorted(target_positive_direction_semantics)}"
        )
    treatment_roles = {
        arm_by_id[contrast_by_id[contrast_id].treatment_arm_id].role
        for contrast_id in target_contrast_ids
    }
    comparator_roles = {
        arm_by_id[contrast_by_id[contrast_id].comparator_arm_id].role
        for contrast_id in target_contrast_ids
    }
    if treatment_roles != {target.treatment_role}:
        reasons.append(
            "target_treatment_role_mapping_not_exactly_one:"
            f"expected={target.treatment_role.value}:"
            f"observed={sorted(role.value for role in treatment_roles)}"
        )
    if comparator_roles != {target.comparator_role}:
        reasons.append(
            "target_comparator_role_mapping_not_exactly_one:"
            f"expected={target.comparator_role.value}:"
            f"observed={sorted(role.value for role in comparator_roles)}"
        )
    if (
        target.claim_contrast_id is not None
        and target_contrast_ids != {target.claim_contrast_id}
    ):
        reasons.append(
            "target_claim_contrast_id_mapping_mismatch:"
            f"expected={target.claim_contrast_id}:observed={sorted(target_contrast_ids)}"
        )
    target_cohorts = {row.cohort_id for row in target_rows}
    cohort_by_id = {row.cohort_id: row for row in roster.cohorts}
    legacy = sorted(
        cohort_id
        for cohort_id in target_cohorts
        if cohort_by_id[cohort_id].identity_basis is CohortIdentityBasis.LEGACY_PLACEHOLDER
    )
    if legacy:
        reasons.append(f"estimable_legacy_placeholder_cohorts:{legacy}")
    development_target_components = _target_component_ids(
        roster=roster,
        assignments=assignments,
        split="development",
    )
    confirmation_target_components = _target_component_ids(
        roster=roster,
        assignments=assignments,
        split="confirmation",
    )
    authority_ledger = _authority_identity_ledger(roster)
    authority_tokens_by_node = _authority_tokens_by_node(authority_ledger)
    linkage_tokens_by_publication = _authority_linkage_tokens_by_publication(
        roster=roster,
        ledger=authority_ledger,
    )
    assignment_by_component = {row.component_id: row for row in assignments}
    component_by_cohort = _component_by_cohort(assignments)
    target_publications_by_component: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        target_publications_by_component[
            component_by_cohort[row.cohort_id].component_id
        ].add(row.publication_id)
    for component_id in sorted(
        development_target_components | confirmation_target_components
    ):
        assignment = assignment_by_component[component_id]
        if not assignment.split_identity_tokens:
            reasons.append(
                "target_component_lacks_authority_scoped_split_identity:"
                f"{component_id}"
            )
        if assignment.authority_identity_conflict_sha256s:
            reasons.append(
                "target_component_authority_identity_conflict:"
                f"{component_id}:"
                f"{assignment.authority_identity_conflict_sha256s}"
            )
        target_publication_ids = sorted(
            target_publications_by_component[component_id]
        )
        for publication_id in target_publication_ids:
            publication_tokens = authority_tokens_by_node[
                f"publication:{publication_id}"
            ]
            if not any(
                parse_canonical_authority_identity(token).kind in {"doi", "pmid"}
                for token in publication_tokens
            ):
                reasons.append(
                    "target_component_publication_lacks_authority_identity:"
                    f"{component_id}:{publication_id}"
                )
        linkage_sets = [
            linkage_tokens_by_publication[publication_id]
            for publication_id in target_publication_ids
        ]
        shared_linkage_tokens = (
            set.intersection(*(set(values) for values in linkage_sets))
            if linkage_sets
            else set()
        )
        if not shared_linkage_tokens:
            reasons.append(
                "target_component_lacks_all_report_authority_linkage:"
                f"{component_id}"
            )
    if not development_target_components:
        reasons.append("development_target_component_split_empty")
    if not confirmation_target_components:
        reasons.append("confirmation_target_component_split_empty")
    if len(confirmation_target_components) < config.confirmation_min_components_total:
        reasons.append(
            "confirmation_target_components_below_minimum:"
            f"{len(confirmation_target_components)}<"
            f"{config.confirmation_min_components_total}"
        )
    levels, level_reasons = _roster_component_levels(
        roster=roster,
        target=target,
        assignments=assignments,
    )
    reasons.extend(level_reasons)
    for moderator in target.moderator_names:
        support: Counter[str] = Counter()
        for component_id, moderator_values in levels.items():
            assignment = assignment_by_component[component_id]
            if assignment.split == "development" and moderator in moderator_values:
                support[moderator_values[moderator]] += 1
        if len(support) < 2:
            reasons.append(f"development_moderator_has_fewer_than_two_levels:{moderator}")
        for level, count in sorted(support.items()):
            if count < config.development_min_components_per_level:
                reasons.append(
                    f"development_component_level_sparse:{moderator}:{level}:"
                    f"{count}<{config.development_min_components_per_level}"
                )
    return sorted(set(reasons))


class ConditionConfirmationMaterializationReceiptV1(ContractModel):
    """Content-silent custody receipt for deterministic private graph partitioning."""

    receipt_version: Literal[
        "condition-confirmation-materialization-receipt-v1"
    ] = "condition-confirmation-materialization-receipt-v1"
    full_graph_outcomes_opened_by_custodian: Literal[True] = True
    effect_outcome_uncertainty_values_embedded: Literal[False] = False
    access_semantics: Literal[
        "independent custodian opened the full graph after target freeze; receipt "
        "contains only identities, counts, hashes, and access declarations"
    ] = (
        "independent custodian opened the full graph after target freeze; receipt "
        "contains only identities, counts, hashes, and access declarations"
    )
    question_id: Annotated[str, Field(min_length=1)]
    target_sha256: str
    claim_spec_sha256: str
    question_config_sha256: str
    corpus_snapshot_sha256: str
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    full_graph_sha256: str
    roster_sha256: str
    component_assignments_sha256: str
    full_partition_sha256: str
    development_partition_sha256: str
    confirmation_partition_sha256: str
    development_graph_sha256: str
    confirmation_graph_sha256: str
    full_component_count: Annotated[int, Field(ge=1)]
    development_component_count: Annotated[int, Field(ge=0)]
    confirmation_component_count: Annotated[int, Field(ge=0)]
    full_publication_count: Annotated[int, Field(ge=1)]
    full_study_count: Annotated[int, Field(ge=0)]
    full_cohort_count: Annotated[int, Field(ge=0)]
    full_estimate_count: Annotated[int, Field(ge=0)]
    development_estimate_count: Annotated[int, Field(ge=0)]
    confirmation_estimate_count: Annotated[int, Field(ge=0)]
    split_algorithm: Literal[
        "sha256-authority-token-set-first8-mod3-zero-is-confirmation-v2"
    ] = SPLIT_ALGORITHM
    split_salt: Literal["literature-multiverse-condition-confirmation-v2"] = (
        SPLIT_SALT
    )
    receipt_sha256: str

    @field_validator(
        "target_sha256",
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "full_graph_sha256",
        "roster_sha256",
        "component_assignments_sha256",
        "full_partition_sha256",
        "development_partition_sha256",
        "confirmation_partition_sha256",
        "development_graph_sha256",
        "confirmation_graph_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> ConditionConfirmationMaterializationReceiptV1:
        if (
            self.development_component_count + self.confirmation_component_count
            != self.full_component_count
        ):
            raise ValueError(
                "condition_confirmation_materialization_component_count_mismatch"
            )
        if (
            self.development_estimate_count + self.confirmation_estimate_count
            != self.full_estimate_count
        ):
            raise ValueError(
                "condition_confirmation_materialization_estimate_count_mismatch"
            )
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("condition_confirmation_materialization_receipt_hash_mismatch")
        return self


def _freeze_materialization_receipt(
    *,
    target: ConditionConfirmationTargetV1,
    roster: LabelFreeGraphRosterV1,
    assignments: Sequence[ConditionComponentAssignmentV1],
    full_partition: ConditionGraphPartitionV1,
    development_partition: ConditionGraphPartitionV1,
    confirmation_partition: ConditionGraphPartitionV1,
    development_graph_sha256: str,
    confirmation_graph_sha256: str,
) -> ConditionConfirmationMaterializationReceiptV1:
    payload: dict[str, Any] = {
        "receipt_version": "condition-confirmation-materialization-receipt-v1",
        "full_graph_outcomes_opened_by_custodian": True,
        "effect_outcome_uncertainty_values_embedded": False,
        "access_semantics": (
            "independent custodian opened the full graph after target freeze; receipt "
            "contains only identities, counts, hashes, and access declarations"
        ),
        "question_id": target.question_id,
        "target_sha256": target.target_sha256,
        "claim_spec_sha256": target.claim_spec_sha256,
        "question_config_sha256": target.question_config_sha256,
        "corpus_snapshot_sha256": target.corpus_snapshot_sha256,
        "corpus_cutoff": target.corpus_cutoff,
        "full_graph_sha256": roster.source_graph_sha256,
        "roster_sha256": roster.roster_sha256,
        "component_assignments_sha256": hash_canonical(assignments),
        "full_partition_sha256": full_partition.partition_sha256,
        "development_partition_sha256": development_partition.partition_sha256,
        "confirmation_partition_sha256": confirmation_partition.partition_sha256,
        "development_graph_sha256": development_graph_sha256,
        "confirmation_graph_sha256": confirmation_graph_sha256,
        "full_component_count": len(assignments),
        "development_component_count": len(development_partition.component_ids),
        "confirmation_component_count": len(confirmation_partition.component_ids),
        "full_publication_count": len(roster.publications),
        "full_study_count": len(roster.studies),
        "full_cohort_count": len(roster.cohorts),
        "full_estimate_count": len(roster.estimates),
        "development_estimate_count": len(development_partition.estimate_ids),
        "confirmation_estimate_count": len(confirmation_partition.estimate_ids),
        "split_algorithm": SPLIT_ALGORITHM,
        "split_salt": SPLIT_SALT,
    }
    return ConditionConfirmationMaterializationReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class ConditionConfirmationPlanV1(ContractModel):
    plan_version: Literal["condition-confirmation-plan-v1"] = (
        "condition-confirmation-plan-v1"
    )
    freeze_state: Literal["confirmation_outcomes_unopened"] = (
        "confirmation_outcomes_unopened"
    )
    target: ConditionConfirmationTargetV1
    target_sha256: str
    claim_spec_sha256: str
    question_config_sha256: str
    corpus_snapshot_sha256: str
    corpus_cutoff: str
    claim_contrast_id: str | None
    config: ConditionConfirmationConfigV1
    config_sha256: str
    pipeline_sha256: str
    external_freeze_anchor: Annotated[str, Field(min_length=1)]
    roster: LabelFreeGraphRosterV1
    roster_sha256: str
    materialization_receipt: ConditionConfirmationMaterializationReceiptV1
    materialization_receipt_sha256: str
    full_graph_sha256: str
    development_graph_sha256: str
    confirmation_graph_sha256: str
    split_algorithm: Literal[
        "sha256-authority-token-set-first8-mod3-zero-is-confirmation-v2"
    ] = SPLIT_ALGORITHM
    split_salt: Literal["literature-multiverse-condition-confirmation-v2"] = SPLIT_SALT
    nominal_confirmation_fraction: Literal[1 / 3] = 1 / 3
    component_assignments: list[ConditionComponentAssignmentV1]
    full_partition: ConditionGraphPartitionV1
    development_partition: ConditionGraphPartitionV1
    confirmation_partition: ConditionGraphPartitionV1
    status: Literal["ready", "insufficient"]
    insufficiency_reasons: list[str]
    access_semantics: Literal[
        "planner opened only the strict label-free roster and content-silent custodian "
        "receipt; no graph, effect, uncertainty, or outcome value opened by planner"
    ] = (
        "planner opened only the strict label-free roster and content-silent custodian "
        "receipt; no graph, effect, uncertainty, or outcome value opened by planner"
    )
    external_anchor_required: Literal[True] = True
    plan_sha256: str

    @field_validator(
        "target_sha256",
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "config_sha256",
        "pipeline_sha256",
        "roster_sha256",
        "materialization_receipt_sha256",
        "full_graph_sha256",
        "development_graph_sha256",
        "confirmation_graph_sha256",
        "plan_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("insufficiency_reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "insufficiency_reasons")

    @field_validator("external_freeze_anchor")
    @classmethod
    def validate_external_anchor(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_confirmation_external_freeze_anchor_empty")
        return normalized

    @model_validator(mode="after")
    def validate_plan(self) -> ConditionConfirmationPlanV1:
        if self.target_sha256 != self.target.target_sha256:
            raise ValueError("condition_confirmation_plan_target_mismatch")
        if (
            self.claim_spec_sha256 != self.target.claim_spec_sha256
            or self.question_config_sha256 != self.target.question_config_sha256
            or self.corpus_snapshot_sha256 != self.target.corpus_snapshot_sha256
            or self.corpus_cutoff != self.target.corpus_cutoff
            or self.claim_contrast_id != self.target.claim_contrast_id
        ):
            raise ValueError("condition_confirmation_plan_claim_corpus_binding_mismatch")
        if self.config_sha256 != self.config.config_sha256:
            raise ValueError("condition_confirmation_plan_config_mismatch")
        if self.roster_sha256 != self.roster.roster_sha256:
            raise ValueError("condition_confirmation_plan_roster_mismatch")
        receipt = self.materialization_receipt
        if self.materialization_receipt_sha256 != receipt.receipt_sha256:
            raise ValueError("condition_confirmation_plan_materialization_receipt_mismatch")
        if (
            receipt.target_sha256 != self.target_sha256
            or receipt.claim_spec_sha256 != self.claim_spec_sha256
            or receipt.question_config_sha256 != self.question_config_sha256
            or receipt.corpus_snapshot_sha256 != self.corpus_snapshot_sha256
            or receipt.corpus_cutoff != self.corpus_cutoff
            or receipt.question_id != self.target.question_id
            or receipt.roster_sha256 != self.roster_sha256
        ):
            raise ValueError(
                "condition_confirmation_plan_materialization_target_roster_mismatch"
            )
        if self.full_graph_sha256 != self.roster.source_graph_sha256:
            raise ValueError("condition_confirmation_plan_source_graph_mismatch")
        if (
            receipt.full_graph_sha256 != self.full_graph_sha256
            or receipt.development_graph_sha256 != self.development_graph_sha256
            or receipt.confirmation_graph_sha256 != self.confirmation_graph_sha256
        ):
            raise ValueError(
                "condition_confirmation_plan_materialization_graph_hash_mismatch"
            )
        expected_assignments = derive_condition_components(
            self.roster,
            question_id=self.target.question_id,
        )
        if self.component_assignments != expected_assignments:
            raise ValueError("condition_confirmation_plan_component_assignment_mismatch")
        if receipt.component_assignments_sha256 != hash_canonical(expected_assignments):
            raise ValueError(
                "condition_confirmation_plan_materialization_component_hash_mismatch"
            )
        expected_full = _freeze_partition("full", expected_assignments)
        expected_development = _freeze_partition(
            "development",
            [row for row in expected_assignments if row.split == "development"],
        )
        expected_confirmation = _freeze_partition(
            "confirmation",
            [row for row in expected_assignments if row.split == "confirmation"],
        )
        if self.full_partition != expected_full:
            raise ValueError("condition_confirmation_plan_full_partition_mismatch")
        if self.development_partition != expected_development:
            raise ValueError("condition_confirmation_plan_development_partition_mismatch")
        if self.confirmation_partition != expected_confirmation:
            raise ValueError("condition_confirmation_plan_confirmation_partition_mismatch")
        if (
            receipt.full_partition_sha256 != expected_full.partition_sha256
            or receipt.development_partition_sha256
            != expected_development.partition_sha256
            or receipt.confirmation_partition_sha256
            != expected_confirmation.partition_sha256
            or receipt.full_component_count != len(expected_assignments)
            or receipt.development_component_count
            != len(expected_development.component_ids)
            or receipt.confirmation_component_count
            != len(expected_confirmation.component_ids)
            or receipt.full_publication_count != len(self.roster.publications)
            or receipt.full_study_count != len(self.roster.studies)
            or receipt.full_cohort_count != len(self.roster.cohorts)
            or receipt.full_estimate_count != len(self.roster.estimates)
            or receipt.development_estimate_count
            != len(expected_development.estimate_ids)
            or receipt.confirmation_estimate_count
            != len(expected_confirmation.estimate_ids)
        ):
            raise ValueError(
                "condition_confirmation_plan_materialization_partition_mismatch"
            )
        expected_reasons = _plan_reasons(
            roster=self.roster,
            target=self.target,
            config=self.config,
            assignments=expected_assignments,
        )
        if self.insufficiency_reasons != expected_reasons:
            raise ValueError("condition_confirmation_plan_reasons_mismatch")
        if (self.status == "ready") != (not expected_reasons):
            raise ValueError("condition_confirmation_plan_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        if hash_canonical(payload) != self.plan_sha256:
            raise ValueError("condition_confirmation_plan_hash_mismatch")
        return self


def prepare_condition_confirmation_plan(
    *,
    target: ConditionConfirmationTargetV1,
    config: ConditionConfirmationConfigV1,
    roster: LabelFreeGraphRosterV1,
    materialization_receipt: ConditionConfirmationMaterializationReceiptV1,
    pipeline_sha256: str,
    external_freeze_anchor: str,
) -> ConditionConfirmationPlanV1:
    """Freeze the full component split from an outcome-blind roster only."""

    try:
        target = ConditionConfirmationTargetV1.model_validate(target.model_dump(mode="json"))
        config = ConditionConfirmationConfigV1.model_validate(config.model_dump(mode="json"))
        roster = LabelFreeGraphRosterV1.model_validate(roster.model_dump(mode="json"))
        materialization_receipt = (
            ConditionConfirmationMaterializationReceiptV1.model_validate(
                materialization_receipt.model_dump(mode="json")
            )
        )
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError("condition_confirmation_prepare_input_tampered") from exc
    _sha256(pipeline_sha256, "pipeline_sha256")
    assignments = derive_condition_components(roster, question_id=target.question_id)
    reasons = _plan_reasons(
        roster=roster,
        target=target,
        config=config,
        assignments=assignments,
    )
    payload: dict[str, Any] = {
        "plan_version": "condition-confirmation-plan-v1",
        "freeze_state": "confirmation_outcomes_unopened",
        "target": target,
        "target_sha256": target.target_sha256,
        "claim_spec_sha256": target.claim_spec_sha256,
        "question_config_sha256": target.question_config_sha256,
        "corpus_snapshot_sha256": target.corpus_snapshot_sha256,
        "corpus_cutoff": target.corpus_cutoff,
        "claim_contrast_id": target.claim_contrast_id,
        "config": config,
        "config_sha256": config.config_sha256,
        "pipeline_sha256": pipeline_sha256,
        "external_freeze_anchor": external_freeze_anchor,
        "roster": roster,
        "roster_sha256": roster.roster_sha256,
        "materialization_receipt": materialization_receipt,
        "materialization_receipt_sha256": materialization_receipt.receipt_sha256,
        "full_graph_sha256": roster.source_graph_sha256,
        "development_graph_sha256": materialization_receipt.development_graph_sha256,
        "confirmation_graph_sha256": (
            materialization_receipt.confirmation_graph_sha256
        ),
        "split_algorithm": SPLIT_ALGORITHM,
        "split_salt": SPLIT_SALT,
        "nominal_confirmation_fraction": 1 / 3,
        "component_assignments": assignments,
        "full_partition": _freeze_partition("full", assignments),
        "development_partition": _freeze_partition(
            "development", [row for row in assignments if row.split == "development"]
        ),
        "confirmation_partition": _freeze_partition(
            "confirmation", [row for row in assignments if row.split == "confirmation"]
        ),
        "status": "ready" if not reasons else "insufficient",
        "insufficiency_reasons": reasons,
        "access_semantics": (
            "planner opened only the strict label-free roster and content-silent "
            "custodian receipt; no graph, effect, uncertainty, or outcome value "
            "opened by planner"
        ),
        "external_anchor_required": True,
    }
    return ConditionConfirmationPlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def partition_evidence_graph(
    graph: EvidenceGraph,
    partition: ConditionGraphPartitionV1,
) -> EvidenceGraph:
    """Materialize one exact graph partition; intended after split preregistration."""

    graph = EvidenceGraph.model_validate(graph.model_dump(mode="json"))
    if not partition.publication_ids:
        raise ConditionConfirmationError("condition_confirmation_partition_has_no_publications")
    publication_ids = set(partition.publication_ids)
    study_ids = set(partition.study_ids)
    cohort_ids = set(partition.cohort_ids)
    arm_ids = set(partition.arm_ids)
    contrast_ids = set(partition.contrast_ids)
    estimate_ids = set(partition.estimate_ids)
    span_ids = set(partition.span_ids)
    return EvidenceGraph(
        publications=[
            row for row in graph.publications if row.publication_id in publication_ids
        ],
        studies=[row for row in graph.studies if row.study_id in study_ids],
        cohorts=[row for row in graph.cohorts if row.cohort_id in cohort_ids],
        arms=[row for row in graph.arms if row.arm_id in arm_ids],
        contrasts=[row for row in graph.contrasts if row.contrast_id in contrast_ids],
        outcome_estimates=[
            row for row in graph.outcome_estimates if row.estimate_id in estimate_ids
        ],
        evidence_spans=[row for row in graph.evidence_spans if row.span_id in span_ids],
    )


def partition_full_graph_for_plan(
    graph: EvidenceGraph,
    plan: ConditionConfirmationPlanV1,
) -> tuple[EvidenceGraph, EvidenceGraph]:
    """Return the exact development and confirmation graphs frozen by a plan."""

    graph = EvidenceGraph.model_validate(graph.model_dump(mode="json"))
    plan = ConditionConfirmationPlanV1.model_validate(plan.model_dump(mode="json"))
    if hash_canonical(graph) != plan.full_graph_sha256:
        raise ConditionConfirmationError("condition_confirmation_full_graph_hash_mismatch")
    return (
        partition_evidence_graph(graph, plan.development_partition),
        partition_evidence_graph(graph, plan.confirmation_partition),
    )


def _graph_indexes(graph: EvidenceGraph) -> dict[str, Any]:
    publication_by_id = {row.publication_id: row for row in graph.publications}
    publication_by_paper = {row.paper_id: row for row in graph.publications}
    study_by_id = {row.study_id: row for row in graph.studies}
    cohort_by_id = {row.cohort_id: row for row in graph.cohorts}
    arm_by_id = {row.arm_id: row for row in graph.arms}
    contrast_by_id = {row.contrast_id: row for row in graph.contrasts}
    return {
        "publication_by_id": publication_by_id,
        "publication_by_paper": publication_by_paper,
        "study_by_id": study_by_id,
        "cohort_by_id": cohort_by_id,
        "arm_by_id": arm_by_id,
        "contrast_by_id": contrast_by_id,
    }


def _estimate_target_projection(
    *,
    estimate: Any,
    contrast_id: str,
    contrast_label: str,
    target: ConditionConfirmationTargetV1,
) -> tuple[bool, dict[str, ModeratorScalar | None]]:
    target_scope = (
        estimate.outcome_name == target.outcome_name
        and contrast_label == target.contrast_label
        and (
            target.claim_contrast_id is None
            or contrast_id == target.claim_contrast_id
        )
    )
    if not target_scope:
        return False, {}
    moderators = {
        name: estimate.effect.moderators.get(name) for name in target.moderator_names
    }
    return True, moderators


def _actual_estimate_status(
    *,
    estimate: Any,
    contrast_id: str,
    contrast_label: str,
    target: ConditionConfirmationTargetV1,
) -> tuple[bool, str, dict[str, ModeratorScalar | None]]:
    target_scope, moderators = _estimate_target_projection(
        estimate=estimate,
        contrast_id=contrast_id,
        contrast_label=contrast_label,
        target=target,
    )
    if not target_scope:
        return False, "out_of_scope", {}
    if any(
        value is None or (isinstance(value, str) and not value.strip())
        for value in moderators.values()
    ):
        return True, "missing_moderator", moderators
    result = harmonize_effect(estimate.effect)
    if result.status == "estimable":
        assert result.effect is not None
        if result.effect.measure is target.measure and result.effect.unit == target.unit:
            return True, "compatible_quantitative", moderators
        return True, "incompatible_quantitative", moderators
    if estimate.effect.estimate is not None or estimate.legacy_reported_direction in {
        "increase",
        "decrease",
    }:
        return True, "directional_only", moderators
    if estimate.effect.availability.value in {"missing", "inconclusive"}:
        return True, "non_estimable", moderators
    return True, "unresolved", moderators


def _project_label_free_roster(
    *,
    full_graph: EvidenceGraph,
    target: ConditionConfirmationTargetV1,
) -> LabelFreeGraphRosterV1:
    publication_by_paper = {row.paper_id: row for row in full_graph.publications}
    cohort_by_id = {row.cohort_id: row for row in full_graph.cohorts}
    contrast_by_id = {row.contrast_id: row for row in full_graph.contrasts}
    estimate_rows: list[RosterEstimateV1] = []
    for estimate in full_graph.outcome_estimates:
        contrast = contrast_by_id[estimate.contrast_id]
        cohort = cohort_by_id[contrast.cohort_id]
        publication = publication_by_paper[estimate.effect.paper_id]
        target_scope, moderators = _estimate_target_projection(
            estimate=estimate,
            contrast_id=contrast.contrast_id,
            contrast_label=contrast.label,
            target=target,
        )
        estimate_rows.append(
            RosterEstimateV1(
                estimate_id=estimate.estimate_id,
                publication_id=publication.publication_id,
                study_id=cohort.study_id,
                cohort_id=cohort.cohort_id,
                contrast_id=contrast.contrast_id,
                target_scope=target_scope,
                moderator_values=(
                    {name: moderators[name] for name in sorted(moderators)}
                    if target_scope
                    else {}
                ),
            )
        )
    return freeze_label_free_graph_roster(
        source_graph_sha256=hash_canonical(full_graph),
        publications=[
            RosterPublicationV1(
                publication_id=row.publication_id,
                paper_id=row.paper_id,
                doi=row.doi,
                pmid=row.pmid,
                doc_id=row.doc_id,
            )
            for row in full_graph.publications
        ],
        studies=[
            RosterStudyV1(
                study_id=row.study_id,
                publication_ids=row.publication_ids,
                registration_ids=row.registration_ids,
            )
            for row in full_graph.studies
        ],
        cohorts=[
            RosterCohortV1(
                cohort_id=row.cohort_id,
                study_id=row.study_id,
                identity_basis=row.identity.basis,
                registry_ids=row.identity.registry_ids,
                dataset_ids=row.identity.dataset_ids,
            )
            for row in full_graph.cohorts
        ],
        arms=[
            RosterArmV1(
                arm_id=row.arm_id,
                cohort_id=row.cohort_id,
                role=row.role,
            )
            for row in full_graph.arms
        ],
        contrasts=[
            RosterContrastV1(
                contrast_id=row.contrast_id,
                cohort_id=row.cohort_id,
                treatment_arm_id=row.treatment_arm_id,
                comparator_arm_id=row.comparator_arm_id,
                label=row.label,
                estimand=row.estimand,
                positive_direction_means=row.positive_direction_means,
            )
            for row in full_graph.contrasts
        ],
        estimates=estimate_rows,
        spans=[
            RosterSpanV1(
                span_id=row.span_id,
                publication_id=row.publication_id,
            )
            for row in full_graph.evidence_spans
        ],
    )


def materialize_condition_confirmation_inputs(
    *,
    full_graph: EvidenceGraph,
    target: ConditionConfirmationTargetV1,
) -> tuple[
    LabelFreeGraphRosterV1,
    EvidenceGraph,
    EvidenceGraph,
    ConditionConfirmationMaterializationReceiptV1,
]:
    """Custodian-only projection and deterministic split of one frozen full graph."""

    try:
        target = ConditionConfirmationTargetV1.model_validate(
            target.model_dump(mode="json")
        )
        full_graph = EvidenceGraph.model_validate(full_graph.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError(
            "condition_confirmation_materialization_input_tampered"
        ) from exc
    roster = _project_label_free_roster(full_graph=full_graph, target=target)
    assignments = derive_condition_components(roster, question_id=target.question_id)
    full_partition = _freeze_partition("full", assignments)
    development_partition = _freeze_partition(
        "development",
        [row for row in assignments if row.split == "development"],
    )
    confirmation_partition = _freeze_partition(
        "confirmation",
        [row for row in assignments if row.split == "confirmation"],
    )
    try:
        development_graph = partition_evidence_graph(full_graph, development_partition)
        confirmation_graph = partition_evidence_graph(full_graph, confirmation_partition)
    except ConditionConfirmationError as exc:
        raise ConditionConfirmationError(
            "condition_confirmation_materialization_empty_partition"
        ) from exc
    receipt = _freeze_materialization_receipt(
        target=target,
        roster=roster,
        assignments=assignments,
        full_partition=full_partition,
        development_partition=development_partition,
        confirmation_partition=confirmation_partition,
        development_graph_sha256=hash_canonical(development_graph),
        confirmation_graph_sha256=hash_canonical(confirmation_graph),
    )
    return roster, development_graph, confirmation_graph, receipt


def validate_condition_confirmation_materialization(
    *,
    full_graph: EvidenceGraph,
    target: ConditionConfirmationTargetV1,
    roster: LabelFreeGraphRosterV1,
    development_graph: EvidenceGraph,
    confirmation_graph: EvidenceGraph,
    receipt: ConditionConfirmationMaterializationReceiptV1,
) -> ConditionConfirmationMaterializationReceiptV1:
    """Exact-replay all custodian outputs from the frozen target and full graph."""

    expected = materialize_condition_confirmation_inputs(
        full_graph=full_graph,
        target=target,
    )
    expected_roster, expected_development, expected_confirmation, expected_receipt = (
        expected
    )
    try:
        observed_roster = LabelFreeGraphRosterV1.model_validate(
            roster.model_dump(mode="json")
        )
        observed_development = EvidenceGraph.model_validate(
            development_graph.model_dump(mode="json")
        )
        observed_confirmation = EvidenceGraph.model_validate(
            confirmation_graph.model_dump(mode="json")
        )
        observed_receipt = ConditionConfirmationMaterializationReceiptV1.model_validate(
            receipt.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError(
            "condition_confirmation_materialization_output_tampered"
        ) from exc
    if (
        observed_roster != expected_roster
        or observed_development != expected_development
        or observed_confirmation != expected_confirmation
        or observed_receipt != expected_receipt
    ):
        raise ConditionConfirmationError(
            "condition_confirmation_materialization_recomputation_mismatch"
        )
    return observed_receipt


def _validate_graph_projection(
    *,
    graph: EvidenceGraph,
    plan: ConditionConfirmationPlanV1,
    partition: ConditionGraphPartitionV1,
    expected_graph_sha256: str,
) -> None:
    if hash_canonical(graph) != expected_graph_sha256:
        raise ConditionConfirmationError(
            f"condition_confirmation_{partition.split}_graph_hash_mismatch"
        )
    observed_ids = {
        "publication_ids": sorted(row.publication_id for row in graph.publications),
        "study_ids": sorted(row.study_id for row in graph.studies),
        "cohort_ids": sorted(row.cohort_id for row in graph.cohorts),
        "arm_ids": sorted(row.arm_id for row in graph.arms),
        "contrast_ids": sorted(row.contrast_id for row in graph.contrasts),
        "estimate_ids": sorted(row.estimate_id for row in graph.outcome_estimates),
        "span_ids": sorted(row.span_id for row in graph.evidence_spans),
    }
    for name, observed in observed_ids.items():
        if observed != getattr(partition, name):
            raise ConditionConfirmationError(
                f"condition_confirmation_{partition.split}_node_partition_mismatch:{name}"
            )
    roster = plan.roster
    roster_publications = {row.publication_id: row for row in roster.publications}
    roster_studies = {row.study_id: row for row in roster.studies}
    roster_cohorts = {row.cohort_id: row for row in roster.cohorts}
    roster_arms = {row.arm_id: row for row in roster.arms}
    roster_contrasts = {row.contrast_id: row for row in roster.contrasts}
    roster_estimates = {row.estimate_id: row for row in roster.estimates}
    roster_spans = {row.span_id: row for row in roster.spans}
    indexes = _graph_indexes(graph)
    for publication in graph.publications:
        expected = roster_publications[publication.publication_id]
        if (
            publication.paper_id != expected.paper_id
            or publication.doi != expected.doi
            or publication.pmid != expected.pmid
            or publication.doc_id != expected.doc_id
        ):
            raise ConditionConfirmationError(
                "condition_confirmation_publication_identity_mismatch:"
                f"{publication.publication_id}"
            )
    for study in graph.studies:
        expected = roster_studies[study.study_id]
        if (
            study.publication_ids != expected.publication_ids
            or study.registration_ids != expected.registration_ids
        ):
            raise ConditionConfirmationError(
                f"condition_confirmation_study_identity_mismatch:{study.study_id}"
            )
    for cohort in graph.cohorts:
        expected = roster_cohorts[cohort.cohort_id]
        if (
            cohort.study_id != expected.study_id
            or cohort.identity.basis is not expected.identity_basis
            or cohort.identity.registry_ids != expected.registry_ids
            or cohort.identity.dataset_ids != expected.dataset_ids
        ):
            raise ConditionConfirmationError(
                f"condition_confirmation_cohort_identity_mismatch:{cohort.cohort_id}"
            )
    for arm in graph.arms:
        expected = roster_arms[arm.arm_id]
        if arm.cohort_id != expected.cohort_id or arm.role is not expected.role:
            raise ConditionConfirmationError(
                f"condition_confirmation_arm_identity_mismatch:{arm.arm_id}"
            )
    for contrast in graph.contrasts:
        expected = roster_contrasts[contrast.contrast_id]
        if (
            contrast.cohort_id != expected.cohort_id
            or contrast.treatment_arm_id != expected.treatment_arm_id
            or contrast.comparator_arm_id != expected.comparator_arm_id
            or contrast.label != expected.label
            or contrast.estimand != expected.estimand
            or contrast.positive_direction_means
            != expected.positive_direction_means
        ):
            raise ConditionConfirmationError(
                f"condition_confirmation_contrast_identity_mismatch:{contrast.contrast_id}"
            )
    for span in graph.evidence_spans:
        if span.publication_id != roster_spans[span.span_id].publication_id:
            raise ConditionConfirmationError(
                f"condition_confirmation_span_identity_mismatch:{span.span_id}"
            )
    publication_by_paper = indexes["publication_by_paper"]
    cohort_by_id = indexes["cohort_by_id"]
    contrast_by_id = indexes["contrast_by_id"]
    for estimate in graph.outcome_estimates:
        expected = roster_estimates[estimate.estimate_id]
        contrast = contrast_by_id[estimate.contrast_id]
        cohort = cohort_by_id[contrast.cohort_id]
        publication = publication_by_paper[estimate.effect.paper_id]
        target_scope, _, moderators = _actual_estimate_status(
            estimate=estimate,
            contrast_id=contrast.contrast_id,
            contrast_label=contrast.label,
            target=plan.target,
        )
        if (
            expected.publication_id != publication.publication_id
            or expected.study_id != cohort.study_id
            or expected.cohort_id != cohort.cohort_id
            or expected.contrast_id != contrast.contrast_id
            or expected.target_scope != target_scope
            or expected.moderator_values != moderators
        ):
            raise ConditionConfirmationError(
                f"condition_confirmation_estimate_projection_mismatch:{estimate.estimate_id}"
            )


def _target_effects(
    graph: EvidenceGraph,
    plan: ConditionConfirmationPlanV1,
) -> tuple[list[HarmonizedEffect], list[str], list[str]]:
    indexes = _graph_indexes(graph)
    contrast_by_id = indexes["contrast_by_id"]
    compatible: list[HarmonizedEffect] = []
    cohort_ids: list[str] = []
    reasons: list[str] = []
    for estimate in sorted(graph.outcome_estimates, key=lambda row: row.estimate_id):
        contrast = contrast_by_id[estimate.contrast_id]
        target_scope, status, _ = _actual_estimate_status(
            estimate=estimate,
            contrast_id=contrast.contrast_id,
            contrast_label=contrast.label,
            target=plan.target,
        )
        if not target_scope:
            continue
        if status != "compatible_quantitative":
            reasons.append(f"confirmation_effect_not_compatible:{estimate.estimate_id}:{status}")
            continue
        result = harmonize_effect(estimate.effect)
        assert result.effect is not None
        compatible.append(result.effect)
        cohort_ids.append(contrast.cohort_id)
    if not compatible:
        reasons.append("no_compatible_quantitative_target_effects")
    return compatible, cohort_ids, sorted(set(reasons))


class NormalPredictiveParametersV1(ContractModel):
    mean: float
    mean_variance: Annotated[float, Field(ge=0)]
    tau_squared: Annotated[float, Field(ge=0)]
    development_component_count: Annotated[int, Field(ge=2)]

    @field_validator("mean", "mean_variance", "tau_squared")
    @classmethod
    def validate_numeric(cls, value: float, info: Any) -> float:
        return _finite(
            value,
            info.field_name,
            nonnegative=info.field_name != "mean",
        )


class ConditionalLevelParametersV1(ContractModel):
    level: Annotated[str, Field(min_length=1)]
    design_vector: list[float]
    fitted_mean: float
    fitted_mean_variance: Annotated[float, Field(ge=0)]
    development_component_count: Annotated[int, Field(ge=2)]

    @field_validator("fitted_mean", "fitted_mean_variance")
    @classmethod
    def validate_numeric(cls, value: float, info: Any) -> float:
        return _finite(
            value,
            info.field_name,
            nonnegative=info.field_name == "fitted_mean_variance",
        )


class CategoricalNormalPredictiveModelV1(ContractModel):
    moderator: Annotated[str, Field(min_length=1)]
    reference_level: Annotated[str, Field(min_length=1)]
    levels: Annotated[list[str], Field(min_length=2)]
    coefficient_names: list[str]
    beta: list[float]
    covariance: list[list[float]]
    tau_squared: Annotated[float, Field(ge=0)]
    residual_degrees_freedom: Annotated[int, Field(ge=2)]
    level_parameters: list[ConditionalLevelParametersV1]

    @field_validator("levels", "coefficient_names")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        if info.field_name == "levels":
            return _sorted_unique(value, info.field_name, allow_empty=False)
        if not value or len(value) != len(set(value)) or any(not row for row in value):
            raise ValueError("condition_confirmation_coefficient_names_invalid")
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> CategoricalNormalPredictiveModelV1:
        dimension = len(self.beta)
        if dimension != len(self.coefficient_names) or dimension != len(self.levels):
            raise ValueError("condition_confirmation_conditional_model_dimension_mismatch")
        if len(self.covariance) != dimension or any(
            len(row) != dimension for row in self.covariance
        ):
            raise ValueError("condition_confirmation_covariance_dimension_mismatch")
        beta = np.asarray(self.beta, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance)):
            raise ValueError("condition_confirmation_model_value_nonfinite")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
            raise ValueError("condition_confirmation_covariance_not_symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
            raise ValueError("condition_confirmation_covariance_not_positive_semidefinite")
        if self.reference_level not in self.levels:
            raise ValueError("condition_confirmation_reference_level_unknown")
        if [row.level for row in self.level_parameters] != self.levels:
            raise ValueError("condition_confirmation_level_parameter_order_mismatch")
        for row in self.level_parameters:
            if len(row.design_vector) != dimension:
                raise ValueError("condition_confirmation_design_vector_dimension_mismatch")
            vector = np.asarray(row.design_vector, dtype=float)
            expected_mean = float(vector @ beta)
            expected_variance = float(vector @ covariance @ vector)
            if not math.isclose(row.fitted_mean, expected_mean, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("condition_confirmation_level_mean_mismatch")
            if not math.isclose(
                row.fitted_mean_variance,
                expected_variance,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("condition_confirmation_level_variance_mismatch")
        return self

    def parameters_for_level(self, level: str) -> ConditionalLevelParametersV1 | None:
        return next((row for row in self.level_parameters if row.level == level), None)


class ModeratorDevelopmentCandidateV1(ContractModel):
    moderator: str
    status: Literal["qualifies", "does_not_qualify", "insufficient"]
    reason: str
    omnibus_p_value: float | None = None
    family_adjusted_omnibus_p_value: float | None = None
    positive_levels: list[str] = Field(default_factory=list)
    negative_levels: list[str] = Field(default_factory=list)

    @field_validator("positive_levels", "negative_levels")
    @classmethod
    def validate_levels(cls, value: list[str], info: Any) -> list[str]:
        return _sorted_unique(value, info.field_name)

    @field_validator("omnibus_p_value", "family_adjusted_omnibus_p_value")
    @classmethod
    def validate_p_value(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("condition_confirmation_candidate_p_value_invalid")
        return value


class ConditionConfirmationFrozenModelV1(ContractModel):
    model_version: Literal["condition-confirmation-frozen-model-v1"] = (
        "condition-confirmation-frozen-model-v1"
    )
    freeze_state: Literal["confirmation_outcomes_unopened"] = (
        "confirmation_outcomes_unopened"
    )
    plan: ConditionConfirmationPlanV1
    plan_sha256: str
    target_sha256: str
    claim_spec_sha256: str
    question_config_sha256: str
    corpus_snapshot_sha256: str
    corpus_cutoff: str
    claim_contrast_id: str | None
    config_sha256: str
    pipeline_sha256: str
    development_graph_sha256: str
    development_effect_input_sha256: str
    status: Literal["fitted", "insufficient"]
    insufficiency_reasons: list[str]
    unconditional: NormalPredictiveParametersV1 | None
    moderator_candidates: list[ModeratorDevelopmentCandidateV1]
    selected_moderator: str | None
    frozen_positive_level: str | None
    frozen_negative_level: str | None
    conditional: CategoricalNormalPredictiveModelV1 | None
    development_family_result: dict[str, JsonValue] | None
    fit_semantics: Literal[
        "development graph only; one conservative contribution per connected "
        "independence component; generalized Paule-Mandel tau and modified "
        "Knapp-Hartung covariance"
    ] = (
        "development graph only; one conservative contribution per connected "
        "independence component; generalized Paule-Mandel tau and modified "
        "Knapp-Hartung covariance"
    )
    model_sha256: str

    @field_validator(
        "plan_sha256",
        "target_sha256",
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "config_sha256",
        "pipeline_sha256",
        "development_graph_sha256",
        "development_effect_input_sha256",
        "model_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("insufficiency_reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "model_insufficiency_reasons")

    @model_validator(mode="after")
    def validate_model(self) -> ConditionConfirmationFrozenModelV1:
        if self.plan_sha256 != self.plan.plan_sha256:
            raise ValueError("condition_confirmation_model_plan_mismatch")
        if self.target_sha256 != self.plan.target_sha256:
            raise ValueError("condition_confirmation_model_target_mismatch")
        if (
            self.claim_spec_sha256 != self.plan.claim_spec_sha256
            or self.question_config_sha256 != self.plan.question_config_sha256
            or self.corpus_snapshot_sha256 != self.plan.corpus_snapshot_sha256
            or self.corpus_cutoff != self.plan.corpus_cutoff
            or self.claim_contrast_id != self.plan.claim_contrast_id
        ):
            raise ValueError("condition_confirmation_model_claim_corpus_binding_mismatch")
        if self.config_sha256 != self.plan.config_sha256:
            raise ValueError("condition_confirmation_model_config_mismatch")
        if self.pipeline_sha256 != self.plan.pipeline_sha256:
            raise ValueError("condition_confirmation_model_pipeline_mismatch")
        if self.development_graph_sha256 != self.plan.development_graph_sha256:
            raise ValueError("condition_confirmation_model_development_graph_mismatch")
        candidate_names = [row.moderator for row in self.moderator_candidates]
        if candidate_names != sorted(set(candidate_names)):
            raise ValueError("condition_confirmation_model_candidates_not_sorted")
        fitted_fields = (
            self.unconditional,
            self.selected_moderator,
            self.frozen_positive_level,
            self.frozen_negative_level,
            self.conditional,
            self.development_family_result,
        )
        if self.status == "fitted":
            if self.insufficiency_reasons or any(value is None for value in fitted_fields):
                raise ValueError("condition_confirmation_fitted_model_fields_incomplete")
            if self.selected_moderator not in self.plan.target.moderator_names:
                raise ValueError("condition_confirmation_selected_moderator_not_prespecified")
            assert self.conditional is not None
            if self.conditional.moderator != self.selected_moderator:
                raise ValueError("condition_confirmation_conditional_moderator_mismatch")
            if self.frozen_positive_level == self.frozen_negative_level:
                raise ValueError("condition_confirmation_frozen_polarities_not_distinct")
        elif not self.insufficiency_reasons or any(value is not None for value in fitted_fields):
            raise ValueError("condition_confirmation_insufficient_model_fields_mismatch")
        payload = self.model_dump(mode="json", exclude={"model_sha256"})
        if hash_canonical(payload) != self.model_sha256:
            raise ValueError("condition_confirmation_model_hash_mismatch")
        return self


def _unconditional_parameters(
    effects: Sequence[HarmonizedEffect],
    component_ids: Sequence[str],
    *,
    correlation: float,
) -> NormalPredictiveParametersV1:
    aggregation = aggregate_one_effect_per_cohort(
        effects,
        component_ids,
        assumed_within_cohort_correlation=correlation,
    )
    if aggregation.status != "ok" or aggregation.n_cohorts < 2:
        raise ConditionConfirmationError(
            f"condition_confirmation_unconditional_aggregation_insufficient:{aggregation.reason}"
        )
    y = np.asarray([row.estimate for row in aggregation.effects], dtype=float)
    variances = np.asarray([row.variance for row in aggregation.effects], dtype=float)
    design = np.ones((len(y), 1), dtype=float)
    tau_squared = _paule_mandel_tau_squared(y, variances, design)
    beta, model_covariance, _, residual_q = _weighted_fit(
        y,
        variances,
        design,
        tau_squared,
    )
    degrees_freedom = len(y) - 1
    scale = max(1.0, residual_q / degrees_freedom)
    covariance = model_covariance * scale
    return NormalPredictiveParametersV1(
        mean=float(beta[0]),
        mean_variance=float(covariance[0, 0]),
        tau_squared=tau_squared,
        development_component_count=aggregation.n_cohorts,
    )


def _conditional_parameters(
    effects: Sequence[HarmonizedEffect],
    component_ids: Sequence[str],
    *,
    moderator: str,
    correlation: float,
) -> CategoricalNormalPredictiveModelV1:
    aggregation = aggregate_one_effect_per_cohort(
        effects,
        component_ids,
        assumed_within_cohort_correlation=correlation,
    )
    if aggregation.status != "ok":
        raise ConditionConfirmationError(
            f"condition_confirmation_conditional_aggregation_insufficient:{aggregation.reason}"
        )
    if any(moderator in row.moderator_conflicts for row in aggregation.effects):
        raise ConditionConfirmationError("condition_confirmation_development_moderator_conflict")
    if any(row.moderators.get(moderator) is None for row in aggregation.effects):
        raise ConditionConfirmationError("condition_confirmation_development_moderator_missing")
    level_by_component = {
        row.cohort_id: _moderator_level_label(row.moderators[moderator])
        for row in aggregation.effects
    }
    levels = sorted(set(level_by_component.values()))
    if len(levels) < 2:
        raise ConditionConfirmationError("condition_confirmation_fewer_than_two_levels")
    reference = levels[0]
    comparison = levels[1:]
    design = np.asarray(
        [
            [
                1.0,
                *(
                    float(level_by_component[row.cohort_id] == level)
                    for level in comparison
                ),
            ]
            for row in aggregation.effects
        ],
        dtype=float,
    )
    y = np.asarray([row.estimate for row in aggregation.effects], dtype=float)
    variances = np.asarray([row.variance for row in aggregation.effects], dtype=float)
    residual_degrees_freedom = len(y) - design.shape[1]
    if residual_degrees_freedom < 2:
        raise ConditionConfirmationError(
            "condition_confirmation_development_residual_degrees_of_freedom_insufficient"
        )
    tau_squared = _paule_mandel_tau_squared(y, variances, design)
    beta, model_covariance, _, residual_q = _weighted_fit(
        y,
        variances,
        design,
        tau_squared,
    )
    scale = max(1.0, residual_q / residual_degrees_freedom)
    covariance = model_covariance * scale
    support = Counter(level_by_component.values())
    level_parameters: list[ConditionalLevelParametersV1] = []
    for level in levels:
        vector = np.asarray(
            [1.0, *(float(level == item) for item in comparison)],
            dtype=float,
        )
        level_parameters.append(
            ConditionalLevelParametersV1(
                level=level,
                design_vector=vector.tolist(),
                fitted_mean=float(vector @ beta),
                fitted_mean_variance=float(vector @ covariance @ vector),
                development_component_count=support[level],
            )
        )
    return CategoricalNormalPredictiveModelV1(
        moderator=moderator,
        reference_level=reference,
        levels=levels,
        coefficient_names=["intercept", *(f"level[{level}]" for level in comparison)],
        beta=beta.tolist(),
        covariance=covariance.tolist(),
        tau_squared=tau_squared,
        residual_degrees_freedom=residual_degrees_freedom,
        level_parameters=level_parameters,
    )


def _candidate_diagnostics(
    family_result: Mapping[str, Any],
) -> list[ModeratorDevelopmentCandidateV1]:
    rows: list[ModeratorDevelopmentCandidateV1] = []
    multiplicity = int(family_result.get("multiplicity_test_count", 0))
    for raw in family_result.get("analyses", []):
        regression = raw.get("regression", {})
        if regression.get("status") != "ok":
            status: Literal["qualifies", "does_not_qualify", "insufficient"] = (
                "insufficient"
            )
            reason = str(regression.get("reason") or "development_regression_insufficient")
            omnibus_p = None
            adjusted_omnibus_p = None
        elif raw.get("qualifies") is True:
            status = "qualifies"
            reason = "development_bonferroni_qualitative_rule_passed"
            omnibus_p = float(regression["omnibus"]["p_value"])
            adjusted_omnibus_p = min(1.0, multiplicity * omnibus_p)
        else:
            status = "does_not_qualify"
            reason = "development_bonferroni_qualitative_rule_not_met"
            omnibus_p = float(regression["omnibus"]["p_value"])
            adjusted_omnibus_p = min(1.0, multiplicity * omnibus_p)
        rows.append(
            ModeratorDevelopmentCandidateV1(
                moderator=str(raw["moderator"]),
                status=status,
                reason=reason,
                omnibus_p_value=omnibus_p,
                family_adjusted_omnibus_p_value=adjusted_omnibus_p,
                positive_levels=sorted(str(value) for value in raw.get("positive_levels", [])),
                negative_levels=sorted(str(value) for value in raw.get("negative_levels", [])),
            )
        )
    return sorted(rows, key=lambda row: row.moderator)


def _freeze_model(
    *,
    plan: ConditionConfirmationPlanV1,
    development_effect_input_sha256: str,
    status: Literal["fitted", "insufficient"],
    reasons: Sequence[str],
    unconditional: NormalPredictiveParametersV1 | None = None,
    candidates: Sequence[ModeratorDevelopmentCandidateV1] = (),
    selected_moderator: str | None = None,
    frozen_positive_level: str | None = None,
    frozen_negative_level: str | None = None,
    conditional: CategoricalNormalPredictiveModelV1 | None = None,
    family_result: Mapping[str, Any] | None = None,
) -> ConditionConfirmationFrozenModelV1:
    payload: dict[str, Any] = {
        "model_version": "condition-confirmation-frozen-model-v1",
        "freeze_state": "confirmation_outcomes_unopened",
        "plan": plan,
        "plan_sha256": plan.plan_sha256,
        "target_sha256": plan.target_sha256,
        "claim_spec_sha256": plan.claim_spec_sha256,
        "question_config_sha256": plan.question_config_sha256,
        "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
        "corpus_cutoff": plan.corpus_cutoff,
        "claim_contrast_id": plan.claim_contrast_id,
        "config_sha256": plan.config_sha256,
        "pipeline_sha256": plan.pipeline_sha256,
        "development_graph_sha256": plan.development_graph_sha256,
        "development_effect_input_sha256": development_effect_input_sha256,
        "status": status,
        "insufficiency_reasons": sorted(set(reasons)),
        "unconditional": unconditional,
        "moderator_candidates": sorted(candidates, key=lambda row: row.moderator),
        "selected_moderator": selected_moderator,
        "frozen_positive_level": frozen_positive_level,
        "frozen_negative_level": frozen_negative_level,
        "conditional": conditional,
        "development_family_result": None if family_result is None else dict(family_result),
        "fit_semantics": (
            "development graph only; one conservative contribution per connected "
            "independence component; generalized Paule-Mandel tau and modified "
            "Knapp-Hartung covariance"
        ),
    }
    return ConditionConfirmationFrozenModelV1.model_validate(
        {**payload, "model_sha256": hash_canonical(payload)}
    )


def fit_condition_confirmation_model(
    plan: ConditionConfirmationPlanV1,
    development_graph: EvidenceGraph,
    *,
    current_pipeline_sha256: str,
) -> ConditionConfirmationFrozenModelV1:
    """Fit the frozen predictive model without accepting a confirmation input."""

    try:
        plan = ConditionConfirmationPlanV1.model_validate(plan.model_dump(mode="json"))
        development_graph = EvidenceGraph.model_validate(
            development_graph.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError("condition_confirmation_fit_input_tampered") from exc
    _sha256(current_pipeline_sha256, "current_pipeline_sha256")
    if current_pipeline_sha256 != plan.pipeline_sha256:
        raise ConditionConfirmationError("condition_confirmation_fit_pipeline_hash_mismatch")
    _validate_graph_projection(
        graph=development_graph,
        plan=plan,
        partition=plan.development_partition,
        expected_graph_sha256=plan.development_graph_sha256,
    )
    effects, cohort_ids, effect_reasons = _target_effects(development_graph, plan)
    component_by_cohort = _component_by_cohort(plan.component_assignments)
    component_ids = [
        component_by_cohort[cohort_id].component_id for cohort_id in cohort_ids
    ]
    effect_input_sha256 = hash_canonical(
        {
            "effects": effects,
            "cohort_ids": cohort_ids,
            "independence_component_ids": component_ids,
            "development_graph_sha256": plan.development_graph_sha256,
        }
    )
    reasons = [*plan.insufficiency_reasons, *effect_reasons]
    if reasons:
        return _freeze_model(
            plan=plan,
            development_effect_input_sha256=effect_input_sha256,
            status="insufficient",
            reasons=reasons,
        )
    try:
        unconditional = _unconditional_parameters(
            effects,
            component_ids,
            correlation=plan.config.assumed_within_cohort_correlation,
        )
        family_result = prespecified_cohort_condition_analysis(
            effects,
            component_ids,
            plan.target.moderator_names,
            familywise_alpha=plan.config.development_familywise_alpha,
            min_cohorts_per_level=plan.config.development_min_components_per_level,
            assumed_within_cohort_correlation=(
                plan.config.assumed_within_cohort_correlation
            ),
        )
    except (MetaAnalysisContractError, ConditionConfirmationError, ValueError) as exc:
        return _freeze_model(
            plan=plan,
            development_effect_input_sha256=effect_input_sha256,
            status="insufficient",
            reasons=[f"development_fit_failed:{exc}"],
        )
    candidates = _candidate_diagnostics(family_result)
    if family_result.get("status") == "insufficient":
        return _freeze_model(
            plan=plan,
            development_effect_input_sha256=effect_input_sha256,
            status="insufficient",
            reasons=[
                "development_moderator_family_insufficient:"
                f"{family_result.get('reason', 'unknown')}"
            ],
        )
    qualifying = [row for row in candidates if row.status == "qualifies"]
    if not qualifying:
        return _freeze_model(
            plan=plan,
            development_effect_input_sha256=effect_input_sha256,
            status="insufficient",
            reasons=["no_development_moderator_passed_bonferroni_qualitative_rule"],
        )
    selected = min(
        qualifying,
        key=lambda row: (
            float(
                row.family_adjusted_omnibus_p_value
                if row.family_adjusted_omnibus_p_value is not None
                else 1.0
            ),
            row.moderator,
        ),
    )
    try:
        conditional = _conditional_parameters(
            effects,
            component_ids,
            moderator=selected.moderator,
            correlation=plan.config.assumed_within_cohort_correlation,
        )
        positive = min(
            selected.positive_levels,
            key=lambda level: (
                -conditional.parameters_for_level(level).fitted_mean,  # type: ignore[union-attr]
                level,
            ),
        )
        negative = min(
            selected.negative_levels,
            key=lambda level: (
                conditional.parameters_for_level(level).fitted_mean,  # type: ignore[union-attr]
                level,
            ),
        )
    except (MetaAnalysisContractError, ConditionConfirmationError, ValueError) as exc:
        return _freeze_model(
            plan=plan,
            development_effect_input_sha256=effect_input_sha256,
            status="insufficient",
            reasons=[f"development_frozen_candidate_fit_failed:{exc}"],
        )
    return _freeze_model(
        plan=plan,
        development_effect_input_sha256=effect_input_sha256,
        status="fitted",
        reasons=[],
        unconditional=unconditional,
        candidates=candidates,
        selected_moderator=selected.moderator,
        frozen_positive_level=positive,
        frozen_negative_level=negative,
        conditional=conditional,
        family_result=family_result,
    )


def validate_condition_confirmation_model(
    *,
    plan: ConditionConfirmationPlanV1,
    development_graph: EvidenceGraph,
    model: ConditionConfirmationFrozenModelV1,
    current_pipeline_sha256: str,
) -> ConditionConfirmationFrozenModelV1:
    """Recompute the complete development fit and require byte-semantic equality."""

    expected = fit_condition_confirmation_model(
        plan,
        development_graph,
        current_pipeline_sha256=current_pipeline_sha256,
    )
    try:
        observed = ConditionConfirmationFrozenModelV1.model_validate(
            model.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError("condition_confirmation_model_tampered") from exc
    if observed != expected:
        raise ConditionConfirmationError("condition_confirmation_model_recomputation_mismatch")
    return observed


class ConditionSignPredictionV1(ContractModel):
    cohort_id: str
    component_id: str
    moderator_level: str
    estimate: float
    sampling_variance: Annotated[float, Field(gt=0)]
    observed_positive: bool
    conditional_positive_probability: Annotated[float, Field(ge=0, le=1)]
    unconditional_positive_probability: Annotated[float, Field(ge=0, le=1)]
    conditional_brier: Annotated[float, Field(ge=0, le=1)]
    unconditional_brier: Annotated[float, Field(ge=0, le=1)]
    paired_brier_difference: Annotated[float, Field(ge=-1, le=1)]

    @field_validator(
        "estimate",
        "sampling_variance",
        "conditional_positive_probability",
        "unconditional_positive_probability",
        "conditional_brier",
        "unconditional_brier",
        "paired_brier_difference",
    )
    @classmethod
    def validate_numeric(cls, value: float, info: Any) -> float:
        return _finite(
            value,
            info.field_name,
            nonnegative=info.field_name == "sampling_variance",
        )

    @model_validator(mode="after")
    def validate_losses(self) -> ConditionSignPredictionV1:
        observed = 1.0 if self.observed_positive else 0.0
        conditional = (self.conditional_positive_probability - observed) ** 2
        unconditional = (self.unconditional_positive_probability - observed) ** 2
        if not math.isclose(self.conditional_brier, conditional, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("condition_confirmation_conditional_brier_mismatch")
        if not math.isclose(
            self.unconditional_brier,
            unconditional,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("condition_confirmation_unconditional_brier_mismatch")
        if not math.isclose(
            self.paired_brier_difference,
            conditional - unconditional,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("condition_confirmation_paired_brier_mismatch")
        return self


class ComponentBrierMeanV1(ContractModel):
    component_id: str
    prediction_count: Annotated[int, Field(ge=1)]
    mean_paired_brier_difference: Annotated[float, Field(ge=-1, le=1)]


class PairedComponentBootstrapV1(ContractModel):
    protocol: Literal["paired-component-bootstrap-v1"] = BOOTSTRAP_PROTOCOL
    resampling_unit: Literal["independent_confirmation_component"] = (
        "independent_confirmation_component"
    )
    component_count: Annotated[int, Field(ge=1)]
    component_means: list[ComponentBrierMeanV1]
    point_delta_brier: Annotated[float, Field(ge=-1, le=1)]
    bootstrap_replicates: Literal[10000] = 10000
    seed: Annotated[int, Field(ge=0)]
    upper_quantile: Literal[0.95] = 0.95
    quantile_method: Literal["higher"] = "higher"
    one_sided_upper_bound: Annotated[float, Field(ge=-1, le=1)]
    minimum_required_improvement: Annotated[float, Field(ge=0)]
    passed: bool

    @model_validator(mode="after")
    def validate_bootstrap(self) -> PairedComponentBootstrapV1:
        if self.component_count != len(self.component_means):
            raise ValueError("condition_confirmation_bootstrap_component_count_mismatch")
        component_ids = [row.component_id for row in self.component_means]
        if component_ids != sorted(set(component_ids)):
            raise ValueError("condition_confirmation_bootstrap_components_not_sorted")
        expected_point = math.fsum(
            row.mean_paired_brier_difference for row in self.component_means
        ) / self.component_count
        if not math.isclose(
            self.point_delta_brier,
            expected_point,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("condition_confirmation_bootstrap_point_mismatch")
        if self.passed != (
            self.one_sided_upper_bound < -self.minimum_required_improvement
        ):
            raise ValueError("condition_confirmation_bootstrap_pass_mismatch")
        return self


class PolarityLevelReplicationV1(ContractModel):
    polarity: Literal["positive", "negative"]
    frozen_level: str
    component_ids: list[str]
    component_count: Annotated[int, Field(ge=1)]
    confidence_level: Literal[0.975] = 0.975
    synthesis: dict[str, JsonValue]
    passed: bool

    @field_validator("component_ids")
    @classmethod
    def validate_components(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "polarity_component_ids", allow_empty=False)

    @model_validator(mode="after")
    def validate_replication(self) -> PolarityLevelReplicationV1:
        if self.component_count != len(self.component_ids):
            raise ValueError("condition_confirmation_polarity_component_count_mismatch")
        if self.synthesis.get("status") != "ok":
            if self.passed:
                raise ValueError("condition_confirmation_insufficient_polarity_cannot_pass")
            return self
        lower = float(self.synthesis["ci_lower"])
        upper = float(self.synthesis["ci_upper"])
        expected = lower > 0 if self.polarity == "positive" else upper < 0
        if self.passed != expected:
            raise ValueError("condition_confirmation_polarity_pass_mismatch")
        return self


class OppositePolarityReplicationV1(ContractModel):
    multiplicity_control: Literal["bonferroni_two_frozen_levels"] = (
        "bonferroni_two_frozen_levels"
    )
    familywise_alpha: Literal[0.05] = 0.05
    per_level_two_sided_confidence: Literal[0.975] = 0.975
    positive: PolarityLevelReplicationV1
    negative: PolarityLevelReplicationV1
    components_containing_both_frozen_levels: list[str]
    passed: bool

    @field_validator("components_containing_both_frozen_levels")
    @classmethod
    def validate_overlap(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "polarity_component_overlap")

    @model_validator(mode="after")
    def validate_result(self) -> OppositePolarityReplicationV1:
        expected = (
            not self.components_containing_both_frozen_levels
            and self.positive.passed
            and self.negative.passed
        )
        if self.passed != expected:
            raise ValueError("condition_confirmation_opposite_polarity_pass_mismatch")
        return self


class ConditionSplitOverlapChecksV1(ContractModel):
    publication_ids: list[str]
    paper_ids: list[str]
    study_ids: list[str]
    cohort_ids: list[str]
    strong_identity_tokens: list[str]
    component_ids: list[str]
    passed: bool

    @field_validator(
        "publication_ids",
        "paper_ids",
        "study_ids",
        "cohort_ids",
        "strong_identity_tokens",
        "component_ids",
    )
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _sorted_unique(value, f"overlap_{info.field_name}")

    @model_validator(mode="after")
    def validate_pass(self) -> ConditionSplitOverlapChecksV1:
        overlaps = (
            self.publication_ids,
            self.paper_ids,
            self.study_ids,
            self.cohort_ids,
            self.strong_identity_tokens,
            self.component_ids,
        )
        if self.passed != all(not values for values in overlaps):
            raise ValueError("condition_confirmation_overlap_status_mismatch")
        return self


class ConditionConfirmationAssessmentV1(ContractModel):
    assessment_version: Literal["condition-confirmation-assessment-v1"] = (
        "condition-confirmation-assessment-v1"
    )
    model: ConditionConfirmationFrozenModelV1
    model_sha256: str
    plan_sha256: str
    target_sha256: str
    claim_spec_sha256: str
    question_config_sha256: str
    corpus_snapshot_sha256: str
    corpus_cutoff: str
    claim_contrast_id: str | None
    config_sha256: str
    pipeline_sha256: str
    full_graph_sha256: str
    development_graph_sha256: str
    confirmation_graph_sha256: str
    confirmation_effect_input_sha256: str
    overlap_checks: ConditionSplitOverlapChecksV1
    predictions: list[ConditionSignPredictionV1]
    brier_comparison: PairedComponentBootstrapV1 | None
    polarity_replication: OppositePolarityReplicationV1 | None
    confirmation_omnibus: dict[str, JsonValue] | None
    confirmation_omnibus_passed: bool | None
    status: Literal["confirmed", "not_confirmed", "insufficient"]
    reasons: list[str]
    interpretation: Literal[
        "held-out predictive association confirmation; not causal proof or scientific truth"
    ] = (
        "held-out predictive association confirmation; not causal proof or scientific truth"
    )
    assessment_sha256: str

    @field_validator(
        "model_sha256",
        "plan_sha256",
        "target_sha256",
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "config_sha256",
        "pipeline_sha256",
        "full_graph_sha256",
        "development_graph_sha256",
        "confirmation_graph_sha256",
        "confirmation_effect_input_sha256",
        "assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        return _sorted_unique(value, "assessment_reasons")

    @model_validator(mode="after")
    def validate_assessment(self) -> ConditionConfirmationAssessmentV1:
        plan = self.model.plan
        if self.model_sha256 != self.model.model_sha256:
            raise ValueError("condition_confirmation_assessment_model_mismatch")
        if (
            self.plan_sha256 != plan.plan_sha256
            or self.target_sha256 != plan.target_sha256
            or self.claim_spec_sha256 != plan.claim_spec_sha256
            or self.question_config_sha256 != plan.question_config_sha256
            or self.corpus_snapshot_sha256 != plan.corpus_snapshot_sha256
            or self.corpus_cutoff != plan.corpus_cutoff
            or self.claim_contrast_id != plan.claim_contrast_id
            or self.config_sha256 != plan.config_sha256
            or self.pipeline_sha256 != plan.pipeline_sha256
            or self.full_graph_sha256 != plan.full_graph_sha256
            or self.development_graph_sha256 != plan.development_graph_sha256
            or self.confirmation_graph_sha256 != plan.confirmation_graph_sha256
        ):
            raise ValueError("condition_confirmation_assessment_context_mismatch")
        prediction_ids = [row.cohort_id for row in self.predictions]
        if prediction_ids != sorted(set(prediction_ids)):
            raise ValueError("condition_confirmation_predictions_not_cohort_unique")
        gates_present = (
            self.brier_comparison is not None
            and self.polarity_replication is not None
            and self.confirmation_omnibus is not None
            and self.confirmation_omnibus_passed is not None
        )
        all_pass = (
            gates_present
            and self.brier_comparison is not None
            and self.brier_comparison.passed
            and self.polarity_replication is not None
            and self.polarity_replication.passed
            and self.confirmation_omnibus_passed is True
        )
        if self.status == "confirmed":
            if self.reasons or not all_pass:
                raise ValueError("condition_confirmation_confirmed_gate_mismatch")
        elif self.status == "not_confirmed":
            if not self.reasons or not gates_present or all_pass:
                raise ValueError("condition_confirmation_not_confirmed_gate_mismatch")
        elif not self.reasons:
            raise ValueError("condition_confirmation_insufficient_requires_reason")
        payload = self.model_dump(mode="json", exclude={"assessment_sha256"})
        if hash_canonical(payload) != self.assessment_sha256:
            raise ValueError("condition_confirmation_assessment_hash_mismatch")
        return self


def _overlap_checks(plan: ConditionConfirmationPlanV1) -> ConditionSplitOverlapChecksV1:
    development = [row for row in plan.component_assignments if row.split == "development"]
    confirmation = [row for row in plan.component_assignments if row.split == "confirmation"]

    def values(rows: Sequence[ConditionComponentAssignmentV1], name: str) -> set[str]:
        return {value for row in rows for value in getattr(row, name)}

    payload = {
        name: sorted(values(development, name) & values(confirmation, name))
        for name in (
            "publication_ids",
            "paper_ids",
            "study_ids",
            "cohort_ids",
            "strong_identity_tokens",
        )
    }
    payload["component_ids"] = sorted(
        {row.component_id for row in development}
        & {row.component_id for row in confirmation}
    )
    return ConditionSplitOverlapChecksV1(
        **payload,
        passed=all(not value for value in payload.values()),
    )


def _prediction_rows(
    *,
    model: ConditionConfirmationFrozenModelV1,
    effects: Sequence[HarmonizedEffect],
    cohort_ids: Sequence[str],
) -> tuple[list[ConditionSignPredictionV1], list[str]]:
    assert model.unconditional is not None
    assert model.conditional is not None
    assert model.selected_moderator is not None
    aggregation = aggregate_one_effect_per_cohort(
        effects,
        cohort_ids,
        assumed_within_cohort_correlation=(
            model.plan.config.assumed_within_cohort_correlation
        ),
    )
    if aggregation.status != "ok":
        return [], [f"confirmation_cohort_aggregation_insufficient:{aggregation.reason}"]
    component_by_cohort = _component_by_cohort(model.plan.component_assignments)
    rows: list[ConditionSignPredictionV1] = []
    reasons: list[str] = []
    normal = NormalDist()
    for effect in aggregation.effects:
        if model.selected_moderator in effect.moderator_conflicts:
            reasons.append(f"confirmation_moderator_conflict:{effect.cohort_id}")
            continue
        value = effect.moderators.get(model.selected_moderator)
        if value is None:
            reasons.append(f"confirmation_moderator_missing:{effect.cohort_id}")
            continue
        level = _moderator_level_label(value)
        level_parameters = model.conditional.parameters_for_level(level)
        if level_parameters is None:
            reasons.append(f"confirmation_moderator_level_unseen:{effect.cohort_id}:{level}")
            continue
        if effect.estimate == 0:
            reasons.append(f"confirmation_exact_zero_sign_ambiguous:{effect.cohort_id}")
            continue
        conditional_variance = (
            effect.variance
            + model.conditional.tau_squared
            + level_parameters.fitted_mean_variance
        )
        unconditional_variance = (
            effect.variance
            + model.unconditional.tau_squared
            + model.unconditional.mean_variance
        )
        if conditional_variance <= 0 or unconditional_variance <= 0:
            reasons.append(f"confirmation_predictive_variance_invalid:{effect.cohort_id}")
            continue
        conditional_probability = normal.cdf(
            level_parameters.fitted_mean / math.sqrt(conditional_variance)
        )
        unconditional_probability = normal.cdf(
            model.unconditional.mean / math.sqrt(unconditional_variance)
        )
        observed = effect.estimate > 0
        target = 1.0 if observed else 0.0
        conditional_brier = (conditional_probability - target) ** 2
        unconditional_brier = (unconditional_probability - target) ** 2
        rows.append(
            ConditionSignPredictionV1(
                cohort_id=effect.cohort_id,
                component_id=component_by_cohort[effect.cohort_id].component_id,
                moderator_level=level,
                estimate=effect.estimate,
                sampling_variance=effect.variance,
                observed_positive=observed,
                conditional_positive_probability=conditional_probability,
                unconditional_positive_probability=unconditional_probability,
                conditional_brier=conditional_brier,
                unconditional_brier=unconditional_brier,
                paired_brier_difference=conditional_brier - unconditional_brier,
            )
        )
    return sorted(rows, key=lambda row: row.cohort_id), sorted(set(reasons))


def _bootstrap_brier(
    *,
    model: ConditionConfirmationFrozenModelV1,
    predictions: Sequence[ConditionSignPredictionV1],
) -> PairedComponentBootstrapV1:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        grouped[row.component_id].append(row.paired_brier_difference)
    component_means = [
        ComponentBrierMeanV1(
            component_id=component_id,
            prediction_count=len(values),
            mean_paired_brier_difference=math.fsum(values) / len(values),
        )
        for component_id, values in sorted(grouped.items())
    ]
    means = np.asarray(
        [row.mean_paired_brier_difference for row in component_means],
        dtype=float,
    )
    seed_bytes = hashlib.sha256(
        model.plan_sha256.encode("ascii")
        + b"\0"
        + model.model_sha256.encode("ascii")
        + b"\0"
        + BOOTSTRAP_PROTOCOL.encode("ascii")
    ).digest()
    seed = int.from_bytes(seed_bytes[:8], "big")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(means),
        size=(model.plan.config.bootstrap_replicates, len(means)),
    )
    bootstrap_means = means[draws].mean(axis=1)
    upper = float(
        np.quantile(
            bootstrap_means,
            model.plan.config.bootstrap_upper_quantile,
            method=model.plan.config.bootstrap_quantile_method,
        )
    )
    return PairedComponentBootstrapV1(
        component_count=len(component_means),
        component_means=component_means,
        point_delta_brier=float(means.mean()),
        seed=seed,
        one_sided_upper_bound=upper,
        minimum_required_improvement=model.plan.config.min_brier_improvement,
        passed=upper < -model.plan.config.min_brier_improvement,
    )


def _polarity_replication(
    *,
    model: ConditionConfirmationFrozenModelV1,
    effects: Sequence[HarmonizedEffect],
    cohort_ids: Sequence[str],
    predictions: Sequence[ConditionSignPredictionV1],
) -> tuple[OppositePolarityReplicationV1 | None, list[str]]:
    assert model.selected_moderator is not None
    assert model.frozen_positive_level is not None
    assert model.frozen_negative_level is not None
    component_by_cohort = _component_by_cohort(model.plan.component_assignments)
    levels_by_component: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        if row.moderator_level in {
            model.frozen_positive_level,
            model.frozen_negative_level,
        }:
            levels_by_component[row.component_id].add(row.moderator_level)
    overlap = sorted(
        component_id
        for component_id, levels in levels_by_component.items()
        if {
            model.frozen_positive_level,
            model.frozen_negative_level,
        }
        <= levels
    )
    if overlap:
        return None, [f"confirmation_component_contains_both_frozen_levels:{overlap}"]

    by_polarity: dict[str, tuple[list[HarmonizedEffect], list[str], str]] = {}
    for polarity, frozen_level in (
        ("positive", model.frozen_positive_level),
        ("negative", model.frozen_negative_level),
    ):
        selected_effects: list[HarmonizedEffect] = []
        selected_components: list[str] = []
        for effect, cohort_id in zip(effects, cohort_ids, strict=True):
            value = effect.moderators.get(model.selected_moderator)
            if value is None or _moderator_level_label(value) != frozen_level:
                continue
            selected_effects.append(effect)
            selected_components.append(component_by_cohort[cohort_id].component_id)
        by_polarity[polarity] = (
            selected_effects,
            selected_components,
            frozen_level,
        )
    reasons: list[str] = []
    for polarity, (_, component_ids, _) in by_polarity.items():
        count = len(set(component_ids))
        if count < model.plan.config.confirmation_min_components_per_polarity:
            reasons.append(
                f"confirmation_{polarity}_level_components_below_minimum:"
                f"{count}<{model.plan.config.confirmation_min_components_per_polarity}"
            )
    if reasons:
        return None, sorted(reasons)

    level_results: dict[str, PolarityLevelReplicationV1] = {}
    confidence = 1.0 - model.plan.config.confirmation_familywise_alpha / 2
    for polarity in ("positive", "negative"):
        selected_effects, component_ids, frozen_level = by_polarity[polarity]
        synthesis = cohort_random_effects_meta_analysis(
            selected_effects,
            component_ids,
            confidence_level=confidence,
            assumed_within_cohort_correlation=(
                model.plan.config.assumed_within_cohort_correlation
            ),
        )
        passed = (
            synthesis.get("status") == "ok"
            and (
                float(synthesis["ci_lower"]) > 0
                if polarity == "positive"
                else float(synthesis["ci_upper"]) < 0
            )
        )
        level_results[polarity] = PolarityLevelReplicationV1(
            polarity=polarity,  # type: ignore[arg-type]
            frozen_level=frozen_level,
            component_ids=sorted(set(component_ids)),
            component_count=len(set(component_ids)),
            synthesis=synthesis,
            passed=passed,
        )
    positive = level_results["positive"]
    negative = level_results["negative"]
    return (
        OppositePolarityReplicationV1(
            positive=positive,
            negative=negative,
            components_containing_both_frozen_levels=[],
            passed=positive.passed and negative.passed,
        ),
        [],
    )


def _confirmation_omnibus(
    *,
    model: ConditionConfirmationFrozenModelV1,
    effects: Sequence[HarmonizedEffect],
    cohort_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, bool | None, list[str]]:
    assert model.selected_moderator is not None
    component_by_cohort = _component_by_cohort(model.plan.component_assignments)
    component_ids = [component_by_cohort[value].component_id for value in cohort_ids]
    result = cohort_categorical_meta_regression(
        effects,
        component_ids,
        model.selected_moderator,
        confidence_level=0.95,
        min_cohorts_per_level=model.plan.config.development_min_components_per_level,
        assumed_within_cohort_correlation=(
            model.plan.config.assumed_within_cohort_correlation
        ),
    )
    if result.get("status") != "ok":
        return (
            None,
            None,
            [f"confirmation_moderator_omnibus_insufficient:{result.get('reason', 'unknown')}"],
        )
    p_value = float(result["omnibus"]["p_value"])
    return result, p_value < model.plan.config.confirmation_familywise_alpha, []


def _freeze_assessment(
    *,
    model: ConditionConfirmationFrozenModelV1,
    confirmation_effect_input_sha256: str,
    overlap_checks: ConditionSplitOverlapChecksV1,
    predictions: Sequence[ConditionSignPredictionV1],
    status: Literal["confirmed", "not_confirmed", "insufficient"],
    reasons: Sequence[str],
    brier_comparison: PairedComponentBootstrapV1 | None = None,
    polarity_replication: OppositePolarityReplicationV1 | None = None,
    confirmation_omnibus: Mapping[str, Any] | None = None,
    confirmation_omnibus_passed: bool | None = None,
) -> ConditionConfirmationAssessmentV1:
    plan = model.plan
    payload: dict[str, Any] = {
        "assessment_version": "condition-confirmation-assessment-v1",
        "model": model,
        "model_sha256": model.model_sha256,
        "plan_sha256": plan.plan_sha256,
        "target_sha256": plan.target_sha256,
        "claim_spec_sha256": plan.claim_spec_sha256,
        "question_config_sha256": plan.question_config_sha256,
        "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
        "corpus_cutoff": plan.corpus_cutoff,
        "claim_contrast_id": plan.claim_contrast_id,
        "config_sha256": plan.config_sha256,
        "pipeline_sha256": plan.pipeline_sha256,
        "full_graph_sha256": plan.full_graph_sha256,
        "development_graph_sha256": plan.development_graph_sha256,
        "confirmation_graph_sha256": plan.confirmation_graph_sha256,
        "confirmation_effect_input_sha256": confirmation_effect_input_sha256,
        "overlap_checks": overlap_checks,
        "predictions": sorted(predictions, key=lambda row: row.cohort_id),
        "brier_comparison": brier_comparison,
        "polarity_replication": polarity_replication,
        "confirmation_omnibus": (
            None if confirmation_omnibus is None else dict(confirmation_omnibus)
        ),
        "confirmation_omnibus_passed": confirmation_omnibus_passed,
        "status": status,
        "reasons": sorted(set(reasons)),
        "interpretation": (
            "held-out predictive association confirmation; not causal proof or "
            "scientific truth"
        ),
    }
    return ConditionConfirmationAssessmentV1.model_validate(
        {**payload, "assessment_sha256": hash_canonical(payload)}
    )


def confirm_condition_dependence(
    *,
    plan: ConditionConfirmationPlanV1,
    model: ConditionConfirmationFrozenModelV1,
    full_graph: EvidenceGraph,
    current_pipeline_sha256: str,
) -> ConditionConfirmationAssessmentV1:
    """Evaluate the frozen model exactly once on its held-out graph partition.

    The caller must keep ``full_graph`` inaccessible until the plan and fitted model
    have been externally anchored.  This function validates every frozen partition,
    reruns the development fit, and uses only confirmation components for all gates.
    """

    try:
        plan = ConditionConfirmationPlanV1.model_validate(plan.model_dump(mode="json"))
        model = ConditionConfirmationFrozenModelV1.model_validate(
            model.model_dump(mode="json")
        )
        full_graph = EvidenceGraph.model_validate(full_graph.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError(
            "condition_confirmation_assessment_input_tampered"
        ) from exc
    _sha256(current_pipeline_sha256, "current_pipeline_sha256")
    if current_pipeline_sha256 != plan.pipeline_sha256:
        raise ConditionConfirmationError(
            "condition_confirmation_assessment_pipeline_hash_mismatch"
        )
    if model.plan_sha256 != plan.plan_sha256 or model.plan != plan:
        raise ConditionConfirmationError("condition_confirmation_assessment_plan_model_mismatch")

    _validate_graph_projection(
        graph=full_graph,
        plan=plan,
        partition=plan.full_partition,
        expected_graph_sha256=plan.full_graph_sha256,
    )
    development_graph, confirmation_graph = partition_full_graph_for_plan(full_graph, plan)
    _validate_graph_projection(
        graph=development_graph,
        plan=plan,
        partition=plan.development_partition,
        expected_graph_sha256=plan.development_graph_sha256,
    )
    _validate_graph_projection(
        graph=confirmation_graph,
        plan=plan,
        partition=plan.confirmation_partition,
        expected_graph_sha256=plan.confirmation_graph_sha256,
    )
    validate_condition_confirmation_model(
        plan=plan,
        development_graph=development_graph,
        model=model,
        current_pipeline_sha256=current_pipeline_sha256,
    )

    effects, cohort_ids, effect_reasons = _target_effects(confirmation_graph, plan)
    effect_input_sha256 = hash_canonical(
        {
            "effects": effects,
            "cohort_ids": cohort_ids,
            "confirmation_graph_sha256": plan.confirmation_graph_sha256,
        }
    )
    overlap_checks = _overlap_checks(plan)
    reasons = [*effect_reasons]
    if not overlap_checks.passed:
        reasons.append("development_confirmation_identity_overlap")
    if model.status != "fitted":
        reasons.extend(
            f"development_model_insufficient:{reason}"
            for reason in model.insufficiency_reasons
        )
    if reasons:
        return _freeze_assessment(
            model=model,
            confirmation_effect_input_sha256=effect_input_sha256,
            overlap_checks=overlap_checks,
            predictions=[],
            status="insufficient",
            reasons=reasons,
        )

    predictions, prediction_reasons = _prediction_rows(
        model=model,
        effects=effects,
        cohort_ids=cohort_ids,
    )
    predicted_components = {row.component_id for row in predictions}
    expected_components = _target_component_ids(
        roster=plan.roster,
        assignments=plan.component_assignments,
        split="confirmation",
    )
    if predicted_components != expected_components:
        missing = sorted(expected_components - predicted_components)
        unexpected = sorted(predicted_components - expected_components)
        prediction_reasons.append(
            "confirmation_prediction_component_partition_mismatch:"
            f"missing={missing}:unexpected={unexpected}"
        )
    if len(predicted_components) < plan.config.confirmation_min_components_total:
        prediction_reasons.append(
            "confirmation_prediction_components_below_minimum:"
            f"{len(predicted_components)}<"
            f"{plan.config.confirmation_min_components_total}"
        )
    if prediction_reasons:
        return _freeze_assessment(
            model=model,
            confirmation_effect_input_sha256=effect_input_sha256,
            overlap_checks=overlap_checks,
            predictions=predictions,
            status="insufficient",
            reasons=prediction_reasons,
        )

    polarity, polarity_reasons = _polarity_replication(
        model=model,
        effects=effects,
        cohort_ids=cohort_ids,
        predictions=predictions,
    )
    omnibus, omnibus_passed, omnibus_reasons = _confirmation_omnibus(
        model=model,
        effects=effects,
        cohort_ids=cohort_ids,
    )
    support_reasons = [*polarity_reasons, *omnibus_reasons]
    if support_reasons or polarity is None or omnibus is None or omnibus_passed is None:
        return _freeze_assessment(
            model=model,
            confirmation_effect_input_sha256=effect_input_sha256,
            overlap_checks=overlap_checks,
            predictions=predictions,
            status="insufficient",
            reasons=(support_reasons or ["confirmation_gate_output_incomplete"]),
        )

    brier = _bootstrap_brier(model=model, predictions=predictions)
    gate_reasons: list[str] = []
    if not brier.passed:
        gate_reasons.append("paired_component_brier_improvement_not_confirmed")
    if not polarity.passed:
        gate_reasons.append("opposite_polarity_replication_not_confirmed")
    if not omnibus_passed:
        gate_reasons.append("confirmation_moderator_omnibus_not_confirmed")
    return _freeze_assessment(
        model=model,
        confirmation_effect_input_sha256=effect_input_sha256,
        overlap_checks=overlap_checks,
        predictions=predictions,
        brier_comparison=brier,
        polarity_replication=polarity,
        confirmation_omnibus=omnibus,
        confirmation_omnibus_passed=omnibus_passed,
        status="not_confirmed" if gate_reasons else "confirmed",
        reasons=gate_reasons,
    )


def validate_condition_confirmation_assessment(
    *,
    plan: ConditionConfirmationPlanV1,
    model: ConditionConfirmationFrozenModelV1,
    full_graph: EvidenceGraph,
    assessment: ConditionConfirmationAssessmentV1,
    current_pipeline_sha256: str,
) -> ConditionConfirmationAssessmentV1:
    """Recompute split, development model, predictions, and every held-out gate."""

    expected = confirm_condition_dependence(
        plan=plan,
        model=model,
        full_graph=full_graph,
        current_pipeline_sha256=current_pipeline_sha256,
    )
    try:
        observed = ConditionConfirmationAssessmentV1.model_validate(
            assessment.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ConditionConfirmationError("condition_confirmation_assessment_tampered") from exc
    if observed != expected:
        raise ConditionConfirmationError(
            "condition_confirmation_assessment_recomputation_mismatch"
        )
    return observed


__all__ = [
    "BOOTSTRAP_PROTOCOL",
    "SPLIT_ALGORITHM",
    "SPLIT_SALT",
    "CategoricalNormalPredictiveModelV1",
    "ConditionComponentAssignmentV1",
    "ConditionConfirmationAssessmentV1",
    "ConditionConfirmationConfigV1",
    "ConditionConfirmationError",
    "ConditionConfirmationFrozenModelV1",
    "ConditionConfirmationMaterializationReceiptV1",
    "ConditionConfirmationPlanV1",
    "ConditionConfirmationTargetV1",
    "ConditionGraphPartitionV1",
    "ConditionSignPredictionV1",
    "LabelFreeGraphRosterV1",
    "RosterArmV1",
    "RosterCohortV1",
    "RosterContrastV1",
    "RosterEstimateV1",
    "RosterPublicationV1",
    "RosterSpanV1",
    "RosterStudyV1",
    "confirm_condition_dependence",
    "derive_condition_components",
    "fit_condition_confirmation_model",
    "freeze_condition_confirmation_config",
    "freeze_condition_confirmation_target",
    "freeze_label_free_graph_roster",
    "materialize_condition_confirmation_inputs",
    "partition_evidence_graph",
    "partition_full_graph_for_plan",
    "prepare_condition_confirmation_plan",
    "validate_condition_confirmation_assessment",
    "validate_condition_confirmation_materialization",
    "validate_condition_confirmation_model",
]
