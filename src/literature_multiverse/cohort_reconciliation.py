"""Conservative cross-publication study and cohort identity reconciliation.

Native extraction deliberately assigns publication-scoped identifiers.  This module
creates a separate, hash-bound derived graph in which reports of the same study/cohort
can share an identity.  Free-text labels are retained as provenance but are never used
as matching keys.  Automatic matching is limited to exact normalized registry or
dataset identifiers; ambiguous or conflicting components remain unmerged until a
reviewer supplies a complete reconciliation artifact.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.evidence_graph import (
    ArmNode,
    CohortIdentity,
    CohortIdentityBasis,
    CohortNode,
    ContrastNode,
    EvidenceGraph,
    OutcomeEstimateNode,
    RiskOfBiasAssessment,
    StudyNode,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


class NativeCohortReconciliationError(ValueError):
    """A reconciliation artifact or derived graph cannot be trusted."""


class ReconciliationNodeKind(StrEnum):
    STUDY = "study"
    COHORT = "cohort"


class ReconciliationGroupBasis(StrEnum):
    SINGLETON = "singleton"
    EXACT_REGISTRY_ID = "exact_registry_id"
    EXACT_DATASET_ID = "exact_dataset_id"
    EXACT_REGISTRY_AND_DATASET_ID = "exact_registry_and_dataset_id"
    IMPLIED_BY_RECONCILED_COHORT = "implied_by_reconciled_cohort"
    REVIEWER_RECONCILED = "reviewer_reconciled"


class NativeReconciliationStatus(StrEnum):
    NO_ESTIMABLE_GRAPH = "no_estimable_graph"
    SINGLE_PUBLICATION_COMPLETE = "single_publication_complete"
    STRONG_IDENTIFIER_RECONCILED_LIMITED = "strong_identifier_reconciled_limited"
    REQUIRES_REVIEWER = "requires_reviewer"
    REVIEWER_COMPLETE = "reviewer_complete"


class ReviewerIdentityGroup(ContractModel):
    """One reviewer-adjudicated identity group over publication-scoped node IDs."""

    member_ids: Annotated[list[str], Field(min_length=1)]
    rationale: Annotated[str, Field(min_length=1)]

    @field_validator("member_ids")
    @classmethod
    def validate_members(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("reviewer_identity_group_members_not_sorted_unique")
        return value

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reviewer_identity_group_rationale_empty")
        return value.strip()


class ReviewerCohortReconciliationArtifact(ContractModel):
    """External reviewer receipt that explicitly partitions every study and cohort."""

    artifact_version: Literal["reviewer-cohort-reconciliation-v1"] = (
        "reviewer-cohort-reconciliation-v1"
    )
    input_corpus_sha256: str
    input_graph_sha256: str
    reviewer_identity_sha256: str
    review_protocol_sha256: str
    completed_at: datetime
    all_studies_and_cohorts_reviewed: Literal[True] = True
    study_groups: Annotated[list[ReviewerIdentityGroup], Field(min_length=1)]
    cohort_groups: Annotated[list[ReviewerIdentityGroup], Field(min_length=1)]
    artifact_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "input_graph_sha256",
        "reviewer_identity_sha256",
        "review_protocol_sha256",
        "artifact_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("reviewer_reconciliation_sha256_invalid")
        return value

    @field_validator("study_groups", "cohort_groups")
    @classmethod
    def validate_group_order(
        cls, value: list[ReviewerIdentityGroup]
    ) -> list[ReviewerIdentityGroup]:
        member_tuples = [tuple(group.member_ids) for group in value]
        if member_tuples != sorted(member_tuples) or len(member_tuples) != len(set(member_tuples)):
            raise ValueError("reviewer_reconciliation_groups_not_sorted_unique")
        flattened = [member for group in value for member in group.member_ids]
        if len(flattened) != len(set(flattened)):
            raise ValueError("reviewer_reconciliation_groups_overlap")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> ReviewerCohortReconciliationArtifact:
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("reviewer_reconciliation_completed_at_requires_timezone")
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if hash_canonical(payload) != self.artifact_sha256:
            raise ValueError("reviewer_reconciliation_artifact_hash_mismatch")
        return self


class NativeIdentityCandidate(ContractModel):
    """A connected candidate component produced only from strong identifiers."""

    node_kind: ReconciliationNodeKind
    member_ids: Annotated[list[str], Field(min_length=2)]
    matched_registry_ids: list[str] = Field(default_factory=list)
    matched_dataset_ids: list[str] = Field(default_factory=list)
    candidate_sha256: str

    @field_validator("member_ids", "matched_registry_ids", "matched_dataset_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_identity_candidate_values_not_sorted_unique")
        return value

    @field_validator("candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_identity_candidate_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_candidate(self) -> NativeIdentityCandidate:
        if not self.matched_registry_ids and not self.matched_dataset_ids:
            raise ValueError("native_identity_candidate_requires_strong_identifier")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("native_identity_candidate_hash_mismatch")
        return self


class NativeReconciliationIssue(ContractModel):
    code: Annotated[str, Field(min_length=1)]
    node_kind: ReconciliationNodeKind
    node_ids: Annotated[list[str], Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]
    resolved_by_reviewer: bool = False

    @field_validator("node_ids")
    @classmethod
    def validate_node_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_reconciliation_issue_ids_not_sorted_unique")
        return value


class NativeReconciledIdentityGroup(ContractModel):
    node_kind: ReconciliationNodeKind
    canonical_id: Annotated[str, Field(min_length=1)]
    member_ids: Annotated[list[str], Field(min_length=1)]
    basis: ReconciliationGroupBasis
    matched_registry_ids: list[str] = Field(default_factory=list)
    matched_dataset_ids: list[str] = Field(default_factory=list)
    reviewer_rationale: str | None = None

    @field_validator("member_ids", "matched_registry_ids", "matched_dataset_ids")
    @classmethod
    def validate_lists(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_reconciled_identity_group_values_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_basis(self) -> NativeReconciledIdentityGroup:
        if len(self.member_ids) == 1:
            if self.basis is not ReconciliationGroupBasis.SINGLETON:
                raise ValueError("singleton_reconciliation_group_requires_singleton_basis")
            if self.canonical_id != self.member_ids[0]:
                raise ValueError("singleton_reconciliation_group_must_preserve_id")
        elif self.basis is ReconciliationGroupBasis.SINGLETON:
            raise ValueError("merged_reconciliation_group_forbids_singleton_basis")
        if self.basis is ReconciliationGroupBasis.EXACT_REGISTRY_ID and not (
            self.matched_registry_ids and not self.matched_dataset_ids
        ):
            raise ValueError("registry_reconciliation_basis_mismatch")
        if self.basis is ReconciliationGroupBasis.EXACT_DATASET_ID and not (
            self.matched_dataset_ids and not self.matched_registry_ids
        ):
            raise ValueError("dataset_reconciliation_basis_mismatch")
        if self.basis is ReconciliationGroupBasis.EXACT_REGISTRY_AND_DATASET_ID and not (
            self.matched_registry_ids and self.matched_dataset_ids
        ):
            raise ValueError("registry_dataset_reconciliation_basis_mismatch")
        if self.basis is ReconciliationGroupBasis.REVIEWER_RECONCILED:
            if self.reviewer_rationale is None or not self.reviewer_rationale.strip():
                raise ValueError("reviewer_reconciled_group_requires_rationale")
        elif self.reviewer_rationale is not None:
            raise ValueError("nonreviewer_reconciled_group_forbids_rationale")
        return self


class NativeCohortReconciliationReceipt(ContractModel):
    """Hash-bound mapping and graph derived from a native typed corpus."""

    receipt_version: Literal["native-cohort-reconciliation-receipt-v1"] = (
        "native-cohort-reconciliation-receipt-v1"
    )
    input_corpus_sha256: str
    input_graph_sha256: str | None
    reviewer_artifact: ReviewerCohortReconciliationArtifact | None = None
    status: NativeReconciliationStatus
    cross_publication_identity_assurance_complete: bool
    candidates: list[NativeIdentityCandidate] = Field(default_factory=list)
    issues: list[NativeReconciliationIssue] = Field(default_factory=list)
    study_groups: list[NativeReconciledIdentityGroup] = Field(default_factory=list)
    cohort_groups: list[NativeReconciledIdentityGroup] = Field(default_factory=list)
    reconciled_graph: EvidenceGraph | None
    reconciled_graph_sha256: str | None
    merged_study_groups: Annotated[int, Field(ge=0)]
    merged_cohort_groups: Annotated[int, Field(ge=0)]
    receipt_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "input_graph_sha256",
        "reconciled_graph_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("native_cohort_reconciliation_sha256_invalid")
        return value

    @field_validator("candidates")
    @classmethod
    def validate_candidates(
        cls, value: list[NativeIdentityCandidate]
    ) -> list[NativeIdentityCandidate]:
        keys = [(item.node_kind.value, tuple(item.member_ids)) for item in value]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("native_cohort_reconciliation_candidates_not_sorted_unique")
        return value

    @field_validator("issues")
    @classmethod
    def validate_issues(
        cls, value: list[NativeReconciliationIssue]
    ) -> list[NativeReconciliationIssue]:
        keys = [
            (item.node_kind.value, item.code, tuple(item.node_ids), item.resolved_by_reviewer)
            for item in value
        ]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("native_cohort_reconciliation_issues_not_sorted_unique")
        return value

    @field_validator("study_groups", "cohort_groups")
    @classmethod
    def validate_groups(
        cls, value: list[NativeReconciledIdentityGroup]
    ) -> list[NativeReconciledIdentityGroup]:
        keys = [tuple(item.member_ids) for item in value]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("native_cohort_reconciliation_groups_not_sorted_unique")
        members = [member for item in value for member in item.member_ids]
        if len(members) != len(set(members)):
            raise ValueError("native_cohort_reconciliation_groups_overlap")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativeCohortReconciliationReceipt:
        if any(group.node_kind is not ReconciliationNodeKind.STUDY for group in self.study_groups):
            raise ValueError("native_reconciliation_study_group_kind_mismatch")
        if any(
            group.node_kind is not ReconciliationNodeKind.COHORT for group in self.cohort_groups
        ):
            raise ValueError("native_reconciliation_cohort_group_kind_mismatch")
        if self.status is NativeReconciliationStatus.NO_ESTIMABLE_GRAPH:
            if self.input_graph_sha256 is None or self.reconciled_graph is None:
                raise ValueError("no_effect_reconciliation_requires_publication_graph")
            if any((self.study_groups, self.cohort_groups, self.candidates, self.issues)):
                raise ValueError("no_effect_reconciliation_forbids_identity_records")
        else:
            if self.input_graph_sha256 is None or self.reconciled_graph is None:
                raise ValueError("graph_reconciliation_requires_graph")
        if (self.reconciled_graph is None) != (self.reconciled_graph_sha256 is None):
            raise ValueError("native_reconciled_graph_hash_presence_mismatch")
        if (
            self.reconciled_graph is not None
            and hash_canonical(self.reconciled_graph) != self.reconciled_graph_sha256
        ):
            raise ValueError("native_reconciled_graph_hash_mismatch")
        expected_complete = self.status in {
            NativeReconciliationStatus.NO_ESTIMABLE_GRAPH,
            NativeReconciliationStatus.SINGLE_PUBLICATION_COMPLETE,
            NativeReconciliationStatus.REVIEWER_COMPLETE,
        }
        if self.cross_publication_identity_assurance_complete != expected_complete:
            raise ValueError("native_reconciliation_assurance_status_mismatch")
        if self.status is NativeReconciliationStatus.REVIEWER_COMPLETE:
            if self.reviewer_artifact is None:
                raise ValueError("reviewer_complete_reconciliation_requires_artifact")
            if any(not item.resolved_by_reviewer for item in self.issues):
                raise ValueError("reviewer_complete_reconciliation_has_unresolved_issue")
        elif self.reviewer_artifact is not None:
            raise ValueError("nonreviewer_reconciliation_forbids_artifact")
        if self.merged_study_groups != sum(
            len(group.member_ids) > 1 for group in self.study_groups
        ):
            raise ValueError("native_reconciliation_merged_study_count_mismatch")
        if self.merged_cohort_groups != sum(
            len(group.member_ids) > 1 for group in self.cohort_groups
        ):
            raise ValueError("native_reconciliation_merged_cohort_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("native_cohort_reconciliation_receipt_hash_mismatch")
        return self


def freeze_reviewer_cohort_reconciliation_artifact(
    *,
    corpus: TypedEvidenceCorpus,
    reviewer_identity_sha256: str,
    review_protocol_sha256: str,
    completed_at: datetime,
    study_groups: list[ReviewerIdentityGroup],
    cohort_groups: list[ReviewerIdentityGroup],
) -> ReviewerCohortReconciliationArtifact:
    """Freeze an external reviewer partition against exact native corpus bytes."""

    if not corpus.graph.outcome_estimates:
        raise NativeCohortReconciliationError("reviewer_reconciliation_requires_estimable_graph")
    payload = {
        "artifact_version": "reviewer-cohort-reconciliation-v1",
        "input_corpus_sha256": corpus.corpus_sha256,
        "input_graph_sha256": hash_canonical(corpus.graph),
        "reviewer_identity_sha256": reviewer_identity_sha256,
        "review_protocol_sha256": review_protocol_sha256,
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "all_studies_and_cohorts_reviewed": True,
        "study_groups": study_groups,
        "cohort_groups": cohort_groups,
    }
    return ReviewerCohortReconciliationArtifact.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def _normalize_identifier(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _freeze_candidate(
    *,
    node_kind: ReconciliationNodeKind,
    member_ids: list[str],
    matched_registry_ids: list[str],
    matched_dataset_ids: list[str],
) -> NativeIdentityCandidate:
    payload = {
        "node_kind": node_kind,
        "member_ids": sorted(member_ids),
        "matched_registry_ids": sorted(matched_registry_ids),
        "matched_dataset_ids": sorted(matched_dataset_ids),
    }
    return NativeIdentityCandidate.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in sorted(self.parent):
            grouped[self.find(value)].append(value)
        return sorted((sorted(values) for values in grouped.values()), key=tuple)


def _strong_components(
    *,
    node_kind: ReconciliationNodeKind,
    node_ids: list[str],
    registry_ids: dict[str, set[str]],
    dataset_ids: dict[str, set[str]],
    publication_ids: dict[str, str],
) -> tuple[list[NativeIdentityCandidate], list[NativeReconciliationIssue], list[list[str]]]:
    union = _UnionFind(node_ids)
    registry_index: dict[str, list[str]] = defaultdict(list)
    dataset_index: dict[str, list[str]] = defaultdict(list)
    issues: list[NativeReconciliationIssue] = []
    for node_id in node_ids:
        for identifier in registry_ids[node_id]:
            if not identifier:
                issues.append(
                    NativeReconciliationIssue(
                        code="empty_registry_identifier",
                        node_kind=node_kind,
                        node_ids=[node_id],
                        detail="An empty registry identifier cannot support reconciliation.",
                    )
                )
            else:
                registry_index[identifier].append(node_id)
        for identifier in dataset_ids[node_id]:
            if not identifier:
                issues.append(
                    NativeReconciliationIssue(
                        code="empty_dataset_identifier",
                        node_kind=node_kind,
                        node_ids=[node_id],
                        detail="An empty dataset identifier cannot support reconciliation.",
                    )
                )
            else:
                dataset_index[identifier].append(node_id)
    for index in (registry_index, dataset_index):
        for members in index.values():
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    # Reconciliation is cross-publication only. Reusing one study
                    # registration across distinct cohorts in the same report never
                    # collapses those cohorts by itself.
                    if publication_ids[left] != publication_ids[right]:
                        union.union(left, right)

    components = [group for group in union.groups() if len(group) > 1]
    candidates: list[NativeIdentityCandidate] = []
    for members in components:
        matched_registry = sorted(
            identifier
            for identifier, ids in registry_index.items()
            if len(set(ids).intersection(members)) >= 2
        )
        matched_dataset = sorted(
            identifier
            for identifier, ids in dataset_index.items()
            if len(set(ids).intersection(members)) >= 2
        )
        candidates.append(
            _freeze_candidate(
                node_kind=node_kind,
                member_ids=members,
                matched_registry_ids=matched_registry,
                matched_dataset_ids=matched_dataset,
            )
        )
        publications = [publication_ids[member] for member in members]
        if len(publications) != len(set(publications)):
            issues.append(
                NativeReconciliationIssue(
                    code="ambiguous_many_to_many_identity",
                    node_kind=node_kind,
                    node_ids=members,
                    detail=(
                        "A strong identifier maps to multiple local identities in one "
                        "publication; automatic reconciliation is forbidden."
                    ),
                )
            )
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                left_registry = registry_ids[left]
                right_registry = registry_ids[right]
                left_dataset = dataset_ids[left]
                right_dataset = dataset_ids[right]
                if (
                    left_registry
                    and right_registry
                    and not left_registry.intersection(right_registry)
                ):
                    issues.append(
                        NativeReconciliationIssue(
                            code="conflicting_registry_identifiers",
                            node_kind=node_kind,
                            node_ids=[left, right],
                            detail=(
                                "Candidate identities carry disjoint non-empty registry "
                                "identifier sets."
                            ),
                        )
                    )
                if left_dataset and right_dataset and not left_dataset.intersection(right_dataset):
                    issues.append(
                        NativeReconciliationIssue(
                            code="conflicting_dataset_identifiers",
                            node_kind=node_kind,
                            node_ids=[left, right],
                            detail=(
                                "Candidate identities carry disjoint non-empty dataset "
                                "identifier sets."
                            ),
                        )
                    )
    candidates.sort(key=lambda item: (item.node_kind.value, tuple(item.member_ids)))
    issues = _sort_issues(issues)
    return candidates, issues, components


def _sort_issues(issues: list[NativeReconciliationIssue]) -> list[NativeReconciliationIssue]:
    unique: dict[tuple[str, str, tuple[str, ...], bool], NativeReconciliationIssue] = {}
    for issue in issues:
        key = (
            issue.node_kind.value,
            issue.code,
            tuple(issue.node_ids),
            issue.resolved_by_reviewer,
        )
        unique[key] = issue
    return [unique[key] for key in sorted(unique)]


def _partition_from_components(node_ids: list[str], components: list[list[str]]) -> list[list[str]]:
    merged = {member for component in components for member in component}
    return sorted(
        [*components, *([node_id] for node_id in node_ids if node_id not in merged)],
        key=tuple,
    )


def _validate_reviewer_partition(
    *,
    groups: list[ReviewerIdentityGroup],
    expected_ids: set[str],
    publication_ids: dict[str, str],
    node_kind: ReconciliationNodeKind,
) -> None:
    observed = {member for group in groups for member in group.member_ids}
    if observed != expected_ids:
        missing = sorted(expected_ids - observed)
        unknown = sorted(observed - expected_ids)
        raise NativeCohortReconciliationError(
            f"reviewer_{node_kind.value}_partition_mismatch:missing={missing}:unknown={unknown}"
        )
    for group in groups:
        publications = [publication_ids[member] for member in group.member_ids]
        if len(publications) != len(set(publications)):
            raise NativeCohortReconciliationError(
                f"reviewer_{node_kind.value}_group_merges_within_publication:{group.member_ids}"
            )


def _canonical_id(
    *,
    node_kind: ReconciliationNodeKind,
    member_ids: list[str],
) -> str:
    if len(member_ids) == 1:
        return member_ids[0]
    digest = hashlib.sha256(
        hash_canonical({"node_kind": node_kind.value, "member_ids": sorted(member_ids)}).encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    return f"{node_kind.value}-reconciled-{digest}"


def _common_or_none(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    return first if all(value == first for value in present[1:]) else None


def _common_risk_or_unassessed(values: list[RiskOfBiasAssessment]) -> RiskOfBiasAssessment:
    first = values[0]
    return first if all(value == first for value in values[1:]) else RiskOfBiasAssessment()


def _group_basis(
    *,
    member_ids: list[str],
    matched_registry: list[str],
    matched_dataset: list[str],
    implied_by_cohort: bool = False,
) -> ReconciliationGroupBasis:
    if len(member_ids) == 1:
        return ReconciliationGroupBasis.SINGLETON
    if matched_registry and matched_dataset:
        return ReconciliationGroupBasis.EXACT_REGISTRY_AND_DATASET_ID
    if matched_registry:
        return ReconciliationGroupBasis.EXACT_REGISTRY_ID
    if matched_dataset:
        return ReconciliationGroupBasis.EXACT_DATASET_ID
    if implied_by_cohort:
        return ReconciliationGroupBasis.IMPLIED_BY_RECONCILED_COHORT
    raise NativeCohortReconciliationError("merged_group_lacks_identity_basis")


def _freeze_groups(
    *,
    node_kind: ReconciliationNodeKind,
    partitions: list[list[str]],
    registry_ids: dict[str, set[str]],
    dataset_ids: dict[str, set[str]],
    reviewer_groups: dict[tuple[str, ...], ReviewerIdentityGroup] | None,
    candidate_by_members: dict[tuple[str, ...], NativeIdentityCandidate],
    implied_groups: set[tuple[str, ...]] | None = None,
) -> list[NativeReconciledIdentityGroup]:
    output: list[NativeReconciledIdentityGroup] = []
    implied_groups = implied_groups or set()
    for members in sorted(partitions, key=tuple):
        key = tuple(members)
        candidate = candidate_by_members.get(key)
        matched_registry = candidate.matched_registry_ids if candidate else []
        matched_dataset = candidate.matched_dataset_ids if candidate else []
        reviewer_group = reviewer_groups.get(key) if reviewer_groups is not None else None
        if len(members) == 1:
            basis = ReconciliationGroupBasis.SINGLETON
            rationale = None
        elif reviewer_group is not None:
            basis = ReconciliationGroupBasis.REVIEWER_RECONCILED
            rationale = reviewer_group.rationale
            # Preserve all shared strong signals even when the reviewer was required.
            matched_registry = sorted(
                identifier
                for identifier in set.union(*(registry_ids[member] for member in members))
                if sum(identifier in registry_ids[member] for member in members) >= 2
            )
            matched_dataset = sorted(
                identifier
                for identifier in set.union(*(dataset_ids[member] for member in members))
                if sum(identifier in dataset_ids[member] for member in members) >= 2
            )
        else:
            basis = _group_basis(
                member_ids=members,
                matched_registry=matched_registry,
                matched_dataset=matched_dataset,
                implied_by_cohort=key in implied_groups,
            )
            rationale = None
        output.append(
            NativeReconciledIdentityGroup(
                node_kind=node_kind,
                canonical_id=_canonical_id(node_kind=node_kind, member_ids=members),
                member_ids=members,
                basis=basis,
                matched_registry_ids=matched_registry,
                matched_dataset_ids=matched_dataset,
                reviewer_rationale=rationale,
            )
        )
    return output


def _rewrite_graph(
    graph: EvidenceGraph,
    *,
    study_groups: list[NativeReconciledIdentityGroup],
    cohort_groups: list[NativeReconciledIdentityGroup],
) -> EvidenceGraph:
    study_by_id = {node.study_id: node for node in graph.studies}
    cohort_by_id = {node.cohort_id: node for node in graph.cohorts}
    study_map = {
        member: group.canonical_id for group in study_groups for member in group.member_ids
    }
    cohort_map = {
        member: group.canonical_id for group in cohort_groups for member in group.member_ids
    }

    studies: list[StudyNode] = []
    for group in study_groups:
        nodes = [study_by_id[member] for member in group.member_ids]
        publication_ids = sorted(
            {publication for node in nodes for publication in node.publication_ids}
        )
        studies.append(
            StudyNode(
                study_id=group.canonical_id,
                publication_ids=publication_ids,
                primary_publication_id=publication_ids[0],
                design=_common_or_none([node.design for node in nodes]),
                registration_ids=sorted(
                    {identifier for node in nodes for identifier in node.registration_ids}
                ),
                risk_of_bias=_common_risk_or_unassessed([node.risk_of_bias for node in nodes]),
            )
        )

    cohorts: list[CohortNode] = []
    for group in cohort_groups:
        nodes = [cohort_by_id[member] for member in group.member_ids]
        parent_studies = {study_map[node.study_id] for node in nodes}
        if len(parent_studies) != 1:
            raise NativeCohortReconciliationError(
                f"reconciled_cohort_has_multiple_parent_studies:{group.member_ids}"
            )
        if len(nodes) == 1:
            identity = nodes[0].identity.model_copy(update={"cohort_id": group.canonical_id})
        else:
            if group.basis is ReconciliationGroupBasis.REVIEWER_RECONCILED:
                identity_basis = CohortIdentityBasis.REVIEWER_RECONCILED
                rationale = group.reviewer_rationale
            elif group.matched_registry_ids:
                identity_basis = CohortIdentityBasis.REPORTED_REGISTRY_ID
                rationale = None
            else:
                identity_basis = CohortIdentityBasis.REPORTED_DATASET_ID
                rationale = None
            identity = CohortIdentity(
                cohort_id=group.canonical_id,
                basis=identity_basis,
                source_labels=sorted(
                    {label for node in nodes for label in node.identity.source_labels}
                ),
                registry_ids=sorted(
                    {identifier for node in nodes for identifier in node.identity.registry_ids}
                ),
                dataset_ids=sorted(
                    {identifier for node in nodes for identifier in node.identity.dataset_ids}
                ),
                rationale=rationale,
            )
        cohorts.append(
            CohortNode(
                identity=identity,
                study_id=next(iter(parent_studies)),
                population_description=_common_or_none(
                    [node.population_description for node in nodes]
                ),
                recruitment_period=_common_or_none([node.recruitment_period for node in nodes]),
                total_sample_size=_common_or_none([node.total_sample_size for node in nodes]),
                risk_of_bias=_common_risk_or_unassessed([node.risk_of_bias for node in nodes]),
            )
        )

    arm_map: dict[str, str] = {}
    arms: list[ArmNode] = []
    for arm in graph.arms:
        canonical_cohort = cohort_map[arm.cohort_id]
        arm_id = (
            arm.arm_id
            if canonical_cohort == arm.cohort_id
            else _canonical_id(
                node_kind=ReconciliationNodeKind.COHORT,
                member_ids=[canonical_cohort, arm.arm_id],
            ).replace("cohort-reconciled", "arm-reconciled", 1)
        )
        arm_map[arm.arm_id] = arm_id
        arms.append(
            ArmNode.model_validate(
                {
                    **arm.model_dump(mode="json"),
                    "arm_id": arm_id,
                    "cohort_id": canonical_cohort,
                }
            )
        )

    contrast_map: dict[str, str] = {}
    contrasts: list[ContrastNode] = []
    for contrast in graph.contrasts:
        canonical_cohort = cohort_map[contrast.cohort_id]
        contrast_id = (
            contrast.contrast_id
            if canonical_cohort == contrast.cohort_id
            else _canonical_id(
                node_kind=ReconciliationNodeKind.COHORT,
                member_ids=[canonical_cohort, contrast.contrast_id],
            ).replace("cohort-reconciled", "contrast-reconciled", 1)
        )
        contrast_map[contrast.contrast_id] = contrast_id
        contrasts.append(
            ContrastNode.model_validate(
                {
                    **contrast.model_dump(mode="json"),
                    "contrast_id": contrast_id,
                    "cohort_id": canonical_cohort,
                    "treatment_arm_id": arm_map[contrast.treatment_arm_id],
                    "comparator_arm_id": arm_map[contrast.comparator_arm_id],
                }
            )
        )

    estimates = [
        OutcomeEstimateNode.model_validate(
            {
                **estimate.model_dump(mode="json"),
                "contrast_id": contrast_map[estimate.contrast_id],
            }
        )
        for estimate in graph.outcome_estimates
    ]
    return EvidenceGraph(
        publications=sorted(graph.publications, key=lambda node: node.publication_id),
        studies=sorted(studies, key=lambda node: node.study_id),
        cohorts=sorted(cohorts, key=lambda node: node.cohort_id),
        arms=sorted(arms, key=lambda node: node.arm_id),
        contrasts=sorted(contrasts, key=lambda node: node.contrast_id),
        outcome_estimates=sorted(estimates, key=lambda node: node.estimate_id),
        evidence_spans=sorted(graph.evidence_spans, key=lambda node: node.span_id),
    )


def _freeze_receipt(payload: dict[str, Any]) -> NativeCohortReconciliationReceipt:
    return NativeCohortReconciliationReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def reconcile_native_cohorts(
    *,
    corpus: TypedEvidenceCorpus,
    reviewer_artifact: ReviewerCohortReconciliationArtifact | None = None,
) -> NativeCohortReconciliationReceipt:
    """Derive and hash a conservative cross-publication identity graph."""

    validated = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    if not validated.graph.outcome_estimates:
        if reviewer_artifact is not None:
            raise NativeCohortReconciliationError(
                "reviewer_reconciliation_requires_estimable_graph"
            )
        empty_graph = validated.graph
        payload = {
            "receipt_version": "native-cohort-reconciliation-receipt-v1",
            "input_corpus_sha256": validated.corpus_sha256,
            "input_graph_sha256": hash_canonical(empty_graph),
            "reviewer_artifact": None,
            "status": NativeReconciliationStatus.NO_ESTIMABLE_GRAPH,
            "cross_publication_identity_assurance_complete": True,
            "candidates": [],
            "issues": [],
            "study_groups": [],
            "cohort_groups": [],
            "reconciled_graph": empty_graph,
            "reconciled_graph_sha256": hash_canonical(empty_graph),
            "merged_study_groups": 0,
            "merged_cohort_groups": 0,
        }
        return _freeze_receipt(payload)

    graph = validated.graph
    graph_sha256 = hash_canonical(graph)
    study_by_id = {node.study_id: node for node in graph.studies}
    cohort_by_id = {node.cohort_id: node for node in graph.cohorts}
    study_publication: dict[str, str] = {}
    input_issues: list[NativeReconciliationIssue] = []
    for study in graph.studies:
        if len(study.publication_ids) != 1:
            input_issues.append(
                NativeReconciliationIssue(
                    code="input_study_not_publication_scoped",
                    node_kind=ReconciliationNodeKind.STUDY,
                    node_ids=[study.study_id],
                    detail=("Native reconciliation requires original publication-scoped studies."),
                )
            )
        else:
            study_publication[study.study_id] = study.publication_ids[0]
    if input_issues:
        raise NativeCohortReconciliationError("native_reconciliation_input_not_publication_scoped")
    cohort_publication = {
        cohort.cohort_id: study_publication[cohort.study_id] for cohort in graph.cohorts
    }

    study_registry = {
        node.study_id: {_normalize_identifier(value) for value in node.registration_ids}
        for node in graph.studies
    }
    study_dataset = {node.study_id: set() for node in graph.studies}
    cohort_registry = {
        node.cohort_id: {_normalize_identifier(value) for value in node.identity.registry_ids}
        for node in graph.cohorts
    }
    cohort_dataset = {
        node.cohort_id: {_normalize_identifier(value) for value in node.identity.dataset_ids}
        for node in graph.cohorts
    }
    study_candidates, study_issues, study_components = _strong_components(
        node_kind=ReconciliationNodeKind.STUDY,
        node_ids=sorted(study_by_id),
        registry_ids=study_registry,
        dataset_ids=study_dataset,
        publication_ids=study_publication,
    )
    cohort_candidates, cohort_issues, cohort_components = _strong_components(
        node_kind=ReconciliationNodeKind.COHORT,
        node_ids=sorted(cohort_by_id),
        registry_ids=cohort_registry,
        dataset_ids=cohort_dataset,
        publication_ids=cohort_publication,
    )
    candidates = sorted(
        [*study_candidates, *cohort_candidates],
        key=lambda item: (item.node_kind.value, tuple(item.member_ids)),
    )
    detected_issues = _sort_issues([*study_issues, *cohort_issues])

    if reviewer_artifact is not None:
        reviewer = ReviewerCohortReconciliationArtifact.model_validate(
            reviewer_artifact.model_dump(mode="json")
        )
        if reviewer.input_corpus_sha256 != validated.corpus_sha256:
            raise NativeCohortReconciliationError("reviewer_reconciliation_corpus_hash_mismatch")
        if reviewer.input_graph_sha256 != graph_sha256:
            raise NativeCohortReconciliationError("reviewer_reconciliation_graph_hash_mismatch")
        _validate_reviewer_partition(
            groups=reviewer.study_groups,
            expected_ids=set(study_by_id),
            publication_ids=study_publication,
            node_kind=ReconciliationNodeKind.STUDY,
        )
        _validate_reviewer_partition(
            groups=reviewer.cohort_groups,
            expected_ids=set(cohort_by_id),
            publication_ids=cohort_publication,
            node_kind=ReconciliationNodeKind.COHORT,
        )
        study_partitions = [group.member_ids for group in reviewer.study_groups]
        cohort_partitions = [group.member_ids for group in reviewer.cohort_groups]
        study_group_index = {
            member: position for position, group in enumerate(study_partitions) for member in group
        }
        for group in cohort_partitions:
            parent_groups = {study_group_index[cohort_by_id[member].study_id] for member in group}
            if len(parent_groups) != 1:
                raise NativeCohortReconciliationError(
                    f"reviewer_cohort_group_crosses_study_groups:{group}"
                )
        reviewer_study = {tuple(group.member_ids): group for group in reviewer.study_groups}
        reviewer_cohort = {tuple(group.member_ids): group for group in reviewer.cohort_groups}
        study_groups = _freeze_groups(
            node_kind=ReconciliationNodeKind.STUDY,
            partitions=study_partitions,
            registry_ids=study_registry,
            dataset_ids=study_dataset,
            reviewer_groups=reviewer_study,
            candidate_by_members={tuple(item.member_ids): item for item in study_candidates},
        )
        cohort_groups = _freeze_groups(
            node_kind=ReconciliationNodeKind.COHORT,
            partitions=cohort_partitions,
            registry_ids=cohort_registry,
            dataset_ids=cohort_dataset,
            reviewer_groups=reviewer_cohort,
            candidate_by_members={tuple(item.member_ids): item for item in cohort_candidates},
        )
        issues = _sort_issues(
            [issue.model_copy(update={"resolved_by_reviewer": True}) for issue in detected_issues]
        )
        status = NativeReconciliationStatus.REVIEWER_COMPLETE
        complete = True
    else:
        reviewer = None
        issues = detected_issues
        if issues:
            study_components = []
            cohort_components = []
            status = NativeReconciliationStatus.REQUIRES_REVIEWER
        else:
            status = (
                NativeReconciliationStatus.SINGLE_PUBLICATION_COMPLETE
                if len(graph.publications) == 1
                else NativeReconciliationStatus.STRONG_IDENTIFIER_RECONCILED_LIMITED
            )
        cohort_partitions = _partition_from_components(sorted(cohort_by_id), cohort_components)
        # A reconciled cohort implies that its publication-scoped parent studies are
        # reports of the same study, even when the cohort carries the only strong ID.
        study_union = _UnionFind(sorted(study_by_id))
        for component in study_components:
            for member in component[1:]:
                study_union.union(component[0], member)
        for component in cohort_components:
            parents = sorted({cohort_by_id[member].study_id for member in component})
            for parent in parents[1:]:
                study_union.union(parents[0], parent)
        study_partitions = study_union.groups()
        implied_study_groups: set[tuple[str, ...]] = set()
        original_study_components = {tuple(component) for component in study_components}
        for group in study_partitions:
            if len(group) > 1 and tuple(group) not in original_study_components:
                implied_study_groups.add(tuple(group))
            publications = [study_publication[member] for member in group]
            if len(publications) != len(set(publications)):
                issues.append(
                    NativeReconciliationIssue(
                        code="ambiguous_many_to_many_identity",
                        node_kind=ReconciliationNodeKind.STUDY,
                        node_ids=group,
                        detail=(
                            "Reconciled cohort candidates imply multiple studies from one "
                            "publication; automatic reconciliation is forbidden."
                        ),
                    )
                )
            for position, left in enumerate(group):
                for right in group[position + 1 :]:
                    if (
                        study_registry[left]
                        and study_registry[right]
                        and not study_registry[left].intersection(study_registry[right])
                    ):
                        issues.append(
                            NativeReconciliationIssue(
                                code="conflicting_registry_identifiers",
                                node_kind=ReconciliationNodeKind.STUDY,
                                node_ids=[left, right],
                                detail=(
                                    "A cohort match implies one study, but the parent "
                                    "studies carry disjoint non-empty registration IDs."
                                ),
                            )
                        )
        if issues and status is not NativeReconciliationStatus.REQUIRES_REVIEWER:
            # Do not retain a partially merged graph when a derived ambiguity appears.
            status = NativeReconciliationStatus.REQUIRES_REVIEWER
            issues = _sort_issues(issues)
            cohort_partitions = [[node_id] for node_id in sorted(cohort_by_id)]
            study_partitions = [[node_id] for node_id in sorted(study_by_id)]
            implied_study_groups = set()
        study_groups = _freeze_groups(
            node_kind=ReconciliationNodeKind.STUDY,
            partitions=study_partitions,
            registry_ids=study_registry,
            dataset_ids=study_dataset,
            reviewer_groups=None,
            candidate_by_members={tuple(item.member_ids): item for item in study_candidates},
            implied_groups=implied_study_groups,
        )
        cohort_groups = _freeze_groups(
            node_kind=ReconciliationNodeKind.COHORT,
            partitions=cohort_partitions,
            registry_ids=cohort_registry,
            dataset_ids=cohort_dataset,
            reviewer_groups=None,
            candidate_by_members={tuple(item.member_ids): item for item in cohort_candidates},
        )
        complete = status is NativeReconciliationStatus.SINGLE_PUBLICATION_COMPLETE

    reconciled_graph = _rewrite_graph(
        graph,
        study_groups=study_groups,
        cohort_groups=cohort_groups,
    )
    payload = {
        "receipt_version": "native-cohort-reconciliation-receipt-v1",
        "input_corpus_sha256": validated.corpus_sha256,
        "input_graph_sha256": graph_sha256,
        "reviewer_artifact": reviewer,
        "status": status,
        "cross_publication_identity_assurance_complete": complete,
        "candidates": candidates,
        "issues": issues,
        "study_groups": study_groups,
        "cohort_groups": cohort_groups,
        "reconciled_graph": reconciled_graph,
        "reconciled_graph_sha256": hash_canonical(reconciled_graph),
        "merged_study_groups": sum(len(group.member_ids) > 1 for group in study_groups),
        "merged_cohort_groups": sum(len(group.member_ids) > 1 for group in cohort_groups),
    }
    return _freeze_receipt(payload)


def reverify_native_cohort_reconciliation(
    *,
    corpus: TypedEvidenceCorpus,
    receipt: NativeCohortReconciliationReceipt,
) -> NativeCohortReconciliationReceipt:
    """Recompute a frozen reconciliation receipt and reject any divergence."""

    frozen = NativeCohortReconciliationReceipt.model_validate(receipt.model_dump(mode="json"))
    replayed = reconcile_native_cohorts(
        corpus=corpus,
        reviewer_artifact=frozen.reviewer_artifact,
    )
    if replayed.model_dump(mode="json") != frozen.model_dump(mode="json"):
        raise NativeCohortReconciliationError(
            f"native_cohort_reconciliation_replay_mismatch:{frozen.receipt_sha256}"
        )
    return replayed


__all__ = [
    "NativeCohortReconciliationError",
    "NativeCohortReconciliationReceipt",
    "NativeIdentityCandidate",
    "NativeReconciledIdentityGroup",
    "NativeReconciliationIssue",
    "NativeReconciliationStatus",
    "ReconciliationGroupBasis",
    "ReconciliationNodeKind",
    "ReviewerCohortReconciliationArtifact",
    "ReviewerIdentityGroup",
    "freeze_reviewer_cohort_reconciliation_artifact",
    "reconcile_native_cohorts",
    "reverify_native_cohort_reconciliation",
]
