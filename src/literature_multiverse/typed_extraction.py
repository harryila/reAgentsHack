"""Native typed publication-extraction artifacts and deterministic graph assembly.

The legacy extraction ledger is intentionally categorical.  This module defines the
normative production boundary for numerical scientific evidence: one hash-bound
fragment per publication, including an explicit record when no safe numerical effect
can be extracted.  Fragments are merged only when node identities are disjoint or
byte-for-byte equivalent; cross-publication study/cohort disagreements require an
external reconciliation artifact rather than a silent guess.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.evidence_graph import EvidenceGraph, PublicationIdentity
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import ContractModel

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TypedExtractionContractError(ValueError):
    """A native extraction fragment or assembled corpus is not reproducible."""


class FragmentStatus(StrEnum):
    ESTIMABLE = "estimable"
    NON_ESTIMABLE = "non_estimable"


class NonEstimabilityReason(StrEnum):
    NO_TARGET_OUTCOME = "no_target_outcome"
    NUMERICAL_RESULT_ABSENT = "numerical_result_absent"
    UNCERTAINTY_ABSENT = "uncertainty_absent"
    INCOMPATIBLE_ESTIMAND = "incompatible_estimand"
    UNGROUNDED_NUMERICAL_RESULT = "ungrounded_numerical_result"
    UNRESOLVED_COHORT_IDENTITY = "unresolved_cohort_identity"
    UNSUPPORTED_EFFECT_FORMAT = "unsupported_effect_format"
    SOURCE_DOCUMENT_INCOMPLETE = "source_document_incomplete"
    OTHER = "other"


class SourceDocumentArtifact(ContractModel):
    """Immutable source document from which a publication fragment was extracted."""

    artifact_path: Annotated[str, Field(min_length=1)]
    sha256: str
    media_type: Annotated[str, Field(min_length=1)]
    source_locator: Annotated[str, Field(min_length=1)]

    @field_validator("artifact_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("source_document_artifact_path_must_be_relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("source_document_sha256_invalid")
        return value


class PublicationEvidenceFragment(ContractModel):
    """One source-grounded typed extraction result for one publication.

    An estimable fragment contains a complete, locally valid one-publication evidence
    graph.  A non-estimable fragment deliberately contains no graph and must state why.
    This keeps eligible zero-effect/zero-extraction papers in the corpus accounting.
    """

    fragment_version: Literal[
        "publication-evidence-fragment-v2",
        "publication-evidence-fragment-v3",
    ] = "publication-evidence-fragment-v2"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    publication_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    publication: PublicationIdentity
    pipeline_fingerprint_sha256: str
    extraction_context_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    source_document: SourceDocumentArtifact
    grounding_receipt_sha256: str | None = None
    status: FragmentStatus
    graph: EvidenceGraph | None = None
    non_estimability_reason: NonEstimabilityReason | None = None
    non_estimability_detail: str | None = None
    extractor_warnings: list[str] = Field(default_factory=list)
    fragment_sha256: str

    @field_validator(
        "pipeline_fingerprint_sha256",
        "extraction_context_sha256",
        "grounding_receipt_sha256",
        "fragment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("publication_fragment_sha256_invalid")
        return value

    @field_validator("extractor_warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("publication_fragment_warnings_not_sorted_unique")
        if any(not warning.strip() for warning in value):
            raise ValueError("publication_fragment_warning_empty")
        return value

    @model_validator(mode="after")
    def validate_fragment(self) -> PublicationEvidenceFragment:
        if self.fragment_version == "publication-evidence-fragment-v2":
            if self.extraction_context_sha256 is not None:
                raise ValueError("publication_fragment_v2_forbids_extraction_context")
        elif self.extraction_context_sha256 is None:
            raise ValueError("publication_fragment_v3_requires_extraction_context")
        if self.publication.publication_id != self.publication_id:
            raise ValueError("publication_fragment_authoritative_publication_id_mismatch")
        if self.publication.paper_id != self.paper_id:
            raise ValueError("publication_fragment_authoritative_paper_id_mismatch")
        if self.status is FragmentStatus.ESTIMABLE:
            if self.grounding_receipt_sha256 is None:
                raise ValueError("estimable_fragment_requires_grounding_receipt")
            if self.graph is None:
                raise ValueError("estimable_fragment_requires_graph")
            if not self.graph.outcome_estimates:
                raise ValueError("estimable_fragment_requires_outcome_estimate")
            if self.non_estimability_reason is not None or self.non_estimability_detail is not None:
                raise ValueError("estimable_fragment_forbids_non_estimability_metadata")
            if len(self.graph.publications) != 1:
                raise ValueError("publication_fragment_graph_requires_one_publication")
            publication = self.graph.publications[0]
            if publication.publication_id != self.publication_id:
                raise ValueError("publication_fragment_publication_id_mismatch")
            if publication.paper_id != self.paper_id:
                raise ValueError("publication_fragment_paper_id_mismatch")
        else:
            if self.graph is not None:
                raise ValueError("non_estimable_fragment_forbids_graph")
            if self.non_estimability_reason is None:
                raise ValueError("non_estimable_fragment_requires_reason")
            if self.non_estimability_reason is NonEstimabilityReason.OTHER and not (
                self.non_estimability_detail and self.non_estimability_detail.strip()
            ):
                raise ValueError("other_non_estimability_requires_detail")
        payload = self.model_dump(mode="json", exclude={"fragment_sha256"})
        if self.fragment_version == "publication-evidence-fragment-v2":
            payload.pop("extraction_context_sha256", None)
        if hash_canonical(payload) != self.fragment_sha256:
            raise ValueError("publication_fragment_hash_mismatch")
        return self


class TypedExtractionIssue(ContractModel):
    publication_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    code: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]


def _assemble_fragment_issues(
    fragments: list[PublicationEvidenceFragment],
) -> list[TypedExtractionIssue]:
    """Project the complete non-estimability ledger into blocking corpus issues."""

    return [
        TypedExtractionIssue(
            publication_id=fragment.publication_id,
            paper_id=fragment.paper_id,
            code=f"non_estimable:{fragment.non_estimability_reason.value}",
            detail=fragment.non_estimability_detail
            or "The source did not support a safely estimable numerical effect.",
        )
        for fragment in sorted(fragments, key=lambda item: item.publication_id)
        if fragment.status is FragmentStatus.NON_ESTIMABLE
        and fragment.non_estimability_reason is not None
    ]


class TypedEvidenceCorpus(ContractModel):
    """Hash-bound output of merging a complete set of publication fragments."""

    corpus_version: Literal[
        "typed-evidence-corpus-v2",
        "typed-evidence-corpus-v3",
    ] = "typed-evidence-corpus-v2"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    pipeline_fingerprint_sha256: str
    extraction_context_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fragments: Annotated[list[PublicationEvidenceFragment], Field(min_length=1)]
    graph: EvidenceGraph
    issues: list[TypedExtractionIssue]
    estimable_publication_ids: list[str]
    non_estimable_publication_ids: list[str]
    corpus_sha256: str

    @field_validator(
        "pipeline_fingerprint_sha256",
        "extraction_context_sha256",
        "corpus_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("typed_evidence_corpus_sha256_invalid")
        return value

    @field_validator("estimable_publication_ids", "non_estimable_publication_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("typed_evidence_corpus_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_corpus(self) -> TypedEvidenceCorpus:
        fragment_versions = {fragment.fragment_version for fragment in self.fragments}
        fragment_contexts = {
            fragment.extraction_context_sha256 for fragment in self.fragments
        }
        if self.corpus_version == "typed-evidence-corpus-v2":
            if self.extraction_context_sha256 is not None:
                raise ValueError("typed_evidence_corpus_v2_forbids_extraction_context")
            if fragment_versions != {"publication-evidence-fragment-v2"}:
                raise ValueError("typed_evidence_corpus_v2_requires_v2_fragments")
        else:
            if self.extraction_context_sha256 is None:
                raise ValueError("typed_evidence_corpus_v3_requires_extraction_context")
            if fragment_versions != {"publication-evidence-fragment-v3"}:
                raise ValueError("typed_evidence_corpus_v3_requires_v3_fragments")
            if fragment_contexts != {self.extraction_context_sha256}:
                raise ValueError("typed_evidence_fragment_extraction_context_mismatch")
        publication_ids = [fragment.publication_id for fragment in self.fragments]
        if publication_ids != sorted(set(publication_ids)):
            raise ValueError("typed_evidence_fragments_not_sorted_unique")
        paper_ids = [fragment.paper_id for fragment in self.fragments]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("typed_evidence_fragment_paper_ids_not_unique")
        if {fragment.question_id for fragment in self.fragments} != {self.question_id}:
            raise ValueError("typed_evidence_fragment_question_mismatch")
        if {fragment.pipeline_fingerprint_sha256 for fragment in self.fragments} != {
            self.pipeline_fingerprint_sha256
        }:
            raise ValueError("typed_evidence_fragment_pipeline_mismatch")
        expected_estimable = sorted(
            fragment.publication_id
            for fragment in self.fragments
            if fragment.status is FragmentStatus.ESTIMABLE
        )
        expected_non_estimable = sorted(
            fragment.publication_id
            for fragment in self.fragments
            if fragment.status is FragmentStatus.NON_ESTIMABLE
        )
        if self.estimable_publication_ids != expected_estimable:
            raise ValueError("typed_evidence_estimable_ids_mismatch")
        if self.non_estimable_publication_ids != expected_non_estimable:
            raise ValueError("typed_evidence_non_estimable_ids_mismatch")
        expected_graph = _assemble_fragment_graph(self.fragments)
        if hash_canonical(self.graph) != hash_canonical(expected_graph):
            raise ValueError("typed_evidence_graph_fragment_projection_mismatch")
        if bool(expected_estimable) != bool(self.graph.outcome_estimates):
            raise ValueError("typed_evidence_graph_estimability_mismatch")
        expected_issues = _assemble_fragment_issues(self.fragments)
        if [issue.model_dump(mode="json") for issue in self.issues] != [
            issue.model_dump(mode="json") for issue in expected_issues
        ]:
            raise ValueError("typed_evidence_issues_fragment_projection_mismatch")
        payload = self.model_dump(mode="json", exclude={"corpus_sha256"})
        if self.corpus_version == "typed-evidence-corpus-v2":
            payload.pop("extraction_context_sha256", None)
        if hash_canonical(payload) != self.corpus_sha256:
            raise ValueError("typed_evidence_corpus_hash_mismatch")
        return self


def freeze_publication_evidence_fragment(
    *,
    question_id: str,
    publication_id: str,
    paper_id: str,
    publication: PublicationIdentity,
    pipeline_fingerprint_sha256: str,
    extraction_context_sha256: str | None = None,
    source_document: SourceDocumentArtifact,
    grounding_receipt_sha256: str | None,
    status: FragmentStatus,
    graph: EvidenceGraph | None = None,
    non_estimability_reason: NonEstimabilityReason | None = None,
    non_estimability_detail: str | None = None,
    extractor_warnings: list[str] | None = None,
) -> PublicationEvidenceFragment:
    """Freeze and self-hash one native extraction artifact."""

    payload: dict[str, Any] = {
        "fragment_version": (
            "publication-evidence-fragment-v3"
            if extraction_context_sha256 is not None
            else "publication-evidence-fragment-v2"
        ),
        "question_id": question_id,
        "publication_id": publication_id,
        "paper_id": paper_id,
        "publication": publication,
        "pipeline_fingerprint_sha256": pipeline_fingerprint_sha256,
        **(
            {"extraction_context_sha256": extraction_context_sha256}
            if extraction_context_sha256 is not None
            else {}
        ),
        "source_document": source_document,
        "grounding_receipt_sha256": grounding_receipt_sha256,
        "status": status,
        "graph": graph,
        "non_estimability_reason": non_estimability_reason,
        "non_estimability_detail": non_estimability_detail,
        "extractor_warnings": sorted(set(extractor_warnings or [])),
    }
    return PublicationEvidenceFragment.model_validate(
        {**payload, "fragment_sha256": hash_canonical(payload)}
    )


def _merge_graphs(graphs: list[EvidenceGraph]) -> EvidenceGraph:
    if not graphs:
        raise TypedExtractionContractError("cannot_merge_empty_graph_list")
    fields = (
        ("publications", "publication_id"),
        ("studies", "study_id"),
        ("cohorts", "cohort_id"),
        ("arms", "arm_id"),
        ("contrasts", "contrast_id"),
        ("outcome_estimates", "estimate_id"),
        ("evidence_spans", "span_id"),
    )
    merged: dict[str, list[Any]] = {}
    for field_name, identity_name in fields:
        by_id: dict[str, Any] = {}
        for graph in graphs:
            for node in getattr(graph, field_name):
                identity = getattr(node, identity_name)
                existing = by_id.get(identity)
                if existing is not None and hash_canonical(existing) != hash_canonical(node):
                    raise TypedExtractionContractError(
                        f"typed_graph_identity_collision:{field_name}:{identity}"
                    )
                by_id[identity] = node
        merged[field_name] = [by_id[key] for key in sorted(by_id)]
    return EvidenceGraph.model_validate({"graph_schema_version": "1", **merged})


def _assemble_fragment_graph(
    fragments: list[PublicationEvidenceFragment],
) -> EvidenceGraph:
    """Project the complete fragment ledger into one deterministic evidence graph.

    Non-estimable publications remain first-class publication nodes even though they
    contribute no study/effect nodes.  All other nodes must come from hash-validated
    estimable fragments, so a caller cannot inject a graph-only estimate and merely
    recompute the outer corpus hash.
    """

    estimable_graphs = [
        fragment.graph
        for fragment in fragments
        if fragment.status is FragmentStatus.ESTIMABLE and fragment.graph is not None
    ]
    if estimable_graphs:
        merged = _merge_graphs(estimable_graphs).model_dump(mode="json")
    else:
        merged = {
            "graph_schema_version": "1",
            "publications": [],
            "studies": [],
            "cohorts": [],
            "arms": [],
            "contrasts": [],
            "outcome_estimates": [],
            "evidence_spans": [],
        }

    publications: dict[str, PublicationIdentity] = {}
    for fragment in fragments:
        existing = publications.get(fragment.publication_id)
        if existing is not None and hash_canonical(existing) != hash_canonical(
            fragment.publication
        ):
            raise TypedExtractionContractError(
                f"typed_graph_identity_collision:publications:{fragment.publication_id}"
            )
        publications[fragment.publication_id] = fragment.publication
    merged["publications"] = [publications[key] for key in sorted(publications)]
    return EvidenceGraph.model_validate(merged)


def assemble_typed_evidence_corpus(
    fragments: list[PublicationEvidenceFragment],
) -> TypedEvidenceCorpus:
    """Validate and deterministically merge a complete publication-fragment ledger."""

    if not fragments:
        raise TypedExtractionContractError("typed_evidence_fragments_empty")
    # Re-parse snapshots so in-place mutation of nested objects cannot bypass hashes.
    validated = [
        PublicationEvidenceFragment.model_validate(fragment.model_dump(mode="json"))
        for fragment in fragments
    ]
    validated.sort(key=lambda fragment: fragment.publication_id)
    questions = {fragment.question_id for fragment in validated}
    pipelines = {fragment.pipeline_fingerprint_sha256 for fragment in validated}
    contexts = {fragment.extraction_context_sha256 for fragment in validated}
    if len(questions) != 1:
        raise TypedExtractionContractError("typed_evidence_questions_mixed")
    if len(pipelines) != 1:
        raise TypedExtractionContractError("typed_evidence_pipelines_mixed")
    if len(contexts) != 1:
        raise TypedExtractionContractError("typed_evidence_extraction_contexts_mixed")
    extraction_context_sha256 = next(iter(contexts))
    estimable = [fragment for fragment in validated if fragment.status is FragmentStatus.ESTIMABLE]
    graph = _assemble_fragment_graph(validated)
    issues = _assemble_fragment_issues(validated)
    payload: dict[str, Any] = {
        "corpus_version": (
            "typed-evidence-corpus-v3"
            if extraction_context_sha256 is not None
            else "typed-evidence-corpus-v2"
        ),
        "question_id": next(iter(questions)),
        "pipeline_fingerprint_sha256": next(iter(pipelines)),
        **(
            {"extraction_context_sha256": extraction_context_sha256}
            if extraction_context_sha256 is not None
            else {}
        ),
        "fragments": validated,
        "graph": graph,
        "issues": issues,
        "estimable_publication_ids": sorted(fragment.publication_id for fragment in estimable),
        "non_estimable_publication_ids": sorted(
            fragment.publication_id
            for fragment in validated
            if fragment.status is FragmentStatus.NON_ESTIMABLE
        ),
    }
    return TypedEvidenceCorpus.model_validate({**payload, "corpus_sha256": hash_canonical(payload)})


def publication_fragment_json_schema() -> dict[str, Any]:
    schema = PublicationEvidenceFragment.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:literature-multiverse:publication-evidence-fragment:v3"
    return schema


__all__ = [
    "FragmentStatus",
    "NonEstimabilityReason",
    "PublicationEvidenceFragment",
    "SourceDocumentArtifact",
    "TypedEvidenceCorpus",
    "TypedExtractionContractError",
    "TypedExtractionIssue",
    "assemble_typed_evidence_corpus",
    "freeze_publication_evidence_fragment",
    "publication_fragment_json_schema",
]
