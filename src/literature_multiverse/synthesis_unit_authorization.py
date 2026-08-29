"""Source-backed authorization of independent synthesis units.

This is intentionally separate from the historical MetaSyn v2 yield contract, which
always abstains from authorizing synthesis.  A receipt here can authorize a group only
after cohort reconciliation and an external source review supply exact, immutable
source citations for every cross-publication identity and independence assertion.
Publication separation, extractor labels, and textual similarity are never evidence.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings
from itertools import combinations
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
    reverify_native_cohort_reconciliation,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import resolve_native_source_document
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


class SynthesisAuthorizationError(ValueError):
    """The source record does not justify a synthesis unit."""


class SourceIdentityCitation(ContractModel):
    """Exact source location supporting one identity assertion."""

    model_config = ConfigDict(str_strip_whitespace=False)

    publication_id: Annotated[str, Field(min_length=1)]
    original_cohort_id: Annotated[str, Field(min_length=1)]
    source_document_sha256: str
    grounding_receipt_sha256: str
    source_locator: Annotated[str, Field(min_length=1)]
    quote: str | None = None
    line_ids: Annotated[list[str], Field(min_length=1)]
    cited_identifier: Annotated[str, Field(min_length=1)]
    source_payload_sha256: str
    cited_lines_sha256: str
    cited_text_sha256: str
    citation_sha256: str

    @field_validator(
        "source_document_sha256",
        "grounding_receipt_sha256",
        "source_payload_sha256",
        "cited_lines_sha256",
        "cited_text_sha256",
        "citation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("synthesis_authorization_citation_sha256_invalid")
        return value

    @field_validator("line_ids")
    @classmethod
    def validate_lines(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("synthesis_authorization_line_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_citation(self) -> SourceIdentityCitation:
        if self.quote is None or not self.quote.strip():
            raise ValueError("synthesis_authorization_citation_requires_quote_and_lines")
        payload = self.model_dump(mode="json", exclude={"citation_sha256"})
        if hash_canonical(payload) != self.citation_sha256:
            raise ValueError("synthesis_authorization_citation_hash_mismatch")
        return self


class SourceIdentityAssertion(ContractModel):
    assertion_version: Literal["source-identity-assertion-v1"] = "source-identity-assertion-v1"
    relationship: Literal["same_cohort", "independent_cohorts"]
    cohort_ids: Annotated[list[str], Field(min_length=2)]
    rationale: Annotated[str, Field(min_length=1)]
    citations: Annotated[list[SourceIdentityCitation], Field(min_length=2)]
    reviewer_identity_sha256: str
    review_protocol_sha256: str
    assertion_sha256: str

    @field_validator("cohort_ids")
    @classmethod
    def validate_cohorts(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_authorization_assertion_cohorts_not_sorted_unique")
        return value

    @field_validator("reviewer_identity_sha256", "review_protocol_sha256", "assertion_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("synthesis_authorization_assertion_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_assertion(self) -> SourceIdentityAssertion:
        if self.relationship == "independent_cohorts" and len(self.cohort_ids) != 2:
            raise ValueError("independence_assertion_requires_exact_pair")
        citation_hashes = [item.citation_sha256 for item in self.citations]
        if citation_hashes != sorted(set(citation_hashes)):
            raise ValueError("synthesis_authorization_citations_not_sorted_unique")
        if not self.rationale.strip():
            raise ValueError("synthesis_authorization_rationale_empty")
        payload = self.model_dump(mode="json", exclude={"assertion_sha256"})
        if hash_canonical(payload) != self.assertion_sha256:
            raise ValueError("synthesis_authorization_assertion_hash_mismatch")
        return self


class SynthesisUnitAuthorizationReceiptV1(ContractModel):
    receipt_version: Literal["source-backed-synthesis-authorization-v1"] = (
        "source-backed-synthesis-authorization-v1"
    )
    input_corpus_sha256: str
    reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    canonical_cohort_ids: Annotated[list[str], Field(min_length=1)]
    assertions: list[SourceIdentityAssertion]
    unresolved_overlap_pairs: list[list[str]]
    reference_labels_accessed: Literal[False] = False
    review_conclusions_accessed: Literal[False] = False
    authorizes_synthesis_input: bool
    authorization_basis: Literal[
        "single_cohort_cross_paper_independence_irrelevant",
        "all_pairwise_independence_source_adjudicated",
        "unresolved_overlap_abstention",
    ]
    receipt_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("synthesis_authorization_receipt_sha256_invalid")
        return value

    @field_validator("estimate_ids", "canonical_cohort_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_authorization_ids_not_sorted_unique")
        return value

    @field_validator("unresolved_overlap_pairs")
    @classmethod
    def validate_pairs(cls, value: list[list[str]]) -> list[list[str]]:
        if any(pair != sorted(set(pair)) or len(pair) != 2 for pair in value):
            raise ValueError("synthesis_authorization_unresolved_pair_invalid")
        if value != sorted(value) or len(value) != len({tuple(pair) for pair in value}):
            raise ValueError("synthesis_authorization_unresolved_pairs_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> SynthesisUnitAuthorizationReceiptV1:
        expected = not self.unresolved_overlap_pairs
        if self.authorizes_synthesis_input != expected:
            raise ValueError("synthesis_authorization_outcome_mismatch")
        if len(self.canonical_cohort_ids) == 1:
            basis = "single_cohort_cross_paper_independence_irrelevant"
        elif expected:
            basis = "all_pairwise_independence_source_adjudicated"
        else:
            basis = "unresolved_overlap_abstention"
        if self.authorization_basis != basis:
            raise ValueError("synthesis_authorization_basis_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("synthesis_authorization_receipt_hash_mismatch")
        return self


class SynthesisUnitAuthorizationReceipt(ContractModel):
    """V2 receipt with an explicit semantic firewall.

    V1 is retained above for historical artifact parsing; new authorization and
    replay paths intentionally emit/accept only V2.
    """

    receipt_version: Literal["source-backed-synthesis-authorization-v2"] = (
        "source-backed-synthesis-authorization-v2"
    )
    input_corpus_sha256: str
    reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    canonical_cohort_ids: Annotated[list[str], Field(min_length=1)]
    assertions: list[SourceIdentityAssertion]
    unresolved_overlap_pairs: list[list[str]]
    publication_source_content_visible: Literal[True] = True
    benchmark_reference_labels_accessed: Literal[False] = False
    benchmark_review_verdicts_accessed: Literal[False] = False
    authorizes_synthesis_input: bool
    authorization_basis: Literal[
        "single_cohort_cross_paper_independence_irrelevant",
        "all_pairwise_independence_source_adjudicated",
        "unresolved_overlap_abstention",
    ]
    receipt_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("synthesis_authorization_receipt_sha256_invalid")
        return value

    @field_validator("estimate_ids", "canonical_cohort_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_authorization_ids_not_sorted_unique")
        return value

    @field_validator("unresolved_overlap_pairs")
    @classmethod
    def validate_pairs(cls, value: list[list[str]]) -> list[list[str]]:
        if any(pair != sorted(set(pair)) or len(pair) != 2 for pair in value):
            raise ValueError("synthesis_authorization_unresolved_pair_invalid")
        if value != sorted(value) or len(value) != len({tuple(pair) for pair in value}):
            raise ValueError("synthesis_authorization_unresolved_pairs_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> SynthesisUnitAuthorizationReceipt:
        expected = not self.unresolved_overlap_pairs
        if self.authorizes_synthesis_input != expected:
            raise ValueError("synthesis_authorization_outcome_mismatch")
        if len(self.canonical_cohort_ids) == 1:
            basis = "single_cohort_cross_paper_independence_irrelevant"
        elif expected:
            basis = "all_pairwise_independence_source_adjudicated"
        else:
            basis = "unresolved_overlap_abstention"
        if self.authorization_basis != basis:
            raise ValueError("synthesis_authorization_basis_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("synthesis_authorization_receipt_hash_mismatch")
        return self


def freeze_source_identity_citation(**values: object) -> SourceIdentityCitation:
    payload = dict(values)
    return SourceIdentityCitation.model_validate(
        {**payload, "citation_sha256": hash_canonical(payload)}
    )


def freeze_source_identity_assertion(**values: object) -> SourceIdentityAssertion:
    payload = {"assertion_version": "source-identity-assertion-v1", **values}
    return SourceIdentityAssertion.model_validate(
        {**payload, "assertion_sha256": hash_canonical(payload)}
    )


def authorize_synthesis_unit(
    *,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    estimate_ids: list[str],
    assertions: list[SourceIdentityAssertion],
    repository_root: Path,
) -> SynthesisUnitAuthorizationReceipt:
    """Authorize only source-adjudicated independent cohorts; otherwise abstain."""

    corpus = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    reconciliation = reverify_native_cohort_reconciliation(corpus=corpus, receipt=reconciliation)
    graph = reconciliation.reconciled_graph
    assert graph is not None and reconciliation.reconciled_graph_sha256 is not None
    selected_ids = sorted(set(estimate_ids))
    if selected_ids != estimate_ids or not selected_ids:
        raise SynthesisAuthorizationError("synthesis_estimate_ids_not_sorted_unique")
    estimate_by_id = {item.estimate_id: item for item in graph.outcome_estimates}
    if any(item not in estimate_by_id for item in selected_ids):
        raise SynthesisAuthorizationError("synthesis_estimate_unknown")
    contrast_to_cohort = {item.contrast_id: item.cohort_id for item in graph.contrasts}
    canonical_cohorts = sorted(
        {contrast_to_cohort[estimate_by_id[item].contrast_id] for item in selected_ids}
    )

    fragments = {item.publication_id: item for item in corpus.fragments}
    resolved_sources = {
        publication_id: resolve_native_source_document(
            repository_root=repository_root,
            source_document=fragment.source_document,
        )
        for publication_id, fragment in fragments.items()
    }
    original_cohort_by_id = {item.cohort_id: item for item in corpus.graph.cohorts}
    original_cohort_to_publication: dict[str, str] = {}
    study_publication = {item.study_id: item.publication_ids[0] for item in corpus.graph.studies}
    for cohort in corpus.graph.cohorts:
        original_cohort_to_publication[cohort.cohort_id] = study_publication[cohort.study_id]
    group_by_canonical = {item.canonical_id: item for item in reconciliation.cohort_groups}

    validated_assertions: list[SourceIdentityAssertion] = []
    same_asserted: set[str] = set()
    independent_pairs: set[tuple[str, str]] = set()
    assertion_hashes = [item.assertion_sha256 for item in assertions]
    if len(assertion_hashes) != len(set(assertion_hashes)):
        raise SynthesisAuthorizationError("duplicate_source_identity_assertion")
    semantic_keys = [(item.relationship, tuple(item.cohort_ids)) for item in assertions]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise SynthesisAuthorizationError("duplicate_source_identity_relationship")
    relationships_by_members: dict[tuple[str, ...], set[str]] = {}
    for relationship, cohort_ids in semantic_keys:
        relationships_by_members.setdefault(cohort_ids, set()).add(relationship)
    if any(len(values) > 1 for values in relationships_by_members.values()):
        raise SynthesisAuthorizationError("conflicting_source_identity_relationships")

    used_assertion_hashes: set[str] = set()
    for raw in assertions:
        assertion = SourceIdentityAssertion.model_validate(raw.model_dump(mode="json"))
        cited_publications = {citation.publication_id for citation in assertion.citations}
        cited_text_by_hash: dict[str, str] = {}
        for citation in assertion.citations:
            fragment = fragments.get(citation.publication_id)
            if fragment is None:
                raise SynthesisAuthorizationError("assertion_citation_publication_unknown")
            if (
                citation.source_document_sha256 != fragment.source_document.sha256
                or citation.grounding_receipt_sha256 != fragment.grounding_receipt_sha256
            ):
                raise SynthesisAuthorizationError("assertion_citation_source_lineage_mismatch")
            source = resolved_sources[citation.publication_id]
            if citation.source_locator != source.source_locator:
                raise SynthesisAuthorizationError("assertion_citation_locator_not_exact")
            if citation.source_payload_sha256 != source.source_payload_sha256:
                raise SynthesisAuthorizationError("assertion_citation_payload_hash_mismatch")
            line_by_id = {line.line_id: line for line in source.lines}
            if any(line_id not in line_by_id for line_id in citation.line_ids):
                raise SynthesisAuthorizationError("assertion_citation_line_unknown")
            cited_lines = [line_by_id[line_id] for line_id in citation.line_ids]
            cited_text = "\n".join(line.text for line in cited_lines)
            if hash_canonical(cited_lines) != citation.cited_lines_sha256:
                raise SynthesisAuthorizationError("assertion_citation_lines_hash_mismatch")
            if hashlib.sha256(cited_text.encode("utf-8")).hexdigest() != citation.cited_text_sha256:
                raise SynthesisAuthorizationError("assertion_citation_text_hash_mismatch")
            assert citation.quote is not None
            if citation.quote not in cited_text or citation.quote not in source.source_text:
                raise SynthesisAuthorizationError("assertion_citation_quote_not_exact")
            cited_text_by_hash[citation.citation_sha256] = cited_text
        if assertion.relationship == "same_cohort":
            matching = [
                group
                for group in reconciliation.cohort_groups
                if group.member_ids == assertion.cohort_ids
            ]
            if len(matching) != 1:
                raise SynthesisAuthorizationError("same_cohort_assertion_not_reconciled_group")
            if matching[0].canonical_id not in canonical_cohorts:
                raise SynthesisAuthorizationError("unused_same_cohort_assertion")
            expected_publications = {
                original_cohort_to_publication[item] for item in assertion.cohort_ids
            }
            if not expected_publications.issubset(cited_publications):
                raise SynthesisAuthorizationError("same_cohort_assertion_missing_member_source")
            relevant_members = assertion.cohort_ids
            same_asserted.add(matching[0].canonical_id)
            used_assertion_hashes.add(assertion.assertion_sha256)
        else:
            if any(item not in canonical_cohorts for item in assertion.cohort_ids):
                raise SynthesisAuthorizationError("unused_independence_assertion")
            expected_publications = {
                original_cohort_to_publication[member]
                for cohort_id in assertion.cohort_ids
                for member in group_by_canonical[cohort_id].member_ids
            }
            if not expected_publications.issubset(cited_publications):
                raise SynthesisAuthorizationError("independence_assertion_missing_cohort_source")
            relevant_members = [
                member
                for cohort_id in assertion.cohort_ids
                for member in group_by_canonical[cohort_id].member_ids
            ]
            independent_pairs.add(tuple(assertion.cohort_ids))
            used_assertion_hashes.add(assertion.assertion_sha256)
        cited_members = {citation.original_cohort_id for citation in assertion.citations}
        if cited_members != set(relevant_members):
            raise SynthesisAuthorizationError(
                "assertion_citations_do_not_cover_exact_original_cohorts"
            )
        for citation in assertion.citations:
            if citation.original_cohort_id not in relevant_members:
                raise SynthesisAuthorizationError("assertion_citation_cohort_outside_assertion")
            if (
                original_cohort_to_publication[citation.original_cohort_id]
                != citation.publication_id
            ):
                raise SynthesisAuthorizationError("assertion_citation_cohort_publication_mismatch")
            normalized = " ".join(
                unicodedata.normalize("NFKC", citation.cited_identifier).strip().casefold().split()
            )
            supported = {
                " ".join(unicodedata.normalize("NFKC", identifier).strip().casefold().split())
                for member in [citation.original_cohort_id]
                for identifier in (
                    *original_cohort_by_id[member].identity.registry_ids,
                    *original_cohort_by_id[member].identity.dataset_ids,
                )
            }
            if normalized not in supported:
                raise SynthesisAuthorizationError(
                    "assertion_citation_identifier_not_in_source_cohort"
                )
            normalized_span = " ".join(
                unicodedata.normalize("NFKC", cited_text_by_hash[citation.citation_sha256])
                .strip()
                .casefold()
                .split()
            )
            if (
                re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
                    normalized_span,
                )
                is None
            ):
                raise SynthesisAuthorizationError(
                    "assertion_citation_identifier_outside_cited_span"
                )
        validated_assertions.append(assertion)

    # Any merged identity used by this unit needs affirmative source evidence.  Strong
    # identifiers from model output alone are not enough.
    for cohort_id in canonical_cohorts:
        if len(group_by_canonical[cohort_id].member_ids) > 1 and cohort_id not in same_asserted:
            raise SynthesisAuthorizationError("merged_cohort_lacks_source_identity_assertion")

    required_pairs = set(combinations(canonical_cohorts, 2))
    if independent_pairs - required_pairs:
        raise SynthesisAuthorizationError("unused_independence_assertion")
    if used_assertion_hashes != set(assertion_hashes):
        raise SynthesisAuthorizationError("unused_source_identity_assertion")
    unresolved = [list(pair) for pair in sorted(required_pairs - independent_pairs)]
    validated_assertions.sort(key=lambda item: item.assertion_sha256)
    basis = (
        "single_cohort_cross_paper_independence_irrelevant"
        if len(canonical_cohorts) == 1
        else (
            "all_pairwise_independence_source_adjudicated"
            if not unresolved
            else "unresolved_overlap_abstention"
        )
    )
    payload = {
        "receipt_version": "source-backed-synthesis-authorization-v2",
        "input_corpus_sha256": corpus.corpus_sha256,
        "reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "estimate_ids": selected_ids,
        "canonical_cohort_ids": canonical_cohorts,
        "assertions": validated_assertions,
        "unresolved_overlap_pairs": unresolved,
        "publication_source_content_visible": True,
        "benchmark_reference_labels_accessed": False,
        "benchmark_review_verdicts_accessed": False,
        "authorizes_synthesis_input": not unresolved,
        "authorization_basis": basis,
    }
    return SynthesisUnitAuthorizationReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def reverify_synthesis_unit_authorization(
    *,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    receipt: SynthesisUnitAuthorizationReceipt,
    repository_root: Path,
) -> SynthesisUnitAuthorizationReceipt:
    """Recompute a receipt from current graph and source bytes and require equality."""

    frozen = SynthesisUnitAuthorizationReceipt.model_validate(receipt.model_dump(mode="json"))
    replayed = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=frozen.estimate_ids,
        assertions=frozen.assertions,
        repository_root=repository_root,
    )
    if replayed != frozen:
        raise SynthesisAuthorizationError("synthesis_authorization_replay_mismatch")
    return replayed


def reverify_synthesis_unit_authorization_v1(
    *,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    receipt: SynthesisUnitAuthorizationReceiptV1,
    repository_root: Path,
) -> SynthesisUnitAuthorizationReceiptV1:
    """Replay a historical V1 receipt for archival V3 validation only.

    The returned object is not current authorization evidence and must never be
    accepted by V2/V4 normative paths.
    """

    warnings.warn(
        "V1 synthesis authorization is historical-only and has no current authority",
        DeprecationWarning,
        stacklevel=2,
    )

    frozen = SynthesisUnitAuthorizationReceiptV1.model_validate(receipt.model_dump(mode="json"))
    current = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=frozen.estimate_ids,
        assertions=frozen.assertions,
        repository_root=repository_root,
    )
    payload = {
        "receipt_version": "source-backed-synthesis-authorization-v1",
        "input_corpus_sha256": current.input_corpus_sha256,
        "reconciliation_receipt_sha256": current.reconciliation_receipt_sha256,
        "reconciled_graph_sha256": current.reconciled_graph_sha256,
        "estimate_ids": current.estimate_ids,
        "canonical_cohort_ids": current.canonical_cohort_ids,
        "assertions": current.assertions,
        "unresolved_overlap_pairs": current.unresolved_overlap_pairs,
        "reference_labels_accessed": False,
        "review_conclusions_accessed": False,
        "authorizes_synthesis_input": current.authorizes_synthesis_input,
        "authorization_basis": current.authorization_basis,
    }
    replayed = SynthesisUnitAuthorizationReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )
    if replayed != frozen:
        raise SynthesisAuthorizationError("synthesis_authorization_v1_replay_mismatch")
    return replayed


__all__ = [
    "SourceIdentityAssertion",
    "SourceIdentityCitation",
    "SynthesisAuthorizationError",
    "SynthesisUnitAuthorizationReceipt",
    "SynthesisUnitAuthorizationReceiptV1",
    "authorize_synthesis_unit",
    "freeze_source_identity_assertion",
    "freeze_source_identity_citation",
    "reverify_synthesis_unit_authorization",
    "reverify_synthesis_unit_authorization_v1",
]
