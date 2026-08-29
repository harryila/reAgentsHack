"""Bounded live-to-frozen invariant validation for the literature harvester.

This module validates transport, normalization, immutable archiving, and exact local
replay.  It intentionally does not estimate retrieval recall or scientific quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, ValidationInfo, field_validator, model_validator

from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    redact_text,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel

from .archive import ArchivedPayload, ImmutableArchive
from .contracts import FullTextSource, HarvestDocument, SearchBatch, SearchSource
from .pipeline import HarvestQuery, HarvestResult, LiteratureHarvester
from .sources import FrozenCorpusSource

FIXED_OPENALEX_QUERY = "Attention Is All You Need"
FIXED_QUERY_FAMILY = "fixed-cs-open-access-probe"
FIXED_RESULT_LIMIT = 1
FIXED_LIVE_PAGE_SIZE = 1
FIXED_REPLAY_PAGE_SIZE = 1
VALIDATION_SCOPE = "invariant_and_transport_validation_only"
HARVESTER_VALIDATION_VERSION = "2"
HARVESTER_VALIDATION_ENTRYPOINT = "scripts/validate_harvester.py"
HARVESTER_VALIDATION_SOURCE_PATHS = (
    "pyproject.toml",
    HARVESTER_VALIDATION_ENTRYPOINT,
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/harvester/__init__.py",
    "src/literature_multiverse/harvester/archive.py",
    "src/literature_multiverse/harvester/contracts.py",
    "src/literature_multiverse/harvester/http.py",
    "src/literature_multiverse/harvester/pipeline.py",
    "src/literature_multiverse/harvester/sources.py",
    "src/literature_multiverse/harvester/validation.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "src/literature_multiverse/search.py",
    "uv.lock",
)
HARVESTER_VALIDATION_HASH_SECURITY_BOUNDARY = (
    "unkeyed reproducibility and tamper-evidence hashes; not signatures, "
    "authorship proof, freshness proof, or rollback protection"
)
# The public result predates the v2 envelope.  Its complete canonical v1 payload is
# pinned so the deterministic migration below cannot bless a modified historical
# scientific result with current source hashes.
PINNED_PUBLIC_V1_PAYLOAD_SHA256 = "77555f925f2b7dd3f8f9345375066b8ea75ea1fc37003dd795cdb5185526b552"


class HarvesterValidationError(RuntimeError):
    """The bounded validation could not finish or persist a valid result."""


class HarvesterValidationRunFailed(HarvesterValidationError):
    """A failed live validation whose evidence was preserved on disk."""

    def __init__(self, summary: HarvesterValidationSummary) -> None:
        self.summary = summary
        super().__init__(summary.failure.error_type if summary.failure else "unknown_failure")


class ValidationQuery(ContractModel):
    provider: str
    source_scope: str
    family: str
    text: str
    result_limit: int = Field(ge=1, le=10)
    live_page_size: int = Field(ge=1, le=10)
    replay_page_size: int = Field(ge=1, le=10)


class ValidationTimestamps(ContractModel):
    started_at: datetime
    live_search_retrieved_at: list[datetime]
    frozen_corpus_written_at: datetime | None
    replay_search_retrieved_at: list[datetime]
    completed_at: datetime

    @field_validator(
        "started_at",
        "frozen_corpus_written_at",
        "completed_at",
    )
    @classmethod
    def validate_scalar_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("harvester_validation_timestamp_requires_timezone")
        return value

    @field_validator("live_search_retrieved_at", "replay_search_retrieved_at")
    @classmethod
    def validate_timestamp_list(cls, values: list[datetime]) -> list[datetime]:
        if any(value.tzinfo is None or value.utcoffset() is None for value in values):
            raise ValueError("harvester_validation_timestamp_requires_timezone")
        return values


class ValidationCounts(ContractModel):
    live_search_pages: int = Field(ge=0)
    replay_search_pages: int = Field(ge=0)
    live_documents: int = Field(ge=0)
    replay_documents: int = Field(ge=0)
    documents_with_archived_full_text: int = Field(ge=0)
    archive_objects: int = Field(ge=0)
    archive_receipts_verified: int = Field(ge=0)


class ValidationFullText(ContractModel):
    status: Literal["archived", "unavailable"]
    resolver: str | None = None
    media_type: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    retrieved_at: datetime | None = None
    trace_payloads: int = Field(ge=0)
    error_count: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("harvester_validation_full_text_sha256_invalid")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def validate_retrieved_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("harvester_validation_timestamp_requires_timezone")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> ValidationFullText:
        content_fields = (self.bytes, self.sha256, self.retrieved_at)
        if self.status == "archived" and any(value is None for value in content_fields):
            raise ValueError("archived_full_text_requires_content_metadata")
        if self.status == "unavailable" and any(value is not None for value in content_fields):
            raise ValueError("unavailable_full_text_forbids_content_metadata")
        return self


class ValidationDocument(ContractModel):
    document_id: str
    live_normalized_sha256: str
    replay_normalized_sha256: str | None
    normalized_identity_equal: bool
    full_text: ValidationFullText

    @field_validator("live_normalized_sha256", "replay_normalized_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("harvester_validation_document_sha256_invalid")
        return value


class ValidationIdentity(ContractModel):
    live_document_ids: list[str]
    replay_document_ids: list[str]
    ordered_document_ids_equal: bool
    normalized_document_hashes_equal: bool
    exact_identity_equivalence: bool

    @model_validator(mode="after")
    def validate_equivalence(self) -> ValidationIdentity:
        expected_ids_equal = self.live_document_ids == self.replay_document_ids
        if self.ordered_document_ids_equal != expected_ids_equal:
            raise ValueError("harvester_validation_id_equivalence_mismatch")
        expected_exact = self.ordered_document_ids_equal and self.normalized_document_hashes_equal
        if self.exact_identity_equivalence != expected_exact:
            raise ValueError("harvester_validation_exact_equivalence_mismatch")
        return self


class ValidationArchive(ContractModel):
    archive_root: str
    frozen_corpus_path: str | None
    frozen_corpus_sha256: str | None
    frozen_corpus_bytes: int | None = Field(default=None, ge=0)
    all_receipts_verified: bool

    @field_validator("frozen_corpus_sha256")
    @classmethod
    def validate_corpus_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("harvester_validation_corpus_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_corpus_fields(self) -> ValidationArchive:
        supplied = (
            self.frozen_corpus_path is not None,
            self.frozen_corpus_sha256 is not None,
            self.frozen_corpus_bytes is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("harvester_validation_partial_corpus_metadata")
        return self


class ValidationFailure(ContractModel):
    error_type: str
    error_message: str
    cache_failure_path: str


class ValidationReproducibility(ContractModel):
    construction: Literal["live_run", "pinned_public_v1_reseal"]
    generator_entrypoint: Literal["scripts/validate_harvester.py"] = HARVESTER_VALIDATION_ENTRYPOINT
    generator_entrypoint_sha256: str
    source_files_sha256: dict[str, str]
    source_bundle_sha256: str
    legacy_payload_sha256: str | None = None
    hash_security_boundary: Literal[
        "unkeyed reproducibility and tamper-evidence hashes; not signatures, "
        "authorship proof, freshness proof, or rollback protection"
    ] = HARVESTER_VALIDATION_HASH_SECURITY_BOUNDARY

    @field_validator(
        "generator_entrypoint_sha256",
        "source_bundle_sha256",
        "legacy_payload_sha256",
    )
    @classmethod
    def validate_reproducibility_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("harvester_validation_reproducibility_sha256_invalid")
        return value

    @field_validator("source_files_sha256")
    @classmethod
    def validate_source_files(cls, value: dict[str, str]) -> dict[str, str]:
        if tuple(value) != HARVESTER_VALIDATION_SOURCE_PATHS:
            raise ValueError("harvester_validation_source_inventory_mismatch")
        if any(not SHA256_RE.fullmatch(digest) for digest in value.values()):
            raise ValueError("harvester_validation_source_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_reproducibility_contract(self) -> ValidationReproducibility:
        if self.generator_entrypoint_sha256 != self.source_files_sha256.get(
            self.generator_entrypoint
        ):
            raise ValueError("harvester_validation_entrypoint_hash_mismatch")
        if self.source_bundle_sha256 != hash_canonical(self.source_files_sha256):
            raise ValueError("harvester_validation_source_bundle_hash_mismatch")
        if (self.construction == "pinned_public_v1_reseal") != (
            self.legacy_payload_sha256 is not None
        ):
            raise ValueError("harvester_validation_legacy_lineage_mismatch")
        if (
            self.legacy_payload_sha256 is not None
            and self.legacy_payload_sha256 != PINNED_PUBLIC_V1_PAYLOAD_SHA256
        ):
            raise ValueError("harvester_validation_legacy_payload_hash_mismatch")
        return self


class HarvesterValidationSummary(ContractModel):
    harvester_validation_version: Literal["2"] = HARVESTER_VALIDATION_VERSION
    status: Literal["complete", "failed"]
    validation_passed: bool
    validation_scope: Literal["invariant_and_transport_validation_only"] = VALIDATION_SCOPE
    retrieval_recall_evidence: Literal[False] = False
    metadata_only_summary: Literal[True] = True
    query: ValidationQuery
    timestamps: ValidationTimestamps
    counts: ValidationCounts
    archive: ValidationArchive
    identity: ValidationIdentity | None
    documents: list[ValidationDocument]
    warnings: list[str]
    failure: ValidationFailure | None = None
    reproducibility: ValidationReproducibility
    artifact_payload_sha256: str

    @field_validator("artifact_payload_sha256")
    @classmethod
    def validate_artifact_payload_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("harvester_validation_artifact_payload_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_status(self, info: ValidationInfo) -> HarvesterValidationSummary:
        if self.status == "complete" and self.failure is not None:
            raise ValueError("complete_harvester_validation_forbids_failure")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed_harvester_validation_requires_failure")
        if self.validation_passed and (
            self.status != "complete"
            or self.identity is None
            or not self.identity.exact_identity_equivalence
            or not self.archive.all_receipts_verified
        ):
            raise ValueError("harvester_validation_passed_invariants_not_met")
        if not (info.context or {}).get("skip_harvester_validation_self_hash", False):
            payload = self.model_dump(mode="json")
            observed = payload.pop("artifact_payload_sha256")
            if observed != hash_canonical(payload):
                raise ValueError("harvester_validation_artifact_payload_hash_mismatch")
        return self


class _RecordingSearchSource:
    def __init__(self, source: SearchSource) -> None:
        self.source = source
        self.documents: list[HarvestDocument] = []
        self.search_retrieved_at: list[datetime] = []
        self._hash_by_id: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self.source.name

    def search(
        self,
        query: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> SearchBatch:
        batch = self.source.search(query, cursor=cursor, limit=limit)
        self.search_retrieved_at.append(batch.response.retrieved_at)
        for document in batch.documents:
            digest = normalized_document_sha256(document)
            previous = self._hash_by_id.get(document.document_id)
            if previous is not None:
                if previous != digest:
                    raise HarvesterValidationError(
                        f"document_identity_changed_between_pages:{document.document_id}"
                    )
                raise HarvesterValidationError(
                    f"duplicate_document_in_bounded_search:{document.document_id}"
                )
            self._hash_by_id[document.document_id] = digest
            self.documents.append(document)
        return batch


def _normalized_document_payload(document: HarvestDocument) -> dict[str, Any]:
    """Return the complete normalized identity; paper summaries expose only its hash."""

    return {
        "document_id": document.document_id,
        "source": document.source,
        "title": document.title,
        "doi": document.doi,
        "pmid": document.pmid,
        "pmcid": document.pmcid,
        "arxiv_id": document.arxiv_id,
        "first_author": document.first_author,
        "pub_year": document.pub_year,
        "article_type": document.article_type,
        "abstract": document.abstract,
        "publication_status": document.publication_status,
        "full_text_urls": list(document.full_text_urls),
        "landing_page_url": document.landing_page_url,
        "identifiers": dict(sorted(document.identifiers.items())),
    }


def normalized_document_sha256(document: HarvestDocument) -> str:
    """Hash all normalized fields, including text omitted from the paper summary."""

    return hash_canonical(_normalized_document_payload(document))


def harvester_validation_source_hashes(
    repository_root: Path | None = None,
) -> dict[str, str]:
    """Hash the exact CLI and complete direct/transitive local runtime surface."""

    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    missing = [
        relative
        for relative in HARVESTER_VALIDATION_SOURCE_PATHS
        if not (root / relative).is_file()
    ]
    if missing:
        raise HarvesterValidationError(
            f"harvester_validation_source_files_missing:{','.join(missing)}"
        )
    return {
        relative: sha256_file(root / relative) for relative in HARVESTER_VALIDATION_SOURCE_PATHS
    }


def _reproducibility_record(
    *,
    construction: Literal["live_run", "pinned_public_v1_reseal"],
    repository_root: Path | None = None,
    legacy_payload_sha256: str | None = None,
) -> ValidationReproducibility:
    sources = harvester_validation_source_hashes(repository_root)
    return ValidationReproducibility(
        construction=construction,
        generator_entrypoint_sha256=sources[HARVESTER_VALIDATION_ENTRYPOINT],
        source_files_sha256=sources,
        source_bundle_sha256=hash_canonical(sources),
        legacy_payload_sha256=legacy_payload_sha256,
    )


def _seal_summary_payload(
    payload: Mapping[str, Any],
    *,
    construction: Literal["live_run", "pinned_public_v1_reseal"] = "live_run",
    repository_root: Path | None = None,
    legacy_payload_sha256: str | None = None,
) -> HarvesterValidationSummary:
    sealed = deepcopy(dict(payload))
    if "reproducibility" in sealed or "artifact_payload_sha256" in sealed:
        raise HarvesterValidationError("harvester_validation_payload_already_sealed")
    sealed["harvester_validation_version"] = HARVESTER_VALIDATION_VERSION
    sealed["reproducibility"] = _reproducibility_record(
        construction=construction,
        repository_root=repository_root,
        legacy_payload_sha256=legacy_payload_sha256,
    ).model_dump(mode="json")
    sealed["artifact_payload_sha256"] = "0" * 64
    normalized = HarvesterValidationSummary.model_validate(
        sealed,
        context={"skip_harvester_validation_self_hash": True},
    ).model_dump(mode="json")
    normalized.pop("artifact_payload_sha256")
    normalized["artifact_payload_sha256"] = hash_canonical(normalized)
    return HarvesterValidationSummary.model_validate(normalized)


def _path_label(path: Path, *, path_base: Path) -> str:
    try:
        return path.resolve().relative_to(path_base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _frozen_corpus_payload(documents: Sequence[HarvestDocument], *, query: str) -> dict[str, Any]:
    return {
        "format": "literature-multiverse-harvester-validation-corpus-v1",
        "normalized_document_schema": "harvest-document-v1",
        "documents": [_normalized_document_payload(document) for document in documents],
        "search_results": {query: [document.document_id for document in documents]},
    }


def _full_text_by_document(result: HarvestResult) -> dict[str, ValidationFullText]:
    observations: dict[str, ValidationFullText] = {}
    for occurrence in result.occurrences:
        raw_provenance = occurrence.raw_metadata.get("_literature_multiverse_harvester")
        provenance = raw_provenance if isinstance(raw_provenance, Mapping) else {}
        raw_full_text = provenance.get("full_text")
        full_text = raw_full_text if isinstance(raw_full_text, Mapping) else {}
        raw_content = full_text.get("content")
        content = raw_content if isinstance(raw_content, Mapping) else None
        trace = full_text.get("trace")
        errors = full_text.get("errors")
        common = {
            "resolver": str(full_text["resolver"]) if full_text.get("resolver") else None,
            "trace_payloads": len(trace) if isinstance(trace, list) else 0,
            "error_count": len(errors) if isinstance(errors, list) else 0,
        }
        if content is None:
            observation = ValidationFullText(status="unavailable", **common)
        else:
            observation = ValidationFullText(
                status="archived",
                media_type=(
                    str(content["media_type"]) if content.get("media_type") is not None else None
                ),
                bytes=int(content["bytes"]),
                sha256=str(content["sha256"]),
                retrieved_at=datetime.fromisoformat(str(content["retrieved_at"])),
                **common,
            )
        previous = observations.get(occurrence.doc_id)
        if previous is not None and previous != observation:
            raise HarvesterValidationError(f"full_text_observation_changed:{occurrence.doc_id}")
        observations[occurrence.doc_id] = observation
    return observations


def _verified_archive_count(
    archive: ImmutableArchive,
    entries: Sequence[ArchivedPayload],
    *,
    path_base: Path,
) -> int:
    unique = {entry.receipt_path: entry for entry in entries}
    on_disk = {
        _path_label(path, path_base=path_base)
        for path in archive.root.joinpath("receipts").rglob("*.json")
    }
    if on_disk != set(unique):
        raise HarvesterValidationError(
            f"archive_receipt_inventory_mismatch:expected={len(unique)}:observed={len(on_disk)}"
        )
    for entry in unique.values():
        archive.verify(entry)
    return len(unique)


def _summary_counts(
    *,
    live_result: HarvestResult | None,
    replay_result: HarvestResult | None,
    live_documents: int,
    replay_documents: int,
    archive_objects: int,
    receipts_verified: int,
) -> ValidationCounts:
    return ValidationCounts(
        live_search_pages=live_result.search_pages if live_result else 0,
        replay_search_pages=replay_result.search_pages if replay_result else 0,
        live_documents=live_documents,
        replay_documents=replay_documents,
        documents_with_archived_full_text=(
            live_result.documents_with_full_text if live_result else 0
        ),
        archive_objects=archive_objects,
        archive_receipts_verified=receipts_verified,
    )


def _archive_summary(
    *,
    archive: ImmutableArchive,
    corpus_path: Path,
    path_base: Path,
    all_receipts_verified: bool,
) -> ValidationArchive:
    if corpus_path.is_file():
        return ValidationArchive(
            archive_root=_path_label(archive.root, path_base=path_base),
            frozen_corpus_path=_path_label(corpus_path, path_base=path_base),
            frozen_corpus_sha256=sha256_file(corpus_path),
            frozen_corpus_bytes=corpus_path.stat().st_size,
            all_receipts_verified=all_receipts_verified,
        )
    return ValidationArchive(
        archive_root=_path_label(archive.root, path_base=path_base),
        frozen_corpus_path=None,
        frozen_corpus_sha256=None,
        frozen_corpus_bytes=None,
        all_receipts_verified=all_receipts_verified,
    )


def build_harvester_validation_summary(
    *,
    query_spec: ValidationQuery,
    started_at: datetime,
    frozen_corpus_written_at: datetime,
    completed_at: datetime,
    live_capture: _RecordingSearchSource,
    replay_capture: _RecordingSearchSource,
    live_result: HarvestResult,
    replay_result: HarvestResult,
    archive: ImmutableArchive,
    archive_entries: Sequence[ArchivedPayload],
    corpus_path: Path,
    path_base: Path,
) -> HarvesterValidationSummary:
    """Verify archive receipts and construct the closed, metadata-only paper view."""

    receipts_verified = _verified_archive_count(archive, archive_entries, path_base=path_base)
    live_ids = [document.document_id for document in live_capture.documents]
    replay_ids = [document.document_id for document in replay_capture.documents]
    live_hashes = {
        document.document_id: normalized_document_sha256(document)
        for document in live_capture.documents
    }
    replay_hashes = {
        document.document_id: normalized_document_sha256(document)
        for document in replay_capture.documents
    }
    ids_equal = live_ids == replay_ids
    hashes_equal = live_hashes == replay_hashes
    identity = ValidationIdentity(
        live_document_ids=live_ids,
        replay_document_ids=replay_ids,
        ordered_document_ids_equal=ids_equal,
        normalized_document_hashes_equal=hashes_equal,
        exact_identity_equivalence=ids_equal and hashes_equal,
    )
    full_text = _full_text_by_document(live_result)
    documents = [
        ValidationDocument(
            document_id=document.document_id,
            live_normalized_sha256=live_hashes[document.document_id],
            replay_normalized_sha256=replay_hashes.get(document.document_id),
            normalized_identity_equal=(
                replay_hashes.get(document.document_id) == live_hashes[document.document_id]
            ),
            full_text=full_text.get(
                document.document_id,
                ValidationFullText(status="unavailable", trace_payloads=0, error_count=0),
            ),
        )
        for document in live_capture.documents
    ]
    unique_entries = {entry.receipt_path: entry for entry in archive_entries}
    warnings = {
        "invariant_transport_validation_only_not_retrieval_recall_evidence",
        "paper_summary_omits_titles_abstracts_and_full_text",
        *live_result.warnings,
        *replay_result.warnings,
    }
    if not live_ids:
        warnings.add("fixed_query_returned_no_documents")
    if not identity.exact_identity_equivalence:
        warnings.add("live_replay_identity_mismatch")
    if not any(document.full_text.status == "archived" for document in documents):
        warnings.add("no_open_full_text_archived")
    passed = bool(live_ids) and identity.exact_identity_equivalence
    return _seal_summary_payload(
        {
            "status": "complete",
            "validation_passed": passed,
            "query": query_spec,
            "timestamps": ValidationTimestamps(
                started_at=started_at,
                live_search_retrieved_at=live_capture.search_retrieved_at,
                frozen_corpus_written_at=frozen_corpus_written_at,
                replay_search_retrieved_at=replay_capture.search_retrieved_at,
                completed_at=completed_at,
            ),
            "counts": _summary_counts(
                live_result=live_result,
                replay_result=replay_result,
                live_documents=len(live_capture.documents),
                replay_documents=len(replay_capture.documents),
                archive_objects=len(unique_entries),
                receipts_verified=receipts_verified,
            ),
            "archive": _archive_summary(
                archive=archive,
                corpus_path=corpus_path,
                path_base=path_base,
                all_receipts_verified=True,
            ),
            "identity": identity,
            "documents": documents,
            "warnings": sorted(warnings),
            "failure": None,
        }
    )


def _failure_summary(
    *,
    error: Exception,
    failure_path: Path,
    query_spec: ValidationQuery,
    started_at: datetime,
    completed_at: datetime,
    frozen_corpus_written_at: datetime | None,
    live_capture: _RecordingSearchSource,
    replay_capture: _RecordingSearchSource | None,
    live_result: HarvestResult | None,
    replay_result: HarvestResult | None,
    archive: ImmutableArchive,
    corpus_path: Path,
    path_base: Path,
) -> HarvesterValidationSummary:
    receipt_count = len(list(archive.root.joinpath("receipts").rglob("*.json")))
    return _seal_summary_payload(
        {
            "status": "failed",
            "validation_passed": False,
            "query": query_spec,
            "timestamps": ValidationTimestamps(
                started_at=started_at,
                live_search_retrieved_at=live_capture.search_retrieved_at,
                frozen_corpus_written_at=frozen_corpus_written_at,
                replay_search_retrieved_at=(
                    replay_capture.search_retrieved_at if replay_capture else []
                ),
                completed_at=completed_at,
            ),
            "counts": _summary_counts(
                live_result=live_result,
                replay_result=replay_result,
                live_documents=len(live_capture.documents),
                replay_documents=len(replay_capture.documents) if replay_capture else 0,
                archive_objects=receipt_count,
                receipts_verified=0,
            ),
            "archive": _archive_summary(
                archive=archive,
                corpus_path=corpus_path,
                path_base=path_base,
                all_receipts_verified=False,
            ),
            "identity": None,
            "documents": [],
            "warnings": [
                "invariant_transport_validation_only_not_retrieval_recall_evidence",
                "live_validation_failed_partial_cache_preserved",
            ],
            "failure": ValidationFailure(
                error_type=type(error).__name__,
                error_message=redact_text(str(error))[:2000],
                cache_failure_path=_path_label(failure_path, path_base=path_base),
            ),
        }
    )


def run_harvester_validation_cycle(
    *,
    live_search_source: SearchSource,
    live_full_text_source: FullTextSource | None,
    query: str,
    query_family: str,
    result_limit: int,
    live_page_size: int,
    replay_page_size: int,
    source_scope: str,
    cache_dir: Path,
    summary_path: Path,
    path_base: Path,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> HarvesterValidationSummary:
    """Run one bounded source query, freeze it, replay it, and verify all receipts."""

    corpus_path = cache_dir / "frozen_metadata_corpus.json"
    failure_path = cache_dir / "live_failure.json"
    archive_root = cache_dir / "archive"
    existing = next(
        (path for path in (summary_path, corpus_path, failure_path, archive_root) if path.exists()),
        None,
    )
    if existing is not None:
        raise OutputExistsError(existing.as_posix())

    started_at = clock()
    query_spec = ValidationQuery(
        provider=live_search_source.name,
        source_scope=source_scope,
        family=query_family,
        text=query,
        result_limit=result_limit,
        live_page_size=live_page_size,
        replay_page_size=replay_page_size,
    )
    archive = ImmutableArchive(archive_root, path_base=path_base)
    live_capture = _RecordingSearchSource(live_search_source)
    replay_capture: _RecordingSearchSource | None = None
    live_result: HarvestResult | None = None
    replay_result: HarvestResult | None = None
    frozen_at: datetime | None = None
    try:
        live_result = LiteratureHarvester(
            live_capture,
            archive,
            full_text_source=live_full_text_source,
            page_size=live_page_size,
        ).run(
            [HarvestQuery(family=query_family, query=query)],
            per_query_limit=result_limit,
        )
        atomic_write_json(
            corpus_path,
            _frozen_corpus_payload(live_capture.documents, query=query),
        )
        frozen_at = clock()
        frozen_source = FrozenCorpusSource(corpus_path, expected_sha256=sha256_file(corpus_path))
        replay_capture = _RecordingSearchSource(frozen_source)
        replay_result = LiteratureHarvester(
            replay_capture,
            archive,
            full_text_source=None,
            page_size=replay_page_size,
        ).run(
            [HarvestQuery(family=query_family, query=query)],
            per_query_limit=result_limit,
        )
        entries = (*live_result.archive_entries, *replay_result.archive_entries)
        summary = build_harvester_validation_summary(
            query_spec=query_spec,
            started_at=started_at,
            frozen_corpus_written_at=frozen_at,
            completed_at=clock(),
            live_capture=live_capture,
            replay_capture=replay_capture,
            live_result=live_result,
            replay_result=replay_result,
            archive=archive,
            archive_entries=entries,
            corpus_path=corpus_path,
            path_base=path_base,
        )
        if summary_contains_forbidden_text_fields(summary):
            raise HarvesterValidationError("paper_summary_contains_forbidden_text_field")
        atomic_write_json(summary_path, summary)
        return summary
    except Exception as error:
        completed_at = clock()
        summary = _failure_summary(
            error=error,
            failure_path=failure_path,
            query_spec=query_spec,
            started_at=started_at,
            completed_at=completed_at,
            frozen_corpus_written_at=frozen_at,
            live_capture=live_capture,
            replay_capture=replay_capture,
            live_result=live_result,
            replay_result=replay_result,
            archive=archive,
            corpus_path=corpus_path,
            path_base=path_base,
        )
        atomic_write_json(
            failure_path,
            {
                "harvester_validation_failure_version": "1",
                "query": query_spec,
                "occurred_at": completed_at,
                "error_type": summary.failure.error_type if summary.failure else "unknown",
                "error_message": summary.failure.error_message if summary.failure else "",
                "partial_cache_preserved": True,
            },
        )
        atomic_write_json(summary_path, summary)
        raise HarvesterValidationRunFailed(summary) from error


def validate_harvester_validation_summary(
    value: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    require_current_sources: bool = True,
) -> HarvesterValidationSummary:
    """Validate structure/self-integrity and, by default, exact current source bytes."""

    try:
        summary = HarvesterValidationSummary.model_validate(value)
    except ValidationError as exc:
        raise HarvesterValidationError("harvester_validation_summary_contract_invalid") from exc
    if summary_contains_forbidden_text_fields(summary):
        raise HarvesterValidationError("paper_summary_contains_forbidden_text_field")
    if require_current_sources:
        current = harvester_validation_source_hashes(repository_root)
        if summary.reproducibility.source_files_sha256 != current:
            raise HarvesterValidationError("harvester_validation_source_lineage_stale")
    return summary


def load_harvester_validation_summary(
    path: Path,
    *,
    repository_root: Path | None = None,
    require_current_sources: bool = True,
) -> HarvesterValidationSummary:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarvesterValidationError(
            f"harvester_validation_summary_unreadable:{path.as_posix()}"
        ) from exc
    if not isinstance(value, Mapping):
        raise HarvesterValidationError("harvester_validation_summary_root_not_object")
    return validate_harvester_validation_summary(
        value,
        repository_root=repository_root,
        require_current_sources=require_current_sources,
    )


def reseal_pinned_public_harvester_validation_summary(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> HarvesterValidationSummary:
    """Deterministically add current v2 envelope to the exact pinned public v1 result.

    A previously sealed v2 artifact may be resealed after source-only hardening.  Its
    self-hash must validate first, and stripping the v2 envelope must recover the exact
    pinned v1 payload.  Therefore this operation cannot legitimize modified scientific
    fields or a different live capture.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarvesterValidationError(
            f"harvester_validation_summary_unreadable:{path.as_posix()}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise HarvesterValidationError("harvester_validation_summary_root_not_object")
    legacy = deepcopy(dict(raw))
    version = legacy.get("harvester_validation_version")
    if version == HARVESTER_VALIDATION_VERSION:
        # Validate self-integrity and the closed v2 structure without requiring the
        # old source hashes to match files that this reseal is intentionally updating.
        HarvesterValidationSummary.model_validate(legacy)
        legacy.pop("artifact_payload_sha256")
        legacy.pop("reproducibility")
        legacy["harvester_validation_version"] = "1"
    elif version != "1":
        raise HarvesterValidationError("harvester_validation_reseal_version_unsupported")
    observed_legacy_hash = hash_canonical(legacy)
    if observed_legacy_hash != PINNED_PUBLIC_V1_PAYLOAD_SHA256:
        raise HarvesterValidationError("harvester_validation_reseal_legacy_payload_mismatch")
    legacy["harvester_validation_version"] = HARVESTER_VALIDATION_VERSION
    summary = _seal_summary_payload(
        legacy,
        construction="pinned_public_v1_reseal",
        repository_root=repository_root,
        legacy_payload_sha256=observed_legacy_hash,
    )
    atomic_write_json(path, summary, force=True)
    return load_harvester_validation_summary(path, repository_root=repository_root)


def summary_contains_forbidden_text_fields(summary: HarvesterValidationSummary) -> bool:
    """Defensive audit for paper artifacts; normalized text belongs only in cache."""

    encoded = json.dumps(summary.model_dump(mode="json"), sort_keys=True).casefold()
    forbidden_keys = ('"abstract"', '"title"', '"full_text_body"', '"body"')
    return any(key in encoded for key in forbidden_keys)


__all__ = [
    "FIXED_LIVE_PAGE_SIZE",
    "FIXED_OPENALEX_QUERY",
    "FIXED_QUERY_FAMILY",
    "FIXED_REPLAY_PAGE_SIZE",
    "FIXED_RESULT_LIMIT",
    "HARVESTER_VALIDATION_ENTRYPOINT",
    "HARVESTER_VALIDATION_SOURCE_PATHS",
    "HARVESTER_VALIDATION_VERSION",
    "HarvesterValidationError",
    "HarvesterValidationRunFailed",
    "HarvesterValidationSummary",
    "ValidationDocument",
    "ValidationFullText",
    "ValidationIdentity",
    "ValidationQuery",
    "ValidationReproducibility",
    "build_harvester_validation_summary",
    "harvester_validation_source_hashes",
    "load_harvester_validation_summary",
    "normalized_document_sha256",
    "reseal_pinned_public_harvester_validation_summary",
    "run_harvester_validation_cycle",
    "summary_contains_forbidden_text_fields",
    "validate_harvester_validation_summary",
]
