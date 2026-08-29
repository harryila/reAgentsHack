"""Bridge immutable archived corpora into native extraction source manifests.

The bridge performs source identity and availability accounting only.  It never emits
an evidence graph, effect estimate, or scientific estimability judgment.  Every input
covered here has previously been opened, so every ledger row is permanently marked as
diagnostic and ineligible for a pristine final holdout.

Two local source layouts are supported:

* Antiox ``papers.parquet`` plus the derived, archived ``source_lines.json`` object;
* revision-pinned MetaSyn Parquet shards, where each row is addressed by physical
  shard, row group, row offset, and corpus ``ID``.

The downstream :class:`NativeSourceManifest` stays minimal and compatible with the
native extractor.  A separate self-hashed :class:`DiagnosticSourceLedger` records
actual containing-artifact hashes, exact locators, semantic row/payload hashes, source
availability, and the diagnostic access contract.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import pyarrow.parquet as pq
from pydantic import Field, field_validator, model_validator

from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_retrieval import (
    MetaSynCorpusError,
    verify_corpus_manifest,
)
from literature_multiverse.models import SHA256_RE, ContractModel, normalize_doi
from literature_multiverse.native_extraction import (
    NativeSourceManifest,
    NativeSourceRecord,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact


class SourceManifestBridgeError(ValueError):
    """An archived source cannot be represented without weakening provenance."""


class SourceCorpusKind(StrEnum):
    ANTIOX = "antiox"
    METASYN = "metasyn"


class SourceContentScope(StrEnum):
    NUMBERED_SOURCE_LINES = "numbered_source_lines"
    FULL_TEXT_SECTIONS = "full_text_sections"
    TITLE_ABSTRACT = "title_abstract"
    ABSTRACT_ONLY = "abstract_only"
    TITLE_ONLY = "title_only"
    UNAVAILABLE = "unavailable"


class DiagnosticAccessState(ContractModel):
    """Permanent access classification for already-opened local inputs."""

    labels_previously_opened: Literal[True] = True
    pristine_final_holdout_eligible: Literal[False] = False
    scientific_role: Literal["diagnostic_only"] = "diagnostic_only"


class SourceArtifactBinding(ContractModel):
    """Actual immutable file bytes consumed while constructing a bridge."""

    role: Annotated[str, Field(min_length=1)]
    artifact_path: Annotated[str, Field(min_length=1)]
    sha256: str
    media_type: Annotated[str, Field(min_length=1)]
    rows: Annotated[int, Field(ge=0)] | None = None

    @field_validator("artifact_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("source_binding_path_must_be_repository_relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("source_binding_sha256_invalid")
        return value


class DiagnosticSourceRecord(ContractModel):
    """One source-only row; no model output or scientific label is present."""

    record_version: Literal["diagnostic-source-record-v1"] = (
        "diagnostic-source-record-v1"
    )
    corpus_kind: SourceCorpusKind
    doc_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    metadata_artifact_path: Annotated[str, Field(min_length=1)]
    metadata_artifact_sha256: str
    metadata_locator: Annotated[str, Field(min_length=1)]
    metadata_row_sha256: str
    source_available: bool
    source_document: SourceDocumentArtifact | None
    source_payload_sha256: str | None
    content_scope: SourceContentScope
    included_in_native_manifest: bool
    manifest_exclusion_reason: str | None
    extraction_attempted: Literal[False] = False
    estimability_status: Literal["not_assessed_source_only"] = (
        "not_assessed_source_only"
    )
    access_state: DiagnosticAccessState
    warnings: list[str] = Field(default_factory=list)
    record_sha256: str

    @field_validator("metadata_artifact_path")
    @classmethod
    def validate_metadata_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("diagnostic_metadata_path_must_be_repository_relative")
        return value

    @field_validator(
        "metadata_artifact_sha256",
        "metadata_row_sha256",
        "source_payload_sha256",
        "record_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"diagnostic_source_sha256_invalid:{info.field_name}")
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("diagnostic_source_warnings_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> DiagnosticSourceRecord:
        if self.source_available != (self.source_document is not None):
            raise ValueError("diagnostic_source_availability_document_mismatch")
        if self.source_available != (self.source_payload_sha256 is not None):
            raise ValueError("diagnostic_source_availability_payload_hash_mismatch")
        if self.source_available == (self.content_scope is SourceContentScope.UNAVAILABLE):
            raise ValueError("diagnostic_source_availability_scope_mismatch")
        if self.included_in_native_manifest:
            if not self.source_available:
                raise ValueError("native_manifest_record_requires_source")
            if self.manifest_exclusion_reason is not None:
                raise ValueError("included_source_forbids_manifest_exclusion_reason")
        elif not self.manifest_exclusion_reason:
            raise ValueError("excluded_source_requires_manifest_exclusion_reason")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if hash_canonical(payload) != self.record_sha256:
            raise ValueError("diagnostic_source_record_hash_mismatch")
        return self


class DiagnosticSourceLedger(ContractModel):
    """Self-hashed bridge ledger bound to a downstream native source manifest."""

    ledger_version: Literal["diagnostic-source-ledger-v2"] = (
        "diagnostic-source-ledger-v2"
    )
    corpus_kind: SourceCorpusKind
    question_id: Annotated[
        str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    ]
    dataset_version: Annotated[str, Field(min_length=1)]
    source_revision: str | None
    license_status: Annotated[str, Field(min_length=1)]
    selection_scope: Annotated[str, Field(min_length=1)]
    access_state: DiagnosticAccessState
    native_source_manifest_sha256: str
    artifacts: Annotated[list[SourceArtifactBinding], Field(min_length=1)]
    records: Annotated[list[DiagnosticSourceRecord], Field(min_length=1)]
    source_records: Annotated[int, Field(ge=1)]
    native_manifest_records: Annotated[int, Field(ge=1)]
    source_available_records: Annotated[int, Field(ge=0)]
    source_absent_records: Annotated[int, Field(ge=0)]
    manifest_excluded_records: Annotated[int, Field(ge=0)]
    content_scope_counts: dict[str, Annotated[int, Field(ge=0)]]
    ledger_sha256: str

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("diagnostic_source_revision_empty")
        return value

    @field_validator("native_source_manifest_sha256", "ledger_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("diagnostic_source_ledger_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_ledger(self) -> DiagnosticSourceLedger:
        artifact_keys = [(item.role, item.artifact_path) for item in self.artifacts]
        if artifact_keys != sorted(set(artifact_keys)):
            raise ValueError("diagnostic_source_artifacts_not_sorted_unique")
        doc_ids = [record.doc_id for record in self.records]
        if doc_ids != sorted(set(doc_ids)):
            raise ValueError("diagnostic_source_records_not_sorted_unique")
        if {record.corpus_kind for record in self.records} != {self.corpus_kind}:
            raise ValueError("diagnostic_source_record_corpus_mismatch")
        if any(record.access_state != self.access_state for record in self.records):
            raise ValueError("diagnostic_source_record_access_state_mismatch")
        expected_counts = {
            "source_records": len(self.records),
            "native_manifest_records": sum(
                record.included_in_native_manifest for record in self.records
            ),
            "source_available_records": sum(
                record.source_available for record in self.records
            ),
            "source_absent_records": sum(
                not record.source_available for record in self.records
            ),
            "manifest_excluded_records": sum(
                not record.included_in_native_manifest for record in self.records
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"diagnostic_source_count_mismatch:{field_name}")
        expected_scopes = dict(
            sorted(Counter(record.content_scope.value for record in self.records).items())
        )
        if self.content_scope_counts != expected_scopes:
            raise ValueError("diagnostic_source_content_scope_counts_mismatch")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if hash_canonical(payload) != self.ledger_sha256:
            raise ValueError("diagnostic_source_ledger_hash_mismatch")
        return self


def _repository_file(path: Path, repository_root: Path) -> tuple[Path, str]:
    root = repository_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SourceManifestBridgeError(f"source_artifact_missing:{candidate}") from exc
    if not resolved.is_file():
        raise SourceManifestBridgeError(f"source_artifact_not_file:{candidate}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceManifestBridgeError(
            f"source_artifact_outside_repository:{resolved}"
        ) from exc
    return resolved, relative


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceManifestBridgeError(f"source_json_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise SourceManifestBridgeError(f"source_json_root_not_object:{path}")
    return value


def _optional_doi(value: Any, warnings: list[str]) -> str | None:
    if value is None or not str(value).strip():
        return None
    try:
        return normalize_doi(str(value))
    except ValueError:
        warnings.append("invalid_doi_omitted")
        return None


def _optional_pmid(value: Any, warnings: list[str]) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip()
    if normalized.isdigit():
        return normalized
    warnings.append("invalid_pmid_omitted")
    return None


def _optional_year(value: Any, warnings: list[str]) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        warnings.append("invalid_publication_year_omitted")
        return None
    if not 1000 <= parsed <= 3000:
        warnings.append("invalid_publication_year_omitted")
        return None
    return parsed


def _optional_title(value: Any, warnings: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        warnings.append("non_string_title_omitted")
        return None
    return value.strip() or None


def _json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _freeze_record(payload: Mapping[str, Any]) -> DiagnosticSourceRecord:
    return DiagnosticSourceRecord.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )


def _freeze_ledger(
    *,
    corpus_kind: SourceCorpusKind,
    question_id: str,
    dataset_version: str,
    source_revision: str | None,
    license_status: str,
    selection_scope: str,
    manifest: NativeSourceManifest,
    artifacts: list[SourceArtifactBinding],
    records: list[DiagnosticSourceRecord],
) -> DiagnosticSourceLedger:
    records = sorted(records, key=lambda record: record.doc_id)
    scope_counts = dict(
        sorted(Counter(record.content_scope.value for record in records).items())
    )
    payload = {
        "ledger_version": "diagnostic-source-ledger-v2",
        "corpus_kind": corpus_kind,
        "question_id": question_id,
        "dataset_version": dataset_version,
        "source_revision": source_revision,
        "license_status": license_status,
        "selection_scope": selection_scope,
        "access_state": DiagnosticAccessState(),
        "native_source_manifest_sha256": hash_canonical(manifest),
        "artifacts": sorted(artifacts, key=lambda item: (item.role, item.artifact_path)),
        "records": records,
        "source_records": len(records),
        "native_manifest_records": sum(
            record.included_in_native_manifest for record in records
        ),
        "source_available_records": sum(record.source_available for record in records),
        "source_absent_records": sum(not record.source_available for record in records),
        "manifest_excluded_records": sum(
            not record.included_in_native_manifest for record in records
        ),
        "content_scope_counts": scope_counts,
    }
    return DiagnosticSourceLedger.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def _validate_expected_hash(
    *, path: str, observed: str, expected: str | None
) -> None:
    if expected is not None:
        if not SHA256_RE.fullmatch(expected):
            raise SourceManifestBridgeError(f"expected_source_hash_invalid:{path}")
        if observed != expected:
            raise SourceManifestBridgeError(
                f"source_artifact_hash_mismatch:{path}:"
                f"expected={expected}:observed={observed}"
            )


def _antiox_lines_payload(value: Any, *, doc_id: str) -> tuple[bool, int]:
    if not isinstance(value, dict):
        raise SourceManifestBridgeError(f"antiox_source_lines_not_object:{doc_id}")
    if not value:
        return False, 0
    for line_id, line in value.items():
        if (
            not isinstance(line_id, str)
            or not line_id.startswith("L")
            or not line_id[1:].isdigit()
            or not isinstance(line, dict)
            or set(line) != {"section", "text"}
            or not isinstance(line["section"], str)
            or not isinstance(line["text"], str)
        ):
            raise SourceManifestBridgeError(
                f"antiox_source_line_contract_invalid:{doc_id}:{line_id}"
            )
    return True, len(value)


def build_antiox_native_source_bridge(
    *,
    question_id: str,
    papers_path: Path,
    source_lines_path: Path,
    repository_root: Path,
    scope: Literal[
        "successful_screened_in",
        "legacy_eligible",
        "all_source_available",
    ] = (
        "successful_screened_in"
    ),
    expected_papers_sha256: str | None = None,
    expected_source_lines_sha256: str | None = None,
) -> tuple[NativeSourceManifest, DiagnosticSourceLedger]:
    """Build an Antiox source manifest without consulting legacy findings."""

    papers_file, papers_relative = _repository_file(papers_path, repository_root)
    lines_file, lines_relative = _repository_file(source_lines_path, repository_root)
    papers_sha256 = sha256_file(papers_file)
    lines_sha256 = sha256_file(lines_file)
    _validate_expected_hash(
        path=papers_relative,
        observed=papers_sha256,
        expected=expected_papers_sha256,
    )
    _validate_expected_hash(
        path=lines_relative,
        observed=lines_sha256,
        expected=expected_source_lines_sha256,
    )
    source_lines = _json_object(lines_file)

    try:
        parquet = pq.ParquetFile(papers_file)
    except Exception as exc:
        raise SourceManifestBridgeError(f"antiox_papers_unreadable:{papers_relative}") from exc
    required_columns = {
        "paper_id",
        "doc_id",
        "doi",
        "pmid",
        "title",
        "pub_year",
        "screen_status",
        "map_status",
    }
    missing_columns = sorted(required_columns - set(parquet.schema_arrow.names))
    if missing_columns:
        raise SourceManifestBridgeError(
            f"antiox_papers_columns_missing:{','.join(missing_columns)}"
        )
    if scope == "legacy_eligible" and "eligible" not in parquet.schema_arrow.names:
        raise SourceManifestBridgeError("antiox_papers_columns_missing:eligible")

    access = DiagnosticAccessState()
    native_records: list[NativeSourceRecord] = []
    ledger_records: list[DiagnosticSourceRecord] = []
    seen_doc_ids: set[str] = set()
    seen_paper_ids: set[str] = set()
    paper_doc_ids: set[str] = set()
    for row_group in range(parquet.metadata.num_row_groups):
        rows = parquet.read_row_group(row_group).to_pylist()
        for row_in_group, row in enumerate(rows):
            doc_id = str(row.get("doc_id") or "").strip()
            paper_id = str(row.get("paper_id") or "").strip()
            if not doc_id or not paper_id:
                raise SourceManifestBridgeError("antiox_paper_identity_missing")
            if doc_id in seen_doc_ids:
                raise SourceManifestBridgeError(f"antiox_doc_id_duplicate:{doc_id}")
            if paper_id in seen_paper_ids:
                raise SourceManifestBridgeError(f"antiox_paper_id_duplicate:{paper_id}")
            seen_doc_ids.add(doc_id)
            seen_paper_ids.add(paper_id)
            paper_doc_ids.add(doc_id)

            warnings: list[str] = []
            publication_id = f"antiox-publication:{doc_id}"
            publication = PublicationIdentity(
                publication_id=publication_id,
                paper_id=paper_id,
                doc_id=doc_id,
                doi=_optional_doi(row.get("doi"), warnings),
                pmid=_optional_pmid(row.get("pmid"), warnings),
                title=_optional_title(row.get("title"), warnings),
                publication_year=_optional_year(row.get("pub_year"), warnings),
            )
            raw_source = source_lines.get(doc_id)
            source_available = False
            line_count = 0
            if raw_source is not None:
                source_available, line_count = _antiox_lines_payload(
                    raw_source,
                    doc_id=doc_id,
                )
            source_document = (
                SourceDocumentArtifact(
                    artifact_path=lines_relative,
                    sha256=lines_sha256,
                    media_type="application/json",
                    source_locator=(
                        f"json:{lines_relative}#/{_json_pointer(doc_id)}"
                    ),
                )
                if source_available
                else None
            )
            in_scope = (
                scope == "all_source_available"
                or (scope == "legacy_eligible" and row.get("eligible") is True)
                or (
                    scope == "successful_screened_in"
                    and
                    row.get("screen_status") == "included"
                    and row.get("map_status") == "success"
                )
            )
            included = in_scope and source_available
            if not in_scope:
                exclusion_reason = f"outside_{scope}_scope"
            elif not source_available:
                exclusion_reason = "source_document_absent_or_empty"
            else:
                exclusion_reason = None
            if raw_source is None:
                warnings.append("source_document_absent")
            elif not source_available:
                warnings.append("source_document_empty")

            if included:
                assert source_document is not None
                native_records.append(
                    NativeSourceRecord(
                        doc_id=doc_id,
                        publication=publication,
                        source_document=source_document,
                    )
                )
            metadata_locator = (
                f"parquet:{papers_relative}#row_group={row_group}"
                f"&row_in_group={row_in_group}&index_base=0"
            )
            ledger_payload = {
                "record_version": "diagnostic-source-record-v1",
                "corpus_kind": SourceCorpusKind.ANTIOX,
                "doc_id": doc_id,
                "publication_id": publication_id,
                "paper_id": paper_id,
                "metadata_artifact_path": papers_relative,
                "metadata_artifact_sha256": papers_sha256,
                "metadata_locator": metadata_locator,
                "metadata_row_sha256": hash_canonical(row),
                "source_available": source_available,
                "source_document": source_document,
                "source_payload_sha256": (
                    hash_canonical(raw_source) if source_available else None
                ),
                "content_scope": (
                    SourceContentScope.NUMBERED_SOURCE_LINES
                    if source_available
                    else SourceContentScope.UNAVAILABLE
                ),
                "included_in_native_manifest": included,
                "manifest_exclusion_reason": exclusion_reason,
                "extraction_attempted": False,
                "estimability_status": "not_assessed_source_only",
                "access_state": access,
                "warnings": sorted(set(warnings)),
            }
            ledger_records.append(_freeze_record(ledger_payload))
            if source_available and line_count < 1:
                raise AssertionError("available Antiox source must contain a line")

    extra_source_doc_ids = sorted(set(source_lines) - paper_doc_ids)
    if extra_source_doc_ids:
        raise SourceManifestBridgeError(
            f"antiox_source_doc_ids_missing_from_papers:{extra_source_doc_ids[:10]}"
        )
    native_records.sort(key=lambda record: record.doc_id)
    if not native_records:
        raise SourceManifestBridgeError("antiox_native_source_manifest_empty")
    manifest = NativeSourceManifest(
        question_id=question_id,
        records=native_records,
    )
    artifacts = [
        SourceArtifactBinding(
            role="publication_metadata",
            artifact_path=papers_relative,
            sha256=papers_sha256,
            media_type="application/vnd.apache.parquet",
            rows=parquet.metadata.num_rows,
        ),
        SourceArtifactBinding(
            role="source_payload",
            artifact_path=lines_relative,
            sha256=lines_sha256,
            media_type="application/json",
            rows=len(source_lines),
        ),
    ]
    ledger = _freeze_ledger(
        corpus_kind=SourceCorpusKind.ANTIOX,
        question_id=question_id,
        dataset_version=(
            f"local-antiox-archive@papers-{papers_sha256[:12]}"
            f"+source-lines-{lines_sha256[:12]}"
        ),
        source_revision=None,
        license_status="local_archived_inputs_redistribution_not_assessed",
        selection_scope=scope,
        manifest=manifest,
        artifacts=artifacts,
        records=ledger_records,
    )
    return manifest, ledger


def _metasyn_content_scope(row: Mapping[str, Any]) -> SourceContentScope:
    title = isinstance(row.get("title"), str) and bool(row["title"].strip())
    abstract = isinstance(row.get("abstract"), str) and bool(row["abstract"].strip())
    sections = row.get("sections")
    has_sections = isinstance(sections, list) and any(
        isinstance(section, dict)
        and isinstance(section.get("text"), str)
        and bool(section["text"].strip())
        for section in sections
    )
    if has_sections:
        return SourceContentScope.FULL_TEXT_SECTIONS
    if title and abstract:
        return SourceContentScope.TITLE_ABSTRACT
    if abstract:
        return SourceContentScope.ABSTRACT_ONLY
    if title:
        return SourceContentScope.TITLE_ONLY
    return SourceContentScope.UNAVAILABLE


def build_metasyn_native_source_bridge(
    *,
    question_id: str,
    corpus_manifest_path: Path,
    repository_root: Path,
    corpus_ids: set[int] | None,
) -> tuple[NativeSourceManifest, DiagnosticSourceLedger]:
    """Build a revision-pinned MetaSyn source manifest for an explicit row subset.

    Passing ``None`` deliberately requests every corpus row.  Command-line callers
    must spell this as ``--all-corpus-rows`` so a 140k-record artifact is not created by
    accident.  No review labels or matched-paper annotations are read here.
    """

    manifest_file, manifest_relative = _repository_file(
        corpus_manifest_path,
        repository_root,
    )
    if corpus_ids is not None:
        if not corpus_ids:
            raise SourceManifestBridgeError("metasyn_corpus_id_selection_empty")
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in corpus_ids
        ):
            raise SourceManifestBridgeError("metasyn_corpus_ids_must_be_nonnegative_integers")
    try:
        corpus_manifest, shard_paths = verify_corpus_manifest(
            manifest_file,
            repository_root=repository_root.resolve(),
        )
    except MetaSynCorpusError as exc:
        raise SourceManifestBridgeError(str(exc)) from exc

    license_file, license_relative = _repository_file(
        Path(corpus_manifest.license_notice.path),
        repository_root,
    )
    shard_by_name = {shard.path: shard for shard in corpus_manifest.shards}
    access = DiagnosticAccessState()
    native_records: list[NativeSourceRecord] = []
    ledger_records: list[DiagnosticSourceRecord] = []
    all_seen_ids: set[int] = set()
    selected_seen_ids: set[int] = set()
    for shard_path in shard_paths:
        shard_relative = shard_path.resolve().relative_to(
            repository_root.resolve()
        ).as_posix()
        shard_contract = shard_by_name[shard_path.name]
        parquet = pq.ParquetFile(shard_path)
        required_columns = {
            "ID",
            "pmid",
            "title",
            "abstract",
            "doi",
            "year",
            "sections",
        }
        missing_columns = sorted(required_columns - set(parquet.schema_arrow.names))
        if missing_columns:
            raise SourceManifestBridgeError(
                f"metasyn_source_columns_missing:{shard_path.name}:"
                f"{','.join(missing_columns)}"
            )
        for row_group in range(parquet.metadata.num_row_groups):
            row_in_group = 0
            try:
                batches = parquet.iter_batches(batch_size=2048, row_groups=[row_group])
                for batch in batches:
                    for row in batch.to_pylist():
                        raw_id = row.get("ID")
                        if (
                            not isinstance(raw_id, int)
                            or isinstance(raw_id, bool)
                            or raw_id < 0
                        ):
                            raise ValueError
                        corpus_id = raw_id
                        if corpus_id in all_seen_ids:
                            raise SourceManifestBridgeError(
                                f"metasyn_corpus_id_duplicate:{corpus_id}"
                            )
                        all_seen_ids.add(corpus_id)
                        selected = corpus_ids is None or corpus_id in corpus_ids
                        if selected:
                            selected_seen_ids.add(corpus_id)
                            warnings: list[str] = []
                            doc_id = f"metasyn-corpus:{corpus_id}"
                            paper_id = doc_id
                            publication_id = f"metasyn-publication:{corpus_id}"
                            publication = PublicationIdentity(
                                publication_id=publication_id,
                                paper_id=paper_id,
                                doc_id=doc_id,
                                doi=_optional_doi(row.get("doi"), warnings),
                                pmid=_optional_pmid(row.get("pmid"), warnings),
                                title=_optional_title(row.get("title"), warnings),
                                publication_year=_optional_year(row.get("year"), warnings),
                            )
                            content_scope = _metasyn_content_scope(row)
                            source_available = (
                                content_scope is not SourceContentScope.UNAVAILABLE
                            )
                            locator = (
                                f"parquet:{shard_relative}#row_group={row_group}"
                                f"&row_in_group={row_in_group}&index_base=0"
                                f"&ID={corpus_id}"
                            )
                            source_document = (
                                SourceDocumentArtifact(
                                    artifact_path=shard_relative,
                                    sha256=shard_contract.sha256,
                                    media_type="application/vnd.apache.parquet",
                                    source_locator=locator,
                                )
                                if source_available
                                else None
                            )
                            if not source_available:
                                warnings.append("source_text_absent")
                            included = source_available
                            source_payload = {
                                "title": row.get("title"),
                                "abstract": row.get("abstract"),
                                "sections": row.get("sections"),
                            }
                            if included:
                                assert source_document is not None
                                native_records.append(
                                    NativeSourceRecord(
                                        doc_id=doc_id,
                                        publication=publication,
                                        source_document=source_document,
                                    )
                                )
                            ledger_payload = {
                                "record_version": "diagnostic-source-record-v1",
                                "corpus_kind": SourceCorpusKind.METASYN,
                                "doc_id": doc_id,
                                "publication_id": publication_id,
                                "paper_id": paper_id,
                                "metadata_artifact_path": shard_relative,
                                "metadata_artifact_sha256": shard_contract.sha256,
                                "metadata_locator": locator,
                                "metadata_row_sha256": hash_canonical(row),
                                "source_available": source_available,
                                "source_document": source_document,
                                "source_payload_sha256": (
                                    hash_canonical(source_payload)
                                    if source_available
                                    else None
                                ),
                                "content_scope": content_scope,
                                "included_in_native_manifest": included,
                                "manifest_exclusion_reason": (
                                    None if included else "source_text_absent"
                                ),
                                "extraction_attempted": False,
                                "estimability_status": "not_assessed_source_only",
                                "access_state": access,
                                "warnings": sorted(set(warnings)),
                            }
                            ledger_records.append(_freeze_record(ledger_payload))
                        row_in_group += 1
            except SourceManifestBridgeError:
                raise
            except (TypeError, ValueError, OverflowError) as exc:
                raise SourceManifestBridgeError(
                    f"metasyn_corpus_row_invalid:{shard_path.name}:"
                    f"row_group={row_group}:row={row_in_group}"
                ) from exc

    if len(all_seen_ids) != corpus_manifest.total_rows:
        raise SourceManifestBridgeError(
            "metasyn_unique_id_count_mismatch:"
            f"expected={corpus_manifest.total_rows}:observed={len(all_seen_ids)}"
        )
    if corpus_ids is not None and selected_seen_ids != corpus_ids:
        missing = sorted(corpus_ids - selected_seen_ids)
        raise SourceManifestBridgeError(f"metasyn_requested_corpus_ids_missing:{missing[:20]}")
    native_records.sort(key=lambda record: record.doc_id)
    if not native_records:
        raise SourceManifestBridgeError("metasyn_native_source_manifest_empty")
    manifest = NativeSourceManifest(
        question_id=question_id,
        records=native_records,
    )

    artifacts = [
        SourceArtifactBinding(
            role="corpus_manifest",
            artifact_path=manifest_relative,
            sha256=sha256_file(manifest_file),
            media_type="application/json",
            rows=None,
        ),
        SourceArtifactBinding(
            role="license_notice",
            artifact_path=license_relative,
            sha256=sha256_file(license_file),
            media_type="text/plain",
            rows=None,
        ),
    ]
    artifacts.extend(
        SourceArtifactBinding(
            role="source_shard",
            artifact_path=path.resolve().relative_to(
                repository_root.resolve()
            ).as_posix(),
            sha256=shard_by_name[path.name].sha256,
            media_type="application/vnd.apache.parquet",
            rows=shard_by_name[path.name].rows,
        )
        for path in shard_paths
    )
    ledger = _freeze_ledger(
        corpus_kind=SourceCorpusKind.METASYN,
        question_id=question_id,
        dataset_version=(
            f"{corpus_manifest.source_repository}@{corpus_manifest.source_revision}"
        ),
        source_revision=corpus_manifest.source_revision,
        license_status=corpus_manifest.license_notice.status,
        selection_scope=(
            "all_corpus_rows" if corpus_ids is None else "explicit_corpus_id_subset"
        ),
        manifest=manifest,
        artifacts=artifacts,
        records=ledger_records,
    )
    return manifest, ledger


__all__ = [
    "DiagnosticAccessState",
    "DiagnosticSourceLedger",
    "DiagnosticSourceRecord",
    "SourceArtifactBinding",
    "SourceContentScope",
    "SourceCorpusKind",
    "SourceManifestBridgeError",
    "build_antiox_native_source_bridge",
    "build_metasyn_native_source_bridge",
]
