"""Replay a frozen acquisition into the native verifier without inventing evidence.

The acquisition boundary is intentionally local and deterministic.  It replays exact
query memberships from a frozen harvester corpus, applies the existing identity and
article-type screen, and joins every included paper to one terminal native extraction.
It does not claim that the frozen search has perfect recall or that deterministic
article-type screening implements free-text PI/ECO eligibility reasoning.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.evidence_graph import AdapterIssueSeverity, PublicationIdentity
from literature_multiverse.harvester.archive import ImmutableArchive
from literature_multiverse.harvester.pipeline import HarvestQuery, LiteratureHarvester
from literature_multiverse.harvester.sources import FrozenCorpusSource, FrozenFullTextSource
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    NativeSourceRecord,
)
from literature_multiverse.native_grounding import (
    NativeGroundingReceipt,
    TypedEvidenceGroundingPackage,
    freeze_grounding_checked_publication_fragment,
    freeze_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.screen import ScreenResult, screen_candidates
from literature_multiverse.typed_extraction import (
    SourceDocumentArtifact,
    assemble_typed_evidence_corpus,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    CorpusAdapterIssue,
    CorpusLoadResult,
    load_corpus,
)


class AcquisitionContractError(ValueError):
    """A frozen acquisition cannot be replayed into a complete native corpus."""


class FrozenArtifactV1(ContractModel):
    path: Annotated[str, Field(min_length=1)]
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or value != path.as_posix()
        ):
            raise ValueError("acquisition_artifact_path_not_canonical_relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("acquisition_artifact_sha256_invalid")
        return value


class FrozenAcquisitionQueryV1(ContractModel):
    family: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    query: Annotated[str, Field(min_length=1)]


class FrozenNativeExtractionRecordV1(ContractModel):
    doc_id: Annotated[str, Field(min_length=1)]
    extraction: NativePublicationExtraction


class FrozenNativeExtractionLedgerV1(ContractModel):
    ledger_version: Literal["frozen-native-extraction-ledger-v1"] = (
        "frozen-native-extraction-ledger-v1"
    )
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    records: Annotated[list[FrozenNativeExtractionRecordV1], Field(min_length=1)]
    ledger_sha256: str

    @field_validator("ledger_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("native_extraction_ledger_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_ledger(self) -> FrozenNativeExtractionLedgerV1:
        doc_ids = [record.doc_id for record in self.records]
        if doc_ids != sorted(set(doc_ids)):
            raise ValueError("native_extraction_ledger_doc_ids_not_sorted_unique")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if hash_canonical(payload) != self.ledger_sha256:
            raise ValueError("native_extraction_ledger_hash_mismatch")
        return self


class FrozenNativeInputV1(ContractModel):
    mode: Literal["frozen_extraction_ledger", "typed_grounding_package"]
    artifact: FrozenArtifactV1


class ProtocolScreenDecisionV1(ContractModel):
    paper_id: Annotated[str, Field(min_length=1)]
    doc_id: Annotated[str, Field(min_length=1)]
    status: Literal["included", "excluded"]
    reason: Annotated[str, Field(min_length=1)]


class ProtocolScreeningReceiptV1(ContractModel):
    """Complete declared PI/ECO decisions over the deterministic identity ledger.

    Version one binds the decision set but has no external trust root for reviewer
    identity or independent adjudication artifacts.  It is therefore never, by
    itself, production authority.
    """

    receipt_version: Literal["protocol-screening-receipt-v1"] = "protocol-screening-receipt-v1"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    protocol_sha256: str
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    provenance: Literal["blinded_human", "benchmark_adjudication", "offline_fixture"]
    adjudicator_count: Annotated[int, Field(ge=1)]
    decisions: Annotated[list[ProtocolScreenDecisionV1], Field(min_length=1)]
    receipt_sha256: str

    @field_validator("protocol_sha256", "receipt_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("protocol_screening_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> ProtocolScreeningReceiptV1:
        keys = [(item.paper_id, item.doc_id) for item in self.decisions]
        if keys != sorted(set(keys)):
            raise ValueError("protocol_screening_decisions_not_sorted_unique")
        if len({item.paper_id for item in self.decisions}) != len(self.decisions):
            raise ValueError("protocol_screening_paper_id_duplicate")
        if len({item.doc_id for item in self.decisions}) != len(self.decisions):
            raise ValueError("protocol_screening_doc_id_duplicate")
        if self.provenance == "blinded_human" and self.adjudicator_count < 2:
            raise ValueError("protocol_screening_blinded_human_requires_two_adjudicators")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("protocol_screening_receipt_hash_mismatch")
        return self

    @property
    def production_authority(self) -> bool:
        # ``provenance`` and ``adjudicator_count`` are self-asserted fields.  Treating
        # them as authority would allow a caller to forge a two-reviewer screen.  A
        # later contract may bind a replayable signed/adjudicated package, but this
        # receipt deliberately cannot remove the release blocker.
        return False


class _FrozenAcquisitionManifestSpecV1(ContractModel):
    manifest_version: Literal["frozen-acquisition-manifest-v1"] = "frozen-acquisition-manifest-v1"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    frozen_corpus: FrozenArtifactV1
    queries: Annotated[list[FrozenAcquisitionQueryV1], Field(min_length=1)]
    per_query_limit: Annotated[int, Field(ge=1, le=1000)]
    page_size: Annotated[int, Field(ge=1, le=200)] = 100
    retrieved_at: datetime
    allowed_article_types: Annotated[list[str], Field(min_length=1)]
    expected_retrieved_doc_ids: Annotated[list[str], Field(min_length=1)]
    expected_included_paper_ids: Annotated[list[str], Field(min_length=1)]
    expected_excluded_paper_ids: list[str] = Field(default_factory=list)
    protocol_screening: FrozenArtifactV1 | None = None
    native_input: FrozenNativeInputV1

    @field_validator("allowed_article_types")
    @classmethod
    def validate_article_types(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("acquisition_article_types_not_sorted_unique")
        return value

    @field_validator(
        "expected_retrieved_doc_ids",
        "expected_included_paper_ids",
        "expected_excluded_paper_ids",
    )
    @classmethod
    def validate_membership(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("acquisition_expected_membership_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_spec(self) -> _FrozenAcquisitionManifestSpecV1:
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("acquisition_retrieved_at_requires_timezone")
        query_keys = [(item.family, item.query) for item in self.queries]
        if query_keys != sorted(set(query_keys)):
            raise ValueError("acquisition_queries_not_sorted_unique")
        if set(self.expected_included_paper_ids) & set(self.expected_excluded_paper_ids):
            raise ValueError("acquisition_expected_screen_membership_overlaps")
        return self


class FrozenAcquisitionManifestV1(_FrozenAcquisitionManifestSpecV1):
    """Exact inputs and expected membership for one acquisition replay."""

    manifest_sha256: str

    @field_validator("manifest_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("acquisition_manifest_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> FrozenAcquisitionManifestV1:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if hash_canonical(payload) != self.manifest_sha256:
            raise ValueError("acquisition_manifest_hash_mismatch")
        return self


class AcquisitionReplayReceiptV1(ContractModel):
    receipt_version: Literal["acquisition-replay-receipt-v1"] = "acquisition-replay-receipt-v1"
    acquisition_manifest_sha256: str
    claim_protocol_sha256: str
    frozen_corpus_sha256: str
    occurrence_membership_sha256: str
    screening_membership_sha256: str
    native_source_manifest_sha256: str
    typed_grounding_package_sha256: str
    native_mode: Literal["frozen_extraction_ledger", "typed_grounding_package"]
    protocol_screening_receipt_sha256: str | None = None
    screening_authority: Literal[
        "deterministic_article_type_only",
        "blinded_human",
        "benchmark_adjudication",
        "offline_fixture",
    ]
    archive_entries: list[dict[str, Any]]
    retrieved_doc_ids: list[str]
    included_paper_ids: list[str]
    excluded_paper_ids: list[str]
    counts: dict[str, int]
    limitations: list[str]
    receipt_sha256: str

    @field_validator(
        "acquisition_manifest_sha256",
        "claim_protocol_sha256",
        "frozen_corpus_sha256",
        "occurrence_membership_sha256",
        "screening_membership_sha256",
        "native_source_manifest_sha256",
        "typed_grounding_package_sha256",
        "protocol_screening_receipt_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and SHA256_RE.fullmatch(value) is None:
            raise ValueError("acquisition_replay_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_screening_authority(self) -> AcquisitionReplayReceiptV1:
        has_receipt = self.protocol_screening_receipt_sha256 is not None
        if has_receipt != (self.screening_authority != "deterministic_article_type_only"):
            raise ValueError("acquisition_screening_authority_receipt_mismatch")
        return self

    @model_validator(mode="after")
    def validate_receipt(self) -> AcquisitionReplayReceiptV1:
        for values in (
            self.retrieved_doc_ids,
            self.included_paper_ids,
            self.excluded_paper_ids,
            self.limitations,
        ):
            if values != sorted(set(values)):
                raise ValueError("acquisition_replay_values_not_sorted_unique")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("acquisition_replay_receipt_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class AcquiredCorpusReplay:
    corpus: CorpusLoadResult
    receipt: AcquisitionReplayReceiptV1
    package_path: Path


def freeze_native_extraction_ledger(
    *, question_id: str, records: list[FrozenNativeExtractionRecordV1]
) -> FrozenNativeExtractionLedgerV1:
    ordered = sorted(records, key=lambda item: item.doc_id)
    payload = {
        "ledger_version": "frozen-native-extraction-ledger-v1",
        "question_id": question_id,
        "records": ordered,
    }
    return FrozenNativeExtractionLedgerV1.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def freeze_protocol_screening_receipt(
    *,
    question_id: str,
    protocol_sha256: str,
    corpus_cutoff: str,
    provenance: Literal["blinded_human", "benchmark_adjudication", "offline_fixture"],
    adjudicator_count: int,
    decisions: list[ProtocolScreenDecisionV1],
) -> ProtocolScreeningReceiptV1:
    payload = {
        "receipt_version": "protocol-screening-receipt-v1",
        "question_id": question_id,
        "protocol_sha256": protocol_sha256,
        "corpus_cutoff": corpus_cutoff,
        "provenance": provenance,
        "adjudicator_count": adjudicator_count,
        "decisions": sorted(decisions, key=lambda item: (item.paper_id, item.doc_id)),
    }
    return ProtocolScreeningReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def freeze_acquisition_manifest(
    payload: dict[str, Any],
) -> FrozenAcquisitionManifestV1:
    """Freeze a validated manifest spec that omits only ``manifest_sha256``."""

    if "manifest_sha256" in payload:
        raise AcquisitionContractError("acquisition_manifest_spec_already_sealed")
    parsed = _FrozenAcquisitionManifestSpecV1.model_validate(payload).model_dump(mode="json")
    return FrozenAcquisitionManifestV1.model_validate(
        {**parsed, "manifest_sha256": hash_canonical(parsed)}
    )


def load_acquisition_manifest(path: Path) -> FrozenAcquisitionManifestV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"acquisition_manifest_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionContractError("acquisition_manifest_root_not_object")
    return FrozenAcquisitionManifestV1.model_validate(payload)


def _repository_file(binding: FrozenArtifactV1, *, repository_root: Path) -> Path:
    root = Path(os.path.abspath(repository_root))
    if not root.is_dir() or root.is_symlink():
        raise AcquisitionContractError("acquisition_repository_root_invalid")
    candidate = root
    for part in PurePosixPath(binding.path).parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise AcquisitionContractError(f"acquisition_artifact_missing:{binding.path}") from exc
        if stat.S_ISLNK(mode):
            raise AcquisitionContractError(f"acquisition_artifact_symlink_forbidden:{binding.path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise AcquisitionContractError(f"acquisition_artifact_outside_repository:{binding.path}")
    observed = sha256_file(resolved)
    if observed != binding.sha256:
        raise AcquisitionContractError(f"acquisition_artifact_hash_mismatch:{binding.path}")
    return resolved


def _screen_payload(result: ScreenResult) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": str(row["doc_id"]),
            "paper_id": str(row["paper_id"]),
            "reason": row["screen_reason"],
            "status": str(row["screen_status"]),
        }
        for row in result.papers
    ]


def _read_protocol_screening(path: Path) -> ProtocolScreeningReceiptV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"protocol_screening_unreadable:{path}") from exc
    return ProtocolScreeningReceiptV1.model_validate(payload)


def _apply_protocol_screening(
    *,
    deterministic: ScreenResult,
    receipt: ProtocolScreeningReceiptV1,
    claim_manifest: ClaimManifest,
    corpus_cutoff: str,
) -> ScreenResult:
    if receipt.question_id != claim_manifest.question_id:
        raise AcquisitionContractError("protocol_screening_question_mismatch")
    if receipt.protocol_sha256 != hash_canonical(claim_manifest.protocol):
        raise AcquisitionContractError("protocol_screening_claim_protocol_mismatch")
    if receipt.corpus_cutoff != corpus_cutoff:
        raise AcquisitionContractError("protocol_screening_corpus_cutoff_mismatch")
    rows_by_key = {
        (str(row["paper_id"]), str(row["doc_id"])): dict(row) for row in deterministic.papers
    }
    decisions = {(item.paper_id, item.doc_id): item for item in receipt.decisions}
    if set(rows_by_key) != set(decisions):
        raise AcquisitionContractError(
            "protocol_screening_membership_mismatch:"
            f"deterministic={sorted(rows_by_key)}:receipt={sorted(decisions)}"
        )
    papers: list[dict[str, Any]] = []
    for key in sorted(rows_by_key):
        row = rows_by_key[key]
        decision = decisions[key]
        if row["screen_status"] == "excluded" and decision.status == "included":
            raise AcquisitionContractError(
                f"protocol_screening_cannot_override_deterministic_exclusion:{key[0]}"
            )
        row["screen_status"] = decision.status
        row["screen_reason"] = (
            None if decision.status == "included" else f"protocol_excluded:{decision.reason}"
        )
        papers.append(row)
    included = sorted(str(row["paper_id"]) for row in papers if row["screen_status"] == "included")
    excluded = sorted(str(row["paper_id"]) for row in papers if row["screen_status"] == "excluded")
    return ScreenResult(
        papers=tuple(papers),
        include_paper_ids=tuple(included),
        exclude_paper_ids=tuple(excluded),
        dedupe_log=deterministic.dedupe_log,
    )


def _full_text_entry(*, doc_id: str, occurrences: list[Any]) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        if occurrence.doc_id != doc_id:
            continue
        harvester = occurrence.raw_metadata.get("_literature_multiverse_harvester")
        full_text = harvester.get("full_text") if isinstance(harvester, dict) else None
        content = full_text.get("content") if isinstance(full_text, dict) else None
        if isinstance(content, dict):
            observed[hash_canonical(content)] = content
    if not observed:
        raise AcquisitionContractError(f"acquisition_included_full_text_missing:{doc_id}")
    if len(observed) != 1:
        raise AcquisitionContractError(f"acquisition_full_text_identity_conflict:{doc_id}")
    return next(iter(observed.values()))


def _publication(row: dict[str, Any]) -> PublicationIdentity:
    doc_id = str(row["doc_id"])
    return PublicationIdentity(
        publication_id=f"harvest-publication:{doc_id}",
        paper_id=str(row["paper_id"]),
        doc_id=doc_id,
        doi=row.get("doi"),
        pmid=row.get("pmid"),
        title=row.get("title"),
        publication_year=row.get("pub_year"),
    )


def _source_manifest_from_harvest(
    *,
    question_id: str,
    screen: ScreenResult,
    occurrences: list[Any],
) -> NativeSourceManifest:
    records: list[NativeSourceRecord] = []
    for raw in screen.papers:
        row = dict(raw)
        if row["screen_status"] != "included":
            continue
        doc_id = str(row["doc_id"])
        content = _full_text_entry(doc_id=doc_id, occurrences=occurrences)
        blob_path = content.get("blob_path")
        digest = content.get("sha256")
        media_type = content.get("media_type")
        if not all(isinstance(value, str) and value for value in (blob_path, digest, media_type)):
            raise AcquisitionContractError(
                f"acquisition_full_text_archive_metadata_invalid:{doc_id}"
            )
        if media_type.partition(";")[0].strip().casefold() not in {
            "application/xml",
            "application/xhtml+xml",
            "text/xml",
            "text/html",
            "text/plain",
        }:
            raise AcquisitionContractError(
                f"acquisition_full_text_media_type_not_natively_groundable:{doc_id}:{media_type}"
            )
        source_document = SourceDocumentArtifact(
            artifact_path=blob_path,
            sha256=digest,
            media_type=media_type,
            source_locator=f"harvest-sha256:{digest}",
        )
        records.append(
            NativeSourceRecord(
                doc_id=doc_id,
                publication=_publication(row),
                source_document=source_document,
            )
        )
    return NativeSourceManifest(question_id=question_id, records=records)


def _read_native_ledger(path: Path) -> FrozenNativeExtractionLedgerV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionContractError(f"native_extraction_ledger_unreadable:{path}") from exc
    return FrozenNativeExtractionLedgerV1.model_validate(payload)


def _build_offline_native_package(
    *,
    manifest: FrozenAcquisitionManifestV1,
    source_manifest: NativeSourceManifest,
    native_ledger_path: Path,
    repository_root: Path,
    pipeline_sha256: str,
    output_dir: Path,
    force: bool,
) -> tuple[TypedEvidenceGroundingPackage, Path]:
    ledger = _read_native_ledger(native_ledger_path)
    if ledger.question_id != manifest.question_id:
        raise AcquisitionContractError("native_extraction_ledger_question_mismatch")
    source_by_doc = {record.doc_id: record for record in source_manifest.records}
    extraction_by_doc = {record.doc_id: record.extraction for record in ledger.records}
    if set(source_by_doc) != set(extraction_by_doc):
        raise AcquisitionContractError(
            "native_extraction_ledger_membership_mismatch:"
            f"source={sorted(source_by_doc)}:extractions={sorted(extraction_by_doc)}"
        )
    fragments = []
    receipts: list[NativeGroundingReceipt] = []
    for doc_id in sorted(source_by_doc):
        source = source_by_doc[doc_id]
        extraction = extraction_by_doc[doc_id]
        grounding = verify_native_publication_grounding(
            repository_root=repository_root,
            source_document=source.source_document,
            extraction=extraction,
        )
        receipts.append(grounding)
        fragments.append(
            freeze_grounding_checked_publication_fragment(
                extraction=extraction,
                grounding_receipt=grounding,
                question_id=manifest.question_id,
                publication=source.publication,
                pipeline_fingerprint_sha256=pipeline_sha256,
                source_document=source.source_document,
            )
        )
    corpus = assemble_typed_evidence_corpus(fragments)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=receipts,
        source_manifest=source_manifest,
        corpus_cutoff=manifest.corpus_cutoff,
    )
    native_dir = output_dir / "acquisition" / "native"
    outputs = {
        native_dir / "native-source-manifest.json": source_manifest,
        native_dir / "typed-evidence-corpus.json": corpus,
        native_dir / "typed-evidence-grounding-package.json": package,
    }
    if not force:
        existing = sorted(path.as_posix() for path in outputs if path.exists())
        if existing:
            raise FileExistsError(f"acquisition_native_outputs_exist:{existing}")
    for path, value in outputs.items():
        atomic_write_json(path, value, force=force)
    return package, native_dir / "typed-evidence-grounding-package.json"


def _package_membership(*, package: TypedEvidenceGroundingPackage, screen: ScreenResult) -> None:
    source_manifest = package.source_manifest
    if source_manifest is None:
        raise AcquisitionContractError("acquisition_native_package_source_manifest_missing")
    expected = sorted(
        (str(row["doc_id"]), str(row["paper_id"]))
        for row in screen.papers
        if row["screen_status"] == "included"
    )
    observed = sorted(
        (record.doc_id, record.publication.paper_id) for record in source_manifest.records
    )
    if observed != expected:
        raise AcquisitionContractError(
            f"acquisition_native_package_screen_membership_mismatch:"
            f"expected={expected}:observed={observed}"
        )


def replay_frozen_acquisition(
    *,
    manifest: FrozenAcquisitionManifestV1,
    claim_manifest: ClaimManifest,
    repository_root: Path,
    pipeline_sha256: str,
    output_dir: Path,
    force: bool = False,
) -> AcquiredCorpusReplay:
    """Replay acquisition and return the exact standard verifier corpus object."""

    if manifest.question_id != claim_manifest.question_id:
        raise AcquisitionContractError("acquisition_claim_question_mismatch")
    if manifest.corpus_cutoff != claim_manifest.protocol.corpus_cutoff:
        raise AcquisitionContractError("acquisition_claim_corpus_cutoff_mismatch")
    root = repository_root.resolve(strict=True)
    destination = output_dir.resolve()
    if not destination.is_relative_to(root):
        raise AcquisitionContractError("acquisition_output_dir_outside_repository")
    corpus_path = _repository_file(manifest.frozen_corpus, repository_root=root)
    native_path = _repository_file(manifest.native_input.artifact, repository_root=root)
    protocol_screening_path = (
        _repository_file(manifest.protocol_screening, repository_root=root)
        if manifest.protocol_screening is not None
        else None
    )

    search_source = FrozenCorpusSource(
        corpus_path,
        expected_sha256=manifest.frozen_corpus.sha256,
        retrieved_at=manifest.retrieved_at,
    )
    for query in manifest.queries:
        exact_ids = search_source.exact_search_result_ids(query.query)
        if exact_ids is None:
            raise AcquisitionContractError(
                f"acquisition_query_missing_exact_frozen_membership:{query.query}"
            )
        if len(exact_ids) > manifest.per_query_limit:
            raise AcquisitionContractError(
                f"acquisition_query_would_truncate:{query.query}:"
                f"results={len(exact_ids)}:limit={manifest.per_query_limit}"
            )
    archive = ImmutableArchive(
        output_dir / "acquisition" / "harvest",
        path_base=root,
    )
    harvester = LiteratureHarvester(
        search_source,
        archive,
        full_text_source=FrozenFullTextSource(corpus_path, retrieved_at=manifest.retrieved_at),
        page_size=manifest.page_size,
    )
    harvest = harvester.run(
        [HarvestQuery(family=item.family, query=item.query) for item in manifest.queries],
        per_query_limit=manifest.per_query_limit,
    )
    occurrences = list(harvest.occurrences)
    retrieved_doc_ids = sorted({item.doc_id for item in occurrences})
    if retrieved_doc_ids != manifest.expected_retrieved_doc_ids:
        raise AcquisitionContractError(
            f"acquisition_retrieved_membership_mismatch:"
            f"expected={manifest.expected_retrieved_doc_ids}:observed={retrieved_doc_ids}"
        )
    deterministic_screen = screen_candidates(
        occurrences,
        allowed_article_types=manifest.allowed_article_types,
        config_sha256=hash_canonical(claim_manifest.protocol),
        created_at=manifest.retrieved_at,
    )
    if any(item.event == "human_review_required" for item in deterministic_screen.dedupe_log):
        raise AcquisitionContractError("acquisition_unresolved_fuzzy_identity")
    screening_receipt = (
        _read_protocol_screening(protocol_screening_path)
        if protocol_screening_path is not None
        else None
    )
    screen = (
        _apply_protocol_screening(
            deterministic=deterministic_screen,
            receipt=screening_receipt,
            claim_manifest=claim_manifest,
            corpus_cutoff=manifest.corpus_cutoff,
        )
        if screening_receipt is not None
        else deterministic_screen
    )
    if list(screen.include_paper_ids) != manifest.expected_included_paper_ids:
        raise AcquisitionContractError("acquisition_included_membership_mismatch")
    if list(screen.exclude_paper_ids) != manifest.expected_excluded_paper_ids:
        raise AcquisitionContractError("acquisition_excluded_membership_mismatch")

    for row in screen.papers:
        if row["screen_status"] == "included":
            _full_text_entry(doc_id=str(row["doc_id"]), occurrences=occurrences)

    if manifest.native_input.mode == "frozen_extraction_ledger":
        source_manifest = _source_manifest_from_harvest(
            question_id=manifest.question_id,
            screen=screen,
            occurrences=occurrences,
        )
        package, package_path = _build_offline_native_package(
            manifest=manifest,
            source_manifest=source_manifest,
            native_ledger_path=native_path,
            repository_root=root,
            pipeline_sha256=pipeline_sha256,
            output_dir=output_dir,
            force=force,
        )
    else:
        try:
            package_payload = json.loads(native_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcquisitionContractError(
                f"acquisition_native_package_unreadable:{native_path}"
            ) from exc
        package = TypedEvidenceGroundingPackage.model_validate(package_payload)
        package_path = native_path
        source_manifest = package.source_manifest
        if source_manifest is None:
            raise AcquisitionContractError("acquisition_native_package_source_manifest_missing")
    if package.corpus.question_id != manifest.question_id:
        raise AcquisitionContractError("acquisition_native_package_question_mismatch")
    if package.corpus_cutoff != manifest.corpus_cutoff:
        raise AcquisitionContractError("acquisition_native_package_cutoff_mismatch")
    _package_membership(package=package, screen=screen)

    corpus = load_corpus(
        package_path,
        legacy_settings=claim_manifest.legacy_adapter,
        repository_root=root,
    )
    screening_authority = (
        "deterministic_article_type_only"
        if screening_receipt is None
        else screening_receipt.provenance
    )
    screening_limitation: str | None = None
    issue: CorpusAdapterIssue | None = None
    if screening_receipt is None:
        screening_limitation = (
            "protocol_eligibility_screening_unverified:the deterministic screen enforces "
            "identity and article-type rules but does not adjudicate free-text inclusion "
            "and exclusion criteria"
        )
        issue = CorpusAdapterIssue(
            severity=AdapterIssueSeverity.BLOCKING,
            code="protocol_eligibility_screening_unverified",
            detail=screening_limitation.partition(":")[2],
        )
    elif not screening_receipt.production_authority:
        screening_limitation = (
            "missing_verified_screening_adjudication_package:the supplied complete "
            f"screening receipt declares {screening_receipt.provenance} provenance, but "
            "does not bind trusted reviewer identities, independent raw decisions, or "
            "an externally replayable adjudication attestation"
        )
        issue = CorpusAdapterIssue(
            severity=AdapterIssueSeverity.BLOCKING,
            code="missing_verified_screening_adjudication_package",
            detail=screening_limitation.partition(":")[2],
        )
    screen_payload = _screen_payload(screen)
    receipt_payload = {
        "receipt_version": "acquisition-replay-receipt-v1",
        "acquisition_manifest_sha256": manifest.manifest_sha256,
        "claim_protocol_sha256": hash_canonical(claim_manifest.protocol),
        "frozen_corpus_sha256": manifest.frozen_corpus.sha256,
        "occurrence_membership_sha256": hash_canonical([item.model_dump() for item in occurrences]),
        "screening_membership_sha256": hash_canonical(screen_payload),
        "native_source_manifest_sha256": hash_canonical(source_manifest),
        "typed_grounding_package_sha256": package.package_sha256,
        "native_mode": manifest.native_input.mode,
        "protocol_screening_receipt_sha256": (
            screening_receipt.receipt_sha256 if screening_receipt is not None else None
        ),
        "screening_authority": screening_authority,
        "archive_entries": [
            entry.model_dump()
            for entry in sorted(harvest.archive_entries, key=lambda item: item.receipt_path)
        ],
        "retrieved_doc_ids": retrieved_doc_ids,
        "included_paper_ids": list(screen.include_paper_ids),
        "excluded_paper_ids": list(screen.exclude_paper_ids),
        "counts": {
            "archive_entries": len(harvest.archive_entries),
            "excluded_papers": len(screen.exclude_paper_ids),
            "included_papers": len(screen.include_paper_ids),
            "native_fragments": len(package.corpus.fragments),
            "native_non_estimable": len(package.corpus.non_estimable_publication_ids),
            "queries": len(manifest.queries),
            "retrieved_documents": len(retrieved_doc_ids),
            "search_pages": harvest.search_pages,
        },
        "limitations": sorted(
            {
                *([screening_limitation] if screening_limitation is not None else []),
                "retrieval_recall_not_established_beyond_exact_frozen_query_membership",
            }
        ),
    }
    receipt = AcquisitionReplayReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    metadata = {
        **corpus.metadata,
        "acquisition_replay_receipt": receipt.model_dump(mode="json"),
        "acquisition_replay_receipt_sha256": receipt.receipt_sha256,
        "acquisition_screening": screen_payload,
        "acquisition_screening_membership_sha256": receipt.screening_membership_sha256,
        "protocol_screening_receipt": (
            screening_receipt.model_dump(mode="json") if screening_receipt is not None else None
        ),
    }
    combined_source_sha256 = hash_canonical(
        {
            "acquisition_replay_receipt_sha256": receipt.receipt_sha256,
            "native_corpus_source_sha256": corpus.source_sha256,
        }
    )
    acquisition_issues = () if issue is None else (issue,)
    issues_by_key = {
        (item.finding_id or "", item.paper_id or "", item.code): item
        for item in (*corpus.adapter_issues, *acquisition_issues)
    }
    acquired = replace(
        corpus,
        source_label=f"acquisition:{manifest.manifest_sha256}",
        source_sha256=combined_source_sha256,
        adapter_issues=tuple(issues_by_key[key] for key in sorted(issues_by_key)),
        metadata=metadata,
    )
    return AcquiredCorpusReplay(
        corpus=acquired,
        receipt=receipt,
        package_path=package_path,
    )


__all__ = [
    "AcquiredCorpusReplay",
    "AcquisitionContractError",
    "AcquisitionReplayReceiptV1",
    "FrozenAcquisitionManifestV1",
    "FrozenAcquisitionQueryV1",
    "FrozenArtifactV1",
    "FrozenNativeExtractionLedgerV1",
    "FrozenNativeExtractionRecordV1",
    "FrozenNativeInputV1",
    "ProtocolScreenDecisionV1",
    "ProtocolScreeningReceiptV1",
    "freeze_acquisition_manifest",
    "freeze_native_extraction_ledger",
    "freeze_protocol_screening_receipt",
    "load_acquisition_manifest",
    "replay_frozen_acquisition",
]
