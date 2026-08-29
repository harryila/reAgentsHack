"""Fail-closed grounding of native extractions against immutable local sources.

This module deliberately supports only the two source locators emitted by the native
source-manifest bridge.  It does not search for relocated files or rows.  A source is
usable only when its repository-relative path, exact file bytes, physical locator, and
row/document identity all agree with the frozen :class:`SourceDocumentArtifact`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl

import pyarrow.parquet as pq
from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationError,
    NativeCohortReconciliationReceipt,
    ReviewerCohortReconciliationArtifact,
    reconcile_native_cohorts,
    reverify_native_cohort_reconciliation,
)
from literature_multiverse.config import QuestionConfig, config_sha256
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.grounding import (
    GroundingContractError,
    ground_evidence,
    normalize_evidence_text,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel, GroundingStatus
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    freeze_native_publication_extraction,
    native_extraction_prompt_replacements,
    native_publication_extraction_json_schema,
)
from literature_multiverse.prompting import render_prompt_text
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    SourceDocumentArtifact,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

_JSON_LOCATOR = re.compile(r"^json:(?P<path>[^#]+)#/(?P<pointer>[^/]*)$")
_PARQUET_LOCATOR = re.compile(r"^parquet:(?P<path>[^#]+)#(?P<query>.+)$")
_LINE_ID = re.compile(r"^L(?P<number>[1-9][0-9]*)$")
_PARQUET_KEYS = frozenset({"row_group", "row_in_group", "index_base", "ID"})
_FORBIDDEN_SECTIONS = frozenset(
    {"abstract", "discussion", "conclusion", "conclusions", "references", "title", "unknown"}
)


class NativeGroundingError(ValueError):
    """A frozen native source cannot be resolved exactly."""


class NativeExtractionArtifactDigest(ContractModel):
    """One private input artifact whose bytes/content were opened during extraction."""

    artifact_id: Annotated[str, Field(pattern=r"^[a-z0-9][-a-z0-9_.:]*$")]
    role: Literal[
        "source_manifest_input",
        "map_output",
        "provider_artifact",
        "provider_execution_receipt",
        "prediction_input_bundle",
        "prediction_ledger",
        "generation_receipt",
    ]
    sha256: str
    hash_basis: Literal["raw_bytes", "canonical_json"]
    byte_count: Annotated[int, Field(ge=0)] | None = None
    execution_ids: list[str] = Field(default_factory=list)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_extraction_artifact_sha256_invalid")
        return value

    @field_validator("execution_ids")
    @classmethod
    def validate_execution_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("native_extraction_artifact_execution_ids_invalid")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> NativeExtractionArtifactDigest:
        if self.hash_basis == "raw_bytes" and self.byte_count is None:
            raise ValueError("raw_native_extraction_artifact_requires_byte_count")
        if self.hash_basis == "canonical_json" and self.byte_count is not None:
            raise ValueError("canonical_native_extraction_artifact_forbids_byte_count")
        if self.role == "map_output" and not self.execution_ids:
            raise ValueError("native_map_artifact_requires_execution_id")
        return self


class NativeRenderedPromptArtifact(ContractModel):
    """Exact private prompt text and the repository template identity that produced it."""

    # Prompt bytes are scientific execution inputs.  The shared contract base strips
    # surrounding whitespace from strings, which would silently change those bytes
    # after ``rendered_prompt_sha256`` was computed.
    model_config = ConfigDict(str_strip_whitespace=False)

    prompt_id: Annotated[str, Field(pattern=r"^[a-z0-9][-a-z0-9_.:]*$")]
    renderer_id: Literal[
        "repository-native-extraction-v1",
        "native-ollama-row-projection-v1",
    ]
    prompt_version: Annotated[str, Field(min_length=1)]
    template_path: Annotated[str, Field(min_length=1)]
    template_sha256: str
    rendered_prompt: Annotated[str, Field(min_length=1)]
    rendered_prompt_sha256: str

    @field_validator("template_path")
    @classmethod
    def validate_template_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("native_prompt_template_path_not_repository_relative")
        return value

    @field_validator("template_sha256", "rendered_prompt_sha256")
    @classmethod
    def validate_prompt_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_extraction_prompt_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_prompt(self) -> NativeRenderedPromptArtifact:
        observed = hashlib.sha256(self.rendered_prompt.encode("utf-8")).hexdigest()
        if observed != self.rendered_prompt_sha256:
            raise ValueError("native_extraction_rendered_prompt_hash_mismatch")
        return self


class NativeEvaluationSchemaArtifact(ContractModel):
    """Exact JSON schema applied at generation or authoritative postvalidation."""

    schema_id: Annotated[str, Field(pattern=r"^[a-z0-9][-a-z0-9_.:]*$")]
    role: Literal["generation_constraint", "official_postvalidation"]
    schema_payload: dict[str, Any]
    schema_sha256: str

    @field_validator("schema_sha256")
    @classmethod
    def validate_schema_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_extraction_schema_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_schema(self) -> NativeEvaluationSchemaArtifact:
        if hash_canonical(self.schema_payload) != self.schema_sha256:
            raise ValueError("native_extraction_schema_hash_mismatch")
        return self


class NativeProviderExecutionReceipt(ContractModel):
    """Self-contained provider/model identity plus the exact raw call ledger."""

    receipt_version: Literal["native-provider-execution-receipt-v1"] = (
        "native-provider-execution-receipt-v1"
    )
    execution_id: Annotated[str, Field(min_length=1)]
    execution_mode: Literal["paperclip_archived", "paperclip_live", "ollama_local"]
    provider_id: Annotated[str, Field(min_length=1)]
    model_id: Annotated[str, Field(min_length=1)]
    model_revision: str | None = None
    runtime_id: Annotated[str, Field(min_length=1)]
    runtime_version: Annotated[str, Field(min_length=1)]
    runtime_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    execution_identity_sha256: str
    raw_call_ledger: dict[str, Any] | list[Any]
    raw_call_ledger_sha256: str
    call_count: Annotated[int, Field(ge=1)]
    receipt_sha256: str

    @field_validator(
        "execution_identity_sha256",
        "raw_call_ledger_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_execution_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_provider_execution_sha256_invalid")
        return value

    @field_validator("runtime_metadata")
    @classmethod
    def validate_runtime_metadata(
        cls, value: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        if any(not key.strip() for key in value):
            raise ValueError("native_provider_runtime_metadata_key_empty")
        if any(isinstance(item, float) and not math.isfinite(item) for item in value.values()):
            raise ValueError("native_provider_runtime_metadata_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativeProviderExecutionReceipt:
        identity_payload = {
            "execution_id": self.execution_id,
            "execution_mode": self.execution_mode,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "provider_id": self.provider_id,
            "runtime_id": self.runtime_id,
            "runtime_metadata": self.runtime_metadata,
            "runtime_version": self.runtime_version,
        }
        if hash_canonical(identity_payload) != self.execution_identity_sha256:
            raise ValueError("native_provider_execution_identity_hash_mismatch")
        if hash_canonical(self.raw_call_ledger) != self.raw_call_ledger_sha256:
            raise ValueError("native_provider_raw_call_ledger_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("native_provider_execution_receipt_hash_mismatch")
        return self


class NativeExtractionExecutionContext(ContractModel):
    """Private, exact inputs that produced every v3 publication fragment."""

    context_version: Literal["native-extraction-execution-context-v1"] = (
        "native-extraction-execution-context-v1"
    )
    extraction_mode: Literal[
        "paperclip_archived",
        "paperclip_live",
        "ollama_local",
    ]
    question_config: QuestionConfig
    question_config_sha256: str
    pipeline_fingerprint_sha256: str
    rendered_prompts: Annotated[list[NativeRenderedPromptArtifact], Field(min_length=1)]
    evaluation_schemas: Annotated[
        list[NativeEvaluationSchemaArtifact], Field(min_length=1)
    ]
    provider_execution_receipts: Annotated[
        list[NativeProviderExecutionReceipt], Field(min_length=1)
    ]
    input_artifacts: Annotated[
        list[NativeExtractionArtifactDigest], Field(min_length=1)
    ]
    source_manifest_content_sha256: str
    source_manifest_records: Annotated[int, Field(ge=1)]
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    context_sha256: str

    @field_validator(
        "question_config_sha256",
        "pipeline_fingerprint_sha256",
        "source_manifest_content_sha256",
        "context_sha256",
    )
    @classmethod
    def validate_context_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_extraction_context_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> NativeExtractionExecutionContext:
        if self.question_config.status != "locked":
            raise ValueError("native_extraction_context_requires_locked_question_config")
        if config_sha256(self.question_config) != self.question_config_sha256:
            raise ValueError("native_extraction_question_config_hash_mismatch")
        for values, label, identity in (
            (self.rendered_prompts, "prompts", lambda item: item.prompt_id),
            (self.evaluation_schemas, "schemas", lambda item: item.schema_id),
            (
                self.provider_execution_receipts,
                "execution_receipts",
                lambda item: item.execution_id,
            ),
            (self.input_artifacts, "input_artifacts", lambda item: item.artifact_id),
        ):
            identities = [identity(item) for item in values]
            if identities != sorted(set(identities)):
                raise ValueError(f"native_extraction_context_{label}_not_sorted_unique")
        if {
            receipt.execution_mode for receipt in self.provider_execution_receipts
        } != {self.extraction_mode}:
            raise ValueError("native_extraction_context_execution_mode_mismatch")
        receipt_execution_ids = {
            receipt.execution_id for receipt in self.provider_execution_receipts
        }
        artifact_execution_ids = {
            execution_id
            for artifact in self.input_artifacts
            for execution_id in artifact.execution_ids
        }
        if not artifact_execution_ids <= receipt_execution_ids:
            raise ValueError("native_extraction_context_artifact_execution_id_unbound")
        if not any(
            schema.role == "official_postvalidation"
            for schema in self.evaluation_schemas
        ):
            raise ValueError("native_extraction_context_official_schema_missing")
        if sum(
            artifact.role == "source_manifest_input"
            for artifact in self.input_artifacts
        ) != 1:
            raise ValueError("native_extraction_context_source_manifest_artifact_required")
        map_artifacts = [
            artifact
            for artifact in self.input_artifacts
            if artifact.role == "map_output"
        ]
        if self.extraction_mode.startswith("paperclip_") and not map_artifacts:
            raise ValueError("paperclip_extraction_context_requires_map_artifact")
        if self.extraction_mode.startswith("paperclip_"):
            map_execution_ids = {
                execution_id
                for artifact in map_artifacts
                for execution_id in artifact.execution_ids
            }
            receipt_artifact_execution_ids = {
                execution_id
                for artifact in self.input_artifacts
                if artifact.role == "provider_execution_receipt"
                for execution_id in artifact.execution_ids
            }
            if map_execution_ids != receipt_execution_ids:
                raise ValueError("paperclip_map_execution_receipt_identity_mismatch")
            if receipt_artifact_execution_ids != receipt_execution_ids:
                raise ValueError("paperclip_receipt_artifact_identity_mismatch")
        if self.extraction_mode == "ollama_local" and map_artifacts:
            raise ValueError("ollama_extraction_context_forbids_map_artifact")
        if self.extraction_mode == "ollama_local":
            if len(self.provider_execution_receipts) != 1:
                raise ValueError("ollama_extraction_context_requires_one_execution_receipt")
            prompt_hashes = {
                prompt.rendered_prompt_sha256 for prompt in self.rendered_prompts
            }
            generation_schema_hashes = {
                schema.schema_sha256
                for schema in self.evaluation_schemas
                if schema.role == "generation_constraint"
            }
            ledger_prompt_hashes: set[str] = set()
            ledger_schema_hashes: set[str] = set()
            ledger_calls = 0
            for receipt in self.provider_execution_receipts:
                raw = receipt.raw_call_ledger
                if not isinstance(raw, dict) or raw.get("ledger_version") != (
                    "native-ollama-raw-call-ledger-v1"
                ):
                    raise ValueError("ollama_extraction_context_raw_ledger_invalid")
                calls = raw.get("generation_receipts")
                if not isinstance(calls, list) or len(calls) != receipt.call_count:
                    raise ValueError("ollama_extraction_context_call_count_mismatch")
                ledger_calls += len(calls)
                for call in calls:
                    if not isinstance(call, dict):
                        raise ValueError("ollama_extraction_context_call_receipt_invalid")
                    prompt_sha256 = call.get("rendered_prompt_sha256")
                    schema_sha256 = call.get("generation_schema_sha256")
                    if not isinstance(prompt_sha256, str) or not isinstance(
                        schema_sha256, str
                    ):
                        raise ValueError("ollama_extraction_context_call_identity_missing")
                    ledger_prompt_hashes.add(prompt_sha256)
                    ledger_schema_hashes.add(schema_sha256)
            if ledger_calls != len(self.rendered_prompts):
                raise ValueError("ollama_extraction_context_prompt_call_count_mismatch")
            if ledger_prompt_hashes != prompt_hashes:
                raise ValueError("ollama_extraction_context_prompt_ledger_mismatch")
            if ledger_schema_hashes != generation_schema_hashes:
                raise ValueError("ollama_extraction_context_schema_ledger_mismatch")
            generation_artifacts = [
                artifact
                for artifact in self.input_artifacts
                if artifact.role == "generation_receipt"
            ]
            prediction_ledgers = [
                artifact
                for artifact in self.input_artifacts
                if artifact.role == "prediction_ledger"
            ]
            if len(prediction_ledgers) != 1:
                raise ValueError("ollama_extraction_context_prediction_ledger_required")
            generation_receipt_hashes = {
                str(call.get("receipt_sha256"))
                for receipt in self.provider_execution_receipts
                for call in receipt.raw_call_ledger["generation_receipts"]
            }
            if (
                len(generation_artifacts) != ledger_calls
                or {artifact.sha256 for artifact in generation_artifacts}
                != generation_receipt_hashes
                or {
                    execution_id
                    for artifact in generation_artifacts
                    for execution_id in artifact.execution_ids
                }
                != receipt_execution_ids
            ):
                raise ValueError("ollama_extraction_context_generation_artifact_mismatch")
            prediction_payload = self.provider_execution_receipts[0].raw_call_ledger[
                "prediction_ledger"
            ]
            if (
                not isinstance(prediction_payload, dict)
                or prediction_ledgers[0].sha256
                != prediction_payload.get("prediction_ledger_sha256")
                or set(prediction_ledgers[0].execution_ids) != receipt_execution_ids
            ):
                raise ValueError("ollama_extraction_context_prediction_artifact_mismatch")
        payload = self.model_dump(mode="json", exclude={"context_sha256"})
        if hash_canonical(payload) != self.context_sha256:
            raise ValueError("native_extraction_execution_context_hash_mismatch")
        return self


def freeze_native_provider_execution_receipt(
    *,
    execution_id: str,
    execution_mode: Literal["paperclip_archived", "paperclip_live", "ollama_local"],
    provider_id: str,
    model_id: str,
    runtime_id: str,
    runtime_version: str,
    raw_call_ledger: dict[str, Any] | list[Any],
    call_count: int,
    model_revision: str | None = None,
    runtime_metadata: dict[str, str | int | float | bool | None] | None = None,
) -> NativeProviderExecutionReceipt:
    metadata = dict(sorted((runtime_metadata or {}).items()))
    identity_payload = {
        "execution_id": execution_id,
        "execution_mode": execution_mode,
        "model_id": model_id,
        "model_revision": model_revision,
        "provider_id": provider_id,
        "runtime_id": runtime_id,
        "runtime_metadata": metadata,
        "runtime_version": runtime_version,
    }
    payload = {
        "receipt_version": "native-provider-execution-receipt-v1",
        **identity_payload,
        "execution_identity_sha256": hash_canonical(identity_payload),
        "raw_call_ledger": raw_call_ledger,
        "raw_call_ledger_sha256": hash_canonical(raw_call_ledger),
        "call_count": call_count,
    }
    return NativeProviderExecutionReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def freeze_native_extraction_execution_context(
    *,
    extraction_mode: Literal["paperclip_archived", "paperclip_live", "ollama_local"],
    question_config: QuestionConfig,
    pipeline_fingerprint_sha256: str,
    rendered_prompts: list[NativeRenderedPromptArtifact],
    evaluation_schemas: list[NativeEvaluationSchemaArtifact],
    provider_execution_receipts: list[NativeProviderExecutionReceipt],
    input_artifacts: list[NativeExtractionArtifactDigest],
    source_manifest_content_sha256: str,
    source_manifest_records: int,
    corpus_cutoff: str,
) -> NativeExtractionExecutionContext:
    payload = {
        "context_version": "native-extraction-execution-context-v1",
        "extraction_mode": extraction_mode,
        "question_config": question_config,
        "question_config_sha256": config_sha256(question_config),
        "pipeline_fingerprint_sha256": pipeline_fingerprint_sha256,
        "rendered_prompts": sorted(rendered_prompts, key=lambda item: item.prompt_id),
        "evaluation_schemas": sorted(
            evaluation_schemas, key=lambda item: item.schema_id
        ),
        "provider_execution_receipts": sorted(
            provider_execution_receipts, key=lambda item: item.execution_id
        ),
        "input_artifacts": sorted(input_artifacts, key=lambda item: item.artifact_id),
        "source_manifest_content_sha256": source_manifest_content_sha256,
        "source_manifest_records": source_manifest_records,
        "corpus_cutoff": corpus_cutoff,
    }
    return NativeExtractionExecutionContext.model_validate(
        {**payload, "context_sha256": hash_canonical(payload)}
    )


class NativeCoordinateCheck(StrEnum):
    EXACT = "exact"
    NOT_PROVIDED = "not_provided"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


class ResolvedSourceLine(ContractModel):
    # Source text is an immutable byte-derived payload.  The shared contract base
    # strips surrounding whitespace, which would change quoted bytes after offsets
    # have already been computed (for example LaTeX lines beginning with tabs).
    model_config = ConfigDict(str_strip_whitespace=False)

    line_id: Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]
    line_number: Annotated[int, Field(ge=1)]
    section: Annotated[str, Field(min_length=1)]
    text: str
    # Character coordinates index the canonical logical ``source_text`` Python
    # string (Unicode code points). Byte coordinates index its UTF-8 encoding.
    # Neither is a physical offset into the JSON/Parquet container; the exact
    # physical container bytes are bound separately by ``artifact_sha256``.
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=0)]
    utf8_byte_start: Annotated[int, Field(ge=0)]
    utf8_byte_end: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_line(self) -> ResolvedSourceLine:
        if self.line_id != f"L{self.line_number}":
            raise ValueError("resolved_source_line_identity_mismatch")
        if self.char_end < self.char_start:
            raise ValueError("resolved_source_line_offsets_not_ordered")
        if self.utf8_byte_end < self.utf8_byte_start:
            raise ValueError("resolved_source_line_byte_offsets_not_ordered")
        return self


class ResolvedNativeSource(ContractModel):
    """Exact logical text resolved from one verified source artifact."""

    model_config = ConfigDict(str_strip_whitespace=False)

    source_kind: Literal["antiox_json_lines", "metasyn_parquet_row"]
    artifact_path: Annotated[str, Field(min_length=1)]
    artifact_sha256: str
    source_locator: Annotated[str, Field(min_length=1)]
    source_payload_sha256: str
    source_text: str
    lines: Annotated[list[ResolvedSourceLine], Field(min_length=1)]

    @field_validator("artifact_sha256", "source_payload_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("resolved_native_source_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_lines(self) -> ResolvedNativeSource:
        numbers = [line.line_number for line in self.lines]
        if numbers != sorted(set(numbers)):
            raise ValueError("resolved_native_source_lines_not_sorted_unique")
        rebuilt = "\n".join(line.text for line in self.lines)
        if rebuilt != self.source_text:
            raise ValueError("resolved_native_source_text_mismatch")
        cursor = 0
        byte_cursor = 0
        for index, line in enumerate(self.lines):
            expected_end = cursor + len(line.text)
            expected_byte_end = byte_cursor + len(line.text.encode("utf-8"))
            if line.char_start != cursor or line.char_end != expected_end:
                raise ValueError("resolved_native_source_line_offset_mismatch")
            if (
                line.utf8_byte_start != byte_cursor
                or line.utf8_byte_end != expected_byte_end
            ):
                raise ValueError("resolved_native_source_line_byte_offset_mismatch")
            cursor = expected_end + (1 if index < len(self.lines) - 1 else 0)
            byte_cursor = expected_byte_end + (1 if index < len(self.lines) - 1 else 0)
        return self


class NativeFindingGroundingResult(ContractModel):
    finding_path: Annotated[str, Field(min_length=1)]
    status: GroundingStatus
    evidence_source_locator: Annotated[str, Field(min_length=1)]
    locator_check: NativeCoordinateCheck
    quote_check: NativeCoordinateCheck
    line_check: NativeCoordinateCheck
    offset_check: NativeCoordinateCheck
    resolved_line_numbers: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    normalized_quote_sha256: str | None = None
    normalized_cited_text_sha256: str | None = None
    issues: list[str] = Field(default_factory=list)
    result_sha256: str

    @field_validator("normalized_quote_sha256", "normalized_cited_text_sha256", "result_sha256")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("native_finding_grounding_sha256_invalid")
        return value

    @field_validator("resolved_line_numbers")
    @classmethod
    def validate_lines(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)):
            raise ValueError("native_finding_grounding_lines_not_sorted_unique")
        return value

    @field_validator("issues")
    @classmethod
    def validate_issues(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("native_finding_grounding_issues_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_result_hash(self) -> NativeFindingGroundingResult:
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("native_finding_grounding_hash_mismatch")
        if self.status is GroundingStatus.EXACT:
            if self.locator_check is not NativeCoordinateCheck.EXACT:
                raise ValueError("exact_native_grounding_requires_locator")
            if self.quote_check is not NativeCoordinateCheck.EXACT:
                raise ValueError("exact_native_grounding_requires_quote")
            coordinate_checks = (self.line_check, self.offset_check)
            if all(item is NativeCoordinateCheck.NOT_PROVIDED for item in coordinate_checks):
                raise ValueError("exact_native_grounding_requires_coordinates")
            if any(
                item in {NativeCoordinateCheck.MISMATCH, NativeCoordinateCheck.UNVERIFIABLE}
                for item in coordinate_checks
            ):
                raise ValueError("exact_native_grounding_forbids_failed_coordinates")
            if self.issues:
                raise ValueError("exact_native_grounding_forbids_issues")
        return self


class NativeGroundingReceipt(ContractModel):
    """Hash-bound evidence that all native findings were checked against source bytes."""

    receipt_version: Literal["native-grounding-receipt-v1"] = "native-grounding-receipt-v1"
    source_document: SourceDocumentArtifact
    expected_source_sha256: str
    observed_source_sha256: str | None
    source_payload_sha256: str | None
    extraction: NativePublicationExtraction
    extraction_sha256: str
    source_verified: bool
    extraction_status: FragmentStatus
    finding_results: list[NativeFindingGroundingResult]
    all_findings_exact: bool
    authorizes_estimable_fragment: bool
    issues: list[str] = Field(default_factory=list)
    receipt_sha256: str

    @field_validator(
        "expected_source_sha256",
        "observed_source_sha256",
        "source_payload_sha256",
        "extraction_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("native_grounding_receipt_sha256_invalid")
        return value

    @field_validator("issues")
    @classmethod
    def validate_issues(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("native_grounding_receipt_issues_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativeGroundingReceipt:
        paths = [result.finding_path for result in self.finding_results]
        if paths != sorted(set(paths)):
            raise ValueError("native_grounding_finding_paths_not_sorted_unique")
        expected_all_exact = bool(self.finding_results) and all(
            result.status is GroundingStatus.EXACT for result in self.finding_results
        )
        if self.all_findings_exact != expected_all_exact:
            raise ValueError("native_grounding_all_exact_mismatch")
        expected_authorization = (
            self.extraction_status is FragmentStatus.ESTIMABLE
            and self.source_verified
            and self.all_findings_exact
            and not self.issues
        )
        if self.authorizes_estimable_fragment != expected_authorization:
            raise ValueError("native_grounding_authorization_mismatch")
        if hash_canonical(self.extraction) != self.extraction_sha256:
            raise ValueError("native_grounding_extraction_hash_mismatch")
        if self.extraction.status is not self.extraction_status:
            raise ValueError("native_grounding_extraction_status_mismatch")
        if self.expected_source_sha256 != self.source_document.sha256:
            raise ValueError("native_grounding_expected_source_hash_mismatch")
        if self.source_verified:
            if self.observed_source_sha256 != self.expected_source_sha256:
                raise ValueError("verified_native_source_observed_hash_mismatch")
            if self.source_payload_sha256 is None:
                raise ValueError("verified_native_source_requires_payload_hash")
            if self.issues:
                raise ValueError("verified_native_source_forbids_source_issues")
        else:
            if self.source_payload_sha256 is not None:
                raise ValueError("unverified_native_source_forbids_payload_hash")
            if not self.issues:
                raise ValueError("unverified_native_source_requires_issue")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("native_grounding_receipt_hash_mismatch")
        return self


class NativeGroundingLinkKind(StrEnum):
    ESTIMABLE_AUTHORIZED = "estimable_authorized"
    EXPECTED_NON_ESTIMABLE_EXTRACTION = "expected_non_estimable_extraction"
    FAILED_ESTIMABLE_GROUNDING = "failed_estimable_grounding"


class NativeGroundingFragmentLink(ContractModel):
    publication_id: Annotated[str, Field(min_length=1)]
    fragment_status: FragmentStatus
    grounding_receipt_sha256: str
    receipt_extraction_status: FragmentStatus
    receipt_authorizes_estimable: bool
    link_kind: NativeGroundingLinkKind

    @field_validator("grounding_receipt_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_grounding_link_sha256_invalid")
        return value


class NativeCorpusGroundingValidation(ContractModel):
    validation_version: Literal["native-corpus-grounding-validation-v1"] = (
        "native-corpus-grounding-validation-v1"
    )
    corpus_sha256: str
    grounding_receipt_set_sha256: str
    links: list[NativeGroundingFragmentLink]
    unlinked_non_estimable_fragments: Annotated[int, Field(ge=0)]
    estimable_authorized_receipts: Annotated[int, Field(ge=0)]
    expected_non_estimable_extraction_receipts: Annotated[int, Field(ge=0)]
    failed_estimable_grounding_receipts: Annotated[int, Field(ge=0)]
    validation_sha256: str

    @field_validator(
        "corpus_sha256",
        "grounding_receipt_set_sha256",
        "validation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_corpus_grounding_validation_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> NativeCorpusGroundingValidation:
        publication_ids = [link.publication_id for link in self.links]
        if publication_ids != sorted(set(publication_ids)):
            raise ValueError("native_corpus_grounding_links_not_sorted_unique")
        expected_counts = {
            NativeGroundingLinkKind.ESTIMABLE_AUTHORIZED: (self.estimable_authorized_receipts),
            NativeGroundingLinkKind.EXPECTED_NON_ESTIMABLE_EXTRACTION: (
                self.expected_non_estimable_extraction_receipts
            ),
            NativeGroundingLinkKind.FAILED_ESTIMABLE_GROUNDING: (
                self.failed_estimable_grounding_receipts
            ),
        }
        for kind, expected in expected_counts.items():
            if sum(link.link_kind is kind for link in self.links) != expected:
                raise ValueError("native_corpus_grounding_link_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"validation_sha256"})
        if hash_canonical(payload) != self.validation_sha256:
            raise ValueError("native_corpus_grounding_validation_hash_mismatch")
        return self


class NativeExtractionContextReceipt(ContractModel):
    """Join one exact extraction execution context to one immutable package core."""

    receipt_version: Literal["native-extraction-context-receipt-v1"] = (
        "native-extraction-context-receipt-v1"
    )
    execution_context: NativeExtractionExecutionContext
    corpus_sha256: str
    grounding_validation_sha256: str
    cohort_reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    source_manifest_sha256: str
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    package_core_sha256: str
    receipt_sha256: str

    @field_validator(
        "corpus_sha256",
        "grounding_validation_sha256",
        "cohort_reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "source_manifest_sha256",
        "package_core_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_receipt_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_extraction_context_receipt_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativeExtractionContextReceipt:
        context = self.execution_context
        if context.source_manifest_content_sha256 != self.source_manifest_sha256:
            raise ValueError("native_extraction_context_source_manifest_mismatch")
        if context.corpus_cutoff != self.corpus_cutoff:
            raise ValueError("native_extraction_context_corpus_cutoff_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("native_extraction_context_receipt_hash_mismatch")
        return self


class TypedEvidenceGroundingPackage(ContractModel):
    """A typed corpus joined to source and cohort-identity replay receipts."""

    package_version: Literal[
        "typed-evidence-grounding-package-v1",
        "typed-evidence-grounding-package-v2",
        "typed-evidence-grounding-package-v3",
        "typed-evidence-grounding-package-v4",
    ] = "typed-evidence-grounding-package-v3"
    corpus: TypedEvidenceCorpus
    grounding_receipts: list[NativeGroundingReceipt]
    grounding_validation: NativeCorpusGroundingValidation
    cohort_reconciliation: NativeCohortReconciliationReceipt | None = None
    source_manifest: NativeSourceManifest | None = None
    source_manifest_sha256: str | None = None
    corpus_cutoff: str | None = None
    extraction_context_receipt: NativeExtractionContextReceipt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    package_sha256: str

    @field_validator("source_manifest_sha256", "package_sha256")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("typed_evidence_grounding_package_sha256_invalid")
        return value

    @field_validator("corpus_cutoff")
    @classmethod
    def validate_corpus_cutoff(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("typed_evidence_grounding_package_corpus_cutoff_empty")
        return value

    @model_validator(mode="after")
    def validate_package(self) -> TypedEvidenceGroundingPackage:
        expected = validate_typed_corpus_grounding(
            corpus=self.corpus,
            grounding_receipts=self.grounding_receipts,
        )
        if expected.validation_sha256 != self.grounding_validation.validation_sha256:
            raise ValueError("typed_evidence_grounding_package_validation_mismatch")
        if self.package_version == "typed-evidence-grounding-package-v1":
            if self.cohort_reconciliation is not None:
                raise ValueError("typed_evidence_grounding_package_v1_forbids_reconciliation")
        elif self.cohort_reconciliation is None:
            raise ValueError("typed_evidence_grounding_package_v2_v3_requires_reconciliation")
        if self.cohort_reconciliation is not None:
            try:
                reverify_native_cohort_reconciliation(
                    corpus=self.corpus,
                    receipt=self.cohort_reconciliation,
                )
            except NativeCohortReconciliationError as exc:
                raise ValueError(
                    f"typed_evidence_grounding_package_reconciliation_invalid:{exc}"
                ) from exc
        membership_fields_present = (
            self.source_manifest is not None,
            self.source_manifest_sha256 is not None,
            self.corpus_cutoff is not None,
        )
        if self.package_version in {
            "typed-evidence-grounding-package-v1",
            "typed-evidence-grounding-package-v2",
        }:
            if any(membership_fields_present):
                raise ValueError(
                    "typed_evidence_grounding_package_v1_v2_forbids_source_membership"
                )
        else:
            if not all(membership_fields_present):
                raise ValueError(
                    "typed_evidence_grounding_package_v3_requires_source_membership"
                )
            assert self.source_manifest is not None
            assert self.source_manifest_sha256 is not None
            if hash_canonical(self.source_manifest) != self.source_manifest_sha256:
                raise ValueError("typed_evidence_grounding_package_source_manifest_hash_mismatch")
            if self.source_manifest.question_id != self.corpus.question_id:
                raise ValueError("typed_evidence_grounding_package_source_question_mismatch")
            records = {
                record.publication.publication_id: record
                for record in self.source_manifest.records
            }
            fragments = {
                fragment.publication_id: fragment for fragment in self.corpus.fragments
            }
            if set(records) != set(fragments):
                raise ValueError("typed_evidence_grounding_package_source_membership_mismatch")
            for publication_id in sorted(records):
                record = records[publication_id]
                fragment = fragments[publication_id]
                if record.publication.model_dump(mode="json") != fragment.publication.model_dump(
                    mode="json"
                ):
                    raise ValueError(
                        "typed_evidence_grounding_package_source_publication_mismatch"
                    )
                if record.source_document.model_dump(
                    mode="json"
                ) != fragment.source_document.model_dump(mode="json"):
                    raise ValueError(
                        "typed_evidence_grounding_package_source_document_mismatch"
                    )
        if self.package_version == "typed-evidence-grounding-package-v4":
            receipt = self.extraction_context_receipt
            if receipt is None:
                raise ValueError("typed_evidence_grounding_package_v4_requires_context")
            if self.corpus.corpus_version != "typed-evidence-corpus-v3":
                raise ValueError("typed_evidence_grounding_package_v4_requires_corpus_v3")
            if self.corpus.extraction_context_sha256 != receipt.execution_context.context_sha256:
                raise ValueError("typed_evidence_grounding_package_context_corpus_mismatch")
            if (
                self.corpus.pipeline_fingerprint_sha256
                != receipt.execution_context.pipeline_fingerprint_sha256
            ):
                raise ValueError("typed_evidence_grounding_package_context_pipeline_mismatch")
            if self.corpus.question_id != receipt.execution_context.question_config.question_id:
                raise ValueError("typed_evidence_grounding_package_context_question_mismatch")
            if self.corpus.corpus_sha256 != receipt.corpus_sha256:
                raise ValueError("typed_evidence_grounding_package_context_corpus_hash_mismatch")
            if self.grounding_validation.validation_sha256 != (
                receipt.grounding_validation_sha256
            ):
                raise ValueError("typed_evidence_grounding_package_context_grounding_mismatch")
            assert self.cohort_reconciliation is not None
            if self.cohort_reconciliation.receipt_sha256 != (
                receipt.cohort_reconciliation_receipt_sha256
            ):
                raise ValueError("typed_evidence_grounding_package_context_reconciliation_mismatch")
            if self.cohort_reconciliation.reconciled_graph_sha256 != (
                receipt.reconciled_graph_sha256
            ):
                raise ValueError("typed_evidence_grounding_package_context_graph_mismatch")
            if self.source_manifest_sha256 != receipt.source_manifest_sha256:
                raise ValueError("typed_evidence_grounding_package_context_manifest_mismatch")
            if self.corpus_cutoff != receipt.corpus_cutoff:
                raise ValueError("typed_evidence_grounding_package_context_cutoff_mismatch")
            core = self.model_dump(
                mode="json",
                exclude={"extraction_context_receipt", "package_sha256"},
            )
            if hash_canonical(core) != receipt.package_core_sha256:
                raise ValueError("typed_evidence_grounding_package_core_hash_mismatch")
        else:
            if self.extraction_context_receipt is not None:
                raise ValueError("legacy_typed_evidence_grounding_package_forbids_context")
            if self.corpus.corpus_version != "typed-evidence-corpus-v2":
                raise ValueError("legacy_typed_evidence_grounding_package_requires_corpus_v2")
        payload = self.model_dump(mode="json", exclude={"package_sha256"})
        if self.package_version == "typed-evidence-grounding-package-v1":
            payload.pop("cohort_reconciliation")
        if self.package_version in {
            "typed-evidence-grounding-package-v1",
            "typed-evidence-grounding-package-v2",
        }:
            payload.pop("source_manifest")
            payload.pop("source_manifest_sha256")
            payload.pop("corpus_cutoff")
        if self.package_version != "typed-evidence-grounding-package-v4":
            payload.pop("extraction_context_receipt", None)
        if hash_canonical(payload) != self.package_sha256:
            raise ValueError("typed_evidence_grounding_package_hash_mismatch")
        return self


class NativeGroundingReplayVerification(ContractModel):
    replay_version: Literal["native-grounding-replay-v4"] = "native-grounding-replay-v4"
    package_sha256: str
    corpus_sha256: str
    grounding_validation_sha256: str
    cohort_reconciliation_receipt_sha256: str | None = None
    reconciled_graph_sha256: str | None = None
    source_manifest_sha256: str | None = None
    source_manifest_records: Annotated[int, Field(ge=0)]
    corpus_cutoff: str | None = None
    extraction_context_sha256: str | None = None
    extraction_context_receipt_sha256: str | None = None
    package_core_sha256: str | None = None
    question_config_sha256: str | None = None
    rendered_prompt_sha256s: list[str] = Field(default_factory=list)
    evaluation_schema_sha256s: list[str] = Field(default_factory=list)
    provider_execution_receipt_sha256s: list[str] = Field(default_factory=list)
    replayed_receipts: Annotated[int, Field(ge=0)]
    projected_estimable_fragments: Annotated[int, Field(ge=0)]
    replay_sha256: str

    @field_validator(
        "package_sha256",
        "corpus_sha256",
        "grounding_validation_sha256",
        "cohort_reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "source_manifest_sha256",
        "extraction_context_sha256",
        "extraction_context_receipt_sha256",
        "package_core_sha256",
        "question_config_sha256",
        "replay_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("native_grounding_replay_sha256_invalid")
        return value

    @field_validator(
        "rendered_prompt_sha256s",
        "evaluation_schema_sha256s",
        "provider_execution_receipt_sha256s",
    )
    @classmethod
    def validate_hash_list(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            SHA256_RE.fullmatch(item) is None for item in value
        ):
            raise ValueError("native_grounding_replay_hash_list_invalid")
        return value

    @model_validator(mode="after")
    def validate_replay(self) -> NativeGroundingReplayVerification:
        payload = self.model_dump(mode="json", exclude={"replay_sha256"})
        if hash_canonical(payload) != self.replay_sha256:
            raise ValueError("native_grounding_replay_hash_mismatch")
        return self


def _reverify_native_extraction_execution_context(
    *,
    context: NativeExtractionExecutionContext,
    repository_root: Path,
) -> None:
    """Recompute every context link available from repository and embedded bytes."""

    validated = NativeExtractionExecutionContext.model_validate(
        context.model_dump(mode="json")
    )
    root = repository_root.resolve(strict=True)
    official = native_publication_extraction_json_schema()
    official_hash = hash_canonical(official)
    official_schemas = [
        schema
        for schema in validated.evaluation_schemas
        if schema.role == "official_postvalidation"
    ]
    if len(official_schemas) != 1 or (
        official_schemas[0].schema_payload != official
        or official_schemas[0].schema_sha256 != official_hash
    ):
        raise NativeGroundingError("native_extraction_official_schema_replay_mismatch")
    for prompt in validated.rendered_prompts:
        template_path = root / prompt.template_path
        if (
            template_path.is_symlink()
            or not template_path.is_file()
            or _sha256_file(template_path) != prompt.template_sha256
        ):
            raise NativeGroundingError("native_extraction_prompt_template_replay_mismatch")
        if prompt.renderer_id != "repository-native-extraction-v1":
            continue
        template = template_path.read_text(encoding="utf-8")
        rendered, version = render_prompt_text(
            template,
            native_extraction_prompt_replacements(validated.question_config),
        )
        if (
            version != prompt.prompt_version
            or rendered != prompt.rendered_prompt
            or hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            != prompt.rendered_prompt_sha256
        ):
            raise NativeGroundingError("native_extraction_rendered_prompt_replay_mismatch")


class _ResolutionFailure(NativeGroundingError):
    def __init__(self, code: str, *, observed_sha256: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.observed_sha256 = observed_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact_path(
    source_document: SourceDocumentArtifact,
    *,
    repository_root: Path,
) -> tuple[Path, str]:
    root = Path(os.path.abspath(repository_root))
    if not root.exists() or not root.is_dir():
        raise _ResolutionFailure("repository_root_not_directory")
    if root.is_symlink():
        raise _ResolutionFailure("repository_root_symlink_forbidden")
    relative = PurePosixPath(source_document.artifact_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise _ResolutionFailure("source_artifact_path_not_canonical_relative")
    if relative.as_posix() != source_document.artifact_path:
        raise _ResolutionFailure("source_artifact_path_not_posix_canonical")

    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise _ResolutionFailure("source_artifact_unreadable_or_missing") from exc
        if stat.S_ISLNK(mode):
            raise _ResolutionFailure("source_artifact_symlink_forbidden")
    try:
        root_resolved = root.resolve(strict=True)
        candidate_resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise _ResolutionFailure("source_artifact_unreadable_or_missing") from exc
    if not candidate_resolved.is_relative_to(root_resolved):
        raise _ResolutionFailure("source_artifact_escapes_repository")
    if not candidate_resolved.is_file():
        raise _ResolutionFailure("source_artifact_not_regular_file")
    try:
        observed = _sha256_file(candidate_resolved)
    except OSError as exc:
        raise _ResolutionFailure("source_artifact_unreadable_or_missing") from exc
    if observed != source_document.sha256:
        raise _ResolutionFailure("source_artifact_hash_mismatch", observed_sha256=observed)
    return candidate_resolved, observed


def _decode_json_pointer_segment(value: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "~":
            output.append(value[index])
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in {"0", "1"}:
            raise _ResolutionFailure("json_source_locator_pointer_escape_invalid")
        output.append("~" if value[index + 1] == "0" else "/")
        index += 2
    return "".join(output)


def _build_resolved_source(
    *,
    source_kind: Literal["antiox_json_lines", "metasyn_parquet_row"],
    source_document: SourceDocumentArtifact,
    observed_sha256: str,
    source_payload: Any,
    raw_lines: list[tuple[int, str, str]],
) -> ResolvedNativeSource:
    if not raw_lines:
        raise _ResolutionFailure(
            "source_payload_has_no_text_lines", observed_sha256=observed_sha256
        )
    lines: list[ResolvedSourceLine] = []
    cursor = 0
    byte_cursor = 0
    for index, (number, section, text) in enumerate(raw_lines):
        end = cursor + len(text)
        byte_end = byte_cursor + len(text.encode("utf-8"))
        lines.append(
            ResolvedSourceLine(
                line_id=f"L{number}",
                line_number=number,
                section=section,
                text=text,
                char_start=cursor,
                char_end=end,
                utf8_byte_start=byte_cursor,
                utf8_byte_end=byte_end,
            )
        )
        cursor = end + (1 if index < len(raw_lines) - 1 else 0)
        byte_cursor = byte_end + (1 if index < len(raw_lines) - 1 else 0)
    return ResolvedNativeSource(
        source_kind=source_kind,
        artifact_path=source_document.artifact_path,
        artifact_sha256=observed_sha256,
        source_locator=source_document.source_locator,
        source_payload_sha256=hash_canonical(source_payload),
        source_text="\n".join(line.text for line in lines),
        lines=lines,
    )


def _resolve_antiox_json(
    *,
    path: Path,
    observed_sha256: str,
    source_document: SourceDocumentArtifact,
    locator_match: re.Match[str],
) -> ResolvedNativeSource:
    if source_document.media_type.partition(";")[0].strip().casefold() != "application/json":
        raise _ResolutionFailure("json_source_media_type_mismatch", observed_sha256=observed_sha256)
    if locator_match.group("path") != source_document.artifact_path:
        raise _ResolutionFailure(
            "json_source_locator_path_mismatch", observed_sha256=observed_sha256
        )
    doc_id = _decode_json_pointer_segment(locator_match.group("pointer"))
    if not doc_id:
        raise _ResolutionFailure(
            "json_source_locator_doc_id_empty", observed_sha256=observed_sha256
        )
    try:
        container = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ResolutionFailure(
            "json_source_artifact_invalid", observed_sha256=observed_sha256
        ) from exc
    if not isinstance(container, dict):
        raise _ResolutionFailure("json_source_root_not_object", observed_sha256=observed_sha256)
    if doc_id not in container:
        raise _ResolutionFailure("json_source_doc_id_missing", observed_sha256=observed_sha256)
    payload = container[doc_id]
    if not isinstance(payload, dict) or not payload:
        raise _ResolutionFailure(
            "json_source_lines_not_nonempty_object", observed_sha256=observed_sha256
        )
    raw_lines: list[tuple[int, str, str]] = []
    seen_numbers: set[int] = set()
    for line_id, line in payload.items():
        match = _LINE_ID.fullmatch(line_id) if isinstance(line_id, str) else None
        if match is None or not isinstance(line, dict) or set(line) != {"section", "text"}:
            raise _ResolutionFailure(
                "json_source_line_schema_invalid", observed_sha256=observed_sha256
            )
        number = int(match.group("number"))
        if number in seen_numbers:
            raise _ResolutionFailure(
                "json_source_line_number_duplicate", observed_sha256=observed_sha256
            )
        section = line["section"]
        text = line["text"]
        if not isinstance(section, str) or not section.strip() or not isinstance(text, str):
            raise _ResolutionFailure(
                "json_source_line_value_invalid", observed_sha256=observed_sha256
            )
        seen_numbers.add(number)
        raw_lines.append((number, section, text))
    raw_lines.sort(key=lambda item: item[0])
    return _build_resolved_source(
        source_kind="antiox_json_lines",
        source_document=source_document,
        observed_sha256=observed_sha256,
        source_payload=payload,
        raw_lines=raw_lines,
    )


def _parse_nonnegative_integer(value: str, *, field: str) -> int:
    if not value or not value.isdigit() or str(int(value)) != value:
        raise _ResolutionFailure(f"parquet_source_locator_{field}_invalid")
    return int(value)


def _section_heading(section: dict[str, Any], *, observed_sha256: str) -> str:
    candidates = [
        section[key].strip()
        for key in ("heading", "section", "title")
        if key in section and isinstance(section[key], str) and section[key].strip()
    ]
    if len(set(candidates)) > 1:
        raise _ResolutionFailure(
            "parquet_source_section_heading_ambiguous", observed_sha256=observed_sha256
        )
    return candidates[0] if candidates else "unknown"


def _resolve_metasyn_parquet(
    *,
    path: Path,
    observed_sha256: str,
    source_document: SourceDocumentArtifact,
    locator_match: re.Match[str],
) -> ResolvedNativeSource:
    media_type = source_document.media_type.partition(";")[0].strip().casefold()
    if media_type not in {"application/vnd.apache.parquet", "application/x-parquet"}:
        raise _ResolutionFailure(
            "parquet_source_media_type_mismatch", observed_sha256=observed_sha256
        )
    if locator_match.group("path") != source_document.artifact_path:
        raise _ResolutionFailure(
            "parquet_source_locator_path_mismatch", observed_sha256=observed_sha256
        )
    try:
        pairs = parse_qsl(locator_match.group("query"), keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise _ResolutionFailure(
            "parquet_source_locator_query_invalid", observed_sha256=observed_sha256
        ) from exc
    if len(pairs) != len(_PARQUET_KEYS) or {key for key, _ in pairs} != _PARQUET_KEYS:
        raise _ResolutionFailure(
            "parquet_source_locator_keys_invalid", observed_sha256=observed_sha256
        )
    query = dict(pairs)
    if query["index_base"] != "0":
        raise _ResolutionFailure(
            "parquet_source_locator_index_base_not_zero", observed_sha256=observed_sha256
        )
    try:
        row_group = _parse_nonnegative_integer(query["row_group"], field="row_group")
        row_in_group = _parse_nonnegative_integer(query["row_in_group"], field="row_in_group")
        expected_id = _parse_nonnegative_integer(query["ID"], field="ID")
    except _ResolutionFailure as exc:
        raise _ResolutionFailure(exc.code, observed_sha256=observed_sha256) from exc
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise _ResolutionFailure(
            "parquet_source_artifact_invalid", observed_sha256=observed_sha256
        ) from exc
    required_columns = {"ID", "title", "abstract", "sections"}
    if not required_columns.issubset(parquet.schema_arrow.names):
        raise _ResolutionFailure("parquet_source_columns_missing", observed_sha256=observed_sha256)
    if row_group >= parquet.metadata.num_row_groups:
        raise _ResolutionFailure(
            "parquet_source_row_group_out_of_range", observed_sha256=observed_sha256
        )
    row_count = parquet.metadata.row_group(row_group).num_rows
    if row_in_group >= row_count:
        raise _ResolutionFailure("parquet_source_row_out_of_range", observed_sha256=observed_sha256)
    try:
        rows = (
            parquet.read_row_group(row_group, columns=["ID", "title", "abstract", "sections"])
            .slice(row_in_group, 1)
            .to_pylist()
        )
    except Exception as exc:
        raise _ResolutionFailure(
            "parquet_source_row_unreadable", observed_sha256=observed_sha256
        ) from exc
    if len(rows) != 1:
        raise _ResolutionFailure("parquet_source_row_not_unique", observed_sha256=observed_sha256)
    row = rows[0]
    observed_id = row.get("ID")
    if (
        not isinstance(observed_id, int)
        or isinstance(observed_id, bool)
        or observed_id != expected_id
    ):
        raise _ResolutionFailure("parquet_source_row_ID_mismatch", observed_sha256=observed_sha256)

    raw_lines: list[tuple[int, str, str]] = []

    def append_text(value: Any, section: str) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise _ResolutionFailure(
                "parquet_source_text_column_invalid", observed_sha256=observed_sha256
            )
        for raw_line in value.splitlines() or [value]:
            if raw_line.strip():
                raw_lines.append((len(raw_lines) + 1, section, raw_line))

    append_text(row.get("title"), "Title")
    append_text(row.get("abstract"), "Abstract")
    sections = row.get("sections")
    if sections is None:
        sections = []
    if not isinstance(sections, list):
        raise _ResolutionFailure(
            "parquet_source_sections_not_list", observed_sha256=observed_sha256
        )
    for section in sections:
        if not isinstance(section, dict) or "text" not in section:
            raise _ResolutionFailure(
                "parquet_source_section_schema_invalid", observed_sha256=observed_sha256
            )
        append_text(section["text"], _section_heading(section, observed_sha256=observed_sha256))
    source_payload = {
        "ID": observed_id,
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "sections": sections,
    }
    return _build_resolved_source(
        source_kind="metasyn_parquet_row",
        source_document=source_document,
        observed_sha256=observed_sha256,
        source_payload=source_payload,
        raw_lines=raw_lines,
    )


def resolve_native_source_document(
    *,
    repository_root: Path,
    source_document: SourceDocumentArtifact,
) -> ResolvedNativeSource:
    """Resolve one exact bridge locator after checking path, symlinks, and file hash."""

    path, observed = _verified_artifact_path(source_document, repository_root=repository_root)
    json_match = _JSON_LOCATOR.fullmatch(source_document.source_locator)
    if json_match is not None:
        return _resolve_antiox_json(
            path=path,
            observed_sha256=observed,
            source_document=source_document,
            locator_match=json_match,
        )
    parquet_match = _PARQUET_LOCATOR.fullmatch(source_document.source_locator)
    if parquet_match is not None:
        return _resolve_metasyn_parquet(
            path=path,
            observed_sha256=observed,
            source_document=source_document,
            locator_match=parquet_match,
        )
    raise NativeGroundingError("native_source_locator_unsupported")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _freeze_finding_result(payload: dict[str, Any]) -> NativeFindingGroundingResult:
    return NativeFindingGroundingResult.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def _verify_finding(
    *,
    finding_path: str,
    evidence: Any,
    source: ResolvedNativeSource,
) -> NativeFindingGroundingResult:
    issues: list[str] = []
    locator_check = NativeCoordinateCheck.EXACT
    if evidence.source_locator != source.source_locator:
        locator_check = NativeCoordinateCheck.MISMATCH
        issues.append("evidence_source_locator_mismatch")

    quote_check = NativeCoordinateCheck.EXACT
    normalized_quote: str | None = None
    if not isinstance(evidence.quote, str) or not evidence.quote.strip():
        quote_check = NativeCoordinateCheck.UNVERIFIABLE
        issues.append("evidence_quote_missing")
    else:
        normalized_quote = normalize_evidence_text(evidence.quote)
        if evidence.quote not in source.source_text:
            quote_check = NativeCoordinateCheck.MISMATCH
            issues.append("evidence_quote_not_exact_source_substring")

    content_lines = {
        line.line_number: {"section": line.section, "text": line.text} for line in source.lines
    }
    line_check = NativeCoordinateCheck.NOT_PROVIDED
    resolved_line_numbers: list[int] = []
    cited_text: str | None = None
    if evidence.line_ids:
        if normalized_quote is None:
            line_check = NativeCoordinateCheck.UNVERIFIABLE
        else:
            try:
                grounded = ground_evidence(evidence.quote, evidence.line_ids, content_lines)
            except (GroundingContractError, TypeError, ValueError):
                line_check = NativeCoordinateCheck.UNVERIFIABLE
                issues.append("evidence_line_ids_invalid_or_unresolved")
            else:
                resolved_line_numbers = sorted(set(grounded["resolved_line_numbers"]))
                cited_text = grounded["normalized_cited_text"] or None
                if "relocated_from_line_numbers" in grounded:
                    line_check = NativeCoordinateCheck.MISMATCH
                    issues.append("evidence_line_relocation_forbidden")
                elif "refined_from_line_numbers" in grounded:
                    line_check = NativeCoordinateCheck.MISMATCH
                    issues.append("evidence_line_refinement_forbidden")
                elif grounded["grounding_status"] != "exact":
                    line_check = (
                        NativeCoordinateCheck.MISMATCH
                        if grounded["grounding_status"] == "mismatch"
                        else NativeCoordinateCheck.UNVERIFIABLE
                    )
                    issues.append(f"evidence_line_grounding_{grounded['grounding_status']}")
                elif evidence.quote not in "\n".join(
                    str(content_lines[number]["text"]) for number in resolved_line_numbers
                ):
                    line_check = NativeCoordinateCheck.MISMATCH
                    issues.append("evidence_quote_not_exact_cited_line_substring")
                elif (
                    grounded["section_flagged"]
                    or grounded["evidence_section"].strip().casefold() in _FORBIDDEN_SECTIONS
                ):
                    line_check = NativeCoordinateCheck.MISMATCH
                    issues.append("evidence_section_forbidden_or_unknown")
                elif evidence.section is not None and (
                    evidence.section.strip().casefold()
                    != grounded["evidence_section"].strip().casefold()
                ):
                    line_check = NativeCoordinateCheck.MISMATCH
                    issues.append("evidence_reported_section_mismatch")
                else:
                    line_check = NativeCoordinateCheck.EXACT

    offset_check = NativeCoordinateCheck.NOT_PROVIDED
    if evidence.char_start is not None and evidence.char_end is not None:
        if normalized_quote is None:
            offset_check = NativeCoordinateCheck.UNVERIFIABLE
        elif evidence.char_end > len(source.source_text):
            offset_check = NativeCoordinateCheck.UNVERIFIABLE
            issues.append("evidence_offsets_out_of_range")
        else:
            raw_offset_text = source.source_text[evidence.char_start : evidence.char_end]
            offset_text = normalize_evidence_text(raw_offset_text)
            if evidence.quote in raw_offset_text:
                offset_check = NativeCoordinateCheck.EXACT
                if cited_text is None:
                    cited_text = offset_text
            else:
                offset_check = NativeCoordinateCheck.MISMATCH
                issues.append("evidence_quote_not_contained_at_offsets")

    if (
        line_check is NativeCoordinateCheck.NOT_PROVIDED
        and offset_check is NativeCoordinateCheck.NOT_PROVIDED
    ):
        issues.append("evidence_coordinates_missing")
    checks = (locator_check, quote_check, line_check, offset_check)
    if any(check is NativeCoordinateCheck.MISMATCH for check in checks):
        status = GroundingStatus.MISMATCH
    elif any(check is NativeCoordinateCheck.UNVERIFIABLE for check in checks) or (
        line_check is NativeCoordinateCheck.NOT_PROVIDED
        and offset_check is NativeCoordinateCheck.NOT_PROVIDED
    ):
        status = GroundingStatus.UNVERIFIABLE
    else:
        status = GroundingStatus.EXACT
    payload = {
        "finding_path": finding_path,
        "status": status,
        "evidence_source_locator": evidence.source_locator,
        "locator_check": locator_check,
        "quote_check": quote_check,
        "line_check": line_check,
        "offset_check": offset_check,
        "resolved_line_numbers": resolved_line_numbers,
        "normalized_quote_sha256": _text_sha256(normalized_quote) if normalized_quote else None,
        "normalized_cited_text_sha256": _text_sha256(cited_text) if cited_text else None,
        "issues": sorted(set(issues)),
    }
    return _freeze_finding_result(payload)


def _freeze_receipt(payload: dict[str, Any]) -> NativeGroundingReceipt:
    return NativeGroundingReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def verify_native_publication_grounding(
    *,
    repository_root: Path,
    source_document: SourceDocumentArtifact,
    extraction: NativePublicationExtraction,
) -> NativeGroundingReceipt:
    """Verify all findings; only a wholly exact estimable extraction is authorized.

    Source failures are represented by a self-hashed, non-authorizing receipt.  The
    function never searches for a moved artifact or a similar Parquet row.
    """

    extraction_sha256 = hash_canonical(extraction)
    observed_sha256: str | None = None
    try:
        source = resolve_native_source_document(
            repository_root=repository_root,
            source_document=source_document,
        )
        observed_sha256 = source.artifact_sha256
    except _ResolutionFailure as exc:
        observed_sha256 = exc.observed_sha256
        issue = exc.code
        source = None
    except NativeGroundingError as exc:
        issue = str(exc)
        source = None

    finding_results: list[NativeFindingGroundingResult] = []
    if source is not None:
        for study in extraction.studies:
            for cohort in study.cohorts:
                for finding in cohort.findings:
                    finding_results.append(
                        _verify_finding(
                            finding_path=f"{study.key}/{cohort.key}/{finding.key}",
                            evidence=finding.evidence,
                            source=source,
                        )
                    )
        finding_results.sort(key=lambda result: result.finding_path)
        receipt_issues: list[str] = []
    else:
        receipt_issues = [issue]
    all_exact = bool(finding_results) and all(
        result.status is GroundingStatus.EXACT for result in finding_results
    )
    source_verified = source is not None
    authorizes = (
        extraction.status is FragmentStatus.ESTIMABLE
        and source_verified
        and all_exact
        and not receipt_issues
    )
    payload = {
        "receipt_version": "native-grounding-receipt-v1",
        "source_document": source_document,
        "expected_source_sha256": source_document.sha256,
        "observed_source_sha256": observed_sha256,
        "source_payload_sha256": source.source_payload_sha256 if source is not None else None,
        "extraction": extraction,
        "extraction_sha256": extraction_sha256,
        "source_verified": source_verified,
        "extraction_status": extraction.status,
        "finding_results": finding_results,
        "all_findings_exact": all_exact,
        "authorizes_estimable_fragment": authorizes,
        "issues": sorted(set(receipt_issues)),
    }
    return _freeze_receipt(payload)


def freeze_grounding_checked_publication_fragment(
    *,
    extraction: NativePublicationExtraction,
    grounding_receipt: NativeGroundingReceipt,
    question_id: str,
    publication: PublicationIdentity,
    pipeline_fingerprint_sha256: str,
    extraction_context_sha256: str | None = None,
    source_document: SourceDocumentArtifact,
) -> PublicationEvidenceFragment:
    """Project an extraction/grounding receipt into its only permitted fragment.

    Ingestion, package validation, and source replay share this projection. A caller
    therefore cannot change the reason, detail, or warnings on a receipt-linked
    non-estimable fragment and merely recompute the surrounding hashes.
    """

    frozen_extraction = NativePublicationExtraction.model_validate(
        extraction.model_dump(mode="json")
    )
    frozen_receipt = NativeGroundingReceipt.model_validate(
        grounding_receipt.model_dump(mode="json")
    )
    if frozen_receipt.extraction.model_dump(mode="json") != frozen_extraction.model_dump(
        mode="json"
    ):
        raise NativeGroundingError("fragment_projection_extraction_receipt_mismatch")
    if hash_canonical(frozen_receipt.source_document) != hash_canonical(source_document):
        raise NativeGroundingError("fragment_projection_source_document_mismatch")

    grounding_warnings = [
        f"grounding_receipt_sha256:{frozen_receipt.receipt_sha256}",
        *(f"grounding_receipt_issue:{issue}" for issue in frozen_receipt.issues),
        *(
            f"grounding_finding_status:{result.finding_path}:{result.status.value}"
            for result in frozen_receipt.finding_results
            if result.status.value != "exact"
        ),
    ]
    if not frozen_receipt.source_verified:
        return freeze_publication_evidence_fragment(
            question_id=question_id,
            publication_id=publication.publication_id,
            paper_id=publication.paper_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_fingerprint_sha256,
            extraction_context_sha256=extraction_context_sha256,
            source_document=source_document,
            grounding_receipt_sha256=frozen_receipt.receipt_sha256,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=NonEstimabilityReason.SOURCE_DOCUMENT_INCOMPLETE,
            non_estimability_detail=(
                "The immutable source artifact could not be resolved and verified; "
                f"see grounding receipt {frozen_receipt.receipt_sha256}."
            ),
            extractor_warnings=[*frozen_extraction.warnings, *grounding_warnings],
        )
    if (
        frozen_extraction.status is FragmentStatus.ESTIMABLE
        and not frozen_receipt.authorizes_estimable_fragment
    ):
        return freeze_publication_evidence_fragment(
            question_id=question_id,
            publication_id=publication.publication_id,
            paper_id=publication.paper_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_fingerprint_sha256,
            extraction_context_sha256=extraction_context_sha256,
            source_document=source_document,
            grounding_receipt_sha256=frozen_receipt.receipt_sha256,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=NonEstimabilityReason.UNGROUNDED_NUMERICAL_RESULT,
            non_estimability_detail=(
                "The extracted numerical result did not pass exact mechanical source "
                "grounding; see grounding receipt "
                f"{frozen_receipt.receipt_sha256}."
            ),
            extractor_warnings=[*frozen_extraction.warnings, *grounding_warnings],
        )
    return freeze_native_publication_extraction(
        payload=frozen_extraction,
        question_id=question_id,
        publication=publication,
        pipeline_fingerprint_sha256=pipeline_fingerprint_sha256,
        extraction_context_sha256=extraction_context_sha256,
        source_document=source_document,
        grounding_receipt_sha256=frozen_receipt.receipt_sha256,
    )


def validate_typed_corpus_grounding(
    *,
    corpus: TypedEvidenceCorpus,
    grounding_receipts: list[NativeGroundingReceipt],
) -> NativeCorpusGroundingValidation:
    """Join every fragment receipt pointer to one actual, self-validating receipt."""

    validated_corpus = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    receipts = [
        NativeGroundingReceipt.model_validate(receipt.model_dump(mode="json"))
        for receipt in grounding_receipts
    ]
    receipt_hashes = [receipt.receipt_sha256 for receipt in receipts]
    if len(receipt_hashes) != len(set(receipt_hashes)):
        raise NativeGroundingError("native_corpus_grounding_receipts_duplicate")
    by_hash = {receipt.receipt_sha256: receipt for receipt in receipts}
    referenced_hashes: list[str] = []
    links: list[NativeGroundingFragmentLink] = []
    unlinked_non_estimable = 0
    for fragment in validated_corpus.fragments:
        receipt_sha256 = fragment.grounding_receipt_sha256
        if receipt_sha256 is None:
            if fragment.status is FragmentStatus.ESTIMABLE:
                raise NativeGroundingError(
                    f"estimable_fragment_grounding_receipt_missing:{fragment.publication_id}"
                )
            unlinked_non_estimable += 1
            continue
        receipt = by_hash.get(receipt_sha256)
        if receipt is None:
            raise NativeGroundingError(
                f"fragment_grounding_receipt_not_found:{fragment.publication_id}"
            )
        if receipt_sha256 in referenced_hashes:
            raise NativeGroundingError(
                f"grounding_receipt_referenced_by_multiple_fragments:{receipt_sha256}"
            )
        referenced_hashes.append(receipt_sha256)
        if hash_canonical(receipt.source_document) != hash_canonical(fragment.source_document):
            raise NativeGroundingError(
                f"fragment_grounding_source_document_mismatch:{fragment.publication_id}"
            )
        if fragment.status is FragmentStatus.ESTIMABLE:
            if receipt.extraction_status is not FragmentStatus.ESTIMABLE:
                raise NativeGroundingError(
                    f"estimable_fragment_receipt_status_mismatch:{fragment.publication_id}"
                )
            if not receipt.authorizes_estimable_fragment:
                raise NativeGroundingError(
                    f"estimable_fragment_receipt_not_authorizing:{fragment.publication_id}"
                )
            link_kind = NativeGroundingLinkKind.ESTIMABLE_AUTHORIZED
        elif receipt.extraction_status is FragmentStatus.ESTIMABLE:
            if receipt.authorizes_estimable_fragment:
                raise NativeGroundingError(
                    f"non_estimable_fragment_receipt_authorizes:{fragment.publication_id}"
                )
            if fragment.non_estimability_reason not in {
                NonEstimabilityReason.UNGROUNDED_NUMERICAL_RESULT,
                NonEstimabilityReason.SOURCE_DOCUMENT_INCOMPLETE,
            }:
                raise NativeGroundingError(
                    f"failed_estimable_grounding_reason_mismatch:{fragment.publication_id}"
                )
            link_kind = NativeGroundingLinkKind.FAILED_ESTIMABLE_GROUNDING
        else:
            if receipt.authorizes_estimable_fragment:
                raise NativeGroundingError(
                    f"non_estimable_extraction_receipt_authorizes:{fragment.publication_id}"
                )
            link_kind = NativeGroundingLinkKind.EXPECTED_NON_ESTIMABLE_EXTRACTION
        projected_fragment = freeze_grounding_checked_publication_fragment(
            extraction=receipt.extraction,
            grounding_receipt=receipt,
            question_id=fragment.question_id,
            publication=fragment.publication,
            pipeline_fingerprint_sha256=fragment.pipeline_fingerprint_sha256,
            extraction_context_sha256=fragment.extraction_context_sha256,
            source_document=fragment.source_document,
        )
        if projected_fragment.model_dump(mode="json") != fragment.model_dump(mode="json"):
            raise NativeGroundingError(
                f"receipt_linked_fragment_projection_mismatch:{fragment.publication_id}"
            )
        links.append(
            NativeGroundingFragmentLink(
                publication_id=fragment.publication_id,
                fragment_status=fragment.status,
                grounding_receipt_sha256=receipt_sha256,
                receipt_extraction_status=receipt.extraction_status,
                receipt_authorizes_estimable=receipt.authorizes_estimable_fragment,
                link_kind=link_kind,
            )
        )
    unreferenced = sorted(set(receipt_hashes) - set(referenced_hashes))
    if unreferenced:
        raise NativeGroundingError(
            "native_corpus_grounding_receipts_unreferenced:" + ",".join(unreferenced)
        )
    links.sort(key=lambda link: link.publication_id)
    receipt_set_payload = [
        receipt.model_dump(mode="json")
        for receipt in sorted(receipts, key=lambda item: item.receipt_sha256)
    ]
    payload: dict[str, Any] = {
        "validation_version": "native-corpus-grounding-validation-v1",
        "corpus_sha256": validated_corpus.corpus_sha256,
        "grounding_receipt_set_sha256": hash_canonical(receipt_set_payload),
        "links": links,
        "unlinked_non_estimable_fragments": unlinked_non_estimable,
        "estimable_authorized_receipts": sum(
            link.link_kind is NativeGroundingLinkKind.ESTIMABLE_AUTHORIZED for link in links
        ),
        "expected_non_estimable_extraction_receipts": sum(
            link.link_kind is NativeGroundingLinkKind.EXPECTED_NON_ESTIMABLE_EXTRACTION
            for link in links
        ),
        "failed_estimable_grounding_receipts": sum(
            link.link_kind is NativeGroundingLinkKind.FAILED_ESTIMABLE_GROUNDING for link in links
        ),
    }
    return NativeCorpusGroundingValidation.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


def freeze_typed_evidence_grounding_package(
    *,
    corpus: TypedEvidenceCorpus,
    grounding_receipts: list[NativeGroundingReceipt],
    reviewer_reconciliation: ReviewerCohortReconciliationArtifact | None = None,
    source_manifest: NativeSourceManifest | None = None,
    corpus_cutoff: str | None = None,
    extraction_context: NativeExtractionExecutionContext | None = None,
) -> TypedEvidenceGroundingPackage:
    """Freeze the only native typed-corpus package accepted by the public verifier."""

    if (source_manifest is None) != (corpus_cutoff is None):
        raise NativeGroundingError(
            "source_manifest_and_corpus_cutoff_must_be_supplied_together"
        )
    if extraction_context is not None and source_manifest is None:
        raise NativeGroundingError(
            "native_extraction_context_requires_source_manifest_and_cutoff"
        )

    validation = validate_typed_corpus_grounding(
        corpus=corpus,
        grounding_receipts=grounding_receipts,
    )
    cohort_reconciliation = reconcile_native_cohorts(
        corpus=corpus,
        reviewer_artifact=reviewer_reconciliation,
    )
    validated_manifest = (
        NativeSourceManifest.model_validate(source_manifest.model_dump(mode="json"))
        if source_manifest is not None
        else None
    )
    package_version = (
        "typed-evidence-grounding-package-v4"
        if extraction_context is not None
        else (
            "typed-evidence-grounding-package-v3"
            if validated_manifest is not None
            else "typed-evidence-grounding-package-v2"
        )
    )
    payload: dict[str, Any] = {
        "package_version": (
            package_version
        ),
        "corpus": corpus,
        "grounding_receipts": grounding_receipts,
        "grounding_validation": validation,
        "cohort_reconciliation": cohort_reconciliation,
    }
    if validated_manifest is not None:
        payload.update(
            {
                "source_manifest": validated_manifest,
                "source_manifest_sha256": hash_canonical(validated_manifest),
                "corpus_cutoff": corpus_cutoff,
            }
        )
    if extraction_context is not None:
        validated_context = NativeExtractionExecutionContext.model_validate(
            extraction_context.model_dump(mode="json")
        )
        if corpus.corpus_version != "typed-evidence-corpus-v3":
            raise NativeGroundingError("native_extraction_context_requires_corpus_v3")
        if corpus.extraction_context_sha256 != validated_context.context_sha256:
            raise NativeGroundingError("native_extraction_context_corpus_link_mismatch")
        assert validated_manifest is not None
        assert corpus_cutoff is not None
        if validated_context.question_config.question_id != corpus.question_id:
            raise NativeGroundingError("native_extraction_context_question_mismatch")
        if validated_context.pipeline_fingerprint_sha256 != (
            corpus.pipeline_fingerprint_sha256
        ):
            raise NativeGroundingError("native_extraction_context_pipeline_mismatch")
        manifest_sha256 = hash_canonical(validated_manifest)
        if validated_context.source_manifest_content_sha256 != manifest_sha256:
            raise NativeGroundingError("native_extraction_context_source_manifest_mismatch")
        if validated_context.source_manifest_records != len(validated_manifest.records):
            raise NativeGroundingError("native_extraction_context_source_count_mismatch")
        if validated_context.corpus_cutoff != corpus_cutoff:
            raise NativeGroundingError("native_extraction_context_corpus_cutoff_mismatch")
        core_sha256 = hash_canonical(payload)
        assert cohort_reconciliation.reconciled_graph_sha256 is not None
        receipt_payload = {
            "receipt_version": "native-extraction-context-receipt-v1",
            "execution_context": validated_context,
            "corpus_sha256": corpus.corpus_sha256,
            "grounding_validation_sha256": validation.validation_sha256,
            "cohort_reconciliation_receipt_sha256": (
                cohort_reconciliation.receipt_sha256
            ),
            "reconciled_graph_sha256": (
                cohort_reconciliation.reconciled_graph_sha256
            ),
            "source_manifest_sha256": manifest_sha256,
            "corpus_cutoff": corpus_cutoff,
            "package_core_sha256": core_sha256,
        }
        payload["extraction_context_receipt"] = (
            NativeExtractionContextReceipt.model_validate(
                {
                    **receipt_payload,
                    "receipt_sha256": hash_canonical(receipt_payload),
                }
            )
        )
    return TypedEvidenceGroundingPackage.model_validate(
        {**payload, "package_sha256": hash_canonical(payload)}
    )


def reverify_typed_evidence_grounding_package(
    *,
    package: TypedEvidenceGroundingPackage,
    repository_root: Path,
) -> NativeGroundingReplayVerification:
    """Replay source grounding and estimable graph projection from current bytes."""

    validated = TypedEvidenceGroundingPackage.model_validate(package.model_dump(mode="json"))
    context_receipt = validated.extraction_context_receipt
    if context_receipt is not None:
        _reverify_native_extraction_execution_context(
            context=context_receipt.execution_context,
            repository_root=repository_root,
        )
    by_receipt: dict[str, NativeGroundingReceipt] = {}
    for frozen_receipt in validated.grounding_receipts:
        replayed = verify_native_publication_grounding(
            repository_root=repository_root,
            source_document=frozen_receipt.source_document,
            extraction=frozen_receipt.extraction,
        )
        if replayed.model_dump(mode="json") != frozen_receipt.model_dump(mode="json"):
            raise NativeGroundingError(
                f"native_grounding_receipt_replay_mismatch:{frozen_receipt.receipt_sha256}"
            )
        by_receipt[frozen_receipt.receipt_sha256] = frozen_receipt

    joined = validate_typed_corpus_grounding(
        corpus=validated.corpus,
        grounding_receipts=validated.grounding_receipts,
    )
    projected = 0
    replayed_fragments = []
    for fragment in validated.corpus.fragments:
        if fragment.grounding_receipt_sha256 is None:
            replayed_fragments.append(fragment)
            continue
        receipt = by_receipt[fragment.grounding_receipt_sha256]
        rebuilt = freeze_grounding_checked_publication_fragment(
            extraction=receipt.extraction,
            grounding_receipt=receipt,
            question_id=fragment.question_id,
            publication=fragment.publication,
            pipeline_fingerprint_sha256=fragment.pipeline_fingerprint_sha256,
            extraction_context_sha256=fragment.extraction_context_sha256,
            source_document=fragment.source_document,
        )
        if rebuilt.model_dump(mode="json") != fragment.model_dump(mode="json"):
            raise NativeGroundingError(
                f"receipt_linked_fragment_projection_mismatch:{fragment.publication_id}"
            )
        replayed_fragments.append(rebuilt)
        if fragment.status is FragmentStatus.ESTIMABLE:
            projected += 1
    reassembled = assemble_typed_evidence_corpus(replayed_fragments)
    if reassembled.model_dump(mode="json") != validated.corpus.model_dump(mode="json"):
        raise NativeGroundingError("typed_evidence_corpus_projection_mismatch")
    reconciliation_receipt_sha256: str | None = None
    reconciled_graph_sha256: str | None = None
    if validated.cohort_reconciliation is not None:
        try:
            reconciliation = reverify_native_cohort_reconciliation(
                corpus=reassembled,
                receipt=validated.cohort_reconciliation,
            )
        except NativeCohortReconciliationError as exc:
            raise NativeGroundingError(f"native_cohort_reconciliation_replay_failed:{exc}") from exc
        reconciliation_receipt_sha256 = reconciliation.receipt_sha256
        reconciled_graph_sha256 = reconciliation.reconciled_graph_sha256
    payload = {
        "replay_version": "native-grounding-replay-v4",
        "package_sha256": validated.package_sha256,
        "corpus_sha256": validated.corpus.corpus_sha256,
        "grounding_validation_sha256": joined.validation_sha256,
        "cohort_reconciliation_receipt_sha256": reconciliation_receipt_sha256,
        "reconciled_graph_sha256": reconciled_graph_sha256,
        "source_manifest_sha256": validated.source_manifest_sha256,
        "source_manifest_records": (
            len(validated.source_manifest.records)
            if validated.source_manifest is not None
            else 0
        ),
        "corpus_cutoff": validated.corpus_cutoff,
        "extraction_context_sha256": (
            context_receipt.execution_context.context_sha256
            if context_receipt is not None
            else None
        ),
        "extraction_context_receipt_sha256": (
            context_receipt.receipt_sha256 if context_receipt is not None else None
        ),
        "package_core_sha256": (
            context_receipt.package_core_sha256 if context_receipt is not None else None
        ),
        "question_config_sha256": (
            context_receipt.execution_context.question_config_sha256
            if context_receipt is not None
            else None
        ),
        "rendered_prompt_sha256s": sorted(
            {
                prompt.rendered_prompt_sha256
                for prompt in (
                    context_receipt.execution_context.rendered_prompts
                    if context_receipt is not None
                    else []
                )
            }
        ),
        "evaluation_schema_sha256s": sorted(
            {
                schema.schema_sha256
                for schema in (
                    context_receipt.execution_context.evaluation_schemas
                    if context_receipt is not None
                    else []
                )
            }
        ),
        "provider_execution_receipt_sha256s": sorted(
            {
                receipt.receipt_sha256
                for receipt in (
                    context_receipt.execution_context.provider_execution_receipts
                    if context_receipt is not None
                    else []
                )
            }
        ),
        "replayed_receipts": len(validated.grounding_receipts),
        "projected_estimable_fragments": projected,
    }
    return NativeGroundingReplayVerification.model_validate(
        {**payload, "replay_sha256": hash_canonical(payload)}
    )


__all__ = [
    "NativeCoordinateCheck",
    "NativeCorpusGroundingValidation",
    "NativeEvaluationSchemaArtifact",
    "NativeExtractionArtifactDigest",
    "NativeExtractionContextReceipt",
    "NativeExtractionExecutionContext",
    "NativeFindingGroundingResult",
    "NativeGroundingError",
    "NativeGroundingFragmentLink",
    "NativeGroundingLinkKind",
    "NativeGroundingReceipt",
    "NativeGroundingReplayVerification",
    "NativeProviderExecutionReceipt",
    "NativeRenderedPromptArtifact",
    "ResolvedNativeSource",
    "ResolvedSourceLine",
    "TypedEvidenceGroundingPackage",
    "freeze_grounding_checked_publication_fragment",
    "freeze_native_extraction_execution_context",
    "freeze_native_provider_execution_receipt",
    "freeze_typed_evidence_grounding_package",
    "resolve_native_source_document",
    "reverify_typed_evidence_grounding_package",
    "validate_typed_corpus_grounding",
    "verify_native_publication_grounding",
]
