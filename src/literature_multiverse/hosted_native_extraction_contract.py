"""Closed contract for a production hosted native-numeric extraction run.

This module is provider neutral and performs no network calls.  It defines the only
hosted-run shape that the native grounding bridge may consume.  Existing diagnostic,
fixture, and source-visible-target artifacts intentionally do not satisfy this
contract and must never be upgraded by a converter.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from literature_multiverse.config import QuestionConfig, config_sha256
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    native_publication_extraction_json_schema,
)
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
from literature_multiverse.schemas import assert_closed_object_schema

RUN_VERSION = "hosted-native-extraction-run-v1"
PROVIDER_IDENTITY_VERSION = "hosted-native-provider-identity-v1"
PROMPT_VERSION = "hosted-native-prompt-artifact-v1"
SCHEMA_VERSION = "hosted-native-schema-artifact-v1"
INTENT_VERSION = "hosted-native-call-intent-v1"
AUTHORIZATION_VERSION = "hosted-native-call-authorization-v1"
TERMINAL_VERSION = "hosted-native-call-terminal-v1"
CALL_VERSION = "hosted-native-call-v1"
NATIVE_PIPELINE_COMPONENT_ID = "native-extraction"
NATIVE_PIPELINE_COMPONENT_VERSION = "13"
HOSTED_NATIVE_ENTRYPOINT = "scripts/build_hosted_native_grounding_package.py"
HOSTED_NATIVE_EXECUTION_MODE = "hosted_exact_once"

REQUIRED_PIPELINE_PATHS = frozenset(
    {
        HOSTED_NATIVE_ENTRYPOINT,
        "src/literature_multiverse/hosted_native_extraction_contract.py",
        "src/literature_multiverse/hosted_native_grounding_bridge.py",
        "src/literature_multiverse/native_extraction.py",
        "src/literature_multiverse/native_grounding.py",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SECRET_VALUE_RE = re.compile(r"(?i)(?:sk-ant-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "anthropic_api_key",
        "authorization",
        "proxy_authorization",
        "x_api_key",
    }
)


class HostedNativeExtractionContractError(ValueError):
    """A hosted extraction run is incomplete, diagnostic, or not exactly bound."""


class _ExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
NonEmpty = Annotated[str, Field(min_length=1)]
Identifier = Annotated[str, Field(pattern=_IDENTIFIER_RE.pattern)]


def _sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _self_hash(model: _ExactModel, field_name: str, error_code: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field_name}))
    if getattr(model, field_name) != expected:
        raise ValueError(error_code)


def _canonical_relative_path(value: str, *, error_code: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(error_code)
    return value


def _assert_credential_free(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).casefold().replace("-", "_")
                if normalized in _SECRET_KEYS:
                    raise ValueError("hosted_native_secret_key_forbidden")
                pending.append(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            pending.extend(item)
        elif isinstance(item, str) and _SECRET_VALUE_RE.search(item):
            raise ValueError("hosted_native_secret_value_forbidden")


def _parse_json_text(value: str, *, error_code: str) -> JsonValue:
    try:
        parsed: JsonValue = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(error_code) from exc
    _assert_credential_free(parsed)
    return parsed


def resolve_json_pointer(document: JsonValue, pointer: str) -> JsonValue:
    """Resolve one RFC 6901 pointer without accepting negative or loose list indexes."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise HostedNativeExtractionContractError("hosted_native_json_pointer_invalid")
    current: JsonValue = document
    for raw_segment in pointer[1:].split("/"):
        segment = raw_segment.replace("~1", "/").replace("~0", "~")
        if re.search(r"~(?:[^01]|$)", raw_segment):
            raise HostedNativeExtractionContractError("hosted_native_json_pointer_escape_invalid")
        if isinstance(current, dict):
            if segment not in current:
                raise HostedNativeExtractionContractError("hosted_native_json_pointer_missing")
            current = current[segment]
        elif isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", segment):
                raise HostedNativeExtractionContractError(
                    "hosted_native_json_pointer_list_index_invalid"
                )
            index = int(segment)
            if index >= len(current):
                raise HostedNativeExtractionContractError("hosted_native_json_pointer_missing")
            current = current[index]
        else:
            raise HostedNativeExtractionContractError("hosted_native_json_pointer_traverses_scalar")
    return current


class HostedNativeProviderIdentityV1(_ExactModel):
    identity_version: Literal["hosted-native-provider-identity-v1"] = PROVIDER_IDENTITY_VERSION
    provider_id: Identifier
    model_id: NonEmpty
    model_revision: str | None = None
    api_base_url: Annotated[str, Field(pattern=r"^https://[^\s]+$")]
    runtime_id: Identifier
    runtime_version: NonEmpty
    runtime_source_paths: Annotated[list[str], Field(min_length=1)]
    sdk_name: Identifier
    sdk_version: NonEmpty
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    identity_sha256: Sha256

    @field_validator("runtime_metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if value != dict(sorted(value.items())):
            raise ValueError("hosted_native_runtime_metadata_not_canonical")
        _assert_credential_free(value)
        return value

    @field_validator("runtime_source_paths")
    @classmethod
    def validate_runtime_source_paths(cls, value: list[str]) -> list[str]:
        normalized = [
            _canonical_relative_path(item, error_code="hosted_native_runtime_source_path_invalid")
            for item in value
        ]
        if normalized != sorted(set(normalized)):
            raise ValueError("hosted_native_runtime_source_paths_not_canonical")
        return normalized

    @model_validator(mode="after")
    def validate_identity(self) -> HostedNativeProviderIdentityV1:
        _self_hash(self, "identity_sha256", "hosted_native_provider_identity_hash_mismatch")
        return self


class HostedNativePromptArtifactV1(_ExactModel):
    artifact_version: Literal["hosted-native-prompt-artifact-v1"] = PROMPT_VERSION
    prompt_id: Identifier
    renderer_id: Literal["hosted-native-extraction-v1"] = "hosted-native-extraction-v1"
    prompt_version: NonEmpty
    template_path: NonEmpty
    template_sha256: Sha256
    rendered_prompt: NonEmpty
    rendered_prompt_sha256: Sha256
    artifact_sha256: Sha256

    @field_validator("template_path")
    @classmethod
    def validate_template_path(cls, value: str) -> str:
        return _canonical_relative_path(
            value, error_code="hosted_native_prompt_template_path_invalid"
        )

    @model_validator(mode="after")
    def validate_prompt(self) -> HostedNativePromptArtifactV1:
        if self.rendered_prompt_sha256 != _sha256_utf8(self.rendered_prompt):
            raise ValueError("hosted_native_rendered_prompt_hash_mismatch")
        _self_hash(self, "artifact_sha256", "hosted_native_prompt_artifact_hash_mismatch")
        return self


class HostedNativeSchemaArtifactV1(_ExactModel):
    artifact_version: Literal["hosted-native-schema-artifact-v1"] = SCHEMA_VERSION
    schema_id: Identifier
    role: Literal["generation_constraint", "official_postvalidation"]
    schema_payload: dict[str, JsonValue]
    schema_sha256: Sha256
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_schema(self) -> HostedNativeSchemaArtifactV1:
        if self.schema_sha256 != hash_canonical(self.schema_payload):
            raise ValueError("hosted_native_schema_hash_mismatch")
        try:
            validator_for(self.schema_payload).check_schema(self.schema_payload)
            assert_closed_object_schema(self.schema_payload)
        except Exception as exc:
            raise ValueError("hosted_native_schema_invalid_or_open") from exc
        _self_hash(self, "artifact_sha256", "hosted_native_schema_artifact_hash_mismatch")
        return self


class HostedNativeCallIntentV1(_ExactModel):
    intent_version: Literal["hosted-native-call-intent-v1"] = INTENT_VERSION
    run_id: Identifier
    request_key: Identifier
    doc_id: NonEmpty
    publication_id: NonEmpty
    source_record_sha256: Sha256
    source_document_sha256: Sha256
    source_locator: NonEmpty
    source_content_scope: Literal["full_text"] = "full_text"
    source_strength_blockers: list[str] = Field(default_factory=list)
    question_config_sha256: Sha256
    source_manifest_sha256: Sha256
    pipeline_fingerprint_sha256: Sha256
    corpus_cutoff: NonEmpty
    provider_identity_sha256: Sha256
    model_id: NonEmpty
    prompt_id: Identifier
    rendered_prompt_sha256: Sha256
    generation_schema_id: Identifier
    generation_schema_sha256: Sha256
    official_postvalidation_schema_sha256: Sha256
    wire_request_utf8: NonEmpty
    wire_request_sha256: Sha256
    maximum_provider_attempts: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    intent_durable_before_transport: Literal[True] = True
    intent_sha256: Sha256

    @field_validator("source_strength_blockers")
    @classmethod
    def validate_source_blockers(cls, value: list[str]) -> list[str]:
        if value:
            raise ValueError("hosted_native_full_text_source_has_blockers")
        return value

    @model_validator(mode="after")
    def validate_intent(self) -> HostedNativeCallIntentV1:
        if self.wire_request_sha256 != _sha256_utf8(self.wire_request_utf8):
            raise ValueError("hosted_native_wire_request_hash_mismatch")
        _parse_json_text(
            self.wire_request_utf8,
            error_code="hosted_native_wire_request_json_invalid",
        )
        _self_hash(self, "intent_sha256", "hosted_native_intent_hash_mismatch")
        return self


class HostedNativeCallAuthorizationV1(_ExactModel):
    authorization_version: Literal["hosted-native-call-authorization-v1"] = AUTHORIZATION_VERSION
    run_id: Identifier
    request_key: Identifier
    intent_sha256: Sha256
    provider_identity_sha256: Sha256
    authorized_provider_attempts: Literal[1] = 1
    application_retries_authorized: Literal[0] = 0
    sdk_retries_authorized: Literal[0] = 0
    exact_request_only: Literal[True] = True
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> HostedNativeCallAuthorizationV1:
        _self_hash(
            self,
            "authorization_sha256",
            "hosted_native_authorization_hash_mismatch",
        )
        return self


class HostedNativeCallTerminalV1(_ExactModel):
    terminal_version: Literal["hosted-native-call-terminal-v1"] = TERMINAL_VERSION
    run_id: Identifier
    request_key: Identifier
    intent_sha256: Sha256
    authorization_sha256: Sha256
    terminal: Literal[True] = True
    outcome: Literal[
        "completed",
        "provider_failed",
        "ambiguous_attempt_poison",
    ]
    attempts_observed: Literal[1] = 1
    no_retry_after_terminal: Literal[True] = True
    provider_request_id: str | None = None
    observed_model_id: NonEmpty | None = None
    raw_response_utf8: str | None = None
    raw_response_sha256: Sha256 | None = None
    structured_output_json_pointer: str | None = None
    parsed_extraction: NativePublicationExtraction | None = None
    parsed_extraction_sha256: Sha256 | None = None
    official_postvalidation_passed: bool
    failure_code: Identifier | None = None
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> HostedNativeCallTerminalV1:
        if self.outcome == "completed":
            if (
                self.raw_response_utf8 is None
                or self.raw_response_sha256 is None
                or self.structured_output_json_pointer is None
                or self.parsed_extraction is None
                or self.parsed_extraction_sha256 is None
                or not self.official_postvalidation_passed
                or self.failure_code is not None
                or self.observed_model_id is None
            ):
                raise ValueError("hosted_native_completed_terminal_shape_invalid")
            if self.raw_response_sha256 != _sha256_utf8(self.raw_response_utf8):
                raise ValueError("hosted_native_raw_response_hash_mismatch")
            response = _parse_json_text(
                self.raw_response_utf8,
                error_code="hosted_native_raw_response_json_invalid",
            )
            try:
                pointed = resolve_json_pointer(response, self.structured_output_json_pointer)
            except HostedNativeExtractionContractError as exc:
                raise ValueError(str(exc)) from exc
            parsed = self.parsed_extraction.model_dump(mode="json")
            if pointed != parsed:
                raise ValueError("hosted_native_structured_output_pointer_mismatch")
            if self.parsed_extraction_sha256 != hash_canonical(parsed):
                raise ValueError("hosted_native_parsed_extraction_hash_mismatch")
        else:
            if (
                self.parsed_extraction is not None
                or self.parsed_extraction_sha256 is not None
                or self.structured_output_json_pointer is not None
                or self.official_postvalidation_passed
                or self.failure_code is None
            ):
                raise ValueError("hosted_native_failed_terminal_shape_invalid")
            if (self.raw_response_utf8 is None) != (self.raw_response_sha256 is None):
                raise ValueError("hosted_native_failed_response_hash_shape_invalid")
            if self.raw_response_utf8 is not None:
                assert self.raw_response_sha256 is not None
                if self.raw_response_sha256 != _sha256_utf8(self.raw_response_utf8):
                    raise ValueError("hosted_native_raw_response_hash_mismatch")
                _parse_json_text(
                    self.raw_response_utf8,
                    error_code="hosted_native_raw_response_json_invalid",
                )
        _self_hash(self, "terminal_sha256", "hosted_native_terminal_hash_mismatch")
        return self


class HostedNativeCallV1(_ExactModel):
    call_version: Literal["hosted-native-call-v1"] = CALL_VERSION
    intent: HostedNativeCallIntentV1
    authorization: HostedNativeCallAuthorizationV1
    terminal: HostedNativeCallTerminalV1
    call_sha256: Sha256

    @model_validator(mode="after")
    def validate_call(self) -> HostedNativeCallV1:
        if not (
            self.intent.run_id == self.authorization.run_id == self.terminal.run_id
            and self.intent.request_key
            == self.authorization.request_key
            == self.terminal.request_key
            and self.authorization.intent_sha256
            == self.terminal.intent_sha256
            == self.intent.intent_sha256
            and self.terminal.authorization_sha256 == self.authorization.authorization_sha256
            and self.authorization.provider_identity_sha256 == self.intent.provider_identity_sha256
        ):
            raise ValueError("hosted_native_call_chain_mismatch")
        _self_hash(self, "call_sha256", "hosted_native_call_hash_mismatch")
        return self


class HostedNativeExtractionRunV1(_ExactModel):
    run_version: Literal["hosted-native-extraction-run-v1"] = RUN_VERSION
    status: Literal["complete_exact_once_hosted_native_extraction_run"] = (
        "complete_exact_once_hosted_native_extraction_run"
    )
    run_id: Identifier
    run_purpose: Literal["production_native_numeric_extraction"] = (
        "production_native_numeric_extraction"
    )
    question_config: QuestionConfig
    question_config_sha256: Sha256
    source_manifest: NativeSourceManifest
    source_manifest_sha256: Sha256
    source_manifest_records: Annotated[int, Field(ge=1)]
    source_membership_sha256: Sha256
    corpus_cutoff: NonEmpty
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: Sha256
    prompts: Annotated[list[HostedNativePromptArtifactV1], Field(min_length=1)]
    prompt_membership_sha256: Sha256
    schemas: Annotated[list[HostedNativeSchemaArtifactV1], Field(min_length=2)]
    schema_membership_sha256: Sha256
    provider_identity: HostedNativeProviderIdentityV1
    provider_identity_sha256: Sha256
    calls: Annotated[list[HostedNativeCallV1], Field(min_length=1)]
    call_membership_sha256: Sha256
    completed_extraction_count: Annotated[int, Field(ge=0)]
    failed_or_ambiguous_count: Annotated[int, Field(ge=0)]
    all_manifest_records_attempted_once: Literal[True] = True
    all_calls_terminal: Literal[True] = True
    provider_calls_made: Literal[True] = True
    source_content_scope: Literal["full_text"] = "full_text"
    source_transmission_authorized: Literal[True] = True
    diagnostic_or_fixture: Literal[False] = False
    code_owned_predictions: Literal[False] = False
    reference_fields_opened: Literal[False] = False
    official_test_labels_opened: Literal[False] = False
    v4_source_provenance_bridge_eligible: Literal[True] = True
    exact_source_grounding_still_required: Literal[True] = True
    scientific_claim_truth_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    run_sha256: Sha256

    @field_validator("prompts")
    @classmethod
    def validate_prompts(
        cls, value: list[HostedNativePromptArtifactV1]
    ) -> list[HostedNativePromptArtifactV1]:
        ids = [item.prompt_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("hosted_native_prompts_not_canonical")
        return value

    @field_validator("schemas")
    @classmethod
    def validate_schemas(
        cls, value: list[HostedNativeSchemaArtifactV1]
    ) -> list[HostedNativeSchemaArtifactV1]:
        ids = [item.schema_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("hosted_native_schemas_not_canonical")
        return value

    @field_validator("calls")
    @classmethod
    def validate_calls(cls, value: list[HostedNativeCallV1]) -> list[HostedNativeCallV1]:
        keys = [(item.intent.doc_id, item.intent.request_key) for item in value]
        if keys != sorted(set(keys)):
            raise ValueError("hosted_native_calls_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_run(self) -> HostedNativeExtractionRunV1:
        if self.question_config.status != "locked":
            raise ValueError("hosted_native_question_config_not_locked")
        if self.question_config_sha256 != config_sha256(self.question_config):
            raise ValueError("hosted_native_question_config_hash_mismatch")
        if self.source_manifest.question_id != self.question_config.question_id:
            raise ValueError("hosted_native_manifest_question_mismatch")
        source_records = [item.model_dump(mode="json") for item in self.source_manifest.records]
        if (
            self.source_manifest_sha256 != hash_canonical(self.source_manifest)
            or self.source_manifest_records != len(source_records)
            or self.source_membership_sha256 != hash_canonical(source_records)
        ):
            raise ValueError("hosted_native_source_membership_mismatch")
        native_components = [
            component
            for component in self.pipeline_fingerprint.components
            if component.component_id == NATIVE_PIPELINE_COMPONENT_ID
        ]
        all_pipeline_paths = {
            file.path
            for component in self.pipeline_fingerprint.components
            for file in component.files
        }
        native_component = native_components[0] if len(native_components) == 1 else None
        native_paths = (
            {file.path for file in native_component.files}
            if native_component is not None
            else set()
        )
        native_entry_points = (
            native_component.settings.get("native_extraction_entry_points")
            if native_component is not None
            else None
        )
        if (
            self.pipeline_fingerprint_sha256 != self.pipeline_fingerprint.pipeline_sha256
            or not (
                REQUIRED_PIPELINE_PATHS | set(self.provider_identity.runtime_source_paths)
            ).issubset(all_pipeline_paths)
            or native_component is None
            or native_component.component_version != NATIVE_PIPELINE_COMPONENT_VERSION
            or not REQUIRED_PIPELINE_PATHS.issubset(native_paths)
            or not isinstance(native_entry_points, list)
            or HOSTED_NATIVE_ENTRYPOINT not in native_entry_points
            or native_component.settings.get("hosted_native_extraction_run_contract") != RUN_VERSION
            or native_component.settings.get("hosted_native_execution_mode")
            != HOSTED_NATIVE_EXECUTION_MODE
        ):
            raise ValueError("hosted_native_pipeline_fingerprint_incomplete")
        if self.prompt_membership_sha256 != hash_canonical(
            [item.artifact_sha256 for item in self.prompts]
        ):
            raise ValueError("hosted_native_prompt_membership_mismatch")
        if self.schema_membership_sha256 != hash_canonical(
            [item.artifact_sha256 for item in self.schemas]
        ):
            raise ValueError("hosted_native_schema_membership_mismatch")
        official = [item for item in self.schemas if item.role == "official_postvalidation"]
        generation = [item for item in self.schemas if item.role == "generation_constraint"]
        expected_official = native_publication_extraction_json_schema()
        if (
            len(official) != 1
            or not generation
            or official[0].schema_payload != expected_official
            or official[0].schema_sha256 != hash_canonical(expected_official)
        ):
            raise ValueError("hosted_native_official_schema_mismatch")
        if self.provider_identity_sha256 != self.provider_identity.identity_sha256:
            raise ValueError("hosted_native_provider_identity_alias_mismatch")
        records_by_doc = {item.doc_id: item for item in self.source_manifest.records}
        prompts_by_id = {item.prompt_id: item for item in self.prompts}
        schemas_by_id = {item.schema_id: item for item in generation}
        if {item.intent.doc_id for item in self.calls} != set(records_by_doc):
            raise ValueError("hosted_native_call_manifest_membership_mismatch")
        official_sha256 = official[0].schema_sha256
        for call in self.calls:
            intent = call.intent
            record = records_by_doc[intent.doc_id]
            prompt = prompts_by_id.get(intent.prompt_id)
            schema = schemas_by_id.get(intent.generation_schema_id)
            if (
                intent.run_id != self.run_id
                or intent.publication_id != record.publication.publication_id
                or intent.source_record_sha256 != hash_canonical(record)
                or intent.source_document_sha256 != record.source_document.sha256
                or intent.source_locator != record.source_document.source_locator
                or intent.question_config_sha256 != self.question_config_sha256
                or intent.source_manifest_sha256 != self.source_manifest_sha256
                or intent.pipeline_fingerprint_sha256 != self.pipeline_fingerprint_sha256
                or intent.corpus_cutoff != self.corpus_cutoff
                or intent.provider_identity_sha256 != self.provider_identity_sha256
                or intent.model_id != self.provider_identity.model_id
                or prompt is None
                or intent.rendered_prompt_sha256 != prompt.rendered_prompt_sha256
                or schema is None
                or intent.generation_schema_sha256 != schema.schema_sha256
                or intent.official_postvalidation_schema_sha256 != official_sha256
                or call.terminal.observed_model_id
                != (
                    self.provider_identity.model_id
                    if call.terminal.outcome == "completed"
                    else call.terminal.observed_model_id
                )
            ):
                raise ValueError("hosted_native_call_lineage_mismatch")
        if self.call_membership_sha256 != hash_canonical([item.call_sha256 for item in self.calls]):
            raise ValueError("hosted_native_call_membership_hash_mismatch")
        completed = sum(item.terminal.outcome == "completed" for item in self.calls)
        if (
            self.completed_extraction_count != completed
            or self.failed_or_ambiguous_count != len(self.calls) - completed
        ):
            raise ValueError("hosted_native_terminal_counts_mismatch")
        _self_hash(self, "run_sha256", "hosted_native_run_hash_mismatch")
        return self


def freeze_hosted_native_provider_identity_v1(
    *,
    provider_id: str,
    model_id: str,
    api_base_url: str,
    runtime_id: str,
    runtime_version: str,
    runtime_source_paths: list[str],
    sdk_name: str,
    sdk_version: str,
    model_revision: str | None = None,
    runtime_metadata: dict[str, JsonValue] | None = None,
) -> HostedNativeProviderIdentityV1:
    payload = {
        "identity_version": PROVIDER_IDENTITY_VERSION,
        "provider_id": provider_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "api_base_url": api_base_url,
        "runtime_id": runtime_id,
        "runtime_version": runtime_version,
        "runtime_source_paths": sorted(runtime_source_paths),
        "sdk_name": sdk_name,
        "sdk_version": sdk_version,
        "runtime_metadata": dict(sorted((runtime_metadata or {}).items())),
    }
    return HostedNativeProviderIdentityV1.model_validate(
        {**payload, "identity_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_prompt_artifact_v1(
    *,
    prompt_id: str,
    prompt_version: str,
    template_path: str,
    template_sha256: str,
    rendered_prompt: str,
) -> HostedNativePromptArtifactV1:
    payload = {
        "artifact_version": PROMPT_VERSION,
        "prompt_id": prompt_id,
        "renderer_id": "hosted-native-extraction-v1",
        "prompt_version": prompt_version,
        "template_path": template_path,
        "template_sha256": template_sha256,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": _sha256_utf8(rendered_prompt),
    }
    return HostedNativePromptArtifactV1.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_schema_artifact_v1(
    *,
    schema_id: str,
    role: Literal["generation_constraint", "official_postvalidation"],
    schema_payload: dict[str, JsonValue],
) -> HostedNativeSchemaArtifactV1:
    payload = {
        "artifact_version": SCHEMA_VERSION,
        "schema_id": schema_id,
        "role": role,
        "schema_payload": schema_payload,
        "schema_sha256": hash_canonical(schema_payload),
    }
    return HostedNativeSchemaArtifactV1.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_call_intent_v1(
    *,
    run_id: str,
    request_key: str,
    question_config: QuestionConfig,
    source_manifest: NativeSourceManifest,
    source_record_index: int,
    pipeline_fingerprint: PipelineFingerprint,
    corpus_cutoff: str,
    provider_identity: HostedNativeProviderIdentityV1,
    prompt: HostedNativePromptArtifactV1,
    generation_schema: HostedNativeSchemaArtifactV1,
    official_schema: HostedNativeSchemaArtifactV1,
    wire_request_utf8: str,
) -> HostedNativeCallIntentV1:
    if source_record_index < 0 or source_record_index >= len(source_manifest.records):
        raise HostedNativeExtractionContractError("hosted_native_source_record_index_invalid")
    if generation_schema.role != "generation_constraint":
        raise HostedNativeExtractionContractError("hosted_native_generation_schema_role_invalid")
    if official_schema.role != "official_postvalidation":
        raise HostedNativeExtractionContractError("hosted_native_official_schema_role_invalid")
    record = source_manifest.records[source_record_index]
    payload = {
        "intent_version": INTENT_VERSION,
        "run_id": run_id,
        "request_key": request_key,
        "doc_id": record.doc_id,
        "publication_id": record.publication.publication_id,
        "source_record_sha256": hash_canonical(record),
        "source_document_sha256": record.source_document.sha256,
        "source_locator": record.source_document.source_locator,
        "source_content_scope": "full_text",
        "source_strength_blockers": [],
        "question_config_sha256": config_sha256(question_config),
        "source_manifest_sha256": hash_canonical(source_manifest),
        "pipeline_fingerprint_sha256": pipeline_fingerprint.pipeline_sha256,
        "corpus_cutoff": corpus_cutoff,
        "provider_identity_sha256": provider_identity.identity_sha256,
        "model_id": provider_identity.model_id,
        "prompt_id": prompt.prompt_id,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
        "generation_schema_id": generation_schema.schema_id,
        "generation_schema_sha256": generation_schema.schema_sha256,
        "official_postvalidation_schema_sha256": official_schema.schema_sha256,
        "wire_request_utf8": wire_request_utf8,
        "wire_request_sha256": _sha256_utf8(wire_request_utf8),
        "maximum_provider_attempts": 1,
        "application_retries": 0,
        "sdk_retries": 0,
        "intent_durable_before_transport": True,
    }
    return HostedNativeCallIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_call_authorization_v1(
    *,
    intent: HostedNativeCallIntentV1,
    provider_identity: HostedNativeProviderIdentityV1,
) -> HostedNativeCallAuthorizationV1:
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "run_id": intent.run_id,
        "request_key": intent.request_key,
        "intent_sha256": intent.intent_sha256,
        "provider_identity_sha256": provider_identity.identity_sha256,
        "authorized_provider_attempts": 1,
        "application_retries_authorized": 0,
        "sdk_retries_authorized": 0,
        "exact_request_only": True,
    }
    return HostedNativeCallAuthorizationV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_completed_terminal_v1(
    *,
    intent: HostedNativeCallIntentV1,
    authorization: HostedNativeCallAuthorizationV1,
    observed_model_id: str,
    raw_response_utf8: str,
    structured_output_json_pointer: str,
    parsed_extraction: NativePublicationExtraction,
    provider_request_id: str | None = None,
) -> HostedNativeCallTerminalV1:
    parsed_payload = parsed_extraction.model_dump(mode="json")
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "run_id": intent.run_id,
        "request_key": intent.request_key,
        "intent_sha256": intent.intent_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "terminal": True,
        "outcome": "completed",
        "attempts_observed": 1,
        "no_retry_after_terminal": True,
        "provider_request_id": provider_request_id,
        "observed_model_id": observed_model_id,
        "raw_response_utf8": raw_response_utf8,
        "raw_response_sha256": _sha256_utf8(raw_response_utf8),
        "structured_output_json_pointer": structured_output_json_pointer,
        "parsed_extraction": parsed_extraction,
        "parsed_extraction_sha256": hash_canonical(parsed_payload),
        "official_postvalidation_passed": True,
        "failure_code": None,
    }
    return HostedNativeCallTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_failed_terminal_v1(
    *,
    intent: HostedNativeCallIntentV1,
    authorization: HostedNativeCallAuthorizationV1,
    outcome: Literal["provider_failed", "ambiguous_attempt_poison"],
    failure_code: str,
    provider_request_id: str | None = None,
    observed_model_id: str | None = None,
    raw_response_utf8: str | None = None,
) -> HostedNativeCallTerminalV1:
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "run_id": intent.run_id,
        "request_key": intent.request_key,
        "intent_sha256": intent.intent_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "terminal": True,
        "outcome": outcome,
        "attempts_observed": 1,
        "no_retry_after_terminal": True,
        "provider_request_id": provider_request_id,
        "observed_model_id": observed_model_id,
        "raw_response_utf8": raw_response_utf8,
        "raw_response_sha256": (
            _sha256_utf8(raw_response_utf8) if raw_response_utf8 is not None else None
        ),
        "structured_output_json_pointer": None,
        "parsed_extraction": None,
        "parsed_extraction_sha256": None,
        "official_postvalidation_passed": False,
        "failure_code": failure_code,
    }
    return HostedNativeCallTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def freeze_hosted_native_call_v1(
    *,
    intent: HostedNativeCallIntentV1,
    authorization: HostedNativeCallAuthorizationV1,
    terminal: HostedNativeCallTerminalV1,
) -> HostedNativeCallV1:
    payload = {
        "call_version": CALL_VERSION,
        "intent": intent,
        "authorization": authorization,
        "terminal": terminal,
    }
    return HostedNativeCallV1.model_validate({**payload, "call_sha256": hash_canonical(payload)})


def freeze_hosted_native_extraction_run_v1(
    *,
    run_id: str,
    question_config: QuestionConfig,
    source_manifest: NativeSourceManifest,
    corpus_cutoff: str,
    pipeline_fingerprint: PipelineFingerprint,
    prompts: list[HostedNativePromptArtifactV1],
    schemas: list[HostedNativeSchemaArtifactV1],
    provider_identity: HostedNativeProviderIdentityV1,
    calls: list[HostedNativeCallV1],
) -> HostedNativeExtractionRunV1:
    ordered_prompts = sorted(prompts, key=lambda item: item.prompt_id)
    ordered_schemas = sorted(schemas, key=lambda item: item.schema_id)
    ordered_calls = sorted(calls, key=lambda item: (item.intent.doc_id, item.intent.request_key))
    records = [item.model_dump(mode="json") for item in source_manifest.records]
    completed = sum(item.terminal.outcome == "completed" for item in ordered_calls)
    payload = {
        "run_version": RUN_VERSION,
        "status": "complete_exact_once_hosted_native_extraction_run",
        "run_id": run_id,
        "run_purpose": "production_native_numeric_extraction",
        "question_config": question_config,
        "question_config_sha256": config_sha256(question_config),
        "source_manifest": source_manifest,
        "source_manifest_sha256": hash_canonical(source_manifest),
        "source_manifest_records": len(records),
        "source_membership_sha256": hash_canonical(records),
        "corpus_cutoff": corpus_cutoff,
        "pipeline_fingerprint": pipeline_fingerprint,
        "pipeline_fingerprint_sha256": pipeline_fingerprint.pipeline_sha256,
        "prompts": ordered_prompts,
        "prompt_membership_sha256": hash_canonical(
            [item.artifact_sha256 for item in ordered_prompts]
        ),
        "schemas": ordered_schemas,
        "schema_membership_sha256": hash_canonical(
            [item.artifact_sha256 for item in ordered_schemas]
        ),
        "provider_identity": provider_identity,
        "provider_identity_sha256": provider_identity.identity_sha256,
        "calls": ordered_calls,
        "call_membership_sha256": hash_canonical([item.call_sha256 for item in ordered_calls]),
        "completed_extraction_count": completed,
        "failed_or_ambiguous_count": len(ordered_calls) - completed,
        "all_manifest_records_attempted_once": True,
        "all_calls_terminal": True,
        "provider_calls_made": True,
        "source_content_scope": "full_text",
        "source_transmission_authorized": True,
        "diagnostic_or_fixture": False,
        "code_owned_predictions": False,
        "reference_fields_opened": False,
        "official_test_labels_opened": False,
        "v4_source_provenance_bridge_eligible": True,
        "exact_source_grounding_still_required": True,
        "scientific_claim_truth_authority": False,
        "claim_release_authority": False,
    }
    return HostedNativeExtractionRunV1.model_validate(
        {**payload, "run_sha256": hash_canonical(payload)}
    )


__all__ = [
    "AUTHORIZATION_VERSION",
    "CALL_VERSION",
    "INTENT_VERSION",
    "PROMPT_VERSION",
    "PROVIDER_IDENTITY_VERSION",
    "REQUIRED_PIPELINE_PATHS",
    "RUN_VERSION",
    "SCHEMA_VERSION",
    "TERMINAL_VERSION",
    "HostedNativeCallAuthorizationV1",
    "HostedNativeCallIntentV1",
    "HostedNativeCallTerminalV1",
    "HostedNativeCallV1",
    "HostedNativeExtractionContractError",
    "HostedNativeExtractionRunV1",
    "HostedNativePromptArtifactV1",
    "HostedNativeProviderIdentityV1",
    "HostedNativeSchemaArtifactV1",
    "freeze_hosted_native_call_authorization_v1",
    "freeze_hosted_native_call_intent_v1",
    "freeze_hosted_native_call_v1",
    "freeze_hosted_native_completed_terminal_v1",
    "freeze_hosted_native_extraction_run_v1",
    "freeze_hosted_native_failed_terminal_v1",
    "freeze_hosted_native_prompt_artifact_v1",
    "freeze_hosted_native_provider_identity_v1",
    "freeze_hosted_native_schema_artifact_v1",
    "resolve_json_pointer",
]
