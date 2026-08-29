"""Exact-once hosted pilot for release-grade native binary extraction.

The pilot is deliberately small: two public PMC full-text records selected from source
text only because each contains an explicit two-arm binary result for one prespecified
H. pylori question.  Selection is purposive and the endpoint is yield only.  The run
therefore has no extraction-accuracy, representativeness, synthesis-conclusion, or
claim-release authority.

No function in this module reads a credential file. Live adapters require an explicit
in-memory API key, never hash or persist it, disable SDK/application retries, and
persist a durable intent before every network attempt. Preparing a workspace is
offline.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, JsonValue, StrictInt, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256,
    ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION,
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
    render_anthropic_prompt_json_model_system,
)
from literature_multiverse.config import QuestionConfig, config_sha256, load_question_config
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.hosted_native_extraction_contract import (
    HostedNativeCallAuthorizationV1,
    HostedNativeCallIntentV1,
    HostedNativeCallTerminalV1,
    HostedNativeCallV1,
    HostedNativeExtractionRunV1,
    HostedNativePromptArtifactV1,
    HostedNativeProviderIdentityV1,
    HostedNativeSchemaArtifactV1,
    freeze_hosted_native_call_authorization_v1,
    freeze_hosted_native_call_intent_v1,
    freeze_hosted_native_call_v1,
    freeze_hosted_native_completed_terminal_v1,
    freeze_hosted_native_extraction_run_v1,
    freeze_hosted_native_failed_terminal_v1,
    freeze_hosted_native_prompt_artifact_v1,
    freeze_hosted_native_provider_identity_v1,
    freeze_hosted_native_schema_artifact_v1,
)
from literature_multiverse.hosted_native_numeric_canary_v4 import (
    CANARY_HARD_CEILING_USD_MICROS,
    HostedNativeNumericCanarySuccessBindingV4,
    HostedNativeNumericCanaryV4Error,
    load_successful_hosted_native_numeric_canary_v4,
    require_hosted_native_numeric_canary_binding_v4,
)
from literature_multiverse.hosted_native_numeric_canary_v4 import (
    DEFAULT_WORKSPACE as DEFAULT_CANARY_WORKSPACE,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    NativeGroundingReceipt,
    resolve_native_source_document,
    verify_native_publication_grounding,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact

RUNTIME_VERSION = "hosted-native-numeric-pilot-runtime-v5"
CONFIG_VERSION = "hosted-native-numeric-yield-pilot-config-v5"
PLAN_VERSION = "hosted-native-numeric-pilot-plan-v5"
RESERVATION_VERSION = "hosted-native-numeric-pilot-reservation-v5"
PRIOR_ACCOUNTING_VERSION = "hosted-native-prior-accounting-receipt-v5"
COUNT_INTENT_VERSION = "hosted-native-numeric-pilot-count-intent-v5"
COUNT_RECEIPT_VERSION = "hosted-native-numeric-pilot-count-receipt-v5"
COUNT_INCIDENT_VERSION = "hosted-native-numeric-pilot-count-incident-v5"
COUNT_CERTIFICATE_VERSION = "hosted-native-numeric-pilot-count-certificate-v5"
GENERATION_AUTHORIZATION_VERSION = "hosted-native-numeric-pilot-generation-authorization-v5"
PROVIDER_OBSERVATION_VERSION = "hosted-native-numeric-pilot-provider-observation-v5"
INCIDENT_VERSION = "hosted-native-numeric-pilot-incident-v5"
VALIDATION_VERSION = "hosted-native-numeric-pilot-validation-v5"
TERMINAL_VERSION = "hosted-native-numeric-pilot-terminal-v5"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/hosted-native-numeric-yield-pilot-v5.json")
DEFAULT_WORKSPACE = Path("data/cache/hosted-native-numeric-yield-pilot-v5-live")
RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/hosted_native_numeric_pilot_v1.py")
SCRIPT_SOURCE_PATH = Path("scripts/run_hosted_native_numeric_pilot_v1.py")
PROMPT_VERSION = "hosted-native-numeric-yield-pilot-prompt-v5"

MODEL = "claude-fable-5"
MODEL_REVISION = "claude-fable-5"
API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
SDK_VERSION = "0.120.2"
EFFORT = "high"
SERVICE_TIER = "standard_only"
TRANSPORT_MODE = "prompt_json_schema"
PROVIDER_GRAMMAR_ENABLED = False
TIMEOUT_SECONDS = 600.0
MAX_OUTPUT_TOKENS = 10_240
TOKEN_COUNT_HEADROOM = 1_024
MAX_CERTIFIED_INPUT_TOKENS = 68_800
INPUT_RATE_USD_PER_MTOK = Decimal("10")
OUTPUT_RATE_USD_PER_MTOK = Decimal("50")
MAXIMUM_PROVIDER_CALLS = 2
PER_CALL_HARD_LIABILITY_USD_MICROS = 1_200_000
NEW_LIABILITY_HARD_CEILING_USD_MICROS = 2_400_000
CANARY_LIABILITY_ALLOCATION_USD_MICROS = 600_000
COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS = 3_000_000
PROJECT_LIABILITY_HARD_STOP_USD_MICROS = 100_000_000
OFFLINE_FRAMING_TOKEN_ALLOWANCE = 2_048
DELIVERED_SCHEMA_MAX_UTF8_BYTES = 8_192
DELIVERED_SCHEMA_MAX_NODE_COUNT = 512
DELIVERED_SCHEMA_MAX_DEPTH = 20
DELIVERED_SCHEMA_MAX_OBJECT_PROPERTIES = 96
DELIVERED_SCHEMA_MAX_ARRAY_SCHEMAS = 16
DELIVERED_SCHEMA_EXPECTED_ARRAY_SCHEMAS = 12
DELIVERED_SCHEMA_EXPECTED_CONST_KEYWORDS = 59
DELIVERED_SCHEMA_EXPECTED_NODE_COUNT = 401
SYSTEM_PROMPT = (
    "You are a bounded scientific evidence extractor. Return exactly one JSON object "
    "that satisfies the supplied schema. Copy numerical evidence only from the frozen "
    "public full text. Never infer, repair, or fabricate a value."
)

PIPELINE_REQUIRED_PATHS = (
    "scripts/build_hosted_native_grounding_package.py",
    "src/literature_multiverse/hosted_native_extraction_contract.py",
    "src/literature_multiverse/hosted_native_grounding_bridge.py",
    "src/literature_multiverse/native_extraction.py",
    "src/literature_multiverse/native_grounding.py",
)
RUNTIME_PIPELINE_PATHS = (
    "scripts/run_hosted_native_numeric_canary_v4.py",
    "scripts/run_hosted_native_numeric_pilot_v1.py",
    "src/literature_multiverse/anthropic_bounded_generation.py",
    "src/literature_multiverse/hosted_native_numeric_canary_v4.py",
    "src/literature_multiverse/hosted_native_numeric_pilot_v1.py",
    "src/literature_multiverse/providers.py",
)
SCIENTIFIC_CONTRACT_PATHS = (
    "configs/benchmarks/hosted-native-numeric-yield-pilot-v5.json",
    "configs/questions/hosted-native-numeric-yield-pilot-v4.yaml",
    "prompts/hosted_native_numeric_yield_pilot_v5.txt",
)

_SECRET_VALUE_RE = re.compile(r"(?i)(?:sk-ant-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "anthropic_api_key",
        "proxy_authorization",
        "x_api_key",
    }
)
_COUNT_PAIR_RE = re.compile(r"\(\s*(?P<events>[0-9]+)\s*/\s*(?P<total>[0-9]+)\s*[,)]")
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_ALLOWED_NON_TEXT_CONTENT_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


class HostedNativeNumericPilotError(ValueError):
    """The frozen pilot, at-most-once state machine, or source check failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
NonEmpty = Annotated[str, Field(min_length=1)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


def _sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field}))
    if getattr(model, field) != expected:
        raise ValueError(code)


def _assert_secret_free(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold().replace("-", "_") in _SECRET_KEYS:
                    raise HostedNativeNumericPilotError("hosted_numeric_secret_key_forbidden")
                pending.append(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            pending.extend(item)
        elif isinstance(item, str) and _SECRET_VALUE_RE.search(item):
            raise HostedNativeNumericPilotError("hosted_numeric_secret_value_forbidden")


def _load_json_object(path: Path, *, code: str) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False) if path.exists() else None
    if (
        metadata is None
        or path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise HostedNativeNumericPilotError(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HostedNativeNumericPilotError(code) from exc
    if not isinstance(value, dict):
        raise HostedNativeNumericPilotError(code)
    _assert_secret_free(value)
    return value


def _safe_repository_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix().startswith("./"):
        raise HostedNativeNumericPilotError("hosted_numeric_repository_path_invalid")
    resolved_root = root.resolve(strict=True)
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise HostedNativeNumericPilotError("hosted_numeric_repository_root_invalid")
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise HostedNativeNumericPilotError("hosted_numeric_repository_symlink_forbidden")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_repository_file_missing") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise HostedNativeNumericPilotError("hosted_numeric_repository_file_invalid")
    return resolved


class PilotSelectionContractV1(_Frozen):
    selection_basis: Literal["public_source_text_only_without_reference_or_prediction_fields"]
    purposive_high_yield: Literal[True]
    source_transmission_authorized_by_operator: Literal[True]
    labels_opened: Literal[False]
    private_predictions_opened: Literal[False]
    selection_outcome_values_persisted: Literal[False]
    yield_only_endpoint: NonEmpty
    accuracy_authority: Literal[False]
    representativeness_authority: Literal[False]
    synthesis_conclusion_authority: Literal[False]
    claim_release_authority: Literal[False]


class PilotProviderConfigV1(_Frozen):
    provider_id: Literal["anthropic_first_party_api"]
    model_id: Literal["claude-fable-5"]
    model_revision: Literal["claude-fable-5"]
    api_base_url: Literal["https://api.anthropic.com"]
    anthropic_api_version: Literal["2023-06-01"]
    sdk_name: Literal["anthropic-python"]
    sdk_version: Literal["0.120.2"]
    effort: Literal["high"]
    service_tier: Literal["standard_only"]
    transport_mode: Literal["prompt_json_schema"]
    provider_grammar_enabled: Literal[False]
    timeout_seconds: Literal[600]
    max_output_tokens_per_call: Literal[10240]
    maximum_certified_input_tokens_per_call: Literal[68800]
    token_count_headroom_per_call: Literal[1024]
    input_rate_usd_per_million_tokens: Decimal
    output_rate_usd_per_million_tokens: Decimal
    maximum_provider_calls: Literal[2]
    application_retries_per_call: Literal[0]
    sdk_retries_per_call: Literal[0]
    maximum_new_liability_usd_micros: Literal[2400000]
    canary_liability_allocation_usd_micros: Literal[600000]
    combined_v4_phase_hard_ceiling_usd_micros: Literal[3000000]
    project_liability_hard_stop_usd_micros: Literal[100000000]

    @model_validator(mode="after")
    def validate_cost_identity(self) -> PilotProviderConfigV1:
        per_call = self.maximum_certified_input_tokens_per_call * int(
            self.input_rate_usd_per_million_tokens
        ) + self.max_output_tokens_per_call * int(self.output_rate_usd_per_million_tokens)
        if (
            self.input_rate_usd_per_million_tokens != INPUT_RATE_USD_PER_MTOK
            or self.output_rate_usd_per_million_tokens != OUTPUT_RATE_USD_PER_MTOK
            or per_call != PER_CALL_HARD_LIABILITY_USD_MICROS
            or per_call * self.maximum_provider_calls != self.maximum_new_liability_usd_micros
            or self.canary_liability_allocation_usd_micros + self.maximum_new_liability_usd_micros
            != self.combined_v4_phase_hard_ceiling_usd_micros
            or self.transport_mode != TRANSPORT_MODE
            or self.provider_grammar_enabled is not PROVIDER_GRAMMAR_ENABLED
        ):
            raise ValueError("hosted_numeric_provider_cost_identity_mismatch")
        return self


class PilotPriorAccountingConfigV1(_Frozen):
    accounting_basis: Literal[
        "root_reconciled_metadata_only_2026-08-29_plus_v3_and_v4_certified_upper_bounds"
    ]
    reported_prior_spend_usd_micros: Literal[38616150]
    unknown_prior_liability_usd_micros: Literal[17482919]
    v3_certified_unresolved_upper_bound_usd_micros: Literal[1932800]
    v4_certified_liability_upper_bound_usd_micros: Literal[1815360]
    reconciled_prior_liability_usd_micros: Literal[59847229]
    credentials_opened: Literal[False]
    source_details_opened: Literal[False]

    @model_validator(mode="after")
    def validate_accounting(self) -> PilotPriorAccountingConfigV1:
        if self.reconciled_prior_liability_usd_micros != (
            self.reported_prior_spend_usd_micros
            + self.unknown_prior_liability_usd_micros
            + self.v3_certified_unresolved_upper_bound_usd_micros
            + self.v4_certified_liability_upper_bound_usd_micros
        ):
            raise ValueError("hosted_numeric_prior_accounting_sum_mismatch")
        return self


class PilotRosterRecordV1(_Frozen):
    doc_id: Annotated[str, Field(pattern=r"^PMC[0-9]+$")]
    publication_id: NonEmpty
    paper_id: NonEmpty
    artifact_path: NonEmpty
    artifact_sha256: Sha256
    media_type: Literal["text/plain"]
    source_locator: Annotated[str, Field(pattern=r"^harvest-sha256:[0-9a-f]{64}$")]
    target_line_id: Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]
    target_clause_start: NonEmpty
    target_clause_stop: NonEmpty
    intervention_label: NonEmpty
    comparator_label: NonEmpty
    event_definition: NonEmpty
    outcome_name: Literal["persistent_h_pylori_detection"]
    timepoint_label: NonEmpty
    timepoint_value: Annotated[float, Field(gt=0)]
    timepoint_unit: Literal["month", "year"]
    follow_up_duration: Literal["three-months", "seven-point-three-years"]
    population_description: NonEmpty
    study_design: NonEmpty
    reported_significance: Literal["significant", "not_reported"]
    selection_reason: Literal[
        "public full text contains an explicit two-arm binary event/total result "
        "for the prespecified endpoint"
    ]

    @model_validator(mode="after")
    def validate_source_binding(self) -> PilotRosterRecordV1:
        if self.source_locator != f"harvest-sha256:{self.artifact_sha256}":
            raise ValueError("hosted_numeric_roster_locator_hash_mismatch")
        relative = PurePosixPath(self.artifact_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.artifact_path
        ):
            raise ValueError("hosted_numeric_roster_artifact_path_invalid")
        return self


class HostedNativeNumericPilotConfigV1(_Frozen):
    config_version: Literal["hosted-native-numeric-yield-pilot-config-v5"]
    run_id: Literal["hosted-native-numeric-yield-pilot-fable5-prompt-json-v5"]
    question_config_path: NonEmpty
    prompt_template_path: NonEmpty
    corpus_cutoff: Literal["2026-08-29T00:00:00Z"]
    selection_contract: PilotSelectionContractV1
    prior_accounting: PilotPriorAccountingConfigV1
    provider: PilotProviderConfigV1
    roster: Annotated[list[PilotRosterRecordV1], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def validate_roster(self) -> HostedNativeNumericPilotConfigV1:
        ids = [item.doc_id for item in self.roster]
        if ids != sorted(set(ids)) or len(ids) != MAXIMUM_PROVIDER_CALLS:
            raise ValueError("hosted_numeric_roster_not_sorted_unique")
        return self

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


def load_hosted_native_numeric_pilot_config_v1(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> HostedNativeNumericPilotConfigV1:
    path = _safe_repository_file(repository_root, config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_config_invalid") from exc
    _assert_secret_free(raw)
    return HostedNativeNumericPilotConfigV1.model_validate(raw)


def _load_validated_canary_prerequisite(
    *,
    repository_root: Path,
    canary_workspace: Path,
    expected_canary_terminal_sha256: str,
) -> tuple[str, HostedNativeNumericCanarySuccessBindingV4]:
    root = repository_root.resolve(strict=True)
    candidate = canary_workspace if canary_workspace.is_absolute() else root / canary_workspace
    if candidate.is_symlink():
        raise HostedNativeNumericPilotError("hosted_numeric_canary_workspace_symlink_forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise HostedNativeNumericPilotError(
            "hosted_numeric_canary_workspace_outside_repository_or_missing"
        ) from exc
    try:
        binding = load_successful_hosted_native_numeric_canary_v4(
            workspace=resolved,
            expected_terminal_sha256=expected_canary_terminal_sha256,
        )
    except HostedNativeNumericCanaryV4Error as exc:
        raise HostedNativeNumericPilotError(
            "hosted_numeric_successful_source_free_canary_required"
        ) from exc
    if (
        binding.certified_request_liability_usd_micros > CANARY_HARD_CEILING_USD_MICROS
        or binding.charged_cost_upper_bound_usd_micros > CANARY_HARD_CEILING_USD_MICROS
        or binding.scientific_authority is not False
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_canary_binding_invalid")
    return relative, binding


def _revalidate_plan_canary_prerequisite(
    *, repository_root: Path, plan: HostedNativeNumericPilotPlanV1
) -> HostedNativeNumericCanarySuccessBindingV4:
    root = repository_root.resolve(strict=True)
    workspace = root / plan.canary_workspace_path
    try:
        observed = require_hosted_native_numeric_canary_binding_v4(
            workspace=workspace,
            expected_binding=plan.canary_success_binding,
        )
    except HostedNativeNumericCanaryV4Error as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_canary_prerequisite_drift") from exc
    if (
        observed.binding_sha256 != plan.canary_success_binding_sha256
        or observed.terminal_sha256 != plan.canary_terminal_sha256
        or observed.terminal_artifact_sha256 != plan.canary_terminal_artifact_sha256
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_canary_prerequisite_drift")
    return observed


def _revalidate_plan_source_artifacts(
    *, repository_root: Path, plan: HostedNativeNumericPilotPlanV1
) -> None:
    expected_manifest = _source_manifest(plan.config)
    if expected_manifest != plan.source_manifest:
        raise HostedNativeNumericPilotError("hosted_numeric_source_manifest_drift")
    try:
        for surface, source_record in zip(
            plan.surfaces,
            plan.source_manifest.records,
            strict=True,
        ):
            source = resolve_native_source_document(
                repository_root=repository_root,
                source_document=source_record.source_document,
            )
            target_line = next(
                (
                    line
                    for line in source.lines
                    if line.line_id == surface.roster_record.target_line_id
                ),
                None,
            )
            if target_line is None:
                raise HostedNativeNumericPilotError("hosted_numeric_target_line_missing")
            start = target_line.text.find(surface.roster_record.target_clause_start)
            stop = target_line.text.find(surface.roster_record.target_clause_stop, start + 1)
            if start < 0 or stop <= start:
                raise HostedNativeNumericPilotError("hosted_numeric_target_clause_not_unique")
            target_clause = target_line.text[start:stop].rstrip()
            if (
                source.source_payload_sha256 != surface.resolved_source_payload_sha256
                or _sha256_utf8(target_line.text) != surface.target_line_sha256
                or _sha256_utf8(target_clause) != surface.target_clause_sha256
            ):
                raise HostedNativeNumericPilotError("hosted_numeric_source_artifact_drift")
    except HostedNativeNumericPilotError:
        raise
    except Exception as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_source_artifact_drift") from exc


def validate_hosted_native_numeric_plan_prerequisites_v4(
    *, repository_root: Path, plan: HostedNativeNumericPilotPlanV1
) -> None:
    """Replay canary, source, and runtime bytes before any source-bearing contact."""

    repository = repository_root.resolve(strict=True)
    _revalidate_plan_canary_prerequisite(repository_root=repository, plan=plan)
    _revalidate_plan_source_artifacts(repository_root=repository, plan=plan)
    require_pipeline_fingerprint_match(
        expected=plan.pipeline_fingerprint,
        root=repository,
    )


def _source_manifest(config: HostedNativeNumericPilotConfigV1) -> NativeSourceManifest:
    records = [
        NativeSourceRecord(
            doc_id=item.doc_id,
            publication=PublicationIdentity(
                publication_id=item.publication_id,
                paper_id=item.paper_id,
                doc_id=item.doc_id,
            ),
            source_document=SourceDocumentArtifact(
                artifact_path=item.artifact_path,
                sha256=item.artifact_sha256,
                media_type=item.media_type,
                source_locator=item.source_locator,
            ),
        )
        for item in config.roster
    ]
    return NativeSourceManifest(question_id="hosted-native-numeric-yield-pilot-v4", records=records)


def _closed_object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


def _const(value: JsonValue) -> dict[str, Any]:
    type_name = (
        "null"
        if value is None
        else "boolean"
        if isinstance(value, bool)
        else "integer"
        if isinstance(value, int)
        else "number"
        if isinstance(value, float)
        else "string"
    )
    return {"type": type_name, "const": value}


def _fixed_list(item_schema: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {
        "type": "array",
        # The pinned Anthropic SDK schema transformer accepts a mapping here, not
        # JSON Schema's boolean ``false`` form. maxItems=0 remains authoritative.
        "items": dict(item_schema) if count else {"type": "string"},
        "minItems": count,
        "maxItems": count,
    }


class HostedNativeSchemaStructuralMetricsV1(_Frozen):
    schema_sha256: Sha256
    schema_utf8_bytes: Annotated[StrictInt, Field(ge=1)]
    node_count: Annotated[StrictInt, Field(ge=1)]
    max_depth: Annotated[StrictInt, Field(ge=0)]
    object_schema_count: Annotated[StrictInt, Field(ge=1)]
    total_object_properties: Annotated[StrictInt, Field(ge=1)]
    array_schema_count: Annotated[StrictInt, Field(ge=1)]
    arrays_with_min_items: Annotated[StrictInt, Field(ge=0)]
    arrays_with_max_items: Annotated[StrictInt, Field(ge=0)]
    const_keyword_count: Annotated[StrictInt, Field(ge=0)]


def measure_hosted_native_schema_structure_v1(
    schema: Mapping[str, Any],
) -> HostedNativeSchemaStructuralMetricsV1:
    """Measure the exact original schema delivered in the prompt-JSON envelope."""

    canonical_schema = json.loads(canonical_json_bytes(dict(schema)))
    counters = {
        "node_count": 0,
        "max_depth": 0,
        "object_schema_count": 0,
        "total_object_properties": 0,
        "array_schema_count": 0,
        "arrays_with_min_items": 0,
        "arrays_with_max_items": 0,
        "const_keyword_count": 0,
    }

    def visit(value: Any, depth: int) -> None:
        counters["node_count"] += 1
        counters["max_depth"] = max(counters["max_depth"], depth)
        if isinstance(value, Mapping):
            if "const" in value:
                counters["const_keyword_count"] += 1
            if value.get("type") == "object":
                counters["object_schema_count"] += 1
                properties = value.get("properties")
                if isinstance(properties, Mapping):
                    counters["total_object_properties"] += len(properties)
            if value.get("type") == "array":
                counters["array_schema_count"] += 1
                if "minItems" in value:
                    counters["arrays_with_min_items"] += 1
                if "maxItems" in value:
                    counters["arrays_with_max_items"] += 1
            for key in sorted(value):
                visit(value[key], depth + 1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, depth + 1)

    visit(canonical_schema, 0)
    return HostedNativeSchemaStructuralMetricsV1(
        schema_sha256=hash_canonical(canonical_schema),
        schema_utf8_bytes=len(canonical_json_bytes(canonical_schema)),
        **counters,
    )


def require_hosted_native_prompt_json_schema_guard_v1(
    schema: Mapping[str, Any],
) -> HostedNativeSchemaStructuralMetricsV1:
    """Fail closed if the frozen full-object prompt schema drifts beyond v5 bounds."""

    metrics = measure_hosted_native_schema_structure_v1(schema)
    if (
        metrics.schema_utf8_bytes > DELIVERED_SCHEMA_MAX_UTF8_BYTES
        or metrics.node_count > DELIVERED_SCHEMA_MAX_NODE_COUNT
        or metrics.max_depth > DELIVERED_SCHEMA_MAX_DEPTH
        or metrics.total_object_properties > DELIVERED_SCHEMA_MAX_OBJECT_PROPERTIES
        or metrics.array_schema_count > DELIVERED_SCHEMA_MAX_ARRAY_SCHEMAS
        or metrics.array_schema_count != DELIVERED_SCHEMA_EXPECTED_ARRAY_SCHEMAS
        or metrics.node_count != DELIVERED_SCHEMA_EXPECTED_NODE_COUNT
        or metrics.arrays_with_min_items != DELIVERED_SCHEMA_EXPECTED_ARRAY_SCHEMAS
        or metrics.arrays_with_max_items != DELIVERED_SCHEMA_EXPECTED_ARRAY_SCHEMAS
        or metrics.const_keyword_count != DELIVERED_SCHEMA_EXPECTED_CONST_KEYWORDS
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_delivered_schema_guard_failed")
    return metrics


def _candidate_generation_schema(item: PilotRosterRecordV1) -> dict[str, Any]:
    null = _const(None)
    empty = _fixed_list({}, 0)
    arm = _closed_object(
        {
            "key": {"type": "string", "enum": ["comparator", "treatment"]},
            "label": {
                "type": "string",
                "enum": [item.comparator_label, item.intervention_label],
            },
            "role": {"type": "string", "enum": ["comparator", "intervention"]},
            "description": null,
            "sample_size": null,
        }
    )
    moderator = _closed_object(
        {
            "name": _const("follow-up-duration"),
            "value": _const(item.follow_up_duration),
        }
    )
    effect = _closed_object(
        {
            "effect_format": _const("odds_ratio"),
            "availability": _const("available"),
            "estimate": null,
            "standard_error": null,
            "variance": null,
            "ci_lower": null,
            "ci_upper": null,
            "ci_level": _const(0.95),
            "unit": null,
            "treatment_mean": null,
            "treatment_sd": null,
            "treatment_n": null,
            "control_mean": null,
            "control_sd": null,
            "control_n": null,
            "treatment_events": {"type": "integer", "minimum": 0, "maximum": 1000000},
            "treatment_total": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "control_events": {"type": "integer", "minimum": 0, "maximum": 1000000},
            "control_total": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "reported_p_value": null,
            "reported_significance": _const(item.reported_significance),
            "equivalence_conclusion": _const("not_tested"),
            "equivalence_margin": null,
            "moderators": _fixed_list(moderator, 1),
            "extraction_method": _const("reported"),
        }
    )
    evidence = _closed_object(
        {
            "source_locator": _const(item.source_locator),
            "quote": {"type": "string", "minLength": 10, "maxLength": 3000},
            "section": _const("Body"),
            "page": null,
            "char_start": null,
            "char_end": null,
            "line_ids": _fixed_list(_const(item.target_line_id), 1),
        }
    )
    finding = _closed_object(
        {
            "key": _const("persistent"),
            "contrast_key": _const("primary"),
            "outcome_name": _const(item.outcome_name),
            "timepoint": _closed_object(
                {
                    "kind": _const("exact"),
                    "value": _const(item.timepoint_value),
                    "lower": null,
                    "upper": null,
                    "unit": _const(item.timepoint_unit),
                    "anchor": _const("after randomized treatment"),
                    "raw_label": _const(item.timepoint_label),
                }
            ),
            "analysis_population": null,
            "effect": effect,
            "evidence": evidence,
        }
    )
    contrast = _closed_object(
        {
            "key": _const("primary"),
            "treatment_arm_key": _const("treatment"),
            "comparator_arm_key": _const("comparator"),
            "label": _const(f"{item.intervention_label} vs {item.comparator_label}"),
            "estimand": _const("odds ratio for persistent H. pylori detection"),
            "positive_direction_means": _const(
                "higher odds of persistent H. pylori detection in the intervention arm"
            ),
        }
    )
    cohort = _closed_object(
        {
            "key": _const("cohort"),
            "source_labels": _fixed_list(_const(f"{item.doc_id} randomized cohort"), 1),
            "registry_ids": empty,
            "dataset_ids": empty,
            "population_description": _const(item.population_description),
            "recruitment_period": null,
            "total_sample_size": null,
            "arms": _fixed_list(arm, 2),
            "contrasts": _fixed_list(contrast, 1),
            "findings": _fixed_list(finding, 1),
        }
    )
    study = _closed_object(
        {
            "key": _const("study"),
            "source_label": _const(f"{item.doc_id} randomized trial"),
            "design": _const(item.study_design),
            "registration_ids": empty,
            "cohorts": _fixed_list(cohort, 1),
        }
    )
    schema = _closed_object(
        {
            "extraction_schema_version": _const("native-publication-extraction-v1"),
            "status": _const("estimable"),
            "studies": _fixed_list(study, 1),
            "non_estimability_reason": null,
            "non_estimability_detail": null,
            "warnings": empty,
        }
    )
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"urn:literature-multiverse:hosted-native-numeric:{item.doc_id}:v1",
            "title": f"Hosted native numeric extraction for {item.doc_id}",
        }
    )
    validator_for(schema).check_schema(schema)
    return schema


def _render_prompt(
    *,
    template: str,
    question: QuestionConfig,
    roster: PilotRosterRecordV1,
    target_clause: str,
    full_text_with_line_ids: str,
) -> str:
    replacements = {
        "RESEARCH_QUESTION": question.research_question,
        "INTERVENTION_LABEL": roster.intervention_label,
        "COMPARATOR_LABEL": roster.comparator_label,
        "EVENT_DEFINITION": roster.event_definition,
        "TIMEPOINT_LABEL": roster.timepoint_label,
        "TARGET_CLAUSE_START": roster.target_clause_start,
        "TARGET_CLAUSE_STOP": roster.target_clause_stop,
        "TARGET_CLAUSE": target_clause,
        "DOC_ID": roster.doc_id,
        "SOURCE_SHA256": roster.artifact_sha256,
        "SOURCE_LOCATOR": roster.source_locator,
        "TARGET_LINE_ID": roster.target_line_id,
        "FULL_TEXT_WITH_LINE_IDS": full_text_with_line_ids,
    }
    rendered = template
    for name, value in replacements.items():
        rendered = rendered.replace("{{" + name + "}}", value)
    if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
        raise HostedNativeNumericPilotError("hosted_numeric_prompt_placeholder_unresolved")
    _assert_secret_free(rendered)
    return rendered


def build_hosted_native_numeric_prompt_json_wire_request_v1(
    *, prompt: str, delivered_schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact no-provider-grammar v5 scientific request surface."""

    require_hosted_native_prompt_json_schema_guard_v1(delivered_schema)
    model_system = render_anthropic_prompt_json_model_system(
        base_system=SYSTEM_PROMPT,
        wire_schema=delivered_schema,
    )
    request = {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": model_system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": EFFORT},
        "service_tier": SERVICE_TIER,
    }
    if "format" in request["output_config"]:
        raise HostedNativeNumericPilotError("hosted_numeric_provider_grammar_forbidden")
    return request


def _wire_kwargs(*, prompt: str, delivered_schema: Mapping[str, Any]) -> dict[str, Any]:
    return build_hosted_native_numeric_prompt_json_wire_request_v1(
        prompt=prompt,
        delivered_schema=delivered_schema,
    )


def _count_kwargs(wire_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: json.loads(canonical_json_bytes(value))
        for name, value in wire_kwargs.items()
        if name not in {"max_tokens", "service_tier"}
    }


class HostedNativeNumericPreparedSurfaceV1(_Frozen):
    source_record_index: Annotated[StrictInt, Field(ge=0, le=1)]
    roster_record: PilotRosterRecordV1
    resolved_source_payload_sha256: Sha256
    target_line_sha256: Sha256
    target_clause_sha256: Sha256
    prompt: HostedNativePromptArtifactV1
    compiled_schema: AnthropicCompiledSchemaV1
    generation_schema: HostedNativeSchemaArtifactV1
    delivered_schema_sha256: Sha256
    schema_structural_metrics: HostedNativeSchemaStructuralMetricsV1
    transport_mode: Literal["prompt_json_schema"] = TRANSPORT_MODE
    provider_grammar_enabled: Literal[False] = PROVIDER_GRAMMAR_ENABLED
    model_system_sha256: Sha256
    model_system_utf8_bytes: Annotated[StrictInt, Field(ge=1)]
    intent: HostedNativeCallIntentV1
    count_request_utf8: NonEmpty
    count_request_sha256: Sha256
    known_count_surface_utf8_bytes: TokenCount
    offline_known_input_token_ceiling: TokenCount
    offline_known_request_liability_usd_micros: Annotated[StrictInt, Field(ge=1)]
    surface_sha256: Sha256

    @model_validator(mode="after")
    def validate_surface(self) -> HostedNativeNumericPreparedSurfaceV1:
        expected_wire = build_hosted_native_numeric_prompt_json_wire_request_v1(
            prompt=self.prompt.rendered_prompt,
            delivered_schema=self.compiled_schema.original_schema,
        )
        expected_wire_utf8 = canonical_json_bytes(expected_wire).decode("utf-8")
        expected_count_utf8 = canonical_json_bytes(_count_kwargs(expected_wire)).decode("utf-8")
        model_system = expected_wire["system"]
        if not isinstance(model_system, str):
            raise ValueError("hosted_numeric_model_system_not_text")
        if (
            self.roster_record.doc_id != self.intent.doc_id
            or self.prompt.rendered_prompt_sha256 != self.intent.rendered_prompt_sha256
            or self.generation_schema.schema_sha256 != self.intent.generation_schema_sha256
            or self.compiled_schema.original_schema != self.generation_schema.schema_payload
            or self.delivered_schema_sha256 != hash_canonical(self.compiled_schema.original_schema)
            or self.delivered_schema_sha256 != self.generation_schema.schema_sha256
            or self.schema_structural_metrics
            != require_hosted_native_prompt_json_schema_guard_v1(
                self.compiled_schema.original_schema
            )
            or self.transport_mode != TRANSPORT_MODE
            or self.provider_grammar_enabled is not PROVIDER_GRAMMAR_ENABLED
            or self.model_system_sha256 != _sha256_utf8(model_system)
            or self.model_system_utf8_bytes != len(model_system.encode("utf-8"))
            or self.intent.wire_request_utf8 != expected_wire_utf8
            or self.intent.wire_request_sha256 != _sha256_utf8(expected_wire_utf8)
            or self.compiled_schema.wire_optional_parameter_count != 0
            or self.compiled_schema.wire_union_parameter_count != 0
            or self.count_request_utf8 != expected_count_utf8
            or self.count_request_sha256 != _sha256_utf8(self.count_request_utf8)
            or self.known_count_surface_utf8_bytes != len(self.count_request_utf8.encode("utf-8"))
            or self.offline_known_input_token_ceiling
            != self.known_count_surface_utf8_bytes + OFFLINE_FRAMING_TOKEN_ALLOWANCE
            or self.offline_known_input_token_ceiling > MAX_CERTIFIED_INPUT_TOKENS
            or self.offline_known_request_liability_usd_micros
            != self.offline_known_input_token_ceiling * int(INPUT_RATE_USD_PER_MTOK)
            + MAX_OUTPUT_TOKENS * int(OUTPUT_RATE_USD_PER_MTOK)
        ):
            raise ValueError("hosted_numeric_prepared_surface_mismatch")
        _self_hash(self, "surface_sha256", "hosted_numeric_surface_hash_mismatch")
        return self


def _expected_hosted_native_provider_identity_v4(
    *,
    config: HostedNativeNumericPilotConfigV1,
    canary_binding: HostedNativeNumericCanarySuccessBindingV4,
) -> HostedNativeProviderIdentityV1:
    return freeze_hosted_native_provider_identity_v1(
        provider_id=config.provider.provider_id,
        model_id=MODEL,
        model_revision=MODEL_REVISION,
        api_base_url=API_BASE_URL,
        runtime_id="hosted-native-numeric-pilot-v5",
        runtime_version=RUNTIME_VERSION,
        sdk_name=config.provider.sdk_name,
        sdk_version=SDK_VERSION,
        runtime_source_paths=sorted(RUNTIME_PIPELINE_PATHS),
        runtime_metadata={
            "application_retries": 0,
            "canary_binding_sha256": canary_binding.binding_sha256,
            "canary_terminal_artifact_sha256": canary_binding.terminal_artifact_sha256,
            "canary_terminal_sha256": canary_binding.terminal_sha256,
            "delivered_schema_role": "original_candidate_prompt_constraint",
            "effort": EFFORT,
            "labels_opened": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "output_format_present_in_call": False,
            "prompt_json_system_envelope_sha256": (ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256),
            "prompt_json_system_envelope_version": (ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION),
            "provider_grammar_enabled": PROVIDER_GRAMMAR_ENABLED,
            "sdk_retries": 0,
            "selection_basis": config.selection_contract.selection_basis,
            "transport_mode": TRANSPORT_MODE,
            "yield_only": True,
        },
    )


class HostedNativeNumericPilotPlanV1(_Frozen):
    plan_version: Literal["hosted-native-numeric-pilot-plan-v5"] = PLAN_VERSION
    status: Literal["offline_prepared_no_provider_calls"] = "offline_prepared_no_provider_calls"
    config: HostedNativeNumericPilotConfigV1
    config_sha256: Sha256
    question_config: QuestionConfig
    question_config_sha256: Sha256
    source_manifest: NativeSourceManifest
    source_manifest_sha256: Sha256
    source_membership_sha256: Sha256
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: Sha256
    canary_workspace_path: NonEmpty
    canary_success_binding: HostedNativeNumericCanarySuccessBindingV4
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    provider_identity: HostedNativeProviderIdentityV1
    official_schema: HostedNativeSchemaArtifactV1
    surfaces: Annotated[
        list[HostedNativeNumericPreparedSurfaceV1], Field(min_length=2, max_length=2)
    ]
    surface_membership_sha256: Sha256
    transport_mode: Literal["prompt_json_schema"] = TRANSPORT_MODE
    provider_grammar_enabled: Literal[False] = PROVIDER_GRAMMAR_ENABLED
    maximum_canary_generation_calls: Literal[1] = 1
    maximum_combined_v4_generation_calls: Literal[3] = 3
    maximum_combined_v4_provider_contacts: Literal[5] = 5
    maximum_provider_calls: Literal[2] = 2
    maximum_new_liability_usd_micros: Literal[2400000] = NEW_LIABILITY_HARD_CEILING_USD_MICROS
    canary_liability_allocation_usd_micros: Literal[600000] = CANARY_LIABILITY_ALLOCATION_USD_MICROS
    combined_v4_phase_hard_ceiling_usd_micros: Literal[3000000] = (
        COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS
    )
    per_call_hard_liability_usd_micros: Literal[1200000] = PER_CALL_HARD_LIABILITY_USD_MICROS
    project_liability_hard_stop_usd_micros: Literal[100000000] = (
        PROJECT_LIABILITY_HARD_STOP_USD_MICROS
    )
    provider_calls_made: Literal[False] = False
    reference_fields_opened: Literal[False] = False
    private_predictions_opened: Literal[False] = False
    yield_only_no_accuracy_representativeness_synthesis_or_release_authority: Literal[True] = True
    plan_sha256: Sha256

    @field_validator("canary_workspace_path")
    @classmethod
    def validate_canary_workspace_path(cls, value: str) -> str:
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != value
            or relative.parts[:2] != ("data", "cache")
        ):
            raise ValueError("hosted_numeric_canary_workspace_path_invalid")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> HostedNativeNumericPilotPlanV1:
        records = [item.model_dump(mode="json") for item in self.source_manifest.records]
        expected_identity = _expected_hosted_native_provider_identity_v4(
            config=self.config,
            canary_binding=self.canary_success_binding,
        )
        if (
            self.config_sha256 != self.config.config_sha256
            or self.question_config_sha256 != config_sha256(self.question_config)
            or self.source_manifest_sha256 != hash_canonical(self.source_manifest)
            or self.source_membership_sha256 != hash_canonical(records)
            or self.pipeline_fingerprint_sha256 != self.pipeline_fingerprint.pipeline_sha256
            or self.canary_success_binding_sha256 != self.canary_success_binding.binding_sha256
            or self.canary_terminal_sha256 != self.canary_success_binding.terminal_sha256
            or self.canary_terminal_artifact_sha256
            != self.canary_success_binding.terminal_artifact_sha256
            or self.canary_success_binding.charged_cost_upper_bound_usd_micros
            > self.canary_liability_allocation_usd_micros
            or self.provider_identity != expected_identity
            or self.source_manifest.question_id != self.question_config.question_id
            or [surface.source_record_index for surface in self.surfaces] != [0, 1]
            or [surface.roster_record.doc_id for surface in self.surfaces]
            != [record.doc_id for record in self.source_manifest.records]
            or self.surface_membership_sha256
            != hash_canonical([item.surface_sha256 for item in self.surfaces])
            or self.transport_mode != self.config.provider.transport_mode
            or self.provider_grammar_enabled is not self.config.provider.provider_grammar_enabled
            or self.maximum_new_liability_usd_micros + self.canary_liability_allocation_usd_micros
            != self.combined_v4_phase_hard_ceiling_usd_micros
            or self.maximum_provider_calls + self.maximum_canary_generation_calls
            != self.maximum_combined_v4_generation_calls
            or self.maximum_combined_v4_provider_contacts
            != self.maximum_provider_calls * 2 + self.maximum_canary_generation_calls
        ):
            raise ValueError("hosted_numeric_plan_alias_mismatch")
        for index, surface in enumerate(self.surfaces):
            expected_intent = freeze_hosted_native_call_intent_v1(
                run_id=self.config.run_id,
                request_key=surface.intent.request_key,
                question_config=self.question_config,
                source_manifest=self.source_manifest,
                source_record_index=index,
                pipeline_fingerprint=self.pipeline_fingerprint,
                corpus_cutoff=self.config.corpus_cutoff,
                provider_identity=self.provider_identity,
                prompt=surface.prompt,
                generation_schema=surface.generation_schema,
                official_schema=self.official_schema,
                wire_request_utf8=surface.intent.wire_request_utf8,
            )
            if surface.intent != expected_intent:
                raise ValueError("hosted_numeric_plan_surface_intent_alias_mismatch")
        _self_hash(self, "plan_sha256", "hosted_numeric_plan_hash_mismatch")
        return self


def _pipeline_fingerprint(repository_root: Path) -> PipelineFingerprint:
    return compute_pipeline_fingerprint(
        root=repository_root,
        components=[
            PipelineComponentSpec(
                component_id="native-extraction",
                component_version="13",
                file_paths=sorted(PIPELINE_REQUIRED_PATHS),
                settings={
                    "authority": "source_provenance_only",
                    "hosted_native_execution_mode": "hosted_exact_once",
                    "hosted_native_extraction_run_contract": ("hosted-native-extraction-run-v1"),
                    "native_extraction_entry_points": [
                        "scripts/build_hosted_native_grounding_package.py"
                    ],
                },
            ),
            PipelineComponentSpec(
                component_id="hosted-native-numeric-pilot-contract",
                component_version="5",
                file_paths=sorted(SCIENTIFIC_CONTRACT_PATHS),
                settings={
                    "accuracy_authority": False,
                    "claim_release_authority": False,
                    "selection_basis": (
                        "public_source_text_only_without_reference_or_prediction_fields"
                    ),
                    "yield_only": True,
                },
            ),
            PipelineComponentSpec(
                component_id="hosted-native-numeric-pilot-runtime",
                component_version=RUNTIME_VERSION,
                file_paths=sorted(RUNTIME_PIPELINE_PATHS),
                settings={
                    "application_retries": 0,
                    "delivered_schema_role": "original_candidate_prompt_constraint",
                    "effort": EFFORT,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "model": MODEL,
                    "provider_grammar_enabled": PROVIDER_GRAMMAR_ENABLED,
                    "sdk_retries": 0,
                    "transport_mode": TRANSPORT_MODE,
                },
            ),
        ],
    )


def freeze_hosted_native_numeric_pilot_plan_v1(
    *,
    repository_root: Path,
    expected_canary_terminal_sha256: str,
    canary_workspace: Path = DEFAULT_CANARY_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> HostedNativeNumericPilotPlanV1:
    root = repository_root.resolve(strict=True)
    canary_workspace_path, canary_binding = _load_validated_canary_prerequisite(
        repository_root=root,
        canary_workspace=canary_workspace,
        expected_canary_terminal_sha256=expected_canary_terminal_sha256,
    )
    config = load_hosted_native_numeric_pilot_config_v1(
        repository_root=root, config_path=config_path
    )
    question = load_question_config(
        _safe_repository_file(root, Path(config.question_config_path)),
        require_locked=True,
    )
    manifest = _source_manifest(config)
    pipeline = _pipeline_fingerprint(root)
    identity = _expected_hosted_native_provider_identity_v4(
        config=config,
        canary_binding=canary_binding,
    )
    official_schema = freeze_hosted_native_schema_artifact_v1(
        schema_id="official-native-publication-extraction-v1",
        role="official_postvalidation",
        schema_payload=native_publication_extraction_json_schema(),
    )
    template_path = Path(config.prompt_template_path)
    template_file = _safe_repository_file(root, template_path)
    template = template_file.read_text(encoding="utf-8")
    surfaces: list[HostedNativeNumericPreparedSurfaceV1] = []
    for index, (roster, source_record) in enumerate(
        zip(config.roster, manifest.records, strict=True)
    ):
        source = resolve_native_source_document(
            repository_root=root,
            source_document=source_record.source_document,
        )
        lines = {line.line_id: line for line in source.lines}
        target_line = lines.get(roster.target_line_id)
        if target_line is None:
            raise HostedNativeNumericPilotError("hosted_numeric_target_line_missing")
        start = target_line.text.find(roster.target_clause_start)
        stop = target_line.text.find(roster.target_clause_stop, start + 1)
        if start < 0 or stop <= start or target_line.text.count(roster.target_clause_start) != 1:
            raise HostedNativeNumericPilotError("hosted_numeric_target_clause_not_unique")
        target_clause = target_line.text[start:stop].rstrip()
        if len(_COUNT_PAIR_RE.findall(target_clause)) < 2:
            raise HostedNativeNumericPilotError("hosted_numeric_target_clause_pair_absent")
        full_text = "\n".join(f"[{line.line_id}] {line.text}" for line in source.lines)
        rendered = _render_prompt(
            template=template,
            question=question,
            roster=roster,
            target_clause=target_clause,
            full_text_with_line_ids=full_text,
        )
        prompt = freeze_hosted_native_prompt_artifact_v1(
            prompt_id=f"prompt-{roster.doc_id.lower()}",
            prompt_version=PROMPT_VERSION,
            template_path=template_path.as_posix(),
            template_sha256=sha256_file(template_file),
            rendered_prompt=rendered,
        )
        original_schema = _candidate_generation_schema(roster)
        compiled = compile_anthropic_bounded_schema(
            original_schema=original_schema,
            full_acceptance_schema_sha256=hash_canonical(
                native_publication_extraction_json_schema()
            ),
        )
        if compiled.wire_optional_parameter_count != 0 or compiled.wire_union_parameter_count != 0:
            raise HostedNativeNumericPilotError("hosted_numeric_schema_complexity_nonzero")
        schema_metrics = require_hosted_native_prompt_json_schema_guard_v1(original_schema)
        generation = freeze_hosted_native_schema_artifact_v1(
            schema_id=f"generation-{roster.doc_id.lower()}-prompt-json-v5",
            role="generation_constraint",
            schema_payload=original_schema,
        )
        wire_kwargs = _wire_kwargs(prompt=rendered, delivered_schema=original_schema)
        wire_request_utf8 = canonical_json_bytes(wire_kwargs).decode("utf-8")
        intent = freeze_hosted_native_call_intent_v1(
            run_id=config.run_id,
            request_key=f"extract-{roster.doc_id.lower()}-fable5-prompt-json-v5",
            question_config=question,
            source_manifest=manifest,
            source_record_index=index,
            pipeline_fingerprint=pipeline,
            corpus_cutoff=config.corpus_cutoff,
            provider_identity=identity,
            prompt=prompt,
            generation_schema=generation,
            official_schema=official_schema,
            wire_request_utf8=wire_request_utf8,
        )
        count_request_utf8 = canonical_json_bytes(_count_kwargs(wire_kwargs)).decode("utf-8")
        known_bytes = len(count_request_utf8.encode("utf-8"))
        offline_tokens = known_bytes + OFFLINE_FRAMING_TOKEN_ALLOWANCE
        if offline_tokens > MAX_CERTIFIED_INPUT_TOKENS:
            raise HostedNativeNumericPilotError("hosted_numeric_known_surface_too_large")
        surface_payload = {
            "source_record_index": index,
            "roster_record": roster,
            "resolved_source_payload_sha256": source.source_payload_sha256,
            "target_line_sha256": _sha256_utf8(target_line.text),
            "target_clause_sha256": _sha256_utf8(target_clause),
            "prompt": prompt,
            "compiled_schema": compiled,
            "generation_schema": generation,
            "delivered_schema_sha256": hash_canonical(original_schema),
            "schema_structural_metrics": schema_metrics,
            "transport_mode": TRANSPORT_MODE,
            "provider_grammar_enabled": PROVIDER_GRAMMAR_ENABLED,
            "model_system_sha256": _sha256_utf8(str(wire_kwargs["system"])),
            "model_system_utf8_bytes": len(str(wire_kwargs["system"]).encode("utf-8")),
            "intent": intent,
            "count_request_utf8": count_request_utf8,
            "count_request_sha256": _sha256_utf8(count_request_utf8),
            "known_count_surface_utf8_bytes": known_bytes,
            "offline_known_input_token_ceiling": offline_tokens,
            "offline_known_request_liability_usd_micros": (
                offline_tokens * int(INPUT_RATE_USD_PER_MTOK)
                + MAX_OUTPUT_TOKENS * int(OUTPUT_RATE_USD_PER_MTOK)
            ),
        }
        surfaces.append(
            HostedNativeNumericPreparedSurfaceV1.model_validate(
                {**surface_payload, "surface_sha256": hash_canonical(surface_payload)}
            )
        )
    plan_payload = {
        "plan_version": PLAN_VERSION,
        "status": "offline_prepared_no_provider_calls",
        "config": config,
        "config_sha256": config.config_sha256,
        "question_config": question,
        "question_config_sha256": config_sha256(question),
        "source_manifest": manifest,
        "source_manifest_sha256": hash_canonical(manifest),
        "source_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in manifest.records]
        ),
        "pipeline_fingerprint": pipeline,
        "pipeline_fingerprint_sha256": pipeline.pipeline_sha256,
        "canary_workspace_path": canary_workspace_path,
        "canary_success_binding": canary_binding,
        "canary_success_binding_sha256": canary_binding.binding_sha256,
        "canary_terminal_sha256": canary_binding.terminal_sha256,
        "canary_terminal_artifact_sha256": canary_binding.terminal_artifact_sha256,
        "provider_identity": identity,
        "official_schema": official_schema,
        "surfaces": surfaces,
        "surface_membership_sha256": hash_canonical([item.surface_sha256 for item in surfaces]),
        "transport_mode": TRANSPORT_MODE,
        "provider_grammar_enabled": PROVIDER_GRAMMAR_ENABLED,
        "maximum_canary_generation_calls": 1,
        "maximum_combined_v4_generation_calls": 3,
        "maximum_combined_v4_provider_contacts": 5,
        "maximum_provider_calls": 2,
        "maximum_new_liability_usd_micros": NEW_LIABILITY_HARD_CEILING_USD_MICROS,
        "canary_liability_allocation_usd_micros": CANARY_LIABILITY_ALLOCATION_USD_MICROS,
        "combined_v4_phase_hard_ceiling_usd_micros": (COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS),
        "per_call_hard_liability_usd_micros": PER_CALL_HARD_LIABILITY_USD_MICROS,
        "project_liability_hard_stop_usd_micros": PROJECT_LIABILITY_HARD_STOP_USD_MICROS,
        "provider_calls_made": False,
        "reference_fields_opened": False,
        "private_predictions_opened": False,
        "yield_only_no_accuracy_representativeness_synthesis_or_release_authority": True,
    }
    return HostedNativeNumericPilotPlanV1.model_validate(
        {**plan_payload, "plan_sha256": hash_canonical(plan_payload)}
    )


def _fresh_workspace(workspace: Path) -> Path:
    absolute = Path(os.path.abspath(workspace))
    if absolute.exists() or absolute.is_symlink():
        raise HostedNativeNumericPilotError("hosted_numeric_workspace_must_be_fresh")
    absolute.mkdir(parents=True, mode=0o700)
    os.chmod(absolute, 0o700)
    return absolute


def _existing_workspace(workspace: Path) -> Path:
    absolute = Path(os.path.abspath(workspace))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_workspace_missing_or_unsafe") from exc
    if (
        resolved != absolute
        or absolute.is_symlink()
        or not absolute.is_dir()
        or stat.S_IMODE(absolute.stat().st_mode) != 0o700
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_workspace_missing_or_unsafe")
    return absolute


def _artifact(workspace: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix().startswith("./"):
        raise HostedNativeNumericPilotError("hosted_numeric_artifact_path_invalid")
    candidate = workspace / relative
    current = workspace
    for part in relative.parts[:-1]:
        current /= part
        if current.exists() and current.is_symlink():
            raise HostedNativeNumericPilotError("hosted_numeric_artifact_symlink_forbidden")
    if candidate.exists() and candidate.is_symlink():
        raise HostedNativeNumericPilotError("hosted_numeric_artifact_symlink_forbidden")
    return candidate


_WORKSPACE_ROOT_ARTIFACTS_V4 = frozenset(
    {
        ".lock",
        "00-prepared.json",
        "01-reservation.json",
        "02-token-count-certificate.json",
        "03-generation-authorization.json",
        "04-hosted-native-extraction-run-v1.json",
        "05-terminal.json",
    }
)
_WORKSPACE_STATE_DIRECTORIES_V4 = frozenset(
    {
        "count-intents",
        "count-receipts",
        "count-incidents",
        "call-authorizations",
        "generation-intents",
        "provider-receipts",
        "validations",
        "incidents",
        "call-terminals",
    }
)


def _validate_workspace_artifact_namespace_v4(
    *, workspace: Path, plan: HostedNativeNumericPilotPlanV1
) -> None:
    """Reject foreign/stale state so old identities cannot be silently ignored."""

    expected_state_files = {f"{surface.intent.request_key}.json" for surface in plan.surfaces}
    for entry in workspace.iterdir():
        if entry.is_symlink():
            raise HostedNativeNumericPilotError("hosted_numeric_workspace_foreign_artifact")
        if entry.name in _WORKSPACE_ROOT_ARTIFACTS_V4:
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise HostedNativeNumericPilotError("hosted_numeric_workspace_foreign_artifact")
            continue
        if entry.name not in _WORKSPACE_STATE_DIRECTORIES_V4:
            raise HostedNativeNumericPilotError("hosted_numeric_workspace_foreign_artifact")
        if not entry.is_dir() or stat.S_IMODE(entry.stat().st_mode) != 0o700:
            raise HostedNativeNumericPilotError("hosted_numeric_workspace_foreign_artifact")
        for child in entry.iterdir():
            metadata = child.stat(follow_symlinks=False)
            if (
                child.name not in expected_state_files
                or child.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise HostedNativeNumericPilotError("hosted_numeric_workspace_foreign_artifact")


@contextmanager
def _workspace_lock(workspace: Path) -> Any:
    descriptor = os.open(workspace / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
            raise HostedNativeNumericPilotError("hosted_numeric_workspace_lock_mode_invalid")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _persist(path: Path, value: Any) -> None:
    serialized = value.model_dump(mode="json") if isinstance(value, ContractModel) else value
    _assert_secret_free(serialized)
    missing: list[Path] = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    if parent.is_symlink():
        raise HostedNativeNumericPilotError("hosted_numeric_artifact_parent_symlink")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    atomic_write_json(path, value)
    os.chmod(path, 0o600)


def prepare_hosted_native_numeric_pilot_v1(
    *,
    repository_root: Path,
    expected_canary_terminal_sha256: str,
    canary_workspace: Path = DEFAULT_CANARY_WORKSPACE,
    workspace: Path = DEFAULT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> HostedNativeNumericPilotPlanV1:
    plan = freeze_hosted_native_numeric_pilot_plan_v1(
        repository_root=repository_root,
        expected_canary_terminal_sha256=expected_canary_terminal_sha256,
        canary_workspace=canary_workspace,
        config_path=config_path,
    )
    root = _fresh_workspace(workspace)
    with _workspace_lock(root):
        _persist(_artifact(root, Path("00-prepared.json")), plan)
    return plan


def load_hosted_native_numeric_pilot_plan_v1(*, workspace: Path) -> HostedNativeNumericPilotPlanV1:
    root = _existing_workspace(workspace)
    return HostedNativeNumericPilotPlanV1.model_validate(
        _load_json_object(
            _artifact(root, Path("00-prepared.json")),
            code="hosted_numeric_prepared_plan_invalid",
        )
    )


class HostedNativePriorAccountingReceiptV1(_Frozen):
    receipt_version: Literal["hosted-native-prior-accounting-receipt-v5"] = PRIOR_ACCOUNTING_VERSION
    accounting_basis: Literal[
        "root_reconciled_metadata_only_2026-08-29_plus_v3_and_v4_certified_upper_bounds"
    ]
    reported_prior_spend_usd_micros: Literal[38616150]
    unknown_prior_liability_usd_micros: Literal[17482919]
    v3_certified_unresolved_upper_bound_usd_micros: Literal[1932800]
    v4_certified_liability_upper_bound_usd_micros: Literal[1815360]
    reconciled_prior_liability_usd_micros: Literal[59847229]
    metadata_only: Literal[True] = True
    credentials_opened: Literal[False] = False
    source_details_opened: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> HostedNativePriorAccountingReceiptV1:
        if self.reconciled_prior_liability_usd_micros != (
            self.reported_prior_spend_usd_micros
            + self.unknown_prior_liability_usd_micros
            + self.v3_certified_unresolved_upper_bound_usd_micros
            + self.v4_certified_liability_upper_bound_usd_micros
        ):
            raise ValueError("hosted_numeric_prior_accounting_sum_mismatch")
        _self_hash(
            self,
            "receipt_sha256",
            "hosted_numeric_prior_accounting_receipt_hash_mismatch",
        )
        return self


def freeze_hosted_native_prior_accounting_receipt_v1(
    config: PilotPriorAccountingConfigV1,
) -> HostedNativePriorAccountingReceiptV1:
    payload = {
        "receipt_version": PRIOR_ACCOUNTING_VERSION,
        "accounting_basis": config.accounting_basis,
        "reported_prior_spend_usd_micros": config.reported_prior_spend_usd_micros,
        "unknown_prior_liability_usd_micros": config.unknown_prior_liability_usd_micros,
        "v3_certified_unresolved_upper_bound_usd_micros": (
            config.v3_certified_unresolved_upper_bound_usd_micros
        ),
        "v4_certified_liability_upper_bound_usd_micros": (
            config.v4_certified_liability_upper_bound_usd_micros
        ),
        "reconciled_prior_liability_usd_micros": (config.reconciled_prior_liability_usd_micros),
        "metadata_only": True,
        "credentials_opened": False,
        "source_details_opened": False,
    }
    return HostedNativePriorAccountingReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class HostedNativeNumericReservationV1(_Frozen):
    reservation_version: Literal["hosted-native-numeric-pilot-reservation-v5"] = RESERVATION_VERSION
    plan_sha256: Sha256
    prior_accounting_receipt: HostedNativePriorAccountingReceiptV1
    prior_accounting_receipt_sha256: Sha256
    reconciled_project_liability_usd_micros: Literal[59847229]
    new_liability_reserved_usd_micros: Literal[2400000] = NEW_LIABILITY_HARD_CEILING_USD_MICROS
    companion_canary_reserved_usd_micros: Literal[600000] = CANARY_LIABILITY_ALLOCATION_USD_MICROS
    combined_v4_phase_reserved_usd_micros: Literal[3000000] = (
        COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS
    )
    project_liability_after_reservation_usd_micros: Annotated[StrictInt, Field(ge=1)]
    project_liability_hard_stop_usd_micros: Literal[100000000] = (
        PROJECT_LIABILITY_HARD_STOP_USD_MICROS
    )
    strict_below_project_hard_stop: Literal[True] = True
    maximum_scientific_token_count_calls: Literal[2] = 2
    maximum_canary_token_count_calls: Literal[0] = 0
    maximum_scientific_generation_calls: Literal[2] = 2
    maximum_canary_generation_calls: Literal[1] = 1
    maximum_combined_v4_generation_calls: Literal[3] = 3
    maximum_combined_v4_provider_contacts: Literal[5] = 5
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    reservation_sha256: Sha256

    @model_validator(mode="after")
    def validate_reservation(self) -> HostedNativeNumericReservationV1:
        expected = (
            self.reconciled_project_liability_usd_micros
            + self.combined_v4_phase_reserved_usd_micros
        )
        if (
            self.prior_accounting_receipt_sha256 != self.prior_accounting_receipt.receipt_sha256
            or self.reconciled_project_liability_usd_micros
            != self.prior_accounting_receipt.reconciled_prior_liability_usd_micros
            or self.new_liability_reserved_usd_micros + self.companion_canary_reserved_usd_micros
            != self.combined_v4_phase_reserved_usd_micros
            or self.project_liability_after_reservation_usd_micros != expected
            or expected >= self.project_liability_hard_stop_usd_micros
            or self.maximum_scientific_generation_calls + self.maximum_canary_generation_calls
            != self.maximum_combined_v4_generation_calls
            or self.maximum_combined_v4_provider_contacts
            != self.maximum_scientific_token_count_calls
            + self.maximum_canary_token_count_calls
            + self.maximum_combined_v4_generation_calls
        ):
            raise ValueError("hosted_numeric_project_liability_hard_stop")
        _self_hash(self, "reservation_sha256", "hosted_numeric_reservation_hash_mismatch")
        return self


def freeze_hosted_native_numeric_reservation_v1(
    *,
    plan: HostedNativeNumericPilotPlanV1,
) -> HostedNativeNumericReservationV1:
    prior = freeze_hosted_native_prior_accounting_receipt_v1(plan.config.prior_accounting)
    payload = {
        "reservation_version": RESERVATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "prior_accounting_receipt": prior,
        "prior_accounting_receipt_sha256": prior.receipt_sha256,
        "reconciled_project_liability_usd_micros": (prior.reconciled_prior_liability_usd_micros),
        "new_liability_reserved_usd_micros": NEW_LIABILITY_HARD_CEILING_USD_MICROS,
        "companion_canary_reserved_usd_micros": CANARY_LIABILITY_ALLOCATION_USD_MICROS,
        "combined_v4_phase_reserved_usd_micros": (COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS),
        "project_liability_after_reservation_usd_micros": (
            prior.reconciled_prior_liability_usd_micros + COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS
        ),
        "project_liability_hard_stop_usd_micros": PROJECT_LIABILITY_HARD_STOP_USD_MICROS,
        "strict_below_project_hard_stop": True,
        "maximum_scientific_token_count_calls": 2,
        "maximum_canary_token_count_calls": 0,
        "maximum_scientific_generation_calls": 2,
        "maximum_canary_generation_calls": 1,
        "maximum_combined_v4_generation_calls": 3,
        "maximum_combined_v4_provider_contacts": 5,
        "application_retries": 0,
        "sdk_retries": 0,
    }
    try:
        return HostedNativeNumericReservationV1.model_validate(
            {**payload, "reservation_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_project_liability_hard_stop") from exc


def reserve_hosted_native_numeric_pilot_v1(
    *,
    workspace: Path,
    expected_plan_sha256: str,
) -> HostedNativeNumericReservationV1:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        if plan.plan_sha256 != expected_plan_sha256:
            raise HostedNativeNumericPilotError("hosted_numeric_expected_plan_hash_mismatch")
        reservation = freeze_hosted_native_numeric_reservation_v1(
            plan=plan,
        )
        path = _artifact(root, Path("01-reservation.json"))
        if path.exists():
            observed = HostedNativeNumericReservationV1.model_validate(
                _load_json_object(path, code="hosted_numeric_reservation_invalid")
            )
            if observed != reservation:
                raise HostedNativeNumericPilotError("hosted_numeric_reservation_drift")
            return observed
        _persist(path, reservation)
        return reservation


def _load_reservation(workspace: Path) -> HostedNativeNumericReservationV1:
    return HostedNativeNumericReservationV1.model_validate(
        _load_json_object(
            _artifact(workspace, Path("01-reservation.json")),
            code="hosted_numeric_reservation_required",
        )
    )


class HostedNativeNumericCountIntentV1(_Frozen):
    intent_version: Literal["hosted-native-numeric-pilot-count-intent-v5"] = COUNT_INTENT_VERSION
    plan_sha256: Sha256
    reservation_sha256: Sha256
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    request_key: NonEmpty
    surface_sha256: Sha256
    model: Literal["claude-fable-5"] = MODEL
    count_request_utf8: NonEmpty
    count_request_sha256: Sha256
    permitted_attempts: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    intent_durable_before_transport: Literal[True] = True
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> HostedNativeNumericCountIntentV1:
        if self.count_request_sha256 != _sha256_utf8(self.count_request_utf8):
            raise ValueError("hosted_numeric_count_request_hash_mismatch")
        _self_hash(self, "intent_sha256", "hosted_numeric_count_intent_hash_mismatch")
        return self


class HostedNativeNumericCountReceiptV1(_Frozen):
    receipt_version: Literal["hosted-native-numeric-pilot-count-receipt-v5"] = COUNT_RECEIPT_VERSION
    intent_sha256: Sha256
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    request_key: NonEmpty
    surface_sha256: Sha256
    counted_input_tokens: Annotated[StrictInt, Field(ge=1)]
    token_count_headroom: Literal[1024] = TOKEN_COUNT_HEADROOM
    certified_input_token_limit: Annotated[StrictInt, Field(ge=1, le=68800)]
    max_output_tokens: Literal[10240] = MAX_OUTPUT_TOKENS
    certified_request_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=1200000)]
    attempts_observed: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> HostedNativeNumericCountReceiptV1:
        expected_limit = self.counted_input_tokens + self.token_count_headroom
        expected_cost = expected_limit * int(
            INPUT_RATE_USD_PER_MTOK
        ) + self.max_output_tokens * int(OUTPUT_RATE_USD_PER_MTOK)
        if (
            self.certified_input_token_limit != expected_limit
            or expected_limit > MAX_CERTIFIED_INPUT_TOKENS
            or self.certified_request_liability_usd_micros != expected_cost
        ):
            raise ValueError("hosted_numeric_count_receipt_cost_mismatch")
        _self_hash(self, "receipt_sha256", "hosted_numeric_count_receipt_hash_mismatch")
        return self


class HostedNativeNumericCountIncidentV1(_Frozen):
    incident_version: Literal["hosted-native-numeric-pilot-count-incident-v5"] = (
        COUNT_INCIDENT_VERSION
    )
    status: Literal["terminal_count_poison"] = "terminal_count_poison"
    plan_sha256: Sha256
    reservation_sha256: Sha256
    request_key: NonEmpty
    intent_sha256: Sha256
    kind: Literal[
        "orphan_count_intent_on_resume",
        "count_provider_call_raised",
        "count_result_invalid",
        "count_exceeds_certified_input_cap",
    ]
    exception_type: str | None = None
    retry_permitted: Literal[False] = False
    incident_sha256: Sha256

    @field_validator("exception_type")
    @classmethod
    def validate_exception_type(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_CODE_RE.fullmatch(value):
            raise ValueError("hosted_numeric_count_exception_type_invalid")
        return value

    @model_validator(mode="after")
    def validate_incident(self) -> HostedNativeNumericCountIncidentV1:
        _self_hash(self, "incident_sha256", "hosted_numeric_count_incident_hash_mismatch")
        return self


class HostedNativeNumericCountCertificateV1(_Frozen):
    certificate_version: Literal["hosted-native-numeric-pilot-count-certificate-v5"] = (
        COUNT_CERTIFICATE_VERSION
    )
    plan_sha256: Sha256
    reservation_sha256: Sha256
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    receipts: Annotated[list[HostedNativeNumericCountReceiptV1], Field(min_length=2, max_length=2)]
    receipt_membership_sha256: Sha256
    certified_total_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=2400000)]
    token_count_calls_made: Literal[2] = 2
    generation_calls_made: Literal[0] = 0
    actual_target_model_tokenizer_used: Literal[True] = True
    certificate_sha256: Sha256

    @model_validator(mode="after")
    def validate_certificate(self) -> HostedNativeNumericCountCertificateV1:
        keys = [item.request_key for item in self.receipts]
        if (
            keys != sorted(set(keys))
            or any(
                item.canary_success_binding_sha256 != self.canary_success_binding_sha256
                or item.canary_terminal_sha256 != self.canary_terminal_sha256
                or item.canary_terminal_artifact_sha256 != self.canary_terminal_artifact_sha256
                for item in self.receipts
            )
            or self.receipt_membership_sha256
            != hash_canonical([item.receipt_sha256 for item in self.receipts])
            or self.certified_total_liability_usd_micros
            != sum(item.certified_request_liability_usd_micros for item in self.receipts)
        ):
            raise ValueError("hosted_numeric_count_certificate_mismatch")
        _self_hash(
            self,
            "certificate_sha256",
            "hosted_numeric_count_certificate_hash_mismatch",
        )
        return self


class HostedNativeNumericTokenCounterProtocol(Protocol):
    def count_tokens(self, count_request: Mapping[str, Any]) -> int: ...


def _require_sdk() -> Any:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise HostedNativeNumericPilotError("hosted_numeric_anthropic_sdk_missing") from exc
    observed = str(getattr(anthropic, "__version__", "unknown"))
    if observed != SDK_VERSION:
        raise HostedNativeNumericPilotError(
            f"hosted_numeric_anthropic_sdk_version_mismatch:{observed}"
        )
    return anthropic


def _anthropic_client(*, api_key: str) -> Any:
    if not isinstance(api_key, str) or not api_key.strip():
        raise HostedNativeNumericPilotError("hosted_numeric_anthropic_api_key_missing")
    if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
        raise HostedNativeNumericPilotError(
            "hosted_numeric_transport_environment_override_forbidden"
        )
    anthropic = _require_sdk()
    http_client = anthropic.DefaultHttpxClient(
        timeout=TIMEOUT_SECONDS,
        trust_env=False,
        follow_redirects=False,
    )
    return anthropic.Anthropic(
        api_key=api_key,
        base_url=API_BASE_URL,
        default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
        http_client=http_client,
        max_retries=0,
        timeout=TIMEOUT_SECONDS,
    )


class AnthropicHostedNativeNumericTokenCounterV1:
    """Pinned actual-model token counter with one call and zero retries."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise HostedNativeNumericPilotError("hosted_numeric_anthropic_api_key_missing")
        self._client = client if client is not None else _anthropic_client(api_key=api_key)

    def count_tokens(self, count_request: Mapping[str, Any]) -> int:
        response = self._client.messages.count_tokens(**dict(count_request))
        return int(response.input_tokens)


def _count_intent(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
) -> HostedNativeNumericCountIntentV1:
    payload = {
        "intent_version": COUNT_INTENT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "canary_success_binding_sha256": plan.canary_success_binding_sha256,
        "canary_terminal_sha256": plan.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": plan.canary_terminal_artifact_sha256,
        "request_key": surface.intent.request_key,
        "surface_sha256": surface.surface_sha256,
        "model": MODEL,
        "count_request_utf8": surface.count_request_utf8,
        "count_request_sha256": surface.count_request_sha256,
        "permitted_attempts": 1,
        "application_retries": 0,
        "sdk_retries": 0,
        "intent_durable_before_transport": True,
    }
    return HostedNativeNumericCountIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _count_receipt(
    *,
    intent: HostedNativeNumericCountIntentV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    counted_input_tokens: int,
) -> HostedNativeNumericCountReceiptV1:
    limit = counted_input_tokens + TOKEN_COUNT_HEADROOM
    payload = {
        "receipt_version": COUNT_RECEIPT_VERSION,
        "intent_sha256": intent.intent_sha256,
        "canary_success_binding_sha256": intent.canary_success_binding_sha256,
        "canary_terminal_sha256": intent.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": intent.canary_terminal_artifact_sha256,
        "request_key": intent.request_key,
        "surface_sha256": surface.surface_sha256,
        "counted_input_tokens": counted_input_tokens,
        "token_count_headroom": TOKEN_COUNT_HEADROOM,
        "certified_input_token_limit": limit,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "certified_request_liability_usd_micros": (
            limit * int(INPUT_RATE_USD_PER_MTOK) + MAX_OUTPUT_TOKENS * int(OUTPUT_RATE_USD_PER_MTOK)
        ),
        "attempts_observed": 1,
        "application_retries": 0,
        "sdk_retries": 0,
    }
    return HostedNativeNumericCountReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _validate_count_receipt_replay_v4(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    intent: HostedNativeNumericCountIntentV1,
    receipt: HostedNativeNumericCountReceiptV1,
) -> None:
    expected_intent = _count_intent(
        plan=plan,
        reservation=reservation,
        surface=surface,
    )
    expected_receipt = _count_receipt(
        intent=expected_intent,
        surface=surface,
        counted_input_tokens=receipt.counted_input_tokens,
    )
    if intent != expected_intent or receipt != expected_receipt:
        raise HostedNativeNumericPilotError("hosted_numeric_count_replay_mismatch")


def _count_incident(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    intent: HostedNativeNumericCountIntentV1,
    kind: Literal[
        "orphan_count_intent_on_resume",
        "count_provider_call_raised",
        "count_result_invalid",
        "count_exceeds_certified_input_cap",
    ],
    exc: Exception | None = None,
) -> HostedNativeNumericCountIncidentV1:
    exception_type = type(exc).__name__ if exc is not None else None
    payload = {
        "incident_version": COUNT_INCIDENT_VERSION,
        "status": "terminal_count_poison",
        "plan_sha256": plan.plan_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "request_key": intent.request_key,
        "intent_sha256": intent.intent_sha256,
        "kind": kind,
        "exception_type": exception_type,
        "retry_permitted": False,
    }
    return HostedNativeNumericCountIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def preflight_hosted_native_numeric_count_v1(
    *,
    repository_root: Path,
    workspace: Path,
    expected_plan_sha256: str,
    expected_reservation_sha256: str,
) -> bool:
    """Validate every offline count gate before credentials or a client exist.

    The return value is true only when at least one target-model token-count
    transport is still required. Existing poison always fails closed. An orphan
    intent is durably poisoned here, before credential loading, and cannot resume.
    """

    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        if (
            plan.plan_sha256 != expected_plan_sha256
            or reservation.plan_sha256 != plan.plan_sha256
            or reservation.reservation_sha256 != expected_reservation_sha256
        ):
            raise HostedNativeNumericPilotError("hosted_numeric_count_anchor_mismatch")
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        certificate_path = _artifact(root, Path("02-token-count-certificate.json"))
        if certificate_path.exists():
            certificate = HostedNativeNumericCountCertificateV1.model_validate(
                _load_json_object(
                    certificate_path,
                    code="hosted_numeric_count_certificate_invalid",
                )
            )
            _validate_count_certificate_state_v4(
                workspace=root,
                plan=plan,
                reservation=reservation,
                certificate=certificate,
            )
            return False
        transport_required = False
        for surface in plan.surfaces:
            key = surface.intent.request_key
            intent_path = _artifact(root, Path("count-intents") / f"{key}.json")
            receipt_path = _artifact(root, Path("count-receipts") / f"{key}.json")
            incident_path = _artifact(root, Path("count-incidents") / f"{key}.json")
            expected_intent = _count_intent(
                plan=plan,
                reservation=reservation,
                surface=surface,
            )
            if incident_path.exists():
                incident = HostedNativeNumericCountIncidentV1.model_validate(
                    _load_json_object(
                        incident_path,
                        code="hosted_numeric_count_incident_invalid",
                    )
                )
                if (
                    incident.plan_sha256 != plan.plan_sha256
                    or incident.reservation_sha256 != reservation.reservation_sha256
                    or incident.request_key != key
                    or incident.intent_sha256 != expected_intent.intent_sha256
                ):
                    raise HostedNativeNumericPilotError(
                        "hosted_numeric_count_incident_anchor_mismatch"
                    )
                raise HostedNativeNumericPilotError("hosted_numeric_count_already_poisoned")
            if receipt_path.exists():
                intent = HostedNativeNumericCountIntentV1.model_validate(
                    _load_json_object(
                        intent_path,
                        code="hosted_numeric_count_intent_invalid",
                    )
                )
                receipt = HostedNativeNumericCountReceiptV1.model_validate(
                    _load_json_object(
                        receipt_path,
                        code="hosted_numeric_count_receipt_invalid",
                    )
                )
                _validate_count_receipt_replay_v4(
                    plan=plan,
                    reservation=reservation,
                    surface=surface,
                    intent=intent,
                    receipt=receipt,
                )
                continue
            if intent_path.exists():
                observed_intent = HostedNativeNumericCountIntentV1.model_validate(
                    _load_json_object(
                        intent_path,
                        code="hosted_numeric_count_intent_invalid",
                    )
                )
                if observed_intent != expected_intent:
                    raise HostedNativeNumericPilotError("hosted_numeric_count_intent_drift")
                incident = _count_incident(
                    plan=plan,
                    reservation=reservation,
                    intent=expected_intent,
                    kind="orphan_count_intent_on_resume",
                )
                _persist(incident_path, incident)
                raise HostedNativeNumericPilotError("hosted_numeric_count_orphan_poison")
            transport_required = True
        return transport_required


def count_hosted_native_numeric_pilot_tokens_v1(
    *,
    repository_root: Path,
    workspace: Path,
    expected_plan_sha256: str,
    expected_reservation_sha256: str,
    counter: HostedNativeNumericTokenCounterProtocol,
) -> HostedNativeNumericCountCertificateV1:
    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        if (
            plan.plan_sha256 != expected_plan_sha256
            or reservation.plan_sha256 != plan.plan_sha256
            or reservation.reservation_sha256 != expected_reservation_sha256
        ):
            raise HostedNativeNumericPilotError("hosted_numeric_count_anchor_mismatch")
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        certificate_path = _artifact(root, Path("02-token-count-certificate.json"))
        if certificate_path.exists():
            certificate = HostedNativeNumericCountCertificateV1.model_validate(
                _load_json_object(certificate_path, code="hosted_numeric_count_certificate_invalid")
            )
            _validate_count_certificate_state_v4(
                workspace=root,
                plan=plan,
                reservation=reservation,
                certificate=certificate,
            )
            return certificate
        receipts: list[HostedNativeNumericCountReceiptV1] = []
        for surface in plan.surfaces:
            key = surface.intent.request_key
            intent_path = _artifact(root, Path("count-intents") / f"{key}.json")
            receipt_path = _artifact(root, Path("count-receipts") / f"{key}.json")
            incident_path = _artifact(root, Path("count-incidents") / f"{key}.json")
            expected_intent = _count_intent(plan=plan, reservation=reservation, surface=surface)
            if receipt_path.exists():
                if incident_path.exists():
                    raise HostedNativeNumericPilotError(
                        "hosted_numeric_count_receipt_incident_conflict"
                    )
                intent = HostedNativeNumericCountIntentV1.model_validate(
                    _load_json_object(intent_path, code="hosted_numeric_count_intent_invalid")
                )
                receipt = HostedNativeNumericCountReceiptV1.model_validate(
                    _load_json_object(receipt_path, code="hosted_numeric_count_receipt_invalid")
                )
                _validate_count_receipt_replay_v4(
                    plan=plan,
                    reservation=reservation,
                    surface=surface,
                    intent=intent,
                    receipt=receipt,
                )
                receipts.append(receipt)
                continue
            if incident_path.exists():
                raise HostedNativeNumericPilotError("hosted_numeric_count_already_poisoned")
            if intent_path.exists():
                observed_intent = HostedNativeNumericCountIntentV1.model_validate(
                    _load_json_object(intent_path, code="hosted_numeric_count_intent_invalid")
                )
                if observed_intent != expected_intent:
                    raise HostedNativeNumericPilotError("hosted_numeric_count_intent_drift")
                incident = _count_incident(
                    plan=plan,
                    reservation=reservation,
                    intent=expected_intent,
                    kind="orphan_count_intent_on_resume",
                )
                _persist(incident_path, incident)
                raise HostedNativeNumericPilotError("hosted_numeric_count_orphan_poison")
            _persist(intent_path, expected_intent)
            try:
                raw_count = counter.count_tokens(json.loads(surface.count_request_utf8))
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                    raise
                incident = _count_incident(
                    plan=plan,
                    reservation=reservation,
                    intent=expected_intent,
                    kind="count_provider_call_raised",
                    exc=exc,
                )
                _persist(incident_path, incident)
                raise HostedNativeNumericPilotError("hosted_numeric_count_provider_poison") from exc
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
                incident = _count_incident(
                    plan=plan,
                    reservation=reservation,
                    intent=expected_intent,
                    kind="count_result_invalid",
                )
                _persist(incident_path, incident)
                raise HostedNativeNumericPilotError("hosted_numeric_count_result_invalid")
            if raw_count + TOKEN_COUNT_HEADROOM > MAX_CERTIFIED_INPUT_TOKENS:
                incident = _count_incident(
                    plan=plan,
                    reservation=reservation,
                    intent=expected_intent,
                    kind="count_exceeds_certified_input_cap",
                )
                _persist(incident_path, incident)
                raise HostedNativeNumericPilotError("hosted_numeric_count_exceeds_cap")
            receipt = _count_receipt(
                intent=expected_intent,
                surface=surface,
                counted_input_tokens=raw_count,
            )
            _persist(receipt_path, receipt)
            receipts.append(receipt)
        certificate = _freeze_count_certificate_v4(
            plan=plan,
            reservation=reservation,
            receipts=receipts,
        )
        _persist(certificate_path, certificate)
        return certificate


def _load_count_certificate(workspace: Path) -> HostedNativeNumericCountCertificateV1:
    return HostedNativeNumericCountCertificateV1.model_validate(
        _load_json_object(
            _artifact(workspace, Path("02-token-count-certificate.json")),
            code="hosted_numeric_count_certificate_required",
        )
    )


def _freeze_count_certificate_v4(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    receipts: Sequence[HostedNativeNumericCountReceiptV1],
) -> HostedNativeNumericCountCertificateV1:
    ordered = sorted(receipts, key=lambda item: item.request_key)
    payload = {
        "certificate_version": COUNT_CERTIFICATE_VERSION,
        "plan_sha256": plan.plan_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "canary_success_binding_sha256": plan.canary_success_binding_sha256,
        "canary_terminal_sha256": plan.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": plan.canary_terminal_artifact_sha256,
        "receipts": ordered,
        "receipt_membership_sha256": hash_canonical([item.receipt_sha256 for item in ordered]),
        "certified_total_liability_usd_micros": sum(
            item.certified_request_liability_usd_micros for item in ordered
        ),
        "token_count_calls_made": 2,
        "generation_calls_made": 0,
        "actual_target_model_tokenizer_used": True,
    }
    return HostedNativeNumericCountCertificateV1.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def _validate_count_certificate_state_v4(
    *,
    workspace: Path,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    certificate: HostedNativeNumericCountCertificateV1,
) -> None:
    if reservation.plan_sha256 != plan.plan_sha256:
        raise HostedNativeNumericPilotError("hosted_numeric_reservation_plan_mismatch")
    receipts_by_key = {item.request_key: item for item in certificate.receipts}
    expected_keys = {surface.intent.request_key for surface in plan.surfaces}
    if (
        certificate.plan_sha256 != plan.plan_sha256
        or certificate.reservation_sha256 != reservation.reservation_sha256
        or certificate.canary_success_binding_sha256 != plan.canary_success_binding_sha256
        or certificate.canary_terminal_sha256 != plan.canary_terminal_sha256
        or certificate.canary_terminal_artifact_sha256 != plan.canary_terminal_artifact_sha256
        or set(receipts_by_key) != expected_keys
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_count_certificate_anchor_mismatch")
    replayed_receipts: list[HostedNativeNumericCountReceiptV1] = []
    for surface in plan.surfaces:
        key = surface.intent.request_key
        intent_path = _artifact(workspace, Path("count-intents") / f"{key}.json")
        receipt_path = _artifact(workspace, Path("count-receipts") / f"{key}.json")
        incident_path = _artifact(workspace, Path("count-incidents") / f"{key}.json")
        if incident_path.exists():
            raise HostedNativeNumericPilotError("hosted_numeric_count_certificate_with_incident")
        expected_intent = _count_intent(
            plan=plan,
            reservation=reservation,
            surface=surface,
        )
        observed_intent = HostedNativeNumericCountIntentV1.model_validate(
            _load_json_object(intent_path, code="hosted_numeric_count_intent_required")
        )
        observed_receipt = HostedNativeNumericCountReceiptV1.model_validate(
            _load_json_object(receipt_path, code="hosted_numeric_count_receipt_required")
        )
        expected_receipt = _count_receipt(
            intent=expected_intent,
            surface=surface,
            counted_input_tokens=observed_receipt.counted_input_tokens,
        )
        if (
            observed_intent != expected_intent
            or observed_receipt != expected_receipt
            or receipts_by_key[key] != observed_receipt
        ):
            raise HostedNativeNumericPilotError("hosted_numeric_count_certificate_replay_mismatch")
        replayed_receipts.append(observed_receipt)
    expected_certificate = _freeze_count_certificate_v4(
        plan=plan,
        reservation=reservation,
        receipts=replayed_receipts,
    )
    if certificate != expected_certificate:
        raise HostedNativeNumericPilotError("hosted_numeric_count_certificate_replay_mismatch")


class HostedNativeNumericAuthorizedRequestV1(_Frozen):
    request_key: NonEmpty
    intent_sha256: Sha256
    count_receipt_sha256: Sha256
    counted_input_tokens: Annotated[StrictInt, Field(ge=1)]
    certified_input_token_limit: Annotated[StrictInt, Field(ge=1, le=68800)]
    max_output_tokens: Literal[10240] = MAX_OUTPUT_TOKENS
    certified_request_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=1200000)]


class HostedNativeNumericGenerationAuthorizationV1(_Frozen):
    authorization_version: Literal["hosted-native-numeric-pilot-generation-authorization-v5"] = (
        GENERATION_AUTHORIZATION_VERSION
    )
    plan_sha256: Sha256
    reservation_sha256: Sha256
    count_certificate_sha256: Sha256
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    canary_charged_cost_upper_bound_usd_micros: Annotated[StrictInt, Field(ge=1, le=600000)]
    authorized_requests: Annotated[
        list[HostedNativeNumericAuthorizedRequestV1], Field(min_length=2, max_length=2)
    ]
    authorized_membership_sha256: Sha256
    certified_maximum_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=2400000)]
    configured_phase_budget_usd_micros: Annotated[StrictInt, Field(ge=1, le=2400000)]
    combined_v4_certified_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=3000000)]
    full_new_liability_reservation_usd_micros: Literal[2400000] = (
        NEW_LIABILITY_HARD_CEILING_USD_MICROS
    )
    project_liability_after_full_reservation_usd_micros: Annotated[
        StrictInt, Field(ge=1, lt=100000000)
    ]
    maximum_provider_attempts: Literal[2] = 2
    maximum_combined_v4_provider_attempts: Literal[3] = 3
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    all_manifest_records_must_receive_terminal_record: Literal[True] = True
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> HostedNativeNumericGenerationAuthorizationV1:
        keys = [item.request_key for item in self.authorized_requests]
        expected_cost = sum(
            item.certified_request_liability_usd_micros for item in self.authorized_requests
        )
        if (
            keys != sorted(set(keys))
            or self.authorized_membership_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.authorized_requests])
            or self.certified_maximum_liability_usd_micros != expected_cost
            or expected_cost > self.configured_phase_budget_usd_micros
            or self.combined_v4_certified_liability_usd_micros
            != expected_cost + self.canary_charged_cost_upper_bound_usd_micros
            or self.combined_v4_certified_liability_usd_micros
            > COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS
            or self.project_liability_after_full_reservation_usd_micros
            >= PROJECT_LIABILITY_HARD_STOP_USD_MICROS
        ):
            raise ValueError("hosted_numeric_generation_authorization_mismatch")
        _self_hash(
            self,
            "authorization_sha256",
            "hosted_numeric_generation_authorization_hash_mismatch",
        )
        return self


def freeze_hosted_native_numeric_generation_authorization_v1(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    certificate: HostedNativeNumericCountCertificateV1,
    phase_budget_usd_micros: int,
) -> HostedNativeNumericGenerationAuthorizationV1:
    if (
        certificate.canary_success_binding_sha256 != plan.canary_success_binding_sha256
        or certificate.canary_terminal_sha256 != plan.canary_terminal_sha256
        or certificate.canary_terminal_artifact_sha256 != plan.canary_terminal_artifact_sha256
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_count_canary_binding_mismatch")
    receipts = {item.request_key: item for item in certificate.receipts}
    requests: list[HostedNativeNumericAuthorizedRequestV1] = []
    for surface in plan.surfaces:
        receipt = receipts.get(surface.intent.request_key)
        if receipt is None or receipt.surface_sha256 != surface.surface_sha256:
            raise HostedNativeNumericPilotError("hosted_numeric_count_surface_mismatch")
        requests.append(
            HostedNativeNumericAuthorizedRequestV1(
                request_key=surface.intent.request_key,
                intent_sha256=surface.intent.intent_sha256,
                count_receipt_sha256=receipt.receipt_sha256,
                counted_input_tokens=receipt.counted_input_tokens,
                certified_input_token_limit=receipt.certified_input_token_limit,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                certified_request_liability_usd_micros=(
                    receipt.certified_request_liability_usd_micros
                ),
            )
        )
    requests.sort(key=lambda item: item.request_key)
    payload = {
        "authorization_version": GENERATION_AUTHORIZATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "count_certificate_sha256": certificate.certificate_sha256,
        "canary_success_binding_sha256": plan.canary_success_binding_sha256,
        "canary_terminal_sha256": plan.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": plan.canary_terminal_artifact_sha256,
        "canary_charged_cost_upper_bound_usd_micros": (
            plan.canary_success_binding.charged_cost_upper_bound_usd_micros
        ),
        "authorized_requests": requests,
        "authorized_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in requests]
        ),
        "certified_maximum_liability_usd_micros": sum(
            item.certified_request_liability_usd_micros for item in requests
        ),
        "configured_phase_budget_usd_micros": phase_budget_usd_micros,
        "combined_v4_certified_liability_usd_micros": (
            sum(item.certified_request_liability_usd_micros for item in requests)
            + plan.canary_success_binding.charged_cost_upper_bound_usd_micros
        ),
        "full_new_liability_reservation_usd_micros": (NEW_LIABILITY_HARD_CEILING_USD_MICROS),
        "project_liability_after_full_reservation_usd_micros": (
            reservation.project_liability_after_reservation_usd_micros
        ),
        "maximum_provider_attempts": 2,
        "maximum_combined_v4_provider_attempts": 3,
        "application_retries_per_request": 0,
        "sdk_retries_per_request": 0,
        "all_manifest_records_must_receive_terminal_record": True,
    }
    try:
        return HostedNativeNumericGenerationAuthorizationV1.model_validate(
            {**payload, "authorization_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise HostedNativeNumericPilotError(
            "hosted_numeric_generation_authorization_rejected"
        ) from exc


def _validate_generation_authorization_state_v4(
    *,
    workspace: Path,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    certificate: HostedNativeNumericCountCertificateV1,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
) -> None:
    _validate_count_certificate_state_v4(
        workspace=workspace,
        plan=plan,
        reservation=reservation,
        certificate=certificate,
    )
    expected = freeze_hosted_native_numeric_generation_authorization_v1(
        plan=plan,
        reservation=reservation,
        certificate=certificate,
        phase_budget_usd_micros=authorization.configured_phase_budget_usd_micros,
    )
    if (
        authorization != expected
        or authorization.plan_sha256 != plan.plan_sha256
        or reservation.plan_sha256 != plan.plan_sha256
        or authorization.reservation_sha256 != reservation.reservation_sha256
        or authorization.count_certificate_sha256 != certificate.certificate_sha256
        or authorization.canary_success_binding_sha256 != plan.canary_success_binding_sha256
        or authorization.canary_terminal_sha256 != plan.canary_terminal_sha256
        or authorization.canary_terminal_artifact_sha256 != plan.canary_terminal_artifact_sha256
        or authorization.canary_charged_cost_upper_bound_usd_micros
        != plan.canary_success_binding.charged_cost_upper_bound_usd_micros
        or authorization.project_liability_after_full_reservation_usd_micros
        != reservation.project_liability_after_reservation_usd_micros
        or authorization.certified_maximum_liability_usd_micros
        != certificate.certified_total_liability_usd_micros
    ):
        raise HostedNativeNumericPilotError(
            "hosted_numeric_generation_authorization_anchor_mismatch"
        )


def _authorized_request_for_surface_v4(
    *,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
) -> HostedNativeNumericAuthorizedRequestV1:
    matches = [
        item
        for item in authorization.authorized_requests
        if item.request_key == surface.intent.request_key
    ]
    if len(matches) != 1 or matches[0].intent_sha256 != surface.intent.intent_sha256:
        raise HostedNativeNumericPilotError("hosted_numeric_authorized_request_missing")
    return matches[0]


def _account_provider_observation_v4(
    *,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    observation: HostedNativeNumericProviderObservationV1,
) -> tuple[int, int, int, int]:
    authorized = _authorized_request_for_surface_v4(
        authorization=authorization,
        surface=surface,
    )
    if observation.observed_cost_usd_micros is None:
        return (0, authorized.certified_request_liability_usd_micros, 1, 0)
    observed = observation.observed_cost_usd_micros
    return (
        observed,
        observed,
        0,
        int(observed > authorized.certified_request_liability_usd_micros),
    )


def _account_unknown_provider_outcome_v4(
    *,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
) -> tuple[int, int, int, int]:
    authorized = _authorized_request_for_surface_v4(
        authorization=authorization,
        surface=surface,
    )
    return (0, authorized.certified_request_liability_usd_micros, 1, 0)


def authorize_hosted_native_numeric_pilot_v1(
    *,
    repository_root: Path,
    workspace: Path,
    expected_plan_sha256: str,
    expected_reservation_sha256: str,
    expected_count_certificate_sha256: str,
    phase_budget_usd_micros: int,
) -> HostedNativeNumericGenerationAuthorizationV1:
    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        certificate = _load_count_certificate(root)
        if (
            plan.plan_sha256 != expected_plan_sha256
            or reservation.reservation_sha256 != expected_reservation_sha256
            or certificate.certificate_sha256 != expected_count_certificate_sha256
            or certificate.plan_sha256 != plan.plan_sha256
            or certificate.reservation_sha256 != reservation.reservation_sha256
        ):
            raise HostedNativeNumericPilotError(
                "hosted_numeric_generation_authorization_anchor_mismatch"
            )
        _validate_count_certificate_state_v4(
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
        )
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        authorization = freeze_hosted_native_numeric_generation_authorization_v1(
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            phase_budget_usd_micros=phase_budget_usd_micros,
        )
        path = _artifact(root, Path("03-generation-authorization.json"))
        if path.exists():
            observed = HostedNativeNumericGenerationAuthorizationV1.model_validate(
                _load_json_object(path, code="hosted_numeric_generation_auth_invalid")
            )
            if observed != authorization:
                raise HostedNativeNumericPilotError("hosted_numeric_generation_authorization_drift")
            return observed
        _persist(path, authorization)
        return authorization


def _load_generation_authorization(
    workspace: Path,
) -> HostedNativeNumericGenerationAuthorizationV1:
    return HostedNativeNumericGenerationAuthorizationV1.model_validate(
        _load_json_object(
            _artifact(workspace, Path("03-generation-authorization.json")),
            code="hosted_numeric_generation_authorization_required",
        )
    )


class HostedNativeNumericUsageV1(_Frozen):
    input_tokens: TokenCount
    output_tokens: TokenCount
    cache_creation_input_tokens: TokenCount = 0
    cache_read_input_tokens: TokenCount = 0

    @field_validator(
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        mode="before",
    )
    @classmethod
    def reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("hosted_numeric_usage_boolean_forbidden")
        return value

    @property
    def conservative_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


class HostedNativeNumericProviderObservationV1(_Frozen):
    observation_version: Literal["hosted-native-numeric-pilot-provider-observation-v5"] = (
        PROVIDER_OBSERVATION_VERSION
    )
    plan_sha256: Sha256
    generation_authorization_sha256: Sha256
    hosted_intent_sha256: Sha256
    hosted_authorization_sha256: Sha256
    request_key: NonEmpty
    response_id: str | None
    response_model: str | None
    stop_reason: str | None
    content_block_count: TokenCount
    text_block_count: TokenCount
    non_text_block_types: Annotated[
        list[Annotated[str, Field(pattern=_SAFE_CODE_RE.pattern)]],
        Field(max_length=8),
    ]
    content_text: str | None
    content_text_sha256: Sha256 | None
    content_sanitization_failure_code: (
        Literal[
            "response_json_invalid",
            "response_content_secret_rejected",
        ]
        | None
    ) = None
    usage: HostedNativeNumericUsageV1 | None
    observed_cost_usd_micros: Annotated[StrictInt, Field(ge=0)] | None
    transport_attempts: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    credential_archived: Literal[False] = False
    observation_sha256: Sha256

    @model_validator(mode="after")
    def validate_observation(self) -> HostedNativeNumericProviderObservationV1:
        if self.content_block_count != self.text_block_count + len(self.non_text_block_types):
            raise ValueError("hosted_numeric_provider_content_block_count_mismatch")
        if (self.content_text is None) != (self.content_text_sha256 is None):
            raise ValueError("hosted_numeric_provider_text_hash_shape_invalid")
        if self.content_sanitization_failure_code is not None and self.content_text is not None:
            raise ValueError("hosted_numeric_provider_sanitized_text_shape_invalid")
        if self.content_text is not None and self.content_text_sha256 != _sha256_utf8(
            self.content_text
        ):
            raise ValueError("hosted_numeric_provider_text_hash_mismatch")
        expected_cost = (
            self.usage.conservative_input_tokens * int(INPUT_RATE_USD_PER_MTOK)
            + self.usage.output_tokens * int(OUTPUT_RATE_USD_PER_MTOK)
            if self.usage is not None
            else None
        )
        if self.observed_cost_usd_micros != expected_cost:
            raise ValueError("hosted_numeric_provider_cost_mismatch")
        _assert_secret_free(self.model_dump(mode="json", exclude={"observation_sha256"}))
        _self_hash(
            self,
            "observation_sha256",
            "hosted_numeric_provider_observation_hash_mismatch",
        )
        return self


@dataclass(frozen=True)
class HostedNativeNumericRawResponseV1:
    response_id: str | None
    response_model: str | None
    stop_reason: str | None
    content_block_count: int
    content_text: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    text_block_count: int = 1
    non_text_block_types: tuple[str, ...] = ()


class HostedNativeNumericGenerationClientProtocol(Protocol):
    def generate(self, wire_request: Mapping[str, Any]) -> HostedNativeNumericRawResponseV1: ...


def _safe_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 512 or _SECRET_VALUE_RE.search(value):
        return None
    return value


class AnthropicHostedNativeNumericGenerationClientV1:
    """One-attempt Fable adapter; transport exceptions cross to the durable executor."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise HostedNativeNumericPilotError("hosted_numeric_anthropic_api_key_missing")
        self._client = client if client is not None else _anthropic_client(api_key=api_key)

    def generate(self, wire_request: Mapping[str, Any]) -> HostedNativeNumericRawResponseV1:
        response = self._client.messages.create(**dict(wire_request))
        content = list(getattr(response, "content", []))
        text_blocks = [block for block in content if getattr(block, "type", None) == "text"]
        text: str | None = None
        if len(text_blocks) == 1:
            candidate = getattr(text_blocks[0], "text", None)
            text = candidate if isinstance(candidate, str) else None
        non_text_block_types = tuple(
            _safe_scalar(getattr(block, "type", None)) or "unknown"
            for block in content
            if getattr(block, "type", None) != "text"
        )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
        cache_creation = (
            getattr(usage, "cache_creation_input_tokens", 0) if usage is not None else 0
        )
        cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage is not None else 0
        return HostedNativeNumericRawResponseV1(
            response_id=_safe_scalar(getattr(response, "id", None)),
            response_model=_safe_scalar(getattr(response, "model", None)),
            stop_reason=_safe_scalar(getattr(response, "stop_reason", None)),
            content_block_count=len(content),
            content_text=text,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            cache_creation_input_tokens=(cache_creation if isinstance(cache_creation, int) else 0),
            cache_read_input_tokens=cache_read if isinstance(cache_read, int) else 0,
            text_block_count=len(text_blocks),
            non_text_block_types=non_text_block_types,
        )


def _provider_observation(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    hosted_intent: HostedNativeCallIntentV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    raw: HostedNativeNumericRawResponseV1,
) -> HostedNativeNumericProviderObservationV1:
    content_text = raw.content_text
    content_sanitization_failure_code: str | None = None
    if content_text is not None:
        try:
            _strict_json_object(content_text)
        except HostedNativeNumericPilotError:
            # Never persist or hash provider text containing a secret-like key or
            # value.  The typed failure is sufficient for exact-once terminalization.
            content_text = None
            content_sanitization_failure_code = "response_content_secret_rejected"
        except (TypeError, ValueError, json.JSONDecodeError):
            # Malformed provider text has no scientific value and may not be safely
            # inspected as structured data.  Retain only its typed terminal class.
            content_text = None
            content_sanitization_failure_code = "response_json_invalid"
    usage: HostedNativeNumericUsageV1 | None = None
    if raw.input_tokens is not None and raw.output_tokens is not None:
        usage = HostedNativeNumericUsageV1(
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cache_creation_input_tokens=raw.cache_creation_input_tokens,
            cache_read_input_tokens=raw.cache_read_input_tokens,
        )
    payload = {
        "observation_version": PROVIDER_OBSERVATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "generation_authorization_sha256": (generation_authorization.authorization_sha256),
        "hosted_intent_sha256": hosted_intent.intent_sha256,
        "hosted_authorization_sha256": hosted_authorization.authorization_sha256,
        "request_key": hosted_intent.request_key,
        "response_id": raw.response_id,
        "response_model": raw.response_model,
        "stop_reason": raw.stop_reason,
        "content_block_count": raw.content_block_count,
        "text_block_count": raw.text_block_count,
        "non_text_block_types": list(raw.non_text_block_types),
        "content_text": content_text,
        "content_text_sha256": (_sha256_utf8(content_text) if content_text is not None else None),
        "content_sanitization_failure_code": content_sanitization_failure_code,
        "usage": usage,
        "observed_cost_usd_micros": (
            usage.conservative_input_tokens * int(INPUT_RATE_USD_PER_MTOK)
            + usage.output_tokens * int(OUTPUT_RATE_USD_PER_MTOK)
            if usage is not None
            else None
        ),
        "transport_attempts": 1,
        "application_retries": 0,
        "sdk_retries": 0,
        "credential_archived": False,
    }
    return HostedNativeNumericProviderObservationV1.model_validate(
        {**payload, "observation_sha256": hash_canonical(payload)}
    )


class HostedNativeNumericIncidentV1(_Frozen):
    incident_version: Literal["hosted-native-numeric-pilot-incident-v5"] = INCIDENT_VERSION
    plan_sha256: Sha256
    generation_authorization_sha256: Sha256
    request_key: NonEmpty
    hosted_intent_sha256: Sha256
    hosted_authorization_sha256: Sha256
    kind: Literal[
        "orphan_generation_intent_on_resume",
        "provider_http_failure",
        "provider_call_ambiguous_exception",
    ]
    terminal_outcome: Literal["provider_failed", "ambiguous_attempt_poison"]
    failure_code: Annotated[str, Field(pattern=_SAFE_CODE_RE.pattern)]
    exception_type: Annotated[str, Field(pattern=_SAFE_CODE_RE.pattern)] | None = None
    provider_http_status: Annotated[StrictInt, Field(ge=400, le=599)] | None = None
    provider_request_id: str | None = None
    retry_permitted: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> HostedNativeNumericIncidentV1:
        if (self.kind == "provider_http_failure") != (self.provider_http_status is not None):
            raise ValueError("hosted_numeric_incident_http_shape_invalid")
        _assert_secret_free(self.model_dump(mode="json", exclude={"incident_sha256"}))
        _self_hash(self, "incident_sha256", "hosted_numeric_incident_hash_mismatch")
        return self


def _strict_json_object(value: str) -> dict[str, Any]:
    def reject_constant(_: str) -> Any:
        raise ValueError("nonfinite_json_constant")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, child in pairs:
            if key in output:
                raise ValueError("duplicate_json_key")
            output[key] = child
        return output

    parsed = json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(parsed, dict):
        raise ValueError("json_root_not_object")
    _assert_secret_free(parsed)
    return parsed


def _expected_extraction_payload(
    *,
    roster: PilotRosterRecordV1,
    treatment_events: int,
    treatment_total: int,
    control_events: int,
    control_total: int,
    quote: str,
) -> dict[str, Any]:
    return {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "estimable",
        "studies": [
            {
                "key": "study",
                "source_label": f"{roster.doc_id} randomized trial",
                "design": roster.study_design,
                "registration_ids": [],
                "cohorts": [
                    {
                        "key": "cohort",
                        "source_labels": [f"{roster.doc_id} randomized cohort"],
                        "registry_ids": [],
                        "dataset_ids": [],
                        "population_description": roster.population_description,
                        "recruitment_period": None,
                        "total_sample_size": None,
                        "arms": [
                            {
                                "key": "comparator",
                                "label": roster.comparator_label,
                                "role": "comparator",
                                "description": None,
                                "sample_size": None,
                            },
                            {
                                "key": "treatment",
                                "label": roster.intervention_label,
                                "role": "intervention",
                                "description": None,
                                "sample_size": None,
                            },
                        ],
                        "contrasts": [
                            {
                                "key": "primary",
                                "treatment_arm_key": "treatment",
                                "comparator_arm_key": "comparator",
                                "label": (
                                    f"{roster.intervention_label} vs {roster.comparator_label}"
                                ),
                                "estimand": ("odds ratio for persistent H. pylori detection"),
                                "positive_direction_means": (
                                    "higher odds of persistent H. pylori detection "
                                    "in the intervention arm"
                                ),
                            }
                        ],
                        "findings": [
                            {
                                "key": "persistent",
                                "contrast_key": "primary",
                                "outcome_name": roster.outcome_name,
                                "timepoint": {
                                    "kind": "exact",
                                    "value": roster.timepoint_value,
                                    "lower": None,
                                    "upper": None,
                                    "unit": roster.timepoint_unit,
                                    "anchor": "after randomized treatment",
                                    "raw_label": roster.timepoint_label,
                                },
                                "analysis_population": None,
                                "effect": {
                                    "effect_format": "odds_ratio",
                                    "availability": "available",
                                    "estimate": None,
                                    "standard_error": None,
                                    "variance": None,
                                    "ci_lower": None,
                                    "ci_upper": None,
                                    "ci_level": 0.95,
                                    "unit": None,
                                    "treatment_mean": None,
                                    "treatment_sd": None,
                                    "treatment_n": None,
                                    "control_mean": None,
                                    "control_sd": None,
                                    "control_n": None,
                                    "treatment_events": treatment_events,
                                    "treatment_total": treatment_total,
                                    "control_events": control_events,
                                    "control_total": control_total,
                                    "reported_p_value": None,
                                    "reported_significance": roster.reported_significance,
                                    "equivalence_conclusion": "not_tested",
                                    "equivalence_margin": None,
                                    "moderators": [
                                        {
                                            "name": "follow-up-duration",
                                            "value": roster.follow_up_duration,
                                        }
                                    ],
                                    "extraction_method": "reported",
                                },
                                "evidence": {
                                    "source_locator": roster.source_locator,
                                    "quote": quote,
                                    "section": "Body",
                                    "page": None,
                                    "char_start": None,
                                    "char_end": None,
                                    "line_ids": [roster.target_line_id],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
        "non_estimability_reason": None,
        "non_estimability_detail": None,
        "warnings": [],
    }


def _validate_target_extraction(
    *,
    repository_root: Path,
    plan: HostedNativeNumericPilotPlanV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    extraction: NativePublicationExtraction,
) -> NativeGroundingReceipt:
    roster = surface.roster_record
    payload = extraction.model_dump(mode="json")
    try:
        finding = payload["studies"][0]["cohorts"][0]["findings"][0]
        effect = finding["effect"]
        evidence = finding["evidence"]
        treatment_events = effect["treatment_events"]
        treatment_total = effect["treatment_total"]
        control_events = effect["control_events"]
        control_total = effect["control_total"]
        quote = evidence["quote"]
    except (IndexError, KeyError, TypeError) as exc:
        raise HostedNativeNumericPilotError("response_target_structure_invalid") from exc
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                treatment_events,
                treatment_total,
                control_events,
                control_total,
            )
        )
        or not isinstance(quote, str)
        or treatment_events > treatment_total
        or control_events > control_total
    ):
        raise HostedNativeNumericPilotError("response_target_structure_invalid")
    observed_arms = payload["studies"][0]["cohorts"][0]["arms"]
    if not isinstance(observed_arms, list):
        raise HostedNativeNumericPilotError("response_target_structure_invalid")
    normalized_payload = json.loads(canonical_json_bytes(payload))
    normalized_payload["studies"][0]["cohorts"][0]["arms"] = sorted(
        observed_arms, key=lambda item: item.get("key", "") if isinstance(item, dict) else ""
    )
    expected = _expected_extraction_payload(
        roster=roster,
        treatment_events=treatment_events,
        treatment_total=treatment_total,
        control_events=control_events,
        control_total=control_total,
        quote=quote,
    )
    if normalized_payload != expected:
        raise HostedNativeNumericPilotError("response_target_structure_invalid")
    source_record = plan.source_manifest.records[surface.source_record_index]
    source = resolve_native_source_document(
        repository_root=repository_root,
        source_document=source_record.source_document,
    )
    if source.source_payload_sha256 != surface.resolved_source_payload_sha256:
        raise HostedNativeNumericPilotError("response_source_payload_drift")
    line = next((item for item in source.lines if item.line_id == roster.target_line_id), None)
    if line is None or _sha256_utf8(line.text) != surface.target_line_sha256:
        raise HostedNativeNumericPilotError("response_target_line_drift")
    start = line.text.find(roster.target_clause_start)
    stop = line.text.find(roster.target_clause_stop, start + 1)
    if start < 0 or stop <= start:
        raise HostedNativeNumericPilotError("response_target_clause_drift")
    clause = line.text[start:stop].rstrip()
    if _sha256_utf8(clause) != surface.target_clause_sha256 or quote not in clause:
        raise HostedNativeNumericPilotError("response_target_association_invalid")
    source_pairs = [
        (int(match.group("events")), int(match.group("total")))
        for match in _COUNT_PAIR_RE.finditer(clause)
    ]
    quote_pairs = [
        (int(match.group("events")), int(match.group("total")))
        for match in _COUNT_PAIR_RE.finditer(quote)
    ]
    if (
        len(source_pairs) < 2
        or len(quote_pairs) < 2
        or quote_pairs[:2] != source_pairs[:2]
        or (treatment_events, treatment_total) != source_pairs[0]
        or (control_events, control_total) != source_pairs[1]
    ):
        raise HostedNativeNumericPilotError("response_target_association_invalid")
    grounding = verify_native_publication_grounding(
        repository_root=repository_root,
        source_document=source_record.source_document,
        extraction=extraction,
    )
    if not grounding.authorizes_estimable_fragment:
        raise HostedNativeNumericPilotError("response_exact_grounding_invalid")
    return grounding


class HostedNativeNumericValidationV1(_Frozen):
    validation_version: Literal["hosted-native-numeric-pilot-validation-v5"] = VALIDATION_VERSION
    plan_sha256: Sha256
    request_key: NonEmpty
    observation_sha256: Sha256
    status: Literal["completed", "provider_failed"]
    failure_code: Annotated[str, Field(pattern=_SAFE_CODE_RE.pattern)] | None
    generation_schema_passed: bool
    official_schema_passed: bool
    target_association_passed: bool
    exact_grounding_passed: bool
    extraction: NativePublicationExtraction | None
    extraction_sha256: Sha256 | None
    grounding_receipt: NativeGroundingReceipt | None
    grounding_receipt_sha256: Sha256 | None
    validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> HostedNativeNumericValidationV1:
        if self.status == "completed":
            if (
                self.failure_code is not None
                or not all(
                    (
                        self.generation_schema_passed,
                        self.official_schema_passed,
                        self.target_association_passed,
                        self.exact_grounding_passed,
                    )
                )
                or self.extraction is None
                or self.extraction_sha256 != hash_canonical(self.extraction)
                or self.grounding_receipt is None
                or self.grounding_receipt_sha256 != self.grounding_receipt.receipt_sha256
            ):
                raise ValueError("hosted_numeric_completed_validation_shape_invalid")
        elif (
            self.failure_code is None
            or self.extraction is not None
            or self.extraction_sha256 is not None
            or self.grounding_receipt is not None
            or self.grounding_receipt_sha256 is not None
        ):
            raise ValueError("hosted_numeric_failed_validation_shape_invalid")
        _self_hash(self, "validation_sha256", "hosted_numeric_validation_hash_mismatch")
        return self


def _validation_failure(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    observation: HostedNativeNumericProviderObservationV1,
    failure_code: str,
    generation_schema_passed: bool = False,
    official_schema_passed: bool = False,
    target_association_passed: bool = False,
    exact_grounding_passed: bool = False,
) -> HostedNativeNumericValidationV1:
    payload = {
        "validation_version": VALIDATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "request_key": observation.request_key,
        "observation_sha256": observation.observation_sha256,
        "status": "provider_failed",
        "failure_code": failure_code,
        "generation_schema_passed": generation_schema_passed,
        "official_schema_passed": official_schema_passed,
        "target_association_passed": target_association_passed,
        "exact_grounding_passed": exact_grounding_passed,
        "extraction": None,
        "extraction_sha256": None,
        "grounding_receipt": None,
        "grounding_receipt_sha256": None,
    }
    return HostedNativeNumericValidationV1.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


def _validate_provider_observation(
    *,
    repository_root: Path,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    observation: HostedNativeNumericProviderObservationV1,
) -> HostedNativeNumericValidationV1:
    if (
        observation.plan_sha256 != plan.plan_sha256
        or observation.generation_authorization_sha256
        != generation_authorization.authorization_sha256
        or observation.hosted_intent_sha256 != surface.intent.intent_sha256
        or observation.hosted_authorization_sha256 != hosted_authorization.authorization_sha256
        or observation.request_key != surface.intent.request_key
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_provider_observation_anchor_mismatch")
    authorized = {item.request_key: item for item in generation_authorization.authorized_requests}[
        surface.intent.request_key
    ]
    if observation.content_sanitization_failure_code is not None:
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code=observation.content_sanitization_failure_code,
        )
    if observation.response_id is None:
        return _validation_failure(
            plan=plan, observation=observation, failure_code="response_identity_invalid"
        )
    if observation.response_model != MODEL:
        return _validation_failure(
            plan=plan, observation=observation, failure_code="response_model_mismatch"
        )
    if observation.stop_reason != "end_turn":
        if observation.stop_reason == "max_tokens":
            failure_code = "response_stop_reason_max_tokens"
        elif observation.stop_reason == "refusal":
            failure_code = "response_stop_reason_refusal"
        else:
            failure_code = "response_stop_reason_invalid"
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code=failure_code,
        )
    if (
        observation.text_block_count != 1
        or observation.content_text is None
        or not set(observation.non_text_block_types).issubset(_ALLOWED_NON_TEXT_CONTENT_BLOCK_TYPES)
    ):
        return _validation_failure(
            plan=plan, observation=observation, failure_code="response_content_invalid"
        )
    if observation.usage is None:
        return _validation_failure(
            plan=plan, observation=observation, failure_code="response_usage_invalid"
        )
    if (
        observation.usage.conservative_input_tokens > authorized.certified_input_token_limit
        or observation.usage.output_tokens > MAX_OUTPUT_TOKENS
        or observation.observed_cost_usd_micros is None
        or observation.observed_cost_usd_micros > authorized.certified_request_liability_usd_micros
    ):
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code="response_usage_exceeds_count_certificate",
        )
    try:
        raw = _strict_json_object(observation.content_text)
    except (TypeError, ValueError, HostedNativeNumericPilotError):
        return _validation_failure(
            plan=plan, observation=observation, failure_code="response_json_invalid"
        )
    try:
        validator_for(surface.compiled_schema.wire_schema)(
            surface.compiled_schema.wire_schema
        ).validate(raw)
        validator_for(surface.compiled_schema.original_schema)(
            surface.compiled_schema.original_schema
        ).validate(raw)
    except Exception:
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code="response_generation_schema_invalid",
        )
    try:
        validator_for(plan.official_schema.schema_payload)(
            plan.official_schema.schema_payload
        ).validate(raw)
    except Exception:
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code="response_official_schema_invalid",
            generation_schema_passed=True,
        )
    try:
        extraction = NativePublicationExtraction.model_validate(raw)
    except ValueError:
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code="response_native_schema_invalid",
            generation_schema_passed=True,
            official_schema_passed=True,
        )
    if raw != extraction.model_dump(mode="json"):
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code="response_native_canonical_mismatch",
            generation_schema_passed=True,
            official_schema_passed=True,
        )
    try:
        grounding = _validate_target_extraction(
            repository_root=repository_root,
            plan=plan,
            surface=surface,
            extraction=extraction,
        )
    except HostedNativeNumericPilotError as exc:
        failure_code = str(exc)
        if not _SAFE_CODE_RE.fullmatch(failure_code):
            failure_code = "response_target_or_grounding_invalid"
        return _validation_failure(
            plan=plan,
            observation=observation,
            failure_code=failure_code,
            generation_schema_passed=True,
            official_schema_passed=True,
        )
    payload = {
        "validation_version": VALIDATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "request_key": observation.request_key,
        "observation_sha256": observation.observation_sha256,
        "status": "completed",
        "failure_code": None,
        "generation_schema_passed": True,
        "official_schema_passed": True,
        "target_association_passed": True,
        "exact_grounding_passed": True,
        "extraction": extraction,
        "extraction_sha256": hash_canonical(extraction),
        "grounding_receipt": grounding,
        "grounding_receipt_sha256": grounding.receipt_sha256,
    }
    return HostedNativeNumericValidationV1.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


def _sanitized_failed_response(
    observation: HostedNativeNumericProviderObservationV1,
) -> str:
    if observation.content_text is not None:
        # Provider text reaches this branch only after strict object parsing and
        # recursive secret scanning in _provider_observation.
        return observation.content_text
    return canonical_json_bytes(
        {
            "content_sanitization_failure_code": (observation.content_sanitization_failure_code),
            "response_id": observation.response_id,
            "response_model": observation.response_model,
            "stop_reason": observation.stop_reason,
        }
    ).decode("utf-8")


def _terminal_from_validation(
    *,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    observation: HostedNativeNumericProviderObservationV1,
    validation: HostedNativeNumericValidationV1,
) -> HostedNativeCallTerminalV1:
    if validation.status == "completed":
        assert validation.extraction is not None
        assert observation.content_text is not None
        return freeze_hosted_native_completed_terminal_v1(
            intent=surface.intent,
            authorization=hosted_authorization,
            observed_model_id=MODEL,
            raw_response_utf8=observation.content_text,
            structured_output_json_pointer="",
            parsed_extraction=validation.extraction,
            provider_request_id=observation.response_id,
        )
    assert validation.failure_code is not None
    return freeze_hosted_native_failed_terminal_v1(
        intent=surface.intent,
        authorization=hosted_authorization,
        outcome="provider_failed",
        failure_code=validation.failure_code,
        provider_request_id=observation.response_id,
        observed_model_id=observation.response_model,
        raw_response_utf8=_sanitized_failed_response(observation),
    )


def _incident_for_exception(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    exc: Exception | None,
    orphan: bool = False,
) -> HostedNativeNumericIncidentV1:
    exception_type = type(exc).__name__ if exc is not None else None
    if exception_type is not None and not _SAFE_CODE_RE.fullmatch(exception_type):
        exception_type = "provider_exception"
    status_raw = getattr(exc, "status_code", None) if exc is not None else None
    status = (
        status_raw
        if isinstance(status_raw, int)
        and not isinstance(status_raw, bool)
        and 400 <= status_raw <= 599
        else None
    )
    provider_request_id = _safe_scalar(
        getattr(exc, "request_id", None) if exc is not None else None
    )
    if orphan:
        kind = "orphan_generation_intent_on_resume"
        terminal_outcome = "ambiguous_attempt_poison"
        failure_code = "orphan_generation_intent_on_resume"
    elif status is not None:
        kind = "provider_http_failure"
        terminal_outcome = "provider_failed"
        failure_code = f"provider_http_{status}"
    else:
        kind = "provider_call_ambiguous_exception"
        terminal_outcome = "ambiguous_attempt_poison"
        failure_code = "provider_call_ambiguous_exception"
    payload = {
        "incident_version": INCIDENT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "generation_authorization_sha256": (generation_authorization.authorization_sha256),
        "request_key": surface.intent.request_key,
        "hosted_intent_sha256": surface.intent.intent_sha256,
        "hosted_authorization_sha256": hosted_authorization.authorization_sha256,
        "kind": kind,
        "terminal_outcome": terminal_outcome,
        "failure_code": failure_code,
        "exception_type": exception_type,
        "provider_http_status": status,
        "provider_request_id": provider_request_id,
        "retry_permitted": False,
    }
    return HostedNativeNumericIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def _terminal_from_incident(
    *,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    incident: HostedNativeNumericIncidentV1,
) -> HostedNativeCallTerminalV1:
    return freeze_hosted_native_failed_terminal_v1(
        intent=surface.intent,
        authorization=hosted_authorization,
        outcome=incident.terminal_outcome,
        failure_code=incident.failure_code,
        provider_request_id=incident.provider_request_id,
    )


def _validate_incident_anchors(
    *,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    incident: HostedNativeNumericIncidentV1,
) -> None:
    if (
        incident.plan_sha256 != plan.plan_sha256
        or incident.generation_authorization_sha256 != generation_authorization.authorization_sha256
        or incident.request_key != surface.intent.request_key
        or incident.hosted_intent_sha256 != surface.intent.intent_sha256
        or incident.hosted_authorization_sha256 != hosted_authorization.authorization_sha256
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_incident_anchor_mismatch")


class HostedNativeNumericTerminalReportV1(_Frozen):
    terminal_version: Literal["hosted-native-numeric-pilot-terminal-v5"] = TERMINAL_VERSION
    status: Literal[
        "complete_with_native_numeric_yield",
        "complete_zero_native_numeric_yield",
        "terminal_scientific_budget_breach",
    ]
    plan_sha256: Sha256
    reservation_sha256: Sha256
    count_certificate_sha256: Sha256
    generation_authorization_sha256: Sha256
    canary_success_binding_sha256: Sha256
    canary_terminal_sha256: Sha256
    canary_terminal_artifact_sha256: Sha256
    hosted_run_sha256: Sha256
    hosted_run_artifact_sha256: Sha256
    manifest_records: Literal[2] = 2
    scientific_call_records_terminally_closed: Literal[2] = 2
    combined_v4_call_records_terminally_closed: Literal[3] = 3
    completed_native_extractions: Annotated[StrictInt, Field(ge=0, le=2)]
    failed_or_ambiguous_extractions: Annotated[StrictInt, Field(ge=0, le=2)]
    release_grade_native_numeric_yield: Annotated[StrictInt, Field(ge=0, le=2)]
    certified_maximum_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=2400000)]
    full_new_liability_reservation_usd_micros: Literal[2400000] = (
        NEW_LIABILITY_HARD_CEILING_USD_MICROS
    )
    companion_canary_reserved_usd_micros: Literal[600000] = CANARY_LIABILITY_ALLOCATION_USD_MICROS
    combined_v4_phase_reserved_usd_micros: Literal[3000000] = (
        COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS
    )
    canary_charged_cost_upper_bound_usd_micros: Annotated[StrictInt, Field(ge=1, le=600000)]
    combined_v4_certified_liability_usd_micros: Annotated[StrictInt, Field(ge=1, le=3000000)]
    observed_generation_cost_usd_micros: Annotated[StrictInt, Field(ge=0)]
    generation_liability_accounted_usd_micros: Annotated[StrictInt, Field(ge=1)]
    combined_v4_liability_accounted_usd_micros: Annotated[StrictInt, Field(ge=1)]
    provider_usage_missing_count: Annotated[StrictInt, Field(ge=0, le=2)]
    request_budget_breach_count: Annotated[StrictInt, Field(ge=0, le=2)]
    scientific_budget_breach_detected: bool
    certified_budget_claim_valid: bool
    all_manifest_records_terminally_closed_at_most_once: Literal[True] = True
    reference_fields_opened: Literal[False] = False
    private_predictions_opened: Literal[False] = False
    accuracy_authority: Literal[False] = False
    representativeness_authority: Literal[False] = False
    synthesis_conclusion_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> HostedNativeNumericTerminalReportV1:
        if (
            self.completed_native_extractions + self.failed_or_ambiguous_extractions != 2
            or self.release_grade_native_numeric_yield
            != (0 if self.scientific_budget_breach_detected else self.completed_native_extractions)
            or self.certified_budget_claim_valid is self.scientific_budget_breach_detected
            or (self.status == "terminal_scientific_budget_breach")
            != self.scientific_budget_breach_detected
            or (
                not self.scientific_budget_breach_detected
                and (self.status == "complete_with_native_numeric_yield")
                != (self.completed_native_extractions > 0)
            )
            or self.full_new_liability_reservation_usd_micros
            + self.companion_canary_reserved_usd_micros
            != self.combined_v4_phase_reserved_usd_micros
            or self.combined_v4_certified_liability_usd_micros
            != self.certified_maximum_liability_usd_micros
            + self.canary_charged_cost_upper_bound_usd_micros
            or self.combined_v4_certified_liability_usd_micros
            > self.combined_v4_phase_reserved_usd_micros
            or self.combined_v4_liability_accounted_usd_micros
            != self.generation_liability_accounted_usd_micros
            + self.canary_charged_cost_upper_bound_usd_micros
            or self.observed_generation_cost_usd_micros
            > self.generation_liability_accounted_usd_micros
            or self.scientific_budget_breach_detected
            != (
                self.request_budget_breach_count > 0
                or self.generation_liability_accounted_usd_micros
                > self.certified_maximum_liability_usd_micros
            )
        ):
            raise ValueError("hosted_numeric_terminal_count_mismatch")
        _self_hash(self, "terminal_sha256", "hosted_numeric_terminal_hash_mismatch")
        return self


@dataclass(frozen=True)
class HostedNativeNumericExecutionOutputV1:
    run: HostedNativeExtractionRunV1
    terminal: HostedNativeNumericTerminalReportV1


def _load_hosted_authorization(path: Path) -> HostedNativeCallAuthorizationV1:
    return HostedNativeCallAuthorizationV1.model_validate(
        _load_json_object(path, code="hosted_numeric_hosted_authorization_invalid")
    )


def _load_hosted_intent(path: Path) -> HostedNativeCallIntentV1:
    return HostedNativeCallIntentV1.model_validate(
        _load_json_object(path, code="hosted_numeric_hosted_intent_invalid")
    )


def _load_provider_observation(path: Path) -> HostedNativeNumericProviderObservationV1:
    return HostedNativeNumericProviderObservationV1.model_validate(
        _load_json_object(path, code="hosted_numeric_provider_observation_invalid")
    )


def _load_validation(path: Path) -> HostedNativeNumericValidationV1:
    return HostedNativeNumericValidationV1.model_validate(
        _load_json_object(path, code="hosted_numeric_validation_invalid")
    )


def _load_incident(path: Path) -> HostedNativeNumericIncidentV1:
    return HostedNativeNumericIncidentV1.model_validate(
        _load_json_object(path, code="hosted_numeric_incident_invalid")
    )


def _load_hosted_terminal(path: Path) -> HostedNativeCallTerminalV1:
    return HostedNativeCallTerminalV1.model_validate(
        _load_json_object(path, code="hosted_numeric_hosted_terminal_invalid")
    )


def _replay_terminal_call_chain_v4(
    *,
    repository_root: Path,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    observation_path: Path,
    validation_path: Path,
    incident_path: Path,
    terminal_path: Path,
) -> tuple[HostedNativeCallTerminalV1, tuple[int, int, int, int]]:
    has_observation = observation_path.exists()
    has_incident = incident_path.exists()
    if has_observation == has_incident:
        raise HostedNativeNumericPilotError("hosted_numeric_terminal_provenance_incomplete")
    terminal = _load_hosted_terminal(terminal_path)
    if has_observation:
        if not validation_path.exists():
            raise HostedNativeNumericPilotError("hosted_numeric_terminal_validation_missing")
        observation = _load_provider_observation(observation_path)
        replayed_validation = _validate_provider_observation(
            repository_root=repository_root,
            plan=plan,
            generation_authorization=generation_authorization,
            surface=surface,
            hosted_authorization=hosted_authorization,
            observation=observation,
        )
        validation = _load_validation(validation_path)
        if validation != replayed_validation:
            raise HostedNativeNumericPilotError("hosted_numeric_validation_replay_mismatch")
        expected_terminal = _terminal_from_validation(
            surface=surface,
            hosted_authorization=hosted_authorization,
            observation=observation,
            validation=validation,
        )
        accounting = _account_provider_observation_v4(
            authorization=generation_authorization,
            surface=surface,
            observation=observation,
        )
    else:
        if validation_path.exists():
            raise HostedNativeNumericPilotError("hosted_numeric_incident_validation_conflict")
        incident = _load_incident(incident_path)
        _validate_incident_anchors(
            plan=plan,
            generation_authorization=generation_authorization,
            surface=surface,
            hosted_authorization=hosted_authorization,
            incident=incident,
        )
        expected_terminal = _terminal_from_incident(
            surface=surface,
            hosted_authorization=hosted_authorization,
            incident=incident,
        )
        accounting = _account_unknown_provider_outcome_v4(
            authorization=generation_authorization,
            surface=surface,
        )
    if terminal != expected_terminal:
        raise HostedNativeNumericPilotError("hosted_numeric_terminal_replay_mismatch")
    return terminal, accounting


def _persist_or_replay_validation_terminal(
    *,
    repository_root: Path,
    workspace: Path,
    plan: HostedNativeNumericPilotPlanV1,
    generation_authorization: HostedNativeNumericGenerationAuthorizationV1,
    surface: HostedNativeNumericPreparedSurfaceV1,
    hosted_authorization: HostedNativeCallAuthorizationV1,
    observation: HostedNativeNumericProviderObservationV1,
    validation_path: Path,
    terminal_path: Path,
) -> HostedNativeCallTerminalV1:
    replayed = _validate_provider_observation(
        repository_root=repository_root,
        plan=plan,
        generation_authorization=generation_authorization,
        surface=surface,
        hosted_authorization=hosted_authorization,
        observation=observation,
    )
    if validation_path.exists():
        observed_validation = _load_validation(validation_path)
        if observed_validation != replayed:
            raise HostedNativeNumericPilotError("hosted_numeric_validation_replay_mismatch")
        validation = observed_validation
    else:
        _persist(validation_path, replayed)
        validation = replayed
    expected_terminal = _terminal_from_validation(
        surface=surface,
        hosted_authorization=hosted_authorization,
        observation=observation,
        validation=validation,
    )
    if terminal_path.exists():
        observed_terminal = _load_hosted_terminal(terminal_path)
        if observed_terminal != expected_terminal:
            raise HostedNativeNumericPilotError("hosted_numeric_terminal_replay_mismatch")
        return observed_terminal
    _persist(terminal_path, expected_terminal)
    return expected_terminal


def _runtime_terminal(
    *,
    workspace: Path,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    certificate: HostedNativeNumericCountCertificateV1,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
    run: HostedNativeExtractionRunV1,
    observed_generation_cost_usd_micros: int,
    generation_liability_accounted_usd_micros: int,
    provider_usage_missing_count: int,
    request_budget_breach_count: int,
) -> HostedNativeNumericTerminalReportV1:
    run_path = _artifact(workspace, Path("04-hosted-native-extraction-run-v1.json"))
    if (
        certificate.canary_success_binding_sha256 != plan.canary_success_binding_sha256
        or authorization.canary_success_binding_sha256 != plan.canary_success_binding_sha256
        or authorization.canary_terminal_sha256 != plan.canary_terminal_sha256
        or authorization.canary_terminal_artifact_sha256 != plan.canary_terminal_artifact_sha256
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_terminal_canary_binding_mismatch")
    completed = run.completed_extraction_count
    scientific_budget_breach_detected = (
        request_budget_breach_count > 0
        or generation_liability_accounted_usd_micros
        > authorization.certified_maximum_liability_usd_micros
    )
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "status": (
            "terminal_scientific_budget_breach"
            if scientific_budget_breach_detected
            else (
                "complete_with_native_numeric_yield"
                if completed
                else "complete_zero_native_numeric_yield"
            )
        ),
        "plan_sha256": plan.plan_sha256,
        "reservation_sha256": reservation.reservation_sha256,
        "count_certificate_sha256": certificate.certificate_sha256,
        "generation_authorization_sha256": authorization.authorization_sha256,
        "canary_success_binding_sha256": plan.canary_success_binding_sha256,
        "canary_terminal_sha256": plan.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": plan.canary_terminal_artifact_sha256,
        "hosted_run_sha256": run.run_sha256,
        "hosted_run_artifact_sha256": sha256_file(run_path),
        "manifest_records": 2,
        "scientific_call_records_terminally_closed": 2,
        "combined_v4_call_records_terminally_closed": 3,
        "completed_native_extractions": completed,
        "failed_or_ambiguous_extractions": run.failed_or_ambiguous_count,
        "release_grade_native_numeric_yield": (
            0 if scientific_budget_breach_detected else completed
        ),
        "certified_maximum_liability_usd_micros": (
            authorization.certified_maximum_liability_usd_micros
        ),
        "full_new_liability_reservation_usd_micros": (NEW_LIABILITY_HARD_CEILING_USD_MICROS),
        "companion_canary_reserved_usd_micros": CANARY_LIABILITY_ALLOCATION_USD_MICROS,
        "combined_v4_phase_reserved_usd_micros": (COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS),
        "canary_charged_cost_upper_bound_usd_micros": (
            plan.canary_success_binding.charged_cost_upper_bound_usd_micros
        ),
        "combined_v4_certified_liability_usd_micros": (
            authorization.combined_v4_certified_liability_usd_micros
        ),
        "observed_generation_cost_usd_micros": observed_generation_cost_usd_micros,
        "generation_liability_accounted_usd_micros": (generation_liability_accounted_usd_micros),
        "combined_v4_liability_accounted_usd_micros": (
            generation_liability_accounted_usd_micros
            + plan.canary_success_binding.charged_cost_upper_bound_usd_micros
        ),
        "provider_usage_missing_count": provider_usage_missing_count,
        "request_budget_breach_count": request_budget_breach_count,
        "scientific_budget_breach_detected": scientific_budget_breach_detected,
        "certified_budget_claim_valid": not scientific_budget_breach_detected,
        "all_manifest_records_terminally_closed_at_most_once": True,
        "reference_fields_opened": False,
        "private_predictions_opened": False,
        "accuracy_authority": False,
        "representativeness_authority": False,
        "synthesis_conclusion_authority": False,
        "claim_release_authority": False,
    }
    return HostedNativeNumericTerminalReportV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _replay_completed_execution_v4(
    *,
    repository_root: Path,
    workspace: Path,
    plan: HostedNativeNumericPilotPlanV1,
    reservation: HostedNativeNumericReservationV1,
    certificate: HostedNativeNumericCountCertificateV1,
    authorization: HostedNativeNumericGenerationAuthorizationV1,
    persist_missing_terminal: bool = False,
) -> HostedNativeNumericExecutionOutputV1:
    """Rebuild a completed run and terminal from every durable call artifact.

    A process may stop after the immutable hosted run is persisted but before the
    summary terminal is written. In that state every provider call is already
    terminally closed. ``persist_missing_terminal`` repairs only the deterministic
    summary artifact and never permits another transport attempt.
    """

    run_path = _artifact(workspace, Path("04-hosted-native-extraction-run-v1.json"))
    terminal_report_path = _artifact(workspace, Path("05-terminal.json"))
    if not run_path.exists():
        raise HostedNativeNumericPilotError("hosted_numeric_hosted_run_required")

    calls: list[HostedNativeCallV1] = []
    observed_total_cost = 0
    generation_liability_accounted = 0
    provider_usage_missing_count = 0
    request_budget_breach_count = 0
    for surface in plan.surfaces:
        key = surface.intent.request_key
        hosted_auth_path = _artifact(
            workspace,
            Path("call-authorizations") / f"{key}.json",
        )
        intent_path = _artifact(workspace, Path("generation-intents") / f"{key}.json")
        observation_path = _artifact(workspace, Path("provider-receipts") / f"{key}.json")
        validation_path = _artifact(workspace, Path("validations") / f"{key}.json")
        incident_path = _artifact(workspace, Path("incidents") / f"{key}.json")
        terminal_path = _artifact(workspace, Path("call-terminals") / f"{key}.json")
        expected_hosted_auth = freeze_hosted_native_call_authorization_v1(
            intent=surface.intent,
            provider_identity=plan.provider_identity,
        )
        observed_hosted_auth = _load_hosted_authorization(hosted_auth_path)
        observed_intent = _load_hosted_intent(intent_path)
        if observed_hosted_auth != expected_hosted_auth:
            raise HostedNativeNumericPilotError("hosted_numeric_call_authorization_drift")
        if observed_intent != surface.intent:
            raise HostedNativeNumericPilotError("hosted_numeric_intent_drift")
        terminal, accounting = _replay_terminal_call_chain_v4(
            repository_root=repository_root,
            plan=plan,
            generation_authorization=authorization,
            surface=surface,
            hosted_authorization=observed_hosted_auth,
            observation_path=observation_path,
            validation_path=validation_path,
            incident_path=incident_path,
            terminal_path=terminal_path,
        )
        observed, accounted, missing, breach = accounting
        observed_total_cost += observed
        generation_liability_accounted += accounted
        provider_usage_missing_count += missing
        request_budget_breach_count += breach
        calls.append(
            freeze_hosted_native_call_v1(
                intent=surface.intent,
                authorization=observed_hosted_auth,
                terminal=terminal,
            )
        )

    expected_run = freeze_hosted_native_extraction_run_v1(
        run_id=plan.config.run_id,
        question_config=plan.question_config,
        source_manifest=plan.source_manifest,
        corpus_cutoff=plan.config.corpus_cutoff,
        pipeline_fingerprint=plan.pipeline_fingerprint,
        prompts=[surface.prompt for surface in plan.surfaces],
        schemas=[
            plan.official_schema,
            *(surface.generation_schema for surface in plan.surfaces),
        ],
        provider_identity=plan.provider_identity,
        calls=calls,
    )
    observed_run = HostedNativeExtractionRunV1.model_validate(
        _load_json_object(run_path, code="hosted_numeric_hosted_run_invalid")
    )
    if observed_run != expected_run:
        raise HostedNativeNumericPilotError("hosted_numeric_hosted_run_replay_mismatch")
    expected_report = _runtime_terminal(
        workspace=workspace,
        plan=plan,
        reservation=reservation,
        certificate=certificate,
        authorization=authorization,
        run=observed_run,
        observed_generation_cost_usd_micros=observed_total_cost,
        generation_liability_accounted_usd_micros=(generation_liability_accounted),
        provider_usage_missing_count=provider_usage_missing_count,
        request_budget_breach_count=request_budget_breach_count,
    )
    if terminal_report_path.exists():
        observed_report = HostedNativeNumericTerminalReportV1.model_validate(
            _load_json_object(
                terminal_report_path,
                code="hosted_numeric_terminal_report_required",
            )
        )
        if observed_report != expected_report:
            raise HostedNativeNumericPilotError("hosted_numeric_terminal_report_replay_mismatch")
    elif persist_missing_terminal:
        _persist(terminal_report_path, expected_report)
        observed_report = expected_report
    else:
        observed_report = expected_report
    return HostedNativeNumericExecutionOutputV1(
        run=observed_run,
        terminal=observed_report,
    )


def preflight_hosted_native_numeric_execution_v1(
    *,
    repository_root: Path,
    workspace: Path,
    expected_plan_sha256: str,
    expected_generation_authorization_sha256: str,
) -> bool:
    """Validate every execution anchor before credentials or a client exist.

    The return value is true only when at least one manifest record has no durable
    generation intent and therefore still requires a provider transport attempt.
    """

    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        certificate = _load_count_certificate(root)
        authorization = _load_generation_authorization(root)
        _validate_generation_authorization_state_v4(
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            authorization=authorization,
        )
        if (
            plan.plan_sha256 != expected_plan_sha256
            or authorization.authorization_sha256 != expected_generation_authorization_sha256
            or authorization.plan_sha256 != plan.plan_sha256
            or authorization.reservation_sha256 != reservation.reservation_sha256
            or authorization.count_certificate_sha256 != certificate.certificate_sha256
            or authorization.canary_success_binding_sha256 != plan.canary_success_binding_sha256
            or authorization.canary_terminal_sha256 != plan.canary_terminal_sha256
            or certificate.canary_success_binding_sha256 != plan.canary_success_binding_sha256
            or reservation.project_liability_after_reservation_usd_micros
            >= PROJECT_LIABILITY_HARD_STOP_USD_MICROS
        ):
            raise HostedNativeNumericPilotError("hosted_numeric_execution_anchor_mismatch")
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        run_path = _artifact(root, Path("04-hosted-native-extraction-run-v1.json"))
        terminal_report_path = _artifact(root, Path("05-terminal.json"))
        if terminal_report_path.exists() and not run_path.exists():
            raise HostedNativeNumericPilotError("hosted_numeric_terminal_without_hosted_run")
        if run_path.exists():
            _replay_completed_execution_v4(
                repository_root=repository,
                workspace=root,
                plan=plan,
                reservation=reservation,
                certificate=certificate,
                authorization=authorization,
            )
            return False
        transport_required = False
        for surface in plan.surfaces:
            key = surface.intent.request_key
            hosted_auth_path = _artifact(
                root,
                Path("call-authorizations") / f"{key}.json",
            )
            intent_path = _artifact(root, Path("generation-intents") / f"{key}.json")
            observation_path = _artifact(root, Path("provider-receipts") / f"{key}.json")
            validation_path = _artifact(root, Path("validations") / f"{key}.json")
            incident_path = _artifact(root, Path("incidents") / f"{key}.json")
            terminal_path = _artifact(root, Path("call-terminals") / f"{key}.json")
            downstream_paths = (
                observation_path,
                validation_path,
                incident_path,
                terminal_path,
            )
            if not intent_path.exists() and any(path.exists() for path in downstream_paths):
                raise HostedNativeNumericPilotError(
                    "hosted_numeric_generation_state_without_intent"
                )
            expected_hosted_auth = freeze_hosted_native_call_authorization_v1(
                intent=surface.intent,
                provider_identity=plan.provider_identity,
            )
            if hosted_auth_path.exists():
                hosted_auth = _load_hosted_authorization(hosted_auth_path)
                if hosted_auth != expected_hosted_auth:
                    raise HostedNativeNumericPilotError("hosted_numeric_call_authorization_drift")
            if not intent_path.exists():
                if any(path.exists() for path in downstream_paths):
                    raise HostedNativeNumericPilotError(
                        "hosted_numeric_generation_state_without_intent"
                    )
                transport_required = True
                continue
            if not hosted_auth_path.exists():
                raise HostedNativeNumericPilotError(
                    "hosted_numeric_generation_state_without_authorization"
                )
            observed_intent = _load_hosted_intent(intent_path)
            if observed_intent != surface.intent:
                raise HostedNativeNumericPilotError("hosted_numeric_intent_drift")
            if terminal_path.exists():
                _replay_terminal_call_chain_v4(
                    repository_root=repository,
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    observation_path=observation_path,
                    validation_path=validation_path,
                    incident_path=incident_path,
                    terminal_path=terminal_path,
                )
            elif observation_path.exists():
                if incident_path.exists():
                    raise HostedNativeNumericPilotError("hosted_numeric_provider_state_conflict")
                observation = _load_provider_observation(observation_path)
                replayed = _validate_provider_observation(
                    repository_root=repository,
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    observation=observation,
                )
                if validation_path.exists() and _load_validation(validation_path) != replayed:
                    raise HostedNativeNumericPilotError("hosted_numeric_validation_replay_mismatch")
            elif incident_path.exists():
                if validation_path.exists():
                    raise HostedNativeNumericPilotError(
                        "hosted_numeric_incident_validation_conflict"
                    )
                _validate_incident_anchors(
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    incident=_load_incident(incident_path),
                )
            elif validation_path.exists():
                raise HostedNativeNumericPilotError(
                    "hosted_numeric_validation_without_provider_state"
                )
        return transport_required


def execute_hosted_native_numeric_pilot_v1(
    *,
    repository_root: Path,
    workspace: Path,
    expected_plan_sha256: str,
    expected_generation_authorization_sha256: str,
    client: HostedNativeNumericGenerationClientProtocol,
) -> HostedNativeNumericExecutionOutputV1:
    """Attempt every manifest record once and materialize the bridge-native run."""

    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        certificate = _load_count_certificate(root)
        authorization = _load_generation_authorization(root)
        _validate_generation_authorization_state_v4(
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            authorization=authorization,
        )
        if (
            plan.plan_sha256 != expected_plan_sha256
            or authorization.authorization_sha256 != expected_generation_authorization_sha256
            or authorization.plan_sha256 != plan.plan_sha256
            or authorization.reservation_sha256 != reservation.reservation_sha256
            or authorization.count_certificate_sha256 != certificate.certificate_sha256
            or authorization.canary_success_binding_sha256 != plan.canary_success_binding_sha256
            or authorization.canary_terminal_sha256 != plan.canary_terminal_sha256
            or certificate.canary_success_binding_sha256 != plan.canary_success_binding_sha256
            or reservation.project_liability_after_reservation_usd_micros
            >= PROJECT_LIABILITY_HARD_STOP_USD_MICROS
        ):
            raise HostedNativeNumericPilotError("hosted_numeric_execution_anchor_mismatch")
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        run_path = _artifact(root, Path("04-hosted-native-extraction-run-v1.json"))
        terminal_report_path = _artifact(root, Path("05-terminal.json"))
        if terminal_report_path.exists() and not run_path.exists():
            raise HostedNativeNumericPilotError("hosted_numeric_terminal_without_hosted_run")
        if run_path.exists():
            return _replay_completed_execution_v4(
                repository_root=repository,
                workspace=root,
                plan=plan,
                reservation=reservation,
                certificate=certificate,
                authorization=authorization,
                persist_missing_terminal=True,
            )
        calls: list[HostedNativeCallV1] = []
        observed_total_cost = 0
        generation_liability_accounted = 0
        provider_usage_missing_count = 0
        request_budget_breach_count = 0

        def add_accounting(accounting: tuple[int, int, int, int]) -> None:
            nonlocal observed_total_cost
            nonlocal generation_liability_accounted
            nonlocal provider_usage_missing_count
            nonlocal request_budget_breach_count
            observed, accounted, missing, breach = accounting
            observed_total_cost += observed
            generation_liability_accounted += accounted
            provider_usage_missing_count += missing
            request_budget_breach_count += breach

        for surface in plan.surfaces:
            key = surface.intent.request_key
            hosted_auth_path = _artifact(root, Path("call-authorizations") / f"{key}.json")
            intent_path = _artifact(root, Path("generation-intents") / f"{key}.json")
            observation_path = _artifact(root, Path("provider-receipts") / f"{key}.json")
            validation_path = _artifact(root, Path("validations") / f"{key}.json")
            incident_path = _artifact(root, Path("incidents") / f"{key}.json")
            terminal_path = _artifact(root, Path("call-terminals") / f"{key}.json")
            downstream_paths = (
                observation_path,
                validation_path,
                incident_path,
                terminal_path,
            )
            if not intent_path.exists() and any(path.exists() for path in downstream_paths):
                raise HostedNativeNumericPilotError(
                    "hosted_numeric_generation_state_without_intent"
                )
            expected_hosted_auth = freeze_hosted_native_call_authorization_v1(
                intent=surface.intent,
                provider_identity=plan.provider_identity,
            )
            if hosted_auth_path.exists():
                hosted_auth = _load_hosted_authorization(hosted_auth_path)
                if hosted_auth != expected_hosted_auth:
                    raise HostedNativeNumericPilotError("hosted_numeric_call_authorization_drift")
            else:
                _persist(hosted_auth_path, expected_hosted_auth)
                hosted_auth = expected_hosted_auth
            if terminal_path.exists():
                if not intent_path.exists():
                    raise HostedNativeNumericPilotError("hosted_numeric_terminal_without_intent")
                observed_intent = _load_hosted_intent(intent_path)
                if observed_intent != surface.intent:
                    raise HostedNativeNumericPilotError("hosted_numeric_intent_drift")
                terminal, accounting = _replay_terminal_call_chain_v4(
                    repository_root=repository,
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    observation_path=observation_path,
                    validation_path=validation_path,
                    incident_path=incident_path,
                    terminal_path=terminal_path,
                )
                calls.append(
                    freeze_hosted_native_call_v1(
                        intent=surface.intent,
                        authorization=hosted_auth,
                        terminal=terminal,
                    )
                )
                add_accounting(accounting)
                continue
            if intent_path.exists():
                observed_intent = _load_hosted_intent(intent_path)
                if observed_intent != surface.intent:
                    raise HostedNativeNumericPilotError("hosted_numeric_intent_drift")
                if observation_path.exists():
                    observation = _load_provider_observation(observation_path)
                    terminal = _persist_or_replay_validation_terminal(
                        repository_root=repository,
                        workspace=root,
                        plan=plan,
                        generation_authorization=authorization,
                        surface=surface,
                        hosted_authorization=hosted_auth,
                        observation=observation,
                        validation_path=validation_path,
                        terminal_path=terminal_path,
                    )
                    add_accounting(
                        _account_provider_observation_v4(
                            authorization=authorization,
                            surface=surface,
                            observation=observation,
                        )
                    )
                elif incident_path.exists():
                    incident = _load_incident(incident_path)
                    _validate_incident_anchors(
                        plan=plan,
                        generation_authorization=authorization,
                        surface=surface,
                        hosted_authorization=hosted_auth,
                        incident=incident,
                    )
                    terminal = _terminal_from_incident(
                        surface=surface,
                        hosted_authorization=hosted_auth,
                        incident=incident,
                    )
                    _persist(terminal_path, terminal)
                    add_accounting(
                        _account_unknown_provider_outcome_v4(
                            authorization=authorization,
                            surface=surface,
                        )
                    )
                else:
                    incident = _incident_for_exception(
                        plan=plan,
                        generation_authorization=authorization,
                        surface=surface,
                        hosted_authorization=hosted_auth,
                        exc=None,
                        orphan=True,
                    )
                    _persist(incident_path, incident)
                    terminal = _terminal_from_incident(
                        surface=surface,
                        hosted_authorization=hosted_auth,
                        incident=incident,
                    )
                    _persist(terminal_path, terminal)
                    add_accounting(
                        _account_unknown_provider_outcome_v4(
                            authorization=authorization,
                            surface=surface,
                        )
                    )
                calls.append(
                    freeze_hosted_native_call_v1(
                        intent=surface.intent,
                        authorization=hosted_auth,
                        terminal=terminal,
                    )
                )
                continue
            _persist(intent_path, surface.intent)
            try:
                raw = client.generate(json.loads(surface.intent.wire_request_utf8))
                observation = _provider_observation(
                    plan=plan,
                    generation_authorization=authorization,
                    hosted_intent=surface.intent,
                    hosted_authorization=hosted_auth,
                    raw=raw,
                )
                _persist(observation_path, observation)
                terminal = _persist_or_replay_validation_terminal(
                    repository_root=repository,
                    workspace=root,
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    observation=observation,
                    validation_path=validation_path,
                    terminal_path=terminal_path,
                )
                add_accounting(
                    _account_provider_observation_v4(
                        authorization=authorization,
                        surface=surface,
                        observation=observation,
                    )
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                    raise
                if observation_path.exists():
                    # A response is already durable. Validation bugs must not be
                    # mislabeled as an ambiguous transport attempt or retried.
                    raise
                incident = _incident_for_exception(
                    plan=plan,
                    generation_authorization=authorization,
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    exc=exc,
                )
                _persist(incident_path, incident)
                terminal = _terminal_from_incident(
                    surface=surface,
                    hosted_authorization=hosted_auth,
                    incident=incident,
                )
                _persist(terminal_path, terminal)
                add_accounting(
                    _account_unknown_provider_outcome_v4(
                        authorization=authorization,
                        surface=surface,
                    )
                )
            calls.append(
                freeze_hosted_native_call_v1(
                    intent=surface.intent,
                    authorization=hosted_auth,
                    terminal=terminal,
                )
            )
        run = freeze_hosted_native_extraction_run_v1(
            run_id=plan.config.run_id,
            question_config=plan.question_config,
            source_manifest=plan.source_manifest,
            corpus_cutoff=plan.config.corpus_cutoff,
            pipeline_fingerprint=plan.pipeline_fingerprint,
            prompts=[surface.prompt for surface in plan.surfaces],
            schemas=[
                plan.official_schema,
                *(surface.generation_schema for surface in plan.surfaces),
            ],
            provider_identity=plan.provider_identity,
            calls=calls,
        )
        _persist(run_path, run)
        report = _runtime_terminal(
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            authorization=authorization,
            run=run,
            observed_generation_cost_usd_micros=observed_total_cost,
            generation_liability_accounted_usd_micros=(generation_liability_accounted),
            provider_usage_missing_count=provider_usage_missing_count,
            request_budget_breach_count=request_budget_breach_count,
        )
        _persist(terminal_report_path, report)
        return HostedNativeNumericExecutionOutputV1(run=run, terminal=report)


def load_hosted_native_numeric_terminal_v1(
    *, repository_root: Path = Path("."), workspace: Path
) -> HostedNativeNumericTerminalReportV1:
    root = _existing_workspace(workspace)
    repository = repository_root.resolve(strict=True)
    if not _artifact(root, Path("05-terminal.json")).exists():
        raise HostedNativeNumericPilotError("hosted_numeric_terminal_report_required")
    with _workspace_lock(root):
        plan = load_hosted_native_numeric_pilot_plan_v1(workspace=root)
        _validate_workspace_artifact_namespace_v4(workspace=root, plan=plan)
        reservation = _load_reservation(root)
        certificate = _load_count_certificate(root)
        authorization = _load_generation_authorization(root)
        _validate_generation_authorization_state_v4(
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            authorization=authorization,
        )
        validate_hosted_native_numeric_plan_prerequisites_v4(
            repository_root=repository,
            plan=plan,
        )
        return _replay_completed_execution_v4(
            repository_root=repository,
            workspace=root,
            plan=plan,
            reservation=reservation,
            certificate=certificate,
            authorization=authorization,
        ).terminal


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_WORKSPACE",
    "AnthropicHostedNativeNumericGenerationClientV1",
    "AnthropicHostedNativeNumericTokenCounterV1",
    "HostedNativeNumericCountCertificateV1",
    "HostedNativeNumericExecutionOutputV1",
    "HostedNativeNumericGenerationAuthorizationV1",
    "HostedNativeNumericPilotConfigV1",
    "HostedNativeNumericPilotError",
    "HostedNativeNumericPilotPlanV1",
    "HostedNativeNumericRawResponseV1",
    "HostedNativeNumericReservationV1",
    "HostedNativeNumericTerminalReportV1",
    "authorize_hosted_native_numeric_pilot_v1",
    "count_hosted_native_numeric_pilot_tokens_v1",
    "execute_hosted_native_numeric_pilot_v1",
    "freeze_hosted_native_numeric_generation_authorization_v1",
    "freeze_hosted_native_numeric_pilot_plan_v1",
    "freeze_hosted_native_numeric_reservation_v1",
    "load_hosted_native_numeric_pilot_config_v1",
    "load_hosted_native_numeric_pilot_plan_v1",
    "load_hosted_native_numeric_terminal_v1",
    "preflight_hosted_native_numeric_count_v1",
    "preflight_hosted_native_numeric_execution_v1",
    "prepare_hosted_native_numeric_pilot_v1",
    "reserve_hosted_native_numeric_pilot_v1",
]
