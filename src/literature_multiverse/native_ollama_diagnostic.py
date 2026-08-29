"""Staged local-Ollama diagnostic for native numerical evidence extraction.

The diagnostic has three physically separate stages:

``prepare``
    Verify a frozen source-bridge run and project only authoritative source-line
    bytes into a self-hashed, label-blind input bundle.
``predict``
    Give that bundle (and nothing else) to an exact localhost model/runtime and
    freeze one immutable generation receipt per publication.
``finalize``
    Post-validate the frozen responses against the official native extraction
    contract, ground every result against current source bytes, build a complete
    v4 grounding package, replay it, and run the unified verifier.

This is a diagnostic over a historically opened, legacy-eligibility-selected subset.
There are no independent numerical gold labels, so this module deliberately exposes
yield and mechanical-grounding measurements rather than extraction accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from literature_multiverse.certificate import write_certificate_artifacts
from literature_multiverse.config import load_config_for_question
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaClientProtocol,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    NativeEvaluationSchemaArtifact,
    NativeExtractionArtifactDigest,
    NativeGroundingReceipt,
    NativeRenderedPromptArtifact,
    freeze_grounding_checked_publication_fragment,
    freeze_native_extraction_execution_context,
    freeze_native_provider_execution_receipt,
    freeze_typed_evidence_grounding_package,
    resolve_native_source_document,
    reverify_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.prompting import render_prompt_text
from literature_multiverse.source_manifest_bridge import DiagnosticSourceLedger
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    assemble_typed_evidence_corpus,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    LegacyAdapterConfig,
    compute_verifier_pipeline_fingerprint,
    load_corpus,
    run_verification,
)

NATIVE_OLLAMA_DIAGNOSTIC_VERSION = "native-antiox-local-ollama-diagnostic-v2"
INPUT_BUNDLE_VERSION = "native-ollama-input-bundle-v1"
GENERATION_RECEIPT_VERSION = "native-ollama-generation-receipt-v1"
PREDICTION_LEDGER_VERSION = "native-ollama-prediction-ledger-v2"
PRIVATE_REPORT_VERSION = "native-ollama-private-report-v2"
PUBLIC_SUMMARY_VERSION = "native-ollama-public-summary-v2"
GENERATION_SCHEMA_ALGORITHM = (
    "official-native-schema-minus-regex-plus-row-enums-fully-typed-singleton-enum-"
    "status-branches-and-zero-projection-nonestimable-only-v5"
)

SCHEMA_COMPATIBILITY_PREFLIGHT_VERSION = (
    "native-ollama-expanded-schema-compatibility-preflight-v1"
)
_SCHEMA_COMPATIBILITY_LINE_ID = "SCHEMA_COMPATIBILITY_LINE"
_SCHEMA_COMPATIBILITY_SOURCE_LOCATOR = "synthetic:schema-compatibility-only"
_SCHEMA_COMPATIBILITY_PROMPT = """\
This is a synthetic JSON-Schema compatibility check. It contains no publication,
scientific claim, source text, eligibility label, or expected answer. Return exactly
this JSON object:
{"extraction_schema_version":"native-publication-extraction-v1","status":"non_estimable","studies":[],"non_estimability_reason":"numerical_result_absent","non_estimability_detail":null,"warnings":[]}
"""

EXPECTED_QUESTION_ID = "antiox-training"
EXPECTED_SELECTION_SCOPE = "legacy_eligible"
EXPECTED_MANIFEST_RECORDS = 19
EXPECTED_CORPUS_CUTOFF = "antiox-legacy-eligible-diagnostic-2026-08-27"
EXPECTED_BRIDGE_RUN_SHA256 = "62d87d08b116d2af95da3e8646a293f9cdaf3507fd8b3e84319b2007367f26b3"

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_MODEL_DIGEST = "baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878"
DEFAULT_OLLAMA_VERSION = "0.15.1"
DEFAULT_GENERATION_CONFIG = OllamaGenerationConfig(
    model=DEFAULT_MODEL,
    model_digest=DEFAULT_MODEL_DIGEST,
    expected_ollama_version=DEFAULT_OLLAMA_VERSION,
    seed=20260827,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    num_ctx=16384,
    num_predict=3072,
    keep_alive="30m",
)

DEFAULT_PROJECTION_CONFIG: dict[str, Any] = {
    "algorithm": "label-blind-source-line-priority-v1",
    "allowed_sections": ["FigureTable", "Methods", "Results"],
    "max_passage_characters": 1800,
    "max_projected_characters": 14000,
    "max_projected_passages": 24,
    "reserved_methods_passages": 3,
    "endpoint_terms": [
        "1-rm",
        "aerobic",
        "endurance",
        "fat-free mass",
        "hypertrophy",
        "lean mass",
        "maximal oxygen",
        "mitochond",
        "muscle mass",
        "performance",
        "power",
        "sprint",
        "strength",
        "time trial",
        "vo2",
    ],
    "exposure_terms": [
        "antioxidant",
        "ascorb",
        "supplement",
        "tocopherol",
        "vitamin c",
        "vitamin e",
    ],
    "comparator_terms": ["control", "placebo"],
}

_SOURCE_BLOCK_TEMPLATE = """

## Frozen authoritative source-line projection

The source projection below is the only publication content available for this call.
Copy `source_locator` exactly as `{source_locator}` for every finding. Cite only exposed
bare line IDs. A quote must be an exact contiguous substring between source-text
delimiters. `section` must match the displayed section exactly or be JSON null. Set
page, char_start, and char_end to JSON null. If the projection does not contain a safely
estimable target result with uncertainty or complete group statistics, return a
non-estimable extraction; never guess or repair missing values.

{passages}
"""

_NUMBER_SIGNAL = re.compile(
    r"(?:\d|%|\bp\s*[<=>\u2264\u2265]|confidence interval|\bci\b|standard deviation|"
    r"\bsd\b|standard error|\bse\b|\u00b1)",
    flags=re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_ABSOLUTE_PATH = re.compile(
    r"(?:(?<![:/A-Za-z0-9])/(?!/)[^\s\"]|[A-Za-z]:[\\/]|\\\\)"
)
_PUBLIC_ARTICLE_ID = re.compile(
    r"(?:\b(?:PMC|PMID)[\s:/_-]*[0-9]+\b|"
    r"\b(?:antiox-publication|antiox-paper|publication|paper):)",
    re.IGNORECASE,
)
_PUBLIC_DOI = re.compile(r"\b10\.[0-9]{4,9}/[^\s\"<>]+", re.IGNORECASE)
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "anchor_papers",
        "expected_direction",
        "expected_directions",
        "expected_finding_count",
        "expected_output",
        "finding_rows",
        "gold_label",
        "labels",
        "legacy_findings",
        "legacy_outputs",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "doc_id",
        "evidence_graph",
        "example_id",
        "input_rows",
        "native_source_manifest",
        "paper_id",
        "parsed_output",
        "passages",
        "prediction",
        "prompt",
        "publication_id",
        "quote",
        "records",
        "response_text",
        "source_lines",
        "source_locator",
    }
)
_PUBLIC_AGGREGATE_CODE = re.compile(r"^[a-z0-9_:.-]{1,160}$")
_PUBLIC_CAVEATS = [
    "The 19-record subset was selected from historically opened legacy "
    "eligibility decisions.",
    "There are no independent native numerical extraction gold labels, so "
    "no accuracy is reported.",
    "Exact grounding is mechanical byte/line containment, not semantic entailment.",
    "The source manifest is a diagnostic subset, not evidence of protocol-wide "
    "retrieval completeness.",
    "The diagnostic uses parsed source lines only; tables, figures, and supplements "
    "are not multimodally read.",
    "Generation call and retry counts cover response-bearing terminal receipts only; "
    "a client-side transport failure after a POST may represent an unobserved local-model "
    "execution and is not durably countable.",
    "Synthetic schema-compatibility preflight calls are outside paper-level generation "
    "counters.",
    "The certificate verifies literature support under the supplied corpus and "
    "abstains; it is not scientific truth.",
]
_INPUT_BUNDLE_FIELDS = {
    "claim_manifest_sha256",
    "config_file_sha256",
    "contains_anchor_expectations",
    "contains_downstream_claim_payload",
    "contains_legacy_directions",
    "contains_legacy_findings",
    "corpus_cutoff",
    "diagnostic_config_path",
    "diagnostic_execution_identity",
    "diagnostic_execution_sha256",
    "diagnostic_version",
    "generation_config",
    "generation_config_sha256",
    "generation_schema_algorithm",
    "input_bundle_sha256",
    "input_bundle_version",
    "official_schema_sha256",
    "pipeline_fingerprint",
    "pipeline_sha256",
    "prediction_stage_can_open_source_or_label_files",
    "pristine_final_holdout_eligible",
    "projection_config",
    "projection_config_sha256",
    "prompt_template_file_sha256",
    "prompt_version",
    "question_config_file_sha256",
    "question_spec_sha256",
    "rendered_base_prompt",
    "rendered_base_prompt_sha256",
    "row_count",
    "rows",
    "scientific_role",
    "selection_labels_previously_opened",
    "selection_scope",
    "source_bridge_run_file_sha256",
    "source_bridge_run_sha256",
    "source_manifest_content_sha256",
    "source_manifest_file_sha256",
    "source_manifest_records",
    "status",
}
_INPUT_ROW_FIELDS = {
    "input_row_sha256",
    "projected_characters",
    "projected_passages",
    "row_key",
    "source_payload_sha256",
    "source_projection",
    "source_projection_sha256",
    "source_record",
}
_PROJECTION_PASSAGE_FIELDS = {
    "line_id",
    "line_number",
    "passage_rank",
    "ranking",
    "section",
    "source_line_end_exclusive",
    "source_line_start",
    "text",
}
_PROJECTION_RANKING_FIELDS = {
    "comparator_term_hits",
    "endpoint_term_hits",
    "exposure_term_hits",
    "numerical_signal",
    "score",
}


class NativeOllamaDiagnosticError(ValueError):
    """A diagnostic artifact or stage boundary violated its frozen contract."""


class _ModelJSONError(ValueError):
    """A model response is not strict, finite, duplicate-free JSON."""


def _strict_model_json_loads(value: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise _ModelJSONError(f"nonfinite_json_constant:{constant}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise _ModelJSONError("duplicate_json_object_key")
            output[key] = item
        return output

    try:
        return json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise _ModelJSONError("json_decode_error") from exc


class NativeDiagnosticConfig(BaseModel):
    """Closed, anchor-free projection of the locked Antiox question configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: Literal["native-antiox-ollama-config-v1"]
    question_id: Literal["antiox-training"]
    question_config_path: Literal["configs/questions/antiox-training.yaml"]
    question_config_file_sha256: str
    question_spec: dict[str, Any]
    question_spec_sha256: str
    prompt_path: Literal["prompts/native_extraction.md"]
    prompt_file_sha256: str
    source_manifest_path: Literal[
        "data/cache/native-source-v1/antiox-eligible/native_source_manifest.json"
    ]
    bridge_run_path: Literal[
        "data/cache/native-source-v1/antiox-eligible/source_manifest_bridge_run.json"
    ]
    bridge_run_sha256: str
    selection_scope: Literal["legacy_eligible"]
    source_manifest_records: Literal[19]
    corpus_cutoff: Literal["antiox-legacy-eligible-diagnostic-2026-08-27"]
    projection: dict[str, Any]
    claim_manifest: dict[str, Any]

    @field_validator(
        "question_config_file_sha256",
        "question_spec_sha256",
        "prompt_file_sha256",
        "bridge_run_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("native_diagnostic_config_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> NativeDiagnosticConfig:
        if self.bridge_run_sha256 != EXPECTED_BRIDGE_RUN_SHA256:
            raise ValueError("native_diagnostic_bridge_run_identity_mismatch")
        if self.projection != DEFAULT_PROJECTION_CONFIG:
            raise ValueError("native_diagnostic_projection_not_frozen")
        if hash_canonical(self.question_spec) != self.question_spec_sha256:
            raise ValueError("native_diagnostic_question_spec_hash_mismatch")
        question_spec = self.question_spec
        if not isinstance(question_spec, dict) or set(question_spec) != {
            "eligibility",
            "moderators",
            "outcomes",
            "research_question",
            "target_relation",
        }:
            raise ValueError("native_diagnostic_question_spec_scope_invalid")
        forbidden = _forbidden_key_path(question_spec, _FORBIDDEN_INPUT_KEYS)
        if forbidden is not None:
            raise ValueError(f"native_diagnostic_question_spec_label_leak:{forbidden}")
        manifest = ClaimManifest.model_validate(self.claim_manifest)
        if (
            manifest.question_id != self.question_id
            or manifest.protocol.corpus_cutoff != self.corpus_cutoff
        ):
            raise ValueError("native_diagnostic_claim_context_mismatch")
        eligibility = question_spec.get("eligibility")
        if not isinstance(eligibility, dict):
            raise ValueError("native_diagnostic_question_eligibility_invalid")
        locked_include = eligibility.get("include")
        locked_exclude = eligibility.get("exclude")
        if (
            not isinstance(locked_include, list)
            or any(not isinstance(value, str) for value in locked_include)
            or sorted(manifest.protocol.inclusion_criteria) != sorted(locked_include)
        ):
            raise ValueError("native_diagnostic_claim_inclusion_config_mismatch")
        if (
            not isinstance(locked_exclude, list)
            or any(not isinstance(value, str) for value in locked_exclude)
            or sorted(manifest.protocol.exclusion_criteria) != sorted(locked_exclude)
        ):
            raise ValueError("native_diagnostic_claim_exclusion_config_mismatch")
        return self


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeOllamaDiagnosticError(f"json_artifact_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise NativeOllamaDiagnosticError(f"json_artifact_not_object:{path}")
    return value


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def _validate_self_hash(value: Mapping[str, Any], field: str, *, artifact: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or hash_canonical(_without_hash(value, field)) != observed:
        raise NativeOllamaDiagnosticError(f"{artifact}_hash_mismatch")


def _forbidden_key_path(value: Any, forbidden: frozenset[str], prefix: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            text = str(key)
            path = f"{prefix}.{text}"
            if text.casefold() in forbidden:
                return path
            nested = _forbidden_key_path(item, forbidden, path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _forbidden_key_path(item, forbidden, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    return None


def _validated_config(path: Path) -> NativeDiagnosticConfig:
    try:
        return NativeDiagnosticConfig.model_validate(_read_json_object(path))
    except ValidationError as exc:
        raise NativeOllamaDiagnosticError("native_diagnostic_config_invalid") from exc


def _verified_repo_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise NativeOllamaDiagnosticError(f"frozen_input_missing:{relative}") from exc
    if (
        not resolved.is_relative_to(resolved_root)
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise NativeOllamaDiagnosticError(f"frozen_input_path_invalid:{relative}")
    if sha256_file(resolved) != expected_sha256:
        raise NativeOllamaDiagnosticError(f"frozen_input_hash_mismatch:{relative}")
    return resolved


def compute_diagnostic_execution_identity(
    *,
    repository_root: Path,
    config_path: Path,
    config: NativeDiagnosticConfig,
) -> dict[str, Any]:
    """Hash the local-model adapter separately from the downstream verifier identity."""

    root = repository_root.resolve(strict=True)
    try:
        relative_config = config_path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise NativeOllamaDiagnosticError(
            "native_diagnostic_config_must_be_repository_relative"
        ) from exc
    paths = sorted(
        {
            relative_config,
            config.prompt_path,
            "scripts/run_native_ollama_diagnostic.py",
            "src/literature_multiverse/local_ollama.py",
            "src/literature_multiverse/native_extraction.py",
            "src/literature_multiverse/native_grounding.py",
            "src/literature_multiverse/native_ollama_diagnostic.py",
            "src/literature_multiverse/source_manifest_bridge.py",
            "src/literature_multiverse/typed_extraction.py",
        }
    )
    files = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise NativeOllamaDiagnosticError(
                f"native_diagnostic_execution_file_invalid:{relative}"
            )
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "execution_identity_version": "native-ollama-execution-identity-v1",
        "files": files,
        "settings": {
            "corpus_cutoff": config.corpus_cutoff,
            "generation_config": DEFAULT_GENERATION_CONFIG.model_dump(mode="json"),
            "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
            "official_schema_sha256": hash_canonical(native_publication_extraction_json_schema()),
            "projection": config.projection,
            "source_bridge_run_sha256": config.bridge_run_sha256,
        },
    }
    return {**payload, "execution_sha256": hash_canonical(payload)}


def _validate_bridge_run(
    bridge: Mapping[str, Any], *, config: NativeDiagnosticConfig, manifest_path: Path
) -> None:
    _validate_self_hash(bridge, "run_sha256", artifact="source_bridge_run")
    if (
        bridge.get("run_sha256") != config.bridge_run_sha256
        or bridge.get("source_manifest_bridge_run_version") != "2"
        or bridge.get("question_id") != config.question_id
        or bridge.get("selection_scope") != config.selection_scope
        or bridge.get("native_manifest_records") != config.source_manifest_records
        or bridge.get("diagnostic_only") is not True
        or bridge.get("labels_previously_opened") is not True
        or bridge.get("pristine_final_holdout_eligible") is not False
        or bridge.get("native_source_manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise NativeOllamaDiagnosticError("source_bridge_run_scope_or_lineage_mismatch")


def _validate_source_ledger(
    *,
    bridge: Mapping[str, Any],
    ledger_path: Path,
    manifest: NativeSourceManifest,
    repository_root: Path,
    config: NativeDiagnosticConfig,
) -> DiagnosticSourceLedger:
    try:
        ledger = DiagnosticSourceLedger.model_validate(_read_json_object(ledger_path))
    except ValidationError as exc:
        raise NativeOllamaDiagnosticError("diagnostic_source_ledger_invalid") from exc
    selected = [record for record in ledger.records if record.included_in_native_manifest]
    selected_by_doc = {record.doc_id: record for record in selected}
    manifest_by_doc = {record.doc_id: record for record in manifest.records}
    bridge_artifacts = [artifact.model_dump(mode="json") for artifact in ledger.artifacts]
    if (
        sha256_file(ledger_path) != bridge.get("diagnostic_source_ledger_file_sha256")
        or ledger.ledger_sha256 != bridge.get("diagnostic_source_ledger_content_sha256")
        or ledger.question_id != config.question_id
        or ledger.selection_scope != config.selection_scope
        or ledger.native_source_manifest_sha256 != hash_canonical(manifest)
        or ledger.source_records != bridge.get("records")
        or ledger.native_manifest_records != bridge.get("native_manifest_records")
        or ledger.source_available_records != bridge.get("source_available_records")
        or ledger.source_absent_records != bridge.get("source_absent_records")
        or ledger.manifest_excluded_records != bridge.get("manifest_excluded_records")
        or ledger.content_scope_counts != bridge.get("content_scope_counts")
        or bridge_artifacts != bridge.get("verified_source_artifacts")
        or set(selected_by_doc) != set(manifest_by_doc)
    ):
        raise NativeOllamaDiagnosticError("diagnostic_source_ledger_scope_or_lineage_mismatch")
    for doc_id, manifest_record in manifest_by_doc.items():
        ledger_record = selected_by_doc[doc_id]
        if (
            ledger_record.publication_id != manifest_record.publication.publication_id
            or ledger_record.paper_id != manifest_record.publication.paper_id
            or ledger_record.source_document != manifest_record.source_document
            or not ledger_record.source_available
        ):
            raise NativeOllamaDiagnosticError(
                "diagnostic_source_ledger_manifest_membership_mismatch"
            )
    for artifact in ledger.artifacts:
        _verified_repo_file(
            repository_root,
            artifact.artifact_path,
            artifact.sha256,
        )
    return ledger


def _split_source_text(text: str, *, maximum: int) -> list[tuple[int, int, str]]:
    if len(text) <= maximum:
        return [(0, len(text), text)]
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in _SENTENCE_BOUNDARY.split(text):
        start = text.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = end
        if len(sentence) <= maximum:
            pieces.append((start, end, sentence))
            continue
        for offset in range(0, len(sentence), maximum):
            chunk = sentence[offset : offset + maximum]
            pieces.append((start + offset, start + offset + len(chunk), chunk))
    return [piece for piece in pieces if piece[2].strip()]


def _term_hits(text: str, terms: Sequence[str]) -> int:
    lowered = text.casefold()
    return sum(term.casefold() in lowered for term in terms)


def project_native_source_lines(
    source: Any,
    projection: Mapping[str, Any] = DEFAULT_PROJECTION_CONFIG,
) -> list[dict[str, Any]]:
    """Select source-only passages without consulting extraction outputs or labels."""

    if dict(projection) != DEFAULT_PROJECTION_CONFIG:
        raise NativeOllamaDiagnosticError("native_projection_configuration_not_frozen")
    allowed = set(projection["allowed_sections"])
    candidates: list[dict[str, Any]] = []
    for line in source.lines:
        if line.section not in allowed:
            continue
        for start, end, text in _split_source_text(
            line.text,
            maximum=int(projection["max_passage_characters"]),
        ):
            endpoint_hits = _term_hits(text, projection["endpoint_terms"])
            exposure_hits = _term_hits(text, projection["exposure_terms"])
            comparator_hits = _term_hits(text, projection["comparator_terms"])
            numerical_signal = bool(_NUMBER_SIGNAL.search(text))
            section_score = {"FigureTable": 6.0, "Results": 5.0, "Methods": 1.0}[line.section]
            score = (
                section_score
                + 5.0 * endpoint_hits
                + 2.0 * exposure_hits
                + 2.0 * comparator_hits
                + 3.0 * float(numerical_signal)
            )
            candidates.append(
                {
                    "line_id": line.line_id,
                    "line_number": line.line_number,
                    "section": line.section,
                    "source_line_start": start,
                    "source_line_end_exclusive": end,
                    "text": text,
                    "ranking": {
                        "score": score,
                        "endpoint_term_hits": endpoint_hits,
                        "exposure_term_hits": exposure_hits,
                        "comparator_term_hits": comparator_hits,
                        "numerical_signal": numerical_signal,
                    },
                }
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["ranking"]["score"]),
            0 if item["section"] == "FigureTable" else 1 if item["section"] == "Results" else 2,
            int(item["line_number"]),
            int(item["source_line_start"]),
        ),
    )
    methods = [item for item in ranked if item["section"] == "Methods"]
    non_methods = [item for item in ranked if item["section"] != "Methods"]
    reserve = int(projection["reserved_methods_passages"])
    ordered = [*methods[:reserve], *non_methods, *methods[reserve:]]
    selected: list[dict[str, Any]] = []
    used = 0
    identities: set[tuple[str, int, int]] = set()
    for item in ordered:
        identity = (
            str(item["line_id"]),
            int(item["source_line_start"]),
            int(item["source_line_end_exclusive"]),
        )
        if identity in identities:
            continue
        length = len(str(item["text"]))
        if used + length > int(projection["max_projected_characters"]):
            continue
        selected.append(item)
        identities.add(identity)
        used += length
        if len(selected) >= int(projection["max_projected_passages"]):
            break
    return [{"passage_rank": index, **item} for index, item in enumerate(selected, start=1)]


def _regex_free_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _regex_free_schema(item)
            for key, item in value.items()
            if key not in {"pattern", "$schema", "$id"}
        }
    if isinstance(value, list):
        return [_regex_free_schema(item) for item in value]
    return value


def generation_schema_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the grammar schema and encode the official status invariants.

    Pydantic model validators correctly reject a non-estimable payload that nevertheless
    contains studies, but those invariants are not present in its generated JSON Schema.
    A small model can therefore satisfy the grammar while contradicting the official
    contract.  The mutually exclusive branches below expose the same top-level invariant
    to constrained decoding; official post-validation remains the authority.
    """

    schema = _regex_free_schema(native_publication_extraction_json_schema())
    passages = row.get("source_projection")
    if not isinstance(passages, list):
        raise NativeOllamaDiagnosticError("native_input_row_projection_invalid")
    line_ids = sorted(
        {
            str(item["line_id"])
            for item in passages
            if isinstance(item, Mapping) and isinstance(item.get("line_id"), str)
        }
    ) or ["NO_EXPOSED_SOURCE_LINE"]
    source_record = row.get("source_record")
    if not isinstance(source_record, Mapping):
        raise NativeOllamaDiagnosticError("native_input_row_source_record_invalid")
    source_document = source_record.get("source_document")
    if not isinstance(source_document, Mapping) or not isinstance(
        source_document.get("source_locator"), str
    ):
        raise NativeOllamaDiagnosticError("native_input_row_source_locator_invalid")
    source_locator = source_document["source_locator"]

    def constrain(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                if isinstance(properties.get("source_locator"), dict):
                    properties["source_locator"].pop("pattern", None)
                    properties["source_locator"]["enum"] = [source_locator]
                line_schema = properties.get("line_ids")
                if isinstance(line_schema, dict) and isinstance(line_schema.get("items"), dict):
                    line_schema["items"].pop("pattern", None)
                    line_schema["items"]["enum"] = line_ids
            for item in node.values():
                constrain(item)
        elif isinstance(node, list):
            for item in node:
                constrain(item)

    constrain(schema)
    estimable_branch = {
        "title": "estimable publication evidence",
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["estimable"]},
            "studies": {"type": "array", "minItems": 1},
            "non_estimability_reason": {"type": "null"},
            "non_estimability_detail": {"type": "null"},
        },
        "required": ["status", "studies"],
    }
    non_estimable_branch = {
        "title": "non-estimable publication evidence",
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["non_estimable"]},
            "studies": {"type": "array", "maxItems": 0},
            "non_estimability_reason": {"type": "string"},
        },
        "required": ["status", "non_estimability_reason"],
    }
    schema["oneOf"] = (
        [estimable_branch, non_estimable_branch]
        if passages
        else [non_estimable_branch]
    )
    if _contains_key(schema, "pattern"):
        raise NativeOllamaDiagnosticError("native_generation_schema_contains_regex")
    return schema


def run_generation_schema_compatibility_preflight(
    *,
    client: OllamaClientProtocol,
    config: OllamaGenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> dict[str, Any]:
    """Compile the expanded grammar with synthetic, source-free sentinel values.

    Ollama's grammar compiler supports a narrower contract than a general JSON Schema
    validator. Exercise the exact expanded official-schema shape before any paper text
    is sent so an unsupported composition keyword or branch shape is an infrastructure
    failure, never a row-level scientific outcome. The short generation is deliberately
    not a prediction receipt and is excluded from response-bearing paper-call counters.
    """

    synthetic_row = {
        "source_projection": [{"line_id": _SCHEMA_COMPATIBILITY_LINE_ID}],
        "source_record": {
            "source_document": {
                "source_locator": _SCHEMA_COMPATIBILITY_SOURCE_LOCATOR,
            }
        },
    }
    schema = generation_schema_for_row(synthetic_row)
    if any(branch.get("type") != "object" for branch in schema["oneOf"]):
        raise NativeOllamaDiagnosticError(
            "native_generation_schema_preflight_branch_type_missing"
        )
    preflight_config = config.model_copy(update={"num_predict": min(config.num_predict, 256)})
    try:
        identity = client.inspect_identity(config)
        result = client.generate(
            prompt=_SCHEMA_COMPATIBILITY_PROMPT,
            output_schema=schema,
            config=preflight_config,
        )
    except LocalOllamaError as exc:
        raise NativeOllamaDiagnosticError(
            "native_generation_schema_compatibility_preflight_failed;"
            "no_scientific_row_request_was_made"
        ) from exc
    if (
        identity.model != config.model
        or identity.model_digest != config.model_digest
        or identity.ollama_version != config.expected_ollama_version
        or result.model != config.model
    ):
        raise NativeOllamaDiagnosticError(
            "native_generation_schema_compatibility_preflight_identity_mismatch"
        )
    return {
        "preflight_version": SCHEMA_COMPATIBILITY_PREFLIGHT_VERSION,
        "status": "passed",
        "schema_sha256": hash_canonical(schema),
        "model_identity_sha256": identity.identity_sha256,
        "generation_config_sha256": preflight_config.config_sha256,
        "contains_publication_content": False,
        "contains_scientific_claim": False,
        "contains_source_text": False,
        "contains_eligibility_or_answer_labels": False,
        "paper_prediction_receipt_written": False,
        "done_reason": result.done_reason,
    }


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def render_prediction_prompt(bundle: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    base_prompt = bundle.get("rendered_base_prompt")
    if not isinstance(base_prompt, str) or not base_prompt:
        raise NativeOllamaDiagnosticError("native_input_bundle_base_prompt_missing")
    source_record = row.get("source_record")
    passages = row.get("source_projection")
    if not isinstance(source_record, Mapping) or not isinstance(passages, list):
        raise NativeOllamaDiagnosticError("native_input_row_invalid")
    source_document = source_record.get("source_document")
    if not isinstance(source_document, Mapping):
        raise NativeOllamaDiagnosticError("native_input_source_document_invalid")
    locator = source_document.get("source_locator")
    if not isinstance(locator, str):
        raise NativeOllamaDiagnosticError("native_input_source_locator_invalid")
    rendered: list[str] = []
    for passage in passages:
        if not isinstance(passage, Mapping):
            raise NativeOllamaDiagnosticError("native_input_passage_invalid")
        line_id = passage.get("line_id")
        section = passage.get("section")
        text = passage.get("text")
        if not all(isinstance(item, str) for item in (line_id, section, text)):
            raise NativeOllamaDiagnosticError("native_input_passage_fields_invalid")
        rendered.append(
            f"LINE_ID: {line_id}\nSECTION: {section}\nBEGIN_EXACT_SOURCE_TEXT\n"
            f"{text}\nEND_EXACT_SOURCE_TEXT"
        )
    source_text = (
        "\n\n".join(rendered)
        if rendered
        else "[NO_ELIGIBLE_SOURCE_PROJECTION] No Results, FigureTable, or Methods text survived."
    )
    return base_prompt + _SOURCE_BLOCK_TEMPLATE.format(
        source_locator=locator,
        passages=source_text,
    )


def prepare_input_bundle(
    *,
    config_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Freeze the exact 19-member, source-only model input population."""

    config = _validated_config(config_path)
    root = repository_root.resolve(strict=True)
    _verified_repo_file(
        root,
        config.question_config_path,
        config.question_config_file_sha256,
    )
    prompt_path = _verified_repo_file(root, config.prompt_path, config.prompt_file_sha256)
    manifest_path = root / config.source_manifest_path
    bridge_path = root / config.bridge_run_path
    ledger_path = manifest_path.with_name("diagnostic_source_ledger.json")
    if (
        not manifest_path.is_file()
        or not bridge_path.is_file()
        or not ledger_path.is_file()
        or manifest_path.is_symlink()
        or bridge_path.is_symlink()
        or ledger_path.is_symlink()
    ):
        raise NativeOllamaDiagnosticError("native_diagnostic_source_inputs_missing")
    bridge = _read_json_object(bridge_path)
    _validate_bridge_run(bridge, config=config, manifest_path=manifest_path)
    try:
        manifest = NativeSourceManifest.model_validate(_read_json_object(manifest_path))
    except ValidationError as exc:
        raise NativeOllamaDiagnosticError("native_source_manifest_invalid") from exc
    if (
        manifest.question_id != config.question_id
        or len(manifest.records) != config.source_manifest_records
        or hash_canonical(manifest) != bridge.get("native_source_manifest_content_sha256")
    ):
        raise NativeOllamaDiagnosticError("native_source_manifest_scope_mismatch")
    _validate_source_ledger(
        bridge=bridge,
        ledger_path=ledger_path,
        manifest=manifest,
        repository_root=root,
        config=config,
    )
    template = prompt_path.read_text(encoding="utf-8")
    rendered_base, prompt_version = render_prompt_text(
        template,
        {"QUESTION_SPEC_JSON": json.dumps(config.question_spec, sort_keys=True)},
    )
    official_schema = native_publication_extraction_json_schema()
    pipeline = compute_verifier_pipeline_fingerprint(root=root)
    claim = ClaimManifest.model_validate(config.claim_manifest)
    execution_identity = compute_diagnostic_execution_identity(
        repository_root=root,
        config_path=config_path,
        config=config,
    )
    rows: list[dict[str, Any]] = []
    for source_record in manifest.records:
        source = resolve_native_source_document(
            repository_root=root,
            source_document=source_record.source_document,
        )
        projection = project_native_source_lines(source, config.projection)
        row_payload = {
            "row_key": hashlib.sha256(
                source_record.publication.publication_id.encode()
            ).hexdigest(),
            "source_record": source_record.model_dump(mode="json"),
            "source_payload_sha256": source.source_payload_sha256,
            "source_projection": projection,
            "source_projection_sha256": hash_canonical(projection),
            "projected_characters": sum(len(item["text"]) for item in projection),
            "projected_passages": len(projection),
        }
        rows.append({**row_payload, "input_row_sha256": hash_canonical(row_payload)})
    rows.sort(key=lambda item: str(item["row_key"]))
    payload = {
        "input_bundle_version": INPUT_BUNDLE_VERSION,
        "diagnostic_version": NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
        "status": "frozen_label_blind_source_only_non_pristine_diagnostic",
        "scientific_role": "diagnostic_only_no_independent_numerical_gold",
        "selection_scope": config.selection_scope,
        "selection_labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "contains_legacy_findings": False,
        "contains_legacy_directions": False,
        "contains_anchor_expectations": False,
        "contains_downstream_claim_payload": False,
        "prediction_stage_can_open_source_or_label_files": False,
        "config_file_sha256": sha256_file(config_path),
        "diagnostic_config_path": config_path.resolve(strict=True).relative_to(root).as_posix(),
        "diagnostic_execution_identity": execution_identity,
        "diagnostic_execution_sha256": execution_identity["execution_sha256"],
        "question_config_file_sha256": config.question_config_file_sha256,
        "question_spec_sha256": config.question_spec_sha256,
        "source_bridge_run_sha256": bridge["run_sha256"],
        "source_bridge_run_file_sha256": sha256_file(bridge_path),
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_content_sha256": hash_canonical(manifest),
        "source_manifest_records": len(manifest.records),
        "corpus_cutoff": config.corpus_cutoff,
        "prompt_template_file_sha256": config.prompt_file_sha256,
        "prompt_version": prompt_version,
        "rendered_base_prompt": rendered_base,
        "rendered_base_prompt_sha256": hashlib.sha256(rendered_base.encode()).hexdigest(),
        "official_schema_sha256": hash_canonical(official_schema),
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_config": DEFAULT_GENERATION_CONFIG.model_dump(mode="json"),
        "generation_config_sha256": DEFAULT_GENERATION_CONFIG.config_sha256,
        "projection_config": deepcopy(config.projection),
        "projection_config_sha256": hash_canonical(config.projection),
        "pipeline_fingerprint": pipeline.model_dump(mode="json"),
        "pipeline_sha256": pipeline.pipeline_sha256,
        "claim_manifest_sha256": hash_canonical(claim),
        "rows": rows,
        "row_count": len(rows),
    }
    bundle = {**payload, "input_bundle_sha256": hash_canonical(payload)}
    return validate_input_bundle(bundle)


def validate_input_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(bundle))
    _validate_self_hash(snapshot, "input_bundle_sha256", artifact="native_input_bundle")
    rendered_base_prompt = snapshot.get("rendered_base_prompt")
    if (
        set(snapshot) != _INPUT_BUNDLE_FIELDS
        or snapshot.get("input_bundle_version") != INPUT_BUNDLE_VERSION
        or snapshot.get("diagnostic_version") != NATIVE_OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("status") != "frozen_label_blind_source_only_non_pristine_diagnostic"
        or snapshot.get("scientific_role") != "diagnostic_only_no_independent_numerical_gold"
        or snapshot.get("selection_scope") != EXPECTED_SELECTION_SCOPE
        or snapshot.get("selection_labels_previously_opened") is not True
        or snapshot.get("pristine_final_holdout_eligible") is not False
        or snapshot.get("contains_legacy_findings") is not False
        or snapshot.get("contains_legacy_directions") is not False
        or snapshot.get("contains_anchor_expectations") is not False
        or snapshot.get("contains_downstream_claim_payload") is not False
        or snapshot.get("prediction_stage_can_open_source_or_label_files") is not False
        or snapshot.get("diagnostic_config_path")
        != "configs/benchmarks/native-antiox-ollama-v1.json"
        or snapshot.get("source_bridge_run_sha256") != EXPECTED_BRIDGE_RUN_SHA256
        or snapshot.get("source_manifest_records") != EXPECTED_MANIFEST_RECORDS
        or snapshot.get("corpus_cutoff") != EXPECTED_CORPUS_CUTOFF
        or snapshot.get("projection_config") != DEFAULT_PROJECTION_CONFIG
        or snapshot.get("projection_config_sha256") != hash_canonical(DEFAULT_PROJECTION_CONFIG)
        or snapshot.get("generation_config") != DEFAULT_GENERATION_CONFIG.model_dump(mode="json")
        or snapshot.get("generation_config_sha256") != DEFAULT_GENERATION_CONFIG.config_sha256
        or snapshot.get("generation_schema_algorithm") != GENERATION_SCHEMA_ALGORITHM
        or snapshot.get("official_schema_sha256")
        != hash_canonical(native_publication_extraction_json_schema())
        or snapshot.get("prompt_version") != "native-extraction-v3"
        or not isinstance(rendered_base_prompt, str)
        or not rendered_base_prompt
        or hashlib.sha256(rendered_base_prompt.encode()).hexdigest()
        != snapshot.get("rendered_base_prompt_sha256")
    ):
        raise NativeOllamaDiagnosticError("native_input_bundle_scope_or_config_mismatch")
    lowered_base_prompt = rendered_base_prompt.casefold()
    if any(token in lowered_base_prompt for token in _FORBIDDEN_INPUT_KEYS):
        raise NativeOllamaDiagnosticError("native_input_bundle_base_prompt_label_leak")
    execution_identity = snapshot.get("diagnostic_execution_identity")
    if (
        not isinstance(execution_identity, Mapping)
        or execution_identity.get("execution_identity_version")
        != "native-ollama-execution-identity-v1"
        or hash_canonical(_without_hash(execution_identity, "execution_sha256"))
        != execution_identity.get("execution_sha256")
        or snapshot.get("diagnostic_execution_sha256") != execution_identity.get("execution_sha256")
    ):
        raise NativeOllamaDiagnosticError("native_input_bundle_execution_identity_invalid")
    execution_files = execution_identity.get("files")
    if not isinstance(execution_files, list):
        raise NativeOllamaDiagnosticError("native_input_bundle_execution_files_invalid")
    expected_execution_paths = sorted(
        {
            str(snapshot.get("diagnostic_config_path")),
            "prompts/native_extraction.md",
            "scripts/run_native_ollama_diagnostic.py",
            "src/literature_multiverse/local_ollama.py",
            "src/literature_multiverse/native_extraction.py",
            "src/literature_multiverse/native_grounding.py",
            "src/literature_multiverse/native_ollama_diagnostic.py",
            "src/literature_multiverse/source_manifest_bridge.py",
            "src/literature_multiverse/typed_extraction.py",
        }
    )
    observed_execution_paths: list[str] = []
    execution_files_by_path: dict[str, Mapping[str, Any]] = {}
    for item in execution_files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"bytes", "path", "sha256"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or _SHA256.fullmatch(str(item["sha256"])) is None
            or type(item.get("bytes")) is not int
            or int(item["bytes"]) < 0
        ):
            raise NativeOllamaDiagnosticError("native_input_bundle_execution_files_invalid")
        path = str(item["path"])
        observed_execution_paths.append(path)
        execution_files_by_path[path] = item
    config_entry = execution_files_by_path.get(str(snapshot.get("diagnostic_config_path")))
    expected_execution_settings = {
        "corpus_cutoff": snapshot.get("corpus_cutoff"),
        "generation_config": snapshot.get("generation_config"),
        "generation_schema_algorithm": snapshot.get("generation_schema_algorithm"),
        "official_schema_sha256": snapshot.get("official_schema_sha256"),
        "projection": snapshot.get("projection_config"),
        "source_bridge_run_sha256": snapshot.get("source_bridge_run_sha256"),
    }
    if (
        observed_execution_paths != expected_execution_paths
        or len(execution_files_by_path) != len(execution_files)
        or config_entry is None
        or config_entry.get("sha256") != snapshot.get("config_file_sha256")
        or execution_identity.get("settings") != expected_execution_settings
    ):
        raise NativeOllamaDiagnosticError(
            "native_input_bundle_execution_file_lineage_mismatch"
        )
    forbidden = _forbidden_key_path(snapshot, _FORBIDDEN_INPUT_KEYS)
    if forbidden is not None:
        raise NativeOllamaDiagnosticError(f"native_input_bundle_label_leak:{forbidden}")
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_MANIFEST_RECORDS:
        raise NativeOllamaDiagnosticError("native_input_bundle_population_incomplete")
    keys: list[str] = []
    receipt_path_keys: list[str] = []
    publication_ids: set[str] = set()
    paper_ids: set[str] = set()
    doc_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _INPUT_ROW_FIELDS:
            raise NativeOllamaDiagnosticError("native_input_bundle_row_invalid")
        _validate_self_hash(row, "input_row_sha256", artifact="native_input_row")
        key = row.get("row_key")
        if not isinstance(key, str) or not _SHA256.fullmatch(key):
            raise NativeOllamaDiagnosticError("native_input_row_key_invalid")
        record = NativeSourceRecord.model_validate(row.get("source_record"))
        if key != hashlib.sha256(record.publication.publication_id.encode()).hexdigest():
            raise NativeOllamaDiagnosticError("native_input_row_key_identity_mismatch")
        projection = row.get("source_projection")
        if not isinstance(projection, list) or hash_canonical(projection) != row.get(
            "source_projection_sha256"
        ):
            raise NativeOllamaDiagnosticError("native_input_row_projection_hash_mismatch")
        projection_identities: set[tuple[str, int, int]] = set()
        for index, passage in enumerate(projection, start=1):
            if not isinstance(passage, Mapping) or set(passage) != _PROJECTION_PASSAGE_FIELDS:
                raise NativeOllamaDiagnosticError("native_input_row_projection_passage_invalid")
            line_id = passage.get("line_id")
            line_number = passage.get("line_number")
            start = passage.get("source_line_start")
            end = passage.get("source_line_end_exclusive")
            text = passage.get("text")
            section = passage.get("section")
            ranking = passage.get("ranking")
            if (
                passage.get("passage_rank") != index
                or not isinstance(line_id, str)
                or re.fullmatch(r"L[1-9][0-9]*", line_id) is None
                or type(line_number) is not int
                or line_number != int(line_id[1:])
                or type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or not isinstance(text, str)
                or not text.strip()
                or len(text) != end - start
                or len(text) > int(DEFAULT_PROJECTION_CONFIG["max_passage_characters"])
                or section not in set(DEFAULT_PROJECTION_CONFIG["allowed_sections"])
                or not isinstance(ranking, Mapping)
                or set(ranking) != _PROJECTION_RANKING_FIELDS
            ):
                raise NativeOllamaDiagnosticError("native_input_row_projection_passage_invalid")
            expected_ranking = {
                "score": (
                    {"FigureTable": 6.0, "Results": 5.0, "Methods": 1.0}[section]
                    + 5.0 * _term_hits(text, DEFAULT_PROJECTION_CONFIG["endpoint_terms"])
                    + 2.0 * _term_hits(text, DEFAULT_PROJECTION_CONFIG["exposure_terms"])
                    + 2.0 * _term_hits(text, DEFAULT_PROJECTION_CONFIG["comparator_terms"])
                    + 3.0 * float(bool(_NUMBER_SIGNAL.search(text)))
                ),
                "endpoint_term_hits": _term_hits(
                    text, DEFAULT_PROJECTION_CONFIG["endpoint_terms"]
                ),
                "exposure_term_hits": _term_hits(
                    text, DEFAULT_PROJECTION_CONFIG["exposure_terms"]
                ),
                "comparator_term_hits": _term_hits(
                    text, DEFAULT_PROJECTION_CONFIG["comparator_terms"]
                ),
                "numerical_signal": bool(_NUMBER_SIGNAL.search(text)),
            }
            if dict(ranking) != expected_ranking:
                raise NativeOllamaDiagnosticError("native_input_row_projection_ranking_invalid")
            identity = (line_id, start, end)
            if identity in projection_identities:
                raise NativeOllamaDiagnosticError("native_input_row_projection_passage_duplicate")
            projection_identities.add(identity)
        if row.get("projected_passages") != len(projection) or row.get(
            "projected_characters"
        ) != sum(
            len(str(item.get("text", ""))) for item in projection if isinstance(item, Mapping)
        ):
            raise NativeOllamaDiagnosticError("native_input_row_projection_count_mismatch")
        if len(projection) > int(DEFAULT_PROJECTION_CONFIG["max_projected_passages"]):
            raise NativeOllamaDiagnosticError("native_input_row_projection_passage_overflow")
        if int(row["projected_characters"]) > int(
            DEFAULT_PROJECTION_CONFIG["max_projected_characters"]
        ):
            raise NativeOllamaDiagnosticError("native_input_row_projection_character_overflow")
        generation_schema_for_row(row)
        keys.append(key)
        receipt_path_keys.append(key[:32])
        publication_ids.add(record.publication.publication_id)
        paper_ids.add(record.publication.paper_id)
        doc_ids.add(record.doc_id)
    if (
        keys != sorted(set(keys))
        or len(publication_ids) != EXPECTED_MANIFEST_RECORDS
        or len(paper_ids) != EXPECTED_MANIFEST_RECORDS
        or len(doc_ids) != EXPECTED_MANIFEST_RECORDS
    ):
        raise NativeOllamaDiagnosticError("native_input_bundle_rows_not_sorted_unique")
    if len(receipt_path_keys) != len(set(receipt_path_keys)):
        raise NativeOllamaDiagnosticError("native_input_bundle_receipt_paths_collide")
    if snapshot.get("row_count") != len(rows):
        raise NativeOllamaDiagnosticError("native_input_bundle_row_count_mismatch")
    try:
        pipeline = PipelineFingerprint.model_validate(snapshot.get("pipeline_fingerprint"))
    except ValidationError as exc:
        raise NativeOllamaDiagnosticError(
            "native_input_bundle_downstream_contract_invalid"
        ) from exc
    if (
        pipeline.pipeline_sha256 != snapshot.get("pipeline_sha256")
        or not isinstance(snapshot.get("claim_manifest_sha256"), str)
        or _SHA256.fullmatch(snapshot["claim_manifest_sha256"]) is None
    ):
        raise NativeOllamaDiagnosticError("native_input_bundle_downstream_context_mismatch")
    return snapshot


def validate_current_diagnostic_context(
    bundle: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Fail before generation when code, config, or downstream pipeline bytes drifted.

    The input bundle's self-hash proves internal integrity, not that the repository still
    contains the bytes frozen during ``prepare``. Recomputing both execution identities
    here prevents an expensive local-model run from starting after repository drift.
    Only code and configuration files are opened; publication sources, manifests, legacy
    outputs, and labels remain unavailable to the prediction stage.
    """

    snapshot = validate_input_bundle(bundle)
    root = repository_root.resolve(strict=True)
    config_path = root / str(snapshot["diagnostic_config_path"])
    config = _validated_config(config_path)
    if (
        snapshot.get("config_file_sha256") != sha256_file(config_path)
        or snapshot.get("question_config_file_sha256")
        != config.question_config_file_sha256
        or snapshot.get("question_spec_sha256") != config.question_spec_sha256
        or snapshot.get("prompt_template_file_sha256") != config.prompt_file_sha256
        or snapshot.get("source_bridge_run_sha256") != config.bridge_run_sha256
        or snapshot.get("source_manifest_records") != config.source_manifest_records
        or snapshot.get("corpus_cutoff") != config.corpus_cutoff
    ):
        raise NativeOllamaDiagnosticError(
            "native_diagnostic_config_context_changed_after_prepare"
        )
    current_execution = compute_diagnostic_execution_identity(
        repository_root=root,
        config_path=config_path,
        config=config,
    )
    if current_execution != snapshot["diagnostic_execution_identity"]:
        raise NativeOllamaDiagnosticError(
            "native_diagnostic_execution_identity_changed_after_prepare"
        )
    try:
        pipeline = PipelineFingerprint.model_validate(snapshot["pipeline_fingerprint"])
        require_pipeline_fingerprint_match(expected=pipeline, root=root)
    except ValueError as exc:
        raise NativeOllamaDiagnosticError(
            "native_diagnostic_downstream_pipeline_changed_after_prepare"
        ) from exc
    return snapshot


def _model_identity_payload(identity: OllamaIdentity) -> dict[str, Any]:
    return {**identity.model_dump(mode="json"), "identity_sha256": identity.identity_sha256}


def _receipt_path(receipts_dir: Path, row_key: str) -> Path:
    return receipts_dir / f"{row_key[:32]}.json"


def _validate_receipt_directory_membership(
    *, receipts_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if receipts_dir.is_symlink() or not receipts_dir.is_dir():
        raise NativeOllamaDiagnosticError("native_generation_receipt_directory_invalid")
    expected = {
        _receipt_path(receipts_dir, str(row["row_key"])).name
        for row in rows
    }
    observed_paths = list(receipts_dir.glob("*.json"))
    if any(path.is_symlink() or not path.is_file() for path in observed_paths):
        raise NativeOllamaDiagnosticError("native_generation_receipt_file_invalid")
    unexpected = sorted(path.name for path in observed_paths if path.name not in expected)
    if unexpected:
        raise NativeOllamaDiagnosticError(
            "native_generation_receipt_files_unexpected:" + ",".join(unexpected)
        )


def _request_identity(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    prompt = render_prediction_prompt(bundle, row)
    schema = generation_schema_for_row(row)
    request = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "keep_alive": config.keep_alive,
        "options": config.request_options(),
    }
    identity_payload = {
        "input_row_sha256": row["input_row_sha256"],
        "model_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation_schema_sha256": hash_canonical(schema),
        "official_schema_sha256": bundle["official_schema_sha256"],
        "request_sha256": hash_canonical(request),
    }
    return prompt, schema, identity_payload


@cache
def _official_schema_validator() -> Any:
    official_schema = native_publication_extraction_json_schema()
    validator_class = validator_for(official_schema)
    try:
        validator_class.check_schema(official_schema)
    except SchemaError as exc:
        raise NativeOllamaDiagnosticError(
            "native_official_extraction_json_schema_invalid"
        ) from exc
    return validator_class(official_schema)


def _official_postvalidate(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    schema_errors = list(_official_schema_validator().iter_errors(value))
    if schema_errors:
        validators = sorted(
            {
                str(error.validator) if error.validator is not None else "unknown"
                for error in schema_errors
            }
        )
        return None, "official_json_schema_validation_error:" + ",".join(validators)
    try:
        extraction = NativePublicationExtraction.model_validate(value)
    except ValidationError as exc:
        error_types = sorted({str(item["type"]) for item in exc.errors(include_url=False)})
        return None, "official_contract_validation_error:" + ",".join(error_types)
    return extraction.model_dump(mode="json"), None


def _make_generation_receipt(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    client: OllamaClientProtocol,
) -> dict[str, Any]:
    prompt, schema, request_identity = _request_identity(
        bundle=bundle,
        row=row,
        config=config,
        identity=identity,
    )
    del prompt
    result: OllamaGenerationResult | None = None
    parsed: Any = None
    official: dict[str, Any] | None = None
    error: str | None = None
    try:
        result = client.generate(
            prompt=render_prediction_prompt(bundle, row), output_schema=schema, config=config
        )
        try:
            parsed = _strict_model_json_loads(result.response_text)
        except _ModelJSONError:
            error = "response_json_decode_error"
        if error is None:
            official, error = _official_postvalidate(parsed)
    except LocalOllamaError as exc:
        detail = str(exc).casefold()
        if "connectionrefusederror" in detail or "connection refused" in detail:
            code = "local_server_connection_refused"
        elif "timeouterror" in detail or "timed out" in detail:
            code = "local_server_timeout"
        elif "http " in detail:
            code = "local_server_http_error"
        elif "invalid json" in detail:
            code = "local_server_invalid_wrapper_json"
        else:
            code = "local_server_contract_error"
        error = f"local_ollama_error:{code}"
    except Exception as exc:
        error = f"client_error:{type(exc).__name__}"
    generation_truncated = result is not None and result.done_reason == "length"
    if generation_truncated:
        # A length-stopped response is terminal even in the unusual case that its
        # prefix happens to parse and satisfy the official contract.  Never promote
        # a truncation into an estimable extraction.
        official = None
        error = "generation_truncated"
        status = "generation_truncated"
    elif result is None:
        status = "execution_failure"
    elif error == "response_json_decode_error":
        status = "response_json_invalid"
    elif error is not None:
        status = "official_schema_invalid"
    else:
        status = "official_schema_valid"
    response_text = result.response_text if result is not None else None
    payload = {
        "generation_receipt_version": GENERATION_RECEIPT_VERSION,
        "diagnostic_version": NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
        "status": status,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "row_key": row["row_key"],
        **request_identity,
        "model": config.model,
        "model_digest": config.model_digest,
        "ollama_version": identity.ollama_version,
        "model_parameter_size": identity.parameter_size,
        "generation_call_attempted_once": True,
        "generation_config": config.model_dump(mode="json"),
        "generation_schema": schema,
        "response_text": response_text,
        "response_text_sha256": (
            hashlib.sha256(response_text.encode()).hexdigest()
            if response_text is not None
            else None
        ),
        "parsed_output": parsed,
        "parsed_output_sha256": hash_canonical(parsed) if parsed is not None else None,
        "official_output": official,
        "official_output_sha256": hash_canonical(official) if official is not None else None,
        "terminal_error": error,
        "generation_truncated": generation_truncated,
        "telemetry": result.model_dump(mode="json") if result is not None else None,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def validate_generation_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    _validate_self_hash(snapshot, "receipt_sha256", artifact="native_generation_receipt")
    _, expected_schema, expected_request = _request_identity(
        bundle=bundle,
        row=row,
        config=config,
        identity=identity,
    )
    if (
        snapshot.get("generation_receipt_version") != GENERATION_RECEIPT_VERSION
        or snapshot.get("diagnostic_version") != NATIVE_OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("input_bundle_sha256") != bundle.get("input_bundle_sha256")
        or snapshot.get("row_key") != row.get("row_key")
        or snapshot.get("model") != config.model
        or snapshot.get("model_digest") != config.model_digest
        or snapshot.get("ollama_version") != identity.ollama_version
        or snapshot.get("model_parameter_size") != identity.parameter_size
        or snapshot.get("generation_call_attempted_once") is not True
        or snapshot.get("generation_config") != config.model_dump(mode="json")
        or snapshot.get("generation_schema") != expected_schema
        or any(snapshot.get(key) != value for key, value in expected_request.items())
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
    ):
        raise NativeOllamaDiagnosticError("native_generation_receipt_context_mismatch")
    response = snapshot.get("response_text")
    if response is not None and (
        not isinstance(response, str)
        or hashlib.sha256(response.encode()).hexdigest() != snapshot.get("response_text_sha256")
    ):
        raise NativeOllamaDiagnosticError("native_generation_receipt_response_hash_mismatch")
    parsed = snapshot.get("parsed_output")
    if parsed is not None and hash_canonical(parsed) != snapshot.get("parsed_output_sha256"):
        raise NativeOllamaDiagnosticError("native_generation_receipt_parsed_hash_mismatch")
    telemetry = snapshot.get("telemetry")
    if response is None:
        expected_status = "execution_failure"
        expected_official = None
        expected_error = snapshot.get("terminal_error")
        if (
            not isinstance(expected_error, str)
            or not expected_error.startswith(("local_ollama_error:", "client_error:"))
            or telemetry is not None
            or parsed is not None
            or snapshot.get("parsed_output_sha256") is not None
            or snapshot.get("response_text_sha256") is not None
            or snapshot.get("generation_truncated") is not False
        ):
            raise NativeOllamaDiagnosticError("native_generation_receipt_execution_failure_invalid")
    else:
        try:
            generation_result = OllamaGenerationResult.model_validate(telemetry)
        except ValidationError as exc:
            raise NativeOllamaDiagnosticError(
                "native_generation_receipt_telemetry_invalid"
            ) from exc
        generation_truncated = generation_result.done_reason == "length"
        if (
            generation_result.response_text != response
            or generation_result.model != config.model
            or snapshot.get("generation_truncated") is not generation_truncated
        ):
            raise NativeOllamaDiagnosticError("native_generation_receipt_telemetry_mismatch")
        try:
            reparsed = _strict_model_json_loads(response)
        except _ModelJSONError:
            if parsed is not None:
                raise NativeOllamaDiagnosticError(
                    "native_generation_receipt_parse_escalation"
                ) from None
            expected_status = "response_json_invalid"
            expected_official = None
            expected_error = "response_json_decode_error"
        else:
            if reparsed != parsed:
                raise NativeOllamaDiagnosticError("native_generation_receipt_reparse_mismatch")
            expected_official, validation_error = _official_postvalidate(reparsed)
            expected_status = (
                "official_schema_valid" if validation_error is None else "official_schema_invalid"
            )
            expected_error = validation_error
        if generation_truncated:
            expected_status = "generation_truncated"
            expected_official = None
            expected_error = "generation_truncated"
    if snapshot.get("status") != expected_status:
        raise NativeOllamaDiagnosticError("native_generation_receipt_status_mismatch")
    if snapshot.get("terminal_error") != expected_error:
        raise NativeOllamaDiagnosticError("native_generation_receipt_terminal_error_mismatch")
    if expected_official != snapshot.get("official_output"):
        raise NativeOllamaDiagnosticError("native_generation_receipt_official_output_mismatch")
    if expected_official is not None and hash_canonical(expected_official) != snapshot.get(
        "official_output_sha256"
    ):
        raise NativeOllamaDiagnosticError("native_generation_receipt_official_hash_mismatch")
    if expected_official is None and snapshot.get("official_output_sha256") is not None:
        raise NativeOllamaDiagnosticError("native_generation_receipt_invalid_official_hash")
    return snapshot


def _ledger_from_receipts(
    *,
    bundle: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = [
        {
            "row_key": receipt["row_key"],
            "input_row_sha256": receipt["input_row_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "status": receipt["status"],
            "official_output_sha256": receipt["official_output_sha256"],
        }
        for receipt in receipts
    ]
    complete = len(receipts) == bundle["row_count"]
    payload = {
        "prediction_ledger_version": PREDICTION_LEDGER_VERSION,
        "diagnostic_version": NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
        "status": "complete_frozen_terminal_ledger" if complete else "partial_resumable_ledger",
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "input_row_count": bundle["row_count"],
        "receipt_count": len(receipts),
        "all_expected_receipts_frozen": complete,
        "generation_call_attempts": len(receipts),
        "generation_retries": 0,
        "generation_omissions": int(bundle["row_count"]) - len(receipts),
        "attempt_accounting_scope": "response_bearing_terminal_receipts_only",
        "prediction_stage_received_legacy_findings": False,
        "prediction_stage_received_legacy_directions": False,
        "prediction_stage_received_anchor_expectations": False,
        "prediction_stage_received_downstream_claim_payload": False,
        "prediction_stage_opened_source_or_label_files": False,
        "generation_config": config.model_dump(mode="json"),
        "generation_config_sha256": config.config_sha256,
        "model_identity": _model_identity_payload(identity),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in receipts).items())),
        "receipts": manifest,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "prediction_ledger_sha256": hash_canonical(payload)}


def validate_prediction_ledger(
    ledger: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(ledger))
    _validate_self_hash(snapshot, "prediction_ledger_sha256", artifact="native_prediction_ledger")
    try:
        config = OllamaGenerationConfig.model_validate(snapshot.get("generation_config"))
        identity_payload = deepcopy(dict(snapshot.get("model_identity", {})))
        identity_sha256 = identity_payload.pop("identity_sha256")
        identity = OllamaIdentity.model_validate(identity_payload)
    except (KeyError, TypeError, ValidationError) as exc:
        raise NativeOllamaDiagnosticError(
            "native_prediction_ledger_model_contract_invalid"
        ) from exc
    if identity.identity_sha256 != identity_sha256:
        raise NativeOllamaDiagnosticError("native_prediction_ledger_model_identity_hash_mismatch")
    receipts = snapshot.get("receipts")
    if not isinstance(receipts, list):
        raise NativeOllamaDiagnosticError("native_prediction_ledger_receipts_invalid")
    row_by_key = {str(row["row_key"]): row for row in bundle["rows"]}
    observed: list[str] = []
    observed_statuses: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise NativeOllamaDiagnosticError("native_prediction_ledger_receipt_row_invalid")
        key = receipt.get("row_key")
        if (
            not isinstance(key, str)
            or key not in row_by_key
            or receipt.get("input_row_sha256") != row_by_key[key]["input_row_sha256"]
        ):
            raise NativeOllamaDiagnosticError("native_prediction_ledger_receipt_identity_invalid")
        status = receipt.get("status")
        if status not in {
            "execution_failure",
            "generation_truncated",
            "official_schema_invalid",
            "official_schema_valid",
            "response_json_invalid",
        }:
            raise NativeOllamaDiagnosticError(
                "native_prediction_ledger_receipt_status_invalid"
            )
        output_sha256 = receipt.get("official_output_sha256")
        if (status == "official_schema_valid") != (
            isinstance(output_sha256, str)
            and _SHA256.fullmatch(output_sha256) is not None
        ):
            raise NativeOllamaDiagnosticError(
                "native_prediction_ledger_official_output_status_mismatch"
            )
        observed.append(key)
        observed_statuses.append(str(status))
    if observed != sorted(set(observed)):
        raise NativeOllamaDiagnosticError("native_prediction_ledger_receipts_not_sorted_unique")
    complete = len(receipts) == bundle["row_count"]
    if (
        snapshot.get("prediction_ledger_version") != PREDICTION_LEDGER_VERSION
        or snapshot.get("diagnostic_version") != NATIVE_OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("input_bundle_sha256") != bundle["input_bundle_sha256"]
        or snapshot.get("input_row_count") != bundle["row_count"]
        or snapshot.get("receipt_count") != len(receipts)
        or snapshot.get("all_expected_receipts_frozen") is not complete
        or snapshot.get("generation_call_attempts") != len(receipts)
        or snapshot.get("generation_retries") != 0
        or snapshot.get("generation_omissions") != bundle["row_count"] - len(receipts)
        or snapshot.get("attempt_accounting_scope")
        != "response_bearing_terminal_receipts_only"
        or snapshot.get("generation_config_sha256") != config.config_sha256
        or config.model_dump(mode="json") != bundle["generation_config"]
        or identity.model != config.model
        or identity.model_digest != config.model_digest
        or identity.ollama_version != config.expected_ollama_version
        or snapshot.get("status_counts")
        != dict(sorted(Counter(observed_statuses).items()))
        or snapshot.get("prediction_stage_received_legacy_findings") is not False
        or snapshot.get("prediction_stage_received_legacy_directions") is not False
        or snapshot.get("prediction_stage_received_anchor_expectations") is not False
        or snapshot.get("prediction_stage_received_downstream_claim_payload") is not False
        or snapshot.get("prediction_stage_opened_source_or_label_files") is not False
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
    ):
        raise NativeOllamaDiagnosticError("native_prediction_ledger_scope_mismatch")
    expected_status = "complete_frozen_terminal_ledger" if complete else "partial_resumable_ledger"
    if snapshot.get("status") != expected_status:
        raise NativeOllamaDiagnosticError("native_prediction_ledger_status_mismatch")
    if require_complete and not complete:
        raise NativeOllamaDiagnosticError("native_prediction_ledger_incomplete")
    return snapshot


def run_prediction_stage(
    *,
    input_bundle: Mapping[str, Any],
    receipts_dir: Path,
    prediction_ledger_path: Path,
    client: OllamaClientProtocol,
    config: OllamaGenerationConfig = DEFAULT_GENERATION_CONFIG,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run/resume exact local generations; every *model response* is terminal.

    A missing local runtime is an orchestration failure, not scientific evidence about
    the publication.  Re-check the exact runtime/model identity immediately before each
    request and stop the stage without freezing a row receipt when that preflight fails.
    This prevents one crashed Ollama process from being converted into a run of permanent
    paper-level extraction failures.  Responses that were actually returned -- including
    invalid JSON and length truncations -- remain immutable and are never retried.
    """

    bundle = validate_input_bundle(input_bundle)
    if config.model_dump(mode="json") != bundle["generation_config"]:
        raise NativeOllamaDiagnosticError("native_prediction_generation_config_mismatch")
    if limit is not None and limit < 1:
        raise NativeOllamaDiagnosticError("native_prediction_limit_not_positive")
    identity = client.inspect_identity(config)
    if (
        identity.model != config.model
        or identity.model_digest != config.model_digest
        or identity.ollama_version != config.expected_ollama_version
    ):
        raise NativeOllamaDiagnosticError("native_prediction_model_identity_mismatch")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    _validate_receipt_directory_membership(
        receipts_dir=receipts_dir,
        rows=bundle["rows"],
    )
    validated: dict[str, dict[str, Any]] = {}
    for row in bundle["rows"]:
        path = _receipt_path(receipts_dir, str(row["row_key"]))
        if not path.exists():
            continue
        validated[str(row["row_key"])] = validate_generation_receipt(
            _read_json_object(path),
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
        )
    missing = [row for row in bundle["rows"] if row["row_key"] not in validated]
    for row in missing if limit is None else missing[:limit]:
        try:
            current_identity = client.inspect_identity(config)
        except LocalOllamaError as exc:
            raise NativeOllamaDiagnosticError(
                "native_prediction_runtime_unavailable_before_request;"
                "no_row_receipt_was_frozen"
            ) from exc
        if current_identity != identity:
            raise NativeOllamaDiagnosticError(
                "native_prediction_runtime_identity_changed_before_request;"
                "no_row_receipt_was_frozen"
            )
        receipt = _make_generation_receipt(
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
            client=client,
        )
        if receipt["status"] == "execution_failure":
            raise NativeOllamaDiagnosticError(
                "native_prediction_transport_failed_without_model_response;"
                "no_row_receipt_was_frozen"
            )
        receipt = validate_generation_receipt(
            receipt,
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
        )
        path = _receipt_path(receipts_dir, str(row["row_key"]))
        atomic_write_json(path, receipt, force=False)
        validated[str(row["row_key"])] = receipt
    ordered = [validated[key] for key in sorted(validated)]
    ledger = _ledger_from_receipts(
        bundle=bundle,
        config=config,
        identity=identity,
        receipts=ordered,
    )
    validate_prediction_ledger(ledger, bundle=bundle, require_complete=False)
    atomic_write_json(
        prediction_ledger_path,
        ledger,
        force=prediction_ledger_path.exists(),
    )
    return ledger


def validate_frozen_receipts(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    receipts_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    bundle = validate_input_bundle(input_bundle)
    ledger = validate_prediction_ledger(
        prediction_ledger,
        bundle=bundle,
        require_complete=True,
    )
    config = OllamaGenerationConfig.model_validate(ledger["generation_config"])
    identity_payload = deepcopy(dict(ledger["model_identity"]))
    identity_payload.pop("identity_sha256")
    identity = OllamaIdentity.model_validate(identity_payload)
    rows = {str(row["row_key"]): row for row in bundle["rows"]}
    _validate_receipt_directory_membership(
        receipts_dir=receipts_dir,
        rows=list(rows.values()),
    )
    receipts: dict[str, dict[str, Any]] = {}
    for summary in ledger["receipts"]:
        key = str(summary["row_key"])
        receipt = validate_generation_receipt(
            _read_json_object(_receipt_path(receipts_dir, key)),
            bundle=bundle,
            row=rows[key],
            config=config,
            identity=identity,
        )
        for field in (
            "input_row_sha256",
            "official_output_sha256",
            "receipt_sha256",
            "row_key",
            "status",
        ):
            if receipt.get(field) != summary.get(field):
                raise NativeOllamaDiagnosticError("native_frozen_receipt_differs_from_ledger")
        receipts[key] = receipt
    if set(receipts) != set(rows):
        raise NativeOllamaDiagnosticError("native_frozen_receipt_population_incomplete")
    return bundle, ledger, receipts


def _fallback_extraction(receipt: Mapping[str, Any]) -> NativePublicationExtraction:
    status = str(receipt["status"])
    return NativePublicationExtraction(
        status=FragmentStatus.NON_ESTIMABLE,
        studies=[],
        non_estimability_reason=NonEstimabilityReason.OTHER,
        non_estimability_detail=(
            "The frozen local-model generation did not pass the official native extraction "
            f"contract ({status}); see private generation receipt {receipt['receipt_sha256']}."
        ),
        warnings=sorted(
            {
                f"generation_receipt_sha256:{receipt['receipt_sha256']}",
                f"local_ollama_terminal_status:{status}",
            }
        ),
    )


def _projection_scope_issues(
    *, extraction: NativePublicationExtraction, row: Mapping[str, Any]
) -> list[str]:
    """Require every estimable quote to occur in bytes actually shown to the model."""

    if extraction.status is FragmentStatus.NON_ESTIMABLE:
        return []
    source_record = row.get("source_record")
    source_document = (
        source_record.get("source_document")
        if isinstance(source_record, Mapping)
        else None
    )
    expected_locator = (
        source_document.get("source_locator")
        if isinstance(source_document, Mapping)
        else None
    )
    projection = row.get("source_projection")
    if not isinstance(projection, list):
        return ["projection_missing"]
    passages = [item for item in projection if isinstance(item, Mapping)]
    issues: set[str] = set()
    for study in extraction.studies:
        for cohort in study.cohorts:
            for finding in cohort.findings:
                evidence = finding.evidence
                if evidence.source_locator != expected_locator:
                    issues.add("source_locator_not_exposed")
                if not evidence.line_ids:
                    issues.add("line_ids_not_exposed")
                    continue
                cited_line_ids = set(evidence.line_ids)
                exposed_line_ids = {
                    str(passage["line_id"])
                    for passage in passages
                    if isinstance(passage.get("line_id"), str)
                }
                matching = [
                    passage
                    for passage in passages
                    if passage.get("line_id") in cited_line_ids
                ]
                if not cited_line_ids.issubset(exposed_line_ids):
                    issues.add("line_id_not_exposed")
                if not isinstance(evidence.quote, str) or not evidence.quote:
                    issues.add("quote_missing")
                    quote_matching: list[Mapping[str, Any]] = []
                else:
                    quote_matching = [
                        passage
                        for passage in matching
                        if isinstance(passage.get("text"), str)
                        and evidence.quote in str(passage["text"])
                    ]
                    if not quote_matching:
                        issues.add("quote_not_exact_exposed_passage")
                if evidence.section is not None and not any(
                    passage.get("section") == evidence.section
                    for passage in quote_matching
                ):
                    issues.add("section_not_exposed_quote_passage")
    return sorted(issues)


def _projection_scope_fallback_extraction(
    *, receipt: Mapping[str, Any], issues: Sequence[str]
) -> NativePublicationExtraction:
    return NativePublicationExtraction(
        status=FragmentStatus.NON_ESTIMABLE,
        studies=[],
        non_estimability_reason=NonEstimabilityReason.UNGROUNDED_NUMERICAL_RESULT,
        non_estimability_detail=(
            "The frozen local-model output used evidence outside the exact source "
            "projection shown to that row; see private generation receipt "
            f"{receipt['receipt_sha256']}."
        ),
        warnings=sorted(
            {
                f"generation_receipt_sha256:{receipt['receipt_sha256']}",
                *(f"projection_scope_issue:{issue}" for issue in issues),
            }
        ),
    )


def _aggregate_generation(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "eval_count",
        "eval_duration_ns",
    )
    sums = {field: 0 for field in fields}
    observed = {field: 0 for field in fields}
    for receipt in receipts:
        telemetry = receipt.get("telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        for field in fields:
            value = telemetry.get(field)
            if isinstance(value, int) and value >= 0:
                sums[field] += value
                observed[field] += 1
    return {
        "attempt_accounting_scope": "response_bearing_terminal_receipts_only",
        "generation_call_attempts": sum(
            receipt.get("generation_call_attempted_once") is True for receipt in receipts
        ),
        "generation_retries": 0,
        "generation_omissions": EXPECTED_MANIFEST_RECORDS - len(receipts),
        "model_execution_attempts": sum(
            receipt.get("telemetry") is not None for receipt in receipts
        ),
        "model_responses_received": sum(
            receipt.get("response_text") is not None for receipt in receipts
        ),
        "terminal_receipts": len(receipts),
        "status_counts": dict(
            sorted(Counter(str(receipt["status"]) for receipt in receipts).items())
        ),
        "generation_truncations": sum(
            receipt.get("generation_truncated") is True for receipt in receipts
        ),
        "reported_field_sums": sums,
        "reported_field_observation_counts": observed,
        "reported_total_duration_seconds": sums["total_duration_ns"] / 1_000_000_000,
        "reported_prompt_tokens": sums["prompt_eval_count"],
        "reported_generated_tokens": sums["eval_count"],
    }


def _process_peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _graph_counts(graph: Any) -> dict[str, int]:
    return {
        "arms": len(graph.arms),
        "cohorts": len(graph.cohorts),
        "contrasts": len(graph.contrasts),
        "evidence_spans": len(graph.evidence_spans),
        "outcome_estimates": len(graph.outcome_estimates),
        "publications": len(graph.publications),
        "studies": len(graph.studies),
    }


def _freeze_ollama_extraction_context(
    *,
    bundle: Mapping[str, Any],
    ledger: Mapping[str, Any],
    receipts_by_key: Mapping[str, Mapping[str, Any]],
    manifest: NativeSourceManifest,
    repository_root: Path,
) -> Any:
    """Bind exact private prompts/schemas and inspected local-model call receipts."""

    root = repository_root.resolve(strict=True)
    question_config = load_config_for_question(
        EXPECTED_QUESTION_ID,
        root=root,
        require_locked=True,
    )
    question_config_path = root / "configs/questions/antiox-training.yaml"
    if sha256_file(question_config_path) != bundle["question_config_file_sha256"]:
        raise NativeOllamaDiagnosticError(
            "native_extraction_context_question_config_file_mismatch"
        )
    prompt_template_path = root / "prompts/native_extraction.md"
    rows = {str(row["row_key"]): row for row in bundle["rows"]}
    rendered_prompts = [
        NativeRenderedPromptArtifact(
            prompt_id=f"row-{key}",
            renderer_id="native-ollama-row-projection-v1",
            prompt_version=str(bundle["prompt_version"]),
            template_path="prompts/native_extraction.md",
            template_sha256=sha256_file(prompt_template_path),
            rendered_prompt=render_prediction_prompt(bundle, rows[key]),
            rendered_prompt_sha256=hashlib.sha256(
                render_prediction_prompt(bundle, rows[key]).encode("utf-8")
            ).hexdigest(),
        )
        for key in sorted(rows)
    ]
    official_schema = native_publication_extraction_json_schema()
    schemas = [
        NativeEvaluationSchemaArtifact(
            schema_id="native-official-postvalidation",
            role="official_postvalidation",
            schema_payload=official_schema,
            schema_sha256=hash_canonical(official_schema),
        )
    ]
    by_generation_hash: dict[str, dict[str, Any]] = {}
    for key in sorted(rows):
        schema = generation_schema_for_row(rows[key])
        by_generation_hash[hash_canonical(schema)] = schema
    schemas.extend(
        NativeEvaluationSchemaArtifact(
            schema_id=f"ollama-generation-{schema_sha256[:20]}",
            role="generation_constraint",
            schema_payload=schema,
            schema_sha256=schema_sha256,
        )
        for schema_sha256, schema in sorted(by_generation_hash.items())
    )
    generation_receipts = [
        dict(receipts_by_key[key]) for key in sorted(receipts_by_key)
    ]
    identity = dict(ledger["model_identity"])
    raw_call_ledger = {
        "ledger_version": "native-ollama-raw-call-ledger-v1",
        "prediction_ledger": dict(ledger),
        "generation_receipts": generation_receipts,
    }
    provider_receipt = freeze_native_provider_execution_receipt(
        execution_id=f"ollama-ledger-{str(ledger['prediction_ledger_sha256'])[:20]}",
        execution_mode="ollama_local",
        provider_id="local-ollama",
        model_id=str(identity["model"]),
        model_revision=str(identity["model_digest"]),
        runtime_id="ollama",
        runtime_version=str(identity["ollama_version"]),
        runtime_metadata={
            "diagnostic_execution_sha256": str(bundle["diagnostic_execution_sha256"]),
            "generation_config_sha256": str(ledger["generation_config_sha256"]),
            "model_format": str(identity["model_format"]),
            "model_family": str(identity["model_family"]),
            "parameter_size": str(identity["parameter_size"]),
            "quantization_level": str(identity["quantization_level"]),
        },
        raw_call_ledger=raw_call_ledger,
        call_count=len(generation_receipts),
    )
    source_manifest_path = root / str(
        _validated_config(root / str(bundle["diagnostic_config_path"])).source_manifest_path
    )
    artifacts = [
        NativeExtractionArtifactDigest(
            artifact_id="source-manifest-input",
            role="source_manifest_input",
            sha256=sha256_file(source_manifest_path),
            hash_basis="raw_bytes",
            byte_count=source_manifest_path.stat().st_size,
        ),
        NativeExtractionArtifactDigest(
            artifact_id="prediction-input-bundle",
            role="prediction_input_bundle",
            sha256=str(bundle["input_bundle_sha256"]),
            hash_basis="canonical_json",
        ),
        NativeExtractionArtifactDigest(
            artifact_id="prediction-ledger",
            role="prediction_ledger",
            sha256=str(ledger["prediction_ledger_sha256"]),
            hash_basis="canonical_json",
            execution_ids=[provider_receipt.execution_id],
        ),
    ]
    artifacts.extend(
        NativeExtractionArtifactDigest(
            artifact_id=f"generation-receipt-{index:03d}",
            role="generation_receipt",
            sha256=str(receipt["receipt_sha256"]),
            hash_basis="canonical_json",
            execution_ids=[provider_receipt.execution_id],
        )
        for index, receipt in enumerate(generation_receipts, start=1)
    )
    return freeze_native_extraction_execution_context(
        extraction_mode="ollama_local",
        question_config=question_config,
        pipeline_fingerprint_sha256=str(bundle["pipeline_sha256"]),
        rendered_prompts=rendered_prompts,
        evaluation_schemas=schemas,
        provider_execution_receipts=[provider_receipt],
        input_artifacts=artifacts,
        source_manifest_content_sha256=hash_canonical(manifest),
        source_manifest_records=len(manifest.records),
        corpus_cutoff=EXPECTED_CORPUS_CUTOFF,
    )


def finalize_diagnostic(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    receipts_dir: Path,
    repository_root: Path,
    private_output_dir: Path,
    generated_at: datetime | None = None,
    budget_minutes: float = 60.0,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build/replay all 19 fragments and issue an explicitly abstaining certificate."""

    started = time.monotonic()
    bundle, ledger, receipts_by_key = validate_frozen_receipts(
        input_bundle=input_bundle,
        prediction_ledger=prediction_ledger,
        receipts_dir=receipts_dir,
    )
    root = repository_root.resolve(strict=True)
    bundle = validate_current_diagnostic_context(bundle, repository_root=root)
    pipeline = PipelineFingerprint.model_validate(bundle["pipeline_fingerprint"])
    config_path = root / str(bundle["diagnostic_config_path"])
    diagnostic_config = _validated_config(config_path)
    rows = {str(row["row_key"]): row for row in bundle["rows"]}
    records_by_key = {
        key: NativeSourceRecord.model_validate(row["source_record"])
        for key, row in rows.items()
    }
    records = sorted(records_by_key.values(), key=lambda record: record.doc_id)
    manifest = NativeSourceManifest(
        question_id=EXPECTED_QUESTION_ID,
        records=records,
    )
    extraction_context = _freeze_ollama_extraction_context(
        bundle=bundle,
        ledger=ledger,
        receipts_by_key=receipts_by_key,
        manifest=manifest,
        repository_root=root,
    )
    grounding_receipts: list[NativeGroundingReceipt] = []
    fragments = []
    attempted_estimable = 0
    downgraded_causes: Counter[str] = Counter()
    for key in sorted(rows):
        row = rows[key]
        receipt = receipts_by_key[key]
        record = records_by_key[key]
        official = receipt.get("official_output")
        extraction = (
            NativePublicationExtraction.model_validate(official)
            if official is not None
            else _fallback_extraction(receipt)
        )
        if extraction.status is FragmentStatus.ESTIMABLE:
            attempted_estimable += 1
            projection_issues = _projection_scope_issues(
                extraction=extraction,
                row=row,
            )
            if projection_issues:
                for issue in projection_issues:
                    downgraded_causes[f"projection_scope:{issue}"] += 1
                extraction = _projection_scope_fallback_extraction(
                    receipt=receipt,
                    issues=projection_issues,
                )
        grounding = verify_native_publication_grounding(
            repository_root=root,
            source_document=record.source_document,
            extraction=extraction,
        )
        grounding_receipts.append(grounding)
        fragment = freeze_grounding_checked_publication_fragment(
            extraction=extraction,
            grounding_receipt=grounding,
            question_id=EXPECTED_QUESTION_ID,
            publication=record.publication,
            pipeline_fingerprint_sha256=pipeline.pipeline_sha256,
            extraction_context_sha256=extraction_context.context_sha256,
            source_document=record.source_document,
        )
        fragments.append(fragment)
        if (
            extraction.status is FragmentStatus.ESTIMABLE
            and fragment.status is not FragmentStatus.ESTIMABLE
        ):
            if not grounding.source_verified:
                downgraded_causes["source_verification_failure"] += 1
            for result in grounding.finding_results:
                if result.status.value != "exact":
                    downgraded_causes[f"grounding_{result.status.value}"] += 1
                    for issue in result.issues:
                        downgraded_causes[f"grounding_issue:{issue}"] += 1
    corpus = assemble_typed_evidence_corpus(fragments)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=grounding_receipts,
        source_manifest=manifest,
        corpus_cutoff=EXPECTED_CORPUS_CUTOFF,
        extraction_context=extraction_context,
    )
    private_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        private_output_dir / "native-extraction-context.json",
        extraction_context,
        force=force,
    )
    package_path = private_output_dir / "typed-evidence-grounding-package.json"
    atomic_write_json(package_path, package, force=force)
    replay = reverify_typed_evidence_grounding_package(package=package, repository_root=root)
    atomic_write_json(private_output_dir / "grounding-replay.json", replay, force=force)
    atomic_write_json(private_output_dir / "pipeline-fingerprint.json", pipeline, force=force)
    claim = ClaimManifest.model_validate(diagnostic_config.claim_manifest)
    if hash_canonical(claim) != bundle["claim_manifest_sha256"]:
        raise NativeOllamaDiagnosticError("native_diagnostic_claim_changed_after_prediction_freeze")
    atomic_write_json(private_output_dir / "claim-manifest.json", claim, force=force)
    loaded = load_corpus(
        package_path,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=root,
    )
    certificate = run_verification(
        manifest=claim,
        corpus=loaded,
        budget_minutes=budget_minutes,
        expected_pipeline_fingerprint=pipeline,
        pipeline_root=root,
        generated_at=generated_at or datetime.now(UTC),
    )
    if certificate.status != "abstained":
        raise NativeOllamaDiagnosticError("native_diagnostic_certificate_must_abstain")
    certificate_dir = private_output_dir / "certificate"
    write_certificate_artifacts(certificate, certificate_dir, force=force)
    generation_receipts = [receipts_by_key[key] for key in sorted(receipts_by_key)]
    grounding_finding_counts = Counter(
        result.status.value for receipt in grounding_receipts for result in receipt.finding_results
    )
    fragment_status_counts = Counter(fragment.status.value for fragment in fragments)
    non_estimability_counts = Counter(
        fragment.non_estimability_reason.value
        for fragment in fragments
        if fragment.non_estimability_reason is not None
    )
    blocker_codes = sorted(
        {
            str(issue["code"])
            for issue in certificate.adapter_issues
            if issue.get("severity") == "blocking" and issue.get("code")
        }
    )
    synthesis_mode = str(certificate.synthesis.get("mode", "unknown"))
    synthesis_status = str(certificate.synthesis.get("status", "unknown"))
    synthesis_reason_raw = certificate.synthesis.get("reason")
    synthesis_reason = (
        str(synthesis_reason_raw) if synthesis_reason_raw is not None else None
    )
    context_receipt = package.extraction_context_receipt
    assert context_receipt is not None
    private_payload = {
        "private_report_version": PRIVATE_REPORT_VERSION,
        "diagnostic_version": NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
        "status": "complete_abstaining_diagnostic",
        "scientific_role": "diagnostic_only_no_independent_numerical_gold",
        "selection_scope": EXPECTED_SELECTION_SCOPE,
        "selection_labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "accuracy_evaluated": False,
        "human_audit_completed": False,
        "retrieval_recall_evaluated": False,
        "multimodal_extraction_evaluated": False,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "source_bridge_run_sha256": bundle["source_bridge_run_sha256"],
        "source_manifest_file_sha256": bundle["source_manifest_file_sha256"],
        "source_manifest_content_sha256": bundle["source_manifest_content_sha256"],
        "source_manifest_records": len(records),
        "corpus_cutoff": EXPECTED_CORPUS_CUTOFF,
        "pipeline_sha256": pipeline.pipeline_sha256,
        "question_config_sha256": extraction_context.question_config_sha256,
        "extraction_context_sha256": extraction_context.context_sha256,
        "extraction_context_receipt_sha256": context_receipt.receipt_sha256,
        "rendered_prompt_set_sha256": hash_canonical(
            sorted(
                prompt.rendered_prompt_sha256
                for prompt in extraction_context.rendered_prompts
            )
        ),
        "evaluation_schema_set_sha256": hash_canonical(
            sorted(
                schema.schema_sha256
                for schema in extraction_context.evaluation_schemas
            )
        ),
        "provider_execution_receipt_sha256s": sorted(
            receipt.receipt_sha256
            for receipt in extraction_context.provider_execution_receipts
        ),
        "diagnostic_execution_sha256": bundle["diagnostic_execution_sha256"],
        "model_identity": ledger["model_identity"],
        "generation_config_sha256": ledger["generation_config_sha256"],
        "official_schema_sha256": bundle["official_schema_sha256"],
        "generation_schema_algorithm": bundle["generation_schema_algorithm"],
        "generation": _aggregate_generation(generation_receipts),
        "extraction": {
            "official_schema_valid_outputs": sum(
                receipt["status"] == "official_schema_valid" for receipt in generation_receipts
            ),
            "official_estimable_attempts": attempted_estimable,
            "fragment_status_counts": dict(sorted(fragment_status_counts.items())),
            "non_estimability_reason_counts": dict(sorted(non_estimability_counts.items())),
            "grounding_receipts": len(grounding_receipts),
            "source_verified_receipts": sum(
                receipt.source_verified for receipt in grounding_receipts
            ),
            "source_verification_failures": sum(
                not receipt.source_verified for receipt in grounding_receipts
            ),
            "authorizing_receipts": sum(
                receipt.authorizes_estimable_fragment for receipt in grounding_receipts
            ),
            "exact_grounded_findings": grounding_finding_counts.get("exact", 0),
            "grounding_finding_status_counts": dict(sorted(grounding_finding_counts.items())),
            "downgraded_estimable_attempts": attempted_estimable
            - len(corpus.estimable_publication_ids),
            "downgrade_cause_counts": dict(sorted(downgraded_causes.items())),
        },
        "typed_corpus_sha256": corpus.corpus_sha256,
        "grounding_package_sha256": package.package_sha256,
        "grounding_replay_sha256": replay.replay_sha256,
        "graph_counts": _graph_counts(loaded.graph),
        "synthesis_mode": synthesis_mode,
        "synthesis_status": synthesis_status,
        "synthesis_reason": synthesis_reason,
        "certificate_status": certificate.status,
        "certificate_version": certificate.certificate_version,
        "certificate_run_id": certificate.run_id,
        "certificate_sha256": certificate.certificate_sha256,
        "certificate_reasons": list(certificate.reasons),
        "certificate_blocker_codes": blocker_codes,
        "complete_corpus_membership_sha256": (
            certificate.complete_corpus_identity.membership_sha256
        ),
        "runtime": {
            "finalization_wall_seconds": time.monotonic() - started,
            "coordinator_process_peak_rss_bytes": _process_peak_rss_bytes(),
            "ollama_server_rss_measured": False,
        },
        "caveats": list(_PUBLIC_CAVEATS),
    }
    private_report = {
        **private_payload,
        "private_report_sha256": hash_canonical(private_payload),
    }
    validate_private_report(private_report)
    public_summary = build_public_summary(private_report)
    validate_public_summary(public_summary)
    atomic_write_json(private_output_dir / "private-report.json", private_report, force=force)
    return private_report, public_summary


def validate_private_report(report: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(report))
    _validate_self_hash(snapshot, "private_report_sha256", artifact="native_private_report")
    context_hash_fields = (
        "evaluation_schema_set_sha256",
        "extraction_context_receipt_sha256",
        "extraction_context_sha256",
        "question_config_sha256",
        "rendered_prompt_set_sha256",
    )
    provider_receipt_hashes = snapshot.get("provider_execution_receipt_sha256s")
    if (
        snapshot.get("private_report_version") != PRIVATE_REPORT_VERSION
        or snapshot.get("diagnostic_version") != NATIVE_OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("status") != "complete_abstaining_diagnostic"
        or snapshot.get("accuracy_evaluated") is not False
        or snapshot.get("human_audit_completed") is not False
        or snapshot.get("retrieval_recall_evaluated") is not False
        or snapshot.get("multimodal_extraction_evaluated") is not False
        or snapshot.get("certificate_status") != "abstained"
        or snapshot.get("certificate_version")
        != "literature-multiverse-verification-v5"
        or not isinstance(snapshot.get("certificate_run_id"), str)
        or re.fullmatch(r"verify-[0-9a-f]{16}", snapshot["certificate_run_id"])
        is None
        or not isinstance(snapshot.get("certificate_reasons"), list)
        or not snapshot["certificate_reasons"]
        or any(
            not isinstance(reason, str) or not reason
            for reason in snapshot["certificate_reasons"]
        )
        or snapshot["certificate_reasons"]
        != sorted(set(snapshot["certificate_reasons"]))
        or not isinstance(snapshot.get("complete_corpus_membership_sha256"), str)
        or _SHA256.fullmatch(snapshot["complete_corpus_membership_sha256"])
        is None
        or snapshot.get("source_manifest_records") != EXPECTED_MANIFEST_RECORDS
        or any(
            not isinstance(snapshot.get(field), str)
            or _SHA256.fullmatch(str(snapshot[field])) is None
            for field in context_hash_fields
        )
        or not isinstance(provider_receipt_hashes, list)
        or provider_receipt_hashes != sorted(set(provider_receipt_hashes))
        or not provider_receipt_hashes
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in provider_receipt_hashes
        )
    ):
        raise NativeOllamaDiagnosticError("native_private_report_scope_mismatch")
    return snapshot


def build_public_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    private = validate_private_report(report)
    payload = {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "diagnostic_version": NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
        "status": private["status"],
        "scientific_role": private["scientific_role"],
        "selection_scope": private["selection_scope"],
        "selection_labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "accuracy_evaluated": False,
        "human_audit_completed": False,
        "retrieval_recall_evaluated": False,
        "multimodal_extraction_evaluated": False,
        "population_count": private["source_manifest_records"],
        "input_bundle_sha256": private["input_bundle_sha256"],
        "prediction_ledger_sha256": private["prediction_ledger_sha256"],
        "source_bridge_run_sha256": private["source_bridge_run_sha256"],
        "source_manifest_file_sha256": private["source_manifest_file_sha256"],
        "source_manifest_content_sha256": private["source_manifest_content_sha256"],
        "corpus_cutoff_sha256": hashlib.sha256(str(private["corpus_cutoff"]).encode()).hexdigest(),
        "pipeline_sha256": private["pipeline_sha256"],
        "question_config_sha256": private["question_config_sha256"],
        "extraction_context_sha256": private["extraction_context_sha256"],
        "extraction_context_receipt_sha256": private[
            "extraction_context_receipt_sha256"
        ],
        "rendered_prompt_set_sha256": private["rendered_prompt_set_sha256"],
        "evaluation_schema_set_sha256": private[
            "evaluation_schema_set_sha256"
        ],
        "provider_execution_receipt_sha256s": private[
            "provider_execution_receipt_sha256s"
        ],
        "diagnostic_execution_sha256": private["diagnostic_execution_sha256"],
        "model": {
            "name": private["model_identity"]["model"],
            "digest": private["model_identity"]["model_digest"],
            "runtime_version": private["model_identity"]["ollama_version"],
            "parameter_size": private["model_identity"]["parameter_size"],
            "quantization_level": private["model_identity"]["quantization_level"],
            "identity_sha256": private["model_identity"]["identity_sha256"],
        },
        "generation_config_sha256": private["generation_config_sha256"],
        "official_schema_sha256": private["official_schema_sha256"],
        "generation_schema_algorithm": private["generation_schema_algorithm"],
        "generation": private["generation"],
        "extraction": private["extraction"],
        "typed_corpus_sha256": private["typed_corpus_sha256"],
        "grounding_package_sha256": private["grounding_package_sha256"],
        "grounding_replay_sha256": private["grounding_replay_sha256"],
        "graph_counts": private["graph_counts"],
        "synthesis_mode": private["synthesis_mode"],
        "synthesis_status": private["synthesis_status"],
        "synthesis_reason": private["synthesis_reason"],
        "certificate_status": private["certificate_status"],
        "certificate_version": private["certificate_version"],
        "certificate_run_id": private["certificate_run_id"],
        "certificate_sha256": private["certificate_sha256"],
        "certificate_reasons": private["certificate_reasons"],
        "certificate_blocker_codes": private["certificate_blocker_codes"],
        "complete_corpus_membership_sha256": private[
            "complete_corpus_membership_sha256"
        ],
        "runtime": private["runtime"],
        "caveats": private["caveats"],
        "private_report_sha256": private["private_report_sha256"],
    }
    return {**payload, "public_summary_sha256": hash_canonical(payload)}


def validate_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(summary))
    _validate_self_hash(snapshot, "public_summary_sha256", artifact="native_public_summary")
    expected_top_level = {
        "accuracy_evaluated",
        "caveats",
        "certificate_blocker_codes",
        "certificate_reasons",
        "certificate_run_id",
        "certificate_sha256",
        "certificate_status",
        "certificate_version",
        "complete_corpus_membership_sha256",
        "corpus_cutoff_sha256",
        "diagnostic_execution_sha256",
        "diagnostic_version",
        "extraction",
        "extraction_context_receipt_sha256",
        "extraction_context_sha256",
        "evaluation_schema_set_sha256",
        "generation",
        "generation_config_sha256",
        "generation_schema_algorithm",
        "graph_counts",
        "grounding_package_sha256",
        "grounding_replay_sha256",
        "human_audit_completed",
        "input_bundle_sha256",
        "model",
        "multimodal_extraction_evaluated",
        "official_schema_sha256",
        "pipeline_sha256",
        "population_count",
        "pristine_final_holdout_eligible",
        "private_report_sha256",
        "prediction_ledger_sha256",
        "provider_execution_receipt_sha256s",
        "question_config_sha256",
        "rendered_prompt_set_sha256",
        "public_summary_sha256",
        "public_summary_version",
        "retrieval_recall_evaluated",
        "runtime",
        "scientific_role",
        "selection_labels_previously_opened",
        "selection_scope",
        "source_bridge_run_sha256",
        "source_manifest_content_sha256",
        "source_manifest_file_sha256",
        "status",
        "synthesis_mode",
        "synthesis_reason",
        "synthesis_status",
        "typed_corpus_sha256",
    }
    if (
        set(snapshot) != expected_top_level
        or snapshot.get("scientific_role")
        != "diagnostic_only_no_independent_numerical_gold"
        or snapshot.get("selection_scope") != EXPECTED_SELECTION_SCOPE
        or snapshot.get("public_summary_version") != PUBLIC_SUMMARY_VERSION
        or snapshot.get("diagnostic_version") != NATIVE_OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("status") != "complete_abstaining_diagnostic"
        or snapshot.get("selection_labels_previously_opened") is not True
        or snapshot.get("pristine_final_holdout_eligible") is not False
        or snapshot.get("accuracy_evaluated") is not False
        or snapshot.get("human_audit_completed") is not False
        or snapshot.get("retrieval_recall_evaluated") is not False
        or snapshot.get("multimodal_extraction_evaluated") is not False
        or snapshot.get("certificate_status") != "abstained"
        or snapshot.get("population_count") != EXPECTED_MANIFEST_RECORDS
        or snapshot.get("generation_schema_algorithm") != GENERATION_SCHEMA_ALGORITHM
        or snapshot.get("generation_config_sha256")
        != DEFAULT_GENERATION_CONFIG.config_sha256
        or snapshot.get("official_schema_sha256")
        != hash_canonical(native_publication_extraction_json_schema())
        or snapshot.get("certificate_version")
        != "literature-multiverse-verification-v5"
        or not isinstance(snapshot.get("certificate_run_id"), str)
        or re.fullmatch(r"verify-[0-9a-f]{16}", snapshot["certificate_run_id"])
        is None
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_scope_mismatch")
    expected_hash_fields = {
        "certificate_sha256",
        "complete_corpus_membership_sha256",
        "corpus_cutoff_sha256",
        "diagnostic_execution_sha256",
        "extraction_context_receipt_sha256",
        "extraction_context_sha256",
        "evaluation_schema_set_sha256",
        "generation_config_sha256",
        "grounding_package_sha256",
        "grounding_replay_sha256",
        "input_bundle_sha256",
        "official_schema_sha256",
        "pipeline_sha256",
        "private_report_sha256",
        "prediction_ledger_sha256",
        "question_config_sha256",
        "rendered_prompt_set_sha256",
        "public_summary_sha256",
        "source_bridge_run_sha256",
        "source_manifest_content_sha256",
        "source_manifest_file_sha256",
        "typed_corpus_sha256",
    }
    if any(
        not isinstance(snapshot.get(field), str)
        or _SHA256.fullmatch(str(snapshot[field])) is None
        for field in expected_hash_fields
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_hash_field_invalid")
    provider_receipt_hashes = snapshot.get("provider_execution_receipt_sha256s")
    if (
        not isinstance(provider_receipt_hashes, list)
        or provider_receipt_hashes != sorted(set(provider_receipt_hashes))
        or not provider_receipt_hashes
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in provider_receipt_hashes
        )
    ):
        raise NativeOllamaDiagnosticError(
            "native_public_summary_provider_receipt_hashes_invalid"
        )
    reasons = snapshot.get("certificate_reasons")
    blocker_codes = snapshot.get("certificate_blocker_codes")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or reasons != sorted(set(reasons))
        or not isinstance(blocker_codes, list)
        or any(not isinstance(code, str) or not code for code in blocker_codes)
        or blocker_codes != sorted(set(blocker_codes))
        or any(_PUBLIC_AGGREGATE_CODE.fullmatch(code) is None for code in blocker_codes)
    ):
        raise NativeOllamaDiagnosticError(
            "native_public_summary_certificate_reasons_invalid"
        )
    generation = snapshot.get("generation")
    expected_generation_fields = {
        "attempt_accounting_scope",
        "generation_call_attempts",
        "generation_omissions",
        "generation_retries",
        "generation_truncations",
        "model_execution_attempts",
        "model_responses_received",
        "reported_field_observation_counts",
        "reported_field_sums",
        "reported_generated_tokens",
        "reported_prompt_tokens",
        "reported_total_duration_seconds",
        "status_counts",
        "terminal_receipts",
    }
    status_counts = generation.get("status_counts") if isinstance(generation, Mapping) else None
    status_counts_valid = isinstance(status_counts, Mapping) and all(
        isinstance(status, str) and type(count) is int and count >= 0
        for status, count in status_counts.items()
    )
    generation_integer_fields = (
        "generation_call_attempts",
        "generation_omissions",
        "generation_retries",
        "generation_truncations",
        "model_execution_attempts",
        "model_responses_received",
        "reported_generated_tokens",
        "reported_prompt_tokens",
        "terminal_receipts",
    )
    generation_integers_valid = isinstance(generation, Mapping) and all(
        type(generation.get(field)) is int and generation[field] >= 0
        for field in generation_integer_fields
    )
    reported_duration = (
        generation.get("reported_total_duration_seconds")
        if isinstance(generation, Mapping)
        else None
    )
    if (
        not isinstance(generation, Mapping)
        or set(generation) != expected_generation_fields
        or generation.get("attempt_accounting_scope")
        != "response_bearing_terminal_receipts_only"
        or not generation_integers_valid
        or not isinstance(reported_duration, (int, float))
        or isinstance(reported_duration, bool)
        or not math.isfinite(reported_duration)
        or reported_duration < 0
        or generation.get("terminal_receipts") != EXPECTED_MANIFEST_RECORDS
        or generation.get("generation_omissions") != 0
        or generation.get("generation_call_attempts") != EXPECTED_MANIFEST_RECORDS
        or generation.get("generation_retries") != 0
        or not status_counts_valid
        or set(status_counts)
        - {
            "execution_failure",
            "generation_truncated",
            "official_schema_invalid",
            "official_schema_valid",
            "response_json_invalid",
        }
        or sum(status_counts.values()) != EXPECTED_MANIFEST_RECORDS
        or generation.get("model_responses_received")
        != generation.get("model_execution_attempts")
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_generation_counts_invalid")
    extraction = snapshot.get("extraction")
    expected_extraction_fields = {
        "authorizing_receipts",
        "downgrade_cause_counts",
        "downgraded_estimable_attempts",
        "exact_grounded_findings",
        "fragment_status_counts",
        "grounding_finding_status_counts",
        "grounding_receipts",
        "non_estimability_reason_counts",
        "official_estimable_attempts",
        "official_schema_valid_outputs",
        "source_verification_failures",
        "source_verified_receipts",
    }
    fragment_counts = (
        extraction.get("fragment_status_counts")
        if isinstance(extraction, Mapping)
        else None
    )
    fragment_counts_valid = isinstance(fragment_counts, Mapping) and all(
        status in {"estimable", "non_estimable"}
        and type(count) is int
        and count >= 0
        for status, count in fragment_counts.items()
    )
    extraction_integer_fields = (
        "authorizing_receipts",
        "downgraded_estimable_attempts",
        "exact_grounded_findings",
        "grounding_receipts",
        "official_estimable_attempts",
        "official_schema_valid_outputs",
        "source_verification_failures",
        "source_verified_receipts",
    )
    extraction_integers_valid = isinstance(extraction, Mapping) and all(
        type(extraction.get(field)) is int and extraction[field] >= 0
        for field in extraction_integer_fields
    )
    if (
        not isinstance(extraction, Mapping)
        or set(extraction) != expected_extraction_fields
        or not extraction_integers_valid
        or extraction.get("grounding_receipts") != EXPECTED_MANIFEST_RECORDS
        or extraction.get("source_verified_receipts", 0)
        + extraction.get("source_verification_failures", 0)
        != EXPECTED_MANIFEST_RECORDS
        or not fragment_counts_valid
        or sum(fragment_counts.values()) != EXPECTED_MANIFEST_RECORDS
        or extraction.get("authorizing_receipts") != fragment_counts.get("estimable", 0)
        or extraction.get("official_estimable_attempts", 0)
        > extraction.get("official_schema_valid_outputs", 0)
        or extraction.get("downgraded_estimable_attempts")
        != extraction.get("official_estimable_attempts", 0)
        - fragment_counts.get("estimable", 0)
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_extraction_counts_invalid")
    graph_counts = snapshot.get("graph_counts")
    expected_graph_count_fields = {
        "arms",
        "cohorts",
        "contrasts",
        "evidence_spans",
        "outcome_estimates",
        "publications",
        "studies",
    }
    if (
        not isinstance(graph_counts, Mapping)
        or set(graph_counts) != expected_graph_count_fields
        or graph_counts.get("publications") != EXPECTED_MANIFEST_RECORDS
        or any(type(count) is not int or count < 0 for count in graph_counts.values())
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_graph_counts_invalid")
    telemetry_fields = {
        "eval_count",
        "eval_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "total_duration_ns",
    }
    for field in ("reported_field_observation_counts", "reported_field_sums"):
        aggregate = generation.get(field)
        if (
            not isinstance(aggregate, Mapping)
            or set(aggregate) != telemetry_fields
            or any(type(count) is not int or count < 0 for count in aggregate.values())
        ):
            raise NativeOllamaDiagnosticError(
                "native_public_summary_generation_telemetry_invalid"
            )
    for field in (
        "downgrade_cause_counts",
        "grounding_finding_status_counts",
        "non_estimability_reason_counts",
    ):
        aggregate = extraction.get(field)
        if (
            not isinstance(aggregate, Mapping)
            or any(
                not isinstance(code, str)
                or _PUBLIC_AGGREGATE_CODE.fullmatch(code) is None
                or type(count) is not int
                or count < 0
                for code, count in aggregate.items()
            )
        ):
            raise NativeOllamaDiagnosticError(
                "native_public_summary_extraction_aggregate_invalid"
            )
    model = snapshot.get("model")
    if (
        not isinstance(model, Mapping)
        or set(model)
        != {
            "digest",
            "identity_sha256",
            "name",
            "parameter_size",
            "quantization_level",
            "runtime_version",
        }
        or model.get("name") != DEFAULT_MODEL
        or model.get("digest") != DEFAULT_MODEL_DIGEST
        or model.get("runtime_version") != DEFAULT_OLLAMA_VERSION
        or not isinstance(model.get("parameter_size"), str)
        or not isinstance(model.get("quantization_level"), str)
        or not isinstance(model.get("identity_sha256"), str)
        or _SHA256.fullmatch(model["identity_sha256"]) is None
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_model_invalid")
    runtime = snapshot.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "coordinator_process_peak_rss_bytes",
            "finalization_wall_seconds",
            "ollama_server_rss_measured",
        }
        or type(runtime.get("coordinator_process_peak_rss_bytes")) is not int
        or runtime["coordinator_process_peak_rss_bytes"] < 0
        or not isinstance(runtime.get("finalization_wall_seconds"), (int, float))
        or isinstance(runtime.get("finalization_wall_seconds"), bool)
        or not math.isfinite(runtime["finalization_wall_seconds"])
        or runtime["finalization_wall_seconds"] < 0
        or runtime.get("ollama_server_rss_measured") is not False
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_runtime_invalid")
    if snapshot.get("caveats") != _PUBLIC_CAVEATS:
        raise NativeOllamaDiagnosticError("native_public_summary_caveats_invalid")
    if snapshot.get("synthesis_mode") not in {
        "directional_sign_synthesis",
        "evidence_graph_contract",
        "insufficient",
        "random_effects_meta_analysis",
    }:
        raise NativeOllamaDiagnosticError("native_public_summary_synthesis_mode_invalid")
    synthesis_reason = snapshot.get("synthesis_reason")
    if (
        snapshot.get("synthesis_status") not in {"insufficient", "ok"}
        or (
            synthesis_reason is not None
            and (
                not isinstance(synthesis_reason, str)
                or _PUBLIC_AGGREGATE_CODE.fullmatch(synthesis_reason) is None
            )
        )
        or (snapshot.get("synthesis_status") == "insufficient")
        != isinstance(synthesis_reason, str)
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_synthesis_status_invalid")
    forbidden = _forbidden_key_path(snapshot, _FORBIDDEN_PUBLIC_KEYS)
    if forbidden is not None:
        raise NativeOllamaDiagnosticError(f"native_public_summary_sensitive_key:{forbidden}")
    serialized = json.dumps(snapshot, sort_keys=True)
    if (
        _PUBLIC_ABSOLUTE_PATH.search(serialized)
        or _PUBLIC_ARTICLE_ID.search(serialized)
        or _PUBLIC_DOI.search(serialized)
    ):
        raise NativeOllamaDiagnosticError("native_public_summary_sensitive_value")
    return snapshot


__all__ = [
    "DEFAULT_GENERATION_CONFIG",
    "DEFAULT_PROJECTION_CONFIG",
    "EXPECTED_CORPUS_CUTOFF",
    "NativeDiagnosticConfig",
    "NativeOllamaDiagnosticError",
    "build_public_summary",
    "finalize_diagnostic",
    "generation_schema_for_row",
    "prepare_input_bundle",
    "project_native_source_lines",
    "render_prediction_prompt",
    "run_prediction_stage",
    "validate_current_diagnostic_context",
    "validate_frozen_receipts",
    "validate_generation_receipt",
    "validate_input_bundle",
    "validate_prediction_ledger",
    "validate_private_report",
    "validate_public_summary",
]
