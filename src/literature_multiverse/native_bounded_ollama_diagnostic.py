"""Staged, bounded local-model diagnostic for native numerical extraction.

This path is deliberately version-separated from the historical one-stage native
diagnostic.  ``prepare`` freezes a source-only bundle; ``predict`` first inventories a
bounded candidate set and then requests exactly one candidate-bound packet per item;
``finalize`` assembles the complete publication or records one typed whole-publication
failure.  Returned invalid, truncated, or unable responses are terminal and are never
retried or partially salvaged.

The Antiox population and schema/model choices are retrospective development over a
historically opened subset.  The aggregate has no extraction-accuracy, semantic-
entailment, calibration, or claim-release authority.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import ValidationError, field_validator, model_validator

from literature_multiverse.lineage import (
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.local_ollama import (
    LOCAL_OLLAMA_CLIENT_VERSION,
    LocalOllamaError,
    OllamaClientProtocol,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    GENERATION_CONTRACT_VERSION,
    INVENTORY_SENTINEL_CAP,
    PACKET_MODELS,
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidateUnableToComplete,
    assemble_candidate_packets,
    inventory_generation_schema,
    packet_generation_schema,
    validate_inventory_for_row,
    validate_packet_for_candidate,
)
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_ollama_diagnostic import (
    prepare_input_bundle as prepare_legacy_source_bundle,
)
from literature_multiverse.native_ollama_diagnostic import (
    validate_current_diagnostic_context as validate_current_legacy_source_context,
)
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
from literature_multiverse.prompting import render_prompt_text
from literature_multiverse.verifier import compute_verifier_pipeline_fingerprint

BOUNDED_DIAGNOSTIC_VERSION = "native-antiox-bounded-ollama-diagnostic-v1"
BOUNDED_INPUT_BUNDLE_VERSION = "native-bounded-input-bundle-v1"
INVENTORY_RECEIPT_VERSION = "native-bounded-inventory-receipt-v1"
PACKET_RECEIPT_VERSION = "native-bounded-packet-receipt-v1"
ATTEMPT_INTENT_VERSION = "native-bounded-pre-call-intent-v1"
PREDICTION_LEDGER_VERSION = "native-bounded-prediction-ledger-v1"
PRIVATE_REPORT_VERSION = "native-bounded-private-report-v1"
PUBLIC_SUMMARY_VERSION = "native-bounded-public-summary-v1"
EXECUTION_IDENTITY_VERSION = "native-bounded-execution-identity-v1"

EXPECTED_MODEL = "qwen2.5:3b-instruct"
EXPECTED_MODEL_DIGEST = (
    "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b"
)
EXPECTED_OLLAMA_VERSION = "0.15.1"
EXPECTED_PARAMETER_SIZE = "3.1B"
EXPECTED_QUANTIZATION = "Q4_K_M"
EXPECTED_MODEL_FORMAT = "gguf"
EXPECTED_MODEL_FAMILY = "qwen2"
EXPECTED_IDENTITY_VERSION = "ollama-local-runtime-identity-v1"
EXPECTED_ROWS = 19

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/native-antiox-bounded-v1.json")
DEFAULT_INVENTORY_CONFIG = OllamaGenerationConfig(
    model=EXPECTED_MODEL,
    model_digest=EXPECTED_MODEL_DIGEST,
    expected_ollama_version=EXPECTED_OLLAMA_VERSION,
    seed=20260827,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    num_ctx=8192,
    num_predict=768,
    keep_alive="30m",
)
DEFAULT_PACKET_CONFIG = DEFAULT_INVENTORY_CONFIG.model_copy(
    update={"num_predict": 2048}
)

_MODULE_ENTRYPOINT = "src/literature_multiverse/native_bounded_ollama_diagnostic.py"
_SCRIPT_ENTRYPOINT = "scripts/run_native_bounded_ollama_diagnostic.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "candidate",
        "candidates",
        "doc_id",
        "evidence",
        "input_rows",
        "inventory",
        "official_output",
        "packet",
        "parsed_output",
        "passages",
        "prompt",
        "publication_id",
        "quote",
        "records",
        "response_text",
        "row_key",
        "source_bundle",
        "source_locator",
        "source_projection",
        "source_record",
    }
)
_PUBLIC_CAVEATS = [
    "The 19-publication population and model/schema choices are retrospective development "
    "over historically opened eligibility decisions; this is not a pristine or "
    "confirmatory evaluation.",
    "There are no independent numerical extraction gold labels, so no extraction "
    "accuracy is reported.",
    "Numeric-support receipts prove exact lexical occurrence and deterministic numeric "
    "equality, not semantic association, treatment/comparator attribution, or entailment.",
    "Candidate descriptor de-duplication conservatively rejects indistinguishable "
    "same-outcome/effect/line anchors and may reduce recall.",
    "Thousands-separated numerals, inequality-bound values, and model-authored computed "
    "statistics are unsupported in bounded v1 and force an abstaining failure.",
    "Parsed source text is not multimodal evidence; tables, figures, and supplements may "
    "remain unavailable.",
    "Every unable, invalid, truncated, missing, conflicting, or unsupported packet blocks "
    "the whole publication; no successful subset is salvaged.",
    "The diagnostic has no calibration or claim-release authority and requires human "
    "verification before scientific use.",
    "Public-only validation proves current code/config lineage and strict aggregate "
    "shape; empirical counts require the private receipt/ledger replay gate.",
]
_INVENTORY_STATUSES = frozenset(
    {
        "generation_truncated",
        "response_json_invalid",
        "inventory_contract_invalid",
        "inventory_below_cap",
        "inventory_no_candidate_non_authorizing",
        "inventory_capacity_or_uncertainty_non_authorizing",
    }
)
_PACKET_STATUSES = frozenset(
    {
        "generation_truncated",
        "response_json_invalid",
        "packet_contract_invalid",
        "packet_unable_to_complete",
        "packet_source_grounding_invalid",
        "packet_completed",
    }
)
_BASE_PUBLICATION_STATUSES = frozenset(
    {
        *_INVENTORY_STATUSES,
        "official_native_v1_estimable",
        "whole_publication_assembly_conflict_or_invalid",
        "whole_publication_non_authorizing",
    }
)
_SOURCE_ADAPTER_FIELDS = {
    "source_adapter_version",
    "scientific_role",
    "upstream_legacy_source_bundle_sha256",
    "upstream_legacy_model_or_generation_authority",
    "question_config_file_sha256",
    "question_spec_sha256",
    "source_bridge_run_sha256",
    "source_manifest_file_sha256",
    "source_manifest_content_sha256",
    "corpus_cutoff",
    "projection_config",
    "projection_config_sha256",
    "rows",
    "row_count",
    "contains_legacy_findings",
    "contains_legacy_directions",
    "contains_anchor_expectations",
    "contains_downstream_claim_payload",
    "source_adapter_sha256",
}
_SOURCE_ROW_FIELDS = {
    "row_key",
    "source_record",
    "source_payload_sha256",
    "source_projection",
    "source_projection_sha256",
    "projected_characters",
    "projected_passages",
    "input_row_sha256",
}
_SOURCE_PASSAGE_FIELDS = {
    "line_id",
    "line_number",
    "passage_rank",
    "ranking",
    "section",
    "source_line_end_exclusive",
    "source_line_start",
    "text",
}
_SOURCE_RANKING_FIELDS = {
    "score",
    "endpoint_term_hits",
    "exposure_term_hits",
    "comparator_term_hits",
    "numerical_signal",
}
_BOUNDED_INPUT_BUNDLE_FIELDS = {
    "input_bundle_version",
    "diagnostic_version",
    "status",
    "scientific_role",
    "selection_labels_previously_opened",
    "pristine_final_holdout_eligible",
    "retrospective_model_schema_selection",
    "prediction_stage_can_open_source_or_label_files",
    "legacy_single_stage_generation_contract_authority",
    "legacy_single_stage_receipts_accepted",
    "config",
    "config_file_sha256",
    "diagnostic_execution_identity",
    "diagnostic_execution_sha256",
    "source_adapter",
    "source_adapter_sha256",
    "source_rows",
    "allowed_outcomes",
    "outcome_positive_directions",
    "allowed_moderators",
    "allowed_sections",
    "inventory_rendered_base_prompt",
    "inventory_rendered_base_prompt_sha256",
    "inventory_prompt_version",
    "packet_rendered_base_prompt",
    "packet_rendered_base_prompt_sha256",
    "packet_prompt_version",
    "official_schema_sha256",
    "downstream_verifier_pipeline",
    "downstream_verifier_pipeline_sha256",
    "input_bundle_sha256",
}
_INVENTORY_RECEIPT_FIELDS = {
    "inventory_receipt_version",
    "diagnostic_version",
    "status",
    "terminal",
    "input_bundle_sha256",
    "row_key",
    "input_row_sha256",
    "request_identity",
    "attempt_intent_sha256",
    "prompt",
    "schema",
    "generation_config",
    "model_identity",
    "response_text",
    "response_text_sha256",
    "telemetry",
    "parsed_output",
    "parsed_output_sha256",
    "validated_inventory",
    "validated_inventory_sha256",
    "terminal_error",
    "generation_truncated",
    "generation_call_attempts",
    "generation_retries",
    "external_provider_calls",
    "external_provider_cost_usd",
    "receipt_sha256",
}
_PACKET_RECEIPT_FIELDS = {
    "packet_receipt_version",
    "diagnostic_version",
    "status",
    "terminal",
    "input_bundle_sha256",
    "row_key",
    "input_row_sha256",
    "inventory_receipt_sha256",
    "candidate",
    "candidate_descriptor_sha256",
    "candidate_index",
    "request_identity",
    "attempt_intent_sha256",
    "prompt",
    "schema",
    "generation_config",
    "model_identity",
    "response_text",
    "response_text_sha256",
    "telemetry",
    "parsed_output",
    "parsed_output_sha256",
    "validated_packet_outcome",
    "validated_packet_outcome_sha256",
    "source_quote_grounding",
    "terminal_error",
    "generation_truncated",
    "generation_call_attempts",
    "generation_retries",
    "external_provider_calls",
    "external_provider_cost_usd",
    "receipt_sha256",
}


class NativeBoundedOllamaDiagnosticError(ValueError):
    """A stage artifact, runtime identity, or all-or-nothing join is unsafe."""


class _ModelJSONError(ValueError):
    pass


class BoundedNativeDiagnosticConfig(ContractModel):
    config_version: Literal["native-antiox-bounded-config-v1"]
    config_sha256: str
    base_config_path: Literal["configs/benchmarks/native-antiox-ollama-v1.json"]
    base_config_file_sha256: str
    inventory_prompt_path: Literal["prompts/native_candidate_inventory.md"]
    inventory_prompt_file_sha256: str
    packet_prompt_path: Literal["prompts/native_candidate_packet.md"]
    packet_prompt_file_sha256: str
    inventory_generation_config: OllamaGenerationConfig
    packet_generation_config: OllamaGenerationConfig
    model_parameter_size: Literal["3.1B"]
    model_quantization_level: Literal["Q4_K_M"]
    diagnostic_scope: Literal["historically_opened_development_only"]
    selection_population_records: Literal[19]
    retrospective_model_schema_selection: Literal[True]

    @field_validator(
        "config_sha256",
        "base_config_file_sha256",
        "inventory_prompt_file_sha256",
        "packet_prompt_file_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("native_bounded_config_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> BoundedNativeDiagnosticConfig:
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if hash_canonical(payload) != self.config_sha256:
            raise ValueError("native_bounded_config_self_hash_mismatch")
        if self.inventory_generation_config != DEFAULT_INVENTORY_CONFIG:
            raise ValueError("native_bounded_inventory_generation_config_not_frozen")
        if self.packet_generation_config != DEFAULT_PACKET_CONFIG:
            raise ValueError("native_bounded_packet_generation_config_not_frozen")
        return self


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_model_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, _ModelJSONError) as exc:
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_json_unreadable:{path.as_posix()}"
        ) from exc
    if not isinstance(payload, dict):
        raise NativeBoundedOllamaDiagnosticError("native_bounded_json_not_object")
    return payload


def load_bounded_json_artifact(path: Path) -> dict[str, Any]:
    """Load an artifact with duplicate-key and non-finite JSON rejection."""

    return _read_json_object(path)


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop(field, None)
    return payload


def _validate_self_hash(
    value: Mapping[str, Any],
    field: str,
    *,
    code: str,
) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or hash_canonical(_without_hash(value, field)) != observed:
        raise NativeBoundedOllamaDiagnosticError(code)


def _strict_model_json_loads(value: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise _ModelJSONError(f"native_bounded_nonfinite_json_constant:{constant}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise _ModelJSONError("native_bounded_duplicate_json_key")
            output[key] = item
        return output

    try:
        return json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise _ModelJSONError("native_bounded_json_decode_error") from exc


def _safe_repo_file(root: Path, relative: str, expected_sha256: str | None = None) -> Path:
    if "\\" in relative:
        raise NativeBoundedOllamaDiagnosticError("native_bounded_path_not_posix")
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts:
        raise NativeBoundedOllamaDiagnosticError("native_bounded_path_invalid")
    current = root.resolve(strict=True)
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise NativeBoundedOllamaDiagnosticError(
                f"native_bounded_source_symlink_forbidden:{relative}"
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_source_missing:{relative}"
        ) from exc
    if not resolved.is_file() or not resolved.is_relative_to(root.resolve(strict=True)):
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_source_path_invalid:{relative}"
        )
    if expected_sha256 is not None and sha256_file(resolved) != expected_sha256:
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_source_hash_mismatch:{relative}"
        )
    return resolved


def load_bounded_diagnostic_config(
    *, config_path: Path, repository_root: Path
) -> BoundedNativeDiagnosticConfig:
    root = repository_root.resolve(strict=True)
    try:
        relative = config_path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_config_not_repository_relative"
        ) from exc
    path = _safe_repo_file(root, relative)
    try:
        config = BoundedNativeDiagnosticConfig.model_validate(_read_json_object(path))
    except ValidationError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_config_invalid"
        ) from exc
    _safe_repo_file(root, config.base_config_path, config.base_config_file_sha256)
    _safe_repo_file(
        root, config.inventory_prompt_path, config.inventory_prompt_file_sha256
    )
    _safe_repo_file(root, config.packet_prompt_path, config.packet_prompt_file_sha256)
    return config


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package = list(current.parts[:-1])
        if level > len(package):
            return None
        parts = package[: len(package) - (level - 1)]
        if module:
            parts.extend(module.split("."))
        candidates = [Path(*parts).with_suffix(".py"), Path(*parts) / "__init__.py"]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        base = Path("src", *module.split("."))
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
    elif module.startswith("scripts."):
        base = Path(*module.split("."))
        candidates = [base.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def bounded_python_dependency_closure(repository_root: Path) -> list[str]:
    pending = [_MODULE_ENTRYPOINT, _SCRIPT_ENTRYPOINT]
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        path = repository_root / relative
        if not path.is_file():
            raise NativeBoundedOllamaDiagnosticError(
                f"native_bounded_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise NativeBoundedOllamaDiagnosticError(
                f"native_bounded_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def compute_bounded_execution_identity(
    *,
    repository_root: Path,
    config_path: Path,
    config: BoundedNativeDiagnosticConfig,
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    relative_config = config_path.resolve(strict=True).relative_to(root).as_posix()
    paths = sorted(
        {
            *bounded_python_dependency_closure(root),
            relative_config,
            config.base_config_path,
            config.inventory_prompt_path,
            config.packet_prompt_path,
            "pyproject.toml",
            "uv.lock",
        }
    )
    files = []
    for relative in paths:
        path = _safe_repo_file(root, relative)
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "execution_identity_version": EXECUTION_IDENTITY_VERSION,
        "files": files,
        "settings": {
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "inventory_generation_config": config.inventory_generation_config.model_dump(
                mode="json"
            ),
            "inventory_sentinel_cap": INVENTORY_SENTINEL_CAP,
            "official_schema_sha256": hash_canonical(
                native_publication_extraction_json_schema()
            ),
            "packet_generation_config": config.packet_generation_config.model_dump(
                mode="json"
            ),
            "retrospective_model_schema_selection": True,
        },
    }
    return {**payload, "execution_sha256": hash_canonical(payload)}


def _canonical_outcomes(question_spec: Mapping[str, Any]) -> list[str]:
    outcomes = question_spec.get("outcomes")
    endpoint_map = outcomes.get("endpoint_map") if isinstance(outcomes, Mapping) else None
    if not isinstance(endpoint_map, Mapping):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_question_endpoint_map_invalid"
        )
    values = sorted(
        {
            str(value)
            for value in endpoint_map.values()
            if isinstance(value, str) and value
        }
    )
    if not values:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_question_outcomes_empty"
        )
    return values


def _canonical_moderators(question_spec: Mapping[str, Any]) -> list[str]:
    raw = question_spec.get("moderators")
    if not isinstance(raw, list):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_question_moderators_invalid"
        )
    names = [
        str(item["name"])
        for item in raw
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]
    if len(names) != len(raw) or names != list(dict.fromkeys(names)):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_question_moderator_names_invalid"
        )
    return sorted(names)


def _outcome_positive_directions(
    question_spec: Mapping[str, Any], outcomes: Sequence[str]
) -> dict[str, str]:
    relation = question_spec.get("target_relation")
    direction = relation.get("increase_definition") if isinstance(relation, Mapping) else None
    if not isinstance(direction, str) or not direction:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_positive_direction_missing"
        )
    return {outcome: direction for outcome in outcomes}


def prepare_bounded_input_bundle(
    *,
    config_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Freeze current code/config/prompts plus the old path's source-only projection.

    The embedded legacy bundle is source lineage only. Its single-stage prompt, schema,
    generation configuration, and prediction formats have no authority in this path.
    """

    root = repository_root.resolve(strict=True)
    config = load_bounded_diagnostic_config(
        config_path=config_path, repository_root=root
    )
    base_config_path = root / config.base_config_path
    source_bundle = prepare_legacy_source_bundle(
        config_path=base_config_path,
        repository_root=root,
    )
    validate_current_legacy_source_context(source_bundle, repository_root=root)
    # The exact label-free question projection is already sealed in the base config and
    # source bundle. Re-open the config, never labels or historical outputs.
    base_config = _read_json_object(base_config_path)
    frozen_question = base_config.get("question_spec")
    if not isinstance(frozen_question, Mapping):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_question_spec_invalid"
        )
    question_json = json.dumps(frozen_question, sort_keys=True)
    inventory_template = (root / config.inventory_prompt_path).read_text(encoding="utf-8")
    packet_template = (root / config.packet_prompt_path).read_text(encoding="utf-8")
    inventory_prompt, inventory_prompt_version = render_prompt_text(
        inventory_template,
        {"QUESTION_SPEC_JSON": question_json},
    )
    packet_prompt, packet_prompt_version = render_prompt_text(
        packet_template,
        {"QUESTION_SPEC_JSON": question_json},
    )
    if "__FROZEN_CANDIDATE_JSON__" not in packet_prompt:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_candidate_sentinel_missing"
        )
    execution = compute_bounded_execution_identity(
        repository_root=root,
        config_path=config_path,
        config=config,
    )
    pipeline = compute_verifier_pipeline_fingerprint(root=root)
    source_adapter_payload = {
        "source_adapter_version": "legacy-native-source-projection-adapter-v1",
        "scientific_role": "upstream_source_preparation_lineage_only",
        "upstream_legacy_source_bundle_sha256": source_bundle["input_bundle_sha256"],
        "upstream_legacy_model_or_generation_authority": False,
        "question_config_file_sha256": source_bundle["question_config_file_sha256"],
        "question_spec_sha256": source_bundle["question_spec_sha256"],
        "source_bridge_run_sha256": source_bundle["source_bridge_run_sha256"],
        "source_manifest_file_sha256": source_bundle["source_manifest_file_sha256"],
        "source_manifest_content_sha256": source_bundle[
            "source_manifest_content_sha256"
        ],
        "corpus_cutoff": source_bundle["corpus_cutoff"],
        "projection_config": source_bundle["projection_config"],
        "projection_config_sha256": source_bundle["projection_config_sha256"],
        "rows": source_bundle["rows"],
        "row_count": source_bundle["row_count"],
        "contains_legacy_findings": False,
        "contains_legacy_directions": False,
        "contains_anchor_expectations": False,
        "contains_downstream_claim_payload": False,
    }
    source_adapter = {
        **source_adapter_payload,
        "source_adapter_sha256": hash_canonical(source_adapter_payload),
    }
    allowed_outcomes = _canonical_outcomes(frozen_question)
    payload = {
        "input_bundle_version": BOUNDED_INPUT_BUNDLE_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "frozen_source_only_nonpristine_development_diagnostic",
        "scientific_role": "retrospective_development_no_accuracy_or_release_authority",
        "selection_labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "retrospective_model_schema_selection": True,
        "prediction_stage_can_open_source_or_label_files": False,
        "legacy_single_stage_generation_contract_authority": False,
        "legacy_single_stage_receipts_accepted": False,
        "config": config.model_dump(mode="json"),
        "config_file_sha256": sha256_file(config_path),
        "diagnostic_execution_identity": execution,
        "diagnostic_execution_sha256": execution["execution_sha256"],
        "source_adapter": source_adapter,
        "source_adapter_sha256": source_adapter["source_adapter_sha256"],
        "source_rows": source_adapter["row_count"],
        "allowed_outcomes": allowed_outcomes,
        "outcome_positive_directions": _outcome_positive_directions(
            frozen_question, allowed_outcomes
        ),
        "allowed_moderators": _canonical_moderators(frozen_question),
        "allowed_sections": sorted(
            set(source_bundle["projection_config"]["allowed_sections"])
        ),
        "inventory_rendered_base_prompt": inventory_prompt,
        "inventory_rendered_base_prompt_sha256": hashlib.sha256(
            inventory_prompt.encode()
        ).hexdigest(),
        "inventory_prompt_version": inventory_prompt_version,
        "packet_rendered_base_prompt": packet_prompt,
        "packet_rendered_base_prompt_sha256": hashlib.sha256(
            packet_prompt.encode()
        ).hexdigest(),
        "packet_prompt_version": packet_prompt_version,
        "official_schema_sha256": hash_canonical(
            native_publication_extraction_json_schema()
        ),
        "downstream_verifier_pipeline": pipeline.model_dump(mode="json"),
        "downstream_verifier_pipeline_sha256": pipeline.pipeline_sha256,
    }
    return validate_bounded_input_bundle(
        {**payload, "input_bundle_sha256": hash_canonical(payload)}
    )


def _validate_execution_identity_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_execution_identity_invalid"
        )
    snapshot = deepcopy(dict(value))
    _validate_self_hash(
        snapshot,
        "execution_sha256",
        code="native_bounded_execution_identity_hash_mismatch",
    )
    files = snapshot.get("files")
    if (
        snapshot.get("execution_identity_version") != EXECUTION_IDENTITY_VERSION
        or not isinstance(files, list)
        or not isinstance(snapshot.get("settings"), Mapping)
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_execution_identity_invalid"
        )
    paths: list[str] = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "bytes"}:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_execution_file_invalid"
            )
        path = item.get("path")
        sha256 = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(path, str)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or not isinstance(sha256, str)
            or not _SHA256.fullmatch(sha256)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_execution_file_invalid"
            )
        paths.append(path)
    if paths != sorted(set(paths)):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_execution_files_not_sorted_unique"
        )
    return snapshot


def _validate_source_adapter(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_ADAPTER_FIELDS:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_source_adapter_invalid"
        )
    adapter = deepcopy(dict(value))
    _validate_self_hash(
        adapter,
        "source_adapter_sha256",
        code="native_bounded_source_adapter_hash_mismatch",
    )
    rows = adapter.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_source_adapter_rows_invalid"
        )
    row_keys: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SOURCE_ROW_FIELDS:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_row_invalid"
            )
        row_snapshot = deepcopy(dict(row))
        observed_row_sha256 = row_snapshot.pop("input_row_sha256", None)
        if hash_canonical(row_snapshot) != observed_row_sha256:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_row_hash_mismatch"
            )
        try:
            source_record = NativeSourceRecord.model_validate(row["source_record"])
        except ValidationError as exc:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_record_invalid"
            ) from exc
        row_key = row.get("row_key")
        if (
            not isinstance(row_key, str)
            or not _SHA256.fullmatch(row_key)
            or row_key
            != hashlib.sha256(
                source_record.publication.publication_id.encode()
            ).hexdigest()
            or not isinstance(row.get("source_payload_sha256"), str)
            or not _SHA256.fullmatch(str(row.get("source_payload_sha256")))
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_row_identity_invalid"
            )
        projection = row.get("source_projection")
        if not isinstance(projection, list):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_projection_invalid"
            )
        passage_ranks: list[int] = []
        projected_characters = 0
        for passage in projection:
            if not isinstance(passage, Mapping) or set(passage) != _SOURCE_PASSAGE_FIELDS:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_source_adapter_passage_invalid"
                )
            ranking = passage.get("ranking")
            if not isinstance(ranking, Mapping) or set(ranking) != _SOURCE_RANKING_FIELDS:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_source_adapter_ranking_invalid"
                )
            line_id = passage.get("line_id")
            section = passage.get("section")
            text = passage.get("text")
            rank = passage.get("passage_rank")
            start = passage.get("source_line_start")
            end = passage.get("source_line_end_exclusive")
            if (
                not isinstance(line_id, str)
                or not line_id
                or not isinstance(section, str)
                or not section
                or not isinstance(text, str)
                or not text
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or len(text) != end - start
            ):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_source_adapter_passage_fields_invalid"
                )
            passage_ranks.append(rank)
            projected_characters += len(text)
        if (
            passage_ranks != list(range(1, len(projection) + 1))
            or hash_canonical(projection) != row.get("source_projection_sha256")
            or row.get("projected_passages") != len(projection)
            or row.get("projected_characters") != projected_characters
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_adapter_projection_replay_mismatch"
            )
        row_keys.append(row_key)
    if row_keys != sorted(set(row_keys)):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_source_adapter_rows_not_sorted_unique"
        )
    return adapter


def validate_bounded_input_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(bundle))
    if set(snapshot) != _BOUNDED_INPUT_BUNDLE_FIELDS:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_input_bundle_fields_invalid"
        )
    _validate_self_hash(
        snapshot,
        "input_bundle_sha256",
        code="native_bounded_input_bundle_hash_mismatch",
    )
    try:
        config = BoundedNativeDiagnosticConfig.model_validate(snapshot.get("config"))
    except (ValidationError, ValueError) as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_input_bundle_nested_contract_invalid"
        ) from exc
    source_adapter = _validate_source_adapter(snapshot.get("source_adapter"))
    execution = _validate_execution_identity_snapshot(
        snapshot.get("diagnostic_execution_identity")
    )
    rows = source_adapter.get("rows")
    if not isinstance(rows, list):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_source_adapter_rows_invalid"
        )
    row_keys = [row.get("row_key") for row in rows if isinstance(row, Mapping)]
    inventory_prompt = snapshot.get("inventory_rendered_base_prompt")
    packet_prompt = snapshot.get("packet_rendered_base_prompt")
    try:
        # Pydantic reparsing is performed by the official fingerprint type's own
        # validator when the current-context function compares it below.
        pipeline = PipelineFingerprint.model_validate(
            snapshot["downstream_verifier_pipeline"]
        )
    except (KeyError, ValidationError, ValueError) as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_downstream_fingerprint_invalid"
        ) from exc
    if (
        snapshot.get("input_bundle_version") != BOUNDED_INPUT_BUNDLE_VERSION
        or snapshot.get("diagnostic_version") != BOUNDED_DIAGNOSTIC_VERSION
        or snapshot.get("status")
        != "frozen_source_only_nonpristine_development_diagnostic"
        or snapshot.get("scientific_role")
        != "retrospective_development_no_accuracy_or_release_authority"
        or snapshot.get("selection_labels_previously_opened") is not True
        or snapshot.get("pristine_final_holdout_eligible") is not False
        or snapshot.get("retrospective_model_schema_selection") is not True
        or snapshot.get("prediction_stage_can_open_source_or_label_files") is not False
        or snapshot.get("legacy_single_stage_generation_contract_authority") is not False
        or snapshot.get("legacy_single_stage_receipts_accepted") is not False
        or snapshot.get("config_file_sha256") is None
        or snapshot.get("source_adapter_sha256")
        != source_adapter.get("source_adapter_sha256")
        or snapshot.get("source_rows") != EXPECTED_ROWS
        or source_adapter.get("row_count") != EXPECTED_ROWS
        or len(rows) != EXPECTED_ROWS
        or row_keys != sorted(set(row_keys))
        or source_adapter.get("source_adapter_version")
        != "legacy-native-source-projection-adapter-v1"
        or source_adapter.get("scientific_role")
        != "upstream_source_preparation_lineage_only"
        or source_adapter.get("upstream_legacy_model_or_generation_authority")
        is not False
        or source_adapter.get("contains_legacy_findings") is not False
        or source_adapter.get("contains_legacy_directions") is not False
        or source_adapter.get("contains_anchor_expectations") is not False
        or source_adapter.get("contains_downstream_claim_payload") is not False
        or snapshot.get("diagnostic_execution_sha256")
        != execution.get("execution_sha256")
        or snapshot.get("official_schema_sha256")
        != hash_canonical(native_publication_extraction_json_schema())
        or snapshot.get("downstream_verifier_pipeline_sha256")
        != pipeline.pipeline_sha256
        or not isinstance(inventory_prompt, str)
        or hashlib.sha256(inventory_prompt.encode()).hexdigest()
        != snapshot.get("inventory_rendered_base_prompt_sha256")
        or not isinstance(packet_prompt, str)
        or hashlib.sha256(packet_prompt.encode()).hexdigest()
        != snapshot.get("packet_rendered_base_prompt_sha256")
        or "__FROZEN_CANDIDATE_JSON__" not in packet_prompt
        or snapshot.get("allowed_outcomes")
        != sorted(set(snapshot.get("allowed_outcomes", [])))
        or not snapshot.get("allowed_outcomes")
        or snapshot.get("outcome_positive_directions")
        != dict(sorted(snapshot.get("outcome_positive_directions", {}).items()))
        or set(snapshot.get("outcome_positive_directions", {}))
        != set(snapshot.get("allowed_outcomes", []))
        or snapshot.get("allowed_moderators")
        != sorted(set(snapshot.get("allowed_moderators", [])))
        or snapshot.get("allowed_sections")
        != sorted(set(snapshot.get("allowed_sections", [])))
        or config.selection_population_records != EXPECTED_ROWS
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_input_bundle_scope_mismatch"
        )
    return snapshot


def validate_current_bounded_context(
    bundle: Mapping[str, Any],
    *,
    repository_root: Path,
    reverify_source_adapter: bool = False,
) -> dict[str, Any]:
    snapshot = validate_bounded_input_bundle(bundle)
    root = repository_root.resolve(strict=True)
    config_payload = snapshot["config"]
    config = BoundedNativeDiagnosticConfig.model_validate(config_payload)
    config_path = root / "configs/benchmarks/native-antiox-bounded-v1.json"
    current_config = load_bounded_diagnostic_config(
        config_path=config_path, repository_root=root
    )
    current_execution = compute_bounded_execution_identity(
        repository_root=root,
        config_path=config_path,
        config=current_config,
    )
    current_pipeline = compute_verifier_pipeline_fingerprint(root=root)
    base_config = _read_json_object(root / current_config.base_config_path)
    frozen_question = base_config.get("question_spec")
    if not isinstance(frozen_question, Mapping):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_current_question_spec_invalid"
        )
    question_json = json.dumps(frozen_question, sort_keys=True)
    inventory_template = _safe_repo_file(
        root,
        current_config.inventory_prompt_path,
        current_config.inventory_prompt_file_sha256,
    ).read_text(encoding="utf-8")
    packet_template = _safe_repo_file(
        root,
        current_config.packet_prompt_path,
        current_config.packet_prompt_file_sha256,
    ).read_text(encoding="utf-8")
    current_inventory_prompt, current_inventory_prompt_version = render_prompt_text(
        inventory_template,
        {"QUESTION_SPEC_JSON": question_json},
    )
    current_packet_prompt, current_packet_prompt_version = render_prompt_text(
        packet_template,
        {"QUESTION_SPEC_JSON": question_json},
    )
    current_outcomes = _canonical_outcomes(frozen_question)
    current_projection = base_config.get("projection")
    if not isinstance(current_projection, Mapping) or not isinstance(
        current_projection.get("allowed_sections"), list
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_current_projection_config_invalid"
        )
    current_projection = deepcopy(dict(current_projection))
    current_sections = sorted(set(current_projection["allowed_sections"]))
    if (
        config != current_config
        or sha256_file(config_path) != snapshot["config_file_sha256"]
        or current_execution != snapshot["diagnostic_execution_identity"]
        or current_pipeline.model_dump(mode="json")
        != snapshot["downstream_verifier_pipeline"]
        or current_pipeline.pipeline_sha256
        != snapshot["downstream_verifier_pipeline_sha256"]
        or snapshot["allowed_outcomes"] != current_outcomes
        or snapshot["outcome_positive_directions"]
        != _outcome_positive_directions(frozen_question, current_outcomes)
        or snapshot["allowed_moderators"] != _canonical_moderators(frozen_question)
        or snapshot["allowed_sections"] != current_sections
        or snapshot["source_adapter"]["projection_config"] != current_projection
        or snapshot["source_adapter"]["projection_config_sha256"]
        != hash_canonical(current_projection)
        or snapshot["source_adapter"]["question_spec_sha256"]
        != hash_canonical(frozen_question)
        or snapshot["source_adapter"]["corpus_cutoff"]
        != base_config.get("corpus_cutoff")
        or snapshot["inventory_rendered_base_prompt"] != current_inventory_prompt
        or snapshot["inventory_prompt_version"] != current_inventory_prompt_version
        or snapshot["inventory_rendered_base_prompt_sha256"]
        != hashlib.sha256(current_inventory_prompt.encode()).hexdigest()
        or snapshot["packet_rendered_base_prompt"] != current_packet_prompt
        or snapshot["packet_prompt_version"] != current_packet_prompt_version
        or snapshot["packet_rendered_base_prompt_sha256"]
        != hashlib.sha256(current_packet_prompt.encode()).hexdigest()
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_current_context_mismatch"
        )
    if reverify_source_adapter:
        current_source_bundle = prepare_legacy_source_bundle(
            config_path=root / current_config.base_config_path,
            repository_root=root,
        )
        validate_current_legacy_source_context(
            current_source_bundle, repository_root=root
        )
        current_source_adapter_payload = deepcopy(dict(snapshot["source_adapter"]))
        current_source_adapter_payload.pop("source_adapter_sha256", None)
        current_source_adapter_payload.update(
            {
                "upstream_legacy_source_bundle_sha256": current_source_bundle[
                    "input_bundle_sha256"
                ],
                "question_config_file_sha256": current_source_bundle[
                    "question_config_file_sha256"
                ],
                "question_spec_sha256": current_source_bundle["question_spec_sha256"],
                "source_bridge_run_sha256": current_source_bundle[
                    "source_bridge_run_sha256"
                ],
                "source_manifest_file_sha256": current_source_bundle[
                    "source_manifest_file_sha256"
                ],
                "source_manifest_content_sha256": current_source_bundle[
                    "source_manifest_content_sha256"
                ],
                "corpus_cutoff": current_source_bundle["corpus_cutoff"],
                "projection_config": current_source_bundle["projection_config"],
                "projection_config_sha256": current_source_bundle[
                    "projection_config_sha256"
                ],
                "rows": current_source_bundle["rows"],
                "row_count": current_source_bundle["row_count"],
            }
        )
        if (
            hash_canonical(current_source_adapter_payload)
            != snapshot["source_adapter_sha256"]
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_current_source_adapter_mismatch"
            )
    return snapshot


def _row_context(
    bundle: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    passages = row.get("source_projection")
    source_record = row.get("source_record")
    if not isinstance(passages, list) or not isinstance(source_record, Mapping):
        raise NativeBoundedOllamaDiagnosticError("native_bounded_input_row_invalid")
    document = source_record.get("source_document")
    if not isinstance(document, Mapping) or not isinstance(
        document.get("source_locator"), str
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_input_row_locator_invalid"
        )
    line_ids = sorted(
        {
            str(passage["line_id"])
            for passage in passages
            if isinstance(passage, Mapping) and isinstance(passage.get("line_id"), str)
        }
    )
    sections = sorted(
        {
            str(passage["section"])
            for passage in passages
            if isinstance(passage, Mapping) and isinstance(passage.get("section"), str)
        }
    )
    return {
        "exposed_line_ids": line_ids,
        "source_locator": document["source_locator"],
        "allowed_outcomes": bundle["allowed_outcomes"],
        "outcome_positive_directions": bundle["outcome_positive_directions"],
        "allowed_moderators": bundle["allowed_moderators"],
        "allowed_sections": sections or bundle["allowed_sections"],
    }


def _render_source_projection(row: Mapping[str, Any]) -> str:
    passages = row.get("source_projection")
    if not isinstance(passages, list):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_source_projection_invalid"
        )
    rendered: list[str] = []
    for passage in passages:
        if not isinstance(passage, Mapping):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_passage_invalid"
            )
        line_id = passage.get("line_id")
        section = passage.get("section")
        text = passage.get("text")
        if not all(isinstance(item, str) for item in (line_id, section, text)):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_source_passage_fields_invalid"
            )
        rendered.append(
            f"LINE_ID: {line_id}\nSECTION: {section}\nBEGIN_EXACT_SOURCE_TEXT\n"
            f"{text}\nEND_EXACT_SOURCE_TEXT"
        )
    return "\n\n".join(rendered) if rendered else "[NO_EXPOSED_SOURCE_PROJECTION]"


def render_inventory_prompt(
    bundle: Mapping[str, Any], row: Mapping[str, Any]
) -> str:
    context = _row_context(bundle, row)
    return (
        str(bundle["inventory_rendered_base_prompt"])
        + "\n\n## Frozen row constraints\n"
        + json.dumps(
            {
                "allowed_outcomes": context["allowed_outcomes"],
                "exposed_line_ids": context["exposed_line_ids"],
            },
            sort_keys=True,
        )
        + "\n\n## Frozen authoritative source projection\n"
        + _render_source_projection(row)
    )


def render_packet_prompt(
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
) -> str:
    base = str(bundle["packet_rendered_base_prompt"])
    rendered = base.replace(
        "__FROZEN_CANDIDATE_JSON__",
        json.dumps(candidate.model_dump(mode="json"), sort_keys=True),
    )
    if "__FROZEN_CANDIDATE_JSON__" in rendered:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_candidate_render_failed"
        )
    context = _row_context(bundle, row)
    return (
        rendered
        + "\n\n## Frozen row constraints\n"
        + json.dumps(
            {
                "allowed_moderators": context["allowed_moderators"],
                "allowed_sections": context["allowed_sections"],
                "source_locator": context["source_locator"],
            },
            sort_keys=True,
        )
        + "\n\n## Frozen authoritative source projection\n"
        + _render_source_projection(row)
    )


def _model_identity_payload(identity: OllamaIdentity) -> dict[str, Any]:
    payload = identity.model_dump(mode="json")
    return {**payload, "identity_sha256": identity.identity_sha256}


def _require_exact_identity(
    identity: OllamaIdentity,
    *,
    config: OllamaGenerationConfig,
) -> None:
    if (
        identity.model != config.model
        or identity.model_digest != config.model_digest
        or identity.ollama_version != config.expected_ollama_version
        or identity.parameter_size != EXPECTED_PARAMETER_SIZE
        or identity.quantization_level != EXPECTED_QUANTIZATION
        or identity.model_format != EXPECTED_MODEL_FORMAT
        or identity.model_family != EXPECTED_MODEL_FAMILY
        or identity.identity_version != EXPECTED_IDENTITY_VERSION
        or identity.client_version != LOCAL_OLLAMA_CLIENT_VERSION
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_runtime_model_identity_mismatch"
        )


def _request_identity(
    *,
    stage: Literal["inventory", "packet"],
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    candidate: NativeCandidateDescriptor | None = None,
    inventory_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "request_identity_version": "native-bounded-request-v1",
        "stage": stage,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "row_key": row["row_key"],
        "input_row_sha256": row["input_row_sha256"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hash_canonical(schema),
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "candidate_descriptor_sha256": (
            candidate.descriptor_sha256 if candidate is not None else None
        ),
        "candidate_index": candidate.candidate_index if candidate is not None else None,
        "inventory_receipt_sha256": inventory_receipt_sha256,
    }
    return {**payload, "request_sha256": hash_canonical(payload)}


def freeze_bounded_pre_call_intent(
    *,
    stage: Literal["inventory", "packet"],
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    candidate: NativeCandidateDescriptor | None = None,
    inventory_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze the sole allowed call attempt before the model-facing POST starts."""

    request = _request_identity(
        stage=stage,
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
        candidate=candidate,
        inventory_receipt_sha256=inventory_receipt_sha256,
    )
    payload = {
        "attempt_intent_version": ATTEMPT_INTENT_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "durable_pre_call_intent_frozen",
        "ambiguous_execution_is_terminal_and_nonresumable": True,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "row_key": row["row_key"],
        "input_row_sha256": row["input_row_sha256"],
        "stage": stage,
        "candidate_index": candidate.candidate_index if candidate is not None else None,
        "candidate_descriptor_sha256": (
            candidate.descriptor_sha256 if candidate is not None else None
        ),
        "inventory_receipt_sha256": inventory_receipt_sha256,
        "request_identity": request,
        "request_sha256": request["request_sha256"],
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "permitted_call_attempts": 1,
        "generation_retries_permitted": 0,
    }
    return {**payload, "attempt_intent_sha256": hash_canonical(payload)}


def validate_bounded_pre_call_intent(
    intent: Mapping[str, Any],
    *,
    stage: Literal["inventory", "packet"],
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    candidate: NativeCandidateDescriptor | None = None,
    inventory_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(intent))
    _validate_self_hash(
        snapshot,
        "attempt_intent_sha256",
        code="native_bounded_attempt_intent_hash_mismatch",
    )
    expected = freeze_bounded_pre_call_intent(
        stage=stage,
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
        candidate=candidate,
        inventory_receipt_sha256=inventory_receipt_sha256,
    )
    if snapshot != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_attempt_intent_replay_mismatch"
        )
    return snapshot


def _generate_with_exact_identity(
    *,
    client: OllamaClientProtocol,
    initial_identity: OllamaIdentity,
    config: OllamaGenerationConfig,
    prompt: str,
    schema: Mapping[str, Any],
) -> OllamaGenerationResult:
    try:
        before = client.inspect_identity(config)
    except LocalOllamaError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_runtime_unavailable_before_request_no_receipt"
        ) from exc
    _require_exact_identity(before, config=config)
    if before != initial_identity:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_identity_changed_before_request_no_receipt"
        )
    try:
        result = client.generate(prompt=prompt, output_schema=schema, config=config)
    except LocalOllamaError as exc:
        # A failed POST can represent an unobserved execution. No durable terminal
        # scientific receipt is written, and orchestration must decide whether to resume.
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_generation_transport_failure_no_receipt"
        ) from exc
    try:
        after = client.inspect_identity(config)
    except LocalOllamaError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_runtime_unavailable_after_response_no_receipt"
        ) from exc
    _require_exact_identity(after, config=config)
    if after != initial_identity or result.model != config.model:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_identity_changed_after_response_no_receipt"
        )
    return result


def _inventory_schema(
    bundle: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    context = _row_context(bundle, row)
    return inventory_generation_schema(
        exposed_line_ids=context["exposed_line_ids"],
        allowed_outcomes=context["allowed_outcomes"],
    )


def _packet_schema(
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
) -> dict[str, Any]:
    context = _row_context(bundle, row)
    return packet_generation_schema(candidate=candidate, **context)


def _classify_inventory_response(
    *,
    result: OllamaGenerationResult,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[str, Any, dict[str, Any] | None, str | None]:
    if result.done_reason == "length":
        return "generation_truncated", None, None, "generation_truncated"
    try:
        parsed = _strict_model_json_loads(result.response_text)
    except _ModelJSONError:
        return "response_json_invalid", None, None, "response_json_invalid"
    try:
        validated = validate_inventory_for_row(
            parsed,
            exposed_line_ids=_row_context(bundle, row)["exposed_line_ids"],
            allowed_outcomes=bundle["allowed_outcomes"],
        )
    except (NativeBoundedGenerationError, ValidationError, ValueError):
        return "inventory_contract_invalid", parsed, None, "inventory_contract_invalid"
    if validated.authorizes_packet_generation():
        status = "inventory_below_cap"
    elif validated.inventory_status == "no_candidate_found":
        status = "inventory_no_candidate_non_authorizing"
    else:
        status = "inventory_capacity_or_uncertainty_non_authorizing"
    return status, parsed, validated.model_dump(mode="json"), None


def _classify_packet_response(
    *,
    result: OllamaGenerationResult,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
) -> tuple[str, Any, dict[str, Any] | None, str | None, dict[str, Any] | None]:
    if result.done_reason == "length":
        return "generation_truncated", None, None, "generation_truncated", None
    try:
        parsed = _strict_model_json_loads(result.response_text)
    except _ModelJSONError:
        return "response_json_invalid", None, None, "response_json_invalid", None
    try:
        validated = validate_packet_for_candidate(
            parsed,
            candidate=candidate,
            **_row_context(bundle, row),
        )
    except (NativeBoundedGenerationError, ValidationError, ValueError):
        return "packet_contract_invalid", parsed, None, "packet_contract_invalid", None
    if isinstance(validated, NativeCandidateUnableToComplete):
        return (
            "packet_unable_to_complete",
            parsed,
            validated.model_dump(mode="json"),
            f"packet_unable_to_complete:{validated.reason}",
            None,
        )
    try:
        grounding = _ground_packet_quote_to_projection(validated, row=row)
    except NativeBoundedOllamaDiagnosticError:
        return (
            "packet_source_grounding_invalid",
            parsed,
            None,
            "packet_source_grounding_invalid",
            None,
        )
    return (
        "packet_completed",
        parsed,
        validated.model_dump(mode="json"),
        None,
        grounding,
    )


def _ground_packet_quote_to_projection(
    packet: Any, *, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the model quote to exactly one cited frozen source passage."""

    projection = row.get("source_projection")
    if not isinstance(projection, list):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_projection_invalid"
        )
    matches: list[dict[str, Any]] = []
    for passage in projection:
        if (
            not isinstance(passage, Mapping)
            or passage.get("line_id") not in set(packet.evidence.line_ids)
            or passage.get("section") != packet.evidence.section
            or not isinstance(passage.get("text"), str)
        ):
            continue
        text = str(passage["text"])
        cursor = 0
        while True:
            start = text.find(packet.evidence.quote, cursor)
            if start < 0:
                break
            end = start + len(packet.evidence.quote)
            source_start = int(passage["source_line_start"]) + start
            source_end = int(passage["source_line_start"]) + end
            matches.append(
                {
                    "passage_rank": passage["passage_rank"],
                    "line_id": passage["line_id"],
                    "section": passage["section"],
                    "passage_char_start": start,
                    "passage_char_end_exclusive": end,
                    "source_line_char_start": source_start,
                    "source_line_char_end_exclusive": source_end,
                    "passage_utf8_start": len(
                        text[:start].encode("utf-8")
                    ),
                    "passage_utf8_end_exclusive": len(
                        text[:end].encode("utf-8")
                    ),
                    "quote_sha256": hashlib.sha256(
                        packet.evidence.quote.encode()
                    ).hexdigest(),
                    "passage_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            cursor = start + 1
    if len(matches) != 1:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_quote_not_unique_in_cited_projection"
        )
    payload = {
        "grounding_version": "native-bounded-projection-quote-grounding-v1",
        "status": "exact_unique_projected_source_match",
        **matches[0],
    }
    return {**payload, "grounding_sha256": hash_canonical(payload)}


def _freeze_inventory_receipt(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    client: OllamaClientProtocol,
    attempt_intent: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = render_inventory_prompt(bundle, row)
    schema = _inventory_schema(bundle, row)
    intent = validate_bounded_pre_call_intent(
        attempt_intent,
        stage="inventory",
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
    )
    request = intent["request_identity"]
    result = _generate_with_exact_identity(
        client=client,
        initial_identity=identity,
        config=config,
        prompt=prompt,
        schema=schema,
    )
    status, parsed, validated, error = _classify_inventory_response(
        result=result, bundle=bundle, row=row
    )
    payload = {
        "inventory_receipt_version": INVENTORY_RECEIPT_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": status,
        "terminal": True,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "row_key": row["row_key"],
        "input_row_sha256": row["input_row_sha256"],
        "request_identity": request,
        "attempt_intent_sha256": intent["attempt_intent_sha256"],
        "prompt": prompt,
        "schema": schema,
        "generation_config": config.model_dump(mode="json"),
        "model_identity": _model_identity_payload(identity),
        "response_text": result.response_text,
        "response_text_sha256": hashlib.sha256(result.response_text.encode()).hexdigest(),
        "telemetry": result.model_dump(mode="json"),
        "parsed_output": parsed,
        "parsed_output_sha256": hash_canonical(parsed) if parsed is not None else None,
        "validated_inventory": validated,
        "validated_inventory_sha256": (
            hash_canonical(validated) if validated is not None else None
        ),
        "terminal_error": error,
        "generation_truncated": result.done_reason == "length",
        "generation_call_attempts": 1,
        "generation_retries": 0,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def _freeze_packet_receipt(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
    inventory_receipt: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    client: OllamaClientProtocol,
    attempt_intent: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = render_packet_prompt(bundle, row, candidate)
    schema = _packet_schema(bundle, row, candidate)
    intent = validate_bounded_pre_call_intent(
        attempt_intent,
        stage="packet",
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
        candidate=candidate,
        inventory_receipt_sha256=str(inventory_receipt["receipt_sha256"]),
    )
    request = intent["request_identity"]
    result = _generate_with_exact_identity(
        client=client,
        initial_identity=identity,
        config=config,
        prompt=prompt,
        schema=schema,
    )
    status, parsed, validated, error, grounding = _classify_packet_response(
        result=result,
        bundle=bundle,
        row=row,
        candidate=candidate,
    )
    payload = {
        "packet_receipt_version": PACKET_RECEIPT_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": status,
        "terminal": True,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "row_key": row["row_key"],
        "input_row_sha256": row["input_row_sha256"],
        "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
        "candidate": candidate.model_dump(mode="json"),
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "candidate_index": candidate.candidate_index,
        "request_identity": request,
        "attempt_intent_sha256": intent["attempt_intent_sha256"],
        "prompt": prompt,
        "schema": schema,
        "generation_config": config.model_dump(mode="json"),
        "model_identity": _model_identity_payload(identity),
        "response_text": result.response_text,
        "response_text_sha256": hashlib.sha256(result.response_text.encode()).hexdigest(),
        "telemetry": result.model_dump(mode="json"),
        "parsed_output": parsed,
        "parsed_output_sha256": hash_canonical(parsed) if parsed is not None else None,
        "validated_packet_outcome": validated,
        "validated_packet_outcome_sha256": (
            hash_canonical(validated) if validated is not None else None
        ),
        "source_quote_grounding": grounding,
        "terminal_error": error,
        "generation_truncated": result.done_reason == "length",
        "generation_call_attempts": 1,
        "generation_retries": 0,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def _identity_from_payload(value: Any) -> OllamaIdentity:
    if not isinstance(value, Mapping):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_identity_invalid"
        )
    payload = deepcopy(dict(value))
    observed = payload.pop("identity_sha256", None)
    try:
        identity = OllamaIdentity.model_validate(payload)
    except ValidationError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_identity_invalid"
        ) from exc
    if identity.identity_sha256 != observed:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_identity_hash_mismatch"
        )
    return identity


def _validate_result_and_hashes(
    snapshot: Mapping[str, Any], *, expected_model: str
) -> OllamaGenerationResult:
    response = snapshot.get("response_text")
    if not isinstance(response, str) or hashlib.sha256(response.encode()).hexdigest() != (
        snapshot.get("response_text_sha256")
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_response_hash_mismatch"
        )
    try:
        result = OllamaGenerationResult.model_validate(snapshot.get("telemetry"))
    except ValidationError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_telemetry_invalid"
        ) from exc
    if result.response_text != response or result.model != expected_model:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_telemetry_response_mismatch"
        )
    parsed = snapshot.get("parsed_output")
    if (parsed is None) != (snapshot.get("parsed_output_sha256") is None):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_parsed_presence_mismatch"
        )
    if parsed is not None and hash_canonical(parsed) != snapshot.get(
        "parsed_output_sha256"
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_parsed_hash_mismatch"
        )
    return result


def validate_inventory_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    expected_identity: OllamaIdentity,
    attempt_intent: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    if set(snapshot) != _INVENTORY_RECEIPT_FIELDS:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_receipt_fields_invalid"
        )
    _validate_self_hash(
        snapshot,
        "receipt_sha256",
        code="native_bounded_inventory_receipt_hash_mismatch",
    )
    identity = _identity_from_payload(snapshot.get("model_identity"))
    _require_exact_identity(identity, config=config)
    if identity != expected_identity:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_receipt_identity_mismatch"
        )
    prompt = render_inventory_prompt(bundle, row)
    schema = _inventory_schema(bundle, row)
    intent = validate_bounded_pre_call_intent(
        attempt_intent,
        stage="inventory",
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
    )
    request = intent["request_identity"]
    result = _validate_result_and_hashes(snapshot, expected_model=config.model)
    status, parsed, validated, error = _classify_inventory_response(
        result=result, bundle=bundle, row=row
    )
    if (
        snapshot.get("inventory_receipt_version") != INVENTORY_RECEIPT_VERSION
        or snapshot.get("diagnostic_version") != BOUNDED_DIAGNOSTIC_VERSION
        or snapshot.get("terminal") is not True
        or snapshot.get("input_bundle_sha256") != bundle["input_bundle_sha256"]
        or snapshot.get("row_key") != row["row_key"]
        or snapshot.get("input_row_sha256") != row["input_row_sha256"]
        or snapshot.get("request_identity") != request
        or snapshot.get("attempt_intent_sha256")
        != intent["attempt_intent_sha256"]
        or snapshot.get("prompt") != prompt
        or snapshot.get("schema") != schema
        or snapshot.get("generation_config") != config.model_dump(mode="json")
        or snapshot.get("status") != status
        or snapshot.get("parsed_output") != parsed
        or snapshot.get("validated_inventory") != validated
        or snapshot.get("validated_inventory_sha256")
        != (hash_canonical(validated) if validated is not None else None)
        or snapshot.get("terminal_error") != error
        or snapshot.get("generation_truncated") is not (result.done_reason == "length")
        or snapshot.get("generation_call_attempts") != 1
        or snapshot.get("generation_retries") != 0
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_receipt_replay_mismatch"
        )
    return snapshot


def _candidate_from_inventory_receipt(
    receipt: Mapping[str, Any], candidate_index: int
) -> NativeCandidateDescriptor:
    inventory_payload = receipt.get("validated_inventory")
    try:
        inventory = NativeCandidateInventory.model_validate(inventory_payload)
    except ValidationError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_candidate_source_invalid"
        ) from exc
    if not inventory.authorizes_packet_generation():
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_does_not_authorize_packets"
        )
    matches = [
        candidate
        for candidate in inventory.candidates
        if candidate.candidate_index == candidate_index
    ]
    if len(matches) != 1:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_candidate_not_in_inventory"
        )
    return matches[0]


def validate_packet_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    inventory_receipt: Mapping[str, Any],
    config: OllamaGenerationConfig,
    expected_identity: OllamaIdentity,
    inventory_attempt_intent: Mapping[str, Any],
    packet_attempt_intent: Mapping[str, Any],
) -> dict[str, Any]:
    inventory_snapshot = validate_inventory_receipt(
        inventory_receipt,
        bundle=bundle,
        row=row,
        config=DEFAULT_INVENTORY_CONFIG,
        expected_identity=expected_identity,
        attempt_intent=inventory_attempt_intent,
    )
    snapshot = deepcopy(dict(receipt))
    if set(snapshot) != _PACKET_RECEIPT_FIELDS:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_receipt_fields_invalid"
        )
    _validate_self_hash(
        snapshot,
        "receipt_sha256",
        code="native_bounded_packet_receipt_hash_mismatch",
    )
    identity = _identity_from_payload(snapshot.get("model_identity"))
    _require_exact_identity(identity, config=config)
    candidate_index = snapshot.get("candidate_index")
    if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_candidate_index_invalid"
        )
    candidate = _candidate_from_inventory_receipt(
        inventory_snapshot, candidate_index
    )
    if identity != expected_identity:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_receipt_identity_mismatch"
        )
    prompt = render_packet_prompt(bundle, row, candidate)
    schema = _packet_schema(bundle, row, candidate)
    packet_intent = validate_bounded_pre_call_intent(
        packet_attempt_intent,
        stage="packet",
        bundle=bundle,
        row=row,
        prompt=prompt,
        schema=schema,
        config=config,
        identity=identity,
        candidate=candidate,
        inventory_receipt_sha256=inventory_snapshot["receipt_sha256"],
    )
    request = packet_intent["request_identity"]
    result = _validate_result_and_hashes(snapshot, expected_model=config.model)
    status, parsed, validated, error, grounding = _classify_packet_response(
        result=result,
        bundle=bundle,
        row=row,
        candidate=candidate,
    )
    if (
        snapshot.get("packet_receipt_version") != PACKET_RECEIPT_VERSION
        or snapshot.get("diagnostic_version") != BOUNDED_DIAGNOSTIC_VERSION
        or snapshot.get("terminal") is not True
        or snapshot.get("input_bundle_sha256") != bundle["input_bundle_sha256"]
        or snapshot.get("row_key") != row["row_key"]
        or snapshot.get("input_row_sha256") != row["input_row_sha256"]
        or snapshot.get("inventory_receipt_sha256")
        != inventory_snapshot["receipt_sha256"]
        or snapshot.get("candidate") != candidate.model_dump(mode="json")
        or snapshot.get("candidate_descriptor_sha256")
        != candidate.descriptor_sha256
        or snapshot.get("request_identity") != request
        or snapshot.get("attempt_intent_sha256")
        != packet_intent["attempt_intent_sha256"]
        or snapshot.get("prompt") != prompt
        or snapshot.get("schema") != schema
        or snapshot.get("generation_config") != config.model_dump(mode="json")
        or snapshot.get("status") != status
        or snapshot.get("parsed_output") != parsed
        or snapshot.get("validated_packet_outcome") != validated
        or snapshot.get("validated_packet_outcome_sha256")
        != (hash_canonical(validated) if validated is not None else None)
        or snapshot.get("source_quote_grounding") != grounding
        or snapshot.get("terminal_error") != error
        or snapshot.get("generation_truncated") is not (result.done_reason == "length")
        or snapshot.get("generation_call_attempts") != 1
        or snapshot.get("generation_retries") != 0
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_receipt_replay_mismatch"
        )
    return snapshot


def _bounded_schema_preflight_specs() -> list[dict[str, Any]]:
    """Return the exact value-free six-call schema compatibility roster."""

    inventory_schema = inventory_generation_schema(
        exposed_line_ids=["SYNTHETIC_LINE"],
        allowed_outcomes=["synthetic_outcome"],
    )
    inventory_config = DEFAULT_INVENTORY_CONFIG.model_copy(
        update={"num_predict": 128}
    )
    specs: list[dict[str, Any]] = [
        {
            "call_id": "00-inventory",
            "kind": "inventory",
            "effect_kind": None,
            "config": inventory_config,
            "prompt": (
            "Synthetic schema compilation only; no paper, claim, label, or source. "
            "Return exactly {\"inventory_version\":\"native-candidate-inventory-v1\","
            "\"inventory_status\":\"no_candidate_found\",\"candidates\":[],"
            "\"has_more_or_uncertain\":false}."
            ),
            "schema": inventory_schema,
        }
    ]
    packet_config = DEFAULT_PACKET_CONFIG.model_copy(update={"num_predict": 128})
    for index, effect_kind in enumerate(sorted(PACKET_MODELS), start=1):
        candidate = NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="synthetic_outcome",
            effect_kind=effect_kind,
            line_ids=["SYNTHETIC_LINE"],
        )
        schema = packet_generation_schema(
            candidate=candidate,
            exposed_line_ids=["SYNTHETIC_LINE"],
            source_locator="synthetic:schema-compilation-only",
            allowed_outcomes=["synthetic_outcome"],
            allowed_moderators=[],
            allowed_sections=["Synthetic"],
            outcome_positive_directions={
                "synthetic_outcome": "larger synthetic target value"
            },
        )
        specs.append(
            {
                "call_id": f"{index:02d}-packet-{effect_kind}",
                "kind": "packet",
                "effect_kind": effect_kind,
                "candidate": candidate,
                "config": packet_config,
                "prompt": (
                "Synthetic schema compilation only; no paper, claim, label, or source. "
                "Return exactly {\"packet_version\":\"native-candidate-packet-v1\","
                "\"packet_status\":\"unable_to_complete\",\"candidate_index\":1,"
                "\"reason\":\"capacity_or_other_uncertainty\"}."
                ),
                "schema": schema,
            }
        )
    return specs


def _preflight_intent(
    spec: Mapping[str, Any],
    identity: OllamaIdentity,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    config = spec["config"]
    payload = {
        "preflight_call_intent_version": "native-bounded-preflight-call-intent-v1",
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "durable_pre_call_intent_frozen",
        "ambiguous_execution_is_terminal_and_nonresumable": True,
        "call_id": spec["call_id"],
        "kind": spec["kind"],
        "effect_kind": spec.get("effect_kind"),
        "prompt_sha256": hashlib.sha256(str(spec["prompt"]).encode()).hexdigest(),
        "schema_sha256": hash_canonical(spec["schema"]),
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "diagnostic_execution_sha256": bundle["diagnostic_execution_sha256"],
        "downstream_verifier_pipeline_sha256": bundle[
            "downstream_verifier_pipeline_sha256"
        ],
        "contains_publication_content": False,
        "contains_scientific_claim": False,
        "contains_source_text": False,
        "contains_eligibility_or_answer_labels": False,
        "permitted_call_attempts": 1,
        "generation_retries_permitted": 0,
    }
    return {**payload, "intent_sha256": hash_canonical(payload)}


def _validate_preflight_result(spec: Mapping[str, Any], result: OllamaGenerationResult) -> None:
    if result.model != spec["config"].model or result.done_reason == "length":
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_preflight_generation_invalid"
        )
    parsed = _strict_model_json_loads(result.response_text)
    if spec["kind"] == "inventory":
        inventory = validate_inventory_for_row(
            parsed,
            exposed_line_ids=["SYNTHETIC_LINE"],
            allowed_outcomes=["synthetic_outcome"],
        )
        if inventory.inventory_status != "no_candidate_found":
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_inventory_schema_preflight_unexpected_output"
            )
        return
    outcome = validate_packet_for_candidate(
        parsed,
        candidate=spec["candidate"],
        exposed_line_ids=["SYNTHETIC_LINE"],
        source_locator="synthetic:schema-compilation-only",
        allowed_outcomes=["synthetic_outcome"],
        allowed_moderators=[],
        allowed_sections=["Synthetic"],
        outcome_positive_directions={
            "synthetic_outcome": "larger synthetic target value"
        },
    )
    if not isinstance(outcome, NativeCandidateUnableToComplete):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_schema_preflight_unexpected_output"
        )


def _freeze_preflight_call_receipt(
    *,
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    identity: OllamaIdentity,
    result: OllamaGenerationResult,
) -> dict[str, Any]:
    _validate_preflight_result(spec, result)
    payload = {
        "preflight_call_receipt_version": "native-bounded-preflight-call-receipt-v1",
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "passed",
        "terminal": True,
        "call_id": spec["call_id"],
        "kind": spec["kind"],
        "effect_kind": spec.get("effect_kind"),
        "intent_sha256": intent["intent_sha256"],
        "prompt": spec["prompt"],
        "schema": spec["schema"],
        "generation_config": spec["config"].model_dump(mode="json"),
        "model_identity": _model_identity_payload(identity),
        "response_text": result.response_text,
        "response_text_sha256": hashlib.sha256(result.response_text.encode()).hexdigest(),
        "telemetry": result.model_dump(mode="json"),
        "generation_call_attempts": 1,
        "generation_retries": 0,
        "contains_publication_content": False,
        "contains_scientific_claim": False,
        "contains_source_text": False,
        "contains_eligibility_or_answer_labels": False,
    }
    return {**payload, "call_receipt_sha256": hash_canonical(payload)}


def _validate_preflight_call_receipt(
    *,
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    identity: OllamaIdentity,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    _validate_self_hash(
        snapshot,
        "call_receipt_sha256",
        code="native_bounded_preflight_call_receipt_hash_mismatch",
    )
    expected_intent = _preflight_intent(spec, identity, bundle)
    if dict(intent) != expected_intent:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_preflight_intent_replay_mismatch"
        )
    result = _validate_result_and_hashes(
        snapshot, expected_model=spec["config"].model
    )
    expected = _freeze_preflight_call_receipt(
        spec=spec,
        intent=expected_intent,
        identity=identity,
        result=result,
    )
    if snapshot != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_preflight_call_receipt_replay_mismatch"
        )
    return snapshot


def _preflight_paths(preflight_dir: Path, call_id: str) -> tuple[Path, Path]:
    return (
        preflight_dir / "pre-call-intents" / f"{call_id}.json",
        preflight_dir / "call-receipts" / f"{call_id}.json",
    )


def _validate_preflight_directory(preflight_dir: Path) -> None:
    _assert_real_directory_or_missing(
        preflight_dir,
        code="native_bounded_preflight_directory_invalid",
    )
    if not preflight_dir.exists():
        return
    expected_ids = {str(spec["call_id"]) for spec in _bounded_schema_preflight_specs()}
    for entry in preflight_dir.iterdir():
        if entry.name in {"preflight-receipt.json", ".preflight.lock"}:
            if entry.is_symlink() or not entry.is_file():
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_preflight_directory_mixing_or_extra"
                )
            continue
        if entry.name not in {"pre-call-intents", "call-receipts"} or (
            entry.is_symlink() or not entry.is_dir()
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_preflight_directory_mixing_or_extra"
            )
        for child in entry.iterdir():
            if (
                child.is_symlink()
                or not child.is_file()
                or child.suffix != ".json"
                or child.stem not in expected_ids
            ):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_preflight_directory_mixing_or_extra"
                )


def validate_bounded_schema_preflight(
    receipt: Mapping[str, Any],
    *,
    preflight_dir: Path,
    identity: OllamaIdentity,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = validate_bounded_input_bundle(bundle)
    _require_exact_identity(identity, config=DEFAULT_INVENTORY_CONFIG)
    _validate_preflight_directory(preflight_dir)
    call_receipts: list[dict[str, Any]] = []
    packet_schema_hashes: dict[str, str] = {}
    for spec in _bounded_schema_preflight_specs():
        intent_path, call_receipt_path = _preflight_paths(
            preflight_dir, str(spec["call_id"])
        )
        if not intent_path.is_file() or not call_receipt_path.is_file():
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_preflight_incomplete_or_ambiguous"
            )
        validated = _validate_preflight_call_receipt(
            receipt=_read_json_object(call_receipt_path),
            spec=spec,
            intent=_read_json_object(intent_path),
            identity=identity,
            bundle=bundle,
        )
        call_receipts.append(validated)
        if spec["effect_kind"] is not None:
            packet_schema_hashes[str(spec["effect_kind"])] = hash_canonical(
                spec["schema"]
            )
    payload = {
        "preflight_version": "native-bounded-schema-preflight-v1",
        "status": "passed",
        "contains_publication_content": False,
        "contains_scientific_claim": False,
        "contains_source_text": False,
        "contains_eligibility_or_answer_labels": False,
        "paper_prediction_receipt_written": False,
        "inventory_schema_sha256": hash_canonical(
            _bounded_schema_preflight_specs()[0]["schema"]
        ),
        "packet_schema_sha256s": packet_schema_hashes,
        "synthetic_calls": len(call_receipts),
        "call_receipt_sha256s": [
            item["call_receipt_sha256"] for item in call_receipts
        ],
        "model_identity_sha256": identity.identity_sha256,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "diagnostic_execution_sha256": bundle["diagnostic_execution_sha256"],
        "downstream_verifier_pipeline_sha256": bundle[
            "downstream_verifier_pipeline_sha256"
        ],
    }
    expected = {**payload, "preflight_sha256": hash_canonical(payload)}
    if dict(receipt) != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_preflight_receipt_replay_mismatch"
        )
    return expected


def _run_bounded_schema_compatibility_preflight_locked(
    *,
    client: OllamaClientProtocol,
    identity: OllamaIdentity,
    preflight_dir: Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Run/resume the six value-free calls with poison-on-ambiguous execution."""

    bundle = validate_bounded_input_bundle(bundle)
    _require_exact_identity(identity, config=DEFAULT_INVENTORY_CONFIG)
    _validate_preflight_directory(preflight_dir)
    aggregate_path = preflight_dir / "preflight-receipt.json"
    if aggregate_path.exists():
        return validate_bounded_schema_preflight(
            _read_json_object(aggregate_path),
            preflight_dir=preflight_dir,
            identity=identity,
            bundle=bundle,
        )
    for spec in _bounded_schema_preflight_specs():
        intent_path, receipt_path = _preflight_paths(
            preflight_dir, str(spec["call_id"])
        )
        if intent_path.exists() and not receipt_path.exists():
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_ambiguous_preflight_execution_workspace_poisoned"
            )
        if receipt_path.exists() and not intent_path.exists():
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_preflight_receipt_without_intent"
            )
        if receipt_path.exists():
            _validate_preflight_call_receipt(
                receipt=_read_json_object(receipt_path),
                spec=spec,
                intent=_read_json_object(intent_path),
                identity=identity,
                bundle=bundle,
            )
            continue
        intent = _preflight_intent(spec, identity, bundle)
        atomic_write_json(intent_path, intent, force=False)
        result = _generate_with_exact_identity(
            client=client,
            initial_identity=identity,
            config=spec["config"],
            prompt=str(spec["prompt"]),
            schema=spec["schema"],
        )
        call_receipt = _freeze_preflight_call_receipt(
            spec=spec,
            intent=intent,
            identity=identity,
            result=result,
        )
        atomic_write_json(receipt_path, call_receipt, force=False)
    specs = _bounded_schema_preflight_specs()
    packet_schema_hashes = {
        str(spec["effect_kind"]): hash_canonical(spec["schema"])
        for spec in specs
        if spec["effect_kind"] is not None
    }
    call_receipts = [
        _read_json_object(_preflight_paths(preflight_dir, str(spec["call_id"]))[1])
        for spec in specs
    ]
    payload = {
        "preflight_version": "native-bounded-schema-preflight-v1",
        "status": "passed",
        "contains_publication_content": False,
        "contains_scientific_claim": False,
        "contains_source_text": False,
        "contains_eligibility_or_answer_labels": False,
        "paper_prediction_receipt_written": False,
        "inventory_schema_sha256": hash_canonical(specs[0]["schema"]),
        "packet_schema_sha256s": packet_schema_hashes,
        "synthetic_calls": len(call_receipts),
        "call_receipt_sha256s": [
            item["call_receipt_sha256"] for item in call_receipts
        ],
        "model_identity_sha256": identity.identity_sha256,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "diagnostic_execution_sha256": bundle["diagnostic_execution_sha256"],
        "downstream_verifier_pipeline_sha256": bundle[
            "downstream_verifier_pipeline_sha256"
        ],
    }
    receipt = {**payload, "preflight_sha256": hash_canonical(payload)}
    atomic_write_json(aggregate_path, receipt, force=False)
    return validate_bounded_schema_preflight(
        receipt,
        preflight_dir=preflight_dir,
        identity=identity,
        bundle=bundle,
    )


def run_bounded_schema_compatibility_preflight(
    *,
    client: OllamaClientProtocol,
    identity: OllamaIdentity,
    preflight_dir: Path,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclusively run/resume the required value-free schema preflight."""

    lock_path = preflight_dir / ".preflight.lock"
    _assert_no_symlink_ancestors(
        preflight_dir, code="native_bounded_preflight_symlink_ancestor_forbidden"
    )
    preflight_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_stage_lock(
        lock_path,
        busy_code="native_bounded_preflight_workspace_locked",
    ):
        return _run_bounded_schema_compatibility_preflight_locked(
            client=client,
            identity=identity,
            preflight_dir=preflight_dir,
            bundle=bundle,
        )


def _inventory_receipt_path(receipts_dir: Path, row_key: str) -> Path:
    if not _SHA256.fullmatch(row_key):
        raise NativeBoundedOllamaDiagnosticError("native_bounded_row_key_invalid")
    return receipts_dir / f"{row_key}.json"


def _packet_receipt_path(
    receipts_dir: Path, row_key: str, candidate_index: int
) -> Path:
    if not _SHA256.fullmatch(row_key) or not 1 <= candidate_index < INVENTORY_SENTINEL_CAP:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_receipt_identity_invalid"
        )
    return receipts_dir / row_key / f"{candidate_index:02d}.json"


def _attempt_intent_path(
    intents_dir: Path,
    *,
    stage: Literal["inventory", "packet"],
    row_key: str,
    candidate_index: int | None = None,
) -> Path:
    if not _SHA256.fullmatch(row_key):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_attempt_intent_row_key_invalid"
        )
    if stage == "inventory":
        if candidate_index is not None:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_inventory_intent_candidate_forbidden"
            )
        return intents_dir / "inventory" / f"{row_key}.json"
    if (
        candidate_index is None
        or isinstance(candidate_index, bool)
        or not 1 <= candidate_index < INVENTORY_SENTINEL_CAP
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_packet_intent_candidate_invalid"
        )
    return intents_dir / "packet" / row_key / f"{candidate_index:02d}.json"


def _assert_no_symlink_ancestors(path: Path, *, code: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise NativeBoundedOllamaDiagnosticError(code)


def _normalized_non_symlink_path(path: Path, *, code: str) -> Path:
    _assert_no_symlink_ancestors(path, code=code)
    return path.absolute().resolve(strict=False)


@contextmanager
def _exclusive_stage_lock(path: Path, *, busy_code: str) -> Any:
    """Hold an OS-enforced exclusive lock across intent, POST, receipt, and ledger."""

    _assert_no_symlink_ancestors(
        path, code="native_bounded_stage_lock_symlink_ancestor_forbidden"
    )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_stage_lock_open_failed"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NativeBoundedOllamaDiagnosticError(busy_code) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_stage_lock_topology(
    *,
    lock_path: Path,
    protected_directories: Sequence[Path],
    protected_files: Sequence[Path] = (),
) -> None:
    lock = _normalized_non_symlink_path(
        lock_path, code="native_bounded_stage_lock_symlink_ancestor_forbidden"
    )
    for directory in protected_directories:
        protected = _normalized_non_symlink_path(
            directory, code="native_bounded_protected_path_symlink_ancestor_forbidden"
        )
        if lock == protected or lock.is_relative_to(protected):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_stage_lock_inside_protected_directory"
            )
    for file_path in protected_files:
        protected = _normalized_non_symlink_path(
            file_path, code="native_bounded_protected_path_symlink_ancestor_forbidden"
        )
        if lock == protected:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_stage_lock_alias_forbidden"
            )


def validate_bounded_prediction_path_topology(
    *,
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
) -> None:
    """Reject aliases/nesting/symlink traversal before any runtime/model contact."""

    inventory = _normalized_non_symlink_path(
        inventory_receipts_dir,
        code="native_bounded_inventory_path_symlink_ancestor_forbidden",
    )
    packets = _normalized_non_symlink_path(
        packet_receipts_dir,
        code="native_bounded_packet_path_symlink_ancestor_forbidden",
    )
    intents = _normalized_non_symlink_path(
        attempt_intents_dir,
        code="native_bounded_attempt_path_symlink_ancestor_forbidden",
    )
    preflight = _normalized_non_symlink_path(
        preflight_dir,
        code="native_bounded_preflight_path_symlink_ancestor_forbidden",
    )
    ledger = _normalized_non_symlink_path(
        prediction_ledger_path,
        code="native_bounded_ledger_path_symlink_ancestor_forbidden",
    )
    directories = (inventory, packets, intents, preflight)
    if len(set(directories)) != len(directories):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_directory_alias_forbidden"
        )
    for left in directories:
        for right in directories:
            if left != right and (left.is_relative_to(right) or right.is_relative_to(left)):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_prediction_directory_nesting_forbidden"
                )
        if ledger == left or ledger.is_relative_to(left) or left.is_relative_to(ledger):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_ledger_receipt_path_overlap_forbidden"
            )
    if prediction_ledger_path.is_symlink() or (
        prediction_ledger_path.exists() and not prediction_ledger_path.is_file()
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_path_invalid"
        )


def validate_bounded_prediction_workspace_layout(
    *,
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
) -> Path:
    """Enforce one canonical workspace, eliminating caller-selected lock domains."""

    workspace = _normalized_non_symlink_path(
        prediction_ledger_path.parent,
        code="native_bounded_workspace_symlink_ancestor_forbidden",
    )
    expected = {
        _normalized_non_symlink_path(
            inventory_receipts_dir,
            code="native_bounded_inventory_path_symlink_ancestor_forbidden",
        ): workspace / "inventory-receipts",
        _normalized_non_symlink_path(
            packet_receipts_dir,
            code="native_bounded_packet_path_symlink_ancestor_forbidden",
        ): workspace / "packet-receipts",
        _normalized_non_symlink_path(
            attempt_intents_dir,
            code="native_bounded_attempt_path_symlink_ancestor_forbidden",
        ): workspace / "pre-call-intents",
        _normalized_non_symlink_path(
            preflight_dir,
            code="native_bounded_preflight_path_symlink_ancestor_forbidden",
        ): workspace / "schema-preflight",
        _normalized_non_symlink_path(
            prediction_ledger_path,
            code="native_bounded_ledger_path_symlink_ancestor_forbidden",
        ): workspace / "prediction-ledger.json",
    }
    if any(observed != required for observed, required in expected.items()):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_workspace_layout_invalid"
        )
    return workspace


def _validate_bounded_receipt_directory_topology(
    *,
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
) -> None:
    paths = (
        _normalized_non_symlink_path(
            inventory_receipts_dir,
            code="native_bounded_inventory_path_symlink_ancestor_forbidden",
        ),
        _normalized_non_symlink_path(
            packet_receipts_dir,
            code="native_bounded_packet_path_symlink_ancestor_forbidden",
        ),
        _normalized_non_symlink_path(
            attempt_intents_dir,
            code="native_bounded_attempt_path_symlink_ancestor_forbidden",
        ),
        _normalized_non_symlink_path(
            preflight_dir,
            code="native_bounded_preflight_path_symlink_ancestor_forbidden",
        ),
    )
    if len(set(paths)) != len(paths):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_receipt_directory_alias_forbidden"
        )
    for left in paths:
        for right in paths:
            if left != right and (left.is_relative_to(right) or right.is_relative_to(left)):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_receipt_directory_nesting_forbidden"
                )


def _assert_real_directory_or_missing(path: Path, *, code: str) -> None:
    _assert_no_symlink_ancestors(path, code=code)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise NativeBoundedOllamaDiagnosticError(code)


def _validate_attempt_directory_membership(
    *,
    intents_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    inventory_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    _assert_real_directory_or_missing(
        intents_dir, code="native_bounded_attempt_intent_directory_invalid"
    )
    if not intents_dir.exists():
        return
    allowed_roots = {"inventory", "packet"}
    for root_entry in intents_dir.iterdir():
        if (
            root_entry.is_symlink()
            or not root_entry.is_dir()
            or root_entry.name not in allowed_roots
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_attempt_intent_directory_mixing_or_extra"
            )
    row_keys = {str(row["row_key"]) for row in rows}
    inventory_dir = intents_dir / "inventory"
    if inventory_dir.exists():
        for entry in inventory_dir.iterdir():
            if (
                entry.is_symlink()
                or not entry.is_file()
                or entry.suffix != ".json"
                or entry.stem not in row_keys
            ):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_attempt_intent_directory_mixing_or_extra"
                )
    packet_dir = intents_dir / "packet"
    if not packet_dir.exists():
        return
    if inventory_receipts is None:
        for row_entry in packet_dir.iterdir():
            if (
                row_entry.is_symlink()
                or not row_entry.is_dir()
                or row_entry.name not in row_keys
            ):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_attempt_intent_directory_mixing_or_extra"
                )
            for entry in row_entry.iterdir():
                if (
                    entry.is_symlink()
                    or not entry.is_file()
                    or entry.suffix != ".json"
                    or not entry.stem.isdecimal()
                    or not 1 <= int(entry.stem) < INVENTORY_SENTINEL_CAP
                ):
                    raise NativeBoundedOllamaDiagnosticError(
                        "native_bounded_attempt_intent_directory_mixing_or_extra"
                    )
        return
    for row_entry in packet_dir.iterdir():
        if (
            row_entry.is_symlink()
            or not row_entry.is_dir()
            or row_entry.name not in row_keys
            or row_entry.name not in inventory_receipts
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_attempt_intent_directory_mixing_or_extra"
            )
        inventory_payload = inventory_receipts[row_entry.name].get(
            "validated_inventory"
        )
        try:
            inventory = NativeCandidateInventory.model_validate(inventory_payload)
        except ValidationError as exc:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_attempt_intent_inventory_invalid"
            ) from exc
        allowed = (
            {f"{candidate.candidate_index:02d}.json" for candidate in inventory.candidates}
            if inventory.authorizes_packet_generation()
            else set()
        )
        for entry in row_entry.iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.name not in allowed:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_attempt_intent_directory_mixing_or_extra"
                )


def _validate_inventory_directory_membership(
    *, receipts_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    _assert_real_directory_or_missing(
        receipts_dir, code="native_bounded_inventory_receipt_directory_invalid"
    )
    if not receipts_dir.exists():
        return
    expected = {f"{row['row_key']}.json" for row in rows}
    for entry in receipts_dir.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.name not in expected:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_inventory_receipt_directory_mixing_or_extra"
            )


def _validate_packet_directory_membership(
    *,
    receipts_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    inventory_receipts: Mapping[str, Mapping[str, Any]],
) -> None:
    _assert_real_directory_or_missing(
        receipts_dir, code="native_bounded_packet_receipt_directory_invalid"
    )
    if not receipts_dir.exists():
        return
    row_by_key = {str(row["row_key"]): row for row in rows}
    for row_entry in receipts_dir.iterdir():
        if (
            row_entry.is_symlink()
            or not row_entry.is_dir()
            or row_entry.name not in row_by_key
            or row_entry.name not in inventory_receipts
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_packet_receipt_directory_mixing_or_extra"
            )
        inventory_payload = inventory_receipts[row_entry.name].get(
            "validated_inventory"
        )
        try:
            inventory = NativeCandidateInventory.model_validate(inventory_payload)
        except ValidationError as exc:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_packet_directory_inventory_invalid"
            ) from exc
        allowed = (
            {f"{candidate.candidate_index:02d}.json" for candidate in inventory.candidates}
            if inventory.authorizes_packet_generation()
            else set()
        )
        for entry in row_entry.iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.name not in allowed:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_packet_receipt_directory_mixing_or_extra"
                )


def _load_inventory_receipts(
    *,
    bundle: Mapping[str, Any],
    receipts_dir: Path,
    identity: OllamaIdentity,
    attempt_intents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = bundle["source_adapter"]["rows"]
    _validate_inventory_directory_membership(receipts_dir=receipts_dir, rows=rows)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["row_key"])
        path = _inventory_receipt_path(receipts_dir, key)
        if not path.exists():
            continue
        intent = attempt_intents.get(key)
        if intent is None:
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_inventory_receipt_without_pre_call_intent"
            )
        output[key] = validate_inventory_receipt(
            _read_json_object(path),
            bundle=bundle,
            row=row,
            config=DEFAULT_INVENTORY_CONFIG,
            expected_identity=identity,
            attempt_intent=intent,
        )
    if set(attempt_intents) != set(output):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_ambiguous_inventory_execution_workspace_poisoned"
        )
    return output


def _load_packet_receipts(
    *,
    bundle: Mapping[str, Any],
    receipts_dir: Path,
    inventory_receipts: Mapping[str, Mapping[str, Any]],
    identity: OllamaIdentity,
    inventory_attempt_intents: Mapping[str, Mapping[str, Any]],
    packet_attempt_intents: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = bundle["source_adapter"]["rows"]
    _validate_packet_directory_membership(
        receipts_dir=receipts_dir,
        rows=rows,
        inventory_receipts=inventory_receipts,
    )
    output: dict[tuple[str, int], dict[str, Any]] = {}
    row_by_key = {str(row["row_key"]): row for row in rows}
    for key, inventory_receipt in sorted(inventory_receipts.items()):
        inventory_payload = inventory_receipt.get("validated_inventory")
        try:
            inventory = NativeCandidateInventory.model_validate(inventory_payload)
        except ValidationError:
            continue
        if not inventory.authorizes_packet_generation():
            continue
        for candidate in inventory.candidates:
            path = _packet_receipt_path(
                receipts_dir, key, candidate.candidate_index
            )
            if not path.exists():
                continue
            intent = packet_attempt_intents.get((key, candidate.candidate_index))
            if intent is None:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_packet_receipt_without_pre_call_intent"
                )
            output[(key, candidate.candidate_index)] = validate_packet_receipt(
                _read_json_object(path),
                bundle=bundle,
                row=row_by_key[key],
                inventory_receipt=inventory_receipt,
                config=DEFAULT_PACKET_CONFIG,
                expected_identity=identity,
                inventory_attempt_intent=inventory_attempt_intents[key],
                packet_attempt_intent=intent,
            )
    if set(packet_attempt_intents) != set(output):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_ambiguous_packet_execution_workspace_poisoned"
        )
    return output


def _load_inventory_attempt_intents(
    *,
    bundle: Mapping[str, Any],
    intents_dir: Path,
    identity: OllamaIdentity,
) -> dict[str, dict[str, Any]]:
    rows = bundle["source_adapter"]["rows"]
    _validate_attempt_directory_membership(intents_dir=intents_dir, rows=rows)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["row_key"])
        path = _attempt_intent_path(
            intents_dir, stage="inventory", row_key=key
        )
        if not path.exists():
            continue
        output[key] = validate_bounded_pre_call_intent(
            _read_json_object(path),
            stage="inventory",
            bundle=bundle,
            row=row,
            prompt=render_inventory_prompt(bundle, row),
            schema=_inventory_schema(bundle, row),
            config=DEFAULT_INVENTORY_CONFIG,
            identity=identity,
        )
    return output


def _load_packet_attempt_intents(
    *,
    bundle: Mapping[str, Any],
    intents_dir: Path,
    inventory_receipts: Mapping[str, Mapping[str, Any]],
    identity: OllamaIdentity,
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = bundle["source_adapter"]["rows"]
    _validate_attempt_directory_membership(
        intents_dir=intents_dir,
        rows=rows,
        inventory_receipts=inventory_receipts,
    )
    row_by_key = {str(row["row_key"]): row for row in rows}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for key, inventory_receipt in sorted(inventory_receipts.items()):
        inventory_payload = inventory_receipt.get("validated_inventory")
        try:
            inventory = NativeCandidateInventory.model_validate(inventory_payload)
        except ValidationError:
            continue
        if not inventory.authorizes_packet_generation():
            continue
        for candidate in inventory.candidates:
            path = _attempt_intent_path(
                intents_dir,
                stage="packet",
                row_key=key,
                candidate_index=candidate.candidate_index,
            )
            if not path.exists():
                continue
            row = row_by_key[key]
            output[(key, candidate.candidate_index)] = (
                validate_bounded_pre_call_intent(
                    _read_json_object(path),
                    stage="packet",
                    bundle=bundle,
                    row=row,
                    prompt=render_packet_prompt(bundle, row, candidate),
                    schema=_packet_schema(bundle, row, candidate),
                    config=DEFAULT_PACKET_CONFIG,
                    identity=identity,
                    candidate=candidate,
                    inventory_receipt_sha256=str(
                        inventory_receipt["receipt_sha256"]
                    ),
                )
            )
    return output


def freeze_bounded_prediction_ledger(
    *,
    bundle: Mapping[str, Any],
    identity: OllamaIdentity,
    inventory_receipts: Mapping[str, Mapping[str, Any]],
    packet_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    rows = bundle["source_adapter"]["rows"]
    row_keys = {str(row["row_key"]) for row in rows}
    if not set(inventory_receipts).issubset(row_keys):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_extra_inventory_receipt"
        )
    allowed_packet_keys: set[tuple[str, int]] = set()
    for key, inventory_receipt in inventory_receipts.items():
        inventory_payload = inventory_receipt.get("validated_inventory")
        try:
            inventory = NativeCandidateInventory.model_validate(inventory_payload)
        except ValidationError:
            continue
        if inventory.authorizes_packet_generation():
            allowed_packet_keys.update(
                (key, candidate.candidate_index) for candidate in inventory.candidates
            )
    if not set(packet_receipts).issubset(allowed_packet_keys):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_extra_packet_receipt"
        )
    row_manifest: list[dict[str, Any]] = []
    inventory_statuses: list[str] = []
    packet_statuses: list[str] = []
    expected_packet_receipts = 0
    complete = True
    for row in rows:
        key = str(row["row_key"])
        inventory_receipt = inventory_receipts.get(key)
        if inventory_receipt is None:
            complete = False
            row_manifest.append(
                {
                    "row_key": key,
                    "input_row_sha256": row["input_row_sha256"],
                    "status": "inventory_pending",
                    "inventory_receipt_sha256": None,
                    "candidate_count": None,
                    "packet_receipts": [],
                }
            )
            continue
        inventory_status = str(inventory_receipt["status"])
        inventory_statuses.append(inventory_status)
        inventory_payload = inventory_receipt.get("validated_inventory")
        inventory = (
            NativeCandidateInventory.model_validate(inventory_payload)
            if inventory_payload is not None
            else None
        )
        packet_manifest: list[dict[str, Any]] = []
        if inventory is not None and inventory.authorizes_packet_generation():
            expected_packet_receipts += len(inventory.candidates)
            for candidate in inventory.candidates:
                receipt = packet_receipts.get((key, candidate.candidate_index))
                if receipt is None:
                    complete = False
                    packet_manifest.append(
                        {
                            "candidate_index": candidate.candidate_index,
                            "candidate_descriptor_sha256": candidate.descriptor_sha256,
                            "status": "packet_pending",
                            "receipt_sha256": None,
                        }
                    )
                else:
                    packet_statuses.append(str(receipt["status"]))
                    packet_manifest.append(
                        {
                            "candidate_index": candidate.candidate_index,
                            "candidate_descriptor_sha256": candidate.descriptor_sha256,
                            "status": receipt["status"],
                            "receipt_sha256": receipt["receipt_sha256"],
                        }
                    )
            row_status = (
                "publication_packet_set_terminal"
                if all(item["receipt_sha256"] is not None for item in packet_manifest)
                else "publication_packets_pending"
            )
        else:
            row_status = inventory_status
        row_manifest.append(
            {
                "row_key": key,
                "input_row_sha256": row["input_row_sha256"],
                "status": row_status,
                "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
                "candidate_count": (
                    len(inventory.candidates) if inventory is not None else None
                ),
                "packet_receipts": packet_manifest,
            }
        )
    payload = {
        "prediction_ledger_version": PREDICTION_LEDGER_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": (
            "complete_frozen_terminal_ledger" if complete else "partial_resumable_ledger"
        ),
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "input_rows": len(rows),
        "inventory_receipts": len(inventory_receipts),
        "expected_packet_receipts": expected_packet_receipts,
        "packet_receipts": len(packet_receipts),
        "all_expected_terminal_receipts_frozen": complete,
        "inventory_generation_call_attempts": len(inventory_receipts),
        "packet_generation_call_attempts": len(packet_receipts),
        "synthetic_preflight_calls_excluded": True,
        "schema_preflight_sha256": preflight_receipt["preflight_sha256"],
        "synthetic_preflight_calls": preflight_receipt["synthetic_calls"],
        "schema_preflight_required_before_paper_calls": True,
        "pre_call_intents_bound_one_to_one": True,
        "generation_retries": 0,
        "inventory_status_counts": dict(sorted(Counter(inventory_statuses).items())),
        "packet_status_counts": dict(sorted(Counter(packet_statuses).items())),
        "model_identity": _model_identity_payload(identity),
        "inventory_generation_config": DEFAULT_INVENTORY_CONFIG.model_dump(mode="json"),
        "packet_generation_config": DEFAULT_PACKET_CONFIG.model_dump(mode="json"),
        "legacy_single_stage_calls_included": 0,
        "prediction_stage_opened_source_or_label_files": False,
        "rows": row_manifest,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "prediction_ledger_sha256": hash_canonical(payload)}


def validate_bounded_prediction_ledger(
    ledger: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    identity: OllamaIdentity,
    inventory_receipts: Mapping[str, Mapping[str, Any]],
    packet_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
    preflight_receipt: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(ledger))
    _validate_self_hash(
        snapshot,
        "prediction_ledger_sha256",
        code="native_bounded_prediction_ledger_hash_mismatch",
    )
    expected = freeze_bounded_prediction_ledger(
        bundle=bundle,
        identity=identity,
        inventory_receipts=inventory_receipts,
        packet_receipts=packet_receipts,
        preflight_receipt=preflight_receipt,
    )
    if snapshot != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_replay_mismatch"
        )
    if require_complete and snapshot["all_expected_terminal_receipts_frozen"] is not True:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_incomplete"
        )
    return snapshot


def validate_bounded_prediction_ledger_prefix(
    ledger: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    identity: OllamaIdentity,
    inventory_receipts: Mapping[str, Mapping[str, Any]],
    packet_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only an exact previously frozen prefix of current immutable receipts."""

    snapshot = deepcopy(dict(ledger))
    _validate_self_hash(
        snapshot,
        "prediction_ledger_sha256",
        code="native_bounded_prediction_ledger_hash_mismatch",
    )
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_prefix_rows_invalid"
        )
    selected_inventory: dict[str, Mapping[str, Any]] = {}
    selected_packets: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("row_key"), str):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_prediction_ledger_prefix_row_invalid"
            )
        key = str(row["row_key"])
        receipt_sha = row.get("inventory_receipt_sha256")
        if receipt_sha is not None:
            current = inventory_receipts.get(key)
            if current is None or current.get("receipt_sha256") != receipt_sha:
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_prediction_ledger_prefix_inventory_mismatch"
                )
            selected_inventory[key] = current
        packet_manifest = row.get("packet_receipts")
        if not isinstance(packet_manifest, list):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_prediction_ledger_prefix_packets_invalid"
            )
        for item in packet_manifest:
            if not isinstance(item, Mapping):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_prediction_ledger_prefix_packet_invalid"
                )
            packet_sha = item.get("receipt_sha256")
            if packet_sha is None:
                continue
            candidate_index = item.get("candidate_index")
            if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_prediction_ledger_prefix_packet_index_invalid"
                )
            current_packet = packet_receipts.get((key, candidate_index))
            if (
                current_packet is None
                or current_packet.get("receipt_sha256") != packet_sha
            ):
                raise NativeBoundedOllamaDiagnosticError(
                    "native_bounded_prediction_ledger_prefix_packet_mismatch"
                )
            selected_packets[(key, candidate_index)] = current_packet
    expected = freeze_bounded_prediction_ledger(
        bundle=bundle,
        identity=identity,
        inventory_receipts=selected_inventory,
        packet_receipts=selected_packets,
        preflight_receipt=preflight_receipt,
    )
    if snapshot != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_ledger_not_exact_receipt_prefix"
        )
    return snapshot


def _run_bounded_prediction_stage_locked(
    *,
    input_bundle: Mapping[str, Any],
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
    repository_root: Path,
    expected_input_bundle_sha256: str,
    client: OllamaClientProtocol,
    inventory_limit: int | None = None,
    packet_limit: int | None = None,
) -> dict[str, Any]:
    """Run/resume only missing requests; every response-bearing receipt is immutable."""

    validate_bounded_prediction_path_topology(
        inventory_receipts_dir=inventory_receipts_dir,
        packet_receipts_dir=packet_receipts_dir,
        attempt_intents_dir=attempt_intents_dir,
        preflight_dir=preflight_dir,
        prediction_ledger_path=prediction_ledger_path,
    )
    bundle = validate_current_bounded_context(
        input_bundle,
        repository_root=repository_root,
        reverify_source_adapter=False,
    )
    if (
        not _SHA256.fullmatch(expected_input_bundle_sha256)
        or bundle["input_bundle_sha256"] != expected_input_bundle_sha256
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prediction_input_bundle_freeze_anchor_mismatch"
        )
    if inventory_limit is not None and inventory_limit < 1:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_inventory_limit_invalid"
        )
    if packet_limit is not None and packet_limit < 1:
        raise NativeBoundedOllamaDiagnosticError("native_bounded_packet_limit_invalid")
    try:
        identity = client.inspect_identity(DEFAULT_INVENTORY_CONFIG)
    except LocalOllamaError as exc:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_runtime_unavailable_before_stage"
        ) from exc
    _require_exact_identity(identity, config=DEFAULT_INVENTORY_CONFIG)
    preflight_receipt = validate_bounded_schema_preflight(
        _read_json_object(preflight_dir / "preflight-receipt.json"),
        preflight_dir=preflight_dir,
        identity=identity,
        bundle=bundle,
    )
    _assert_real_directory_or_missing(
        inventory_receipts_dir,
        code="native_bounded_inventory_receipt_directory_invalid",
    )
    _assert_real_directory_or_missing(
        packet_receipts_dir,
        code="native_bounded_packet_receipt_directory_invalid",
    )
    _assert_real_directory_or_missing(
        attempt_intents_dir,
        code="native_bounded_attempt_intent_directory_invalid",
    )
    inventory_receipts_dir.mkdir(parents=True, exist_ok=True)
    packet_receipts_dir.mkdir(parents=True, exist_ok=True)
    attempt_intents_dir.mkdir(parents=True, exist_ok=True)
    rows = bundle["source_adapter"]["rows"]
    inventory_attempt_intents = _load_inventory_attempt_intents(
        bundle=bundle,
        intents_dir=attempt_intents_dir,
        identity=identity,
    )
    inventory_receipts = _load_inventory_receipts(
        bundle=bundle,
        receipts_dir=inventory_receipts_dir,
        identity=identity,
        attempt_intents=inventory_attempt_intents,
    )
    packet_attempt_intents = _load_packet_attempt_intents(
        bundle=bundle,
        intents_dir=attempt_intents_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
    )
    packet_receipts = _load_packet_receipts(
        bundle=bundle,
        receipts_dir=packet_receipts_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
        inventory_attempt_intents=inventory_attempt_intents,
        packet_attempt_intents=packet_attempt_intents,
    )
    if prediction_ledger_path.exists():
        validate_bounded_prediction_ledger_prefix(
            _read_json_object(prediction_ledger_path),
            bundle=bundle,
            identity=identity,
            inventory_receipts=inventory_receipts,
            packet_receipts=packet_receipts,
            preflight_receipt=preflight_receipt,
        )
    missing_inventory = [
        row for row in rows if str(row["row_key"]) not in inventory_receipts
    ]
    if inventory_limit is not None:
        missing_inventory = missing_inventory[:inventory_limit]
    for row in missing_inventory:
        prompt = render_inventory_prompt(bundle, row)
        schema = _inventory_schema(bundle, row)
        intent = freeze_bounded_pre_call_intent(
            stage="inventory",
            bundle=bundle,
            row=row,
            prompt=prompt,
            schema=schema,
            config=DEFAULT_INVENTORY_CONFIG,
            identity=identity,
        )
        intent_path = _attempt_intent_path(
            attempt_intents_dir,
            stage="inventory",
            row_key=str(row["row_key"]),
        )
        atomic_write_json(intent_path, intent, force=False)
        inventory_attempt_intents[str(row["row_key"])] = intent
        receipt = _freeze_inventory_receipt(
            bundle=bundle,
            row=row,
            config=DEFAULT_INVENTORY_CONFIG,
            identity=identity,
            client=client,
            attempt_intent=intent,
        )
        path = _inventory_receipt_path(
            inventory_receipts_dir, str(row["row_key"])
        )
        atomic_write_json(path, receipt, force=False)
        inventory_receipts[str(row["row_key"])] = receipt
    # Packet intents are loaded a second time because newly frozen inventories may
    # legitimately create a packet candidate roster during this invocation.
    packet_attempt_intents = _load_packet_attempt_intents(
        bundle=bundle,
        intents_dir=attempt_intents_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
    )
    packet_receipts = _load_packet_receipts(
        bundle=bundle,
        receipts_dir=packet_receipts_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
        inventory_attempt_intents=inventory_attempt_intents,
        packet_attempt_intents=packet_attempt_intents,
    )
    row_by_key = {str(row["row_key"]): row for row in rows}
    missing_packets: list[
        tuple[str, NativeCandidateDescriptor, Mapping[str, Any]]
    ] = []
    for key, inventory_receipt in sorted(inventory_receipts.items()):
        payload = inventory_receipt.get("validated_inventory")
        if payload is None:
            continue
        inventory = NativeCandidateInventory.model_validate(payload)
        if not inventory.authorizes_packet_generation():
            continue
        for candidate in inventory.candidates:
            if (key, candidate.candidate_index) not in packet_receipts:
                missing_packets.append((key, candidate, inventory_receipt))
    if packet_limit is not None:
        missing_packets = missing_packets[:packet_limit]
    for key, candidate, inventory_receipt in missing_packets:
        row = row_by_key[key]
        prompt = render_packet_prompt(bundle, row, candidate)
        schema = _packet_schema(bundle, row, candidate)
        intent = freeze_bounded_pre_call_intent(
            stage="packet",
            bundle=bundle,
            row=row,
            prompt=prompt,
            schema=schema,
            config=DEFAULT_PACKET_CONFIG,
            identity=identity,
            candidate=candidate,
            inventory_receipt_sha256=str(inventory_receipt["receipt_sha256"]),
        )
        intent_path = _attempt_intent_path(
            attempt_intents_dir,
            stage="packet",
            row_key=key,
            candidate_index=candidate.candidate_index,
        )
        atomic_write_json(intent_path, intent, force=False)
        packet_attempt_intents[(key, candidate.candidate_index)] = intent
        receipt = _freeze_packet_receipt(
            bundle=bundle,
            row=row,
            candidate=candidate,
            inventory_receipt=inventory_receipt,
            config=DEFAULT_PACKET_CONFIG,
            identity=identity,
            client=client,
            attempt_intent=intent,
        )
        path = _packet_receipt_path(
            packet_receipts_dir, key, candidate.candidate_index
        )
        atomic_write_json(path, receipt, force=False)
        packet_receipts[(key, candidate.candidate_index)] = receipt
    ledger = freeze_bounded_prediction_ledger(
        bundle=bundle,
        identity=identity,
        inventory_receipts=inventory_receipts,
        packet_receipts=packet_receipts,
        preflight_receipt=preflight_receipt,
    )
    atomic_write_json(prediction_ledger_path, ledger, force=True)
    return validate_bounded_prediction_ledger(
        ledger,
        bundle=bundle,
        identity=identity,
        inventory_receipts=inventory_receipts,
        packet_receipts=packet_receipts,
        preflight_receipt=preflight_receipt,
        require_complete=False,
    )


def run_bounded_prediction_stage(
    *,
    input_bundle: Mapping[str, Any],
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
    repository_root: Path,
    expected_input_bundle_sha256: str,
    client: OllamaClientProtocol,
    inventory_limit: int | None = None,
    packet_limit: int | None = None,
) -> dict[str, Any]:
    """Exclusively run one predictor so every durable intent authorizes one POST."""

    workspace = validate_bounded_prediction_workspace_layout(
        inventory_receipts_dir=inventory_receipts_dir,
        packet_receipts_dir=packet_receipts_dir,
        attempt_intents_dir=attempt_intents_dir,
        preflight_dir=preflight_dir,
        prediction_ledger_path=prediction_ledger_path,
    )
    prediction_lock_path = workspace / ".native-bounded-prediction.lock"
    _validate_stage_lock_topology(
        lock_path=prediction_lock_path,
        protected_directories=(
            inventory_receipts_dir,
            packet_receipts_dir,
            attempt_intents_dir,
            preflight_dir,
        ),
        protected_files=(prediction_ledger_path,),
    )
    prediction_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_stage_lock(
        prediction_lock_path,
        busy_code="native_bounded_prediction_workspace_locked",
    ):
        return _run_bounded_prediction_stage_locked(
            input_bundle=input_bundle,
            inventory_receipts_dir=inventory_receipts_dir,
            packet_receipts_dir=packet_receipts_dir,
            attempt_intents_dir=attempt_intents_dir,
            preflight_dir=preflight_dir,
            prediction_ledger_path=prediction_ledger_path,
            repository_root=repository_root,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
            client=client,
            inventory_limit=inventory_limit,
            packet_limit=packet_limit,
        )


def _finalize_bounded_diagnostic_locked(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    repository_root: Path,
    expected_input_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reverify exact source lineage and assemble every publication all-or-nothing."""

    _validate_bounded_receipt_directory_topology(
        inventory_receipts_dir=inventory_receipts_dir,
        packet_receipts_dir=packet_receipts_dir,
        attempt_intents_dir=attempt_intents_dir,
        preflight_dir=preflight_dir,
    )
    bundle = validate_current_bounded_context(
        input_bundle,
        repository_root=repository_root,
        reverify_source_adapter=True,
    )
    if (
        not _SHA256.fullmatch(expected_input_bundle_sha256)
        or bundle["input_bundle_sha256"] != expected_input_bundle_sha256
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_finalizer_input_bundle_freeze_anchor_mismatch"
        )
    identity = _identity_from_payload(prediction_ledger.get("model_identity"))
    _require_exact_identity(identity, config=DEFAULT_INVENTORY_CONFIG)
    preflight_receipt = validate_bounded_schema_preflight(
        _read_json_object(preflight_dir / "preflight-receipt.json"),
        preflight_dir=preflight_dir,
        identity=identity,
        bundle=bundle,
    )
    _assert_real_directory_or_missing(
        attempt_intents_dir,
        code="native_bounded_attempt_intent_directory_invalid",
    )
    inventory_attempt_intents = _load_inventory_attempt_intents(
        bundle=bundle,
        intents_dir=attempt_intents_dir,
        identity=identity,
    )
    inventory_receipts = _load_inventory_receipts(
        bundle=bundle,
        receipts_dir=inventory_receipts_dir,
        identity=identity,
        attempt_intents=inventory_attempt_intents,
    )
    packet_attempt_intents = _load_packet_attempt_intents(
        bundle=bundle,
        intents_dir=attempt_intents_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
    )
    packet_receipts = _load_packet_receipts(
        bundle=bundle,
        receipts_dir=packet_receipts_dir,
        inventory_receipts=inventory_receipts,
        identity=identity,
        inventory_attempt_intents=inventory_attempt_intents,
        packet_attempt_intents=packet_attempt_intents,
    )
    ledger = validate_bounded_prediction_ledger(
        prediction_ledger,
        bundle=bundle,
        identity=identity,
        inventory_receipts=inventory_receipts,
        packet_receipts=packet_receipts,
        preflight_receipt=preflight_receipt,
        require_complete=True,
    )
    private_rows: list[dict[str, Any]] = []
    publication_statuses: list[str] = []
    candidate_counts: list[int] = []
    official_publications = 0
    official_findings = 0
    for row in bundle["source_adapter"]["rows"]:
        key = str(row["row_key"])
        inventory_receipt = inventory_receipts[key]
        inventory_payload = inventory_receipt.get("validated_inventory")
        inventory = (
            NativeCandidateInventory.model_validate(inventory_payload)
            if inventory_payload is not None
            else None
        )
        packet_manifest: list[dict[str, Any]] = []
        official_output: dict[str, Any] | None = None
        blocking_code: str | None = None
        if inventory is None:
            blocking_code = str(inventory_receipt["status"])
        elif not inventory.authorizes_packet_generation():
            blocking_code = str(inventory_receipt["status"])
            candidate_counts.append(len(inventory.candidates))
        else:
            candidate_counts.append(len(inventory.candidates))
            outcomes = []
            for candidate in inventory.candidates:
                packet_receipt = packet_receipts[(key, candidate.candidate_index)]
                packet_manifest.append(
                    {
                        "candidate_index": candidate.candidate_index,
                        "candidate_descriptor_sha256": candidate.descriptor_sha256,
                        "receipt_sha256": packet_receipt["receipt_sha256"],
                        "status": packet_receipt["status"],
                    }
                )
                if packet_receipt["status"] == "packet_completed":
                    outcome = validate_packet_for_candidate(
                        packet_receipt["validated_packet_outcome"],
                        candidate=candidate,
                        **_row_context(bundle, row),
                    )
                    expected_grounding = _ground_packet_quote_to_projection(
                        outcome,
                        row=row,
                    )
                    if packet_receipt.get("source_quote_grounding") != expected_grounding:
                        raise NativeBoundedOllamaDiagnosticError(
                            "native_bounded_finalizer_source_grounding_mismatch"
                        )
                    outcomes.append(outcome)
            if len(outcomes) != len(inventory.candidates):
                statuses = sorted(
                    {
                        str(item["status"])
                        for item in packet_manifest
                        if item["status"] != "packet_completed"
                    }
                )
                blocking_code = "packet_set_non_authorizing:" + "+".join(statuses)
            else:
                try:
                    official = assemble_candidate_packets(
                        inventory=inventory,
                        packets=outcomes,
                        **_row_context(bundle, row),
                    )
                    official = NativePublicationExtraction.model_validate(
                        official.model_dump(mode="json")
                    )
                except (NativeBoundedGenerationError, ValidationError, ValueError):
                    blocking_code = "whole_publication_assembly_conflict_or_invalid"
                else:
                    official_output = official.model_dump(mode="json")
                    official_publications += 1
                    official_findings += sum(
                        len(cohort.findings)
                        for study in official.studies
                        for cohort in study.cohorts
                    )
        status = (
            "official_native_v1_estimable"
            if official_output is not None
            else blocking_code or "whole_publication_non_authorizing"
        )
        publication_statuses.append(status)
        private_rows.append(
            {
                "row_key": key,
                "input_row_sha256": row["input_row_sha256"],
                "status": status,
                "blocking_code": blocking_code,
                "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
                "packet_receipts": packet_manifest,
                "official_output": official_output,
                "official_output_sha256": (
                    hash_canonical(official_output)
                    if official_output is not None
                    else None
                ),
            }
        )
    private_payload = {
        "private_report_version": PRIVATE_REPORT_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "complete_all_or_nothing_publication_assembly",
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "source_adapter_sha256": bundle["source_adapter_sha256"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "rows": private_rows,
        "row_count": len(private_rows),
        "publication_status_counts": dict(
            sorted(Counter(publication_statuses).items())
        ),
        "candidate_count_distribution": dict(
            sorted(Counter(str(count) for count in candidate_counts).items())
        ),
        "official_native_v1_estimable_publications": official_publications,
        "official_native_v1_findings": official_findings,
        "partial_packet_salvage_count": 0,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    private_report = {
        **private_payload,
        "private_report_sha256": hash_canonical(private_payload),
    }
    config = BoundedNativeDiagnosticConfig.model_validate(bundle["config"])
    public_payload = {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "diagnostic_version": BOUNDED_DIAGNOSTIC_VERSION,
        "status": "complete_retrospective_development_diagnostic",
        "artifact_scope": "aggregate_only_content_silent",
        "scientific_role": "diagnostic_only_no_accuracy_calibration_or_release_authority",
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "source_adapter_sha256": bundle["source_adapter_sha256"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "private_report_sha256": private_report["private_report_sha256"],
        "config_file_sha256": bundle["config_file_sha256"],
        "config_sha256": config.config_sha256,
        "diagnostic_execution_sha256": bundle["diagnostic_execution_sha256"],
        "official_schema_sha256": bundle["official_schema_sha256"],
        "downstream_verifier_pipeline_sha256": bundle[
            "downstream_verifier_pipeline_sha256"
        ],
        "model": {
            "name": EXPECTED_MODEL,
            "digest": EXPECTED_MODEL_DIGEST,
            "runtime_version": EXPECTED_OLLAMA_VERSION,
            "parameter_size": EXPECTED_PARAMETER_SIZE,
            "quantization_level": EXPECTED_QUANTIZATION,
            "identity": _model_identity_payload(identity),
        },
        "generation": {
            "inventory_generation_config_sha256": DEFAULT_INVENTORY_CONFIG.config_sha256,
            "packet_generation_config_sha256": DEFAULT_PACKET_CONFIG.config_sha256,
            "inventory_num_predict_cap": DEFAULT_INVENTORY_CONFIG.num_predict,
            "packet_num_predict_cap": DEFAULT_PACKET_CONFIG.num_predict,
            "cap_hit_is_terminal": True,
            "inventory_generation_calls": ledger[
                "inventory_generation_call_attempts"
            ],
            "packet_generation_calls": ledger["packet_generation_call_attempts"],
            "paper_generation_calls": (
                ledger["inventory_generation_call_attempts"]
                + ledger["packet_generation_call_attempts"]
            ),
            "synthetic_preflight_calls": ledger["synthetic_preflight_calls"],
            "synthetic_preflight_status": "passed_and_bound_before_paper_calls",
            "schema_preflight_sha256": ledger["schema_preflight_sha256"],
            "synthetic_preflight_calls_excluded_from_paper_counts": True,
            "legacy_single_stage_calls_included": 0,
            "generation_retries": 0,
        },
        "population": {
            "publications": EXPECTED_ROWS,
            "selection_labels_previously_opened": True,
            "pristine_final_holdout_eligible": False,
            "retrospective_model_schema_selection": True,
        },
        "inventory_status_counts": ledger["inventory_status_counts"],
        "packet_status_counts": ledger["packet_status_counts"],
        "publication_status_counts": private_report["publication_status_counts"],
        "candidate_count_distribution": private_report[
            "candidate_count_distribution"
        ],
        "official_native_v1_estimable_publications": official_publications,
        "official_native_v1_findings": official_findings,
        "partial_packet_salvage_count": 0,
        "lexical_numeric_support_verified_for_promoted_packets": True,
        "exact_source_projection_grounding_verified_for_promoted_packets": True,
        "semantic_entailment_verified": False,
        "independent_numerical_gold_available": False,
        "extraction_accuracy_reported": False,
        "release_probability_authority": False,
        "claim_release_authority": False,
        "public_validation_scope": "current_code_config_and_aggregate_shape_only",
        "empirical_counts_require_private_receipt_replay": True,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "caveats": list(_PUBLIC_CAVEATS),
    }
    summary = {
        **public_payload,
        "summary_sha256": hash_canonical(public_payload),
    }
    return private_report, validate_bounded_public_summary(
        summary, repository_root=repository_root
    )


def finalize_bounded_diagnostic(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
    repository_root: Path,
    expected_input_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exclusively replay/finalize without racing a predictor."""

    workspace = validate_bounded_prediction_workspace_layout(
        inventory_receipts_dir=inventory_receipts_dir,
        packet_receipts_dir=packet_receipts_dir,
        attempt_intents_dir=attempt_intents_dir,
        preflight_dir=preflight_dir,
        prediction_ledger_path=prediction_ledger_path,
    )
    prediction_lock_path = workspace / ".native-bounded-prediction.lock"
    _validate_stage_lock_topology(
        lock_path=prediction_lock_path,
        protected_directories=(
            inventory_receipts_dir,
            packet_receipts_dir,
            attempt_intents_dir,
            preflight_dir,
        ),
    )
    prediction_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_stage_lock(
        prediction_lock_path,
        busy_code="native_bounded_prediction_workspace_locked",
    ):
        if load_bounded_json_artifact(prediction_ledger_path) != dict(
            prediction_ledger
        ):
            raise NativeBoundedOllamaDiagnosticError(
                "native_bounded_prediction_ledger_file_payload_mismatch"
            )
        return _finalize_bounded_diagnostic_locked(
            input_bundle=input_bundle,
            prediction_ledger=prediction_ledger,
            inventory_receipts_dir=inventory_receipts_dir,
            packet_receipts_dir=packet_receipts_dir,
            attempt_intents_dir=attempt_intents_dir,
            preflight_dir=preflight_dir,
            repository_root=repository_root,
            expected_input_bundle_sha256=expected_input_bundle_sha256,
        )


_PUBLIC_SUMMARY_FIELDS = {
    "public_summary_version",
    "diagnostic_version",
    "status",
    "artifact_scope",
    "scientific_role",
    "input_bundle_sha256",
    "source_adapter_sha256",
    "prediction_ledger_sha256",
    "private_report_sha256",
    "config_file_sha256",
    "config_sha256",
    "diagnostic_execution_sha256",
    "official_schema_sha256",
    "downstream_verifier_pipeline_sha256",
    "model",
    "generation",
    "population",
    "inventory_status_counts",
    "packet_status_counts",
    "publication_status_counts",
    "candidate_count_distribution",
    "official_native_v1_estimable_publications",
    "official_native_v1_findings",
    "partial_packet_salvage_count",
    "lexical_numeric_support_verified_for_promoted_packets",
    "exact_source_projection_grounding_verified_for_promoted_packets",
    "semantic_entailment_verified",
    "independent_numerical_gold_available",
    "extraction_accuracy_reported",
    "release_probability_authority",
    "claim_release_authority",
    "public_validation_scope",
    "empirical_counts_require_private_receipt_replay",
    "external_provider_calls",
    "external_provider_cost_usd",
    "caveats",
    "summary_sha256",
}
_PUBLIC_MODEL_FIELDS = {
    "name",
    "digest",
    "runtime_version",
    "parameter_size",
    "quantization_level",
    "identity",
}
_PUBLIC_GENERATION_FIELDS = {
    "inventory_generation_config_sha256",
    "packet_generation_config_sha256",
    "inventory_num_predict_cap",
    "packet_num_predict_cap",
    "cap_hit_is_terminal",
    "inventory_generation_calls",
    "packet_generation_calls",
    "paper_generation_calls",
    "synthetic_preflight_calls",
    "synthetic_preflight_status",
    "schema_preflight_sha256",
    "synthetic_preflight_calls_excluded_from_paper_counts",
    "legacy_single_stage_calls_included",
    "generation_retries",
}
_PUBLIC_POPULATION_FIELDS = {
    "publications",
    "selection_labels_previously_opened",
    "pristine_final_holdout_eligible",
    "retrospective_model_schema_selection",
}


def _forbidden_public_key_path(value: Any, prefix: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                return path
            nested = _forbidden_public_key_path(item, path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _forbidden_public_key_path(item, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    return None


def _absolute_public_value_path(value: Any, prefix: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            nested = _absolute_public_value_path(item, f"{prefix}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _absolute_public_value_path(item, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    elif isinstance(value, str) and value.startswith("/"):
        return prefix
    return None


def _require_exact_public_fields(
    value: Any, expected: set[str], *, code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NativeBoundedOllamaDiagnosticError(code)
    return value


def _exact_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_count_map(
    value: Any,
    *,
    allowed_keys: frozenset[str] | None,
    key_predicate: Any = None,
    code: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise NativeBoundedOllamaDiagnosticError(code)
    output: dict[str, int] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not _exact_nonnegative_int(count)
            or (allowed_keys is not None and key not in allowed_keys)
            or (key_predicate is not None and not key_predicate(key))
        ):
            raise NativeBoundedOllamaDiagnosticError(code)
        output[key] = count
    if list(output) != sorted(output):
        raise NativeBoundedOllamaDiagnosticError(code)
    return output


def _valid_publication_status(value: str) -> bool:
    if value in _BASE_PUBLICATION_STATUSES:
        return True
    prefix = "packet_set_non_authorizing:"
    if not value.startswith(prefix):
        return False
    parts = value[len(prefix) :].split("+")
    return (
        bool(parts)
        and parts == sorted(set(parts))
        and set(parts).issubset(_PACKET_STATUSES - {"packet_completed"})
    )


def _valid_candidate_count_key(value: str) -> bool:
    return value.isdecimal() and value == str(int(value)) and 0 <= int(value) < (
        INVENTORY_SENTINEL_CAP
    )


def validate_bounded_public_summary(
    summary: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Public-only integrity/current-lineage gate; aggregate outcomes remain diagnostic."""

    snapshot = deepcopy(dict(summary))
    _require_exact_public_fields(
        snapshot,
        _PUBLIC_SUMMARY_FIELDS,
        code="native_bounded_public_summary_fields_invalid",
    )
    _validate_self_hash(
        snapshot,
        "summary_sha256",
        code="native_bounded_public_summary_hash_mismatch",
    )
    forbidden = _forbidden_public_key_path(snapshot)
    if forbidden is not None:
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_public_summary_not_aggregate_only:{forbidden}"
        )
    absolute = _absolute_public_value_path(snapshot)
    if absolute is not None:
        raise NativeBoundedOllamaDiagnosticError(
            f"native_bounded_public_summary_absolute_path_forbidden:{absolute}"
        )
    root = repository_root.resolve(strict=True)
    config_path = root / DEFAULT_CONFIG_PATH
    config = load_bounded_diagnostic_config(
        config_path=config_path,
        repository_root=root,
    )
    execution = compute_bounded_execution_identity(
        repository_root=root,
        config_path=config_path,
        config=config,
    )
    pipeline = compute_verifier_pipeline_fingerprint(root=root)
    model = _require_exact_public_fields(
        snapshot["model"],
        _PUBLIC_MODEL_FIELDS,
        code="native_bounded_public_model_fields_invalid",
    )
    generation = _require_exact_public_fields(
        snapshot["generation"],
        _PUBLIC_GENERATION_FIELDS,
        code="native_bounded_public_generation_fields_invalid",
    )
    population = _require_exact_public_fields(
        snapshot["population"],
        _PUBLIC_POPULATION_FIELDS,
        code="native_bounded_public_population_fields_invalid",
    )
    identity = _identity_from_payload(model["identity"])
    _require_exact_identity(identity, config=DEFAULT_INVENTORY_CONFIG)
    inventory_counts = _validate_count_map(
        snapshot["inventory_status_counts"],
        allowed_keys=_INVENTORY_STATUSES,
        code="native_bounded_public_inventory_counts_invalid",
    )
    packet_counts = _validate_count_map(
        snapshot["packet_status_counts"],
        allowed_keys=_PACKET_STATUSES,
        code="native_bounded_public_packet_counts_invalid",
    )
    publication_counts = _validate_count_map(
        snapshot["publication_status_counts"],
        allowed_keys=None,
        key_predicate=_valid_publication_status,
        code="native_bounded_public_publication_counts_invalid",
    )
    candidate_distribution = _validate_count_map(
        snapshot["candidate_count_distribution"],
        allowed_keys=None,
        key_predicate=_valid_candidate_count_key,
        code="native_bounded_public_candidate_distribution_invalid",
    )
    hash_fields = {
        "input_bundle_sha256",
        "source_adapter_sha256",
        "prediction_ledger_sha256",
        "private_report_sha256",
        "config_file_sha256",
        "config_sha256",
        "diagnostic_execution_sha256",
        "official_schema_sha256",
        "downstream_verifier_pipeline_sha256",
        "summary_sha256",
    }
    if any(
        not isinstance(snapshot.get(field), str)
        or not _SHA256.fullmatch(str(snapshot.get(field)))
        for field in hash_fields
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_summary_sha256_invalid"
        )
    generation_int_fields = {
        "inventory_num_predict_cap",
        "packet_num_predict_cap",
        "inventory_generation_calls",
        "packet_generation_calls",
        "paper_generation_calls",
        "synthetic_preflight_calls",
        "legacy_single_stage_calls_included",
        "generation_retries",
    }
    if any(not _exact_nonnegative_int(generation[field]) for field in generation_int_fields):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_generation_integer_invalid"
        )
    public_int_fields = {
        "official_native_v1_estimable_publications",
        "official_native_v1_findings",
        "partial_packet_salvage_count",
        "external_provider_calls",
    }
    if any(not _exact_nonnegative_int(snapshot[field]) for field in public_int_fields):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_summary_integer_invalid"
        )
    if not isinstance(snapshot["external_provider_cost_usd"], float):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_summary_cost_type_invalid"
        )
    if (
        snapshot.get("public_summary_version") != PUBLIC_SUMMARY_VERSION
        or snapshot.get("diagnostic_version") != BOUNDED_DIAGNOSTIC_VERSION
        or snapshot.get("status")
        != "complete_retrospective_development_diagnostic"
        or snapshot.get("artifact_scope") != "aggregate_only_content_silent"
        or snapshot.get("scientific_role")
        != "diagnostic_only_no_accuracy_calibration_or_release_authority"
        or snapshot.get("config_file_sha256") != sha256_file(config_path)
        or snapshot.get("config_sha256") != config.config_sha256
        or snapshot.get("diagnostic_execution_sha256")
        != execution["execution_sha256"]
        or snapshot.get("official_schema_sha256")
        != hash_canonical(native_publication_extraction_json_schema())
        or snapshot.get("downstream_verifier_pipeline_sha256")
        != pipeline.pipeline_sha256
        or {key: value for key, value in model.items() if key != "identity"}
        != {
            "name": EXPECTED_MODEL,
            "digest": EXPECTED_MODEL_DIGEST,
            "runtime_version": EXPECTED_OLLAMA_VERSION,
            "parameter_size": EXPECTED_PARAMETER_SIZE,
            "quantization_level": EXPECTED_QUANTIZATION,
        }
        or generation.get("inventory_generation_config_sha256")
        != DEFAULT_INVENTORY_CONFIG.config_sha256
        or generation.get("packet_generation_config_sha256")
        != DEFAULT_PACKET_CONFIG.config_sha256
        or generation.get("inventory_num_predict_cap")
        != DEFAULT_INVENTORY_CONFIG.num_predict
        or generation.get("packet_num_predict_cap") != DEFAULT_PACKET_CONFIG.num_predict
        or generation.get("cap_hit_is_terminal") is not True
        or generation.get("paper_generation_calls")
        != generation.get("inventory_generation_calls", 0)
        + generation.get("packet_generation_calls", 0)
        or generation.get("synthetic_preflight_calls") != 1 + len(PACKET_MODELS)
        or generation.get("synthetic_preflight_status")
        != "passed_and_bound_before_paper_calls"
        or not isinstance(generation.get("schema_preflight_sha256"), str)
        or not _SHA256.fullmatch(str(generation.get("schema_preflight_sha256")))
        or generation.get("synthetic_preflight_calls_excluded_from_paper_counts")
        is not True
        or generation.get("legacy_single_stage_calls_included") != 0
        or generation.get("generation_retries") != 0
        or population.get("publications") != EXPECTED_ROWS
        or population.get("selection_labels_previously_opened") is not True
        or population.get("pristine_final_holdout_eligible") is not False
        or population.get("retrospective_model_schema_selection") is not True
        or sum(int(value) for value in inventory_counts.values()) != EXPECTED_ROWS
        or sum(int(value) for value in publication_counts.values()) != EXPECTED_ROWS
        or sum(int(value) for value in packet_counts.values())
        != generation.get("packet_generation_calls")
        or snapshot.get("official_native_v1_estimable_publications", -1)
        != publication_counts.get("official_native_v1_estimable", 0)
        or snapshot.get("official_native_v1_findings", -1)
        < snapshot.get("official_native_v1_estimable_publications", 0)
        or snapshot.get("partial_packet_salvage_count") != 0
        or snapshot.get("lexical_numeric_support_verified_for_promoted_packets")
        is not True
        or snapshot.get(
            "exact_source_projection_grounding_verified_for_promoted_packets"
        )
        is not True
        or snapshot.get("semantic_entailment_verified") is not False
        or snapshot.get("independent_numerical_gold_available") is not False
        or snapshot.get("extraction_accuracy_reported") is not False
        or snapshot.get("release_probability_authority") is not False
        or snapshot.get("claim_release_authority") is not False
        or snapshot.get("public_validation_scope")
        != "current_code_config_and_aggregate_shape_only"
        or snapshot.get("empirical_counts_require_private_receipt_replay") is not True
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
        or snapshot.get("caveats") != _PUBLIC_CAVEATS
        or sum(candidate_distribution.values()) > EXPECTED_ROWS
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_summary_semantic_mismatch"
        )
    return snapshot


def validate_bounded_finalized_artifacts_with_private_replay(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    inventory_receipts_dir: Path,
    packet_receipts_dir: Path,
    attempt_intents_dir: Path,
    preflight_dir: Path,
    prediction_ledger_path: Path,
    private_report: Mapping[str, Any],
    public_summary: Mapping[str, Any],
    repository_root: Path,
    expected_input_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay every private receipt before authorizing empirical aggregate counts."""

    expected_private, expected_public = finalize_bounded_diagnostic(
        input_bundle=input_bundle,
        prediction_ledger=prediction_ledger,
        inventory_receipts_dir=inventory_receipts_dir,
        packet_receipts_dir=packet_receipts_dir,
        attempt_intents_dir=attempt_intents_dir,
        preflight_dir=preflight_dir,
        prediction_ledger_path=prediction_ledger_path,
        repository_root=repository_root,
        expected_input_bundle_sha256=expected_input_bundle_sha256,
    )
    if dict(private_report) != expected_private:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_private_report_full_replay_mismatch"
        )
    if dict(public_summary) != expected_public:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_summary_private_replay_mismatch"
        )
    return expected_private, expected_public
