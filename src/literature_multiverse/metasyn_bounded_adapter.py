"""Provider-neutral bounded extraction adapter for the frozen MetaSyn pilot.

This module never calls a model and never opens review conclusions, directions, aggregate
effects, or test data.  It turns the private label-blind prepare bundle into exact per-row
inventory and packet contracts, validates caller-supplied structured outputs, enforces unique
quote grounding, and joins packets all-or-nothing at publication level.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_typed_pilot import (
    EXPECTED_SELECTED_COMPONENTS,
    EXPECTED_SELECTED_PAPERS,
    EXPECTED_SELECTED_QUESTIONS,
    PREPARE_BUNDLE_FILENAME,
    MetaSynPilotQuestionSpecV1,
    MetaSynPilotSourceProjectionRowV1,
    MetaSynTypedPilotPrepareBundleV1,
    compute_metasyn_typed_pilot_pipeline_fingerprint,
    validate_metasyn_typed_pilot_prepare,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidatePacketOutcome,
    NativeCandidateUnableToComplete,
    assemble_candidate_packets,
    canonical_packet_snapshot,
    inventory_generation_schema,
    packet_generation_schema,
    validate_inventory_for_row,
    validate_packet_for_candidate,
)
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    native_publication_extraction_json_schema,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.prompting import render_prompt_text

ADAPTER_VERSION = "metasyn-bounded-provider-neutral-adapter-v1"
ADAPTER_BUNDLE_VERSION = "metasyn-bounded-provider-neutral-bundle-v1"
ROW_CONTEXT_VERSION = "metasyn-bounded-row-context-v1"
INVENTORY_RECEIPT_VERSION = "metasyn-bounded-inventory-validation-v1"
PACKET_CALL_VERSION = "metasyn-bounded-packet-call-v1"
PACKET_RECEIPT_VERSION = "metasyn-bounded-packet-validation-v1"
GROUNDING_VERSION = "metasyn-bounded-unique-quote-grounding-v1"
PUBLICATION_RESULT_VERSION = "metasyn-bounded-publication-result-v1"
YIELD_REPORT_VERSION = "metasyn-bounded-private-yield-report-v1"
ADAPTER_COMPONENT_VERSION = "1"

INVENTORY_PROMPT_PATH = "prompts/metasyn_candidate_inventory.md"
PACKET_PROMPT_PATH = "prompts/metasyn_candidate_packet.md"
MAX_RENDERED_PROMPT_CHARACTERS = 48_000

_ADAPTER_DEPENDENCY_ENTRYPOINTS = (
    "src/literature_multiverse/metasyn_bounded_adapter.py",
)
_ADAPTER_NON_PYTHON_INPUTS = (
    INVENTORY_PROMPT_PATH,
    PACKET_PROMPT_PATH,
    "pyproject.toml",
    "uv.lock",
)

_FORBIDDEN_REFERENCE_KEYS = frozenset(
    {
        "CI_Lower",
        "CI_Upper",
        "Conclusion_Summary",
        "Effect_Direction",
        "Effect_Size_Category",
        "Effect_Size_Type",
        "Effect_Size_Value",
        "Heterogeneity_Level",
        "I2_Value",
        "Key_Insights",
        "P_Value",
        "Q_Value",
        "Statistical_Significance",
        "Tau2_Value",
        "conclusion_paragraph",
        "reference_verdict",
    }
)


class MetaSynBoundedAdapterError(ValueError):
    """A frozen row, model-facing contract, grounding join, or yield roster is unsafe."""


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _adapter_python_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_ADAPTER_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if not source_path.is_file():
            raise MetaSynBoundedAdapterError(
                f"metasyn_adapter_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"), filename=relative
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynBoundedAdapterError(
                f"metasyn_adapter_dependency_unreadable:{relative}"
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


def compute_metasyn_bounded_adapter_fingerprint(
    *, repository_root: Path, upstream_pilot_pipeline_sha256: str
) -> PipelineFingerprint:
    if not SHA256_RE.fullmatch(upstream_pilot_pipeline_sha256):
        raise MetaSynBoundedAdapterError("metasyn_adapter_upstream_pipeline_invalid")
    root = repository_root.resolve(strict=True)
    component = PipelineComponentSpec(
        component_id="metasyn-bounded-provider-neutral-adapter",
        component_version=ADAPTER_COMPONENT_VERSION,
        file_paths=sorted(
            {
                *_adapter_python_dependency_closure(root),
                *_ADAPTER_NON_PYTHON_INPUTS,
            }
        ),
        settings={
            "all_or_nothing_publication_join": True,
            "dependency_closure_entrypoints": list(_ADAPTER_DEPENDENCY_ENTRYPOINTS),
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name)
                for name in ("jsonschema", "pydantic")
            },
            "official_native_schema_sha256": hash_canonical(
                native_publication_extraction_json_schema()
            ),
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "provider_calls": 0,
            "reference_fields_opened": False,
            "upstream_pilot_pipeline_sha256": upstream_pilot_pipeline_sha256,
            "yield_only_no_accuracy_or_release_authority": True,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_forbidden_reference_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_REFERENCE_KEYS:
                raise MetaSynBoundedAdapterError(
                    f"metasyn_adapter_forbidden_reference_key:{key}"
                )
            _reject_forbidden_reference_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_reference_keys(item)


def _render_projection(source_row: MetaSynPilotSourceProjectionRowV1) -> str:
    projection = source_row.projection
    if not projection.passages:
        return "[NO_EXPOSED_SOURCE_PROJECTION]"
    rendered: list[str] = []
    for passage in projection.passages:
        rendered.append(
            f"LINE_ID: {passage.line_id}\n"
            f"SECTION_ENUM: {passage.exposed_section}\n"
            f"SOURCE_SECTION: {json.dumps(passage.section, ensure_ascii=False)}\n"
            "BEGIN_EXACT_SOURCE_TEXT\n"
            f"{passage.text}\n"
            "END_EXACT_SOURCE_TEXT"
        )
    return "\n\n".join(rendered)


def _row_inventory_prompt(
    *, base_prompt: str, source_row: MetaSynPilotSourceProjectionRowV1
) -> str:
    projection = source_row.projection
    constraints = {
        "allowed_outcomes": projection.allowed_outcomes,
        "exposed_line_ids": projection.exposed_line_ids,
        "release_grade_source_grounding_eligible": (
            projection.release_grade_source_grounding_eligible
        ),
        "source_strength": projection.source_strength,
        "source_strength_blockers": projection.source_strength_blockers,
    }
    return (
        base_prompt
        + "\n\n## Frozen row constraints\n"
        + json.dumps(constraints, sort_keys=True)
        + "\n\n## Frozen authoritative source projection\n"
        + _render_projection(source_row)
    )


class MetaSynBoundedRowContextV1(ContractModel):
    row_context_version: Literal["metasyn-bounded-row-context-v1"] = (
        ROW_CONTEXT_VERSION
    )
    question_bundle_sha256: str
    question_spec: MetaSynPilotQuestionSpecV1
    question_spec_sha256: str
    independence_component_id: Annotated[str, Field(min_length=1, max_length=256)]
    independence_component_review_ids: Annotated[
        list[int], Field(min_length=1, max_length=512)
    ]
    independence_component_membership_sha256: str
    source_row: MetaSynPilotSourceProjectionRowV1
    source_row_sha256: str
    projection_sha256: str
    source_locator: Annotated[str, Field(min_length=1, max_length=512)]
    allowed_outcomes: Annotated[list[str], Field(min_length=1, max_length=16)]
    outcome_positive_directions: dict[str, Annotated[str, Field(min_length=1)]]
    allowed_moderators: list[str]
    allowed_sections: list[
        Literal["Abstract", "FigureTable", "Methods", "Results", "Title"]
    ]
    inventory_prompt_version: Literal["metasyn-candidate-inventory-v1"]
    inventory_prompt: Annotated[
        str, Field(min_length=1, max_length=MAX_RENDERED_PROMPT_CHARACTERS)
    ]
    inventory_prompt_sha256: str
    packet_base_prompt_version: Literal["metasyn-candidate-packet-v1"]
    packet_base_prompt: Annotated[
        str, Field(min_length=1, max_length=MAX_RENDERED_PROMPT_CHARACTERS)
    ]
    packet_base_prompt_sha256: str
    inventory_schema: dict[str, Any]
    inventory_schema_sha256: str
    row_context_sha256: str

    @field_validator(
        "question_bundle_sha256",
        "question_spec_sha256",
        "independence_component_membership_sha256",
        "source_row_sha256",
        "projection_sha256",
        "inventory_prompt_sha256",
        "packet_base_prompt_sha256",
        "inventory_schema_sha256",
        "row_context_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_adapter_sha256_invalid:{info.field_name}")
        return value

    @field_validator("allowed_outcomes", "allowed_moderators", "allowed_sections")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_adapter_not_sorted_unique:{info.field_name}")
        return value

    @field_validator("independence_component_review_ids")
    @classmethod
    def validate_component_review_ids(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)) or any(item < 0 for item in value):
            raise ValueError("metasyn_adapter_component_review_ids_not_sorted_unique")
        return value

    @field_validator("outcome_positive_directions")
    @classmethod
    def validate_directions(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())):
            raise ValueError("metasyn_adapter_positive_directions_not_sorted")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> MetaSynBoundedRowContextV1:
        projection = self.source_row.projection
        if self.independence_component_membership_sha256 != hash_canonical(
            self.independence_component_review_ids
        ):
            raise ValueError("metasyn_adapter_component_membership_hash_mismatch")
        if self.question_spec.review_id not in self.independence_component_review_ids:
            raise ValueError("metasyn_adapter_question_outside_component")
        if self.question_spec_sha256 != self.question_spec.question_spec_sha256:
            raise ValueError("metasyn_adapter_question_hash_alias_mismatch")
        if self.source_row_sha256 != self.source_row.source_row_sha256:
            raise ValueError("metasyn_adapter_source_row_hash_alias_mismatch")
        if self.projection_sha256 != projection.projection_sha256:
            raise ValueError("metasyn_adapter_projection_hash_alias_mismatch")
        if self.source_row.question_id != self.question_spec.question_id:
            raise ValueError("metasyn_adapter_source_question_mismatch")
        if self.source_locator != projection.source_locator:
            raise ValueError("metasyn_adapter_source_locator_mismatch")
        if self.allowed_outcomes != projection.allowed_outcomes:
            raise ValueError("metasyn_adapter_outcome_projection_mismatch")
        if self.allowed_outcomes != self.question_spec.allowed_outcomes:
            raise ValueError("metasyn_adapter_outcome_question_mismatch")
        if set(self.outcome_positive_directions) != set(self.allowed_outcomes):
            raise ValueError("metasyn_adapter_positive_direction_membership_mismatch")
        if self.outcome_positive_directions != (
            self.question_spec.positive_direction_means_by_outcome_id
        ):
            raise ValueError("metasyn_adapter_positive_direction_question_mismatch")
        if self.allowed_moderators != projection.allowed_moderators:
            raise ValueError("metasyn_adapter_moderator_projection_mismatch")
        if self.allowed_sections != projection.exposed_sections:
            raise ValueError("metasyn_adapter_section_projection_mismatch")
        if _sha256_text(self.inventory_prompt) != self.inventory_prompt_sha256:
            raise ValueError("metasyn_adapter_inventory_prompt_hash_mismatch")
        if _sha256_text(self.packet_base_prompt) != self.packet_base_prompt_sha256:
            raise ValueError("metasyn_adapter_packet_prompt_hash_mismatch")
        if "__FROZEN_CANDIDATE_JSON__" not in self.packet_base_prompt:
            raise ValueError("metasyn_adapter_packet_candidate_sentinel_missing")
        expected_schema = inventory_generation_schema(
            exposed_line_ids=projection.exposed_line_ids,
            allowed_outcomes=projection.allowed_outcomes,
        )
        if self.inventory_schema != expected_schema:
            raise ValueError("metasyn_adapter_inventory_schema_mismatch")
        if hash_canonical(self.inventory_schema) != self.inventory_schema_sha256:
            raise ValueError("metasyn_adapter_inventory_schema_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_context_sha256"})
        if hash_canonical(payload) != self.row_context_sha256:
            raise ValueError("metasyn_adapter_row_context_hash_mismatch")
        return self

    @property
    def row_key(self) -> str:
        return f"{self.question_spec.question_id}::{self.source_row.doc_id}"

    @property
    def component_descriptor(self) -> dict[str, Any]:
        return {
            "independence_component_id": self.independence_component_id,
            "independence_component_review_ids": self.independence_component_review_ids,
            "independence_component_membership_sha256": (
                self.independence_component_membership_sha256
            ),
        }


class MetaSynBoundedAdapterBundleV1(ContractModel):
    adapter_bundle_version: Literal[
        "metasyn-bounded-provider-neutral-bundle-v1"
    ] = ADAPTER_BUNDLE_VERSION
    adapter_version: Literal["metasyn-bounded-provider-neutral-adapter-v1"] = (
        ADAPTER_VERSION
    )
    status: Literal["provider_neutral_inputs_frozen_reference_fields_unopened"] = (
        "provider_neutral_inputs_frozen_reference_fields_unopened"
    )
    prepare_bundle_sha256: str
    upstream_pilot_pipeline_sha256: str
    adapter_pipeline_fingerprint: PipelineFingerprint
    adapter_pipeline_sha256: str
    official_native_schema_sha256: str
    inventory_prompt_path: Literal["prompts/metasyn_candidate_inventory.md"] = (
        INVENTORY_PROMPT_PATH
    )
    inventory_prompt_file_sha256: str
    packet_prompt_path: Literal["prompts/metasyn_candidate_packet.md"] = (
        PACKET_PROMPT_PATH
    )
    packet_prompt_file_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    question_membership_sha256: str
    component_membership_sha256: str
    row_membership_sha256: str
    row_contexts: Annotated[list[MetaSynBoundedRowContextV1], Field(min_length=1)]
    reference_fields_unopened: Literal[True] = True
    model_calls_made: Literal[False] = False
    directional_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal[
        "contract_grounding_publication_and_synthesis_input_yield_only"
    ] = "contract_grounding_publication_and_synthesis_input_yield_only"
    adapter_bundle_sha256: str

    @field_validator(
        "prepare_bundle_sha256",
        "upstream_pilot_pipeline_sha256",
        "adapter_pipeline_sha256",
        "official_native_schema_sha256",
        "inventory_prompt_file_sha256",
        "packet_prompt_file_sha256",
        "question_membership_sha256",
        "component_membership_sha256",
        "row_membership_sha256",
        "adapter_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_adapter_bundle_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynBoundedAdapterBundleV1:
        if self.adapter_pipeline_sha256 != self.adapter_pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_adapter_pipeline_hash_alias_mismatch")
        component = self.adapter_pipeline_fingerprint.components
        if len(component) != 1 or component[0].component_id != (
            "metasyn-bounded-provider-neutral-adapter"
        ):
            raise ValueError("metasyn_adapter_pipeline_component_mismatch")
        settings = component[0].settings
        if settings.get("upstream_pilot_pipeline_sha256") != (
            self.upstream_pilot_pipeline_sha256
        ):
            raise ValueError("metasyn_adapter_upstream_pipeline_setting_mismatch")
        if settings.get("official_native_schema_sha256") != (
            self.official_native_schema_sha256
        ):
            raise ValueError("metasyn_adapter_official_schema_setting_mismatch")
        keys = [row.row_key for row in self.row_contexts]
        if keys != sorted(set(keys)):
            raise ValueError("metasyn_adapter_rows_not_sorted_unique")
        questions = sorted({row.question_spec.question_id for row in self.row_contexts})
        components_by_id: dict[str, dict[str, Any]] = {}
        for row in self.row_contexts:
            prior = components_by_id.setdefault(
                row.independence_component_id, row.component_descriptor
            )
            if prior != row.component_descriptor:
                raise ValueError("metasyn_adapter_component_descriptor_conflict")
        components = [components_by_id[key] for key in sorted(components_by_id)]
        if len(questions) != self.question_count or len(components) != self.component_count:
            raise ValueError("metasyn_adapter_question_or_component_count_mismatch")
        if len(keys) != self.publication_count:
            raise ValueError("metasyn_adapter_publication_count_mismatch")
        if self.question_membership_sha256 != hash_canonical(questions):
            raise ValueError("metasyn_adapter_question_membership_hash_mismatch")
        if self.component_membership_sha256 != hash_canonical(components):
            raise ValueError("metasyn_adapter_component_membership_hash_mismatch")
        if self.row_membership_sha256 != hash_canonical(keys):
            raise ValueError("metasyn_adapter_row_membership_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"adapter_bundle_sha256"})
        if hash_canonical(payload) != self.adapter_bundle_sha256:
            raise ValueError("metasyn_adapter_bundle_hash_mismatch")
        return self


def _read_prompt(root: Path, relative: str) -> tuple[str, str]:
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise MetaSynBoundedAdapterError("metasyn_adapter_prompt_path_unsafe")
    try:
        return path.read_text(encoding="utf-8"), sha256_file(path)
    except (OSError, UnicodeDecodeError) as exc:
        raise MetaSynBoundedAdapterError("metasyn_adapter_prompt_unreadable") from exc


def freeze_metasyn_bounded_row_context(
    *,
    question_bundle_sha256: str,
    question_spec: MetaSynPilotQuestionSpecV1,
    independence_component_id: str,
    independence_component_review_ids: Sequence[int],
    independence_component_membership_sha256: str,
    source_row: MetaSynPilotSourceProjectionRowV1,
    inventory_template: str,
    packet_template: str,
) -> MetaSynBoundedRowContextV1:
    """Freeze one exact per-question/per-publication model-facing surface."""

    question_spec = MetaSynPilotQuestionSpecV1.model_validate(question_spec)
    source_row = MetaSynPilotSourceProjectionRowV1.model_validate(source_row)
    question_payload = question_spec.model_dump(mode="json")
    _reject_forbidden_reference_keys(question_payload)
    question_json = json.dumps(
        question_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    inventory_base, inventory_version = render_prompt_text(
        inventory_template, {"QUESTION_SPEC_JSON": question_json}
    )
    packet_base, packet_version = render_prompt_text(
        packet_template, {"QUESTION_SPEC_JSON": question_json}
    )
    # ContractModel strips outer string whitespace. Canonicalize before hashing so the
    # stored prompt bytes and their receipt agree exactly.
    inventory_base = inventory_base.strip()
    packet_base = packet_base.strip()
    if "__FROZEN_CANDIDATE_JSON__" not in packet_base:
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_packet_candidate_sentinel_missing"
        )
    projection = source_row.projection
    inventory_prompt = _row_inventory_prompt(
        base_prompt=inventory_base, source_row=source_row
    )
    schema = inventory_generation_schema(
        exposed_line_ids=projection.exposed_line_ids,
        allowed_outcomes=projection.allowed_outcomes,
    )
    row_payload: dict[str, Any] = {
        "row_context_version": ROW_CONTEXT_VERSION,
        "question_bundle_sha256": question_bundle_sha256,
        "question_spec": question_spec,
        "question_spec_sha256": question_spec.question_spec_sha256,
        "independence_component_id": independence_component_id,
        "independence_component_review_ids": sorted(
            independence_component_review_ids
        ),
        "independence_component_membership_sha256": (
            independence_component_membership_sha256
        ),
        "source_row": source_row,
        "source_row_sha256": source_row.source_row_sha256,
        "projection_sha256": projection.projection_sha256,
        "source_locator": projection.source_locator,
        "allowed_outcomes": projection.allowed_outcomes,
        "outcome_positive_directions": (
            question_spec.positive_direction_means_by_outcome_id
        ),
        "allowed_moderators": projection.allowed_moderators,
        "allowed_sections": projection.exposed_sections,
        "inventory_prompt_version": inventory_version,
        "inventory_prompt": inventory_prompt,
        "inventory_prompt_sha256": _sha256_text(inventory_prompt),
        "packet_base_prompt_version": packet_version,
        "packet_base_prompt": packet_base,
        "packet_base_prompt_sha256": _sha256_text(packet_base),
        "inventory_schema": schema,
        "inventory_schema_sha256": hash_canonical(schema),
    }
    return MetaSynBoundedRowContextV1.model_validate(
        {**row_payload, "row_context_sha256": hash_canonical(row_payload)}
    )


def freeze_metasyn_bounded_adapter_bundle(
    *,
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1 | Mapping[str, Any],
    repository_root: Path,
) -> MetaSynBoundedAdapterBundleV1:
    """Freeze all 32 rows from a prepare bundle already externally replayed by its custodian."""

    root = repository_root.resolve(strict=True)
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(prepare_bundle)
    if prepared.access_state.reference_fields_unopened is not True:
        raise MetaSynBoundedAdapterError("metasyn_adapter_reference_fields_opened")
    current_pilot = compute_metasyn_typed_pilot_pipeline_fingerprint(root=root)
    if current_pilot != prepared.pilot_pipeline_fingerprint:
        raise MetaSynBoundedAdapterError("metasyn_adapter_upstream_pipeline_stale")
    inventory_template, inventory_file_sha256 = _read_prompt(
        root, INVENTORY_PROMPT_PATH
    )
    packet_template, packet_file_sha256 = _read_prompt(root, PACKET_PROMPT_PATH)
    adapter_pipeline = compute_metasyn_bounded_adapter_fingerprint(
        repository_root=root,
        upstream_pilot_pipeline_sha256=prepared.pilot_pipeline_sha256,
    )
    row_contexts: list[MetaSynBoundedRowContextV1] = []
    component_descriptors: dict[str, dict[str, Any]] = {}
    for question in prepared.questions:
        question_spec = question.question_spec
        descriptor = {
            "independence_component_id": question.independence_component_id,
            "independence_component_review_ids": (
                question.independence_component_review_ids
            ),
            "independence_component_membership_sha256": (
                question.independence_component_membership_sha256
            ),
        }
        prior_descriptor = component_descriptors.setdefault(
            question.independence_component_id, descriptor
        )
        if prior_descriptor != descriptor:
            raise MetaSynBoundedAdapterError(
                "metasyn_adapter_component_descriptor_conflict"
            )
        for source_row in question.source_rows:
            row_contexts.append(
                freeze_metasyn_bounded_row_context(
                    question_bundle_sha256=question.question_bundle_sha256,
                    question_spec=question_spec,
                    independence_component_id=question.independence_component_id,
                    independence_component_review_ids=(
                        question.independence_component_review_ids
                    ),
                    independence_component_membership_sha256=(
                        question.independence_component_membership_sha256
                    ),
                    source_row=source_row,
                    inventory_template=inventory_template,
                    packet_template=packet_template,
                )
            )
    row_contexts.sort(key=lambda row: row.row_key)
    question_ids = sorted({row.question_spec.question_id for row in row_contexts})
    row_keys = [row.row_key for row in row_contexts]
    payload: dict[str, Any] = {
        "adapter_bundle_version": ADAPTER_BUNDLE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "status": "provider_neutral_inputs_frozen_reference_fields_unopened",
        "prepare_bundle_sha256": prepared.prepare_bundle_sha256,
        "upstream_pilot_pipeline_sha256": prepared.pilot_pipeline_sha256,
        "adapter_pipeline_fingerprint": adapter_pipeline,
        "adapter_pipeline_sha256": adapter_pipeline.pipeline_sha256,
        "official_native_schema_sha256": hash_canonical(
            native_publication_extraction_json_schema()
        ),
        "inventory_prompt_path": INVENTORY_PROMPT_PATH,
        "inventory_prompt_file_sha256": inventory_file_sha256,
        "packet_prompt_path": PACKET_PROMPT_PATH,
        "packet_prompt_file_sha256": packet_file_sha256,
        "question_count": len(question_ids),
        "component_count": len(component_descriptors),
        "publication_count": len(row_contexts),
        "question_membership_sha256": hash_canonical(question_ids),
        "component_membership_sha256": hash_canonical(
            [component_descriptors[key] for key in sorted(component_descriptors)]
        ),
        "row_membership_sha256": hash_canonical(row_keys),
        "row_contexts": row_contexts,
        "reference_fields_unopened": True,
        "model_calls_made": False,
        "directional_accuracy_authority": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
    }
    return MetaSynBoundedAdapterBundleV1.model_validate(
        {**payload, "adapter_bundle_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_bounded_adapter_bundle_from_workspace(
    *,
    repository_root: Path,
    workspace: Path,
) -> MetaSynBoundedAdapterBundleV1:
    """Replay the private prepare workspace, then freeze the exact adapter roster.

    This is the executable boundary.  The lower-level bundle factory deliberately accepts
    only a prepare object whose external replay has already happened in the same process.
    """

    root = repository_root.resolve(strict=True)
    receipt = validate_metasyn_typed_pilot_prepare(
        repository_root=root, workspace=workspace
    )
    private = workspace if workspace.is_absolute() else root / workspace
    bundle_path = private.resolve(strict=True) / PREPARE_BUNDLE_FILENAME
    try:
        raw = bundle_path.read_bytes()
    except OSError as exc:
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_prepare_bundle_unreadable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != receipt.prepare_bundle_file_sha256:
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_prepare_bundle_post_replay_file_mismatch"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_prepare_bundle_json_invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_prepare_bundle_json_not_object"
        )
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(payload)
    if prepared.prepare_bundle_sha256 != receipt.prepare_bundle_sha256:
        raise MetaSynBoundedAdapterError(
            "metasyn_adapter_prepare_bundle_receipt_mismatch"
        )
    return freeze_metasyn_bounded_adapter_bundle(
        prepare_bundle=prepared, repository_root=root
    )


def validate_metasyn_bounded_adapter_bundle_external_replay(
    *,
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    repository_root: Path,
    workspace: Path,
) -> MetaSynBoundedAdapterBundleV1:
    canonical = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    replayed = freeze_metasyn_bounded_adapter_bundle_from_workspace(
        repository_root=repository_root, workspace=workspace
    )
    if replayed != canonical:
        raise MetaSynBoundedAdapterError("metasyn_adapter_external_replay_mismatch")
    return canonical


class MetaSynInventoryValidationReceiptV1(ContractModel):
    receipt_version: Literal["metasyn-bounded-inventory-validation-v1"] = (
        INVENTORY_RECEIPT_VERSION
    )
    row_context_sha256: str
    inventory_schema_sha256: str
    inventory: NativeCandidateInventory
    inventory_sha256: str
    status: Literal[
        "candidates_authorized",
        "no_candidate_non_authorizing",
        "capacity_or_uncertainty_non_authorizing",
    ]
    receipt_sha256: str

    @field_validator(
        "row_context_sha256",
        "inventory_schema_sha256",
        "inventory_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_inventory_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynInventoryValidationReceiptV1:
        if self.inventory_sha256 != hash_canonical(self.inventory):
            raise ValueError("metasyn_inventory_payload_hash_mismatch")
        expected_status = (
            "candidates_authorized"
            if self.inventory.authorizes_packet_generation()
            else "no_candidate_non_authorizing"
            if self.inventory.inventory_status == "no_candidate_found"
            else "capacity_or_uncertainty_non_authorizing"
        )
        if self.status != expected_status:
            raise ValueError("metasyn_inventory_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("metasyn_inventory_receipt_hash_mismatch")
        return self


def freeze_metasyn_inventory_validation_receipt(
    *, row: MetaSynBoundedRowContextV1, value: Any
) -> MetaSynInventoryValidationReceiptV1:
    row = MetaSynBoundedRowContextV1.model_validate(row)
    inventory = validate_inventory_for_row(
        value,
        exposed_line_ids=row.source_row.projection.exposed_line_ids,
        allowed_outcomes=row.allowed_outcomes,
    )
    status = (
        "candidates_authorized"
        if inventory.authorizes_packet_generation()
        else "no_candidate_non_authorizing"
        if inventory.inventory_status == "no_candidate_found"
        else "capacity_or_uncertainty_non_authorizing"
    )
    payload: dict[str, Any] = {
        "receipt_version": INVENTORY_RECEIPT_VERSION,
        "row_context_sha256": row.row_context_sha256,
        "inventory_schema_sha256": row.inventory_schema_sha256,
        "inventory": inventory,
        "inventory_sha256": hash_canonical(inventory),
        "status": status,
    }
    return MetaSynInventoryValidationReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_metasyn_inventory_validation_receipt(
    *,
    receipt: MetaSynInventoryValidationReceiptV1 | Mapping[str, Any],
    row: MetaSynBoundedRowContextV1,
) -> MetaSynInventoryValidationReceiptV1:
    canonical = MetaSynInventoryValidationReceiptV1.model_validate(receipt)
    replayed = freeze_metasyn_inventory_validation_receipt(
        row=row, value=canonical.inventory
    )
    if replayed != canonical:
        raise MetaSynBoundedAdapterError("metasyn_inventory_receipt_replay_mismatch")
    return canonical


class MetaSynPacketCallV1(ContractModel):
    packet_call_version: Literal["metasyn-bounded-packet-call-v1"] = (
        PACKET_CALL_VERSION
    )
    row_context_sha256: str
    inventory_receipt_sha256: str
    candidate: NativeCandidateDescriptor
    candidate_sha256: str
    rendered_prompt: Annotated[
        str, Field(min_length=1, max_length=MAX_RENDERED_PROMPT_CHARACTERS)
    ]
    rendered_prompt_sha256: str
    packet_schema: dict[str, Any]
    packet_schema_sha256: str
    packet_call_sha256: str

    @field_validator(
        "row_context_sha256",
        "inventory_receipt_sha256",
        "candidate_sha256",
        "rendered_prompt_sha256",
        "packet_schema_sha256",
        "packet_call_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_packet_call_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_call(self) -> MetaSynPacketCallV1:
        if self.candidate_sha256 != self.candidate.descriptor_sha256:
            raise ValueError("metasyn_packet_candidate_hash_mismatch")
        if _sha256_text(self.rendered_prompt) != self.rendered_prompt_sha256:
            raise ValueError("metasyn_packet_prompt_hash_mismatch")
        if hash_canonical(self.packet_schema) != self.packet_schema_sha256:
            raise ValueError("metasyn_packet_schema_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"packet_call_sha256"})
        if hash_canonical(payload) != self.packet_call_sha256:
            raise ValueError("metasyn_packet_call_hash_mismatch")
        return self


def freeze_metasyn_packet_call(
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
    candidate_index: int,
) -> MetaSynPacketCallV1:
    row = MetaSynBoundedRowContextV1.model_validate(row)
    inventory_receipt = validate_metasyn_inventory_validation_receipt(
        receipt=inventory_receipt, row=row
    )
    if not inventory_receipt.inventory.authorizes_packet_generation():
        raise MetaSynBoundedAdapterError("metasyn_packet_inventory_non_authorizing")
    by_index = {
        item.candidate_index: item for item in inventory_receipt.inventory.candidates
    }
    candidate = by_index.get(candidate_index)
    if candidate is None:
        raise MetaSynBoundedAdapterError("metasyn_packet_candidate_not_in_inventory")
    candidate_json = json.dumps(
        candidate.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
    )
    rendered = row.packet_base_prompt.replace(
        "__FROZEN_CANDIDATE_JSON__", candidate_json
    )
    if "__FROZEN_CANDIDATE_JSON__" in rendered:
        raise MetaSynBoundedAdapterError("metasyn_packet_candidate_render_failed")
    constraints = {
        "allowed_moderators": row.allowed_moderators,
        "allowed_sections": row.allowed_sections,
        "outcome_positive_direction": row.outcome_positive_directions[
            candidate.outcome_name
        ],
        "source_locator": row.source_locator,
        "source_strength": row.source_row.source_projection_strength,
    }
    rendered = (
        rendered
        + "\n\n## Frozen row constraints\n"
        + json.dumps(constraints, sort_keys=True)
        + "\n\n## Frozen authoritative source projection\n"
        + _render_projection(row.source_row)
    )
    schema = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=row.source_row.projection.exposed_line_ids,
        source_locator=row.source_locator,
        allowed_outcomes=row.allowed_outcomes,
        allowed_moderators=row.allowed_moderators,
        allowed_sections=row.allowed_sections,
        outcome_positive_directions=row.outcome_positive_directions,
    )
    payload: dict[str, Any] = {
        "packet_call_version": PACKET_CALL_VERSION,
        "row_context_sha256": row.row_context_sha256,
        "inventory_receipt_sha256": inventory_receipt.receipt_sha256,
        "candidate": candidate,
        "candidate_sha256": candidate.descriptor_sha256,
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": _sha256_text(rendered),
        "packet_schema": schema,
        "packet_schema_sha256": hash_canonical(schema),
    }
    return MetaSynPacketCallV1.model_validate(
        {**payload, "packet_call_sha256": hash_canonical(payload)}
    )


class MetaSynUniqueQuoteGroundingV1(ContractModel):
    grounding_version: Literal["metasyn-bounded-unique-quote-grounding-v1"] = (
        GROUNDING_VERSION
    )
    projection_sha256: str
    packet_payload_sha256: str
    line_id: str
    passage_rank: int
    exposed_section: Literal["Abstract", "FigureTable", "Methods", "Results", "Title"]
    quote_sha256: str
    passage_quote_start: Annotated[int, Field(ge=0)]
    passage_quote_end_exclusive: Annotated[int, Field(gt=0)]
    source_char_start: Annotated[int, Field(ge=0)]
    source_char_end_exclusive: Annotated[int, Field(gt=0)]
    source_utf8_byte_start: Annotated[int, Field(ge=0)]
    source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    unique_matching_occurrences: Literal[1] = 1
    grounding_sha256: str

    @field_validator(
        "projection_sha256",
        "packet_payload_sha256",
        "quote_sha256",
        "grounding_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_grounding_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_grounding(self) -> MetaSynUniqueQuoteGroundingV1:
        if self.passage_quote_end_exclusive <= self.passage_quote_start:
            raise ValueError("metasyn_grounding_passage_offsets_invalid")
        if self.source_char_end_exclusive <= self.source_char_start:
            raise ValueError("metasyn_grounding_source_offsets_invalid")
        if self.source_utf8_byte_end_exclusive <= self.source_utf8_byte_start:
            raise ValueError("metasyn_grounding_source_byte_offsets_invalid")
        payload = self.model_dump(mode="json", exclude={"grounding_sha256"})
        if hash_canonical(payload) != self.grounding_sha256:
            raise ValueError("metasyn_grounding_hash_mismatch")
        return self


def _unique_quote_grounding(
    *, row: MetaSynBoundedRowContextV1, packet: NativeCandidatePacketOutcome
) -> MetaSynUniqueQuoteGroundingV1:
    if isinstance(packet, NativeCandidateUnableToComplete):
        raise MetaSynBoundedAdapterError("metasyn_grounding_unable_packet_forbidden")
    evidence = packet.evidence
    matches: list[tuple[Any, int]] = []
    for passage in row.source_row.projection.passages:
        cursor = 0
        while True:
            index = passage.text.find(evidence.quote, cursor)
            if index < 0:
                break
            matches.append((passage, index))
            cursor = index + 1
    if len(matches) != 1:
        raise MetaSynBoundedAdapterError(
            f"metasyn_packet_quote_not_unique_in_frozen_projection:{len(matches)}"
        )
    passage, start = matches[0]
    if (
        passage.line_id not in set(evidence.line_ids)
        or passage.exposed_section != evidence.section
    ):
        raise MetaSynBoundedAdapterError(
            "metasyn_packet_unique_quote_not_in_cited_projection"
        )
    end = start + len(evidence.quote)
    source_char_start = passage.source_char_start + start
    source_char_end = source_char_start + len(evidence.quote)
    source_byte_start = passage.source_utf8_byte_start + len(
        passage.text[:start].encode("utf-8")
    )
    source_byte_end = source_byte_start + len(evidence.quote.encode("utf-8"))
    packet_sha256 = hash_canonical(canonical_packet_snapshot(packet))
    payload: dict[str, Any] = {
        "grounding_version": GROUNDING_VERSION,
        "projection_sha256": row.projection_sha256,
        "packet_payload_sha256": packet_sha256,
        "line_id": passage.line_id,
        "passage_rank": passage.passage_rank,
        "exposed_section": passage.exposed_section,
        "quote_sha256": _sha256_text(evidence.quote),
        "passage_quote_start": start,
        "passage_quote_end_exclusive": end,
        "source_char_start": source_char_start,
        "source_char_end_exclusive": source_char_end,
        "source_utf8_byte_start": source_byte_start,
        "source_utf8_byte_end_exclusive": source_byte_end,
        "unique_matching_occurrences": 1,
    }
    return MetaSynUniqueQuoteGroundingV1.model_validate(
        {**payload, "grounding_sha256": hash_canonical(payload)}
    )


class MetaSynPacketValidationReceiptV1(ContractModel):
    receipt_version: Literal["metasyn-bounded-packet-validation-v1"] = (
        PACKET_RECEIPT_VERSION
    )
    packet_call_sha256: str
    candidate_index: int
    candidate_sha256: str
    packet_status: Literal["completed", "unable_to_complete"]
    packet_payload: dict[str, Any]
    packet_payload_sha256: str
    quote_grounding: MetaSynUniqueQuoteGroundingV1 | None
    quote_grounding_sha256: str | None
    receipt_sha256: str

    @field_validator(
        "packet_call_sha256",
        "candidate_sha256",
        "packet_payload_sha256",
        "quote_grounding_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_packet_receipt_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynPacketValidationReceiptV1:
        if self.packet_payload_sha256 != hash_canonical(self.packet_payload):
            raise ValueError("metasyn_packet_payload_hash_mismatch")
        completed = self.packet_status == "completed"
        if completed != (self.quote_grounding is not None):
            raise ValueError("metasyn_packet_grounding_presence_mismatch")
        expected_grounding = (
            self.quote_grounding.grounding_sha256
            if self.quote_grounding is not None
            else None
        )
        if self.quote_grounding_sha256 != expected_grounding:
            raise ValueError("metasyn_packet_grounding_hash_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("metasyn_packet_receipt_hash_mismatch")
        return self


def _validate_packet_call_external(
    *,
    call: MetaSynPacketCallV1,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
) -> MetaSynPacketCallV1:
    canonical = MetaSynPacketCallV1.model_validate(call)
    expected = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=canonical.candidate.candidate_index,
    )
    if expected != canonical:
        raise MetaSynBoundedAdapterError("metasyn_packet_call_replay_mismatch")
    return canonical


def freeze_metasyn_packet_validation_receipt(
    *,
    call: MetaSynPacketCallV1,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
    value: Any,
) -> MetaSynPacketValidationReceiptV1:
    call = _validate_packet_call_external(
        call=call, row=row, inventory_receipt=inventory_receipt
    )
    packet = validate_packet_for_candidate(
        value,
        candidate=call.candidate,
        exposed_line_ids=row.source_row.projection.exposed_line_ids,
        source_locator=row.source_locator,
        allowed_outcomes=row.allowed_outcomes,
        allowed_moderators=row.allowed_moderators,
        allowed_sections=row.allowed_sections,
        outcome_positive_directions=row.outcome_positive_directions,
    )
    packet_payload = canonical_packet_snapshot(packet)
    grounding = (
        None
        if isinstance(packet, NativeCandidateUnableToComplete)
        else _unique_quote_grounding(row=row, packet=packet)
    )
    payload: dict[str, Any] = {
        "receipt_version": PACKET_RECEIPT_VERSION,
        "packet_call_sha256": call.packet_call_sha256,
        "candidate_index": call.candidate.candidate_index,
        "candidate_sha256": call.candidate_sha256,
        "packet_status": packet.packet_status,
        "packet_payload": packet_payload,
        "packet_payload_sha256": hash_canonical(packet_payload),
        "quote_grounding": grounding,
        "quote_grounding_sha256": (
            grounding.grounding_sha256 if grounding is not None else None
        ),
    }
    return MetaSynPacketValidationReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_metasyn_packet_validation_receipt(
    *,
    receipt: MetaSynPacketValidationReceiptV1 | Mapping[str, Any],
    call: MetaSynPacketCallV1,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
) -> MetaSynPacketValidationReceiptV1:
    canonical = MetaSynPacketValidationReceiptV1.model_validate(receipt)
    replayed = freeze_metasyn_packet_validation_receipt(
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=canonical.packet_payload,
    )
    if replayed != canonical:
        raise MetaSynBoundedAdapterError("metasyn_packet_receipt_replay_mismatch")
    return canonical


class MetaSynPublicationResultV1(ContractModel):
    result_version: Literal["metasyn-bounded-publication-result-v1"] = (
        PUBLICATION_RESULT_VERSION
    )
    row_context_sha256: str
    inventory_receipt: MetaSynInventoryValidationReceiptV1
    inventory_receipt_sha256: str
    packet_receipts: list[MetaSynPacketValidationReceiptV1]
    packet_receipt_sha256s: list[str]
    status: Literal[
        "typed_publication_output",
        "abstained_inventory_no_candidate",
        "abstained_inventory_uncertain",
        "abstained_packet_set_incomplete",
        "abstained_packet_unable",
    ]
    blocking_reasons: list[str]
    official_output: NativePublicationExtraction | None
    official_output_sha256: str | None
    result_sha256: str

    @field_validator(
        "row_context_sha256",
        "inventory_receipt_sha256",
        "official_output_sha256",
        "result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_publication_sha256_invalid:{info.field_name}")
        return value

    @field_validator("packet_receipt_sha256s", "blocking_reasons")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_publication_not_sorted_unique:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> MetaSynPublicationResultV1:
        if self.inventory_receipt_sha256 != self.inventory_receipt.receipt_sha256:
            raise ValueError("metasyn_publication_inventory_hash_alias_mismatch")
        expected_packets = sorted(item.receipt_sha256 for item in self.packet_receipts)
        if self.packet_receipt_sha256s != expected_packets:
            raise ValueError("metasyn_publication_packet_hashes_mismatch")
        typed = self.status == "typed_publication_output"
        if typed != (self.official_output is not None):
            raise ValueError("metasyn_publication_output_presence_mismatch")
        expected_output_sha256 = (
            hash_canonical(self.official_output)
            if self.official_output is not None
            else None
        )
        if self.official_output_sha256 != expected_output_sha256:
            raise ValueError("metasyn_publication_output_hash_mismatch")
        if typed == bool(self.blocking_reasons):
            raise ValueError("metasyn_publication_blocking_reason_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("metasyn_publication_result_hash_mismatch")
        return self


def freeze_metasyn_publication_result(
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
    packet_receipts: Sequence[MetaSynPacketValidationReceiptV1],
) -> MetaSynPublicationResultV1:
    row = MetaSynBoundedRowContextV1.model_validate(row)
    inventory_receipt = validate_metasyn_inventory_validation_receipt(
        receipt=inventory_receipt, row=row
    )
    inventory = inventory_receipt.inventory
    validated_receipts: list[MetaSynPacketValidationReceiptV1] = []
    by_index = {item.candidate_index: item for item in packet_receipts}
    if len(by_index) != len(packet_receipts):
        raise MetaSynBoundedAdapterError("metasyn_publication_packet_index_duplicate")
    candidate_indices = {item.candidate_index for item in inventory.candidates}
    if not set(by_index).issubset(candidate_indices):
        raise MetaSynBoundedAdapterError("metasyn_publication_packet_not_in_inventory")
    for candidate in inventory.candidates:
        receipt = by_index.get(candidate.candidate_index)
        if receipt is None:
            continue
        call = freeze_metasyn_packet_call(
            row=row,
            inventory_receipt=inventory_receipt,
            candidate_index=candidate.candidate_index,
        )
        validated_receipts.append(
            validate_metasyn_packet_validation_receipt(
                receipt=receipt,
                call=call,
                row=row,
                inventory_receipt=inventory_receipt,
            )
        )
    official: NativePublicationExtraction | None = None
    if inventory_receipt.status == "no_candidate_non_authorizing":
        if packet_receipts:
            raise MetaSynBoundedAdapterError(
                "metasyn_publication_non_authorizing_inventory_has_packets"
            )
        status = "abstained_inventory_no_candidate"
        reasons = ["inventory_no_candidate_is_not_nonestimability_proof"]
    elif inventory_receipt.status == "capacity_or_uncertainty_non_authorizing":
        if packet_receipts:
            raise MetaSynBoundedAdapterError(
                "metasyn_publication_non_authorizing_inventory_has_packets"
            )
        status = "abstained_inventory_uncertain"
        reasons = ["inventory_capacity_or_uncertainty_non_authorizing"]
    elif len(validated_receipts) != len(inventory.candidates):
        status = "abstained_packet_set_incomplete"
        reasons = ["publication_packet_membership_incomplete"]
    elif any(item.packet_status == "unable_to_complete" for item in validated_receipts):
        status = "abstained_packet_unable"
        reasons = ["at_least_one_candidate_packet_unable_to_complete"]
    else:
        candidates_by_index = {
            item.candidate_index: item for item in inventory.candidates
        }
        packet_values = [
            validate_packet_for_candidate(
                item.packet_payload,
                candidate=candidates_by_index[item.candidate_index],
                exposed_line_ids=row.source_row.projection.exposed_line_ids,
                source_locator=row.source_locator,
                allowed_outcomes=row.allowed_outcomes,
                allowed_moderators=row.allowed_moderators,
                allowed_sections=row.allowed_sections,
                outcome_positive_directions=row.outcome_positive_directions,
            )
            for item in sorted(
                validated_receipts, key=lambda item: item.candidate_index
            )
        ]
        official = assemble_candidate_packets(
            inventory=inventory,
            packets=packet_values,
            exposed_line_ids=row.source_row.projection.exposed_line_ids,
            source_locator=row.source_locator,
            allowed_outcomes=row.allowed_outcomes,
            allowed_moderators=row.allowed_moderators,
            allowed_sections=row.allowed_sections,
            outcome_positive_directions=row.outcome_positive_directions,
        )
        status = "typed_publication_output"
        reasons = []
    validated_receipts.sort(key=lambda item: item.candidate_index)
    payload: dict[str, Any] = {
        "result_version": PUBLICATION_RESULT_VERSION,
        "row_context_sha256": row.row_context_sha256,
        "inventory_receipt": inventory_receipt,
        "inventory_receipt_sha256": inventory_receipt.receipt_sha256,
        "packet_receipts": validated_receipts,
        "packet_receipt_sha256s": sorted(
            item.receipt_sha256 for item in validated_receipts
        ),
        "status": status,
        "blocking_reasons": sorted(reasons),
        "official_output": official,
        "official_output_sha256": hash_canonical(official) if official is not None else None,
    }
    return MetaSynPublicationResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def validate_metasyn_publication_result(
    *,
    result: MetaSynPublicationResultV1 | Mapping[str, Any],
    row: MetaSynBoundedRowContextV1,
) -> MetaSynPublicationResultV1:
    canonical = MetaSynPublicationResultV1.model_validate(result)
    replayed = freeze_metasyn_publication_result(
        row=row,
        inventory_receipt=canonical.inventory_receipt,
        packet_receipts=canonical.packet_receipts,
    )
    if replayed != canonical:
        raise MetaSynBoundedAdapterError(
            "metasyn_publication_result_external_replay_mismatch"
        )
    return canonical


class MetaSynBoundedPrivateYieldReportV1(ContractModel):
    report_version: Literal["metasyn-bounded-private-yield-report-v1"] = (
        YIELD_REPORT_VERSION
    )
    adapter_bundle_sha256: str
    row_membership_sha256: str
    publication_results: Annotated[list[MetaSynPublicationResultV1], Field(min_length=1)]
    publication_result_sha256s: list[str]
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    inventory_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    typed_publication_output_count: Annotated[int, Field(ge=0)]
    release_grade_typed_publication_count: Annotated[int, Field(ge=0)]
    diagnostic_only_typed_publication_count: Annotated[int, Field(ge=0)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    questions_with_any_release_grade_typed_publication: Annotated[int, Field(ge=0)]
    synthesis_attempt_input_publication_count: Annotated[int, Field(ge=0)]
    reference_fields_unopened: Literal[True] = True
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    claim_release_authority: Literal[False] = False
    synthesis_input_caveat: Literal[
        "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_correctness"
    ] = "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_correctness"
    report_sha256: str

    @field_validator(
        "adapter_bundle_sha256",
        "row_membership_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_yield_sha256_invalid:{info.field_name}")
        return value

    @field_validator("publication_result_sha256s")
    @classmethod
    def validate_result_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not SHA256_RE.fullmatch(item) for item in value):
            raise ValueError("metasyn_yield_result_hashes_invalid")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynBoundedPrivateYieldReportV1:
        expected_hashes = sorted(item.result_sha256 for item in self.publication_results)
        if self.publication_result_sha256s != expected_hashes:
            raise ValueError("metasyn_yield_result_hash_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("metasyn_yield_report_hash_mismatch")
        return self


def freeze_metasyn_bounded_private_yield_report(
    *,
    adapter_bundle: MetaSynBoundedAdapterBundleV1,
    publication_results: Sequence[MetaSynPublicationResultV1],
) -> MetaSynBoundedPrivateYieldReportV1:
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    rows_by_hash = {row.row_context_sha256: row for row in adapter.row_contexts}
    results_by_row: dict[str, MetaSynPublicationResultV1] = {}
    for raw in publication_results:
        result = MetaSynPublicationResultV1.model_validate(raw)
        row = rows_by_hash.get(result.row_context_sha256)
        if row is None or result.row_context_sha256 in results_by_row:
            raise MetaSynBoundedAdapterError("metasyn_yield_result_roster_mismatch")
        replayed = freeze_metasyn_publication_result(
            row=row,
            inventory_receipt=result.inventory_receipt,
            packet_receipts=result.packet_receipts,
        )
        if replayed != result:
            raise MetaSynBoundedAdapterError("metasyn_yield_publication_replay_mismatch")
        results_by_row[result.row_context_sha256] = result
    if set(results_by_row) != set(rows_by_hash):
        raise MetaSynBoundedAdapterError("metasyn_yield_publication_roster_incomplete")
    ordered = [results_by_row[row.row_context_sha256] for row in adapter.row_contexts]
    inventory_counts = Counter(result.inventory_receipt.status for result in ordered)
    packet_counts = Counter(
        receipt.packet_status for result in ordered for receipt in result.packet_receipts
    )
    typed_rows = [
        rows_by_hash[result.row_context_sha256]
        for result in ordered
        if result.status == "typed_publication_output"
    ]
    release_grade_rows = [
        row
        for row in typed_rows
        if row.source_row.release_grade_source_grounding_eligible
    ]
    questions_with_release_grade = {
        row.question_spec.question_id for row in release_grade_rows
    }
    typed_findings = sum(
        len(cohort.findings)
        for result in ordered
        if result.official_output is not None
        for study in result.official_output.studies
        for cohort in study.cohorts
    )
    payload: dict[str, Any] = {
        "report_version": YIELD_REPORT_VERSION,
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "row_membership_sha256": adapter.row_membership_sha256,
        "publication_results": ordered,
        "publication_result_sha256s": sorted(item.result_sha256 for item in ordered),
        "question_count": adapter.question_count,
        "publication_count": adapter.publication_count,
        "inventory_status_counts": dict(sorted(inventory_counts.items())),
        "packet_status_counts": dict(sorted(packet_counts.items())),
        "typed_publication_output_count": len(typed_rows),
        "release_grade_typed_publication_count": len(release_grade_rows),
        "diagnostic_only_typed_publication_count": len(typed_rows)
        - len(release_grade_rows),
        "typed_finding_count": typed_findings,
        "questions_with_any_release_grade_typed_publication": len(
            questions_with_release_grade
        ),
        "synthesis_attempt_input_publication_count": len(release_grade_rows),
        "reference_fields_unopened": True,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "synthesis_input_caveat": (
            "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_"
            "correctness"
        ),
    }
    return MetaSynBoundedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def validate_metasyn_bounded_private_yield_report(
    *,
    report: MetaSynBoundedPrivateYieldReportV1 | Mapping[str, Any],
    adapter_bundle: MetaSynBoundedAdapterBundleV1,
) -> MetaSynBoundedPrivateYieldReportV1:
    """Replay every publication result and every yield-only aggregate."""

    canonical = MetaSynBoundedPrivateYieldReportV1.model_validate(report)
    replayed = freeze_metasyn_bounded_private_yield_report(
        adapter_bundle=adapter_bundle,
        publication_results=canonical.publication_results,
    )
    if replayed != canonical:
        raise MetaSynBoundedAdapterError("metasyn_yield_report_external_replay_mismatch")
    return canonical


__all__ = [
    "MetaSynBoundedAdapterBundleV1",
    "MetaSynBoundedAdapterError",
    "MetaSynBoundedPrivateYieldReportV1",
    "MetaSynBoundedRowContextV1",
    "MetaSynInventoryValidationReceiptV1",
    "MetaSynPacketCallV1",
    "MetaSynPacketValidationReceiptV1",
    "MetaSynPublicationResultV1",
    "MetaSynUniqueQuoteGroundingV1",
    "compute_metasyn_bounded_adapter_fingerprint",
    "freeze_metasyn_bounded_adapter_bundle",
    "freeze_metasyn_bounded_adapter_bundle_from_workspace",
    "freeze_metasyn_bounded_private_yield_report",
    "freeze_metasyn_bounded_row_context",
    "freeze_metasyn_inventory_validation_receipt",
    "freeze_metasyn_packet_call",
    "freeze_metasyn_packet_validation_receipt",
    "freeze_metasyn_publication_result",
    "validate_metasyn_bounded_adapter_bundle_external_replay",
    "validate_metasyn_bounded_private_yield_report",
    "validate_metasyn_inventory_validation_receipt",
    "validate_metasyn_packet_validation_receipt",
    "validate_metasyn_publication_result",
]
