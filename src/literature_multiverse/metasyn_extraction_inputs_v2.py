"""Provider-neutral MetaSyn v2 extraction inputs over the immutable v5 source surface.

This module creates no provider client and writes no runtime artifacts.  It externally
replays :class:`MetaSynV5SourceSurfaceV1`, builds one passage-anchored projection-v2
for every one of its 32 rows, and freezes exact inventory prompts and schemas.  A
separate candidate binder accepts only a validated successor inventory receipt and
produces one packet prompt/schema bound to that candidate's p2 passages.

Only extraction-relevant protocol fields are model-facing.  V5 hosted receipts, row
results, provider outputs, reference fields, test labels, conclusions, and benchmark
effect results are not inputs to this component.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    MetaSynPassageCandidateV2,
    metasyn_candidate_inventory_schema_bundle_v2,
    validate_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_projection_v2 import (
    FrozenMetaSynProjectionV2,
    freeze_metasyn_projection_v2,
    freeze_projection_v2_lineage_binding,
    validate_metasyn_projection_v2_external_replay,
)
from literature_multiverse.metasyn_v5_source_surface import (
    EXPECTED_V5_EXECUTION_BUNDLE_SHA256,
    MetaSynV5SourceSurfaceRowV1,
    MetaSynV5SourceSurfaceV1,
    freeze_metasyn_v5_source_surface,
    validate_metasyn_v5_source_surface,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_packet_grounding_v2 import (
    PacketGroundingSchemaBundleV2,
    PacketPassageCandidateBindingV2,
    freeze_packet_grounding_schema_bundle_v2,
    freeze_passage_packet_candidate_binding_v2,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

EXTRACTION_INPUTS_VERSION = "metasyn-provider-neutral-extraction-inputs-v2"
EXTRACTION_ROW_INPUT_VERSION = "metasyn-provider-neutral-extraction-row-v2"
QUESTION_SURFACE_VERSION = "metasyn-extraction-question-surface-v2"
SOURCE_STRENGTH_SURFACE_VERSION = "metasyn-source-strength-surface-v2"
PASSAGE_SURFACE_VERSION = "metasyn-model-passage-surface-v2"
PROJECTION_SURFACE_VERSION = "metasyn-model-projection-surface-v2"
PROMPT_BINDING_VERSION = "metasyn-prompt-template-binding-v2"
INVENTORY_INPUT_VERSION = "metasyn-inventory-input-surface-v2"
CANDIDATE_PASSAGE_SURFACE_VERSION = "metasyn-candidate-passage-surface-v2"
PACKET_INPUT_VERSION = "metasyn-packet-candidate-input-v2"
EXTRACTION_INPUTS_COMPONENT_VERSION = "1"

INVENTORY_PROMPT_PATH = Path("prompts/metasyn_candidate_inventory_v2.md")
PACKET_PROMPT_PATH = Path("prompts/metasyn_candidate_packet_v2.md")
INVENTORY_PROMPT_VERSION = "metasyn-passage-candidate-inventory-v2"
PACKET_PROMPT_VERSION = "metasyn-passage-candidate-packet-v2"

EXPECTED_PUBLICATION_COUNT = 32
EXPECTED_QUESTION_COUNT = 10
EXPECTED_COMPONENT_COUNT = 10
MAX_PROMPT_CHARACTERS = 100_000

QUESTION_PROTOCOL_FIELD_WHITELIST: tuple[str, ...] = (
    "allowed_outcomes",
    "comparator_role",
    "comparison",
    "contrast_estimand",
    "contrast_orientation",
    "exclusion_criteria",
    "inclusion_criteria",
    "intervention_or_exposure",
    "outcome_id_to_text",
    "population",
    "positive_direction_means_by_outcome_id",
    "question_id",
    "question_spec_sha256",
    "relation_kind",
    "research_question",
    "treatment_role",
)

FORBIDDEN_MODEL_FACING_FIELD_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            "abstract",
            "ci_lower",
            "ci_upper",
            "conclusion",
            "conclusion_paragraph",
            "conclusion_summary",
            "effect_direction",
            "effect_size_category",
            "effect_size_type",
            "effect_size_value",
            "heterogeneity_level",
            "i2_value",
            "key_insights",
            "official_test",
            "official_test_label",
            "official_test_labels",
            "p_value",
            "provider_output",
            "provider_outputs",
            "q_value",
            "reference_field",
            "reference_fields",
            "result_label",
            "row_result",
            "row_results",
            "statistical_significance",
            "tau2_value",
            "test_label",
            "test_labels",
        }
    )
)

_INVENTORY_PROMPT_TOKENS = ("PROJECTION_V2_JSON", "QUESTION_SPEC_JSON")
_PACKET_PROMPT_TOKENS = (
    "CANDIDATE_BINDING_JSON",
    "CANDIDATE_PASSAGE_SURFACE_JSON",
    "PROJECTION_V2_JSON",
    "QUESTION_SURFACE_JSON",
)
_PROMPT_TOKEN_RE = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")
_EXTRACTION_INPUTS_ENTRYPOINTS = ("src/literature_multiverse/metasyn_extraction_inputs_v2.py",)
_EXTRACTION_INPUTS_NON_PYTHON_FILES = (
    INVENTORY_PROMPT_PATH.as_posix(),
    PACKET_PROMPT_PATH.as_posix(),
    "pyproject.toml",
    "uv.lock",
)
_INSTALLED_DEPENDENCIES = ("anthropic", "jsonschema", "pyarrow", "pydantic")


class MetaSynExtractionInputsV2Error(ValueError):
    """A successor input, prompt, schema, or replay boundary failed closed."""


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_extraction_inputs_v2_hash_invalid:{field_name}")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode):
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_repository_root_symlink_forbidden"
        )
    if not resolved.is_dir():
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_repository_root_not_directory"
        )
    return resolved


def _checked_repository_file(*, root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative_path.startswith("./")
        or relative.as_posix() != relative_path
    ):
        raise MetaSynExtractionInputsV2Error("metasyn_extraction_inputs_v2_file_path_unsafe")
    candidate = root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise MetaSynExtractionInputsV2Error(
                f"metasyn_extraction_inputs_v2_file_missing:{relative_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynExtractionInputsV2Error(
                f"metasyn_extraction_inputs_v2_file_symlink_forbidden:{relative_path}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MetaSynExtractionInputsV2Error(
            f"metasyn_extraction_inputs_v2_file_missing:{relative_path}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise MetaSynExtractionInputsV2Error(
            f"metasyn_extraction_inputs_v2_file_outside_repository:{relative_path}"
        )
    if not resolved.is_file():
        raise MetaSynExtractionInputsV2Error(
            f"metasyn_extraction_inputs_v2_path_not_file:{relative_path}"
        )
    return resolved


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _reject_forbidden_model_facing_fields(value: Any, *, location: str) -> None:
    forbidden = set(FORBIDDEN_MODEL_FACING_FIELD_NAMES)
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(str(key))
            if normalized in forbidden:
                raise MetaSynExtractionInputsV2Error(
                    f"metasyn_extraction_inputs_v2_forbidden_model_field:{location}:{key}"
                )
            _reject_forbidden_model_facing_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_model_facing_fields(child, location=f"{location}[{index}]")


def _prompt_requirements(
    kind: Literal["inventory", "packet"],
) -> tuple[str, tuple[str, ...], str]:
    if kind == "inventory":
        return (
            INVENTORY_PROMPT_VERSION,
            _INVENTORY_PROMPT_TOKENS,
            INVENTORY_PROMPT_PATH.as_posix(),
        )
    return PACKET_PROMPT_VERSION, _PACKET_PROMPT_TOKENS, PACKET_PROMPT_PATH.as_posix()


def _render_prompt(template: str, replacements: Mapping[str, Any]) -> str:
    observed = sorted(set(_PROMPT_TOKEN_RE.findall(template)))
    if observed != sorted(replacements):
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_prompt_token_contract_mismatch"
        )
    rendered = template
    for token in sorted(replacements):
        rendered = rendered.replace(f"[[{token}]]", _canonical_json(replacements[token]))
    rendered = rendered.strip()
    if _PROMPT_TOKEN_RE.search(rendered):
        raise MetaSynExtractionInputsV2Error("metasyn_extraction_inputs_v2_prompt_token_unresolved")
    if not rendered or len(rendered) > MAX_PROMPT_CHARACTERS:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_rendered_prompt_size_invalid"
        )
    return rendered


class _ExactContractModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


class MetaSynPromptTemplateBindingV2(_ExactContractModel):
    binding_version: Literal["metasyn-prompt-template-binding-v2"] = PROMPT_BINDING_VERSION
    prompt_kind: Literal["inventory", "packet"]
    prompt_version: str
    prompt_path: Annotated[str, Field(min_length=1, max_length=2048)]
    prompt_file_sha256: str
    prompt_template: Annotated[str, Field(min_length=1, max_length=32_000)]
    prompt_template_sha256: str
    required_tokens: list[str]
    prompt_binding_sha256: str

    @field_validator("prompt_file_sha256", "prompt_template_sha256", "prompt_binding_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("required_tokens")
    @classmethod
    def validate_tokens(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_extraction_inputs_v2_prompt_tokens_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> MetaSynPromptTemplateBindingV2:
        expected_version, expected_tokens, expected_path = _prompt_requirements(self.prompt_kind)
        if (
            self.prompt_version != expected_version
            or self.prompt_path != expected_path
            or self.required_tokens != sorted(expected_tokens)
        ):
            raise ValueError("metasyn_extraction_inputs_v2_prompt_contract_mismatch")
        marker = f"Prompt version: `{self.prompt_version}`"
        if marker not in self.prompt_template:
            raise ValueError("metasyn_extraction_inputs_v2_prompt_version_marker_missing")
        if sorted(set(_PROMPT_TOKEN_RE.findall(self.prompt_template))) != (self.required_tokens):
            raise ValueError("metasyn_extraction_inputs_v2_prompt_template_tokens_mismatch")
        if _sha256_text(self.prompt_template) != self.prompt_template_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_prompt_template_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"prompt_binding_sha256"})
        if hash_canonical(payload) != self.prompt_binding_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_prompt_binding_hash_mismatch")
        return self


def _freeze_prompt_binding(
    *, root: Path, kind: Literal["inventory", "packet"]
) -> MetaSynPromptTemplateBindingV2:
    version, tokens, relative = _prompt_requirements(kind)
    path = _checked_repository_file(root=root, relative_path=relative)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MetaSynExtractionInputsV2Error(
            f"metasyn_extraction_inputs_v2_prompt_unreadable:{relative}"
        ) from exc
    template = raw.strip()
    payload = {
        "binding_version": PROMPT_BINDING_VERSION,
        "prompt_kind": kind,
        "prompt_version": version,
        "prompt_path": relative,
        "prompt_file_sha256": sha256_file(path),
        "prompt_template": template,
        "prompt_template_sha256": _sha256_text(template),
        "required_tokens": sorted(tokens),
    }
    return MetaSynPromptTemplateBindingV2.model_validate(
        {**payload, "prompt_binding_sha256": hash_canonical(payload)}
    )


class MetaSynExtractionQuestionSurfaceV2(ContractModel):
    question_surface_version: Literal["metasyn-extraction-question-surface-v2"] = (
        QUESTION_SURFACE_VERSION
    )
    question_id: Annotated[str, Field(min_length=1, max_length=128)]
    question_spec_sha256: str
    research_question: Annotated[str, Field(min_length=1, max_length=4096)]
    population: Annotated[str, Field(min_length=1, max_length=4096)]
    relation_kind: Literal["intervention", "exposure"]
    intervention_or_exposure: Annotated[str, Field(min_length=1, max_length=4096)]
    comparison: Annotated[str, Field(min_length=1, max_length=4096)]
    treatment_role: Literal["intervention_or_exposure"]
    comparator_role: Literal["comparator"]
    contrast_orientation: Literal["intervention_or_exposure_minus_comparator"]
    contrast_estimand: Literal[
        "between_group_effect_intervention_or_exposure_vs_comparator_on_reported_measure"
    ]
    inclusion_criteria: Annotated[str, Field(max_length=16_384)] | None
    exclusion_criteria: Annotated[str, Field(max_length=16_384)] | None
    allowed_outcome_ids: Annotated[list[str], Field(min_length=1, max_length=16)]
    allowed_outcome_text_by_id: dict[str, Annotated[str, Field(min_length=1)]]
    raw_positive_direction_meaning_by_outcome_id: dict[str, Annotated[str, Field(min_length=1)]]
    outcome_membership_sha256: str
    question_surface_sha256: str

    @field_validator(
        "question_spec_sha256",
        "outcome_membership_sha256",
        "question_surface_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("allowed_outcome_ids")
    @classmethod
    def validate_outcome_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_extraction_inputs_v2_outcome_ids_not_canonical")
        return value

    @field_validator(
        "allowed_outcome_text_by_id",
        "raw_positive_direction_meaning_by_outcome_id",
    )
    @classmethod
    def validate_maps(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())):
            raise ValueError("metasyn_extraction_inputs_v2_question_map_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynExtractionQuestionSurfaceV2:
        if set(self.allowed_outcome_text_by_id) != set(self.allowed_outcome_ids):
            raise ValueError("metasyn_extraction_inputs_v2_outcome_text_membership_mismatch")
        if set(self.raw_positive_direction_meaning_by_outcome_id) != set(self.allowed_outcome_ids):
            raise ValueError("metasyn_extraction_inputs_v2_outcome_direction_membership_mismatch")
        if hash_canonical(self.allowed_outcome_text_by_id) != (self.outcome_membership_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_outcome_membership_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"question_surface_sha256"})
        _reject_forbidden_model_facing_fields(payload, location="question_surface")
        if hash_canonical(payload) != self.question_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_question_surface_hash_mismatch")
        return self


class MetaSynSourceStrengthSurfaceV2(ContractModel):
    source_strength_surface_version: Literal["metasyn-source-strength-surface-v2"] = (
        SOURCE_STRENGTH_SURFACE_VERSION
    )
    source_content_scope: Literal["full_text_sections", "title_abstract"]
    oracle_selection_full_text_scope: bool
    source_projection_strength: Literal[
        "full_text_textual_grounding",
        "diagnostic_title_abstract_grounding",
        "diagnostic_unrecognized_sections",
        "no_eligible_source_passage",
    ]
    release_grade_source_grounding_eligible: bool
    source_strength_blockers: list[str]
    projection_v2_selection_complete: bool
    projection_v2_omitted_passage_count: Annotated[int, Field(ge=0)]
    projection_v2_omitted_source_characters: Annotated[int, Field(ge=0)]
    source_strength_surface_sha256: str

    @field_validator("source_strength_surface_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "source_strength_surface_sha256")

    @field_validator("source_strength_blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_extraction_inputs_v2_source_blockers_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynSourceStrengthSurfaceV2:
        if self.oracle_selection_full_text_scope != (
            self.source_content_scope == "full_text_sections"
        ):
            raise ValueError("metasyn_extraction_inputs_v2_source_scope_alias_mismatch")
        if self.projection_v2_selection_complete != (self.projection_v2_omitted_passage_count == 0):
            raise ValueError("metasyn_extraction_inputs_v2_projection_selection_complete_mismatch")
        payload = self.model_dump(mode="json", exclude={"source_strength_surface_sha256"})
        if hash_canonical(payload) != self.source_strength_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_source_strength_hash_mismatch")
        return self


class MetaSynModelPassageSurfaceV2(_ExactContractModel):
    passage_surface_version: Literal["metasyn-model-passage-surface-v2"] = PASSAGE_SURFACE_VERSION
    passage_id: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    prompt_rank: Annotated[int, Field(ge=1)]
    passage_lineage_sha256: str
    section_enums: list[str]
    upstream_line_ids: list[Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]]
    exact_source_occurrence_count: Annotated[int, Field(ge=1)]
    text: Annotated[str, Field(min_length=1, max_length=512)]
    text_sha256: str
    passage_surface_sha256: str

    @field_validator("passage_lineage_sha256", "text_sha256", "passage_surface_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("section_enums", "upstream_line_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if not value or value != sorted(set(value)):
            raise ValueError(
                f"metasyn_extraction_inputs_v2_passage_values_not_canonical:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynModelPassageSurfaceV2:
        if _sha256_text(self.text) != self.text_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_passage_text_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"passage_surface_sha256"})
        if hash_canonical(payload) != self.passage_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_passage_surface_hash_mismatch")
        return self


class MetaSynModelProjectionSurfaceV2(_ExactContractModel):
    projection_surface_version: Literal["metasyn-model-projection-surface-v2"] = (
        PROJECTION_SURFACE_VERSION
    )
    projection_v2_sha256: str
    projection_prompt_source_sha256: str
    source_strength: MetaSynSourceStrengthSurfaceV2
    source_strength_surface_sha256: str
    selected_passage_count: Annotated[int, Field(ge=1)]
    omitted_passage_count: Annotated[int, Field(ge=0)]
    selection_complete: bool
    passage_ids: Annotated[list[str], Field(min_length=1)]
    passages: Annotated[list[MetaSynModelPassageSurfaceV2], Field(min_length=1)]
    passage_membership_sha256: str
    projection_surface_sha256: str

    @field_validator(
        "projection_v2_sha256",
        "projection_prompt_source_sha256",
        "source_strength_surface_sha256",
        "passage_membership_sha256",
        "projection_surface_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("passage_ids")
    @classmethod
    def validate_passage_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("metasyn_extraction_inputs_v2_passage_ids_not_unique")
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynModelProjectionSurfaceV2:
        if self.source_strength_surface_sha256 != (
            self.source_strength.source_strength_surface_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_source_strength_alias_mismatch")
        if self.passages != sorted(self.passages, key=lambda item: item.prompt_rank):
            raise ValueError("metasyn_extraction_inputs_v2_passages_not_in_prompt_order")
        if [item.prompt_rank for item in self.passages] != list(range(1, len(self.passages) + 1)):
            raise ValueError("metasyn_extraction_inputs_v2_passage_prompt_ranks_invalid")
        if self.passage_ids != [item.passage_id for item in self.passages]:
            raise ValueError("metasyn_extraction_inputs_v2_passage_id_alias_mismatch")
        if self.selected_passage_count != len(self.passages):
            raise ValueError("metasyn_extraction_inputs_v2_selected_passage_count_mismatch")
        if self.selection_complete != (self.omitted_passage_count == 0):
            raise ValueError("metasyn_extraction_inputs_v2_selection_complete_mismatch")
        membership = [
            item.model_dump(mode="json")
            for item in sorted(self.passages, key=lambda passage: passage.passage_id)
        ]
        if hash_canonical(membership) != self.passage_membership_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_passage_membership_mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_surface_sha256"})
        _reject_forbidden_model_facing_fields(payload, location="projection_surface")
        if hash_canonical(payload) != self.projection_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_projection_surface_hash_mismatch")
        return self


def _validate_inventory_schema_bundle(
    value: Mapping[str, Any],
    *,
    allowed_outcome_ids: list[str],
    passage_ids: list[str],
) -> dict[str, Any]:
    canonical = json.loads(_canonical_json(dict(value)))
    expected = metasyn_candidate_inventory_schema_bundle_v2(
        allowed_outcome_ids=allowed_outcome_ids,
        passage_ids=passage_ids,
    )
    if canonical != expected:
        raise ValueError("metasyn_extraction_inputs_v2_inventory_schema_replay_mismatch")
    try:
        for key in ("provider_schema", "full_acceptance_schema"):
            schema = canonical[key]
            validator_for(schema).check_schema(schema)
            validator_for(schema)(schema).validate(canonical["valid_example"])
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise ValueError("metasyn_extraction_inputs_v2_inventory_schema_invalid") from exc
    return canonical


class MetaSynInventoryInputSurfaceV2(_ExactContractModel):
    inventory_input_version: Literal["metasyn-inventory-input-surface-v2"] = INVENTORY_INPUT_VERSION
    prompt_binding_sha256: str
    question_surface_sha256: str
    projection_surface_sha256: str
    projection_v2_sha256: str
    rendered_prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)]
    rendered_prompt_sha256: str
    rendered_prompt_characters: Annotated[int, Field(ge=1)]
    inventory_schema_bundle: dict[str, Any]
    inventory_schema_bundle_sha256: str
    inventory_input_sha256: str

    @field_validator(
        "prompt_binding_sha256",
        "question_surface_sha256",
        "projection_surface_sha256",
        "projection_v2_sha256",
        "rendered_prompt_sha256",
        "inventory_schema_bundle_sha256",
        "inventory_input_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_input(self) -> MetaSynInventoryInputSurfaceV2:
        if (
            _sha256_text(self.rendered_prompt) != self.rendered_prompt_sha256
            or len(self.rendered_prompt) != self.rendered_prompt_characters
        ):
            raise ValueError("metasyn_extraction_inputs_v2_inventory_prompt_hash_mismatch")
        if self.inventory_schema_bundle.get("schema_bundle_sha256") != (
            self.inventory_schema_bundle_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_inventory_schema_hash_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"inventory_input_sha256"})
        if hash_canonical(payload) != self.inventory_input_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_inventory_input_hash_mismatch")
        return self


class MetaSynExtractionRowInputV2(_ExactContractModel):
    row_input_version: Literal["metasyn-provider-neutral-extraction-row-v2"] = (
        EXTRACTION_ROW_INPUT_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: Annotated[str, Field(min_length=1, max_length=512)]
    upstream_source_surface_row_sha256: str
    upstream_row_context_sha256: str
    upstream_question_bundle_sha256: str
    upstream_question_spec_sha256: str
    upstream_component_binding_sha256: str
    upstream_source_row_sha256: str
    upstream_projection_sha256: str
    upstream_artifact_binding_sha256: str
    question_surface: MetaSynExtractionQuestionSurfaceV2
    question_surface_sha256: str
    source_strength: MetaSynSourceStrengthSurfaceV2
    source_strength_surface_sha256: str
    projection_v2: FrozenMetaSynProjectionV2
    projection_v2_sha256: str
    projection_surface: MetaSynModelProjectionSurfaceV2
    projection_surface_sha256: str
    inventory_input: MetaSynInventoryInputSurfaceV2
    inventory_input_sha256: str
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    v5_hosted_outputs_consumed: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    row_input_sha256: str

    @field_validator(
        "upstream_source_surface_row_sha256",
        "upstream_row_context_sha256",
        "upstream_question_bundle_sha256",
        "upstream_question_spec_sha256",
        "upstream_component_binding_sha256",
        "upstream_source_row_sha256",
        "upstream_projection_sha256",
        "upstream_artifact_binding_sha256",
        "question_surface_sha256",
        "source_strength_surface_sha256",
        "projection_v2_sha256",
        "projection_surface_sha256",
        "inventory_input_sha256",
        "row_input_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_row(self) -> MetaSynExtractionRowInputV2:
        aliases = {
            "question_surface_sha256": self.question_surface.question_surface_sha256,
            "source_strength_surface_sha256": (self.source_strength.source_strength_surface_sha256),
            "projection_v2_sha256": self.projection_v2.projection_sha256,
            "projection_surface_sha256": (self.projection_surface.projection_surface_sha256),
            "inventory_input_sha256": self.inventory_input.inventory_input_sha256,
        }
        if any(getattr(self, field) != expected for field, expected in aliases.items()):
            raise ValueError("metasyn_extraction_inputs_v2_row_hash_alias_mismatch")
        if self.question_surface.question_spec_sha256 != (self.upstream_question_spec_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_question_lineage_mismatch")
        if self.projection_v2.lineage_binding.upstream_row_context_sha256 != (
            self.upstream_row_context_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_row_context_lineage_mismatch")
        if self.projection_v2.lineage_binding.upstream_source_row_sha256 != (
            self.upstream_source_row_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_source_row_lineage_mismatch")
        if self.projection_v2.lineage_binding.upstream_projection_sha256 != (
            self.upstream_projection_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_projection_lineage_mismatch")
        if self.projection_v2.question_id != self.question_surface.question_id:
            raise ValueError("metasyn_extraction_inputs_v2_projection_question_mismatch")
        if self.projection_surface.projection_v2_sha256 != self.projection_v2_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_projection_surface_alias_mismatch")
        if self.projection_surface.source_strength != self.source_strength:
            raise ValueError("metasyn_extraction_inputs_v2_source_strength_surface_mismatch")
        if self.inventory_input.question_surface_sha256 != self.question_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_inventory_question_mismatch")
        if self.inventory_input.projection_surface_sha256 != self.projection_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_inventory_projection_mismatch")
        if self.inventory_input.projection_v2_sha256 != self.projection_v2_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_inventory_projection_hash_mismatch")
        _validate_inventory_schema_bundle(
            self.inventory_input.inventory_schema_bundle,
            allowed_outcome_ids=self.question_surface.allowed_outcome_ids,
            passage_ids=sorted(self.projection_surface.passage_ids),
        )
        payload = self.model_dump(mode="json", exclude={"row_input_sha256"})
        if hash_canonical(payload) != self.row_input_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_row_input_hash_mismatch")
        return self


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


def _extraction_inputs_python_dependency_closure(repository_root: Path) -> list[str]:
    root = _canonical_root(repository_root)
    pending = list(_EXTRACTION_INPUTS_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        path = _checked_repository_file(root=root, relative_path=relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynExtractionInputsV2Error(
                f"metasyn_extraction_inputs_v2_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def _installed_dependency_versions() -> dict[str, str]:
    return {dependency: distribution_version(dependency) for dependency in _INSTALLED_DEPENDENCIES}


def _pipeline_settings(
    *,
    source_surface: MetaSynV5SourceSurfaceV1,
    inventory_prompt_binding: MetaSynPromptTemplateBindingV2,
    packet_prompt_binding: MetaSynPromptTemplateBindingV2,
) -> dict[str, Any]:
    return {
        "all_32_rows_required": True,
        "installed_dependency_versions": _installed_dependency_versions(),
        "inventory_prompt_binding_sha256": (inventory_prompt_binding.prompt_binding_sha256),
        "official_test_labels_opened": False,
        "packet_prompt_binding_sha256": packet_prompt_binding.prompt_binding_sha256,
        "provider_calls_permitted": False,
        "question_protocol_field_whitelist": list(QUESTION_PROTOCOL_FIELD_WHITELIST),
        "reference_fields_unopened": True,
        "source_surface_external_replay_required": True,
        "source_surface_pipeline_sha256": source_surface.source_surface_pipeline_sha256,
        "source_surface_sha256": source_surface.source_surface_sha256,
        "v5_execution_bundle_sha256": source_surface.v5_execution_bundle_sha256,
        "v5_hosted_outputs_consumed": False,
    }


def compute_metasyn_extraction_inputs_v2_pipeline_fingerprint(
    *,
    source_surface: MetaSynV5SourceSurfaceV1,
    root: Path | None = None,
) -> PipelineFingerprint:
    repository_root = _canonical_root(root or Path(__file__).resolve().parents[2])
    canonical_surface = MetaSynV5SourceSurfaceV1.model_validate(
        source_surface.model_dump(mode="json")
    )
    inventory_binding = _freeze_prompt_binding(root=repository_root, kind="inventory")
    packet_binding = _freeze_prompt_binding(root=repository_root, kind="packet")
    files = sorted(
        set(_extraction_inputs_python_dependency_closure(repository_root))
        | set(_EXTRACTION_INPUTS_NON_PYTHON_FILES)
    )
    component = PipelineComponentSpec(
        component_id="metasyn-provider-neutral-extraction-inputs-v2",
        component_version=EXTRACTION_INPUTS_COMPONENT_VERSION,
        file_paths=files,
        settings=_pipeline_settings(
            source_surface=canonical_surface,
            inventory_prompt_binding=inventory_binding,
            packet_prompt_binding=packet_binding,
        ),
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


class MetaSynExtractionInputsV2(_ExactContractModel):
    extraction_inputs_version: Literal["metasyn-provider-neutral-extraction-inputs-v2"] = (
        EXTRACTION_INPUTS_VERSION
    )
    status: Literal["frozen_provider_neutral_inputs_no_provider_calls"] = (
        "frozen_provider_neutral_inputs_no_provider_calls"
    )
    upstream_source_surface_sha256: str
    upstream_source_surface_pipeline_sha256: str
    upstream_v5_execution_bundle_sha256: str
    upstream_v5_adapter_bundle_sha256: str
    upstream_v5_runtime_pipeline_sha256: str
    upstream_v5_question_membership_sha256: str
    upstream_v5_component_membership_sha256: str
    upstream_v5_row_membership_sha256: str
    upstream_source_artifact_membership_sha256: str
    extraction_inputs_pipeline_fingerprint: PipelineFingerprint
    extraction_inputs_pipeline_sha256: str
    inventory_prompt_binding: MetaSynPromptTemplateBindingV2
    inventory_prompt_binding_sha256: str
    packet_prompt_binding: MetaSynPromptTemplateBindingV2
    packet_prompt_binding_sha256: str
    question_protocol_field_whitelist: list[str]
    forbidden_model_facing_field_names: list[str]
    question_count: Literal[10] = EXPECTED_QUESTION_COUNT
    component_count: Literal[10] = EXPECTED_COMPONENT_COUNT
    publication_count: Literal[32] = EXPECTED_PUBLICATION_COUNT
    rows: Annotated[list[MetaSynExtractionRowInputV2], Field(min_length=32, max_length=32)]
    row_input_hash_membership_sha256: str
    projection_v2_membership_sha256: str
    question_surface_membership_sha256: str
    inventory_prompt_membership_sha256: str
    inventory_schema_membership_sha256: str
    upstream_v5_source_surface_consumed: Literal[True] = True
    upstream_v5_source_surface_external_replayed: Literal[True] = True
    direct_v5_hosted_execution_bundle_consumed: Literal[False] = False
    v5_hosted_call_receipts_consumed: Literal[False] = False
    v5_hosted_row_results_consumed: Literal[False] = False
    v5_hosted_provider_outputs_consumed: Literal[False] = False
    provider_calls_made: Literal[False] = False
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    extraction_inputs_sha256: str

    @field_validator(
        "upstream_source_surface_sha256",
        "upstream_source_surface_pipeline_sha256",
        "upstream_v5_execution_bundle_sha256",
        "upstream_v5_adapter_bundle_sha256",
        "upstream_v5_runtime_pipeline_sha256",
        "upstream_v5_question_membership_sha256",
        "upstream_v5_component_membership_sha256",
        "upstream_v5_row_membership_sha256",
        "upstream_source_artifact_membership_sha256",
        "extraction_inputs_pipeline_sha256",
        "inventory_prompt_binding_sha256",
        "packet_prompt_binding_sha256",
        "row_input_hash_membership_sha256",
        "projection_v2_membership_sha256",
        "question_surface_membership_sha256",
        "inventory_prompt_membership_sha256",
        "inventory_schema_membership_sha256",
        "extraction_inputs_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("question_protocol_field_whitelist", "forbidden_model_facing_field_names")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_extraction_inputs_v2_values_not_canonical:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynExtractionInputsV2:
        if self.upstream_v5_execution_bundle_sha256 != (EXPECTED_V5_EXECUTION_BUNDLE_SHA256):
            raise ValueError("metasyn_extraction_inputs_v2_v5_anchor_mismatch")
        if self.inventory_prompt_binding.prompt_kind != "inventory":
            raise ValueError("metasyn_extraction_inputs_v2_inventory_prompt_kind_mismatch")
        if self.packet_prompt_binding.prompt_kind != "packet":
            raise ValueError("metasyn_extraction_inputs_v2_packet_prompt_kind_mismatch")
        if self.inventory_prompt_binding_sha256 != (
            self.inventory_prompt_binding.prompt_binding_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_inventory_binding_alias_mismatch")
        if self.packet_prompt_binding_sha256 != (self.packet_prompt_binding.prompt_binding_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_packet_binding_alias_mismatch")
        if self.question_protocol_field_whitelist != sorted(QUESTION_PROTOCOL_FIELD_WHITELIST):
            raise ValueError("metasyn_extraction_inputs_v2_question_whitelist_mismatch")
        if self.forbidden_model_facing_field_names != list(FORBIDDEN_MODEL_FACING_FIELD_NAMES):
            raise ValueError("metasyn_extraction_inputs_v2_forbidden_fields_mismatch")

        fingerprint = self.extraction_inputs_pipeline_fingerprint
        if self.extraction_inputs_pipeline_sha256 != fingerprint.pipeline_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_pipeline_hash_alias_mismatch")
        if len(fingerprint.components) != 1:
            raise ValueError("metasyn_extraction_inputs_v2_pipeline_component_count_mismatch")
        component = fingerprint.components[0]
        expected_settings = {
            "all_32_rows_required": True,
            "installed_dependency_versions": _installed_dependency_versions(),
            "inventory_prompt_binding_sha256": self.inventory_prompt_binding_sha256,
            "official_test_labels_opened": False,
            "packet_prompt_binding_sha256": self.packet_prompt_binding_sha256,
            "provider_calls_permitted": False,
            "question_protocol_field_whitelist": list(QUESTION_PROTOCOL_FIELD_WHITELIST),
            "reference_fields_unopened": True,
            "source_surface_external_replay_required": True,
            "source_surface_pipeline_sha256": (self.upstream_source_surface_pipeline_sha256),
            "source_surface_sha256": self.upstream_source_surface_sha256,
            "v5_execution_bundle_sha256": self.upstream_v5_execution_bundle_sha256,
            "v5_hosted_outputs_consumed": False,
        }
        if (
            component.component_id != "metasyn-provider-neutral-extraction-inputs-v2"
            or component.component_version != EXTRACTION_INPUTS_COMPONENT_VERSION
            or component.settings != expected_settings
        ):
            raise ValueError("metasyn_extraction_inputs_v2_pipeline_component_mismatch")

        if len(self.rows) != self.publication_count:
            raise ValueError("metasyn_extraction_inputs_v2_publication_count_mismatch")
        if [row.row_ordinal for row in self.rows] != list(range(EXPECTED_PUBLICATION_COUNT)):
            raise ValueError("metasyn_extraction_inputs_v2_row_ordinals_mismatch")
        row_keys = [row.row_key for row in self.rows]
        if row_keys != sorted(set(row_keys)):
            raise ValueError("metasyn_extraction_inputs_v2_rows_not_canonical")
        if hash_canonical(row_keys) != self.upstream_v5_row_membership_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_upstream_row_membership_mismatch")
        question_ids = sorted({row.question_surface.question_id for row in self.rows})
        if (
            len(question_ids) != self.question_count
            or hash_canonical(question_ids) != self.upstream_v5_question_membership_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_question_membership_mismatch")
        expected_memberships = {
            "row_input_hash_membership_sha256": hash_canonical(
                [row.row_input_sha256 for row in self.rows]
            ),
            "projection_v2_membership_sha256": hash_canonical(
                [row.projection_v2_sha256 for row in self.rows]
            ),
            "question_surface_membership_sha256": hash_canonical(
                sorted({row.question_surface.question_surface_sha256 for row in self.rows})
            ),
            "inventory_prompt_membership_sha256": hash_canonical(
                [row.inventory_input.rendered_prompt_sha256 for row in self.rows]
            ),
            "inventory_schema_membership_sha256": hash_canonical(
                [row.inventory_input.inventory_schema_bundle_sha256 for row in self.rows]
            ),
        }
        if any(
            getattr(self, field) != expected for field, expected in expected_memberships.items()
        ):
            raise ValueError("metasyn_extraction_inputs_v2_membership_hash_mismatch")

        for row in self.rows:
            expected_prompt = _render_prompt(
                self.inventory_prompt_binding.prompt_template,
                {
                    "QUESTION_SPEC_JSON": row.question_surface.model_dump(mode="json"),
                    "PROJECTION_V2_JSON": row.projection_surface.model_dump(mode="json"),
                },
            )
            if expected_prompt != row.inventory_input.rendered_prompt:
                raise ValueError("metasyn_extraction_inputs_v2_inventory_prompt_replay_mismatch")
            if row.inventory_input.prompt_binding_sha256 != (self.inventory_prompt_binding_sha256):
                raise ValueError("metasyn_extraction_inputs_v2_inventory_prompt_binding_mismatch")

        payload = self.model_dump(mode="json", exclude={"extraction_inputs_sha256"})
        if hash_canonical(payload) != self.extraction_inputs_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_bundle_hash_mismatch")
        return self


def _freeze_question_surface(
    row: MetaSynV5SourceSurfaceRowV1,
) -> MetaSynExtractionQuestionSurfaceV2:
    question = row.question_spec
    accessed = {
        "allowed_outcomes",
        "comparison",
        "contrast_estimand",
        "contrast_orientation",
        "exclusion_criteria",
        "inclusion_criteria",
        "intervention_or_exposure",
        "outcome_id_to_text",
        "population",
        "positive_direction_means_by_outcome_id",
        "question_id",
        "question_spec_sha256",
        "relation_kind",
        "research_question",
        "treatment_role",
        "comparator_role",
    }
    if accessed != set(QUESTION_PROTOCOL_FIELD_WHITELIST):
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_question_projection_not_whitelisted"
        )
    payload = {
        "question_surface_version": QUESTION_SURFACE_VERSION,
        "question_id": question.question_id,
        "question_spec_sha256": question.question_spec_sha256,
        "research_question": question.research_question,
        "population": question.population,
        "relation_kind": question.relation_kind,
        "intervention_or_exposure": question.intervention_or_exposure,
        "comparison": question.comparison,
        "treatment_role": question.treatment_role,
        "comparator_role": question.comparator_role,
        "contrast_orientation": question.contrast_orientation,
        "contrast_estimand": question.contrast_estimand,
        "inclusion_criteria": question.inclusion_criteria,
        "exclusion_criteria": question.exclusion_criteria,
        "allowed_outcome_ids": list(question.allowed_outcomes),
        "allowed_outcome_text_by_id": dict(question.outcome_id_to_text),
        "raw_positive_direction_meaning_by_outcome_id": dict(
            question.positive_direction_means_by_outcome_id
        ),
        "outcome_membership_sha256": hash_canonical(question.outcome_id_to_text),
    }
    _reject_forbidden_model_facing_fields(payload, location="question_surface")
    return MetaSynExtractionQuestionSurfaceV2.model_validate(
        {**payload, "question_surface_sha256": hash_canonical(payload)}
    )


def _freeze_source_strength(
    *, row: MetaSynV5SourceSurfaceRowV1, projection_v2: FrozenMetaSynProjectionV2
) -> MetaSynSourceStrengthSurfaceV2:
    source = row.source_row
    payload = {
        "source_strength_surface_version": SOURCE_STRENGTH_SURFACE_VERSION,
        "source_content_scope": source.source_content_scope,
        "oracle_selection_full_text_scope": source.oracle_selection_full_text_scope,
        "source_projection_strength": source.source_projection_strength,
        "release_grade_source_grounding_eligible": (source.release_grade_source_grounding_eligible),
        "source_strength_blockers": list(source.source_strength_blockers),
        "projection_v2_selection_complete": projection_v2.selection_complete,
        "projection_v2_omitted_passage_count": projection_v2.omitted_passage_count,
        "projection_v2_omitted_source_characters": (projection_v2.omitted_source_characters),
    }
    return MetaSynSourceStrengthSurfaceV2.model_validate(
        {**payload, "source_strength_surface_sha256": hash_canonical(payload)}
    )


def _freeze_projection_surface(
    *,
    projection_v2: FrozenMetaSynProjectionV2,
    source_strength: MetaSynSourceStrengthSurfaceV2,
) -> MetaSynModelProjectionSurfaceV2:
    passages: list[MetaSynModelPassageSurfaceV2] = []
    for passage in sorted(
        (item for item in projection_v2.passages if item.selection_status == "selected"),
        key=lambda item: item.prompt_rank or 0,
    ):
        if passage.prompt_rank is None:  # pragma: no cover - projection contract guards it
            raise MetaSynExtractionInputsV2Error(
                "metasyn_extraction_inputs_v2_selected_passage_rank_missing"
            )
        payload = {
            "passage_surface_version": PASSAGE_SURFACE_VERSION,
            "passage_id": passage.passage_anchor,
            "prompt_rank": passage.prompt_rank,
            "passage_lineage_sha256": passage.passage_lineage_sha256,
            "section_enums": list(passage.exposed_sections),
            "upstream_line_ids": list(passage.line_ids),
            "exact_source_occurrence_count": passage.origin_count,
            "text": passage.text,
            "text_sha256": passage.text_sha256,
        }
        passages.append(
            MetaSynModelPassageSurfaceV2.model_validate(
                {**payload, "passage_surface_sha256": hash_canonical(payload)}
            )
        )
    if not passages:
        raise MetaSynExtractionInputsV2Error("metasyn_extraction_inputs_v2_no_selected_passage")
    membership = [
        item.model_dump(mode="json")
        for item in sorted(passages, key=lambda passage: passage.passage_id)
    ]
    payload = {
        "projection_surface_version": PROJECTION_SURFACE_VERSION,
        "projection_v2_sha256": projection_v2.projection_sha256,
        "projection_prompt_source_sha256": projection_v2.prompt_source_sha256,
        "source_strength": source_strength,
        "source_strength_surface_sha256": (source_strength.source_strength_surface_sha256),
        "selected_passage_count": len(passages),
        "omitted_passage_count": projection_v2.omitted_passage_count,
        "selection_complete": projection_v2.selection_complete,
        "passage_ids": [item.passage_id for item in passages],
        "passages": passages,
        "passage_membership_sha256": hash_canonical(membership),
    }
    _reject_forbidden_model_facing_fields(payload, location="projection_surface")
    return MetaSynModelProjectionSurfaceV2.model_validate(
        {**payload, "projection_surface_sha256": hash_canonical(payload)}
    )


def _freeze_inventory_input(
    *,
    question_surface: MetaSynExtractionQuestionSurfaceV2,
    projection_v2: FrozenMetaSynProjectionV2,
    projection_surface: MetaSynModelProjectionSurfaceV2,
    prompt_binding: MetaSynPromptTemplateBindingV2,
) -> MetaSynInventoryInputSurfaceV2:
    prompt = _render_prompt(
        prompt_binding.prompt_template,
        {
            "QUESTION_SPEC_JSON": question_surface.model_dump(mode="json"),
            "PROJECTION_V2_JSON": projection_surface.model_dump(mode="json"),
        },
    )
    schema = metasyn_candidate_inventory_schema_bundle_v2(
        allowed_outcome_ids=question_surface.allowed_outcome_ids,
        passage_ids=sorted(projection_surface.passage_ids),
    )
    _validate_inventory_schema_bundle(
        schema,
        allowed_outcome_ids=question_surface.allowed_outcome_ids,
        passage_ids=sorted(projection_surface.passage_ids),
    )
    payload = {
        "inventory_input_version": INVENTORY_INPUT_VERSION,
        "prompt_binding_sha256": prompt_binding.prompt_binding_sha256,
        "question_surface_sha256": question_surface.question_surface_sha256,
        "projection_surface_sha256": projection_surface.projection_surface_sha256,
        "projection_v2_sha256": projection_v2.projection_sha256,
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": _sha256_text(prompt),
        "rendered_prompt_characters": len(prompt),
        "inventory_schema_bundle": schema,
        "inventory_schema_bundle_sha256": schema["schema_bundle_sha256"],
    }
    return MetaSynInventoryInputSurfaceV2.model_validate(
        {**payload, "inventory_input_sha256": hash_canonical(payload)}
    )


def _freeze_row_input(
    *,
    source_surface: MetaSynV5SourceSurfaceV1,
    row: MetaSynV5SourceSurfaceRowV1,
    inventory_prompt_binding: MetaSynPromptTemplateBindingV2,
) -> MetaSynExtractionRowInputV2:
    projection_v1 = row.source_row.projection
    lineage = freeze_projection_v2_lineage_binding(
        upstream_execution_bundle_sha256=source_surface.v5_execution_bundle_sha256,
        upstream_row_context_sha256=row.upstream_row_context_sha256,
        upstream_source_row_sha256=row.source_row_sha256,
        projection=projection_v1,
    )
    projection_v2 = freeze_metasyn_projection_v2(
        projection=projection_v1,
        lineage_binding=lineage,
    )
    validate_metasyn_projection_v2_external_replay(
        projection_v2=projection_v2,
        projection_v1=projection_v1,
        lineage_binding=lineage,
    )
    question_surface = _freeze_question_surface(row)
    source_strength = _freeze_source_strength(row=row, projection_v2=projection_v2)
    projection_surface = _freeze_projection_surface(
        projection_v2=projection_v2,
        source_strength=source_strength,
    )
    inventory_input = _freeze_inventory_input(
        question_surface=question_surface,
        projection_v2=projection_v2,
        projection_surface=projection_surface,
        prompt_binding=inventory_prompt_binding,
    )
    payload = {
        "row_input_version": EXTRACTION_ROW_INPUT_VERSION,
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "upstream_source_surface_row_sha256": row.source_surface_row_sha256,
        "upstream_row_context_sha256": row.upstream_row_context_sha256,
        "upstream_question_bundle_sha256": row.question_bundle_sha256,
        "upstream_question_spec_sha256": row.question_spec_sha256,
        "upstream_component_binding_sha256": row.component_binding_sha256,
        "upstream_source_row_sha256": row.source_row_sha256,
        "upstream_projection_sha256": row.projection_sha256,
        "upstream_artifact_binding_sha256": row.artifact_binding_sha256,
        "question_surface": question_surface,
        "question_surface_sha256": question_surface.question_surface_sha256,
        "source_strength": source_strength,
        "source_strength_surface_sha256": (source_strength.source_strength_surface_sha256),
        "projection_v2": projection_v2,
        "projection_v2_sha256": projection_v2.projection_sha256,
        "projection_surface": projection_surface,
        "projection_surface_sha256": projection_surface.projection_surface_sha256,
        "inventory_input": inventory_input,
        "inventory_input_sha256": inventory_input.inventory_input_sha256,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "v5_hosted_outputs_consumed": False,
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynExtractionRowInputV2.model_validate(
        {**payload, "row_input_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_extraction_inputs_v2(
    *,
    repository_root: Path | None = None,
    source_surface: MetaSynV5SourceSurfaceV1 | Mapping[str, Any] | None = None,
) -> MetaSynExtractionInputsV2:
    """Externally replay v5 source lineage and freeze all provider-neutral inputs."""

    root = _canonical_root(repository_root or Path(__file__).resolve().parents[2])
    if source_surface is None:
        canonical_source = freeze_metasyn_v5_source_surface(repository_root=root)
    else:
        canonical_source = validate_metasyn_v5_source_surface(
            source_surface=(
                source_surface
                if isinstance(source_surface, dict)
                else source_surface.model_dump(mode="json")
                if isinstance(source_surface, MetaSynV5SourceSurfaceV1)
                else dict(source_surface)
            ),
            repository_root=root,
            external_replay=True,
        )
    if (
        len(canonical_source.rows) != EXPECTED_PUBLICATION_COUNT
        or not canonical_source.reference_fields_unopened
        or canonical_source.official_test_labels_opened
    ):
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_source_surface_boundary_invalid"
        )

    inventory_binding = _freeze_prompt_binding(root=root, kind="inventory")
    packet_binding = _freeze_prompt_binding(root=root, kind="packet")
    rows = [
        _freeze_row_input(
            source_surface=canonical_source,
            row=row,
            inventory_prompt_binding=inventory_binding,
        )
        for row in canonical_source.rows
    ]
    fingerprint = compute_metasyn_extraction_inputs_v2_pipeline_fingerprint(
        source_surface=canonical_source,
        root=root,
    )
    payload = {
        "extraction_inputs_version": EXTRACTION_INPUTS_VERSION,
        "status": "frozen_provider_neutral_inputs_no_provider_calls",
        "upstream_source_surface_sha256": canonical_source.source_surface_sha256,
        "upstream_source_surface_pipeline_sha256": (
            canonical_source.source_surface_pipeline_sha256
        ),
        "upstream_v5_execution_bundle_sha256": (canonical_source.v5_execution_bundle_sha256),
        "upstream_v5_adapter_bundle_sha256": canonical_source.v5_adapter_bundle_sha256,
        "upstream_v5_runtime_pipeline_sha256": (canonical_source.v5_runtime_pipeline_sha256),
        "upstream_v5_question_membership_sha256": (canonical_source.v5_question_membership_sha256),
        "upstream_v5_component_membership_sha256": (
            canonical_source.v5_component_membership_sha256
        ),
        "upstream_v5_row_membership_sha256": (canonical_source.v5_row_membership_sha256),
        "upstream_source_artifact_membership_sha256": (
            canonical_source.source_artifact_membership_sha256
        ),
        "extraction_inputs_pipeline_fingerprint": fingerprint,
        "extraction_inputs_pipeline_sha256": fingerprint.pipeline_sha256,
        "inventory_prompt_binding": inventory_binding,
        "inventory_prompt_binding_sha256": inventory_binding.prompt_binding_sha256,
        "packet_prompt_binding": packet_binding,
        "packet_prompt_binding_sha256": packet_binding.prompt_binding_sha256,
        "question_protocol_field_whitelist": sorted(QUESTION_PROTOCOL_FIELD_WHITELIST),
        "forbidden_model_facing_field_names": list(FORBIDDEN_MODEL_FACING_FIELD_NAMES),
        "question_count": canonical_source.question_count,
        "component_count": canonical_source.component_count,
        "publication_count": canonical_source.publication_count,
        "rows": rows,
        "row_input_hash_membership_sha256": hash_canonical([row.row_input_sha256 for row in rows]),
        "projection_v2_membership_sha256": hash_canonical(
            [row.projection_v2_sha256 for row in rows]
        ),
        "question_surface_membership_sha256": hash_canonical(
            sorted({row.question_surface_sha256 for row in rows})
        ),
        "inventory_prompt_membership_sha256": hash_canonical(
            [row.inventory_input.rendered_prompt_sha256 for row in rows]
        ),
        "inventory_schema_membership_sha256": hash_canonical(
            [row.inventory_input.inventory_schema_bundle_sha256 for row in rows]
        ),
        "upstream_v5_source_surface_consumed": True,
        "upstream_v5_source_surface_external_replayed": True,
        "direct_v5_hosted_execution_bundle_consumed": False,
        "v5_hosted_call_receipts_consumed": False,
        "v5_hosted_row_results_consumed": False,
        "v5_hosted_provider_outputs_consumed": False,
        "provider_calls_made": False,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynExtractionInputsV2.model_validate(
        {**payload, "extraction_inputs_sha256": hash_canonical(payload)}
    )


def validate_metasyn_extraction_inputs_v2(
    *,
    extraction_inputs: MetaSynExtractionInputsV2 | Mapping[str, Any],
    repository_root: Path | None = None,
    external_replay: bool = True,
) -> MetaSynExtractionInputsV2:
    """Validate and optionally rebuild every row, prompt, schema, and fingerprint."""

    try:
        canonical = MetaSynExtractionInputsV2.model_validate(
            extraction_inputs.model_dump(mode="json")
            if isinstance(extraction_inputs, MetaSynExtractionInputsV2)
            else extraction_inputs
        )
    except ValueError as exc:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_contract_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_extraction_inputs_v2(repository_root=repository_root)
        if replayed != canonical:
            raise MetaSynExtractionInputsV2Error(
                "metasyn_extraction_inputs_v2_external_replay_mismatch"
            )
    return canonical


class MetaSynCandidatePassageSurfaceV2(_ExactContractModel):
    candidate_passage_surface_version: Literal["metasyn-candidate-passage-surface-v2"] = (
        CANDIDATE_PASSAGE_SURFACE_VERSION
    )
    projection_v2_sha256: str
    candidate_binding_sha256: str
    source_strength: MetaSynSourceStrengthSurfaceV2
    source_strength_surface_sha256: str
    passage_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    passages: Annotated[list[MetaSynModelPassageSurfaceV2], Field(min_length=1, max_length=4)]
    passage_membership_sha256: str
    candidate_passage_surface_sha256: str

    @field_validator(
        "projection_v2_sha256",
        "candidate_binding_sha256",
        "source_strength_surface_sha256",
        "passage_membership_sha256",
        "candidate_passage_surface_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("passage_ids")
    @classmethod
    def validate_passage_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_ids_not_unique")
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynCandidatePassageSurfaceV2:
        if self.source_strength_surface_sha256 != (
            self.source_strength.source_strength_surface_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_source_strength_mismatch")
        if self.passages != sorted(self.passages, key=lambda item: item.prompt_rank):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passages_not_in_prompt_order")
        if self.passage_ids != [item.passage_id for item in self.passages]:
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_alias_mismatch")
        membership = [
            item.model_dump(mode="json")
            for item in sorted(self.passages, key=lambda passage: passage.passage_id)
        ]
        if hash_canonical(membership) != self.passage_membership_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_membership_mismatch")
        payload = self.model_dump(mode="json", exclude={"candidate_passage_surface_sha256"})
        if hash_canonical(payload) != self.candidate_passage_surface_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_surface_hash_mismatch")
        return self


class MetaSynPacketCandidateInputV2(_ExactContractModel):
    packet_input_version: Literal["metasyn-packet-candidate-input-v2"] = PACKET_INPUT_VERSION
    extraction_inputs_sha256: str
    extraction_inputs_pipeline_sha256: str
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: Annotated[str, Field(min_length=1, max_length=512)]
    row_input_sha256: str
    inventory_receipt_sha256: str
    candidate: MetaSynPassageCandidateV2
    candidate_descriptor_sha256: str
    candidate_binding: PacketPassageCandidateBindingV2
    candidate_binding_sha256: str
    projection_surface: MetaSynModelProjectionSurfaceV2
    projection_surface_sha256: str
    candidate_passage_surface: MetaSynCandidatePassageSurfaceV2
    candidate_passage_surface_sha256: str
    packet_prompt_binding_sha256: str
    rendered_prompt: Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)]
    rendered_prompt_sha256: str
    rendered_prompt_characters: Annotated[int, Field(ge=1)]
    packet_schema_bundle: PacketGroundingSchemaBundleV2
    packet_schema_bundle_sha256: str
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    v5_hosted_outputs_consumed: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    packet_input_sha256: str

    @field_validator(
        "extraction_inputs_sha256",
        "extraction_inputs_pipeline_sha256",
        "row_input_sha256",
        "inventory_receipt_sha256",
        "candidate_descriptor_sha256",
        "candidate_binding_sha256",
        "projection_surface_sha256",
        "candidate_passage_surface_sha256",
        "packet_prompt_binding_sha256",
        "rendered_prompt_sha256",
        "packet_schema_bundle_sha256",
        "packet_input_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_input(self) -> MetaSynPacketCandidateInputV2:
        if self.candidate_descriptor_sha256 != self.candidate.descriptor_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_candidate_descriptor_hash_mismatch")
        if self.candidate_binding_sha256 != self.candidate_binding.binding_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_candidate_binding_hash_mismatch")
        if self.candidate_binding.candidate_descriptor_sha256 != (self.candidate_descriptor_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_binding_descriptor_mismatch")
        if self.candidate_passage_surface_sha256 != (
            self.candidate_passage_surface.candidate_passage_surface_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_hash_mismatch")
        if self.projection_surface_sha256 != (self.projection_surface.projection_surface_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_packet_projection_surface_hash_mismatch")
        if self.projection_surface.projection_v2_sha256 != (
            self.candidate_binding.projection_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_packet_projection_binding_mismatch")
        if self.candidate_passage_surface.candidate_binding_sha256 != (
            self.candidate_binding_sha256
        ):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_binding_mismatch")
        if set(self.candidate_passage_surface.passage_ids) != set(self.candidate.passage_ids):
            raise ValueError("metasyn_extraction_inputs_v2_candidate_passage_identity_mismatch")
        projection_passages = {
            passage.passage_id: passage for passage in self.projection_surface.passages
        }
        if any(
            projection_passages.get(passage.passage_id) != passage
            for passage in self.candidate_passage_surface.passages
        ):
            raise ValueError(
                "metasyn_extraction_inputs_v2_candidate_passage_outside_full_projection"
            )
        if self.packet_schema_bundle_sha256 != (self.packet_schema_bundle.schema_bundle_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_packet_schema_hash_mismatch")
        if self.packet_schema_bundle.candidate_binding_sha256 != (self.candidate_binding_sha256):
            raise ValueError("metasyn_extraction_inputs_v2_packet_schema_binding_mismatch")
        if (
            _sha256_text(self.rendered_prompt) != self.rendered_prompt_sha256
            or len(self.rendered_prompt) != self.rendered_prompt_characters
        ):
            raise ValueError("metasyn_extraction_inputs_v2_packet_prompt_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"packet_input_sha256"})
        if hash_canonical(payload) != self.packet_input_sha256:
            raise ValueError("metasyn_extraction_inputs_v2_packet_input_hash_mismatch")
        return self


def _canonical_extraction_inputs(
    value: MetaSynExtractionInputsV2 | Mapping[str, Any],
) -> MetaSynExtractionInputsV2:
    try:
        return MetaSynExtractionInputsV2.model_validate(
            value.model_dump(mode="json") if isinstance(value, MetaSynExtractionInputsV2) else value
        )
    except ValueError as exc:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_bundle_contract_invalid"
        ) from exc


def _canonical_inventory_receipt(
    *,
    receipt: MetaSynCandidateInventoryReceiptV2 | Mapping[str, Any],
    row: MetaSynExtractionRowInputV2,
) -> MetaSynCandidateInventoryReceiptV2:
    return validate_metasyn_candidate_inventory_receipt_v2(
        receipt,
        row_context_sha256=row.upstream_row_context_sha256,
        projection_v2_sha256=row.projection_v2_sha256,
        allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
        passage_text_by_id={
            passage.passage_id: passage.text for passage in row.projection_surface.passages
        },
    )


def freeze_metasyn_packet_candidate_input_v2(
    *,
    extraction_inputs: MetaSynExtractionInputsV2 | Mapping[str, Any],
    row_ordinal: int,
    inventory_receipt: MetaSynCandidateInventoryReceiptV2 | Mapping[str, Any],
    candidate_index: int,
) -> MetaSynPacketCandidateInputV2:
    """Bind one authorized inventory candidate to an exact packet prompt/schema."""

    bundle = _canonical_extraction_inputs(extraction_inputs)
    if isinstance(row_ordinal, bool) or not 0 <= row_ordinal < len(bundle.rows):
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_packet_row_ordinal_invalid"
        )
    row = bundle.rows[row_ordinal]
    receipt = _canonical_inventory_receipt(receipt=inventory_receipt, row=row)
    if receipt.status != "candidates_authorized":
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_inventory_not_packet_authorizing"
        )
    candidates = {
        candidate.candidate_index: candidate for candidate in receipt.inventory.candidates
    }
    candidate = candidates.get(candidate_index)
    if candidate is None:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_candidate_index_not_authorized"
        )
    outcome_text = row.question_surface.allowed_outcome_text_by_id.get(
        candidate.canonical_outcome_id
    )
    if outcome_text is None or candidate.outcome_concept_quote not in outcome_text:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_candidate_outcome_not_frozen"
        )
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=row.projection_v2,
    )
    passage_by_id = {passage.passage_id: passage for passage in row.projection_surface.passages}
    try:
        passages = sorted(
            (passage_by_id[item] for item in candidate.passage_ids),
            key=lambda passage: passage.prompt_rank,
        )
    except KeyError as exc:  # pragma: no cover - inventory receipt validation guards it
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_candidate_passage_not_frozen"
        ) from exc
    passage_membership = [
        item.model_dump(mode="json")
        for item in sorted(passages, key=lambda passage: passage.passage_id)
    ]
    candidate_passage_payload = {
        "candidate_passage_surface_version": CANDIDATE_PASSAGE_SURFACE_VERSION,
        "projection_v2_sha256": row.projection_v2_sha256,
        "candidate_binding_sha256": binding.binding_sha256,
        "source_strength": row.source_strength,
        "source_strength_surface_sha256": row.source_strength_surface_sha256,
        "passage_ids": [passage.passage_id for passage in passages],
        "passages": passages,
        "passage_membership_sha256": hash_canonical(passage_membership),
    }
    candidate_passage_surface = MetaSynCandidatePassageSurfaceV2.model_validate(
        {
            **candidate_passage_payload,
            "candidate_passage_surface_sha256": hash_canonical(candidate_passage_payload),
        }
    )
    prompt = _render_prompt(
        bundle.packet_prompt_binding.prompt_template,
        {
            "QUESTION_SURFACE_JSON": row.question_surface.model_dump(mode="json"),
            "CANDIDATE_BINDING_JSON": binding.model_dump(mode="json"),
            "PROJECTION_V2_JSON": row.projection_surface.model_dump(mode="json"),
            "CANDIDATE_PASSAGE_SURFACE_JSON": (candidate_passage_surface.model_dump(mode="json")),
        },
    )
    schema = freeze_packet_grounding_schema_bundle_v2(binding=binding)
    payload = {
        "packet_input_version": PACKET_INPUT_VERSION,
        "extraction_inputs_sha256": bundle.extraction_inputs_sha256,
        "extraction_inputs_pipeline_sha256": (bundle.extraction_inputs_pipeline_sha256),
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "row_input_sha256": row.row_input_sha256,
        "inventory_receipt_sha256": receipt.receipt_sha256,
        "candidate": candidate,
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "candidate_binding": binding,
        "candidate_binding_sha256": binding.binding_sha256,
        "projection_surface": row.projection_surface,
        "projection_surface_sha256": row.projection_surface_sha256,
        "candidate_passage_surface": candidate_passage_surface,
        "candidate_passage_surface_sha256": (
            candidate_passage_surface.candidate_passage_surface_sha256
        ),
        "packet_prompt_binding_sha256": (bundle.packet_prompt_binding.prompt_binding_sha256),
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": _sha256_text(prompt),
        "rendered_prompt_characters": len(prompt),
        "packet_schema_bundle": schema,
        "packet_schema_bundle_sha256": schema.schema_bundle_sha256,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "v5_hosted_outputs_consumed": False,
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPacketCandidateInputV2.model_validate(
        {**payload, "packet_input_sha256": hash_canonical(payload)}
    )


def validate_metasyn_packet_candidate_input_v2(
    *,
    packet_input: MetaSynPacketCandidateInputV2 | Mapping[str, Any],
    extraction_inputs: MetaSynExtractionInputsV2 | Mapping[str, Any],
    inventory_receipt: MetaSynCandidateInventoryReceiptV2 | Mapping[str, Any],
) -> MetaSynPacketCandidateInputV2:
    """Replay a candidate-specific packet surface from its authorized inventory."""

    try:
        canonical = MetaSynPacketCandidateInputV2.model_validate(
            packet_input.model_dump(mode="json")
            if isinstance(packet_input, MetaSynPacketCandidateInputV2)
            else packet_input
        )
    except ValueError as exc:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_packet_contract_invalid"
        ) from exc
    replayed = freeze_metasyn_packet_candidate_input_v2(
        extraction_inputs=extraction_inputs,
        row_ordinal=canonical.row_ordinal,
        inventory_receipt=inventory_receipt,
        candidate_index=canonical.candidate.candidate_index,
    )
    if replayed != canonical:
        raise MetaSynExtractionInputsV2Error(
            "metasyn_extraction_inputs_v2_packet_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "EXTRACTION_INPUTS_VERSION",
    "FORBIDDEN_MODEL_FACING_FIELD_NAMES",
    "INVENTORY_PROMPT_PATH",
    "PACKET_PROMPT_PATH",
    "QUESTION_PROTOCOL_FIELD_WHITELIST",
    "MetaSynCandidatePassageSurfaceV2",
    "MetaSynExtractionInputsV2",
    "MetaSynExtractionInputsV2Error",
    "MetaSynExtractionQuestionSurfaceV2",
    "MetaSynExtractionRowInputV2",
    "MetaSynInventoryInputSurfaceV2",
    "MetaSynModelPassageSurfaceV2",
    "MetaSynModelProjectionSurfaceV2",
    "MetaSynPacketCandidateInputV2",
    "MetaSynPromptTemplateBindingV2",
    "MetaSynSourceStrengthSurfaceV2",
    "compute_metasyn_extraction_inputs_v2_pipeline_fingerprint",
    "freeze_metasyn_extraction_inputs_v2",
    "freeze_metasyn_packet_candidate_input_v2",
    "validate_metasyn_extraction_inputs_v2",
    "validate_metasyn_packet_candidate_input_v2",
]
