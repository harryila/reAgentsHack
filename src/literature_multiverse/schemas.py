"""Generate strict, topic-aware extraction and normalized finding schemas."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    create_model,
    model_validator,
)

from literature_multiverse.config import ModeratorSpec, QuestionConfig, config_sha256
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import CanonicalDirection, ContractModel, FindingRow
from literature_multiverse.paths import PATHS


class ExtractionFindingFixed(ContractModel):
    """Model-supplied fields only; every nullable key remains required."""

    study_type: str | None
    species: str | None
    model: str | None
    population_state: str | None
    intervention: str | None
    intervention_class: str | None
    comparator: str | None
    dose_raw: str | None
    duration_raw: str | None
    timing_context: str | None
    outcome_name: str = Field(min_length=1)
    outcome_family: str | None
    timepoint_raw: str | None
    effect_direction: CanonicalDirection
    effect_size_raw: str | None
    p_value: float | None = Field(ge=0, le=1)
    significant: bool | None
    sample_size: int | None = Field(ge=1)
    evidence_quote: str | None
    evidence_lines: list[str] | None
    confidence: float | None = Field(ge=0, le=1)


class ExtractionEnvelopeFixed(ContractModel):
    eligible: bool
    exclusion_reason: str | None

    @model_validator(mode="after")
    def validate_eligibility_findings(self) -> ExtractionEnvelopeFixed:
        findings = getattr(self, "findings", None)
        if not self.eligible and findings:
            raise ValueError("ineligible_extraction_must_have_zero_findings")
        return self


def _literal_annotation(values: tuple[Any, ...]) -> Any:
    return Literal.__getitem__(values)  # type: ignore[attr-defined]


def _moderator_annotation(spec: ModeratorSpec) -> Any:
    if spec.type == "categorical":
        assert spec.allowed_values is not None
        return _literal_annotation(tuple(spec.allowed_values)) | None
    if spec.type == "bool":
        return StrictBool | None
    if spec.type == "int":
        return StrictInt | None
    return StrictFloat | None


@lru_cache(maxsize=32)
def _models_for_canonical_config(
    config_json: str,
) -> tuple[type[ContractModel], type[ContractModel], type[FindingRow]]:
    config = QuestionConfig.model_validate_json(config_json)
    safe_name = config.question_id.replace("-", "_")
    moderator_fields = {
        spec.name: (_moderator_annotation(spec), ...) for spec in config.moderators
    }
    moderator_model = create_model(
        f"Moderators_{safe_name}",
        __config__=ConfigDict(extra="forbid", validate_assignment=True),
        **moderator_fields,
    )
    extraction_finding = create_model(
        f"ExtractionFinding_{safe_name}",
        __base__=ExtractionFindingFixed,
        moderators=(moderator_model, ...),
    )
    extraction_envelope = create_model(
        f"ExtractionEnvelope_{safe_name}",
        __base__=ExtractionEnvelopeFixed,
        findings=(list[extraction_finding], ...),
    )
    normalized_finding = create_model(
        f"FindingRow_{safe_name}",
        __base__=FindingRow,
        moderators=(moderator_model, ...),
    )
    return extraction_envelope, extraction_finding, normalized_finding


def build_extraction_models(
    config: QuestionConfig,
) -> tuple[type[ContractModel], type[ContractModel]]:
    canonical = config.model_dump_json(exclude_none=False)
    envelope, finding, _ = _models_for_canonical_config(canonical)
    return envelope, finding


def finding_row_model(config: QuestionConfig) -> type[FindingRow]:
    canonical = config.model_dump_json(exclude_none=False)
    _, _, finding = _models_for_canonical_config(canonical)
    return finding


def generate_extraction_schema(config: QuestionConfig) -> dict[str, Any]:
    """Return the JSON Schema 2020-12 contract sent to the extraction worker."""

    envelope, _ = build_extraction_models(config)
    schema = envelope.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"urn:literature-multiverse:extraction:{config.question_id}:v1"
    schema["title"] = f"Literature Multiverse extraction envelope — {config.question_id}"
    schema["x-question-config-sha256"] = config_sha256(config)
    schema["x-schema-version"] = "1"
    assert_closed_object_schema(schema)
    properties = schema.get("properties", {})
    forbidden_identity = {"paper_id", "doc_id", "finding_id", "map_result_id"}
    if forbidden_identity & set(properties):
        raise AssertionError("model_identity_leaked_into_extraction_envelope")
    return schema


def assert_closed_object_schema(schema: dict[str, Any]) -> None:
    """Prove recursively that every object node rejects unknown properties."""

    def visit(node: Any, location: str) -> None:
        if isinstance(node, dict):
            is_object = node.get("type") == "object" or "properties" in node
            if is_object and node.get("additionalProperties") is not False:
                raise ValueError(f"schema_object_not_closed:{location}")
            for key, value in node.items():
                visit(value, f"{location}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{location}/{index}")

    visit(schema, "#")


def schema_sha256(config: QuestionConfig) -> str:
    return hash_canonical(generate_extraction_schema(config))


def write_extraction_schema(
    config: QuestionConfig, *, path: Path | None = None, force: bool = False
) -> Path:
    target = path or PATHS.schema_path(config.question_id)
    atomic_write_json(target, generate_extraction_schema(config), force=force)
    return target


def validate_extraction_payload(payload: Any, config: QuestionConfig) -> ContractModel:
    envelope, _ = build_extraction_models(config)
    return envelope.model_validate(payload)


def validate_finding_row(payload: Any, config: QuestionConfig) -> FindingRow:
    model = finding_row_model(config)
    return model.model_validate(payload)
