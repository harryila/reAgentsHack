"""Deterministic, question-aware projection of verified native source text.

The projector is deliberately separate from provider/model orchestration.  It consumes
an already verified :class:`ResolvedNativeSource` and an exact P/I(E)/C/O specification,
then freezes the only source passages a bounded extraction model may see.  Scientific
outputs, reference directions, conclusions, effect values, and audit labels are not
inputs to this module.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import ResolvedNativeSource, ResolvedSourceLine

PROJECTION_ALGORITHM = "question-aware-source-projection-v1"
PROJECTION_SPEC_VERSION = "native-question-projection-spec-v1"
SOURCE_PROJECTION_VERSION = "native-frozen-source-projection-v1"

MAX_TERMS_PER_GROUP = 128
MAX_PASSAGE_CHARACTERS = 1_800
MAX_PROJECTED_CHARACTERS = 14_000
MAX_PROJECTED_PASSAGES = 24
RESERVED_METHODS_PASSAGES = 3
MAX_QUESTION_ID_CHARACTERS = 128
MAX_QUESTION_FIELD_CHARACTERS = 4_096
MAX_OUTCOMES = 16
MAX_OUTCOME_ID_CHARACTERS = 64
MAX_OUTCOME_TEXT_CHARACTERS = 4_096
MAX_DIRECTION_SEMANTICS_CHARACTERS = 512
MAX_ROLE_CHARACTERS = 128
MAX_ESTIMAND_CHARACTERS = 512
MAX_MODERATORS = 8
MAX_MODERATOR_CHARACTERS = 64
MAX_ROW_ID_CHARACTERS = 256
MAX_ARTIFACT_PATH_CHARACTERS = 2_048
MAX_SOURCE_LOCATOR_CHARACTERS = 2_048
MAX_SECTION_CHARACTERS = 256
RESERVED_FALLBACK_PASSAGES = 2
EARLY_RESULTS_PASSAGES = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_SIGNAL = re.compile(
    r"(?:\d|%|\bp\s*[<=>\u2264\u2265]|confidence interval|\bci\b|standard deviation|"
    r"\bsd\b|standard error|\bse\b|\u00b1)",
    flags=re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# This vocabulary removes syntax and generic review words only.  It contains no
# domain-, intervention-, outcome-, effect-, or label-specific scientific terms.
_STOPWORDS = frozenset(
    _TOKEN_RE.findall(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having how if in into is it its itself more most no nor not
    of off on once only or other our out over same should so some such than that the their
    them then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would study studies research
    systematic review reviews meta analysis analyses effect effects evaluate evaluates
    evaluated assessing assessed assess association associations relationship
    """
    )
)

_METHODS_HEADING_TOKENS = frozenset(
    {
        "design",
        "material",
        "materials",
        "method",
        "methods",
        "participant",
        "participants",
        "patient",
        "patients",
        "protocol",
    }
)
_RESULTS_HEADING_TOKENS = frozenset(
    {"result", "results", "finding", "findings", "outcome", "outcomes", "efficacy"}
)
_TABLE_HEADING_TOKENS = frozenset(
    {"table", "tables", "figure", "figures", "supplement", "supplementary"}
)

SourceModality = Literal[
    "full_text_recognized_sections",
    "full_text_unrecognized_sections",
    "title_abstract",
    "abstract_only",
    "title_only",
    "sectioned_source_recognized",
    "sectioned_source_unrecognized",
]
SectionFamily = Literal[
    "title",
    "abstract",
    "methods",
    "results",
    "table_or_figure",
    "other",
]
ExposedSection = Literal["Abstract", "FigureTable", "Methods", "Results", "Title"]
SourceStrength = Literal[
    "full_text_textual_grounding",
    "diagnostic_title_abstract_grounding",
    "diagnostic_unrecognized_sections",
    "no_eligible_source_passage",
]


class NativeQuestionProjectionError(ValueError):
    """A question specification or source cannot be projected without ambiguity."""


class CanonicalOutcomeV1(ContractModel):
    """A bounded model-facing ID bound to the untruncated scientific outcome text."""

    outcome_id: Annotated[
        str,
        Field(
            pattern=r"^outcome-[0-9]{2}$",
            max_length=MAX_OUTCOME_ID_CHARACTERS,
        ),
    ]
    outcome_text: Annotated[str, Field(min_length=1, max_length=MAX_OUTCOME_TEXT_CHARACTERS)]
    positive_direction_means: Annotated[
        str,
        Field(min_length=1, max_length=MAX_DIRECTION_SEMANTICS_CHARACTERS),
    ]


class QuestionProjectionFieldsV1(ContractModel):
    """Exact scientific fields from which every projection term is derived."""

    population: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_FIELD_CHARACTERS)]
    intervention_or_exposure: Annotated[
        str, Field(min_length=1, max_length=MAX_QUESTION_FIELD_CHARACTERS)
    ]
    comparison: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_FIELD_CHARACTERS)]
    treatment_role: Annotated[str, Field(min_length=1, max_length=MAX_ROLE_CHARACTERS)]
    comparator_role: Annotated[str, Field(min_length=1, max_length=MAX_ROLE_CHARACTERS)]
    contrast_estimand: Annotated[
        str, Field(min_length=1, max_length=MAX_ESTIMAND_CHARACTERS)
    ]
    outcomes: Annotated[
        list[CanonicalOutcomeV1], Field(min_length=1, max_length=MAX_OUTCOMES)
    ]

    @field_validator("outcomes")
    @classmethod
    def validate_outcomes(cls, value: list[CanonicalOutcomeV1]) -> list[CanonicalOutcomeV1]:
        ids = [item.outcome_id for item in value]
        normalized_text = [" ".join(item.outcome_text.casefold().split()) for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("question_projection_outcomes_not_sorted_unique")
        if len(normalized_text) != len(set(normalized_text)):
            raise ValueError("question_projection_outcome_text_not_casefold_unique")
        return value


class QuestionProjectionSpecV1(ContractModel):
    """Self-hashed P/I(E)/C/O-only projection policy for one question."""

    projection_spec_version: Literal["native-question-projection-spec-v1"] = (
        PROJECTION_SPEC_VERSION
    )
    projection_algorithm: Literal["question-aware-source-projection-v1"] = (
        PROJECTION_ALGORITHM
    )
    question_id: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_ID_CHARACTERS)]
    question_fields: QuestionProjectionFieldsV1
    question_spec_sha256: str
    population_terms: list[str]
    intervention_or_exposure_terms: list[str]
    comparison_terms: list[str]
    outcome_terms: list[str]
    allowed_outcomes: Annotated[
        list[
            Annotated[str, Field(min_length=1, max_length=MAX_OUTCOME_ID_CHARACTERS)]
        ],
        Field(min_length=1, max_length=MAX_OUTCOMES),
    ]
    allowed_moderators: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_MODERATOR_CHARACTERS)]],
        Field(max_length=MAX_MODERATORS),
    ]
    max_passage_characters: Literal[1800] = MAX_PASSAGE_CHARACTERS
    max_projected_characters: Literal[14000] = MAX_PROJECTED_CHARACTERS
    max_projected_passages: Literal[24] = MAX_PROJECTED_PASSAGES
    reserved_methods_passages: Literal[3] = RESERVED_METHODS_PASSAGES
    projection_spec_sha256: str

    @field_validator("question_spec_sha256", "projection_spec_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"question_projection_sha256_invalid:{info.field_name}")
        return value

    @field_validator(
        "population_terms",
        "intervention_or_exposure_terms",
        "comparison_terms",
        "outcome_terms",
        "allowed_moderators",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if len(value) > MAX_TERMS_PER_GROUP:
            raise ValueError(f"question_projection_terms_exceed_cap:{info.field_name}")
        if value != sorted(set(value)) or any(not term for term in value):
            raise ValueError(f"question_projection_values_not_sorted_unique:{info.field_name}")
        return value

    @field_validator("allowed_outcomes")
    @classmethod
    def validate_allowed_outcomes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("question_projection_allowed_outcomes_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_spec(self) -> QuestionProjectionSpecV1:
        fields_payload = self.question_fields.model_dump(mode="json")
        if hash_canonical(fields_payload) != self.question_spec_sha256:
            raise ValueError("question_projection_question_spec_hash_mismatch")
        expected_terms = _term_groups(self.question_fields)
        for field_name, expected in expected_terms.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"question_projection_term_derivation_mismatch:{field_name}")
        if self.allowed_outcomes != [item.outcome_id for item in self.question_fields.outcomes]:
            raise ValueError("question_projection_allowed_outcomes_mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_spec_sha256"})
        if hash_canonical(payload) != self.projection_spec_sha256:
            raise ValueError("question_projection_spec_hash_mismatch")
        return self


class FrozenProjectedPassageV1(ContractModel):
    """One exact contiguous source substring exposed to the model."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    passage_rank: Annotated[int, Field(ge=1, le=MAX_PROJECTED_PASSAGES)]
    line_id: Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]
    line_number: Annotated[int, Field(ge=1)]
    section: Annotated[str, Field(min_length=1, max_length=MAX_SECTION_CHARACTERS)]
    section_family: SectionFamily
    exposed_section: ExposedSection
    line_char_start: Annotated[int, Field(ge=0)]
    line_char_end_exclusive: Annotated[int, Field(ge=0)]
    source_char_start: Annotated[int, Field(ge=0)]
    source_char_end_exclusive: Annotated[int, Field(ge=0)]
    source_utf8_byte_start: Annotated[int, Field(ge=0)]
    source_utf8_byte_end_exclusive: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1, max_length=MAX_PASSAGE_CHARACTERS)]
    text_sha256: str
    source_line_sha256: str
    population_term_hits: Annotated[int, Field(ge=0)]
    intervention_or_exposure_term_hits: Annotated[int, Field(ge=0)]
    comparison_term_hits: Annotated[int, Field(ge=0)]
    outcome_term_hits: Annotated[int, Field(ge=0)]
    numerical_signal: bool
    priority_score: int

    @field_validator("text_sha256", "source_line_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"projected_passage_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_passage(self) -> FrozenProjectedPassageV1:
        if self.line_id != f"L{self.line_number}":
            raise ValueError("projected_passage_line_identity_mismatch")
        if self.line_char_end_exclusive <= self.line_char_start:
            raise ValueError("projected_passage_line_offsets_invalid")
        if self.source_char_end_exclusive <= self.source_char_start:
            raise ValueError("projected_passage_source_offsets_invalid")
        if self.source_utf8_byte_end_exclusive <= self.source_utf8_byte_start:
            raise ValueError("projected_passage_source_byte_offsets_invalid")
        if _sha256_text(self.text) != self.text_sha256:
            raise ValueError("projected_passage_text_hash_mismatch")
        if self.exposed_section != _exposed_section(self.section_family):
            raise ValueError("projected_passage_exposed_section_mismatch")
        return self


class FrozenSourceProjectionV1(ContractModel):
    """Self-hashed, exact source surface for one question/publication row."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    source_projection_version: Literal["native-frozen-source-projection-v1"] = (
        SOURCE_PROJECTION_VERSION
    )
    projection_algorithm: Literal["question-aware-source-projection-v1"] = (
        PROJECTION_ALGORITHM
    )
    row_id: Annotated[str, Field(min_length=1, max_length=MAX_ROW_ID_CHARACTERS)]
    row_source_identity_sha256: str
    question_id: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_ID_CHARACTERS)]
    question_spec_sha256: str
    projection_spec_sha256: str
    source_kind: Literal["antiox_json_lines", "metasyn_parquet_row"]
    source_modality: SourceModality
    source_strength: SourceStrength
    release_grade_source_grounding_eligible: bool
    source_strength_blockers: list[str]
    artifact_path: Annotated[
        str, Field(min_length=1, max_length=MAX_ARTIFACT_PATH_CHARACTERS)
    ]
    artifact_sha256: str
    source_locator: Annotated[
        str, Field(min_length=1, max_length=MAX_SOURCE_LOCATOR_CHARACTERS)
    ]
    source_payload_sha256: str
    source_text_sha256: str
    allowed_outcomes: Annotated[
        list[
            Annotated[str, Field(min_length=1, max_length=MAX_OUTCOME_ID_CHARACTERS)]
        ],
        Field(min_length=1, max_length=MAX_OUTCOMES),
    ]
    allowed_moderators: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_MODERATOR_CHARACTERS)]],
        Field(max_length=MAX_MODERATORS),
    ]
    projection_status: Literal["ready", "no_eligible_source_passage"]
    passages: list[FrozenProjectedPassageV1]
    exposed_line_ids: list[str]
    exposed_sections: list[ExposedSection]
    projected_characters: Annotated[int, Field(ge=0, le=MAX_PROJECTED_CHARACTERS)]
    projection_sha256: str

    @field_validator(
        "row_source_identity_sha256",
        "question_spec_sha256",
        "projection_spec_sha256",
        "artifact_sha256",
        "source_payload_sha256",
        "source_text_sha256",
        "projection_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"frozen_source_projection_sha256_invalid:{info.field_name}")
        return value

    @field_validator("allowed_outcomes")
    @classmethod
    def validate_outcomes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("frozen_source_projection_outcomes_not_sorted_unique")
        return value

    @field_validator(
        "allowed_moderators",
        "exposed_line_ids",
        "exposed_sections",
        "source_strength_blockers",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError(f"frozen_source_projection_values_not_sorted_unique:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> FrozenSourceProjectionV1:
        ranks = [passage.passage_rank for passage in self.passages]
        if ranks != list(range(1, len(self.passages) + 1)):
            raise ValueError("frozen_source_projection_passage_ranks_invalid")
        identities = [
            (passage.line_number, passage.line_char_start, passage.line_char_end_exclusive)
            for passage in self.passages
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("frozen_source_projection_passage_duplicate")
        expected_line_ids = sorted({passage.line_id for passage in self.passages})
        if self.exposed_line_ids != expected_line_ids:
            raise ValueError("frozen_source_projection_line_ids_mismatch")
        expected_sections = sorted({passage.exposed_section for passage in self.passages})
        if self.exposed_sections != expected_sections:
            raise ValueError("frozen_source_projection_sections_mismatch")
        if self.projected_characters != sum(len(passage.text) for passage in self.passages):
            raise ValueError("frozen_source_projection_character_count_mismatch")
        ready = bool(self.passages)
        if ready != (self.projection_status == "ready"):
            raise ValueError("frozen_source_projection_status_mismatch")
        expected_strength, expected_eligible, expected_blockers = _source_strength(
            modality=self.source_modality,
            ready=ready,
        )
        if self.source_strength != expected_strength:
            raise ValueError("frozen_source_projection_strength_mismatch")
        if self.release_grade_source_grounding_eligible != expected_eligible:
            raise ValueError("frozen_source_projection_release_grade_flag_mismatch")
        if self.source_strength_blockers != expected_blockers:
            raise ValueError("frozen_source_projection_strength_blockers_mismatch")
        expected_identity = hash_canonical(
            {
                "row_id": self.row_id,
                "question_id": self.question_id,
                "source_locator": self.source_locator,
                "artifact_sha256": self.artifact_sha256,
                "source_payload_sha256": self.source_payload_sha256,
            }
        )
        if self.row_source_identity_sha256 != expected_identity:
            raise ValueError("frozen_source_projection_row_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_sha256"})
        if hash_canonical(payload) != self.projection_sha256:
            raise ValueError("frozen_source_projection_hash_mismatch")
        return self


def _terms(value: str) -> list[str]:
    terms = sorted(
        {
            token
            for token in _TOKEN_RE.findall(value.casefold())
            if len(token) > 1 and token not in _STOPWORDS
        }
    )
    if len(terms) > MAX_TERMS_PER_GROUP:
        raise NativeQuestionProjectionError("question_projection_terms_exceed_cap")
    return terms


def _term_groups(fields: QuestionProjectionFieldsV1) -> dict[str, list[str]]:
    return {
        "population_terms": _terms(fields.population),
        "intervention_or_exposure_terms": _terms(fields.intervention_or_exposure),
        "comparison_terms": _terms(fields.comparison),
        "outcome_terms": _terms(" ".join(item.outcome_text for item in fields.outcomes)),
    }


def freeze_canonical_outcomes(
    *,
    outcome_texts: Sequence[str],
    positive_direction_means_by_outcome: Mapping[str, str] | None,
) -> list[CanonicalOutcomeV1]:
    """Normalize outcome identity without truncating or consulting scientific labels."""

    normalized: dict[str, str] = {}
    for raw in outcome_texts:
        if not isinstance(raw, str):
            raise NativeQuestionProjectionError("question_projection_outcome_text_invalid")
        value = raw.strip()
        if not value:
            raise NativeQuestionProjectionError("question_projection_outcome_text_empty")
        if len(value) > MAX_OUTCOME_TEXT_CHARACTERS:
            raise NativeQuestionProjectionError("question_projection_outcome_text_exceeds_cap")
        key = " ".join(value.casefold().split())
        current = normalized.get(key)
        if current is None or value < current:
            normalized[key] = value
    if not normalized:
        raise NativeQuestionProjectionError("question_projection_outcomes_empty")
    if len(normalized) > MAX_OUTCOMES:
        raise NativeQuestionProjectionError("question_projection_outcome_count_exceeds_cap")

    supplied: dict[str, str] = {}
    if positive_direction_means_by_outcome is not None:
        for raw_text, raw_semantics in positive_direction_means_by_outcome.items():
            if not isinstance(raw_text, str) or not isinstance(raw_semantics, str):
                raise NativeQuestionProjectionError(
                    "question_projection_direction_semantics_invalid"
                )
            key = " ".join(raw_text.strip().casefold().split())
            semantics = raw_semantics.strip()
            if not key or not semantics:
                raise NativeQuestionProjectionError(
                    "question_projection_direction_semantics_empty"
                )
            if len(semantics) > MAX_DIRECTION_SEMANTICS_CHARACTERS:
                raise NativeQuestionProjectionError(
                    "question_projection_direction_semantics_exceeds_cap"
                )
            if key in supplied and supplied[key] != semantics:
                raise NativeQuestionProjectionError(
                    "question_projection_direction_semantics_conflict"
                )
            supplied[key] = semantics
        if set(supplied) != set(normalized):
            raise NativeQuestionProjectionError(
                "question_projection_direction_semantics_membership_mismatch"
            )

    outcomes: list[CanonicalOutcomeV1] = []
    for index, key in enumerate(sorted(normalized), start=1):
        outcomes.append(
            CanonicalOutcomeV1(
                outcome_id=f"outcome-{index:02d}",
                outcome_text=normalized[key],
                positive_direction_means=(
                    supplied[key]
                    if supplied
                    else "not_prespecified_from_question_metadata"
                ),
            )
        )
    return outcomes


def freeze_question_projection_spec(
    *,
    question_id: str,
    population: str,
    intervention_or_exposure: str,
    comparison: str,
    outcome_texts: Sequence[str],
    treatment_role: str,
    comparator_role: str,
    contrast_estimand: str,
    positive_direction_means_by_outcome: Mapping[str, str] | None = None,
    allowed_moderators: Sequence[str] = (),
) -> QuestionProjectionSpecV1:
    """Freeze the only scientific fields allowed to influence source projection."""

    canonical_outcomes = freeze_canonical_outcomes(
        outcome_texts=outcome_texts,
        positive_direction_means_by_outcome=positive_direction_means_by_outcome,
    )
    normalized_moderators = sorted(
        {item.strip() for item in allowed_moderators if isinstance(item, str) and item.strip()}
    )
    if len(normalized_moderators) != len(allowed_moderators):
        raise NativeQuestionProjectionError(
            "question_projection_moderators_empty_or_duplicate"
        )
    fields = QuestionProjectionFieldsV1(
        population=population,
        intervention_or_exposure=intervention_or_exposure,
        comparison=comparison,
        treatment_role=treatment_role,
        comparator_role=comparator_role,
        contrast_estimand=contrast_estimand,
        outcomes=canonical_outcomes,
    )
    terms = _term_groups(fields)
    payload: dict[str, Any] = {
        "projection_spec_version": PROJECTION_SPEC_VERSION,
        "projection_algorithm": PROJECTION_ALGORITHM,
        "question_id": question_id,
        "question_fields": fields,
        "question_spec_sha256": hash_canonical(fields.model_dump(mode="json")),
        **terms,
        "allowed_outcomes": [item.outcome_id for item in fields.outcomes],
        "allowed_moderators": normalized_moderators,
        "max_passage_characters": MAX_PASSAGE_CHARACTERS,
        "max_projected_characters": MAX_PROJECTED_CHARACTERS,
        "max_projected_passages": MAX_PROJECTED_PASSAGES,
        "reserved_methods_passages": RESERVED_METHODS_PASSAGES,
    }
    return QuestionProjectionSpecV1.model_validate(
        {**payload, "projection_spec_sha256": hash_canonical(payload)}
    )


def _section_family(section: str) -> SectionFamily:
    tokens = set(_TOKEN_RE.findall(section.casefold()))
    if "title" in tokens:
        return "title"
    if "abstract" in tokens or "summary" in tokens:
        return "abstract"
    if tokens & _TABLE_HEADING_TOKENS:
        return "table_or_figure"
    if tokens & _RESULTS_HEADING_TOKENS:
        return "results"
    if tokens & _METHODS_HEADING_TOKENS:
        return "methods"
    return "other"


def _source_modality(source: ResolvedNativeSource) -> SourceModality:
    families = {_section_family(line.section) for line in source.lines}
    recognized_full_text = bool(
        families.intersection({"methods", "results", "table_or_figure"})
    )
    if source.source_kind == "metasyn_parquet_row":
        if recognized_full_text:
            return "full_text_recognized_sections"
        if families - {"title", "abstract"}:
            return "full_text_unrecognized_sections"
        if {"title", "abstract"} <= families:
            return "title_abstract"
        if "abstract" in families:
            return "abstract_only"
        return "title_only"
    return (
        "sectioned_source_recognized"
        if recognized_full_text
        else "sectioned_source_unrecognized"
    )


def _source_strength(
    *, modality: SourceModality, ready: bool
) -> tuple[SourceStrength, bool, list[str]]:
    if not ready:
        return "no_eligible_source_passage", False, ["no_eligible_source_passage"]
    if modality in {"full_text_recognized_sections", "sectioned_source_recognized"}:
        return "full_text_textual_grounding", True, []
    if modality in {
        "full_text_unrecognized_sections",
        "sectioned_source_unrecognized",
    }:
        return (
            "diagnostic_unrecognized_sections",
            False,
            ["no_recognized_methods_results_or_table_section"],
        )
    return (
        "diagnostic_title_abstract_grounding",
        False,
        ["title_or_abstract_only_not_release_grade"],
    )


def _exposed_section(family: SectionFamily) -> ExposedSection:
    mapping: dict[SectionFamily, ExposedSection] = {
        "title": "Title",
        "abstract": "Abstract",
        "methods": "Methods",
        "results": "Results",
        "table_or_figure": "FigureTable",
        # Other sections are never candidates, so this branch is defensive.
        "other": "Abstract",
    }
    return mapping[family]


def _split_exact_text(value: str, *, maximum: int) -> list[tuple[int, int, str]]:
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(value):
        sentence_end = match.start()
        sentence = value[cursor:sentence_end]
        if sentence:
            pieces.extend(_chunk_exact(sentence, base=cursor, maximum=maximum))
        cursor = match.end()
    tail = value[cursor:]
    if tail:
        pieces.extend(_chunk_exact(tail, base=cursor, maximum=maximum))
    if not pieces and value:
        pieces.extend(_chunk_exact(value, base=0, maximum=maximum))
    return [piece for piece in pieces if piece[2]]


def _chunk_exact(value: str, *, base: int, maximum: int) -> list[tuple[int, int, str]]:
    return [
        (base + start, base + min(start + maximum, len(value)), value[start : start + maximum])
        for start in range(0, len(value), maximum)
        if value[start : start + maximum]
    ]


def _hit_count(text: str, terms: Sequence[str]) -> int:
    tokens = set(_TOKEN_RE.findall(text.casefold()))
    return sum(term in tokens for term in terms)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_passages(
    source: ResolvedNativeSource,
    spec: QuestionProjectionSpecV1,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for line in source.lines:
        family = _section_family(line.section)
        if family == "other":
            continue
        for start, end, text in _split_exact_text(
            line.text,
            maximum=spec.max_passage_characters,
        ):
            hits = {
                "population_term_hits": _hit_count(text, spec.population_terms),
                "intervention_or_exposure_term_hits": _hit_count(
                    text, spec.intervention_or_exposure_terms
                ),
                "comparison_term_hits": _hit_count(text, spec.comparison_terms),
                "outcome_term_hits": _hit_count(text, spec.outcome_terms),
            }
            numerical = bool(_NUMBER_SIGNAL.search(text))
            relevant = sum(hits.values()) > 0
            if family in {"methods", "results", "table_or_figure"}:
                eligible = relevant or numerical
            else:
                # Title/Abstract are an explicit fallback, not a source of unrelated text.
                eligible = relevant or numerical
            if not eligible:
                continue
            base_score = {
                "table_or_figure": 600,
                "results": 500,
                "methods": 100,
                "abstract": 80,
                "title": 40,
            }[family]
            score = (
                base_score
                + 50 * hits["outcome_term_hits"]
                + 20 * hits["intervention_or_exposure_term_hits"]
                + 20 * hits["comparison_term_hits"]
                + 10 * hits["population_term_hits"]
                + 30 * int(numerical)
            )
            source_char_start = line.char_start + start
            source_char_end = line.char_start + end
            byte_start = line.utf8_byte_start + len(line.text[:start].encode("utf-8"))
            byte_end = line.utf8_byte_start + len(line.text[:end].encode("utf-8"))
            candidates.append(
                {
                    "line": line,
                    "section_family": family,
                    "line_char_start": start,
                    "line_char_end_exclusive": end,
                    "source_char_start": source_char_start,
                    "source_char_end_exclusive": source_char_end,
                    "source_utf8_byte_start": byte_start,
                    "source_utf8_byte_end_exclusive": byte_end,
                    "text": text,
                    "priority_score": score,
                    "numerical_signal": numerical,
                    **hits,
                }
            )
    return candidates


def _ordered_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["priority_score"]),
            int(row["line"].line_number),
            int(row["line_char_start"]),
            int(row["line_char_end_exclusive"]),
        ),
    )
    results = [
        row
        for row in ranked
        if row["section_family"] in {"results", "table_or_figure"}
    ]
    methods = [row for row in ranked if row["section_family"] == "methods"]
    fallback = [
        row for row in ranked if row["section_family"] in {"title", "abstract"}
    ]
    # Title/Abstract are retained as bounded context even when a weak Methods hit
    # exists.  They remain explicitly diagnostic in the frozen source-strength flag.
    early = [
        *results[:EARLY_RESULTS_PASSAGES],
        *fallback[:RESERVED_FALLBACK_PASSAGES],
        *methods[:RESERVED_METHODS_PASSAGES],
    ]
    early_ids = {id(row) for row in early}
    return [*early, *(row for row in ranked if id(row) not in early_ids)]


def project_resolved_source_for_question(
    *,
    row_id: str,
    source: ResolvedNativeSource,
    spec: QuestionProjectionSpecV1,
) -> FrozenSourceProjectionV1:
    """Freeze a deterministic, bounded, exact projection for one source row."""

    source = ResolvedNativeSource.model_validate(source.model_dump(mode="json"))
    spec = QuestionProjectionSpecV1.model_validate(spec.model_dump(mode="json"))
    selected: list[Mapping[str, Any]] = []
    projected_characters = 0
    seen: set[tuple[int, int, int]] = set()
    for candidate in _ordered_candidates(_candidate_passages(source, spec)):
        line: ResolvedSourceLine = candidate["line"]
        identity = (
            line.line_number,
            int(candidate["line_char_start"]),
            int(candidate["line_char_end_exclusive"]),
        )
        if identity in seen:
            continue
        text = str(candidate["text"])
        if projected_characters + len(text) > spec.max_projected_characters:
            continue
        selected.append(candidate)
        seen.add(identity)
        projected_characters += len(text)
        if len(selected) >= spec.max_projected_passages:
            break

    passages: list[FrozenProjectedPassageV1] = []
    for rank, candidate in enumerate(selected, start=1):
        line = candidate["line"]
        assert isinstance(line, ResolvedSourceLine)
        text = str(candidate["text"])
        passages.append(
            FrozenProjectedPassageV1(
                passage_rank=rank,
                line_id=line.line_id,
                line_number=line.line_number,
                section=line.section,
                section_family=candidate["section_family"],
                exposed_section=_exposed_section(candidate["section_family"]),
                line_char_start=candidate["line_char_start"],
                line_char_end_exclusive=candidate["line_char_end_exclusive"],
                source_char_start=candidate["source_char_start"],
                source_char_end_exclusive=candidate["source_char_end_exclusive"],
                source_utf8_byte_start=candidate["source_utf8_byte_start"],
                source_utf8_byte_end_exclusive=candidate[
                    "source_utf8_byte_end_exclusive"
                ],
                text=text,
                text_sha256=_sha256_text(text),
                source_line_sha256=_sha256_text(line.text),
                population_term_hits=candidate["population_term_hits"],
                intervention_or_exposure_term_hits=candidate[
                    "intervention_or_exposure_term_hits"
                ],
                comparison_term_hits=candidate["comparison_term_hits"],
                outcome_term_hits=candidate["outcome_term_hits"],
                numerical_signal=candidate["numerical_signal"],
                priority_score=candidate["priority_score"],
            )
        )

    identity_payload = {
        "row_id": row_id,
        "question_id": spec.question_id,
        "source_locator": source.source_locator,
        "artifact_sha256": source.artifact_sha256,
        "source_payload_sha256": source.source_payload_sha256,
    }
    modality = _source_modality(source)
    strength, release_grade_eligible, strength_blockers = _source_strength(
        modality=modality,
        ready=bool(passages),
    )
    payload: dict[str, Any] = {
        "source_projection_version": SOURCE_PROJECTION_VERSION,
        "projection_algorithm": PROJECTION_ALGORITHM,
        "row_id": row_id,
        "row_source_identity_sha256": hash_canonical(identity_payload),
        "question_id": spec.question_id,
        "question_spec_sha256": spec.question_spec_sha256,
        "projection_spec_sha256": spec.projection_spec_sha256,
        "source_kind": source.source_kind,
        "source_modality": modality,
        "source_strength": strength,
        "release_grade_source_grounding_eligible": release_grade_eligible,
        "source_strength_blockers": strength_blockers,
        "artifact_path": source.artifact_path,
        "artifact_sha256": source.artifact_sha256,
        "source_locator": source.source_locator,
        "source_payload_sha256": source.source_payload_sha256,
        "source_text_sha256": _sha256_text(source.source_text),
        "allowed_outcomes": spec.allowed_outcomes,
        "allowed_moderators": spec.allowed_moderators,
        "projection_status": "ready" if passages else "no_eligible_source_passage",
        "passages": passages,
        "exposed_line_ids": sorted({passage.line_id for passage in passages}),
        "exposed_sections": sorted(
            {passage.exposed_section for passage in passages}
        ),
        "projected_characters": sum(len(passage.text) for passage in passages),
    }
    return FrozenSourceProjectionV1.model_validate(
        {**payload, "projection_sha256": hash_canonical(payload)}
    )


def validate_frozen_source_projection_external_replay(
    *,
    projection: FrozenSourceProjectionV1 | Mapping[str, Any],
    source: ResolvedNativeSource | Mapping[str, Any],
    spec: QuestionProjectionSpecV1 | Mapping[str, Any],
) -> FrozenSourceProjectionV1:
    """Recompute every selected byte, offset, hit, score, and hash from source input."""

    canonical = FrozenSourceProjectionV1.model_validate(projection)
    canonical_source = ResolvedNativeSource.model_validate(source)
    canonical_spec = QuestionProjectionSpecV1.model_validate(spec)
    replayed = project_resolved_source_for_question(
        row_id=canonical.row_id,
        source=canonical_source,
        spec=canonical_spec,
    )
    if replayed != canonical:
        raise NativeQuestionProjectionError(
            "frozen_source_projection_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "MAX_PASSAGE_CHARACTERS",
    "MAX_PROJECTED_CHARACTERS",
    "MAX_PROJECTED_PASSAGES",
    "CanonicalOutcomeV1",
    "FrozenProjectedPassageV1",
    "FrozenSourceProjectionV1",
    "NativeQuestionProjectionError",
    "QuestionProjectionFieldsV1",
    "QuestionProjectionSpecV1",
    "freeze_canonical_outcomes",
    "freeze_question_projection_spec",
    "project_resolved_source_for_question",
    "validate_frozen_source_projection_external_replay",
]
