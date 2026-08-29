"""Label-blind, passage-anchored v2 views of frozen MetaSyn projections.

This module is intentionally downstream of ``FrozenSourceProjectionV1``.  It never
reopens a corpus row, benchmark reference field, review conclusion, or provider
response.  It converts the already-frozen model-facing source surface into a finer,
exactly replayable prompt surface by:

* splitting long v1 passages into exact, non-overlapping anchored segments;
* representing every occurrence of identical text while exposing that text once;
* selecting by the existing 14,000 source-character ceiling, with no passage-count
  ceiling; and
* recording every omitted segment explicitly (although a valid v1 input normally
  fits in full after deduplication).

The v1 projector and all v1 artifacts remain immutable.  A v2 object has no claim,
accuracy, eligibility, or synthesis authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_question_projection import FrozenSourceProjectionV1

PROJECTION_V2_VERSION = "metasyn-frozen-source-projection-v2"
PROJECTION_V2_ALGORITHM = "frozen-v1-exact-dedup-passage-anchor-v2"
LINEAGE_BINDING_VERSION = "metasyn-projection-v2-lineage-binding-v1"
PROMPT_RENDER_VERSION = "metasyn-projection-v2-prompt-surface-v1"
DIAGNOSTIC_VERSION = "metasyn-projection-v2-label-blind-diagnostic-v1"

MAX_SELECTED_SOURCE_CHARACTERS = 14_000
MAX_ANCHOR_TEXT_CHARACTERS = 512
MIN_SOFT_SPLIT_CHARACTERS = 256
MAX_ANCHORED_PASSAGES = 4_096
MAX_ORIGINS_PER_ANCHOR = 4_096
MAX_SECTION_CHARACTERS = 256
MAX_SOURCE_LOCATOR_CHARACTERS = 2_048
MAX_ROW_ID_CHARACTERS = 256
DEFAULT_FAILURE_STRATIFIED_ROWS = (9, 10, 18, 27, 29)

_ANCHOR_RE = re.compile(r"^p2-[0-9a-f]{64}$")
_SAFE_BREAK_RE = re.compile(r"\s")


class MetaSynProjectionV2Error(ValueError):
    """The frozen input cannot be transformed without weakening lineage."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_projection_v2_sha256_invalid:{field_name}")
    return value


class ProjectionV2LineageBinding(ContractModel):
    """Opaque parent hashes required before a v1 projection may be transformed."""

    binding_version: Literal["metasyn-projection-v2-lineage-binding-v1"] = (
        LINEAGE_BINDING_VERSION
    )
    upstream_execution_bundle_sha256: str
    upstream_row_context_sha256: str
    upstream_source_row_sha256: str
    upstream_projection_sha256: str
    row_source_identity_sha256: str
    question_spec_sha256: str
    source_payload_sha256: str
    source_text_sha256: str
    binding_sha256: str

    @field_validator(
        "upstream_execution_bundle_sha256",
        "upstream_row_context_sha256",
        "upstream_source_row_sha256",
        "upstream_projection_sha256",
        "row_source_identity_sha256",
        "question_spec_sha256",
        "source_payload_sha256",
        "source_text_sha256",
        "binding_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> ProjectionV2LineageBinding:
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if hash_canonical(payload) != self.binding_sha256:
            raise ValueError("metasyn_projection_v2_lineage_binding_hash_mismatch")
        return self


def freeze_projection_v2_lineage_binding(
    *,
    upstream_execution_bundle_sha256: str,
    upstream_row_context_sha256: str,
    upstream_source_row_sha256: str,
    projection: FrozenSourceProjectionV1 | Mapping[str, Any],
) -> ProjectionV2LineageBinding:
    """Bind a v2 transform to the enclosing immutable v1 row lineage."""

    canonical = FrozenSourceProjectionV1.model_validate(projection)
    payload = {
        "binding_version": LINEAGE_BINDING_VERSION,
        "upstream_execution_bundle_sha256": upstream_execution_bundle_sha256,
        "upstream_row_context_sha256": upstream_row_context_sha256,
        "upstream_source_row_sha256": upstream_source_row_sha256,
        "upstream_projection_sha256": canonical.projection_sha256,
        "row_source_identity_sha256": canonical.row_source_identity_sha256,
        "question_spec_sha256": canonical.question_spec_sha256,
        "source_payload_sha256": canonical.source_payload_sha256,
        "source_text_sha256": canonical.source_text_sha256,
    }
    return ProjectionV2LineageBinding.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


class PassageOriginV2(ContractModel):
    """One exact source occurrence represented by an anchored prompt passage."""

    upstream_passage_rank: Annotated[int, Field(ge=1)]
    upstream_parent_text_sha256: str
    upstream_parent_text_characters: Annotated[int, Field(ge=1)]
    parent_char_start: Annotated[int, Field(ge=0)]
    parent_char_end_exclusive: Annotated[int, Field(ge=1)]
    line_id: Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]
    line_number: Annotated[int, Field(ge=1)]
    section: Annotated[str, Field(min_length=1, max_length=MAX_SECTION_CHARACTERS)]
    section_family: Literal[
        "title", "abstract", "methods", "results", "table_or_figure", "other"
    ]
    exposed_section: Literal["Abstract", "FigureTable", "Methods", "Results", "Title"]
    line_char_start: Annotated[int, Field(ge=0)]
    line_char_end_exclusive: Annotated[int, Field(ge=1)]
    source_char_start: Annotated[int, Field(ge=0)]
    source_char_end_exclusive: Annotated[int, Field(ge=1)]
    source_utf8_byte_start: Annotated[int, Field(ge=0)]
    source_utf8_byte_end_exclusive: Annotated[int, Field(ge=1)]
    segment_text_sha256: str
    segment_characters: Annotated[int, Field(ge=1, le=MAX_ANCHOR_TEXT_CHARACTERS)]
    segment_utf8_bytes: Annotated[int, Field(ge=1)]
    source_line_sha256: str
    origin_sha256: str

    @field_validator(
        "upstream_parent_text_sha256",
        "segment_text_sha256",
        "source_line_sha256",
        "origin_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_origin(self) -> PassageOriginV2:
        if self.line_id != f"L{self.line_number}":
            raise ValueError("metasyn_projection_v2_origin_line_identity_mismatch")
        if not (
            0
            <= self.parent_char_start
            < self.parent_char_end_exclusive
            <= self.upstream_parent_text_characters
        ):
            raise ValueError("metasyn_projection_v2_origin_parent_offsets_invalid")
        if self.parent_char_end_exclusive - self.parent_char_start != self.segment_characters:
            raise ValueError("metasyn_projection_v2_origin_parent_character_count_mismatch")
        if self.line_char_end_exclusive - self.line_char_start != self.segment_characters:
            raise ValueError("metasyn_projection_v2_origin_line_character_count_mismatch")
        if self.source_char_end_exclusive - self.source_char_start != self.segment_characters:
            raise ValueError("metasyn_projection_v2_origin_source_character_count_mismatch")
        if (
            self.source_utf8_byte_end_exclusive - self.source_utf8_byte_start
            != self.segment_utf8_bytes
        ):
            raise ValueError("metasyn_projection_v2_origin_source_byte_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"origin_sha256"})
        if hash_canonical(payload) != self.origin_sha256:
            raise ValueError("metasyn_projection_v2_origin_hash_mismatch")
        return self


def _origin_sort_key(origin: PassageOriginV2) -> tuple[int, int, int, int, str]:
    return (
        origin.upstream_passage_rank,
        origin.parent_char_start,
        origin.source_utf8_byte_start,
        origin.source_utf8_byte_end_exclusive,
        origin.origin_sha256,
    )


class AnchoredPassageV2(ContractModel):
    """One exact prompt text with every duplicate occurrence bound to it."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    candidate_order: Annotated[int, Field(ge=1, le=MAX_ANCHORED_PASSAGES)]
    selection_status: Literal["selected", "omitted_character_budget"]
    prompt_rank: Annotated[int, Field(ge=1, le=MAX_ANCHORED_PASSAGES)] | None
    passage_anchor: Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]
    row_source_identity_sha256: str
    source_text_sha256: str
    upstream_projection_sha256: str
    line_ids: list[Annotated[str, Field(pattern=r"^L[1-9][0-9]*$")]]
    sections: list[Annotated[str, Field(min_length=1, max_length=MAX_SECTION_CHARACTERS)]]
    section_families: list[
        Literal["title", "abstract", "methods", "results", "table_or_figure", "other"]
    ]
    exposed_sections: list[
        Literal["Abstract", "FigureTable", "Methods", "Results", "Title"]
    ]
    source_line_sha256s: list[str]
    text: Annotated[str, Field(min_length=1, max_length=MAX_ANCHOR_TEXT_CHARACTERS)]
    text_sha256: str
    text_characters: Annotated[int, Field(ge=1, le=MAX_ANCHOR_TEXT_CHARACTERS)]
    text_utf8_bytes: Annotated[int, Field(ge=1)]
    origins: Annotated[
        list[PassageOriginV2],
        Field(min_length=1, max_length=MAX_ORIGINS_PER_ANCHOR),
    ]
    origin_count: Annotated[int, Field(ge=1, le=MAX_ORIGINS_PER_ANCHOR)]
    duplicate_occurrence_count: Annotated[int, Field(ge=0)]
    origin_set_sha256: str
    passage_lineage_sha256: str

    @field_validator(
        "row_source_identity_sha256",
        "source_text_sha256",
        "upstream_projection_sha256",
        "text_sha256",
        "origin_set_sha256",
        "passage_lineage_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator(
        "line_ids",
        "sections",
        "section_families",
        "exposed_sections",
        "source_line_sha256s",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if not value or value != sorted(set(value)):
            raise ValueError(
                f"metasyn_projection_v2_passage_values_not_sorted_unique:{info.field_name}"
            )
        if info.field_name == "source_line_sha256s":
            for item in value:
                _validate_sha256(item, field_name=info.field_name)
        return value

    @model_validator(mode="after")
    def validate_passage(self) -> AnchoredPassageV2:
        if self.selection_status == "selected" and self.prompt_rank is None:
            raise ValueError("metasyn_projection_v2_selected_prompt_rank_missing")
        if self.selection_status != "selected" and self.prompt_rank is not None:
            raise ValueError("metasyn_projection_v2_omitted_prompt_rank_present")
        if _sha256_text(self.text) != self.text_sha256:
            raise ValueError("metasyn_projection_v2_passage_text_hash_mismatch")
        if len(self.text) != self.text_characters:
            raise ValueError("metasyn_projection_v2_passage_text_character_count_mismatch")
        if len(self.text.encode("utf-8")) != self.text_utf8_bytes:
            raise ValueError("metasyn_projection_v2_passage_text_byte_count_mismatch")
        if self.origins != sorted(self.origins, key=_origin_sort_key):
            raise ValueError("metasyn_projection_v2_passage_origins_not_canonical")
        if len({item.origin_sha256 for item in self.origins}) != len(self.origins):
            raise ValueError("metasyn_projection_v2_passage_origin_duplicate")
        if self.origin_count != len(self.origins):
            raise ValueError("metasyn_projection_v2_passage_origin_count_mismatch")
        if self.duplicate_occurrence_count != self.origin_count - 1:
            raise ValueError("metasyn_projection_v2_passage_duplicate_count_mismatch")
        expected_metadata = {
            "line_ids": sorted({item.line_id for item in self.origins}),
            "sections": sorted({item.section for item in self.origins}),
            "section_families": sorted({item.section_family for item in self.origins}),
            "exposed_sections": sorted({item.exposed_section for item in self.origins}),
            "source_line_sha256s": sorted(
                {item.source_line_sha256 for item in self.origins}
            ),
        }
        if any(getattr(self, key) != value for key, value in expected_metadata.items()):
            raise ValueError("metasyn_projection_v2_passage_origin_metadata_mismatch")
        for origin in self.origins:
            if (
                origin.segment_text_sha256 != self.text_sha256
                or origin.segment_characters != self.text_characters
                or origin.segment_utf8_bytes != self.text_utf8_bytes
            ):
                raise ValueError("metasyn_projection_v2_passage_origin_text_mismatch")
        origin_payload = [item.model_dump(mode="json") for item in self.origins]
        if hash_canonical(origin_payload) != self.origin_set_sha256:
            raise ValueError("metasyn_projection_v2_passage_origin_set_hash_mismatch")
        anchor_payload = {
            "anchor_version": "metasyn-passage-anchor-v2",
            "row_source_identity_sha256": self.row_source_identity_sha256,
            "source_text_sha256": self.source_text_sha256,
            "text_sha256": self.text_sha256,
        }
        expected_anchor = f"p2-{hash_canonical(anchor_payload)}"
        if self.passage_anchor != expected_anchor or not _ANCHOR_RE.fullmatch(
            self.passage_anchor
        ):
            raise ValueError("metasyn_projection_v2_passage_anchor_mismatch")
        lineage_payload = self.model_dump(mode="json", exclude={"passage_lineage_sha256"})
        if hash_canonical(lineage_payload) != self.passage_lineage_sha256:
            raise ValueError("metasyn_projection_v2_passage_lineage_hash_mismatch")
        return self


def _render_prompt_surface(passages: Sequence[AnchoredPassageV2]) -> str:
    selected = sorted(
        (item for item in passages if item.selection_status == "selected"),
        key=lambda item: (item.prompt_rank or 0, item.passage_anchor),
    )
    blocks: list[str] = []
    for passage in selected:
        blocks.append(
            "\n".join(
                (
                    f"PASSAGE_ANCHOR: {passage.passage_anchor}",
                    f"PASSAGE_LINEAGE_SHA256: {passage.passage_lineage_sha256}",
                    f"SECTION_ENUMS: {','.join(passage.exposed_sections)}",
                    f"UPSTREAM_LINE_IDS: {','.join(passage.line_ids)}",
                    f"EXACT_SOURCE_OCCURRENCE_COUNT: {passage.origin_count}",
                    "BEGIN_EXACT_SOURCE_TEXT",
                    passage.text,
                    "END_EXACT_SOURCE_TEXT",
                )
            )
        )
    return "\n\n".join(blocks)


class FrozenMetaSynProjectionV2(ContractModel):
    """A self-hashed, fully accounted v2 prompt surface."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    projection_version: Literal["metasyn-frozen-source-projection-v2"] = (
        PROJECTION_V2_VERSION
    )
    projection_algorithm: Literal["frozen-v1-exact-dedup-passage-anchor-v2"] = (
        PROJECTION_V2_ALGORITHM
    )
    prompt_render_version: Literal["metasyn-projection-v2-prompt-surface-v1"] = (
        PROMPT_RENDER_VERSION
    )
    selection_policy: Literal[
        "v1-rank-then-exact-segment-character-budget-only-no-passage-cap"
    ] = "v1-rank-then-exact-segment-character-budget-only-no-passage-cap"
    scientific_authority: Literal["source_surface_diagnostic_only"] = (
        "source_surface_diagnostic_only"
    )
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    row_id: Annotated[str, Field(min_length=1, max_length=MAX_ROW_ID_CHARACTERS)]
    question_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_locator: Annotated[
        str, Field(min_length=1, max_length=MAX_SOURCE_LOCATOR_CHARACTERS)
    ]
    lineage_binding: ProjectionV2LineageBinding
    lineage_binding_sha256: str
    max_selected_source_characters: Literal[14000] = MAX_SELECTED_SOURCE_CHARACTERS
    max_anchor_text_characters: Literal[512] = MAX_ANCHOR_TEXT_CHARACTERS
    upstream_passage_count: Annotated[int, Field(ge=0)]
    upstream_projected_characters: Annotated[int, Field(ge=0)]
    unique_upstream_parent_text_count: Annotated[int, Field(ge=0)]
    exact_duplicate_parent_passages_removed: Annotated[int, Field(ge=0)]
    exact_duplicate_parent_characters_removed: Annotated[int, Field(ge=0)]
    expanded_origin_segment_count: Annotated[int, Field(ge=0)]
    anchored_passage_count: Annotated[int, Field(ge=0, le=MAX_ANCHORED_PASSAGES)]
    selected_passage_count: Annotated[int, Field(ge=0, le=MAX_ANCHORED_PASSAGES)]
    omitted_passage_count: Annotated[int, Field(ge=0, le=MAX_ANCHORED_PASSAGES)]
    selected_source_characters: Annotated[
        int, Field(ge=0, le=MAX_SELECTED_SOURCE_CHARACTERS)
    ]
    omitted_source_characters: Annotated[int, Field(ge=0)]
    unspent_source_character_budget: Annotated[
        int, Field(ge=0, le=MAX_SELECTED_SOURCE_CHARACTERS)
    ]
    selection_complete: bool
    all_upstream_occurrences_bound: Literal[True] = True
    all_exact_duplicate_parent_passages_deduplicated: Literal[True] = True
    expanded_beyond_v1_passage_cap: bool
    selected_passage_anchors: list[str]
    omitted_passage_anchors: list[str]
    passages: Annotated[list[AnchoredPassageV2], Field(max_length=MAX_ANCHORED_PASSAGES)]
    prompt_source_characters: Annotated[int, Field(ge=0)]
    prompt_source_sha256: str
    projection_sha256: str

    @field_validator(
        "lineage_binding_sha256",
        "prompt_source_sha256",
        "projection_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @field_validator("selected_passage_anchors", "omitted_passage_anchors")
    @classmethod
    def validate_anchor_list(cls, value: list[str], info: Any) -> list[str]:
        if len(value) != len(set(value)) or any(not _ANCHOR_RE.fullmatch(item) for item in value):
            raise ValueError(f"metasyn_projection_v2_anchor_list_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> FrozenMetaSynProjectionV2:
        if self.lineage_binding_sha256 != self.lineage_binding.binding_sha256:
            raise ValueError("metasyn_projection_v2_binding_alias_mismatch")
        if self.passages != sorted(self.passages, key=lambda item: item.candidate_order):
            raise ValueError("metasyn_projection_v2_passages_not_canonical")
        if [item.candidate_order for item in self.passages] != list(
            range(1, len(self.passages) + 1)
        ):
            raise ValueError("metasyn_projection_v2_candidate_orders_invalid")
        if len({item.passage_anchor for item in self.passages}) != len(self.passages):
            raise ValueError("metasyn_projection_v2_passage_anchor_duplicate")
        if len({item.text_sha256 for item in self.passages}) != len(self.passages):
            raise ValueError("metasyn_projection_v2_prompt_passage_duplicate")
        selected = [item for item in self.passages if item.selection_status == "selected"]
        omitted = [
            item for item in self.passages if item.selection_status == "omitted_character_budget"
        ]
        if [item.prompt_rank for item in selected] != list(range(1, len(selected) + 1)):
            raise ValueError("metasyn_projection_v2_prompt_ranks_invalid")
        if self.selected_passage_anchors != [item.passage_anchor for item in selected]:
            raise ValueError("metasyn_projection_v2_selected_anchor_membership_mismatch")
        if self.omitted_passage_anchors != [item.passage_anchor for item in omitted]:
            raise ValueError("metasyn_projection_v2_omitted_anchor_membership_mismatch")
        if self.anchored_passage_count != len(self.passages):
            raise ValueError("metasyn_projection_v2_anchored_count_mismatch")
        if self.selected_passage_count != len(selected):
            raise ValueError("metasyn_projection_v2_selected_count_mismatch")
        if self.omitted_passage_count != len(omitted):
            raise ValueError("metasyn_projection_v2_omitted_count_mismatch")
        if self.selected_source_characters != sum(item.text_characters for item in selected):
            raise ValueError("metasyn_projection_v2_selected_character_count_mismatch")
        if self.omitted_source_characters != sum(item.text_characters for item in omitted):
            raise ValueError("metasyn_projection_v2_omitted_character_count_mismatch")
        if self.unspent_source_character_budget != (
            self.max_selected_source_characters - self.selected_source_characters
        ):
            raise ValueError("metasyn_projection_v2_unspent_budget_mismatch")
        if self.selection_complete != (not omitted):
            raise ValueError("metasyn_projection_v2_selection_complete_mismatch")
        if self.expanded_beyond_v1_passage_cap != (len(selected) > 24):
            raise ValueError("metasyn_projection_v2_expansion_flag_mismatch")

        origins = [origin for passage in self.passages for origin in passage.origins]
        if self.expanded_origin_segment_count != len(origins):
            raise ValueError("metasyn_projection_v2_origin_segment_count_mismatch")
        by_parent: dict[int, list[PassageOriginV2]] = defaultdict(list)
        for origin in origins:
            by_parent[origin.upstream_passage_rank].append(origin)
        if self.upstream_passage_count != len(by_parent):
            raise ValueError("metasyn_projection_v2_upstream_passage_count_mismatch")
        parent_hashes: dict[str, tuple[int, int]] = {}
        upstream_characters = 0
        for rank, parent_origins in sorted(by_parent.items()):
            lengths = {item.upstream_parent_text_characters for item in parent_origins}
            hashes = {item.upstream_parent_text_sha256 for item in parent_origins}
            if len(lengths) != 1 or len(hashes) != 1:
                raise ValueError("metasyn_projection_v2_parent_lineage_ambiguous")
            parent_length = next(iter(lengths))
            parent_hash = next(iter(hashes))
            segments = sorted(
                (item.parent_char_start, item.parent_char_end_exclusive)
                for item in parent_origins
            )
            cursor = 0
            for start, end in segments:
                if start != cursor:
                    raise ValueError(
                        f"metasyn_projection_v2_parent_character_coverage_gap:{rank}"
                    )
                cursor = end
            if cursor != parent_length:
                raise ValueError(
                    f"metasyn_projection_v2_parent_character_coverage_incomplete:{rank}"
                )
            upstream_characters += parent_length
            current = parent_hashes.get(parent_hash)
            if current is None:
                parent_hashes[parent_hash] = (parent_length, rank)
            elif current[0] != parent_length:
                raise ValueError("metasyn_projection_v2_parent_hash_length_conflict")
        if self.upstream_projected_characters != upstream_characters:
            raise ValueError("metasyn_projection_v2_upstream_character_count_mismatch")
        if self.unique_upstream_parent_text_count != len(parent_hashes):
            raise ValueError("metasyn_projection_v2_unique_parent_count_mismatch")
        if self.exact_duplicate_parent_passages_removed != (
            self.upstream_passage_count - len(parent_hashes)
        ):
            raise ValueError("metasyn_projection_v2_removed_parent_count_mismatch")
        unique_parent_characters = sum(item[0] for item in parent_hashes.values())
        if self.exact_duplicate_parent_characters_removed != (
            upstream_characters - unique_parent_characters
        ):
            raise ValueError("metasyn_projection_v2_removed_parent_characters_mismatch")

        rendered = _render_prompt_surface(self.passages)
        if len(rendered) != self.prompt_source_characters:
            raise ValueError("metasyn_projection_v2_prompt_character_count_mismatch")
        if _sha256_text(rendered) != self.prompt_source_sha256:
            raise ValueError("metasyn_projection_v2_prompt_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_sha256"})
        if hash_canonical(payload) != self.projection_sha256:
            raise ValueError("metasyn_projection_v2_projection_hash_mismatch")
        return self


def _split_exact_for_anchor(value: str) -> list[tuple[int, int, str]]:
    """Partition text exactly, preferring whitespace without deleting it."""

    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(value):
        hard_end = min(cursor + MAX_ANCHOR_TEXT_CHARACTERS, len(value))
        end = hard_end
        if hard_end < len(value):
            window = value[cursor:hard_end]
            candidates = [match.end() for match in _SAFE_BREAK_RE.finditer(window)]
            safe = [point for point in candidates if point >= MIN_SOFT_SPLIT_CHARACTERS]
            if safe:
                end = cursor + safe[-1]
        text = value[cursor:end]
        if not text:
            raise MetaSynProjectionV2Error("metasyn_projection_v2_empty_exact_segment")
        pieces.append((cursor, end, text))
        cursor = end
    if "".join(item[2] for item in pieces) != value:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_exact_partition_mismatch")
    return pieces


def _freeze_origin(*, passage: Any, start: int, end: int, text: str) -> PassageOriginV2:
    byte_prefix = len(passage.text[:start].encode("utf-8"))
    byte_length = len(text.encode("utf-8"))
    payload = {
        "upstream_passage_rank": passage.passage_rank,
        "upstream_parent_text_sha256": passage.text_sha256,
        "upstream_parent_text_characters": len(passage.text),
        "parent_char_start": start,
        "parent_char_end_exclusive": end,
        "line_id": passage.line_id,
        "line_number": passage.line_number,
        "section": passage.section,
        "section_family": passage.section_family,
        "exposed_section": passage.exposed_section,
        "line_char_start": passage.line_char_start + start,
        "line_char_end_exclusive": passage.line_char_start + end,
        "source_char_start": passage.source_char_start + start,
        "source_char_end_exclusive": passage.source_char_start + end,
        "source_utf8_byte_start": passage.source_utf8_byte_start + byte_prefix,
        "source_utf8_byte_end_exclusive": (
            passage.source_utf8_byte_start + byte_prefix + byte_length
        ),
        "segment_text_sha256": _sha256_text(text),
        "segment_characters": len(text),
        "segment_utf8_bytes": byte_length,
        "source_line_sha256": passage.source_line_sha256,
    }
    return PassageOriginV2.model_validate(
        {**payload, "origin_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_projection_v2(
    *,
    projection: FrozenSourceProjectionV1 | Mapping[str, Any],
    lineage_binding: ProjectionV2LineageBinding | Mapping[str, Any],
) -> FrozenMetaSynProjectionV2:
    """Deduplicate and passage-anchor one immutable MetaSyn v1 projection."""

    upstream = FrozenSourceProjectionV1.model_validate(projection)
    binding = ProjectionV2LineageBinding.model_validate(lineage_binding)
    if upstream.source_kind != "metasyn_parquet_row":
        raise MetaSynProjectionV2Error("metasyn_projection_v2_source_kind_forbidden")
    expected = {
        "upstream_projection_sha256": upstream.projection_sha256,
        "row_source_identity_sha256": upstream.row_source_identity_sha256,
        "question_spec_sha256": upstream.question_spec_sha256,
        "source_payload_sha256": upstream.source_payload_sha256,
        "source_text_sha256": upstream.source_text_sha256,
    }
    for field_name, value in expected.items():
        if getattr(binding, field_name) != value:
            raise MetaSynProjectionV2Error(
                f"metasyn_projection_v2_lineage_binding_mismatch:{field_name}"
            )

    # Hash buckets are verified against exact text to fail closed even under an
    # impossible digest collision or a coherently malformed caller mapping.
    grouped: dict[str, list[tuple[str, PassageOriginV2, tuple[int, int]]]] = defaultdict(list)
    parent_text_by_hash: dict[str, str] = {}
    for passage in upstream.passages:
        existing_parent = parent_text_by_hash.get(passage.text_sha256)
        if existing_parent is not None and existing_parent != passage.text:
            raise MetaSynProjectionV2Error("metasyn_projection_v2_parent_hash_collision")
        parent_text_by_hash[passage.text_sha256] = passage.text
        for start, end, text in _split_exact_for_anchor(passage.text):
            text_sha = _sha256_text(text)
            origin = _freeze_origin(
                passage=passage,
                start=start,
                end=end,
                text=text,
            )
            grouped[text_sha].append(
                (
                    text,
                    origin,
                    (passage.passage_rank, start),
                )
            )

    candidates: list[dict[str, Any]] = []
    for text_sha, rows in grouped.items():
        texts = {item[0] for item in rows}
        if len(texts) != 1:
            raise MetaSynProjectionV2Error("metasyn_projection_v2_segment_hash_collision")
        origins = sorted((item[1] for item in rows), key=_origin_sort_key)
        first_order = min(item[2] for item in rows)
        candidates.append(
            {
                "text": next(iter(texts)),
                "text_sha256": text_sha,
                "origins": origins,
                "first_order": first_order,
            }
        )
    candidates.sort(key=lambda item: (item["first_order"], item["text_sha256"]))
    if len(candidates) > MAX_ANCHORED_PASSAGES:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_anchor_count_exceeds_cap")

    passages: list[AnchoredPassageV2] = []
    selected_characters = 0
    prompt_rank = 0
    for candidate_order, candidate in enumerate(candidates, start=1):
        text = str(candidate["text"])
        origins = list(candidate["origins"])
        selected = selected_characters + len(text) <= MAX_SELECTED_SOURCE_CHARACTERS
        if selected:
            selected_characters += len(text)
            prompt_rank += 1
        anchor_payload = {
            "anchor_version": "metasyn-passage-anchor-v2",
            "row_source_identity_sha256": upstream.row_source_identity_sha256,
            "source_text_sha256": upstream.source_text_sha256,
            "text_sha256": candidate["text_sha256"],
        }
        origin_payload = [item.model_dump(mode="json") for item in origins]
        base: dict[str, Any] = {
            "candidate_order": candidate_order,
            "selection_status": (
                "selected" if selected else "omitted_character_budget"
            ),
            "prompt_rank": prompt_rank if selected else None,
            "passage_anchor": f"p2-{hash_canonical(anchor_payload)}",
            "row_source_identity_sha256": upstream.row_source_identity_sha256,
            "source_text_sha256": upstream.source_text_sha256,
            "upstream_projection_sha256": upstream.projection_sha256,
            "line_ids": sorted({item.line_id for item in origins}),
            "sections": sorted({item.section for item in origins}),
            "section_families": sorted({item.section_family for item in origins}),
            "exposed_sections": sorted({item.exposed_section for item in origins}),
            "source_line_sha256s": sorted(
                {item.source_line_sha256 for item in origins}
            ),
            "text": text,
            "text_sha256": candidate["text_sha256"],
            "text_characters": len(text),
            "text_utf8_bytes": len(text.encode("utf-8")),
            "origins": origins,
            "origin_count": len(origins),
            "duplicate_occurrence_count": len(origins) - 1,
            "origin_set_sha256": hash_canonical(origin_payload),
        }
        passages.append(
            AnchoredPassageV2.model_validate(
                {**base, "passage_lineage_sha256": hash_canonical(base)}
            )
        )

    selected = [item for item in passages if item.selection_status == "selected"]
    omitted = [
        item for item in passages if item.selection_status == "omitted_character_budget"
    ]
    parent_hashes: dict[str, int] = {}
    for item in upstream.passages:
        current = parent_hashes.get(item.text_sha256)
        if current is not None and current != len(item.text):
            raise MetaSynProjectionV2Error("metasyn_projection_v2_parent_hash_length_conflict")
        parent_hashes[item.text_sha256] = len(item.text)
    upstream_characters = sum(len(item.text) for item in upstream.passages)
    unique_parent_characters = sum(parent_hashes.values())
    rendered = _render_prompt_surface(passages)
    payload: dict[str, Any] = {
        "projection_version": PROJECTION_V2_VERSION,
        "projection_algorithm": PROJECTION_V2_ALGORITHM,
        "prompt_render_version": PROMPT_RENDER_VERSION,
        "selection_policy": (
            "v1-rank-then-exact-segment-character-budget-only-no-passage-cap"
        ),
        "scientific_authority": "source_surface_diagnostic_only",
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
        "row_id": upstream.row_id,
        "question_id": upstream.question_id,
        "source_locator": upstream.source_locator,
        "lineage_binding": binding,
        "lineage_binding_sha256": binding.binding_sha256,
        "max_selected_source_characters": MAX_SELECTED_SOURCE_CHARACTERS,
        "max_anchor_text_characters": MAX_ANCHOR_TEXT_CHARACTERS,
        "upstream_passage_count": len(upstream.passages),
        "upstream_projected_characters": upstream_characters,
        "unique_upstream_parent_text_count": len(parent_hashes),
        "exact_duplicate_parent_passages_removed": (
            len(upstream.passages) - len(parent_hashes)
        ),
        "exact_duplicate_parent_characters_removed": (
            upstream_characters - unique_parent_characters
        ),
        "expanded_origin_segment_count": sum(
            len(item.origins) for item in passages
        ),
        "anchored_passage_count": len(passages),
        "selected_passage_count": len(selected),
        "omitted_passage_count": len(omitted),
        "selected_source_characters": sum(item.text_characters for item in selected),
        "omitted_source_characters": sum(item.text_characters for item in omitted),
        "unspent_source_character_budget": (
            MAX_SELECTED_SOURCE_CHARACTERS
            - sum(item.text_characters for item in selected)
        ),
        "selection_complete": not omitted,
        "all_upstream_occurrences_bound": True,
        "all_exact_duplicate_parent_passages_deduplicated": True,
        "expanded_beyond_v1_passage_cap": len(selected) > 24,
        "selected_passage_anchors": [item.passage_anchor for item in selected],
        "omitted_passage_anchors": [item.passage_anchor for item in omitted],
        "passages": passages,
        "prompt_source_characters": len(rendered),
        "prompt_source_sha256": _sha256_text(rendered),
    }
    return FrozenMetaSynProjectionV2.model_validate(
        {**payload, "projection_sha256": hash_canonical(payload)}
    )


def render_metasyn_projection_v2_prompt_surface(
    projection: FrozenMetaSynProjectionV2 | Mapping[str, Any],
) -> str:
    """Return the exact anchored source surface whose hash is in the v2 object."""

    canonical = FrozenMetaSynProjectionV2.model_validate(projection)
    rendered = _render_prompt_surface(canonical.passages)
    if _sha256_text(rendered) != canonical.prompt_source_sha256:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_prompt_external_hash_mismatch")
    return rendered


def validate_metasyn_projection_v2_external_replay(
    *,
    projection_v2: FrozenMetaSynProjectionV2 | Mapping[str, Any],
    projection_v1: FrozenSourceProjectionV1 | Mapping[str, Any],
    lineage_binding: ProjectionV2LineageBinding | Mapping[str, Any],
) -> FrozenMetaSynProjectionV2:
    """Rebuild all segments, anchors, omissions, prompt bytes, and hashes."""

    canonical = FrozenMetaSynProjectionV2.model_validate(projection_v2)
    replayed = freeze_metasyn_projection_v2(
        projection=projection_v1,
        lineage_binding=lineage_binding,
    )
    if replayed != canonical:
        raise MetaSynProjectionV2Error(
            "metasyn_projection_v2_external_replay_mismatch"
        )
    return canonical


class ProjectionV2RowDiagnostic(ContractModel):
    row_ordinal: Annotated[int, Field(ge=0)]
    row_id: str
    upstream_projection_sha256: str
    projection_v2_sha256: str
    upstream_passage_count: Annotated[int, Field(ge=0)]
    unique_upstream_parent_text_count: Annotated[int, Field(ge=0)]
    exact_duplicate_parent_passages_removed: Annotated[int, Field(ge=0)]
    anchored_passage_count: Annotated[int, Field(ge=0)]
    selected_passage_count: Annotated[int, Field(ge=0)]
    omitted_passage_count: Annotated[int, Field(ge=0)]
    expanded_beyond_v1_passage_cap: bool
    target_surface_definition: Literal[
        "upstream_numerical_signal_and_outcome_term_hits_gt_zero"
    ] = "upstream_numerical_signal_and_outcome_term_hits_gt_zero"
    target_surface_parent_passage_count: Annotated[int, Field(ge=0)]
    retained_target_surface_parent_passage_count: Annotated[int, Field(ge=0)]
    target_surface_source_characters: Annotated[int, Field(ge=0)]
    retained_target_surface_source_characters: Annotated[int, Field(ge=0)]
    all_target_surface_source_characters_retained: bool
    target_surface_retention_is_accuracy: Literal[False] = False
    row_diagnostic_sha256: str

    @field_validator(
        "upstream_projection_sha256",
        "projection_v2_sha256",
        "row_diagnostic_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_row(self) -> ProjectionV2RowDiagnostic:
        if (
            self.retained_target_surface_parent_passage_count
            > self.target_surface_parent_passage_count
        ):
            raise ValueError("metasyn_projection_v2_diagnostic_parent_retention_invalid")
        if self.retained_target_surface_source_characters > self.target_surface_source_characters:
            raise ValueError("metasyn_projection_v2_diagnostic_character_retention_invalid")
        expected = (
            self.retained_target_surface_source_characters
            == self.target_surface_source_characters
        )
        if self.all_target_surface_source_characters_retained != expected:
            raise ValueError("metasyn_projection_v2_diagnostic_retention_flag_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_diagnostic_sha256"})
        if hash_canonical(payload) != self.row_diagnostic_sha256:
            raise ValueError("metasyn_projection_v2_row_diagnostic_hash_mismatch")
        return self


class ProjectionV2DiagnosticReport(ContractModel):
    diagnostic_version: Literal["metasyn-projection-v2-label-blind-diagnostic-v1"] = (
        DIAGNOSTIC_VERSION
    )
    status: Literal["valid_label_blind_projection_surface_diagnostic"] = (
        "valid_label_blind_projection_surface_diagnostic"
    )
    source_scope: Literal["frozen_v1_projection_only"] = "frozen_v1_projection_only"
    official_test_labels_opened: Literal[False] = False
    reference_fields_unopened: Literal[True] = True
    extraction_accuracy_reported: Literal[False] = False
    claim_release_authority: Literal[False] = False
    synthesis_authority: Literal[False] = False
    target_surface_retention_is_accuracy: Literal[False] = False
    execution_bundle_sha256: str
    row_ordinals: list[Annotated[int, Field(ge=0)]]
    rows: list[ProjectionV2RowDiagnostic]
    total_upstream_passages: Annotated[int, Field(ge=0)]
    total_exact_duplicate_parent_passages_removed: Annotated[int, Field(ge=0)]
    total_anchored_passages: Annotated[int, Field(ge=0)]
    total_selected_passages: Annotated[int, Field(ge=0)]
    total_omitted_passages: Annotated[int, Field(ge=0)]
    rows_with_selection_complete: Annotated[int, Field(ge=0)]
    rows_with_all_target_surface_characters_retained: Annotated[int, Field(ge=0)]
    report_sha256: str

    @field_validator("execution_bundle_sha256", "report_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_report(self) -> ProjectionV2DiagnosticReport:
        if self.row_ordinals != sorted(set(self.row_ordinals)):
            raise ValueError("metasyn_projection_v2_diagnostic_rows_not_sorted_unique")
        if [item.row_ordinal for item in self.rows] != self.row_ordinals:
            raise ValueError("metasyn_projection_v2_diagnostic_row_membership_mismatch")
        totals = {
            "total_upstream_passages": sum(item.upstream_passage_count for item in self.rows),
            "total_exact_duplicate_parent_passages_removed": sum(
                item.exact_duplicate_parent_passages_removed for item in self.rows
            ),
            "total_anchored_passages": sum(item.anchored_passage_count for item in self.rows),
            "total_selected_passages": sum(item.selected_passage_count for item in self.rows),
            "total_omitted_passages": sum(item.omitted_passage_count for item in self.rows),
            "rows_with_selection_complete": sum(
                item.omitted_passage_count == 0 for item in self.rows
            ),
            "rows_with_all_target_surface_characters_retained": sum(
                item.all_target_surface_source_characters_retained for item in self.rows
            ),
        }
        for field_name, expected in totals.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"metasyn_projection_v2_diagnostic_total_mismatch:{field_name}"
                )
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("metasyn_projection_v2_diagnostic_report_hash_mismatch")
        return self


def _canonical_execution_bundle(value: Mapping[str, Any]) -> Mapping[str, Any]:
    bundle_sha = value.get("execution_bundle_sha256")
    if not isinstance(bundle_sha, str) or not SHA256_RE.fullmatch(bundle_sha):
        raise MetaSynProjectionV2Error("metasyn_projection_v2_execution_bundle_hash_missing")
    payload = {key: item for key, item in value.items() if key != "execution_bundle_sha256"}
    if hash_canonical(payload) != bundle_sha:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_execution_bundle_hash_mismatch")
    if value.get("official_test_labels_opened") is not False:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_official_labels_boundary_invalid")
    if value.get("reference_fields_unopened") is not True:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_reference_boundary_invalid")
    return value


def _selected_parent_coverage(projection: FrozenMetaSynProjectionV2) -> dict[int, int]:
    coverage: dict[int, int] = defaultdict(int)
    for passage in projection.passages:
        if passage.selection_status != "selected":
            continue
        for origin in passage.origins:
            coverage[origin.upstream_passage_rank] += origin.segment_characters
    return dict(coverage)


def diagnose_execution_bundle_projection_v2(
    *,
    execution_bundle: Mapping[str, Any],
    row_ordinals: Sequence[int] = DEFAULT_FAILURE_STRATIFIED_ROWS,
) -> ProjectionV2DiagnosticReport:
    """Run a projection-only diagnostic without opening reference or output labels."""

    bundle = _canonical_execution_bundle(execution_bundle)
    ordinals = list(row_ordinals)
    if ordinals != sorted(set(ordinals)) or not ordinals:
        raise MetaSynProjectionV2Error("metasyn_projection_v2_diagnostic_rows_invalid")
    adapter_bundle = bundle.get("adapter_bundle")
    if not isinstance(adapter_bundle, Mapping):
        raise MetaSynProjectionV2Error("metasyn_projection_v2_adapter_bundle_missing")
    contexts = adapter_bundle.get("row_contexts")
    if not isinstance(contexts, list):
        raise MetaSynProjectionV2Error("metasyn_projection_v2_row_contexts_missing")

    rows: list[ProjectionV2RowDiagnostic] = []
    for ordinal in ordinals:
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < len(contexts)
        ):
            raise MetaSynProjectionV2Error(
                f"metasyn_projection_v2_row_ordinal_out_of_bounds:{ordinal}"
            )
        context = contexts[ordinal]
        if not isinstance(context, Mapping):
            raise MetaSynProjectionV2Error("metasyn_projection_v2_row_context_invalid")
        source_row = context.get("source_row")
        if not isinstance(source_row, Mapping):
            raise MetaSynProjectionV2Error("metasyn_projection_v2_source_row_missing")
        projection_value = source_row.get("projection")
        if not isinstance(projection_value, Mapping):
            raise MetaSynProjectionV2Error("metasyn_projection_v2_upstream_projection_missing")
        upstream = FrozenSourceProjectionV1.model_validate(projection_value)
        if source_row.get("projection_sha256") != upstream.projection_sha256:
            raise MetaSynProjectionV2Error(
                "metasyn_projection_v2_source_row_projection_alias_mismatch"
            )
        if context.get("projection_sha256") != upstream.projection_sha256:
            raise MetaSynProjectionV2Error(
                "metasyn_projection_v2_context_projection_alias_mismatch"
            )
        row_context_sha = context.get("row_context_sha256")
        source_row_sha = source_row.get("source_row_sha256")
        if not isinstance(row_context_sha, str) or not isinstance(source_row_sha, str):
            raise MetaSynProjectionV2Error("metasyn_projection_v2_parent_lineage_hash_missing")
        binding = freeze_projection_v2_lineage_binding(
            upstream_execution_bundle_sha256=str(bundle["execution_bundle_sha256"]),
            upstream_row_context_sha256=row_context_sha,
            upstream_source_row_sha256=source_row_sha,
            projection=upstream,
        )
        v2 = freeze_metasyn_projection_v2(
            projection=upstream,
            lineage_binding=binding,
        )
        validate_metasyn_projection_v2_external_replay(
            projection_v2=v2,
            projection_v1=upstream,
            lineage_binding=binding,
        )

        target_ranks = {
            item.passage_rank
            for item in upstream.passages
            if item.numerical_signal and item.outcome_term_hits > 0
        }
        target_lengths = {
            item.passage_rank: len(item.text)
            for item in upstream.passages
            if item.passage_rank in target_ranks
        }
        selected_coverage = _selected_parent_coverage(v2)
        retained_ranks = {
            rank
            for rank, length in target_lengths.items()
            if selected_coverage.get(rank, 0) == length
        }
        row_payload: dict[str, Any] = {
            "row_ordinal": ordinal,
            "row_id": upstream.row_id,
            "upstream_projection_sha256": upstream.projection_sha256,
            "projection_v2_sha256": v2.projection_sha256,
            "upstream_passage_count": v2.upstream_passage_count,
            "unique_upstream_parent_text_count": v2.unique_upstream_parent_text_count,
            "exact_duplicate_parent_passages_removed": (
                v2.exact_duplicate_parent_passages_removed
            ),
            "anchored_passage_count": v2.anchored_passage_count,
            "selected_passage_count": v2.selected_passage_count,
            "omitted_passage_count": v2.omitted_passage_count,
            "expanded_beyond_v1_passage_cap": v2.expanded_beyond_v1_passage_cap,
            "target_surface_definition": (
                "upstream_numerical_signal_and_outcome_term_hits_gt_zero"
            ),
            "target_surface_parent_passage_count": len(target_ranks),
            "retained_target_surface_parent_passage_count": len(retained_ranks),
            "target_surface_source_characters": sum(target_lengths.values()),
            "retained_target_surface_source_characters": sum(
                target_lengths[rank] for rank in retained_ranks
            ),
            "all_target_surface_source_characters_retained": (
                retained_ranks == target_ranks
            ),
            "target_surface_retention_is_accuracy": False,
        }
        rows.append(
            ProjectionV2RowDiagnostic.model_validate(
                {
                    **row_payload,
                    "row_diagnostic_sha256": hash_canonical(row_payload),
                }
            )
        )

    payload: dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "valid_label_blind_projection_surface_diagnostic",
        "source_scope": "frozen_v1_projection_only",
        "official_test_labels_opened": False,
        "reference_fields_unopened": True,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "synthesis_authority": False,
        "target_surface_retention_is_accuracy": False,
        "execution_bundle_sha256": bundle["execution_bundle_sha256"],
        "row_ordinals": ordinals,
        "rows": rows,
        "total_upstream_passages": sum(item.upstream_passage_count for item in rows),
        "total_exact_duplicate_parent_passages_removed": sum(
            item.exact_duplicate_parent_passages_removed for item in rows
        ),
        "total_anchored_passages": sum(item.anchored_passage_count for item in rows),
        "total_selected_passages": sum(item.selected_passage_count for item in rows),
        "total_omitted_passages": sum(item.omitted_passage_count for item in rows),
        "rows_with_selection_complete": sum(
            item.omitted_passage_count == 0 for item in rows
        ),
        "rows_with_all_target_surface_characters_retained": sum(
            item.all_target_surface_source_characters_retained for item in rows
        ),
    }
    return ProjectionV2DiagnosticReport.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def load_and_diagnose_execution_bundle_projection_v2(
    path: Path,
    *,
    row_ordinals: Sequence[int] = DEFAULT_FAILURE_STRATIFIED_ROWS,
) -> ProjectionV2DiagnosticReport:
    """Read one private execution bundle and return an aggregate-only diagnostic."""

    if not path.is_file():
        raise MetaSynProjectionV2Error("metasyn_projection_v2_execution_bundle_missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetaSynProjectionV2Error(
            "metasyn_projection_v2_execution_bundle_unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise MetaSynProjectionV2Error("metasyn_projection_v2_execution_bundle_not_object")
    return diagnose_execution_bundle_projection_v2(
        execution_bundle=value,
        row_ordinals=row_ordinals,
    )


__all__ = [
    "DEFAULT_FAILURE_STRATIFIED_ROWS",
    "MAX_ANCHOR_TEXT_CHARACTERS",
    "MAX_SELECTED_SOURCE_CHARACTERS",
    "AnchoredPassageV2",
    "FrozenMetaSynProjectionV2",
    "MetaSynProjectionV2Error",
    "PassageOriginV2",
    "ProjectionV2DiagnosticReport",
    "ProjectionV2LineageBinding",
    "ProjectionV2RowDiagnostic",
    "diagnose_execution_bundle_projection_v2",
    "freeze_metasyn_projection_v2",
    "freeze_projection_v2_lineage_binding",
    "load_and_diagnose_execution_bundle_projection_v2",
    "render_metasyn_projection_v2_prompt_surface",
    "validate_metasyn_projection_v2_external_replay",
]
