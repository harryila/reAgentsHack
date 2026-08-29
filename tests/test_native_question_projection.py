from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.native_grounding import ResolvedNativeSource, ResolvedSourceLine
from literature_multiverse.native_question_projection import (
    FrozenSourceProjectionV1,
    NativeQuestionProjectionError,
    freeze_question_projection_spec,
    project_resolved_source_for_question,
    validate_frozen_source_projection_external_replay,
)


def _source(*sections_and_text: tuple[str, str]) -> ResolvedNativeSource:
    lines: list[ResolvedSourceLine] = []
    char_cursor = 0
    byte_cursor = 0
    for line_number, (section, text) in enumerate(sections_and_text, start=1):
        char_end = char_cursor + len(text)
        byte_end = byte_cursor + len(text.encode("utf-8"))
        lines.append(
            ResolvedSourceLine(
                line_id=f"L{line_number}",
                line_number=line_number,
                section=section,
                text=text,
                char_start=char_cursor,
                char_end=char_end,
                utf8_byte_start=byte_cursor,
                utf8_byte_end=byte_end,
            )
        )
        char_cursor = char_end + 1
        byte_cursor = byte_end + 1
    source_text = "\n".join(text for _, text in sections_and_text)
    return ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path="data/cache/metasyn/corpus.parquet",
        artifact_sha256=hashlib.sha256(b"artifact").hexdigest(),
        source_locator="metasyn://corpus/42",
        source_payload_sha256=hash_canonical(
            {"sections": list(sections_and_text)}
        ),
        source_text=source_text,
        lines=lines,
    )


def _spec(**overrides: Any):
    values: dict[str, Any] = {
        "question_id": "metasyn-review-000042",
        "population": "adults with hypertension",
        "intervention_or_exposure": "drug alpha",
        "comparison": "placebo",
        "outcome_texts": ["Systolic blood pressure"],
        "treatment_role": "intervention_or_exposure",
        "comparator_role": "comparator",
        "contrast_estimand": "not_prespecified_in_protocol_metadata",
    }
    values.update(overrides)
    return freeze_question_projection_spec(**values)


def test_projection_keeps_exact_offsets_and_abstract_fallback_with_weak_methods() -> None:
    source = _source(
        ("Title", "Drug alpha for hypertension"),
        (
            "Abstract",
            "Drug alpha reduced systolic blood pressure by 5 mmHg compared with placebo.",
        ),
        ("Materials and Methods", "Systolic blood pressure was measured in adults."),
    )
    spec = _spec()

    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=spec
    )

    assert projection.projection_status == "ready"
    assert projection.source_modality == "full_text_recognized_sections"
    assert projection.release_grade_source_grounding_eligible is True
    assert "Abstract" in projection.exposed_sections
    assert "Methods" in projection.exposed_sections
    assert any(item.section_family == "abstract" for item in projection.passages)
    for passage in projection.passages:
        assert (
            source.source_text[
                passage.source_char_start : passage.source_char_end_exclusive
            ]
            == passage.text
        )
        source_bytes = source.source_text.encode("utf-8")
        assert (
            source_bytes[
                passage.source_utf8_byte_start : passage.source_utf8_byte_end_exclusive
            ].decode("utf-8")
            == passage.text
        )


def test_unrecognized_full_text_headings_never_claim_release_grade_grounding() -> None:
    source = _source(
        ("Title", "Drug alpha for hypertension"),
        ("Abstract", "Systolic blood pressure was assessed in adults."),
        ("Discussion", "Drug alpha may be useful."),
    )
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=_spec()
    )

    assert projection.source_modality == "full_text_unrecognized_sections"
    assert projection.source_strength == "diagnostic_unrecognized_sections"
    assert projection.release_grade_source_grounding_eligible is False
    assert projection.source_strength_blockers == [
        "no_recognized_methods_results_or_table_section"
    ]
    assert all(item.section_family != "other" for item in projection.passages)


def test_title_abstract_projection_is_explicitly_diagnostic() -> None:
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42",
        source=_source(
            ("Title", "Drug alpha for hypertension"),
            ("Abstract", "Systolic blood pressure was 120 mmHg in adults."),
        ),
        spec=_spec(),
    )

    assert projection.source_modality == "title_abstract"
    assert projection.source_strength == "diagnostic_title_abstract_grounding"
    assert projection.release_grade_source_grounding_eligible is False
    assert projection.source_strength_blockers == [
        "title_or_abstract_only_not_release_grade"
    ]


def test_results_and_table_headings_normalize_to_exact_bounded_section_enum() -> None:
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42",
        source=_source(
            ("RESULTS AND FINDINGS", "Systolic blood pressure changed by 4 mmHg."),
            (
                "Supplementary Table S1",
                "Adults receiving drug alpha had systolic blood pressure of 120 mmHg.",
            ),
        ),
        spec=_spec(),
    )

    assert projection.exposed_sections == ["FigureTable", "Results"]
    assert {item.section_family for item in projection.passages} == {
        "results",
        "table_or_figure",
    }


def test_projection_enforces_fixed_passage_and_character_caps() -> None:
    source = _source(
        *(
            (
                f"Results {index}",
                "Drug alpha changed systolic blood pressure by 4 mmHg. " * 80,
            )
            for index in range(1, 40)
        )
    )
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=_spec()
    )

    assert len(projection.passages) <= 24
    assert projection.projected_characters <= 14_000
    assert all(len(item.text) <= 1_800 for item in projection.passages)


def test_projection_can_freeze_zero_eligible_passages_without_substitution() -> None:
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42",
        source=_source(("Title", "Unrelated qualitative commentary")),
        spec=_spec(),
    )

    assert projection.projection_status == "no_eligible_source_passage"
    assert projection.passages == []
    assert projection.exposed_line_ids == []
    assert projection.exposed_sections == []
    assert projection.release_grade_source_grounding_eligible is False
    assert projection.source_strength_blockers == ["no_eligible_source_passage"]


def test_outcome_ids_are_bounded_and_verbatim_text_is_casefold_deduplicated() -> None:
    spec = _spec(
        outcome_texts=["Mortality at 90 days", " mortality at 90 days "],
        positive_direction_means_by_outcome={
            "MORTALITY AT 90 DAYS": "higher_raw_value_means_more_deaths"
        },
    )

    assert spec.allowed_outcomes == ["outcome-01"]
    assert len(spec.allowed_outcomes[0]) <= 64
    assert len(spec.question_fields.outcomes) == 1
    assert spec.question_fields.outcomes[0].outcome_text == "Mortality at 90 days"
    assert (
        spec.question_fields.outcomes[0].positive_direction_means
        == "higher_raw_value_means_more_deaths"
    )


def test_projection_inputs_have_finite_caps() -> None:
    with pytest.raises(ValidationError, match="string_too_long"):
        _spec(question_id="q" * 129)
    with pytest.raises(NativeQuestionProjectionError, match="outcome_text_exceeds_cap"):
        _spec(outcome_texts=["x" * 4097])


def test_external_replay_rejects_coherently_rehashed_score_tamper() -> None:
    source = _source(
        ("Results", "Drug alpha changed systolic blood pressure by 4 mmHg."),
    )
    spec = _spec()
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=spec
    )
    tampered = projection.model_dump(mode="json")
    tampered["passages"][0]["priority_score"] += 1
    tampered["projection_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "projection_sha256"}
    )
    coherent = FrozenSourceProjectionV1.model_validate(tampered)

    with pytest.raises(
        NativeQuestionProjectionError,
        match="frozen_source_projection_external_replay_mismatch",
    ):
        validate_frozen_source_projection_external_replay(
            projection=coherent, source=source, spec=spec
        )


def test_projection_is_deterministic_and_contains_no_reference_label_fields() -> None:
    source = _source(
        ("Results", "Drug alpha changed systolic blood pressure by 4 mmHg."),
    )
    spec = _spec()
    first = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=spec
    )
    second = project_resolved_source_for_question(
        row_id="metasyn-corpus:42", source=source, spec=spec
    )

    assert first == second
    payload = first.model_dump_json().casefold()
    for forbidden in (
        "effect_direction",
        "conclusion_summary",
        "statistical_significance",
        "reference_verdict",
        "ground_truth",
    ):
        assert forbidden not in payload
