from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_projection_v2 import (
    DEFAULT_FAILURE_STRATIFIED_ROWS,
    FrozenMetaSynProjectionV2,
    MetaSynProjectionV2Error,
    ProjectionV2LineageBinding,
    diagnose_execution_bundle_projection_v2,
    freeze_metasyn_projection_v2,
    freeze_projection_v2_lineage_binding,
    load_and_diagnose_execution_bundle_projection_v2,
    render_metasyn_projection_v2_prompt_surface,
    validate_metasyn_projection_v2_external_replay,
)
from literature_multiverse.native_grounding import (
    ResolvedNativeSource,
    ResolvedSourceLine,
)
from literature_multiverse.native_question_projection import (
    FrozenSourceProjectionV1,
    freeze_question_projection_spec,
    project_resolved_source_for_question,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FAILURE_BUNDLE = (
    REPOSITORY_ROOT
    / "data/cache/metasyn/bounded-anthropic-yield-v5/execution-bundle.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source(*sections_and_text: tuple[str, str]) -> ResolvedNativeSource:
    lines: list[ResolvedSourceLine] = []
    character_cursor = 0
    byte_cursor = 0
    for line_number, (section, source_text) in enumerate(
        sections_and_text, start=1
    ):
        character_end = character_cursor + len(source_text)
        byte_end = byte_cursor + len(source_text.encode("utf-8"))
        lines.append(
            ResolvedSourceLine(
                line_id=f"L{line_number}",
                line_number=line_number,
                section=section,
                text=source_text,
                char_start=character_cursor,
                char_end=character_end,
                utf8_byte_start=byte_cursor,
                utf8_byte_end=byte_end,
            )
        )
        character_cursor = character_end + 1
        byte_cursor = byte_end + 1
    joined = "\n".join(source_text for _, source_text in sections_and_text)
    return ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path="data/cache/metasyn/projection-v2-private-fixture.parquet",
        artifact_sha256=_sha("projection-v2-fixture-artifact"),
        source_locator="metasyn://projection-v2-private-fixture/1",
        source_payload_sha256=hash_canonical(
            {"sections": list(sections_and_text)}
        ),
        source_text=joined,
        lines=lines,
    )


def _projection(*sections_and_text: tuple[str, str]) -> FrozenSourceProjectionV1:
    source = _source(*sections_and_text)
    spec = freeze_question_projection_spec(
        question_id="projection-v2-private-fixture-question",
        population="adults with hypertension",
        intervention_or_exposure="drug alpha",
        comparison="placebo",
        outcome_texts=["Systolic blood pressure"],
        treatment_role="intervention_or_exposure",
        comparator_role="comparator",
        contrast_estimand="not_prespecified_in_protocol_metadata",
    )
    return project_resolved_source_for_question(
        row_id="metasyn-projection-v2-private-fixture:1",
        source=source,
        spec=spec,
    )


def _binding(projection: FrozenSourceProjectionV1) -> ProjectionV2LineageBinding:
    return freeze_projection_v2_lineage_binding(
        upstream_execution_bundle_sha256=_sha("execution-bundle"),
        upstream_row_context_sha256=_sha("row-context"),
        upstream_source_row_sha256=_sha("source-row"),
        projection=projection,
    )


def _load_frozen_failure_bundle() -> dict[str, Any]:
    if not FROZEN_FAILURE_BUNDLE.is_file():
        pytest.skip("private frozen failure bundle is not present")
    value = json.loads(FROZEN_FAILURE_BUNDLE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_exact_duplicate_text_is_exposed_once_with_every_origin_bound() -> None:
    duplicate = (
        "Adults with hypertension receiving drug alpha versus placebo had "
        "systolic blood pressure change of -4 mmHg"
    )
    upstream = _projection(("Results", duplicate), ("Table 1", duplicate))
    binding = _binding(upstream)

    first = freeze_metasyn_projection_v2(
        projection=upstream,
        lineage_binding=binding,
    )
    second = freeze_metasyn_projection_v2(
        projection=upstream,
        lineage_binding=binding,
    )

    assert first == second
    assert first.upstream_passage_count == 2
    assert first.unique_upstream_parent_text_count == 1
    assert first.exact_duplicate_parent_passages_removed == 1
    assert first.anchored_passage_count == 1
    assert first.selected_passage_count == 1
    assert first.omitted_passage_count == 0
    anchored = first.passages[0]
    assert anchored.origin_count == 2
    assert anchored.duplicate_occurrence_count == 1
    assert anchored.line_ids == ["L1", "L2"]
    assert anchored.sections == ["Results", "Table 1"]
    assert anchored.exposed_sections == ["FigureTable", "Results"]
    assert render_metasyn_projection_v2_prompt_surface(first).count(duplicate) == 1
    assert validate_metasyn_projection_v2_external_replay(
        projection_v2=first,
        projection_v1=upstream,
        lineage_binding=binding,
    ) == first


def test_exact_partition_expands_past_v1_passage_cap_without_source_loss() -> None:
    rows: list[tuple[str, str]] = []
    for index in range(1, 25):
        prefix = (
            "Adults with hypertension received drug alpha versus placebo and "
            "systolic blood pressure changed by 4 mmHg "
        )
        token = f"unique-marker-{index:02d} "
        text = (prefix + token * 100)[:540]
        rows.append((f"Results {index}", text))
    upstream = _projection(*rows)

    assert len(upstream.passages) == 24
    assert upstream.projected_characters == 24 * 540
    v2 = freeze_metasyn_projection_v2(
        projection=upstream,
        lineage_binding=_binding(upstream),
    )

    assert v2.selected_passage_count > 24
    assert v2.expanded_beyond_v1_passage_cap is True
    assert v2.selection_complete is True
    assert v2.omitted_passage_count == 0
    assert v2.selected_source_characters == upstream.projected_characters
    assert v2.expanded_origin_segment_count == 48
    assert all(len(item.text) <= 512 for item in v2.passages)
    assert sum(
        origin.segment_characters
        for passage in v2.passages
        for origin in passage.origins
    ) == upstream.projected_characters


def test_coherently_rehashed_wrong_parent_binding_fails_closed() -> None:
    upstream = _projection(
        (
            "Results",
            "Adults taking drug alpha rather than placebo had systolic blood "
            "pressure change of -4 mmHg",
        )
    )
    raw_binding = _binding(upstream).model_dump(mode="json")
    raw_binding["upstream_projection_sha256"] = _sha("different-projection")
    binding_payload = {
        key: value
        for key, value in raw_binding.items()
        if key != "binding_sha256"
    }
    raw_binding["binding_sha256"] = hash_canonical(binding_payload)
    wrong_parent = ProjectionV2LineageBinding.model_validate(raw_binding)

    with pytest.raises(
        MetaSynProjectionV2Error,
        match="lineage_binding_mismatch:upstream_projection_sha256",
    ):
        freeze_metasyn_projection_v2(
            projection=upstream,
            lineage_binding=wrong_parent,
        )


def test_coherently_rehashed_origin_coverage_gap_fails_closed() -> None:
    long_text = (
        "Adults with hypertension received drug alpha versus placebo and systolic "
        "blood pressure changed by 4 mmHg "
        + "unique-provenance-token " * 30
    )
    upstream = _projection(("Results", long_text))
    frozen = freeze_metasyn_projection_v2(
        projection=upstream,
        lineage_binding=_binding(upstream),
    )
    raw = frozen.model_dump(mode="json")
    passage = raw["passages"][0]
    origin = passage["origins"][0]
    for field_name in (
        "parent_char_start",
        "parent_char_end_exclusive",
        "line_char_start",
        "line_char_end_exclusive",
        "source_char_start",
        "source_char_end_exclusive",
        "source_utf8_byte_start",
        "source_utf8_byte_end_exclusive",
    ):
        origin[field_name] += 1
    origin_payload = {
        key: value for key, value in origin.items() if key != "origin_sha256"
    }
    origin["origin_sha256"] = hash_canonical(origin_payload)
    passage["origin_set_sha256"] = hash_canonical(passage["origins"])
    passage_payload = {
        key: value
        for key, value in passage.items()
        if key != "passage_lineage_sha256"
    }
    passage["passage_lineage_sha256"] = hash_canonical(passage_payload)

    with pytest.raises(ValueError, match="parent_character_coverage_gap"):
        FrozenMetaSynProjectionV2.model_validate(raw)


def test_failure_stratified_frozen_diagnostic_is_label_blind_and_replayable() -> None:
    first = load_and_diagnose_execution_bundle_projection_v2(
        FROZEN_FAILURE_BUNDLE
    )
    second = load_and_diagnose_execution_bundle_projection_v2(
        FROZEN_FAILURE_BUNDLE
    )

    assert first == second
    assert first.row_ordinals == list(DEFAULT_FAILURE_STRATIFIED_ROWS)
    assert first.official_test_labels_opened is False
    assert first.reference_fields_unopened is True
    assert first.extraction_accuracy_reported is False
    assert first.claim_release_authority is False
    assert first.synthesis_authority is False
    assert first.target_surface_retention_is_accuracy is False
    assert first.total_upstream_passages == 120
    assert first.total_exact_duplicate_parent_passages_removed == 12
    assert first.total_omitted_passages == 0
    assert first.rows_with_selection_complete == 5
    assert first.rows_with_all_target_surface_characters_retained == 5
    by_ordinal = {row.row_ordinal: row for row in first.rows}
    assert {
        ordinal: row.exact_duplicate_parent_passages_removed
        for ordinal, row in by_ordinal.items()
    } == {9: 7, 10: 0, 18: 3, 27: 1, 29: 1}
    assert by_ordinal[27].expanded_beyond_v1_passage_cap is True
    assert by_ordinal[29].expanded_beyond_v1_passage_cap is True
    assert all(
        row.all_target_surface_source_characters_retained for row in first.rows
    )

    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True)
    assert '"text"' not in serialized
    assert "official_conclusion" not in serialized
    assert "reference_verdict" not in serialized
    assert "ground_truth" not in serialized


def test_diagnostic_fails_closed_when_label_boundary_is_rehashed_open() -> None:
    bundle = deepcopy(_load_frozen_failure_bundle())
    bundle["official_test_labels_opened"] = True
    payload = {
        key: value
        for key, value in bundle.items()
        if key != "execution_bundle_sha256"
    }
    bundle["execution_bundle_sha256"] = hash_canonical(payload)

    with pytest.raises(
        MetaSynProjectionV2Error,
        match="official_labels_boundary_invalid",
    ):
        diagnose_execution_bundle_projection_v2(execution_bundle=bundle)
