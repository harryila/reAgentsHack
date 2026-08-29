from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.metasyn_extraction_inputs_v2 as inputs_module
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    MetaSynCandidateInventoryV2,
    freeze_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    FORBIDDEN_MODEL_FACING_FIELD_NAMES,
    QUESTION_PROTOCOL_FIELD_WHITELIST,
    MetaSynExtractionInputsV2,
    MetaSynExtractionInputsV2Error,
    MetaSynPacketCandidateInputV2,
    _extraction_inputs_python_dependency_closure,
    _freeze_prompt_binding,
    freeze_metasyn_extraction_inputs_v2,
    freeze_metasyn_packet_candidate_input_v2,
    validate_metasyn_extraction_inputs_v2,
    validate_metasyn_packet_candidate_input_v2,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def extraction_inputs() -> MetaSynExtractionInputsV2:
    # Normative integration path: this performs the real immutable-v5 external replay,
    # rehashes source bytes, and freezes all 32 successor rows without provider calls.
    return freeze_metasyn_extraction_inputs_v2(repository_root=REPOSITORY_ROOT)


def _candidate_fixture(
    bundle: MetaSynExtractionInputsV2,
) -> tuple[int, MetaSynCandidateInventoryReceiptV2]:
    selected_row = None
    selected_ids: list[str] | None = None
    for row in bundle.rows:
        ordered = row.projection_surface.passage_ids
        for left, right in combinations(ordered, 2):
            pair_in_prompt_order = [left, right]
            if pair_in_prompt_order != sorted(pair_in_prompt_order):
                selected_row = row
                selected_ids = sorted(pair_in_prompt_order)
                break
        if selected_row is not None:
            break
    if selected_row is None or selected_ids is None:  # pragma: no cover - real roster guards
        selected_row = bundle.rows[0]
        selected_ids = [selected_row.projection_surface.passage_ids[0]]

    outcome_id = selected_row.question_surface.allowed_outcome_ids[0]
    outcome_text = selected_row.question_surface.allowed_outcome_text_by_id[outcome_id]
    candidate = {
        "candidate_index": 1,
        "canonical_outcome_id": outcome_id,
        "outcome_concept_quote": outcome_text[: min(128, len(outcome_text))],
        "effect_kind": "direct_confidence_interval",
        "passage_ids": selected_ids,
    }
    inventory = MetaSynCandidateInventoryV2(
        inventory_status="candidates_found",
        candidates=[candidate],
        has_more_or_uncertain=False,
    )
    receipt = freeze_metasyn_candidate_inventory_receipt_v2(
        row_context_sha256=selected_row.upstream_row_context_sha256,
        projection_v2_sha256=selected_row.projection_v2_sha256,
        allowed_outcome_text_by_id=(selected_row.question_surface.allowed_outcome_text_by_id),
        passage_text_by_id={
            passage.passage_id: passage.text for passage in selected_row.projection_surface.passages
        },
        value=inventory,
    )
    return selected_row.row_ordinal, receipt


@pytest.fixture(scope="session")
def candidate_receipt(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> tuple[int, MetaSynCandidateInventoryReceiptV2]:
    return _candidate_fixture(extraction_inputs)


@pytest.fixture(scope="session")
def packet_input(
    extraction_inputs: MetaSynExtractionInputsV2,
    candidate_receipt: tuple[int, MetaSynCandidateInventoryReceiptV2],
) -> MetaSynPacketCandidateInputV2:
    row_ordinal, receipt = candidate_receipt
    return freeze_metasyn_packet_candidate_input_v2(
        extraction_inputs=extraction_inputs,
        row_ordinal=row_ordinal,
        inventory_receipt=receipt,
        candidate_index=1,
    )


def _rehash_bundle(payload: dict[str, Any]) -> None:
    payload["extraction_inputs_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "extraction_inputs_sha256"}
    )


def _rehash_packet(payload: dict[str, Any]) -> None:
    payload["packet_input_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "packet_input_sha256"}
    )


def _model_facing_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {item for child in value.values() for item in _model_facing_keys(child)}
    if isinstance(value, list):
        return {item for child in value for item in _model_facing_keys(child)}
    return set()


def test_real_external_replay_freezes_all_32_provider_neutral_rows(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    bundle = extraction_inputs
    assert (bundle.question_count, bundle.component_count, bundle.publication_count) == (
        10,
        10,
        32,
    )
    assert len(bundle.rows) == 32
    assert [row.row_ordinal for row in bundle.rows] == list(range(32))
    assert [row.row_key for row in bundle.rows] == sorted({row.row_key for row in bundle.rows})
    assert bundle.upstream_v5_source_surface_consumed is True
    assert bundle.upstream_v5_source_surface_external_replayed is True
    assert bundle.direct_v5_hosted_execution_bundle_consumed is False
    assert bundle.v5_hosted_call_receipts_consumed is False
    assert bundle.v5_hosted_row_results_consumed is False
    assert bundle.v5_hosted_provider_outputs_consumed is False
    assert bundle.provider_calls_made is False
    assert bundle.reference_fields_unopened is True
    assert bundle.official_test_labels_opened is False
    assert bundle.extraction_accuracy_authority is False
    assert bundle.scientific_effectiveness_authority is False
    assert bundle.claim_release_authority is False


def test_question_surface_is_the_exact_closed_extraction_protocol_whitelist(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    assert extraction_inputs.question_protocol_field_whitelist == sorted(
        QUESTION_PROTOCOL_FIELD_WHITELIST
    )
    assert extraction_inputs.forbidden_model_facing_field_names == list(
        FORBIDDEN_MODEL_FACING_FIELD_NAMES
    )
    forbidden = set(FORBIDDEN_MODEL_FACING_FIELD_NAMES)
    for row in extraction_inputs.rows:
        question = row.question_surface
        assert question.allowed_outcome_ids == sorted(question.allowed_outcome_text_by_id)
        assert set(question.allowed_outcome_ids) == set(
            question.raw_positive_direction_meaning_by_outcome_id
        )
        model_keys = {
            inputs_module._normalized_key(key)
            for key in _model_facing_keys(question.model_dump(mode="json"))
        }
        assert model_keys.isdisjoint(forbidden)
        assert "review_id" not in model_keys
        assert "search_end_date" not in model_keys
        assert "clinical_benefit_direction_by_outcome_id" not in model_keys


def test_every_row_has_projection_v2_source_strength_prompt_and_schema(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    for row in extraction_inputs.rows:
        assert row.projection_v2.extraction_accuracy_authority is False
        assert row.projection_v2.claim_release_authority is False
        assert row.projection_v2_sha256 == row.projection_v2.projection_sha256
        assert row.projection_surface.projection_v2_sha256 == row.projection_v2_sha256
        assert row.source_strength == row.projection_surface.source_strength
        assert row.inventory_input.projection_v2_sha256 == row.projection_v2_sha256
        assert (
            row.inventory_input.inventory_schema_bundle_sha256
            == (row.inventory_input.inventory_schema_bundle["schema_bundle_sha256"])
        )
        assert "[[QUESTION_SPEC_JSON]]" not in row.inventory_input.rendered_prompt
        assert "[[PROJECTION_V2_JSON]]" not in row.inventory_input.rendered_prompt
        assert row.question_surface.question_id in row.inventory_input.rendered_prompt
        assert row.projection_v2_sha256 in row.inventory_input.rendered_prompt
        context = row.inventory_input.inventory_schema_bundle["context_binding"]
        assert context["allowed_outcome_ids"] == row.question_surface.allowed_outcome_ids
        assert context["passage_ids"] == sorted(row.projection_surface.passage_ids)


def test_full_projection_passages_preserve_contiguous_prompt_order(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    at_least_one_hash_order_differs = False
    for row in extraction_inputs.rows:
        passages = row.projection_surface.passages
        assert [item.prompt_rank for item in passages] == list(range(1, len(passages) + 1))
        assert row.projection_surface.passage_ids == [item.passage_id for item in passages]
        at_least_one_hash_order_differs |= row.projection_surface.passage_ids != sorted(
            row.projection_surface.passage_ids
        )
    assert at_least_one_hash_order_differs


def test_pipeline_fingerprint_is_ast_closed_and_binds_prompts_and_dependencies(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    closure = _extraction_inputs_python_dependency_closure(REPOSITORY_ROOT)
    assert "src/literature_multiverse/metasyn_extraction_inputs_v2.py" in closure
    assert "src/literature_multiverse/metasyn_v5_source_surface.py" in closure
    assert "src/literature_multiverse/metasyn_projection_v2.py" in closure
    assert "src/literature_multiverse/metasyn_candidate_inventory_v2.py" in closure
    assert "src/literature_multiverse/native_packet_grounding_v2.py" in closure
    component = extraction_inputs.extraction_inputs_pipeline_fingerprint.components[0]
    paths = [item.path for item in component.files]
    assert "prompts/metasyn_candidate_inventory_v2.md" in paths
    assert "prompts/metasyn_candidate_packet_v2.md" in paths
    assert "pyproject.toml" in paths
    assert "uv.lock" in paths
    assert set(component.settings["installed_dependency_versions"]) == {
        "anthropic",
        "jsonschema",
        "pyarrow",
        "pydantic",
    }


def test_packet_binder_uses_authorized_candidate_full_projection_and_subset(
    extraction_inputs: MetaSynExtractionInputsV2,
    candidate_receipt: tuple[int, MetaSynCandidateInventoryReceiptV2],
    packet_input: MetaSynPacketCandidateInputV2,
) -> None:
    row_ordinal, receipt = candidate_receipt
    row = extraction_inputs.rows[row_ordinal]
    packet = packet_input
    assert packet.inventory_receipt_sha256 == receipt.receipt_sha256
    assert packet.candidate_binding_sha256 == packet.candidate_binding.binding_sha256
    assert packet.candidate_binding.passage_ids == sorted(packet.candidate_binding.passage_ids)
    assert set(packet.candidate_passage_surface.passage_ids) == set(
        packet.candidate_binding.passage_ids
    )
    assert [passage.prompt_rank for passage in packet.candidate_passage_surface.passages] == (
        sorted(passage.prompt_rank for passage in packet.candidate_passage_surface.passages)
    )
    assert packet.projection_surface == row.projection_surface
    assert packet.projection_surface_sha256 == row.projection_surface_sha256
    assert packet.packet_schema_bundle.candidate_binding_sha256 == (packet.candidate_binding_sha256)
    assert packet.candidate_binding_sha256 in packet.rendered_prompt
    assert packet.projection_surface_sha256 in packet.rendered_prompt
    assert "[[PROJECTION_V2_JSON]]" not in packet.rendered_prompt
    candidate_ids = set(packet.candidate_passage_surface.passage_ids)
    noncandidate = next(
        passage
        for passage in packet.projection_surface.passages
        if passage.passage_id not in candidate_ids
    )
    assert noncandidate.text in packet.rendered_prompt
    assert "identity claims" in packet.rendered_prompt.casefold()
    assert "full exposed projection" in packet.rendered_prompt.casefold()


def test_packet_input_replays_exactly(
    extraction_inputs: MetaSynExtractionInputsV2,
    candidate_receipt: tuple[int, MetaSynCandidateInventoryReceiptV2],
    packet_input: MetaSynPacketCandidateInputV2,
) -> None:
    _, receipt = candidate_receipt
    assert (
        validate_metasyn_packet_candidate_input_v2(
            packet_input=packet_input.model_dump(mode="json"),
            extraction_inputs=extraction_inputs,
            inventory_receipt=receipt,
        )
        == packet_input
    )


def test_non_authorizing_inventory_cannot_create_packet(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    row = extraction_inputs.rows[0]
    inventory = MetaSynCandidateInventoryV2(
        inventory_status="no_candidate_found",
        candidates=[],
        has_more_or_uncertain=False,
    )
    receipt = freeze_metasyn_candidate_inventory_receipt_v2(
        row_context_sha256=row.upstream_row_context_sha256,
        projection_v2_sha256=row.projection_v2_sha256,
        allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
        passage_text_by_id={
            passage.passage_id: passage.text for passage in row.projection_surface.passages
        },
        value=inventory,
    )
    with pytest.raises(
        MetaSynExtractionInputsV2Error,
        match="metasyn_extraction_inputs_v2_inventory_not_packet_authorizing",
    ):
        freeze_metasyn_packet_candidate_input_v2(
            extraction_inputs=extraction_inputs,
            row_ordinal=0,
            inventory_receipt=receipt,
            candidate_index=1,
        )


def test_packet_prompt_coherent_tamper_is_caught_by_replay(
    extraction_inputs: MetaSynExtractionInputsV2,
    candidate_receipt: tuple[int, MetaSynCandidateInventoryReceiptV2],
    packet_input: MetaSynPacketCandidateInputV2,
) -> None:
    _, receipt = candidate_receipt
    payload = packet_input.model_dump(mode="json")
    payload["rendered_prompt"] += "\ncoherent-tamper"
    payload["rendered_prompt_sha256"] = inputs_module._sha256_text(payload["rendered_prompt"])
    payload["rendered_prompt_characters"] = len(payload["rendered_prompt"])
    _rehash_packet(payload)
    coherent = MetaSynPacketCandidateInputV2.model_validate(payload)
    with pytest.raises(
        MetaSynExtractionInputsV2Error,
        match="metasyn_extraction_inputs_v2_packet_external_replay_mismatch",
    ):
        validate_metasyn_packet_candidate_input_v2(
            packet_input=coherent,
            extraction_inputs=extraction_inputs,
            inventory_receipt=receipt,
        )


def test_bundle_hash_and_membership_tamper_fail_closed(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> None:
    payload = extraction_inputs.model_dump(mode="json")
    payload["rows"].pop()
    _rehash_bundle(payload)
    with pytest.raises(ValueError):
        MetaSynExtractionInputsV2.model_validate(payload)

    payload = extraction_inputs.model_dump(mode="json")
    payload["inventory_prompt_membership_sha256"] = "0" * 64
    _rehash_bundle(payload)
    with pytest.raises(ValueError, match="metasyn_extraction_inputs_v2_membership"):
        MetaSynExtractionInputsV2.model_validate(payload)


def test_coherently_rehashed_upstream_lineage_tamper_fails_external_replay(
    extraction_inputs: MetaSynExtractionInputsV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = extraction_inputs.model_dump(mode="json")
    row = payload["rows"][0]
    row["upstream_artifact_binding_sha256"] = "0" * 64
    row["row_input_sha256"] = hash_canonical(
        {key: value for key, value in row.items() if key != "row_input_sha256"}
    )
    payload["row_input_hash_membership_sha256"] = hash_canonical(
        [item["row_input_sha256"] for item in payload["rows"]]
    )
    _rehash_bundle(payload)
    coherent = MetaSynExtractionInputsV2.model_validate(payload)

    # The session fixture already exercised the real external replay.  Pin its exact
    # result here so this test isolates the replay comparison rather than repeating the
    # expensive source-byte pass.
    monkeypatch.setattr(
        inputs_module,
        "freeze_metasyn_extraction_inputs_v2",
        lambda **_kwargs: extraction_inputs,
    )
    with pytest.raises(
        MetaSynExtractionInputsV2Error,
        match="metasyn_extraction_inputs_v2_external_replay_mismatch",
    ):
        validate_metasyn_extraction_inputs_v2(
            extraction_inputs=coherent,
            repository_root=REPOSITORY_ROOT,
            external_replay=True,
        )


def test_prompt_symlink_fails_closed(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    real = tmp_path / "real-inventory.md"
    real.write_text(
        "Prompt version: `metasyn-passage-candidate-inventory-v2`\n"
        "[[QUESTION_SPEC_JSON]] [[PROJECTION_V2_JSON]]",
        encoding="utf-8",
    )
    link = prompts / "metasyn_candidate_inventory_v2.md"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(
        MetaSynExtractionInputsV2Error,
        match="metasyn_extraction_inputs_v2_file_symlink_forbidden",
    ):
        _freeze_prompt_binding(root=tmp_path, kind="inventory")
