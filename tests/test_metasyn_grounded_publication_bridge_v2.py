from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    freeze_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    freeze_metasyn_packet_candidate_input_v2,
)
from literature_multiverse.metasyn_grounded_publication_bridge_v2 import (
    BRIDGE_MODULE_PATH,
    MetaSynGroundedPublicationBridgeV2Error,
    MetaSynGroundedPublicationCorpusBridgeV2,
    compute_metasyn_grounded_publication_bridge_v2_pipeline_fingerprint,
    freeze_metasyn_grounded_publication_bridge_v2,
    validate_metasyn_grounded_publication_bridge_v2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    freeze_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.native_packet_assembly_v2 import (
    assemble_native_packet_v2,
    freeze_packet_assembly_protocol_orientation_v2,
    replay_metasyn_question_projection_spec_v2,
)
from literature_multiverse.native_packet_grounding_v2 import (
    freeze_passage_packet_grounding_receipt_v2,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def execution_bundle() -> MetaSynPassageHostedExecutionBundleV2:
    return freeze_metasyn_passage_hosted_execution_bundle_v2(repository_root=ROOT)


def _no_candidate_receipts(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> dict[str, MetaSynCandidateInventoryReceiptV2]:
    output: dict[str, MetaSynCandidateInventoryReceiptV2] = {}
    for row in bundle.extraction_inputs.rows:
        output[row.row_key] = freeze_metasyn_candidate_inventory_receipt_v2(
            row_context_sha256=row.upstream_row_context_sha256,
            projection_v2_sha256=row.projection_v2_sha256,
            allowed_outcome_text_by_id=(row.question_surface.allowed_outcome_text_by_id),
            passage_text_by_id={
                passage.passage_id: passage.text for passage in row.projection_surface.passages
            },
            value={
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            },
        )
    return output


@pytest.fixture(scope="module")
def no_candidate_receipts(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
) -> dict[str, MetaSynCandidateInventoryReceiptV2]:
    return _no_candidate_receipts(execution_bundle)


@pytest.fixture(scope="module")
def zero_yield_bridge(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    no_candidate_receipts: dict[str, MetaSynCandidateInventoryReceiptV2],
) -> MetaSynGroundedPublicationCorpusBridgeV2:
    return freeze_metasyn_grounded_publication_bridge_v2(
        execution_bundle=execution_bundle,
        inventory_receipts_by_row=no_candidate_receipts,
        candidate_terminals_by_row={
            row.row_key: [] for row in execution_bundle.extraction_inputs.rows
        },
        repository_root=ROOT,
    )


def _rehash(payload: dict[str, Any], field: str) -> None:
    payload[field] = hash_canonical({key: value for key, value in payload.items() if key != field})


def test_all_32_publications_are_retained_in_ten_question_scoped_corpora(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    bridge = zero_yield_bridge
    assert bridge.publication_count == len(bridge.publication_joins) == 32
    assert bridge.question_count == len(bridge.question_corpora) == 10
    assert sum(len(item.compatibility_corpus.fragments) for item in bridge.question_corpora) == 32
    assert (
        sum(len(item.compatibility_corpus.graph.publications) for item in bridge.question_corpora)
        == 32
    )
    assert all(
        len({fragment.question_id for fragment in item.compatibility_corpus.fragments}) == 1
        for item in bridge.question_corpora
    )
    assert len({item.publication.publication_id for item in bridge.publication_joins}) == 32
    assert len({item.publication.paper_id for item in bridge.publication_joins}) == 32


def test_bridge_fingerprint_rejects_symlink_repository_root(tmp_path: Path) -> None:
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(
        MetaSynGroundedPublicationBridgeV2Error,
        match="repository_root_symlink",
    ):
        compute_metasyn_grounded_publication_bridge_v2_pipeline_fingerprint(
            repository_root=linked_root,
            execution_bundle_sha256="a" * 64,
            source_surface_sha256="b" * 64,
            inventory_receipt_membership_sha256="c" * 64,
            terminal_membership_sha256="d" * 64,
        )


def test_zero_yield_is_preserved_as_non_authorizing_not_numeric_absence(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    bridge = zero_yield_bridge
    assert bridge.inventoried_candidate_count == 0
    assert bridge.authorized_candidate_count == 0
    assert bridge.terminal_candidate_count == 0
    assert bridge.completed_candidate_count == 0
    assert bridge.quantitative_effect_count == 0
    assert bridge.estimable_publication_count == 0
    assert bridge.quantitative_kernel_compatibility is False
    for publication in bridge.publication_joins:
        assert publication.inventory_status == "no_candidate_non_authorizing"
        assert publication.candidate_terminals == []
        assert publication.compatibility_fragment.status == "non_estimable"
        assert publication.compatibility_fragment.non_estimability_reason in {
            "other",
            "source_document_incomplete",
        }
        assert publication.compatibility_fragment.non_estimability_reason != (
            "numerical_result_absent"
        )
        assert (
            "no_candidate_non_authorizing_no_absence_claim"
            in publication.compatibility_fragment.non_estimability_detail
            or "projection_or_grounding_surface_incomplete"
            in publication.compatibility_fragment.non_estimability_detail
        )
        assert publication.compatibility_fragment.graph is None
    assert all(
        not item.compatibility_corpus.graph.outcome_estimates for item in bridge.question_corpora
    )


def test_bridge_authority_is_explicitly_fail_closed(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    bridge = zero_yield_bridge
    assert bridge.exact_projection_authority is True
    assert bridge.graph_construction_authority is True
    assert bridge.extraction_accuracy_authority is False
    assert bridge.scientific_effectiveness_authority is False
    assert bridge.synthesis_input_authority is False
    assert bridge.claim_release_authority is False
    assert bridge.legacy_v4_grounding_package_emitted is False
    assert bridge.reference_fields_unopened is True
    assert bridge.official_test_labels_opened is False
    assert bridge.v5_hosted_outputs_consumed is False
    assert all(
        item.synthesis_input_authority is False and item.claim_release_authority is False
        for item in bridge.publication_joins
    )


def test_every_publication_join_preserves_exact_protocol_source_and_artifact_aliases(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    for source_row, extraction_row, publication in zip(
        zero_yield_bridge.source_surface.rows,
        zero_yield_bridge.execution_bundle.extraction_inputs.rows,
        zero_yield_bridge.publication_joins,
        strict=True,
    ):
        source_record = source_row.source_row.source_record
        assert publication.row_key == source_row.row_key == extraction_row.row_key
        assert publication.publication_source_identity_sha256 == (
            source_row.row_source_identity_sha256
        )
        assert publication.publication == source_record.publication
        assert publication.source_document == source_record.source_document
        assert publication.source_document.sha256 == (
            source_row.artifact_binding.observed_artifact_sha256
        )
        assert publication.protocol_outcome_text_by_id == (
            extraction_row.question_surface.allowed_outcome_text_by_id
        )


def test_bridge_pipeline_fingerprint_binds_additive_file_and_runtime_inputs(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    fingerprint = zero_yield_bridge.bridge_pipeline_fingerprint
    assert len(fingerprint.components) == 1
    component = fingerprint.components[0]
    assert component.component_id == "metasyn-grounded-publication-corpus-bridge-v2"
    assert [item.path for item in component.files] == [BRIDGE_MODULE_PATH]
    assert component.settings["execution_bundle_sha256"] == (
        zero_yield_bridge.execution_bundle_sha256
    )
    assert component.settings["source_surface_sha256"] == (zero_yield_bridge.source_surface_sha256)
    assert component.settings["inventory_receipt_membership_sha256"] == (
        zero_yield_bridge.inventory_receipt_membership_sha256
    )
    assert component.settings["terminal_membership_sha256"] == (
        zero_yield_bridge.terminal_membership_sha256
    )
    assert component.settings["claim_release_authority"] is False


def test_saved_bridge_externally_replays_every_join_and_source_byte(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    replayed = validate_metasyn_grounded_publication_bridge_v2(
        bridge=zero_yield_bridge.model_dump(mode="json"),
        repository_root=ROOT,
        external_replay=True,
    )
    assert replayed == zero_yield_bridge


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("synthesis_input_authority", True),
        ("claim_release_authority", True),
        ("official_test_labels_opened", True),
        ("legacy_v4_grounding_package_emitted", True),
    ],
)
def test_coherent_outer_rehash_cannot_escalate_authority(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
    field: str,
    value: bool,
) -> None:
    payload = zero_yield_bridge.model_dump(mode="json")
    payload[field] = value
    _rehash(payload, "bridge_sha256")
    with pytest.raises(ValidationError):
        MetaSynGroundedPublicationCorpusBridgeV2.model_validate(payload)


def test_missing_row_from_inventory_or_terminal_roster_fails_closed(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    no_candidate_receipts: dict[str, MetaSynCandidateInventoryReceiptV2],
) -> None:
    missing_key = execution_bundle.extraction_inputs.rows[0].row_key
    inventories = dict(no_candidate_receipts)
    inventories.pop(missing_key)
    terminals = {row.row_key: [] for row in execution_bundle.extraction_inputs.rows}
    with pytest.raises(
        MetaSynGroundedPublicationBridgeV2Error,
        match="inventory_row_roster_incomplete",
    ):
        freeze_metasyn_grounded_publication_bridge_v2(
            execution_bundle=execution_bundle,
            inventory_receipts_by_row=inventories,
            candidate_terminals_by_row=terminals,
            repository_root=ROOT,
        )


def test_authorized_candidate_requires_one_exact_terminal_and_abstention_is_retained(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    no_candidate_receipts: dict[str, MetaSynCandidateInventoryReceiptV2],
) -> None:
    row = execution_bundle.extraction_inputs.rows[0]
    passage = row.projection_surface.passages[0]
    outcome_id = row.question_surface.allowed_outcome_ids[0]
    outcome_text = row.question_surface.allowed_outcome_text_by_id[outcome_id]
    inventories = dict(no_candidate_receipts)
    inventory = freeze_metasyn_candidate_inventory_receipt_v2(
        row_context_sha256=row.upstream_row_context_sha256,
        projection_v2_sha256=row.projection_v2_sha256,
        allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
        passage_text_by_id={item.passage_id: item.text for item in row.projection_surface.passages},
        value={
            "inventory_status": "candidates_found",
            "candidates": [
                {
                    "candidate_index": 1,
                    "canonical_outcome_id": outcome_id,
                    "outcome_concept_quote": outcome_text[: min(128, len(outcome_text))],
                    "effect_kind": "direct_standard_error",
                    "passage_ids": [passage.passage_id],
                }
            ],
            "has_more_or_uncertain": False,
        },
    )
    inventories[row.row_key] = inventory
    empty_terminals = {item.row_key: [] for item in execution_bundle.extraction_inputs.rows}
    with pytest.raises(
        MetaSynGroundedPublicationBridgeV2Error,
        match="terminal_candidate_roster_mismatch",
    ):
        freeze_metasyn_grounded_publication_bridge_v2(
            execution_bundle=execution_bundle,
            inventory_receipts_by_row=inventories,
            candidate_terminals_by_row=empty_terminals,
            repository_root=ROOT,
        )

    packet = freeze_metasyn_packet_candidate_input_v2(
        extraction_inputs=execution_bundle.extraction_inputs,
        row_ordinal=row.row_ordinal,
        inventory_receipt=inventory,
        candidate_index=1,
    )
    grounding = freeze_passage_packet_grounding_receipt_v2(
        model_outcome={
            "outcome_version": "native-packet-grounding-model-outcome-v2",
            "packet_status": "unable_to_complete",
            "candidate_binding_sha256": packet.candidate_binding_sha256,
            "reason": "source_support_incomplete",
        },
        candidate=packet.candidate,
        projection=row.projection_v2,
    )
    protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
    orientation = freeze_packet_assembly_protocol_orientation_v2(
        question_surface=row.question_surface
    )
    assembly = assemble_native_packet_v2(
        candidate=packet.candidate,
        projection=row.projection_v2,
        protocol=protocol,
        protocol_orientation=orientation,
        analysis_policy=execution_bundle.assembly_analysis_policy,
        grounding_receipt=grounding,
    )
    terminals = dict(empty_terminals)
    terminals[row.row_key] = [
        {
            "packet_input": packet,
            "grounding_receipt": grounding,
            "assembly_receipt": assembly,
        }
    ]
    bridge = freeze_metasyn_grounded_publication_bridge_v2(
        execution_bundle=execution_bundle,
        inventory_receipts_by_row=inventories,
        candidate_terminals_by_row=terminals,
        repository_root=ROOT,
    )
    joined = bridge.publication_joins[0]
    assert bridge.inventoried_candidate_count == 1
    assert bridge.authorized_candidate_count == 1
    assert bridge.terminal_candidate_count == bridge.abstained_candidate_count == 1
    assert bridge.completed_candidate_count == bridge.quantitative_effect_count == 0
    assert joined.candidate_descriptor_sha256s == [packet.candidate_descriptor_sha256]
    assert joined.candidate_terminals[0].terminal_status == "unable_to_assemble"
    assert joined.candidate_terminals[0].terminal_blockers == [
        "assembly:grounding_abstained",
        "grounding:source_support_incomplete",
    ]
    assert joined.compatibility_fragment.status == "non_estimable"


def test_terminal_adapter_rejects_extra_runtime_fields_before_join(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    no_candidate_receipts: dict[str, MetaSynCandidateInventoryReceiptV2],
) -> None:
    row = execution_bundle.extraction_inputs.rows[0]
    terminals: dict[str, list[dict[str, Any]]] = {
        item.row_key: [] for item in execution_bundle.extraction_inputs.rows
    }
    terminals[row.row_key] = [
        {
            "packet_input": {},
            "grounding_receipt": {},
            "assembly_receipt": {},
            "provider_debug_payload": "forbidden",
        }
    ]
    with pytest.raises(
        MetaSynGroundedPublicationBridgeV2Error,
        match="terminal_input_fields_invalid",
    ):
        freeze_metasyn_grounded_publication_bridge_v2(
            execution_bundle=execution_bundle,
            inventory_receipts_by_row=no_candidate_receipts,
            candidate_terminals_by_row=terminals,
            repository_root=ROOT,
        )


def test_dropped_publication_fails_even_after_coherent_outer_rehash(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
) -> None:
    payload = deepcopy(zero_yield_bridge.model_dump(mode="json"))
    payload["publication_joins"] = payload["publication_joins"][:-1]
    payload["publication_count"] = 31
    payload["publication_join_membership_sha256"] = hash_canonical(
        [item["publication_join_sha256"] for item in payload["publication_joins"]]
    )
    _rehash(payload, "bridge_sha256")
    with pytest.raises(ValidationError):
        MetaSynGroundedPublicationCorpusBridgeV2.model_validate(payload)
