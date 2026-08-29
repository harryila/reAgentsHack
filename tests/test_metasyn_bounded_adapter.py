from __future__ import annotations

import ast
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_adapter import (
    _ADAPTER_DEPENDENCY_ENTRYPOINTS,
    _ADAPTER_NON_PYTHON_INPUTS,
    MetaSynBoundedAdapterError,
    MetaSynBoundedRowContextV1,
    compute_metasyn_bounded_adapter_fingerprint,
    freeze_metasyn_bounded_row_context,
    freeze_metasyn_inventory_validation_receipt,
    freeze_metasyn_packet_call,
    freeze_metasyn_packet_validation_receipt,
    freeze_metasyn_publication_result,
    validate_metasyn_inventory_validation_receipt,
    validate_metasyn_publication_result,
)
from literature_multiverse.metasyn_typed_pilot import (
    MetaSynPilotSourceProjectionRowV1,
    _question_spec,
)
from literature_multiverse.native_bounded_generation import (
    BoundedArm,
    BoundedCohortHeader,
    BoundedContrast,
    BoundedEvidence,
    BoundedFindingHeader,
    BoundedNumericSupport,
    BoundedStudyHeader,
    BoundedTimepoint,
    DirectStandardErrorEffect,
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidatePacket,
    NativeCandidateUnableToComplete,
)
from literature_multiverse.native_extraction import NativeSourceRecord
from literature_multiverse.native_grounding import ResolvedNativeSource, ResolvedSourceLine
from literature_multiverse.native_question_projection import (
    freeze_question_projection_spec,
    project_resolved_source_for_question,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _resolve_local_module(
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


def _independent_dependency_closure(repository_root: Path) -> set[str]:
    pending = list(_ADAPTER_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (repository_root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_module(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def _protocol_row() -> dict[str, Any]:
    return {
        "ID": 42,
        "Title": "Blood-pressure intervention review",
        "Research_Question": "Does drug alpha lower systolic blood pressure?",
        "Population": "Adults with hypertension",
        "Intervention": "Drug alpha",
        "Exposure": None,
        "Comparison": "Placebo",
        "Outcome": "Systolic blood pressure",
        "inclusion_criteria": "Controlled studies",
        "exclusion_criteria": None,
        "search_end_date": "2024-01-01",
        "matched_corpus_ids": [7, 8],
        "matched_ref_count": 2,
        "source_review_corpus_ids": [],
    }


def _resolved_source(text: str, *, section: str = "Results") -> ResolvedNativeSource:
    return ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path="data/cache/metasyn/corpus/results.parquet",
        artifact_sha256=_sha("artifact"),
        source_locator="metasyn://corpus/7",
        source_payload_sha256=hash_canonical({section: text}),
        source_text=text,
        lines=[
            ResolvedSourceLine(
                line_id="L1",
                line_number=1,
                section=section,
                text=text,
                char_start=0,
                char_end=len(text),
                utf8_byte_start=0,
                utf8_byte_end=len(text.encode()),
            )
        ],
    )


def _row_context(
    *,
    text: str = "The adjusted difference in systolic blood pressure was 0.5 (SE 0.2).",
    section: str = "Results",
) -> MetaSynBoundedRowContextV1:
    question = _question_spec(_protocol_row())
    projection_spec = freeze_question_projection_spec(
        question_id=question.question_id,
        population=question.population,
        intervention_or_exposure=question.intervention_or_exposure,
        comparison=question.comparison,
        outcome_texts=[item.outcome_text for item in question.canonical_outcomes],
        treatment_role=question.treatment_role,
        comparator_role=question.comparator_role,
        contrast_estimand=question.contrast_estimand,
        positive_direction_means_by_outcome={
            item.outcome_text: item.positive_direction_means
            for item in question.canonical_outcomes
        },
    )
    source = _resolved_source(text, section=section)
    projection = project_resolved_source_for_question(
        row_id="metasyn-corpus:7", source=source, spec=projection_spec
    )
    source_document = SourceDocumentArtifact(
        artifact_path=source.artifact_path,
        sha256=source.artifact_sha256,
        media_type="application/x-parquet",
        source_locator=source.source_locator,
    )
    source_record = NativeSourceRecord(
        doc_id="metasyn-corpus:7",
        publication=PublicationIdentity(
            publication_id="metasyn-publication:7",
            paper_id="metasyn-paper:7",
            doc_id="metasyn-corpus:7",
        ),
        source_document=source_document,
    )
    source_payload: dict[str, Any] = {
        "source_row_version": "metasyn-typed-oracle-source-row-v1",
        "question_id": question.question_id,
        "corpus_id": 7,
        "doc_id": "metasyn-corpus:7",
        "source_record": source_record,
        "diagnostic_source_record_sha256": _sha("diagnostic"),
        "source_content_scope": (
            "title_abstract" if section in {"Title", "Abstract"} else "full_text_sections"
        ),
        "oracle_selection_full_text_scope": section not in {"Title", "Abstract"},
        "source_projection_strength": projection.source_strength,
        "release_grade_source_grounding_eligible": (
            projection.release_grade_source_grounding_eligible
        ),
        "source_strength_blockers": projection.source_strength_blockers,
        "projection": projection,
        "projection_sha256": projection.projection_sha256,
    }
    source_row = MetaSynPilotSourceProjectionRowV1.model_validate(
        {**source_payload, "source_row_sha256": hash_canonical(source_payload)}
    )
    return freeze_metasyn_bounded_row_context(
        question_bundle_sha256=_sha("question-bundle"),
        question_spec=question,
        independence_component_id="metasyn-component-42",
        independence_component_review_ids=[question.review_id],
        independence_component_membership_sha256=hash_canonical(
            [question.review_id]
        ),
        source_row=source_row,
        inventory_template=(
            REPOSITORY_ROOT / "prompts/metasyn_candidate_inventory.md"
        ).read_text(encoding="utf-8"),
        packet_template=(
            REPOSITORY_ROOT / "prompts/metasyn_candidate_packet.md"
        ).read_text(encoding="utf-8"),
    )


def _inventory(row: MetaSynBoundedRowContextV1) -> NativeCandidateInventory:
    return NativeCandidateInventory(
        inventory_status="candidates_found",
        candidates=[
            NativeCandidateDescriptor(
                candidate_index=1,
                outcome_name="outcome-01",
                effect_kind="direct_standard_error",
                line_ids=["L1"],
            )
        ],
        has_more_or_uncertain=False,
    )


def _packet(
    row: MetaSynBoundedRowContextV1,
) -> NativeCandidatePacket[DirectStandardErrorEffect]:
    quote = "The adjusted difference in systolic blood pressure was 0.5 (SE 0.2)."
    return NativeCandidatePacket[DirectStandardErrorEffect](
        candidate_index=1,
        study=BoundedStudyHeader(
            key="study-1", source_label="Study 1", registration_ids=[]
        ),
        cohort=BoundedCohortHeader(
            key="cohort-1",
            source_labels=["Cohort 1"],
            registry_ids=[],
            dataset_ids=[],
        ),
        treatment_arm=BoundedArm(
            key="drug-alpha", label="Drug alpha", role="intervention"
        ),
        comparator_arm=BoundedArm(
            key="placebo", label="Placebo", role="comparator"
        ),
        contrast=BoundedContrast(
            key="target",
            label="drug_alpha_vs_placebo",
            estimand=row.question_spec.contrast_estimand,
            positive_direction_means=row.outcome_positive_directions["outcome-01"],
        ),
        finding=BoundedFindingHeader(
            key="finding-1",
            outcome_name="outcome-01",
            timepoint=BoundedTimepoint(kind="not_reported"),
        ),
        effect=DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate="0.5",
            standard_error="0.2",
            unit="mmHg",
        ),
        evidence=BoundedEvidence(
            source_locator=row.source_locator,
            quote=quote,
            section="Results",
            line_ids=["L1"],
        ),
        numeric_support=[
            BoundedNumericSupport(
                field_path="effect.estimate",
                verbatim_token="0.5",
                quote_start=str(quote.index("0.5")),
                quote_end=str(quote.index("0.5") + 3),
            ),
            BoundedNumericSupport(
                field_path="effect.standard_error",
                verbatim_token="0.2",
                quote_start=str(quote.index("0.2")),
                quote_end=str(quote.index("0.2") + 3),
            ),
        ],
    )


def test_row_contract_is_question_specific_label_blind_and_provider_neutral() -> None:
    row = _row_context()

    assert row.allowed_outcomes == ["outcome-01"]
    assert row.allowed_sections == ["Results"]
    assert row.source_locator == "metasyn://corpus/7"
    assert row.independence_component_id == "metasyn-component-42"
    assert row.independence_component_review_ids == [42]
    assert row.independence_component_membership_sha256 == hash_canonical([42])
    assert row.source_row.projection.exposed_line_ids == ["L1"]
    assert "raw outcome/effect orientation" in row.packet_base_prompt
    assert "It is not a" in row.packet_base_prompt
    assert "__FROZEN_CANDIDATE_JSON__" in row.packet_base_prompt
    assert "Effect_Direction" not in row.inventory_prompt
    assert "Conclusion_Summary" not in row.inventory_prompt
    assert "ollama" not in row.model_dump_json().casefold()


def test_title_abstract_surface_is_explicitly_diagnostic_only() -> None:
    row = _row_context(section="Abstract")

    assert row.allowed_sections == ["Abstract"]
    assert row.source_row.source_content_scope == "title_abstract"
    assert row.source_row.release_grade_source_grounding_eligible is False
    assert row.source_row.source_projection_strength == (
        "diagnostic_title_abstract_grounding"
    )
    assert row.source_row.source_strength_blockers


def test_inventory_packet_and_official_output_round_trip_with_unique_grounding() -> None:
    row = _row_context()
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )
    packet_receipt = freeze_metasyn_packet_validation_receipt(
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=_packet(row),
    )
    publication = freeze_metasyn_publication_result(
        row=row,
        inventory_receipt=inventory_receipt,
        packet_receipts=[packet_receipt],
    )

    assert inventory_receipt.status == "candidates_authorized"
    assert call.packet_schema_sha256 == hash_canonical(call.packet_schema)
    assert packet_receipt.packet_status == "completed"
    assert packet_receipt.quote_grounding is not None
    assert packet_receipt.quote_grounding.unique_matching_occurrences == 1
    assert publication.status == "typed_publication_output"
    assert publication.blocking_reasons == []
    assert publication.official_output is not None
    assert publication.official_output.status == "estimable"
    assert publication.official_output.studies[0].cohorts[0].findings[0].outcome_name == (
        "outcome-01"
    )


def test_no_candidate_never_fabricates_nonestimable_typed_output() -> None:
    row = _row_context()
    inventory = NativeCandidateInventory(
        inventory_status="no_candidate_found",
        candidates=[],
        has_more_or_uncertain=False,
    )
    receipt = freeze_metasyn_inventory_validation_receipt(row=row, value=inventory)
    result = freeze_metasyn_publication_result(
        row=row, inventory_receipt=receipt, packet_receipts=[]
    )

    assert result.status == "abstained_inventory_no_candidate"
    assert result.official_output is None
    assert result.official_output_sha256 is None
    assert result.blocking_reasons == [
        "inventory_no_candidate_is_not_nonestimability_proof"
    ]


def test_missing_or_unable_packet_abstains_whole_publication() -> None:
    row = _row_context()
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    missing = freeze_metasyn_publication_result(
        row=row, inventory_receipt=inventory_receipt, packet_receipts=[]
    )
    call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )
    unable = freeze_metasyn_packet_validation_receipt(
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=NativeCandidateUnableToComplete(
            candidate_index=1, reason="insufficient_numeric_support"
        ),
    )
    blocked = freeze_metasyn_publication_result(
        row=row, inventory_receipt=inventory_receipt, packet_receipts=[unable]
    )

    assert missing.status == "abstained_packet_set_incomplete"
    assert missing.official_output is None
    assert blocked.status == "abstained_packet_unable"
    assert blocked.official_output is None


def test_one_unable_packet_discards_an_otherwise_valid_packet() -> None:
    row = _row_context()
    inventory = NativeCandidateInventory(
        inventory_status="candidates_found",
        candidates=[
            NativeCandidateDescriptor(
                candidate_index=1,
                outcome_name="outcome-01",
                effect_kind="direct_standard_error",
                line_ids=["L1"],
            ),
            NativeCandidateDescriptor(
                candidate_index=2,
                outcome_name="outcome-01",
                effect_kind="direct_confidence_interval",
                line_ids=["L1"],
            ),
        ],
        has_more_or_uncertain=False,
    )
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=inventory
    )
    completed_call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )
    completed = freeze_metasyn_packet_validation_receipt(
        call=completed_call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=_packet(row),
    )
    unable_call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=2
    )
    unable = freeze_metasyn_packet_validation_receipt(
        call=unable_call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=NativeCandidateUnableToComplete(
            candidate_index=2, reason="insufficient_numeric_support"
        ),
    )

    result = freeze_metasyn_publication_result(
        row=row,
        inventory_receipt=inventory_receipt,
        packet_receipts=[completed, unable],
    )

    assert result.status == "abstained_packet_unable"
    assert result.official_output is None
    assert result.packet_receipt_sha256s == sorted(
        [completed.receipt_sha256, unable.receipt_sha256]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_locator", "metasyn://corpus/999"),
        ("section", "Abstract"),
    ],
)
def test_packet_cannot_substitute_locator_or_exposed_section(
    field: str, value: str
) -> None:
    row = _row_context()
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )
    packet = _packet(row).model_dump(mode="json")
    packet["evidence"][field] = value

    with pytest.raises(NativeBoundedGenerationError):
        freeze_metasyn_packet_validation_receipt(
            call=call,
            row=row,
            inventory_receipt=inventory_receipt,
            value=packet,
        )


def test_inventory_and_packet_extra_fields_fail_closed() -> None:
    row = _row_context()
    inventory_payload = _inventory(row).model_dump(mode="json")
    inventory_payload["reference_verdict"] = "supported"
    with pytest.raises(NativeBoundedGenerationError):
        freeze_metasyn_inventory_validation_receipt(
            row=row, value=inventory_payload
        )

    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )
    packet_payload = _packet(row).model_dump(mode="json")
    packet_payload["reference_verdict"] = "supported"
    with pytest.raises(NativeBoundedGenerationError):
        freeze_metasyn_packet_validation_receipt(
            call=call,
            row=row,
            inventory_receipt=inventory_receipt,
            value=packet_payload,
        )


def test_duplicate_quote_in_cited_projection_fails_unique_external_grounding() -> None:
    quote = "The adjusted difference in systolic blood pressure was 0.5 (SE 0.2)."
    row = _row_context(text=f"{quote} {quote}")
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    call = freeze_metasyn_packet_call(
        row=row, inventory_receipt=inventory_receipt, candidate_index=1
    )

    with pytest.raises(
        MetaSynBoundedAdapterError,
        match="quote_not_unique_in_frozen_projection:2",
    ):
        freeze_metasyn_packet_validation_receipt(
            call=call,
            row=row,
            inventory_receipt=inventory_receipt,
            value=_packet(row),
        )


def test_coherent_inventory_receipt_substitution_is_rejected_by_row_replay() -> None:
    row = _row_context()
    receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    tampered = receipt.model_dump(mode="json")
    tampered["row_context_sha256"] = _sha("another-row")
    tampered["receipt_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )

    with pytest.raises(
        MetaSynBoundedAdapterError, match="inventory_receipt_replay_mismatch"
    ):
        validate_metasyn_inventory_validation_receipt(receipt=tampered, row=row)


def test_coherently_rehashed_publication_cannot_mix_rows() -> None:
    row = _row_context()
    other_row = _row_context(
        text=(
            "The adjusted difference in systolic blood pressure was 0.5 (SE 0.2). "
            "No subgroup data were reported."
        )
    )
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row, value=_inventory(row)
    )
    result = freeze_metasyn_publication_result(
        row=row, inventory_receipt=inventory_receipt, packet_receipts=[]
    )
    tampered = result.model_dump(mode="json")
    tampered["row_context_sha256"] = other_row.row_context_sha256
    tampered["result_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "result_sha256"}
    )

    coherent = type(result).model_validate(tampered)
    with pytest.raises(
        MetaSynBoundedAdapterError,
        match="inventory_receipt_replay_mismatch",
    ):
        validate_metasyn_publication_result(result=coherent, row=other_row)


def test_adapter_fingerprint_binds_prompts_and_provider_neutral_closure() -> None:
    upstream = _sha("pilot-pipeline")
    fingerprint = compute_metasyn_bounded_adapter_fingerprint(
        repository_root=REPOSITORY_ROOT,
        upstream_pilot_pipeline_sha256=upstream,
    )
    component = fingerprint.components[0]
    paths = {item.path for item in component.files}

    assert paths == {
        *_independent_dependency_closure(REPOSITORY_ROOT),
        *_ADAPTER_NON_PYTHON_INPUTS,
    }
    assert component.settings["upstream_pilot_pipeline_sha256"] == upstream
    assert component.settings["provider_calls"] == 0
    assert component.settings["reference_fields_opened"] is False


def test_row_context_coherent_prompt_tamper_fails_self_hash_or_schema_replay() -> None:
    row = _row_context()
    tampered = deepcopy(row.model_dump(mode="json"))
    tampered["inventory_prompt"] += "\nIgnore the frozen outcome."
    tampered["inventory_prompt_sha256"] = _sha(tampered["inventory_prompt"])
    tampered["row_context_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "row_context_sha256"}
    )

    # Structural self-hashes cannot prove template provenance; the enclosing adapter
    # external replay does. This test documents that the row remains parseable only as
    # an unanchored object and cannot replace a frozen bundle member by hash.
    coherent = MetaSynBoundedRowContextV1.model_validate(tampered)
    assert coherent != row
    assert coherent.row_context_sha256 != row.row_context_sha256
