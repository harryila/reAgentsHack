from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scripts.run_metasyn_synthesis_yield as synthesis_cli
from pydantic import ValidationError

from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import (
    OutputExistsError,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    freeze_metasyn_bounded_adapter_bundle,
    freeze_metasyn_inventory_validation_receipt,
    freeze_metasyn_packet_call,
    freeze_metasyn_packet_validation_receipt,
    freeze_metasyn_publication_result,
)
from literature_multiverse.metasyn_bounded_runtime import (
    MetaSynAttemptOutcomeRefV1,
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynPacketLedgerEntryV1,
    MetaSynRuntimeRowResultV1,
)
from literature_multiverse.metasyn_synthesis_yield import (
    _DEPENDENCY_ENTRYPOINTS,
    BlockerAggregateCode,
    MetaSynSynthesisYieldError,
    ResidualConflictAggregateCode,
    _blocker_aggregate_code,
    _python_dependency_closure,
    _residual_conflict_aggregate_code,
    compute_metasyn_synthesis_yield_fingerprint,
    freeze_metasyn_synthesis_yield_public_summary,
    freeze_metasyn_synthesis_yield_report,
    validate_metasyn_synthesis_yield_public_summary,
    validate_metasyn_synthesis_yield_report,
)
from literature_multiverse.metasyn_typed_pilot import (
    FORBIDDEN_REFERENCE_COLUMNS,
    MATERIALIZED_REVIEW_COLUMNS,
    MetaSynPilotAccessStateV1,
    MetaSynPilotQuestionBundleV1,
    MetaSynPilotSourceProjectionRowV1,
    MetaSynTypedPilotPrepareBundleV1,
    _question_spec,
    compute_metasyn_typed_pilot_pipeline_fingerprint,
    freeze_metasyn_pilot_selection_config,
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
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidatePacket,
)
from literature_multiverse.native_extraction import (
    NativeSourceManifest,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import resolve_native_source_document
from literature_multiverse.native_question_projection import (
    freeze_question_projection_spec,
    project_resolved_source_for_question,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _protocol_row(review_id: int) -> dict[str, Any]:
    return {
        "ID": review_id,
        "Title": f"Synthetic review {review_id}",
        "Research_Question": f"Does intervention {review_id} change outcome {review_id}?",
        "Population": f"Population {review_id}",
        "Intervention": f"Intervention {review_id}",
        "Exposure": None,
        "Comparison": f"Comparator {review_id}",
        "Outcome": f"Outcome {review_id}",
        "inclusion_criteria": "Controlled studies",
        "exclusion_criteria": None,
        "search_end_date": "2025-01-01",
        "matched_corpus_ids": [review_id * 10, review_id * 10 + 1],
        "matched_ref_count": 2,
        "source_review_corpus_ids": [],
    }


def _source_text(review_id: int, *, two_lines: bool = False) -> str:
    first = (
        f"Outcome {review_id} adjusted difference at 4 weeks was 0.5 "
        "with standard error 0.2."
    )
    if not two_lines:
        return first
    return (
        first
        + "\n"
        + f"Outcome {review_id} adjusted difference at 4 weeks was 0.7 "
        "with standard error 0.3."
    )


def _write_source(
    *, directory: Path, corpus_id: int, review_id: int, full_text: bool, two_lines: bool
) -> SourceDocumentArtifact:
    relative = directory.relative_to(REPOSITORY_ROOT) / f"{corpus_id}.parquet"
    path = REPOSITORY_ROOT / relative
    text = _source_text(review_id, two_lines=two_lines)
    row = {
        "ID": corpus_id,
        "title": None,
        "abstract": None if full_text else text,
        "sections": ([{"heading": "Results", "text": text}] if full_text else []),
    }
    pq.write_table(pa.Table.from_pylist([row]), path)
    return SourceDocumentArtifact(
        artifact_path=relative.as_posix(),
        sha256=sha256_file(path),
        media_type="application/vnd.apache.parquet",
        source_locator=(
            f"parquet:{relative.as_posix()}#row_group=0&row_in_group=0"
            f"&index_base=0&ID={corpus_id}"
        ),
    )


def _source_row(
    *,
    question: Any,
    corpus_id: int,
    source_document: SourceDocumentArtifact,
) -> MetaSynPilotSourceProjectionRowV1:
    source_record = NativeSourceRecord(
        doc_id=f"metasyn-corpus:{corpus_id}",
        publication=PublicationIdentity(
            publication_id=f"metasyn-publication:{corpus_id}",
            paper_id=f"metasyn-paper:{corpus_id}",
            doc_id=f"metasyn-corpus:{corpus_id}",
        ),
        source_document=source_document,
    )
    resolved = resolve_native_source_document(
        repository_root=REPOSITORY_ROOT, source_document=source_document
    )
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
    projection = project_resolved_source_for_question(
        row_id=source_record.doc_id, source=resolved, spec=projection_spec
    )
    full_text = projection.source_modality.startswith("full_text_")
    payload: dict[str, Any] = {
        "source_row_version": "metasyn-typed-oracle-source-row-v1",
        "question_id": question.question_id,
        "corpus_id": corpus_id,
        "doc_id": source_record.doc_id,
        "source_record": source_record,
        "diagnostic_source_record_sha256": _sha(f"diagnostic-{corpus_id}"),
        "source_content_scope": (
            "full_text_sections" if full_text else "title_abstract"
        ),
        "oracle_selection_full_text_scope": full_text,
        "source_projection_strength": projection.source_strength,
        "release_grade_source_grounding_eligible": (
            projection.release_grade_source_grounding_eligible
        ),
        "source_strength_blockers": projection.source_strength_blockers,
        "projection": projection,
        "projection_sha256": projection.projection_sha256,
    }
    return MetaSynPilotSourceProjectionRowV1.model_validate(
        {**payload, "source_row_sha256": hash_canonical(payload)}
    )


def _question_bundle(
    *, question: Any, source_rows: list[MetaSynPilotSourceProjectionRowV1]
) -> MetaSynPilotQuestionBundleV1:
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
    records = sorted(
        [item.source_record for item in source_rows], key=lambda item: item.doc_id
    )
    manifest = NativeSourceManifest(question_id=question.question_id, records=records)
    corpus_ids = sorted(item.corpus_id for item in source_rows)
    component_ids = [question.review_id]
    payload: dict[str, Any] = {
        "question_bundle_version": "metasyn-typed-oracle-question-bundle-v1",
        "question_spec": question,
        "question_spec_sha256": question.question_spec_sha256,
        "projection_spec": projection_spec,
        "projection_spec_sha256": projection_spec.projection_spec_sha256,
        "independence_component_id": f"metasyn-component-{question.review_id}",
        "independence_component_review_ids": component_ids,
        "independence_component_membership_sha256": hash_canonical(component_ids),
        "oracle_corpus_ids": corpus_ids,
        "oracle_roster_membership_sha256": hash_canonical(
            {"question_id": question.question_id, "oracle_corpus_ids": corpus_ids}
        ),
        "source_manifest": manifest,
        "source_manifest_sha256": hash_canonical(manifest),
        "source_rows": sorted(source_rows, key=lambda item: item.corpus_id),
    }
    return MetaSynPilotQuestionBundleV1.model_validate(
        {**payload, "question_bundle_sha256": hash_canonical(payload)}
    )


def _prepare_bundle(
    questions: list[MetaSynPilotQuestionBundleV1],
) -> MetaSynTypedPilotPrepareBundleV1:
    pilot = compute_metasyn_typed_pilot_pipeline_fingerprint(root=REPOSITORY_ROOT)
    selection = freeze_metasyn_pilot_selection_config()
    access = MetaSynPilotAccessStateV1(
        materialized_review_columns=list(MATERIALIZED_REVIEW_COLUMNS),
        forbidden_reference_columns=sorted(FORBIDDEN_REFERENCE_COLUMNS),
    )
    source_rows = [row for question in questions for row in question.source_rows]
    modalities = dict(sorted(Counter(row.source_content_scope for row in source_rows).items()))
    strengths = dict(
        sorted(Counter(row.source_projection_strength for row in source_rows).items())
    )
    question_ids = [item.question_spec.question_id for item in questions]
    component_ids = [item.independence_component_id for item in questions]
    repository_inputs = {
        "corpus_manifest": "configs/benchmarks/metasyn-corpus-c8fa07d.json",
        "reviews_train": "data/cache/metasyn/reviews-train.parquet",
        "screening_fit_receipt": "data/cache/metasyn/screening/fit.json",
        "screening_winner_rankings": "data/cache/metasyn/screening/rankings.jsonl",
    }
    repository_inputs = dict(sorted(repository_inputs.items()))
    payload: dict[str, Any] = {
        "prepare_bundle_version": "metasyn-typed-oracle-prepare-bundle-v1",
        "pilot_version": "metasyn-typed-oracle-pilot-v1",
        "status": "prepared_predictions_and_reference_fields_unopened",
        "pilot_pipeline_fingerprint": pilot,
        "pilot_pipeline_sha256": pilot.pipeline_sha256,
        "downstream_verifier_pipeline_sha256": pilot.components[0].settings[
            "downstream_verifier_pipeline_sha256"
        ],
        "official_native_extraction_schema_sha256": hash_canonical(
            native_publication_extraction_json_schema()
        ),
        "selection_config": selection,
        "selection_config_sha256": selection.selection_config_sha256,
        "access_state": access,
        "repository_inputs": repository_inputs,
        "repository_input_sha256s": dict(
            sorted((key, _sha(f"input-{key}")) for key in repository_inputs)
        ),
        "screening_fit_payload_sha256": _sha("screening-fit"),
        "screening_winner_rankings_sha256": _sha("screening-rankings"),
        "corpus_source_revision": "synthetic-fixture-revision",
        "calibration_source_inventory_sha256": _sha("calibration-source"),
        "calibration_question_count": 161,
        "calibration_component_count": 10,
        "selected_question_count": 10,
        "selected_component_count": 10,
        "selected_paper_count": 32,
        "selected_unique_paper_count": 32,
        "source_modality_counts": modalities,
        "source_strength_counts": strengths,
        "release_grade_source_grounding_count": sum(
            row.release_grade_source_grounding_eligible for row in source_rows
        ),
        "selected_question_membership_sha256": hash_canonical(question_ids),
        "selected_component_membership_sha256": hash_canonical(sorted(component_ids)),
        "selected_oracle_roster_membership_sha256": hash_canonical(
            sorted(item.oracle_roster_membership_sha256 for item in questions)
        ),
        "questions": questions,
    }
    return MetaSynTypedPilotPrepareBundleV1.model_validate(
        {**payload, "prepare_bundle_sha256": hash_canonical(payload)}
    )


def _support(*, field: str, token: str, quote: str) -> BoundedNumericSupport:
    start = quote.rindex(token) if field == "finding.timepoint.value" else quote.index(token)
    return BoundedNumericSupport(
        field_path=field,
        verbatim_token=token,
        quote_start=str(start),
        quote_end=str(start + len(token)),
    )


def _packet(
    *,
    row: Any,
    candidate_index: int,
    estimate: str,
    standard_error: str,
    line_id: str,
    cohort_key: str,
    contrast_label: str,
) -> NativeCandidatePacket[DirectStandardErrorEffect]:
    passage = next(
        item for item in row.source_row.projection.passages if item.line_id == line_id
    )
    quote = passage.text
    return NativeCandidatePacket[DirectStandardErrorEffect](
        candidate_index=candidate_index,
        study=BoundedStudyHeader(
            key="study-1", source_label="Synthetic Study", registration_ids=[]
        ),
        cohort=BoundedCohortHeader(
            key=cohort_key,
            source_labels=[cohort_key],
            registry_ids=[],
            dataset_ids=[],
        ),
        treatment_arm=BoundedArm(
            key=f"treatment-{cohort_key}", label="Intervention", role="intervention"
        ),
        comparator_arm=BoundedArm(
            key=f"control-{cohort_key}", label="Comparator", role="comparator"
        ),
        contrast=BoundedContrast(
            key=f"contrast-{cohort_key}",
            label=contrast_label,
            estimand=row.question_spec.contrast_estimand,
            positive_direction_means=row.outcome_positive_directions["outcome-01"],
        ),
        finding=BoundedFindingHeader(
            key=f"finding-{candidate_index}",
            outcome_name="outcome-01",
            timepoint=BoundedTimepoint(kind="exact", value="4", unit="week"),
        ),
        effect=DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate=estimate,
            standard_error=standard_error,
            unit="points",
        ),
        evidence=BoundedEvidence(
            source_locator=row.source_locator,
            quote=quote,
            section=passage.exposed_section,
            line_ids=[line_id],
        ),
        numeric_support=[
            _support(field="effect.estimate", token=estimate, quote=quote),
            _support(
                field="effect.standard_error", token=standard_error, quote=quote
            ),
            _support(field="finding.timepoint.value", token="4", quote=quote),
        ],
    )


def _attempt_ref(label: str, status: str) -> MetaSynAttemptOutcomeRefV1:
    return MetaSynAttemptOutcomeRefV1(
        attempt_id=_sha(f"attempt-{label}"),
        state="response",
        attempt_intent_sha256=_sha(f"intent-{label}"),
        response_receipt_sha256=_sha(f"receipt-{label}"),
        response_status=status,
        incident_sha256=None,
        incident_kind=None,
    )


def _runtime_row(
    *, row: Any, mode: str, contrast_label: str = "intervention_vs_comparator"
) -> MetaSynRuntimeRowResultV1:
    if mode == "runtime_blocked":
        adapter_result = None
        inventory_attempt = _attempt_ref(
            row.row_context_sha256, "inventory_generation_truncated"
        )
        packet_attempts = []
        status = "runtime_inventory_blocked"
        blockers = ["inventory_response:inventory_generation_truncated"]
        findings = 0
    elif mode == "no_candidate":
        inventory = NativeCandidateInventory(
            inventory_status="no_candidate_found",
            candidates=[],
            has_more_or_uncertain=False,
        )
        inventory_receipt = freeze_metasyn_inventory_validation_receipt(
            row=row, value=inventory
        )
        adapter_result = freeze_metasyn_publication_result(
            row=row, inventory_receipt=inventory_receipt, packet_receipts=[]
        )
        inventory_attempt = _attempt_ref(
            row.row_context_sha256, "inventory_valid_no_candidate_non_authorizing"
        )
        packet_attempts: list[MetaSynPacketLedgerEntryV1] = []
        status = "adapter_inventory_no_candidate"
        blockers = adapter_result.blocking_reasons
        findings = 0
    else:
        candidate_count = 2 if mode == "two_candidates" else 1
        candidates = [
            NativeCandidateDescriptor(
                candidate_index=index,
                outcome_name="outcome-01",
                effect_kind="direct_standard_error",
                line_ids=[f"L{index}"],
            )
            for index in range(1, candidate_count + 1)
        ]
        inventory = NativeCandidateInventory(
            inventory_status="candidates_found",
            candidates=candidates,
            has_more_or_uncertain=False,
        )
        inventory_receipt = freeze_metasyn_inventory_validation_receipt(
            row=row, value=inventory
        )
        receipts = []
        packet_attempts = []
        for candidate in candidates:
            index = candidate.candidate_index
            estimate, standard_error = (
                ("0.5", "0.2") if index == 1 else ("0.7", "0.3")
            )
            call = freeze_metasyn_packet_call(
                row=row,
                inventory_receipt=inventory_receipt,
                candidate_index=index,
            )
            receipt = freeze_metasyn_packet_validation_receipt(
                call=call,
                row=row,
                inventory_receipt=inventory_receipt,
                value=_packet(
                    row=row,
                    candidate_index=index,
                    estimate=estimate,
                    standard_error=standard_error,
                    line_id=f"L{index}",
                    cohort_key=f"cohort-{index}",
                    contrast_label=contrast_label,
                ),
            )
            receipts.append(receipt)
            packet_attempts.append(
                MetaSynPacketLedgerEntryV1(
                    candidate_index=index,
                    candidate_sha256=candidate.descriptor_sha256,
                    outcome=_attempt_ref(
                        f"{row.row_context_sha256}-packet-{index}",
                        "packet_completed",
                    ),
                )
            )
        adapter_result = freeze_metasyn_publication_result(
            row=row,
            inventory_receipt=inventory_receipt,
            packet_receipts=receipts,
        )
        inventory_attempt = _attempt_ref(
            row.row_context_sha256, "inventory_valid_candidates"
        )
        status = "typed_publication_output"
        blockers = []
        findings = candidate_count
    payload: dict[str, Any] = {
        "row_result_version": "metasyn-bounded-runtime-row-result-v1",
        "row_context_sha256": row.row_context_sha256,
        "question_spec_sha256": row.question_spec_sha256,
        "question_bundle_sha256": row.question_bundle_sha256,
        "source_row_sha256": row.source_row_sha256,
        "independence_component_membership_sha256": (
            row.independence_component_membership_sha256
        ),
        "source_strength": row.source_row.source_projection_strength,
        "release_grade_source_grounding_eligible": (
            row.source_row.release_grade_source_grounding_eligible
        ),
        "status": status,
        "runtime_blockers": sorted(blockers),
        "inventory_attempt": inventory_attempt,
        "packet_attempts": packet_attempts,
        "adapter_publication_result": adapter_result,
        "adapter_publication_result_sha256": (
            adapter_result.result_sha256 if adapter_result is not None else None
        ),
        "observed_source_generation_calls": 1 + len(packet_attempts),
        "possible_ambiguous_source_generation_calls": 0,
        "typed_finding_count": findings,
        "synthesis_input_eligible": (
            status == "typed_publication_output"
            and row.source_row.release_grade_source_grounding_eligible
        ),
    }
    return MetaSynRuntimeRowResultV1.model_validate(
        {**payload, "row_result_sha256": hash_canonical(payload)}
    )


def _runtime_report(
    *, adapter: MetaSynBoundedAdapterBundleV1, modes: dict[str, tuple[str, str]]
) -> MetaSynBoundedPrivateYieldReportV1:
    rows = [
        _runtime_row(
            row=row,
            mode=modes.get(row.row_key, ("no_candidate", ""))[0],
            contrast_label=modes.get(row.row_key, ("no_candidate", ""))[1]
            or "intervention_vs_comparator",
        )
        for row in adapter.row_contexts
    ]
    rows.sort(key=lambda item: item.row_context_sha256)
    typed = [item for item in rows if item.status == "typed_publication_output"]
    release_grade = [
        item for item in typed if item.release_grade_source_grounding_eligible
    ]
    inventory_counts = dict(
        sorted(Counter(item.inventory_attempt.response_status for item in rows).items())
    )
    packet_counts = dict(
        sorted(
            Counter(
                packet.outcome.response_status
                for item in rows
                for packet in item.packet_attempts
            ).items()
        )
    )
    observed = sum(item.observed_source_generation_calls for item in rows)
    downstream = adapter.row_contexts[0].question_spec.question_spec_sha256
    # The exact downstream hash is replaced by the prepare-bound value by the fixture.
    payload: dict[str, Any] = {
        "report_version": "metasyn-bounded-runtime-private-yield-report-v1",
        "status": "complete_32_row_yield_only_runtime_report",
        "execution_bundle_sha256": _sha("execution-bundle"),
        "runtime_pipeline_sha256": _sha("runtime-pipeline"),
        "config_sha256": _sha("runtime-config"),
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "downstream_verifier_pipeline_sha256": downstream,
        "preflight_sha256": _sha("preflight"),
        "prediction_ledger_sha256": _sha("ledger"),
        "model_identity_sha256": _sha("model"),
        "question_count": 10,
        "component_count": 10,
        "publication_count": 32,
        "row_membership_sha256": adapter.row_membership_sha256,
        "row_results": rows,
        "row_result_sha256s": sorted(item.row_result_sha256 for item in rows),
        "row_status_counts": dict(sorted(Counter(item.status for item in rows).items())),
        "inventory_response_status_counts": inventory_counts,
        "packet_response_status_counts": packet_counts,
        "ambiguity_incident_kind_counts": {},
        "typed_publication_output_count": len(typed),
        "release_grade_typed_publication_count": len(release_grade),
        "diagnostic_only_typed_publication_count": len(typed) - len(release_grade),
        "typed_finding_count": sum(item.typed_finding_count for item in typed),
        "questions_with_any_release_grade_typed_publication": len(
            {item.question_spec_sha256 for item in release_grade}
        ),
        "synthesis_attempt_input_publication_count": len(release_grade),
        "observed_source_generation_calls": observed,
        "possible_ambiguous_source_generation_calls": 0,
        "total_possible_source_generation_call_attempts": observed,
        "synthetic_preflight_call_attempts": 2,
        "generation_retries": 0,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
        "synthesis_input_caveat": (
            "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_"
            "correctness"
        ),
    }
    return MetaSynBoundedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def _rebind_runtime_downstream(
    runtime: MetaSynBoundedPrivateYieldReportV1, downstream_sha256: str
) -> MetaSynBoundedPrivateYieldReportV1:
    payload = runtime.model_dump(mode="json", exclude={"report_sha256"})
    payload["downstream_verifier_pipeline_sha256"] = downstream_sha256
    return MetaSynBoundedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


@dataclass(frozen=True)
class SyntheticInputs:
    directory: Path
    prepare: MetaSynTypedPilotPrepareBundleV1
    adapter: MetaSynBoundedAdapterBundleV1
    runtime: MetaSynBoundedPrivateYieldReportV1
    stale_source_path: Path


@pytest.fixture(scope="module")
def synthetic_inputs() -> Any:
    directory = (
        REPOSITORY_ROOT
        / "data"
        / "cache"
        / "metasyn"
        / f"synthesis-yield-test-{uuid.uuid4().hex}"
    )
    directory.mkdir(parents=True)
    try:
        questions: list[MetaSynPilotQuestionBundleV1] = []
        corpus_id = 1000
        for question_index in range(10):
            review_id = 100 + question_index
            question = _question_spec(_protocol_row(review_id))
            paper_count = 4 if question_index < 2 else 3
            source_rows = []
            for paper_index in range(paper_count):
                full_text = paper_index < 2 or (
                    question_index < 2 and paper_index == 2
                )
                document = _write_source(
                    directory=directory,
                    corpus_id=corpus_id,
                    review_id=review_id,
                    full_text=full_text,
                    two_lines=question_index == 0 and paper_index == 0,
                )
                source_rows.append(
                    _source_row(
                        question=question,
                        corpus_id=corpus_id,
                        source_document=document,
                    )
                )
                corpus_id += 1
            questions.append(_question_bundle(question=question, source_rows=source_rows))
        prepare = _prepare_bundle(questions)
        adapter = freeze_metasyn_bounded_adapter_bundle(
            prepare_bundle=prepare, repository_root=REPOSITORY_ROOT
        )
        by_question: dict[str, list[Any]] = {}
        for row in adapter.row_contexts:
            by_question.setdefault(row.question_spec.question_id, []).append(row)
        for rows in by_question.values():
            rows.sort(key=lambda item: item.source_row.corpus_id)
        modes: dict[str, tuple[str, str]] = {}
        question_ids = sorted(by_question)
        modes[by_question[question_ids[0]][0].row_key] = (
            "two_candidates",
            "intervention_vs_comparator",
        )
        for row in by_question[question_ids[1]][:2]:
            modes[row.row_key] = ("one_candidate", "intervention_vs_comparator")
        modes[by_question[question_ids[2]][0].row_key] = (
            "one_candidate",
            "intervention_vs_comparator",
        )
        modes[by_question[question_ids[2]][1].row_key] = (
            "one_candidate",
            "alternate_intervention_vs_comparator",
        )
        diagnostic_row = by_question[question_ids[3]][-1]
        assert not diagnostic_row.source_row.release_grade_source_grounding_eligible
        modes[diagnostic_row.row_key] = (
            "one_candidate",
            "intervention_vs_comparator",
        )
        stale_row = by_question[question_ids[4]][0]
        modes[stale_row.row_key] = (
            "one_candidate",
            "intervention_vs_comparator",
        )
        runtime_blocked_row = by_question[question_ids[5]][0]
        modes[runtime_blocked_row.row_key] = (
            "runtime_blocked",
            "intervention_vs_comparator",
        )
        runtime = _runtime_report(adapter=adapter, modes=modes)
        runtime = _rebind_runtime_downstream(
            runtime, prepare.downstream_verifier_pipeline_sha256
        )
        stale_path = (
            REPOSITORY_ROOT
            / stale_row.source_row.source_record.source_document.artifact_path
        )
        stale_path.write_bytes(stale_path.read_bytes() + b"stale")
        yield SyntheticInputs(
            directory=directory,
            prepare=prepare,
            adapter=adapter,
            runtime=runtime,
            stale_source_path=stale_path,
        )
    finally:
        shutil.rmtree(directory)


def test_full_fake_roster_replays_sources_and_yields_only_safe_synthesis(
    synthetic_inputs: SyntheticInputs,
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )

    assert report.terminal_fragment_count == 32
    assert report.runtime_contract_typed_publication_count == 7
    assert report.diagnostic_only_typed_publication_count == 1
    assert report.original_source_grounding_attempt_count == 7
    # The title/abstract row is replayed but its adapter output is deliberately
    # non-estimable, so only the five intact full-text outputs authorize promotion.
    assert report.original_source_grounding_authorized_count == 5
    assert report.release_grade_estimable_publication_count == 5
    assert report.graph_construction_completed_question_count == 10
    assert report.synthesis_completed_group_count == 1
    assert report.synthesis_completion_mode_counts == {
        "random_effects_meta_analysis": 1
    }
    assert report.synthesis_group_stage_counts[
        "blocked_cross_publication_identity"
    ] == 1
    assert report.residual_conflict_counts["multiple_effect_strata"] == 1
    assert report.blocker_counts["runtime_terminal_exclusion"] == 10
    stages = Counter(
        publication.stage
        for question in report.question_reports
        for publication in question.publication_records
    )
    assert stages["diagnostic_only_fragment_excluded"] == 1
    assert stages["original_source_grounding_failed"] == 1
    completed = [
        group
        for question in report.question_reports
        for group in question.compatibility_groups
        if group.synthesis_completed
    ]
    assert len(completed) == 1
    assert len(completed[0].paper_ids) == 1
    assert len(completed[0].cohort_ids) == 2
    grounded = next(
        publication
        for question in report.question_reports
        for publication in question.publication_records
        if publication.stage == "release_grade_fragment_estimable"
    )
    assert grounded.adapter_quote_groundings
    assert grounded.adapter_quote_groundings[0].source_char_start >= 0
    assert grounded.original_source_grounding_receipt is not None
    assert grounded.terminal_fragment.graph is not None
    assert grounded.terminal_fragment.graph.evidence_spans[0].quote

    assert validate_metasyn_synthesis_yield_report(
        report=report,
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    ) == report


def test_public_summary_is_aggregate_only_and_exactly_replayable(
    synthetic_inputs: SyntheticInputs,
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    summary = freeze_metasyn_synthesis_yield_public_summary(report=report)
    payload = summary.model_dump(mode="json")
    forbidden = {
        "question_id",
        "publication_id",
        "paper_id",
        "study_id",
        "cohort_id",
        "estimate_id",
        "source_locator",
        "quote",
        "effect_direction",
        "conclusion_summary",
    }

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {str(key).casefold() for key in value} | set().union(
                *(keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert not keys(payload).intersection(forbidden)
    assert summary.direction_agreement_reported is False
    assert summary.extraction_accuracy_reported is False
    assert summary.claim_release_authority is False
    assert summary.synthesis_private_report_sha256 == report.report_sha256
    assert validate_metasyn_synthesis_yield_public_summary(
        summary=summary, report=report
    ) == summary

    tampered = payload | {"runtime_private_report_sha256": "0" * 64}
    tampered["summary_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "summary_sha256"}
    )
    with pytest.raises(
        MetaSynSynthesisYieldError,
        match="public_summary_external_replay_mismatch",
    ):
        validate_metasyn_synthesis_yield_public_summary(
            summary=tampered, report=report
        )


@pytest.mark.parametrize(
    ("field", "injected_key"),
    [
        ("blocker_counts", "metasyn-review-deadbeef"),
        ("blocker_counts", "/Users/private/source.txt"),
        ("blocker_counts", "outcome improved in the intervention arm"),
        ("residual_conflict_counts", "parquet:private-source#row=17"),
    ],
)
def test_public_summary_rejects_rehashed_dynamic_aggregate_keys(
    synthetic_inputs: SyntheticInputs,
    field: str,
    injected_key: str,
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    summary = freeze_metasyn_synthesis_yield_public_summary(report=report)
    payload = summary.model_dump(mode="json")
    payload[field][injected_key] = 1
    payload["summary_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "summary_sha256"}
    )

    with pytest.raises(ValidationError):
        type(summary).model_validate(payload)


def test_public_aggregate_vocabulary_is_closed_at_construction_and_validation(
    synthetic_inputs: SyntheticInputs,
) -> None:
    with pytest.raises(
        MetaSynSynthesisYieldError,
        match="unknown_blocker_aggregate_code",
    ):
        _blocker_aggregate_code("unregistered scientific narrative")
    with pytest.raises(
        MetaSynSynthesisYieldError,
        match="unknown_residual_conflict_aggregate_code",
    ):
        _residual_conflict_aggregate_code("metasyn-publication:private")

    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    summary = freeze_metasyn_synthesis_yield_public_summary(report=report)
    payload = summary.model_dump(mode="json")
    assert set(payload["blocker_counts"]).issubset(
        {item.value for item in BlockerAggregateCode}
    )
    assert set(payload["residual_conflict_counts"]).issubset(
        {item.value for item in ResidualConflictAggregateCode}
    )

    injected = json.loads(json.dumps(payload))
    injected["publication_stage_counts"]["private trial name"] = 1
    injected["summary_sha256"] = hash_canonical(
        {key: value for key, value in injected.items() if key != "summary_sha256"}
    )
    with pytest.raises(ValidationError):
        type(summary).model_validate(injected)


def test_upstream_identity_mismatch_fails_before_source_or_synthesis(
    synthetic_inputs: SyntheticInputs,
) -> None:
    payload = synthetic_inputs.adapter.model_dump(
        mode="json", exclude={"adapter_bundle_sha256"}
    )
    payload["prepare_bundle_sha256"] = _sha("different-prepare")
    forged = MetaSynBoundedAdapterBundleV1.model_validate(
        {**payload, "adapter_bundle_sha256": hash_canonical(payload)}
    )
    runtime_payload = synthetic_inputs.runtime.model_dump(
        mode="json", exclude={"report_sha256"}
    )
    runtime_payload["adapter_bundle_sha256"] = forged.adapter_bundle_sha256
    forged_runtime = MetaSynBoundedPrivateYieldReportV1.model_validate(
        {**runtime_payload, "report_sha256": hash_canonical(runtime_payload)}
    )
    with pytest.raises(
        MetaSynSynthesisYieldError, match="adapter_prepare_mismatch"
    ):
        freeze_metasyn_synthesis_yield_report(
            repository_root=REPOSITORY_ROOT,
            runtime_report=forged_runtime,
            adapter_bundle=forged,
            prepare_bundle=synthetic_inputs.prepare,
        )


def test_fingerprint_binds_dependency_closure_and_all_upstream_anchors(
    synthetic_inputs: SyntheticInputs,
) -> None:
    fingerprint = compute_metasyn_synthesis_yield_fingerprint(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    component = fingerprint.components[0]
    paths = {item.path for item in component.files}

    assert set(_DEPENDENCY_ENTRYPOINTS).issubset(paths)
    assert set(_python_dependency_closure(REPOSITORY_ROOT)).issubset(paths)
    assert {
        "src/literature_multiverse/native_grounding.py",
        "src/literature_multiverse/cohort_reconciliation.py",
        "src/literature_multiverse/meta_analysis.py",
        "src/literature_multiverse/typed_extraction.py",
    }.issubset(paths)
    assert component.settings["runtime_private_report_sha256"] == (
        synthetic_inputs.runtime.report_sha256
    )
    assert component.settings["adapter_bundle_sha256"] == (
        synthetic_inputs.adapter.adapter_bundle_sha256
    )
    assert component.settings["prepare_bundle_sha256"] == (
        synthetic_inputs.prepare.prepare_bundle_sha256
    )
    assert component.settings[
        "reference_direction_conclusion_and_test_labels_opened"
    ] is False


def test_self_hash_rejects_private_group_tampering(
    synthetic_inputs: SyntheticInputs,
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    payload = report.model_dump(mode="json")
    target = next(
        group
        for question in payload["question_reports"]
        for group in question["compatibility_groups"]
        if group["synthesis_completed"]
    )
    target["synthesis_sha256"] = _sha("forged-synthesis")
    with pytest.raises(ValidationError, match="group_hash_mismatch"):
        type(report).model_validate(payload)


def test_provider_neutral_cli_writes_only_fixed_outputs_without_overwrite(
    synthetic_inputs: SyntheticInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    public = freeze_metasyn_synthesis_yield_public_summary(report=report)
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs: Any) -> tuple[Any, Any]:
        calls.append(kwargs)
        return report, public

    monkeypatch.setattr(
        synthesis_cli,
        "evaluate_current_metasyn_synthesis_yield",
        fake_evaluate,
    )
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    runtime_workspace = repository_root / "data/cache/metasyn/finalized-runtime"
    pilot_workspace = repository_root / "data/cache/metasyn/pilot"
    private_output = repository_root / synthesis_cli.PRIVATE_OUTPUT_RELATIVE
    public_output = repository_root / synthesis_cli.PUBLIC_OUTPUT_RELATIVE
    execution_sha256 = _sha("execution-bundle")
    args = [
        "--repository-root",
        str(repository_root),
        "--runtime-workspace",
        str(runtime_workspace),
        "--pilot-workspace",
        str(pilot_workspace),
        "--expected-execution-bundle-sha256",
        execution_sha256,
        "--private-output",
        synthesis_cli.PRIVATE_OUTPUT_RELATIVE.as_posix(),
        "--public-output",
        synthesis_cli.PUBLIC_OUTPUT_RELATIVE.as_posix(),
    ]

    assert synthesis_cli.main(args) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["provider_calls_made"] is False
    assert status["reference_fields_opened"] is False
    assert status["private_report_sha256"] == report.report_sha256
    assert status["public_summary_sha256"] == public.summary_sha256
    assert calls == [
        {
            "repository_root": repository_root.resolve(),
            "runtime_workspace": runtime_workspace,
            "pilot_workspace": pilot_workspace,
            "expected_execution_bundle_sha256": execution_sha256,
        }
    ]
    assert type(report).model_validate(json.loads(private_output.read_bytes())) == report
    assert type(public).model_validate(json.loads(public_output.read_bytes())) == public
    public_payload = json.loads(public_output.read_bytes())

    def nested_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {str(key).casefold() for key in value} | set().union(
                *(nested_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value)) if value else set()
        return set()

    forbidden = {
        "question_id",
        "publication_id",
        "paper_id",
        "study_id",
        "cohort_id",
        "estimate_id",
        "source_locator",
        "quote",
    }
    assert not nested_keys(public_payload).intersection(forbidden)

    private_before = private_output.read_bytes()
    public_before = public_output.read_bytes()
    with pytest.raises(OutputExistsError, match="lineage_output_exists"):
        synthesis_cli.main(args)
    assert len(calls) == 1  # Existing outputs fail before expensive external replay.
    assert private_output.read_bytes() == private_before
    assert public_output.read_bytes() == public_before


def test_cli_rejects_absolute_escape_and_swapped_outputs_before_evaluation_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    calls: list[dict[str, Any]] = []

    def forbidden_evaluate(**kwargs: Any) -> tuple[Any, Any]:
        calls.append(kwargs)
        raise AssertionError("evaluator must not run for an invalid output boundary")

    monkeypatch.setattr(
        synthesis_cli,
        "evaluate_current_metasyn_synthesis_yield",
        forbidden_evaluate,
    )
    base = [
        "--repository-root",
        str(repository_root),
        "--runtime-workspace",
        "data/cache/metasyn/finalized-runtime",
        "--pilot-workspace",
        "data/cache/metasyn/pilot",
        "--expected-execution-bundle-sha256",
        _sha("execution-bundle"),
    ]
    cases = [
        ["--private-output", str(tmp_path / "absolute-private.json")],
        ["--public-output", str(repository_root / synthesis_cli.PUBLIC_OUTPUT_RELATIVE)],
        ["--private-output", "../escaped-private.json"],
        ["--public-output", "artifacts/diagnostics/../escaped-public.json"],
        [
            "--private-output",
            synthesis_cli.PUBLIC_OUTPUT_RELATIVE.as_posix(),
            "--public-output",
            synthesis_cli.PRIVATE_OUTPUT_RELATIVE.as_posix(),
        ],
    ]

    for override in cases:
        with pytest.raises(ValueError, match="must_equal_repository_relative_path"):
            synthesis_cli.main(base + override)
        assert calls == []
        assert not (repository_root / "data").exists()
        assert not (repository_root / "artifacts").exists()
    assert not (tmp_path / "absolute-private.json").exists()


@pytest.mark.parametrize(
    ("symlink_relative", "error_kind"),
    [
        (Path("data/cache"), "private"),
        (Path("artifacts"), "public"),
    ],
)
def test_cli_rejects_fixed_output_symlink_ancestor_before_evaluation_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_relative: Path,
    error_kind: str,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = repository_root / symlink_relative
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to(outside, target_is_directory=True)
    calls: list[dict[str, Any]] = []

    def forbidden_evaluate(**kwargs: Any) -> tuple[Any, Any]:
        calls.append(kwargs)
        raise AssertionError("evaluator must not run through an output symlink")

    monkeypatch.setattr(
        synthesis_cli,
        "evaluate_current_metasyn_synthesis_yield",
        forbidden_evaluate,
    )
    args = [
        "--repository-root",
        str(repository_root),
        "--runtime-workspace",
        "data/cache/metasyn/finalized-runtime",
        "--pilot-workspace",
        "data/cache/metasyn/pilot",
        "--expected-execution-bundle-sha256",
        _sha("execution-bundle"),
    ]

    with pytest.raises(
        ValueError,
        match=f"metasyn_synthesis_{error_kind}_output_symlink_forbidden",
    ):
        synthesis_cli.main(args)
    assert calls == []
    assert list(outside.iterdir()) == []


def test_cli_creates_no_output_topology_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()

    def stop_during_evaluation(**kwargs: Any) -> tuple[Any, Any]:
        assert kwargs["repository_root"] == repository_root.resolve()
        assert not (repository_root / "data").exists()
        assert not (repository_root / "artifacts").exists()
        raise RuntimeError("synthetic_evaluation_stop")

    monkeypatch.setattr(
        synthesis_cli,
        "evaluate_current_metasyn_synthesis_yield",
        stop_during_evaluation,
    )
    args = [
        "--repository-root",
        str(repository_root),
        "--runtime-workspace",
        "data/cache/metasyn/finalized-runtime",
        "--pilot-workspace",
        "data/cache/metasyn/pilot",
        "--expected-execution-bundle-sha256",
        _sha("execution-bundle"),
    ]

    with pytest.raises(RuntimeError, match="synthetic_evaluation_stop"):
        synthesis_cli.main(args)
    assert not (repository_root / "data").exists()
    assert not (repository_root / "artifacts").exists()


def test_cli_rechecks_symlink_ancestors_after_evaluation_before_write(
    synthetic_inputs: SyntheticInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=REPOSITORY_ROOT,
        runtime_report=synthetic_inputs.runtime,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    public = freeze_metasyn_synthesis_yield_public_summary(report=report)
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    def insert_symlink_during_evaluation(**kwargs: Any) -> tuple[Any, Any]:
        assert not (repository_root / "artifacts").exists()
        (repository_root / "artifacts").symlink_to(
            outside,
            target_is_directory=True,
        )
        return report, public

    monkeypatch.setattr(
        synthesis_cli,
        "evaluate_current_metasyn_synthesis_yield",
        insert_symlink_during_evaluation,
    )
    args = [
        "--repository-root",
        str(repository_root),
        "--runtime-workspace",
        "data/cache/metasyn/finalized-runtime",
        "--pilot-workspace",
        "data/cache/metasyn/pilot",
        "--expected-execution-bundle-sha256",
        _sha("execution-bundle"),
    ]

    with pytest.raises(
        ValueError,
        match="metasyn_synthesis_public_output_symlink_forbidden",
    ):
        synthesis_cli.main(args)
    assert not (repository_root / synthesis_cli.PRIVATE_OUTPUT_RELATIVE).exists()
    assert list(outside.iterdir()) == []


def test_cli_output_pair_rolls_back_own_file_and_preserves_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"
    original_install = synthesis_cli._install_staged_no_overwrite

    def racing_install(*, temporary: Path, destination: Path) -> None:
        if destination == public_output:
            destination.write_text("racing-writer\n", encoding="utf-8")
        original_install(temporary=temporary, destination=destination)

    monkeypatch.setattr(
        synthesis_cli,
        "_install_staged_no_overwrite",
        racing_install,
    )
    with pytest.raises(OutputExistsError, match="lineage_output_exists"):
        synthesis_cli._write_output_pair_no_overwrite(
            private_output=private_output,
            private_value={"private": True},
            public_output=public_output,
            public_value={"public": True},
        )
    assert not private_output.exists()
    assert public_output.read_text(encoding="utf-8") == "racing-writer\n"
    assert not list(tmp_path.glob(".*.tmp"))
