from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.metasyn_bounded_adapter import (
    ADAPTER_BUNDLE_VERSION,
    ADAPTER_VERSION,
    INVENTORY_PROMPT_PATH,
    PACKET_PROMPT_PATH,
    MetaSynBoundedAdapterBundleV1,
    compute_metasyn_bounded_adapter_fingerprint,
    freeze_metasyn_bounded_row_context,
    freeze_metasyn_inventory_validation_receipt,
    freeze_metasyn_packet_call,
    freeze_metasyn_packet_validation_receipt,
)
from literature_multiverse.metasyn_bounded_runtime import (
    MetaSynBoundedExecutionBundleV1,
    MetaSynBoundedRuntimeError,
    _assess_preflight_result,
    _packet_response_classification,
    _preflight_spec_roster,
    _request_surface,
    finalize_metasyn_bounded_yield_runtime,
    freeze_metasyn_attempt_intent,
    freeze_metasyn_bounded_execution_bundle,
    freeze_metasyn_generation_receipt,
    load_metasyn_bounded_runtime_config,
    metasyn_runtime_paths,
    run_metasyn_bounded_prediction_stage,
    run_metasyn_schema_preflight,
    validate_current_metasyn_bounded_public_yield_summary,
    validate_metasyn_bounded_finalized_runtime,
    write_metasyn_bounded_execution_bundle,
)
from literature_multiverse.metasyn_typed_pilot import (
    MetaSynPilotSourceProjectionRowV1,
    _question_spec,
    compute_metasyn_typed_pilot_pipeline_fingerprint,
)
from literature_multiverse.native_bounded_schema_v2 import (
    INVENTORY_PROVIDER_SCHEMA_V2,
    PACKET_PROVIDER_SCHEMA_V2,
    PROVIDER_GRAMMAR_SCOPE_V2,
    schema_v2_contract,
    synthetic_schema_v2_preflight_fingerprint,
)
from literature_multiverse.native_extraction import (
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    ResolvedNativeSource,
    ResolvedSourceLine,
)
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


def _source_row(*, review_id: int, corpus_id: int) -> MetaSynPilotSourceProjectionRowV1:
    question = _question_spec(_protocol_row(review_id))
    text = (
        f"The adjusted difference for outcome {review_id} was 0.5 with standard error 0.2."
    )
    artifact_path = f"data/cache/metasyn/synthetic/{corpus_id}.json"
    source_locator = f"metasyn://corpus/{corpus_id}"
    resolved = ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path=artifact_path,
        artifact_sha256=_sha(f"artifact-{corpus_id}"),
        source_locator=source_locator,
        source_payload_sha256=hash_canonical({"Results": text}),
        source_text=text,
        lines=[
            ResolvedSourceLine(
                line_id="L1",
                line_number=1,
                section="Results",
                text=text,
                char_start=0,
                char_end=len(text),
                utf8_byte_start=0,
                utf8_byte_end=len(text.encode()),
            )
        ],
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
    doc_id = f"metasyn-corpus:{corpus_id}"
    projection = project_resolved_source_for_question(
        row_id=doc_id, source=resolved, spec=projection_spec
    )
    source_record = NativeSourceRecord(
        doc_id=doc_id,
        publication=PublicationIdentity(
            publication_id=f"metasyn-publication:{corpus_id}",
            paper_id=f"metasyn-paper:{corpus_id}",
            doc_id=doc_id,
        ),
        source_document=SourceDocumentArtifact(
            artifact_path=artifact_path,
            sha256=resolved.artifact_sha256,
            media_type="application/x-parquet",
            source_locator=source_locator,
        ),
    )
    payload: dict[str, Any] = {
        "source_row_version": "metasyn-typed-oracle-source-row-v1",
        "question_id": question.question_id,
        "corpus_id": corpus_id,
        "doc_id": doc_id,
        "source_record": source_record,
        "diagnostic_source_record_sha256": _sha(f"diagnostic-{corpus_id}"),
        "source_content_scope": "full_text_sections",
        "oracle_selection_full_text_scope": True,
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


@pytest.fixture(scope="session")
def execution_bundle() -> MetaSynBoundedExecutionBundleV1:
    root = REPOSITORY_ROOT
    inventory_template = (root / INVENTORY_PROMPT_PATH).read_text(encoding="utf-8")
    packet_template = (root / PACKET_PROMPT_PATH).read_text(encoding="utf-8")
    row_contexts = []
    corpus_id = 1000
    for question_index in range(10):
        review_id = 100 + question_index
        question = _question_spec(_protocol_row(review_id))
        paper_count = 4 if question_index < 2 else 3
        for _ in range(paper_count):
            source_row = _source_row(review_id=review_id, corpus_id=corpus_id)
            row_contexts.append(
                freeze_metasyn_bounded_row_context(
                    question_bundle_sha256=_sha(f"question-bundle-{review_id}"),
                    question_spec=question,
                    independence_component_id=f"metasyn-component-{review_id}",
                    independence_component_review_ids=[review_id],
                    independence_component_membership_sha256=hash_canonical(
                        [review_id]
                    ),
                    source_row=source_row,
                    inventory_template=inventory_template,
                    packet_template=packet_template,
                )
            )
            corpus_id += 1
    row_contexts.sort(key=lambda row: row.row_key)
    pilot = compute_metasyn_typed_pilot_pipeline_fingerprint(root=root)
    adapter_pipeline = compute_metasyn_bounded_adapter_fingerprint(
        repository_root=root,
        upstream_pilot_pipeline_sha256=pilot.pipeline_sha256,
    )
    components = {
        row.independence_component_id: row.component_descriptor for row in row_contexts
    }
    question_ids = sorted({row.question_spec.question_id for row in row_contexts})
    row_keys = [row.row_key for row in row_contexts]
    adapter_payload: dict[str, Any] = {
        "adapter_bundle_version": ADAPTER_BUNDLE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "status": "provider_neutral_inputs_frozen_reference_fields_unopened",
        "prepare_bundle_sha256": _sha("synthetic-prepare-bundle"),
        "upstream_pilot_pipeline_sha256": pilot.pipeline_sha256,
        "adapter_pipeline_fingerprint": adapter_pipeline,
        "adapter_pipeline_sha256": adapter_pipeline.pipeline_sha256,
        "official_native_schema_sha256": hash_canonical(
            native_publication_extraction_json_schema()
        ),
        "inventory_prompt_path": INVENTORY_PROMPT_PATH,
        "inventory_prompt_file_sha256": sha256_file(root / INVENTORY_PROMPT_PATH),
        "packet_prompt_path": PACKET_PROMPT_PATH,
        "packet_prompt_file_sha256": sha256_file(root / PACKET_PROMPT_PATH),
        "question_count": 10,
        "component_count": 10,
        "publication_count": 32,
        "question_membership_sha256": hash_canonical(question_ids),
        "component_membership_sha256": hash_canonical(
            [components[key] for key in sorted(components)]
        ),
        "row_membership_sha256": hash_canonical(row_keys),
        "row_contexts": row_contexts,
        "reference_fields_unopened": True,
        "model_calls_made": False,
        "directional_accuracy_authority": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
    }
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(
        {
            **adapter_payload,
            "adapter_bundle_sha256": hash_canonical(adapter_payload),
        }
    )
    config, config_file_sha = load_metasyn_bounded_runtime_config(
        repository_root=root
    )
    return freeze_metasyn_bounded_execution_bundle(
        adapter_bundle=adapter,
        runtime_config=config,
        config_file_sha256=config_file_sha,
        pilot_workspace_relative="data/cache/metasyn/synthetic-runtime-pilot",
        repository_root=root,
    )


@pytest.fixture()
def runtime_workspace(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    import literature_multiverse.metasyn_bounded_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "validate_metasyn_bounded_adapter_bundle_external_replay",
        lambda **kwargs: MetaSynBoundedAdapterBundleV1.model_validate(
            kwargs["adapter_bundle"]
        ),
    )
    parent = (
        REPOSITORY_ROOT
        / "data"
        / "cache"
        / "metasyn"
        / f"bounded-runtime-test-{uuid.uuid4().hex}"
    )
    workspace = parent / "execution"
    write_metasyn_bounded_execution_bundle(
        execution_bundle=execution_bundle,
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
    )
    try:
        yield workspace
    finally:
        shutil.rmtree(parent)


def _find_property(schema: Any, name: str) -> dict[str, Any]:
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping) and isinstance(properties.get(name), Mapping):
            return dict(properties[name])
        for value in schema.values():
            try:
                return _find_property(value, name)
            except KeyError:
                pass
    elif isinstance(schema, list):
        for value in schema:
            try:
                return _find_property(value, name)
            except KeyError:
                pass
    raise KeyError(name)


class FakeOllamaClient:
    def __init__(
        self,
        identity: OllamaIdentity,
        *,
        source_modes: dict[int, str] | None = None,
        inventory_done_reasons: dict[int, str | None] | None = None,
        packet_done_reasons: dict[int, str | None] | None = None,
        synthetic_done_reason: str | None = "stop",
        synthetic_mode: str = "valid",
        fail_transport_once: bool = False,
    ) -> None:
        self.identity = identity
        self.source_modes = source_modes or {}
        self.inventory_done_reasons = inventory_done_reasons or {}
        self.packet_done_reasons = packet_done_reasons or {}
        self.synthetic_done_reason = synthetic_done_reason
        self.synthetic_mode = synthetic_mode
        self.fail_transport_once = fail_transport_once
        self.source_inventory_calls = 0
        self.source_packet_calls = 0
        self.synthetic_calls = 0
        self.output_schema_markers: list[str | None] = []

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        assert config.model == self.identity.model
        return self.identity

    def generate(
        self,
        *,
        prompt: str,
        output_schema: Mapping[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        synthetic = prompt.startswith(
            "Synthetic provider-schema whole-request compatibility"
        )
        packet_schema = "candidate-packet" in str(output_schema.get("$id", ""))
        self.output_schema_markers.append(
            output_schema.get("x-literature-multiverse-generation-schema-version")
        )
        if synthetic:
            self.synthetic_calls += 1
            payload = json.loads(prompt.split("SYNTHETIC_JSON:", maxsplit=1)[1])
            response_text = json.dumps(payload)
            if self.synthetic_mode == "invalid_json":
                response_text = "{"
            elif self.synthetic_mode == "provider_only_inventory":
                assert self.synthetic_calls == 1
                # The provider grammar now preserves the three inventory-state
                # branches, so exercise a remaining intentional widening:
                # provider candidate indices are bounded but contiguity remains
                # authoritative only in the full schema/typed validator.
                payload["inventory_status"] = "candidates_found"
                payload["candidates"] = [
                    {
                        "candidate_index": 2,
                        "outcome_name": "synthetic_outcome",
                        "effect_kind": "direct_standard_error",
                        "line_ids": ["SYNTHETIC_LINE"],
                    }
                ]
                response_text = json.dumps(payload)
            elif (
                self.synthetic_mode == "omit_declared_packet_defaults"
                and self.synthetic_calls > 1
            ):
                omitted_nonnull_defaults = {
                    "registration_ids": [],
                    "registry_ids": [],
                    "dataset_ids": [],
                    "moderators": [],
                    "reported_significance": "not_reported",
                    "equivalence_conclusion": "not_tested",
                    "extraction_method": "reported",
                }

                def omit_declared_defaults(value: Any) -> None:
                    if isinstance(value, dict):
                        for key in list(value):
                            item = value[key]
                            if item is None or (
                                key in omitted_nonnull_defaults
                                and item == omitted_nonnull_defaults[key]
                            ):
                                del value[key]
                            else:
                                omit_declared_defaults(item)
                    elif isinstance(value, list):
                        for item in value:
                            omit_declared_defaults(item)

                omit_declared_defaults(payload)
                response_text = json.dumps(payload)
            return OllamaGenerationResult(
                model=config.model,
                response_text=response_text,
                done=True,
                done_reason=self.synthetic_done_reason,
            )
        if self.fail_transport_once:
            self.fail_transport_once = False
            raise LocalOllamaError("synthetic transport failure")
        if packet_schema:
            self.source_packet_calls += 1
            payload = {
                "packet_version": "native-candidate-packet-v1",
                "packet_status": "unable_to_complete",
                "candidate_index": 1,
                "reason": "capacity_or_other_uncertainty",
            }
            return OllamaGenerationResult(
                model=config.model,
                response_text=json.dumps(payload),
                done=True,
                done_reason=self.packet_done_reasons.get(
                    self.source_packet_calls, "stop"
                ),
            )
        self.source_inventory_calls += 1
        mode = self.source_modes.get(self.source_inventory_calls, "no_candidate")
        if mode == "truncated":
            return OllamaGenerationResult(
                model=config.model,
                response_text="{",
                done=True,
                done_reason="length",
            )
        if mode == "invalid_json":
            return OllamaGenerationResult(
                model=config.model,
                response_text="{not-json",
                done=True,
                done_reason="stop",
            )
        if mode in {"candidate", "provider_only_inventory"}:
            outcome = _find_property(output_schema, "outcome_name")["enum"][0]
            line_id = _find_property(output_schema, "line_ids")["items"]["enum"][0]
            payload = {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": (
                    "no_candidate_found"
                    if mode == "provider_only_inventory"
                    else "candidates_found"
                ),
                "candidates": [
                    {
                        "candidate_index": 1,
                        "outcome_name": outcome,
                        "effect_kind": "direct_standard_error",
                        "line_ids": [line_id],
                    }
                ],
                "has_more_or_uncertain": False,
            }
        else:
            payload = {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            }
        return OllamaGenerationResult(
            model=config.model,
            response_text=json.dumps(payload),
            done=True,
            done_reason=self.inventory_done_reasons.get(
                self.source_inventory_calls, "stop"
            ),
        )


def _client(
    execution_bundle: MetaSynBoundedExecutionBundleV1, **kwargs: Any
) -> FakeOllamaClient:
    return FakeOllamaClient(
        OllamaIdentity.model_validate(
            execution_bundle.runtime_config.expected_model_identity.model_dump(
                mode="json"
            )
        ),
        **kwargs,
    )


def test_config_full_roster_and_structural_preflight_are_exactly_bound(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
) -> None:
    config, _ = load_metasyn_bounded_runtime_config(
        repository_root=REPOSITORY_ROOT
    )
    specs = _preflight_spec_roster(execution_bundle)

    assert config.config_sha256 == (
        "5a59d6a9842db5ec9723e0c1c4a0d55ec52905fee9e302996dcc473aefa13af7"
    )
    assert execution_bundle.question_count == 10
    assert execution_bundle.component_count == 10
    assert execution_bundle.publication_count == 32
    assert len(execution_bundle.adapter_bundle.row_contexts) == 32
    assert len(specs) == 8
    assert sum(spec["kind"] == "inventory" for spec in specs) == 3
    assert {
        spec["inventory_state"]
        for spec in specs
        if spec["kind"] == "inventory"
    } == {"candidates_found", "no_candidate_found", "overflow_or_uncertain"}
    assert all(
        spec["inventory_state"] == spec["valid_example"]["inventory_status"]
        for spec in specs
        if spec["kind"] == "inventory"
    )
    assert {spec["effect_kind"] for spec in specs if spec["kind"] == "packet"} == {
        "binary_group_statistics",
        "continuous_group_statistics",
        "direct_confidence_interval",
        "direct_standard_error",
        "direct_variance",
    }
    assert execution_bundle.native_schema_v2_contract == schema_v2_contract()
    assert execution_bundle.provider_grammar_scope == PROVIDER_GRAMMAR_SCOPE_V2
    assert (
        execution_bundle.schema_v2_preflight_fingerprint
        == synthetic_schema_v2_preflight_fingerprint()
    )
    assert all(spec["schema"] == spec["provider_schema"] for spec in specs)
    assert all(
        spec["provider_schema_sha256"]
        != spec["full_acceptance_schema_sha256"]
        for spec in specs
    )


def test_fake_full_roster_no_candidate_run_finalizes_yield_only(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(execution_bundle)
    preflight = run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )
    report, summary = finalize_metasyn_bounded_yield_runtime(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )
    replayed_report, replayed_summary = validate_metasyn_bounded_finalized_runtime(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )

    assert preflight["structural_skeleton_count"] == 8
    assert preflight["status"] == (
        "passed_eight_call_three_inventory_state_provider_schema_canonical_compatibility_preflight"
    )
    assert preflight["canonical_fixture_equal_call_count"] == 8
    assert preflight["raw_fixture_equal_call_count"] == 8
    assert preflight["declared_default_omission_call_count"] == 0
    assert preflight["omitted_declared_default_path_count"] == 0
    assert preflight["all_canonical_fixtures_equal"] is True
    assert (
        preflight["all_raw_differences_are_declared_pydantic_default_omissions"]
        is True
    )
    assert preflight["whole_request_compatibility_only"] is True
    assert preflight["provider_keyword_enforcement_validated"] is False
    assert preflight["production_context_schema_compilation_validated"] is False
    assert preflight["production_enum_or_cardinality_compilation_validated"] is False
    assert (
        preflight[
            "declared_pydantic_default_omission_equivalent_after_validation"
        ]
        is True
    )
    assert preflight["nondefault_omission_equivalent_after_validation"] is False
    assert preflight["structural_skeleton_coverage"][
        "whole_request_compatibility_only"
    ] is True
    assert preflight["structural_skeleton_coverage"][
        "provider_keyword_enforcement_validated"
    ] is False
    assert preflight["structural_skeleton_coverage"][
        "production_enum_or_cardinality_compilation_validated"
    ] is False
    assert "all_production_structures_covered" not in json.dumps(preflight)
    assert preflight["structural_skeleton_coverage"]["inventory_state_fixture_count"] == 3
    assert preflight["structural_skeleton_coverage"]["inventory_state_fixtures"] == [
        "no_candidate_found",
        "candidates_found",
        "overflow_or_uncertain",
    ]
    assert client.synthetic_calls == 8
    assert client.source_inventory_calls == 32
    assert client.source_packet_calls == 0
    assert set(client.output_schema_markers) == {
        INVENTORY_PROVIDER_SCHEMA_V2,
        PACKET_PROVIDER_SCHEMA_V2,
    }
    assert ledger.all_rows_terminal is True
    assert ledger.observed_source_generation_calls == 32
    assert report.row_status_counts == {"adapter_inventory_no_candidate": 32}
    assert report.typed_publication_output_count == 0
    assert summary.row_status_counts == report.row_status_counts
    assert summary.direction_agreement_reported is False
    assert summary.extraction_accuracy_reported is False
    assert summary.claim_release_authority is False
    assert replayed_report == report
    assert replayed_summary == summary
    assert (
        validate_current_metasyn_bounded_public_yield_summary(
            summary=summary,
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            expected_execution_bundle_sha256=(
                execution_bundle.execution_bundle_sha256
            ),
        )
        == summary
    )
    public_keys: set[str] = set()

    def collect_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                public_keys.add(str(key))
                collect_keys(item)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item)

    collect_keys(summary.model_dump(mode="json"))
    assert not public_keys.intersection(
        {
            "candidate",
            "doc_id",
            "evidence",
            "official_output",
            "packet_payload",
            "passages",
            "prompt",
            "question_spec",
            "quote",
            "response_text",
            "row_contexts",
            "source_locator",
        }
    )
    assert not (runtime_workspace / "public-summary.json").exists()


def test_preflight_accepts_only_proven_declared_default_omissions_and_replays(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(
        execution_bundle, synthetic_mode="omit_declared_packet_defaults"
    )
    preflight = run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )

    assert preflight["raw_fixture_equal_call_count"] == 3
    assert preflight["canonical_fixture_equal_call_count"] == 8
    assert preflight["declared_default_omission_call_count"] == 5
    assert preflight["omitted_declared_default_path_count"] == 129
    assert client.synthetic_calls == 8

    binary = json.loads(
        (
            metasyn_runtime_paths(runtime_workspace)["preflight_dir"]
            / "call-receipts"
            / "03-packet-binary_group_statistics-v2.json"
        ).read_text(encoding="utf-8")
    )
    assert binary["status"] == "passed"
    assert binary["raw_fixture_equal"] is False
    assert binary["canonical_fixture_equal"] is True
    assert binary["omitted_declared_default_path_count"] == 25
    assert binary["omitted_declared_default_paths"] == [
        "cohort.dataset_ids",
        "cohort.population_description",
        "cohort.recruitment_period",
        "cohort.registry_ids",
        "cohort.total_sample_size",
        "comparator_arm.description",
        "comparator_arm.sample_size",
        "contrast.estimand",
        "effect.equivalence_conclusion",
        "effect.equivalence_margin",
        "effect.extraction_method",
        "effect.moderators",
        "effect.reported_p_value",
        "effect.reported_significance",
        "finding.analysis_population",
        "finding.timepoint.anchor",
        "finding.timepoint.lower",
        "finding.timepoint.raw_label",
        "finding.timepoint.unit",
        "finding.timepoint.upper",
        "finding.timepoint.value",
        "study.design",
        "study.registration_ids",
        "treatment_arm.description",
        "treatment_arm.sample_size",
    ]
    assert binary["omitted_declared_default_paths_sha256"] == hash_canonical(
        binary["omitted_declared_default_paths"]
    )

    replayed = run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    assert replayed == preflight
    assert client.synthetic_calls == 8


def test_preflight_canonical_gate_rejects_every_nondefault_raw_difference(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
) -> None:
    specs = _preflight_spec_roster(execution_bundle)

    def assess(spec_index: int, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        spec = specs[spec_index]
        return _assess_preflight_result(
            spec=spec,
            result=OllamaGenerationResult(
                model=spec["config"].model,
                response_text=json.dumps(payload),
                done=True,
                done_reason="stop",
            ),
        )

    missing_required = deepcopy(specs[3]["valid_example"])
    del missing_required["candidate_index"]
    assert assess(3, missing_required)[0] == (
        "full_acceptance_or_typed_validation_invalid"
    )

    omitted_nondefault = deepcopy(specs[3]["valid_example"])
    del omitted_nondefault["study"]["source_label"]
    assert assess(3, omitted_nondefault)[0] == (
        "full_acceptance_or_typed_validation_invalid"
    )

    extra_field = deepcopy(specs[3]["valid_example"])
    extra_field["unexpected"] = "not permitted"
    assert assess(3, extra_field)[0] == (
        "full_acceptance_or_typed_validation_invalid"
    )

    changed_science = deepcopy(specs[3]["valid_example"])
    changed_science["study"]["source_label"] = "Different study"
    status, comparison = assess(3, changed_science)
    assert status == "canonical_semantic_fixture_mismatch"
    assert comparison["canonical_fixture_equal"] is False

    changed_default = deepcopy(specs[3]["valid_example"])
    changed_default["effect"]["reported_significance"] = "significant"
    assert assess(3, changed_default)[0] == "canonical_semantic_fixture_mismatch"

    invalid_lexeme = deepcopy(specs[3]["valid_example"])
    invalid_lexeme["effect"]["treatment_total"] = "022"
    assert assess(3, invalid_lexeme)[0] == (
        "full_acceptance_or_typed_validation_invalid"
    )

    normalized_whitespace = deepcopy(specs[3]["valid_example"])
    normalized_whitespace["evidence"]["quote"] = (
        f" {normalized_whitespace['evidence']['quote']} "
    )
    assert assess(3, normalized_whitespace)[0] == (
        "full_acceptance_or_typed_validation_invalid"
    )

    alternate_lexeme = deepcopy(specs[6]["valid_example"])
    alternate_lexeme["effect"]["standard_error"] = "0.20"
    alternate_lexeme["evidence"]["quote"] = alternate_lexeme["evidence"][
        "quote"
    ].replace("standard_error=0.2", "standard_error=0.20")
    support = next(
        item
        for item in alternate_lexeme["numeric_support"]
        if item["field_path"] == "effect.standard_error"
    )
    support["verbatim_token"] = "0.20"
    support["quote_end"] = str(int(support["quote_end"]) + 1)
    status, comparison = assess(6, alternate_lexeme)
    assert status == "canonical_semantic_fixture_mismatch"
    assert comparison["canonical_fixture_equal"] is False

    branch_mismatch_spec = deepcopy(specs[1])
    branch_mismatch_spec["inventory_state"] = "no_candidate_found"
    status, _ = _assess_preflight_result(
        spec=branch_mismatch_spec,
        result=OllamaGenerationResult(
            model=branch_mismatch_spec["config"].model,
            response_text=json.dumps(branch_mismatch_spec["valid_example"]),
            done=True,
            done_reason="stop",
        ),
    )
    assert status == "full_acceptance_or_typed_validation_invalid"


def test_runtime_intent_and_receipt_bind_provider_full_bundle_and_context_hashes(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
) -> None:
    row = execution_bundle.adapter_bundle.row_contexts[0]
    identity = OllamaIdentity.model_validate(
        execution_bundle.runtime_config.expected_model_identity.model_dump(mode="json")
    )
    prompt, schema_bundle, binding, packet_call = _request_surface(
        row=row,
        stage="inventory",
    )
    assert packet_call is None
    assert prompt == row.inventory_prompt
    assert schema_bundle["provider_schema"][
        "x-literature-multiverse-generation-schema-version"
    ] == INVENTORY_PROVIDER_SCHEMA_V2
    assert schema_bundle["full_acceptance_schema"][
        "x-literature-multiverse-generation-schema-version"
    ] == "native-candidate-inventory-generation-schema-v2"

    intent = freeze_metasyn_attempt_intent(
        execution_bundle=execution_bundle,
        row=row,
        stage="inventory",
        identity=identity,
    )
    assert intent.schema_sha256 == binding["provider_schema_sha256"]
    assert intent.provider_schema_sha256 == binding["provider_schema_sha256"]
    assert intent.full_acceptance_schema_sha256 == binding[
        "full_acceptance_schema_sha256"
    ]
    assert intent.schema_bundle_sha256 == binding["schema_bundle_sha256"]
    assert intent.schema_context_binding_sha256 == binding[
        "context_binding_sha256"
    ]

    result = OllamaGenerationResult(
        model=execution_bundle.runtime_config.expected_model_identity.model,
        response_text=json.dumps(
            {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            }
        ),
        done=True,
        done_reason="stop",
    )
    receipt = freeze_metasyn_generation_receipt(
        execution_bundle=execution_bundle,
        row=row,
        intent=intent,
        identity=identity,
        generation_result=result,
    )
    assert receipt.schema_sha256 == intent.provider_schema_sha256
    assert receipt.provider_schema_sha256 == intent.provider_schema_sha256
    assert receipt.full_acceptance_schema_sha256 == (
        intent.full_acceptance_schema_sha256
    )
    assert receipt.schema_bundle_sha256 == intent.schema_bundle_sha256
    assert receipt.schema_context_binding_sha256 == (
        intent.schema_context_binding_sha256
    )


def test_provider_widened_inventory_is_rejected_by_full_v2_authority(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(
        execution_bundle,
        source_modes={1: "provider_only_inventory"},
    )
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
        inventory_limit=1,
    )

    assert client.source_inventory_calls == 1
    assert ledger.inventory_response_status_counts == {
        "inventory_contract_invalid": 1
    }
    attempted = [
        row for row in ledger.rows if row.inventory.response_status is not None
    ]
    assert len(attempted) == 1
    assert attempted[0].status == "runtime_inventory_blocked"


def test_runtime_rejects_adapter_acceptable_whitespace_normalization(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
) -> None:
    row = execution_bundle.adapter_bundle.row_contexts[0]
    outcome = row.allowed_outcomes[0]
    passage = row.source_row.projection.passages[0]
    candidate_payload = {
        "candidate_index": 1,
        "outcome_name": outcome,
        "effect_kind": "direct_standard_error",
        "line_ids": [passage.line_id],
    }
    inventory_receipt = freeze_metasyn_inventory_validation_receipt(
        row=row,
        value={
            "inventory_version": "native-candidate-inventory-v1",
            "inventory_status": "candidates_found",
            "candidates": [candidate_payload],
            "has_more_or_uncertain": False,
        },
    )
    call = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=1,
    )
    quote = passage.text
    estimate_start = quote.index("0.5")
    standard_error_start = quote.index("0.2")
    packet = {
        "packet_version": "native-candidate-packet-v1",
        "packet_status": "completed",
        "candidate_index": 1,
        "study": {
            "key": "study-1",
            "source_label": "Synthetic study",
            "design": None,
            "registration_ids": [],
        },
        "cohort": {
            "key": "cohort-1",
            "source_labels": ["Synthetic cohort"],
            "registry_ids": [],
            "dataset_ids": [],
            "population_description": None,
            "recruitment_period": None,
            "total_sample_size": None,
        },
        "treatment_arm": {
            "key": "treatment",
            "label": "Treatment",
            "role": "intervention",
            "description": None,
            "sample_size": None,
        },
        "comparator_arm": {
            "key": "control",
            "label": "Control",
            "role": "control",
            "description": None,
            "sample_size": None,
        },
        "contrast": {
            "key": "contrast-1",
            "label": "Treatment versus control",
            "estimand": None,
            "positive_direction_means": row.outcome_positive_directions[outcome],
        },
        "finding": {
            "key": "finding-1",
            "outcome_name": outcome,
            "timepoint": {
                "kind": "not_reported",
                "value": None,
                "lower": None,
                "upper": None,
                "unit": None,
                "anchor": None,
                "raw_label": None,
            },
            "analysis_population": None,
        },
        "effect": {
            "reported_p_value": None,
            "reported_significance": "not_reported",
            "equivalence_conclusion": "not_tested",
            "equivalence_margin": None,
            "moderators": [],
            "extraction_method": "reported",
            "effect_kind": "direct_standard_error",
            "effect_format": "mean_difference",
            "estimate": "0.5",
            "standard_error": "0.2",
            "unit": None,
        },
        "evidence": {
            "source_locator": row.source_locator,
            # Pydantic strips this, so v1 accepts and grounds the canonical quote.
            "quote": f" {quote} ",
            "section": passage.exposed_section,
            "line_ids": [passage.line_id],
        },
        "numeric_support": [
            {
                "field_path": "effect.estimate",
                "verbatim_token": "0.5",
                "normalization": "identity",
                "quote_start": str(estimate_start),
                "quote_end": str(estimate_start + 3),
            },
            {
                "field_path": "effect.standard_error",
                "verbatim_token": "0.2",
                "normalization": "identity",
                "quote_start": str(standard_error_start),
                "quote_end": str(standard_error_start + 3),
            },
        ],
    }
    legacy_receipt = freeze_metasyn_packet_validation_receipt(
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
        value=packet,
    )
    assert legacy_receipt.packet_status == "completed"

    status, receipt, error = _packet_response_classification(
        row=row,
        inventory_receipt=inventory_receipt,
        packet_call=call,
        result=OllamaGenerationResult(
            model=execution_bundle.runtime_config.expected_model_identity.model,
            response_text=json.dumps(packet),
            done=True,
            done_reason="stop",
        ),
    )
    assert status == "packet_contract_invalid"
    assert receipt is None
    assert error == "packet_contract_invalid"


def test_ambiguous_attempt_is_poisoned_once_and_other_rows_resume(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    first_client = _client(execution_bundle, fail_transport_once=True)
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=first_client,
    )
    partial = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=first_client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )

    second_client = _client(execution_bundle)
    complete = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=second_client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )
    report, _ = finalize_metasyn_bounded_yield_runtime(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )

    assert partial.possible_ambiguous_source_generation_calls == 1
    assert partial.observed_source_generation_calls == 0
    assert partial.all_rows_terminal is False
    assert complete.all_rows_terminal is True
    assert complete.possible_ambiguous_source_generation_calls == 1
    assert complete.observed_source_generation_calls == 31
    assert complete.total_possible_source_generation_call_attempts == 32
    assert second_client.source_inventory_calls == 31
    assert report.row_status_counts == {
        "adapter_inventory_no_candidate": 31,
        "runtime_inventory_blocked": 1,
    }


def test_truncated_invalid_and_unable_are_terminal_without_fabricated_inventory(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(
        execution_bundle,
        source_modes={1: "truncated", 2: "invalid_json", 3: "candidate"},
    )
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )
    report, summary = finalize_metasyn_bounded_yield_runtime(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )

    assert ledger.all_rows_terminal is True
    assert ledger.inventory_response_status_counts == {
        "generation_truncated": 1,
        "inventory_valid_candidates": 1,
        "inventory_valid_no_candidate_non_authorizing": 29,
        "response_json_invalid": 1,
    }
    assert ledger.packet_response_status_counts == {
        "packet_unable_to_complete": 1
    }
    assert report.row_status_counts == {
        "adapter_inventory_no_candidate": 29,
        "adapter_packet_unable": 1,
        "runtime_inventory_blocked": 2,
    }
    assert report.typed_publication_output_count == 0
    assert summary.row_status_counts == report.row_status_counts
    inventory_blocked = [
        row for row in report.row_results if row.status == "runtime_inventory_blocked"
    ]
    assert len(inventory_blocked) == 2
    assert all(row.adapter_publication_result is None for row in inventory_blocked)


@pytest.mark.parametrize(
    ("done_reason", "expected_status"),
    [
        (None, "generation_terminal_reason_invalid"),
        ("unknown", "generation_terminal_reason_invalid"),
        ("load", "generation_terminal_reason_invalid"),
        ("length", "generation_truncated"),
    ],
)
def test_inventory_requires_exact_stop_terminal_reason(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
    done_reason: str | None,
    expected_status: str,
) -> None:
    client = _client(
        execution_bundle,
        inventory_done_reasons={1: done_reason},
    )
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
        inventory_limit=1,
    )

    attempted = [
        row for row in ledger.rows if row.inventory.response_status is not None
    ]
    assert client.source_inventory_calls == 1
    assert ledger.inventory_response_status_counts == {expected_status: 1}
    assert len(attempted) == 1
    assert attempted[0].status == "runtime_inventory_blocked"
    assert attempted[0].terminal is True
    assert attempted[0].inventory.response_status == expected_status
    assert attempted[0].packets == []


@pytest.mark.parametrize(
    ("done_reason", "expected_status"),
    [
        (None, "generation_terminal_reason_invalid"),
        ("unknown", "generation_terminal_reason_invalid"),
        ("load", "generation_terminal_reason_invalid"),
        ("length", "generation_truncated"),
    ],
)
def test_packet_requires_exact_stop_terminal_reason(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
    done_reason: str | None,
    expected_status: str,
) -> None:
    client = _client(
        execution_bundle,
        source_modes={1: "candidate"},
        packet_done_reasons={1: done_reason},
    )
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
        inventory_limit=1,
        packet_limit=1,
    )

    attempted = [row for row in ledger.rows if row.packets]
    assert client.source_inventory_calls == 1
    assert client.source_packet_calls == 1
    assert ledger.inventory_response_status_counts == {
        "inventory_valid_candidates": 1
    }
    assert ledger.packet_response_status_counts == {expected_status: 1}
    assert len(attempted) == 1
    assert attempted[0].status == "runtime_packet_blocked"
    assert attempted[0].terminal is True
    assert len(attempted[0].packets) == 1
    assert attempted[0].packets[0].outcome.response_status == expected_status


@pytest.mark.parametrize(
    ("done_reason", "synthetic_mode", "expected_status"),
    [
        (None, "valid", "generation_terminal_reason_invalid"),
        ("unknown", "valid", "generation_terminal_reason_invalid"),
        ("length", "valid", "generation_truncated"),
        ("stop", "invalid_json", "response_json_invalid"),
        (
            "stop",
            "provider_only_inventory",
            "full_acceptance_or_typed_validation_invalid",
        ),
    ],
)
def test_preflight_observed_invalid_response_is_terminal_and_replayable(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
    done_reason: str | None,
    synthetic_mode: str,
    expected_status: str,
) -> None:
    client = _client(
        execution_bundle,
        synthetic_done_reason=done_reason,
        synthetic_mode=synthetic_mode,
    )

    with pytest.raises(
        MetaSynBoundedRuntimeError,
        match=(
            "metasyn_runtime_preflight_terminal_response_failed:"
            f"{expected_status}"
        ),
    ):
        run_metasyn_schema_preflight(
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
        )

    assert client.synthetic_calls == 1
    assert client.source_inventory_calls == 0
    assert client.source_packet_calls == 0
    spec = _preflight_spec_roster(execution_bundle)[0]
    call_paths = {
        key: path
        for key, path in metasyn_runtime_paths(runtime_workspace).items()
        if key in {"preflight_dir"}
    }
    receipt_path = (
        call_paths["preflight_dir"]
        / "call-receipts"
        / f"{spec['call_id']}.json"
    )
    incident_path = (
        call_paths["preflight_dir"]
        / "ambiguity-incidents"
        / f"{spec['call_id']}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == expected_status
    assert receipt["terminal"] is True
    assert receipt["response_observed"] is True
    assert receipt["terminal_error"] == expected_status
    assert receipt["generation_call_attempts"] == 1
    assert not incident_path.exists()

    if synthetic_mode == "provider_only_inventory":
        widened = json.loads(receipt["generation_result"]["response_text"])
        Draft202012Validator(spec["provider_schema"]).validate(widened)

    with pytest.raises(
        MetaSynBoundedRuntimeError,
        match=(
            "metasyn_runtime_preflight_terminal_response_failed:"
            f"{expected_status}"
        ),
    ):
        run_metasyn_schema_preflight(
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
        )

    assert client.synthetic_calls == 1


def test_prediction_requires_preflight_and_exact_bundle_anchor(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(execution_bundle)
    with pytest.raises(MetaSynBoundedRuntimeError):
        run_metasyn_bounded_prediction_stage(
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
            expected_execution_bundle_sha256="0" * 64,
        )
    with pytest.raises(MetaSynBoundedRuntimeError):
        run_metasyn_bounded_prediction_stage(
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
            expected_execution_bundle_sha256=(
                execution_bundle.execution_bundle_sha256
            ),
        )


def test_orphan_source_intent_becomes_terminal_incident_without_retry(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(execution_bundle)
    run_metasyn_schema_preflight(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )
    row = execution_bundle.adapter_bundle.row_contexts[0]
    identity = OllamaIdentity.model_validate(
        execution_bundle.runtime_config.expected_model_identity.model_dump(mode="json")
    )
    intent = freeze_metasyn_attempt_intent(
        execution_bundle=execution_bundle,
        row=row,
        stage="inventory",
        identity=identity,
    )
    intent_path = (
        metasyn_runtime_paths(runtime_workspace)["attempt_intents"]
        / f"{intent.attempt_id}.json"
    )
    atomic_write_json(intent_path, intent.model_dump(mode="json"), force=False)

    ledger = run_metasyn_bounded_prediction_stage(
        workspace=runtime_workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=(
            execution_bundle.execution_bundle_sha256
        ),
    )

    assert ledger.all_rows_terminal is True
    assert ledger.observed_source_generation_calls == 31
    assert ledger.possible_ambiguous_source_generation_calls == 1
    assert ledger.ambiguity_incident_kind_counts == {
        "orphan_intent_observed_on_resume": 1
    }
    assert client.source_inventory_calls == 31


def test_workspace_mixing_is_rejected_before_any_preflight_call(
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    runtime_workspace: Path,
) -> None:
    client = _client(execution_bundle)
    (runtime_workspace / "foreign-artifact.txt").write_text("mixed", encoding="utf-8")

    with pytest.raises(MetaSynBoundedRuntimeError, match="mixing_or_extra"):
        run_metasyn_schema_preflight(
            workspace=runtime_workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
        )

    assert client.synthetic_calls == 0
    assert client.source_inventory_calls == 0


def test_cli_has_no_public_output_or_label_access_surface(capsys: Any) -> None:
    from scripts.run_metasyn_bounded_runtime import main

    with pytest.raises(SystemExit) as preflight_exit:
        main(["preflight", "--help"])
    preflight_output = capsys.readouterr().out.casefold()
    normalized_preflight_output = " ".join(preflight_output.split())

    assert preflight_exit.value.code == 0
    assert "compact provider" in normalized_preflight_output
    assert "full v2 acceptance stack" in normalized_preflight_output
    assert "whole-request compatibility only" in normalized_preflight_output
    assert "not provider keyword enforcement" in normalized_preflight_output

    with pytest.raises(SystemExit) as raised:
        main(["finalize", "--help"])
    output = capsys.readouterr().out.casefold()

    assert raised.value.code == 0
    assert "public-summary-output" not in output
    assert "review conclusion" not in output
    assert "test label" not in output
    assert "do not write/register" in output
