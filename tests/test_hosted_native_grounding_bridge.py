from __future__ import annotations

import hashlib
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from literature_multiverse.config import load_config_for_question
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.hosted_native_extraction_contract import (
    REQUIRED_PIPELINE_PATHS,
    HostedNativeExtractionRunV1,
    freeze_hosted_native_call_authorization_v1,
    freeze_hosted_native_call_intent_v1,
    freeze_hosted_native_call_v1,
    freeze_hosted_native_completed_terminal_v1,
    freeze_hosted_native_extraction_run_v1,
    freeze_hosted_native_failed_terminal_v1,
    freeze_hosted_native_prompt_artifact_v1,
    freeze_hosted_native_provider_identity_v1,
    freeze_hosted_native_schema_artifact_v1,
)
from literature_multiverse.hosted_native_grounding_bridge import (
    HostedNativeGroundingBridgeError,
    build_hosted_native_grounding_package_v1,
    validate_hosted_native_extraction_run_v1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    reverify_typed_evidence_grounding_package,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    compute_pipeline_fingerprint,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact
from literature_multiverse.verifier import (
    ClaimManifest,
    LegacyAdapterConfig,
    load_corpus,
    run_verification,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _copy_pipeline_surface(root: Path) -> None:
    for relative in sorted(REQUIRED_PIPELINE_PATHS | {"prompts/native_extraction.md"}):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, destination)


def _source(root: Path) -> tuple[NativeSourceManifest, str]:
    path = root / "archive" / "source_lines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    quote = "The mean difference in peak power was 2.5 points (SE 0.5)."
    path.write_text(
        json.dumps(
            {
                "PMC/1": {
                    "L1": {"section": "Methods", "text": "A randomized trial."},
                    "L20": {"section": "Results", "text": quote},
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_document = SourceDocumentArtifact(
        artifact_path="archive/source_lines.json",
        sha256=sha256_file(path),
        media_type="application/json",
        source_locator="json:archive/source_lines.json#/PMC~11",
    )
    manifest = NativeSourceManifest(
        question_id="fixture-a",
        records=[
            NativeSourceRecord(
                doc_id="PMC/1",
                publication=PublicationIdentity(
                    publication_id="publication-hosted-1",
                    paper_id="paper-hosted-1",
                    doc_id="PMC/1",
                    title="Hosted bridge fixture",
                ),
                source_document=source_document,
            )
        ],
    )
    return manifest, quote


def _extraction(locator: str, quote: str) -> NativePublicationExtraction:
    return NativePublicationExtraction.model_validate(
        {
            "extraction_schema_version": "native-publication-extraction-v1",
            "status": "estimable",
            "studies": [
                {
                    "key": "trial",
                    "source_label": "Hosted bridge fixture",
                    "design": "randomized trial",
                    "registration_ids": ["NCT00000001"],
                    "cohorts": [
                        {
                            "key": "cohort",
                            "source_labels": ["reported cohort"],
                            "registry_ids": ["NCT00000001"],
                            "dataset_ids": [],
                            "population_description": None,
                            "recruitment_period": None,
                            "total_sample_size": 20,
                            "arms": [
                                {
                                    "key": "treatment",
                                    "label": "Treatment",
                                    "role": "intervention",
                                    "description": None,
                                    "sample_size": 10,
                                },
                                {
                                    "key": "control",
                                    "label": "Control",
                                    "role": "control",
                                    "description": None,
                                    "sample_size": 10,
                                },
                            ],
                            "contrasts": [
                                {
                                    "key": "primary",
                                    "treatment_arm_key": "treatment",
                                    "comparator_arm_key": "control",
                                    "label": "treatment_vs_control",
                                    "estimand": "mean difference",
                                    "positive_direction_means": "higher peak power",
                                }
                            ],
                            "findings": [
                                {
                                    "key": "result",
                                    "contrast_key": "primary",
                                    "outcome_name": "peak_power",
                                    "timepoint": {
                                        "kind": "exact",
                                        "value": 4,
                                        "unit": "week",
                                    },
                                    "analysis_population": "all randomized participants",
                                    "effect": {
                                        "effect_format": "mean_difference",
                                        "availability": "available",
                                        "estimate": 2.5,
                                        "standard_error": 0.5,
                                        "unit": "points",
                                        "reported_significance": "not_reported",
                                        "equivalence_conclusion": "not_tested",
                                        "moderators": [],
                                        "extraction_method": "reported",
                                    },
                                    "evidence": {
                                        "source_locator": locator,
                                        "quote": quote,
                                        "section": "Results",
                                        "page": None,
                                        "char_start": None,
                                        "char_end": None,
                                        "line_ids": ["L20"],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            "non_estimability_reason": None,
            "non_estimability_detail": None,
            "warnings": [],
        }
    )


def _run(
    root: Path,
    *,
    failed: bool = False,
    pipeline_component_version: str = "13",
) -> HostedNativeExtractionRunV1:
    _copy_pipeline_surface(root)
    manifest, quote = _source(root)
    question = load_config_for_question("fixture-a", require_locked=True)
    pipeline = compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="native-extraction",
                component_version=pipeline_component_version,
                file_paths=sorted(REQUIRED_PIPELINE_PATHS),
                settings={
                    "fixture_contract_only": True,
                    "hosted_native_execution_mode": "hosted_exact_once",
                    "hosted_native_extraction_run_contract": ("hosted-native-extraction-run-v1"),
                    "native_extraction_entry_points": [
                        "scripts/build_hosted_native_grounding_package.py"
                    ],
                },
            )
        ],
    )
    prompt_path = root / "prompts/native_extraction.md"
    prompt = freeze_hosted_native_prompt_artifact_v1(
        prompt_id="hosted-native-default",
        prompt_version="1",
        template_path="prompts/native_extraction.md",
        template_sha256=sha256_file(prompt_path),
        rendered_prompt="Extract only exact native numerical evidence.",
    )
    official_payload = native_publication_extraction_json_schema()
    official = freeze_hosted_native_schema_artifact_v1(
        schema_id="native-official-postvalidation",
        role="official_postvalidation",
        schema_payload=official_payload,
    )
    generation = freeze_hosted_native_schema_artifact_v1(
        schema_id="hosted-generation-pmc-1",
        role="generation_constraint",
        schema_payload=official_payload,
    )
    provider = freeze_hosted_native_provider_identity_v1(
        provider_id="fixture-hosted-provider",
        model_id="fixture-hosted-model",
        model_revision="fixture-revision",
        api_base_url="https://provider.invalid",
        runtime_id="fixture-hosted-runtime",
        runtime_version="1",
        runtime_source_paths=["src/literature_multiverse/hosted_native_extraction_contract.py"],
        sdk_name="fixture-sdk",
        sdk_version="1",
        runtime_metadata={"effort": "high"},
    )
    wire_request = json.dumps(
        {
            "model": provider.model_id,
            "prompt": prompt.rendered_prompt,
            "schema": generation.schema_payload,
            "source_document_sha256": manifest.records[0].source_document.sha256,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    intent = freeze_hosted_native_call_intent_v1(
        run_id="hosted-run-1",
        request_key="pmc-1",
        question_config=question,
        source_manifest=manifest,
        source_record_index=0,
        pipeline_fingerprint=pipeline,
        corpus_cutoff="fixture-full-text-cutoff-v1",
        provider_identity=provider,
        prompt=prompt,
        generation_schema=generation,
        official_schema=official,
        wire_request_utf8=wire_request,
    )
    authorization = freeze_hosted_native_call_authorization_v1(
        intent=intent,
        provider_identity=provider,
    )
    if failed:
        terminal = freeze_hosted_native_failed_terminal_v1(
            intent=intent,
            authorization=authorization,
            outcome="provider_failed",
            failure_code="provider_unavailable",
        )
    else:
        extraction = _extraction(manifest.records[0].source_document.source_locator, quote)
        raw_response = json.dumps(
            extraction.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        terminal = freeze_hosted_native_completed_terminal_v1(
            intent=intent,
            authorization=authorization,
            observed_model_id=provider.model_id,
            raw_response_utf8=raw_response,
            structured_output_json_pointer="",
            parsed_extraction=extraction,
            provider_request_id="request-fixture-1",
        )
    call = freeze_hosted_native_call_v1(
        intent=intent,
        authorization=authorization,
        terminal=terminal,
    )
    return freeze_hosted_native_extraction_run_v1(
        run_id="hosted-run-1",
        question_config=question,
        source_manifest=manifest,
        corpus_cutoff="fixture-full-text-cutoff-v1",
        pipeline_fingerprint=pipeline,
        prompts=[prompt],
        schemas=[generation, official],
        provider_identity=provider,
        calls=[call],
    )


def test_completed_hosted_run_builds_and_replays_standard_v4_package(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    result = build_hosted_native_grounding_package_v1(
        run=run,
        repository_root=tmp_path,
    )

    assert result.package.package_version == "typed-evidence-grounding-package-v4"
    assert result.corpus.estimable_publication_ids == ["publication-hosted-1"]
    assert result.receipt.v4_source_provenance_input_eligible is True
    assert result.receipt.claim_release_authority is False
    replay = reverify_typed_evidence_grounding_package(
        package=result.package,
        repository_root=tmp_path,
    )
    assert replay.extraction_context_sha256 == result.extraction_context.context_sha256

    package_path = tmp_path / "typed_evidence_grounding_package.json"
    atomic_write_json(package_path, result.package)
    loaded = load_corpus(
        package_path,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )
    assert loaded.source_format == "typed_evidence_grounding_package_json"
    assert loaded.provenance_assurance.status == "source_replayed_native_grounding"
    assert loaded.metadata["grounding_package_version"] == ("typed-evidence-grounding-package-v4")
    assert loaded.extraction_context is not None
    assert loaded.extraction_context.extraction_mode == "hosted_exact_once"
    claim = ClaimManifest.model_validate(
        {
            "question_id": "fixture-a",
            "population_id": "fixture-population",
            "domain": "fixture-domain",
            "claim": {
                "statement": "The synthetic intervention increases peak power.",
                "direction": "increase",
                "outcome_name": "peak_power",
                "estimand": "mean difference",
            },
            "protocol": {
                "corpus_cutoff": "fixture-full-text-cutoff-v1",
                "inclusion_criteria": [
                    "controlled primary research with a declared synthetic intervention"
                ],
                "exclusion_criteria": ["reviews, protocols, and reports without a comparator"],
            },
        }
    )
    certificate = run_verification(
        manifest=claim,
        corpus=loaded,
        budget_minutes=30,
        pipeline_root=REPOSITORY_ROOT,
    )
    assert certificate.status == "abstained"
    assert certificate.corpus["provenance_assurance"]["release_eligible"] is True


def test_terminal_provider_failure_is_complete_but_non_estimable(
    tmp_path: Path,
) -> None:
    result = build_hosted_native_grounding_package_v1(
        run=_run(tmp_path, failed=True),
        repository_root=tmp_path,
    )

    assert result.package.package_version == "typed-evidence-grounding-package-v4"
    assert result.corpus.estimable_publication_ids == []
    assert result.corpus.non_estimable_publication_ids == ["publication-hosted-1"]
    assert result.receipt.failed_or_ambiguous_count == 1
    assert result.receipt.claim_release_authority is False
    assert result.fragments[0].non_estimability_detail == (
        "hosted_exact_once_terminal:provider_failed:provider_unavailable"
    )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (
            ("diagnostic_or_fixture",),
            True,
            "literal_error",
        ),
        (
            ("provider_identity", "model_id"),
            "different-model",
            "hosted_native_provider_identity_hash_mismatch",
        ),
        (
            ("calls", 0, "intent", "generation_schema_sha256"),
            "0" * 64,
            "hosted_native_intent_hash_mismatch",
        ),
    ],
)
def test_tampered_authority_model_or_schema_fails_closed(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    error: str,
) -> None:
    payload = deepcopy(_run(tmp_path).model_dump(mode="json"))
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError, match=error):
        HostedNativeExtractionRunV1.model_validate(payload)


def test_source_or_prompt_byte_drift_fails_external_replay(tmp_path: Path) -> None:
    run = _run(tmp_path)
    (tmp_path / "archive/source_lines.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HostedNativeGroundingBridgeError, match="source_artifact_hash_mismatch"):
        validate_hosted_native_extraction_run_v1(run=run, repository_root=tmp_path)

    other = tmp_path / "other"
    run = _run(other)
    prompt_path = other / "prompts/native_extraction.md"
    prompt_path.write_text("changed", encoding="utf-8")
    with pytest.raises(
        HostedNativeGroundingBridgeError,
        match="prompt_template_hash_mismatch",
    ):
        validate_hosted_native_extraction_run_v1(run=run, repository_root=other)


def test_stale_v12_pipeline_cannot_freeze_a_hosted_native_run(tmp_path: Path) -> None:
    with pytest.raises(
        ValidationError,
        match="hosted_native_pipeline_fingerprint_incomplete",
    ):
        _run(tmp_path, pipeline_component_version="12")


def test_v13_hosted_bridge_byte_tamper_fails_external_replay(tmp_path: Path) -> None:
    run = _run(tmp_path)
    bridge_path = tmp_path / "src/literature_multiverse/hosted_native_grounding_bridge.py"
    bridge_path.write_text(
        bridge_path.read_text(encoding="utf-8") + "\n# simulated v13 byte drift\n",
        encoding="utf-8",
    )

    with pytest.raises(
        HostedNativeGroundingBridgeError,
        match=re.escape(
            "file_sha256_mismatch:src/literature_multiverse/hosted_native_grounding_bridge.py"
        ),
    ):
        validate_hosted_native_extraction_run_v1(run=run, repository_root=tmp_path)


def test_missing_terminal_call_cannot_be_rehashed_into_a_complete_run(
    tmp_path: Path,
) -> None:
    payload = _run(tmp_path).model_dump(mode="json")
    payload["calls"] = []
    payload["call_membership_sha256"] = hashlib.sha256(b"[]").hexdigest()
    with pytest.raises(ValidationError):
        HostedNativeExtractionRunV1.model_validate(payload)


def test_existing_diagnostic_contract_cannot_be_upgraded() -> None:
    diagnostic = {
        "terminal_version": "metasyn-contextual-frontier-terminal-report-v1",
        "terminal": True,
        "status": "typed_graph_mechanics_completed",
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    with pytest.raises(ValidationError):
        HostedNativeExtractionRunV1.model_validate(diagnostic)


def test_response_pointer_must_resolve_exact_postvalidated_extraction(
    tmp_path: Path,
) -> None:
    payload = _run(tmp_path).calls[0].terminal.model_dump(mode="json")
    payload["raw_response_utf8"] = json.dumps(
        {"not_the_extraction": payload["parsed_extraction"]},
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["raw_response_sha256"] = hashlib.sha256(
        payload["raw_response_utf8"].encode("utf-8")
    ).hexdigest()
    payload["terminal_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "terminal_sha256"}
    )
    with pytest.raises(ValidationError, match="structured_output_pointer_mismatch"):
        type(_run(tmp_path).calls[0].terminal).model_validate(payload)
