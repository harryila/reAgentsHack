from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.build_typed_evidence_corpus import main as build_typed_corpus_main
from scripts.s3_extract_typed import main as typed_extract_main

from literature_multiverse.cli import main as cli_main
from literature_multiverse.config import QuestionConfig
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    freeze_native_publication_extraction,
    native_extraction_prompt_replacements,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    NativeExtractionExecutionContext,
    NativeGroundingError,
    NativeGroundingReceipt,
    TypedEvidenceGroundingPackage,
    freeze_native_provider_execution_receipt,
    freeze_typed_evidence_grounding_package,
)
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.typed_extraction import (
    PublicationEvidenceFragment,
    SourceDocumentArtifact,
    assemble_typed_evidence_corpus,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    LegacyAdapterConfig,
    ScientificClaim,
    VerificationContractError,
    VerificationProtocol,
    compute_verifier_pipeline_fingerprint,
    load_corpus,
    verifier_pipeline_components,
)

PIPELINE_HASH = "d" * 64
SOURCE_HASH = "e" * 64
GROUNDING_HASH = "f" * 64


def _payload() -> dict[str, object]:
    return {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "estimable",
        "studies": [
            {
                "key": "trial-1",
                "source_label": "Randomized trial",
                "design": "parallel randomized controlled trial",
                "registration_ids": ["NCT00000001"],
                "cohorts": [
                    {
                        "key": "main-cohort",
                        "source_labels": ["intention-to-treat population"],
                        "registry_ids": ["NCT00000001"],
                        "dataset_ids": [],
                        "population_description": "Adults with the target condition",
                        "recruitment_period": None,
                        "total_sample_size": 100,
                        "arms": [
                            {
                                "key": "treatment",
                                "label": "Intervention",
                                "role": "intervention",
                                "description": None,
                                "sample_size": 50,
                            },
                            {
                                "key": "control",
                                "label": "Placebo",
                                "role": "control",
                                "description": None,
                                "sample_size": 50,
                            },
                        ],
                        "contrasts": [
                            {
                                "key": "primary",
                                "treatment_arm_key": "treatment",
                                "comparator_arm_key": "control",
                                "label": "intervention_vs_placebo",
                                "estimand": "intention-to-treat difference",
                                "positive_direction_means": (
                                    "higher performance under intervention"
                                ),
                            }
                        ],
                        "findings": [
                            {
                                "key": "performance-week-4",
                                "contrast_key": "primary",
                                "outcome_name": "performance",
                                "timepoint": {
                                    "kind": "exact",
                                    "value": 4,
                                    "unit": "week",
                                },
                                "analysis_population": "intention-to-treat",
                                "effect": {
                                    "effect_format": "hedges_g",
                                    "availability": "available",
                                    "estimate": 0.35,
                                    "standard_error": 0.1,
                                    "variance": None,
                                    "ci_lower": None,
                                    "ci_upper": None,
                                    "ci_level": 0.95,
                                    "unit": None,
                                    "treatment_mean": None,
                                    "treatment_sd": None,
                                    "treatment_n": None,
                                    "control_mean": None,
                                    "control_sd": None,
                                    "control_n": None,
                                    "treatment_events": None,
                                    "treatment_total": None,
                                    "control_events": None,
                                    "control_total": None,
                                    "reported_p_value": 0.002,
                                    "reported_significance": "significant",
                                    "equivalence_conclusion": "not_tested",
                                    "equivalence_margin": None,
                                    "moderators": [{"name": "dose", "value": "high"}],
                                    "extraction_method": "reported",
                                },
                                "evidence": {
                                    "source_locator": "article.pdf#page=7;table=2;row=4",
                                    "quote": "The standardized effect was 0.35 (SE 0.10).",
                                    "section": "Results",
                                    "page": 7,
                                    "char_start": None,
                                    "char_end": None,
                                    "line_ids": ["L210"],
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


def _publication() -> PublicationIdentity:
    return PublicationIdentity(
        publication_id="publication-authoritative",
        paper_id="paper-authoritative",
        doc_id="document-authoritative",
        doi="10.1000/native",
        title="A real trial",
    )


def _source() -> SourceDocumentArtifact:
    return SourceDocumentArtifact(
        artifact_path="data/raw/documents/article.pdf",
        sha256=SOURCE_HASH,
        media_type="application/pdf",
        source_locator="archive:article",
    )


def _write_cli_source(tmp_path: Path) -> tuple[Path, SourceDocumentArtifact]:
    path = tmp_path / "archive" / "source_lines.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "document-authoritative": {
                    "L210": {
                        "section": "Results",
                        "text": "The standardized effect was 0.35 (SE 0.10).",
                    }
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, SourceDocumentArtifact(
        artifact_path="archive/source_lines.json",
        sha256=sha256_file(path),
        media_type="application/json",
        source_locator=("json:archive/source_lines.json#/document-authoritative"),
    )


def _write_cli_map(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                "Map results [native-map-1]",
                "--- [1] [success] A real trial ---",
                "doc_id: document-authoritative",
                json.dumps(payload),
            ]
        ),
        encoding="utf-8",
    )


def _write_cli_manifest(
    path: Path,
    *,
    question_id: str,
    source_document: SourceDocumentArtifact,
) -> None:
    path.write_text(
        json.dumps(
            {
                "source_manifest_version": "native-source-manifest-v1",
                "question_id": question_id,
                "records": [
                    {
                        "doc_id": "document-authoritative",
                        "publication": _publication().model_dump(mode="json"),
                        "source_document": source_document.model_dump(mode="json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_archived_execution_receipt(path: Path) -> None:
    receipt = freeze_native_provider_execution_receipt(
        execution_id="native-map-1",
        execution_mode="paperclip_archived",
        provider_id="paperclip",
        model_id="archived-test-model",
        model_revision="immutable-test-revision",
        runtime_id="paperclip-cli",
        runtime_version="test-v1",
        runtime_metadata={"fixture": True},
        raw_call_ledger={
            "map_result_id": "native-map-1",
            "response_count": 1,
            "transport_status": "archived_success",
        },
        call_count=1,
    )
    path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")


def test_native_payload_builds_complete_typed_fragment_with_injected_identity() -> None:
    payload = NativePublicationExtraction.model_validate(_payload())

    fragment = freeze_native_publication_extraction(
        payload=payload,
        question_id="native-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=_source(),
        grounding_receipt_sha256=GROUNDING_HASH,
    )

    assert fragment.status == "estimable"
    assert fragment.paper_id == "paper-authoritative"
    assert fragment.graph is not None
    assert fragment.graph.publications == [_publication()]
    estimate = fragment.graph.outcome_estimates[0]
    assert estimate.effect.estimate == 0.35
    assert estimate.effect.moderators == {"dose": "high"}
    assert estimate.effect.paper_id == "paper-authoritative"
    assert estimate.effect.provenance.source_locator.endswith("table=2;row=4")
    assert fragment.graph.cohorts[0].identity.registry_ids == ["NCT00000001"]


def test_native_non_estimable_payload_preserves_explicit_reason() -> None:
    payload = NativePublicationExtraction(
        status="non_estimable",
        studies=[],
        non_estimability_reason="uncertainty_absent",
        non_estimability_detail="Only an unsupported point estimate was reported.",
    )

    fragment = freeze_native_publication_extraction(
        payload=payload,
        question_id="native-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=_source(),
        grounding_receipt_sha256=None,
    )

    assert fragment.status == "non_estimable"
    assert fragment.graph is None
    assert fragment.non_estimability_reason == "uncertainty_absent"


def test_native_payload_rejects_unknown_contrast_reference() -> None:
    payload = deepcopy(_payload())
    payload["studies"][0]["cohorts"][0]["findings"][0]["contrast_key"] = "missing"

    with pytest.raises(ValidationError, match="native_finding_contrast_unknown"):
        NativePublicationExtraction.model_validate(payload)


def test_native_payload_rejects_invalid_effect_uncertainty_combination() -> None:
    payload = deepcopy(_payload())
    effect = payload["studies"][0]["cohorts"][0]["findings"][0]["effect"]
    effect["variance"] = 0.01

    validated = NativePublicationExtraction.model_validate(payload)
    with pytest.raises(ValidationError, match="direct_uncertainty_sources_mutually_exclusive"):
        freeze_native_publication_extraction(
            payload=validated,
            question_id="native-question",
            publication=_publication(),
            pipeline_fingerprint_sha256=PIPELINE_HASH,
            source_document=_source(),
            grounding_receipt_sha256=GROUNDING_HASH,
        )


def test_native_extraction_schema_is_closed_and_versioned() -> None:
    schema = native_publication_extraction_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(":v1")


def test_native_extraction_prompt_binds_frozen_question(
    repo_root: Path, fixture_config: QuestionConfig
) -> None:
    rendered = render_prompt_file(
        repo_root / "prompts" / "native_extraction.md",
        native_extraction_prompt_replacements(fixture_config),
    )

    assert rendered.prompt_version == "native-extraction-v3"
    assert fixture_config.research_question in rendered.text
    assert "[[" not in rendered.text


def test_archived_native_extraction_cli_builds_real_graph_artifacts(
    tmp_path: Path, fixture_config: QuestionConfig
) -> None:
    source_path, source_document = _write_cli_source(tmp_path)
    payload = deepcopy(_payload())
    evidence = payload["studies"][0]["cohorts"][0]["findings"][0]["evidence"]
    evidence["source_locator"] = source_document.source_locator
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    output = tmp_path / "typed"

    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    corpus = json.loads((output / "typed_evidence_corpus.json").read_text())
    run = json.loads((output / "native_extraction_run.json").read_text())
    receipts = [
        json.loads(line) for line in (output / "grounding_receipts.jsonl").read_text().splitlines()
    ]
    assert corpus["graph"]["outcome_estimates"][0]["effect"]["estimate"] == 0.35
    assert corpus["fragments"][0]["grounding_receipt_sha256"] == receipts[0]["receipt_sha256"]
    assert receipts[0]["authorizes_estimable_fragment"] is True
    assert run["counts"]["estimable_publications"] == 1
    assert run["counts"]["grounding_authorizing_receipts"] == 1
    assert run["grounding_receipts_sha256"] == sha256_file(output / "grounding_receipts.jsonl")
    assert (output / "rendered_native_extraction_prompt.md").is_file()
    loaded = load_corpus(
        output,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )
    assert loaded.source_format == "typed_evidence_grounding_package_json"
    assert loaded.metadata["grounding_receipts"] == 1
    assert loaded.provenance_assurance.status == "source_replayed_native_grounding"
    assert loaded.provenance_assurance.replay_sha256 == loaded.metadata["grounding_replay_sha256"]
    assert loaded.provenance_release_eligible() is False
    assert "native_extraction_context_unbound" in {
        issue.code for issue in loaded.adapter_issues
    }
    assert not any(issue.code == "unverified_source_provenance" for issue in loaded.adapter_issues)
    rebuilt = tmp_path / "rebuilt"
    assert (
        build_typed_corpus_main(
            [
                "--fragments",
                str(output / "publication_fragments.jsonl"),
                "--grounding-receipts",
                str(output / "grounding_receipts.jsonl"),
                "--output-dir",
                str(rebuilt),
            ]
        )
        == 0
    )
    rebuilt_loaded = load_corpus(
        rebuilt,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )
    assert rebuilt_loaded.metadata["grounding_receipts"] == 1
    assert rebuilt_loaded.provenance_release_eligible() is False
    assert any(
        issue.code == "native_source_manifest_membership_unbound"
        for issue in rebuilt_loaded.adapter_issues
    )

    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        VerificationContractError,
        match="grounding_receipt_replay_mismatch",
    ):
        load_corpus(
            output,
            legacy_settings=LegacyAdapterConfig(),
            repository_root=tmp_path,
        )


def test_archived_native_extraction_cli_with_receipt_builds_replayable_v4_package(
    tmp_path: Path,
    fixture_config: QuestionConfig,
    repo_root: Path,
) -> None:
    _source_path, source_document = _write_cli_source(tmp_path)
    payload = deepcopy(_payload())
    finding = payload["studies"][0]["cohorts"][0]["findings"][0]
    endpoint = fixture_config.outcomes.included_primary_endpoints[0]
    finding["outcome_name"] = endpoint
    finding["effect"]["moderators"] = []
    evidence = finding["evidence"]
    evidence["source_locator"] = source_document.source_locator
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    execution_receipt = tmp_path / "execution-receipt.json"
    _write_archived_execution_receipt(execution_receipt)
    pipeline_paths = {
        path
        for component in verifier_pipeline_components()
        for path in component.file_paths
    }
    for relative in sorted(pipeline_paths):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / relative, destination)
    fingerprint = compute_verifier_pipeline_fingerprint(root=tmp_path)
    output = tmp_path / "typed-v4"

    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--execution-receipt",
                str(execution_receipt),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                fingerprint.pipeline_sha256,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    package = json.loads(
        (output / "typed_evidence_grounding_package.json").read_text()
    )
    corpus = json.loads((output / "typed_evidence_corpus.json").read_text())
    context = json.loads((output / "native_extraction_context.json").read_text())
    run = json.loads((output / "native_extraction_run.json").read_text())
    assert package["package_version"] == "typed-evidence-grounding-package-v4"
    assert corpus["corpus_version"] == "typed-evidence-corpus-v3"
    assert corpus["extraction_context_sha256"] == context["context_sha256"]
    assert run["provenance_release_eligible"] is True
    assert run["extraction_context_sha256"] == context["context_sha256"]
    loaded = load_corpus(
        output,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )
    assert loaded.provenance_release_eligible() is True
    assert loaded.extraction_context is not None
    assert loaded.metadata["grounding_package_version"] == (
        "typed-evidence-grounding-package-v4"
    )
    assert "raw_call_ledger" not in json.dumps(loaded.certificate_payload())

    rebuilt = tmp_path / "rebuilt-v4"
    assert (
        build_typed_corpus_main(
            [
                "--fragments",
                str(output / "publication_fragments.jsonl"),
                "--grounding-receipts",
                str(output / "grounding_receipts.jsonl"),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--extraction-context",
                str(output / "native_extraction_context.json"),
                "--output-dir",
                str(rebuilt),
            ]
        )
        == 0
    )
    rebuilt_loaded = load_corpus(
        rebuilt,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )
    assert rebuilt_loaded.provenance_release_eligible() is True

    claim = ClaimManifest(
        question_id=fixture_config.question_id,
        population_id="fixture-population",
        domain="synthetic",
        claim=ScientificClaim(
            statement="The intervention increases the configured fixture endpoint.",
            direction="increase",
            outcome_name=endpoint,
            estimand="intention-to-treat difference",
        ),
        protocol=VerificationProtocol(
            corpus_cutoff="fixture-corpus-v1",
            inclusion_criteria=fixture_config.eligibility.include,
            exclusion_criteria=fixture_config.eligibility.exclude,
        ),
    )
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")
    certificate_dir = tmp_path / "certificate"
    assert (
        cli_main(
            [
                "verify",
                "--claim",
                str(claim_path),
                "--corpus",
                str(output),
                "--budget-minutes",
                "30",
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(certificate_dir),
            ]
        )
        == 0
    )
    certificate = json.loads(
        (certificate_dir / "verification-certificate.json").read_text()
    )
    assert certificate["corpus"]["provenance_assurance"]["release_eligible"] is True
    assert not any(
        issue["code"].startswith("native_claim_")
        or issue["code"].startswith("native_protocol_")
        or issue["code"].startswith("native_extracted_")
        for issue in certificate["adapter_issues"]
    )

    incompatible_claim = claim.model_copy(
        update={
            "protocol": claim.protocol.model_copy(
                update={"inclusion_criteria": ["different inclusion protocol"]}
            )
        }
    )
    incompatible_path = tmp_path / "incompatible-claim.json"
    incompatible_path.write_text(
        incompatible_claim.model_dump_json(indent=2), encoding="utf-8"
    )
    incompatible_dir = tmp_path / "incompatible-certificate"
    assert (
        cli_main(
            [
                "verify",
                "--claim",
                str(incompatible_path),
                "--corpus",
                str(output),
                "--budget-minutes",
                "30",
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(incompatible_dir),
            ]
        )
        == 0
    )
    incompatible_certificate = json.loads(
        (incompatible_dir / "verification-certificate.json").read_text()
    )
    assert incompatible_certificate["status"] == "abstained"
    assert "adapter:native_protocol_inclusion_config_mismatch" in (
        incompatible_certificate["reasons"]
    )

    tampered_package = deepcopy(package)
    tampered_context = tampered_package["extraction_context_receipt"][
        "execution_context"
    ]
    provider = tampered_context["provider_execution_receipts"][0]
    provider["raw_call_ledger"]["transport_status"] = "tampered"
    provider["raw_call_ledger_sha256"] = hash_canonical(
        provider["raw_call_ledger"]
    )
    provider["receipt_sha256"] = hash_canonical(
        {key: value for key, value in provider.items() if key != "receipt_sha256"}
    )
    tampered_context["context_sha256"] = hash_canonical(
        {
            key: value
            for key, value in tampered_context.items()
            if key != "context_sha256"
        }
    )
    context_receipt = tampered_package["extraction_context_receipt"]
    context_receipt["receipt_sha256"] = hash_canonical(
        {
            key: value
            for key, value in context_receipt.items()
            if key != "receipt_sha256"
        }
    )
    tampered_package["package_sha256"] = hash_canonical(
        {
            key: value
            for key, value in tampered_package.items()
            if key != "package_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="typed_evidence_grounding_package_context_corpus_mismatch",
    ):
        TypedEvidenceGroundingPackage.model_validate(tampered_package)

    context_payload = deepcopy(context)
    map_artifact = next(
        artifact
        for artifact in context_payload["input_artifacts"]
        if artifact["role"] == "map_output"
    )
    map_artifact["execution_ids"] = ["different-execution"]
    context_payload["context_sha256"] = hash_canonical(
        {
            key: value
            for key, value in context_payload.items()
            if key != "context_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="native_extraction_context_artifact_execution_id_unbound",
    ):
        NativeExtractionExecutionContext.model_validate(context_payload)


def test_archived_native_extraction_cli_downgrades_grounding_mismatch(
    tmp_path: Path, fixture_config: QuestionConfig
) -> None:
    _source_path, source_document = _write_cli_source(tmp_path)
    payload = deepcopy(_payload())
    evidence = payload["studies"][0]["cohorts"][0]["findings"][0]["evidence"]
    evidence["source_locator"] = source_document.source_locator
    evidence["quote"] = "The standardized effect was 9.99 (SE 0.10)."
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    output = tmp_path / "typed"

    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    corpus = json.loads((output / "typed_evidence_corpus.json").read_text())
    fragment = corpus["fragments"][0]
    receipt = json.loads((output / "grounding_receipts.jsonl").read_text())
    run = json.loads((output / "native_extraction_run.json").read_text())
    assert len(corpus["graph"]["publications"]) == 1
    assert corpus["graph"]["outcome_estimates"] == []
    assert fragment["status"] == "non_estimable"
    assert fragment["non_estimability_reason"] == "ungrounded_numerical_result"
    assert fragment["grounding_receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt["authorizes_estimable_fragment"] is False
    assert run["counts"]["grounding_failed_estimable_receipts"] == 1
    assert run["counts"]["grounding_expected_non_estimable_extraction_receipts"] == 0
    assert (output / "evidence_graph.json").is_file()
    assert (output / "reconciled_evidence_graph.json").is_file()
    assert run["cohort_reconciliation_status"] == "no_estimable_graph"


def test_archived_native_extraction_cli_counts_expected_non_estimable_separately(
    tmp_path: Path, fixture_config: QuestionConfig
) -> None:
    _source_path, source_document = _write_cli_source(tmp_path)
    payload = {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "non_estimable",
        "studies": [],
        "non_estimability_reason": "numerical_result_absent",
        "non_estimability_detail": None,
        "warnings": [],
    }
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    output = tmp_path / "typed"

    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    run = json.loads((output / "native_extraction_run.json").read_text())
    corpus = json.loads((output / "typed_evidence_corpus.json").read_text())
    assert corpus["fragments"][0]["non_estimability_reason"] == "numerical_result_absent"
    assert run["counts"]["grounding_expected_non_estimable_extraction_receipts"] == 1
    assert run["counts"]["grounding_failed_estimable_receipts"] == 0


def test_public_loader_rejects_rehashed_extraction_graph_projection_tampering(
    tmp_path: Path, fixture_config: QuestionConfig
) -> None:
    _source_path, source_document = _write_cli_source(tmp_path)
    payload = deepcopy(_payload())
    evidence = payload["studies"][0]["cohorts"][0]["findings"][0]["evidence"]
    evidence["source_locator"] = source_document.source_locator
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    output = tmp_path / "typed"
    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    receipt_payload = json.loads((output / "grounding_receipts.jsonl").read_text())
    receipt_payload["extraction"]["studies"][0]["cohorts"][0]["findings"][0]["effect"][
        "estimate"
    ] = 0.99
    receipt_payload["extraction_sha256"] = hash_canonical(receipt_payload["extraction"])
    receipt_payload.pop("receipt_sha256")
    tampered_receipt = NativeGroundingReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": hash_canonical(receipt_payload),
        }
    )
    corpus_payload = json.loads((output / "typed_evidence_corpus.json").read_text())
    fragment_payload = corpus_payload["fragments"][0]
    fragment_payload["grounding_receipt_sha256"] = tampered_receipt.receipt_sha256
    fragment_payload.pop("fragment_sha256")
    tampered_fragment = PublicationEvidenceFragment.model_validate(
        {
            **fragment_payload,
            "fragment_sha256": hash_canonical(fragment_payload),
        }
    )
    tampered_corpus = assemble_typed_evidence_corpus([tampered_fragment])
    with pytest.raises(
        NativeGroundingError,
        match="receipt_linked_fragment_projection_mismatch",
    ):
        freeze_typed_evidence_grounding_package(
            corpus=tampered_corpus,
            grounding_receipts=[tampered_receipt],
        )


def test_archived_native_extraction_cli_downgrades_source_hash_drift(
    tmp_path: Path, fixture_config: QuestionConfig
) -> None:
    source_path, source_document = _write_cli_source(tmp_path)
    payload = deepcopy(_payload())
    evidence = payload["studies"][0]["cohorts"][0]["findings"][0]["evidence"]
    evidence["source_locator"] = source_document.source_locator
    map_output = tmp_path / "native-map.txt"
    _write_cli_map(map_output, payload)
    source_manifest = tmp_path / "source-manifest.json"
    _write_cli_manifest(
        source_manifest,
        question_id=fixture_config.question_id,
        source_document=source_document,
    )
    source_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "typed"

    assert (
        typed_extract_main(
            [
                "--question",
                fixture_config.question_id,
                "--map-output",
                str(map_output),
                "--source-manifest",
                str(source_manifest),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--pipeline-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    corpus = json.loads((output / "typed_evidence_corpus.json").read_text())
    fragment = corpus["fragments"][0]
    receipt = json.loads((output / "grounding_receipts.jsonl").read_text())
    run = json.loads((output / "native_extraction_run.json").read_text())
    assert fragment["non_estimability_reason"] == "source_document_incomplete"
    assert receipt["issues"] == ["source_artifact_hash_mismatch"]
    assert receipt["observed_source_sha256"] != receipt["expected_source_sha256"]
    assert fragment["grounding_receipt_sha256"] == receipt["receipt_sha256"]
    assert run["counts"]["grounding_failed_estimable_receipts"] == 1


def test_live_native_extraction_rejects_unverified_hash_before_provider_call(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="live_native_extraction_requires_verified_pipeline_artifact",
    ):
        typed_extract_main(
            [
                "--question",
                "fixture-a",
                "--live",
                "--from-result",
                "s_would_call_provider",
                "--source-manifest",
                str(tmp_path / "not-opened.json"),
                "--corpus-cutoff",
                "fixture-corpus-v1",
                "--pipeline-fingerprint",
                PIPELINE_HASH,
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
