#!/usr/bin/env python3
"""Run or ingest native extraction and build typed publication fragments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.extract import parse_map_file, reconcile_envelopes
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.live import live_map_to_results_file
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    native_extraction_prompt_replacements,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    NativeEvaluationSchemaArtifact,
    NativeExtractionArtifactDigest,
    NativeGroundingReceipt,
    NativeProviderExecutionReceipt,
    NativeRenderedPromptArtifact,
    freeze_grounding_checked_publication_fragment,
    freeze_native_extraction_execution_context,
    freeze_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--map-output",
        type=Path,
        nargs="+",
        help="archived terminal map result file(s)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="explicitly launch and archive a provider map",
    )
    parser.add_argument("--from-result", action="append")
    parser.add_argument("--resume-map-id", action="append")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--execution-receipt",
        type=Path,
        action="append",
        help=(
            "Explicit self-hashed provider/model execution receipt; repeat for batches. "
            "Without it, the command writes an analysis-only v3 package."
        ),
    )
    parser.add_argument(
        "--corpus-cutoff",
        required=True,
        help="Exact frozen corpus-cutoff identifier bound into the grounding package",
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument(
        "--pipeline-fingerprint",
        help="low-level offline-only hash; prefer a computed fingerprint artifact",
    )
    identity.add_argument(
        "--pipeline-fingerprint-artifact",
        type=Path,
        help="computed fingerprint JSON that is reverified against current files",
    )
    parser.add_argument("--pipeline-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"native_source_manifest_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("native_source_manifest_must_be_object")
    return payload


def _pipeline_identity(args: argparse.Namespace) -> tuple[str, str, dict[str, Any] | None]:
    if args.pipeline_fingerprint_artifact is None:
        assert args.pipeline_fingerprint is not None
        if not _SHA256.fullmatch(args.pipeline_fingerprint):
            raise ValueError("native_pipeline_fingerprint_invalid")
        if args.live:
            raise ValueError("live_native_extraction_requires_verified_pipeline_artifact")
        return args.pipeline_fingerprint, "caller_supplied_hash_offline_only", None
    payload = _json_object(args.pipeline_fingerprint_artifact)
    if "pipeline_fingerprint" in payload:
        nested = payload["pipeline_fingerprint"]
        if not isinstance(nested, dict):
            raise ValueError("native_pipeline_fingerprint_wrapper_invalid")
        payload = nested
    expected = PipelineFingerprint.model_validate(payload)
    proof = require_pipeline_fingerprint_match(
        expected=expected,
        root=args.pipeline_root,
    )
    assert proof.computed_pipeline_sha256 is not None
    return (
        proof.computed_pipeline_sha256,
        "verified_computed_pipeline_artifact",
        proof.model_dump(mode="json"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.live and not (args.from_result or args.resume_map_id):
        raise ValueError("live_native_extraction_requires_from_result_or_resume_map_id")
    if not args.live and (args.from_result or args.resume_map_id):
        raise ValueError("provider_result_ids_require_live_mode")
    if args.from_result and args.resume_map_id:
        raise ValueError("from_result_and_resume_map_id_are_mutually_exclusive")
    if args.concurrency is not None and args.concurrency < 1:
        raise ValueError("concurrency_must_be_positive")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds_must_be_positive")
    pipeline_sha256, pipeline_basis, pipeline_proof = _pipeline_identity(args)
    config = load_config_for_question(args.question, require_locked=True)
    if args.live:
        config.authorize_stage("s3", explicit_fixture=False, live_provider=True)
    source_manifest = NativeSourceManifest.model_validate(_json_object(args.source_manifest))
    if source_manifest.question_id != config.question_id:
        raise ValueError("native_source_manifest_question_mismatch")
    source_by_doc = {record.doc_id: record for record in source_manifest.records}

    schema = native_publication_extraction_json_schema()
    prompt_template_path = Path("prompts/native_extraction.md")
    prompt = render_prompt_file(
        prompt_template_path,
        native_extraction_prompt_replacements(config),
    )
    provider_artifacts: list[Path] = []
    map_paths: list[Path]
    if args.live:
        provider_dir = args.output_dir / "provider"
        batches = (
            [("resume", value) for value in args.resume_map_id]
            if args.resume_map_id
            else [("from", value) for value in args.from_result]
        )
        map_paths = []
        for index, (kind, value) in enumerate(batches, start=1):
            live = live_map_to_results_file(
                archive_dir=provider_dir,
                archive_stem=f"native-map-{pipeline_sha256[:8]}-b{index:02d}",
                from_result=value if kind == "from" else None,
                schema_json=json.dumps(schema, sort_keys=True) if kind == "from" else None,
                prompt=prompt.text if kind == "from" else None,
                concurrency=args.concurrency if kind == "from" else None,
                resume_map_id=value if kind == "resume" else None,
                retry_failed=kind == "resume",
                timeout_seconds=args.timeout_seconds,
                force=args.force,
            )
            map_paths.append(live.results_path)
            provider_artifacts.extend(live.artifacts)
    else:
        assert args.map_output is not None
        map_paths = args.map_output
    envelope_batches = [parse_map_file(path) for path in map_paths]
    envelopes = reconcile_envelopes(
        envelope_batches,
        expected_doc_ids=sorted(source_by_doc),
    )

    extraction_context = None
    execution_receipts: list[NativeProviderExecutionReceipt] = []
    if args.execution_receipt:
        for receipt_path in args.execution_receipt:
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError("native_execution_receipt_file_invalid")
            execution_receipts.append(
                NativeProviderExecutionReceipt.model_validate(_json_object(receipt_path))
            )
        execution_mode = "paperclip_live" if args.live else "paperclip_archived"
        if {receipt.execution_mode for receipt in execution_receipts} != {
            execution_mode
        }:
            raise ValueError("native_execution_receipt_mode_mismatch")
        observed_map_ids = sorted({envelope.map_result_id for envelope in envelopes})
        expected_execution_ids = sorted(
            [
                *(args.from_result or []),
                *(args.resume_map_id or []),
            ]
            if args.live
            else observed_map_ids
        )
        if sorted(receipt.execution_id for receipt in execution_receipts) != (
            expected_execution_ids
        ):
            raise ValueError("native_execution_receipt_batch_identity_mismatch")
        artifacts: list[NativeExtractionArtifactDigest] = [
            NativeExtractionArtifactDigest(
                artifact_id="source-manifest-input",
                role="source_manifest_input",
                sha256=sha256_file(args.source_manifest),
                hash_basis="raw_bytes",
                byte_count=args.source_manifest.stat().st_size,
            )
        ]
        for index, path in enumerate(map_paths, start=1):
            map_ids = sorted(
                {envelope.map_result_id for envelope in parse_map_file(path)}
            )
            artifacts.append(
                NativeExtractionArtifactDigest(
                    artifact_id=f"map-output-{index:03d}",
                    role="map_output",
                    sha256=sha256_file(path),
                    hash_basis="raw_bytes",
                    byte_count=path.stat().st_size,
                    execution_ids=map_ids,
                )
            )
        for index, path in enumerate(sorted(provider_artifacts), start=1):
            artifacts.append(
                NativeExtractionArtifactDigest(
                    artifact_id=f"provider-artifact-{index:03d}",
                    role="provider_artifact",
                    sha256=sha256_file(path),
                    hash_basis="raw_bytes",
                    byte_count=path.stat().st_size,
                )
            )
        for index, path in enumerate(args.execution_receipt, start=1):
            receipt = execution_receipts[index - 1]
            artifacts.append(
                NativeExtractionArtifactDigest(
                    artifact_id=f"provider-execution-receipt-{index:03d}",
                    role="provider_execution_receipt",
                    sha256=sha256_file(path),
                    hash_basis="raw_bytes",
                    byte_count=path.stat().st_size,
                    execution_ids=[receipt.execution_id],
                )
            )
        schema_hash = hash_canonical(schema)
        extraction_context = freeze_native_extraction_execution_context(
            extraction_mode=execution_mode,
            question_config=config,
            pipeline_fingerprint_sha256=pipeline_sha256,
            rendered_prompts=[
                NativeRenderedPromptArtifact(
                    prompt_id="native-extraction-default",
                    renderer_id="repository-native-extraction-v1",
                    prompt_version=prompt.prompt_version,
                    template_path="prompts/native_extraction.md",
                    template_sha256=sha256_file(prompt_template_path),
                    rendered_prompt=prompt.text,
                    rendered_prompt_sha256=prompt.sha256,
                )
            ],
            evaluation_schemas=[
                NativeEvaluationSchemaArtifact(
                    schema_id="native-official-postvalidation",
                    role="official_postvalidation",
                    schema_payload=schema,
                    schema_sha256=schema_hash,
                ),
                NativeEvaluationSchemaArtifact(
                    schema_id="paperclip-generation-constraint",
                    role="generation_constraint",
                    schema_payload=schema,
                    schema_sha256=schema_hash,
                ),
            ],
            provider_execution_receipts=execution_receipts,
            input_artifacts=artifacts,
            source_manifest_content_sha256=hash_canonical(source_manifest),
            source_manifest_records=len(source_manifest.records),
            corpus_cutoff=args.corpus_cutoff,
        )

    fragments = []
    grounding_receipts: list[NativeGroundingReceipt] = []
    for envelope in envelopes:
        source = source_by_doc[envelope.doc_id]
        if envelope.successful:
            if envelope.payload is None:
                raise ValueError("native_successful_map_payload_missing")
            extraction = NativePublicationExtraction.model_validate(envelope.payload)
            grounding = verify_native_publication_grounding(
                repository_root=args.pipeline_root,
                source_document=source.source_document,
                extraction=extraction,
            )
            grounding_receipts.append(grounding)
            fragment = freeze_grounding_checked_publication_fragment(
                extraction=extraction,
                grounding_receipt=grounding,
                question_id=config.question_id,
                publication=source.publication,
                pipeline_fingerprint_sha256=pipeline_sha256,
                extraction_context_sha256=(
                    extraction_context.context_sha256
                    if extraction_context is not None
                    else None
                ),
                source_document=source.source_document,
            )
        else:
            fragment = freeze_publication_evidence_fragment(
                question_id=config.question_id,
                publication_id=source.publication.publication_id,
                paper_id=source.publication.paper_id,
                publication=source.publication,
                pipeline_fingerprint_sha256=pipeline_sha256,
                extraction_context_sha256=(
                    extraction_context.context_sha256
                    if extraction_context is not None
                    else None
                ),
                source_document=source.source_document,
                grounding_receipt_sha256=None,
                status=FragmentStatus.NON_ESTIMABLE,
                non_estimability_reason=(NonEstimabilityReason.SOURCE_DOCUMENT_INCOMPLETE),
                non_estimability_detail=(
                    f"Extraction provider returned terminal status {envelope.status}."
                ),
                extractor_warnings=[f"provider_status:{envelope.status}"],
            )
        fragments.append(fragment)

    corpus = assemble_typed_evidence_corpus(fragments)
    grounding_package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=grounding_receipts,
        source_manifest=source_manifest,
        corpus_cutoff=args.corpus_cutoff,
        extraction_context=extraction_context,
    )
    output_dir: Path = args.output_dir
    atomic_write_json(output_dir / "native_extraction.schema.json", schema, force=args.force)
    atomic_write_text(
        output_dir / "rendered_native_extraction_prompt.md",
        prompt.text,
        force=args.force,
    )
    atomic_write_jsonl(output_dir / "publication_fragments.jsonl", fragments, force=args.force)
    grounding_receipts_path = output_dir / "grounding_receipts.jsonl"
    atomic_write_jsonl(grounding_receipts_path, grounding_receipts, force=args.force)
    atomic_write_json(output_dir / "typed_evidence_corpus.json", corpus, force=args.force)
    if extraction_context is not None:
        atomic_write_json(
            output_dir / "native_extraction_context.json",
            extraction_context,
            force=args.force,
        )
    atomic_write_json(
        output_dir / "typed_evidence_grounding_package.json",
        grounding_package,
        force=args.force,
    )
    atomic_write_json(output_dir / "evidence_graph.json", corpus.graph, force=args.force)
    cohort_reconciliation = grounding_package.cohort_reconciliation
    assert cohort_reconciliation is not None
    assert cohort_reconciliation.reconciled_graph is not None
    atomic_write_json(
        output_dir / "reconciled_evidence_graph.json",
        cohort_reconciliation.reconciled_graph,
        force=args.force,
    )

    run_payload: dict[str, Any] = {
        "native_extraction_run_version": "3",
        "question_id": config.question_id,
        "pipeline_fingerprint_sha256": pipeline_sha256,
        "pipeline_identity_basis": pipeline_basis,
        "pipeline_verification": pipeline_proof,
        "config_sha256": config_sha256(config),
        "prompt_sha256": prompt.sha256,
        "schema_sha256": hash_canonical(schema),
        "extraction_context_sha256": (
            extraction_context.context_sha256
            if extraction_context is not None
            else None
        ),
        "extraction_context_receipt_sha256": (
            grounding_package.extraction_context_receipt.receipt_sha256
            if grounding_package.extraction_context_receipt is not None
            else None
        ),
        "provider_execution_receipt_sha256s": sorted(
            receipt.receipt_sha256 for receipt in execution_receipts
        ),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_manifest_content_sha256": grounding_package.source_manifest_sha256,
        "source_manifest_records": len(source_manifest.records),
        "corpus_cutoff": args.corpus_cutoff,
        "map_output_sha256s": {path.as_posix(): sha256_file(path) for path in sorted(map_paths)},
        "provider_artifact_sha256s": {
            path.as_posix(): sha256_file(path) for path in sorted(provider_artifacts)
        },
        "corpus_sha256": corpus.corpus_sha256,
        "grounding_package_sha256": grounding_package.package_sha256,
        "grounding_package_version": grounding_package.package_version,
        "provenance_release_eligible": (
            grounding_package.package_version
            == "typed-evidence-grounding-package-v4"
        ),
        "grounding_validation_sha256": (grounding_package.grounding_validation.validation_sha256),
        "cohort_reconciliation_status": cohort_reconciliation.status.value,
        "cohort_reconciliation_receipt_sha256": cohort_reconciliation.receipt_sha256,
        "cross_publication_identity_assurance_complete": (
            cohort_reconciliation.cross_publication_identity_assurance_complete
        ),
        "reconciled_graph_sha256": cohort_reconciliation.reconciled_graph_sha256,
        "merged_study_groups": cohort_reconciliation.merged_study_groups,
        "merged_cohort_groups": cohort_reconciliation.merged_cohort_groups,
        "grounding_receipts_sha256": sha256_file(grounding_receipts_path),
        "counts": {
            "fragments": len(corpus.fragments),
            "grounding_receipts": len(grounding_receipts),
            "grounding_authorizing_receipts": sum(
                receipt.authorizes_estimable_fragment for receipt in grounding_receipts
            ),
            "grounding_expected_non_estimable_extraction_receipts": sum(
                receipt.extraction_status is FragmentStatus.NON_ESTIMABLE
                for receipt in grounding_receipts
            ),
            "grounding_failed_estimable_receipts": sum(
                receipt.extraction_status is FragmentStatus.ESTIMABLE
                and not receipt.authorizes_estimable_fragment
                for receipt in grounding_receipts
            ),
            "grounding_non_authorizing_receipts": sum(
                not receipt.authorizes_estimable_fragment for receipt in grounding_receipts
            ),
            "grounding_source_verified_receipts": sum(
                receipt.source_verified for receipt in grounding_receipts
            ),
            "grounding_finding_results": sum(
                len(receipt.finding_results) for receipt in grounding_receipts
            ),
            "grounding_exact_findings": sum(
                result.status.value == "exact"
                for receipt in grounding_receipts
                for result in receipt.finding_results
            ),
            "grounding_mismatch_findings": sum(
                result.status.value == "mismatch"
                for receipt in grounding_receipts
                for result in receipt.finding_results
            ),
            "grounding_unverifiable_findings": sum(
                result.status.value == "unverifiable"
                for receipt in grounding_receipts
                for result in receipt.finding_results
            ),
            "estimable_publications": len(corpus.estimable_publication_ids),
            "non_estimable_publications": len(corpus.non_estimable_publication_ids),
        },
    }
    atomic_write_json(
        output_dir / "native_extraction_run.json",
        {**run_payload, "run_sha256": hash_canonical(run_payload)},
        force=args.force,
    )
    print(
        json.dumps(
            {
                "corpus_sha256": corpus.corpus_sha256,
                "estimable_publications": len(corpus.estimable_publication_ids),
                "non_estimable_publications": len(corpus.non_estimable_publication_ids),
                "output": (output_dir / "typed_evidence_grounding_package.json").as_posix(),
                "question_id": corpus.question_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
