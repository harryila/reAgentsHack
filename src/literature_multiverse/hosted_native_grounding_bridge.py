"""Offline bridge from a hosted exact-once run to native grounding package v4.

The bridge never calls a provider and never consults labels.  It externally replays
the computed pipeline identity, prompt templates, complete source manifest, exact
source grounding, terminal-call membership, and deterministic typed-graph projection.
Passing this boundary establishes source-provenance input authority only; it does not
itself authorize a scientific claim release.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.cohort_reconciliation import (
    ReviewerCohortReconciliationArtifact,
)
from literature_multiverse.hosted_native_extraction_contract import (
    HostedNativeExtractionRunV1,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import (
    NativeEvaluationSchemaArtifact,
    NativeExtractionArtifactDigest,
    NativeExtractionExecutionContext,
    NativeGroundingReceipt,
    NativeRenderedPromptArtifact,
    TypedEvidenceGroundingPackage,
    freeze_grounding_checked_publication_fragment,
    freeze_native_extraction_execution_context,
    freeze_native_provider_execution_receipt,
    freeze_typed_evidence_grounding_package,
    resolve_native_source_document,
    reverify_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprintError,
    PipelineFingerprintVerification,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

BRIDGE_VERSION = "hosted-native-grounding-bridge-v1"
BRIDGE_RECEIPT_VERSION = "hosted-native-grounding-bridge-receipt-v1"


class HostedNativeGroundingBridgeError(ValueError):
    """The hosted run cannot truthfully become a native grounding package."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class HostedNativeGroundingBridgeReceiptV1(_Frozen):
    receipt_version: Literal["hosted-native-grounding-bridge-receipt-v1"] = BRIDGE_RECEIPT_VERSION
    bridge_version: Literal["hosted-native-grounding-bridge-v1"] = BRIDGE_VERSION
    hosted_run_sha256: Sha256
    hosted_run_artifact_sha256: Sha256
    provider_identity_sha256: Sha256
    pipeline_fingerprint_sha256: Sha256
    pipeline_verification_sha256: Sha256
    question_config_sha256: Sha256
    source_manifest_sha256: Sha256
    source_membership_sha256: Sha256
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    extraction_context_sha256: Sha256
    provider_execution_receipt_sha256: Sha256
    terminal_call_membership_sha256: Sha256
    terminal_call_count: Annotated[int, Field(ge=1)]
    completed_extraction_count: Annotated[int, Field(ge=0)]
    failed_or_ambiguous_count: Annotated[int, Field(ge=0)]
    grounding_receipt_count: Annotated[int, Field(ge=0)]
    estimable_fragment_count: Annotated[int, Field(ge=0)]
    non_estimable_fragment_count: Annotated[int, Field(ge=0)]
    typed_corpus_sha256: Sha256
    grounding_validation_sha256: Sha256
    cohort_reconciliation_receipt_sha256: Sha256
    reconciled_graph_sha256: Sha256
    grounding_package_version: Literal["typed-evidence-grounding-package-v4"] = (
        "typed-evidence-grounding-package-v4"
    )
    grounding_package_sha256: Sha256
    grounding_replay_sha256: Sha256
    complete_exact_once_execution_replayed: Literal[True] = True
    complete_source_membership_replayed: Literal[True] = True
    exact_source_grounding_replayed: Literal[True] = True
    v4_source_provenance_input_eligible: Literal[True] = True
    remaining_release_gates_external: Literal[True] = True
    extraction_accuracy_benchmark_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> HostedNativeGroundingBridgeReceiptV1:
        if self.terminal_call_count != (
            self.completed_extraction_count + self.failed_or_ambiguous_count
        ):
            raise ValueError("hosted_native_bridge_terminal_count_mismatch")
        if self.terminal_call_count != (
            self.estimable_fragment_count + self.non_estimable_fragment_count
        ):
            raise ValueError("hosted_native_bridge_fragment_count_mismatch")
        expected = hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"}))
        if self.receipt_sha256 != expected:
            raise ValueError("hosted_native_bridge_receipt_hash_mismatch")
        return self


@dataclass(frozen=True)
class HostedNativeGroundingBridgeOutputV1:
    run: HostedNativeExtractionRunV1
    pipeline_verification: PipelineFingerprintVerification
    extraction_context: NativeExtractionExecutionContext
    fragments: tuple[PublicationEvidenceFragment, ...]
    grounding_receipts: tuple[NativeGroundingReceipt, ...]
    corpus: TypedEvidenceCorpus
    package: TypedEvidenceGroundingPackage
    receipt: HostedNativeGroundingBridgeReceiptV1


def _checked_template(root: Path, relative: str, expected_sha256: str) -> None:
    current = root
    for part in relative.split("/"):
        current /= part
        if current.is_symlink():
            raise HostedNativeGroundingBridgeError(
                "hosted_native_prompt_template_symlink_forbidden"
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise HostedNativeGroundingBridgeError("hosted_native_prompt_template_missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HostedNativeGroundingBridgeError("hosted_native_prompt_template_invalid")
    if sha256_file(resolved) != expected_sha256:
        raise HostedNativeGroundingBridgeError("hosted_native_prompt_template_hash_mismatch")


def validate_hosted_native_extraction_run_v1(
    *,
    run: HostedNativeExtractionRunV1 | dict[str, Any],
    repository_root: Path,
) -> tuple[HostedNativeExtractionRunV1, PipelineFingerprintVerification]:
    """Reparse and externally replay a future production hosted run."""

    try:
        validated = HostedNativeExtractionRunV1.model_validate(
            run.model_dump(mode="json") if isinstance(run, HostedNativeExtractionRunV1) else run
        )
        candidate_root = Path(os.path.abspath(repository_root))
        if candidate_root.is_symlink():
            raise HostedNativeGroundingBridgeError("hosted_native_repository_root_invalid")
        root = candidate_root.resolve(strict=True)
        if not root.is_dir():
            raise HostedNativeGroundingBridgeError("hosted_native_repository_root_invalid")
        verification = require_pipeline_fingerprint_match(
            expected=validated.pipeline_fingerprint,
            root=root,
        )
        for prompt in validated.prompts:
            _checked_template(root, prompt.template_path, prompt.template_sha256)
        for record in validated.source_manifest.records:
            resolve_native_source_document(
                repository_root=root,
                source_document=record.source_document,
            )
    except (OSError, ValueError, PipelineFingerprintError) as exc:
        if isinstance(exc, HostedNativeGroundingBridgeError):
            raise
        raise HostedNativeGroundingBridgeError(
            f"hosted_native_run_external_replay_failed:{exc}"
        ) from exc
    if verification.computed_pipeline_sha256 != validated.pipeline_fingerprint_sha256:
        raise HostedNativeGroundingBridgeError("hosted_native_pipeline_verification_alias_mismatch")
    return validated, verification


def _execution_context(
    *,
    run: HostedNativeExtractionRunV1,
) -> NativeExtractionExecutionContext:
    provider = run.provider_identity
    receipt = freeze_native_provider_execution_receipt(
        execution_id=run.run_id,
        execution_mode="hosted_exact_once",
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        runtime_id=provider.runtime_id,
        runtime_version=provider.runtime_version,
        runtime_metadata={
            "hosted_run_sha256": run.run_sha256,
            "provider_identity_sha256": run.provider_identity_sha256,
        },
        raw_call_ledger=run.model_dump(mode="json"),
        call_count=len(run.calls),
    )
    prompts = [
        NativeRenderedPromptArtifact(
            prompt_id=item.prompt_id,
            renderer_id=item.renderer_id,
            prompt_version=item.prompt_version,
            template_path=item.template_path,
            template_sha256=item.template_sha256,
            rendered_prompt=item.rendered_prompt,
            rendered_prompt_sha256=item.rendered_prompt_sha256,
        )
        for item in run.prompts
    ]
    schemas = [
        NativeEvaluationSchemaArtifact(
            schema_id=item.schema_id,
            role=item.role,
            schema_payload=item.schema_payload,
            schema_sha256=item.schema_sha256,
        )
        for item in run.schemas
    ]
    artifacts = [
        NativeExtractionArtifactDigest(
            artifact_id="hosted-execution-run",
            role="hosted_execution_run",
            sha256=hash_canonical(run),
            hash_basis="canonical_json",
            execution_ids=[run.run_id],
        ),
        NativeExtractionArtifactDigest(
            artifact_id="pipeline-fingerprint",
            role="pipeline_fingerprint",
            sha256=hash_canonical(run.pipeline_fingerprint),
            hash_basis="canonical_json",
            execution_ids=[run.run_id],
        ),
        NativeExtractionArtifactDigest(
            artifact_id="provider-execution-receipt",
            role="provider_execution_receipt",
            sha256=hash_canonical(receipt),
            hash_basis="canonical_json",
            execution_ids=[run.run_id],
        ),
        NativeExtractionArtifactDigest(
            artifact_id="source-manifest-input",
            role="source_manifest_input",
            sha256=hash_canonical(run.source_manifest),
            hash_basis="canonical_json",
        ),
    ]
    return freeze_native_extraction_execution_context(
        extraction_mode="hosted_exact_once",
        question_config=run.question_config,
        pipeline_fingerprint_sha256=run.pipeline_fingerprint_sha256,
        rendered_prompts=prompts,
        evaluation_schemas=schemas,
        provider_execution_receipts=[receipt],
        input_artifacts=artifacts,
        source_manifest_content_sha256=run.source_manifest_sha256,
        source_manifest_records=run.source_manifest_records,
        corpus_cutoff=run.corpus_cutoff,
    )


def build_hosted_native_grounding_package_v1(
    *,
    run: HostedNativeExtractionRunV1 | dict[str, Any],
    repository_root: Path,
    reviewer_reconciliation: ReviewerCohortReconciliationArtifact | None = None,
) -> HostedNativeGroundingBridgeOutputV1:
    """Build and immediately replay one standard typed grounding package v4."""

    validated, pipeline_verification = validate_hosted_native_extraction_run_v1(
        run=run,
        repository_root=repository_root,
    )
    context = _execution_context(run=validated)
    records_by_doc = {item.doc_id: item for item in validated.source_manifest.records}
    fragments: list[PublicationEvidenceFragment] = []
    grounding_receipts: list[NativeGroundingReceipt] = []
    for call in validated.calls:
        record = records_by_doc[call.intent.doc_id]
        terminal = call.terminal
        if terminal.outcome == "completed":
            extraction = terminal.parsed_extraction
            assert extraction is not None
            grounding = verify_native_publication_grounding(
                repository_root=repository_root,
                source_document=record.source_document,
                extraction=extraction,
            )
            grounding_receipts.append(grounding)
            fragment = freeze_grounding_checked_publication_fragment(
                extraction=extraction,
                grounding_receipt=grounding,
                question_id=validated.question_config.question_id,
                publication=record.publication,
                pipeline_fingerprint_sha256=validated.pipeline_fingerprint_sha256,
                extraction_context_sha256=context.context_sha256,
                source_document=record.source_document,
            )
        else:
            failure_code = terminal.failure_code
            assert failure_code is not None
            detail = f"hosted_exact_once_terminal:{terminal.outcome}:{failure_code}"
            warning = f"hosted_exact_once:{terminal.outcome}:{failure_code}"
            fragment = freeze_publication_evidence_fragment(
                question_id=validated.question_config.question_id,
                publication_id=record.publication.publication_id,
                paper_id=record.publication.paper_id,
                publication=record.publication,
                pipeline_fingerprint_sha256=validated.pipeline_fingerprint_sha256,
                extraction_context_sha256=context.context_sha256,
                source_document=record.source_document,
                grounding_receipt_sha256=None,
                status=FragmentStatus.NON_ESTIMABLE,
                non_estimability_reason=NonEstimabilityReason.OTHER,
                non_estimability_detail=detail,
                extractor_warnings=[warning],
            )
        fragments.append(fragment)
    corpus = assemble_typed_evidence_corpus(fragments)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=grounding_receipts,
        reviewer_reconciliation=reviewer_reconciliation,
        source_manifest=validated.source_manifest,
        corpus_cutoff=validated.corpus_cutoff,
        extraction_context=context,
    )
    if package.package_version != "typed-evidence-grounding-package-v4":
        raise HostedNativeGroundingBridgeError("hosted_native_bridge_did_not_emit_v4")
    replay = reverify_typed_evidence_grounding_package(
        package=package,
        repository_root=repository_root,
    )
    reconciliation = package.cohort_reconciliation
    assert reconciliation is not None
    assert reconciliation.reconciled_graph_sha256 is not None
    native_provider_receipt = context.provider_execution_receipts[0]
    receipt_payload = {
        "receipt_version": BRIDGE_RECEIPT_VERSION,
        "bridge_version": BRIDGE_VERSION,
        "hosted_run_sha256": validated.run_sha256,
        "hosted_run_artifact_sha256": hash_canonical(validated),
        "provider_identity_sha256": validated.provider_identity_sha256,
        "pipeline_fingerprint_sha256": validated.pipeline_fingerprint_sha256,
        "pipeline_verification_sha256": pipeline_verification.verification_sha256,
        "question_config_sha256": validated.question_config_sha256,
        "source_manifest_sha256": validated.source_manifest_sha256,
        "source_membership_sha256": validated.source_membership_sha256,
        "corpus_cutoff": validated.corpus_cutoff,
        "extraction_context_sha256": context.context_sha256,
        "provider_execution_receipt_sha256": native_provider_receipt.receipt_sha256,
        "terminal_call_membership_sha256": validated.call_membership_sha256,
        "terminal_call_count": len(validated.calls),
        "completed_extraction_count": validated.completed_extraction_count,
        "failed_or_ambiguous_count": validated.failed_or_ambiguous_count,
        "grounding_receipt_count": len(grounding_receipts),
        "estimable_fragment_count": len(corpus.estimable_publication_ids),
        "non_estimable_fragment_count": len(corpus.non_estimable_publication_ids),
        "typed_corpus_sha256": corpus.corpus_sha256,
        "grounding_validation_sha256": package.grounding_validation.validation_sha256,
        "cohort_reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "grounding_package_version": package.package_version,
        "grounding_package_sha256": package.package_sha256,
        "grounding_replay_sha256": replay.replay_sha256,
        "complete_exact_once_execution_replayed": True,
        "complete_source_membership_replayed": True,
        "exact_source_grounding_replayed": True,
        "v4_source_provenance_input_eligible": True,
        "remaining_release_gates_external": True,
        "extraction_accuracy_benchmark_authority": False,
        "scientific_claim_truth_authority": False,
        "claim_release_authority": False,
    }
    bridge_receipt = HostedNativeGroundingBridgeReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    return HostedNativeGroundingBridgeOutputV1(
        run=validated,
        pipeline_verification=pipeline_verification,
        extraction_context=context,
        fragments=tuple(sorted(fragments, key=lambda item: item.publication_id)),
        grounding_receipts=tuple(
            sorted(grounding_receipts, key=lambda item: hash_canonical(item.source_document))
        ),
        corpus=corpus,
        package=package,
        receipt=bridge_receipt,
    )


__all__ = [
    "BRIDGE_RECEIPT_VERSION",
    "BRIDGE_VERSION",
    "HostedNativeGroundingBridgeError",
    "HostedNativeGroundingBridgeOutputV1",
    "HostedNativeGroundingBridgeReceiptV1",
    "build_hosted_native_grounding_package_v1",
    "validate_hosted_native_extraction_run_v1",
]
