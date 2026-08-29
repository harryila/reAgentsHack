"""Offline construction and replay of real corpus-pipeline composition joins.

This module is the artifact-derived boundary between a hosted exact-once native
extraction package and the verifier.  Callers provide files, never lineage hashes:
the package, embedded hosted run, external bridge receipt, source bytes, extraction
pipeline, current verifier pipeline, and code-owned join policy are all replayed
before a composition join is emitted.

The resulting artifact proves only code identity and corpus lineage.  Like the
underlying join contract, it has no extraction-accuracy, calibration, scientific,
or claim-release authority.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.corpus_pipeline_composition import (
    CORPUS_INGRESS_PROJECTION_VERSION,
    CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION,
    PIPELINE_COMPOSITION_CONTEXT_VERSION,
    CorpusIngressProjectionV1,
    CorpusPipelineCompositionJoinV1,
    build_composed_calibration_pipeline_verification_v1,
    build_corpus_ingress_projection_v1,
    build_corpus_pipeline_composition_join_v1,
    validate_corpus_pipeline_composition_join_v1,
)
from literature_multiverse.hosted_native_extraction_contract import (
    HostedNativeExtractionRunV1,
)
from literature_multiverse.hosted_native_grounding_bridge import (
    BRIDGE_RECEIPT_VERSION,
    BRIDGE_VERSION,
    HostedNativeGroundingBridgeReceiptV1,
    validate_hosted_native_extraction_run_v1,
)
from literature_multiverse.lineage import hash_canonical, sha256_bytes
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import (
    NativeGroundingReplayVerification,
    TypedEvidenceGroundingPackage,
    reverify_typed_evidence_grounding_package,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
    validate_pipeline_verification_integrity,
)

EXTERNAL_REPLAY_RECEIPT_VERSION = "corpus-pipeline-composition-external-replay-receipt-v1"
JOIN_POLICY_COMPONENT_ID = "corpus-pipeline-composition-policy"
JOIN_POLICY_COMPONENT_VERSION = "1"
HOSTED_INGRESS_INTERFACE = "hosted-exact-once-native-grounding-v4"

JOIN_POLICY_FILE_PATHS = (
    "scripts/build_corpus_pipeline_composition.py",
    "src/literature_multiverse/corpus_pipeline_composition.py",
    "src/literature_multiverse/corpus_pipeline_composition_runtime.py",
    "src/literature_multiverse/pipeline_fingerprint.py",
)

# These are deliberately literal code-owned settings.  They make semantic policy
# drift observable even if a future edit happens not to change file membership.
JOIN_POLICY_SETTINGS: dict[str, Any] = {
    "bridge_alias_validation": "external-receipt-all-fields",
    "bridge_receipt_contract": BRIDGE_RECEIPT_VERSION,
    "bridge_runtime_contract": BRIDGE_VERSION,
    "calibration_pipeline_identity": "extraction-plus-verifier-core-plus-join-policy",
    "caller_supplied_lineage_hashes": False,
    "corpus_ingress_contract": CORPUS_INGRESS_PROJECTION_VERSION,
    "corpus_ingress_interface": HOSTED_INGRESS_INTERFACE,
    "corpus_pipeline_join_contract": CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION,
    "extraction_pipeline_proof": "embedded-hosted-run-external-replay",
    "grounding_package_contract": "typed-evidence-grounding-package-v4",
    "grounding_replay_required": True,
    "offline_only": True,
    "release_authority": False,
    "pipeline_composition_context_contract": PIPELINE_COMPOSITION_CONTEXT_VERSION,
    "verifier_core_pipeline_proof": "current-code-compute-then-external-replay",
}

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class CorpusPipelineCompositionRuntimeError(ValueError):
    """A real package could not be composed or replayed without trusting aliases."""


class _FrozenContract(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


class CorpusPipelineCompositionExternalReplayReceiptV1(_FrozenContract):
    """Self-hashed evidence that the portable join was externally replayed."""

    receipt_version: Literal["corpus-pipeline-composition-external-replay-receipt-v1"] = (
        EXTERNAL_REPLAY_RECEIPT_VERSION
    )
    external_replay_completed: Literal[True] = True
    ingress_interface: Literal["hosted-exact-once-native-grounding-v4"] = HOSTED_INGRESS_INTERFACE
    grounding_package_file_sha256: Sha256
    hosted_bridge_receipt_file_sha256: Sha256
    corpus_source_sha256: Sha256
    grounding_package_sha256: Sha256
    typed_corpus_sha256: Sha256
    hosted_bridge_receipt_sha256: Sha256
    grounding_replay_sha256: Sha256
    effective_graph_sha256: Sha256
    extraction_pipeline_verification_sha256: Sha256
    verifier_core_pipeline_verification_sha256: Sha256
    join_policy_pipeline_verification_sha256: Sha256
    composed_pipeline_fingerprint: PipelineFingerprint
    composed_pipeline_verification: PipelineFingerprintVerification
    composed_pipeline_sha256: Sha256
    composed_pipeline_verification_sha256: Sha256
    calibration_pipeline_sha256: Sha256
    release_pipeline_sha256: Sha256
    composition_join: CorpusPipelineCompositionJoinV1
    composition_join_sha256: Sha256
    corpus_ingress_projection_sha256: Sha256
    public_corpus_loader_match_completed: Literal[True] = True
    scientific_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> CorpusPipelineCompositionExternalReplayReceiptV1:
        try:
            join = validate_corpus_pipeline_composition_join_v1(self.composition_join)
        except ValueError as exc:
            raise ValueError("composition_runtime_join_invalid") from exc
        if join.external_replay_required is not True or join.external_replay_completed is not False:
            raise ValueError("composition_runtime_bare_join_replay_state_invalid")
        aliases = {
            "composition_join_sha256": join.join_sha256,
            "corpus_ingress_projection_sha256": (
                join.corpus_ingress_projection_sha256
            ),
            "corpus_source_sha256": join.corpus_ingress.corpus_source_sha256,
            "grounding_package_sha256": join.corpus_ingress.grounding_package_sha256,
            "typed_corpus_sha256": join.corpus_ingress.typed_corpus_sha256,
            "hosted_bridge_receipt_sha256": (join.corpus_ingress.hosted_bridge_receipt_sha256),
            "grounding_replay_sha256": join.corpus_ingress.grounding_replay_sha256,
            "effective_graph_sha256": join.corpus_ingress.effective_graph_sha256,
            "extraction_pipeline_verification_sha256": (
                join.extraction_pipeline_verification.verification_sha256
            ),
            "verifier_core_pipeline_verification_sha256": (
                join.pipeline_composition_context.verifier_core_pipeline_verification.verification_sha256
            ),
            "join_policy_pipeline_verification_sha256": (
                join.pipeline_composition_context.join_policy_pipeline_verification.verification_sha256
            ),
        }
        for field, expected in aliases.items():
            if getattr(self, field) != expected:
                raise ValueError(f"composition_runtime_{field}_alias_mismatch")
        try:
            composed_proof = validate_pipeline_verification_integrity(
                self.composed_pipeline_verification
            )
            composed_fingerprint = PipelineFingerprint.model_validate(
                self.composed_pipeline_fingerprint.model_dump(mode="json")
            )
        except ValueError as exc:
            raise ValueError("composition_runtime_composed_pipeline_invalid") from exc
        if (
            composed_proof.status != "matched"
            or composed_proof.issues
            or composed_proof.computed is None
            or composed_proof.computed != composed_fingerprint
            or composed_proof.expected_pipeline_sha256
            != composed_fingerprint.pipeline_sha256
            or composed_proof.computed_pipeline_sha256
            != composed_fingerprint.pipeline_sha256
            or self.composed_pipeline_sha256
            != composed_fingerprint.pipeline_sha256
            or self.composed_pipeline_verification_sha256
            != composed_proof.verification_sha256
            or self.calibration_pipeline_sha256
            != composed_fingerprint.pipeline_sha256
            or self.release_pipeline_sha256
            != composed_fingerprint.pipeline_sha256
        ):
            raise ValueError("composition_runtime_composed_pipeline_alias_mismatch")
        frozen_components = [
            component
            for component in composed_fingerprint.components
            if component.component_id == "frozen-corpus-extraction"
        ]
        if len(frozen_components) != 1:
            raise ValueError("composition_runtime_frozen_extraction_component_missing")
        settings = frozen_components[0].settings
        composition = join.pipeline_composition_context
        expected_settings = {
            "composition_contract": composition.composition_version,
            "composition_context_sha256": composition.composition_context_sha256,
            "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
            "extraction_pipeline_verification_sha256": (
                join.extraction_pipeline_verification.verification_sha256
            ),
            "join_policy_pipeline_sha256": join.join_policy_pipeline_sha256,
            "join_policy_pipeline_verification_sha256": (
                composition.join_policy_pipeline_verification.verification_sha256
            ),
            "native_ingress_interface": self.ingress_interface,
            "per_corpus_join_excluded_from_calibration_identity": True,
            "verifier_core_pipeline_sha256": join.verifier_core_pipeline_sha256,
            "verifier_core_pipeline_verification_sha256": (
                composition.verifier_core_pipeline_verification.verification_sha256
            ),
        }
        if settings != expected_settings:
            raise ValueError("composition_runtime_frozen_extraction_settings_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("composition_runtime_receipt_hash_mismatch")
        return self


def join_policy_pipeline_components() -> tuple[PipelineComponentSpec, ...]:
    """Return the exact code-owned manifest for composition policy v1."""

    return (
        PipelineComponentSpec(
            component_id=JOIN_POLICY_COMPONENT_ID,
            component_version=JOIN_POLICY_COMPONENT_VERSION,
            file_paths=list(JOIN_POLICY_FILE_PATHS),
            settings=dict(JOIN_POLICY_SETTINGS),
        ),
    )


def _repository_root(value: Path) -> Path:
    candidate = Path(os.path.abspath(value))
    if candidate.is_symlink():
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_repository_root_symlink_forbidden"
        )
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_repository_root_missing"
        ) from exc
    if not root.is_dir():
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_repository_root_not_directory"
        )
    return root


def _json_object(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise CorpusPipelineCompositionRuntimeError(f"composition_runtime_{role}_file_invalid")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusPipelineCompositionRuntimeError(
            f"composition_runtime_{role}_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise CorpusPipelineCompositionRuntimeError(f"composition_runtime_{role}_not_object")
    return value, sha256_bytes(raw)


def _matched_join_policy_proof(root: Path) -> PipelineFingerprintVerification:
    components = join_policy_pipeline_components()
    frozen = compute_pipeline_fingerprint(root=root, components=list(components))
    return require_pipeline_fingerprint_match(
        expected=frozen,
        root=root,
        current_components=components,
    )


def _matched_verifier_core_proof(
    root: Path,
) -> tuple[PipelineFingerprintVerification, tuple[PipelineComponentSpec, ...]]:
    # Local import prevents an import cycle when the public verifier later dispatches
    # to this additive runtime.
    from literature_multiverse.verifier import (
        compute_verifier_pipeline_fingerprint,
        verifier_pipeline_components,
    )

    components = verifier_pipeline_components()
    frozen = compute_verifier_pipeline_fingerprint(root=root)
    return (
        require_pipeline_fingerprint_match(
            expected=frozen,
            root=root,
            current_components=components,
        ),
        components,
    )


def _terminal_fragment_membership_sha256(
    package: TypedEvidenceGroundingPackage,
) -> str:
    # This is intentionally byte-for-byte the public verifier loader convention.
    membership = [
        {
            "fragment_sha256": fragment.fragment_sha256,
            "paper_id": fragment.paper_id,
            "publication_id": fragment.publication_id,
            "status": fragment.status.value,
        }
        for fragment in package.corpus.fragments
    ]
    return hash_canonical(membership)


def _require_ingress_matches_corpus_load_result(
    *,
    ingress: CorpusIngressProjectionV1,
    corpus: Any,
) -> None:
    """Compare an ingress projection with the exact public-loader object in use."""

    metadata = getattr(corpus, "metadata", None)
    extraction_context = getattr(corpus, "extraction_context", None)
    graph = getattr(corpus, "graph", None)
    if (
        not isinstance(metadata, dict)
        or extraction_context is None
        or graph is None
        or getattr(corpus, "source_format", None)
        != "typed_evidence_grounding_package_json"
    ):
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_loader_shape_invalid"
        )
    provider_receipts = extraction_context.provider_execution_receipts
    if len(provider_receipts) != 1:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_provider_receipt_invalid"
        )
    try:
        run = HostedNativeExtractionRunV1.model_validate(
            provider_receipts[0].raw_call_ledger
        )
    except ValueError as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_hosted_run_invalid"
        ) from exc
    native_manifest = metadata.get("native_source_manifest")
    if not isinstance(native_manifest, dict) or not isinstance(
        native_manifest.get("records"), list
    ):
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_manifest_invalid"
        )
    observed = {
        "corpus_id": getattr(corpus, "corpus_id", None),
        "question_id": run.question_config.question_id,
        "corpus_cutoff": metadata.get("native_corpus_cutoff"),
        "corpus_source_sha256": getattr(corpus, "source_sha256", None),
        "grounding_package_sha256": metadata.get("grounding_package_sha256"),
        "typed_corpus_sha256": metadata.get("typed_evidence_corpus_sha256"),
        "extraction_pipeline_sha256": metadata.get("pipeline_fingerprint_sha256"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "source_membership_sha256": hash_canonical(native_manifest["records"]),
        "question_config_sha256": metadata.get("question_config_sha256"),
        "extraction_context_sha256": metadata.get("extraction_context_sha256"),
        "extraction_context_receipt_sha256": metadata.get(
            "extraction_context_receipt_sha256"
        ),
        "hosted_run_sha256": run.run_sha256,
        "terminal_call_membership_sha256": run.call_membership_sha256,
        "terminal_fragment_membership_sha256": metadata.get(
            "terminal_fragment_membership_sha256"
        ),
        "grounding_validation_sha256": metadata.get(
            "grounding_validation_sha256"
        ),
        "grounding_replay_sha256": metadata.get("grounding_replay_sha256"),
        "cohort_reconciliation_receipt_sha256": metadata.get(
            "cohort_reconciliation_receipt_sha256"
        ),
        "reconciled_graph_sha256": metadata.get("reconciled_graph_sha256"),
        "effective_graph_sha256": hash_canonical(graph),
    }
    for field, value in observed.items():
        if getattr(ingress, field) != value:
            raise CorpusPipelineCompositionRuntimeError(
                f"composition_runtime_public_corpus_{field}_mismatch"
            )
    try:
        release_eligible = corpus.provenance_release_eligible()
    except (AttributeError, TypeError) as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_provenance_invalid"
        ) from exc
    if release_eligible is not True:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_public_corpus_provenance_ineligible"
        )


def require_external_replay_receipt_matches_corpus_load_result_v1(
    *,
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
    corpus: Any,
) -> CorpusPipelineCompositionExternalReplayReceiptV1:
    """Mandatory downstream comparator for the actual loaded verifier corpus."""

    try:
        validated = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
            receipt.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_receipt_contract_invalid"
        ) from exc
    _require_ingress_matches_corpus_load_result(
        ingress=validated.composition_join.corpus_ingress,
        corpus=corpus,
    )
    return validated


_BRIDGE_FIXED_CONTRACT_FIELDS = {
    "bridge_version",
    "claim_release_authority",
    "complete_exact_once_execution_replayed",
    "complete_source_membership_replayed",
    "exact_source_grounding_replayed",
    "extraction_accuracy_benchmark_authority",
    "receipt_sha256",
    "receipt_version",
    "remaining_release_gates_external",
    "scientific_claim_truth_authority",
    "v4_source_provenance_input_eligible",
}


def _validate_bridge_aliases(
    *,
    bridge: HostedNativeGroundingBridgeReceiptV1,
    run: HostedNativeExtractionRunV1,
    package: TypedEvidenceGroundingPackage,
    replay: NativeGroundingReplayVerification,
    extraction_proof: PipelineFingerprintVerification,
) -> None:
    context_receipt = package.extraction_context_receipt
    reconciliation = package.cohort_reconciliation
    if context_receipt is None or reconciliation is None:
        raise CorpusPipelineCompositionRuntimeError("composition_runtime_v4_lineage_incomplete")
    context = context_receipt.execution_context
    if len(context.provider_execution_receipts) != 1:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_hosted_provider_receipt_cardinality_invalid"
        )
    provider_receipt = context.provider_execution_receipts[0]
    expected: dict[str, Any] = {
        "hosted_run_sha256": run.run_sha256,
        "hosted_run_artifact_sha256": hash_canonical(run),
        "provider_identity_sha256": run.provider_identity_sha256,
        "pipeline_fingerprint_sha256": run.pipeline_fingerprint_sha256,
        "pipeline_verification_sha256": extraction_proof.verification_sha256,
        "question_config_sha256": run.question_config_sha256,
        "source_manifest_sha256": run.source_manifest_sha256,
        "source_membership_sha256": run.source_membership_sha256,
        "corpus_cutoff": run.corpus_cutoff,
        "extraction_context_sha256": context.context_sha256,
        "provider_execution_receipt_sha256": provider_receipt.receipt_sha256,
        "terminal_call_membership_sha256": run.call_membership_sha256,
        "terminal_call_count": len(run.calls),
        "completed_extraction_count": run.completed_extraction_count,
        "failed_or_ambiguous_count": run.failed_or_ambiguous_count,
        "grounding_receipt_count": len(package.grounding_receipts),
        "estimable_fragment_count": len(package.corpus.estimable_publication_ids),
        "non_estimable_fragment_count": len(package.corpus.non_estimable_publication_ids),
        "typed_corpus_sha256": package.corpus.corpus_sha256,
        "grounding_validation_sha256": (package.grounding_validation.validation_sha256),
        "cohort_reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "grounding_package_version": package.package_version,
        "grounding_package_sha256": package.package_sha256,
        "grounding_replay_sha256": replay.replay_sha256,
    }
    model_fields = set(HostedNativeGroundingBridgeReceiptV1.model_fields)
    covered_fields = set(expected) | _BRIDGE_FIXED_CONTRACT_FIELDS
    if model_fields != covered_fields:
        missing = sorted(model_fields - covered_fields)
        extra = sorted(covered_fields - model_fields)
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_bridge_alias_schema_uncovered:"
            f"missing={','.join(missing)}:extra={','.join(extra)}"
        )
    for field, observed in expected.items():
        if getattr(bridge, field) != observed:
            raise CorpusPipelineCompositionRuntimeError(
                f"composition_runtime_bridge_{field}_mismatch"
            )


def _build_artifact_payload(
    *,
    repository_root: Path,
    grounding_package_path: Path,
    hosted_bridge_receipt_path: Path,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    companion_papers = grounding_package_path.parent / "papers.parquet"
    if companion_papers.is_symlink() or companion_papers.is_file():
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_companion_papers_parquet_unsupported"
        )
    package_raw, package_file_sha256 = _json_object(
        grounding_package_path,
        role="grounding_package",
    )
    bridge_raw, bridge_file_sha256 = _json_object(
        hosted_bridge_receipt_path,
        role="hosted_bridge_receipt",
    )
    try:
        package = TypedEvidenceGroundingPackage.model_validate(package_raw)
        bridge = HostedNativeGroundingBridgeReceiptV1.model_validate(bridge_raw)
    except ValueError as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_input_contract_invalid"
        ) from exc
    if package.package_version != "typed-evidence-grounding-package-v4":
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_requires_grounding_package_v4"
        )
    context_receipt = package.extraction_context_receipt
    reconciliation = package.cohort_reconciliation
    if context_receipt is None or reconciliation is None:
        raise CorpusPipelineCompositionRuntimeError("composition_runtime_v4_lineage_incomplete")
    context = context_receipt.execution_context
    if context.extraction_mode != "hosted_exact_once":
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_requires_hosted_exact_once"
        )
    if len(context.provider_execution_receipts) != 1:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_hosted_provider_receipt_cardinality_invalid"
        )
    try:
        replay = reverify_typed_evidence_grounding_package(
            package=package,
            repository_root=root,
        )
        run, extraction_proof = validate_hosted_native_extraction_run_v1(
            run=context.provider_execution_receipts[0].raw_call_ledger,
            repository_root=root,
        )
        extraction_proof = validate_pipeline_verification_integrity(extraction_proof)
    except ValueError as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_package_external_replay_failed"
        ) from exc
    _validate_bridge_aliases(
        bridge=bridge,
        run=run,
        package=package,
        replay=replay,
        extraction_proof=extraction_proof,
    )
    if reconciliation.reconciled_graph_sha256 is None:
        raise CorpusPipelineCompositionRuntimeError("composition_runtime_reconciled_graph_missing")
    verifier_core_proof, verifier_core_components = _matched_verifier_core_proof(
        root
    )
    join_policy_proof = _matched_join_policy_proof(root)
    ingress = build_corpus_ingress_projection_v1(
        ingress_interface=HOSTED_INGRESS_INTERFACE,
        corpus_id=package.corpus.question_id,
        question_id=package.corpus.question_id,
        corpus_cutoff=context.corpus_cutoff,
        corpus_source_sha256=hash_canonical({"evidence": package_file_sha256}),
        grounding_package_sha256=package.package_sha256,
        typed_corpus_sha256=package.corpus.corpus_sha256,
        extraction_pipeline_sha256=run.pipeline_fingerprint_sha256,
        extraction_pipeline_verification_sha256=(extraction_proof.verification_sha256),
        source_manifest_sha256=run.source_manifest_sha256,
        source_membership_sha256=run.source_membership_sha256,
        question_config_sha256=run.question_config_sha256,
        extraction_context_sha256=context.context_sha256,
        extraction_context_receipt_sha256=context_receipt.receipt_sha256,
        hosted_run_sha256=run.run_sha256,
        terminal_call_membership_sha256=run.call_membership_sha256,
        terminal_fragment_membership_sha256=(_terminal_fragment_membership_sha256(package)),
        grounding_validation_sha256=package.grounding_validation.validation_sha256,
        grounding_replay_sha256=replay.replay_sha256,
        cohort_reconciliation_receipt_sha256=reconciliation.receipt_sha256,
        reconciled_graph_sha256=reconciliation.reconciled_graph_sha256,
        effective_graph_sha256=reconciliation.reconciled_graph_sha256,
        hosted_bridge_receipt_sha256=bridge.receipt_sha256,
    )
    # The composition receipt is checked against the same loader object the public
    # verifier consumes.  A sibling papers.parquet was rejected above so the loader
    # source digest cannot depend on whether a caller passed a file or directory.
    from literature_multiverse.verifier import LegacyAdapterConfig, load_corpus

    loaded_corpus = load_corpus(
        grounding_package_path,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=root,
    )
    _require_ingress_matches_corpus_load_result(
        ingress=ingress,
        corpus=loaded_corpus,
    )
    join = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=extraction_proof,
        verifier_core_pipeline_verification=verifier_core_proof,
        join_policy_pipeline_verification=join_policy_proof,
        corpus_ingress=ingress,
    )
    composed_pipeline_proof = (
        build_composed_calibration_pipeline_verification_v1(
            repository_root=root,
            verifier_core_components=verifier_core_components,
            join_policy_components=join_policy_pipeline_components(),
            ingress_interface=HOSTED_INGRESS_INTERFACE,
            extraction_pipeline_verification=extraction_proof,
            verifier_core_pipeline_verification=verifier_core_proof,
            join_policy_pipeline_verification=join_policy_proof,
        )
    )
    composed_pipeline = composed_pipeline_proof.computed
    if composed_pipeline is None:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_composed_pipeline_missing"
        )
    return {
        "receipt_version": EXTERNAL_REPLAY_RECEIPT_VERSION,
        "external_replay_completed": True,
        "ingress_interface": HOSTED_INGRESS_INTERFACE,
        "grounding_package_file_sha256": package_file_sha256,
        "hosted_bridge_receipt_file_sha256": bridge_file_sha256,
        "corpus_source_sha256": ingress.corpus_source_sha256,
        "grounding_package_sha256": package.package_sha256,
        "typed_corpus_sha256": package.corpus.corpus_sha256,
        "hosted_bridge_receipt_sha256": bridge.receipt_sha256,
        "grounding_replay_sha256": replay.replay_sha256,
        "effective_graph_sha256": ingress.effective_graph_sha256,
        "extraction_pipeline_verification_sha256": (extraction_proof.verification_sha256),
        "verifier_core_pipeline_verification_sha256": (verifier_core_proof.verification_sha256),
        "join_policy_pipeline_verification_sha256": (join_policy_proof.verification_sha256),
        "composed_pipeline_fingerprint": composed_pipeline,
        "composed_pipeline_verification": composed_pipeline_proof,
        "composed_pipeline_sha256": composed_pipeline.pipeline_sha256,
        "composed_pipeline_verification_sha256": (
            composed_pipeline_proof.verification_sha256
        ),
        "calibration_pipeline_sha256": composed_pipeline.pipeline_sha256,
        "release_pipeline_sha256": composed_pipeline.pipeline_sha256,
        "composition_join": join,
        "composition_join_sha256": join.join_sha256,
        "corpus_ingress_projection_sha256": ingress.ingress_projection_sha256,
        "public_corpus_loader_match_completed": True,
        "scientific_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }


def build_corpus_pipeline_composition_external_replay_receipt_v1(
    *,
    repository_root: Path,
    grounding_package_path: Path,
    hosted_bridge_receipt_path: Path,
) -> CorpusPipelineCompositionExternalReplayReceiptV1:
    """Replay real artifacts and current code, then build one closed join artifact."""

    try:
        payload = _build_artifact_payload(
            repository_root=repository_root,
            grounding_package_path=grounding_package_path,
            hosted_bridge_receipt_path=hosted_bridge_receipt_path,
        )
        return CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
            {**payload, "receipt_sha256": hash_canonical(payload)}
        )
    except CorpusPipelineCompositionRuntimeError:
        raise
    except ValueError as exc:
        raise CorpusPipelineCompositionRuntimeError("composition_runtime_build_failed") from exc


def load_corpus_pipeline_composition_external_replay_receipt_v1(
    path: Path,
) -> CorpusPipelineCompositionExternalReplayReceiptV1:
    """Load and internally validate one persisted runtime artifact."""

    raw, _ = _json_object(path, role="composition_artifact")
    try:
        return CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(raw)
    except ValueError as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_artifact_contract_invalid"
        ) from exc


def validate_corpus_pipeline_composition_external_replay_receipt_v1(
    *,
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
    repository_root: Path,
    grounding_package_path: Path,
    hosted_bridge_receipt_path: Path,
) -> CorpusPipelineCompositionExternalReplayReceiptV1:
    """Rebuild from source artifacts/current code and require exact equality."""

    try:
        validated = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
            receipt.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_artifact_contract_invalid"
        ) from exc
    rebuilt = build_corpus_pipeline_composition_external_replay_receipt_v1(
        repository_root=repository_root,
        grounding_package_path=grounding_package_path,
        hosted_bridge_receipt_path=hosted_bridge_receipt_path,
    )
    if validated.model_dump(mode="json") != rebuilt.model_dump(mode="json"):
        raise CorpusPipelineCompositionRuntimeError(
            "composition_runtime_artifact_external_replay_mismatch"
        )
    return validated


__all__ = [
    "EXTERNAL_REPLAY_RECEIPT_VERSION",
    "HOSTED_INGRESS_INTERFACE",
    "JOIN_POLICY_COMPONENT_ID",
    "JOIN_POLICY_COMPONENT_VERSION",
    "JOIN_POLICY_FILE_PATHS",
    "JOIN_POLICY_SETTINGS",
    "CorpusPipelineCompositionExternalReplayReceiptV1",
    "CorpusPipelineCompositionRuntimeError",
    "build_corpus_pipeline_composition_external_replay_receipt_v1",
    "join_policy_pipeline_components",
    "load_corpus_pipeline_composition_external_replay_receipt_v1",
    "validate_corpus_pipeline_composition_external_replay_receipt_v1",
]
