"""V3 join from byte-replayed source authorization to existing synthesis mechanics."""

from __future__ import annotations

import platform
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import (
    TypedEvidenceGroundingPackage,
    resolve_native_source_document,
    reverify_typed_evidence_grounding_package,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.synthesis_unit_authorization import (
    SynthesisUnitAuthorizationReceiptV1,
    reverify_synthesis_unit_authorization_v1,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


class SourceAuthorizedSynthesisV3Error(ValueError):
    pass


_PIPELINE_FILES = [
    "pyproject.toml",
    "scripts/run_source_authorized_synthesis_v3.py",
    "scripts/validate_source_authorized_synthesis_v3.py",
    "uv.lock",
    "src/literature_multiverse/cohort_reconciliation.py",
    "src/literature_multiverse/evidence_graph.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/meta_analysis.py",
    "src/literature_multiverse/native_grounding.py",
    "src/literature_multiverse/pipeline_fingerprint.py",
    "src/literature_multiverse/source_authorized_synthesis_v3.py",
    "src/literature_multiverse/synthesis_unit_authorization.py",
    "src/literature_multiverse/typed_extraction.py",
]


def compute_source_authorized_synthesis_v3_fingerprint(
    *,
    root: Path,
) -> PipelineFingerprint:
    return compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="source-authorized-synthesis-v3",
                component_version="1",
                file_paths=sorted(_PIPELINE_FILES),
                settings={
                    "source_authorization_scope": "independence_and_input_only",
                    "reference_labels_accessed": False,
                    "review_conclusions_accessed": False,
                    "python_version": platform.python_version(),
                    "numpy_version": distribution_version("numpy"),
                    "scipy_version": distribution_version("scipy"),
                    "pydantic_version": distribution_version("pydantic"),
                },
            )
        ],
    )


class SourceAuthorizedSynthesisUnitV3(ContractModel):
    unit_version: Literal["source-authorized-synthesis-unit-v3"] = (
        "source-authorized-synthesis-unit-v3"
    )
    authorization_receipt_sha256: str
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    status: Literal["synthesized", "abstained_unresolved_independence"]
    synthesis: dict[str, Any] | None
    synthesis_sha256: str | None
    unresolved_overlap_pairs: list[list[str]]
    unit_sha256: str

    @field_validator("authorization_receipt_sha256", "synthesis_sha256", "unit_sha256")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("source_authorized_v3_hash_invalid")
        return value

    @field_validator("estimate_ids")
    @classmethod
    def validate_estimate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("source_authorized_v3_unit_estimates_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_unit(self) -> SourceAuthorizedSynthesisUnitV3:
        synthesized = self.status == "synthesized"
        if synthesized != (self.synthesis is not None and self.synthesis_sha256 is not None):
            raise ValueError("source_authorized_v3_synthesis_presence_mismatch")
        if self.synthesis is not None and hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("source_authorized_v3_synthesis_hash_mismatch")
        if synthesized == bool(self.unresolved_overlap_pairs):
            raise ValueError("source_authorized_v3_unresolved_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"unit_sha256"})
        if hash_canonical(payload) != self.unit_sha256:
            raise ValueError("source_authorized_v3_unit_hash_mismatch")
        return self


class SourceAuthorizedSynthesisReportV3(ContractModel):
    report_version: Literal["source-authorized-synthesis-report-v3"] = (
        "source-authorized-synthesis-report-v3"
    )
    scientific_role: Literal["source_authorized_synthesis_yield_only"] = (
        "source_authorized_synthesis_yield_only"
    )
    public_private_role: Literal["private_source_bound"] = "private_source_bound"
    input_corpus_sha256: str
    grounding_package_sha256: str
    reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    requested_estimate_ids: Annotated[list[str], Field(min_length=1)]
    authorization_receipt_sha256s: Annotated[list[str], Field(min_length=1)]
    authorized_receipt_sha256s: list[str]
    source_map: dict[str, str]
    pipeline_fingerprint: PipelineFingerprint
    runtime: dict[str, str]
    reference_labels_accessed: Literal[False] = False
    review_conclusions_accessed: Literal[False] = False
    supports_accuracy_claim: Literal[False] = False
    supports_release_claim: Literal[False] = False
    verifier_replay_status: Literal["not_constructed_missing_claim_calibration_and_audit_contract"]
    units: Annotated[list[SourceAuthorizedSynthesisUnitV3], Field(min_length=1)]
    report_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "grounding_package_sha256",
        "reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("source_authorized_v3_report_hash_invalid")
        return value

    @field_validator("source_map")
    @classmethod
    def validate_source_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or list(value) != sorted(value):
            raise ValueError("source_authorized_v3_source_map_not_sorted")
        if any(not key or not SHA256_RE.fullmatch(digest) for key, digest in value.items()):
            raise ValueError("source_authorized_v3_source_map_invalid")
        return value

    @field_validator(
        "requested_estimate_ids",
        "authorization_receipt_sha256s",
        "authorized_receipt_sha256s",
    )
    @classmethod
    def validate_sorted_unique_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError(f"source_authorized_v3_not_sorted_unique:{info.field_name}")
        if "receipt" in info.field_name and any(not SHA256_RE.fullmatch(item) for item in value):
            raise ValueError(f"source_authorized_v3_invalid_hash:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> SourceAuthorizedSynthesisReportV3:
        if set(self.runtime) != {"python", "numpy", "scipy", "pydantic"} or any(
            not value.strip() for value in self.runtime.values()
        ):
            raise ValueError("source_authorized_v3_runtime_invalid")
        receipts = set(self.authorization_receipt_sha256s)
        authorized = set(self.authorized_receipt_sha256s)
        if not authorized <= receipts:
            raise ValueError("source_authorized_v3_authorized_not_subset")
        unit_receipts = [unit.authorization_receipt_sha256 for unit in self.units]
        if unit_receipts != sorted(set(unit_receipts)):
            raise ValueError("source_authorized_v3_units_not_sorted_unique")
        if set(unit_receipts) != receipts:
            raise ValueError("source_authorized_v3_units_receipt_coverage_mismatch")
        synthesized = {
            unit.authorization_receipt_sha256 for unit in self.units if unit.status == "synthesized"
        }
        if synthesized != authorized:
            raise ValueError("source_authorized_v3_synthesized_authorized_mismatch")
        memberships = [item for unit in self.units for item in unit.estimate_ids]
        if len(memberships) != len(set(memberships)):
            raise ValueError("source_authorized_v3_unit_estimate_overlap")
        if sorted(memberships) != self.requested_estimate_ids:
            raise ValueError("source_authorized_v3_unit_estimate_coverage_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("source_authorized_v3_report_hash_mismatch")
        return self


def _freeze_unit(payload: dict[str, Any]) -> SourceAuthorizedSynthesisUnitV3:
    return SourceAuthorizedSynthesisUnitV3.model_validate(
        {**payload, "unit_sha256": hash_canonical(payload)}
    )


def run_source_authorized_synthesis_v3(
    *,
    corpus: TypedEvidenceCorpus,
    grounding_package: TypedEvidenceGroundingPackage,
    reconciliation: NativeCohortReconciliationReceipt,
    receipts: list[SynthesisUnitAuthorizationReceiptV1],
    requested_estimate_ids: list[str],
    authorized_receipt_sha256s: list[str],
    repository_root: Path,
    pipeline_root: Path | None = None,
) -> SourceAuthorizedSynthesisReportV3:
    corpus = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    grounding_package = TypedEvidenceGroundingPackage.model_validate(
        grounding_package.model_dump(mode="json")
    )
    reverify_typed_evidence_grounding_package(
        package=grounding_package, repository_root=repository_root
    )
    if grounding_package.corpus != corpus:
        raise SourceAuthorizedSynthesisV3Error("v3_grounding_package_corpus_mismatch")
    if grounding_package.cohort_reconciliation != reconciliation:
        raise SourceAuthorizedSynthesisV3Error("v3_grounding_package_reconciliation_mismatch")
    requested = sorted(set(requested_estimate_ids))
    if requested != requested_estimate_ids or not requested:
        raise SourceAuthorizedSynthesisV3Error("v3_requested_estimates_not_sorted_unique")
    receipt_hashes = [item.receipt_sha256 for item in receipts]
    if receipt_hashes != sorted(set(receipt_hashes)):
        raise SourceAuthorizedSynthesisV3Error("v3_authorization_receipts_not_sorted_unique")
    authorized_hashes = sorted(set(authorized_receipt_sha256s))
    if authorized_hashes != authorized_receipt_sha256s:
        raise SourceAuthorizedSynthesisV3Error("v3_authorized_hashes_not_sorted_unique")
    if not set(authorized_hashes) <= set(receipt_hashes):
        raise SourceAuthorizedSynthesisV3Error("v3_authorized_receipt_unknown")

    replayed = [
        reverify_synthesis_unit_authorization_v1(
            corpus=corpus,
            reconciliation=reconciliation,
            receipt=receipt,
            repository_root=repository_root,
        )
        for receipt in receipts
    ]
    for receipt in replayed:
        declared = receipt.receipt_sha256 in authorized_hashes
        if declared != receipt.authorizes_synthesis_input:
            raise SourceAuthorizedSynthesisV3Error(
                "v3_authorized_set_disagrees_with_receipt_outcome"
            )
        if receipt.reference_labels_accessed or receipt.review_conclusions_accessed:
            raise SourceAuthorizedSynthesisV3Error("v3_label_or_conclusion_access_forbidden")

    membership = [item for receipt in replayed for item in receipt.estimate_ids]
    if len(membership) != len(set(membership)):
        raise SourceAuthorizedSynthesisV3Error("v3_overlapping_estimate_membership")
    if sorted(membership) != requested:
        raise SourceAuthorizedSynthesisV3Error("v3_requested_estimate_coverage_mismatch")
    graph = reconciliation.reconciled_graph
    if graph is None or reconciliation.reconciled_graph_sha256 is None:
        raise SourceAuthorizedSynthesisV3Error("v3_reconciled_graph_missing")
    graph_estimates = {item.estimate_id: item for item in graph.outcome_estimates}
    if any(item not in graph_estimates for item in requested):
        raise SourceAuthorizedSynthesisV3Error("v3_requested_estimate_unknown")

    units: list[SourceAuthorizedSynthesisUnitV3] = []
    for receipt in replayed:
        if not receipt.authorizes_synthesis_input:
            units.append(
                _freeze_unit(
                    {
                        "unit_version": "source-authorized-synthesis-unit-v3",
                        "authorization_receipt_sha256": receipt.receipt_sha256,
                        "estimate_ids": receipt.estimate_ids,
                        "status": "abstained_unresolved_independence",
                        "synthesis": None,
                        "synthesis_sha256": None,
                        "unresolved_overlap_pairs": receipt.unresolved_overlap_pairs,
                    }
                )
            )
            continue
        selected = [graph_estimates[item] for item in receipt.estimate_ids]
        outcomes = {item.outcome_name for item in selected}
        if len(outcomes) != 1:
            raise SourceAuthorizedSynthesisV3Error("v3_unit_mixes_outcomes")
        selected_graph = EvidenceGraph.model_validate(
            {
                **graph.model_dump(mode="json"),
                "outcome_estimates": selected,
            }
        )
        synthesis = synthesize_evidence_graph(selected_graph, outcome_name=next(iter(outcomes)))
        units.append(
            _freeze_unit(
                {
                    "unit_version": "source-authorized-synthesis-unit-v3",
                    "authorization_receipt_sha256": receipt.receipt_sha256,
                    "estimate_ids": receipt.estimate_ids,
                    "status": "synthesized",
                    "synthesis": synthesis,
                    "synthesis_sha256": hash_canonical(synthesis),
                    "unresolved_overlap_pairs": [],
                }
            )
        )
    units.sort(key=lambda item: item.authorization_receipt_sha256)
    effective_pipeline_root = pipeline_root or Path(__file__).resolve().parents[2]
    fingerprint = compute_source_authorized_synthesis_v3_fingerprint(root=effective_pipeline_root)
    source_map = {
        fragment.publication_id: resolve_native_source_document(
            repository_root=repository_root,
            source_document=fragment.source_document,
        ).source_payload_sha256
        for fragment in corpus.fragments
    }
    payload = {
        "report_version": "source-authorized-synthesis-report-v3",
        "scientific_role": "source_authorized_synthesis_yield_only",
        "public_private_role": "private_source_bound",
        "input_corpus_sha256": corpus.corpus_sha256,
        "grounding_package_sha256": grounding_package.package_sha256,
        "reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "requested_estimate_ids": requested,
        "authorization_receipt_sha256s": receipt_hashes,
        "authorized_receipt_sha256s": authorized_hashes,
        "source_map": dict(sorted(source_map.items())),
        "pipeline_fingerprint": fingerprint,
        "runtime": {
            "python": platform.python_version(),
            "numpy": distribution_version("numpy"),
            "scipy": distribution_version("scipy"),
            "pydantic": distribution_version("pydantic"),
        },
        "reference_labels_accessed": False,
        "review_conclusions_accessed": False,
        "supports_accuracy_claim": False,
        "supports_release_claim": False,
        "verifier_replay_status": ("not_constructed_missing_claim_calibration_and_audit_contract"),
        "units": units,
    }
    return SourceAuthorizedSynthesisReportV3.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def reverify_source_authorized_synthesis_v3_report(
    *,
    report: SourceAuthorizedSynthesisReportV3,
    corpus: TypedEvidenceCorpus,
    grounding_package: TypedEvidenceGroundingPackage,
    reconciliation: NativeCohortReconciliationReceipt,
    receipts: list[SynthesisUnitAuthorizationReceiptV1],
    repository_root: Path,
    pipeline_root: Path | None = None,
) -> SourceAuthorizedSynthesisReportV3:
    frozen = SourceAuthorizedSynthesisReportV3.model_validate(report.model_dump(mode="json"))
    replayed = run_source_authorized_synthesis_v3(
        corpus=corpus,
        grounding_package=grounding_package,
        reconciliation=reconciliation,
        receipts=receipts,
        requested_estimate_ids=frozen.requested_estimate_ids,
        authorized_receipt_sha256s=frozen.authorized_receipt_sha256s,
        repository_root=repository_root,
        pipeline_root=pipeline_root,
    )
    if replayed != frozen:
        raise SourceAuthorizedSynthesisV3Error("source_authorized_v3_report_replay_mismatch")
    return replayed


__all__ = [
    "SourceAuthorizedSynthesisReportV3",
    "SourceAuthorizedSynthesisV3Error",
    "compute_source_authorized_synthesis_v3_fingerprint",
    "reverify_source_authorized_synthesis_v3_report",
    "run_source_authorized_synthesis_v3",
]
