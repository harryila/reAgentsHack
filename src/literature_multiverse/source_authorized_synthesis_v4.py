"""V4 semantic-firewall projection of the source-authorized synthesis join."""

from __future__ import annotations

import ast
import platform
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
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
    SynthesisUnitAuthorizationReceipt,
    reverify_synthesis_unit_authorization,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


class SourceAuthorizedSynthesisV4Error(ValueError):
    pass


_DEPENDENCY_ENTRYPOINTS = (
    "src/literature_multiverse/source_authorized_synthesis_v4.py",
    "scripts/run_source_authorized_synthesis_v4.py",
    "scripts/validate_source_authorized_synthesis_v4.py",
)
_NON_PYTHON_INPUTS = ("pyproject.toml", "uv.lock")


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            raise SourceAuthorizedSynthesisV4Error(
                f"source_authorized_v4_relative_import_invalid:{current_path}:{module}"
            )
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
    raise SourceAuthorizedSynthesisV4Error(
        f"source_authorized_v4_local_dependency_missing:{current_path}:{module}"
    )


def _python_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source = repository_root / relative
        if not source.is_file():
            raise SourceAuthorizedSynthesisV4Error(
                f"source_authorized_v4_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise SourceAuthorizedSynthesisV4Error(
                f"source_authorized_v4_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


class SourceAuthorizedSynthesisUnitV4(ContractModel):
    unit_version: Literal["source-authorized-synthesis-unit-v4"] = (
        "source-authorized-synthesis-unit-v4"
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
            raise ValueError("source_authorized_v4_unit_hash_invalid")
        return value

    @field_validator("estimate_ids")
    @classmethod
    def validate_estimate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("source_authorized_v4_unit_estimates_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_unit(self) -> SourceAuthorizedSynthesisUnitV4:
        synthesized = self.status == "synthesized"
        if synthesized != (self.synthesis is not None and self.synthesis_sha256 is not None):
            raise ValueError("source_authorized_v4_synthesis_presence_mismatch")
        if self.synthesis is not None and hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("source_authorized_v4_synthesis_hash_mismatch")
        if synthesized == bool(self.unresolved_overlap_pairs):
            raise ValueError("source_authorized_v4_unresolved_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"unit_sha256"})
        if hash_canonical(payload) != self.unit_sha256:
            raise ValueError("source_authorized_v4_unit_hash_mismatch")
        return self


class SourceAuthorizedSynthesisReportV4(ContractModel):
    report_version: Literal["source-authorized-synthesis-report-v4"] = (
        "source-authorized-synthesis-report-v4"
    )
    scientific_role: Literal["source_authorized_synthesis_yield_only"]
    public_private_role: Literal["private_source_bound"]
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
    publication_source_content_visible: Literal[True] = True
    benchmark_reference_labels_accessed: Literal[False] = False
    benchmark_review_verdicts_accessed: Literal[False] = False
    supports_accuracy_claim: Literal[False] = False
    supports_release_claim: Literal[False] = False
    verifier_replay_status: Literal["not_constructed_missing_claim_calibration_and_audit_contract"]
    units: Annotated[list[SourceAuthorizedSynthesisUnitV4], Field(min_length=1)]
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
            raise ValueError("source_authorized_v4_report_hash_invalid")
        return value

    @field_validator("source_map")
    @classmethod
    def validate_source_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or list(value) != sorted(value):
            raise ValueError("source_authorized_v4_source_map_not_sorted")
        if any(not key or not SHA256_RE.fullmatch(digest) for key, digest in value.items()):
            raise ValueError("source_authorized_v4_source_map_invalid")
        return value

    @field_validator(
        "requested_estimate_ids",
        "authorization_receipt_sha256s",
        "authorized_receipt_sha256s",
    )
    @classmethod
    def validate_sorted_unique_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError(f"source_authorized_v4_not_sorted_unique:{info.field_name}")
        if "receipt" in info.field_name and any(not SHA256_RE.fullmatch(item) for item in value):
            raise ValueError(f"source_authorized_v4_invalid_hash:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> SourceAuthorizedSynthesisReportV4:
        if set(self.runtime) != {"python", "numpy", "scipy", "pydantic"} or any(
            not value.strip() for value in self.runtime.values()
        ):
            raise ValueError("source_authorized_v4_runtime_invalid")
        receipts = set(self.authorization_receipt_sha256s)
        authorized = set(self.authorized_receipt_sha256s)
        if not authorized <= receipts:
            raise ValueError("source_authorized_v4_authorized_not_subset")
        unit_receipts = [unit.authorization_receipt_sha256 for unit in self.units]
        if unit_receipts != sorted(set(unit_receipts)) or set(unit_receipts) != receipts:
            raise ValueError("source_authorized_v4_units_receipt_coverage_mismatch")
        synthesized = {
            unit.authorization_receipt_sha256 for unit in self.units if unit.status == "synthesized"
        }
        if synthesized != authorized:
            raise ValueError("source_authorized_v4_synthesized_authorized_mismatch")
        memberships = [item for unit in self.units for item in unit.estimate_ids]
        if (
            len(memberships) != len(set(memberships))
            or sorted(memberships) != self.requested_estimate_ids
        ):
            raise ValueError("source_authorized_v4_unit_estimate_coverage_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("source_authorized_v4_report_hash_mismatch")
        return self


def compute_source_authorized_synthesis_v4_fingerprint(*, root: Path) -> PipelineFingerprint:
    return compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="source-authorized-synthesis-v4",
                component_version="1",
                file_paths=sorted({*_python_dependency_closure(root), *_NON_PYTHON_INPUTS}),
                settings={
                    "source_authorization_scope": "independence_and_input_only",
                    "publication_source_content_visible": True,
                    "benchmark_reference_labels_accessed": False,
                    "benchmark_review_verdicts_accessed": False,
                    "python_version": platform.python_version(),
                    "numpy_version": distribution_version("numpy"),
                    "scipy_version": distribution_version("scipy"),
                    "pydantic_version": distribution_version("pydantic"),
                },
            )
        ],
    )


def run_source_authorized_synthesis_v4(
    *,
    corpus: TypedEvidenceCorpus,
    grounding_package: TypedEvidenceGroundingPackage,
    reconciliation: NativeCohortReconciliationReceipt,
    receipts: list[SynthesisUnitAuthorizationReceipt],
    requested_estimate_ids: list[str],
    authorized_receipt_sha256s: list[str],
    repository_root: Path,
    pipeline_root: Path | None = None,
) -> SourceAuthorizedSynthesisReportV4:
    corpus = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    grounding_package = TypedEvidenceGroundingPackage.model_validate(
        grounding_package.model_dump(mode="json")
    )
    reverify_typed_evidence_grounding_package(
        package=grounding_package, repository_root=repository_root
    )
    if grounding_package.corpus != corpus:
        raise SourceAuthorizedSynthesisV4Error("v4_grounding_package_corpus_mismatch")
    if grounding_package.cohort_reconciliation != reconciliation:
        raise SourceAuthorizedSynthesisV4Error("v4_grounding_package_reconciliation_mismatch")
    requested = sorted(set(requested_estimate_ids))
    if requested != requested_estimate_ids or not requested:
        raise SourceAuthorizedSynthesisV4Error("v4_requested_estimates_not_sorted_unique")
    receipt_hashes = [item.receipt_sha256 for item in receipts]
    if receipt_hashes != sorted(set(receipt_hashes)):
        raise SourceAuthorizedSynthesisV4Error("v4_authorization_receipts_not_sorted_unique")
    authorized_hashes = sorted(set(authorized_receipt_sha256s))
    if authorized_hashes != authorized_receipt_sha256s:
        raise SourceAuthorizedSynthesisV4Error("v4_authorized_hashes_not_sorted_unique")
    if not set(authorized_hashes) <= set(receipt_hashes):
        raise SourceAuthorizedSynthesisV4Error("v4_authorized_receipt_unknown")
    replayed = [
        reverify_synthesis_unit_authorization(
            corpus=corpus,
            reconciliation=reconciliation,
            receipt=item,
            repository_root=repository_root,
        )
        for item in receipts
    ]
    for receipt in replayed:
        declared = receipt.receipt_sha256 in authorized_hashes
        if declared != receipt.authorizes_synthesis_input:
            raise SourceAuthorizedSynthesisV4Error(
                "v4_authorized_set_disagrees_with_receipt_outcome"
            )
    membership = [item for receipt in replayed for item in receipt.estimate_ids]
    if len(membership) != len(set(membership)):
        raise SourceAuthorizedSynthesisV4Error("v4_overlapping_estimate_membership")
    if sorted(membership) != requested:
        raise SourceAuthorizedSynthesisV4Error("v4_requested_estimate_coverage_mismatch")
    graph = reconciliation.reconciled_graph
    if graph is None or reconciliation.reconciled_graph_sha256 is None:
        raise SourceAuthorizedSynthesisV4Error("v4_reconciled_graph_missing")
    graph_estimates = {item.estimate_id: item for item in graph.outcome_estimates}
    if any(item not in graph_estimates for item in requested):
        raise SourceAuthorizedSynthesisV4Error("v4_requested_estimate_unknown")

    units = []
    for receipt in replayed:
        if not receipt.authorizes_synthesis_input:
            unit_payload = {
                "unit_version": "source-authorized-synthesis-unit-v4",
                "authorization_receipt_sha256": receipt.receipt_sha256,
                "estimate_ids": receipt.estimate_ids,
                "status": "abstained_unresolved_independence",
                "synthesis": None,
                "synthesis_sha256": None,
                "unresolved_overlap_pairs": receipt.unresolved_overlap_pairs,
            }
        else:
            selected = [graph_estimates[item] for item in receipt.estimate_ids]
            outcomes = {item.outcome_name for item in selected}
            if len(outcomes) != 1:
                raise SourceAuthorizedSynthesisV4Error("v4_unit_mixes_outcomes")
            selected_graph = EvidenceGraph.model_validate(
                {**graph.model_dump(mode="json"), "outcome_estimates": selected}
            )
            synthesis = synthesize_evidence_graph(selected_graph, outcome_name=next(iter(outcomes)))
            unit_payload = {
                "unit_version": "source-authorized-synthesis-unit-v4",
                "authorization_receipt_sha256": receipt.receipt_sha256,
                "estimate_ids": receipt.estimate_ids,
                "status": "synthesized",
                "synthesis": synthesis,
                "synthesis_sha256": hash_canonical(synthesis),
                "unresolved_overlap_pairs": [],
            }
        units.append(
            SourceAuthorizedSynthesisUnitV4.model_validate(
                {**unit_payload, "unit_sha256": hash_canonical(unit_payload)}
            )
        )
    units.sort(key=lambda item: item.authorization_receipt_sha256)
    source_map = {
        fragment.publication_id: resolve_native_source_document(
            repository_root=repository_root,
            source_document=fragment.source_document,
        ).source_payload_sha256
        for fragment in corpus.fragments
    }
    payload = {
        "report_version": "source-authorized-synthesis-report-v4",
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
        "pipeline_fingerprint": compute_source_authorized_synthesis_v4_fingerprint(
            root=pipeline_root or Path(__file__).resolve().parents[2]
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": distribution_version("numpy"),
            "scipy": distribution_version("scipy"),
            "pydantic": distribution_version("pydantic"),
        },
        "publication_source_content_visible": True,
        "benchmark_reference_labels_accessed": False,
        "benchmark_review_verdicts_accessed": False,
        "supports_accuracy_claim": False,
        "supports_release_claim": False,
        "verifier_replay_status": ("not_constructed_missing_claim_calibration_and_audit_contract"),
        "units": units,
    }
    return SourceAuthorizedSynthesisReportV4.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def reverify_source_authorized_synthesis_v4_report(
    *,
    report: SourceAuthorizedSynthesisReportV4,
    corpus: TypedEvidenceCorpus,
    grounding_package: TypedEvidenceGroundingPackage,
    reconciliation: NativeCohortReconciliationReceipt,
    receipts: list[SynthesisUnitAuthorizationReceipt],
    repository_root: Path,
    pipeline_root: Path | None = None,
) -> SourceAuthorizedSynthesisReportV4:
    frozen = SourceAuthorizedSynthesisReportV4.model_validate(report.model_dump(mode="json"))
    replayed = run_source_authorized_synthesis_v4(
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
        raise SourceAuthorizedSynthesisV4Error("source_authorized_v4_report_replay_mismatch")
    return replayed


__all__ = [
    "SourceAuthorizedSynthesisReportV4",
    "SourceAuthorizedSynthesisUnitV4",
    "SourceAuthorizedSynthesisV4Error",
    "compute_source_authorized_synthesis_v4_fingerprint",
    "reverify_source_authorized_synthesis_v4_report",
    "run_source_authorized_synthesis_v4",
]
