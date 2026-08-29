"""Private, source-replayed human review for synthesis-unit authorization.

The statistical verifier cannot infer cohort independence from separate publications,
different registry identifiers, model confidence, or textual similarity.  This module
prepares a private packet containing only the exact source identity materials needed
for requested synthesis units, freezes two independent blank review forms, requires a
third reviewer for every scientific-decision disagreement, and projects only complete
agreed/adjudicated decisions into ``SourceIdentityAssertion`` v1 objects.

Reviewer packets may contain source text and identifiers.  Writers in this module
therefore accept only the repository's ignored ``data/cache/synthesis-authorization-review``
tree.  The public projection is aggregate-only and contains no source, publication,
cohort, synthesis-unit, or reviewer identifiers.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import re
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version as distribution_version
from itertools import combinations, pairwise
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
    reverify_native_cohort_reconciliation,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import (
    ResolvedSourceLine,
    resolve_native_source_document,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintError,
    compute_pipeline_fingerprint,
)
from literature_multiverse.synthesis_unit_authorization import (
    SourceIdentityAssertion,
    SourceIdentityCitation,
    SynthesisUnitAuthorizationReceipt,
    authorize_synthesis_unit,
    freeze_source_identity_assertion,
    freeze_source_identity_citation,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus

SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_COMPONENT_VERSION = "1"
SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_ENTRYPOINTS = (
    "scripts/evaluate_synthesis_authorization_review.py",
    "scripts/prepare_synthesis_authorization_review.py",
    "src/literature_multiverse/synthesis_authorization_review.py",
)
SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_NON_PYTHON_INPUTS = (
    "pyproject.toml",
    "uv.lock",
)


class SynthesisAuthorizationReviewError(ValueError):
    """A review packet or transition cannot support an authorization assertion."""


class ReviewRelationship(StrEnum):
    SAME_COHORT = "same_cohort"
    INDEPENDENT_COHORTS = "independent_cohorts"
    UNKNOWN = "unknown"


class ReviewTargetKind(StrEnum):
    MERGED_COHORT_IDENTITY = "merged_cohort_identity"
    CANONICAL_COHORT_INDEPENDENCE = "canonical_cohort_independence"


class ReviewerSlot(StrEnum):
    REVIEWER_A = "reviewer_a"
    REVIEWER_B = "reviewer_b"
    ADJUDICATOR = "adjudicator"


def _module_name_for_repository_path(relative: str) -> tuple[str, bool] | None:
    path = Path(relative)
    parts = list(path.parts)
    if parts[:2] == ["src", "literature_multiverse"]:
        module_parts = parts[1:]
    elif parts[:1] == ["scripts"]:
        module_parts = parts
    else:
        return None
    is_package = module_parts[-1] == "__init__.py"
    if is_package:
        module_parts = module_parts[:-1]
    elif module_parts[-1].endswith(".py"):
        module_parts[-1] = module_parts[-1][:-3]
    else:
        return None
    return ".".join(module_parts), is_package


def _local_module_path(repository_root: Path, module: str) -> str | None:
    if module == "literature_multiverse" or module.startswith("literature_multiverse."):
        base = Path("src", *module.split("."))
    elif module == "scripts" or module.startswith("scripts."):
        base = Path(*module.split("."))
    else:
        return None
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _absolute_import_module(*, current_path: str, module: str, level: int) -> str | None:
    if not level:
        return module
    current = _module_name_for_repository_path(current_path)
    if current is None:
        return None
    current_module, is_package = current
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    if level > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - (level - 1)]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _synthesis_authorization_review_python_dependency_closure(
    repository_root: Path,
) -> list[str]:
    """Resolve the complete in-repository import closure from the public entry points."""

    pending = list(SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_ENTRYPOINTS)
    observed: set[str] = set()
    package_init = "src/literature_multiverse/__init__.py"
    if (repository_root / package_init).is_file():
        pending.append(package_init)
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise SynthesisAuthorizationReviewError(
                f"synthesis_review_pipeline_dependency_missing:{relative}"
            )
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise SynthesisAuthorizationReviewError(
                f"synthesis_review_pipeline_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependency = _local_module_path(repository_root, alias.name)
                    if dependency is not None:
                        if dependency not in observed:
                            pending.append(dependency)
                    elif alias.name.startswith(("literature_multiverse.", "scripts.")):
                        raise SynthesisAuthorizationReviewError(
                            f"synthesis_review_pipeline_local_import_missing:{alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                absolute = _absolute_import_module(
                    current_path=relative,
                    module=node.module or "",
                    level=node.level,
                )
                if absolute is None:
                    continue
                dependency = _local_module_path(repository_root, absolute)
                if dependency is not None:
                    if dependency not in observed:
                        pending.append(dependency)
                elif absolute.startswith(("literature_multiverse.", "scripts.")):
                    raise SynthesisAuthorizationReviewError(
                        f"synthesis_review_pipeline_local_import_missing:{absolute}"
                    )
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    alias_module = f"{absolute}.{alias.name}" if absolute else alias.name
                    alias_dependency = _local_module_path(repository_root, alias_module)
                    if alias_dependency is not None and alias_dependency not in observed:
                        pending.append(alias_dependency)
    return sorted(observed)


def compute_synthesis_authorization_review_pipeline_fingerprint(
    root: Path,
) -> PipelineFingerprint:
    """Hash the AST-closed review runtime and its exact lock/config inputs."""

    try:
        repository_root = root.resolve(strict=True)
        python_files = _synthesis_authorization_review_python_dependency_closure(repository_root)
        component = PipelineComponentSpec(
            component_id="synthesis-authorization-source-review",
            component_version=SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_COMPONENT_VERSION,
            file_paths=sorted(
                {
                    *python_files,
                    *SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_NON_PYTHON_INPUTS,
                }
            ),
            settings={
                "affirmative_decisions_require_exact_source_replay": True,
                "agreement_basis": "relationship_label_after_independent_source_validation",
                "dependency_closure_entrypoints": list(
                    SYNTHESIS_AUTHORIZATION_REVIEW_PIPELINE_ENTRYPOINTS
                ),
                "distinct_publication_registry_or_dataset_ids_imply_independence": False,
                "in_repository_ast_dependency_closure_bound": True,
                "installed_dependency_versions": {"pydantic": distribution_version("pydantic")},
                "private_review_root": "data/cache/synthesis-authorization-review",
                "public_summary_root": ("artifacts/diagnostics/synthesis-authorization-review"),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "substantive_disagreement_policy": "distinct_third_adjudicator_required",
                "support_divergence_policy": "preserve_both_valid_source_provenances",
                "unknown_or_missing_policy": "abstain",
            },
        )
        return compute_pipeline_fingerprint(
            root=repository_root,
            components=[component],
        )
    except (OSError, PipelineFingerprintError) as exc:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_pipeline_fingerprint_unverifiable"
        ) from exc


def _validate_pipeline_binding(
    fingerprint: PipelineFingerprint,
    fingerprint_sha256: str,
    *,
    code: str,
) -> None:
    if fingerprint.pipeline_sha256 != fingerprint_sha256:
        raise ValueError(code)


def _require_current_pipeline_binding(
    *,
    repository_root: Path,
    fingerprint: PipelineFingerprint,
    fingerprint_sha256: str,
) -> PipelineFingerprint:
    current = compute_synthesis_authorization_review_pipeline_fingerprint(repository_root)
    if current != fingerprint or current.pipeline_sha256 != fingerprint_sha256:
        raise SynthesisAuthorizationReviewError("synthesis_review_pipeline_fingerprint_mismatch")
    return current


def _validate_sha256(value: str, *, code: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(code)
    return value


def _validate_utc(value: datetime, *, code: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError(code)
    return value.astimezone(UTC)


def _utc_json(value: datetime) -> str:
    """Match Pydantic's canonical UTC JSON representation before hashing."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_identifier(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _strictly_under(path: Path, root: Path, *, code: str) -> Path:
    """Resolve a prospective output and require it to be strictly below ``root``."""

    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise SynthesisAuthorizationReviewError(code)
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise SynthesisAuthorizationReviewError(f"{code}_symlink")
    return resolved_path


class RequestedSynthesisUnit(ContractModel):
    synthesis_unit_id: Annotated[str, Field(min_length=1)]
    estimate_ids: Annotated[list[str], Field(min_length=1)]

    @field_validator("estimate_ids")
    @classmethod
    def validate_estimates(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_estimate_ids_not_sorted_unique")
        return value


class SynthesisAuthorizationReviewRequest(ContractModel):
    request_version: Literal["synthesis-authorization-review-request-v1"] = (
        "synthesis-authorization-review-request-v1"
    )
    synthesis_units: Annotated[list[RequestedSynthesisUnit], Field(min_length=1)]
    request_sha256: str

    @field_validator("request_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_request_sha256_invalid")

    @field_validator("synthesis_units")
    @classmethod
    def validate_units(cls, value: list[RequestedSynthesisUnit]) -> list[RequestedSynthesisUnit]:
        identifiers = [item.synthesis_unit_id for item in value]
        if identifiers != sorted(set(identifiers)):
            raise ValueError("synthesis_review_unit_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_self_hash(self) -> SynthesisAuthorizationReviewRequest:
        payload = self.model_dump(mode="json", exclude={"request_sha256"})
        if hash_canonical(payload) != self.request_sha256:
            raise ValueError("synthesis_review_request_hash_mismatch")
        return self


def freeze_synthesis_authorization_review_request(
    synthesis_units: list[RequestedSynthesisUnit],
) -> SynthesisAuthorizationReviewRequest:
    payload = {
        "request_version": "synthesis-authorization-review-request-v1",
        "synthesis_units": sorted(synthesis_units, key=lambda item: item.synthesis_unit_id),
    }
    return SynthesisAuthorizationReviewRequest.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )


class SynthesisAuthorizationReviewProtocol(ContractModel):
    protocol_version: Literal["synthesis-authorization-source-review-v1"] = (
        "synthesis-authorization-source-review-v1"
    )
    decision_labels: tuple[
        Literal["same_cohort"],
        Literal["independent_cohorts"],
        Literal["unknown"],
    ] = (
        "same_cohort",
        "independent_cohorts",
        "unknown",
    )
    independent_reviewers: Literal[2] = 2
    disagreement_adjudicators: Literal[1] = 1
    source_identity_visible: Literal[True] = True
    system_scores_hidden: Literal[True] = True
    peer_decisions_hidden_during_independent_review: Literal[True] = True
    exact_citation_required_for_affirmative_decision: Literal[True] = True
    citation_required_for_every_original_cohort: Literal[True] = True
    measured_person_minutes_required: Literal[True] = True
    distinct_publications_do_not_imply_independence: Literal[True] = True
    distinct_registry_or_dataset_ids_do_not_imply_independence: Literal[True] = True
    unknown_or_missing_requires_abstention: Literal[True] = True
    instructions: Annotated[list[str], Field(min_length=1)]
    protocol_sha256: str

    @field_validator("protocol_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_protocol_sha256_invalid")

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("synthesis_review_protocol_instructions_invalid")
        return value

    @model_validator(mode="after")
    def validate_self_hash(self) -> SynthesisAuthorizationReviewProtocol:
        payload = self.model_dump(mode="json", exclude={"protocol_sha256"})
        if hash_canonical(payload) != self.protocol_sha256:
            raise ValueError("synthesis_review_protocol_hash_mismatch")
        return self


def default_synthesis_authorization_review_protocol() -> SynthesisAuthorizationReviewProtocol:
    payload = {
        "protocol_version": "synthesis-authorization-source-review-v1",
        "decision_labels": ("same_cohort", "independent_cohorts", "unknown"),
        "independent_reviewers": 2,
        "disagreement_adjudicators": 1,
        "source_identity_visible": True,
        "system_scores_hidden": True,
        "peer_decisions_hidden_during_independent_review": True,
        "exact_citation_required_for_affirmative_decision": True,
        "citation_required_for_every_original_cohort": True,
        "measured_person_minutes_required": True,
        "distinct_publications_do_not_imply_independence": True,
        "distinct_registry_or_dataset_ids_do_not_imply_independence": True,
        "unknown_or_missing_requires_abstention": True,
        "instructions": [
            (
                "Review the cited source bytes; do not use model confidence, rank, "
                "influence, or synthesis scores."
            ),
            (
                "Choose same_cohort only when the sources establish that every listed "
                "original cohort is the same participant or sample cohort."
            ),
            (
                "Choose independent_cohorts only when the sources affirmatively establish "
                "non-overlapping participant or sample cohorts."
            ),
            (
                "Different publications, labels, registry identifiers, or dataset "
                "identifiers are not affirmative evidence of independence."
            ),
            (
                "For an affirmative decision, cite exact source lines and an exact "
                "source-reported registry or dataset identifier for every original cohort."
            ),
            (
                "Choose unknown whenever the available source bytes do not establish the "
                "requested relationship; unknown cannot authorize synthesis."
            ),
            (
                "Independent agreement is defined on the relationship label after each "
                "reviewer's affirmative evidence passes exact source replay; differing "
                "valid rationales and citations remain separately preserved and reported."
            ),
            (
                "Record actual start and completion timestamps; review_minutes must equal "
                "elapsed person-minutes."
            ),
        ],
    }
    return SynthesisAuthorizationReviewProtocol.model_validate(
        {**payload, "protocol_sha256": hash_canonical(payload)}
    )


class ReviewSourceMaterial(ContractModel):
    """Private, exact source projection shown to reviewers without system scores."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    source_material_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]
    artifact_path: Annotated[str, Field(min_length=1)]
    source_kind: Annotated[str, Field(min_length=1)]
    source_document_sha256: str
    grounding_receipt_sha256: str
    source_locator: Annotated[str, Field(min_length=1)]
    source_payload_sha256: str
    lines: Annotated[list[ResolvedSourceLine], Field(min_length=1)]
    source_material_sha256: str

    @field_validator(
        "source_document_sha256",
        "grounding_receipt_sha256",
        "source_payload_sha256",
        "source_material_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_source_sha256_invalid")

    @model_validator(mode="after")
    def validate_self_hash(self) -> ReviewSourceMaterial:
        payload = self.model_dump(mode="json", exclude={"source_material_sha256"})
        if hash_canonical(payload) != self.source_material_sha256:
            raise ValueError("synthesis_review_source_material_hash_mismatch")
        return self


class OriginalCohortSourceMaterial(ContractModel):
    original_cohort_id: Annotated[str, Field(min_length=1)]
    canonical_cohort_id: Annotated[str, Field(min_length=1)]
    publication_id: Annotated[str, Field(min_length=1)]
    source_material_id: Annotated[str, Field(min_length=1)]
    source_material_sha256: str
    registry_ids: list[str]
    dataset_ids: list[str]
    cohort_source_sha256: str

    @field_validator("source_material_sha256", "cohort_source_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_cohort_source_sha256_invalid")

    @field_validator("registry_ids", "dataset_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_source_identifiers_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_self_hash(self) -> OriginalCohortSourceMaterial:
        payload = self.model_dump(mode="json", exclude={"cohort_source_sha256"})
        if hash_canonical(payload) != self.cohort_source_sha256:
            raise ValueError("synthesis_review_cohort_source_hash_mismatch")
        return self


class SynthesisAuthorizationReviewTarget(ContractModel):
    target_id: Annotated[str, Field(min_length=1)]
    synthesis_unit_id: Annotated[str, Field(min_length=1)]
    target_kind: ReviewTargetKind
    required_relationship: Literal["same_cohort", "independent_cohorts"]
    assertion_cohort_ids: Annotated[list[str], Field(min_length=2)]
    canonical_cohort_ids: Annotated[list[str], Field(min_length=1)]
    original_cohort_ids: Annotated[list[str], Field(min_length=2)]
    publication_ids: Annotated[list[str], Field(min_length=1)]
    target_sha256: str

    @field_validator(
        "assertion_cohort_ids",
        "canonical_cohort_ids",
        "original_cohort_ids",
        "publication_ids",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_target_ids_not_sorted_unique")
        return value

    @field_validator("target_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_target_sha256_invalid")

    @model_validator(mode="after")
    def validate_target(self) -> SynthesisAuthorizationReviewTarget:
        if self.target_kind is ReviewTargetKind.MERGED_COHORT_IDENTITY:
            if self.required_relationship != "same_cohort" or len(self.canonical_cohort_ids) != 1:
                raise ValueError("synthesis_review_merged_target_semantics_invalid")
            if self.assertion_cohort_ids != self.original_cohort_ids:
                raise ValueError("synthesis_review_merged_target_assertion_ids_invalid")
        else:
            if (
                self.required_relationship != "independent_cohorts"
                or len(self.canonical_cohort_ids) != 2
                or self.assertion_cohort_ids != self.canonical_cohort_ids
            ):
                raise ValueError("synthesis_review_independence_target_semantics_invalid")
        payload = self.model_dump(mode="json", exclude={"target_sha256"})
        if hash_canonical(payload) != self.target_sha256:
            raise ValueError("synthesis_review_target_hash_mismatch")
        return self


class FrozenRequestedSynthesisUnit(ContractModel):
    synthesis_unit_id: Annotated[str, Field(min_length=1)]
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    canonical_cohort_ids: Annotated[list[str], Field(min_length=1)]
    target_ids: Annotated[list[str], Field(min_length=1)]
    unit_sha256: str

    @field_validator("estimate_ids", "canonical_cohort_ids", "target_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_frozen_unit_ids_not_sorted_unique")
        return value

    @field_validator("unit_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_unit_sha256_invalid")

    @model_validator(mode="after")
    def validate_self_hash(self) -> FrozenRequestedSynthesisUnit:
        payload = self.model_dump(mode="json", exclude={"unit_sha256"})
        if hash_canonical(payload) != self.unit_sha256:
            raise ValueError("synthesis_review_unit_hash_mismatch")
        return self


class SynthesisAuthorizationReviewPacket(ContractModel):
    packet_version: Literal["private-synthesis-authorization-review-packet-v1"] = (
        "private-synthesis-authorization-review-packet-v1"
    )
    created_at: datetime
    input_corpus_sha256: str
    reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    request_sha256: str
    review_protocol: SynthesisAuthorizationReviewProtocol
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: str
    source_manifest_sha256: str
    requested_units: Annotated[list[FrozenRequestedSynthesisUnit], Field(min_length=1)]
    targets: Annotated[list[SynthesisAuthorizationReviewTarget], Field(min_length=1)]
    cohort_sources: Annotated[list[OriginalCohortSourceMaterial], Field(min_length=2)]
    source_materials: Annotated[list[ReviewSourceMaterial], Field(min_length=1)]
    source_identity_visible: Literal[True] = True
    publication_source_content_visible: Literal[True] = True
    system_scores_included: Literal[False] = False
    benchmark_reference_labels_accessed: Literal[False] = False
    benchmark_review_verdicts_accessed: Literal[False] = False
    packet_sha256: str

    @field_validator(
        "input_corpus_sha256",
        "reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "request_sha256",
        "pipeline_fingerprint_sha256",
        "source_manifest_sha256",
        "packet_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_packet_sha256_invalid")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, code="synthesis_review_packet_created_at_not_utc")

    @model_validator(mode="after")
    def validate_packet(self) -> SynthesisAuthorizationReviewPacket:
        _validate_pipeline_binding(
            self.pipeline_fingerprint,
            self.pipeline_fingerprint_sha256,
            code="synthesis_review_packet_pipeline_fingerprint_hash_mismatch",
        )
        unit_ids = [item.synthesis_unit_id for item in self.requested_units]
        target_ids = [item.target_id for item in self.targets]
        cohort_ids = [item.original_cohort_id for item in self.cohort_sources]
        source_ids = [item.source_material_id for item in self.source_materials]
        for values, code in (
            (unit_ids, "synthesis_review_packet_units_not_sorted_unique"),
            (target_ids, "synthesis_review_packet_targets_not_sorted_unique"),
            (cohort_ids, "synthesis_review_packet_cohorts_not_sorted_unique"),
            (source_ids, "synthesis_review_packet_sources_not_sorted_unique"),
        ):
            if values != sorted(set(values)):
                raise ValueError(code)
        if {target.target_id for target in self.targets} != {
            target_id for unit in self.requested_units for target_id in unit.target_ids
        }:
            raise ValueError("synthesis_review_packet_target_roster_mismatch")
        source_manifest = hash_canonical(
            {
                "cohort_sources": self.cohort_sources,
                "source_materials": self.source_materials,
            }
        )
        if source_manifest != self.source_manifest_sha256:
            raise ValueError("synthesis_review_source_manifest_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"packet_sha256"})
        if hash_canonical(payload) != self.packet_sha256:
            raise ValueError("synthesis_review_packet_hash_mismatch")
        return self


class ReviewCitationForm(ContractModel):
    """Editable citation slot; immutable source fields are prefilled by preparation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )

    publication_id: Annotated[str, Field(min_length=1)]
    original_cohort_id: Annotated[str, Field(min_length=1)]
    source_document_sha256: str
    grounding_receipt_sha256: str
    source_locator: Annotated[str, Field(min_length=1)]
    source_payload_sha256: str
    eligible_source_identifiers: list[str]
    quote: str | None = None
    line_ids: list[str] = Field(default_factory=list)
    cited_identifier: str | None = None

    @field_validator("source_document_sha256", "grounding_receipt_sha256", "source_payload_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_form_source_sha256_invalid")

    @field_validator("eligible_source_identifiers", "line_ids")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_form_values_not_sorted_unique")
        return value


class ReviewDecisionForm(ContractModel):
    target_id: Annotated[str, Field(min_length=1)]
    synthesis_unit_id: Annotated[str, Field(min_length=1)]
    target_kind: ReviewTargetKind
    relationship_under_review: Literal["same_cohort", "independent_cohorts"]
    assertion_cohort_ids: Annotated[list[str], Field(min_length=2)]
    original_cohort_ids: Annotated[list[str], Field(min_length=2)]
    relationship: ReviewRelationship | None = None
    rationale: str | None = None
    citations: Annotated[list[ReviewCitationForm], Field(min_length=2)]
    review_started_at: datetime | None = None
    review_completed_at: datetime | None = None
    review_minutes: float | None = None

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: list[ReviewCitationForm]) -> list[ReviewCitationForm]:
        ids = [item.original_cohort_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("synthesis_review_form_citations_not_sorted_unique")
        return value

    @field_validator("assertion_cohort_ids", "original_cohort_ids")
    @classmethod
    def validate_identity_rosters(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_form_identity_roster_not_sorted_unique")
        return value

    @field_validator("review_minutes")
    @classmethod
    def validate_minutes(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("synthesis_review_form_minutes_nonfinite")
        return value


class BlankSynthesisAuthorizationReviewTemplate(ContractModel):
    template_version: Literal["synthesis-authorization-review-template-v1"] = (
        "synthesis-authorization-review-template-v1"
    )
    packet_sha256: str
    review_protocol_sha256: str
    reviewer_slot: ReviewerSlot
    input_transition_sha256: str
    reviewer_identity_sha256: None = None
    submitted_at: None = None
    decisions: Annotated[list[ReviewDecisionForm], Field(min_length=1)]
    template_sha256: str

    @field_validator(
        "packet_sha256", "review_protocol_sha256", "input_transition_sha256", "template_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_template_sha256_invalid")

    @model_validator(mode="after")
    def validate_template(self) -> BlankSynthesisAuthorizationReviewTemplate:
        target_ids = [item.target_id for item in self.decisions]
        if target_ids != sorted(set(target_ids)):
            raise ValueError("synthesis_review_template_targets_not_sorted_unique")
        for decision in self.decisions:
            if any(
                value is not None
                for value in (
                    decision.relationship,
                    decision.rationale,
                    decision.review_started_at,
                    decision.review_completed_at,
                    decision.review_minutes,
                )
            ):
                raise ValueError("synthesis_review_template_decision_not_blank")
            for citation in decision.citations:
                if (
                    citation.quote is not None
                    or citation.line_ids
                    or citation.cited_identifier is not None
                ):
                    raise ValueError("synthesis_review_template_citation_not_blank")
        payload = self.model_dump(mode="json", exclude={"template_sha256"})
        if hash_canonical(payload) != self.template_sha256:
            raise ValueError("synthesis_review_template_hash_mismatch")
        return self


class SubmittedSynthesisAuthorizationReviewForm(ContractModel):
    template_version: Literal["synthesis-authorization-review-template-v1"] = (
        "synthesis-authorization-review-template-v1"
    )
    packet_sha256: str
    review_protocol_sha256: str
    reviewer_slot: ReviewerSlot
    input_transition_sha256: str
    reviewer_identity_sha256: str
    submitted_at: datetime
    decisions: Annotated[list[ReviewDecisionForm], Field(min_length=1)]
    template_sha256: str

    @field_validator(
        "packet_sha256",
        "review_protocol_sha256",
        "input_transition_sha256",
        "reviewer_identity_sha256",
        "template_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_submission_sha256_invalid")

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, code="synthesis_review_submitted_at_not_utc")

    @field_validator("decisions")
    @classmethod
    def validate_decisions(cls, value: list[ReviewDecisionForm]) -> list[ReviewDecisionForm]:
        ids = [item.target_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("synthesis_review_submission_targets_not_sorted_unique")
        return value


class FrozenSynthesisAuthorizationReviewDecision(ContractModel):
    target_id: Annotated[str, Field(min_length=1)]
    relationship: ReviewRelationship
    rationale: Annotated[str, Field(min_length=1)]
    citations: list[SourceIdentityCitation]
    review_started_at: datetime
    review_completed_at: datetime
    review_minutes: Annotated[float, Field(gt=0)]
    scientific_decision_sha256: str
    decision_sha256: str

    @field_validator("review_started_at", "review_completed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _validate_utc(value, code="synthesis_review_decision_time_not_utc")

    @field_validator("scientific_decision_sha256", "decision_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_decision_sha256_invalid")

    @field_validator("review_minutes")
    @classmethod
    def validate_minutes(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("synthesis_review_decision_minutes_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> FrozenSynthesisAuthorizationReviewDecision:
        if self.review_completed_at <= self.review_started_at:
            raise ValueError("synthesis_review_decision_time_not_ordered")
        elapsed = (self.review_completed_at - self.review_started_at).total_seconds() / 60.0
        if not math.isclose(self.review_minutes, elapsed, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("synthesis_review_minutes_not_measured_elapsed_time")
        citation_hashes = [item.citation_sha256 for item in self.citations]
        if citation_hashes != sorted(set(citation_hashes)):
            raise ValueError("synthesis_review_decision_citations_not_sorted_unique")
        scientific_payload = {
            "target_id": self.target_id,
            "relationship": self.relationship,
            "rationale": self.rationale,
            "citations": self.citations,
        }
        if hash_canonical(scientific_payload) != self.scientific_decision_sha256:
            raise ValueError("synthesis_review_scientific_decision_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("synthesis_review_decision_hash_mismatch")
        return self


class FrozenSynthesisAuthorizationReviewSubmission(ContractModel):
    submission_version: Literal["synthesis-authorization-review-submission-v1"] = (
        "synthesis-authorization-review-submission-v1"
    )
    packet_sha256: str
    template_sha256: str
    review_protocol_sha256: str
    reviewer_slot: ReviewerSlot
    reviewer_identity_sha256: str
    input_transition_sha256: str
    submitted_at: datetime
    submitted_form_payload_sha256: str
    decisions: Annotated[list[FrozenSynthesisAuthorizationReviewDecision], Field(min_length=1)]
    submission_sha256: str

    @field_validator(
        "packet_sha256",
        "template_sha256",
        "review_protocol_sha256",
        "reviewer_identity_sha256",
        "input_transition_sha256",
        "submitted_form_payload_sha256",
        "submission_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_frozen_submission_sha256_invalid")

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, code="synthesis_review_frozen_submitted_at_not_utc")

    @model_validator(mode="after")
    def validate_submission(self) -> FrozenSynthesisAuthorizationReviewSubmission:
        ids = [item.target_id for item in self.decisions]
        if ids != sorted(set(ids)):
            raise ValueError("synthesis_review_frozen_submission_targets_not_sorted_unique")
        if self.submitted_at < max(item.review_completed_at for item in self.decisions):
            raise ValueError("synthesis_review_submission_precedes_decision_completion")
        intervals = sorted(
            (item.review_started_at, item.review_completed_at) for item in self.decisions
        )
        if any(right_start < left_end for (_, left_end), (right_start, _) in pairwise(intervals)):
            raise ValueError("synthesis_review_reviewer_intervals_overlap")
        payload = self.model_dump(mode="json", exclude={"submission_sha256"})
        if hash_canonical(payload) != self.submission_sha256:
            raise ValueError("synthesis_review_submission_hash_mismatch")
        return self


class SynthesisAuthorizationReviewComparison(ContractModel):
    transition_version: Literal["synthesis-authorization-review-comparison-v1"] = (
        "synthesis-authorization-review-comparison-v1"
    )
    packet_sha256: str
    reviewer_a_submission_sha256: str
    reviewer_b_submission_sha256: str
    agreed_target_ids: list[str]
    support_divergence_target_ids: list[str]
    disagreement_target_ids: list[str]
    predecessor_transition_sha256: str
    transition_sha256: str

    @field_validator(
        "packet_sha256",
        "reviewer_a_submission_sha256",
        "reviewer_b_submission_sha256",
        "predecessor_transition_sha256",
        "transition_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_comparison_sha256_invalid")

    @field_validator(
        "agreed_target_ids", "support_divergence_target_ids", "disagreement_target_ids"
    )
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_comparison_targets_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_transition(self) -> SynthesisAuthorizationReviewComparison:
        if set(self.agreed_target_ids).intersection(self.disagreement_target_ids):
            raise ValueError("synthesis_review_comparison_target_sets_overlap")
        if not set(self.support_divergence_target_ids).issubset(self.agreed_target_ids):
            raise ValueError("synthesis_review_support_divergence_outside_label_agreement")
        if self.predecessor_transition_sha256 != self.packet_sha256:
            raise ValueError("synthesis_review_comparison_predecessor_mismatch")
        payload = self.model_dump(mode="json", exclude={"transition_sha256"})
        if hash_canonical(payload) != self.transition_sha256:
            raise ValueError("synthesis_review_comparison_hash_mismatch")
        return self


class ResolutionSourceDecision(ContractModel):
    """Exact reviewer-decision provenance retained in a panel resolution."""

    reviewer_slot: ReviewerSlot
    decision_sha256: str
    scientific_decision_sha256: str
    relationship: ReviewRelationship
    rationale: Annotated[str, Field(min_length=1)]
    citation_sha256s: list[str]
    reference_sha256: str

    @field_validator("decision_sha256", "scientific_decision_sha256", "reference_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_resolution_reference_sha_invalid")

    @field_validator("citation_sha256s")
    @classmethod
    def validate_citation_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not SHA256_RE.fullmatch(item) for item in value):
            raise ValueError("synthesis_review_resolution_reference_citations_invalid")
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> ResolutionSourceDecision:
        payload = self.model_dump(mode="json", exclude={"reference_sha256"})
        if hash_canonical(payload) != self.reference_sha256:
            raise ValueError("synthesis_review_resolution_reference_hash_mismatch")
        return self


def _combined_agreement_rationale(references: list[ResolutionSourceDecision]) -> str:
    labels = {
        ReviewerSlot.REVIEWER_A: "Reviewer A",
        ReviewerSlot.REVIEWER_B: "Reviewer B",
    }
    return "\n\n".join(f"{labels[item.reviewer_slot]}: {item.rationale}" for item in references)


class ResolvedSynthesisAuthorizationReviewDecision(ContractModel):
    target_id: Annotated[str, Field(min_length=1)]
    resolution_source: Literal["independent_agreement", "third_adjudication"]
    relationship: ReviewRelationship
    rationale: Annotated[str, Field(min_length=1)]
    citations: list[SourceIdentityCitation]
    source_decisions: Annotated[list[ResolutionSourceDecision], Field(min_length=2, max_length=3)]
    support_evidence_diverged: bool
    panel_identity_sha256: str
    resolution_sha256: str

    @field_validator("panel_identity_sha256", "resolution_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_resolution_sha256_invalid")

    @model_validator(mode="after")
    def validate_resolution(self) -> ResolvedSynthesisAuthorizationReviewDecision:
        hashes = [item.citation_sha256 for item in self.citations]
        if hashes != sorted(set(hashes)):
            raise ValueError("synthesis_review_resolution_citations_not_sorted_unique")
        slots = [item.reviewer_slot for item in self.source_decisions]
        expected_slots = (
            [ReviewerSlot.REVIEWER_A, ReviewerSlot.REVIEWER_B]
            if self.resolution_source == "independent_agreement"
            else [ReviewerSlot.REVIEWER_A, ReviewerSlot.REVIEWER_B, ReviewerSlot.ADJUDICATOR]
        )
        if slots != expected_slots:
            raise ValueError("synthesis_review_resolution_source_decision_slots_invalid")
        initial = self.source_decisions[:2]
        expected_divergence = (
            initial[0].scientific_decision_sha256 != initial[1].scientific_decision_sha256
        )
        if self.support_evidence_diverged != expected_divergence:
            raise ValueError("synthesis_review_resolution_support_divergence_mismatch")
        if self.resolution_source == "independent_agreement":
            if len({item.relationship for item in initial}) != 1:
                raise ValueError("synthesis_review_resolution_agreement_label_mismatch")
            expected_citations = sorted(
                {citation_sha256 for item in initial for citation_sha256 in item.citation_sha256s}
            )
            if self.relationship is not initial[0].relationship:
                raise ValueError("synthesis_review_resolution_agreement_relationship_mismatch")
            if self.rationale != _combined_agreement_rationale(initial):
                raise ValueError("synthesis_review_resolution_agreement_rationale_mismatch")
        else:
            adjudicator = self.source_decisions[2]
            expected_citations = adjudicator.citation_sha256s
            if (
                self.relationship is not adjudicator.relationship
                or self.rationale != adjudicator.rationale
            ):
                raise ValueError("synthesis_review_resolution_adjudication_projection_mismatch")
        if hashes != expected_citations:
            raise ValueError("synthesis_review_resolution_citation_projection_mismatch")
        payload = self.model_dump(mode="json", exclude={"resolution_sha256"})
        if hash_canonical(payload) != self.resolution_sha256:
            raise ValueError("synthesis_review_resolution_hash_mismatch")
        return self


class SynthesisAuthorizationReviewedUnit(ContractModel):
    synthesis_unit_id: Annotated[str, Field(min_length=1)]
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    target_ids: Annotated[list[str], Field(min_length=1)]
    assertions: list[SourceIdentityAssertion]
    blocker_codes: list[str]
    authorization_receipt: SynthesisUnitAuthorizationReceipt | None
    authorizes_synthesis_input: bool
    unit_review_sha256: str

    @field_validator("estimate_ids", "target_ids", "blocker_codes")
    @classmethod
    def validate_sorted_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("synthesis_review_unit_outcome_values_not_sorted_unique")
        return value

    @field_validator("unit_review_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_unit_outcome_sha256_invalid")

    @model_validator(mode="after")
    def validate_outcome(self) -> SynthesisAuthorizationReviewedUnit:
        hashes = [item.assertion_sha256 for item in self.assertions]
        if hashes != sorted(set(hashes)):
            raise ValueError("synthesis_review_unit_assertions_not_sorted_unique")
        expected = bool(
            self.authorization_receipt is not None
            and self.authorization_receipt.authorizes_synthesis_input
            and not self.blocker_codes
        )
        if self.authorizes_synthesis_input != expected:
            raise ValueError("synthesis_review_unit_authorization_outcome_mismatch")
        payload = self.model_dump(mode="json", exclude={"unit_review_sha256"})
        if hash_canonical(payload) != self.unit_review_sha256:
            raise ValueError("synthesis_review_unit_outcome_hash_mismatch")
        return self


class PrivateSynthesisAuthorizationReviewEvaluation(ContractModel):
    evaluation_version: Literal["private-synthesis-authorization-review-evaluation-v1"] = (
        "private-synthesis-authorization-review-evaluation-v1"
    )
    status: Literal["awaiting_adjudication", "complete"]
    packet_sha256: str
    review_protocol_sha256: str
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: str
    reviewer_a_submission: FrozenSynthesisAuthorizationReviewSubmission
    reviewer_b_submission: FrozenSynthesisAuthorizationReviewSubmission
    comparison: SynthesisAuthorizationReviewComparison
    adjudication_template_sha256: str | None
    adjudicator_submission: FrozenSynthesisAuthorizationReviewSubmission | None
    resolutions: list[ResolvedSynthesisAuthorizationReviewDecision]
    unit_outcomes: Annotated[list[SynthesisAuthorizationReviewedUnit], Field(min_length=1)]
    final_transition_sha256: str
    evaluation_sha256: str

    @field_validator(
        "packet_sha256",
        "review_protocol_sha256",
        "pipeline_fingerprint_sha256",
        "adjudication_template_sha256",
        "final_transition_sha256",
        "evaluation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None:
            return _validate_sha256(value, code="synthesis_review_evaluation_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_evaluation(self) -> PrivateSynthesisAuthorizationReviewEvaluation:
        _validate_pipeline_binding(
            self.pipeline_fingerprint,
            self.pipeline_fingerprint_sha256,
            code="synthesis_review_evaluation_pipeline_fingerprint_hash_mismatch",
        )
        reviewer_a = self.reviewer_a_submission
        reviewer_b = self.reviewer_b_submission
        if reviewer_a.reviewer_slot is not ReviewerSlot.REVIEWER_A:
            raise ValueError("synthesis_review_evaluation_reviewer_a_slot_mismatch")
        if reviewer_b.reviewer_slot is not ReviewerSlot.REVIEWER_B:
            raise ValueError("synthesis_review_evaluation_reviewer_b_slot_mismatch")
        for submission in (reviewer_a, reviewer_b):
            if (
                submission.packet_sha256 != self.packet_sha256
                or submission.review_protocol_sha256 != self.review_protocol_sha256
                or submission.input_transition_sha256 != self.packet_sha256
            ):
                raise ValueError("synthesis_review_evaluation_submission_lineage_mismatch")
        if reviewer_a.reviewer_identity_sha256 == reviewer_b.reviewer_identity_sha256:
            raise ValueError("synthesis_review_evaluation_reviewer_identity_collision")
        if (
            self.comparison.packet_sha256 != self.packet_sha256
            or self.comparison.reviewer_a_submission_sha256 != reviewer_a.submission_sha256
            or self.comparison.reviewer_b_submission_sha256 != reviewer_b.submission_sha256
        ):
            raise ValueError("synthesis_review_evaluation_comparison_lineage_mismatch")
        by_a = {item.target_id: item for item in reviewer_a.decisions}
        by_b = {item.target_id: item for item in reviewer_b.decisions}
        if set(by_a) != set(by_b):
            raise ValueError("synthesis_review_evaluation_reviewer_target_roster_mismatch")
        expected_agreed = sorted(
            target_id
            for target_id in by_a
            if by_a[target_id].relationship is by_b[target_id].relationship
        )
        expected_support_divergence = sorted(
            target_id
            for target_id in expected_agreed
            if by_a[target_id].scientific_decision_sha256
            != by_b[target_id].scientific_decision_sha256
        )
        expected_disagreements = sorted(set(by_a) - set(expected_agreed))
        if (
            self.comparison.agreed_target_ids != expected_agreed
            or self.comparison.support_divergence_target_ids != expected_support_divergence
            or self.comparison.disagreement_target_ids != expected_disagreements
        ):
            raise ValueError("synthesis_review_evaluation_comparison_semantics_mismatch")

        adjudicator = self.adjudicator_submission
        by_adjudicator: dict[str, FrozenSynthesisAuthorizationReviewDecision] = {}
        if adjudicator is not None:
            if (
                adjudicator.reviewer_slot is not ReviewerSlot.ADJUDICATOR
                or adjudicator.packet_sha256 != self.packet_sha256
                or adjudicator.review_protocol_sha256 != self.review_protocol_sha256
                or adjudicator.input_transition_sha256 != self.comparison.transition_sha256
            ):
                raise ValueError("synthesis_review_evaluation_adjudicator_lineage_mismatch")
            if adjudicator.reviewer_identity_sha256 in {
                reviewer_a.reviewer_identity_sha256,
                reviewer_b.reviewer_identity_sha256,
            }:
                raise ValueError("synthesis_review_evaluation_adjudicator_identity_collision")
            by_adjudicator = {item.target_id: item for item in adjudicator.decisions}
            if set(by_adjudicator) != set(expected_disagreements):
                raise ValueError("synthesis_review_evaluation_adjudicator_target_roster_mismatch")
        elif not expected_disagreements and self.adjudication_template_sha256 is not None:
            raise ValueError("synthesis_review_evaluation_unnecessary_adjudication_template")

        resolution_ids = [item.target_id for item in self.resolutions]
        if resolution_ids != sorted(set(resolution_ids)):
            raise ValueError("synthesis_review_resolutions_not_sorted_unique")
        expected_resolution_ids = sorted(
            [*expected_agreed, *(expected_disagreements if adjudicator is not None else [])]
        )
        if resolution_ids != expected_resolution_ids:
            raise ValueError("synthesis_review_resolution_roster_mismatch")
        for resolution in self.resolutions:
            if resolution.target_id in by_adjudicator:
                assert adjudicator is not None
                expected_resolution = _freeze_resolution(
                    reviewer_a_decision=by_a[resolution.target_id],
                    reviewer_b_decision=by_b[resolution.target_id],
                    adjudicator_decision=by_adjudicator[resolution.target_id],
                    reviewer_identity_hashes=[
                        reviewer_a.reviewer_identity_sha256,
                        reviewer_b.reviewer_identity_sha256,
                        adjudicator.reviewer_identity_sha256,
                    ],
                    comparison_sha256=self.comparison.transition_sha256,
                )
            else:
                expected_resolution = _freeze_resolution(
                    reviewer_a_decision=by_a[resolution.target_id],
                    reviewer_b_decision=by_b[resolution.target_id],
                    reviewer_identity_hashes=[
                        reviewer_a.reviewer_identity_sha256,
                        reviewer_b.reviewer_identity_sha256,
                    ],
                    comparison_sha256=self.comparison.transition_sha256,
                )
            if resolution != expected_resolution:
                raise ValueError("synthesis_review_resolution_source_decision_mismatch")

        unit_ids = [item.synthesis_unit_id for item in self.unit_outcomes]
        if unit_ids != sorted(set(unit_ids)):
            raise ValueError("synthesis_review_evaluation_units_not_sorted_unique")
        outcome_targets = [
            target_id for outcome in self.unit_outcomes for target_id in outcome.target_ids
        ]
        if len(outcome_targets) != len(set(outcome_targets)) or set(outcome_targets) != set(by_a):
            raise ValueError("synthesis_review_evaluation_unit_target_roster_mismatch")
        for outcome in self.unit_outcomes:
            receipt = outcome.authorization_receipt
            if receipt is not None and (
                receipt.estimate_ids != outcome.estimate_ids
                or receipt.assertions != outcome.assertions
            ):
                raise ValueError("synthesis_review_evaluation_unit_receipt_projection_mismatch")
        awaiting = bool(
            self.comparison.disagreement_target_ids and self.adjudicator_submission is None
        )
        if (self.status == "awaiting_adjudication") != awaiting:
            raise ValueError("synthesis_review_evaluation_status_mismatch")
        if awaiting != (self.adjudication_template_sha256 is not None):
            raise ValueError("synthesis_review_adjudication_template_presence_mismatch")
        expected_transition = hash_canonical(
            {
                "comparison_transition_sha256": self.comparison.transition_sha256,
                "adjudicator_submission_sha256": (
                    self.adjudicator_submission.submission_sha256
                    if self.adjudicator_submission is not None
                    else None
                ),
                "resolution_sha256s": [item.resolution_sha256 for item in self.resolutions],
                "unit_review_sha256s": [item.unit_review_sha256 for item in self.unit_outcomes],
            }
        )
        if self.final_transition_sha256 != expected_transition:
            raise ValueError("synthesis_review_final_transition_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"evaluation_sha256"})
        if hash_canonical(payload) != self.evaluation_sha256:
            raise ValueError("synthesis_review_evaluation_hash_mismatch")
        return self


class SynthesisAuthorizationReviewPublicSummary(ContractModel):
    summary_version: Literal["synthesis-authorization-review-public-summary-v1"] = (
        "synthesis-authorization-review-public-summary-v1"
    )
    status: Literal["awaiting_adjudication", "complete"]
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: str
    synthesis_unit_count: Annotated[int, Field(ge=1)]
    review_target_count: Annotated[int, Field(ge=1)]
    same_cohort_target_count: Annotated[int, Field(ge=0)]
    independence_target_count: Annotated[int, Field(ge=0)]
    relationship_agreement_count: Annotated[int, Field(ge=0)]
    support_divergence_count: Annotated[int, Field(ge=0)]
    disagreement_count: Annotated[int, Field(ge=0)]
    adjudicated_count: Annotated[int, Field(ge=0)]
    resolved_unknown_count: Annotated[int, Field(ge=0)]
    resolved_contradiction_count: Annotated[int, Field(ge=0)]
    source_assertion_count: Annotated[int, Field(ge=0)]
    authorized_unit_count: Annotated[int, Field(ge=0)]
    abstained_unit_count: Annotated[int, Field(ge=0)]
    independent_reviewer_person_minutes: Annotated[float, Field(gt=0)]
    adjudication_person_minutes: Annotated[float, Field(ge=0)]
    total_person_minutes: Annotated[float, Field(gt=0)]
    median_minutes_per_reviewed_target: Annotated[float, Field(gt=0)]
    review_time_basis: Literal["timestamp_derived_person_minutes"] = (
        "timestamp_derived_person_minutes"
    )
    aggregate_only: Literal[True] = True
    contains_source_text: Literal[False] = False
    contains_source_identifiers: Literal[False] = False
    contains_publication_or_cohort_identifiers: Literal[False] = False
    contains_synthesis_unit_identifiers: Literal[False] = False
    contains_reviewer_identities: Literal[False] = False
    summary_sha256: str

    @field_validator("pipeline_fingerprint_sha256", "summary_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_public_sha256_invalid")

    @model_validator(mode="after")
    def validate_summary(self) -> SynthesisAuthorizationReviewPublicSummary:
        _validate_pipeline_binding(
            self.pipeline_fingerprint,
            self.pipeline_fingerprint_sha256,
            code="synthesis_review_public_pipeline_fingerprint_hash_mismatch",
        )
        if (
            self.same_cohort_target_count + self.independence_target_count
            != self.review_target_count
        ):
            raise ValueError("synthesis_review_public_target_count_mismatch")
        if self.authorized_unit_count + self.abstained_unit_count != self.synthesis_unit_count:
            raise ValueError("synthesis_review_public_unit_count_mismatch")
        if self.relationship_agreement_count + self.disagreement_count != self.review_target_count:
            raise ValueError("synthesis_review_public_agreement_count_mismatch")
        if self.support_divergence_count > self.relationship_agreement_count:
            raise ValueError("synthesis_review_public_support_divergence_count_invalid")
        if not math.isclose(
            self.independent_reviewer_person_minutes + self.adjudication_person_minutes,
            self.total_person_minutes,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("synthesis_review_public_time_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"summary_sha256"})
        if hash_canonical(payload) != self.summary_sha256:
            raise ValueError("synthesis_review_public_summary_hash_mismatch")
        return self


class PrivateReviewFileReference(ContractModel):
    role: Literal["packet", "reviewer_a_template", "reviewer_b_template"]
    path: Annotated[str, Field(min_length=1)]
    file_sha256: str
    object_sha256: str

    @field_validator("file_sha256", "object_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_manifest_file_sha256_invalid")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
            raise ValueError("synthesis_review_manifest_private_path_unsafe")
        return value


class SynthesisAuthorizationReviewManifest(ContractModel):
    manifest_version: Literal["private-synthesis-authorization-review-manifest-v1"] = (
        "private-synthesis-authorization-review-manifest-v1"
    )
    created_at: datetime
    input_corpus_sha256: str
    reconciliation_receipt_sha256: str
    reconciled_graph_sha256: str
    source_manifest_sha256: str
    packet_sha256: str
    review_protocol_sha256: str
    pipeline_fingerprint: PipelineFingerprint
    pipeline_fingerprint_sha256: str
    review_target_count: Annotated[int, Field(ge=1)]
    source_identity_visible: Literal[True] = True
    publication_source_content_visible: Literal[True] = True
    system_scores_included: Literal[False] = False
    benchmark_reference_labels_accessed: Literal[False] = False
    benchmark_review_verdicts_accessed: Literal[False] = False
    private_files: Annotated[list[PrivateReviewFileReference], Field(min_length=3, max_length=3)]
    manifest_sha256: str

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_utc(value, code="synthesis_review_manifest_created_at_not_utc")

    @field_validator(
        "input_corpus_sha256",
        "reconciliation_receipt_sha256",
        "reconciled_graph_sha256",
        "source_manifest_sha256",
        "packet_sha256",
        "review_protocol_sha256",
        "pipeline_fingerprint_sha256",
        "manifest_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, code="synthesis_review_manifest_sha256_invalid")

    @model_validator(mode="after")
    def validate_manifest(self) -> SynthesisAuthorizationReviewManifest:
        _validate_pipeline_binding(
            self.pipeline_fingerprint,
            self.pipeline_fingerprint_sha256,
            code="synthesis_review_manifest_pipeline_fingerprint_hash_mismatch",
        )
        roles = [item.role for item in self.private_files]
        if roles != sorted(set(roles)) or set(roles) != {
            "packet",
            "reviewer_a_template",
            "reviewer_b_template",
        }:
            raise ValueError("synthesis_review_manifest_roles_invalid")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if hash_canonical(payload) != self.manifest_sha256:
            raise ValueError("synthesis_review_manifest_hash_mismatch")
        return self


def _freeze_source_material(**values: Any) -> ReviewSourceMaterial:
    payload = dict(values)
    return ReviewSourceMaterial.model_validate(
        {**payload, "source_material_sha256": hash_canonical(payload)}
    )


def _freeze_cohort_source(**values: Any) -> OriginalCohortSourceMaterial:
    payload = dict(values)
    return OriginalCohortSourceMaterial.model_validate(
        {**payload, "cohort_source_sha256": hash_canonical(payload)}
    )


def _freeze_target(**values: Any) -> SynthesisAuthorizationReviewTarget:
    payload = dict(values)
    return SynthesisAuthorizationReviewTarget.model_validate(
        {**payload, "target_sha256": hash_canonical(payload)}
    )


def _freeze_unit(**values: Any) -> FrozenRequestedSynthesisUnit:
    payload = dict(values)
    return FrozenRequestedSynthesisUnit.model_validate(
        {**payload, "unit_sha256": hash_canonical(payload)}
    )


def freeze_synthesis_authorization_review_packet(
    *,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    request: SynthesisAuthorizationReviewRequest,
    repository_root: Path,
    created_at: datetime,
    review_protocol: SynthesisAuthorizationReviewProtocol | None = None,
) -> SynthesisAuthorizationReviewPacket:
    """Build the exact private target/source projection without model-side scores."""

    created_at = _validate_utc(created_at, code="synthesis_review_packet_created_at_not_utc")
    corpus = TypedEvidenceCorpus.model_validate(corpus.model_dump(mode="json"))
    reconciliation = reverify_native_cohort_reconciliation(corpus=corpus, receipt=reconciliation)
    request = SynthesisAuthorizationReviewRequest.model_validate(request.model_dump(mode="json"))
    protocol = review_protocol or default_synthesis_authorization_review_protocol()
    protocol = SynthesisAuthorizationReviewProtocol.model_validate(protocol.model_dump(mode="json"))
    pipeline_fingerprint = compute_synthesis_authorization_review_pipeline_fingerprint(
        repository_root
    )
    graph = reconciliation.reconciled_graph
    if graph is None or reconciliation.reconciled_graph_sha256 is None:
        raise SynthesisAuthorizationReviewError("synthesis_review_requires_reconciled_graph")

    estimate_by_id = {item.estimate_id: item for item in graph.outcome_estimates}
    contrast_to_cohort = {item.contrast_id: item.cohort_id for item in graph.contrasts}
    group_by_canonical = {item.canonical_id: item for item in reconciliation.cohort_groups}
    original_cohort_by_id = {item.cohort_id: item for item in corpus.graph.cohorts}
    original_study_by_id = {item.study_id: item for item in corpus.graph.studies}
    fragment_by_publication = {item.publication_id: item for item in corpus.fragments}

    targets: list[SynthesisAuthorizationReviewTarget] = []
    frozen_units: list[FrozenRequestedSynthesisUnit] = []
    needed_original_cohorts: set[str] = set()
    for requested in request.synthesis_units:
        if any(item not in estimate_by_id for item in requested.estimate_ids):
            raise SynthesisAuthorizationReviewError("synthesis_review_request_estimate_unknown")
        canonical_ids = sorted(
            {
                contrast_to_cohort[estimate_by_id[estimate_id].contrast_id]
                for estimate_id in requested.estimate_ids
            }
        )
        unit_targets: list[SynthesisAuthorizationReviewTarget] = []
        for canonical_id in canonical_ids:
            group = group_by_canonical.get(canonical_id)
            if group is None:
                raise SynthesisAuthorizationReviewError(
                    "synthesis_review_canonical_cohort_missing_from_reconciliation"
                )
            needed_original_cohorts.update(group.member_ids)
            if len(group.member_ids) > 1:
                target_payload = {
                    "synthesis_unit_id": requested.synthesis_unit_id,
                    "target_kind": ReviewTargetKind.MERGED_COHORT_IDENTITY,
                    "required_relationship": "same_cohort",
                    "assertion_cohort_ids": group.member_ids,
                    "canonical_cohort_ids": [canonical_id],
                    "original_cohort_ids": group.member_ids,
                }
                publication_ids = sorted(
                    {
                        original_study_by_id[
                            original_cohort_by_id[item].study_id
                        ].primary_publication_id
                        for item in group.member_ids
                    }
                )
                target_id = f"review-target-{hash_canonical(target_payload)[:24]}"
                unit_targets.append(
                    _freeze_target(
                        target_id=target_id,
                        **target_payload,
                        publication_ids=publication_ids,
                    )
                )
        for left, right in combinations(canonical_ids, 2):
            original_ids = sorted(
                [*group_by_canonical[left].member_ids, *group_by_canonical[right].member_ids]
            )
            needed_original_cohorts.update(original_ids)
            publication_ids = sorted(
                {
                    original_study_by_id[
                        original_cohort_by_id[item].study_id
                    ].primary_publication_id
                    for item in original_ids
                }
            )
            target_payload = {
                "synthesis_unit_id": requested.synthesis_unit_id,
                "target_kind": ReviewTargetKind.CANONICAL_COHORT_INDEPENDENCE,
                "required_relationship": "independent_cohorts",
                "assertion_cohort_ids": [left, right],
                "canonical_cohort_ids": [left, right],
                "original_cohort_ids": original_ids,
            }
            target_id = f"review-target-{hash_canonical(target_payload)[:24]}"
            unit_targets.append(
                _freeze_target(
                    target_id=target_id,
                    **target_payload,
                    publication_ids=publication_ids,
                )
            )
        unit_targets.sort(key=lambda item: item.target_id)
        if not unit_targets:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_request_has_no_review_targets"
            )
        targets.extend(unit_targets)
        frozen_units.append(
            _freeze_unit(
                synthesis_unit_id=requested.synthesis_unit_id,
                estimate_ids=requested.estimate_ids,
                canonical_cohort_ids=canonical_ids,
                target_ids=[item.target_id for item in unit_targets],
            )
        )

    target_ids = [item.target_id for item in targets]
    if len(target_ids) != len(set(target_ids)):
        raise SynthesisAuthorizationReviewError("synthesis_review_target_identity_collision")

    publication_by_original_cohort: dict[str, str] = {}
    canonical_by_original: dict[str, str] = {}
    for canonical, group in group_by_canonical.items():
        for member in group.member_ids:
            canonical_by_original[member] = canonical
    for cohort_id in sorted(needed_original_cohorts):
        cohort = original_cohort_by_id[cohort_id]
        study = original_study_by_id[cohort.study_id]
        if len(study.publication_ids) != 1:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_original_study_publication_identity_ambiguous"
            )
        publication_by_original_cohort[cohort_id] = study.primary_publication_id

    source_materials: list[ReviewSourceMaterial] = []
    source_id_by_publication: dict[str, str] = {}
    source_by_publication: dict[str, ReviewSourceMaterial] = {}
    for publication_id in sorted(set(publication_by_original_cohort.values())):
        fragment = fragment_by_publication.get(publication_id)
        if fragment is None or fragment.grounding_receipt_sha256 is None:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_source_fragment_or_grounding_missing"
            )
        source = resolve_native_source_document(
            repository_root=repository_root,
            source_document=fragment.source_document,
        )
        source_identity = {
            "publication_id": publication_id,
            "source_payload_sha256": source.source_payload_sha256,
        }
        source_material_id = f"review-source-{hash_canonical(source_identity)[:20]}"
        material = _freeze_source_material(
            source_material_id=source_material_id,
            publication_id=publication_id,
            artifact_path=fragment.source_document.artifact_path,
            source_kind=source.source_kind,
            source_document_sha256=fragment.source_document.sha256,
            grounding_receipt_sha256=fragment.grounding_receipt_sha256,
            source_locator=source.source_locator,
            source_payload_sha256=source.source_payload_sha256,
            lines=source.lines,
        )
        source_materials.append(material)
        source_id_by_publication[publication_id] = source_material_id
        source_by_publication[publication_id] = material

    cohort_sources: list[OriginalCohortSourceMaterial] = []
    for cohort_id in sorted(needed_original_cohorts):
        cohort = original_cohort_by_id[cohort_id]
        publication_id = publication_by_original_cohort[cohort_id]
        material = source_by_publication[publication_id]
        cohort_sources.append(
            _freeze_cohort_source(
                original_cohort_id=cohort_id,
                canonical_cohort_id=canonical_by_original[cohort_id],
                publication_id=publication_id,
                source_material_id=source_id_by_publication[publication_id],
                source_material_sha256=material.source_material_sha256,
                registry_ids=cohort.identity.registry_ids,
                dataset_ids=cohort.identity.dataset_ids,
            )
        )

    targets.sort(key=lambda item: item.target_id)
    frozen_units.sort(key=lambda item: item.synthesis_unit_id)
    source_materials.sort(key=lambda item: item.source_material_id)
    source_manifest_sha256 = hash_canonical(
        {"cohort_sources": cohort_sources, "source_materials": source_materials}
    )
    payload = {
        "packet_version": "private-synthesis-authorization-review-packet-v1",
        "created_at": _utc_json(created_at),
        "input_corpus_sha256": corpus.corpus_sha256,
        "reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "request_sha256": request.request_sha256,
        "review_protocol": protocol,
        "pipeline_fingerprint": pipeline_fingerprint,
        "pipeline_fingerprint_sha256": pipeline_fingerprint.pipeline_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "requested_units": frozen_units,
        "targets": targets,
        "cohort_sources": cohort_sources,
        "source_materials": source_materials,
        "source_identity_visible": True,
        "publication_source_content_visible": True,
        "system_scores_included": False,
        "benchmark_reference_labels_accessed": False,
        "benchmark_review_verdicts_accessed": False,
    }
    return SynthesisAuthorizationReviewPacket.model_validate(
        {**payload, "packet_sha256": hash_canonical(payload)}
    )


def _citation_slot(
    cohort: OriginalCohortSourceMaterial,
    source: ReviewSourceMaterial,
) -> ReviewCitationForm:
    return ReviewCitationForm(
        publication_id=cohort.publication_id,
        original_cohort_id=cohort.original_cohort_id,
        source_document_sha256=source.source_document_sha256,
        grounding_receipt_sha256=source.grounding_receipt_sha256,
        source_locator=source.source_locator,
        source_payload_sha256=source.source_payload_sha256,
        eligible_source_identifiers=sorted({*cohort.registry_ids, *cohort.dataset_ids}),
    )


def freeze_blank_synthesis_authorization_review_template(
    *,
    packet: SynthesisAuthorizationReviewPacket,
    reviewer_slot: ReviewerSlot,
    target_ids: list[str] | None = None,
    input_transition_sha256: str | None = None,
) -> BlankSynthesisAuthorizationReviewTemplate:
    packet = SynthesisAuthorizationReviewPacket.model_validate(packet.model_dump(mode="json"))
    selected_ids = sorted(target_ids or [item.target_id for item in packet.targets])
    target_by_id = {item.target_id: item for item in packet.targets}
    if not selected_ids or any(item not in target_by_id for item in selected_ids):
        raise SynthesisAuthorizationReviewError("synthesis_review_template_target_set_invalid")
    cohort_by_id = {item.original_cohort_id: item for item in packet.cohort_sources}
    source_by_id = {item.source_material_id: item for item in packet.source_materials}
    decisions = []
    for target_id in selected_ids:
        target = target_by_id[target_id]
        citations = [
            _citation_slot(
                cohort_by_id[cohort_id],
                source_by_id[cohort_by_id[cohort_id].source_material_id],
            )
            for cohort_id in target.original_cohort_ids
        ]
        decisions.append(
            ReviewDecisionForm(
                target_id=target_id,
                synthesis_unit_id=target.synthesis_unit_id,
                target_kind=target.target_kind,
                relationship_under_review=target.required_relationship,
                assertion_cohort_ids=target.assertion_cohort_ids,
                original_cohort_ids=target.original_cohort_ids,
                citations=citations,
            )
        )
    payload = {
        "template_version": "synthesis-authorization-review-template-v1",
        "packet_sha256": packet.packet_sha256,
        "review_protocol_sha256": packet.review_protocol.protocol_sha256,
        "reviewer_slot": reviewer_slot,
        "input_transition_sha256": input_transition_sha256 or packet.packet_sha256,
        "reviewer_identity_sha256": None,
        "submitted_at": None,
        "decisions": decisions,
    }
    return BlankSynthesisAuthorizationReviewTemplate.model_validate(
        {**payload, "template_sha256": hash_canonical(payload)}
    )


def prepare_synthesis_authorization_review(
    *,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    request: SynthesisAuthorizationReviewRequest,
    repository_root: Path,
    output_dir: Path,
    created_at: datetime,
    review_protocol: SynthesisAuthorizationReviewProtocol | None = None,
) -> SynthesisAuthorizationReviewManifest:
    """Freeze a new ignored private workspace; existing paths are never overwritten."""

    private_root = repository_root / "data/cache/synthesis-authorization-review"
    output_dir = _strictly_under(
        output_dir,
        private_root,
        code="synthesis_review_private_output_outside_ignored_root",
    )
    if output_dir.exists():
        raise SynthesisAuthorizationReviewError("synthesis_review_output_directory_exists")
    packet = freeze_synthesis_authorization_review_packet(
        corpus=corpus,
        reconciliation=reconciliation,
        request=request,
        repository_root=repository_root,
        created_at=created_at,
        review_protocol=review_protocol,
    )
    reviewer_a = freeze_blank_synthesis_authorization_review_template(
        packet=packet,
        reviewer_slot=ReviewerSlot.REVIEWER_A,
    )
    reviewer_b = freeze_blank_synthesis_authorization_review_template(
        packet=packet,
        reviewer_slot=ReviewerSlot.REVIEWER_B,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "packet": output_dir / "packet.private.json",
        "reviewer_a_template": output_dir / "reviewer-a-decisions.private.json",
        "reviewer_b_template": output_dir / "reviewer-b-decisions.private.json",
    }
    atomic_write_json(paths["packet"], packet)
    atomic_write_json(paths["reviewer_a_template"], reviewer_a)
    atomic_write_json(paths["reviewer_b_template"], reviewer_b)
    object_hashes = {
        "packet": packet.packet_sha256,
        "reviewer_a_template": reviewer_a.template_sha256,
        "reviewer_b_template": reviewer_b.template_sha256,
    }
    references = [
        PrivateReviewFileReference(
            role=role,
            path=paths[role].name,
            file_sha256=sha256_file(paths[role]),
            object_sha256=object_hashes[role],
        )
        for role in sorted(paths)
    ]
    manifest_payload = {
        "manifest_version": "private-synthesis-authorization-review-manifest-v1",
        "created_at": _utc_json(packet.created_at),
        "input_corpus_sha256": packet.input_corpus_sha256,
        "reconciliation_receipt_sha256": packet.reconciliation_receipt_sha256,
        "reconciled_graph_sha256": packet.reconciled_graph_sha256,
        "source_manifest_sha256": packet.source_manifest_sha256,
        "packet_sha256": packet.packet_sha256,
        "review_protocol_sha256": packet.review_protocol.protocol_sha256,
        "pipeline_fingerprint": packet.pipeline_fingerprint,
        "pipeline_fingerprint_sha256": packet.pipeline_fingerprint_sha256,
        "review_target_count": len(packet.targets),
        "source_identity_visible": True,
        "publication_source_content_visible": True,
        "system_scores_included": False,
        "benchmark_reference_labels_accessed": False,
        "benchmark_review_verdicts_accessed": False,
        "private_files": references,
    }
    manifest = SynthesisAuthorizationReviewManifest.model_validate(
        {**manifest_payload, "manifest_sha256": hash_canonical(manifest_payload)}
    )
    atomic_write_json(output_dir / "manifest.private.json", manifest)
    return manifest


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SynthesisAuthorizationReviewError(f"{code}_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SynthesisAuthorizationReviewError(f"{code}_json_invalid") from exc
    if not isinstance(value, dict):
        raise SynthesisAuthorizationReviewError(f"{code}_json_requires_object")
    return value


def verify_synthesis_authorization_review_manifest(
    *, manifest_path: Path, repository_root: Path
) -> tuple[
    SynthesisAuthorizationReviewManifest,
    SynthesisAuthorizationReviewPacket,
    BlankSynthesisAuthorizationReviewTemplate,
    BlankSynthesisAuthorizationReviewTemplate,
]:
    """Rehash the immutable packet and both blank templates without following links."""

    manifest = SynthesisAuthorizationReviewManifest.model_validate(
        _read_object(manifest_path, code="synthesis_review_manifest")
    )
    _require_current_pipeline_binding(
        repository_root=repository_root,
        fingerprint=manifest.pipeline_fingerprint,
        fingerprint_sha256=manifest.pipeline_fingerprint_sha256,
    )
    root = manifest_path.parent.resolve(strict=True)
    by_role: dict[str, dict[str, Any]] = {}
    for reference in manifest.private_files:
        candidate = manifest_path.parent / reference.path
        if candidate.is_symlink():
            raise SynthesisAuthorizationReviewError("synthesis_review_private_file_symlink")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_private_file_missing"
            ) from exc
        if root not in resolved.parents or not resolved.is_file():
            raise SynthesisAuthorizationReviewError("synthesis_review_private_file_path_unsafe")
        if sha256_file(resolved) != reference.file_sha256:
            raise SynthesisAuthorizationReviewError("synthesis_review_private_file_hash_mismatch")
        by_role[reference.role] = _read_object(resolved, code="synthesis_review_private_file")
    packet = SynthesisAuthorizationReviewPacket.model_validate(by_role["packet"])
    reviewer_a = BlankSynthesisAuthorizationReviewTemplate.model_validate(
        by_role["reviewer_a_template"]
    )
    reviewer_b = BlankSynthesisAuthorizationReviewTemplate.model_validate(
        by_role["reviewer_b_template"]
    )
    if (
        packet.pipeline_fingerprint != manifest.pipeline_fingerprint
        or packet.pipeline_fingerprint_sha256 != manifest.pipeline_fingerprint_sha256
    ):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_manifest_packet_pipeline_mismatch"
        )
    if (
        packet.packet_sha256 != manifest.packet_sha256
        or reviewer_a.template_sha256
        != next(
            item.object_sha256
            for item in manifest.private_files
            if item.role == "reviewer_a_template"
        )
        or reviewer_b.template_sha256
        != next(
            item.object_sha256
            for item in manifest.private_files
            if item.role == "reviewer_b_template"
        )
        or packet.packet_sha256
        != next(item.object_sha256 for item in manifest.private_files if item.role == "packet")
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_manifest_object_hash_mismatch")
    if (
        reviewer_a.reviewer_slot is not ReviewerSlot.REVIEWER_A
        or reviewer_b.reviewer_slot is not ReviewerSlot.REVIEWER_B
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_manifest_reviewer_slot_mismatch")
    if any(
        value != manifest.review_protocol_sha256
        for value in (
            packet.review_protocol.protocol_sha256,
            reviewer_a.review_protocol_sha256,
            reviewer_b.review_protocol_sha256,
        )
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_manifest_protocol_mismatch")
    if any(
        item != packet.packet_sha256
        for item in (reviewer_a.input_transition_sha256, reviewer_b.input_transition_sha256)
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_initial_transition_mismatch")
    return manifest, packet, reviewer_a, reviewer_b


def reverify_synthesis_authorization_review_packet(
    *,
    packet: SynthesisAuthorizationReviewPacket,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    repository_root: Path,
) -> SynthesisAuthorizationReviewPacket:
    """Replay request projection and current source bytes, requiring exact packet equality."""

    request = freeze_synthesis_authorization_review_request(
        [
            RequestedSynthesisUnit(
                synthesis_unit_id=item.synthesis_unit_id,
                estimate_ids=item.estimate_ids,
            )
            for item in packet.requested_units
        ]
    )
    if request.request_sha256 != packet.request_sha256:
        raise SynthesisAuthorizationReviewError("synthesis_review_packet_request_replay_mismatch")
    replayed = freeze_synthesis_authorization_review_packet(
        corpus=corpus,
        reconciliation=reconciliation,
        request=request,
        repository_root=repository_root,
        created_at=packet.created_at,
        review_protocol=packet.review_protocol,
    )
    if replayed != packet:
        raise SynthesisAuthorizationReviewError("synthesis_review_packet_replay_mismatch")
    return replayed


def _citation_is_blank(citation: ReviewCitationForm) -> bool:
    return citation.quote is None and not citation.line_ids and citation.cited_identifier is None


def _freeze_review_citation(
    *,
    citation: ReviewCitationForm,
    expected: ReviewCitationForm,
    source: ReviewSourceMaterial,
) -> SourceIdentityCitation:
    immutable = (
        "publication_id",
        "original_cohort_id",
        "source_document_sha256",
        "grounding_receipt_sha256",
        "source_locator",
        "source_payload_sha256",
        "eligible_source_identifiers",
    )
    if any(getattr(citation, field) != getattr(expected, field) for field in immutable):
        raise SynthesisAuthorizationReviewError("synthesis_review_citation_prefill_tampered")
    if (
        citation.quote is None
        or not citation.quote.strip()
        or not citation.line_ids
        or not citation.cited_identifier
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_citation_incomplete")
    line_by_id = {item.line_id: item for item in source.lines}
    if any(line_id not in line_by_id for line_id in citation.line_ids):
        raise SynthesisAuthorizationReviewError("synthesis_review_citation_line_unknown")
    cited_lines = [line_by_id[line_id] for line_id in citation.line_ids]
    cited_text = "\n".join(item.text for item in cited_lines)
    if citation.quote not in cited_text:
        raise SynthesisAuthorizationReviewError("synthesis_review_citation_quote_not_exact")
    normalized_identifier = _normalize_identifier(citation.cited_identifier)
    supported = {_normalize_identifier(item) for item in citation.eligible_source_identifiers}
    if normalized_identifier not in supported:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_citation_identifier_not_source_reported"
        )
    normalized_text = _normalize_identifier(cited_text)
    if (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_identifier)}(?![a-z0-9])",
            normalized_text,
        )
        is None
    ):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_citation_identifier_outside_exact_lines"
        )
    return freeze_source_identity_citation(
        publication_id=citation.publication_id,
        original_cohort_id=citation.original_cohort_id,
        source_document_sha256=citation.source_document_sha256,
        grounding_receipt_sha256=citation.grounding_receipt_sha256,
        source_locator=citation.source_locator,
        quote=citation.quote,
        line_ids=citation.line_ids,
        cited_identifier=citation.cited_identifier,
        source_payload_sha256=citation.source_payload_sha256,
        cited_lines_sha256=hash_canonical(cited_lines),
        cited_text_sha256=hashlib.sha256(cited_text.encode("utf-8")).hexdigest(),
    )


def freeze_completed_synthesis_authorization_review_submission(
    *,
    packet: SynthesisAuthorizationReviewPacket,
    template: BlankSynthesisAuthorizationReviewTemplate,
    form: SubmittedSynthesisAuthorizationReviewForm,
    not_before: datetime | None = None,
) -> FrozenSynthesisAuthorizationReviewSubmission:
    """Validate one edited template and freeze its exact source-replayed decisions."""

    if (
        form.packet_sha256 != packet.packet_sha256
        or form.template_sha256 != template.template_sha256
        or form.review_protocol_sha256 != packet.review_protocol.protocol_sha256
        or form.reviewer_slot is not template.reviewer_slot
        or form.input_transition_sha256 != template.input_transition_sha256
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_submission_lineage_mismatch")
    form_target_ids = [item.target_id for item in form.decisions]
    template_target_ids = [item.target_id for item in template.decisions]
    if form_target_ids != template_target_ids:
        raise SynthesisAuthorizationReviewError("synthesis_review_submission_target_set_mismatch")
    target_by_id = {item.target_id: item for item in packet.targets}
    blank_by_target = {item.target_id: item for item in template.decisions}
    source_by_id = {item.source_material_id: item for item in packet.source_materials}
    cohort_by_id = {item.original_cohort_id: item for item in packet.cohort_sources}
    frozen_decisions: list[FrozenSynthesisAuthorizationReviewDecision] = []
    for decision in form.decisions:
        target = target_by_id[decision.target_id]
        blank = blank_by_target[decision.target_id]
        expected_target_fields = {
            "synthesis_unit_id": target.synthesis_unit_id,
            "target_kind": target.target_kind,
            "relationship_under_review": target.required_relationship,
            "assertion_cohort_ids": target.assertion_cohort_ids,
            "original_cohort_ids": target.original_cohort_ids,
        }
        if any(
            getattr(decision, field) != expected
            for field, expected in expected_target_fields.items()
        ):
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_decision_target_prefill_tampered"
            )
        if (
            decision.relationship is None
            or decision.rationale is None
            or not decision.rationale.strip()
        ):
            raise SynthesisAuthorizationReviewError("synthesis_review_decision_incomplete")
        if (
            decision.review_started_at is None
            or decision.review_completed_at is None
            or decision.review_minutes is None
        ):
            raise SynthesisAuthorizationReviewError("synthesis_review_decision_timing_incomplete")
        started = _validate_utc(
            decision.review_started_at, code="synthesis_review_decision_started_at_not_utc"
        )
        completed = _validate_utc(
            decision.review_completed_at, code="synthesis_review_decision_completed_at_not_utc"
        )
        threshold = not_before or packet.created_at
        if started < threshold:
            raise SynthesisAuthorizationReviewError("synthesis_review_decision_precedes_transition")
        if [item.original_cohort_id for item in decision.citations] != target.original_cohort_ids:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_decision_original_cohort_roster_mismatch"
            )
        blank_citations = {item.original_cohort_id: item for item in blank.citations}
        built: list[SourceIdentityCitation] = []
        blank_flags = [_citation_is_blank(item) for item in decision.citations]
        if any(blank_flags) and not all(blank_flags):
            raise SynthesisAuthorizationReviewError("synthesis_review_partial_citation_roster")
        if decision.relationship is not ReviewRelationship.UNKNOWN and any(blank_flags):
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_affirmative_decision_requires_all_citations"
            )
        if not all(blank_flags):
            for citation in decision.citations:
                cohort = cohort_by_id[citation.original_cohort_id]
                source = source_by_id[cohort.source_material_id]
                built.append(
                    _freeze_review_citation(
                        citation=citation,
                        expected=blank_citations[citation.original_cohort_id],
                        source=source,
                    )
                )
        built.sort(key=lambda item: item.citation_sha256)
        scientific_payload = {
            "target_id": decision.target_id,
            "relationship": decision.relationship,
            "rationale": decision.rationale.strip(),
            "citations": built,
        }
        decision_payload = {
            **scientific_payload,
            "review_started_at": _utc_json(started),
            "review_completed_at": _utc_json(completed),
            "review_minutes": decision.review_minutes,
            "scientific_decision_sha256": hash_canonical(scientific_payload),
        }
        frozen_decisions.append(
            FrozenSynthesisAuthorizationReviewDecision.model_validate(
                {**decision_payload, "decision_sha256": hash_canonical(decision_payload)}
            )
        )
    frozen_decisions.sort(key=lambda item: item.target_id)
    payload = {
        "submission_version": "synthesis-authorization-review-submission-v1",
        "packet_sha256": packet.packet_sha256,
        "template_sha256": template.template_sha256,
        "review_protocol_sha256": packet.review_protocol.protocol_sha256,
        "reviewer_slot": form.reviewer_slot,
        "reviewer_identity_sha256": form.reviewer_identity_sha256,
        "input_transition_sha256": form.input_transition_sha256,
        "submitted_at": _utc_json(form.submitted_at),
        "submitted_form_payload_sha256": hash_canonical(form),
        "decisions": frozen_decisions,
    }
    return FrozenSynthesisAuthorizationReviewSubmission.model_validate(
        {**payload, "submission_sha256": hash_canonical(payload)}
    )


def _freeze_comparison(
    *,
    packet: SynthesisAuthorizationReviewPacket,
    reviewer_a: FrozenSynthesisAuthorizationReviewSubmission,
    reviewer_b: FrozenSynthesisAuthorizationReviewSubmission,
) -> SynthesisAuthorizationReviewComparison:
    by_a = {item.target_id: item for item in reviewer_a.decisions}
    by_b = {item.target_id: item for item in reviewer_b.decisions}
    if set(by_a) != set(by_b) or set(by_a) != {item.target_id for item in packet.targets}:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_comparison_target_roster_mismatch"
        )
    agreed = sorted(
        target_id
        for target_id in by_a
        if by_a[target_id].relationship is by_b[target_id].relationship
    )
    support_divergence = sorted(
        target_id
        for target_id in agreed
        if by_a[target_id].scientific_decision_sha256 != by_b[target_id].scientific_decision_sha256
    )
    disagreements = sorted(set(by_a) - set(agreed))
    payload = {
        "transition_version": "synthesis-authorization-review-comparison-v1",
        "packet_sha256": packet.packet_sha256,
        "reviewer_a_submission_sha256": reviewer_a.submission_sha256,
        "reviewer_b_submission_sha256": reviewer_b.submission_sha256,
        "agreed_target_ids": agreed,
        "support_divergence_target_ids": support_divergence,
        "disagreement_target_ids": disagreements,
        "predecessor_transition_sha256": packet.packet_sha256,
    }
    return SynthesisAuthorizationReviewComparison.model_validate(
        {**payload, "transition_sha256": hash_canonical(payload)}
    )


def _freeze_resolution_source_decision(
    *,
    reviewer_slot: ReviewerSlot,
    decision: FrozenSynthesisAuthorizationReviewDecision,
) -> ResolutionSourceDecision:
    payload = {
        "reviewer_slot": reviewer_slot,
        "decision_sha256": decision.decision_sha256,
        "scientific_decision_sha256": decision.scientific_decision_sha256,
        "relationship": decision.relationship,
        "rationale": decision.rationale,
        "citation_sha256s": [item.citation_sha256 for item in decision.citations],
    }
    return ResolutionSourceDecision.model_validate(
        {**payload, "reference_sha256": hash_canonical(payload)}
    )


def _freeze_resolution(
    *,
    reviewer_a_decision: FrozenSynthesisAuthorizationReviewDecision,
    reviewer_b_decision: FrozenSynthesisAuthorizationReviewDecision,
    reviewer_identity_hashes: list[str],
    comparison_sha256: str,
    adjudicator_decision: FrozenSynthesisAuthorizationReviewDecision | None = None,
) -> ResolvedSynthesisAuthorizationReviewDecision:
    if reviewer_a_decision.target_id != reviewer_b_decision.target_id:
        raise SynthesisAuthorizationReviewError("synthesis_review_resolution_target_mismatch")
    references = [
        _freeze_resolution_source_decision(
            reviewer_slot=ReviewerSlot.REVIEWER_A,
            decision=reviewer_a_decision,
        ),
        _freeze_resolution_source_decision(
            reviewer_slot=ReviewerSlot.REVIEWER_B,
            decision=reviewer_b_decision,
        ),
    ]
    if adjudicator_decision is None:
        if reviewer_a_decision.relationship is not reviewer_b_decision.relationship:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_resolution_requires_adjudication_for_label_disagreement"
            )
        resolution_source = "independent_agreement"
        relationship = reviewer_a_decision.relationship
        citation_by_hash = {
            item.citation_sha256: item
            for decision in (reviewer_a_decision, reviewer_b_decision)
            for item in decision.citations
        }
        citations = [citation_by_hash[key] for key in sorted(citation_by_hash)]
        rationale = _combined_agreement_rationale(references)
    else:
        if adjudicator_decision.target_id != reviewer_a_decision.target_id:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_adjudication_resolution_target_mismatch"
            )
        resolution_source = "third_adjudication"
        relationship = adjudicator_decision.relationship
        citations = adjudicator_decision.citations
        rationale = adjudicator_decision.rationale
        references.append(
            _freeze_resolution_source_decision(
                reviewer_slot=ReviewerSlot.ADJUDICATOR,
                decision=adjudicator_decision,
            )
        )
    panel_identity = hash_canonical(
        {
            "reviewer_identity_sha256s": sorted(reviewer_identity_hashes),
            "comparison_transition_sha256": comparison_sha256,
            "target_id": reviewer_a_decision.target_id,
            "resolution_source": resolution_source,
        }
    )
    payload = {
        "target_id": reviewer_a_decision.target_id,
        "resolution_source": resolution_source,
        "relationship": relationship,
        "rationale": rationale,
        "citations": citations,
        "source_decisions": references,
        "support_evidence_diverged": (
            reviewer_a_decision.scientific_decision_sha256
            != reviewer_b_decision.scientific_decision_sha256
        ),
        "panel_identity_sha256": panel_identity,
    }
    return ResolvedSynthesisAuthorizationReviewDecision.model_validate(
        {**payload, "resolution_sha256": hash_canonical(payload)}
    )


def _unit_outcomes(
    *,
    packet: SynthesisAuthorizationReviewPacket,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    repository_root: Path,
    resolutions: list[ResolvedSynthesisAuthorizationReviewDecision],
    unresolved_target_ids: set[str],
) -> list[SynthesisAuthorizationReviewedUnit]:
    target_by_id = {item.target_id: item for item in packet.targets}
    resolution_by_id = {item.target_id: item for item in resolutions}
    outcomes: list[SynthesisAuthorizationReviewedUnit] = []
    for unit in packet.requested_units:
        assertions: list[SourceIdentityAssertion] = []
        blockers: set[str] = set()
        unresolved_same_cohort = False
        for target_id in unit.target_ids:
            target = target_by_id[target_id]
            resolution = resolution_by_id.get(target_id)
            if target_id in unresolved_target_ids or resolution is None:
                blockers.add("unresolved_reviewer_disagreement")
                unresolved_same_cohort |= target.required_relationship == "same_cohort"
                continue
            if resolution.relationship is ReviewRelationship.UNKNOWN:
                blockers.add("source_relationship_unknown")
                unresolved_same_cohort |= target.required_relationship == "same_cohort"
                continue
            if resolution.relationship.value != target.required_relationship:
                blockers.add("source_relationship_contradicts_reconciled_target")
                unresolved_same_cohort |= target.required_relationship == "same_cohort"
                continue
            assertion = freeze_source_identity_assertion(
                relationship=target.required_relationship,
                cohort_ids=target.assertion_cohort_ids,
                rationale=resolution.rationale,
                citations=resolution.citations,
                reviewer_identity_sha256=resolution.panel_identity_sha256,
                review_protocol_sha256=packet.review_protocol.protocol_sha256,
            )
            assertions.append(assertion)
        assertions.sort(key=lambda item: item.assertion_sha256)
        receipt = None
        if not unresolved_same_cohort:
            receipt = authorize_synthesis_unit(
                corpus=corpus,
                reconciliation=reconciliation,
                estimate_ids=unit.estimate_ids,
                assertions=assertions,
                repository_root=repository_root,
            )
            if not receipt.authorizes_synthesis_input:
                blockers.add("source_independence_unresolved")
        payload = {
            "synthesis_unit_id": unit.synthesis_unit_id,
            "estimate_ids": unit.estimate_ids,
            "target_ids": unit.target_ids,
            "assertions": assertions,
            "blocker_codes": sorted(blockers),
            "authorization_receipt": receipt,
            "authorizes_synthesis_input": bool(
                receipt is not None and receipt.authorizes_synthesis_input and not blockers
            ),
        }
        outcomes.append(
            SynthesisAuthorizationReviewedUnit.model_validate(
                {**payload, "unit_review_sha256": hash_canonical(payload)}
            )
        )
    return outcomes


def evaluate_synthesis_authorization_review(
    *,
    manifest_path: Path,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    repository_root: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    adjudicator_path: Path | None = None,
) -> tuple[
    PrivateSynthesisAuthorizationReviewEvaluation,
    SynthesisAuthorizationReviewPublicSummary,
    BlankSynthesisAuthorizationReviewTemplate | None,
]:
    """Replay all private inputs and return private state, aggregate public state, and conflicts."""

    private_root = repository_root / "data/cache/synthesis-authorization-review"
    for path, code in (
        (manifest_path, "synthesis_review_manifest_outside_ignored_root"),
        (reviewer_a_path, "synthesis_review_reviewer_a_outside_ignored_root"),
        (reviewer_b_path, "synthesis_review_reviewer_b_outside_ignored_root"),
    ):
        _strictly_under(path, private_root, code=code)
    if adjudicator_path is not None:
        _strictly_under(
            adjudicator_path,
            private_root,
            code="synthesis_review_adjudicator_outside_ignored_root",
        )
    manifest, packet, reviewer_a_template, reviewer_b_template = (
        verify_synthesis_authorization_review_manifest(
            manifest_path=manifest_path,
            repository_root=repository_root,
        )
    )
    packet = reverify_synthesis_authorization_review_packet(
        packet=packet,
        corpus=corpus,
        reconciliation=reconciliation,
        repository_root=repository_root,
    )
    if (
        corpus.corpus_sha256 != manifest.input_corpus_sha256
        or reconciliation.receipt_sha256 != manifest.reconciliation_receipt_sha256
    ):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_manifest_scientific_input_mismatch"
        )
    reviewer_a_form = SubmittedSynthesisAuthorizationReviewForm.model_validate(
        _read_object(reviewer_a_path, code="synthesis_review_reviewer_a")
    )
    reviewer_b_form = SubmittedSynthesisAuthorizationReviewForm.model_validate(
        _read_object(reviewer_b_path, code="synthesis_review_reviewer_b")
    )
    reviewer_a = freeze_completed_synthesis_authorization_review_submission(
        packet=packet,
        template=reviewer_a_template,
        form=reviewer_a_form,
    )
    reviewer_b = freeze_completed_synthesis_authorization_review_submission(
        packet=packet,
        template=reviewer_b_template,
        form=reviewer_b_form,
    )
    if reviewer_a.reviewer_identity_sha256 == reviewer_b.reviewer_identity_sha256:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_independent_reviewers_not_distinct"
        )
    comparison = _freeze_comparison(packet=packet, reviewer_a=reviewer_a, reviewer_b=reviewer_b)
    adjudication_template = None
    adjudicator = None
    if comparison.disagreement_target_ids:
        adjudication_template = freeze_blank_synthesis_authorization_review_template(
            packet=packet,
            reviewer_slot=ReviewerSlot.ADJUDICATOR,
            target_ids=comparison.disagreement_target_ids,
            input_transition_sha256=comparison.transition_sha256,
        )
        if adjudicator_path is not None:
            form = SubmittedSynthesisAuthorizationReviewForm.model_validate(
                _read_object(adjudicator_path, code="synthesis_review_adjudicator")
            )
            adjudicator = freeze_completed_synthesis_authorization_review_submission(
                packet=packet,
                template=adjudication_template,
                form=form,
                not_before=max(reviewer_a.submitted_at, reviewer_b.submitted_at),
            )
            if adjudicator.reviewer_identity_sha256 in {
                reviewer_a.reviewer_identity_sha256,
                reviewer_b.reviewer_identity_sha256,
            }:
                raise SynthesisAuthorizationReviewError(
                    "synthesis_review_adjudicator_not_identity_independent"
                )
    elif adjudicator_path is not None:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_unnecessary_adjudication_forbidden"
        )

    by_a = {item.target_id: item for item in reviewer_a.decisions}
    by_b = {item.target_id: item for item in reviewer_b.decisions}
    by_adjudicator = (
        {item.target_id: item for item in adjudicator.decisions} if adjudicator is not None else {}
    )
    resolutions: list[ResolvedSynthesisAuthorizationReviewDecision] = []
    for target_id in comparison.agreed_target_ids:
        resolutions.append(
            _freeze_resolution(
                reviewer_a_decision=by_a[target_id],
                reviewer_b_decision=by_b[target_id],
                reviewer_identity_hashes=[
                    reviewer_a.reviewer_identity_sha256,
                    reviewer_b.reviewer_identity_sha256,
                ],
                comparison_sha256=comparison.transition_sha256,
            )
        )
    if adjudicator is not None:
        for target_id in comparison.disagreement_target_ids:
            resolutions.append(
                _freeze_resolution(
                    reviewer_a_decision=by_a[target_id],
                    reviewer_b_decision=by_b[target_id],
                    adjudicator_decision=by_adjudicator[target_id],
                    reviewer_identity_hashes=[
                        reviewer_a.reviewer_identity_sha256,
                        reviewer_b.reviewer_identity_sha256,
                        adjudicator.reviewer_identity_sha256,
                    ],
                    comparison_sha256=comparison.transition_sha256,
                )
            )
    resolutions.sort(key=lambda item: item.target_id)
    unresolved = set(comparison.disagreement_target_ids) if adjudicator is None else set()
    outcomes = _unit_outcomes(
        packet=packet,
        corpus=corpus,
        reconciliation=reconciliation,
        repository_root=repository_root,
        resolutions=resolutions,
        unresolved_target_ids=unresolved,
    )
    final_transition = hash_canonical(
        {
            "comparison_transition_sha256": comparison.transition_sha256,
            "adjudicator_submission_sha256": (
                adjudicator.submission_sha256 if adjudicator is not None else None
            ),
            "resolution_sha256s": [item.resolution_sha256 for item in resolutions],
            "unit_review_sha256s": [item.unit_review_sha256 for item in outcomes],
        }
    )
    private_payload = {
        "evaluation_version": "private-synthesis-authorization-review-evaluation-v1",
        "status": "awaiting_adjudication" if unresolved else "complete",
        "packet_sha256": packet.packet_sha256,
        "review_protocol_sha256": packet.review_protocol.protocol_sha256,
        "pipeline_fingerprint": packet.pipeline_fingerprint,
        "pipeline_fingerprint_sha256": packet.pipeline_fingerprint_sha256,
        "reviewer_a_submission": reviewer_a,
        "reviewer_b_submission": reviewer_b,
        "comparison": comparison,
        "adjudication_template_sha256": (
            adjudication_template.template_sha256 if unresolved and adjudication_template else None
        ),
        "adjudicator_submission": adjudicator,
        "resolutions": resolutions,
        "unit_outcomes": outcomes,
        "final_transition_sha256": final_transition,
    }
    private = PrivateSynthesisAuthorizationReviewEvaluation.model_validate(
        {**private_payload, "evaluation_sha256": hash_canonical(private_payload)}
    )

    target_by_id = {item.target_id: item for item in packet.targets}
    reviewer_minutes = [
        item.review_minutes
        for submission in (reviewer_a, reviewer_b)
        for item in submission.decisions
    ]
    adjudication_minutes = (
        sum(item.review_minutes for item in adjudicator.decisions)
        if adjudicator is not None
        else 0.0
    )
    all_minutes = [
        *reviewer_minutes,
        *([item.review_minutes for item in adjudicator.decisions] if adjudicator else []),
    ]
    assertion_count = sum(len(item.assertions) for item in outcomes)
    public_payload = {
        "summary_version": "synthesis-authorization-review-public-summary-v1",
        "status": private.status,
        "pipeline_fingerprint": packet.pipeline_fingerprint,
        "pipeline_fingerprint_sha256": packet.pipeline_fingerprint_sha256,
        "synthesis_unit_count": len(packet.requested_units),
        "review_target_count": len(packet.targets),
        "same_cohort_target_count": sum(
            item.required_relationship == "same_cohort" for item in packet.targets
        ),
        "independence_target_count": sum(
            item.required_relationship == "independent_cohorts" for item in packet.targets
        ),
        "relationship_agreement_count": len(comparison.agreed_target_ids),
        "support_divergence_count": len(comparison.support_divergence_target_ids),
        "disagreement_count": len(comparison.disagreement_target_ids),
        "adjudicated_count": len(comparison.disagreement_target_ids) if adjudicator else 0,
        "resolved_unknown_count": sum(
            item.relationship is ReviewRelationship.UNKNOWN for item in resolutions
        ),
        "resolved_contradiction_count": sum(
            item.relationship.value != target_by_id[item.target_id].required_relationship
            and item.relationship is not ReviewRelationship.UNKNOWN
            for item in resolutions
        ),
        "source_assertion_count": assertion_count,
        "authorized_unit_count": sum(item.authorizes_synthesis_input for item in outcomes),
        "abstained_unit_count": sum(not item.authorizes_synthesis_input for item in outcomes),
        "independent_reviewer_person_minutes": sum(reviewer_minutes),
        "adjudication_person_minutes": adjudication_minutes,
        "total_person_minutes": sum(all_minutes),
        "median_minutes_per_reviewed_target": median(all_minutes),
        "review_time_basis": "timestamp_derived_person_minutes",
        "aggregate_only": True,
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "contains_publication_or_cohort_identifiers": False,
        "contains_synthesis_unit_identifiers": False,
        "contains_reviewer_identities": False,
    }
    public = SynthesisAuthorizationReviewPublicSummary.model_validate(
        {**public_payload, "summary_sha256": hash_canonical(public_payload)}
    )
    return private, public, adjudication_template if unresolved else None


def write_synthesis_authorization_review_evaluation(
    *,
    repository_root: Path,
    private_output: Path,
    public_output: Path,
    private: PrivateSynthesisAuthorizationReviewEvaluation,
    public: SynthesisAuthorizationReviewPublicSummary,
    adjudication_template_output: Path | None = None,
    adjudication_template: BlankSynthesisAuthorizationReviewTemplate | None = None,
) -> None:
    """Write immutable private/public projections to their mandated disjoint roots."""

    if (
        private.pipeline_fingerprint != public.pipeline_fingerprint
        or private.pipeline_fingerprint_sha256 != public.pipeline_fingerprint_sha256
    ):
        raise SynthesisAuthorizationReviewError("synthesis_review_private_public_pipeline_mismatch")
    private_root = repository_root / "data/cache/synthesis-authorization-review"
    public_root = repository_root / "artifacts/diagnostics/synthesis-authorization-review"
    private_output = _strictly_under(
        private_output,
        private_root,
        code="synthesis_review_private_evaluation_outside_ignored_root",
    )
    public_output = _strictly_under(
        public_output,
        public_root,
        code="synthesis_review_public_output_outside_diagnostic_root",
    )
    if (adjudication_template_output is None) != (adjudication_template is None):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_adjudication_template_output_presence_mismatch"
        )
    if adjudication_template_output is not None:
        adjudication_template_output = _strictly_under(
            adjudication_template_output,
            private_root,
            code="synthesis_review_adjudication_template_outside_ignored_root",
        )
    outputs = [private_output, public_output]
    if adjudication_template_output is not None:
        outputs.append(adjudication_template_output)
    if len(outputs) != len(set(outputs)) or any(path.exists() for path in outputs):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_evaluation_output_exists_or_aliases"
        )
    atomic_write_json(private_output, private)
    if adjudication_template_output is not None and adjudication_template is not None:
        atomic_write_json(adjudication_template_output, adjudication_template)
    atomic_write_json(public_output, public)


def reverify_synthesis_authorization_review_evaluation(
    *,
    manifest_path: Path,
    corpus: TypedEvidenceCorpus,
    reconciliation: NativeCohortReconciliationReceipt,
    repository_root: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    private_evaluation: PrivateSynthesisAuthorizationReviewEvaluation,
    public_summary: SynthesisAuthorizationReviewPublicSummary,
    adjudicator_path: Path | None = None,
    adjudication_template: BlankSynthesisAuthorizationReviewTemplate | None = None,
) -> tuple[
    PrivateSynthesisAuthorizationReviewEvaluation,
    SynthesisAuthorizationReviewPublicSummary,
    BlankSynthesisAuthorizationReviewTemplate | None,
]:
    """Replay every source and transition; standalone self-hash validation is insufficient.

    This is the normative downstream validation path.  It reconstructs the packet from
    the corpus, reconciliation, and current source bytes; rebuilds both reviewer
    submissions and any adjudication; reruns source-backed synthesis authorization;
    and then requires exact equality of the private and aggregate public projections.
    """

    frozen_private = PrivateSynthesisAuthorizationReviewEvaluation.model_validate(
        private_evaluation.model_dump(mode="json")
    )
    frozen_public = SynthesisAuthorizationReviewPublicSummary.model_validate(
        public_summary.model_dump(mode="json")
    )
    current_pipeline = _require_current_pipeline_binding(
        repository_root=repository_root,
        fingerprint=frozen_private.pipeline_fingerprint,
        fingerprint_sha256=frozen_private.pipeline_fingerprint_sha256,
    )
    if (
        frozen_public.pipeline_fingerprint != current_pipeline
        or frozen_public.pipeline_fingerprint_sha256 != current_pipeline.pipeline_sha256
    ):
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_public_pipeline_fingerprint_mismatch"
        )
    replayed_private, replayed_public, replayed_template = evaluate_synthesis_authorization_review(
        manifest_path=manifest_path,
        corpus=corpus,
        reconciliation=reconciliation,
        repository_root=repository_root,
        reviewer_a_path=reviewer_a_path,
        reviewer_b_path=reviewer_b_path,
        adjudicator_path=adjudicator_path,
    )
    if replayed_private != frozen_private:
        raise SynthesisAuthorizationReviewError(
            "synthesis_review_private_evaluation_replay_mismatch"
        )
    if replayed_public != frozen_public:
        raise SynthesisAuthorizationReviewError("synthesis_review_public_summary_replay_mismatch")
    if adjudication_template is not None:
        frozen_template = BlankSynthesisAuthorizationReviewTemplate.model_validate(
            adjudication_template.model_dump(mode="json")
        )
        if replayed_template != frozen_template:
            raise SynthesisAuthorizationReviewError(
                "synthesis_review_adjudication_template_replay_mismatch"
            )
    return replayed_private, replayed_public, replayed_template


__all__ = [
    "BlankSynthesisAuthorizationReviewTemplate",
    "PrivateSynthesisAuthorizationReviewEvaluation",
    "RequestedSynthesisUnit",
    "ReviewRelationship",
    "ReviewerSlot",
    "SubmittedSynthesisAuthorizationReviewForm",
    "SynthesisAuthorizationReviewError",
    "SynthesisAuthorizationReviewManifest",
    "SynthesisAuthorizationReviewPacket",
    "SynthesisAuthorizationReviewProtocol",
    "SynthesisAuthorizationReviewPublicSummary",
    "SynthesisAuthorizationReviewRequest",
    "compute_synthesis_authorization_review_pipeline_fingerprint",
    "default_synthesis_authorization_review_protocol",
    "evaluate_synthesis_authorization_review",
    "freeze_blank_synthesis_authorization_review_template",
    "freeze_completed_synthesis_authorization_review_submission",
    "freeze_synthesis_authorization_review_packet",
    "freeze_synthesis_authorization_review_request",
    "prepare_synthesis_authorization_review",
    "reverify_synthesis_authorization_review_evaluation",
    "reverify_synthesis_authorization_review_packet",
    "verify_synthesis_authorization_review_manifest",
    "write_synthesis_authorization_review_evaluation",
]
