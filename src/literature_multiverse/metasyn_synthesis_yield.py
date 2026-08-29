"""Fail-closed MetaSyn typed-graph and synthesis-yield evaluation.

The bounded runtime deliberately stops at publication-level extraction yield.  This
module is the provider-neutral downstream bridge: it joins the exact finalized runtime
report to the externally replayed adapter and prepare bundles, replays every promoted
extraction against the original source artifact, constructs a complete terminal
publication ledger for each question, and attempts synthesis only within compatible
effect strata whose cross-publication cohort identity is resolved.

The diagnostic never opens MetaSyn directions, conclusions, aggregate effects, or test
labels.  It reports mechanics and yield only.  In particular, a completed synthesis is
not an accuracy, truth, calibration, clinical-benefit, or claim-release result.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
    reconcile_native_cohorts,
)
from literature_multiverse.effects import harmonize_effect
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    MetaSynBoundedRowContextV1,
    MetaSynUniqueQuoteGroundingV1,
    validate_metasyn_bounded_adapter_bundle_external_replay,
)
from literature_multiverse.metasyn_bounded_runtime import (
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynRuntimeRowResultV1,
    load_current_metasyn_bounded_execution_bundle,
    validate_metasyn_bounded_finalized_runtime,
)
from literature_multiverse.metasyn_typed_pilot import (
    EXPECTED_SELECTED_COMPONENTS,
    EXPECTED_SELECTED_PAPERS,
    EXPECTED_SELECTED_QUESTIONS,
    PREPARE_BUNDLE_FILENAME,
    MetaSynPilotQuestionBundleV1,
    MetaSynTypedPilotPrepareBundleV1,
    validate_metasyn_typed_pilot_prepare,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_grounding import (
    NativeGroundingReceipt,
    freeze_grounding_checked_publication_fragment,
    verify_native_publication_grounding,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

EVALUATION_VERSION = "metasyn-typed-synthesis-yield-v1"
PUBLIC_SUMMARY_VERSION = "metasyn-typed-synthesis-yield-public-v1"
PUBLICATION_RECORD_VERSION = "metasyn-synthesis-publication-record-v1"
GROUP_RECORD_VERSION = "metasyn-synthesis-compatibility-group-v1"
QUESTION_RECORD_VERSION = "metasyn-synthesis-question-yield-v1"
EVALUATION_COMPONENT_VERSION = "1"
COMPATIBILITY_RULE_VERSION = "metasyn-effect-compatibility-v1"

_MODULE_ENTRYPOINT = "src/literature_multiverse/metasyn_synthesis_yield.py"
_DEPENDENCY_ENTRYPOINTS = (_MODULE_ENTRYPOINT,)
_NON_PYTHON_INPUTS = ("pyproject.toml", "uv.lock")

PublicationStage = Literal[
    "release_grade_fragment_estimable",
    "diagnostic_only_fragment_excluded",
    "runtime_terminal_fragment_excluded",
    "original_source_grounding_failed",
]
GroupStage = Literal[
    "synthesis_completed",
    "synthesis_attempted_insufficient",
    "blocked_harmonization",
    "blocked_cross_publication_identity",
]
SynthesisCompletionMode = Literal[
    "directional_sign_synthesis",
    "random_effects_meta_analysis",
]


class BlockerAggregateCode(StrEnum):
    """Closed, identifier-free public families for question-level blockers."""

    CROSS_PUBLICATION_IDENTITY = "cross_publication_identity_unresolved"
    DIAGNOSTIC_SOURCE = "diagnostic_source_exclusion"
    EFFECT_HARMONIZATION = "effect_harmonization_failure"
    NO_ESTIMABLE_GRAPH = "no_estimable_graph"
    ORIGINAL_SOURCE_GROUNDING = "original_source_grounding_failure"
    RUNTIME_TERMINAL = "runtime_terminal_exclusion"
    SYNTHESIS_INSUFFICIENT = "synthesis_insufficient"


class ResidualConflictAggregateCode(StrEnum):
    """Closed, identifier-free public families for preserved conflicts."""

    COHORT_RECONCILIATION = "cohort_reconciliation_issue"
    MULTIPLE_CONTRAST_STRATA = "multiple_contrast_strata"
    MULTIPLE_EFFECT_STRATA = "multiple_effect_strata"
    UNHARMONIZABLE_EFFECT_STRATUM = "unharmonizable_effect_stratum"
    UNRESOLVED_COHORT_IDENTITY = "unresolved_cohort_identity"


class MetaSynSynthesisYieldError(ValueError):
    """An upstream identity, source replay, graph, or yield invariant failed."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
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
    return None


def _python_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source = repository_root / relative
        if not source.is_file():
            raise MetaSynSynthesisYieldError(
                f"metasyn_synthesis_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynSynthesisYieldError(
                f"metasyn_synthesis_dependency_unreadable:{relative}"
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


def compute_metasyn_synthesis_yield_fingerprint(
    *,
    repository_root: Path,
    runtime_report: MetaSynBoundedPrivateYieldReportV1,
    adapter_bundle: MetaSynBoundedAdapterBundleV1,
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1,
) -> PipelineFingerprint:
    """Bind this evaluator's full local dependency closure and all upstream identities."""

    root = repository_root.resolve(strict=True)
    runtime = MetaSynBoundedPrivateYieldReportV1.model_validate(runtime_report)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(prepare_bundle)
    component = PipelineComponentSpec(
        component_id="metasyn-typed-synthesis-yield",
        component_version=EVALUATION_COMPONENT_VERSION,
        file_paths=sorted(
            {*_python_dependency_closure(root), *_NON_PYTHON_INPUTS}
        ),
        settings={
            "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
            "all_runtime_rows_projected_to_terminal_fragments": True,
            "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
            "dependency_closure_entrypoints": list(_DEPENDENCY_ENTRYPOINTS),
            "downstream_verifier_pipeline_sha256": (
                runtime.downstream_verifier_pipeline_sha256
            ),
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name)
                for name in ("numpy", "pydantic", "scipy")
            },
            "original_source_grounding_replay_required": True,
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "prepare_bundle_sha256": prepared.prepare_bundle_sha256,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "reference_direction_conclusion_and_test_labels_opened": False,
            "runtime_private_report_sha256": runtime.report_sha256,
            "runtime_pipeline_sha256": runtime.runtime_pipeline_sha256,
            "synthesis_scope": (
                "yield_only_full_text_source_replayed_compatible_identity_resolved"
            ),
            "title_abstract_synthesis_input_permitted": False,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


class MetaSynSynthesisPublicationRecordV1(ContractModel):
    record_version: Literal["metasyn-synthesis-publication-record-v1"] = (
        PUBLICATION_RECORD_VERSION
    )
    question_id: Annotated[str, Field(min_length=1, max_length=64)]
    question_spec_sha256: str
    question_bundle_sha256: str
    row_context_sha256: str
    runtime_row_result_sha256: str
    source_row_sha256: str
    publication_id: Annotated[str, Field(min_length=1, max_length=256)]
    paper_id: Annotated[str, Field(min_length=1, max_length=256)]
    source_locator: Annotated[str, Field(min_length=1, max_length=2048)]
    source_document_sha256: str
    runtime_status: str
    runtime_contract_typed: bool
    release_grade_source_grounding_eligible: bool
    adapter_publication_result_sha256: str | None
    adapter_quote_groundings: list[MetaSynUniqueQuoteGroundingV1]
    adapter_quote_grounding_sha256s: list[str]
    original_source_grounding_receipt: NativeGroundingReceipt | None
    original_source_grounding_receipt_sha256: str | None
    original_source_grounding_authorized: bool
    stage: PublicationStage
    blockers: list[str]
    terminal_fragment: PublicationEvidenceFragment
    terminal_fragment_sha256: str
    record_sha256: str

    @field_validator(
        "question_spec_sha256",
        "question_bundle_sha256",
        "row_context_sha256",
        "runtime_row_result_sha256",
        "source_row_sha256",
        "source_document_sha256",
        "adapter_publication_result_sha256",
        "original_source_grounding_receipt_sha256",
        "terminal_fragment_sha256",
        "record_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"metasyn_synthesis_publication_hash_invalid:{info.field_name}"
            )
        return value

    @field_validator("adapter_quote_grounding_sha256s", "blockers")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"metasyn_synthesis_publication_not_sorted_unique:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_record(self) -> MetaSynSynthesisPublicationRecordV1:
        expected_grounding_hashes = sorted(
            item.grounding_sha256 for item in self.adapter_quote_groundings
        )
        if self.adapter_quote_grounding_sha256s != expected_grounding_hashes:
            raise ValueError("metasyn_synthesis_adapter_grounding_hashes_mismatch")
        receipt_sha = (
            self.original_source_grounding_receipt.receipt_sha256
            if self.original_source_grounding_receipt is not None
            else None
        )
        if self.original_source_grounding_receipt_sha256 != receipt_sha:
            raise ValueError("metasyn_synthesis_source_receipt_hash_mismatch")
        expected_authorized = bool(
            self.original_source_grounding_receipt is not None
            and self.original_source_grounding_receipt.authorizes_estimable_fragment
        )
        if self.original_source_grounding_authorized != expected_authorized:
            raise ValueError("metasyn_synthesis_source_authorization_mismatch")
        fragment = self.terminal_fragment
        if (
            fragment.question_id != self.question_id
            or fragment.publication_id != self.publication_id
            or fragment.paper_id != self.paper_id
            or fragment.source_document.source_locator != self.source_locator
            or fragment.source_document.sha256 != self.source_document_sha256
        ):
            raise ValueError("metasyn_synthesis_terminal_fragment_lineage_mismatch")
        if self.terminal_fragment_sha256 != fragment.fragment_sha256:
            raise ValueError("metasyn_synthesis_terminal_fragment_hash_mismatch")
        estimable = fragment.status is FragmentStatus.ESTIMABLE
        if estimable != (self.stage == "release_grade_fragment_estimable"):
            raise ValueError("metasyn_synthesis_publication_stage_fragment_mismatch")
        if self.stage == "release_grade_fragment_estimable" and not (
            self.runtime_contract_typed
            and self.release_grade_source_grounding_eligible
            and self.original_source_grounding_authorized
            and not self.blockers
        ):
            raise ValueError("metasyn_synthesis_estimable_stage_not_authorized")
        if self.stage != "release_grade_fragment_estimable" and not self.blockers:
            raise ValueError("metasyn_synthesis_excluded_stage_requires_blocker")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if hash_canonical(payload) != self.record_sha256:
            raise ValueError("metasyn_synthesis_publication_record_hash_mismatch")
        return self


class MetaSynCompatibilityGroupYieldV1(ContractModel):
    group_version: Literal["metasyn-synthesis-compatibility-group-v1"] = (
        GROUP_RECORD_VERSION
    )
    compatibility_rule_version: Literal["metasyn-effect-compatibility-v1"] = (
        COMPATIBILITY_RULE_VERSION
    )
    compatibility_sha256: str
    outcome_name: Annotated[str, Field(min_length=1, max_length=64)]
    contrast_label: Annotated[str, Field(min_length=1, max_length=256)]
    contrast_estimand: str | None
    positive_direction_means: Annotated[str, Field(min_length=1, max_length=256)]
    timepoint_sha256: str
    analysis_population: str | None
    harmonization_status: Literal["estimable", "insufficient"]
    harmonized_measure: str | None
    harmonized_unit: str | None
    harmonization_reason: str | None
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    paper_ids: Annotated[list[str], Field(min_length=1)]
    cohort_ids: Annotated[list[str], Field(min_length=1)]
    group_graph_sha256: str
    cross_publication_identity_assurance: Literal[
        "single_publication_intrinsic",
        "reconciliation_receipt_complete",
        "unresolved_blocked",
    ]
    synthesis_input_eligible: bool
    synthesis_attempted: bool
    synthesis_completed: bool
    stage: GroupStage
    blocker: str | None
    synthesis_status: str | None
    synthesis_mode: str | None
    synthesis_reason: str | None
    synthesis_sha256: str | None
    group_sha256: str

    @field_validator(
        "compatibility_sha256",
        "timepoint_sha256",
        "group_graph_sha256",
        "synthesis_sha256",
        "group_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_synthesis_group_hash_invalid:{info.field_name}")
        return value

    @field_validator("estimate_ids", "paper_ids", "cohort_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_synthesis_group_ids_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_group(self) -> MetaSynCompatibilityGroupYieldV1:
        estimable = self.harmonization_status == "estimable"
        if estimable != (
            self.harmonized_measure is not None and self.harmonization_reason is None
        ):
            raise ValueError("metasyn_synthesis_group_harmonization_fields_mismatch")
        if not estimable and (
            self.harmonized_measure is not None or self.harmonization_reason is None
        ):
            raise ValueError("metasyn_synthesis_group_insufficient_fields_mismatch")
        eligible = estimable and self.cross_publication_identity_assurance != (
            "unresolved_blocked"
        )
        if self.synthesis_input_eligible != eligible:
            raise ValueError("metasyn_synthesis_group_eligibility_mismatch")
        if self.synthesis_attempted != eligible:
            raise ValueError("metasyn_synthesis_group_attempt_mismatch")
        if self.synthesis_completed and not self.synthesis_attempted:
            raise ValueError("metasyn_synthesis_group_completion_without_attempt")
        if self.synthesis_attempted != (self.synthesis_sha256 is not None):
            raise ValueError("metasyn_synthesis_group_result_hash_presence_mismatch")
        expected_stage: GroupStage
        if not estimable:
            expected_stage = "blocked_harmonization"
        elif self.cross_publication_identity_assurance == "unresolved_blocked":
            expected_stage = "blocked_cross_publication_identity"
        elif self.synthesis_completed:
            expected_stage = "synthesis_completed"
        else:
            expected_stage = "synthesis_attempted_insufficient"
        if self.stage != expected_stage:
            raise ValueError("metasyn_synthesis_group_stage_mismatch")
        if (self.blocker is None) != self.synthesis_input_eligible:
            raise ValueError("metasyn_synthesis_group_blocker_presence_mismatch")
        if self.synthesis_attempted != (
            self.synthesis_status is not None and self.synthesis_mode is not None
        ):
            raise ValueError("metasyn_synthesis_group_result_fields_mismatch")
        payload = self.model_dump(mode="json", exclude={"group_sha256"})
        if hash_canonical(payload) != self.group_sha256:
            raise ValueError("metasyn_synthesis_group_hash_mismatch")
        return self


class MetaSynQuestionSynthesisYieldV1(ContractModel):
    question_report_version: Literal["metasyn-synthesis-question-yield-v1"] = (
        QUESTION_RECORD_VERSION
    )
    question_id: Annotated[str, Field(min_length=1, max_length=64)]
    question_spec_sha256: str
    question_bundle_sha256: str
    independence_component_id: Annotated[str, Field(min_length=1, max_length=256)]
    independence_component_membership_sha256: str
    oracle_roster_membership_sha256: str
    publication_records: Annotated[
        list[MetaSynSynthesisPublicationRecordV1], Field(min_length=2, max_length=4)
    ]
    publication_record_sha256s: list[str]
    terminal_corpus: TypedEvidenceCorpus
    terminal_corpus_sha256: str
    cohort_reconciliation: NativeCohortReconciliationReceipt
    cohort_reconciliation_sha256: str
    compatibility_groups: list[MetaSynCompatibilityGroupYieldV1]
    compatibility_group_sha256s: list[str]
    graph_construction_completed: Literal[True] = True
    graph_estimate_count: Annotated[int, Field(ge=0)]
    graph_study_count: Annotated[int, Field(ge=0)]
    graph_cohort_count: Annotated[int, Field(ge=0)]
    release_grade_estimable_publication_count: Annotated[int, Field(ge=0, le=4)]
    synthesis_input_group_count: Annotated[int, Field(ge=0)]
    synthesis_attempted_group_count: Annotated[int, Field(ge=0)]
    synthesis_completed_group_count: Annotated[int, Field(ge=0)]
    blockers: list[str]
    residual_conflicts: list[str]
    question_report_sha256: str

    @field_validator(
        "question_spec_sha256",
        "question_bundle_sha256",
        "independence_component_membership_sha256",
        "oracle_roster_membership_sha256",
        "terminal_corpus_sha256",
        "cohort_reconciliation_sha256",
        "question_report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_synthesis_question_hash_invalid:{info.field_name}")
        return value

    @field_validator(
        "publication_record_sha256s",
        "compatibility_group_sha256s",
        "blockers",
        "residual_conflicts",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_synthesis_question_values_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_question(self) -> MetaSynQuestionSynthesisYieldV1:
        publications = [item.publication_id for item in self.publication_records]
        if publications != sorted(set(publications)):
            raise ValueError("metasyn_synthesis_question_publication_roster_invalid")
        if self.publication_record_sha256s != sorted(
            item.record_sha256 for item in self.publication_records
        ):
            raise ValueError("metasyn_synthesis_question_publication_hashes_mismatch")
        if self.compatibility_group_sha256s != sorted(
            item.group_sha256 for item in self.compatibility_groups
        ):
            raise ValueError("metasyn_synthesis_question_group_hashes_mismatch")
        if self.terminal_corpus_sha256 != self.terminal_corpus.corpus_sha256:
            raise ValueError("metasyn_synthesis_question_corpus_hash_mismatch")
        if self.cohort_reconciliation_sha256 != (
            self.cohort_reconciliation.receipt_sha256
        ):
            raise ValueError("metasyn_synthesis_question_reconciliation_hash_mismatch")
        expected_fragments = sorted(
            item.terminal_fragment_sha256 for item in self.publication_records
        )
        if expected_fragments != sorted(
            fragment.fragment_sha256 for fragment in self.terminal_corpus.fragments
        ):
            raise ValueError("metasyn_synthesis_question_terminal_roster_mismatch")
        graph = self.terminal_corpus.graph
        if (
            self.graph_estimate_count != len(graph.outcome_estimates)
            or self.graph_study_count != len(graph.studies)
            or self.graph_cohort_count != len(graph.cohorts)
        ):
            raise ValueError("metasyn_synthesis_question_graph_counts_mismatch")
        if self.release_grade_estimable_publication_count != sum(
            item.stage == "release_grade_fragment_estimable"
            for item in self.publication_records
        ):
            raise ValueError("metasyn_synthesis_question_estimable_count_mismatch")
        if self.synthesis_input_group_count != sum(
            item.synthesis_input_eligible for item in self.compatibility_groups
        ):
            raise ValueError("metasyn_synthesis_question_input_count_mismatch")
        if self.synthesis_attempted_group_count != sum(
            item.synthesis_attempted for item in self.compatibility_groups
        ):
            raise ValueError("metasyn_synthesis_question_attempt_count_mismatch")
        if self.synthesis_completed_group_count != sum(
            item.synthesis_completed for item in self.compatibility_groups
        ):
            raise ValueError("metasyn_synthesis_question_completion_count_mismatch")
        grouped_estimates = sorted(
            estimate_id
            for group in self.compatibility_groups
            for estimate_id in group.estimate_ids
        )
        if grouped_estimates != sorted(
            estimate.estimate_id for estimate in graph.outcome_estimates
        ):
            raise ValueError("metasyn_synthesis_question_group_coverage_mismatch")
        payload = self.model_dump(mode="json", exclude={"question_report_sha256"})
        if hash_canonical(payload) != self.question_report_sha256:
            raise ValueError("metasyn_synthesis_question_report_hash_mismatch")
        return self


def _blocker_aggregate_code(value: str) -> BlockerAggregateCode:
    """Collapse a private internal blocker into a closed public-safe family."""

    if value.startswith(("runtime_terminal:", "runtime:")):
        return BlockerAggregateCode.RUNTIME_TERMINAL
    if value == "diagnostic_source_surface_not_synthesis_eligible" or value.startswith(
        "source_strength:"
    ):
        return BlockerAggregateCode.DIAGNOSTIC_SOURCE
    if value == "original_source_grounding_not_authorizing" or value.startswith(
        ("original_source_grounding:", "original_source_finding:")
    ):
        return BlockerAggregateCode.ORIGINAL_SOURCE_GROUNDING
    if value.startswith("harmonization:"):
        return BlockerAggregateCode.EFFECT_HARMONIZATION
    if value == "cross_publication_cohort_identity_unresolved":
        return BlockerAggregateCode.CROSS_PUBLICATION_IDENTITY
    if value.startswith("synthesis:"):
        return BlockerAggregateCode.SYNTHESIS_INSUFFICIENT
    if value == "no_release_grade_source_replayed_graph_estimates":
        return BlockerAggregateCode.NO_ESTIMABLE_GRAPH
    raise MetaSynSynthesisYieldError(
        "metasyn_synthesis_unknown_blocker_aggregate_code"
    )


def _residual_conflict_aggregate_code(
    value: str,
) -> ResidualConflictAggregateCode:
    """Collapse a private reconciliation detail into a closed public-safe family."""

    if value.startswith("cohort_reconciliation:"):
        return ResidualConflictAggregateCode.COHORT_RECONCILIATION
    mapping = {
        "multiple_effect_compatibility_strata_preserved": (
            ResidualConflictAggregateCode.MULTIPLE_EFFECT_STRATA
        ),
        "unresolved_cross_publication_cohort_identity_preserved": (
            ResidualConflictAggregateCode.UNRESOLVED_COHORT_IDENTITY
        ),
        "unharmonizable_effect_stratum_preserved": (
            ResidualConflictAggregateCode.UNHARMONIZABLE_EFFECT_STRATUM
        ),
        "multiple_contrast_orientation_or_estimand_strata_preserved": (
            ResidualConflictAggregateCode.MULTIPLE_CONTRAST_STRATA
        ),
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_unknown_residual_conflict_aggregate_code"
        ) from exc


def _aggregate_blocker_counts(
    questions: Sequence[MetaSynQuestionSynthesisYieldV1],
) -> dict[BlockerAggregateCode, int]:
    counts = Counter(
        code
        for question in questions
        for code in {
            _blocker_aggregate_code(blocker) for blocker in question.blockers
        }
    )
    return dict(sorted(counts.items()))


def _aggregate_residual_conflict_counts(
    questions: Sequence[MetaSynQuestionSynthesisYieldV1],
) -> dict[ResidualConflictAggregateCode, int]:
    counts = Counter(
        code
        for question in questions
        for code in {
            _residual_conflict_aggregate_code(conflict)
            for conflict in question.residual_conflicts
        }
    )
    return dict(sorted(counts.items()))


class MetaSynSynthesisYieldReportV1(ContractModel):
    evaluation_version: Literal["metasyn-typed-synthesis-yield-v1"] = (
        EVALUATION_VERSION
    )
    status: Literal["complete_label_blind_synthesis_yield_evaluation"] = (
        "complete_label_blind_synthesis_yield_evaluation"
    )
    evaluation_pipeline_fingerprint: PipelineFingerprint
    evaluation_pipeline_sha256: str
    runtime_private_report_sha256: str
    runtime_pipeline_sha256: str
    adapter_bundle_sha256: str
    prepare_bundle_sha256: str
    downstream_verifier_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    question_reports: Annotated[
        list[MetaSynQuestionSynthesisYieldV1], Field(min_length=10, max_length=10)
    ]
    question_report_sha256s: Annotated[list[str], Field(min_length=10, max_length=10)]
    runtime_contract_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    diagnostic_only_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    original_source_grounding_attempt_count: Annotated[int, Field(ge=0, le=32)]
    original_source_grounding_authorized_count: Annotated[int, Field(ge=0, le=32)]
    release_grade_estimable_publication_count: Annotated[int, Field(ge=0, le=32)]
    terminal_fragment_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    graph_construction_completed_question_count: Literal[10] = (
        EXPECTED_SELECTED_QUESTIONS
    )
    questions_with_estimable_graph: Annotated[int, Field(ge=0, le=10)]
    graph_estimate_count: Annotated[int, Field(ge=0)]
    compatibility_group_count: Annotated[int, Field(ge=0)]
    synthesis_input_group_count: Annotated[int, Field(ge=0)]
    synthesis_attempted_group_count: Annotated[int, Field(ge=0)]
    synthesis_completed_group_count: Annotated[int, Field(ge=0)]
    questions_with_completed_synthesis: Annotated[int, Field(ge=0, le=10)]
    publication_stage_counts: dict[PublicationStage, Annotated[int, Field(ge=1)]]
    synthesis_group_stage_counts: dict[GroupStage, Annotated[int, Field(ge=1)]]
    synthesis_completion_mode_counts: dict[
        SynthesisCompletionMode, Annotated[int, Field(ge=1)]
    ]
    blocker_counts: dict[BlockerAggregateCode, Annotated[int, Field(ge=1)]]
    residual_conflict_counts: dict[
        ResidualConflictAggregateCode, Annotated[int, Field(ge=1)]
    ]
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    truth_or_clinical_benefit_reported: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal[
        "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
    ] = "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
    report_sha256: str

    @field_validator(
        "evaluation_pipeline_sha256",
        "runtime_private_report_sha256",
        "runtime_pipeline_sha256",
        "adapter_bundle_sha256",
        "prepare_bundle_sha256",
        "downstream_verifier_pipeline_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_synthesis_report_hash_invalid:{info.field_name}")
        return value

    @field_validator("question_report_sha256s")
    @classmethod
    def validate_question_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not SHA256_RE.fullmatch(item) for item in value
        ):
            raise ValueError("metasyn_synthesis_report_question_hashes_invalid")
        return value

    @field_validator(
        "publication_stage_counts",
        "synthesis_group_stage_counts",
        "synthesis_completion_mode_counts",
        "blocker_counts",
        "residual_conflict_counts",
    )
    @classmethod
    def validate_sorted_counts(cls, value: dict[str, int], info: Any) -> dict[str, int]:
        if value != dict(sorted(value.items())) or any(count < 0 for count in value.values()):
            raise ValueError(f"metasyn_synthesis_report_counts_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynSynthesisYieldReportV1:
        if self.evaluation_pipeline_sha256 != (
            self.evaluation_pipeline_fingerprint.pipeline_sha256
        ):
            raise ValueError("metasyn_synthesis_report_pipeline_hash_mismatch")
        ids = [item.question_id for item in self.question_reports]
        if ids != sorted(set(ids)):
            raise ValueError("metasyn_synthesis_report_question_roster_invalid")
        if self.question_report_sha256s != sorted(
            item.question_report_sha256 for item in self.question_reports
        ):
            raise ValueError("metasyn_synthesis_report_question_hash_alias_mismatch")
        publications = [
            record
            for question in self.question_reports
            for record in question.publication_records
        ]
        groups = [
            group
            for question in self.question_reports
            for group in question.compatibility_groups
        ]
        expected = {
            "runtime_contract_typed_publication_count": sum(
                row.runtime_contract_typed for row in publications
            ),
            "diagnostic_only_typed_publication_count": sum(
                row.runtime_contract_typed
                and not row.release_grade_source_grounding_eligible
                for row in publications
            ),
            "original_source_grounding_attempt_count": sum(
                row.original_source_grounding_receipt is not None for row in publications
            ),
            "original_source_grounding_authorized_count": sum(
                row.original_source_grounding_authorized for row in publications
            ),
            "release_grade_estimable_publication_count": sum(
                row.stage == "release_grade_fragment_estimable" for row in publications
            ),
            "terminal_fragment_count": len(publications),
            "graph_construction_completed_question_count": sum(
                row.graph_construction_completed for row in self.question_reports
            ),
            "questions_with_estimable_graph": sum(
                row.graph_estimate_count > 0 for row in self.question_reports
            ),
            "graph_estimate_count": sum(
                row.graph_estimate_count for row in self.question_reports
            ),
            "compatibility_group_count": len(groups),
            "synthesis_input_group_count": sum(
                row.synthesis_input_eligible for row in groups
            ),
            "synthesis_attempted_group_count": sum(row.synthesis_attempted for row in groups),
            "synthesis_completed_group_count": sum(row.synthesis_completed for row in groups),
            "questions_with_completed_synthesis": sum(
                row.synthesis_completed_group_count > 0 for row in self.question_reports
            ),
        }
        for name, count in expected.items():
            if getattr(self, name) != count:
                raise ValueError(f"metasyn_synthesis_report_aggregate_mismatch:{name}")
        expected_publication_stages = dict(
            sorted(Counter(row.stage for row in publications).items())
        )
        expected_group_stages = dict(sorted(Counter(row.stage for row in groups).items()))
        expected_modes = dict(
            sorted(
                Counter(
                    row.synthesis_mode
                    for row in groups
                    if row.synthesis_completed and row.synthesis_mode is not None
                ).items()
            )
        )
        expected_blockers = _aggregate_blocker_counts(self.question_reports)
        expected_residuals = _aggregate_residual_conflict_counts(
            self.question_reports
        )
        if (
            self.publication_stage_counts != expected_publication_stages
            or self.synthesis_group_stage_counts != expected_group_stages
            or self.synthesis_completion_mode_counts != expected_modes
            or self.blocker_counts != expected_blockers
            or self.residual_conflict_counts != expected_residuals
        ):
            raise ValueError("metasyn_synthesis_report_counter_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("metasyn_synthesis_report_hash_mismatch")
        return self


_PUBLIC_CAVEATS = (
    "This is a retrospective calibration-split oracle-corpus mechanics-and-yield "
    "diagnostic, not a pristine retrieval or end-to-end evaluation.",
    "Review conclusions, directions, aggregate effects, and official test labels remain "
    "unopened; no accuracy or direction agreement is reported.",
    "Only full-text typed publications that pass original-source replay can contribute "
    "estimates to a synthesis attempt.",
    "Title/abstract outputs remain diagnostic and every blocked or failed publication "
    "remains in the terminal roster.",
    "Effect strata are kept separate by outcome, timepoint, analysis population, "
    "contrast orientation, estimand, and harmonized measure/unit.",
    "Unresolved cross-publication cohort identity blocks pooling; residual conflicts "
    "are retained rather than collapsed.",
    "A completed statistical synthesis establishes execution yield only, not scientific "
    "correctness, calibration, truth, clinical benefit, or release authority.",
)


class MetaSynSynthesisYieldPublicSummaryV1(ContractModel):
    summary_version: Literal["metasyn-typed-synthesis-yield-public-v1"] = (
        PUBLIC_SUMMARY_VERSION
    )
    status: Literal["aggregate_only_label_blind_synthesis_yield"] = (
        "aggregate_only_label_blind_synthesis_yield"
    )
    evaluation_pipeline_sha256: str
    runtime_private_report_sha256: str
    synthesis_private_report_sha256: str
    adapter_bundle_sha256: str
    prepare_bundle_sha256: str
    downstream_verifier_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    runtime_contract_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    diagnostic_only_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    original_source_grounding_attempt_count: Annotated[int, Field(ge=0, le=32)]
    original_source_grounding_authorized_count: Annotated[int, Field(ge=0, le=32)]
    release_grade_estimable_publication_count: Annotated[int, Field(ge=0, le=32)]
    terminal_fragment_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    graph_construction_completed_question_count: Literal[10] = (
        EXPECTED_SELECTED_QUESTIONS
    )
    questions_with_estimable_graph: Annotated[int, Field(ge=0, le=10)]
    graph_estimate_count: Annotated[int, Field(ge=0)]
    compatibility_group_count: Annotated[int, Field(ge=0)]
    synthesis_input_group_count: Annotated[int, Field(ge=0)]
    synthesis_attempted_group_count: Annotated[int, Field(ge=0)]
    synthesis_completed_group_count: Annotated[int, Field(ge=0)]
    questions_with_completed_synthesis: Annotated[int, Field(ge=0, le=10)]
    publication_stage_counts: dict[PublicationStage, Annotated[int, Field(ge=1)]]
    synthesis_group_stage_counts: dict[GroupStage, Annotated[int, Field(ge=1)]]
    synthesis_completion_mode_counts: dict[
        SynthesisCompletionMode, Annotated[int, Field(ge=1)]
    ]
    blocker_counts: dict[BlockerAggregateCode, Annotated[int, Field(ge=1)]]
    residual_conflict_counts: dict[
        ResidualConflictAggregateCode, Annotated[int, Field(ge=1)]
    ]
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    truth_or_clinical_benefit_reported: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal[
        "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
    ] = "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
    caveats: Annotated[list[str], Field(min_length=7, max_length=7)]
    summary_sha256: str

    @field_validator(
        "evaluation_pipeline_sha256",
        "runtime_private_report_sha256",
        "synthesis_private_report_sha256",
        "adapter_bundle_sha256",
        "prepare_bundle_sha256",
        "downstream_verifier_pipeline_sha256",
        "summary_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_synthesis_public_hash_invalid:{info.field_name}")
        return value

    @field_validator(
        "publication_stage_counts",
        "synthesis_group_stage_counts",
        "synthesis_completion_mode_counts",
        "blocker_counts",
        "residual_conflict_counts",
    )
    @classmethod
    def validate_sorted_counts(cls, value: dict[str, int], info: Any) -> dict[str, int]:
        if value != dict(sorted(value.items())) or any(count < 0 for count in value.values()):
            raise ValueError(f"metasyn_synthesis_public_counts_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> MetaSynSynthesisYieldPublicSummaryV1:
        if self.caveats != list(_PUBLIC_CAVEATS):
            raise ValueError("metasyn_synthesis_public_caveats_mismatch")
        if sum(self.publication_stage_counts.values()) != self.publication_count:
            raise ValueError("metasyn_synthesis_public_publication_count_mismatch")
        if sum(self.synthesis_group_stage_counts.values()) != self.compatibility_group_count:
            raise ValueError("metasyn_synthesis_public_group_count_mismatch")
        runtime_terminal = self.publication_stage_counts.get(
            "runtime_terminal_fragment_excluded", 0
        )
        if self.runtime_contract_typed_publication_count != (
            self.publication_count - runtime_terminal
        ):
            raise ValueError("metasyn_synthesis_public_runtime_typed_count_mismatch")
        if self.diagnostic_only_typed_publication_count != (
            self.publication_stage_counts.get("diagnostic_only_fragment_excluded", 0)
        ):
            raise ValueError("metasyn_synthesis_public_diagnostic_count_mismatch")
        if self.release_grade_estimable_publication_count != (
            self.publication_stage_counts.get("release_grade_fragment_estimable", 0)
        ):
            raise ValueError("metasyn_synthesis_public_estimable_count_mismatch")
        if self.original_source_grounding_attempt_count != (
            self.runtime_contract_typed_publication_count
        ):
            raise ValueError("metasyn_synthesis_public_grounding_attempt_count_mismatch")
        if not (
            self.release_grade_estimable_publication_count
            <= self.original_source_grounding_authorized_count
            <= self.original_source_grounding_attempt_count
        ):
            raise ValueError("metasyn_synthesis_public_grounding_count_order_invalid")
        attempted_stages = sum(
            self.synthesis_group_stage_counts.get(stage, 0)
            for stage in (
                "synthesis_attempted_insufficient",
                "synthesis_completed",
            )
        )
        if not (
            self.synthesis_input_group_count
            == self.synthesis_attempted_group_count
            == attempted_stages
        ):
            raise ValueError("metasyn_synthesis_public_attempted_count_mismatch")
        if self.synthesis_completed_group_count != (
            self.synthesis_group_stage_counts.get("synthesis_completed", 0)
        ) or self.synthesis_completed_group_count != sum(
            self.synthesis_completion_mode_counts.values()
        ):
            raise ValueError("metasyn_synthesis_public_completed_count_mismatch")
        if (
            self.questions_with_estimable_graph > self.graph_estimate_count
            or self.questions_with_completed_synthesis
            > self.synthesis_completed_group_count
            or any(count > self.question_count for count in self.blocker_counts.values())
            or any(
                count > self.question_count
                for count in self.residual_conflict_counts.values()
            )
        ):
            raise ValueError("metasyn_synthesis_public_question_count_bound_invalid")
        payload = self.model_dump(mode="json", exclude={"summary_sha256"})
        if hash_canonical(payload) != self.summary_sha256:
            raise ValueError("metasyn_synthesis_public_summary_hash_mismatch")
        return self


def _validate_exact_input_join(
    *,
    runtime: MetaSynBoundedPrivateYieldReportV1,
    adapter: MetaSynBoundedAdapterBundleV1,
    prepared: MetaSynTypedPilotPrepareBundleV1,
) -> tuple[
    dict[str, MetaSynBoundedRowContextV1],
    dict[str, MetaSynRuntimeRowResultV1],
    dict[str, MetaSynPilotQuestionBundleV1],
]:
    if runtime.adapter_bundle_sha256 != adapter.adapter_bundle_sha256:
        raise MetaSynSynthesisYieldError("metasyn_synthesis_runtime_adapter_mismatch")
    if adapter.prepare_bundle_sha256 != prepared.prepare_bundle_sha256:
        raise MetaSynSynthesisYieldError("metasyn_synthesis_adapter_prepare_mismatch")
    if adapter.upstream_pilot_pipeline_sha256 != prepared.pilot_pipeline_sha256:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_adapter_prepare_pipeline_mismatch"
        )
    if adapter.official_native_schema_sha256 != (
        prepared.official_native_extraction_schema_sha256
    ):
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_adapter_prepare_native_schema_mismatch"
        )
    if runtime.row_membership_sha256 != adapter.row_membership_sha256:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_runtime_adapter_membership_mismatch"
        )
    downstream = {
        runtime.downstream_verifier_pipeline_sha256,
        prepared.downstream_verifier_pipeline_sha256,
    }
    if len(downstream) != 1:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_downstream_verifier_identity_mismatch"
        )
    adapter_rows = {row.row_context_sha256: row for row in adapter.row_contexts}
    runtime_rows = {row.row_context_sha256: row for row in runtime.row_results}
    if len(adapter_rows) != EXPECTED_SELECTED_PAPERS or set(adapter_rows) != set(
        runtime_rows
    ):
        raise MetaSynSynthesisYieldError("metasyn_synthesis_runtime_row_roster_mismatch")
    questions = {item.question_spec.question_id: item for item in prepared.questions}
    if len(questions) != EXPECTED_SELECTED_QUESTIONS:
        raise MetaSynSynthesisYieldError("metasyn_synthesis_question_roster_mismatch")
    prepared_sources = {
        row.source_row_sha256: (question, row)
        for question in prepared.questions
        for row in question.source_rows
    }
    if len(prepared_sources) != EXPECTED_SELECTED_PAPERS:
        raise MetaSynSynthesisYieldError("metasyn_synthesis_prepare_source_roster_mismatch")
    for row_hash, adapter_row in adapter_rows.items():
        runtime_row = runtime_rows[row_hash]
        prepared_pair = prepared_sources.get(adapter_row.source_row_sha256)
        if prepared_pair is None:
            raise MetaSynSynthesisYieldError(
                "metasyn_synthesis_adapter_source_not_in_prepare"
            )
        question, source_row = prepared_pair
        if source_row != adapter_row.source_row:
            raise MetaSynSynthesisYieldError(
                "metasyn_synthesis_adapter_prepare_source_snapshot_mismatch"
            )
        if (
            adapter_row.question_bundle_sha256 != question.question_bundle_sha256
            or adapter_row.question_spec != question.question_spec
            or adapter_row.independence_component_id
            != question.independence_component_id
            or adapter_row.independence_component_membership_sha256
            != question.independence_component_membership_sha256
        ):
            raise MetaSynSynthesisYieldError(
                "metasyn_synthesis_adapter_prepare_question_lineage_mismatch"
            )
        if (
            runtime_row.question_spec_sha256 != adapter_row.question_spec_sha256
            or runtime_row.question_bundle_sha256 != adapter_row.question_bundle_sha256
            or runtime_row.source_row_sha256 != adapter_row.source_row_sha256
            or runtime_row.independence_component_membership_sha256
            != adapter_row.independence_component_membership_sha256
        ):
            raise MetaSynSynthesisYieldError(
                "metasyn_synthesis_runtime_adapter_row_lineage_mismatch"
            )
    return adapter_rows, runtime_rows, questions


def _quote_groundings(
    runtime_row: MetaSynRuntimeRowResultV1,
) -> list[MetaSynUniqueQuoteGroundingV1]:
    result = runtime_row.adapter_publication_result
    if result is None:
        return []
    values = [
        receipt.quote_grounding
        for receipt in result.packet_receipts
        if receipt.quote_grounding is not None
    ]
    values.sort(key=lambda item: item.grounding_sha256)
    return values


def _freeze_publication_record(
    *,
    repository_root: Path,
    pipeline_sha256: str,
    adapter_row: MetaSynBoundedRowContextV1,
    runtime_row: MetaSynRuntimeRowResultV1,
) -> MetaSynSynthesisPublicationRecordV1:
    source = adapter_row.source_row.source_record
    publication = source.publication
    source_document = source.source_document
    typed = runtime_row.status == "typed_publication_output"
    result = runtime_row.adapter_publication_result
    official = result.official_output if result is not None else None
    quote_groundings = _quote_groundings(runtime_row)
    grounding_receipt: NativeGroundingReceipt | None = None
    if typed:
        if official is None:
            raise MetaSynSynthesisYieldError(
                "metasyn_synthesis_typed_runtime_output_missing"
            )
        grounding_receipt = verify_native_publication_grounding(
            repository_root=repository_root,
            source_document=source_document,
            extraction=official,
        )
    if typed and adapter_row.source_row.release_grade_source_grounding_eligible:
        assert official is not None
        assert grounding_receipt is not None
        fragment = freeze_grounding_checked_publication_fragment(
            extraction=official,
            grounding_receipt=grounding_receipt,
            question_id=adapter_row.question_spec.question_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_sha256,
            source_document=source_document,
        )
        if fragment.status is FragmentStatus.ESTIMABLE:
            stage: PublicationStage = "release_grade_fragment_estimable"
            blockers: list[str] = []
        else:
            stage = "original_source_grounding_failed"
            blockers = [
                "original_source_grounding_not_authorizing",
                *(f"original_source_grounding:{item}" for item in grounding_receipt.issues),
                *(
                    f"original_source_finding:{item.status.value}"
                    for item in grounding_receipt.finding_results
                    if item.status.value != "exact"
                ),
            ]
    elif typed:
        assert grounding_receipt is not None
        fragment = freeze_publication_evidence_fragment(
            question_id=adapter_row.question_spec.question_id,
            publication_id=publication.publication_id,
            paper_id=publication.paper_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_sha256,
            source_document=source_document,
            grounding_receipt_sha256=grounding_receipt.receipt_sha256,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=NonEstimabilityReason.SOURCE_DOCUMENT_INCOMPLETE,
            non_estimability_detail=(
                "The frozen source surface is diagnostic title/abstract or otherwise not "
                "release-grade full text; it is excluded from synthesis regardless of "
                "structural extraction and lexical grounding."
            ),
            extractor_warnings=[
                "diagnostic_source_surface_excluded_from_synthesis",
                *adapter_row.source_row.source_strength_blockers,
            ],
        )
        stage = "diagnostic_only_fragment_excluded"
        blockers = [
            "diagnostic_source_surface_not_synthesis_eligible",
            *(
                f"source_strength:{item}"
                for item in adapter_row.source_row.source_strength_blockers
            ),
        ]
    else:
        detail = (
            "The bounded runtime terminated this publication without a complete typed "
            f"output ({runtime_row.status}); the row remains in the corpus accounting "
            "but contributes no graph estimate."
        )
        fragment = freeze_publication_evidence_fragment(
            question_id=adapter_row.question_spec.question_id,
            publication_id=publication.publication_id,
            paper_id=publication.paper_id,
            publication=publication,
            pipeline_fingerprint_sha256=pipeline_sha256,
            source_document=source_document,
            grounding_receipt_sha256=None,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=NonEstimabilityReason.OTHER,
            non_estimability_detail=detail,
            extractor_warnings=[
                f"runtime_terminal_status:{runtime_row.status}",
                *(f"runtime_blocker:{item}" for item in runtime_row.runtime_blockers),
            ],
        )
        stage = "runtime_terminal_fragment_excluded"
        blockers = [
            f"runtime_terminal:{runtime_row.status}",
            *(f"runtime:{item}" for item in runtime_row.runtime_blockers),
        ]
    blockers = sorted(set(blockers))
    payload: dict[str, Any] = {
        "record_version": PUBLICATION_RECORD_VERSION,
        "question_id": adapter_row.question_spec.question_id,
        "question_spec_sha256": adapter_row.question_spec_sha256,
        "question_bundle_sha256": adapter_row.question_bundle_sha256,
        "row_context_sha256": adapter_row.row_context_sha256,
        "runtime_row_result_sha256": runtime_row.row_result_sha256,
        "source_row_sha256": adapter_row.source_row_sha256,
        "publication_id": publication.publication_id,
        "paper_id": publication.paper_id,
        "source_locator": source_document.source_locator,
        "source_document_sha256": source_document.sha256,
        "runtime_status": runtime_row.status,
        "runtime_contract_typed": typed,
        "release_grade_source_grounding_eligible": (
            adapter_row.source_row.release_grade_source_grounding_eligible
        ),
        "adapter_publication_result_sha256": (
            result.result_sha256 if result is not None else None
        ),
        "adapter_quote_groundings": quote_groundings,
        "adapter_quote_grounding_sha256s": sorted(
            item.grounding_sha256 for item in quote_groundings
        ),
        "original_source_grounding_receipt": grounding_receipt,
        "original_source_grounding_receipt_sha256": (
            grounding_receipt.receipt_sha256 if grounding_receipt is not None else None
        ),
        "original_source_grounding_authorized": bool(
            grounding_receipt is not None
            and grounding_receipt.authorizes_estimable_fragment
        ),
        "stage": stage,
        "blockers": blockers,
        "terminal_fragment": fragment,
        "terminal_fragment_sha256": fragment.fragment_sha256,
    }
    return MetaSynSynthesisPublicationRecordV1.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )


def _compatibility_descriptor(
    graph: EvidenceGraph, estimate_id: str
) -> tuple[dict[str, Any], str]:
    estimates = {item.estimate_id: item for item in graph.outcome_estimates}
    contrasts = {item.contrast_id: item for item in graph.contrasts}
    estimate = estimates[estimate_id]
    contrast = contrasts[estimate.contrast_id]
    harmonized = harmonize_effect(estimate.effect)
    descriptor: dict[str, Any] = {
        "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
        "outcome_name": estimate.outcome_name,
        "contrast_label": contrast.label,
        "contrast_estimand": contrast.estimand,
        "positive_direction_means": contrast.positive_direction_means,
        "timepoint_sha256": hash_canonical(estimate.timepoint),
        "analysis_population": estimate.analysis_population,
        "harmonization_status": harmonized.status,
        "harmonized_measure": (
            harmonized.effect.measure.value if harmonized.effect is not None else None
        ),
        "harmonized_unit": (
            harmonized.effect.unit if harmonized.effect is not None else None
        ),
        "harmonization_reason": harmonized.reason,
    }
    return descriptor, hash_canonical(descriptor)


def _subgraph(graph: EvidenceGraph, estimate_ids: Sequence[str]) -> EvidenceGraph:
    selected_ids = set(estimate_ids)
    estimates = [
        item for item in graph.outcome_estimates if item.estimate_id in selected_ids
    ]
    if len(estimates) != len(selected_ids):
        raise MetaSynSynthesisYieldError("metasyn_synthesis_group_estimate_missing")
    contrast_ids = {item.contrast_id for item in estimates}
    contrasts = [item for item in graph.contrasts if item.contrast_id in contrast_ids]
    cohort_ids = {item.cohort_id for item in contrasts}
    cohorts = [item for item in graph.cohorts if item.cohort_id in cohort_ids]
    study_ids = {item.study_id for item in cohorts}
    studies = [item for item in graph.studies if item.study_id in study_ids]
    publication_ids = {
        publication_id for item in studies for publication_id in item.publication_ids
    }
    paper_ids = {item.effect.paper_id for item in estimates}
    publications = [
        item
        for item in graph.publications
        if item.publication_id in publication_ids or item.paper_id in paper_ids
    ]
    arm_ids = {
        arm_id
        for item in contrasts
        for arm_id in (item.treatment_arm_id, item.comparator_arm_id)
    }
    arms = [item for item in graph.arms if item.arm_id in arm_ids]
    span_ids = {span_id for item in estimates for span_id in item.evidence_span_ids}
    for study in studies:
        for domain in study.risk_of_bias.domains:
            span_ids.update(domain.evidence_span_ids)
    for estimate in estimates:
        for domain in estimate.risk_of_bias.domains:
            span_ids.update(domain.evidence_span_ids)
    spans = [item for item in graph.evidence_spans if item.span_id in span_ids]
    return EvidenceGraph(
        publications=sorted(publications, key=lambda item: item.publication_id),
        studies=sorted(studies, key=lambda item: item.study_id),
        cohorts=sorted(cohorts, key=lambda item: item.cohort_id),
        arms=sorted(arms, key=lambda item: item.arm_id),
        contrasts=sorted(contrasts, key=lambda item: item.contrast_id),
        outcome_estimates=sorted(estimates, key=lambda item: item.estimate_id),
        evidence_spans=sorted(spans, key=lambda item: item.span_id),
    )


def _freeze_compatibility_groups(
    *,
    graph: EvidenceGraph,
    reconciliation: NativeCohortReconciliationReceipt,
) -> list[MetaSynCompatibilityGroupYieldV1]:
    grouped: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for estimate in graph.outcome_estimates:
        descriptor, digest = _compatibility_descriptor(graph, estimate.estimate_id)
        existing = grouped.get(digest)
        if existing is None:
            grouped[digest] = (descriptor, [estimate.estimate_id])
        else:
            if existing[0] != descriptor:
                raise MetaSynSynthesisYieldError(
                    "metasyn_synthesis_compatibility_hash_collision"
                )
            existing[1].append(estimate.estimate_id)
    output: list[MetaSynCompatibilityGroupYieldV1] = []
    for compatibility_sha in sorted(grouped):
        descriptor, estimate_ids = grouped[compatibility_sha]
        estimate_ids = sorted(estimate_ids)
        group_graph = _subgraph(graph, estimate_ids)
        paper_ids = sorted(
            {item.effect.paper_id for item in group_graph.outcome_estimates}
        )
        cohort_ids = sorted({item.cohort_id for item in group_graph.cohorts})
        if len(paper_ids) == 1:
            assurance = "single_publication_intrinsic"
        elif reconciliation.cross_publication_identity_assurance_complete:
            assurance = "reconciliation_receipt_complete"
        else:
            assurance = "unresolved_blocked"
        synthesis: dict[str, Any] | None = None
        if descriptor["harmonization_status"] != "estimable":
            blocker = f"harmonization:{descriptor['harmonization_reason']}"
            stage: GroupStage = "blocked_harmonization"
        elif assurance == "unresolved_blocked":
            blocker = "cross_publication_cohort_identity_unresolved"
            stage = "blocked_cross_publication_identity"
        else:
            blocker = None
            synthesis = synthesize_evidence_graph(
                group_graph,
                outcome_name=descriptor["outcome_name"],
                require_explicit_timepoint=True,
            )
            stage = (
                "synthesis_completed"
                if synthesis.get("status") == "ok"
                else "synthesis_attempted_insufficient"
            )
        completed = bool(synthesis is not None and synthesis.get("status") == "ok")
        payload: dict[str, Any] = {
            "group_version": GROUP_RECORD_VERSION,
            "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
            "compatibility_sha256": compatibility_sha,
            **descriptor,
            "estimate_ids": estimate_ids,
            "paper_ids": paper_ids,
            "cohort_ids": cohort_ids,
            "group_graph_sha256": hash_canonical(group_graph),
            "cross_publication_identity_assurance": assurance,
            "synthesis_input_eligible": synthesis is not None,
            "synthesis_attempted": synthesis is not None,
            "synthesis_completed": completed,
            "stage": stage,
            "blocker": blocker,
            "synthesis_status": synthesis.get("status") if synthesis is not None else None,
            "synthesis_mode": synthesis.get("mode") if synthesis is not None else None,
            "synthesis_reason": synthesis.get("reason") if synthesis is not None else None,
            "synthesis_sha256": hash_canonical(synthesis) if synthesis is not None else None,
        }
        output.append(
            MetaSynCompatibilityGroupYieldV1.model_validate(
                {**payload, "group_sha256": hash_canonical(payload)}
            )
        )
    return output


def _question_blockers(
    publication_records: Sequence[MetaSynSynthesisPublicationRecordV1],
    groups: Sequence[MetaSynCompatibilityGroupYieldV1],
) -> list[str]:
    blockers = {
        blocker
        for record in publication_records
        for blocker in record.blockers
    }
    blockers.update(group.blocker for group in groups if group.blocker is not None)
    blockers.update(
        f"synthesis:{group.synthesis_reason}"
        for group in groups
        if group.synthesis_attempted
        and not group.synthesis_completed
        and group.synthesis_reason is not None
    )
    if not groups:
        blockers.add("no_release_grade_source_replayed_graph_estimates")
    return sorted(blockers)


def _residual_conflicts(
    *,
    graph: EvidenceGraph,
    reconciliation: NativeCohortReconciliationReceipt,
    groups: Sequence[MetaSynCompatibilityGroupYieldV1],
) -> list[str]:
    conflicts = {
        f"cohort_reconciliation:{issue.code}" for issue in reconciliation.issues
    }
    by_outcome = Counter(group.outcome_name for group in groups)
    if any(count > 1 for count in by_outcome.values()):
        conflicts.add("multiple_effect_compatibility_strata_preserved")
    if any(
        group.cross_publication_identity_assurance == "unresolved_blocked"
        for group in groups
    ):
        conflicts.add("unresolved_cross_publication_cohort_identity_preserved")
    if any(group.harmonization_status == "insufficient" for group in groups):
        conflicts.add("unharmonizable_effect_stratum_preserved")
    contrast_orientations = {
        (item.label, item.estimand, item.positive_direction_means)
        for item in graph.contrasts
    }
    if len(contrast_orientations) > 1:
        conflicts.add("multiple_contrast_orientation_or_estimand_strata_preserved")
    return sorted(conflicts)


def _freeze_question_report(
    *,
    repository_root: Path,
    pipeline_sha256: str,
    question: MetaSynPilotQuestionBundleV1,
    adapter_rows: Mapping[str, MetaSynBoundedRowContextV1],
    runtime_rows: Mapping[str, MetaSynRuntimeRowResultV1],
) -> MetaSynQuestionSynthesisYieldV1:
    question_adapter_rows = sorted(
        (
            row
            for row in adapter_rows.values()
            if row.question_spec.question_id == question.question_spec.question_id
        ),
        key=lambda item: item.source_row.source_record.publication.publication_id,
    )
    if len(question_adapter_rows) != len(question.oracle_corpus_ids):
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_question_runtime_oracle_roster_mismatch"
        )
    records = [
        _freeze_publication_record(
            repository_root=repository_root,
            pipeline_sha256=pipeline_sha256,
            adapter_row=row,
            runtime_row=runtime_rows[row.row_context_sha256],
        )
        for row in question_adapter_rows
    ]
    corpus = assemble_typed_evidence_corpus(
        [record.terminal_fragment for record in records]
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    if reconciliation.reconciled_graph is None:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_reconciled_graph_missing"
        )
    groups = _freeze_compatibility_groups(
        graph=reconciliation.reconciled_graph,
        reconciliation=reconciliation,
    )
    blockers = _question_blockers(records, groups)
    conflicts = _residual_conflicts(
        graph=reconciliation.reconciled_graph,
        reconciliation=reconciliation,
        groups=groups,
    )
    graph = corpus.graph
    payload: dict[str, Any] = {
        "question_report_version": QUESTION_RECORD_VERSION,
        "question_id": question.question_spec.question_id,
        "question_spec_sha256": question.question_spec_sha256,
        "question_bundle_sha256": question.question_bundle_sha256,
        "independence_component_id": question.independence_component_id,
        "independence_component_membership_sha256": (
            question.independence_component_membership_sha256
        ),
        "oracle_roster_membership_sha256": question.oracle_roster_membership_sha256,
        "publication_records": records,
        "publication_record_sha256s": sorted(item.record_sha256 for item in records),
        "terminal_corpus": corpus,
        "terminal_corpus_sha256": corpus.corpus_sha256,
        "cohort_reconciliation": reconciliation,
        "cohort_reconciliation_sha256": reconciliation.receipt_sha256,
        "compatibility_groups": groups,
        "compatibility_group_sha256s": sorted(item.group_sha256 for item in groups),
        "graph_construction_completed": True,
        "graph_estimate_count": len(graph.outcome_estimates),
        "graph_study_count": len(graph.studies),
        "graph_cohort_count": len(graph.cohorts),
        "release_grade_estimable_publication_count": sum(
            item.stage == "release_grade_fragment_estimable" for item in records
        ),
        "synthesis_input_group_count": sum(item.synthesis_input_eligible for item in groups),
        "synthesis_attempted_group_count": sum(item.synthesis_attempted for item in groups),
        "synthesis_completed_group_count": sum(item.synthesis_completed for item in groups),
        "blockers": blockers,
        "residual_conflicts": conflicts,
    }
    return MetaSynQuestionSynthesisYieldV1.model_validate(
        {**payload, "question_report_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_synthesis_yield_report(
    *,
    repository_root: Path,
    runtime_report: MetaSynBoundedPrivateYieldReportV1 | Mapping[str, Any],
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1 | Mapping[str, Any],
) -> MetaSynSynthesisYieldReportV1:
    """Replay source grounding and freeze the exact 10-question yield report in memory."""

    root = repository_root.resolve(strict=True)
    runtime = MetaSynBoundedPrivateYieldReportV1.model_validate(runtime_report)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(prepare_bundle)
    adapter_rows, runtime_rows, questions = _validate_exact_input_join(
        runtime=runtime, adapter=adapter, prepared=prepared
    )
    fingerprint = compute_metasyn_synthesis_yield_fingerprint(
        repository_root=root,
        runtime_report=runtime,
        adapter_bundle=adapter,
        prepare_bundle=prepared,
    )
    question_reports = [
        _freeze_question_report(
            repository_root=root,
            pipeline_sha256=fingerprint.pipeline_sha256,
            question=questions[question_id],
            adapter_rows=adapter_rows,
            runtime_rows=runtime_rows,
        )
        for question_id in sorted(questions)
    ]
    publications = [
        record
        for question in question_reports
        for record in question.publication_records
    ]
    groups = [
        group for question in question_reports for group in question.compatibility_groups
    ]
    payload: dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "status": "complete_label_blind_synthesis_yield_evaluation",
        "evaluation_pipeline_fingerprint": fingerprint,
        "evaluation_pipeline_sha256": fingerprint.pipeline_sha256,
        "runtime_private_report_sha256": runtime.report_sha256,
        "runtime_pipeline_sha256": runtime.runtime_pipeline_sha256,
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "prepare_bundle_sha256": prepared.prepare_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            runtime.downstream_verifier_pipeline_sha256
        ),
        "question_count": EXPECTED_SELECTED_QUESTIONS,
        "component_count": EXPECTED_SELECTED_COMPONENTS,
        "publication_count": EXPECTED_SELECTED_PAPERS,
        "question_reports": question_reports,
        "question_report_sha256s": sorted(
            item.question_report_sha256 for item in question_reports
        ),
        "runtime_contract_typed_publication_count": sum(
            item.runtime_contract_typed for item in publications
        ),
        "diagnostic_only_typed_publication_count": sum(
            item.runtime_contract_typed
            and not item.release_grade_source_grounding_eligible
            for item in publications
        ),
        "original_source_grounding_attempt_count": sum(
            item.original_source_grounding_receipt is not None for item in publications
        ),
        "original_source_grounding_authorized_count": sum(
            item.original_source_grounding_authorized for item in publications
        ),
        "release_grade_estimable_publication_count": sum(
            item.stage == "release_grade_fragment_estimable" for item in publications
        ),
        "terminal_fragment_count": len(publications),
        "graph_construction_completed_question_count": len(question_reports),
        "questions_with_estimable_graph": sum(
            item.graph_estimate_count > 0 for item in question_reports
        ),
        "graph_estimate_count": sum(item.graph_estimate_count for item in question_reports),
        "compatibility_group_count": len(groups),
        "synthesis_input_group_count": sum(item.synthesis_input_eligible for item in groups),
        "synthesis_attempted_group_count": sum(item.synthesis_attempted for item in groups),
        "synthesis_completed_group_count": sum(item.synthesis_completed for item in groups),
        "questions_with_completed_synthesis": sum(
            item.synthesis_completed_group_count > 0 for item in question_reports
        ),
        "publication_stage_counts": dict(
            sorted(Counter(item.stage for item in publications).items())
        ),
        "synthesis_group_stage_counts": dict(
            sorted(Counter(item.stage for item in groups).items())
        ),
        "synthesis_completion_mode_counts": dict(
            sorted(
                Counter(
                    item.synthesis_mode
                    for item in groups
                    if item.synthesis_completed and item.synthesis_mode is not None
                ).items()
            )
        ),
        "blocker_counts": _aggregate_blocker_counts(question_reports),
        "residual_conflict_counts": _aggregate_residual_conflict_counts(
            question_reports
        ),
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "truth_or_clinical_benefit_reported": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
        ),
    }
    return MetaSynSynthesisYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_synthesis_yield_public_summary(
    *, report: MetaSynSynthesisYieldReportV1 | Mapping[str, Any]
) -> MetaSynSynthesisYieldPublicSummaryV1:
    """Derive an identifier- and source-free aggregate summary in memory."""

    canonical = MetaSynSynthesisYieldReportV1.model_validate(report)
    copied_fields = (
        "runtime_contract_typed_publication_count",
        "diagnostic_only_typed_publication_count",
        "original_source_grounding_attempt_count",
        "original_source_grounding_authorized_count",
        "release_grade_estimable_publication_count",
        "terminal_fragment_count",
        "graph_construction_completed_question_count",
        "questions_with_estimable_graph",
        "graph_estimate_count",
        "compatibility_group_count",
        "synthesis_input_group_count",
        "synthesis_attempted_group_count",
        "synthesis_completed_group_count",
        "questions_with_completed_synthesis",
        "publication_stage_counts",
        "synthesis_group_stage_counts",
        "synthesis_completion_mode_counts",
        "blocker_counts",
        "residual_conflict_counts",
    )
    payload: dict[str, Any] = {
        "summary_version": PUBLIC_SUMMARY_VERSION,
        "status": "aggregate_only_label_blind_synthesis_yield",
        "evaluation_pipeline_sha256": canonical.evaluation_pipeline_sha256,
        "runtime_private_report_sha256": canonical.runtime_private_report_sha256,
        "synthesis_private_report_sha256": canonical.report_sha256,
        "adapter_bundle_sha256": canonical.adapter_bundle_sha256,
        "prepare_bundle_sha256": canonical.prepare_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            canonical.downstream_verifier_pipeline_sha256
        ),
        "question_count": canonical.question_count,
        "component_count": canonical.component_count,
        "publication_count": canonical.publication_count,
        **{field: getattr(canonical, field) for field in copied_fields},
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "truth_or_clinical_benefit_reported": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_graph_synthesis_input_and_synthesis_completion_yield_only"
        ),
        "caveats": list(_PUBLIC_CAVEATS),
    }
    return MetaSynSynthesisYieldPublicSummaryV1.model_validate(
        {**payload, "summary_sha256": hash_canonical(payload)}
    )


def validate_metasyn_synthesis_yield_report(
    *,
    report: MetaSynSynthesisYieldReportV1 | Mapping[str, Any],
    repository_root: Path,
    runtime_report: MetaSynBoundedPrivateYieldReportV1 | Mapping[str, Any],
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1 | Mapping[str, Any],
) -> MetaSynSynthesisYieldReportV1:
    """Externally replay the entire source→fragment→graph→synthesis yield closure."""

    canonical = MetaSynSynthesisYieldReportV1.model_validate(report)
    replayed = freeze_metasyn_synthesis_yield_report(
        repository_root=repository_root,
        runtime_report=runtime_report,
        adapter_bundle=adapter_bundle,
        prepare_bundle=prepare_bundle,
    )
    if replayed != canonical:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_yield_external_replay_mismatch"
        )
    return canonical


def validate_metasyn_synthesis_yield_public_summary(
    *,
    summary: MetaSynSynthesisYieldPublicSummaryV1 | Mapping[str, Any],
    report: MetaSynSynthesisYieldReportV1 | Mapping[str, Any],
) -> MetaSynSynthesisYieldPublicSummaryV1:
    canonical = MetaSynSynthesisYieldPublicSummaryV1.model_validate(summary)
    expected = freeze_metasyn_synthesis_yield_public_summary(report=report)
    if canonical != expected:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_public_summary_external_replay_mismatch"
        )
    return canonical


def _read_prepare_bundle_after_external_replay(
    *, repository_root: Path, pilot_workspace: Path
) -> MetaSynTypedPilotPrepareBundleV1:
    root = repository_root.resolve(strict=True)
    receipt = validate_metasyn_typed_pilot_prepare(
        repository_root=root, workspace=pilot_workspace
    )
    workspace = pilot_workspace if pilot_workspace.is_absolute() else root / pilot_workspace
    path = workspace.resolve(strict=True) / PREPARE_BUNDLE_FILENAME
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_prepare_bundle_unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise MetaSynSynthesisYieldError("metasyn_synthesis_prepare_bundle_not_object")
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(payload)
    if (
        hashlib.sha256(raw).hexdigest() != receipt.prepare_bundle_file_sha256
        or prepared.prepare_bundle_sha256 != receipt.prepare_bundle_sha256
    ):
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_prepare_bundle_post_replay_mismatch"
        )
    return prepared


def evaluate_current_metasyn_synthesis_yield(
    *,
    repository_root: Path,
    runtime_workspace: Path,
    pilot_workspace: Path,
    expected_execution_bundle_sha256: str,
) -> tuple[MetaSynSynthesisYieldReportV1, MetaSynSynthesisYieldPublicSummaryV1]:
    """Replay the finalized private runtime and derive both reports without writing."""

    root = repository_root.resolve(strict=True)
    runtime_report, _ = validate_metasyn_bounded_finalized_runtime(
        workspace=runtime_workspace,
        repository_root=root,
        expected_execution_bundle_sha256=expected_execution_bundle_sha256,
    )
    _, execution_bundle = load_current_metasyn_bounded_execution_bundle(
        workspace=runtime_workspace, repository_root=root, external_replay=True
    )
    if execution_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynSynthesisYieldError(
            "metasyn_synthesis_execution_bundle_anchor_mismatch"
        )
    adapter = validate_metasyn_bounded_adapter_bundle_external_replay(
        adapter_bundle=execution_bundle.adapter_bundle,
        repository_root=root,
        workspace=pilot_workspace,
    )
    prepared = _read_prepare_bundle_after_external_replay(
        repository_root=root, pilot_workspace=pilot_workspace
    )
    report = freeze_metasyn_synthesis_yield_report(
        repository_root=root,
        runtime_report=runtime_report,
        adapter_bundle=adapter,
        prepare_bundle=prepared,
    )
    return report, freeze_metasyn_synthesis_yield_public_summary(report=report)


__all__ = [
    "COMPATIBILITY_RULE_VERSION",
    "EVALUATION_VERSION",
    "MetaSynCompatibilityGroupYieldV1",
    "MetaSynQuestionSynthesisYieldV1",
    "MetaSynSynthesisPublicationRecordV1",
    "MetaSynSynthesisYieldError",
    "MetaSynSynthesisYieldPublicSummaryV1",
    "MetaSynSynthesisYieldReportV1",
    "compute_metasyn_synthesis_yield_fingerprint",
    "evaluate_current_metasyn_synthesis_yield",
    "freeze_metasyn_synthesis_yield_public_summary",
    "freeze_metasyn_synthesis_yield_report",
    "validate_metasyn_synthesis_yield_public_summary",
    "validate_metasyn_synthesis_yield_report",
]
