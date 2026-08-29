"""Label-blind quantitative-mechanics join for grounded MetaSyn v2 artifacts.

This additive boundary externally replays a complete
``MetaSynGroundedPublicationCorpusBridgeV2``, reconciles publication-scoped
study/cohort identities, and invokes the repository's real cohort-unit synthesis
kernel only when independence is explicitly resolved.  It reports mechanics and
yield only.  A computed result is not synthesis-input authorization, a scientific
conclusion, an accuracy claim, or claim-release authority.

The compatibility bridge necessarily carries legacy storage sentinels for optional
scientific fields.  This module never treats those sentinels as observations: it
does not request moderator analysis, does not classify reported significance, and
does not interpret equivalence.  Directional fallback may be retained in a raw
kernel receipt for replay, but it never counts as completed quantitative mechanics.
"""

from __future__ import annotations

import ast
import os
import platform
import stat
from collections.abc import Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
    NativeReconciliationStatus,
    ReviewerCohortReconciliationArtifact,
    reconcile_native_cohorts,
    reverify_native_cohort_reconciliation,
)
from literature_multiverse.evidence_graph import (
    ArmRole,
    EvidenceGraph,
    OutcomeEstimateNode,
    OutcomeTimepoint,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.metasyn_grounded_publication_bridge_v2 import (
    MetaSynGroundedPublicationCorpusBridgeV2,
    MetaSynGroundedQuestionCorpusV2,
    validate_metasyn_grounded_publication_bridge_v2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

ANALYSIS_VERSION = "metasyn-grounded-analysis-v2"
ANALYSIS_COMPONENT_VERSION = "1"
QUESTION_ANALYSIS_VERSION = "metasyn-grounded-question-analysis-v2"
UNIT_SIGNATURE_VERSION = "metasyn-grounded-quantitative-unit-signature-v2"
UNIT_VERSION = "metasyn-grounded-quantitative-kernel-unit-v2"

EXPECTED_QUESTION_COUNT = 10
EXPECTED_PUBLICATION_COUNT = 32

ANALYSIS_MODULE_PATH = "src/literature_multiverse/metasyn_grounded_analysis_v2.py"
ANALYSIS_CLI_PATH = "scripts/run_metasyn_grounded_analysis_v2.py"
_DEPENDENCY_ENTRYPOINTS = (ANALYSIS_MODULE_PATH, ANALYSIS_CLI_PATH)
_NON_PYTHON_INPUTS = ("pyproject.toml", "uv.lock")
_INSTALLED_DEPENDENCIES = ("numpy", "pydantic", "scipy")

_MECHANICS_POLICY = {
    "assumed_within_cohort_correlation": 1.0,
    "confidence_level": 0.95,
    "directional_fallback_completed_quantitative_mechanics": False,
    "moderator_analysis_permitted": False,
    "require_explicit_timepoint": True,
    "reviewer_artifact_absence_infers_independence": False,
}


class MetaSynGroundedAnalysisV2Error(ValueError):
    """The grounded analysis join cannot be replayed without weakening a boundary."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    if getattr(model, field_name) != hash_canonical(
        model.model_dump(mode="json", exclude={field_name})
    ):
        raise ValueError(f"metasyn_grounded_analysis_v2_self_hash_mismatch:{field_name}")


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_grounded_analysis_v2_sha256_invalid:{field_name}")
    return value


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise MetaSynGroundedAnalysisV2Error(
                "metasyn_grounded_analysis_v2_repository_root_symlink"
            )
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_repository_root_invalid"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_repository_root_not_directory"
        )
    return resolved


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            raise MetaSynGroundedAnalysisV2Error(
                f"metasyn_grounded_analysis_v2_relative_import_invalid:{current_path}:{module}"
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
    raise MetaSynGroundedAnalysisV2Error(
        f"metasyn_grounded_analysis_v2_local_dependency_missing:{current_path}:{module}"
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
            raise MetaSynGroundedAnalysisV2Error(
                f"metasyn_grounded_analysis_v2_dependency_missing:{relative}"
            )
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynGroundedAnalysisV2Error(
                f"metasyn_grounded_analysis_v2_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
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


class MetaSynQuantitativeUnitSignatureV2(_FrozenExactModel):
    """Exact scientific grouping dimensions used before kernel invocation."""

    signature_version: Literal["metasyn-grounded-quantitative-unit-signature-v2"] = (
        UNIT_SIGNATURE_VERSION
    )
    outcome_name: str
    contrast_label: str
    contrast_estimand: str | None
    positive_direction_means: str
    treatment_arm_role: ArmRole
    comparator_arm_role: ArmRole
    timepoint: OutcomeTimepoint
    analysis_population: str | None
    signature_sha256: Sha256

    @model_validator(mode="after")
    def validate_signature(self) -> MetaSynQuantitativeUnitSignatureV2:
        _self_hash(self, "signature_sha256")
        return self


class MetaSynQuantitativeKernelUnitV2(_FrozenExactModel):
    """One compatible unit's kernel mechanics or exact fail-closed abstention."""

    unit_version: Literal["metasyn-grounded-quantitative-kernel-unit-v2"] = UNIT_VERSION
    question_id: str
    question_corpus_sha256: Sha256
    reconciliation_receipt_sha256: Sha256
    reconciled_graph_sha256: Sha256
    signature: MetaSynQuantitativeUnitSignatureV2
    signature_sha256: Sha256
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    contrast_ids: Annotated[list[str], Field(min_length=1)]
    cohort_ids: Annotated[list[str], Field(min_length=1)]
    paper_ids: Annotated[list[str], Field(min_length=1)]
    input_effect_count: Annotated[int, Field(ge=1)]
    input_cohort_count: Annotated[int, Field(ge=1)]
    input_publication_count: Annotated[int, Field(ge=1)]
    cohort_independence_resolved: bool
    kernel_invoked: bool
    status: Literal["quantitative_kernel_completed", "abstained"]
    abstention_reasons: list[str]
    kernel_result: dict[str, Any] | None
    kernel_result_sha256: Sha256 | None
    require_explicit_timepoint: Literal[True] = True
    confidence_level: Literal[0.95] = 0.95
    assumed_within_cohort_correlation: Literal[1.0] = 1.0
    prespecified_moderators: list[Literal["__never__"]] = Field(default_factory=list)
    moderator_inference_performed: Literal[False] = False
    reported_significance_consumed: Literal[False] = False
    equivalence_consumed: Literal[False] = False
    legacy_optional_sentinels_interpreted: Literal[False] = False
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_mechanics_authority: Literal[True] = True
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    accuracy_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    unit_sha256: Sha256

    @field_validator(
        "estimate_ids", "contrast_ids", "cohort_ids", "paper_ids", "abstention_reasons"
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"metasyn_grounded_analysis_v2_unit_list_not_canonical:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_unit(self) -> MetaSynQuantitativeKernelUnitV2:
        if self.signature_sha256 != self.signature.signature_sha256:
            raise ValueError("metasyn_grounded_analysis_v2_unit_signature_alias_mismatch")
        if self.input_effect_count != len(self.estimate_ids):
            raise ValueError("metasyn_grounded_analysis_v2_unit_effect_count_mismatch")
        if self.input_cohort_count != len(self.cohort_ids):
            raise ValueError("metasyn_grounded_analysis_v2_unit_cohort_count_mismatch")
        if self.input_publication_count != len(self.paper_ids):
            raise ValueError("metasyn_grounded_analysis_v2_unit_publication_count_mismatch")
        if self.kernel_invoked != (
            self.kernel_result is not None and self.kernel_result_sha256 is not None
        ):
            raise ValueError("metasyn_grounded_analysis_v2_kernel_presence_mismatch")
        if self.kernel_result is not None and self.kernel_result_sha256 != hash_canonical(
            self.kernel_result
        ):
            raise ValueError("metasyn_grounded_analysis_v2_kernel_hash_mismatch")
        completed = self.status == "quantitative_kernel_completed"
        if completed:
            quantitative = (
                self.kernel_result.get("quantitative")
                if isinstance(self.kernel_result, dict)
                else None
            )
            if (
                not self.cohort_independence_resolved
                or not self.kernel_invoked
                or self.kernel_result.get("status") != "ok"
                or self.kernel_result.get("mode") != "random_effects_meta_analysis"
                or not isinstance(quantitative, dict)
                or quantitative.get("status") != "ok"
                or self.abstention_reasons
            ):
                raise ValueError("metasyn_grounded_analysis_v2_completed_kernel_contract_mismatch")
        elif not self.abstention_reasons:
            raise ValueError("metasyn_grounded_analysis_v2_abstention_requires_reason")
        if self.kernel_invoked and self.kernel_result.get("condition_analysis") is not None:
            raise ValueError("metasyn_grounded_analysis_v2_moderator_analysis_forbidden")
        _self_hash(self, "unit_sha256")
        return self


class MetaSynGroundedQuestionAnalysisV2(_FrozenExactModel):
    """Question-scoped reconciliation and quantitative-mechanics ledger."""

    question_analysis_version: Literal["metasyn-grounded-question-analysis-v2"] = (
        QUESTION_ANALYSIS_VERSION
    )
    question_id: str
    question_corpus_sha256: Sha256
    compatibility_corpus_sha256: Sha256
    publication_join_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    publication_ids: Annotated[list[str], Field(min_length=1)]
    publication_count: Annotated[int, Field(ge=1)]
    quantitative_effect_count: Annotated[int, Field(ge=0)]
    bridge_coverage_blockers: list[str]
    reviewer_artifact_sha256: Sha256 | None
    reconciliation: NativeCohortReconciliationReceipt
    reconciliation_receipt_sha256: Sha256
    reconciliation_status: NativeReconciliationStatus
    reconciled_graph_sha256: Sha256
    cohort_independence_resolved: bool
    units: list[MetaSynQuantitativeKernelUnitV2]
    unit_membership_sha256: Sha256
    unit_count: Annotated[int, Field(ge=0)]
    kernel_invoked_unit_count: Annotated[int, Field(ge=0)]
    kernel_completed_unit_count: Annotated[int, Field(ge=0)]
    kernel_abstained_unit_count: Annotated[int, Field(ge=0)]
    mechanics_status: Literal[
        "no_quantitative_units",
        "cohort_independence_unresolved",
        "kernel_completed",
        "kernel_abstained",
        "mixed",
    ]
    mechanics_blockers: list[str]
    scientific_authority_blockers: Annotated[list[str], Field(min_length=1)]
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: bool
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    accuracy_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    question_analysis_sha256: Sha256

    @field_validator(
        "publication_join_sha256s",
        "publication_ids",
        "bridge_coverage_blockers",
        "mechanics_blockers",
        "scientific_authority_blockers",
    )
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"metasyn_grounded_analysis_v2_question_list_not_canonical:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_question(self) -> MetaSynGroundedQuestionAnalysisV2:
        if self.publication_count != len(self.publication_ids):
            raise ValueError("metasyn_grounded_analysis_v2_question_publication_count_mismatch")
        if self.reconciliation_receipt_sha256 != self.reconciliation.receipt_sha256:
            raise ValueError("metasyn_grounded_analysis_v2_reconciliation_hash_alias_mismatch")
        if self.reconciliation_status is not self.reconciliation.status:
            raise ValueError("metasyn_grounded_analysis_v2_reconciliation_status_alias_mismatch")
        if self.reconciled_graph_sha256 != self.reconciliation.reconciled_graph_sha256:
            raise ValueError("metasyn_grounded_analysis_v2_graph_hash_alias_mismatch")
        if self.cohort_independence_resolved != (
            self.reconciliation.cross_publication_identity_assurance_complete
        ):
            raise ValueError("metasyn_grounded_analysis_v2_independence_alias_mismatch")
        expected_reviewer = (
            self.reconciliation.reviewer_artifact.artifact_sha256
            if self.reconciliation.reviewer_artifact is not None
            else None
        )
        if self.reviewer_artifact_sha256 != expected_reviewer:
            raise ValueError("metasyn_grounded_analysis_v2_reviewer_hash_alias_mismatch")
        if [item.signature_sha256 for item in self.units] != sorted(
            {item.signature_sha256 for item in self.units}
        ):
            raise ValueError("metasyn_grounded_analysis_v2_units_not_canonical")
        if self.unit_membership_sha256 != hash_canonical([item.unit_sha256 for item in self.units]):
            raise ValueError("metasyn_grounded_analysis_v2_unit_membership_mismatch")
        invoked = sum(item.kernel_invoked for item in self.units)
        completed = sum(item.status == "quantitative_kernel_completed" for item in self.units)
        abstained = len(self.units) - completed
        if (
            self.unit_count,
            self.kernel_invoked_unit_count,
            self.kernel_completed_unit_count,
            self.kernel_abstained_unit_count,
        ) != (len(self.units), invoked, completed, abstained):
            raise ValueError("metasyn_grounded_analysis_v2_question_unit_counts_mismatch")
        expected_status = _question_mechanics_status(
            unit_count=len(self.units),
            completed_count=completed,
            independence_resolved=self.cohort_independence_resolved,
        )
        if self.mechanics_status != expected_status:
            raise ValueError("metasyn_grounded_analysis_v2_question_status_mismatch")
        if self.quantitative_kernel_compatibility != bool(self.quantitative_effect_count):
            raise ValueError("metasyn_grounded_analysis_v2_question_compatibility_mismatch")
        _self_hash(self, "question_analysis_sha256")
        return self


class MetaSynGroundedAnalysisV2(_FrozenExactModel):
    """Complete 10-question/32-publication label-blind mechanics artifact."""

    analysis_version: Literal["metasyn-grounded-analysis-v2"] = ANALYSIS_VERSION
    status: Literal["externally_replayable_label_blind_mechanics_only"] = (
        "externally_replayable_label_blind_mechanics_only"
    )
    bridge: MetaSynGroundedPublicationCorpusBridgeV2
    bridge_sha256: Sha256
    execution_bundle_sha256: Sha256
    source_surface_sha256: Sha256
    inventory_receipt_membership_sha256: Sha256
    terminal_membership_sha256: Sha256
    bridge_pipeline_sha256: Sha256
    publication_join_membership_sha256: Sha256
    question_corpus_membership_sha256: Sha256
    reviewer_artifact_membership_sha256: Sha256
    analysis_pipeline_fingerprint: PipelineFingerprint
    analysis_pipeline_sha256: Sha256
    question_analyses: Annotated[
        list[MetaSynGroundedQuestionAnalysisV2],
        Field(min_length=EXPECTED_QUESTION_COUNT, max_length=EXPECTED_QUESTION_COUNT),
    ]
    question_analysis_membership_sha256: Sha256
    question_count: Literal[10] = EXPECTED_QUESTION_COUNT
    publication_count: Literal[32] = EXPECTED_PUBLICATION_COUNT
    inventoried_candidate_count: Annotated[int, Field(ge=0)]
    authorized_candidate_count: Annotated[int, Field(ge=0)]
    terminal_candidate_count: Annotated[int, Field(ge=0)]
    completed_candidate_count: Annotated[int, Field(ge=0)]
    abstained_candidate_count: Annotated[int, Field(ge=0)]
    estimable_publication_count: Annotated[int, Field(ge=0)]
    quantitative_effect_count: Annotated[int, Field(ge=0)]
    quantitative_unit_count: Annotated[int, Field(ge=0)]
    kernel_invoked_unit_count: Annotated[int, Field(ge=0)]
    kernel_completed_unit_count: Annotated[int, Field(ge=0)]
    kernel_abstained_unit_count: Annotated[int, Field(ge=0)]
    questions_with_completed_kernel_mechanics: Annotated[int, Field(ge=0, le=10)]
    questions_with_unresolved_independence: Annotated[int, Field(ge=0, le=10)]
    bridge_external_replay_validated: Literal[True] = True
    complete_terminal_coverage_preserved: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    benchmark_conclusions_accessed: Literal[False] = False
    moderator_inference_performed: Literal[False] = False
    reported_significance_consumed: Literal[False] = False
    equivalence_consumed: Literal[False] = False
    directional_fallback_completed_quantitative_mechanics: Literal[False] = False
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: bool
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    accuracy_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    analysis_sha256: Sha256

    @model_validator(mode="after")
    def validate_analysis(self) -> MetaSynGroundedAnalysisV2:
        bridge_aliases = {
            "bridge_sha256": self.bridge.bridge_sha256,
            "execution_bundle_sha256": self.bridge.execution_bundle_sha256,
            "source_surface_sha256": self.bridge.source_surface_sha256,
            "inventory_receipt_membership_sha256": (
                self.bridge.inventory_receipt_membership_sha256
            ),
            "terminal_membership_sha256": self.bridge.terminal_membership_sha256,
            "bridge_pipeline_sha256": self.bridge.bridge_pipeline_sha256,
            "publication_join_membership_sha256": (self.bridge.publication_join_membership_sha256),
            "question_corpus_membership_sha256": (self.bridge.question_corpus_membership_sha256),
        }
        if any(getattr(self, key) != value for key, value in bridge_aliases.items()):
            raise ValueError("metasyn_grounded_analysis_v2_bridge_hash_alias_mismatch")
        if self.analysis_pipeline_sha256 != self.analysis_pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_grounded_analysis_v2_pipeline_hash_alias_mismatch")
        if len(self.analysis_pipeline_fingerprint.components) != 1:
            raise ValueError("metasyn_grounded_analysis_v2_pipeline_component_count_mismatch")
        component = self.analysis_pipeline_fingerprint.components[0]
        if (
            component.component_id != "metasyn-grounded-analysis-v2"
            or component.component_version != ANALYSIS_COMPONENT_VERSION
            or not {ANALYSIS_MODULE_PATH, ANALYSIS_CLI_PATH}.issubset(
                {item.path for item in component.files}
            )
            or component.settings
            != _pipeline_settings(
                bridge=self.bridge,
                reviewer_artifact_membership_sha256=(self.reviewer_artifact_membership_sha256),
            )
        ):
            raise ValueError("metasyn_grounded_analysis_v2_pipeline_component_mismatch")
        if [item.question_id for item in self.question_analyses] != sorted(
            {item.question_id for item in self.question_analyses}
        ):
            raise ValueError("metasyn_grounded_analysis_v2_question_roster_invalid")
        if self.question_analysis_membership_sha256 != hash_canonical(
            [item.question_analysis_sha256 for item in self.question_analyses]
        ):
            raise ValueError("metasyn_grounded_analysis_v2_question_membership_mismatch")
        bridge_corpora = {item.question_id: item for item in self.bridge.question_corpora}
        for item in self.question_analyses:
            corpus = bridge_corpora.get(item.question_id)
            if (
                corpus is None
                or item.question_corpus_sha256 != corpus.question_corpus_sha256
                or item.compatibility_corpus_sha256 != corpus.compatibility_corpus_sha256
                or item.publication_join_sha256s != corpus.publication_join_sha256s
                or item.publication_ids != corpus.publication_ids
            ):
                raise ValueError("metasyn_grounded_analysis_v2_question_corpus_alias_mismatch")
        bridge_counts = (
            self.bridge.inventoried_candidate_count,
            self.bridge.authorized_candidate_count,
            self.bridge.terminal_candidate_count,
            self.bridge.completed_candidate_count,
            self.bridge.abstained_candidate_count,
            self.bridge.estimable_publication_count,
            self.bridge.quantitative_effect_count,
        )
        if (
            self.inventoried_candidate_count,
            self.authorized_candidate_count,
            self.terminal_candidate_count,
            self.completed_candidate_count,
            self.abstained_candidate_count,
            self.estimable_publication_count,
            self.quantitative_effect_count,
        ) != bridge_counts:
            raise ValueError("metasyn_grounded_analysis_v2_terminal_count_alias_mismatch")
        units = [unit for question in self.question_analyses for unit in question.units]
        invoked = sum(unit.kernel_invoked for unit in units)
        completed = sum(unit.status == "quantitative_kernel_completed" for unit in units)
        unresolved_questions = sum(
            not item.cohort_independence_resolved and item.quantitative_effect_count > 0
            for item in self.question_analyses
        )
        if (
            self.quantitative_unit_count,
            self.kernel_invoked_unit_count,
            self.kernel_completed_unit_count,
            self.kernel_abstained_unit_count,
            self.questions_with_completed_kernel_mechanics,
            self.questions_with_unresolved_independence,
        ) != (
            len(units),
            invoked,
            completed,
            len(units) - completed,
            sum(item.kernel_completed_unit_count > 0 for item in self.question_analyses),
            unresolved_questions,
        ):
            raise ValueError("metasyn_grounded_analysis_v2_mechanics_count_mismatch")
        if sum(item.publication_count for item in self.question_analyses) != 32:
            raise ValueError("metasyn_grounded_analysis_v2_publication_coverage_mismatch")
        if self.quantitative_kernel_compatibility != self.bridge.quantitative_kernel_compatibility:
            raise ValueError("metasyn_grounded_analysis_v2_compatibility_alias_mismatch")
        expected_reviewer_membership = hash_canonical(
            {
                item.question_id: item.reviewer_artifact_sha256
                for item in self.question_analyses
                if item.reviewer_artifact_sha256 is not None
            }
        )
        if self.reviewer_artifact_membership_sha256 != expected_reviewer_membership:
            raise ValueError("metasyn_grounded_analysis_v2_reviewer_membership_mismatch")
        _self_hash(self, "analysis_sha256")
        return self


def _pipeline_settings(
    *,
    bridge: MetaSynGroundedPublicationCorpusBridgeV2,
    reviewer_artifact_membership_sha256: str,
) -> dict[str, Any]:
    _validate_sha256(
        reviewer_artifact_membership_sha256,
        "reviewer_artifact_membership_sha256",
    )
    return {
        "accuracy_claim_authority": False,
        "bridge_pipeline_sha256": bridge.bridge_pipeline_sha256,
        "bridge_sha256": bridge.bridge_sha256,
        "claim_release_authority": False,
        "execution_bundle_sha256": bridge.execution_bundle_sha256,
        "installed_dependency_versions": {
            name: distribution_version(name) for name in _INSTALLED_DEPENDENCIES
        },
        "inventory_receipt_membership_sha256": (bridge.inventory_receipt_membership_sha256),
        "mechanics_policy": dict(_MECHANICS_POLICY),
        "official_test_labels_opened": False,
        "platform_python_version": platform.python_version(),
        "publication_join_membership_sha256": (bridge.publication_join_membership_sha256),
        "question_corpus_membership_sha256": (bridge.question_corpus_membership_sha256),
        "reviewer_artifact_membership_sha256": (reviewer_artifact_membership_sha256),
        "scientific_synthesis_authority": False,
        "source_surface_sha256": bridge.source_surface_sha256,
        "synthesis_input_authority": False,
        "terminal_membership_sha256": bridge.terminal_membership_sha256,
        "yield_and_mechanics_only": True,
    }


def compute_metasyn_grounded_analysis_v2_pipeline_fingerprint(
    *,
    repository_root: Path,
    bridge: MetaSynGroundedPublicationCorpusBridgeV2 | Mapping[str, Any],
    reviewer_artifact_membership_sha256: str,
) -> PipelineFingerprint:
    """Compute an AST-closed identity for reconciliation and kernel mechanics."""

    root = _canonical_root(repository_root)
    canonical = MetaSynGroundedPublicationCorpusBridgeV2.model_validate(
        bridge.model_dump(mode="json")
        if isinstance(bridge, MetaSynGroundedPublicationCorpusBridgeV2)
        else bridge
    )
    files = sorted(set(_python_dependency_closure(root)) | set(_NON_PYTHON_INPUTS))
    return compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="metasyn-grounded-analysis-v2",
                component_version=ANALYSIS_COMPONENT_VERSION,
                file_paths=files,
                settings=_pipeline_settings(
                    bridge=canonical,
                    reviewer_artifact_membership_sha256=(reviewer_artifact_membership_sha256),
                ),
            )
        ],
    )


def _unit_signature(
    *, graph: EvidenceGraph, estimate: OutcomeEstimateNode
) -> MetaSynQuantitativeUnitSignatureV2:
    contrasts = {item.contrast_id: item for item in graph.contrasts}
    arms = {item.arm_id: item for item in graph.arms}
    contrast = contrasts[estimate.contrast_id]
    payload = {
        "signature_version": UNIT_SIGNATURE_VERSION,
        "outcome_name": estimate.outcome_name,
        "contrast_label": contrast.label,
        "contrast_estimand": contrast.estimand,
        "positive_direction_means": contrast.positive_direction_means,
        "treatment_arm_role": arms[contrast.treatment_arm_id].role,
        "comparator_arm_role": arms[contrast.comparator_arm_id].role,
        "timepoint": estimate.timepoint,
        "analysis_population": estimate.analysis_population,
    }
    return MetaSynQuantitativeUnitSignatureV2.model_validate(
        {**payload, "signature_sha256": hash_canonical(payload)}
    )


def _subgraph_for_estimates(
    graph: EvidenceGraph, estimates: Sequence[OutcomeEstimateNode]
) -> EvidenceGraph:
    payload = graph.model_dump(mode="json")
    payload["outcome_estimates"] = [item.model_dump(mode="json") for item in estimates]
    return EvidenceGraph.model_validate(payload)


def _kernel_abstention_reasons(result: Mapping[str, Any]) -> list[str]:
    reasons: set[str] = set()
    quantitative = result.get("quantitative")
    if isinstance(quantitative, Mapping) and quantitative.get("reason"):
        reasons.add(f"quantitative_kernel:{quantitative['reason']}")
    if result.get("status") != "ok" and result.get("reason"):
        reasons.add(f"evidence_graph_contract:{result['reason']}")
    if result.get("mode") == "directional_sign_synthesis":
        reasons.add("directional_fallback_non_authorizing")
    elif result.get("mode") != "random_effects_meta_analysis":
        reasons.add(f"quantitative_kernel_mode_not_completed:{result.get('mode')}")
    if not reasons:
        reasons.add("quantitative_kernel_not_completed")
    return sorted(reasons)


def _freeze_kernel_units(
    *,
    question: MetaSynGroundedQuestionCorpusV2,
    reconciliation: NativeCohortReconciliationReceipt,
) -> list[MetaSynQuantitativeKernelUnitV2]:
    graph = reconciliation.reconciled_graph
    graph_sha256 = reconciliation.reconciled_graph_sha256
    if graph is None or graph_sha256 is None:
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_reconciled_graph_missing"
        )
    groups: dict[str, tuple[MetaSynQuantitativeUnitSignatureV2, list[OutcomeEstimateNode]]] = {}
    for estimate in graph.outcome_estimates:
        signature = _unit_signature(graph=graph, estimate=estimate)
        existing = groups.get(signature.signature_sha256)
        if existing is None:
            groups[signature.signature_sha256] = (signature, [estimate])
        else:
            if existing[0] != signature:
                raise MetaSynGroundedAnalysisV2Error(
                    "metasyn_grounded_analysis_v2_unit_signature_hash_collision"
                )
            existing[1].append(estimate)

    contrast_index = {item.contrast_id: item for item in graph.contrasts}
    units: list[MetaSynQuantitativeKernelUnitV2] = []
    for signature_sha256 in sorted(groups):
        signature, estimates = groups[signature_sha256]
        estimates = sorted(estimates, key=lambda item: item.estimate_id)
        contrast_ids = sorted({item.contrast_id for item in estimates})
        cohort_ids = sorted({contrast_index[item.contrast_id].cohort_id for item in estimates})
        paper_ids = sorted({item.effect.paper_id for item in estimates})
        independence_resolved = reconciliation.cross_publication_identity_assurance_complete
        if independence_resolved:
            selected_graph = _subgraph_for_estimates(graph, estimates)
            kernel_result = synthesize_evidence_graph(
                selected_graph,
                outcome_name=signature.outcome_name,
                require_explicit_timepoint=True,
                confidence_level=0.95,
                assumed_within_cohort_correlation=1.0,
                prespecified_moderators=(),
            )
            if kernel_result.get("condition_analysis") is not None:
                raise MetaSynGroundedAnalysisV2Error(
                    "metasyn_grounded_analysis_v2_unrequested_condition_analysis"
                )
            quantitative = kernel_result.get("quantitative")
            completed = (
                kernel_result.get("status") == "ok"
                and kernel_result.get("mode") == "random_effects_meta_analysis"
                and isinstance(quantitative, Mapping)
                and quantitative.get("status") == "ok"
            )
            status = "quantitative_kernel_completed" if completed else "abstained"
            reasons = [] if completed else _kernel_abstention_reasons(kernel_result)
            kernel_hash = hash_canonical(kernel_result)
        else:
            kernel_result = None
            kernel_hash = None
            status = "abstained"
            reasons = [
                f"cohort_independence_not_resolved:{reconciliation.status.value}",
                *(["reviewer_artifact_absent"] if reconciliation.reviewer_artifact is None else []),
                *(
                    f"cohort_reconciliation_issue:{issue.code}"
                    for issue in reconciliation.issues
                    if not issue.resolved_by_reviewer
                ),
            ]
            reasons = sorted(set(reasons))
        payload = {
            "unit_version": UNIT_VERSION,
            "question_id": question.question_id,
            "question_corpus_sha256": question.question_corpus_sha256,
            "reconciliation_receipt_sha256": reconciliation.receipt_sha256,
            "reconciled_graph_sha256": graph_sha256,
            "signature": signature,
            "signature_sha256": signature.signature_sha256,
            "estimate_ids": [item.estimate_id for item in estimates],
            "contrast_ids": contrast_ids,
            "cohort_ids": cohort_ids,
            "paper_ids": paper_ids,
            "input_effect_count": len(estimates),
            "input_cohort_count": len(cohort_ids),
            "input_publication_count": len(paper_ids),
            "cohort_independence_resolved": independence_resolved,
            "kernel_invoked": kernel_result is not None,
            "status": status,
            "abstention_reasons": reasons,
            "kernel_result": kernel_result,
            "kernel_result_sha256": kernel_hash,
            "require_explicit_timepoint": True,
            "confidence_level": 0.95,
            "assumed_within_cohort_correlation": 1.0,
            "prespecified_moderators": [],
            "moderator_inference_performed": False,
            "reported_significance_consumed": False,
            "equivalence_consumed": False,
            "legacy_optional_sentinels_interpreted": False,
            "graph_construction_authority": True,
            "quantitative_kernel_mechanics_authority": True,
            "synthesis_input_authority": False,
            "scientific_synthesis_authority": False,
            "accuracy_claim_authority": False,
            "claim_release_authority": False,
        }
        units.append(
            MetaSynQuantitativeKernelUnitV2.model_validate(
                {**payload, "unit_sha256": hash_canonical(payload)}
            )
        )
    return units


def _question_mechanics_status(
    *, unit_count: int, completed_count: int, independence_resolved: bool
) -> str:
    if unit_count == 0:
        return "no_quantitative_units"
    if not independence_resolved:
        return "cohort_independence_unresolved"
    if completed_count == unit_count:
        return "kernel_completed"
    if completed_count == 0:
        return "kernel_abstained"
    return "mixed"


def freeze_metasyn_grounded_question_analysis_v2(
    *,
    question_corpus: MetaSynGroundedQuestionCorpusV2 | Mapping[str, Any],
    reviewer_artifact: ReviewerCohortReconciliationArtifact | Mapping[str, Any] | None = None,
) -> MetaSynGroundedQuestionAnalysisV2:
    """Freeze one question's reconciliation and kernel-mechanics ledger."""

    question = MetaSynGroundedQuestionCorpusV2.model_validate(
        question_corpus.model_dump(mode="json")
        if isinstance(question_corpus, MetaSynGroundedQuestionCorpusV2)
        else question_corpus
    )
    reviewer = (
        None
        if reviewer_artifact is None
        else ReviewerCohortReconciliationArtifact.model_validate(
            reviewer_artifact.model_dump(mode="json")
            if isinstance(reviewer_artifact, ReviewerCohortReconciliationArtifact)
            else reviewer_artifact
        )
    )
    reconciliation = reconcile_native_cohorts(
        corpus=question.compatibility_corpus,
        reviewer_artifact=reviewer,
    )
    reconciliation = reverify_native_cohort_reconciliation(
        corpus=question.compatibility_corpus,
        receipt=reconciliation,
    )
    units = _freeze_kernel_units(question=question, reconciliation=reconciliation)
    completed = sum(item.status == "quantitative_kernel_completed" for item in units)
    invoked = sum(item.kernel_invoked for item in units)
    mechanics_status = _question_mechanics_status(
        unit_count=len(units),
        completed_count=completed,
        independence_resolved=(reconciliation.cross_publication_identity_assurance_complete),
    )
    mechanics_blockers = {blocker for unit in units for blocker in unit.abstention_reasons}
    if not units:
        mechanics_blockers.add("no_quantitative_effects")
    scientific_blockers = {
        "bridge_synthesis_input_authority_false",
        "claim_release_authority_false",
        "optional_scientific_fields_not_authorized",
        *question.coverage_blockers,
    }
    payload = {
        "question_analysis_version": QUESTION_ANALYSIS_VERSION,
        "question_id": question.question_id,
        "question_corpus_sha256": question.question_corpus_sha256,
        "compatibility_corpus_sha256": question.compatibility_corpus_sha256,
        "publication_join_sha256s": question.publication_join_sha256s,
        "publication_ids": question.publication_ids,
        "publication_count": len(question.publication_ids),
        "quantitative_effect_count": question.quantitative_effect_count,
        "bridge_coverage_blockers": question.coverage_blockers,
        "reviewer_artifact_sha256": (reviewer.artifact_sha256 if reviewer is not None else None),
        "reconciliation": reconciliation,
        "reconciliation_receipt_sha256": reconciliation.receipt_sha256,
        "reconciliation_status": reconciliation.status,
        "reconciled_graph_sha256": reconciliation.reconciled_graph_sha256,
        "cohort_independence_resolved": (
            reconciliation.cross_publication_identity_assurance_complete
        ),
        "units": units,
        "unit_membership_sha256": hash_canonical([item.unit_sha256 for item in units]),
        "unit_count": len(units),
        "kernel_invoked_unit_count": invoked,
        "kernel_completed_unit_count": completed,
        "kernel_abstained_unit_count": len(units) - completed,
        "mechanics_status": mechanics_status,
        "mechanics_blockers": sorted(mechanics_blockers),
        "scientific_authority_blockers": sorted(scientific_blockers),
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": bool(question.quantitative_effect_count),
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "accuracy_claim_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedQuestionAnalysisV2.model_validate(
        {
            **payload,
            "question_analysis_sha256": hash_canonical(payload),
        }
    )


def _canonical_reviewer_map(
    *,
    question_ids: set[str],
    values: Mapping[str, ReviewerCohortReconciliationArtifact | Mapping[str, Any]] | None,
) -> dict[str, ReviewerCohortReconciliationArtifact]:
    if values is None:
        return {}
    if set(values) - question_ids:
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_reviewer_question_unknown"
        )
    try:
        return {
            question_id: ReviewerCohortReconciliationArtifact.model_validate(
                value.model_dump(mode="json")
                if isinstance(value, ReviewerCohortReconciliationArtifact)
                else value
            )
            for question_id, value in sorted(values.items())
        }
    except ValueError as exc:
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_reviewer_artifact_invalid"
        ) from exc


def freeze_metasyn_grounded_analysis_v2(
    *,
    bridge: MetaSynGroundedPublicationCorpusBridgeV2 | Mapping[str, Any],
    repository_root: Path,
    reviewer_artifacts_by_question: Mapping[
        str, ReviewerCohortReconciliationArtifact | Mapping[str, Any]
    ]
    | None = None,
) -> MetaSynGroundedAnalysisV2:
    """Externally replay a complete bridge and freeze mechanics-only analysis."""

    root = _canonical_root(repository_root)
    canonical_bridge = validate_metasyn_grounded_publication_bridge_v2(
        bridge=bridge,
        repository_root=root,
        external_replay=True,
    )
    question_ids = {item.question_id for item in canonical_bridge.question_corpora}
    reviewers = _canonical_reviewer_map(
        question_ids=question_ids,
        values=reviewer_artifacts_by_question,
    )
    reviewer_membership = hash_canonical(
        {question_id: artifact.artifact_sha256 for question_id, artifact in reviewers.items()}
    )
    fingerprint = compute_metasyn_grounded_analysis_v2_pipeline_fingerprint(
        repository_root=root,
        bridge=canonical_bridge,
        reviewer_artifact_membership_sha256=reviewer_membership,
    )
    questions = [
        freeze_metasyn_grounded_question_analysis_v2(
            question_corpus=question,
            reviewer_artifact=reviewers.get(question.question_id),
        )
        for question in canonical_bridge.question_corpora
    ]
    questions.sort(key=lambda item: item.question_id)
    units = [unit for question in questions for unit in question.units]
    completed = sum(item.status == "quantitative_kernel_completed" for item in units)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "externally_replayable_label_blind_mechanics_only",
        "bridge": canonical_bridge,
        "bridge_sha256": canonical_bridge.bridge_sha256,
        "execution_bundle_sha256": canonical_bridge.execution_bundle_sha256,
        "source_surface_sha256": canonical_bridge.source_surface_sha256,
        "inventory_receipt_membership_sha256": (
            canonical_bridge.inventory_receipt_membership_sha256
        ),
        "terminal_membership_sha256": canonical_bridge.terminal_membership_sha256,
        "bridge_pipeline_sha256": canonical_bridge.bridge_pipeline_sha256,
        "publication_join_membership_sha256": (canonical_bridge.publication_join_membership_sha256),
        "question_corpus_membership_sha256": (canonical_bridge.question_corpus_membership_sha256),
        "reviewer_artifact_membership_sha256": reviewer_membership,
        "analysis_pipeline_fingerprint": fingerprint,
        "analysis_pipeline_sha256": fingerprint.pipeline_sha256,
        "question_analyses": questions,
        "question_analysis_membership_sha256": hash_canonical(
            [item.question_analysis_sha256 for item in questions]
        ),
        "question_count": EXPECTED_QUESTION_COUNT,
        "publication_count": EXPECTED_PUBLICATION_COUNT,
        "inventoried_candidate_count": canonical_bridge.inventoried_candidate_count,
        "authorized_candidate_count": canonical_bridge.authorized_candidate_count,
        "terminal_candidate_count": canonical_bridge.terminal_candidate_count,
        "completed_candidate_count": canonical_bridge.completed_candidate_count,
        "abstained_candidate_count": canonical_bridge.abstained_candidate_count,
        "estimable_publication_count": canonical_bridge.estimable_publication_count,
        "quantitative_effect_count": canonical_bridge.quantitative_effect_count,
        "quantitative_unit_count": len(units),
        "kernel_invoked_unit_count": sum(item.kernel_invoked for item in units),
        "kernel_completed_unit_count": completed,
        "kernel_abstained_unit_count": len(units) - completed,
        "questions_with_completed_kernel_mechanics": sum(
            item.kernel_completed_unit_count > 0 for item in questions
        ),
        "questions_with_unresolved_independence": sum(
            not item.cohort_independence_resolved and item.quantitative_effect_count > 0
            for item in questions
        ),
        "bridge_external_replay_validated": True,
        "complete_terminal_coverage_preserved": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "benchmark_conclusions_accessed": False,
        "moderator_inference_performed": False,
        "reported_significance_consumed": False,
        "equivalence_consumed": False,
        "directional_fallback_completed_quantitative_mechanics": False,
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": (canonical_bridge.quantitative_kernel_compatibility),
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "accuracy_claim_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedAnalysisV2.model_validate(
        {**payload, "analysis_sha256": hash_canonical(payload)}
    )


def validate_metasyn_grounded_analysis_v2(
    *,
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> MetaSynGroundedAnalysisV2:
    """Validate an analysis artifact and, by default, replay the whole bridge."""

    try:
        canonical = MetaSynGroundedAnalysisV2.model_validate(
            analysis.model_dump(mode="json")
            if isinstance(analysis, MetaSynGroundedAnalysisV2)
            else analysis
        )
    except ValueError as exc:
        raise MetaSynGroundedAnalysisV2Error(
            "metasyn_grounded_analysis_v2_contract_invalid"
        ) from exc
    if external_replay:
        reviewers = {
            item.question_id: item.reconciliation.reviewer_artifact
            for item in canonical.question_analyses
            if item.reconciliation.reviewer_artifact is not None
        }
        replayed = freeze_metasyn_grounded_analysis_v2(
            bridge=canonical.bridge,
            repository_root=repository_root,
            reviewer_artifacts_by_question=reviewers,
        )
        if replayed != canonical:
            raise MetaSynGroundedAnalysisV2Error(
                "metasyn_grounded_analysis_v2_external_replay_mismatch"
            )
    return canonical


__all__ = [
    "ANALYSIS_CLI_PATH",
    "ANALYSIS_MODULE_PATH",
    "ANALYSIS_VERSION",
    "MetaSynGroundedAnalysisV2",
    "MetaSynGroundedAnalysisV2Error",
    "MetaSynGroundedQuestionAnalysisV2",
    "MetaSynQuantitativeKernelUnitV2",
    "MetaSynQuantitativeUnitSignatureV2",
    "compute_metasyn_grounded_analysis_v2_pipeline_fingerprint",
    "freeze_metasyn_grounded_analysis_v2",
    "freeze_metasyn_grounded_question_analysis_v2",
    "validate_metasyn_grounded_analysis_v2",
]
