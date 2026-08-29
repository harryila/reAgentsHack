"""Provider-neutral hosted MetaSyn graph and synthesis-yield evaluation.

This evaluator is deliberately separate from the frozen local-runtime v1 diagnostic.
It consumes the externally replayed hosted runtime, projects every hosted terminal row
through a self-hashed provider-neutral row view, replays eligible extractions against
the original source artifacts, and then reuses the immutable v1 graph, cohort, effect
compatibility, and synthesis mechanics.  It never opens review conclusions, effect
directions, aggregate reference effects, or official test labels.

The private report contains the complete identifier-bearing 10-question/32-publication
terminal accounting.  Its public companion contains aggregate counts and cryptographic
lineage only.  Neither artifact has accuracy, truth, calibration, or release authority.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynPublicationResultV1,
    validate_metasyn_bounded_adapter_bundle_external_replay,
    validate_metasyn_bounded_private_yield_report,
    validate_metasyn_publication_result,
)
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    MetaSynHostedPrivateYieldReportV1,
    load_current_metasyn_hosted_execution_bundle,
)
from literature_multiverse.metasyn_synthesis_yield import (
    _PUBLIC_CAVEATS,
    MetaSynCompatibilityGroupYieldV1,
    MetaSynQuestionSynthesisYieldV1,
    MetaSynSynthesisYieldError,
    MetaSynSynthesisYieldPublicSummaryV1,
    MetaSynSynthesisYieldReportV1,
    _aggregate_blocker_counts,
    _aggregate_residual_conflict_counts,
    _freeze_question_report,
    _question_blockers,
    _residual_conflicts,
    _subgraph,
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
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

EVALUATION_VERSION = "metasyn-hosted-synthesis-yield-v2"
PUBLIC_SUMMARY_VERSION = "metasyn-hosted-synthesis-yield-public-v2"
ROW_VIEW_VERSION = "metasyn-hosted-synthesis-row-view-v2"
EVALUATION_COMPONENT_VERSION = "3"
SYNTHESIS_UNIT_AUTHORIZATION_RECEIPT_VERSION = (
    "metasyn-synthesis-unit-authorization-receipt-v2"
)

_MODULE_ENTRYPOINT = "src/literature_multiverse/metasyn_synthesis_yield_v2.py"
_DEPENDENCY_ENTRYPOINTS = (_MODULE_ENTRYPOINT,)
_NON_PYTHON_INPUTS = ("pyproject.toml", "uv.lock")

HostedRowStatus = Literal[
    "typed_publication_output",
    "adapter_inventory_no_candidate",
    "adapter_inventory_uncertain",
    "adapter_packet_unable",
    "runtime_inventory_blocked",
    "runtime_packet_blocked",
]


class MetaSynSynthesisYieldV2Error(MetaSynSynthesisYieldError):
    """A hosted lineage, exact-join, source replay, or aggregate invariant failed."""


def _canonical_hosted_report(
    value: MetaSynHostedPrivateYieldReportV1 | Mapping[str, Any],
) -> MetaSynHostedPrivateYieldReportV1:
    """Fresh-validate mappings and instances; never trust model construction state."""

    payload: Mapping[str, Any]
    if isinstance(value, MetaSynHostedPrivateYieldReportV1):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    return MetaSynHostedPrivateYieldReportV1.model_validate(payload)


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
            raise MetaSynSynthesisYieldV2Error(
                f"metasyn_synthesis_v2_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynSynthesisYieldV2Error(
                f"metasyn_synthesis_v2_dependency_unreadable:{relative}"
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


class MetaSynHostedSynthesisRowViewV2(ContractModel):
    """Self-hashed scientific view of one hosted terminal row.

    The v1 publication freezer only needs this provider-neutral surface.  Transport,
    token, and price fields remain in the externally validated hosted report and cannot
    affect scientific graph construction.
    """

    row_view_version: Literal["metasyn-hosted-synthesis-row-view-v2"] = (
        ROW_VIEW_VERSION
    )
    hosted_runtime_private_report_sha256: str
    hosted_row_result_sha256: str
    provider_neutral_yield_report_sha256: str | None
    row_context_sha256: str
    question_spec_sha256: str
    question_bundle_sha256: str
    source_row_sha256: str
    independence_component_membership_sha256: str
    status: HostedRowStatus
    runtime_blockers: list[str]
    adapter_publication_result: MetaSynPublicationResultV1 | None
    adapter_publication_result_sha256: str | None
    row_result_sha256: str

    @field_validator(
        "hosted_runtime_private_report_sha256",
        "hosted_row_result_sha256",
        "provider_neutral_yield_report_sha256",
        "row_context_sha256",
        "question_spec_sha256",
        "question_bundle_sha256",
        "source_row_sha256",
        "independence_component_membership_sha256",
        "adapter_publication_result_sha256",
        "row_result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"metasyn_synthesis_v2_row_hash_invalid:{info.field_name}"
            )
        return value

    @field_validator("runtime_blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_synthesis_v2_row_blockers_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_row_view(self) -> MetaSynHostedSynthesisRowViewV2:
        result = self.adapter_publication_result
        expected_result_sha = result.result_sha256 if result is not None else None
        if self.adapter_publication_result_sha256 != expected_result_sha:
            raise ValueError("metasyn_synthesis_v2_row_result_hash_alias_mismatch")
        if result is not None and result.row_context_sha256 != self.row_context_sha256:
            raise ValueError("metasyn_synthesis_v2_row_result_context_mismatch")
        typed = self.status == "typed_publication_output"
        if typed != bool(
            result is not None and result.status == "typed_publication_output"
        ):
            raise ValueError("metasyn_synthesis_v2_row_typed_status_mismatch")
        if typed == bool(self.runtime_blockers):
            raise ValueError("metasyn_synthesis_v2_row_blocker_presence_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_result_sha256"})
        if hash_canonical(payload) != self.row_result_sha256:
            raise ValueError("metasyn_synthesis_v2_row_view_hash_mismatch")
        return self


class MetaSynSynthesisUnitAuthorizationReceiptV2(ContractModel):
    """Private proof that a compatibility group is not yet safe to synthesize.

    The current hosted schema grounds numeric findings but does not independently
    ground the analytical grain: study/cohort identity, arm identity, protocol
    contrast binding, or pairwise independence.  V2 therefore records the complete
    hash join and fails closed.  A later receipt version may authorize synthesis only
    after carrying source- or reviewer-authenticated structural support.
    """

    receipt_version: Literal[
        "metasyn-synthesis-unit-authorization-receipt-v2"
    ] = SYNTHESIS_UNIT_AUTHORIZATION_RECEIPT_VERSION
    authorization_scope: Literal[
        "structural_identity_and_independence_not_yet_source_authorized"
    ] = "structural_identity_and_independence_not_yet_source_authorized"
    question_spec_sha256: str
    question_bundle_sha256: str
    compatibility_sha256: str
    group_graph_sha256: str
    estimate_ids: Annotated[list[str], Field(min_length=1)]
    publication_record_sha256s: Annotated[list[str], Field(min_length=1)]
    runtime_row_result_sha256s: Annotated[list[str], Field(min_length=1)]
    adapter_publication_result_sha256s: Annotated[list[str], Field(min_length=1)]
    source_row_sha256s: Annotated[list[str], Field(min_length=1)]
    source_document_sha256s: Annotated[list[str], Field(min_length=1)]
    original_source_grounding_receipt_sha256s: Annotated[
        list[str], Field(min_length=1)
    ]
    cohort_reconciliation_sha256: str
    canonical_unit_membership_sha256: str
    structural_claim_support_receipt_sha256s: Annotated[
        list[str], Field(max_length=0)
    ] = Field(default_factory=list)
    pairwise_independence_support_receipt_sha256s: Annotated[
        list[str], Field(max_length=0)
    ] = Field(default_factory=list)
    structural_source_authorization_complete: Literal[False] = False
    protocol_contrast_binding_complete: Literal[False] = False
    reconciliation_coverage_complete: Literal[True] = True
    pairwise_independence_complete: Literal[False] = False
    authorizes_synthesis_input: Literal[False] = False
    issues: Annotated[list[str], Field(min_length=1)]
    receipt_sha256: str

    @field_validator(
        "question_spec_sha256",
        "question_bundle_sha256",
        "compatibility_sha256",
        "group_graph_sha256",
        "cohort_reconciliation_sha256",
        "canonical_unit_membership_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_receipt_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"metasyn_synthesis_v2_authorization_hash_invalid:{info.field_name}"
            )
        return value

    @field_validator(
        "estimate_ids",
        "publication_record_sha256s",
        "runtime_row_result_sha256s",
        "adapter_publication_result_sha256s",
        "source_row_sha256s",
        "source_document_sha256s",
        "original_source_grounding_receipt_sha256s",
        "structural_claim_support_receipt_sha256s",
        "pairwise_independence_support_receipt_sha256s",
        "issues",
    )
    @classmethod
    def validate_sorted_unique_receipt_values(
        cls, value: list[str], info: Any
    ) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError(
                "metasyn_synthesis_v2_authorization_values_invalid:"
                f"{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_authorization_receipt(
        self,
    ) -> MetaSynSynthesisUnitAuthorizationReceiptV2:
        required_issues = {
            "pairwise_synthesis_unit_independence_not_source_authorized",
            "protocol_contrast_not_source_authorized",
            "structural_claims_not_source_authorized",
        }
        if not required_issues.issubset(self.issues):
            raise ValueError(
                "metasyn_synthesis_v2_authorization_required_issues_missing"
            )
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError(
                "metasyn_synthesis_v2_authorization_receipt_hash_mismatch"
            )
        return self


def _freeze_row_views(
    *,
    hosted: MetaSynHostedPrivateYieldReportV1,
    adapter: MetaSynBoundedAdapterBundleV1,
) -> list[MetaSynHostedSynthesisRowViewV2]:
    hosted_rows = {row.row_context_sha256: row for row in hosted.row_results}
    adapter_rows = {row.row_context_sha256: row for row in adapter.row_contexts}
    if (
        len(hosted_rows) != EXPECTED_SELECTED_PAPERS
        or set(hosted_rows) != set(adapter_rows)
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_hosted_adapter_row_roster_mismatch"
        )
    output: list[MetaSynHostedSynthesisRowViewV2] = []
    for row_context_sha256 in sorted(adapter_rows):
        adapter_row = adapter_rows[row_context_sha256]
        hosted_row = hosted_rows[row_context_sha256]
        if (
            hosted_row.question_spec_sha256 != adapter_row.question_spec_sha256
            or hosted_row.question_bundle_sha256
            != adapter_row.question_bundle_sha256
            or hosted_row.source_row_sha256 != adapter_row.source_row_sha256
            or hosted_row.release_grade_source_grounding_eligible
            != adapter_row.source_row.release_grade_source_grounding_eligible
        ):
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_hosted_adapter_row_lineage_mismatch"
            )
        if hosted_row.adapter_publication_result is not None:
            replayed_result = validate_metasyn_publication_result(
                result=hosted_row.adapter_publication_result,
                row=adapter_row,
            )
            if replayed_result != hosted_row.adapter_publication_result:
                raise MetaSynSynthesisYieldV2Error(
                    "metasyn_synthesis_v2_hosted_publication_replay_mismatch"
                )
        payload: dict[str, Any] = {
            "row_view_version": ROW_VIEW_VERSION,
            "hosted_runtime_private_report_sha256": hosted.report_sha256,
            "hosted_row_result_sha256": hosted_row.row_result_sha256,
            "provider_neutral_yield_report_sha256": (
                hosted.provider_neutral_yield_report_sha256
            ),
            "row_context_sha256": adapter_row.row_context_sha256,
            "question_spec_sha256": adapter_row.question_spec_sha256,
            "question_bundle_sha256": adapter_row.question_bundle_sha256,
            "source_row_sha256": adapter_row.source_row_sha256,
            "independence_component_membership_sha256": (
                adapter_row.independence_component_membership_sha256
            ),
            "status": hosted_row.status,
            "runtime_blockers": sorted(hosted_row.blockers),
            "adapter_publication_result": hosted_row.adapter_publication_result,
            "adapter_publication_result_sha256": (
                hosted_row.adapter_publication_result_sha256
            ),
        }
        output.append(
            MetaSynHostedSynthesisRowViewV2.model_validate(
                {**payload, "row_result_sha256": hash_canonical(payload)}
            )
        )
    return output


def _validate_prepare_adapter_join(
    *,
    hosted: MetaSynHostedPrivateYieldReportV1,
    adapter: MetaSynBoundedAdapterBundleV1,
    prepared: MetaSynTypedPilotPrepareBundleV1,
) -> dict[str, MetaSynPilotQuestionBundleV1]:
    if hosted.adapter_bundle_sha256 != adapter.adapter_bundle_sha256:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_hosted_adapter_mismatch"
        )
    if hosted.row_membership_sha256 != adapter.row_membership_sha256:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_hosted_adapter_membership_mismatch"
        )
    if adapter.prepare_bundle_sha256 != prepared.prepare_bundle_sha256:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_adapter_prepare_mismatch"
        )
    if adapter.upstream_pilot_pipeline_sha256 != prepared.pilot_pipeline_sha256:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_adapter_prepare_pipeline_mismatch"
        )
    if adapter.official_native_schema_sha256 != (
        prepared.official_native_extraction_schema_sha256
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_adapter_prepare_schema_mismatch"
        )
    if hosted.downstream_verifier_pipeline_sha256 != (
        prepared.downstream_verifier_pipeline_sha256
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_downstream_verifier_mismatch"
        )
    if (
        adapter.question_count != EXPECTED_SELECTED_QUESTIONS
        or adapter.component_count != EXPECTED_SELECTED_COMPONENTS
        or adapter.publication_count != EXPECTED_SELECTED_PAPERS
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_adapter_cardinality_mismatch"
        )
    questions = {item.question_spec.question_id: item for item in prepared.questions}
    if len(questions) != EXPECTED_SELECTED_QUESTIONS:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_prepare_question_roster_mismatch"
        )
    prepared_sources = {
        row.source_row_sha256: (question, row)
        for question in prepared.questions
        for row in question.source_rows
    }
    if len(prepared_sources) != EXPECTED_SELECTED_PAPERS:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_prepare_source_roster_mismatch"
        )
    for adapter_row in adapter.row_contexts:
        prepared_pair = prepared_sources.get(adapter_row.source_row_sha256)
        if prepared_pair is None:
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_adapter_source_not_in_prepare"
            )
        question, source_row = prepared_pair
        if source_row != adapter_row.source_row:
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_adapter_prepare_source_snapshot_mismatch"
            )
        if (
            adapter_row.question_bundle_sha256 != question.question_bundle_sha256
            or adapter_row.question_spec != question.question_spec
            or adapter_row.independence_component_id
            != question.independence_component_id
            or adapter_row.independence_component_membership_sha256
            != question.independence_component_membership_sha256
        ):
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_adapter_prepare_question_lineage_mismatch"
            )
    return questions


def _validate_provider_neutral_report(
    *,
    hosted: MetaSynHostedPrivateYieldReportV1,
    adapter: MetaSynBoundedAdapterBundleV1,
) -> MetaSynBoundedPrivateYieldReportV1 | None:
    report = hosted.provider_neutral_yield_report
    report_sha256 = hosted.provider_neutral_yield_report_sha256
    all_rows_have_adapter_results = all(
        row.adapter_publication_result is not None for row in hosted.row_results
    )
    if (report is None) != (report_sha256 is None):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_provider_report_presence_mismatch"
        )
    if (report is not None) != all_rows_have_adapter_results:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_provider_report_completeness_mismatch"
        )
    if report is None:
        return None
    canonical = validate_metasyn_bounded_private_yield_report(
        report=report, adapter_bundle=adapter
    )
    if canonical.report_sha256 != report_sha256:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_provider_report_hash_mismatch"
        )
    hosted_results = {
        row.row_context_sha256: row.adapter_publication_result
        for row in hosted.row_results
    }
    provider_results = {
        result.row_context_sha256: result
        for result in canonical.publication_results
    }
    if set(hosted_results) != set(provider_results) or any(
        hosted_results[key] != provider_results[key] for key in provider_results
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_provider_hosted_result_roster_mismatch"
        )
    return canonical


def _freeze_non_authorizing_synthesis_unit_receipt(
    *,
    question: MetaSynQuestionSynthesisYieldV1,
    group: MetaSynCompatibilityGroupYieldV1,
) -> MetaSynSynthesisUnitAuthorizationReceiptV2:
    graph = question.cohort_reconciliation.reconciled_graph
    if graph is None:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_authorization_graph_missing"
        )
    estimates = {item.estimate_id: item for item in graph.outcome_estimates}
    contrasts = {item.contrast_id: item for item in graph.contrasts}
    cohorts = {item.cohort_id: item for item in graph.cohorts}
    records = {item.paper_id: item for item in question.publication_records}
    if set(group.estimate_ids) - set(estimates):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_authorization_estimate_missing"
        )
    relevant_records = []
    unit_membership = []
    paper_ids: set[str] = set()
    for estimate_id in group.estimate_ids:
        estimate = estimates[estimate_id]
        contrast = contrasts.get(estimate.contrast_id)
        if contrast is None or contrast.cohort_id not in cohorts:
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_authorization_contrast_join_missing"
            )
        cohort = cohorts[contrast.cohort_id]
        record = records.get(estimate.effect.paper_id)
        if record is None:
            raise MetaSynSynthesisYieldV2Error(
                "metasyn_synthesis_v2_authorization_publication_join_missing"
            )
        relevant_records.append(record)
        paper_ids.add(estimate.effect.paper_id)
        unit_membership.append(
            {
                "cohort_id": cohort.cohort_id,
                "contrast_id": contrast.contrast_id,
                "estimate_id": estimate.estimate_id,
                "paper_id": estimate.effect.paper_id,
                "study_id": cohort.study_id,
            }
        )
    unique_records = {
        item.record_sha256: item for item in relevant_records
    }
    if any(
        item.adapter_publication_result_sha256 is None
        or item.original_source_grounding_receipt_sha256 is None
        for item in unique_records.values()
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_authorization_record_receipt_missing"
        )
    issues = {
        "pairwise_synthesis_unit_independence_not_source_authorized",
        "protocol_contrast_not_source_authorized",
        "structural_claims_not_source_authorized",
    }
    paper_cohorts: dict[str, set[str]] = {}
    for graph_estimate in graph.outcome_estimates:
        graph_contrast = contrasts.get(graph_estimate.contrast_id)
        if graph_contrast is not None:
            paper_cohorts.setdefault(graph_estimate.effect.paper_id, set()).add(
                graph_contrast.cohort_id
            )
    if any(len(paper_cohorts.get(paper_id, set())) > 1 for paper_id in paper_ids):
        issues.add(
            "within_publication_cohort_independence_not_source_authorized"
        )
    if group.analysis_population is None:
        issues.add("analysis_population_not_source_authorized")
    payload: dict[str, Any] = {
        "receipt_version": SYNTHESIS_UNIT_AUTHORIZATION_RECEIPT_VERSION,
        "authorization_scope": (
            "structural_identity_and_independence_not_yet_source_authorized"
        ),
        "question_spec_sha256": question.question_spec_sha256,
        "question_bundle_sha256": question.question_bundle_sha256,
        "compatibility_sha256": group.compatibility_sha256,
        "group_graph_sha256": group.group_graph_sha256,
        "estimate_ids": sorted(group.estimate_ids),
        "publication_record_sha256s": sorted(unique_records),
        "runtime_row_result_sha256s": sorted(
            {item.runtime_row_result_sha256 for item in unique_records.values()}
        ),
        "adapter_publication_result_sha256s": sorted(
            {
                item.adapter_publication_result_sha256
                for item in unique_records.values()
                if item.adapter_publication_result_sha256 is not None
            }
        ),
        "source_row_sha256s": sorted(
            {item.source_row_sha256 for item in unique_records.values()}
        ),
        "source_document_sha256s": sorted(
            {item.source_document_sha256 for item in unique_records.values()}
        ),
        "original_source_grounding_receipt_sha256s": sorted(
            {
                item.original_source_grounding_receipt_sha256
                for item in unique_records.values()
                if item.original_source_grounding_receipt_sha256 is not None
            }
        ),
        "cohort_reconciliation_sha256": question.cohort_reconciliation_sha256,
        "canonical_unit_membership_sha256": hash_canonical(
            sorted(unit_membership, key=lambda item: item["estimate_id"])
        ),
        "structural_claim_support_receipt_sha256s": [],
        "pairwise_independence_support_receipt_sha256s": [],
        "structural_source_authorization_complete": False,
        "protocol_contrast_binding_complete": False,
        "reconciliation_coverage_complete": True,
        "pairwise_independence_complete": False,
        "authorizes_synthesis_input": False,
        "issues": sorted(issues),
    }
    return MetaSynSynthesisUnitAuthorizationReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _freeze_provisional_single_estimate_group(
    *,
    question: MetaSynQuestionSynthesisYieldV1,
    group: MetaSynCompatibilityGroupYieldV1,
    estimate_id: str,
) -> MetaSynCompatibilityGroupYieldV1:
    graph = question.cohort_reconciliation.reconciled_graph
    if graph is None:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_provisional_group_graph_missing"
        )
    subgroup = _subgraph(graph, [estimate_id])
    estimate = subgroup.outcome_estimates[0]
    contrast = next(
        item for item in subgroup.contrasts if item.contrast_id == estimate.contrast_id
    )
    arms = {item.arm_id: item for item in subgroup.arms}
    treatment = arms[contrast.treatment_arm_id]
    comparator = arms[contrast.comparator_arm_id]
    compatibility_sha256 = hash_canonical(
        {
            "base_compatibility_sha256": group.compatibility_sha256,
            "provisional_policy": (
                "single_estimate_until_protocol_contrast_and_structural_identity_"
                "are_source_authorized"
            ),
            "estimate_id": estimate_id,
            "analysis_population": estimate.analysis_population,
            "contrast": {
                "label": contrast.label,
                "estimand": contrast.estimand,
                "positive_direction_means": contrast.positive_direction_means,
                "treatment_arm": treatment.model_dump(mode="json"),
                "comparator_arm": comparator.model_dump(mode="json"),
            },
        }
    )
    payload = group.model_dump(mode="python", exclude={"group_sha256"})
    payload.update(
        {
            "compatibility_sha256": compatibility_sha256,
            "estimate_ids": [estimate_id],
            "paper_ids": [estimate.effect.paper_id],
            "cohort_ids": sorted(item.cohort_id for item in subgroup.cohorts),
            "group_graph_sha256": hash_canonical(subgroup),
            "cross_publication_identity_assurance": "unresolved_blocked",
            "synthesis_input_eligible": False,
            "synthesis_attempted": False,
            "synthesis_completed": False,
            "stage": (
                "blocked_harmonization"
                if group.harmonization_status != "estimable"
                else "blocked_cross_publication_identity"
            ),
            "blocker": (
                group.blocker
                if group.harmonization_status != "estimable"
                else "cross_publication_cohort_identity_unresolved"
            ),
            "synthesis_status": None,
            "synthesis_mode": None,
            "synthesis_reason": None,
            "synthesis_sha256": None,
        }
    )
    return MetaSynCompatibilityGroupYieldV1.model_validate(
        {**payload, "group_sha256": hash_canonical(payload)}
    )


def _enforce_v2_synthesis_unit_authorization(
    question: MetaSynQuestionSynthesisYieldV1,
) -> tuple[
    MetaSynQuestionSynthesisYieldV1,
    list[MetaSynSynthesisUnitAuthorizationReceiptV2],
]:
    groups = [
        _freeze_provisional_single_estimate_group(
            question=question,
            group=group,
            estimate_id=estimate_id,
        )
        for group in question.compatibility_groups
        for estimate_id in group.estimate_ids
    ]
    groups.sort(key=lambda item: item.compatibility_sha256)
    receipts = [
        _freeze_non_authorizing_synthesis_unit_receipt(
            question=question,
            group=group,
        )
        for group in groups
    ]
    graph = question.cohort_reconciliation.reconciled_graph
    if graph is None:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_authorization_graph_missing"
        )
    payload = question.model_dump(mode="python", exclude={"question_report_sha256"})
    payload.update(
        {
            "compatibility_groups": groups,
            "compatibility_group_sha256s": sorted(
                item.group_sha256 for item in groups
            ),
            "synthesis_input_group_count": sum(
                item.synthesis_input_eligible for item in groups
            ),
            "synthesis_attempted_group_count": sum(
                item.synthesis_attempted for item in groups
            ),
            "synthesis_completed_group_count": sum(
                item.synthesis_completed for item in groups
            ),
            "blockers": _question_blockers(question.publication_records, groups),
            "residual_conflicts": _residual_conflicts(
                graph=graph,
                reconciliation=question.cohort_reconciliation,
                groups=groups,
            ),
        }
    )
    blocked = MetaSynQuestionSynthesisYieldV1.model_validate(
        {**payload, "question_report_sha256": hash_canonical(payload)}
    )
    return blocked, receipts


def compute_metasyn_synthesis_yield_v2_fingerprint(
    *,
    repository_root: Path,
    hosted_runtime_report: MetaSynHostedPrivateYieldReportV1,
    adapter_bundle: MetaSynBoundedAdapterBundleV1,
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1,
    row_views: Sequence[MetaSynHostedSynthesisRowViewV2],
) -> PipelineFingerprint:
    """Bind the v2 dependency closure and every scientific upstream identity."""

    root = repository_root.resolve(strict=True)
    hosted = _canonical_hosted_report(hosted_runtime_report)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(prepare_bundle)
    views = [MetaSynHostedSynthesisRowViewV2.model_validate(item) for item in row_views]
    view_membership = hash_canonical(
        sorted(item.row_result_sha256 for item in views)
    )
    component = PipelineComponentSpec(
        component_id="metasyn-hosted-synthesis-yield",
        component_version=EVALUATION_COMPONENT_VERSION,
        file_paths=sorted({*_python_dependency_closure(root), *_NON_PYTHON_INPUTS}),
        settings={
            "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
            "all_hosted_rows_projected_to_terminal_fragments": True,
            "dependency_closure_entrypoints": list(_DEPENDENCY_ENTRYPOINTS),
            "downstream_verifier_pipeline_sha256": (
                hosted.downstream_verifier_pipeline_sha256
            ),
            "hosted_execution_bundle_sha256": hosted.execution_bundle_sha256,
            "hosted_runtime_private_report_sha256": hosted.report_sha256,
            "hosted_runtime_pipeline_sha256": hosted.runtime_pipeline_sha256,
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name)
                for name in ("numpy", "pydantic", "scipy")
            },
            "original_source_grounding_replay_required": True,
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "prepare_bundle_sha256": prepared.prepare_bundle_sha256,
            "provider_neutral_yield_report_sha256": (
                hosted.provider_neutral_yield_report_sha256
            ),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "reference_direction_conclusion_and_test_labels_opened": False,
            "row_view_membership_sha256": view_membership,
            "synthesis_unit_authorization_contract": (
                SYNTHESIS_UNIT_AUTHORIZATION_RECEIPT_VERSION
            ),
            "synthesis_unit_authorization_policy": (
                "fail_closed_until_structural_claims_protocol_binding_and_"
                "pairwise_independence_are_source_or_reviewer_authorized"
            ),
            "compatibility_group_policy": (
                "provisional_single_estimate_strata_until_protocol_contrast_"
                "binding_is_source_authorized"
            ),
            "synthesis_scope": (
                "hosted_yield_only_full_text_source_replayed_compatible_"
                "identity_resolved"
            ),
            "title_abstract_synthesis_input_permitted": False,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


class MetaSynSynthesisYieldReportV2(MetaSynSynthesisYieldReportV1):
    evaluation_version: Literal["metasyn-hosted-synthesis-yield-v2"] = (
        EVALUATION_VERSION
    )
    status: Literal["complete_hosted_label_blind_synthesis_yield_evaluation"] = (
        "complete_hosted_label_blind_synthesis_yield_evaluation"
    )
    hosted_runtime_private_report_sha256: str
    hosted_runtime_pipeline_sha256: str
    hosted_execution_bundle_sha256: str
    provider_neutral_yield_report_present: bool
    provider_neutral_yield_report_sha256: str | None
    row_view_membership_sha256: str
    row_views: Annotated[
        list[MetaSynHostedSynthesisRowViewV2], Field(min_length=32, max_length=32)
    ]
    row_view_sha256s: Annotated[list[str], Field(min_length=32, max_length=32)]
    synthesis_unit_authorization_receipts: list[
        MetaSynSynthesisUnitAuthorizationReceiptV2
    ]
    synthesis_unit_authorization_receipt_sha256s: list[str]
    synthesis_unit_authorization_receipt_count: Annotated[int, Field(ge=0)]
    authorized_synthesis_input_group_count: Literal[0] = 0
    compatibility_groups_are_provisional_single_estimate_strata: Literal[True] = True

    @field_validator(
        "hosted_runtime_private_report_sha256",
        "hosted_runtime_pipeline_sha256",
        "hosted_execution_bundle_sha256",
        "provider_neutral_yield_report_sha256",
        "row_view_membership_sha256",
    )
    @classmethod
    def validate_v2_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"metasyn_synthesis_v2_report_hash_invalid:{info.field_name}"
            )
        return value

    @field_validator("row_view_sha256s")
    @classmethod
    def validate_row_view_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not SHA256_RE.fullmatch(item) for item in value
        ):
            raise ValueError("metasyn_synthesis_v2_row_view_hashes_invalid")
        return value

    @field_validator("synthesis_unit_authorization_receipt_sha256s")
    @classmethod
    def validate_authorization_receipt_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not SHA256_RE.fullmatch(item) for item in value
        ):
            raise ValueError(
                "metasyn_synthesis_v2_authorization_receipt_hashes_invalid"
            )
        return value

    @model_validator(mode="after")
    def validate_v2_report(self) -> MetaSynSynthesisYieldReportV2:
        if (
            self.runtime_private_report_sha256
            != self.hosted_runtime_private_report_sha256
            or self.runtime_pipeline_sha256 != self.hosted_runtime_pipeline_sha256
        ):
            raise ValueError("metasyn_synthesis_v2_hosted_hash_alias_mismatch")
        if self.provider_neutral_yield_report_present != (
            self.provider_neutral_yield_report_sha256 is not None
        ):
            raise ValueError("metasyn_synthesis_v2_provider_presence_mismatch")
        if self.provider_neutral_yield_report_present and any(
            item.adapter_publication_result is None for item in self.row_views
        ):
            raise ValueError(
                "metasyn_synthesis_v2_provider_report_with_missing_row_result"
            )
        contexts = [item.row_context_sha256 for item in self.row_views]
        if contexts != sorted(set(contexts)):
            raise ValueError("metasyn_synthesis_v2_row_view_roster_invalid")
        hashes = sorted(item.row_result_sha256 for item in self.row_views)
        if self.row_view_sha256s != hashes:
            raise ValueError("metasyn_synthesis_v2_row_view_hash_alias_mismatch")
        if self.row_view_membership_sha256 != hash_canonical(hashes):
            raise ValueError("metasyn_synthesis_v2_row_view_membership_mismatch")
        publication_hashes = sorted(
            record.runtime_row_result_sha256
            for question in self.question_reports
            for record in question.publication_records
        )
        if publication_hashes != hashes:
            raise ValueError("metasyn_synthesis_v2_terminal_row_coverage_mismatch")
        if any(
            item.hosted_runtime_private_report_sha256
            != self.hosted_runtime_private_report_sha256
            or item.provider_neutral_yield_report_sha256
            != self.provider_neutral_yield_report_sha256
            for item in self.row_views
        ):
            raise ValueError("metasyn_synthesis_v2_row_view_lineage_mismatch")
        views_by_context = {item.row_context_sha256: item for item in self.row_views}
        records_by_context = {
            record.row_context_sha256: (question, record)
            for question in self.question_reports
            for record in question.publication_records
        }
        if set(views_by_context) != set(records_by_context):
            raise ValueError("metasyn_synthesis_v2_context_join_roster_mismatch")
        for context, view in views_by_context.items():
            question, record = records_by_context[context]
            if (
                view.question_spec_sha256 != question.question_spec_sha256
                or view.question_spec_sha256 != record.question_spec_sha256
                or view.question_bundle_sha256 != question.question_bundle_sha256
                or view.question_bundle_sha256 != record.question_bundle_sha256
                or view.source_row_sha256 != record.source_row_sha256
                or view.row_result_sha256 != record.runtime_row_result_sha256
                or view.status != record.runtime_status
                or view.adapter_publication_result_sha256
                != record.adapter_publication_result_sha256
                or view.independence_component_membership_sha256
                != question.independence_component_membership_sha256
            ):
                raise ValueError("metasyn_synthesis_v2_context_join_mismatch")
        receipt_hashes = sorted(
            item.receipt_sha256
            for item in self.synthesis_unit_authorization_receipts
        )
        if (
            self.synthesis_unit_authorization_receipt_sha256s != receipt_hashes
            or self.synthesis_unit_authorization_receipt_count != len(receipt_hashes)
        ):
            raise ValueError(
                "metasyn_synthesis_v2_authorization_receipt_alias_mismatch"
            )
        expected_receipts = sorted(
            (
                _freeze_non_authorizing_synthesis_unit_receipt(
                    question=question,
                    group=group,
                )
                for question in self.question_reports
                for group in question.compatibility_groups
            ),
            key=lambda item: item.receipt_sha256,
        )
        observed_receipts = sorted(
            self.synthesis_unit_authorization_receipts,
            key=lambda item: item.receipt_sha256,
        )
        if observed_receipts != expected_receipts:
            raise ValueError(
                "metasyn_synthesis_v2_authorization_receipt_replay_mismatch"
            )
        if (
            self.synthesis_unit_authorization_receipt_count
            != self.compatibility_group_count
            or self.compatibility_group_count != self.graph_estimate_count
            or any(
                len(group.estimate_ids) != 1
                for question in self.question_reports
                for group in question.compatibility_groups
            )
            or self.authorized_synthesis_input_group_count != 0
            or self.synthesis_input_group_count != 0
            or self.synthesis_attempted_group_count != 0
            or self.synthesis_completed_group_count != 0
        ):
            raise ValueError(
                "metasyn_synthesis_v2_unauthorized_synthesis_not_blocked"
            )
        return self


class MetaSynSynthesisYieldPublicSummaryV2(MetaSynSynthesisYieldPublicSummaryV1):
    summary_version: Literal["metasyn-hosted-synthesis-yield-public-v2"] = (
        PUBLIC_SUMMARY_VERSION
    )
    status: Literal["aggregate_only_hosted_label_blind_synthesis_yield"] = (
        "aggregate_only_hosted_label_blind_synthesis_yield"
    )
    hosted_runtime_private_report_sha256: str
    hosted_runtime_pipeline_sha256: str
    hosted_execution_bundle_sha256: str
    provider_neutral_yield_report_present: bool
    provider_neutral_yield_report_sha256: str | None
    row_view_membership_sha256: str
    synthesis_unit_authorization_receipt_count: Annotated[int, Field(ge=0)]
    authorized_synthesis_input_group_count: Literal[0] = 0
    structural_synthesis_authorization_required: Literal[True] = True
    structural_authorization_details_public: Literal[False] = False
    compatibility_groups_are_provisional_single_estimate_strata: Literal[True] = True

    @field_validator(
        "hosted_runtime_private_report_sha256",
        "hosted_runtime_pipeline_sha256",
        "hosted_execution_bundle_sha256",
        "provider_neutral_yield_report_sha256",
        "row_view_membership_sha256",
    )
    @classmethod
    def validate_v2_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"metasyn_synthesis_v2_public_hash_invalid:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_v2_summary(self) -> MetaSynSynthesisYieldPublicSummaryV2:
        if self.runtime_private_report_sha256 != (
            self.hosted_runtime_private_report_sha256
        ):
            raise ValueError("metasyn_synthesis_v2_public_hosted_hash_alias_mismatch")
        if self.provider_neutral_yield_report_present != (
            self.provider_neutral_yield_report_sha256 is not None
        ):
            raise ValueError("metasyn_synthesis_v2_public_provider_presence_mismatch")
        if (
            self.synthesis_unit_authorization_receipt_count
            != self.compatibility_group_count
            or self.compatibility_group_count != self.graph_estimate_count
            or self.authorized_synthesis_input_group_count != 0
            or self.synthesis_input_group_count != 0
            or self.synthesis_attempted_group_count != 0
            or self.synthesis_completed_group_count != 0
        ):
            raise ValueError(
                "metasyn_synthesis_v2_public_unauthorized_synthesis_not_blocked"
            )
        return self


def _aggregate_payload(
    questions: Sequence[MetaSynQuestionSynthesisYieldV1],
) -> dict[str, Any]:
    publications = [
        record for question in questions for record in question.publication_records
    ]
    groups = [
        group for question in questions for group in question.compatibility_groups
    ]
    return {
        "runtime_contract_typed_publication_count": sum(
            item.runtime_contract_typed for item in publications
        ),
        "diagnostic_only_typed_publication_count": sum(
            item.runtime_contract_typed
            and not item.release_grade_source_grounding_eligible
            for item in publications
        ),
        "original_source_grounding_attempt_count": sum(
            item.original_source_grounding_receipt is not None
            for item in publications
        ),
        "original_source_grounding_authorized_count": sum(
            item.original_source_grounding_authorized for item in publications
        ),
        "release_grade_estimable_publication_count": sum(
            item.stage == "release_grade_fragment_estimable"
            for item in publications
        ),
        "terminal_fragment_count": len(publications),
        "graph_construction_completed_question_count": len(questions),
        "questions_with_estimable_graph": sum(
            item.graph_estimate_count > 0 for item in questions
        ),
        "graph_estimate_count": sum(item.graph_estimate_count for item in questions),
        "compatibility_group_count": len(groups),
        "synthesis_input_group_count": sum(
            item.synthesis_input_eligible for item in groups
        ),
        "synthesis_attempted_group_count": sum(
            item.synthesis_attempted for item in groups
        ),
        "synthesis_completed_group_count": sum(
            item.synthesis_completed for item in groups
        ),
        "questions_with_completed_synthesis": sum(
            item.synthesis_completed_group_count > 0 for item in questions
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
        "blocker_counts": _aggregate_blocker_counts(questions),
        "residual_conflict_counts": _aggregate_residual_conflict_counts(questions),
    }


def freeze_metasyn_synthesis_yield_v2_report(
    *,
    repository_root: Path,
    hosted_runtime_report: MetaSynHostedPrivateYieldReportV1 | Mapping[str, Any],
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1 | Mapping[str, Any],
) -> MetaSynSynthesisYieldReportV2:
    """Freeze the complete hosted 10-question/32-publication yield report in memory."""

    root = repository_root.resolve(strict=True)
    hosted = _canonical_hosted_report(hosted_runtime_report)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(prepare_bundle)
    questions = _validate_prepare_adapter_join(
        hosted=hosted, adapter=adapter, prepared=prepared
    )
    _validate_provider_neutral_report(hosted=hosted, adapter=adapter)
    row_views = _freeze_row_views(hosted=hosted, adapter=adapter)
    row_view_membership = hash_canonical(
        sorted(item.row_result_sha256 for item in row_views)
    )
    fingerprint = compute_metasyn_synthesis_yield_v2_fingerprint(
        repository_root=root,
        hosted_runtime_report=hosted,
        adapter_bundle=adapter,
        prepare_bundle=prepared,
        row_views=row_views,
    )
    adapter_rows = {row.row_context_sha256: row for row in adapter.row_contexts}
    view_rows = {row.row_context_sha256: row for row in row_views}
    raw_question_reports = [
        _freeze_question_report(
            repository_root=root,
            pipeline_sha256=fingerprint.pipeline_sha256,
            question=questions[question_id],
            adapter_rows=adapter_rows,
            runtime_rows=view_rows,
        )
        for question_id in sorted(questions)
    ]
    question_reports: list[MetaSynQuestionSynthesisYieldV1] = []
    synthesis_unit_authorization_receipts: list[
        MetaSynSynthesisUnitAuthorizationReceiptV2
    ] = []
    for raw_question in raw_question_reports:
        blocked_question, receipts = _enforce_v2_synthesis_unit_authorization(
            raw_question
        )
        question_reports.append(blocked_question)
        synthesis_unit_authorization_receipts.extend(receipts)
    synthesis_unit_authorization_receipts.sort(
        key=lambda item: item.receipt_sha256
    )
    aggregate = _aggregate_payload(question_reports)
    payload: dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "status": "complete_hosted_label_blind_synthesis_yield_evaluation",
        "evaluation_pipeline_fingerprint": fingerprint,
        "evaluation_pipeline_sha256": fingerprint.pipeline_sha256,
        "runtime_private_report_sha256": hosted.report_sha256,
        "runtime_pipeline_sha256": hosted.runtime_pipeline_sha256,
        "hosted_runtime_private_report_sha256": hosted.report_sha256,
        "hosted_runtime_pipeline_sha256": hosted.runtime_pipeline_sha256,
        "hosted_execution_bundle_sha256": hosted.execution_bundle_sha256,
        "provider_neutral_yield_report_present": (
            hosted.provider_neutral_yield_report is not None
        ),
        "provider_neutral_yield_report_sha256": (
            hosted.provider_neutral_yield_report_sha256
        ),
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "prepare_bundle_sha256": prepared.prepare_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            hosted.downstream_verifier_pipeline_sha256
        ),
        "row_view_membership_sha256": row_view_membership,
        "row_views": row_views,
        "row_view_sha256s": sorted(item.row_result_sha256 for item in row_views),
        "synthesis_unit_authorization_receipts": (
            synthesis_unit_authorization_receipts
        ),
        "synthesis_unit_authorization_receipt_sha256s": sorted(
            item.receipt_sha256 for item in synthesis_unit_authorization_receipts
        ),
        "synthesis_unit_authorization_receipt_count": len(
            synthesis_unit_authorization_receipts
        ),
        "authorized_synthesis_input_group_count": 0,
        "compatibility_groups_are_provisional_single_estimate_strata": True,
        "question_count": EXPECTED_SELECTED_QUESTIONS,
        "component_count": EXPECTED_SELECTED_COMPONENTS,
        "publication_count": EXPECTED_SELECTED_PAPERS,
        "question_reports": question_reports,
        "question_report_sha256s": sorted(
            item.question_report_sha256 for item in question_reports
        ),
        **aggregate,
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
    return MetaSynSynthesisYieldReportV2.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_synthesis_yield_v2_public_summary(
    *, report: MetaSynSynthesisYieldReportV2 | Mapping[str, Any]
) -> MetaSynSynthesisYieldPublicSummaryV2:
    """Derive the exact identifier- and source-free aggregate companion."""

    canonical = MetaSynSynthesisYieldReportV2.model_validate(report)
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
    # The inherited caveat contract is intentionally retained byte-for-byte.  Every
    # statement applies equally to hosted generation and prevents provider-specific
    # mechanics from acquiring scientific authority.
    payload: dict[str, Any] = {
        "summary_version": PUBLIC_SUMMARY_VERSION,
        "status": "aggregate_only_hosted_label_blind_synthesis_yield",
        "evaluation_pipeline_sha256": canonical.evaluation_pipeline_sha256,
        "runtime_private_report_sha256": canonical.runtime_private_report_sha256,
        "hosted_runtime_private_report_sha256": (
            canonical.hosted_runtime_private_report_sha256
        ),
        "hosted_runtime_pipeline_sha256": canonical.hosted_runtime_pipeline_sha256,
        "hosted_execution_bundle_sha256": canonical.hosted_execution_bundle_sha256,
        "provider_neutral_yield_report_present": (
            canonical.provider_neutral_yield_report_present
        ),
        "provider_neutral_yield_report_sha256": (
            canonical.provider_neutral_yield_report_sha256
        ),
        "row_view_membership_sha256": canonical.row_view_membership_sha256,
        "synthesis_unit_authorization_receipt_count": (
            canonical.synthesis_unit_authorization_receipt_count
        ),
        "authorized_synthesis_input_group_count": (
            canonical.authorized_synthesis_input_group_count
        ),
        "structural_synthesis_authorization_required": True,
        "structural_authorization_details_public": False,
        "compatibility_groups_are_provisional_single_estimate_strata": True,
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
    return MetaSynSynthesisYieldPublicSummaryV2.model_validate(
        {**payload, "summary_sha256": hash_canonical(payload)}
    )


def validate_metasyn_synthesis_yield_v2_report(
    *,
    report: MetaSynSynthesisYieldReportV2 | Mapping[str, Any],
    repository_root: Path,
    hosted_runtime_report: MetaSynHostedPrivateYieldReportV1 | Mapping[str, Any],
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    prepare_bundle: MetaSynTypedPilotPrepareBundleV1 | Mapping[str, Any],
) -> MetaSynSynthesisYieldReportV2:
    """Replay source grounding, graph construction, grouping, and synthesis exactly."""

    canonical = MetaSynSynthesisYieldReportV2.model_validate(report)
    replayed = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=repository_root,
        hosted_runtime_report=hosted_runtime_report,
        adapter_bundle=adapter_bundle,
        prepare_bundle=prepare_bundle,
    )
    if replayed != canonical:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_external_replay_mismatch"
        )
    return canonical


def validate_metasyn_synthesis_yield_v2_public_summary(
    *,
    summary: MetaSynSynthesisYieldPublicSummaryV2 | Mapping[str, Any],
    report: MetaSynSynthesisYieldReportV2 | Mapping[str, Any],
) -> MetaSynSynthesisYieldPublicSummaryV2:
    canonical = MetaSynSynthesisYieldPublicSummaryV2.model_validate(summary)
    expected = freeze_metasyn_synthesis_yield_v2_public_summary(report=report)
    if canonical != expected:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_public_external_replay_mismatch"
        )
    return canonical


def _read_prepare_bundle_after_external_replay(
    *, repository_root: Path, pilot_workspace: Path
) -> MetaSynTypedPilotPrepareBundleV1:
    root = repository_root.resolve(strict=True)
    receipt = validate_metasyn_typed_pilot_prepare(
        repository_root=root, workspace=pilot_workspace
    )
    workspace = (
        pilot_workspace if pilot_workspace.is_absolute() else root / pilot_workspace
    )
    path = workspace.resolve(strict=True) / PREPARE_BUNDLE_FILENAME
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_prepare_bundle_unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_prepare_bundle_not_object"
        )
    prepared = MetaSynTypedPilotPrepareBundleV1.model_validate(payload)
    if (
        hashlib.sha256(raw).hexdigest() != receipt.prepare_bundle_file_sha256
        or prepared.prepare_bundle_sha256 != receipt.prepare_bundle_sha256
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_prepare_bundle_post_replay_mismatch"
        )
    return prepared


def evaluate_current_metasyn_synthesis_yield_v2(
    *,
    repository_root: Path,
    hosted_runtime_workspace: Path,
    pilot_workspace: Path,
    expected_execution_bundle_sha256: str,
) -> tuple[MetaSynSynthesisYieldReportV2, MetaSynSynthesisYieldPublicSummaryV2]:
    """Externally validate every upstream artifact, then derive without writing."""

    from literature_multiverse.metasyn_bounded_hosted_runtime import (
        validate_finalized_metasyn_hosted_runtime,
    )

    root = repository_root.resolve(strict=True)
    hosted = validate_finalized_metasyn_hosted_runtime(
        workspace=hosted_runtime_workspace,
        repository_root=root,
        expected_execution_bundle_sha256=expected_execution_bundle_sha256,
    )
    _, execution_bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=hosted_runtime_workspace,
        repository_root=root,
        external_replay=True,
    )
    if (
        execution_bundle.execution_bundle_sha256
        != expected_execution_bundle_sha256
        or hosted.execution_bundle_sha256 != expected_execution_bundle_sha256
    ):
        raise MetaSynSynthesisYieldV2Error(
            "metasyn_synthesis_v2_execution_bundle_anchor_mismatch"
        )
    adapter = validate_metasyn_bounded_adapter_bundle_external_replay(
        adapter_bundle=execution_bundle.adapter_bundle,
        repository_root=root,
        workspace=pilot_workspace,
    )
    prepared = _read_prepare_bundle_after_external_replay(
        repository_root=root, pilot_workspace=pilot_workspace
    )
    _validate_provider_neutral_report(hosted=hosted, adapter=adapter)
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=root,
        hosted_runtime_report=hosted,
        adapter_bundle=adapter,
        prepare_bundle=prepared,
    )
    public = freeze_metasyn_synthesis_yield_v2_public_summary(report=report)
    validate_metasyn_synthesis_yield_v2_public_summary(
        summary=public, report=report
    )
    return report, public


__all__ = [
    "EVALUATION_VERSION",
    "PUBLIC_SUMMARY_VERSION",
    "MetaSynHostedSynthesisRowViewV2",
    "MetaSynSynthesisUnitAuthorizationReceiptV2",
    "MetaSynSynthesisYieldPublicSummaryV2",
    "MetaSynSynthesisYieldReportV2",
    "MetaSynSynthesisYieldV2Error",
    "compute_metasyn_synthesis_yield_v2_fingerprint",
    "evaluate_current_metasyn_synthesis_yield_v2",
    "freeze_metasyn_synthesis_yield_v2_public_summary",
    "freeze_metasyn_synthesis_yield_v2_report",
    "validate_metasyn_synthesis_yield_v2_public_summary",
    "validate_metasyn_synthesis_yield_v2_report",
]
