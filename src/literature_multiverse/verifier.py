"""Unified, provider-free orchestration for scientific claim verification.

This module owns the supported path from a claim manifest and frozen corpus artifact to
the evidence graph, statistical synthesis, graph-derived audit priorities, fail-closed
release assessment, and verification certificate.  Network retrieval and model-backed
extraction remain replaceable upstream adapters; this boundary consumes their frozen
outputs and never opens the corpus during a run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from literature_multiverse.budgeted_verification import (
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ReleaseGuardConfig,
)
from literature_multiverse.calibration import FrozenCalibrationBundle
from literature_multiverse.certificate import (
    CertificateLineageStage,
    VerificationCertificate,
    freeze_verification_certificate,
)
from literature_multiverse.claim_release import (
    AuditResolutionReceipt,
    ClaimReleaseConfig,
    ClaimTarget,
    TargetDirection,
    assess_claim_release,
)
from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    AdapterIssueSeverity,
    CohortIdentity,
    CohortIdentityBasis,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
    adapt_finding_row,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.meta_analysis import (
    build_graph_counterfactual_audit_plan,
    synthesize_evidence_graph,
)
from literature_multiverse.models import SHA256_RE, ContractModel, FindingRow
from literature_multiverse.records import read_parquet_records


class VerificationContractError(ValueError):
    """The unified verifier cannot safely complete the requested run."""


Scalar = str | int | float | bool | None


class ScientificClaim(ContractModel):
    """The AI-generated statement whose literature support is being assessed."""

    statement: Annotated[str, Field(min_length=1)]
    direction: TargetDirection
    outcome_name: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)] | None = None
    estimand: str | None = None
    conditions: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, value: dict[str, Scalar]) -> dict[str, Scalar]:
        if any(not key.strip() for key in value):
            raise ValueError("claim_condition_name_empty")
        for item in value.values():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("claim_condition_value_nonfinite")
        return value


class VerificationProtocol(ContractModel):
    """Frozen search/screening boundary associated with the claim."""

    corpus_cutoff: Annotated[str, Field(min_length=1)]
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("inclusion_criteria", "exclusion_criteria")
    @classmethod
    def validate_criteria(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("verification_protocol_criterion_empty")
        if len(value) != len(set(value)):
            raise ValueError("verification_protocol_criteria_not_unique")
        return value


class AuditPolicyConfig(ContractModel):
    """Prospective inputs for audit ranking; no correctness labels are accepted."""

    error_probability: Annotated[float, Field(ge=0, le=1)] = 0.5
    probability_basis: ProbabilityBasis = ProbabilityBasis.HEURISTIC
    probability_source: Annotated[str, Field(min_length=1)] = (
        "unvalidated-default-error-probability"
    )
    verification_minutes_per_item: Annotated[float, Field(gt=0)] = 10.0
    item_error_probabilities: dict[str, float] = Field(default_factory=dict)
    item_verification_minutes: dict[str, float] = Field(default_factory=dict)
    decision_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5

    @model_validator(mode="after")
    def validate_item_overrides(self) -> AuditPolicyConfig:
        if any(not item_id.strip() for item_id in self.item_error_probabilities):
            raise ValueError("audit_item_error_probability_id_empty")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in self.item_error_probabilities.values()
        ):
            raise ValueError("audit_item_error_probability_invalid")
        if any(not item_id.strip() for item_id in self.item_verification_minutes):
            raise ValueError("audit_item_verification_cost_id_empty")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.item_verification_minutes.values()
        ):
            raise ValueError("audit_item_verification_cost_invalid")
        return self


class LegacyAdapterConfig(ContractModel):
    """Explicit orientation used when lifting the old categorical findings ledger."""

    contrast_label: Annotated[str, Field(min_length=1)] = "intervention_vs_comparator"
    positive_direction_means: Annotated[str, Field(min_length=1)] = (
        "higher outcome under the intervention"
    )
    treatment_label: Annotated[str, Field(min_length=1)] = "intervention"
    comparator_label: Annotated[str, Field(min_length=1)] = "comparator"


class AuditGuardConfig(ContractModel):
    """Serializable view of :class:`ReleaseGuardConfig`."""

    max_unresolved_item_influence: Annotated[float, Field(ge=0, le=1)] = 0.05
    max_unresolved_expected_claim_loss: Annotated[float, Field(ge=0)] = 0.05
    block_counterfactual_conclusion_flips: bool = True
    require_calibrated_error_probabilities: bool = True
    require_error_probability_upper_bounds: bool = True
    max_residual_decision_risk: Annotated[float, Field(ge=0, le=1)] = 0.05

    def to_runtime(self) -> ReleaseGuardConfig:
        return ReleaseGuardConfig(**self.model_dump(mode="python"))


class ClaimManifest(ContractModel):
    """Closed YAML/JSON input contract for ``lm verify``."""

    claim_manifest_version: Literal["1"] = "1"
    question_id: Annotated[
        str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    ]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    claim: ScientificClaim
    protocol: VerificationProtocol
    audit: AuditPolicyConfig = Field(default_factory=AuditPolicyConfig)
    legacy_adapter: LegacyAdapterConfig = Field(default_factory=LegacyAdapterConfig)
    release: ClaimReleaseConfig = Field(default_factory=ClaimReleaseConfig)
    audit_guard: AuditGuardConfig = Field(default_factory=AuditGuardConfig)
    pipeline_sha256: str | None = None

    @field_validator("pipeline_sha256")
    @classmethod
    def validate_pipeline_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("claim_manifest_pipeline_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_condition_contract(self) -> ClaimManifest:
        condition_names = sorted(self.claim.conditions)
        declared = self.release.prespecified_condition_moderators
        if condition_names and declared != condition_names:
            raise ValueError(
                "claim_conditions_must_match_prespecified_condition_moderators"
            )
        return self


class CorpusEligibilityRecord(ContractModel):
    """One retrieved paper and its terminal screening state."""

    paper_id: Annotated[str, Field(min_length=1)]
    title: str | None = None
    status: Literal["included", "excluded", "pending"]
    reason: Annotated[str, Field(min_length=1)]
    source: str | None = None


class CorpusAdapterIssue(ContractModel):
    """A graph-conversion caveat tied to its source evidence item."""

    severity: AdapterIssueSeverity
    code: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]
    paper_id: str | None = None
    finding_id: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusLoadResult:
    """Frozen corpus payload ready for orchestration."""

    corpus_id: str
    source_label: str
    source_format: str
    source_sha256: str
    graph: EvidenceGraph
    eligibility: tuple[CorpusEligibilityRecord, ...]
    adapter_issues: tuple[CorpusAdapterIssue, ...]
    metadata: dict[str, Any]

    def certificate_payload(self) -> dict[str, Any]:
        included = sum(item.status == "included" for item in self.eligibility)
        excluded = sum(item.status == "excluded" for item in self.eligibility)
        pending = sum(item.status == "pending" for item in self.eligibility)
        return {
            "corpus_id": self.corpus_id,
            "source_label": self.source_label,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "eligibility": [item.model_dump(mode="json") for item in self.eligibility],
            "eligibility_counts": {
                "excluded": excluded,
                "included": included,
                "pending": pending,
            },
            "graph_counts": {
                "cohorts": len(self.graph.cohorts),
                "estimates": len(self.graph.outcome_estimates),
                "publications": len(self.graph.publications),
                "studies": len(self.graph.studies),
            },
            "metadata": self.metadata,
        }


def load_claim_manifest(path: Path) -> ClaimManifest:
    """Load a closed claim YAML/JSON document without resolving external references."""

    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationContractError(f"claim_manifest_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise VerificationContractError("claim_manifest_must_be_object")
    return ClaimManifest.model_validate(payload)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _json_plain(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if type(value).__name__ == "NAType":
        return None
    if isinstance(value, dict):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    return value


def _finding_from_flat_record(record: dict[str, Any]) -> FindingRow:
    payload = {key: _json_plain(value) for key, value in record.items()}
    if "moderators" not in payload:
        moderator_keys = sorted(key for key in payload if key.startswith("mod__"))
        payload["moderators"] = {
            key.removeprefix("mod__"): payload.pop(key) for key in moderator_keys
        }
    return FindingRow.model_validate(payload)


def _paper_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record["paper_id"]): {key: _json_plain(value) for key, value in record.items()}
        for record in records
        if record.get("paper_id")
    }


def _eligibility_from_papers(
    records: list[dict[str, Any]],
) -> tuple[CorpusEligibilityRecord, ...]:
    output: list[CorpusEligibilityRecord] = []
    for raw in records:
        record = {key: _json_plain(value) for key, value in raw.items()}
        eligible = record.get("eligible")
        screen_status = str(record.get("screen_status") or "")
        if eligible is True:
            status: Literal["included", "excluded", "pending"] = "included"
            reason = "eligible_after_screening_and_extraction"
        elif eligible is False:
            status = "excluded"
            reason = str(record.get("exclusion_reason") or "ineligible_after_mapping")
        elif screen_status == "excluded":
            status = "excluded"
            reason = str(record.get("screen_reason") or "excluded_during_screening")
        else:
            status = "pending"
            reason = "terminal_eligibility_not_available"
        output.append(
            CorpusEligibilityRecord(
                paper_id=str(record["paper_id"]),
                title=(str(record["title"]) if record.get("title") else None),
                status=status,
                reason=reason,
                source=(str(record["source"]) if record.get("source") else None),
            )
        )
    return tuple(sorted(output, key=lambda item: item.paper_id))


def _merge_graphs(graphs: list[EvidenceGraph]) -> EvidenceGraph:
    if not graphs:
        raise VerificationContractError("legacy_corpus_contains_no_findings")

    def merge(field: str, identity: str) -> list[Any]:
        values: dict[str, Any] = {}
        for graph in graphs:
            for item in getattr(graph, field):
                key = str(getattr(item, identity))
                if key in values and values[key] != item:
                    raise VerificationContractError(
                        f"evidence_graph_identity_collision:{field}:{key}"
                    )
                values[key] = item
        return [values[key] for key in sorted(values)]

    return EvidenceGraph(
        publications=merge("publications", "publication_id"),
        studies=merge("studies", "study_id"),
        cohorts=merge("cohorts", "cohort_id"),
        arms=merge("arms", "arm_id"),
        contrasts=merge("contrasts", "contrast_id"),
        outcome_estimates=merge("outcome_estimates", "estimate_id"),
        evidence_spans=merge("evidence_spans", "span_id"),
    )


def adapt_legacy_findings(
    findings: list[FindingRow],
    *,
    settings: LegacyAdapterConfig,
    papers: dict[str, dict[str, Any]] | None = None,
) -> tuple[EvidenceGraph, tuple[CorpusAdapterIssue, ...]]:
    """Lift the old categorical ledger without inventing cohorts or effect sizes."""

    papers = papers or {}
    graphs: list[EvidenceGraph] = []
    issues: list[CorpusAdapterIssue] = []
    for finding in sorted(findings, key=lambda item: item.finding_id):
        item_key = _stable_id("legacy", finding.finding_id)
        paper_key = _stable_id("publication", finding.paper_id)
        metadata = papers.get(finding.paper_id, {})
        publication = PublicationIdentity(
            publication_id=paper_key,
            paper_id=finding.paper_id,
            doc_id=finding.doc_id,
            doi=metadata.get("doi"),
            pmid=(str(metadata["pmid"]) if metadata.get("pmid") else None),
            title=(str(metadata["title"]) if metadata.get("title") else None),
            publication_year=(
                int(metadata["pub_year"]) if metadata.get("pub_year") is not None else None
            ),
        )
        context = GraphAdapterContext(
            publication=publication,
            study_id=f"study-{item_key}",
            cohort_identity=CohortIdentity(
                cohort_id=f"cohort-{item_key}",
                basis=CohortIdentityBasis.LEGACY_PLACEHOLDER,
                rationale=(
                    "The legacy FindingRow does not encode a reconciled participant/sample "
                    "identity; human reconciliation is required before synthesis."
                ),
            ),
            treatment_arm_id=f"arm-treatment-{item_key}",
            comparator_arm_id=f"arm-comparator-{item_key}",
            contrast_id=f"contrast-{item_key}",
            contrast_label=settings.contrast_label,
            positive_direction_means=settings.positive_direction_means,
            treatment_label=finding.intervention or settings.treatment_label,
            comparator_label=finding.comparator or settings.comparator_label,
        )
        result = adapt_finding_row(finding, context=context)
        graphs.append(result.graph)
        issues.extend(
            CorpusAdapterIssue(
                severity=issue.severity,
                code=issue.code,
                detail=issue.detail,
                paper_id=finding.paper_id,
                finding_id=finding.finding_id,
            )
            for issue in result.issues
        )
    return _merge_graphs(graphs), tuple(
        sorted(issues, key=lambda item: (item.finding_id or "", item.code))
    )


def _graph_eligibility(graph: EvidenceGraph) -> tuple[CorpusEligibilityRecord, ...]:
    return tuple(
        CorpusEligibilityRecord(
            paper_id=publication.paper_id,
            title=publication.title,
            status="included",
            reason="present_in_typed_evidence_graph",
        )
        for publication in sorted(graph.publications, key=lambda item: item.paper_id)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationContractError(f"corpus_json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise VerificationContractError("corpus_json_must_be_object")
    return payload


def _read_jsonl_findings(path: Path) -> list[FindingRow]:
    findings: list[FindingRow] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise VerificationContractError(f"corpus_jsonl_unreadable:{path}") from exc
    for position, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationContractError(
                f"corpus_jsonl_invalid_json:line={position}"
            ) from exc
        if not isinstance(payload, dict):
            raise VerificationContractError(f"corpus_jsonl_row_not_object:line={position}")
        findings.append(_finding_from_flat_record(payload))
    return findings


def load_corpus(path: Path, *, legacy_settings: LegacyAdapterConfig) -> CorpusLoadResult:
    """Load a typed graph/bundle or conservatively adapt normalized legacy findings."""

    source_path = path
    paper_path: Path | None = None
    if path.is_dir():
        graph_path = path / "evidence_graph.json"
        findings_path = path / "findings.parquet"
        if graph_path.is_file():
            source_path = graph_path
        elif findings_path.is_file():
            source_path = findings_path
        else:
            raise VerificationContractError(
                "corpus_directory_requires_evidence_graph_json_or_findings_parquet"
            )
        candidate_papers = path / "papers.parquet"
        paper_path = candidate_papers if candidate_papers.is_file() else None
    elif not path.is_file():
        raise VerificationContractError(f"corpus_path_not_found:{path}")

    paper_records = read_parquet_records(paper_path) if paper_path is not None else []
    paper_metadata = _paper_metadata(paper_records)
    eligibility = _eligibility_from_papers(paper_records)
    source_hash_parts = {"evidence": sha256_file(source_path)}
    if paper_path is not None:
        source_hash_parts["papers"] = sha256_file(paper_path)
    source_sha256 = hash_canonical(source_hash_parts)

    suffix = source_path.suffix.casefold()
    if suffix == ".json":
        payload = _read_json_object(source_path)
        if "graph_schema_version" in payload:
            graph = EvidenceGraph.model_validate(payload)
            bundle_metadata: dict[str, Any] = {}
            issues: tuple[CorpusAdapterIssue, ...] = ()
            corpus_id = path.name
            source_format = "evidence_graph_json"
        elif "graph" in payload:
            allowed = {
                "adapter_issues",
                "corpus_bundle_version",
                "corpus_id",
                "eligibility",
                "graph",
                "metadata",
            }
            extras = sorted(set(payload) - allowed)
            if extras:
                raise VerificationContractError(f"corpus_bundle_unknown_keys:{extras}")
            graph = EvidenceGraph.model_validate(payload["graph"])
            corpus_id = str(payload.get("corpus_id") or path.name)
            source_format = "verification_corpus_bundle_json"
            raw_metadata = payload.get("metadata", {})
            if not isinstance(raw_metadata, dict):
                raise VerificationContractError("corpus_bundle_metadata_must_be_object")
            bundle_metadata = raw_metadata
            raw_issues = payload.get("adapter_issues", [])
            if not isinstance(raw_issues, list):
                raise VerificationContractError("corpus_bundle_adapter_issues_must_be_list")
            issues = tuple(CorpusAdapterIssue.model_validate(item) for item in raw_issues)
            if "eligibility" in payload:
                raw_eligibility = payload["eligibility"]
                if not isinstance(raw_eligibility, list):
                    raise VerificationContractError(
                        "corpus_bundle_eligibility_must_be_list"
                    )
                eligibility = tuple(
                    sorted(
                        (
                            CorpusEligibilityRecord.model_validate(item)
                            for item in raw_eligibility
                        ),
                        key=lambda item: item.paper_id,
                    )
                )
        else:
            raise VerificationContractError(
                "corpus_json_requires_graph_schema_or_graph_bundle"
            )
    elif suffix == ".jsonl":
        findings = _read_jsonl_findings(source_path)
        graph, issues = adapt_legacy_findings(
            findings,
            settings=legacy_settings,
            papers=paper_metadata,
        )
        corpus_id = path.name
        source_format = "legacy_findings_jsonl"
        bundle_metadata = {"legacy_finding_count": len(findings)}
    elif suffix == ".parquet":
        findings = [
            _finding_from_flat_record(record) for record in read_parquet_records(source_path)
        ]
        graph, issues = adapt_legacy_findings(
            findings,
            settings=legacy_settings,
            papers=paper_metadata,
        )
        corpus_id = path.name
        source_format = "legacy_findings_parquet"
        bundle_metadata = {"legacy_finding_count": len(findings)}
    else:
        raise VerificationContractError(f"corpus_format_not_supported:{suffix}")

    if not eligibility:
        eligibility = _graph_eligibility(graph)
    return CorpusLoadResult(
        corpus_id=corpus_id,
        source_label=path.as_posix(),
        source_format=source_format,
        source_sha256=source_sha256,
        graph=graph,
        eligibility=eligibility,
        adapter_issues=issues,
        metadata=bundle_metadata,
    )


def _disagreement_score(estimate: Any, direction: TargetDirection) -> float:
    numeric = estimate.effect.estimate
    if numeric is not None:
        if numeric == 0:
            return 0.5
        supports = numeric > 0 if direction is TargetDirection.INCREASE else numeric < 0
        return 0.0 if supports else 1.0
    legacy = estimate.legacy_reported_direction
    if legacy is None:
        return 0.5
    if legacy.value in {"mixed", "unclear", "no_effect"}:
        return 0.5
    supports = legacy.value == direction.value
    return 0.0 if supports else 1.0


def build_graph_counterfactual_audits(
    graph: EvidenceGraph,
    *,
    target: ClaimTarget,
    release_config: ClaimReleaseConfig,
    policy: AuditPolicyConfig,
) -> tuple[ClaimModel, list[AuditCandidate], list[dict[str, Any]], dict[str, Any]]:
    """Rerun the real synthesis after removing each matching evidence estimate."""

    selected = sorted(
        (
            estimate
            for estimate in graph.outcome_estimates
            if estimate.outcome_name == target.outcome_name
            and (target.contrast_id is None or estimate.contrast_id == target.contrast_id)
        ),
        key=lambda item: item.estimate_id,
    )
    selected_ids = {estimate.estimate_id for estimate in selected}
    unknown_error_overrides = sorted(set(policy.item_error_probabilities) - selected_ids)
    unknown_cost_overrides = sorted(set(policy.item_verification_minutes) - selected_ids)
    if unknown_error_overrides or unknown_cost_overrides:
        raise VerificationContractError(
            "audit_item_override_identity_unknown:"
            f"errors={unknown_error_overrides}:costs={unknown_cost_overrides}"
        )

    synthesis_kwargs: dict[str, Any] = {
        "outcome_name": target.outcome_name,
        "contrast_id": target.contrast_id,
        "require_explicit_timepoint": release_config.require_explicit_timepoint,
        "confidence_level": release_config.confidence_level,
        "assumed_within_paper_correlation": (
            release_config.assumed_within_paper_correlation
        ),
        "prespecified_moderators": release_config.prespecified_condition_moderators,
        "condition_familywise_alpha": release_config.condition_familywise_alpha,
        "condition_min_papers_per_level": (
            release_config.condition_min_papers_per_level
        ),
    }
    if not selected:
        synthesis = synthesize_evidence_graph(graph, **synthesis_kwargs)
        return (
            ClaimModel(
                intercept=0.0,
                decision_threshold=policy.decision_threshold,
                claim_id=f"{target.outcome_name}-{target.direction.value}",
            ),
            [],
            [],
            synthesis,
        )

    errors = {
        estimate.estimate_id: policy.item_error_probabilities.get(
            estimate.estimate_id, policy.error_probability
        )
        for estimate in selected
    }
    costs = {
        estimate.estimate_id: policy.item_verification_minutes.get(
            estimate.estimate_id, policy.verification_minutes_per_item
        )
        for estimate in selected
    }
    disagreement = {
        estimate.estimate_id: _disagreement_score(estimate, target.direction)
        for estimate in selected
    }
    plan = build_graph_counterfactual_audit_plan(
        graph,
        outcome_name=target.outcome_name,
        target_direction=target.direction.value,
        error_probabilities=errors,
        verification_costs=costs,
        probability_basis=policy.probability_basis,
        probability_source=policy.probability_source,
        disagreement_scores=disagreement,
        contrast_id=target.contrast_id,
        require_explicit_timepoint=release_config.require_explicit_timepoint,
        require_prediction_interval_stability=(
            release_config.require_prediction_interval_stability
        ),
        confidence_level=release_config.confidence_level,
        assumed_within_paper_correlation=(
            release_config.assumed_within_paper_correlation
        ),
        prespecified_moderators=release_config.prespecified_condition_moderators,
        condition_familywise_alpha=release_config.condition_familywise_alpha,
        condition_min_papers_per_level=(
            release_config.condition_min_papers_per_level
        ),
        claim_id=f"{target.outcome_name}-{target.direction.value}",
    )
    counterfactual_rows = [
        {
            "baseline_decision": plan.baseline_decision.model_dump(mode="json"),
            "baseline_synthesis_sha256": hash_canonical(plan.baseline_synthesis),
            "counterfactual_decision": plan.counterfactual_decisions[
                candidate.item_id
            ].model_dump(mode="json"),
            "counterfactual_synthesis": plan.counterfactual_syntheses[
                candidate.item_id
            ],
            "counterfactual_synthesis_sha256": hash_canonical(
                plan.counterfactual_syntheses[candidate.item_id]
            ),
            "item_id": candidate.item_id,
            "scenario": "leave_one_out_actual_synthesis_rerun",
        }
        for candidate in plan.candidates
    ]
    return (
        plan.claim_model,
        list(plan.candidates),
        counterfactual_rows,
        plan.baseline_synthesis,
    )


def _pipeline_identity(
    manifest: ClaimManifest,
    bundle: FrozenCalibrationBundle | None,
) -> tuple[str, str]:
    if manifest.pipeline_sha256 is not None:
        return manifest.pipeline_sha256, "declared_in_claim_manifest"
    if bundle is not None:
        return bundle.pipeline_sha256, "inherited_from_frozen_calibration_bundle"
    digest = hash_canonical(
        {
            "audit_policy": manifest.audit,
            "claim_release": manifest.release,
            "contract": "unified-verifier-v1",
            "evidence_graph_schema_version": "1",
        }
    )
    return digest, "versioned_contract_and_policy_digest"


def run_verification(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    frozen_calibration_bundle: FrozenCalibrationBundle | None = None,
    audit_resolution_receipts: list[AuditResolutionReceipt] | None = None,
    generated_at: datetime | None = None,
) -> VerificationCertificate:
    """Execute the complete frozen-corpus verifier and freeze its certificate."""

    if not math.isfinite(budget_minutes) or budget_minutes < 0:
        raise VerificationContractError("verification_budget_minutes_invalid")
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise VerificationContractError("verification_generated_at_requires_timezone")
    receipts = audit_resolution_receipts or []
    pipeline_sha256, pipeline_basis = _pipeline_identity(
        manifest, frozen_calibration_bundle
    )
    target = ClaimTarget(
        direction=manifest.claim.direction,
        outcome_name=manifest.claim.outcome_name,
        contrast_id=manifest.claim.contrast_id,
    )
    claim_model, candidates, counterfactuals, synthesis = (
        build_graph_counterfactual_audits(
        corpus.graph,
        target=target,
        release_config=manifest.release,
        policy=manifest.audit,
        )
    )
    assessment = assess_claim_release(
        graph=corpus.graph,
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        pipeline_sha256=pipeline_sha256,
        target=target,
        audit_candidates=candidates,
        claim_model=claim_model,
        audit_resolution_receipts=receipts,
        audit_budget=budget_minutes,
        frozen_calibration_bundle=frozen_calibration_bundle,
        config=manifest.release,
        audit_guard_config=manifest.audit_guard.to_runtime(),
    )
    synthesis_hash = hash_canonical(synthesis)
    if assessment.synthesis_sha256 != synthesis_hash:
        raise VerificationContractError("orchestrator_release_synthesis_hash_mismatch")

    reasons = list(assessment.reasons)
    reasons.extend(
        f"adapter:{issue.code}"
        for issue in corpus.adapter_issues
        if issue.severity is AdapterIssueSeverity.BLOCKING
    )
    reasons = sorted(set(reasons))
    status: Literal["released", "abstained"] = (
        "released" if assessment.status.value == "released" and not reasons else "abstained"
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_hash = hash_canonical(manifest_payload)
    graph_hash = hash_canonical(corpus.graph)
    candidate_payload = [asdict(candidate) for candidate in candidates]
    candidate_hash = hash_canonical(candidate_payload)
    release_input_hash = hash_canonical(
        {
            "audit_candidates": candidate_payload,
            "audit_receipts": receipts,
            "budget_minutes": budget_minutes,
            "pipeline_sha256": pipeline_sha256,
            "target": target,
        }
    )
    lineage = [
        CertificateLineageStage(
            stage="corpus_to_evidence_graph",
            input_sha256s=dict(
                sorted(
                    {
                        "claim_manifest": manifest_hash,
                        "corpus_source": corpus.source_sha256,
                    }.items()
                )
            ),
            output_sha256s={"evidence_graph": graph_hash},
            method=f"{corpus.source_format}:closed-corpus-adapter",
        ),
        CertificateLineageStage(
            stage="evidence_graph_to_synthesis",
            input_sha256s={"evidence_graph": graph_hash},
            output_sha256s={"synthesis": synthesis_hash},
            method=str(synthesis.get("mode", "insufficient")),
        ),
        CertificateLineageStage(
            stage="counterfactual_verification_priority",
            input_sha256s=dict(
                sorted(
                    {
                        "evidence_graph": graph_hash,
                        "synthesis": synthesis_hash,
                    }.items()
                )
            ),
            output_sha256s={"audit_candidates": candidate_hash},
            method="actual-leave-one-out-synthesis-reruns",
        ),
        CertificateLineageStage(
            stage="risk_controlled_release",
            input_sha256s={"release_inputs": release_input_hash},
            output_sha256s={"release_decision": assessment.decision_sha256},
            method="prospective-claim-release-v1",
        ),
    ]
    corpus_payload = corpus.certificate_payload()
    corpus_payload["declared_corpus_cutoff"] = manifest.protocol.corpus_cutoff
    corpus_payload["pipeline_identity_basis"] = pipeline_basis
    return freeze_verification_certificate(
        generated_at=generated_at,
        status=status,
        reasons=reasons,
        claim_manifest=manifest_payload,
        corpus=corpus_payload,
        corpus_sha256=corpus.source_sha256,
        evidence_graph=corpus.graph,
        adapter_issues=[issue.model_dump(mode="json") for issue in corpus.adapter_issues],
        synthesis=synthesis,
        counterfactual_reruns=counterfactuals,
        audit_candidates=candidate_payload,
        release_assessment=assessment,
        lineage=lineage,
    )


def build_offline_fixture() -> tuple[ClaimManifest, CorpusLoadResult]:
    """Construct a deterministic three-study fixture with no provider dependencies."""

    graphs: list[EvidenceGraph] = []
    for index, estimate in enumerate((0.45, 0.62, 0.51), start=1):
        suffix = str(index)
        context = GraphAdapterContext(
            publication=PublicationIdentity(
                publication_id=f"fixture-publication-{suffix}",
                paper_id=f"fixture-paper-{suffix}",
                doc_id=f"fixture-document-{suffix}",
                title=f"Offline verifier fixture study {suffix}",
            ),
            study_id=f"fixture-study-{suffix}",
            cohort_identity=CohortIdentity(
                cohort_id=f"fixture-cohort-{suffix}",
                basis=CohortIdentityBasis.REVIEWER_RECONCILED,
                rationale="Deterministic fixture identity; not an empirical study.",
            ),
            treatment_arm_id=f"fixture-treatment-{suffix}",
            comparator_arm_id=f"fixture-control-{suffix}",
            contrast_id=f"fixture-contrast-{suffix}",
            contrast_label="intervention_vs_control",
            positive_direction_means="higher fixture outcome under intervention",
            treatment_label="fixture intervention",
            comparator_label="fixture control",
            timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week"),
        )
        effect = EffectEvidence(
            paper_id=f"fixture-paper-{suffix}",
            finding_id=f"fixture-finding-{suffix}",
            outcome="fixture_outcome",
            contrast="intervention_vs_control",
            effect_format="hedges_g",
            availability="available",
            estimate=estimate,
            standard_error=0.08,
            reported_significance="significant",
            provenance={
                "source_locator": f"fixture-paper-{suffix}#page=1",
                "source_quote": f"The planted standardized estimate was {estimate}.",
            },
        )
        graphs.append(adapt_effect_evidence(effect, context=context).graph)
    graph = _merge_graphs(graphs)
    manifest = ClaimManifest(
        question_id="offline-verifier-fixture",
        population_id="offline-fixture-population",
        domain="synthetic",
        claim=ScientificClaim(
            statement="The fixture intervention increases the fixture outcome.",
            direction=TargetDirection.INCREASE,
            outcome_name="fixture_outcome",
        ),
        protocol=VerificationProtocol(
            corpus_cutoff="deterministic-offline-fixture-v1",
            inclusion_criteria=["all three generated fixture studies"],
            exclusion_criteria=[],
        ),
    )
    source_hash = hash_canonical(
        {"fixture": "unified-verifier-v1", "graph": graph.model_dump(mode="json")}
    )
    corpus = CorpusLoadResult(
        corpus_id="offline-verifier-fixture-v1",
        source_label="embedded:offline-verifier-fixture-v1",
        source_format="embedded_typed_evidence_graph",
        source_sha256=source_hash,
        graph=graph,
        eligibility=_graph_eligibility(graph),
        adapter_issues=(),
        metadata={"empirical_evidence": False, "purpose": "offline_integration_test"},
    )
    return manifest, corpus


__all__ = [
    "AuditGuardConfig",
    "AuditPolicyConfig",
    "ClaimManifest",
    "CorpusAdapterIssue",
    "CorpusEligibilityRecord",
    "CorpusLoadResult",
    "LegacyAdapterConfig",
    "ScientificClaim",
    "VerificationContractError",
    "VerificationProtocol",
    "adapt_legacy_findings",
    "build_graph_counterfactual_audits",
    "build_offline_fixture",
    "load_claim_manifest",
    "load_corpus",
    "run_verification",
]
