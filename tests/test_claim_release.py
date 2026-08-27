from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime

import pytest
import scripts.assess_claim_release as cli
from pydantic import ValidationError

from literature_multiverse.budgeted_verification import (
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ScenarioKind,
)
from literature_multiverse.calibration import (
    FrozenCalibrationBundle,
    RiskExample,
    freeze_calibration_bundle,
)
from literature_multiverse.claim_release import (
    CLAIM_RELEASE_RISK_FEATURE_NAMES,
    AuditResolutionReceipt,
    ClaimReleaseConfig,
    ClaimReleaseContractError,
    ClaimTarget,
    assess_claim_release,
    evidence_item_sha256s,
    freeze_audit_resolution_receipt,
)
from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

PIPELINE_SHA256 = "a" * 64


def _context(suffix: str) -> GraphAdapterContext:
    return GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=f"paper-{suffix}",
            doc_id=f"document-{suffix}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=f"cohort-{suffix}",
            basis="reviewer_reconciled",
            rationale="Registry and recruitment details reconciled before synthesis.",
        ),
        treatment_arm_id=f"arm-{suffix}-treatment",
        comparator_arm_id=f"arm-{suffix}-control",
        contrast_id=f"contrast-{suffix}",
        contrast_label="intervention_vs_control",
        positive_direction_means="higher outcome under intervention",
        treatment_label="intervention",
        comparator_label="control",
        timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week"),
    )


def _evidence(
    suffix: str,
    *,
    estimate: float | None = 0.8,
    standard_error: float | None = 0.05,
) -> EffectEvidence:
    available = estimate is not None
    return EffectEvidence(
        paper_id=f"paper-{suffix}",
        finding_id=f"finding-{suffix}",
        outcome="performance",
        contrast="intervention_vs_control",
        effect_format="hedges_g" if available else "unspecified",
        availability="available" if available else "inconclusive",
        estimate=estimate,
        standard_error=standard_error if available else None,
        reported_significance="not_significant",
        provenance={
            "source_locator": f"paper-{suffix}.pdf#page=4",
            "source_quote": (
                f"The standardized estimate was {estimate}."
                if available
                else "No detectable difference was reported."
            ),
        },
    )


def _graph(
    suffixes: Sequence[str] = ("a", "b", "c"),
    *,
    estimate: float | None = 0.8,
    standard_error: float | None = 0.05,
) -> EvidenceGraph:
    graphs = [
        adapt_effect_evidence(
            _evidence(suffix, estimate=estimate, standard_error=standard_error),
            context=_context(suffix),
        ).graph
        for suffix in suffixes
    ]
    return EvidenceGraph(
        publications=[item for graph in graphs for item in graph.publications],
        studies=[item for graph in graphs for item in graph.studies],
        cohorts=[item for graph in graphs for item in graph.cohorts],
        arms=[item for graph in graphs for item in graph.arms],
        contrasts=[item for graph in graphs for item in graph.contrasts],
        outcome_estimates=[item for graph in graphs for item in graph.outcome_estimates],
        evidence_spans=[item for graph in graphs for item in graph.evidence_spans],
    )


def _audit_candidates(
    graph: EvidenceGraph,
    *,
    basis: ProbabilityBasis = ProbabilityBasis.HEURISTIC,
    baseline: float = 0.3,
) -> list[AuditCandidate]:
    return [
        AuditCandidate(
            item_id=estimate.estimate_id,
            baseline_contribution=baseline,
            counterfactual_contribution=0.0,
            error_probability=0.2,
            probability_basis=basis,
            probability_source="prospective-error-model-v1",
            verification_cost=1.0,
            cost_unit="minutes",
            disagreement_score=0.1,
            scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
            scenario_source="prespecified-leave-one-out-rerun",
        )
        for estimate in graph.outcome_estimates
    ]


def _feature_row(value: float) -> dict[str, float]:
    return {name: value for name in CLAIM_RELEASE_RISK_FEATURE_NAMES}


def _bundle(
    *, threshold: float = 1.0, label_source: str = "benchmark_annotation"
) -> FrozenCalibrationBundle:
    rows: list[RiskExample] = []
    for index in range(8):
        unsupported = index >= 4
        rows.append(
            RiskExample(
                question_id=f"development-{index}",
                split="development",
                population_id="prospective-population-v1",
                domain="biomedicine",
                pipeline_sha256=PIPELINE_SHA256,
                paper_ids=[f"frozen-development-paper-{index}"],
                features=_feature_row(1.0 if unsupported else 0.0),
                unsupported_claim=unsupported,
                label_source=label_source,
            )
        )
    for index in range(4):
        rows.append(
            RiskExample(
                question_id=f"calibration-{index}",
                split="calibration",
                population_id="prospective-population-v1",
                domain="biomedicine",
                pipeline_sha256=PIPELINE_SHA256,
                paper_ids=[f"frozen-calibration-paper-{index}"],
                features=_feature_row(0.0),
                unsupported_claim=False,
                label_source=label_source,
            )
        )
    return freeze_calibration_bundle(
        rows,
        alpha=0.99,
        delta=0.5,
        seed=3,
        candidate_thresholds=[threshold],
    )


def _resolution_receipts(
    graph: EvidenceGraph,
    preliminary,
) -> list[AuditResolutionReceipt]:
    evidence_hashes = evidence_item_sha256s(graph)
    return [
        freeze_audit_resolution_receipt(
            item_id=estimate.estimate_id,
            provenance="benchmark_adjudication",
            adjudicator_count=1,
            completed_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
            adjudication_protocol_sha256="b" * 64,
            adjudication_artifact_sha256=hash_canonical(
                {"unit_test_adjudication": estimate.estimate_id}
            ),
            audited_evidence_item_sha256=evidence_hashes[estimate.estimate_id],
            audited_graph_sha256=preliminary.evidence_graph_sha256,
            audited_synthesis_sha256=preliminary.synthesis_sha256,
            current_evidence_item_sha256=evidence_hashes[estimate.estimate_id],
            current_graph_sha256=preliminary.evidence_graph_sha256,
            current_synthesis_sha256=preliminary.synthesis_sha256,
            current_candidate_input_sha256=preliminary.audit.candidate_input_sha256,
        )
        for estimate in sorted(graph.outcome_estimates, key=lambda row: row.estimate_id)
    ]


def _assess(
    graph: EvidenceGraph,
    *,
    candidates: Sequence[AuditCandidate] | None = None,
    receipts: Sequence[AuditResolutionReceipt] | None = None,
    resolve_all: bool = True,
    bundle: FrozenCalibrationBundle | None = None,
    budget: float = 10.0,
    config: ClaimReleaseConfig | None = None,
    claim_model: ClaimModel | None = None,
):
    candidates = list(candidates if candidates is not None else _audit_candidates(graph))
    selected_claim_model = claim_model or ClaimModel(
        intercept=-0.2, claim_id="performance-increase"
    )

    def run(selected_receipts: Sequence[AuditResolutionReceipt]):
        return assess_claim_release(
            graph=graph,
            question_id="prospective-question",
            population_id="prospective-population-v1",
            domain="biomedicine",
            pipeline_sha256=PIPELINE_SHA256,
            target=ClaimTarget(direction="increase", outcome_name="performance"),
            audit_candidates=candidates,
            claim_model=selected_claim_model,
            audit_resolution_receipts=list(selected_receipts),
            audit_budget=budget,
            frozen_calibration_bundle=bundle,
            config=config,
        )

    if receipts is not None:
        return run(receipts)
    preliminary = run([])
    if not resolve_all or not candidates:
        return preliminary
    return run(_resolution_receipts(graph, preliminary))


def test_releases_only_when_evidence_audit_and_frozen_risk_gates_pass() -> None:
    graph = _graph()
    result = _assess(graph, bundle=_bundle())

    assert result.status == "released"
    assert result.reasons == []
    assert result.evidence.classification == "supported"
    assert result.evidence.ci_lower > 0
    assert result.evidence.prediction_interval_lower > 0
    assert result.audit.status == "eligible"
    assert result.audit.resolved_item_ids == result.audit.expected_item_ids
    assert len(result.audit.resolution_receipts) == len(result.audit.expected_item_ids)
    assert result.audit.resolution_ledger_sha256 is not None
    assert result.audit.expected_item_ids == sorted(
        estimate.estimate_id for estimate in graph.outcome_estimates
    )
    assert result.paper_ids == ["paper-a", "paper-b", "paper-c"]
    assert result.calibration.status == "released"
    assert result.calibration.label_source == "benchmark_annotation"
    assert "scientific truth" in result.release_semantics
    assert len(result.decision_sha256) == 64


def test_selected_but_unresolved_influential_candidate_blocks_release() -> None:
    graph = _graph()
    candidates = _audit_candidates(
        graph,
        basis=ProbabilityBasis.CALIBRATED,
        baseline=0.3,
    )
    result = _assess(
        graph,
        candidates=candidates,
        resolve_all=False,
        bundle=_bundle(),
        budget=10,
    )

    assert result.audit.selected_item_ids == sorted(candidate.item_id for candidate in candidates)
    assert result.audit.resolved_item_ids == []
    assert result.audit.status == "blocked"
    assert result.audit.unresolved_high_influence_item_ids
    assert result.status == "abstained"


def test_heuristic_unresolved_candidate_blocks_even_when_influence_is_small() -> None:
    graph = _graph()
    candidates = _audit_candidates(graph, basis=ProbabilityBasis.HEURISTIC, baseline=0.001)
    result = _assess(
        graph,
        candidates=candidates,
        resolve_all=False,
        bundle=_bundle(),
        claim_model=ClaimModel(intercept=0.2, claim_id="performance-increase"),
    )

    assert result.audit.unresolved_noncalibrated_item_ids == sorted(
        candidate.item_id for candidate in candidates
    )
    assert "unresolved_error_probabilities_not_calibrated" in result.audit.reasons
    assert result.status == "abstained"


def test_zero_or_understated_influence_cannot_release_unresolved_estimates() -> None:
    graph = _graph()
    candidates = [
        replace(
            candidate,
            baseline_contribution=0.0,
            counterfactual_contribution=0.0,
            probability_basis=ProbabilityBasis.CALIBRATED,
        )
        for candidate in _audit_candidates(graph)
    ]
    result = _assess(
        graph,
        candidates=candidates,
        resolve_all=False,
        bundle=_bundle(),
        claim_model=ClaimModel(intercept=0.2, claim_id="performance-increase"),
    )

    assert all(row.probability_influence == 0 for row in result.audit.ranking)
    assert result.audit.unresolved_item_ids == result.audit.expected_item_ids
    assert result.audit.status == "blocked"
    assert "all_matching_estimates_require_completed_resolution_receipts" in (
        result.audit.reasons
    )
    assert result.status == "abstained"


def test_supported_synthesis_rejects_inconsistent_claim_model_baseline() -> None:
    with pytest.raises(ClaimReleaseContractError, match="baseline_conclusion_inconsistent"):
        _assess(
            _graph(),
            resolve_all=False,
            bundle=None,
            claim_model=ClaimModel(intercept=-100.0, claim_id="misaligned-model"),
        )


def test_simulation_calibration_bundle_cannot_authorize_scientific_release() -> None:
    result = _assess(_graph(), bundle=_bundle(label_source="simulation"))

    assert result.evidence.classification == "supported"
    assert result.audit.status == "eligible"
    assert result.calibration.status == "abstained"
    assert result.calibration.label_source == "simulation"
    assert result.calibration.reason == (
        "simulation_calibration_not_valid_for_scientific_release"
    )
    assert result.status == "abstained"


def test_frozen_risk_policy_can_abstain_after_other_gates_pass() -> None:
    graph = _graph()
    result = _assess(graph, bundle=_bundle(threshold=0.2))

    assert result.evidence.classification == "supported"
    assert result.audit.status == "eligible"
    assert result.calibration.status == "abstained"
    assert result.calibration.reason == "risk_above_threshold"
    assert result.status == "abstained"


def test_absent_frozen_calibration_bundle_always_abstains() -> None:
    result = _assess(_graph(), bundle=None)

    assert result.calibration.status == "not_run"
    assert result.calibration.reason == "frozen_calibration_bundle_absent"
    assert result.status == "abstained"


def test_missing_or_extra_audit_identity_raises_contract_error() -> None:
    graph = _graph()
    candidates = _audit_candidates(graph)

    with pytest.raises(ClaimReleaseContractError, match="identity_coverage_mismatch"):
        _assess(graph, candidates=candidates[:-1], resolve_all=False, bundle=None)

    extra = replace(candidates[0], item_id="not-in-evidence-graph")
    with pytest.raises(ClaimReleaseContractError, match="identity_coverage_mismatch"):
        _assess(
            graph,
            candidates=[*candidates, extra],
            resolve_all=False,
            bundle=None,
        )


def test_id_only_resolution_is_not_a_valid_completed_receipt() -> None:
    with pytest.raises(ValidationError):
        AuditResolutionReceipt.model_validate(
            {
                "item_id": "estimate-finding-a",
                "receipt_sha256": "a" * 64,
            }
        )


def test_resolution_receipt_must_bind_the_current_candidate_snapshot() -> None:
    graph = _graph()
    preliminary = _assess(graph, resolve_all=False, bundle=_bundle())
    receipts = _resolution_receipts(graph, preliminary)
    payload = receipts[0].model_dump(mode="json", exclude={"receipt_sha256"})
    payload["current_candidate_input_sha256"] = "f" * 64
    bad = AuditResolutionReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )

    with pytest.raises(ClaimReleaseContractError, match="current_candidate_mismatch"):
        _assess(
            graph,
            receipts=[bad, *receipts[1:]],
            bundle=_bundle(),
        )


def test_changed_resolution_snapshot_requires_correction_lineage() -> None:
    graph = _graph()
    preliminary = _assess(graph, resolve_all=False, bundle=None)
    receipt = _resolution_receipts(graph, preliminary)[0]
    payload = receipt.model_dump(mode="json", exclude={"receipt_sha256"})
    payload["audited_graph_sha256"] = "e" * 64
    payload["receipt_sha256"] = hash_canonical(payload)

    with pytest.raises(ValidationError, match="correction_lineage_mismatch"):
        AuditResolutionReceipt.model_validate(payload)


def test_blinded_human_receipt_requires_two_adjudicators() -> None:
    graph = _graph()
    preliminary = _assess(graph, resolve_all=False, bundle=None)
    evidence_hash = evidence_item_sha256s(graph)["estimate-finding-a"]

    with pytest.raises(ValidationError, match="requires_two_adjudicators"):
        freeze_audit_resolution_receipt(
            item_id="estimate-finding-a",
            provenance="blinded_human",
            adjudicator_count=1,
            completed_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
            adjudication_protocol_sha256="b" * 64,
            adjudication_artifact_sha256="c" * 64,
            audited_evidence_item_sha256=evidence_hash,
            audited_graph_sha256=preliminary.evidence_graph_sha256,
            audited_synthesis_sha256=preliminary.synthesis_sha256,
            current_evidence_item_sha256=evidence_hash,
            current_graph_sha256=preliminary.evidence_graph_sha256,
            current_synthesis_sha256=preliminary.synthesis_sha256,
            current_candidate_input_sha256=preliminary.audit.candidate_input_sha256,
        )


def test_graph_insufficiency_is_not_evaluable_instead_of_manufacturing_claim() -> None:
    graph = _graph(("a",))
    result = _assess(graph, bundle=None)

    assert result.evidence.classification == "not_evaluable"
    assert result.evidence.reason == "fewer_than_two_nonzero_paper_signs"
    assert result.status == "abstained"


def test_no_effect_target_is_forbidden_and_zero_or_nonestimable_evidence_abstains() -> None:
    with pytest.raises(ValidationError):
        ClaimTarget(direction="no_effect", outcome_name="performance")

    exact_zero = _assess(_graph(estimate=0.0), bundle=None)
    assert exact_zero.evidence.classification == "inconclusive"
    assert exact_zero.evidence.reason == "confidence_interval_includes_null"

    nonestimable = _assess(_graph(estimate=None), bundle=None)
    assert nonestimable.evidence.classification == "not_evaluable"
    assert nonestimable.status == "abstained"


def test_prediction_interval_requirement_can_block_ci_only_support() -> None:
    graph = _graph(("a", "b"))
    default = _assess(graph, bundle=None)
    relaxed = _assess(
        graph,
        bundle=None,
        config=ClaimReleaseConfig(require_prediction_interval_stability=False),
    )

    assert default.evidence.classification == "inconclusive"
    assert default.evidence.reason == "prediction_interval_required_but_unavailable"
    assert relaxed.evidence.classification == "supported"


def test_directional_fallback_uses_exact_sign_interval_and_cannot_fake_precision() -> None:
    graph = _graph(("a", "b", "c", "d", "e", "f"), standard_error=None)

    strict = _assess(graph, bundle=None)
    relaxed = _assess(
        graph,
        bundle=None,
        config=ClaimReleaseConfig(require_prediction_interval_stability=False),
    )

    assert strict.evidence.mode == "directional_sign_synthesis"
    assert strict.evidence.directional_ci_lower > 0.5
    assert strict.evidence.classification == "inconclusive"
    assert strict.evidence.reason == (
        "directional_fallback_cannot_satisfy_prediction_interval_requirement"
    )
    assert relaxed.evidence.classification == "supported"


def test_calibration_feature_schema_mismatch_fails_closed() -> None:
    graph = _graph()
    bundle = _bundle()
    payload = bundle.model_dump(mode="json")
    payload["feature_names"] = payload["feature_names"][:-1]

    # A corrupted bundle cannot pass its own lineage validator.
    with pytest.raises(ValidationError):
        FrozenCalibrationBundle.model_validate(payload)

    # A valid bundle trained for a different feature schema is rejected explicitly.
    rows = [
        RiskExample(
            question_id=f"{split}-{index}",
            split=split,
            population_id="prospective-population-v1",
            domain="biomedicine",
            pipeline_sha256=PIPELINE_SHA256,
            paper_ids=[f"different-schema-paper-{split}-{index}"],
            features={"different_feature": float(index)},
            unsupported_claim=(split == "development" and index >= 2),
            label_source="simulation",
        )
        for split, width in (("development", 4), ("calibration", 2))
        for index in range(width)
    ]
    different_bundle = freeze_calibration_bundle(
        rows,
        alpha=0.99,
        delta=0.5,
        candidate_thresholds=[1.0],
    )
    with pytest.raises(ClaimReleaseContractError, match="feature_schema_mismatch"):
        _assess(graph, bundle=different_bundle)


def test_closed_request_cli_writes_the_same_hash_bound_release(tmp_path, capsys) -> None:
    graph = _graph()
    candidates = _audit_candidates(graph)
    bundle = _bundle()
    prepared = _assess(graph, candidates=candidates, bundle=bundle)
    request = {
        "graph": graph,
        "question_id": "prospective-question",
        "population_id": "prospective-population-v1",
        "domain": "biomedicine",
        "pipeline_sha256": PIPELINE_SHA256,
        "target": {"direction": "increase", "outcome_name": "performance"},
        "audit_candidates": [asdict(candidate) for candidate in candidates],
        "claim_model": asdict(
            ClaimModel(intercept=-0.2, claim_id="performance-increase")
        ),
        "audit_resolution_receipts": [
            receipt.model_dump(mode="json")
            for receipt in prepared.audit.resolution_receipts
        ],
        "audit_budget": 10.0,
        "frozen_calibration_bundle": bundle,
    }
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "assessment.json"
    atomic_write_json(input_path, request)

    assert (
        cli.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assessment = json.loads(output_path.read_text())
    serialized_request = json.loads(input_path.read_text())
    assert summary["status"] == "released"
    assert summary["decision_sha256"] == assessment["decision_sha256"]
    assert "unsupported_claim" not in serialized_request
    assert "audit_oracle" not in serialized_request

    legacy_request = dict(request)
    legacy_request.pop("audit_resolution_receipts")
    legacy_request["resolved_audit_item_ids"] = [
        candidate.item_id for candidate in candidates
    ]
    legacy_path = tmp_path / "legacy-id-only-request.json"
    atomic_write_json(legacy_path, legacy_request)
    with pytest.raises(ValueError, match="claim_release_request_keys_invalid"):
        cli.main(
            [
                "--input",
                str(legacy_path),
                "--output",
                str(tmp_path / "legacy-assessment.json"),
            ]
        )
