from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from scripts.run_metasyn_grounded_analysis_v2 import _reviewers

import literature_multiverse.metasyn_grounded_analysis_v2 as analysis_module
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    freeze_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_grounded_analysis_v2 import (
    ANALYSIS_CLI_PATH,
    ANALYSIS_MODULE_PATH,
    MetaSynGroundedAnalysisV2,
    freeze_metasyn_grounded_analysis_v2,
    freeze_metasyn_grounded_question_analysis_v2,
    validate_metasyn_grounded_analysis_v2,
)
from literature_multiverse.metasyn_grounded_publication_bridge_v2 import (
    MetaSynGroundedPublicationCorpusBridgeV2,
    MetaSynGroundedQuestionCorpusV2,
    freeze_metasyn_grounded_publication_bridge_v2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    freeze_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    freeze_native_publication_extraction,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    SourceDocumentArtifact,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_HASH = "a" * 64
GROUNDING_HASH = "b" * 64


def _cohort_payload(
    key: str,
    *,
    registry_ids: list[str],
    estimate: float,
    standard_error: float | None = 0.1,
) -> dict[str, object]:
    effect: dict[str, object] = {
        "effect_format": "hedges_g",
        "estimate": estimate,
    }
    if standard_error is not None:
        effect["standard_error"] = standard_error
    return {
        "key": key,
        "source_labels": [f"Reported cohort {key}"],
        "registry_ids": registry_ids,
        "dataset_ids": [],
        "total_sample_size": 100,
        "arms": [
            {"key": "tx", "label": "Treatment", "role": "intervention"},
            {"key": "control", "label": "Control", "role": "control"},
        ],
        "contrasts": [
            {
                "key": "primary",
                "treatment_arm_key": "tx",
                "comparator_arm_key": "control",
                "label": "treatment_vs_control",
                "positive_direction_means": "higher values favor treatment",
            }
        ],
        "findings": [
            {
                "key": "finding",
                "contrast_key": "primary",
                "outcome_name": "outcome",
                "timepoint": {"kind": "exact", "value": 4, "unit": "week"},
                "effect": effect,
                "evidence": {
                    "source_locator": f"source:{key}",
                    "quote": f"Effect for {key} was {estimate}.",
                    "line_ids": ["L1"],
                },
            }
        ],
    }


def _fragment(
    publication_number: int,
    *,
    cohorts: list[dict[str, object]],
    study_registration_ids: list[str] | None = None,
):
    publication_id = f"publication-{publication_number}"
    extraction = NativePublicationExtraction.model_validate(
        {
            "status": "estimable",
            "studies": [
                {
                    "key": "study",
                    "source_label": f"Study report {publication_number}",
                    "registration_ids": study_registration_ids or [],
                    "cohorts": cohorts,
                }
            ],
        }
    )
    publication = PublicationIdentity(
        publication_id=publication_id,
        paper_id=f"paper-{publication_number}",
        doc_id=f"doc-{publication_number}",
    )
    source = SourceDocumentArtifact(
        artifact_path=f"data/source-{publication_number}.json",
        sha256=f"{publication_number:x}" * 64,
        media_type="application/json",
        source_locator=f"json:data/source-{publication_number}.json#/doc-{publication_number}",
    )
    return freeze_native_publication_extraction(
        payload=extraction,
        question_id="analysis-question",
        publication=publication,
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=GROUNDING_HASH,
    )


def _question_corpus(corpus: TypedEvidenceCorpus) -> MetaSynGroundedQuestionCorpusV2:
    payload = {
        "question_corpus_version": "metasyn-grounded-question-corpus-v2",
        "question_id": corpus.question_id,
        "publication_join_sha256s": [
            hash_canonical({"publication_id": item.publication_id}) for item in corpus.fragments
        ],
        "publication_ids": sorted(item.publication_id for item in corpus.fragments),
        "compatibility_corpus": corpus,
        "compatibility_corpus_sha256": corpus.corpus_sha256,
        "estimable_publication_count": len(corpus.estimable_publication_ids),
        "non_estimable_publication_count": len(corpus.non_estimable_publication_ids),
        "quantitative_effect_count": len(corpus.graph.outcome_estimates),
        "coverage_blockers": [
            "equivalence_conclusion_not_extracted",
            "moderators_not_extracted",
            "reported_significance_conclusion_not_extracted",
        ],
        "exact_projection_authority": True,
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": bool(corpus.graph.outcome_estimates),
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedQuestionCorpusV2.model_validate(
        {**payload, "question_corpus_sha256": hash_canonical(payload)}
    )


def _two_cohort_single_publication_question(
    *, standard_error: float | None = 0.1
) -> MetaSynGroundedQuestionCorpusV2:
    corpus = assemble_typed_evidence_corpus(
        [
            _fragment(
                1,
                cohorts=[
                    _cohort_payload(
                        "cohort-a",
                        registry_ids=["NCT-SINGLE-STUDY"],
                        estimate=0.2,
                        standard_error=standard_error,
                    ),
                    _cohort_payload(
                        "cohort-b",
                        registry_ids=["NCT-SINGLE-STUDY"],
                        estimate=0.4,
                        standard_error=standard_error,
                    ),
                ],
                study_registration_ids=["NCT-SINGLE-STUDY"],
            )
        ]
    )
    return _question_corpus(corpus)


def test_actual_two_cohort_kernel_runs_without_scientific_authority() -> None:
    question = _two_cohort_single_publication_question()

    result = freeze_metasyn_grounded_question_analysis_v2(question_corpus=question)

    assert result.reconciliation_status == "single_publication_complete"
    assert result.cohort_independence_resolved is True
    assert result.mechanics_status == "kernel_completed"
    assert result.kernel_completed_unit_count == 1
    unit = result.units[0]
    assert unit.status == "quantitative_kernel_completed"
    assert unit.kernel_result["mode"] == "random_effects_meta_analysis"
    assert unit.kernel_result["quantitative"]["status"] == "ok"
    assert unit.kernel_result["quantitative"]["n_cohorts"] == 2
    assert unit.kernel_result["condition_analysis"] is None
    assert unit.prespecified_moderators == []
    assert unit.reported_significance_consumed is False
    assert unit.equivalence_consumed is False
    assert unit.synthesis_input_authority is False
    assert unit.scientific_synthesis_authority is False
    assert unit.claim_release_authority is False


def test_multi_publication_without_reviewer_abstains_before_kernel() -> None:
    corpus = assemble_typed_evidence_corpus(
        [
            _fragment(
                1,
                cohorts=[_cohort_payload("cohort-a", registry_ids=["NCT-ONE"], estimate=0.2)],
                study_registration_ids=["NCT-ONE"],
            ),
            _fragment(
                2,
                cohorts=[_cohort_payload("cohort-b", registry_ids=["NCT-TWO"], estimate=0.4)],
                study_registration_ids=["NCT-TWO"],
            ),
        ]
    )

    result = freeze_metasyn_grounded_question_analysis_v2(question_corpus=_question_corpus(corpus))

    assert result.reconciliation_status == "strong_identifier_reconciled_limited"
    assert result.cohort_independence_resolved is False
    assert result.mechanics_status == "cohort_independence_unresolved"
    assert result.kernel_invoked_unit_count == 0
    assert result.kernel_completed_unit_count == 0
    assert result.units[0].kernel_result is None
    assert result.units[0].abstention_reasons == [
        "cohort_independence_not_resolved:strong_identifier_reconciled_limited",
        "reviewer_artifact_absent",
    ]


def test_directional_fallback_is_retained_but_never_counts_as_quantitative() -> None:
    question = _two_cohort_single_publication_question(standard_error=None)

    result = freeze_metasyn_grounded_question_analysis_v2(question_corpus=question)

    assert result.cohort_independence_resolved is True
    assert result.mechanics_status == "kernel_abstained"
    assert result.kernel_invoked_unit_count == 1
    assert result.kernel_completed_unit_count == 0
    unit = result.units[0]
    assert unit.status == "abstained"
    assert unit.kernel_result["mode"] == "directional_sign_synthesis"
    assert "directional_fallback_non_authorizing" in unit.abstention_reasons
    assert "quantitative_kernel:no_estimable_effects" in unit.abstention_reasons


def test_empty_effect_question_preserves_publication_and_reason() -> None:
    publication = PublicationIdentity(
        publication_id="publication-missing", paper_id="paper-missing"
    )
    source = SourceDocumentArtifact(
        artifact_path="data/source-missing.json",
        sha256="c" * 64,
        media_type="application/json",
        source_locator="json:data/source-missing.json",
    )
    fragment = freeze_publication_evidence_fragment(
        question_id="analysis-question",
        publication_id=publication.publication_id,
        paper_id=publication.paper_id,
        publication=publication,
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.NO_TARGET_OUTCOME,
    )
    question = _question_corpus(assemble_typed_evidence_corpus([fragment]))

    result = freeze_metasyn_grounded_question_analysis_v2(question_corpus=question)

    assert result.publication_count == 1
    assert result.quantitative_effect_count == 0
    assert result.mechanics_status == "no_quantitative_units"
    assert result.mechanics_blockers == ["no_quantitative_effects"]
    assert result.units == []


@pytest.fixture(scope="module")
def execution_bundle() -> MetaSynPassageHostedExecutionBundleV2:
    return freeze_metasyn_passage_hosted_execution_bundle_v2(repository_root=ROOT)


@pytest.fixture(scope="module")
def zero_yield_bridge(
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
) -> MetaSynGroundedPublicationCorpusBridgeV2:
    inventories: dict[str, MetaSynCandidateInventoryReceiptV2] = {}
    for row in execution_bundle.extraction_inputs.rows:
        inventories[row.row_key] = freeze_metasyn_candidate_inventory_receipt_v2(
            row_context_sha256=row.upstream_row_context_sha256,
            projection_v2_sha256=row.projection_v2_sha256,
            allowed_outcome_text_by_id=(row.question_surface.allowed_outcome_text_by_id),
            passage_text_by_id={
                passage.passage_id: passage.text for passage in row.projection_surface.passages
            },
            value={
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            },
        )
    return freeze_metasyn_grounded_publication_bridge_v2(
        execution_bundle=execution_bundle,
        inventory_receipts_by_row=inventories,
        candidate_terminals_by_row={
            row.row_key: [] for row in execution_bundle.extraction_inputs.rows
        },
        repository_root=ROOT,
    )


def test_real_bridge_join_preserves_all_ten_questions_and_32_publications(
    zero_yield_bridge: MetaSynGroundedPublicationCorpusBridgeV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def validated_bridge(**kwargs: Any) -> MetaSynGroundedPublicationCorpusBridgeV2:
        calls.append(kwargs["external_replay"])
        assert kwargs["bridge"] == zero_yield_bridge
        return zero_yield_bridge

    monkeypatch.setattr(
        analysis_module,
        "validate_metasyn_grounded_publication_bridge_v2",
        validated_bridge,
    )
    analysis = freeze_metasyn_grounded_analysis_v2(
        bridge=zero_yield_bridge,
        repository_root=ROOT,
    )

    assert calls == [True]
    assert analysis.question_count == len(analysis.question_analyses) == 10
    assert analysis.publication_count == 32
    assert sum(item.publication_count for item in analysis.question_analyses) == 32
    assert analysis.quantitative_effect_count == 0
    assert analysis.quantitative_unit_count == 0
    assert analysis.kernel_completed_unit_count == 0
    assert all(
        item.mechanics_status == "no_quantitative_units"
        and item.mechanics_blockers == ["no_quantitative_effects"]
        for item in analysis.question_analyses
    )
    component = analysis.analysis_pipeline_fingerprint.components[0]
    paths = {item.path for item in component.files}
    assert {ANALYSIS_MODULE_PATH, ANALYSIS_CLI_PATH}.issubset(paths)
    assert component.settings["yield_and_mechanics_only"] is True
    assert analysis.synthesis_input_authority is False
    assert analysis.scientific_synthesis_authority is False
    assert analysis.accuracy_claim_authority is False
    assert analysis.claim_release_authority is False
    assert (
        validate_metasyn_grounded_analysis_v2(
            analysis=analysis,
            repository_root=ROOT,
            external_replay=False,
        )
        == analysis
    )

    tampered = analysis.model_dump(mode="json")
    tampered["claim_release_authority"] = True
    tampered["analysis_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "analysis_sha256"}
    )
    with pytest.raises(ValidationError):
        MetaSynGroundedAnalysisV2.model_validate(tampered)


def test_cli_reviewer_map_shape_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "reviewers.json"
    path.write_text(json.dumps({"artifacts_by_question": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewer_map_keys_invalid"):
        _reviewers(path)
