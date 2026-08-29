from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from scipy.stats import beta

from literature_multiverse.item_risk_calibration import (
    ItemRiskCalibrationError,
    ItemRiskCandidate,
    calibrate_item_risk_bounds,
    make_fixed_risk_bin_family,
    score_item_risk_bound,
    seal_item_risk_calibration_unit,
    seal_item_risk_candidate,
    seal_shift_assessment,
    validate_item_risk_calibration_bundle_integrity,
    verified_audit_cell_rate_ucl_fields,
    verified_audit_probability_fields,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    compute_pipeline_fingerprint,
    verify_pipeline_fingerprint,
)

SCORE_MODEL_SHA256 = "b" * 64
ADJUDICATION_PROTOCOL_SHA256 = "c" * 64
SAMPLING_PROTOCOL_SHA256 = "d" * 64
SHIFT_DETECTOR_SHA256 = "e" * 64
DEFINITION_ARTIFACT_SHA256 = "f" * 64


def _unit(
    index: int,
    *,
    split: str,
    score: float,
    error: bool,
    pipeline_sha256: str,
    domain: str = "cardiology",
):
    return seal_item_risk_calibration_unit(
        split=split,
        item_id=f"item-{index}",
        question_id=f"question-{index}",
        paper_id=f"paper-{index}",
        population_id="biomed-v1",
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        score_model_sha256=SCORE_MODEL_SHA256,
        score_input_sha256=f"{index:x}".rjust(64, "0"),
        risk_score=score,
        observed_error=error,
        label_source="expert_adjudication",
        adjudication_protocol_sha256=ADJUDICATION_PROTOCOL_SHA256,
        adjudication_artifact_sha256=f"{index + 100:x}".rjust(64, "0"),
    )


def _pipeline_verification(tmp_path: Path):
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text("FROZEN = True\n", encoding="utf-8")
    fingerprint = compute_pipeline_fingerprint(
        root=tmp_path,
        components=[
            PipelineComponentSpec(
                component_id="verifier",
                component_version="1",
                file_paths=["pipeline.py"],
                settings={"mode": "test"},
            )
        ],
    )
    return verify_pipeline_fingerprint(expected=fingerprint, root=tmp_path)


def _bundle(tmp_path: Path):
    verification = _pipeline_verification(tmp_path)
    family = make_fixed_risk_bin_family(
        edges=[0.0, 0.5, 1.0],
        score_name="prospective_extraction_error_score",
        score_model_sha256=SCORE_MODEL_SHA256,
        definition_source="prespecified",
        definition_artifact_sha256=DEFINITION_ARTIFACT_SHA256,
    )
    units = [
        _unit(
            1,
            split="development",
            score=0.1,
            error=False,
            pipeline_sha256=verification.expected_pipeline_sha256,
        ),
        _unit(
            2,
            split="development",
            score=0.8,
            error=True,
            pipeline_sha256=verification.expected_pipeline_sha256,
        ),
        _unit(
            3,
            split="calibration",
            score=0.1,
            error=False,
            pipeline_sha256=verification.expected_pipeline_sha256,
        ),
        _unit(
            4,
            split="calibration",
            score=0.2,
            error=False,
            pipeline_sha256=verification.expected_pipeline_sha256,
        ),
        _unit(
            5,
            split="calibration",
            score=0.4,
            error=True,
            pipeline_sha256=verification.expected_pipeline_sha256,
        ),
    ]
    return (
        calibrate_item_risk_bounds(
            units,
            pipeline_verification=verification,
            bin_family=family,
            familywise_delta=0.05,
            sampling_protocol_sha256=SAMPLING_PROTOCOL_SHA256,
            error_event_definition="Any adjudicated material extraction error in the item.",
            shift_detector_id="frozen-domain-monitor-v1",
            shift_detector_sha256=SHIFT_DETECTOR_SHA256,
        ),
        verification,
    )


def _candidate(
    bundle,
    *,
    score: float = 0.2,
    shift_status: str = "no_shift_detected",
    question_id: str = "prospective-question",
    paper_id: str = "prospective-paper",
):
    assessment = seal_shift_assessment(
        bundle=bundle,
        candidate_population_id="biomed-v1",
        candidate_domain="cardiology",
        status=shift_status,
        assessment_input_sha256="1" * 64,
        assessment_artifact_sha256=None if shift_status == "not_assessed" else "2" * 64,
    )
    return seal_item_risk_candidate(
        item_id="prospective-item",
        question_id=question_id,
        paper_id=paper_id,
        population_id="biomed-v1",
        domain="cardiology",
        pipeline_sha256=bundle.pipeline_sha256,
        score_model_sha256=SCORE_MODEL_SHA256,
        score_input_sha256="3" * 64,
        risk_score=score,
        shift_assessment=assessment,
    )


def test_calibration_emits_simultaneous_group_average_cell_rate_ucl(tmp_path: Path) -> None:
    bundle, verification = _bundle(tmp_path)
    candidate = _candidate(bundle)

    bound = score_item_risk_bound(
        candidate=candidate, bundle=bundle, pipeline_verification=verification
    )

    expected = float(beta.ppf(1.0 - 0.025, 2, 2))
    assert bound.status == "cell_rate_ucl_available"
    assert bound.usable_for_scheduling
    assert not bound.usable_for_release
    assert bound.rate_basis == "calibrated_cell_rate_ucl"
    assert bound.estimand == "group_average_item_error_rate_within_domain_score_bin"
    assert bound.cell_calibration_units == 3
    assert bound.cell_observed_errors == 1
    assert bound.familywise_delta == pytest.approx(0.05)
    assert bound.family_cell_count == 2
    assert bound.cellwise_delta == pytest.approx(0.025)
    assert math.isclose(bound.upper_cell_error_rate or 0, expected)
    fields = verified_audit_cell_rate_ucl_fields(
        bound=bound, bundle=bundle, pipeline_verification=verification
    )
    assert fields["item_cell_rate_ucl"] == bound.upper_cell_error_rate
    assert fields["risk_bound_sha256"] == bound.risk_bound_sha256
    with pytest.raises(ItemRiskCalibrationError, match="not_release_probability"):
        verified_audit_probability_fields(
            bound=bound, bundle=bundle, pipeline_verification=verification
        )


def test_raw_score_is_not_promoted_when_calibration_bin_is_empty(tmp_path: Path) -> None:
    bundle, verification = _bundle(tmp_path)
    bound = score_item_risk_bound(
        candidate=_candidate(bundle, score=0.8),
        bundle=bundle,
        pipeline_verification=verification,
    )

    assert bound.status == "empty_calibration_bin"
    assert bound.upper_cell_error_rate is None
    assert bound.rate_basis is None
    assert not bound.usable_for_scheduling
    assert not bound.usable_for_release
    with pytest.raises(ItemRiskCalibrationError, match="not_scheduling_eligible"):
        verified_audit_cell_rate_ucl_fields(
            bound=bound, bundle=bundle, pipeline_verification=verification
        )


def test_shift_must_be_assessed_and_must_not_be_detected(tmp_path: Path) -> None:
    bundle, verification = _bundle(tmp_path)

    unassessed = score_item_risk_bound(
        candidate=_candidate(bundle, shift_status="not_assessed"),
        bundle=bundle,
        pipeline_verification=verification,
    )
    shifted = score_item_risk_bound(
        candidate=_candidate(bundle, shift_status="shift_detected"),
        bundle=bundle,
        pipeline_verification=verification,
    )

    assert unassessed.status == "shift_not_assessed"
    assert shifted.status == "shift_detected"
    assert unassessed.upper_cell_error_rate is None
    assert shifted.upper_cell_error_rate is None


def test_pipeline_mismatch_fails_closed_without_bound(tmp_path: Path) -> None:
    bundle, verification = _bundle(tmp_path)
    valid = _candidate(bundle)
    payload = valid.model_dump(mode="json", exclude={"candidate_sha256"})
    payload["pipeline_sha256"] = "9" * 64
    payload["shift_assessment"] = None
    mismatched = seal_item_risk_candidate(
        item_id=payload["item_id"],
        question_id=payload["question_id"],
        paper_id=payload["paper_id"],
        population_id=payload["population_id"],
        domain=payload["domain"],
        pipeline_sha256=payload["pipeline_sha256"],
        score_model_sha256=payload["score_model_sha256"],
        score_input_sha256=payload["score_input_sha256"],
        risk_score=payload["risk_score"],
        shift_assessment=None,
    )

    bound = score_item_risk_bound(
        candidate=mismatched, bundle=bundle, pipeline_verification=verification
    )

    assert bound.status == "pipeline_mismatch"
    assert bound.upper_cell_error_rate is None


def test_changed_pipeline_verification_cannot_authorize_bound(tmp_path: Path) -> None:
    bundle, matched = _bundle(tmp_path)
    assert matched.computed is not None
    (tmp_path / "pipeline.py").write_text("FROZEN = False\n", encoding="utf-8")
    changed = verify_pipeline_fingerprint(expected=matched.computed, root=tmp_path)

    bound = score_item_risk_bound(
        candidate=_candidate(bundle),
        bundle=bundle,
        pipeline_verification=changed,
    )

    assert changed.status == "mismatch"
    assert bound.status == "pipeline_mismatch"
    assert bound.upper_cell_error_rate is None


@pytest.mark.parametrize(
    ("candidate_kwargs", "expected_status"),
    [
        ({"question_id": "question-3"}, "calibration_question_overlap"),
        ({"paper_id": "paper-3"}, "calibration_paper_overlap"),
        ({"question_id": "question-1"}, "calibration_question_overlap"),
        ({"paper_id": "paper-1"}, "calibration_paper_overlap"),
    ],
)
def test_prospective_candidate_cannot_reuse_development_or_calibration_identity(
    tmp_path: Path,
    candidate_kwargs: dict[str, str],
    expected_status: str,
) -> None:
    bundle, verification = _bundle(tmp_path)

    bound = score_item_risk_bound(
        candidate=_candidate(bundle, **candidate_kwargs),
        bundle=bundle,
        pipeline_verification=verification,
    )

    assert bound.status == expected_status
    assert not bound.usable_for_scheduling
    assert bound.upper_cell_error_rate is None


def test_opaque_raw_score_never_becomes_release_probability(tmp_path: Path) -> None:
    bundle, verification = _bundle(tmp_path)
    low = score_item_risk_bound(
        candidate=_candidate(bundle, score=0.01),
        bundle=bundle,
        pipeline_verification=verification,
    )
    high = score_item_risk_bound(
        candidate=_candidate(bundle, score=0.49),
        bundle=bundle,
        pipeline_verification=verification,
    )

    assert low.status == high.status == "cell_rate_ucl_available"
    assert low.raw_risk_score != high.raw_risk_score
    assert low.score_semantics == high.score_semantics == (
        "externally_supplied_scheduling_score_not_recomputed"
    )
    assert not low.usable_for_release and not high.usable_for_release
    for bound in (low, high):
        with pytest.raises(ItemRiskCalibrationError, match="not_release_probability"):
            verified_audit_probability_fields(
                bound=bound,
                bundle=bundle,
                pipeline_verification=verification,
            )


def test_manifest_probability_is_forbidden_by_prospective_contract(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    payload = _candidate(bundle).model_dump(mode="json")
    payload["error_probability"] = 0.001

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ItemRiskCandidate.model_validate(payload)


def test_calibration_rejects_repeated_question_or_paper_identity(tmp_path: Path) -> None:
    verification = _pipeline_verification(tmp_path)
    family = make_fixed_risk_bin_family(
        edges=[0.0, 1.0],
        score_name="risk",
        score_model_sha256=SCORE_MODEL_SHA256,
        definition_source="prespecified",
        definition_artifact_sha256=DEFINITION_ARTIFACT_SHA256,
    )
    development = _unit(
        10,
        split="development",
        score=0.1,
        error=False,
        pipeline_sha256=verification.expected_pipeline_sha256,
    )
    calibration = _unit(
        11,
        split="calibration",
        score=0.2,
        error=False,
        pipeline_sha256=verification.expected_pipeline_sha256,
    )
    duplicate = seal_item_risk_calibration_unit(
        split="calibration",
        item_id="item-12",
        question_id=development.question_id,
        paper_id="paper-12",
        population_id="biomed-v1",
        domain="cardiology",
        pipeline_sha256=verification.expected_pipeline_sha256,
        score_model_sha256=SCORE_MODEL_SHA256,
        score_input_sha256="4" * 64,
        risk_score=0.3,
        observed_error=False,
        label_source="expert_adjudication",
        adjudication_protocol_sha256=ADJUDICATION_PROTOCOL_SHA256,
        adjudication_artifact_sha256="5" * 64,
    )

    with pytest.raises(ItemRiskCalibrationError, match="question_disjoint"):
        calibrate_item_risk_bounds(
            [development, calibration, duplicate],
            pipeline_verification=verification,
            bin_family=family,
            familywise_delta=0.05,
            sampling_protocol_sha256=SAMPLING_PROTOCOL_SHA256,
            error_event_definition="Material error.",
            shift_detector_id="monitor",
            shift_detector_sha256=SHIFT_DETECTOR_SHA256,
        )


def test_nested_bundle_mutation_invalidates_proof(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    bundle.units.reverse()

    with pytest.raises(ItemRiskCalibrationError, match="integrity_changed"):
        validate_item_risk_calibration_bundle_integrity(bundle)
