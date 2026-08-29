from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.calibrate_item_risk as cli

from literature_multiverse.item_risk_artifacts import (
    ExternalShiftDetectorReceipt,
    ItemRiskArtifactError,
    ItemRiskCalibrationRunReceipt,
    ItemRiskScoringRunReceipt,
    LegacyItemRiskScoringRunReceiptV1,
    ShiftAssessmentRunReceipt,
)
from literature_multiverse.item_risk_calibration import seal_item_risk_calibration_unit
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    compute_pipeline_fingerprint,
)

SCORE_MODEL_SHA256 = "b" * 64
ADJUDICATION_PROTOCOL_SHA256 = "c" * 64
SAMPLING_PROTOCOL_SHA256 = "d" * 64
SHIFT_DETECTOR_SHA256 = "e" * 64


def _pipeline(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "pipeline-root"
    root.mkdir()
    (root / "pipeline.py").write_text("FROZEN = True\n", encoding="utf-8")
    expected = compute_pipeline_fingerprint(
        root=root,
        components=[
            PipelineComponentSpec(
                component_id="item-risk-pipeline",
                component_version="1",
                file_paths=["pipeline.py"],
                settings={"score_model_sha256": SCORE_MODEL_SHA256},
            )
        ],
    )
    expected_path = tmp_path / "expected-pipeline.json"
    atomic_write_json(expected_path, expected)
    return root, expected_path, expected.pipeline_sha256


def _freeze_bins(tmp_path: Path) -> Path:
    definition_path = tmp_path / "bin-definition.json"
    output_path = tmp_path / "fixed-bins.json"
    atomic_write_json(
        definition_path,
        {
            "definition_version": "item-risk-bin-definition-v1",
            "definition_source": "development_only",
            "source_split": "development",
            "labels_used": True,
            "label_source": "expert_adjudication",
            "simulation": False,
            "score_name": "prospective_extraction_error_score",
            "score_model_sha256": SCORE_MODEL_SHA256,
            "edges": [0.0, 0.5, 1.0],
        },
    )
    assert (
        cli.main(
            [
                "freeze-bins",
                "--definition",
                str(definition_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    return output_path


def _unit(
    index: int, *, split: str, pipeline_sha256: str, score: float, error: bool
):
    return seal_item_risk_calibration_unit(
        split=split,
        item_id=f"item-{index}",
        question_id=f"question-{index}",
        paper_id=f"paper-{index}",
        population_id="biomed-v1",
        domain="cardiology",
        pipeline_sha256=pipeline_sha256,
        score_model_sha256=SCORE_MODEL_SHA256,
        score_input_sha256=f"{index:x}".rjust(64, "0"),
        risk_score=score,
        observed_error=error,
        label_source="expert_adjudication",
        adjudication_protocol_sha256=ADJUDICATION_PROTOCOL_SHA256,
        adjudication_artifact_sha256=f"{index + 100:x}".rjust(64, "0"),
    )


def _calibrate(
    tmp_path: Path, *, root: Path, expected: Path, pipeline_sha256: str, bins: Path
) -> tuple[Path, Path, Path]:
    development_path = tmp_path / "development.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    output_path = tmp_path / "calibration-run.json"
    atomic_write_jsonl(
        development_path,
        [
            _unit(
                1,
                split="development",
                pipeline_sha256=pipeline_sha256,
                score=0.1,
                error=False,
            ),
            _unit(
                2,
                split="development",
                pipeline_sha256=pipeline_sha256,
                score=0.7,
                error=True,
            ),
        ],
    )
    atomic_write_jsonl(
        calibration_path,
        [
            _unit(
                3,
                split="calibration",
                pipeline_sha256=pipeline_sha256,
                score=0.1,
                error=False,
            ),
            _unit(
                4,
                split="calibration",
                pipeline_sha256=pipeline_sha256,
                score=0.2,
                error=False,
            ),
        ],
    )
    assert (
        cli.main(
            [
                "calibrate",
                "--expected-pipeline",
                str(expected),
                "--pipeline-root",
                str(root),
                "--fixed-bins",
                str(bins),
                "--development-units",
                str(development_path),
                "--calibration-units",
                str(calibration_path),
                "--familywise-delta",
                "0.05",
                "--sampling-protocol-sha256",
                SAMPLING_PROTOCOL_SHA256,
                "--error-event-definition",
                "Any adjudicated material extraction error in the item.",
                "--shift-detector-id",
                "frozen-domain-monitor-v1",
                "--shift-detector-sha256",
                SHIFT_DETECTOR_SHA256,
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    return output_path, development_path, calibration_path


def _candidate_file(tmp_path: Path, *, pipeline_sha256: str) -> Path:
    candidate_path = tmp_path / "prospective.jsonl"
    atomic_write_jsonl(
        candidate_path,
        [
            {
                "input_version": "prospective-item-risk-input-v1",
                "source_split": "prospective",
                "simulation": False,
                "item_id": "prospective-item",
                "question_id": "prospective-question",
                "paper_id": "prospective-paper",
                "population_id": "biomed-v1",
                "domain": "cardiology",
                "pipeline_sha256": pipeline_sha256,
                "score_model_sha256": SCORE_MODEL_SHA256,
                "score_input_sha256": "1" * 64,
                "risk_score": 0.2,
            }
        ],
    )
    return candidate_path


def _assess_shift(tmp_path: Path, calibration_run: Path, candidates: Path) -> Path:
    calibration = ItemRiskCalibrationRunReceipt.model_validate_json(
        calibration_run.read_text(encoding="utf-8")
    )
    detector_artifact = tmp_path / "detector-output.json"
    atomic_write_json(detector_artifact, {"decision": "no_shift_detected"})
    receipt_payload = {
        "receipt_version": "external-item-risk-shift-v1",
        "calibration_bundle_sha256": calibration.bundle.bundle_sha256,
        "detector_id": calibration.bundle.shift_detector_id,
        "detector_sha256": calibration.bundle.shift_detector_sha256,
        "candidate_population_id": "biomed-v1",
        "candidate_domain": "cardiology",
        "candidate_input_file_sha256": sha256_file(candidates),
        "status": "no_shift_detected",
        "detector_artifact_sha256": sha256_file(detector_artifact),
        "source_split": "prospective",
        "labels_opened": False,
        "simulation": False,
    }
    detector_receipt = ExternalShiftDetectorReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": hash_canonical(receipt_payload),
        }
    )
    detector_receipt_path = tmp_path / "detector-receipt.json"
    atomic_write_json(detector_receipt_path, detector_receipt)
    output_path = tmp_path / "shift-assessment.json"
    assert (
        cli.main(
            [
                "assess-shift",
                "--calibration-run",
                str(calibration_run),
                "--detector-receipt",
                str(detector_receipt_path),
                "--detector-artifact",
                str(detector_artifact),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    return output_path


def test_cli_artifact_flow_produces_recomputable_scheduling_only_cell_ucl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, expected, pipeline_sha256 = _pipeline(tmp_path)
    bins = _freeze_bins(tmp_path)
    calibration_run, _, _ = _calibrate(
        tmp_path,
        root=root,
        expected=expected,
        pipeline_sha256=pipeline_sha256,
        bins=bins,
    )
    candidates = _candidate_file(tmp_path, pipeline_sha256=pipeline_sha256)
    shift = _assess_shift(tmp_path, calibration_run, candidates)
    score_output = tmp_path / "risk-bounds.json"

    assert (
        cli.main(
            [
                "score",
                "--calibration-run",
                str(calibration_run),
                "--expected-pipeline",
                str(expected),
                "--pipeline-root",
                str(root),
                "--shift-assessment",
                str(shift),
                "--candidates",
                str(candidates),
                "--output",
                str(score_output),
            ]
        )
        == 0
    )

    calibration = ItemRiskCalibrationRunReceipt.model_validate_json(
        calibration_run.read_text(encoding="utf-8")
    )
    assessment = ShiftAssessmentRunReceipt.model_validate_json(
        shift.read_text(encoding="utf-8")
    )
    scoring = ItemRiskScoringRunReceipt.model_validate_json(
        score_output.read_text(encoding="utf-8")
    )
    assert calibration.access_order.index("pipeline_fingerprint_recomputed_and_matched") < (
        calibration.access_order.index("calibration_units_opened")
    )
    assert assessment.assessment.calibration_bundle_sha256 == calibration.bundle.bundle_sha256
    assert scoring.receipt_version == "item-risk-scoring-run-v2"
    assert scoring.calibration_bundle == calibration.bundle
    assert scoring.calibration_bundle_sha256 == calibration.bundle.bundle_sha256
    assert scoring.candidates[0].candidate_sha256 == scoring.candidate_sha256s[0]
    assert scoring.bounds[0].status == "cell_rate_ucl_available"
    assert scoring.bounds[0].usable_for_scheduling
    assert not scoring.bounds[0].usable_for_release
    assert scoring.bounds[0].upper_cell_error_rate is not None
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert summaries[-1]["status_counts"] == {"cell_rate_ucl_available": 1}

    tampered = scoring.model_dump(mode="json")
    tampered_bound = dict(tampered["bounds"][0])
    tampered_bound["raw_risk_score"] = 0.3
    tampered_bound["risk_bound_sha256"] = hash_canonical(
        {
            key: value
            for key, value in tampered_bound.items()
            if key != "risk_bound_sha256"
        }
    )
    tampered["bounds"][0] = tampered_bound
    tampered["receipt_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="bound_recomputation_mismatch"):
        ItemRiskScoringRunReceipt.model_validate(tampered)

    legacy_payload = {
        "receipt_version": "item-risk-scoring-run-v1",
        "calibration_run_file_sha256": scoring.calibration_run_file_sha256,
        "calibration_run_receipt_sha256": scoring.calibration_run_receipt_sha256,
        "expected_pipeline_file_sha256": scoring.expected_pipeline_file_sha256,
        "shift_run_file_sha256": scoring.shift_run_file_sha256,
        "shift_run_receipt_sha256": scoring.shift_run_receipt_sha256,
        "candidate_input_file_sha256": scoring.candidate_input_file_sha256,
        "candidate_count": scoring.candidate_count,
        "pipeline_verification": scoring.pipeline_verification,
        "candidate_sha256s": scoring.candidate_sha256s,
        "bounds": [
            {
                "candidate_sha256": scoring.candidate_sha256s[0],
                "legacy_probability_basis": "calibrated_upper_bound",
            }
        ],
        "access_order": scoring.access_order,
    }
    legacy = LegacyItemRiskScoringRunReceiptV1.model_validate(
        {**legacy_payload, "receipt_sha256": hash_canonical(legacy_payload)}
    )
    assert legacy.diagnostic_only
    with pytest.raises(ValueError):
        ItemRiskScoringRunReceipt.model_validate(legacy.model_dump(mode="json"))


def test_calibrate_verifies_pipeline_before_any_label_file_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, expected, pipeline_sha256 = _pipeline(tmp_path)
    bins = _freeze_bins(tmp_path)
    capsys.readouterr()
    (root / "pipeline.py").write_text("FROZEN = False\n", encoding="utf-8")
    label_opens: list[Path] = []
    original = cli._read_jsonl_payloads

    def tracked(path: Path, **kwargs):
        label_opens.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(cli, "_read_jsonl_payloads", tracked)
    with pytest.raises(ItemRiskArtifactError, match="pipeline_fingerprint_not_matched"):
        cli.main(
            [
                "calibrate",
                "--expected-pipeline",
                str(expected),
                "--pipeline-root",
                str(root),
                "--fixed-bins",
                str(bins),
                "--development-units",
                str(tmp_path / "never-open-development.jsonl"),
                "--calibration-units",
                str(tmp_path / "never-open-calibration.jsonl"),
                "--familywise-delta",
                "0.05",
                "--sampling-protocol-sha256",
                SAMPLING_PROTOCOL_SHA256,
                "--error-event-definition",
                "Material error.",
                "--shift-detector-id",
                "frozen-domain-monitor-v1",
                "--shift-detector-sha256",
                SHIFT_DETECTOR_SHA256,
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
    assert label_opens == []
    assert not (tmp_path / "must-not-exist.json").exists()
    assert pipeline_sha256


def test_calibrate_requires_physically_separate_split_files(tmp_path: Path) -> None:
    root, expected, pipeline_sha256 = _pipeline(tmp_path)
    bins = _freeze_bins(tmp_path)
    shared = tmp_path / "shared-labels.jsonl"
    atomic_write_jsonl(
        shared,
        [
            _unit(
                1,
                split="development",
                pipeline_sha256=pipeline_sha256,
                score=0.1,
                error=False,
            )
        ],
    )
    with pytest.raises(ItemRiskArtifactError, match="physically_separate"):
        cli.main(
            [
                "calibrate",
                "--expected-pipeline",
                str(expected),
                "--pipeline-root",
                str(root),
                "--fixed-bins",
                str(bins),
                "--development-units",
                str(shared),
                "--calibration-units",
                str(shared),
                "--familywise-delta",
                "0.05",
                "--sampling-protocol-sha256",
                SAMPLING_PROTOCOL_SHA256,
                "--error-event-definition",
                "Material error.",
                "--shift-detector-id",
                "frozen-domain-monitor-v1",
                "--shift-detector-sha256",
                SHIFT_DETECTOR_SHA256,
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )


def test_score_rejects_manifest_probability_and_outputs_never_overwrite(
    tmp_path: Path,
) -> None:
    definition = tmp_path / "definition.json"
    output = tmp_path / "fixed.json"
    atomic_write_json(
        definition,
        {
            "definition_version": "item-risk-bin-definition-v1",
            "definition_source": "prespecified",
            "source_split": "none",
            "labels_used": False,
            "label_source": None,
            "simulation": False,
            "score_name": "risk",
            "score_model_sha256": SCORE_MODEL_SHA256,
            "edges": [0.0, 1.0],
        },
    )
    cli.main(
        ["freeze-bins", "--definition", str(definition), "--output", str(output)]
    )
    with pytest.raises(OutputExistsError):
        cli.main(
            ["freeze-bins", "--definition", str(definition), "--output", str(output)]
        )

    probability_payload = {
        "input_version": "prospective-item-risk-input-v1",
        "source_split": "prospective",
        "simulation": False,
        "item_id": "x",
        "question_id": "q",
        "paper_id": "p",
        "population_id": "pop",
        "domain": "domain",
        "pipeline_sha256": "1" * 64,
        "score_model_sha256": SCORE_MODEL_SHA256,
        "score_input_sha256": "2" * 64,
        "risk_score": 0.2,
        "error_probability": 0.01,
    }
    with pytest.raises(ItemRiskArtifactError, match="manifest_probability_forbidden"):
        cli._forbid_manifest_probabilities([probability_payload])


def test_simulation_and_test_scope_are_not_accepted(tmp_path: Path) -> None:
    definition = tmp_path / "simulation-definition.json"
    atomic_write_json(
        definition,
        {
            "definition_version": "item-risk-bin-definition-v1",
            "definition_source": "prespecified",
            "source_split": "none",
            "labels_used": False,
            "label_source": None,
            "simulation": True,
            "score_name": "risk",
            "score_model_sha256": SCORE_MODEL_SHA256,
            "edges": [0.0, 1.0],
        },
    )
    with pytest.raises(ItemRiskArtifactError, match="contract_invalid"):
        cli.main(
            [
                "freeze-bins",
                "--definition",
                str(definition),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )

    candidate = {
        "input_version": "prospective-item-risk-input-v1",
        "source_split": "test",
        "simulation": False,
        "item_id": "x",
        "question_id": "q",
        "paper_id": "p",
        "population_id": "pop",
        "domain": "domain",
        "pipeline_sha256": "1" * 64,
        "score_model_sha256": SCORE_MODEL_SHA256,
        "score_input_sha256": "2" * 64,
        "risk_score": 0.2,
    }
    with pytest.raises(ItemRiskArtifactError, match="contract_invalid"):
        cli._validate_rows(
            [candidate],
            cli.ProspectiveItemRiskInput,
            purpose="prospective_item_risk_candidates",
        )

    valid_unit = _unit(
        9,
        split="calibration",
        pipeline_sha256="3" * 64,
        score=0.2,
        error=False,
    ).model_dump(mode="json")
    for forbidden_field, forbidden_value in (
        ("split", "test"),
        ("label_source", "simulation"),
    ):
        invalid_unit = {**valid_unit, forbidden_field: forbidden_value}
        payload = {key: value for key, value in invalid_unit.items() if key != "unit_sha256"}
        invalid_unit["unit_sha256"] = hash_canonical(payload)
        with pytest.raises(ItemRiskArtifactError, match="contract_invalid"):
            cli._validate_rows(
                [invalid_unit],
                cli.ItemRiskCalibrationUnit,
                purpose="calibration_units",
            )
