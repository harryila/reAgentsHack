from __future__ import annotations

import json

import pytest
import scripts.calibrate_risk_gate as cli

from literature_multiverse.calibration import CalibrationContractError, RiskExample
from literature_multiverse.lineage import atomic_write_jsonl


def _rows() -> list[RiskExample]:
    rows: list[RiskExample] = []
    counter = 0
    for split, count in (("development", 12), ("calibration", 12), ("test", 8)):
        for index in range(count):
            unsupported = index >= count // 2
            score = 0.85 if unsupported else 0.05
            rows.append(
                RiskExample(
                    question_id=f"{split}-q-{index}",
                    split=split,
                    population_id="cli-simulation-v1",
                    domain="cli-domain",
                    pipeline_sha256="c" * 64,
                    paper_ids=[f"cli-paper-{counter}"],
                    features={"uncertainty": score},
                    unsupported_claim=unsupported,
                    label_source="simulation",
                )
            )
            counter += 1
    return rows


def test_cli_freeze_then_evaluate_test_as_separate_inputs(tmp_path, capsys) -> None:
    rows = _rows()
    freeze_input = tmp_path / "development-calibration.jsonl"
    test_input = tmp_path / "test.jsonl"
    bundle_path = tmp_path / "bundle.json"
    evaluation_path = tmp_path / "test-evaluation.json"
    atomic_write_jsonl(freeze_input, [row for row in rows if row.split != "test"])
    atomic_write_jsonl(test_input, [row for row in rows if row.split == "test"])

    assert (
        cli.main(
            [
                "freeze",
                "--input",
                str(freeze_input),
                "--output",
                str(bundle_path),
                "--alpha",
                "0.5",
                "--candidate-threshold",
                "0.5",
            ]
        )
        == 0
    )
    freeze_summary = json.loads(capsys.readouterr().out)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert freeze_summary["stage"] == "frozen_before_test_access"
    assert freeze_summary["bundle_sha256"] == bundle["bundle_sha256"]

    assert (
        cli.main(
            [
                "evaluate-test",
                "--bundle",
                str(bundle_path),
                "--expected-freeze-sha256",
                bundle["bundle_sha256"],
                "--input",
                str(test_input),
                "--output",
                str(evaluation_path),
            ]
        )
        == 0
    )
    test_summary = json.loads(capsys.readouterr().out)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert test_summary["stage"] == "held_out_test_after_freeze"
    assert evaluation["frozen_bundle_sha256"] == bundle["bundle_sha256"]
    assert evaluation["test_evaluation"]["total"] == 8


def test_cli_checks_external_freeze_hash_before_opening_test(tmp_path, monkeypatch) -> None:
    rows = _rows()
    freeze_input = tmp_path / "development-calibration.jsonl"
    bundle_path = tmp_path / "bundle.json"
    atomic_write_jsonl(freeze_input, [row for row in rows if row.split != "test"])
    cli.main(
        [
            "freeze",
            "--input",
            str(freeze_input),
            "--output",
            str(bundle_path),
            "--alpha",
            "0.5",
            "--candidate-threshold",
            "0.5",
        ]
    )
    monkeypatch.setattr(
        cli,
        "_read_jsonl",
        lambda _path: pytest.fail("test input opened before freeze hash verification"),
    )
    with pytest.raises(CalibrationContractError, match="expected_freeze_sha256_mismatch"):
        cli.main(
            [
                "evaluate-test",
                "--bundle",
                str(bundle_path),
                "--expected-freeze-sha256",
                "0" * 64,
                "--input",
                str(tmp_path / "must-not-open.jsonl"),
                "--output",
                str(tmp_path / "must-not-write.json"),
            ]
        )


def test_one_shot_cli_is_restricted_to_simulation_labels(tmp_path) -> None:
    input_path = tmp_path / "real-labels.jsonl"
    rows = [row.model_copy(update={"label_source": "benchmark_annotation"}) for row in _rows()]
    atomic_write_jsonl(input_path, rows)
    with pytest.raises(
        CalibrationContractError, match="diagnostic_one_shot_requires_simulation_labels"
    ):
        cli.main(
            [
                "diagnostic-one-shot",
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "diagnostic.json"),
            ]
        )
