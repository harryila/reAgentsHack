from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.calibrate_adaptive_release as cli
from tests.test_adaptive_calibration import _contexts, _trajectory
from tests.test_adaptive_calibration_v2 import (
    _context as _context_v2,
)
from tests.test_adaptive_calibration_v2 import (
    _reference as _reference_v2,
)
from tests.test_adaptive_calibration_v2 import (
    _visible as _visible_v2,
)

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationBundleV2,
    AdaptiveCalibrationError,
    AdaptiveDevelopmentFreeze,
    AdaptiveDevelopmentFreezeV2,
    GateCompleteCalibrationRosterV2,
    freeze_question_reference_verdict,
    join_labeled_question_trajectory_v2,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)


def _inputs(tmp_path: Path) -> dict[str, object]:
    contexts = _contexts()
    development = [
        _trajectory(
            question_index=index,
            split="development",
            contexts=contexts,
            reference=("supported" if index % 3 else "contradicted"),
        )
        for index in range(1, 7)
    ]
    calibration = [
        _trajectory(
            question_index=index,
            split="calibration",
            contexts=contexts,
            reference=("supported" if index % 2 == 0 else "contradicted"),
        )
        for index in range(7, 12)
    ]
    calibration.sort(key=lambda row: row.visible.question_id)
    contexts_path = tmp_path / "policy-contexts.json"
    development_path = tmp_path / "development.jsonl"
    calibration_visible_path = tmp_path / "calibration-visible.jsonl"
    calibration_labels_path = tmp_path / "calibration-labels.private.jsonl"
    atomic_write_json(contexts_path, contexts)
    atomic_write_jsonl(development_path, development)
    atomic_write_jsonl(
        calibration_visible_path,
        [row.visible for row in calibration],
    )
    atomic_write_jsonl(
        calibration_labels_path,
        [row.reference for row in calibration],
    )
    return {
        "contexts": contexts,
        "development": development,
        "calibration": calibration,
        "contexts_path": contexts_path,
        "development_path": development_path,
        "calibration_visible_path": calibration_visible_path,
        "calibration_labels_path": calibration_labels_path,
    }


def _freeze(
    tmp_path: Path,
    inputs: dict[str, object],
    *,
    calibration_visible_path: Path | None = None,
) -> tuple[Path, AdaptiveDevelopmentFreeze]:
    output = tmp_path / "adaptive-development-freeze.json"
    assert (
        cli.main(
            [
                "freeze-development",
                "--development-trajectories",
                str(inputs["development_path"]),
                "--policy-contexts",
                str(inputs["contexts_path"]),
                "--calibration-visible-trajectories",
                str(calibration_visible_path or inputs["calibration_visible_path"]),
                "--alpha",
                "0.99",
                "--delta",
                "0.5",
                "--candidate-threshold",
                "adaptive=1.0",
                "--candidate-threshold",
                "random=1.0",
                "--seed",
                "9",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    return output, AdaptiveDevelopmentFreeze.model_validate_json(output.read_text(encoding="utf-8"))


def _build_args(
    *,
    freeze_path: Path,
    freeze_sha256: str,
    labels_path: Path,
    output: Path,
) -> list[str]:
    return [
        "build-calibration-bundle",
        "--development-freeze",
        str(freeze_path),
        "--expected-development-freeze-sha256",
        freeze_sha256,
        "--calibration-labels",
        str(labels_path),
        "--output",
        str(output),
    ]


def _inputs_v2(tmp_path: Path) -> dict[str, object]:
    context = _context_v2()
    development_visible = [_visible_v2(101, split="development", context=context)]
    calibration_visible = [
        _visible_v2(110, split="calibration", context=context, decision="supported"),
        _visible_v2(111, split="calibration", context=context, decision="supported"),
    ]
    development = [
        join_labeled_question_trajectory_v2(
            visible=row,
            reference=_reference_v2(row),
        )
        for row in development_visible
    ]
    references = [_reference_v2(row) for row in calibration_visible]
    assessment_receipts: list[object] = []
    paths = {
        "contexts_path": tmp_path / "policy-contexts-v2.json",
        "development_path": tmp_path / "development-v2.jsonl",
        "calibration_visible_path": tmp_path / "calibration-visible-v2.jsonl",
        "calibration_assessment_receipts_path": (
            tmp_path / "calibration-assessment-receipts-v2.private.jsonl"
        ),
        "calibration_labels_path": tmp_path / "calibration-labels-v2.private.jsonl",
    }
    atomic_write_json(paths["contexts_path"], [context])
    atomic_write_jsonl(paths["development_path"], development)
    atomic_write_jsonl(paths["calibration_visible_path"], calibration_visible)
    atomic_write_jsonl(
        paths["calibration_assessment_receipts_path"],
        assessment_receipts,
    )
    atomic_write_jsonl(paths["calibration_labels_path"], references)
    return {
        **paths,
        "context": context,
        "development": development,
        "calibration_visible": calibration_visible,
        "calibration_assessment_receipts": assessment_receipts,
        "references": references,
    }


def _freeze_v2(
    tmp_path: Path,
    inputs: dict[str, object],
) -> tuple[Path, AdaptiveDevelopmentFreezeV2]:
    output = tmp_path / "adaptive-development-freeze-v2.json"
    assert (
        cli.main(
            [
                "freeze-development-v2",
                "--development-trajectories",
                str(inputs["development_path"]),
                "--policy-contexts",
                str(inputs["contexts_path"]),
                "--calibration-visible-trajectories",
                str(inputs["calibration_visible_path"]),
                "--alpha",
                "0.99",
                "--delta",
                "0.5",
                "--candidate-threshold",
                "adaptive=1.0",
                "--seed",
                "19",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    return (
        output,
        AdaptiveDevelopmentFreezeV2.model_validate_json(output.read_text(encoding="utf-8")),
    )


def test_staged_cli_builds_typed_artifacts_and_prints_self_hashed_receipts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _inputs(tmp_path)
    freeze_path, freeze = _freeze(tmp_path, inputs)
    freeze_receipt = json.loads(capsys.readouterr().out)

    assert freeze_receipt["receipt_sha256"] == hash_canonical(
        {key: value for key, value in freeze_receipt.items() if key != "receipt_sha256"}
    )
    assert freeze_receipt["calibration_labels_opened"] is False
    assert freeze_receipt["development_freeze_sha256"] == (freeze.development_freeze_sha256)
    assert freeze_receipt["development_freeze_file_sha256"] == sha256_file(freeze_path)

    bundle_path = tmp_path / "adaptive-calibration-bundle.json"
    assert (
        cli.main(
            _build_args(
                freeze_path=freeze_path,
                freeze_sha256=freeze.development_freeze_sha256,
                labels_path=inputs["calibration_labels_path"],  # type: ignore[arg-type]
                output=bundle_path,
            )
        )
        == 0
    )
    bundle_receipt = json.loads(capsys.readouterr().out)
    bundle = AdaptiveCalibrationBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    assert bundle_receipt["receipt_sha256"] == hash_canonical(
        {key: value for key, value in bundle_receipt.items() if key != "receipt_sha256"}
    )
    assert bundle_receipt["bundle_sha256"] == bundle.bundle_sha256
    assert bundle_receipt["bundle_file_sha256"] == sha256_file(bundle_path)
    assert bundle_receipt["test_labels_opened"] is False
    assert bundle_receipt["access_order"].index(
        "external_development_freeze_sha256_matched"
    ) < bundle_receipt["access_order"].index("calibration_labels_opened_after_freeze_match")


def test_expected_freeze_hash_is_checked_before_calibration_label_file_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    freeze_path, _ = _freeze(tmp_path, inputs)

    monkeypatch.setattr(
        cli,
        "_read_question_references",
        lambda _path: pytest.fail("calibration labels opened before external freeze match"),
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="expected_development_freeze_sha256_mismatch",
    ):
        cli.main(
            _build_args(
                freeze_path=freeze_path,
                freeze_sha256="0" * 64,
                labels_path=tmp_path / "must-remain-unopened.jsonl",
                output=tmp_path / "must-not-write.json",
            )
        )


def test_freeze_rejects_test_rows_and_cross_split_publication_overlap(
    tmp_path: Path,
) -> None:
    test_inputs = _inputs(tmp_path / "test-split")
    test_row = _trajectory(
        question_index=20,
        split="test",
        contexts=test_inputs["contexts"],  # type: ignore[arg-type]
        reference="supported",
    )
    test_visible_path = tmp_path / "test-visible.jsonl"
    atomic_write_jsonl(test_visible_path, [test_row.visible])
    with pytest.raises(AdaptiveCalibrationError, match="adaptive_trajectory_split_mismatch"):
        _freeze(tmp_path / "test-output", test_inputs, calibration_visible_path=test_visible_path)

    overlap_inputs = _inputs(tmp_path / "overlap")
    overlapping_row = _trajectory(
        question_index=30,
        split="calibration",
        contexts=overlap_inputs["contexts"],  # type: ignore[arg-type]
        reference="supported",
        source_manifest_extra="publication-1",
    )
    overlap_visible_path = tmp_path / "overlap-visible.jsonl"
    atomic_write_jsonl(overlap_visible_path, [overlapping_row.visible])
    with pytest.raises(
        AdaptiveCalibrationError,
        match="complete_corpus_publication_cross_split_overlap",
    ):
        _freeze(
            tmp_path / "overlap-output",
            overlap_inputs,
            calibration_visible_path=overlap_visible_path,
        )


def test_calibration_rejects_extra_test_label_and_self_hash_tampering(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    freeze_path, freeze = _freeze(tmp_path, inputs)
    labels = [row.reference for row in inputs["calibration"]]  # type: ignore[union-attr]
    labels.append(
        freeze_question_reference_verdict(
            question_id="question-held-out-test",
            verdict="supported",
            label_source="expert_adjudication",
            adjudication_protocol_sha256="d" * 64,
            adjudication_artifact_sha256="e" * 64,
        )
    )
    labels.sort(key=lambda row: row.question_id)
    extra_path = tmp_path / "labels-with-test-row.jsonl"
    atomic_write_jsonl(extra_path, labels)
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_calibration_label_roster_mismatch",
    ):
        cli.main(
            _build_args(
                freeze_path=freeze_path,
                freeze_sha256=freeze.development_freeze_sha256,
                labels_path=extra_path,
                output=tmp_path / "extra-output.json",
            )
        )

    tampered = inputs["calibration"][0].reference.model_dump(mode="json")  # type: ignore[index,union-attr]
    tampered["verdict"] = "tampered-without-rehash"
    tampered_path = tmp_path / "tampered-label.jsonl"
    atomic_write_jsonl(tampered_path, [tampered])
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_calibration_labels_contract_invalid",
    ):
        cli.main(
            _build_args(
                freeze_path=freeze_path,
                freeze_sha256=freeze.development_freeze_sha256,
                labels_path=tampered_path,
                output=tmp_path / "tampered-output.json",
            )
        )


def test_output_replacement_requires_explicit_force_and_never_aliases_input(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    freeze_path, freeze = _freeze(tmp_path, inputs)
    output = tmp_path / "bundle.json"
    atomic_write_json(output, {"preexisting": True})
    arguments = _build_args(
        freeze_path=freeze_path,
        freeze_sha256=freeze.development_freeze_sha256,
        labels_path=inputs["calibration_labels_path"],  # type: ignore[arg-type]
        output=output,
    )
    with pytest.raises(OutputExistsError):
        cli.main(arguments)

    assert cli.main([*arguments, "--force"]) == 0
    AdaptiveCalibrationBundle.model_validate_json(output.read_text(encoding="utf-8"))

    with pytest.raises(AdaptiveCalibrationError, match="adaptive_output_must_not_alias_input"):
        cli.main(
            [
                *_build_args(
                    freeze_path=freeze_path,
                    freeze_sha256=freeze.development_freeze_sha256,
                    labels_path=inputs["calibration_labels_path"],  # type: ignore[arg-type]
                    output=inputs["calibration_labels_path"],  # type: ignore[arg-type]
                ),
                "--force",
            ]
        )


def test_v2_staged_cli_freezes_outcome_roster_before_opening_references(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _inputs_v2(tmp_path)
    freeze_path, freeze = _freeze_v2(tmp_path, inputs)
    freeze_receipt = json.loads(capsys.readouterr().out)
    assert freeze_receipt["calibration_assessment_receipts_opened"] is False
    assert freeze_receipt["calibration_labels_opened"] is False
    assert freeze_receipt["test_rows_accepted"] is False

    gate_roster_path = tmp_path / "gate-complete-roster-v2.json"
    assert (
        cli.main(
            [
                "freeze-terminal-gates-v2",
                "--development-freeze",
                str(freeze_path),
                "--expected-development-freeze-sha256",
                freeze.development_freeze_sha256,
                "--calibration-assessment-receipts",
                str(inputs["calibration_assessment_receipts_path"]),
                "--output",
                str(gate_roster_path),
            ]
        )
        == 0
    )
    gate_receipt = json.loads(capsys.readouterr().out)
    gate_roster = GateCompleteCalibrationRosterV2.model_validate_json(
        gate_roster_path.read_text(encoding="utf-8")
    )
    assert gate_receipt["calibration_assessment_receipts_opened"] is True
    assert gate_receipt["calibration_labels_opened"] is False
    assert gate_receipt["gate_complete_roster_sha256"] == (gate_roster.gate_roster_sha256)
    assert gate_receipt["access_order"].index(
        "external_development_freeze_sha256_matched"
    ) < gate_receipt["access_order"].index(
        "calibration_assessment_receipts_opened_after_source_and_development_freezes_matched"
    )

    bundle_path = tmp_path / "adaptive-calibration-bundle-v2.json"
    assert (
        cli.main(
            [
                "build-calibration-bundle-v2",
                "--development-freeze",
                str(freeze_path),
                "--expected-development-freeze-sha256",
                freeze.development_freeze_sha256,
                "--gate-complete-roster",
                str(gate_roster_path),
                "--expected-gate-complete-roster-sha256",
                gate_roster.gate_roster_sha256,
                "--calibration-labels",
                str(inputs["calibration_labels_path"]),
                "--output",
                str(bundle_path),
            ]
        )
        == 0
    )
    bundle_receipt = json.loads(capsys.readouterr().out)
    bundle = AdaptiveCalibrationBundleV2.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    assert bundle_receipt["bundle_sha256"] == bundle.bundle_sha256
    assert bundle_receipt["test_labels_opened"] is False
    assert bundle_receipt["condition_release_domains"] == ["medicine"]
    assert bundle_receipt["simultaneous_test_count"] == 2
    assert bundle_receipt["candidates_with_complete_condition_domain_support"] == 0
    assert bundle_receipt["access_order"].index(
        "external_gate_complete_roster_sha256_and_lineage_matched"
    ) < bundle_receipt["access_order"].index(
        "calibration_labels_opened_after_both_external_freeze_matches"
    )


def test_v2_external_hashes_are_checked_before_later_stage_files_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs_v2(tmp_path)
    freeze_path, freeze = _freeze_v2(tmp_path, inputs)

    original_jsonl_reader = cli._read_jsonl_models

    def fail_on_terminal_results(*args: object, **kwargs: object) -> object:
        purpose = kwargs.get("purpose")
        if purpose == "adaptive_v2_calibration_assessment_receipts":
            pytest.fail("terminal outcomes opened before external freeze match")
        return original_jsonl_reader(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_read_jsonl_models", fail_on_terminal_results)
    with pytest.raises(
        AdaptiveCalibrationError,
        match="expected_v2_development_freeze_sha256_mismatch",
    ):
        cli.main(
            [
                "freeze-terminal-gates-v2",
                "--development-freeze",
                str(freeze_path),
                "--expected-development-freeze-sha256",
                "0" * 64,
                "--calibration-assessment-receipts",
                str(inputs["calibration_assessment_receipts_path"]),
                "--output",
                str(tmp_path / "must-not-exist-gates.json"),
            ]
        )

    monkeypatch.setattr(cli, "_read_jsonl_models", original_jsonl_reader)
    gate_roster_path = tmp_path / "gate-roster-v2.json"
    assert (
        cli.main(
            [
                "freeze-terminal-gates-v2",
                "--development-freeze",
                str(freeze_path),
                "--expected-development-freeze-sha256",
                freeze.development_freeze_sha256,
                "--calibration-assessment-receipts",
                str(inputs["calibration_assessment_receipts_path"]),
                "--output",
                str(gate_roster_path),
            ]
        )
        == 0
    )
    GateCompleteCalibrationRosterV2.model_validate_json(
        gate_roster_path.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        cli,
        "_read_question_references_v2",
        lambda _path: pytest.fail(
            "calibration references opened before external gate-roster match"
        ),
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="expected_v2_gate_complete_roster_sha256_mismatch",
    ):
        cli.main(
            [
                "build-calibration-bundle-v2",
                "--development-freeze",
                str(freeze_path),
                "--expected-development-freeze-sha256",
                freeze.development_freeze_sha256,
                "--gate-complete-roster",
                str(gate_roster_path),
                "--expected-gate-complete-roster-sha256",
                "f" * 64,
                "--calibration-labels",
                str(inputs["calibration_labels_path"]),
                "--output",
                str(tmp_path / "must-not-exist-bundle.json"),
            ]
        )


def test_v2_cli_never_auto_downgrades_a_v1_freeze(tmp_path: Path) -> None:
    v1_inputs = _inputs(tmp_path / "v1")
    v1_path, v1_freeze = _freeze(tmp_path / "v1", v1_inputs)
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_v2_development_freeze_contract_invalid",
    ):
        cli.main(
            [
                "freeze-terminal-gates-v2",
                "--development-freeze",
                str(v1_path),
                "--expected-development-freeze-sha256",
                v1_freeze.development_freeze_sha256,
                "--calibration-assessment-receipts",
                str(tmp_path / "must-remain-unopened.jsonl"),
                "--output",
                str(tmp_path / "must-not-write.json"),
            ]
        )
