"""Offline contract tests for the MetaSyn review-level benchmark adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.metasyn_benchmark import _parser as metasyn_cli_parser
from scripts.metasyn_benchmark import main as metasyn_cli_main

from literature_multiverse.calibration import (
    CalibrationContractError,
    validate_split_integrity,
)
from literature_multiverse.metasyn_benchmark import (
    FORBIDDEN_MODEL_COLUMNS,
    MetaSynBenchmarkError,
    MetaSynPrediction,
    build_metasyn_risk_examples,
    evaluate_metasyn_predictions,
    freeze_fixed_direction_baseline,
    load_metasyn_inputs,
    load_metasyn_labels,
    load_metasyn_manifest,
    prepare_metasyn_benchmark,
)


def _row(
    review_id: int,
    papers: list[int],
    direction: str,
    *,
    source_reviews: list[int] | None = None,
) -> dict[str, object]:
    return {
        "ID": review_id,
        "Title": f"Review title {review_id}",
        "Abstract": f"Leaking abstract {review_id}",
        "Population": "adults",
        "Intervention": "intervention",
        "Exposure": None,
        "Comparison": "control",
        "Outcome": "continuous outcome",
        "Effect_Direction": direction,
        "Conclusion_Summary": f"Leaking conclusion {review_id}",
        "Heterogeneity_Level": "High",
        "Research_Question": f"Does intervention {review_id} change the outcome?",
        "matched_corpus_ids": papers,
        "matched_ref_count": len(set(papers)),
        "study_count": float(len(papers)),
        "source_review_corpus_ids": source_reviews or [],
    }


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    # Reviews 1 and 2 are transitively connected to official test review 20 and
    # therefore must both be quarantined.  Reviews 3 and 4 must stay together.
    train_rows = [
        _row(1, [100, 101], "Positive"),
        _row(2, [101, 102], "Negative"),
        _row(3, [200], "Mixed"),
        _row(4, [200, 201], "Positive"),
        _row(5, [300], "NR"),
        _row(6, [400], "Negative"),
        _row(7, [500], "Positive"),
        _row(8, [600], "Mixed", source_reviews=[990]),
        _row(9, [800], "Negative"),
        _row(10, [900], "Positive"),
    ]
    test_rows = [
        _row(20, [100], "Positive"),
        _row(21, [700], "Negative"),
        _row(22, [700, 701], "Mixed"),
    ]
    train_path = tmp_path / "reviews-train.parquet"
    test_path = tmp_path / "reviews-test.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(test_rows).to_parquet(test_path, index=False)
    return train_path, test_path


def _prepared(tmp_path: Path) -> tuple[Path, object]:
    train_path, test_path = _source_files(tmp_path)
    output_dir = tmp_path / "benchmark"
    manifest = prepare_metasyn_benchmark(
        train_parquet=train_path,
        test_parquet=test_path,
        output_dir=output_dir,
        seed=1,
        calibration_fraction=0.5,
    )
    return output_dir / "manifest.json", manifest


def test_prepare_preserves_official_test_and_quarantines_connected_train(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _prepared(tmp_path)

    assert manifest.test.review_ids == [20, 21, 22]
    assert [row.review_id for row in manifest.quarantined_official_train] == [1, 2]
    assert manifest.development.rows + manifest.calibration.rows == 8
    assert set(manifest.development.review_ids) & set(manifest.calibration.review_ids) == set()

    labels = load_metasyn_labels(manifest_path)
    split_by_id = {label.review_id: label.split for label in labels}
    assert split_by_id[3] == split_by_id[4]
    paper_splits: dict[int, str] = {}
    for label in labels:
        for paper_id in label.gold_matched_corpus_ids:
            assert paper_id not in paper_splits or paper_splits[paper_id] == label.split
            paper_splits[paper_id] = label.split


def test_model_inputs_exclude_all_labels_and_gold_retrieval_ids(tmp_path: Path) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    allowed = {
        "benchmark_input_version",
        "question_id",
        "review_id",
        "research_question",
        "population",
        "intervention",
        "exposure",
        "comparison",
        "outcome",
    }
    for artifact in (manifest.development, manifest.calibration, manifest.test):
        path = manifest_path.parent / artifact.path
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            assert set(payload) == allowed
            assert not set(FORBIDDEN_MODEL_COLUMNS) & set(payload)
            assert "Leaking" not in line


def test_evaluator_flags_missing_nr_abstention_and_retrieval_states(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _prepared(tmp_path)
    labels = load_metasyn_labels(manifest_path)
    eligible = [label for label in labels if label.gold_direction != "NR"]
    predictions = [
        MetaSynPrediction(
            review_id=eligible[0].review_id,
            predicted_direction=eligible[0].gold_direction,
            retrieved_corpus_ids=eligible[0].gold_matched_corpus_ids,
        ),
        MetaSynPrediction(
            review_id=eligible[1].review_id,
            predicted_direction="NR",
            retrieved_corpus_ids=[],
        ),
        MetaSynPrediction(
            review_id=eligible[2].review_id,
            predicted_direction="Abstain",
            retrieved_corpus_ids=None,
        ),
    ]

    evaluation = evaluate_metasyn_predictions(
        manifest_path=manifest_path,
        predictions=predictions,
        evaluation_split="all",
    )

    assert evaluation["direction"]["correct"] == 1
    assert evaluation["direction"]["answered"] == 1
    assert evaluation["direction"]["predicted_nr"] == 1
    assert evaluation["direction"]["abstained"] == 1
    assert evaluation["direction"]["missing"] == len(eligible) - 3
    assert evaluation["direction"]["gold_nr_excluded"] == 1
    assert evaluation["retrieval"]["explicit_empty"] == 1
    assert evaluation["retrieval"]["missing"] == len(labels) - 2
    assert evaluation["retrieval"]["micro_recall_missing_as_zero"] is not None
    assert "not an error label for scientific truth" in evaluation["loss_interpretation"]


def test_real_predictions_emit_only_observed_benchmark_risk_rows(tmp_path: Path) -> None:
    manifest_path, _ = _prepared(tmp_path)
    labels = load_metasyn_labels(manifest_path)
    predictions = [
        MetaSynPrediction(
            review_id=label.review_id,
            predicted_direction=(
                "Positive" if label.gold_direction == "NR" else label.gold_direction
            ),
            retrieved_corpus_ids=label.gold_matched_corpus_ids,
            risk_features={"grounding_failure": 0.1, "instability": 0.2},
        )
        for label in labels
    ]
    modified_index = next(
        index for index, label in enumerate(labels) if label.gold_direction != "NR"
    )
    modified_review_id = labels[modified_index].review_id
    predictions[modified_index] = predictions[modified_index].model_copy(
        update={
            "predicted_direction": "Negative",
            "retrieved_corpus_ids": [987_654],
        }
    )

    examples = build_metasyn_risk_examples(
        manifest_path=manifest_path,
        predictions=predictions,
        pipeline_sha256="a" * 64,
    )

    assert len(examples) == sum(label.gold_direction != "NR" for label in labels)
    assert {row.label_source for row in examples} == {"benchmark_annotation"}
    assert all(row.paper_ids for row in examples)
    first = next(row for row in examples if row.question_id.endswith(f"{modified_review_id:06d}"))
    assert first.paper_ids == ["metasyn-corpus:987654"]
    validate_split_integrity(examples)


def test_risk_rows_reject_actual_retrieval_crossover(tmp_path: Path) -> None:
    manifest_path, _ = _prepared(tmp_path)
    labels = load_metasyn_labels(manifest_path)
    development = next(
        label for label in labels if label.split == "development" and label.gold_direction != "NR"
    )
    calibration = next(
        label for label in labels if label.split == "calibration" and label.gold_direction != "NR"
    )
    predictions = [
        MetaSynPrediction(
            review_id=label.review_id,
            predicted_direction=label.gold_direction,
            retrieved_corpus_ids=[999_999],
            risk_features={"instability": 0.2},
        )
        for label in (development, calibration)
    ]

    with pytest.raises(CalibrationContractError, match="paper_crosses_risk_split"):
        build_metasyn_risk_examples(
            manifest_path=manifest_path,
            predictions=predictions,
            pipeline_sha256="a" * 64,
        )


def test_hash_locked_private_labels_detect_mutation(tmp_path: Path) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    labels_path = manifest_path.parent / manifest.evaluator_labels.path
    labels_path.write_text(labels_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(MetaSynBenchmarkError, match="artifact_hash_mismatch"):
        load_metasyn_manifest(manifest_path)


def test_model_loader_opens_only_requested_split(tmp_path: Path) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    labels_path = manifest_path.parent / manifest.evaluator_labels.path
    test_path = manifest_path.parent / manifest.test.path
    labels_path.rename(labels_path.with_suffix(".inaccessible"))
    test_path.rename(test_path.with_suffix(".inaccessible"))

    development = load_metasyn_inputs(manifest_path, split="development")
    calibration = load_metasyn_inputs(manifest_path, split="calibration")

    assert len(development) == manifest.development.rows
    assert len(calibration) == manifest.calibration.rows
    assert {row.review_id for row in development}.isdisjoint(row.review_id for row in calibration)
    with pytest.raises(MetaSynBenchmarkError, match="artifact_missing"):
        load_metasyn_inputs(manifest_path, split="test")


def test_fixed_direction_freeze_never_opens_labels_or_other_splits(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _prepared(tmp_path)
    inaccessible = (
        manifest.evaluator_labels,
        manifest.development,
        manifest.calibration,
    )
    for artifact in inaccessible:
        path = manifest_path.parent / artifact.path
        path.rename(path.with_suffix(".inaccessible"))
    output_dir = tmp_path / "fixed-positive"

    receipt = freeze_fixed_direction_baseline(
        manifest_path=manifest_path,
        split="test",
        direction="Positive",
        selection_note="Explicit fixture control.",
        output_dir=output_dir,
    )

    assert receipt.labels_opened is False
    assert receipt.retrieval_ids_emitted is False
    assert receipt.risk_features_emitted is False
    assert receipt.rows == manifest.test.rows
    assert receipt.model_input_artifact_sha256 == manifest.test.sha256
    prediction_lines = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    predictions = [json.loads(line) for line in prediction_lines]
    assert len(predictions) == manifest.test.rows
    assert all(
        set(row) == {"prediction_version", "review_id", "predicted_direction"}
        for row in predictions
    )
    assert {row["predicted_direction"] for row in predictions} == {"Positive"}
    assert (
        json.loads((output_dir / "freeze_receipt.json").read_text(encoding="utf-8"))[
            "labels_opened"
        ]
        is False
    )
    with pytest.raises(MetaSynBenchmarkError, match="outputs_exist"):
        freeze_fixed_direction_baseline(
            manifest_path=manifest_path,
            split="test",
            direction="Positive",
            selection_note="Explicit fixture control.",
            output_dir=output_dir,
        )


def test_fixed_direction_cli_requires_explicit_class() -> None:
    with pytest.raises(SystemExit):
        metasyn_cli_parser().parse_args(
            [
                "freeze-fixed-direction",
                "--manifest",
                "manifest.json",
                "--split",
                "test",
                "--selection-note",
                "fixture",
                "--output-dir",
                "out",
            ]
        )

    args = metasyn_cli_parser().parse_args(
        [
            "freeze-fixed-direction",
            "--manifest",
            "manifest.json",
            "--split",
            "test",
            "--direction",
            "Mixed",
            "--selection-note",
            "fixture",
            "--output-dir",
            "out",
        ]
    )
    assert args.direction == "Mixed"


def test_unknown_prediction_id_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _prepared(tmp_path)
    with pytest.raises(MetaSynBenchmarkError, match="not_in_manifest"):
        evaluate_metasyn_predictions(
            manifest_path=manifest_path,
            predictions=[MetaSynPrediction(review_id=999_999)],
        )


def test_cli_evaluates_and_emits_observed_risk_example(tmp_path: Path) -> None:
    manifest_path, _ = _prepared(tmp_path)
    test_labels = [row for row in load_metasyn_labels(manifest_path) if row.split == "test"]
    label = next(row for row in test_labels if row.gold_direction != "NR")
    prediction = MetaSynPrediction(
        review_id=label.review_id,
        predicted_direction=label.gold_direction,
        retrieved_corpus_ids=label.gold_matched_corpus_ids,
        risk_features={"instability": 0.1},
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(prediction.model_dump_json() + "\n", encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    risks_path = tmp_path / "risks.jsonl"

    assert (
        metasyn_cli_main(
            [
                "evaluate",
                "--manifest",
                manifest_path.as_posix(),
                "--predictions",
                predictions_path.as_posix(),
                "--output",
                evaluation_path.as_posix(),
                "--risk-examples-output",
                risks_path.as_posix(),
                "--pipeline-sha256",
                "a" * 64,
            ]
        )
        == 0
    )
    assert json.loads(evaluation_path.read_text(encoding="utf-8"))["direction"]["correct"] == 1
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["evaluated_split"] == "test"
    assert evaluation["direction"]["eligible_gold"] == len(test_labels)
    risk = json.loads(risks_path.read_text(encoding="utf-8"))
    assert risk["label_source"] == "benchmark_annotation"
    assert risk["domain"] == "metasyn_systematic_reviews"
