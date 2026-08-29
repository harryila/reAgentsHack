from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scripts.run_evidencebench_diagnostic as eb_cli
from pydantic import ValidationError

import literature_multiverse.evidencebench_diagnostic as eb
from literature_multiverse.evidencebench_diagnostic import (
    EvidenceBenchDiagnosticError,
    EvidenceBenchPredictionRowV1,
    EvidenceBenchPublicSummaryV1,
    audit_evidencebench_run,
    materialize_evidencebench_test,
    predict_evidencebench_test,
    prepare_evidencebench_plan,
    rank_question,
    score_evidencebench_test,
    write_evidencebench_materialization,
)
from literature_multiverse.lineage import OutputExistsError, sha256_file


def _question(split: str, index: int, *, results: bool = True) -> dict[str, Any]:
    prefix = f"evidencebench_{split}_id_{index}"
    all_aspects = [f"{prefix}_aspect_0", f"{prefix}_aspect_1"]
    results_aspects = [all_aspects[1]] if results else None
    sentences = [
        "Background and study design.",
        "Vitamin treatment improves the measured outcome.",
        "The unrelated comparator description appears here.",
        "Results show the measured outcome increased with vitamin treatment.",
        "Discussion and limitations.",
        "Methods appendix.",
        "A second unrelated observation.",
        "Reference material.",
        "Neutral sentence.",
        "Closing sentence.",
        "Extra candidate sentence.",
    ]
    sentence_map = {
        "1": [all_aspects[0]],
        "3": [all_aspects[1]],
    }
    return {
        "hypothesis": "Vitamin treatment improves the measured outcome.",
        "paper_as_candidate_pool": sentences,
        "aspect_list_ids": all_aspects,
        "results_aspect_list_ids": results_aspects,
        "aspect2sentence_indices": {
            all_aspects[0]: [1],
            all_aspects[1]: [3],
        },
        "sentence_index2aspects": sentence_map,
        "evidence_retrieval_at_optimal_evaluation": {
            "optimal": 2,
            "one_selection_of_sentences": [1, 3],
            "covered_aspects": all_aspects,
        },
        "evidence_retrieval_at_10_evaluation": {
            "one_selection_of_sentences": [1, 3, 0, 2, 4, 5, 6, 7, 8, 9],
            "covered_aspects": all_aspects,
        },
        "results_evidence_retrieval_at_optimal_evaluation": (
            {
                "optimal": 1,
                "one_selection_of_sentences": [3],
                "covered_aspects": [all_aspects[1]],
            }
            if results
            else None
        ),
        "results_evidence_retrieval_at_5_evaluation": (
            {
                "one_selection_of_sentences": [3, 0, 1, 2, 4],
                "covered_aspects": [all_aspects[1]],
            }
            if results
            else None
        ),
        "sentence_types_in_candidate_pool": ["normal_paragraph"] * len(sentences),
        "paper_id": f"paper-{index}",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def synthetic_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    development_path = tmp_path / "dev.json"
    test_path = tmp_path / "test.json"
    development = {
        "evidencebench_dev_id_0": _question("dev", 0),
        "evidencebench_dev_id_1": _question("dev", 1, results=False),
    }
    test = {
        "evidencebench_test_id_0": _question("test", 0),
        "evidencebench_test_id_1": _question("test", 1, results=False),
    }
    _write_json(development_path, development)
    _write_json(test_path, test)
    monkeypatch.setattr(eb, "EVIDENCEBENCH_DEV_SHA256", sha256_file(development_path))
    monkeypatch.setattr(eb, "EVIDENCEBENCH_TEST_SHA256", sha256_file(test_path))
    monkeypatch.setattr(eb, "EVIDENCEBENCH_DEV_ROWS", 2)
    monkeypatch.setattr(eb, "EVIDENCEBENCH_TEST_ROWS", 2)
    monkeypatch.setattr(eb, "_protocol_config_sha256", lambda: "b" * 64)
    monkeypatch.setattr(eb, "_implementation_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        eb,
        "_runtime_versions",
        lambda: {
            "numpy": "test",
            "pydantic": "test",
            "python": "test",
            "scikit-learn": "test",
            "scipy": "test",
        },
    )
    return {"development": development_path, "test": test_path}


def _run_stages(paths: dict[str, Path]):
    test_sha = sha256_file(paths["test"])
    plan = prepare_evidencebench_plan(
        development_path=paths["development"],
        expected_test_sha256=test_sha,
        expected_test_rows=2,
        bootstrap_replicates=1000,
    )
    visible, gold, materialization = materialize_evidencebench_test(
        plan=plan,
        raw_test_path=paths["test"],
    )
    predictions, prediction_receipt = predict_evidencebench_test(
        plan=plan,
        visible=visible,
        materialization_receipt=materialization,
    )
    summary = score_evidencebench_test(
        plan=plan,
        gold=gold,
        materialization_receipt=materialization,
        predictions=predictions,
        prediction_receipt=prediction_receipt,
    )
    return plan, visible, gold, materialization, predictions, prediction_receipt, summary


def test_prepare_api_cannot_receive_test_path_and_excludes_empty_results(
    synthetic_pins: dict[str, Path],
) -> None:
    assert "test_path" not in inspect.signature(prepare_evidencebench_plan).parameters
    plan = prepare_evidencebench_plan(
        development_path=synthetic_pins["development"],
        expected_test_sha256=sha256_file(synthetic_pins["test"]),
        expected_test_rows=2,
        bootstrap_replicates=1000,
    )
    assert plan.test_labels_opened_by_prepare is False
    assert {row.results_metric_question_count for row in plan.development_results} == {1}
    serialized = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    assert "Vitamin treatment" not in serialized
    assert "paper-0" not in serialized


def test_staged_run_is_deterministic_and_public_summary_is_aggregate_only(
    synthetic_pins: dict[str, Path],
) -> None:
    first = _run_stages(synthetic_pins)
    second = _run_stages(synthetic_pins)

    def canonical(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return [canonical(row) for row in value]

    assert [canonical(value) for value in first] == [
        canonical(value) for value in second
    ]
    summary = first[-1]
    assert summary.question_count == 2
    assert summary.results_metric_question_count == 1
    assert summary.output_is_aggregate_only is True
    assert len(summary.selected_method_paired_deltas) == 5
    assert {
        row.comparator_method_id for row in summary.selected_method_paired_deltas
    } == {
        "bm25-v1",
        "tfidf-word-v1",
        "tfidf-char-v1",
        "rrf-word-char-v1",
        "first-sentences-control-v1",
        "deterministic-random-control-v1",
    } - {summary.selected_method_id}
    public_payload = summary.model_dump(mode="json")
    serialized = json.dumps(public_payload, sort_keys=True)
    for forbidden in (
        "Vitamin treatment",
        "paper-0",
        "evidencebench_test_id_0",
        "aspect_0",
    ):
        assert forbidden not in serialized

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for item in value.values() for key in keys(item)}
        if isinstance(value, list):
            return {key for item in value for key in keys(item)}
        return set()

    assert {"candidate_sentences", "sentence_index_to_aspects"}.isdisjoint(
        keys(public_payload)
    )


def test_visible_projection_and_private_gold_are_disjoint(
    synthetic_pins: dict[str, Path],
) -> None:
    _plan, visible, gold, receipt, *_rest = _run_stages(synthetic_pins)
    visible_payload = json.dumps(
        [row.model_dump(mode="json") for row in visible], sort_keys=True
    )
    gold_payload = json.dumps([row.model_dump(mode="json") for row in gold], sort_keys=True)
    assert "aspect_ids" not in visible_payload
    assert "sentence_index_to_aspects" not in visible_payload
    assert "Vitamin treatment" not in gold_payload
    assert "candidate_sentences" not in gold_payload
    assert receipt.visible_projection_contains_gold_labels is False
    assert receipt.private_gold_contains_candidate_text is False


def test_prediction_tamper_and_out_of_range_index_fail_closed(
    synthetic_pins: dict[str, Path],
) -> None:
    plan, _visible, gold, receipt, predictions, prediction_receipt, _summary = (
        _run_stages(synthetic_pins)
    )
    tampered = [row.model_copy(deep=True) for row in predictions]
    tampered[0].method_rankings["bm25-v1"] = [999]
    with pytest.raises(EvidenceBenchDiagnosticError, match="predictions_mismatch"):
        score_evidencebench_test(
            plan=plan,
            gold=gold,
            materialization_receipt=receipt,
            predictions=tampered,
            prediction_receipt=prediction_receipt,
        )
    forged_payload = [row.model_dump(mode="json") for row in tampered]
    forged_receipt_payload = prediction_receipt.model_dump(
        mode="json", exclude={"receipt_sha256"}
    )
    forged_receipt_payload["predictions_sha256"] = eb.hash_canonical(forged_payload)
    forged_receipt = eb.EvidenceBenchPredictionReceiptV1.model_validate(
        {
            **forged_receipt_payload,
            "receipt_sha256": eb.hash_canonical(forged_receipt_payload),
        }
    )
    with pytest.raises(EvidenceBenchDiagnosticError, match="index_out_of_range"):
        score_evidencebench_test(
            plan=plan,
            gold=gold,
            materialization_receipt=receipt,
            predictions=tampered,
            prediction_receipt=forged_receipt,
        )


def test_materialization_writer_preflights_aliases_and_existing_outputs(
    synthetic_pins: dict[str, Path], tmp_path: Path
) -> None:
    _plan, visible, gold, receipt, *_rest = _run_stages(synthetic_pins)
    alias = tmp_path / "alias.json"
    with pytest.raises(EvidenceBenchDiagnosticError, match="output_paths_alias"):
        write_evidencebench_materialization(
            visible_path=alias,
            gold_path=alias,
            receipt_path=tmp_path / "receipt.json",
            visible=visible,
            gold=gold,
            receipt=receipt,
        )
    existing = tmp_path / "existing.json"
    existing.write_text("owned", encoding="utf-8")
    untouched = tmp_path / "untouched.json"
    with pytest.raises(OutputExistsError):
        write_evidencebench_materialization(
            visible_path=untouched,
            gold_path=existing,
            receipt_path=tmp_path / "receipt.json",
            visible=visible,
            gold=gold,
            receipt=receipt,
        )
    assert not untouched.exists()
    assert existing.read_text(encoding="utf-8") == "owned"


def test_public_summary_tamper_is_rejected(synthetic_pins: dict[str, Path]) -> None:
    summary = _run_stages(synthetic_pins)[-1]
    payload = summary.model_dump(mode="json")
    payload["question_count"] = 1
    with pytest.raises(ValidationError):
        EvidenceBenchPublicSummaryV1.model_validate(payload)


def test_rehashed_duplicate_method_rows_and_denominator_tamper_are_rejected(
    synthetic_pins: dict[str, Path],
) -> None:
    summary = _run_stages(synthetic_pins)[-1]
    duplicate_payload = summary.model_dump(mode="json", exclude={"summary_sha256"})
    duplicate_payload["test_results"].append(duplicate_payload["test_results"][0])
    with pytest.raises(ValidationError, match="method_results_incomplete"):
        EvidenceBenchPublicSummaryV1.model_validate(
            {
                **duplicate_payload,
                "summary_sha256": eb.hash_canonical(duplicate_payload),
            }
        )

    plan = _run_stages(synthetic_pins)[0]
    plan_payload = plan.model_dump(mode="json", exclude={"plan_sha256"})
    plan_payload["development_results"].append(
        plan_payload["development_results"][0]
    )
    with pytest.raises(ValidationError, match="development_results_incomplete"):
        eb.EvidenceBenchFrozenPlanV1.model_validate(
            {**plan_payload, "plan_sha256": eb.hash_canonical(plan_payload)}
        )

    denominator_payload = summary.model_dump(
        mode="json", exclude={"summary_sha256"}
    )
    denominator_payload["test_results"][0]["results_metric_question_count"] = 2
    with pytest.raises(ValidationError, match="results_denominator_mismatch"):
        EvidenceBenchPublicSummaryV1.model_validate(
            {
                **denominator_payload,
                "summary_sha256": eb.hash_canonical(denominator_payload),
            }
        )


def test_ranker_ignores_gold_because_visible_contract_has_no_gold_fields() -> None:
    question = eb.EvidenceBenchVisibleQuestionV1(
        question_id="evidencebench_test_id_0",
        paper_id="paper",
        hypothesis="treatment improves outcome",
        candidate_sentences=["unrelated", "treatment improves outcome"],
        sentence_types=["normal_paragraph", "normal_paragraph"],
    )
    first = rank_question(question)
    second = rank_question(question)
    assert first == second
    assert first["tfidf-word-v1"][0] == 1
    assert set(first) == {
        "bm25-v1",
        "tfidf-word-v1",
        "tfidf-char-v1",
        "rrf-word-char-v1",
        "first-sentences-control-v1",
        "deterministic-random-control-v1",
    }


def test_prediction_model_requires_complete_frozen_roster() -> None:
    with pytest.raises(ValidationError, match="prediction_methods_incomplete"):
        EvidenceBenchPredictionRowV1(
            question_id="evidencebench_test_id_0",
            method_rankings={"bm25-v1": [0]},
        )


def test_environment_drift_fails_before_test_file_is_parsed(
    synthetic_pins: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = prepare_evidencebench_plan(
        development_path=synthetic_pins["development"],
        expected_test_sha256=sha256_file(synthetic_pins["test"]),
        expected_test_rows=2,
        bootstrap_replicates=1000,
    )
    monkeypatch.setattr(eb, "_implementation_sha256", lambda: "c" * 64)

    def forbidden_reader(_path: Path) -> dict[str, Any]:
        raise AssertionError("test file was parsed before environment validation")

    monkeypatch.setattr(eb, "_load_json_object", forbidden_reader)
    with pytest.raises(EvidenceBenchDiagnosticError, match="implementation_drift"):
        materialize_evidencebench_test(
            plan=plan,
            raw_test_path=synthetic_pins["test"],
        )


def test_protocol_config_drift_fails_before_test_file_is_parsed(
    synthetic_pins: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = prepare_evidencebench_plan(
        development_path=synthetic_pins["development"],
        expected_test_sha256=sha256_file(synthetic_pins["test"]),
        expected_test_rows=2,
        bootstrap_replicates=1000,
    )
    monkeypatch.setattr(eb, "_protocol_config_sha256", lambda: "c" * 64)

    def forbidden_reader(_path: Path) -> dict[str, Any]:
        raise AssertionError("test file was parsed before config validation")

    monkeypatch.setattr(eb, "_load_json_object", forbidden_reader)
    with pytest.raises(EvidenceBenchDiagnosticError, match="protocol_config_drift"):
        materialize_evidencebench_test(
            plan=plan,
            raw_test_path=synthetic_pins["test"],
        )


def test_oracle_coverage_and_forward_inverse_mapping_are_recomputed(
    synthetic_pins: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(synthetic_pins["development"].read_text(encoding="utf-8"))
    row = payload["evidencebench_dev_id_0"]
    row["evidence_retrieval_at_10_evaluation"]["covered_aspects"] = []
    _write_json(synthetic_pins["development"], payload)
    monkeypatch.setattr(
        eb,
        "EVIDENCEBENCH_DEV_SHA256",
        sha256_file(synthetic_pins["development"]),
    )
    with pytest.raises(EvidenceBenchDiagnosticError, match="oracle_coverage_mismatch"):
        prepare_evidencebench_plan(
            development_path=synthetic_pins["development"],
            expected_test_sha256=sha256_file(synthetic_pins["test"]),
            expected_test_rows=2,
            bootstrap_replicates=1000,
        )


def test_score_cli_rejects_bad_prediction_receipt_before_opening_gold(
    synthetic_pins: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _visible, _gold, materialization, predictions, receipt, _summary = (
        _run_stages(synthetic_pins)
    )
    predictions_path = tmp_path / "predictions.json"
    receipt_path = tmp_path / "prediction-receipt.json"
    gold_path = tmp_path / "private-gold.json"
    _write_json(
        predictions_path, [row.model_dump(mode="json") for row in predictions]
    )
    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload["receipt_sha256"] = "0" * 64
    _write_json(receipt_path, receipt_payload)
    gold_path.write_text("this must never be parsed", encoding="utf-8")

    def forbidden_gold_reader(_path: Path):
        raise AssertionError("private gold opened before prediction freeze validation")

    monkeypatch.setattr(eb_cli, "_gold", forbidden_gold_reader)
    args = SimpleNamespace(
        prediction_receipt=receipt_path,
        predictions=predictions_path,
        gold=gold_path,
    )
    with pytest.raises(ValidationError, match="prediction_receipt_hash_mismatch"):
        eb_cli._score_stage(
            args,
            plan=plan,
            materialization_receipt=materialization,
        )


def test_exact_replay_audit_receipt_binds_public_summary(
    synthetic_pins: dict[str, Path],
) -> None:
    plan, visible, gold, materialization, predictions, prediction_receipt, summary = (
        _run_stages(synthetic_pins)
    )
    receipt = audit_evidencebench_run(
        development_path=synthetic_pins["development"],
        raw_test_path=synthetic_pins["test"],
        plan=plan,
        visible=visible,
        gold=gold,
        materialization_receipt=materialization,
        predictions=predictions,
        prediction_receipt=prediction_receipt,
        summary=summary,
    )
    assert receipt.exact_replay_status == "passed"
    assert receipt.summary_sha256 == summary.summary_sha256
    assert receipt.test_gold_opened_only_after_prediction_freeze_validation is True
    validated_summary, validated_receipt = eb.validate_evidencebench_public_bundle(
        summary=summary,
        audit_receipt=receipt,
    )
    assert validated_summary.summary_sha256 == summary.summary_sha256
    assert validated_receipt.receipt_sha256 == receipt.receipt_sha256

    forged_payload = receipt.model_dump(mode="json", exclude={"receipt_sha256"})
    forged_payload["summary_sha256"] = "0" * 64
    forged_receipt = eb.EvidenceBenchReplayAuditReceiptV1.model_validate(
        {
            **forged_payload,
            "receipt_sha256": eb.hash_canonical(forged_payload),
        }
    )
    with pytest.raises(EvidenceBenchDiagnosticError, match="binding_mismatch"):
        eb.validate_evidencebench_public_bundle(
            summary=summary,
            audit_receipt=forged_receipt,
        )


def test_exact_replay_rejects_rehashed_summary_tamper(
    synthetic_pins: dict[str, Path],
) -> None:
    plan, visible, gold, materialization, predictions, prediction_receipt, summary = (
        _run_stages(synthetic_pins)
    )
    payload = summary.model_dump(mode="json", exclude={"summary_sha256"})
    payload["oracle_upper_bounds"]["all_aspect_recall_at_10"] = 0.5
    forged = EvidenceBenchPublicSummaryV1.model_validate(
        {**payload, "summary_sha256": eb.hash_canonical(payload)}
    )
    with pytest.raises(EvidenceBenchDiagnosticError, match="replay_mismatch:summary"):
        audit_evidencebench_run(
            development_path=synthetic_pins["development"],
            raw_test_path=synthetic_pins["test"],
            plan=plan,
            visible=visible,
            gold=gold,
            materialization_receipt=materialization,
            predictions=predictions,
            prediction_receipt=prediction_receipt,
            summary=forged,
        )


def test_full_staged_cli_synthetic_smoke(
    synthetic_pins: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_prepare = prepare_evidencebench_plan

    def synthetic_prepare(**kwargs: Any):
        return real_prepare(
            **kwargs,
            expected_test_sha256=sha256_file(synthetic_pins["test"]),
            expected_test_rows=2,
        )

    monkeypatch.setattr(eb_cli, "prepare_evidencebench_plan", synthetic_prepare)
    plan = tmp_path / "run" / "plan.json"
    visible = tmp_path / "run" / "visible.json"
    gold = tmp_path / "run" / "gold.json"
    materialization = tmp_path / "run" / "materialization.json"
    predictions = tmp_path / "run" / "predictions.json"
    prediction_receipt = tmp_path / "run" / "prediction.json"
    summary = tmp_path / "public" / "summary.json"
    audit_receipt = tmp_path / "public" / "audit.json"

    stages = [
        [
            "prepare",
            "--development",
            str(synthetic_pins["development"]),
            "--plan",
            str(plan),
            "--bootstrap-replicates",
            "1000",
        ],
        [
            "materialize",
            "--plan",
            str(plan),
            "--test",
            str(synthetic_pins["test"]),
            "--visible",
            str(visible),
            "--gold",
            str(gold),
            "--receipt",
            str(materialization),
        ],
        [
            "predict",
            "--plan",
            str(plan),
            "--materialization-receipt",
            str(materialization),
            "--visible",
            str(visible),
            "--predictions",
            str(predictions),
            "--receipt",
            str(prediction_receipt),
        ],
        [
            "score",
            "--plan",
            str(plan),
            "--materialization-receipt",
            str(materialization),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--prediction-receipt",
            str(prediction_receipt),
            "--output",
            str(summary),
        ],
        [
            "audit",
            "--development",
            str(synthetic_pins["development"]),
            "--test",
            str(synthetic_pins["test"]),
            "--plan",
            str(plan),
            "--visible",
            str(visible),
            "--gold",
            str(gold),
            "--materialization-receipt",
            str(materialization),
            "--predictions",
            str(predictions),
            "--prediction-receipt",
            str(prediction_receipt),
            "--summary",
            str(summary),
            "--receipt",
            str(audit_receipt),
        ],
    ]
    for stage in stages:
        monkeypatch.setattr(
            sys, "argv", ["run_evidencebench_diagnostic.py", *stage]
        )
        eb_cli.main()
        capsys.readouterr()

    public_summary = EvidenceBenchPublicSummaryV1.model_validate(
        json.loads(summary.read_text(encoding="utf-8"))
    )
    replay_receipt = eb.EvidenceBenchReplayAuditReceiptV1.model_validate(
        json.loads(audit_receipt.read_text(encoding="utf-8"))
    )
    assert replay_receipt.summary_sha256 == public_summary.summary_sha256
    with pytest.raises(OutputExistsError):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_evidencebench_diagnostic.py",
                "score",
                "--plan",
                str(plan),
                "--materialization-receipt",
                str(materialization),
                "--gold",
                str(gold),
                "--predictions",
                str(predictions),
                "--prediction-receipt",
                str(prediction_receipt),
                "--output",
                str(summary),
            ],
        )
        eb_cli.main()
