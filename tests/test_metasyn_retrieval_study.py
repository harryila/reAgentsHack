"""Staged MetaSyn retrieval selection and calibration contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import literature_multiverse.metasyn_retrieval_study as study
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_benchmark import prepare_metasyn_benchmark
from literature_multiverse.metasyn_retrieval_study import (
    CANDIDATE_IDS,
    MetaSynRetrievalStudyError,
    freeze_candidate_predictions,
    reciprocal_rank_fusion,
    run_retrieval_study,
    validate_candidate_freeze,
)


def _review_row(review_id: int, *, gold_id: int, source_id: int) -> dict[str, Any]:
    return {
        "ID": review_id,
        "Title": f"Fixture review {review_id}",
        "Abstract": "Evaluator-only review abstract",
        "Population": f"population {review_id}",
        "Intervention": f"therapy {review_id}",
        "Exposure": None,
        "Comparison": "control",
        "Outcome": f"outcome {review_id}",
        "Effect_Direction": "Positive",
        "Research_Question": f"Does therapy {review_id} improve outcome {review_id}?",
        "matched_corpus_ids": [gold_id],
        "matched_ref_count": 1,
        "study_count": 1.0,
        "source_review_corpus_ids": [source_id],
    }


def _study_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    review_cache = tmp_path / "reviews"
    review_cache.mkdir()
    train = review_cache / "reviews-train.parquet"
    test = review_cache / "reviews-test.parquet"
    pd.DataFrame(
        [_review_row(index, gold_id=index, source_id=1_050 + index) for index in range(1, 13)]
    ).to_parquet(train, index=False)
    pd.DataFrame(
        [
            _review_row(100 + index, gold_id=100 + index, source_id=1_080 + index)
            for index in range(2)
        ]
    ).to_parquet(test, index=False)
    benchmark_dir = tmp_path / "benchmark"
    prepare_metasyn_benchmark(
        train_parquet=train,
        test_parquet=test,
        output_dir=benchmark_dir,
        seed=1,
        calibration_fraction=0.5,
    )

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_path = corpus_dir / "part.parquet"
    pd.DataFrame(
        [
            {
                "ID": index,
                "title": f"therapy {index} controlled trial",
                "abstract": f"population {index} outcome {index} control",
            }
            for index in range(1_105)
        ]
    ).to_parquet(corpus_path, index=False)
    license_path = tmp_path / "LICENSE.fixture"
    license_path.write_text("fixture only\n", encoding="utf-8")
    corpus_manifest = tmp_path / "corpus-manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "corpus_manifest_version": "1",
                "dataset": "MetaSyn corpus",
                "source_repository": "fixture/metasyn",
                "source_revision": "a" * 40,
                "local_root": "corpus",
                "license_notice": {
                    "path": "LICENSE.fixture",
                    "sha256": sha256_file(license_path),
                    "status": "local_evaluation_only_third_party_terms_apply",
                },
                "shards": [
                    {
                        "path": corpus_path.name,
                        "rows": 1_105,
                        "sha256": sha256_file(corpus_path),
                    }
                ],
                "total_rows": 1_105,
            }
        ),
        encoding="utf-8",
    )
    return benchmark_dir / "manifest.json", corpus_manifest, review_cache


def test_reciprocal_rank_fusion_is_ordered_and_tie_stable() -> None:
    fused = reciprocal_rank_fusion(
        {
            "a": {1: [2, 1, 4]},
            "b": {1: [1, 2, 3]},
        },
        top_k=3,
        rank_constant=60,
    )

    assert fused == {1: [1, 2, 3]}


def test_source_lineage_binds_complete_direct_runtime_surface() -> None:
    assert set(study._source_code_hashes()) == {
        "pyproject.toml",
        "scripts/run_metasyn_retrieval_study.py",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/calibration.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/metasyn_benchmark.py",
        "src/literature_multiverse/metasyn_retrieval.py",
        "src/literature_multiverse/metasyn_retrieval_study.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/paths.py",
        "uv.lock",
    }


def test_freeze_does_not_require_evaluator_labels_or_test_inputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    benchmark_manifest, corpus_manifest, review_cache = _study_fixture(tmp_path)
    manifest = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
    for field in ("evaluator_labels", "test"):
        path = benchmark_manifest.parent / manifest[field]["path"]
        path.rename(path.with_suffix(".inaccessible"))

    def fake_rank(*, questions: Any, exclusions: Any, top_k: int, **_: Any) -> Any:
        rankings = {}
        for question in questions:
            eligible = [
                value for value in range(1_105) if value not in exclusions[question.review_id]
            ]
            rankings[question.review_id] = eligible[:top_k]
        return rankings, {"documents": 1_105, "fixture": True}

    monkeypatch.setattr("literature_multiverse.metasyn_retrieval_study.tfidf_rank", fake_rank)
    monkeypatch.setattr("literature_multiverse.metasyn_retrieval_study.bm25_rank", fake_rank)
    receipt = freeze_candidate_predictions(
        benchmark_manifest_path=benchmark_manifest,
        corpus_manifest_path=corpus_manifest,
        repository_root=tmp_path,
        review_cache_dir=review_cache,
        work_dir=tmp_path / "study",
    )

    assert receipt["labels_read_by_freeze_stage"] is False
    assert receipt["official_test_model_inputs_opened"] is False
    assert receipt["official_test_evaluated"] is False
    assert receipt["freeze_payload_sha256"] == hash_canonical(
        {key: value for key, value in receipt.items() if key != "freeze_payload_sha256"}
    )
    monkeypatch.setattr(
        "literature_multiverse.metasyn_retrieval_study._source_code_hashes",
        lambda: {"source": "f" * 64},
    )
    with pytest.raises(MetaSynRetrievalStudyError, match="source_code_drift"):
        validate_candidate_freeze(work_dir=tmp_path / "study")


def test_full_study_selects_on_development_and_scores_one_calibration_candidate(
    tmp_path: Path,
) -> None:
    benchmark_manifest, corpus_manifest, review_cache = _study_fixture(tmp_path)
    work_dir = tmp_path / "study"
    public_summary = tmp_path / "summary.json"

    summary = run_retrieval_study(
        benchmark_manifest_path=benchmark_manifest,
        corpus_manifest_path=corpus_manifest,
        repository_root=tmp_path,
        review_cache_dir=review_cache,
        work_dir=work_dir,
        public_summary_path=public_summary,
        bootstrap_replicates=100,
    )

    assert summary["selection_protocol"]["development_compared_candidates"] == list(CANDIDATE_IDS)
    assert summary["selection_protocol"]["calibration_scored_candidates"] == [
        summary["selection_protocol"]["selected_candidate"]
    ]
    assert summary["selection_protocol"]["official_test_evaluated"] is False
    assert summary["contains_question_text"] is False
    assert summary["contains_article_text"] is False
    assert summary["public_summary_payload_sha256"] == hash_canonical(
        {key: value for key, value in summary.items() if key != "public_summary_payload_sha256"}
    )
    rendered = public_summary.read_text(encoding="utf-8")
    assert "Does therapy" not in rendered
    assert "controlled trial" not in rendered
    calibration = json.loads((work_dir / "calibration_evaluation.json").read_text(encoding="utf-8"))
    assert calibration["calibration_candidate_comparison_computed"] is False
    assert list(calibration["calibration_result"]["candidates"]) == [
        summary["selection_protocol"]["selected_candidate"]
    ]
