"""Contracts for staged, protocol-aware MetaSyn screening reranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import literature_multiverse.metasyn_screening_study as study
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_benchmark import prepare_metasyn_benchmark
from literature_multiverse.metasyn_screening_study import (
    BASELINE_ID,
    FEATURE_NAMES,
    MetaSynScreeningStudyError,
    _point_metrics,
    _ScreeningLabel,
    _validate_feature_rows,
    _validate_public_redaction,
    deterministic_rerank,
    evaluate_frozen_winner,
    fit_and_freeze_winner,
    prepare_label_blind_features,
    validate_fit,
    validate_prepare,
    validate_public_summary,
)


def _review_row(review_id: int) -> dict[str, Any]:
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
        "matched_corpus_ids": [review_id],
        "matched_ref_count": 1,
        "study_count": 1.0,
        "source_review_corpus_ids": [1_000 + review_id],
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    review_cache = tmp_path / "reviews"
    review_cache.mkdir()
    train_path = review_cache / "reviews-train.parquet"
    test_path = review_cache / "reviews-test.parquet"
    pd.DataFrame([_review_row(index) for index in range(1, 31)]).to_parquet(
        train_path, index=False
    )
    pd.DataFrame([_review_row(index) for index in range(100, 102)]).to_parquet(
        test_path, index=False
    )
    benchmark_dir = tmp_path / "benchmark"
    manifest = prepare_metasyn_benchmark(
        train_parquet=train_path,
        test_parquet=test_path,
        output_dir=benchmark_dir,
        seed=20_260_827,
        calibration_fraction=0.5,
    )
    assert manifest.development.rows >= 5
    assert manifest.calibration.rows >= 5

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_path = corpus_dir / "part.parquet"
    pd.DataFrame(
        [
            {
                "ID": index,
                "title": f"therapy {index} trial",
                "abstract": f"population {index} outcome {index} compared with control",
            }
            for index in range(1_105)
        ]
    ).to_parquet(corpus_path, index=False)
    license_path = tmp_path / "LICENSE.fixture"
    license_path.write_text("fixture only\n", encoding="utf-8")
    corpus_manifest = tmp_path / "corpus-manifest.json"
    atomic_write_json(
        corpus_manifest,
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
        },
    )

    retrieval_dir = tmp_path / "retrieval"
    ranking_path = retrieval_dir / "ordered_rankings.private.jsonl"
    included_review_ids = sorted(
        [*manifest.development.review_ids, *manifest.calibration.review_ids]
    )
    source_order = list(range(1_000))
    atomic_write_jsonl(
        ranking_path,
        [
            {
                "review_id": review_id,
                "bm25_source_ranking": source_order,
                "tfidf_source_ranking": [*source_order[1:], source_order[0]],
                "rrf_output_ranking": source_order[:200],
            }
            for review_id in included_review_ids
        ],
    )
    retrieval_payload = {
        "metasyn_retrieval_candidate_freeze_version": "1",
        "stage": "label_blind_candidate_freeze",
        "labels_read_by_freeze_stage": False,
        "official_test_model_inputs_opened": False,
        "official_test_evaluated": False,
        "benchmark_manifest_sha256": sha256_file(benchmark_dir / "manifest.json"),
        "corpus_manifest_sha256": sha256_file(corpus_manifest),
        "ordered_rankings_path": ranking_path.relative_to(retrieval_dir).as_posix(),
        "ordered_rankings_sha256": sha256_file(ranking_path),
        "freeze_payload_sha256": "placeholder",
        "candidates": {
            "rrf-tfidf-bm25-fixed-v1": {"config": {"evaluated_depth": 200}}
        },
    }
    retrieval_payload["freeze_payload_sha256"] = hash_canonical(
        {
            key: value
            for key, value in retrieval_payload.items()
            if key != "freeze_payload_sha256"
        }
    )
    atomic_write_json(retrieval_dir / "freeze_receipt.json", retrieval_payload)
    return {
        "benchmark": benchmark_dir / "manifest.json",
        "benchmark_dir": benchmark_dir,
        "corpus_manifest": corpus_manifest,
        "repository_root": tmp_path,
        "retrieval": retrieval_dir,
        "review_cache": review_cache,
        "test_parquet": test_path,
        "work": tmp_path / "screening",
        "public": tmp_path / "public.json",
    }


def test_full_staged_study_respects_access_boundaries_and_public_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    manifest = json.loads(paths["benchmark"].read_text(encoding="utf-8"))
    # Neither feature preparation nor subsequent label stages are allowed to depend
    # on the official-test file or the shared all-split evaluator artifact.
    for path in (
        paths["benchmark_dir"] / manifest["test"]["path"],
        paths["benchmark_dir"] / manifest["evaluator_labels"]["path"],
        paths["test_parquet"],
    ):
        path.rename(path.with_suffix(path.suffix + ".inaccessible"))

    prepare = prepare_label_blind_features(
        benchmark_manifest_path=paths["benchmark"],
        corpus_manifest_path=paths["corpus_manifest"],
        repository_root=paths["repository_root"],
        retrieval_work_dir=paths["retrieval"],
        work_dir=paths["work"],
    )
    assert prepare["access_boundary"]["development_labels_opened"] is False
    assert prepare["access_boundary"]["calibration_labels_opened"] is False
    feature_rendered = (paths["work"] / "pair_features.private.jsonl").read_text(
        encoding="utf-8"
    )
    for forbidden in ("Effect_Direction", "matched_corpus_ids", "Conclusion_Summary"):
        assert forbidden not in feature_rendered

    observed_splits: list[str] = []
    original_loader = study._load_split_labels_from_train

    def recording_loader(**kwargs: Any) -> Any:
        observed_splits.append(kwargs["split"])
        return original_loader(**kwargs)

    monkeypatch.setattr(study, "_load_split_labels_from_train", recording_loader)
    with pytest.raises(
        MetaSynScreeningStudyError, match="fit_configuration_differs_from_prepare_freeze"
    ):
        fit_and_freeze_winner(
            benchmark_manifest_path=paths["benchmark"],
            review_cache_dir=paths["review_cache"],
            work_dir=paths["work"],
            cv_folds=3,
        )
    fit = fit_and_freeze_winner(
        benchmark_manifest_path=paths["benchmark"],
        review_cache_dir=paths["review_cache"],
        work_dir=paths["work"],
    )
    assert observed_splits == ["development"]
    assert fit["access_boundary"]["calibration_labels_materialized"] is False
    assert fit["selected_candidate"] in study.CANDIDATE_IDS
    for fold in fit["cv_folds"]:
        assert not set(fold["training_component_ids"]) & set(
            fold["validation_component_ids"]
        )

    summary = evaluate_frozen_winner(
        benchmark_manifest_path=paths["benchmark"],
        review_cache_dir=paths["review_cache"],
        repository_root=paths["repository_root"],
        work_dir=paths["work"],
        public_summary_path=paths["public"],
        bootstrap_replicates=10_000,
    )
    assert observed_splits == ["development", "calibration"]
    assert summary["protocol"]["official_test_inputs_opened_by_this_study"] is False
    assert summary["protocol"]["official_test_labels_opened_by_this_study"] is False
    assert summary["protocol"]["official_test_evaluated"] is False
    assert summary["network_calls"] == 0
    assert summary["provider_calls"] == 0
    assert (
        summary["interpretation_limits"][
            "official_test_labels_historically_opened_elsewhere_in_repository"
        ]
        is True
    )
    assert summary["public_summary_payload_sha256"] == hash_canonical(
        {
            key: value
            for key, value in summary.items()
            if key != "public_summary_payload_sha256"
        }
    )
    assert set(summary["lineage"]["source_code_sha256s"]) == {
        "pyproject.toml",
        "scripts/run_metasyn_screening_study.py",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/calibration.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/metasyn_benchmark.py",
        "src/literature_multiverse/metasyn_retrieval.py",
        "src/literature_multiverse/metasyn_screening_study.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/paths.py",
        "uv.lock",
    }
    validate_public_summary(
        repository_root=paths["repository_root"],
        work_dir=paths["work"],
        public_summary_path=paths["public"],
    )
    rendered = paths["public"].read_text(encoding="utf-8")
    assert "Does therapy" not in rendered
    assert "metasyn-review-" not in rendered
    assert tmp_path.as_posix() not in rendered


def test_feature_winner_and_public_tampering_fail_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    prepare_label_blind_features(
        benchmark_manifest_path=paths["benchmark"],
        corpus_manifest_path=paths["corpus_manifest"],
        repository_root=paths["repository_root"],
        retrieval_work_dir=paths["retrieval"],
        work_dir=paths["work"],
    )
    feature_path = paths["work"] / "pair_features.private.jsonl"
    original_features = feature_path.read_bytes()
    feature_path.write_bytes(original_features + b"\n")
    with pytest.raises(MetaSynScreeningStudyError, match="feature_rows_hash_mismatch"):
        validate_prepare(work_dir=paths["work"])
    feature_path.write_bytes(original_features)

    fit_and_freeze_winner(
        benchmark_manifest_path=paths["benchmark"],
        review_cache_dir=paths["review_cache"],
        work_dir=paths["work"],
    )
    winner_path = paths["work"] / "winner_rankings.private.jsonl"
    original_winner = winner_path.read_bytes()
    winner_path.write_bytes(original_winner + b"\n")
    with pytest.raises(MetaSynScreeningStudyError, match="winner_rankings_hash_mismatch"):
        validate_fit(work_dir=paths["work"])
    winner_path.write_bytes(original_winner)

    evaluate_frozen_winner(
        benchmark_manifest_path=paths["benchmark"],
        review_cache_dir=paths["review_cache"],
        repository_root=paths["repository_root"],
        work_dir=paths["work"],
        public_summary_path=paths["public"],
        bootstrap_replicates=10_000,
    )
    public = json.loads(paths["public"].read_text(encoding="utf-8"))
    public["status"] = "tampered"
    public["public_summary_payload_sha256"] = hash_canonical(
        {
            key: value
            for key, value in public.items()
            if key != "public_summary_payload_sha256"
        }
    )
    paths["public"].write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(MetaSynScreeningStudyError, match="content_mismatch"):
        validate_public_summary(
            repository_root=paths["repository_root"],
            work_dir=paths["work"],
            public_summary_path=paths["public"],
        )


def test_deterministic_reranking_uses_corpus_id_tie_break() -> None:
    rows = [
        {
            "corpus_id": corpus_id,
            "original_rrf_rank": rank,
            "features": {name: 0.0 for name in FEATURE_NAMES},
        }
        for rank, corpus_id in enumerate(reversed(range(200)), start=1)
    ]

    first = deterministic_rerank(rows, [0.5] * 200)
    second = deterministic_rerank(rows, [0.5] * 200)

    assert first == list(range(200))
    assert second == first


def test_denominators_include_zero_retrieval_question_in_absolute_macro() -> None:
    baseline = {1: list(range(200)), 2: list(range(200))}
    selected = {1: list(range(200)), 2: list(range(200))}
    labels = [
        _ScreeningLabel(review_id=1, component_id="a", gold_corpus_ids=(500,)),
        _ScreeningLabel(review_id=2, component_id="b", gold_corpus_ids=(1,)),
    ]

    metrics = _point_metrics(
        ranking=selected, baseline=baseline, labels=labels, depth=10
    )

    assert metrics["question_macro_absolute_recall"] == 0.5
    assert metrics["micro_absolute_recall"] == 0.5
    assert metrics["questions_with_zero_retrievable_matched_references"] == 1
    assert metrics["conditional_questions"] == 1
    assert metrics["question_macro_conditional_survival"] == 1.0


def test_missing_feature_rows_and_public_identifier_leak_are_rejected(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "split": "development",
            "review_id": 1,
            "corpus_id": rank,
            "original_rrf_rank": rank,
            "features": {name: 0.0 for name in FEATURE_NAMES},
        }
        for rank in range(1, 200)
    ]
    with pytest.raises(MetaSynScreeningStudyError, match="ranking_depth_or_rank_invalid"):
        _validate_feature_rows(
            rows, expected_by_split={"development": 1, "calibration": 0}
        )
    with pytest.raises(MetaSynScreeningStudyError, match="forbidden_key:question_id"):
        _validate_public_redaction(
            {"question_id": "metasyn-review-000001"}, repository_root=tmp_path
        )


def test_baseline_is_in_prespecified_candidate_family() -> None:
    assert BASELINE_ID in study.CANDIDATE_IDS
    assert set(study._candidate_configs()) == set(study.CANDIDATE_IDS)
