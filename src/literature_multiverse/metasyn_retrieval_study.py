"""Staged, leakage-explicit MetaSyn retrieval selection and evaluation.

The study has three irreversible logical stages:

1. freeze fixed, label-blind candidate rankings for development and calibration;
2. select exactly one candidate using development matched-paper labels; and
3. evaluate that frozen selection once on calibration.

The official test split is never scored.  All benchmark labels were opened before this
study was designed, so the protocol improves procedural auditability but cannot turn
calibration into a pristine holdout.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow
import sklearn

from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_benchmark import (
    MetaSynEvaluatorLabel,
    MetaSynPrediction,
    load_metasyn_inputs,
    load_metasyn_manifest_metadata,
    load_metasyn_predictions,
)
from literature_multiverse.metasyn_retrieval import (
    bm25_rank,
    load_source_review_exclusions,
    tfidf_config,
    tfidf_rank,
    verify_corpus_manifest,
)

CandidateId = Literal[
    "bm25-fixed-v1",
    "rrf-tfidf-bm25-fixed-v1",
    "tfidf-fixed-v1",
]

STUDY_VERSION = "1"
OUTPUT_DEPTH = 200
FUSION_SOURCE_DEPTH = 1_000
RRF_RANK_CONSTANT = 60
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_827
CANDIDATE_IDS: tuple[CandidateId, ...] = (
    "bm25-fixed-v1",
    "rrf-tfidf-bm25-fixed-v1",
    "tfidf-fixed-v1",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "abstract",
        "article_id",
        "corpus_id",
        "gold_matched_corpus_ids",
        "outcome",
        "population",
        "question_id",
        "research_question",
        "retrieved_corpus_ids",
        "review_id",
        "title",
    }
)


class MetaSynRetrievalStudyError(ValueError):
    """A staged retrieval artifact or evaluation contract is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynRetrievalStudyError(f"json_artifact_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise MetaSynRetrievalStudyError(f"json_artifact_not_object:{path}")
    return payload


def _attach_payload_hash(payload: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in payload:
        raise MetaSynRetrievalStudyError(f"payload_hash_field_already_present:{field}")
    return {**payload, field: hash_canonical(payload)}


def _verify_payload_hash(payload: Mapping[str, Any], *, field: str) -> None:
    observed = payload.get(field)
    unhashed = {key: value for key, value in payload.items() if key != field}
    if not isinstance(observed, str) or observed != hash_canonical(unhashed):
        raise MetaSynRetrievalStudyError(f"payload_hash_mismatch:{field}")


def _candidate_configs() -> dict[CandidateId, dict[str, Any]]:
    return {
        "bm25-fixed-v1": {
            "algorithm": "streaming_bm25_title_abstract_unigram_bigram_v1",
            "b": 0.75,
            "evaluated_depth": OUTPUT_DEPTH,
            "k1": 1.2,
            "k3": 8.0,
            "ranking_depth": FUSION_SOURCE_DEPTH,
        },
        "rrf-tfidf-bm25-fixed-v1": {
            "algorithm": "deterministic_reciprocal_rank_fusion_v1",
            "evaluated_depth": OUTPUT_DEPTH,
            "input_candidates": ["bm25-fixed-v1", "tfidf-fixed-v1"],
            "input_ranking_depth": FUSION_SOURCE_DEPTH,
            "rank_constant": RRF_RANK_CONSTANT,
            "stable_tie_break": "descending_fusion_score_then_ascending_corpus_id",
        },
        "tfidf-fixed-v1": {
            **tfidf_config(top_k=FUSION_SOURCE_DEPTH),
            "evaluated_depth": OUTPUT_DEPTH,
        },
    }


def reciprocal_rank_fusion(
    rankings: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    top_k: int = OUTPUT_DEPTH,
    rank_constant: int = RRF_RANK_CONSTANT,
) -> dict[int, list[int]]:
    """Fuse complete ordered lists with deterministic reciprocal-rank fusion."""

    if top_k < 1 or rank_constant < 1:
        raise ValueError("invalid_reciprocal_rank_fusion_configuration")
    if len(rankings) < 2:
        raise MetaSynRetrievalStudyError("fusion_requires_at_least_two_rankers")
    methods = sorted(rankings)
    review_sets = [set(rankings[method]) for method in methods]
    if not review_sets or any(review_set != review_sets[0] for review_set in review_sets[1:]):
        raise MetaSynRetrievalStudyError("fusion_review_universe_mismatch")

    fused: dict[int, list[int]] = {}
    for review_id in sorted(review_sets[0]):
        scores: defaultdict[int, float] = defaultdict(float)
        for method in methods:
            ordered = list(rankings[method][review_id])
            if len(ordered) != len(set(ordered)):
                raise MetaSynRetrievalStudyError(
                    f"fusion_input_contains_duplicate_document:{method}"
                )
            for rank, document_id in enumerate(ordered, start=1):
                scores[int(document_id)] += 1.0 / (rank_constant + rank)
        if len(scores) < top_k:
            raise MetaSynRetrievalStudyError("fusion_union_smaller_than_output_depth")
        fused[review_id] = [
            document_id
            for document_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
                :top_k
            ]
        ]
    return fused


def _validate_rankings(
    rankings: Mapping[int, Sequence[int]],
    *,
    expected_review_ids: set[int],
    expected_depth: int,
    exclusions: Mapping[int, set[int]],
) -> None:
    if set(rankings) != expected_review_ids:
        raise MetaSynRetrievalStudyError("ranking_review_universe_mismatch")
    for review_id, ordered in rankings.items():
        if len(ordered) != expected_depth or len(set(ordered)) != expected_depth:
            raise MetaSynRetrievalStudyError("ranking_depth_or_uniqueness_invalid")
        if set(ordered) & exclusions[review_id]:
            raise MetaSynRetrievalStudyError("source_review_exclusion_leaked_into_ranking")


def _prediction_payloads(
    rankings: Mapping[int, Sequence[int]], *, depth: int
) -> list[dict[str, Any]]:
    return [
        MetaSynPrediction(
            review_id=review_id,
            retrieved_corpus_ids=sorted(rankings[review_id][:depth]),
        ).model_dump(mode="json", exclude_none=True)
        for review_id in sorted(rankings)
    ]


def _source_code_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    paths = {
        "pyproject.toml": repository_root / "pyproject.toml",
        "scripts/run_metasyn_retrieval_study.py": (
            repository_root / "scripts/run_metasyn_retrieval_study.py"
        ),
        "src/literature_multiverse/__init__.py": package_root / "__init__.py",
        "src/literature_multiverse/calibration.py": package_root / "calibration.py",
        "src/literature_multiverse/lineage.py": package_root / "lineage.py",
        "src/literature_multiverse/metasyn_benchmark.py": package_root / "metasyn_benchmark.py",
        "src/literature_multiverse/metasyn_retrieval.py": package_root / "metasyn_retrieval.py",
        "src/literature_multiverse/metasyn_retrieval_study.py": Path(__file__).resolve(),
        "src/literature_multiverse/models.py": package_root / "models.py",
        "src/literature_multiverse/paths.py": package_root / "paths.py",
        "uv.lock": repository_root / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def freeze_candidate_predictions(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    review_cache_dir: Path,
    work_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze all candidates without opening the evaluator-label artifact."""

    receipt_path = work_dir / "freeze_receipt.json"
    rankings_path = work_dir / "ordered_rankings.private.jsonl"
    prediction_paths = {
        candidate: work_dir / "predictions" / f"{candidate}.jsonl" for candidate in CANDIDATE_IDS
    }
    targets = [receipt_path, rankings_path, *prediction_paths.values()]
    existing = [path.as_posix() for path in targets if path.exists()]
    if existing and not force:
        raise MetaSynRetrievalStudyError(f"freeze_outputs_exist:{existing}")

    # These two calls open only the named model-facing files.  In particular they do
    # not hash or open the evaluator-label artifact or official-test model inputs.
    development = load_metasyn_inputs(benchmark_manifest_path, split="development")
    calibration = load_metasyn_inputs(benchmark_manifest_path, split="calibration")
    questions = sorted([*development, *calibration], key=lambda item: item.review_id)
    benchmark = load_metasyn_manifest_metadata(benchmark_manifest_path)
    corpus, shard_paths = verify_corpus_manifest(
        corpus_manifest_path, repository_root=repository_root
    )
    exclusions, source_review_hashes = load_source_review_exclusions(
        benchmark_manifest_path=benchmark_manifest_path,
        split="development",
        review_cache_dir=review_cache_dir,
        expected_review_ids={question.review_id for question in questions},
    )

    tfidf_rankings, tfidf_diagnostics = tfidf_rank(
        questions=questions,
        shard_paths=shard_paths,
        exclusions=exclusions,
        top_k=FUSION_SOURCE_DEPTH,
    )
    bm25_rankings, bm25_diagnostics = bm25_rank(
        questions=questions,
        shard_paths=shard_paths,
        exclusions=exclusions,
        top_k=FUSION_SOURCE_DEPTH,
        k1=1.2,
        b=0.75,
        k3=8.0,
    )
    rrf_rankings = reciprocal_rank_fusion(
        {"bm25-fixed-v1": bm25_rankings, "tfidf-fixed-v1": tfidf_rankings},
        top_k=OUTPUT_DEPTH,
        rank_constant=RRF_RANK_CONSTANT,
    )
    expected_ids = {question.review_id for question in questions}
    _validate_rankings(
        tfidf_rankings,
        expected_review_ids=expected_ids,
        expected_depth=FUSION_SOURCE_DEPTH,
        exclusions=exclusions,
    )
    _validate_rankings(
        bm25_rankings,
        expected_review_ids=expected_ids,
        expected_depth=FUSION_SOURCE_DEPTH,
        exclusions=exclusions,
    )
    _validate_rankings(
        rrf_rankings,
        expected_review_ids=expected_ids,
        expected_depth=OUTPUT_DEPTH,
        exclusions=exclusions,
    )
    candidate_rankings: dict[CandidateId, dict[int, list[int]]] = {
        "bm25-fixed-v1": bm25_rankings,
        "rrf-tfidf-bm25-fixed-v1": rrf_rankings,
        "tfidf-fixed-v1": tfidf_rankings,
    }

    ranking_rows = [
        {
            "review_id": review_id,
            "bm25_source_ranking": bm25_rankings[review_id],
            "rrf_output_ranking": rrf_rankings[review_id],
            "tfidf_source_ranking": tfidf_rankings[review_id],
        }
        for review_id in sorted(expected_ids)
    ]
    atomic_write_jsonl(rankings_path, ranking_rows, force=force)
    candidate_artifacts: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_IDS:
        payloads = _prediction_payloads(candidate_rankings[candidate], depth=OUTPUT_DEPTH)
        path = prediction_paths[candidate]
        atomic_write_jsonl(path, payloads, force=force)
        candidate_artifacts[candidate] = {
            "config": _candidate_configs()[candidate],
            "config_sha256": hash_canonical(_candidate_configs()[candidate]),
            "ordered_ranking_sha256": hash_canonical(candidate_rankings[candidate]),
            "predictions_path": path.relative_to(work_dir).as_posix(),
            "predictions_sha256": sha256_file(path),
            "rows": len(payloads),
        }

    exclusions_per_question = [len(exclusions[question.review_id]) for question in questions]
    payload = {
        "metasyn_retrieval_candidate_freeze_version": STUDY_VERSION,
        "stage": "label_blind_candidate_freeze",
        "scientific_role": "retrospective_development_selection_calibration_evaluation",
        "splits_frozen": ["development", "calibration"],
        "official_test_model_inputs_opened": False,
        "official_test_evaluated": False,
        "labels_read_by_freeze_stage": False,
        "labels_previously_opened_before_study_design": True,
        "pristine_holdout_eligible": False,
        "gold_matched_ids_used_for_ranking": False,
        "source_review_ids_used_only_as_exclusions": True,
        "source_review_parquet_columns_opened": ["ID", "source_review_corpus_ids"],
        "model_fields_used": [
            "research_question",
            "population",
            "intervention",
            "exposure",
            "comparison",
            "outcome",
        ],
        "rows_by_split": {
            "development": len(development),
            "calibration": len(calibration),
        },
        "benchmark_source_review_inventory": {
            "official_train": benchmark.source_train.rows,
            "official_test_not_used": benchmark.source_test.rows,
            "quarantined_official_train_not_used": len(benchmark.quarantined_official_train),
        },
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "model_input_sha256s": {
            "development": benchmark.development.sha256,
            "calibration": benchmark.calibration.sha256,
        },
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_source_revision": corpus.source_revision,
        "corpus_rows": corpus.total_rows,
        "corpus_shard_sha256s": {shard.path: shard.sha256 for shard in corpus.shards},
        "source_review_sha256s": source_review_hashes,
        "source_review_exclusions": {
            "total": sum(exclusions_per_question),
            "questions_with_at_least_one": sum(value > 0 for value in exclusions_per_question),
            "maximum_per_question": max(exclusions_per_question, default=0),
        },
        "candidate_selection_rule": {
            "development_metric": "question_weighted_macro_matched_subset_recall_at_200",
            "maximize": True,
            "tie_break": "ascending_candidate_id",
            "candidate_ids": list(CANDIDATE_IDS),
            "within_run_candidate_or_hyperparameter_tuning": False,
        },
        "candidates": candidate_artifacts,
        "ordered_rankings_path": rankings_path.relative_to(work_dir).as_posix(),
        "ordered_rankings_sha256": sha256_file(rankings_path),
        "diagnostics": {
            "bm25": bm25_diagnostics,
            "tfidf": tfidf_diagnostics,
        },
        "source_code_sha256s": _source_code_hashes(),
        "runtime_versions": {
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "tracked_summary_contains_question_or_article_text": False,
    }
    receipt = _attach_payload_hash(payload, field="freeze_payload_sha256")
    atomic_write_json(receipt_path, receipt, force=force)
    return receipt


def validate_candidate_freeze(*, work_dir: Path) -> dict[str, Any]:
    receipt_path = work_dir / "freeze_receipt.json"
    receipt = _load_json(receipt_path)
    _verify_payload_hash(receipt, field="freeze_payload_sha256")
    if receipt.get("metasyn_retrieval_candidate_freeze_version") != STUDY_VERSION:
        raise MetaSynRetrievalStudyError("candidate_freeze_version_unsupported")
    if receipt.get("official_test_evaluated") is not False:
        raise MetaSynRetrievalStudyError("official_test_must_not_be_evaluated")
    if receipt.get("labels_read_by_freeze_stage") is not False:
        raise MetaSynRetrievalStudyError("candidate_freeze_accessed_labels")
    if receipt.get("source_code_sha256s") != _source_code_hashes():
        raise MetaSynRetrievalStudyError("candidate_freeze_source_code_drift")
    rankings_path = work_dir / str(receipt.get("ordered_rankings_path"))
    if sha256_file(rankings_path) != receipt.get("ordered_rankings_sha256"):
        raise MetaSynRetrievalStudyError("ordered_rankings_hash_mismatch")

    expected_rows = sum(int(value) for value in receipt["rows_by_split"].values())
    universes: list[set[int]] = []
    for candidate in CANDIDATE_IDS:
        artifact = receipt.get("candidates", {}).get(candidate)
        if not isinstance(artifact, dict):
            raise MetaSynRetrievalStudyError(f"candidate_artifact_missing:{candidate}")
        if artifact.get("config_sha256") != hash_canonical(artifact.get("config")):
            raise MetaSynRetrievalStudyError(f"candidate_config_hash_mismatch:{candidate}")
        path = work_dir / str(artifact.get("predictions_path"))
        if sha256_file(path) != artifact.get("predictions_sha256"):
            raise MetaSynRetrievalStudyError(f"candidate_predictions_hash_mismatch:{candidate}")
        predictions = load_metasyn_predictions(path)
        if len(predictions) != expected_rows or len(predictions) != artifact.get("rows"):
            raise MetaSynRetrievalStudyError(f"candidate_prediction_rows_mismatch:{candidate}")
        for prediction in predictions:
            if (
                prediction.retrieved_corpus_ids is None
                or len(prediction.retrieved_corpus_ids) != OUTPUT_DEPTH
            ):
                raise MetaSynRetrievalStudyError(f"candidate_prediction_depth_mismatch:{candidate}")
        universes.append({prediction.review_id for prediction in predictions})
    if any(universe != universes[0] for universe in universes[1:]):
        raise MetaSynRetrievalStudyError("candidate_prediction_universe_mismatch")
    return receipt


def _load_split_labels(
    *, manifest_path: Path, split: Literal["development", "calibration"]
) -> tuple[list[MetaSynEvaluatorLabel], dict[str, Any]]:
    """Load one logical split from the historical shared evaluator artifact.

    The shared JSONL physically contains all three splits, so its bytes must be hashed
    and scanned even though only records bearing the requested split are retained.
    This limitation is disclosed in every stage receipt.
    """

    benchmark = load_metasyn_manifest_metadata(manifest_path)
    label_path = manifest_path.parent / benchmark.evaluator_labels.path
    observed_hash = sha256_file(label_path)
    if observed_hash != benchmark.evaluator_labels.sha256:
        raise MetaSynRetrievalStudyError("evaluator_label_artifact_hash_mismatch")
    labels: list[MetaSynEvaluatorLabel] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MetaSynRetrievalStudyError("evaluator_label_artifact_unreadable") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetaSynRetrievalStudyError(
                f"evaluator_label_json_invalid:line={line_number}"
            ) from exc
        if payload.get("split") != split:
            continue
        try:
            labels.append(MetaSynEvaluatorLabel.model_validate(payload))
        except ValueError as exc:
            raise MetaSynRetrievalStudyError(
                f"evaluator_label_invalid:split={split}:line={line_number}"
            ) from exc
    expected = getattr(benchmark, split)
    labels.sort(key=lambda item: item.review_id)
    if [label.review_id for label in labels] != expected.review_ids:
        raise MetaSynRetrievalStudyError(f"evaluator_label_review_ids_mismatch:{split}")
    if sorted({label.component_id for label in labels}) != expected.component_ids:
        raise MetaSynRetrievalStudyError(f"evaluator_label_components_mismatch:{split}")
    access = {
        "evaluator_labels_sha256": observed_hash,
        "logical_split_retained": split,
        "records_retained": len(labels),
        "shared_artifact_physically_contains_all_splits": True,
        "nonrequested_gold_records_scored": False,
    }
    return labels, access


def _load_candidate_predictions(
    *, work_dir: Path, freeze: Mapping[str, Any], candidate: CandidateId
) -> list[MetaSynPrediction]:
    artifact = freeze["candidates"][candidate]
    path = work_dir / artifact["predictions_path"]
    if sha256_file(path) != artifact["predictions_sha256"]:
        raise MetaSynRetrievalStudyError(f"candidate_predictions_hash_mismatch:{candidate}")
    return load_metasyn_predictions(path)


def _cluster_bootstrap(
    *,
    labels: Sequence[MetaSynEvaluatorLabel],
    records_by_candidate: Mapping[str, Sequence[tuple[int, int]]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, Any]]]:
    """Paired percentile bootstrap over precomputed benchmark components."""

    if replicates < 1:
        raise ValueError("bootstrap_replicates_must_be_positive")
    if not labels:
        raise MetaSynRetrievalStudyError("bootstrap_labels_empty")
    candidate_names = sorted(records_by_candidate)
    if any(len(records_by_candidate[name]) != len(labels) for name in candidate_names):
        raise MetaSynRetrievalStudyError("bootstrap_record_count_mismatch")

    clusters = sorted({label.component_id for label in labels})
    cluster_index = {component: index for index, component in enumerate(clusters)}
    counts = np.zeros(len(clusters), dtype=np.int64)
    gold = np.zeros(len(clusters), dtype=np.int64)
    recall_sums = {
        candidate: np.zeros(len(clusters), dtype=np.float64) for candidate in candidate_names
    }
    hits = {candidate: np.zeros(len(clusters), dtype=np.int64) for candidate in candidate_names}
    for row_index, label in enumerate(labels):
        index = cluster_index[label.component_id]
        counts[index] += 1
        gold_count = len(label.gold_matched_corpus_ids)
        gold[index] += gold_count
        for candidate in candidate_names:
            candidate_hits, candidate_gold = records_by_candidate[candidate][row_index]
            if candidate_gold != gold_count:
                raise MetaSynRetrievalStudyError("bootstrap_gold_count_mismatch")
            hits[candidate][index] += candidate_hits
            recall_sums[candidate][index] += candidate_hits / candidate_gold

    rng = np.random.default_rng(seed)
    macro_samples = {
        candidate: np.empty(replicates, dtype=np.float64) for candidate in candidate_names
    }
    micro_samples = {
        candidate: np.empty(replicates, dtype=np.float64) for candidate in candidate_names
    }
    batch_size = 512
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        sampled = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        question_denominator = counts[sampled].sum(axis=1)
        gold_denominator = gold[sampled].sum(axis=1)
        for candidate in candidate_names:
            macro_samples[candidate][start:stop] = (
                recall_sums[candidate][sampled].sum(axis=1) / question_denominator
            )
            micro_samples[candidate][start:stop] = (
                hits[candidate][sampled].sum(axis=1) / gold_denominator
            )

    intervals: dict[str, dict[str, list[float]]] = {}
    for candidate in candidate_names:
        intervals[candidate] = {
            "macro_recall_at_200": [
                float(value) for value in np.quantile(macro_samples[candidate], [0.025, 0.975])
            ],
            "micro_recall_at_200": [
                float(value) for value in np.quantile(micro_samples[candidate], [0.025, 0.975])
            ],
        }

    differences: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(candidate_names):
        for right in candidate_names[left_index + 1 :]:
            key = f"{left}_minus_{right}"
            macro_delta = macro_samples[left] - macro_samples[right]
            micro_delta = micro_samples[left] - micro_samples[right]
            differences[key] = {
                "macro_recall_at_200_difference": float(
                    recall_sums[left].sum() / counts.sum() - recall_sums[right].sum() / counts.sum()
                ),
                "macro_recall_at_200_interval_95": [
                    float(value) for value in np.quantile(macro_delta, [0.025, 0.975])
                ],
                "micro_recall_at_200_difference": float(
                    hits[left].sum() / gold.sum() - hits[right].sum() / gold.sum()
                ),
                "micro_recall_at_200_interval_95": [
                    float(value) for value in np.quantile(micro_delta, [0.025, 0.975])
                ],
            }
    return intervals, differences


def _score_candidates(
    *,
    labels: Sequence[MetaSynEvaluatorLabel],
    predictions: Mapping[CandidateId, Sequence[MetaSynPrediction]],
    exclusions: Mapping[int, set[int]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected_ids = {label.review_id for label in labels}
    records: dict[str, list[tuple[int, int]]] = {}
    raw_metrics: dict[str, dict[str, Any]] = {}
    for candidate, rows in predictions.items():
        by_id = {row.review_id: row for row in rows}
        missing = sorted(expected_ids - set(by_id))
        if missing:
            raise MetaSynRetrievalStudyError(
                f"candidate_missing_split_predictions:{candidate}:count={len(missing)}"
            )
        candidate_records: list[tuple[int, int]] = []
        zero_recall = full_recall = total_hits = total_gold = 0
        for label in labels:
            retrieved = by_id[label.review_id].retrieved_corpus_ids
            if retrieved is None or len(retrieved) != OUTPUT_DEPTH:
                raise MetaSynRetrievalStudyError(f"candidate_prediction_depth_mismatch:{candidate}")
            gold = set(label.gold_matched_corpus_ids)
            hit_count = len(gold & set(retrieved))
            candidate_records.append((hit_count, len(gold)))
            total_hits += hit_count
            total_gold += len(gold)
            zero_recall += int(hit_count == 0)
            full_recall += int(hit_count == len(gold))
        records[candidate] = candidate_records
        recalls = [hit_count / gold_count for hit_count, gold_count in candidate_records]
        raw_metrics[candidate] = {
            "questions": len(labels),
            "retrieval_depth": OUTPUT_DEPTH,
            "matched_references": total_gold,
            "matched_references_retrieved": total_hits,
            "macro_recall_at_200": float(sum(recalls) / len(recalls)),
            "micro_recall_at_200": float(total_hits / total_gold),
            "questions_with_zero_recall": zero_recall,
            "questions_with_full_recall": full_recall,
        }

    gold_exclusion_overlaps = [
        len(set(label.gold_matched_corpus_ids) & exclusions[label.review_id]) for label in labels
    ]
    intervals, paired_differences = _cluster_bootstrap(
        labels=labels,
        records_by_candidate=records,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    for candidate in raw_metrics:
        raw_metrics[candidate]["cluster_bootstrap_interval_95"] = intervals[candidate]

    cluster_sizes: defaultdict[str, int] = defaultdict(int)
    for label in labels:
        cluster_sizes[label.component_id] += 1
    return {
        "estimand": (
            "question-weighted mean of per-question Recall@200 against MetaSyn's "
            "released matched-paper subset"
        ),
        "candidates": {key: raw_metrics[key] for key in sorted(raw_metrics)},
        "paired_cluster_bootstrap_differences": paired_differences,
        "uncertainty": {
            "method": "paired_nonparametric_percentile_cluster_bootstrap",
            "resampling_unit": "pre-split MetaSyn review component",
            "question_is_observation_unit": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "clusters": len(cluster_sizes),
            "singleton_clusters": sum(size == 1 for size in cluster_sizes.values()),
            "maximum_questions_per_cluster": max(cluster_sizes.values()),
            "interpretation": (
                "descriptive sampling uncertainty for this retrospective benchmark; "
                "not a pristine-holdout or exhaustive-corpus guarantee"
            ),
        },
        "source_review_exclusion_audit": {
            "gold_matched_references_also_excluded": sum(gold_exclusion_overlaps),
            "questions_with_gold_exclusion_overlap": sum(
                overlap > 0 for overlap in gold_exclusion_overlaps
            ),
        },
    }


def select_candidate_on_development(
    *,
    benchmark_manifest_path: Path,
    review_cache_dir: Path,
    work_dir: Path,
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score every frozen candidate on development and freeze one selection."""

    output_path = work_dir / "development_selection.json"
    if output_path.exists() and not force:
        raise MetaSynRetrievalStudyError(f"development_selection_exists:{output_path}")
    freeze = validate_candidate_freeze(work_dir=work_dir)
    labels, label_access = _load_split_labels(
        manifest_path=benchmark_manifest_path, split="development"
    )
    expected_ids = {label.review_id for label in labels}
    exclusions, source_hashes = load_source_review_exclusions(
        benchmark_manifest_path=benchmark_manifest_path,
        split="development",
        review_cache_dir=review_cache_dir,
        expected_review_ids=expected_ids,
    )
    predictions = {
        candidate: _load_candidate_predictions(
            work_dir=work_dir, freeze=freeze, candidate=candidate
        )
        for candidate in CANDIDATE_IDS
    }
    results = _score_candidates(
        labels=labels,
        predictions=predictions,
        exclusions=exclusions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    selected = min(
        CANDIDATE_IDS,
        key=lambda candidate: (
            -results["candidates"][candidate]["macro_recall_at_200"],
            candidate,
        ),
    )
    selected_artifact = freeze["candidates"][selected]
    payload = {
        "metasyn_retrieval_development_selection_version": STUDY_VERSION,
        "stage": "development_only_selection",
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "freeze_receipt_sha256": sha256_file(work_dir / "freeze_receipt.json"),
        "development_labels_scored": True,
        "calibration_labels_scored": False,
        "official_test_evaluated": False,
        "all_labels_previously_opened_before_study_design": True,
        "pristine_holdout_eligible": False,
        "label_access": label_access,
        "source_review_sha256s": source_hashes,
        "selection_rule": freeze["candidate_selection_rule"],
        "development_results": results,
        "selected_candidate_id": selected,
        "selected_candidate_config_sha256": selected_artifact["config_sha256"],
        "selected_candidate_predictions_sha256": selected_artifact["predictions_sha256"],
        "calibration_selection_feedback_allowed": False,
    }
    receipt = _attach_payload_hash(payload, field="development_selection_payload_sha256")
    atomic_write_json(output_path, receipt, force=force)
    return receipt


def validate_development_selection(*, work_dir: Path) -> dict[str, Any]:
    freeze = validate_candidate_freeze(work_dir=work_dir)
    path = work_dir / "development_selection.json"
    selection = _load_json(path)
    _verify_payload_hash(selection, field="development_selection_payload_sha256")
    if selection.get("metasyn_retrieval_development_selection_version") != STUDY_VERSION:
        raise MetaSynRetrievalStudyError("development_selection_version_unsupported")
    if selection.get("freeze_payload_sha256") != freeze.get("freeze_payload_sha256"):
        raise MetaSynRetrievalStudyError("development_selection_freeze_mismatch")
    selected = selection.get("selected_candidate_id")
    if selected not in CANDIDATE_IDS:
        raise MetaSynRetrievalStudyError("development_selected_candidate_invalid")
    if selection.get("calibration_labels_scored") is not False:
        raise MetaSynRetrievalStudyError("development_selection_used_calibration")
    if selection.get("official_test_evaluated") is not False:
        raise MetaSynRetrievalStudyError("development_selection_used_official_test")
    return selection


def evaluate_selected_on_calibration(
    *,
    benchmark_manifest_path: Path,
    review_cache_dir: Path,
    work_dir: Path,
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED + 1,
) -> dict[str, Any]:
    """Evaluate only the development-selected candidate on calibration."""

    output_path = work_dir / "calibration_evaluation.json"
    if output_path.exists() and not force:
        raise MetaSynRetrievalStudyError(f"calibration_evaluation_exists:{output_path}")
    freeze = validate_candidate_freeze(work_dir=work_dir)
    selection = validate_development_selection(work_dir=work_dir)
    candidate: CandidateId = selection["selected_candidate_id"]
    labels, label_access = _load_split_labels(
        manifest_path=benchmark_manifest_path, split="calibration"
    )
    expected_ids = {label.review_id for label in labels}
    exclusions, source_hashes = load_source_review_exclusions(
        benchmark_manifest_path=benchmark_manifest_path,
        split="calibration",
        review_cache_dir=review_cache_dir,
        expected_review_ids=expected_ids,
    )
    predictions = {
        candidate: _load_candidate_predictions(
            work_dir=work_dir, freeze=freeze, candidate=candidate
        )
    }
    results = _score_candidates(
        labels=labels,
        predictions=predictions,
        exclusions=exclusions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    payload = {
        "metasyn_retrieval_calibration_evaluation_version": STUDY_VERSION,
        "stage": "single_calibration_evaluation",
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "development_selection_payload_sha256": selection["development_selection_payload_sha256"],
        "development_selection_receipt_sha256": sha256_file(
            work_dir / "development_selection.json"
        ),
        "selected_candidate_id": candidate,
        "selected_candidate_config_sha256": freeze["candidates"][candidate]["config_sha256"],
        "selected_candidate_predictions_sha256": freeze["candidates"][candidate][
            "predictions_sha256"
        ],
        "development_candidate_comparison_computed": True,
        "calibration_candidate_comparison_computed": False,
        "calibration_selected_candidate_evaluations_in_this_protocol": 1,
        "calibration_selection_feedback_allowed": False,
        "official_test_evaluated": False,
        "all_labels_previously_opened_before_study_design": True,
        "pristine_holdout_eligible": False,
        "label_access": label_access,
        "source_review_sha256s": source_hashes,
        "calibration_result": results,
    }
    receipt = _attach_payload_hash(payload, field="calibration_evaluation_payload_sha256")
    atomic_write_json(output_path, receipt, force=force)
    return receipt


def validate_calibration_evaluation(*, work_dir: Path) -> dict[str, Any]:
    selection = validate_development_selection(work_dir=work_dir)
    path = work_dir / "calibration_evaluation.json"
    evaluation = _load_json(path)
    _verify_payload_hash(evaluation, field="calibration_evaluation_payload_sha256")
    if evaluation.get("metasyn_retrieval_calibration_evaluation_version") != STUDY_VERSION:
        raise MetaSynRetrievalStudyError("calibration_evaluation_version_unsupported")
    if evaluation.get("development_selection_payload_sha256") != selection.get(
        "development_selection_payload_sha256"
    ):
        raise MetaSynRetrievalStudyError("calibration_development_selection_mismatch")
    if evaluation.get("selected_candidate_id") != selection.get("selected_candidate_id"):
        raise MetaSynRetrievalStudyError("calibration_selected_candidate_mismatch")
    if evaluation.get("calibration_candidate_comparison_computed") is not False:
        raise MetaSynRetrievalStudyError("calibration_compared_unselected_candidates")
    if evaluation.get("official_test_evaluated") is not False:
        raise MetaSynRetrievalStudyError("calibration_evaluated_official_test")
    return evaluation


def _assert_public_metadata_only(payload: Any, *, repository_root: Path) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise MetaSynRetrievalStudyError(f"public_summary_forbidden_key:{key}")
            _assert_public_metadata_only(value, repository_root=repository_root)
    elif isinstance(payload, list):
        for value in payload:
            _assert_public_metadata_only(value, repository_root=repository_root)
    elif isinstance(payload, str) and repository_root.as_posix() in payload:
        raise MetaSynRetrievalStudyError("public_summary_contains_absolute_repository_path")


def build_public_summary(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    work_dir: Path,
    output_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Emit a deterministic aggregate-only summary of the completed study."""

    freeze = validate_candidate_freeze(work_dir=work_dir)
    selection = validate_development_selection(work_dir=work_dir)
    calibration = validate_calibration_evaluation(work_dir=work_dir)
    benchmark = load_metasyn_manifest_metadata(benchmark_manifest_path)
    candidate: CandidateId = selection["selected_candidate_id"]
    payload = {
        "metasyn_retrieval_public_summary_version": STUDY_VERSION,
        "status": "complete_retrospective_nonpristine",
        "task": "MetaSyn matched-subset lexical retrieval at depth 200",
        "selection_protocol": {
            "candidate_predictions_frozen_before_scoring": True,
            "development_compared_candidates": list(CANDIDATE_IDS),
            "development_selection_metric": freeze["candidate_selection_rule"][
                "development_metric"
            ],
            "development_tie_break": freeze["candidate_selection_rule"]["tie_break"],
            "selected_candidate": candidate,
            "calibration_scored_candidates": [candidate],
            "calibration_selection_feedback_allowed": False,
            "official_test_evaluated": False,
        },
        "access_boundary": {
            "all_development_calibration_and_test_labels_previously_opened": True,
            "pristine_final_holdout_eligible": False,
            "procedural_split_does_not_restore_pristine_status": True,
            "shared_evaluator_file_physically_contains_all_splits": True,
            "official_test_gold_not_scored": True,
        },
        "dataset_boundary": {
            "dataset": "THUIR/MetaSyn",
            "source_revision": freeze["corpus_source_revision"],
            "corpus_rows": freeze["corpus_rows"],
            "pinned_source_reviews": (benchmark.source_train.rows + benchmark.source_test.rows),
            "official_train_source_reviews": benchmark.source_train.rows,
            "official_test_source_reviews_not_evaluated": benchmark.source_test.rows,
            "quarantined_official_train_reviews_not_evaluated": len(
                benchmark.quarantined_official_train
            ),
            "development_reviews": benchmark.development.rows,
            "calibration_reviews": benchmark.calibration.rows,
            "recall_denominator": "released_matched_paper_subset_not_exhaustive_eligibility",
            "claim_scope": (
                "retrieval agreement with MetaSyn's released matched-paper identifiers; "
                "not exhaustive eligible-study recall or scientific correctness"
            ),
        },
        "source_review_exclusions": freeze["source_review_exclusions"],
        "candidate_configs": {
            key: {
                "config": freeze["candidates"][key]["config"],
                "config_sha256": freeze["candidates"][key]["config_sha256"],
            }
            for key in CANDIDATE_IDS
        },
        "development_results": selection["development_results"],
        "selected_calibration_result": calibration["calibration_result"],
        "lineage": {
            "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
            "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "corpus_shard_sha256s": freeze["corpus_shard_sha256s"],
            "model_input_sha256s": freeze["model_input_sha256s"],
            "source_review_sha256s": freeze["source_review_sha256s"],
            "evaluator_labels_sha256": selection["label_access"]["evaluator_labels_sha256"],
            "freeze_receipt_sha256": sha256_file(work_dir / "freeze_receipt.json"),
            "development_selection_receipt_sha256": sha256_file(
                work_dir / "development_selection.json"
            ),
            "calibration_evaluation_receipt_sha256": sha256_file(
                work_dir / "calibration_evaluation.json"
            ),
            "selected_predictions_sha256": freeze["candidates"][candidate]["predictions_sha256"],
            "source_code_sha256s": freeze["source_code_sha256s"],
        },
        "runtime_versions": freeze["runtime_versions"],
        "network_calls": 0,
        "provider_calls": 0,
        "contains_question_text": False,
        "contains_article_text": False,
        "contains_per_question_or_per_article_identifiers": False,
        "timestamps_in_scientific_payload": False,
    }
    _assert_public_metadata_only(payload, repository_root=repository_root)
    summary = _attach_payload_hash(payload, field="public_summary_payload_sha256")
    atomic_write_json(output_path, summary, force=force)
    return summary


def run_retrieval_study(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    review_cache_dir: Path,
    work_dir: Path,
    public_summary_path: Path,
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run or validate the ordered three-stage study and publish its summary."""

    freeze_path = work_dir / "freeze_receipt.json"
    selection_path = work_dir / "development_selection.json"
    calibration_path = work_dir / "calibration_evaluation.json"
    if force or not freeze_path.exists():
        freeze_candidate_predictions(
            benchmark_manifest_path=benchmark_manifest_path,
            corpus_manifest_path=corpus_manifest_path,
            repository_root=repository_root,
            review_cache_dir=review_cache_dir,
            work_dir=work_dir,
            force=force,
        )
    else:
        validate_candidate_freeze(work_dir=work_dir)
    if force or not selection_path.exists():
        select_candidate_on_development(
            benchmark_manifest_path=benchmark_manifest_path,
            review_cache_dir=review_cache_dir,
            work_dir=work_dir,
            force=force,
            bootstrap_replicates=bootstrap_replicates,
        )
    else:
        validate_development_selection(work_dir=work_dir)
    if force or not calibration_path.exists():
        evaluate_selected_on_calibration(
            benchmark_manifest_path=benchmark_manifest_path,
            review_cache_dir=review_cache_dir,
            work_dir=work_dir,
            force=force,
            bootstrap_replicates=bootstrap_replicates,
        )
    else:
        validate_calibration_evaluation(work_dir=work_dir)
    return build_public_summary(
        benchmark_manifest_path=benchmark_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        repository_root=repository_root,
        work_dir=work_dir,
        output_path=public_summary_path,
        force=True,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_IDS",
    "FUSION_SOURCE_DEPTH",
    "OUTPUT_DEPTH",
    "RRF_RANK_CONSTANT",
    "MetaSynRetrievalStudyError",
    "build_public_summary",
    "evaluate_selected_on_calibration",
    "freeze_candidate_predictions",
    "reciprocal_rank_fusion",
    "run_retrieval_study",
    "select_candidate_on_development",
    "validate_calibration_evaluation",
    "validate_candidate_freeze",
    "validate_development_selection",
]
