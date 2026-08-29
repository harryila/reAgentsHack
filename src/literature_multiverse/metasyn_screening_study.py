"""Staged, protocol-aware MetaSyn screening reranking study.

This module deliberately operates *after* the frozen retrieval study.  It may only
reorder the 200 documents returned by that study's fixed RRF candidate.  The three
logical stages are:

1. ``prepare`` freezes numeric, label-blind features for development and calibration;
2. ``fit`` opens development labels, performs component-disjoint cross-validation,
   selects one prespecified reranker, and freezes its calibration ordering; and
3. ``evaluate`` opens calibration labels once and compares the frozen winner with the
   original RRF order.

The official test input and label artifacts are never opened by this implementation.
MetaSyn development and calibration were historically opened before this protocol,
so the result is retrospective and cannot be represented as pristine holdout evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_benchmark import (
    MetaSynQuestionInput,
    load_metasyn_inputs,
    load_metasyn_manifest_metadata,
)
from literature_multiverse.metasyn_retrieval import verify_corpus_manifest

ScreeningCandidate = Literal[
    "logistic-l2-balanced-v1",
    "monotonic-hist-gradient-balanced-v1",
    "rrf-passthrough-v1",
]

STUDY_VERSION = "1"
RETRIEVAL_DEPTH = 200
EVALUATION_DEPTHS: tuple[int, ...] = (10, 20, 50, 100, 200)
SELECTION_DEPTHS: tuple[int, ...] = (10, 20, 50, 100)
CV_FOLDS = 5
CV_SEED = 20_260_827
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_829
BASELINE_ID: ScreeningCandidate = "rrf-passthrough-v1"
CANDIDATE_IDS: tuple[ScreeningCandidate, ...] = (
    "logistic-l2-balanced-v1",
    "monotonic-hist-gradient-balanced-v1",
    BASELINE_ID,
)

FEATURE_NAMES: tuple[str, ...] = (
    "rrf_position_reciprocal",
    "bm25_rank_reciprocal",
    "tfidf_rank_reciprocal",
    "rrf_source_score",
    "source_rank_agreement",
    "query_title_jaccard",
    "query_abstract_jaccard",
    "query_title_recall",
    "query_abstract_recall",
    "title_query_precision",
    "abstract_query_precision",
    "population_abstract_recall",
    "intervention_abstract_recall",
    "exposure_abstract_recall",
    "comparison_abstract_recall",
    "outcome_abstract_recall",
    "protocol_fields_with_abstract_overlap_fraction",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only
    or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you
    your yours yourself yourselves study studies review reviews systematic meta analysis
    investigate investigating examined examine evaluates evaluate assessed assess aim aims
    """.split()  # noqa: SIM905 - freezing this readable vocabulary is intentional.
)
_PROTOCOL_FIELDS: tuple[str, ...] = (
    "population",
    "intervention",
    "exposure",
    "comparison",
    "outcome",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "abstract",
        "component_id",
        "corpus_id",
        "gold_matched_corpus_ids",
        "ordered_corpus_ids",
        "question_id",
        "research_question",
        "review_id",
        "title",
    }
)


class MetaSynScreeningStudyError(ValueError):
    """A source, stage boundary, or aggregate artifact violated the protocol."""


@dataclass(frozen=True, slots=True)
class _ScreeningLabel:
    review_id: int
    component_id: str
    gold_corpus_ids: tuple[int, ...]


class _UnionFind:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynScreeningStudyError(f"json_artifact_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise MetaSynScreeningStudyError(f"json_artifact_not_object:{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MetaSynScreeningStudyError(f"jsonl_artifact_unreadable:{path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetaSynScreeningStudyError(
                f"jsonl_artifact_invalid:{path}:line={line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise MetaSynScreeningStudyError(
                f"jsonl_artifact_row_not_object:{path}:line={line_number}"
            )
        rows.append(row)
    return rows


def _attach_hash(payload: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in payload:
        raise MetaSynScreeningStudyError(f"payload_hash_field_present:{field}")
    return {**payload, field: hash_canonical(payload)}


def _verify_hash(payload: Mapping[str, Any], *, field: str) -> None:
    observed = payload.get(field)
    content = {key: value for key, value in payload.items() if key != field}
    if not isinstance(observed, str) or observed != hash_canonical(content):
        raise MetaSynScreeningStudyError(f"payload_hash_mismatch:{field}")


def _work_artifact(work_dir: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise MetaSynScreeningStudyError(f"artifact_path_invalid:{field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise MetaSynScreeningStudyError(f"artifact_path_unsafe:{field}")
    return work_dir / relative


def _source_code_hashes() -> dict[str, str]:
    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[2]
    paths = {
        "pyproject.toml": repository_root / "pyproject.toml",
        "scripts/run_metasyn_screening_study.py": (
            repository_root / "scripts/run_metasyn_screening_study.py"
        ),
        "src/literature_multiverse/__init__.py": (
            repository_root / "src/literature_multiverse/__init__.py"
        ),
        "src/literature_multiverse/calibration.py": (
            repository_root / "src/literature_multiverse/calibration.py"
        ),
        "src/literature_multiverse/lineage.py": (
            repository_root / "src/literature_multiverse/lineage.py"
        ),
        "src/literature_multiverse/metasyn_benchmark.py": (
            repository_root / "src/literature_multiverse/metasyn_benchmark.py"
        ),
        "src/literature_multiverse/metasyn_retrieval.py": (
            repository_root / "src/literature_multiverse/metasyn_retrieval.py"
        ),
        "src/literature_multiverse/metasyn_screening_study.py": module_path,
        "src/literature_multiverse/models.py": (
            repository_root / "src/literature_multiverse/models.py"
        ),
        "src/literature_multiverse/paths.py": (
            repository_root / "src/literature_multiverse/paths.py"
        ),
        "uv.lock": repository_root / "uv.lock",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise MetaSynScreeningStudyError(f"source_code_missing:{sorted(missing)}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _candidate_configs() -> dict[ScreeningCandidate, dict[str, Any]]:
    common = {
        "candidate_set": "frozen_rrf_top_200_only",
        "features": list(FEATURE_NAMES),
        "stable_tie_break": "descending_score_then_ascending_corpus_id",
    }
    return {
        "logistic-l2-balanced-v1": {
            **common,
            "estimator": "standard_scaler_plus_logistic_regression",
            "C": 0.25,
            "class_weight": "balanced",
            "max_iter": 500,
            "solver": "lbfgs",
        },
        "monotonic-hist-gradient-balanced-v1": {
            **common,
            "estimator": "hist_gradient_boosting_classifier",
            "class_weight": "balanced",
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "monotonic_cst": [1] * len(FEATURE_NAMES),
        },
        BASELINE_ID: {
            **common,
            "estimator": "identity_original_rrf_order",
        },
    }


def _feature_config() -> dict[str, Any]:
    return {
        "feature_extractor": "deterministic_protocol_lexical_overlap_v1",
        "candidate_source": "rrf-tfidf-bm25-fixed-v1_top_200",
        "candidate_document_fields": ["title", "abstract"],
        "protocol_fields": ["research_question", *_PROTOCOL_FIELDS],
        "forbidden_fields": [
            "conclusion",
            "effect_direction",
            "effect_size",
            "gold_matched_corpus_ids",
            "label",
        ],
        "token_pattern": _TOKEN_RE.pattern,
        "stopwords": sorted(_STOPWORDS),
        "feature_names": list(FEATURE_NAMES),
        "rank_constant": 60,
        "retrieval_depth": RETRIEVAL_DEPTH,
    }


def _selection_config() -> dict[str, Any]:
    return {
        "candidate_family": {
            candidate: {
                "config": config,
                "config_sha256": hash_canonical(config),
            }
            for candidate, config in _candidate_configs().items()
        },
        "cross_validation": "group_k_fold_with_shuffle",
        "folds": CV_FOLDS,
        "seed": CV_SEED,
        "groups": "MetaSyn_connected_review_components",
        "metric": "unweighted_mean_question_macro_absolute_recall_at_10_20_50_100",
        "maximize": True,
        "tie_break": "ascending_candidate_id",
        "depth_200_excluded_because_invariant_under_within_set_reranking": True,
    }


def _evaluation_config() -> dict[str, Any]:
    return {
        "depths": list(EVALUATION_DEPTHS),
        "prespecified_baseline": BASELINE_ID,
        "bootstrap": "paired_nonparametric_percentile_component_cluster_bootstrap",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_bootstrap_replicates": 10_000,
    }


def _terms(value: str | None) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall((value or "").casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _overlap_features(
    *,
    question: MetaSynQuestionInput,
    title: str | None,
    abstract: str | None,
    original_rrf_rank: int,
    bm25_rank: int,
    tfidf_rank: int,
) -> dict[str, float]:
    query_fields = {
        field: _terms(str(getattr(question, field) or ""))
        for field in _PROTOCOL_FIELDS
    }
    query = _terms(question.research_question)
    for tokens in query_fields.values():
        query.update(tokens)
    title_terms = _terms(title)
    abstract_terms = _terms(abstract)
    query_title = query & title_terms
    query_abstract = query & abstract_terms
    field_abstract_recalls = {
        field: _ratio(len(tokens & abstract_terms), len(tokens))
        for field, tokens in query_fields.items()
    }
    nonempty_fields = [tokens for tokens in query_fields.values() if tokens]
    fields_with_overlap = sum(bool(tokens & abstract_terms) for tokens in nonempty_fields)
    features = {
        "rrf_position_reciprocal": 1.0 / original_rrf_rank,
        "bm25_rank_reciprocal": 1.0 / (60 + bm25_rank),
        "tfidf_rank_reciprocal": 1.0 / (60 + tfidf_rank),
        "rrf_source_score": 1.0 / (60 + bm25_rank) + 1.0 / (60 + tfidf_rank),
        "source_rank_agreement": 1.0 / (1.0 + abs(bm25_rank - tfidf_rank)),
        "query_title_jaccard": _ratio(
            len(query_title), len(query | title_terms)
        ),
        "query_abstract_jaccard": _ratio(
            len(query_abstract), len(query | abstract_terms)
        ),
        "query_title_recall": _ratio(len(query_title), len(query)),
        "query_abstract_recall": _ratio(len(query_abstract), len(query)),
        "title_query_precision": _ratio(len(query_title), len(title_terms)),
        "abstract_query_precision": _ratio(len(query_abstract), len(abstract_terms)),
        "population_abstract_recall": field_abstract_recalls["population"],
        "intervention_abstract_recall": field_abstract_recalls["intervention"],
        "exposure_abstract_recall": field_abstract_recalls["exposure"],
        "comparison_abstract_recall": field_abstract_recalls["comparison"],
        "outcome_abstract_recall": field_abstract_recalls["outcome"],
        "protocol_fields_with_abstract_overlap_fraction": _ratio(
            fields_with_overlap, len(nonempty_fields)
        ),
    }
    if tuple(features) != FEATURE_NAMES:
        raise MetaSynScreeningStudyError("feature_order_contract_changed")
    if any(not math.isfinite(value) for value in features.values()):
        raise MetaSynScreeningStudyError("nonfinite_label_blind_feature")
    return features


def _validate_retrieval_freeze(
    *,
    retrieval_work_dir: Path,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    receipt_path = retrieval_work_dir / "freeze_receipt.json"
    receipt = _load_json(receipt_path)
    _verify_hash(receipt, field="freeze_payload_sha256")
    if receipt.get("stage") != "label_blind_candidate_freeze":
        raise MetaSynScreeningStudyError("retrieval_freeze_stage_invalid")
    if receipt.get("labels_read_by_freeze_stage") is not False:
        raise MetaSynScreeningStudyError("retrieval_freeze_used_labels")
    if receipt.get("official_test_model_inputs_opened") is not False:
        raise MetaSynScreeningStudyError("retrieval_freeze_opened_official_test_inputs")
    if receipt.get("official_test_evaluated") is not False:
        raise MetaSynScreeningStudyError("retrieval_freeze_evaluated_official_test")
    if receipt.get("benchmark_manifest_sha256") != sha256_file(benchmark_manifest_path):
        raise MetaSynScreeningStudyError("retrieval_benchmark_manifest_hash_mismatch")
    if receipt.get("corpus_manifest_sha256") != sha256_file(corpus_manifest_path):
        raise MetaSynScreeningStudyError("retrieval_corpus_manifest_hash_mismatch")
    candidate = receipt.get("candidates", {}).get("rrf-tfidf-bm25-fixed-v1")
    if not isinstance(candidate, dict):
        raise MetaSynScreeningStudyError("retrieval_rrf_candidate_missing")
    if candidate.get("config", {}).get("evaluated_depth") != RETRIEVAL_DEPTH:
        raise MetaSynScreeningStudyError("retrieval_rrf_depth_mismatch")
    rankings_path = retrieval_work_dir / str(receipt.get("ordered_rankings_path"))
    if sha256_file(rankings_path) != receipt.get("ordered_rankings_sha256"):
        raise MetaSynScreeningStudyError("retrieval_ordered_rankings_hash_mismatch")
    return receipt, rankings_path


def _load_ordered_rankings(
    path: Path, *, expected_review_ids: set[int]
) -> dict[int, dict[str, Any]]:
    rows = _load_jsonl(path)
    by_review: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            review_id = int(row["review_id"])
            bm25 = [int(value) for value in row["bm25_source_ranking"]]
            tfidf = [int(value) for value in row["tfidf_source_ranking"]]
            rrf = [int(value) for value in row["rrf_output_ranking"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise MetaSynScreeningStudyError("retrieval_ranking_row_invalid") from exc
        if review_id in by_review:
            raise MetaSynScreeningStudyError(f"retrieval_ranking_duplicate_review:{review_id}")
        if len(bm25) != 1_000 or len(set(bm25)) != 1_000:
            raise MetaSynScreeningStudyError("bm25_source_ranking_invalid")
        if len(tfidf) != 1_000 or len(set(tfidf)) != 1_000:
            raise MetaSynScreeningStudyError("tfidf_source_ranking_invalid")
        if len(rrf) != RETRIEVAL_DEPTH or len(set(rrf)) != RETRIEVAL_DEPTH:
            raise MetaSynScreeningStudyError("rrf_output_ranking_invalid")
        if not set(rrf) <= (set(bm25) | set(tfidf)):
            raise MetaSynScreeningStudyError("rrf_document_absent_from_source_rankings")
        by_review[review_id] = {
            "bm25": bm25,
            "tfidf": tfidf,
            "rrf": rrf,
        }
    if set(by_review) != expected_review_ids:
        missing = len(expected_review_ids - set(by_review))
        extra = len(set(by_review) - expected_review_ids)
        raise MetaSynScreeningStudyError(
            f"retrieval_ranking_universe_mismatch:missing={missing}:extra={extra}"
        )
    return by_review


def _load_candidate_documents(
    *, shard_paths: Sequence[Path], required_ids: set[int]
) -> dict[int, tuple[str | None, str | None]]:
    documents: dict[int, tuple[str | None, str | None]] = {}
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_path)
        try:
            batches = parquet.iter_batches(batch_size=8192, columns=["ID", "title", "abstract"])
            for batch in batches:
                for row in batch.to_pylist():
                    corpus_id = int(row["ID"])
                    if corpus_id not in required_ids:
                        continue
                    if corpus_id in documents:
                        raise MetaSynScreeningStudyError(
                            f"candidate_document_id_duplicate:{corpus_id}"
                        )
                    documents[corpus_id] = (
                        None if row["title"] is None else str(row["title"]),
                        None if row["abstract"] is None else str(row["abstract"]),
                    )
        except MetaSynScreeningStudyError:
            raise
        except Exception as exc:
            raise MetaSynScreeningStudyError(
                f"candidate_document_scan_failed:{shard_path.name}"
            ) from exc
    missing = required_ids - set(documents)
    if missing:
        raise MetaSynScreeningStudyError(f"candidate_documents_missing:count={len(missing)}")
    return documents


def prepare_label_blind_features(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    retrieval_work_dir: Path,
    work_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze development/calibration pair features without opening any labels."""

    receipt_path = work_dir / "prepare_receipt.json"
    feature_path = work_dir / "pair_features.private.jsonl"
    existing = [path.as_posix() for path in (receipt_path, feature_path) if path.exists()]
    if existing and not force:
        raise MetaSynScreeningStudyError(f"prepare_outputs_exist:{existing}")

    retrieval, rankings_path = _validate_retrieval_freeze(
        retrieval_work_dir=retrieval_work_dir,
        benchmark_manifest_path=benchmark_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
    )
    benchmark = load_metasyn_manifest_metadata(benchmark_manifest_path)
    inputs = {
        split: load_metasyn_inputs(benchmark_manifest_path, split=split)
        for split in ("development", "calibration")
    }
    question_by_id = {
        question.review_id: question
        for split in ("development", "calibration")
        for question in inputs[split]
    }
    expected_ids = set(question_by_id)
    if len(expected_ids) != sum(len(rows) for rows in inputs.values()):
        raise MetaSynScreeningStudyError("question_crosses_development_calibration")
    rankings = _load_ordered_rankings(rankings_path, expected_review_ids=expected_ids)
    required_corpus_ids = {
        corpus_id
        for review_id in expected_ids
        for corpus_id in rankings[review_id]["rrf"]
    }
    corpus, shard_paths = verify_corpus_manifest(
        corpus_manifest_path, repository_root=repository_root
    )
    documents = _load_candidate_documents(
        shard_paths=shard_paths, required_ids=required_corpus_ids
    )
    split_by_review = {
        question.review_id: split
        for split, questions in inputs.items()
        for question in questions
    }
    feature_rows: list[dict[str, Any]] = []
    for review_id in sorted(expected_ids):
        ordered = rankings[review_id]
        bm25_rank = {
            corpus_id: rank
            for rank, corpus_id in enumerate(ordered["bm25"], start=1)
        }
        tfidf_rank = {
            corpus_id: rank
            for rank, corpus_id in enumerate(ordered["tfidf"], start=1)
        }
        for rrf_rank, corpus_id in enumerate(ordered["rrf"], start=1):
            # An RRF result can theoretically occur in only one source list.  The
            # frozen source depth plus one is the prespecified missing-rank value.
            source_missing_rank = 1_001
            title, abstract = documents[corpus_id]
            feature_rows.append(
                {
                    "split": split_by_review[review_id],
                    "review_id": review_id,
                    "corpus_id": corpus_id,
                    "original_rrf_rank": rrf_rank,
                    "features": _overlap_features(
                        question=question_by_id[review_id],
                        title=title,
                        abstract=abstract,
                        original_rrf_rank=rrf_rank,
                        bm25_rank=bm25_rank.get(corpus_id, source_missing_rank),
                        tfidf_rank=tfidf_rank.get(corpus_id, source_missing_rank),
                    ),
                }
            )
    atomic_write_jsonl(feature_path, feature_rows, force=force)
    source_code = _source_code_hashes()
    feature_config = _feature_config()
    selection_config = _selection_config()
    payload = {
        "metasyn_screening_prepare_version": STUDY_VERSION,
        "stage": "label_blind_pair_feature_freeze",
        "scientific_role": "retrospective_nonpristine_screening_reranking",
        "retrieval_freeze_payload_sha256": retrieval["freeze_payload_sha256"],
        "retrieval_freeze_receipt_sha256": sha256_file(
            retrieval_work_dir / "freeze_receipt.json"
        ),
        "retrieval_ordered_rankings_sha256": sha256_file(rankings_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_source_revision": corpus.source_revision,
        "corpus_rows": corpus.total_rows,
        "corpus_shard_sha256s": {
            shard.path: shard.sha256 for shard in corpus.shards
        },
        "model_input_sha256s": {
            "development": benchmark.development.sha256,
            "calibration": benchmark.calibration.sha256,
        },
        "feature_config": feature_config,
        "feature_config_sha256": hash_canonical(feature_config),
        "development_selection_config": selection_config,
        "development_selection_config_sha256": hash_canonical(selection_config),
        "calibration_evaluation_config": _evaluation_config(),
        "calibration_evaluation_config_sha256": hash_canonical(_evaluation_config()),
        "feature_rows_path": feature_path.relative_to(work_dir).as_posix(),
        "feature_rows_sha256": sha256_file(feature_path),
        "feature_rows": len(feature_rows),
        "questions_by_split": {
            split: len(inputs[split]) for split in ("development", "calibration")
        },
        "unique_candidate_documents": len(required_corpus_ids),
        "candidate_documents_with_title": sum(
            bool(title and title.strip()) for title, _ in documents.values()
        ),
        "candidate_documents_with_abstract": sum(
            bool(abstract and abstract.strip()) for _, abstract in documents.values()
        ),
        "access_boundary": {
            "development_labels_opened": False,
            "calibration_labels_opened": False,
            "official_test_inputs_opened": False,
            "official_test_labels_opened": False,
            "official_test_evaluated": False,
            "gold_fields_used_for_features": False,
            "candidate_conclusion_or_effect_fields_used": False,
            "labels_historically_opened_before_protocol": True,
            "pristine_holdout_eligible": False,
        },
        "source_code_sha256s": source_code,
        "runtime_versions": {
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    receipt = _attach_hash(payload, field="prepare_payload_sha256")
    atomic_write_json(receipt_path, receipt, force=force)
    return receipt


def _validate_feature_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_by_split: Mapping[str, int]
) -> None:
    expected_questions: dict[str, set[int]] = defaultdict(set)
    ranks: dict[tuple[str, int], list[int]] = defaultdict(list)
    corpus_ids: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        if set(row) != {"split", "review_id", "corpus_id", "original_rrf_rank", "features"}:
            raise MetaSynScreeningStudyError("feature_row_schema_invalid")
        split = row["split"]
        if split not in {"development", "calibration"}:
            raise MetaSynScreeningStudyError("feature_row_split_invalid")
        try:
            review_id = int(row["review_id"])
            corpus_id = int(row["corpus_id"])
            rank = int(row["original_rrf_rank"])
        except (TypeError, ValueError) as exc:
            raise MetaSynScreeningStudyError("feature_row_identity_invalid") from exc
        if review_id < 0 or corpus_id < 0:
            raise MetaSynScreeningStudyError("feature_row_identity_negative")
        features = row["features"]
        if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
            raise MetaSynScreeningStudyError("feature_row_features_invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in features.values()
        ):
            raise MetaSynScreeningStudyError("feature_row_feature_nonfinite")
        key = (split, review_id)
        expected_questions[split].add(review_id)
        ranks[key].append(rank)
        corpus_ids[key].append(corpus_id)
    if {
        split: len(expected_questions[split])
        for split in ("development", "calibration")
    } != dict(expected_by_split):
        raise MetaSynScreeningStudyError("feature_question_count_mismatch")
    for key in ranks:
        if sorted(ranks[key]) != list(range(1, RETRIEVAL_DEPTH + 1)):
            raise MetaSynScreeningStudyError("feature_ranking_depth_or_rank_invalid")
        if len(corpus_ids[key]) != len(set(corpus_ids[key])):
            raise MetaSynScreeningStudyError("feature_ranking_duplicate_document")


def validate_prepare(*, work_dir: Path) -> dict[str, Any]:
    receipt = _load_json(work_dir / "prepare_receipt.json")
    _verify_hash(receipt, field="prepare_payload_sha256")
    if receipt.get("metasyn_screening_prepare_version") != STUDY_VERSION:
        raise MetaSynScreeningStudyError("prepare_version_unsupported")
    access = receipt.get("access_boundary", {})
    if any(
        access.get(field) is not False
        for field in (
            "development_labels_opened",
            "calibration_labels_opened",
            "official_test_inputs_opened",
            "official_test_labels_opened",
            "official_test_evaluated",
            "gold_fields_used_for_features",
            "candidate_conclusion_or_effect_fields_used",
        )
    ):
        raise MetaSynScreeningStudyError("prepare_access_boundary_invalid")
    if receipt.get("source_code_sha256s") != _source_code_hashes():
        raise MetaSynScreeningStudyError("prepare_source_code_drift")
    if receipt.get("feature_config_sha256") != hash_canonical(receipt.get("feature_config")):
        raise MetaSynScreeningStudyError("prepare_feature_config_hash_mismatch")
    expected_selection = _selection_config()
    if receipt.get("development_selection_config") != expected_selection or receipt.get(
        "development_selection_config_sha256"
    ) != hash_canonical(expected_selection):
        raise MetaSynScreeningStudyError("prepare_selection_config_mismatch")
    expected_evaluation = _evaluation_config()
    if receipt.get("calibration_evaluation_config") != expected_evaluation or receipt.get(
        "calibration_evaluation_config_sha256"
    ) != hash_canonical(expected_evaluation):
        raise MetaSynScreeningStudyError("prepare_evaluation_config_mismatch")
    feature_path = _work_artifact(
        work_dir, receipt.get("feature_rows_path"), field="feature_rows_path"
    )
    if sha256_file(feature_path) != receipt.get("feature_rows_sha256"):
        raise MetaSynScreeningStudyError("prepare_feature_rows_hash_mismatch")
    rows = _load_jsonl(feature_path)
    if len(rows) != receipt.get("feature_rows"):
        raise MetaSynScreeningStudyError("prepare_feature_row_count_mismatch")
    _validate_feature_rows(rows, expected_by_split=receipt["questions_by_split"])
    return receipt


def _normalize_link_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


def _integer_ids(value: Any, *, field: str, review_id: int) -> tuple[int, ...]:
    if value is None:
        raw: list[Any] = []
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        raw = converted if isinstance(converted, list) else [converted]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raise MetaSynScreeningStudyError(f"label_{field}_invalid:review={review_id}")
    try:
        ids = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as exc:
        raise MetaSynScreeningStudyError(
            f"label_{field}_invalid:review={review_id}"
        ) from exc
    if any(item < 0 for item in ids):
        raise MetaSynScreeningStudyError(f"label_{field}_negative:review={review_id}")
    return ids


def _load_split_labels_from_train(
    *,
    benchmark_manifest_path: Path,
    review_cache_dir: Path,
    split: Literal["development", "calibration"],
) -> tuple[list[_ScreeningLabel], dict[str, Any]]:
    """Materialize only one official-train split using a Parquet row predicate.

    Development and calibration share the physical train Parquet.  This loader asks
    Arrow to materialize only the exact manifest-listed rows and never opens the
    official-test Parquet or shared evaluator-label JSONL.  The physical co-location
    limitation is carried into every receipt.
    """

    benchmark = load_metasyn_manifest_metadata(benchmark_manifest_path)
    artifact = getattr(benchmark, split)
    source_path = review_cache_dir / benchmark.source_train.filename
    if sha256_file(source_path) != benchmark.source_train.sha256:
        raise MetaSynScreeningStudyError("source_train_parquet_hash_mismatch")
    columns = [
        "ID",
        "Title",
        "Research_Question",
        "matched_corpus_ids",
        "matched_ref_count",
        "source_review_corpus_ids",
    ]
    try:
        table = pq.read_table(
            source_path,
            columns=columns,
            filters=[("ID", "in", list(artifact.review_ids))],
        )
        rows = table.to_pylist()
    except Exception as exc:
        raise MetaSynScreeningStudyError(f"source_train_split_read_failed:{split}") from exc
    ids = [int(row["ID"]) for row in rows]
    if len(ids) != len(set(ids)) or sorted(ids) != artifact.review_ids:
        raise MetaSynScreeningStudyError(f"source_train_split_rows_mismatch:{split}")

    union_find = _UnionFind(ids)
    owner: dict[str, int] = {}
    normalized_rows: dict[int, tuple[int, ...]] = {}
    for row in rows:
        review_id = int(row["ID"])
        gold = _integer_ids(
            row["matched_corpus_ids"], field="matched_corpus_ids", review_id=review_id
        )
        if not gold:
            raise MetaSynScreeningStudyError(f"label_gold_empty:review={review_id}")
        if int(row["matched_ref_count"]) != len(gold):
            raise MetaSynScreeningStudyError(f"label_gold_count_mismatch:review={review_id}")
        source_ids = _integer_ids(
            row["source_review_corpus_ids"],
            field="source_review_corpus_ids",
            review_id=review_id,
        )
        tokens = [f"paper:{item}" for item in gold]
        tokens.extend(f"source-review:{item}" for item in source_ids)
        title_key = _normalize_link_text(row["Title"])
        question_key = _normalize_link_text(row["Research_Question"])
        if title_key:
            tokens.append(f"title:{title_key}")
        if question_key:
            tokens.append(f"question:{question_key}")
        for token in tokens:
            prior = owner.setdefault(token, review_id)
            union_find.union(review_id, prior)
        normalized_rows[review_id] = gold

    grouped: dict[int, list[int]] = defaultdict(list)
    for review_id in ids:
        grouped[union_find.find(review_id)].append(review_id)
    component_by_review: dict[int, str] = {}
    for members in grouped.values():
        ordered_members = sorted(members)
        component_id = f"metasyn-component-{hash_canonical(ordered_members)[:20]}"
        for review_id in ordered_members:
            component_by_review[review_id] = component_id
    if sorted(set(component_by_review.values())) != artifact.component_ids:
        raise MetaSynScreeningStudyError(f"source_train_component_mismatch:{split}")
    labels = [
        _ScreeningLabel(
            review_id=review_id,
            component_id=component_by_review[review_id],
            gold_corpus_ids=normalized_rows[review_id],
        )
        for review_id in sorted(ids)
    ]
    access = {
        "logical_split_materialized": split,
        "rows_materialized": len(labels),
        "columns_materialized": columns,
        "manifest_row_predicate_applied": True,
        "official_test_parquet_opened": False,
        "shared_evaluator_label_jsonl_opened": False,
        "source_train_parquet_physically_colocates_development_and_calibration": True,
        "strict_storage_level_unopened_claim_possible": False,
        "source_train_parquet_sha256": benchmark.source_train.sha256,
    }
    return labels, access


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    review_ids = np.asarray([int(row["review_id"]) for row in rows], dtype=np.int64)
    corpus_ids = np.asarray([int(row["corpus_id"]) for row in rows], dtype=np.int64)
    original_ranks = np.asarray(
        [int(row["original_rrf_rank"]) for row in rows], dtype=np.int64
    )
    return matrix, review_ids, corpus_ids, original_ranks


def _binary_targets(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[_ScreeningLabel]
) -> np.ndarray:
    gold_by_review = {label.review_id: set(label.gold_corpus_ids) for label in labels}
    row_review_ids = {int(row["review_id"]) for row in rows}
    if row_review_ids != set(gold_by_review):
        raise MetaSynScreeningStudyError("feature_label_question_universe_mismatch")
    targets = np.asarray(
        [
            int(int(row["corpus_id"]) in gold_by_review[int(row["review_id"])])
            for row in rows
        ],
        dtype=np.int8,
    )
    if targets.min(initial=0) != 0 or targets.max(initial=0) != 1:
        raise MetaSynScreeningStudyError("pair_target_not_binary")
    if not targets.any() or targets.all():
        raise MetaSynScreeningStudyError("pair_training_requires_both_classes")
    return targets


def _make_estimator(candidate: ScreeningCandidate, *, seed: int) -> Any:
    if candidate == "logistic-l2-balanced-v1":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.25,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if candidate == "monotonic-hist-gradient-balanced-v1":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=False,
            l2_regularization=0.1,
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            monotonic_cst=[1] * len(FEATURE_NAMES),
            random_state=seed,
        )
    if candidate == BASELINE_ID:
        return None
    raise MetaSynScreeningStudyError(f"screening_candidate_unknown:{candidate}")


def deterministic_rerank(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> list[int]:
    """Return a score order with a stable ascending-corpus-ID tie break."""

    if len(rows) != len(scores) or len(rows) != RETRIEVAL_DEPTH:
        raise MetaSynScreeningStudyError("rerank_rows_or_scores_wrong_depth")
    pairs: list[tuple[float, int]] = []
    for row, score in zip(rows, scores, strict=True):
        rendered = float(score)
        if not math.isfinite(rendered):
            raise MetaSynScreeningStudyError("rerank_score_nonfinite")
        pairs.append((rendered, int(row["corpus_id"])))
    if len({corpus_id for _, corpus_id in pairs}) != RETRIEVAL_DEPTH:
        raise MetaSynScreeningStudyError("rerank_documents_not_unique")
    return [corpus_id for _, corpus_id in sorted(pairs, key=lambda item: (-item[0], item[1]))]


def _scores_for_candidate(
    candidate: ScreeningCandidate,
    *,
    estimator: Any,
    matrix: np.ndarray,
    original_ranks: np.ndarray,
) -> np.ndarray:
    if candidate == BASELINE_ID:
        return -original_ranks.astype(np.float64)
    probabilities = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    if probabilities.shape != (len(matrix), 2):
        raise MetaSynScreeningStudyError("reranker_probability_shape_invalid")
    return probabilities[:, 1]


def _rank_rows_by_question(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> dict[int, list[int]]:
    grouped_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    grouped_scores: dict[int, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        review_id = int(row["review_id"])
        grouped_rows[review_id].append(row)
        grouped_scores[review_id].append(float(score))
    return {
        review_id: deterministic_rerank(grouped_rows[review_id], grouped_scores[review_id])
        for review_id in sorted(grouped_rows)
    }


def _selection_metrics(
    *, rankings: Mapping[int, Sequence[int]], labels: Sequence[_ScreeningLabel]
) -> dict[str, Any]:
    if set(rankings) != {label.review_id for label in labels}:
        raise MetaSynScreeningStudyError("selection_ranking_label_universe_mismatch")
    depth_metrics: dict[str, dict[str, float | int]] = {}
    for depth in EVALUATION_DEPTHS:
        recalls: list[float] = []
        hits = gold_count = zero = full = 0
        for label in labels:
            ranking = list(rankings[label.review_id])
            if len(ranking) != RETRIEVAL_DEPTH or len(set(ranking)) != RETRIEVAL_DEPTH:
                raise MetaSynScreeningStudyError("selection_ranking_depth_invalid")
            gold = set(label.gold_corpus_ids)
            retained = len(gold & set(ranking[:depth]))
            recalls.append(retained / len(gold))
            hits += retained
            gold_count += len(gold)
            zero += int(retained == 0)
            full += int(retained == len(gold))
        depth_metrics[str(depth)] = {
            "question_macro_absolute_recall": float(np.mean(recalls)),
            "micro_absolute_recall": float(hits / gold_count),
            "matched_references_retained": hits,
            "matched_references_total": gold_count,
            "questions_with_zero_retained": zero,
            "questions_with_full_inclusion": full,
        }
    selection_score = float(
        np.mean(
            [
                depth_metrics[str(depth)]["question_macro_absolute_recall"]
                for depth in SELECTION_DEPTHS
            ]
        )
    )
    return {
        "selection_score": selection_score,
        "selection_score_definition": (
            "unweighted_mean_question_macro_absolute_recall_at_10_20_50_100"
        ),
        "depths": depth_metrics,
    }


def _cross_validated_rankings(
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[_ScreeningLabel],
    folds: int,
    seed: int,
) -> tuple[dict[ScreeningCandidate, dict[int, list[int]]], list[dict[str, Any]]]:
    label_by_review = {label.review_id: label for label in labels}
    question_ids = np.asarray(sorted(label_by_review), dtype=np.int64)
    groups = np.asarray(
        [label_by_review[int(review_id)].component_id for review_id in question_ids],
        dtype=object,
    )
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise MetaSynScreeningStudyError("component_disjoint_cv_requires_two_components")
    fold_count = min(folds, len(unique_groups))
    splitter = GroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    matrix, row_review_ids, _, original_ranks = _feature_matrix(rows)
    targets = _binary_targets(rows, labels)
    rankings: dict[ScreeningCandidate, dict[int, list[int]]] = {
        candidate: {} for candidate in CANDIDATE_IDS
    }
    fold_receipts: list[dict[str, Any]] = []
    for fold_index, (train_question_indices, test_question_indices) in enumerate(
        splitter.split(question_ids, groups=groups)
    ):
        train_questions = set(question_ids[train_question_indices].tolist())
        test_questions = set(question_ids[test_question_indices].tolist())
        train_groups = set(groups[train_question_indices].tolist())
        test_groups = set(groups[test_question_indices].tolist())
        if train_questions & test_questions or train_groups & test_groups:
            raise MetaSynScreeningStudyError("component_disjoint_cv_overlap")
        train_mask = np.isin(row_review_ids, np.asarray(sorted(train_questions)))
        test_mask = np.isin(row_review_ids, np.asarray(sorted(test_questions)))
        if not train_mask.any() or not test_mask.any():
            raise MetaSynScreeningStudyError("component_disjoint_cv_fold_empty")
        if len(set(targets[train_mask].tolist())) != 2:
            raise MetaSynScreeningStudyError("component_disjoint_cv_training_class_missing")
        test_rows = [row for row, keep in zip(rows, test_mask, strict=True) if keep]
        for candidate in CANDIDATE_IDS:
            estimator = _make_estimator(candidate, seed=seed + fold_index)
            if estimator is not None:
                estimator.fit(matrix[train_mask], targets[train_mask])
            scores = _scores_for_candidate(
                candidate,
                estimator=estimator,
                matrix=matrix[test_mask],
                original_ranks=original_ranks[test_mask],
            )
            candidate_rankings = _rank_rows_by_question(test_rows, scores)
            if set(candidate_rankings) != test_questions:
                raise MetaSynScreeningStudyError("component_disjoint_cv_test_universe_mismatch")
            rankings[candidate].update(candidate_rankings)
        fold_receipts.append(
            {
                "fold": fold_index,
                "training_questions": len(train_questions),
                "validation_questions": len(test_questions),
                "training_components": len(train_groups),
                "validation_components": len(test_groups),
                "training_component_ids": sorted(train_groups),
                "validation_component_ids": sorted(test_groups),
                "component_overlap": 0,
            }
        )
    for candidate, candidate_rankings in rankings.items():
        if set(candidate_rankings) != set(question_ids.tolist()):
            raise MetaSynScreeningStudyError(f"candidate_oof_rows_missing:{candidate}")
    return rankings, fold_receipts


def _fit_full_and_rank(
    *,
    candidate: ScreeningCandidate,
    development_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[_ScreeningLabel],
    seed: int,
) -> dict[str, dict[int, list[int]]]:
    development_matrix, _, _, development_ranks = _feature_matrix(development_rows)
    calibration_matrix, _, _, calibration_ranks = _feature_matrix(calibration_rows)
    targets = _binary_targets(development_rows, labels)
    estimator = _make_estimator(candidate, seed=seed)
    if estimator is not None:
        estimator.fit(development_matrix, targets)
    development_scores = _scores_for_candidate(
        candidate,
        estimator=estimator,
        matrix=development_matrix,
        original_ranks=development_ranks,
    )
    calibration_scores = _scores_for_candidate(
        candidate,
        estimator=estimator,
        matrix=calibration_matrix,
        original_ranks=calibration_ranks,
    )
    return {
        "development": _rank_rows_by_question(development_rows, development_scores),
        "calibration": _rank_rows_by_question(calibration_rows, calibration_scores),
    }


def fit_and_freeze_winner(
    *,
    benchmark_manifest_path: Path,
    review_cache_dir: Path,
    work_dir: Path,
    force: bool = False,
    cv_folds: int = CV_FOLDS,
    cv_seed: int = CV_SEED,
) -> dict[str, Any]:
    """Select on development only and freeze exactly one calibration ordering."""

    receipt_path = work_dir / "fit_receipt.json"
    oof_path = work_dir / "development_oof_rankings.private.jsonl"
    winner_path = work_dir / "winner_rankings.private.jsonl"
    existing = [
        path.as_posix() for path in (receipt_path, oof_path, winner_path) if path.exists()
    ]
    if existing and not force:
        raise MetaSynScreeningStudyError(f"fit_outputs_exist:{existing}")
    prepare = validate_prepare(work_dir=work_dir)
    frozen_selection = prepare["development_selection_config"]
    if cv_folds != frozen_selection["folds"] or cv_seed != frozen_selection["seed"]:
        raise MetaSynScreeningStudyError("fit_configuration_differs_from_prepare_freeze")
    feature_path = work_dir / prepare["feature_rows_path"]
    feature_rows = _load_jsonl(feature_path)
    development_rows = [row for row in feature_rows if row["split"] == "development"]
    calibration_rows = [row for row in feature_rows if row["split"] == "calibration"]
    labels, label_access = _load_split_labels_from_train(
        benchmark_manifest_path=benchmark_manifest_path,
        review_cache_dir=review_cache_dir,
        split="development",
    )
    oof_rankings, fold_receipts = _cross_validated_rankings(
        rows=development_rows,
        labels=labels,
        folds=cv_folds,
        seed=cv_seed,
    )
    results = {
        candidate: _selection_metrics(rankings=oof_rankings[candidate], labels=labels)
        for candidate in CANDIDATE_IDS
    }
    selected: ScreeningCandidate = min(
        CANDIDATE_IDS,
        key=lambda candidate: (-results[candidate]["selection_score"], candidate),
    )
    frozen_rankings = _fit_full_and_rank(
        candidate=selected,
        development_rows=development_rows,
        calibration_rows=calibration_rows,
        labels=labels,
        seed=cv_seed,
    )
    original_by_split: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(dict)
    for row in feature_rows:
        split = str(row["split"])
        review_id = int(row["review_id"])
        original_by_split[split].setdefault(review_id, []).append(
            (int(row["original_rrf_rank"]), int(row["corpus_id"]))
        )
    normalized_original = {
        split: {
            review_id: [
                corpus_id for _, corpus_id in sorted(ordered_rank_pairs)
            ]
            for review_id, ordered_rank_pairs in by_review.items()
        }
        for split, by_review in original_by_split.items()
    }
    for split in ("development", "calibration"):
        if set(frozen_rankings[split]) != set(normalized_original[split]):
            raise MetaSynScreeningStudyError("winner_ranking_question_universe_mismatch")
        for review_id, ranking in frozen_rankings[split].items():
            if set(ranking) != set(normalized_original[split][review_id]):
                raise MetaSynScreeningStudyError("winner_changed_retrieval_candidate_set")

    oof_rows = [
        {
            "review_id": review_id,
            "candidate_rankings": {
                candidate: oof_rankings[candidate][review_id]
                for candidate in CANDIDATE_IDS
            },
        }
        for review_id in sorted(oof_rankings[BASELINE_ID])
    ]
    winner_rows = [
        {
            "split": split,
            "review_id": review_id,
            "selected_candidate": selected,
            "ordered_corpus_ids": frozen_rankings[split][review_id],
            "original_rrf_ordered_corpus_ids": normalized_original[split][review_id],
        }
        for split in ("development", "calibration")
        for review_id in sorted(frozen_rankings[split])
    ]
    atomic_write_jsonl(oof_path, oof_rows, force=force)
    atomic_write_jsonl(winner_path, winner_rows, force=force)
    configs = _candidate_configs()
    label_identity = [
        {
            "review_id": label.review_id,
            "component_id": label.component_id,
            "gold_corpus_ids": list(label.gold_corpus_ids),
        }
        for label in labels
    ]
    payload = {
        "metasyn_screening_fit_version": STUDY_VERSION,
        "stage": "development_only_grouped_cv_selection_and_winner_freeze",
        "prepare_payload_sha256": prepare["prepare_payload_sha256"],
        "prepare_receipt_sha256": sha256_file(work_dir / "prepare_receipt.json"),
        "feature_rows_sha256": sha256_file(feature_path),
        "development_label_identity_sha256": hash_canonical(label_identity),
        "development_label_access": label_access,
        "candidate_family": frozen_selection["candidate_family"],
        "selection_rule": {
            "candidate_ids": list(frozen_selection["candidate_family"]),
            "cross_validation": frozen_selection["cross_validation"],
            "folds_requested": frozen_selection["folds"],
            "folds_executed": len(fold_receipts),
            "seed": frozen_selection["seed"],
            "groups": frozen_selection["groups"],
            "metric": frozen_selection["metric"],
            "maximize": frozen_selection["maximize"],
            "tie_break": frozen_selection["tie_break"],
            "depth_200_excluded_because_invariant_under_within_set_reranking": (
                frozen_selection[
                    "depth_200_excluded_because_invariant_under_within_set_reranking"
                ]
            ),
        },
        "cv_folds": fold_receipts,
        "development_oof_results": results,
        "selected_candidate": selected,
        "selected_candidate_config_sha256": hash_canonical(configs[selected]),
        "development_oof_rankings_path": oof_path.relative_to(work_dir).as_posix(),
        "development_oof_rankings_sha256": sha256_file(oof_path),
        "winner_rankings_path": winner_path.relative_to(work_dir).as_posix(),
        "winner_rankings_sha256": sha256_file(winner_path),
        "winner_rankings_rows": len(winner_rows),
        "access_boundary": {
            "development_labels_materialized": True,
            "calibration_labels_materialized": False,
            "official_test_inputs_opened": False,
            "official_test_labels_opened": False,
            "official_test_evaluated": False,
            "calibration_features_scored_by_development_trained_winner": True,
            "calibration_outcomes_used_for_selection": False,
            "labels_historically_opened_before_protocol": True,
            "pristine_holdout_eligible": False,
        },
        "implicit_negative_contract": {
            "positive_pairs": "released_MetaSyn_matched_identifiers_within_RRF_top_200",
            "negative_pairs": "all_other_RRF_top_200_candidates",
            "unknown_eligibility_possible": True,
            "released_matching_is_not_exhaustive_eligibility_annotation": True,
        },
        "source_code_sha256s": _source_code_hashes(),
    }
    receipt = _attach_hash(payload, field="fit_payload_sha256")
    atomic_write_json(receipt_path, receipt, force=force)
    return receipt


def _load_winner_rankings(
    *, work_dir: Path, fit: Mapping[str, Any]
) -> dict[str, dict[str, dict[int, list[int]]]]:
    path = _work_artifact(
        work_dir, fit.get("winner_rankings_path"), field="winner_rankings_path"
    )
    if sha256_file(path) != fit.get("winner_rankings_sha256"):
        raise MetaSynScreeningStudyError("winner_rankings_hash_mismatch")
    rows = _load_jsonl(path)
    if len(rows) != fit.get("winner_rankings_rows"):
        raise MetaSynScreeningStudyError("winner_rankings_row_count_mismatch")
    selected = fit.get("selected_candidate")
    output: dict[str, dict[str, dict[int, list[int]]]] = {
        "selected": defaultdict(dict),
        "baseline": defaultdict(dict),
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if set(row) != {
            "split",
            "review_id",
            "selected_candidate",
            "ordered_corpus_ids",
            "original_rrf_ordered_corpus_ids",
        }:
            raise MetaSynScreeningStudyError("winner_ranking_row_schema_invalid")
        split = str(row["split"])
        review_id = int(row["review_id"])
        if split not in {"development", "calibration"} or (split, review_id) in seen:
            raise MetaSynScreeningStudyError("winner_ranking_row_identity_invalid")
        seen.add((split, review_id))
        if row["selected_candidate"] != selected:
            raise MetaSynScreeningStudyError("winner_ranking_selected_candidate_mismatch")
        winner = [int(value) for value in row["ordered_corpus_ids"]]
        baseline = [int(value) for value in row["original_rrf_ordered_corpus_ids"]]
        if (
            len(winner) != RETRIEVAL_DEPTH
            or len(set(winner)) != RETRIEVAL_DEPTH
            or len(baseline) != RETRIEVAL_DEPTH
            or len(set(baseline)) != RETRIEVAL_DEPTH
        ):
            raise MetaSynScreeningStudyError("winner_ranking_depth_invalid")
        if set(winner) != set(baseline):
            raise MetaSynScreeningStudyError("winner_changed_retrieval_candidate_set")
        output["selected"][split][review_id] = winner
        output["baseline"][split][review_id] = baseline
    return output


def validate_fit(*, work_dir: Path) -> dict[str, Any]:
    prepare = validate_prepare(work_dir=work_dir)
    receipt = _load_json(work_dir / "fit_receipt.json")
    _verify_hash(receipt, field="fit_payload_sha256")
    if receipt.get("metasyn_screening_fit_version") != STUDY_VERSION:
        raise MetaSynScreeningStudyError("fit_version_unsupported")
    if receipt.get("prepare_payload_sha256") != prepare.get("prepare_payload_sha256"):
        raise MetaSynScreeningStudyError("fit_prepare_mismatch")
    if receipt.get("prepare_receipt_sha256") != sha256_file(
        work_dir / "prepare_receipt.json"
    ):
        raise MetaSynScreeningStudyError("fit_prepare_receipt_hash_mismatch")
    if receipt.get("feature_rows_sha256") != prepare.get("feature_rows_sha256"):
        raise MetaSynScreeningStudyError("fit_feature_rows_mismatch")
    if receipt.get("source_code_sha256s") != _source_code_hashes():
        raise MetaSynScreeningStudyError("fit_source_code_drift")
    access = receipt.get("access_boundary", {})
    if access.get("calibration_labels_materialized") is not False:
        raise MetaSynScreeningStudyError("fit_opened_calibration_labels")
    if any(
        access.get(field) is not False
        for field in (
            "official_test_inputs_opened",
            "official_test_labels_opened",
            "official_test_evaluated",
            "calibration_outcomes_used_for_selection",
        )
    ):
        raise MetaSynScreeningStudyError("fit_access_boundary_invalid")
    selected = receipt.get("selected_candidate")
    if selected not in CANDIDATE_IDS:
        raise MetaSynScreeningStudyError("fit_selected_candidate_invalid")
    results = receipt.get("development_oof_results", {})
    configs = _candidate_configs()
    family = receipt.get("candidate_family", {})
    if family != prepare["development_selection_config"]["candidate_family"]:
        raise MetaSynScreeningStudyError("fit_candidate_family_not_prepare_frozen")
    for candidate in CANDIDATE_IDS:
        record = family.get(candidate)
        if not isinstance(record, dict):
            raise MetaSynScreeningStudyError(f"fit_candidate_config_missing:{candidate}")
        if record.get("config") != configs[candidate] or record.get(
            "config_sha256"
        ) != hash_canonical(configs[candidate]):
            raise MetaSynScreeningStudyError(f"fit_candidate_config_mismatch:{candidate}")
    if receipt.get("selected_candidate_config_sha256") != hash_canonical(
        configs[selected]
    ):
        raise MetaSynScreeningStudyError("fit_selected_candidate_config_mismatch")
    expected = min(
        CANDIDATE_IDS,
        key=lambda candidate: (-results[candidate]["selection_score"], candidate),
    )
    if selected != expected:
        raise MetaSynScreeningStudyError("fit_selection_rule_not_replayed")
    rule = receipt.get("selection_rule", {})
    frozen_rule = prepare["development_selection_config"]
    expected_rule_fields = {
        "candidate_ids": list(frozen_rule["candidate_family"]),
        "cross_validation": frozen_rule["cross_validation"],
        "folds_requested": frozen_rule["folds"],
        "seed": frozen_rule["seed"],
        "groups": frozen_rule["groups"],
        "metric": frozen_rule["metric"],
        "maximize": frozen_rule["maximize"],
        "tie_break": frozen_rule["tie_break"],
        "depth_200_excluded_because_invariant_under_within_set_reranking": frozen_rule[
            "depth_200_excluded_because_invariant_under_within_set_reranking"
        ],
    }
    if any(rule.get(key) != value for key, value in expected_rule_fields.items()):
        raise MetaSynScreeningStudyError("fit_selection_rule_not_prepare_frozen")
    folds = receipt.get("cv_folds")
    if not isinstance(folds, list) or not folds:
        raise MetaSynScreeningStudyError("fit_cv_folds_missing")
    if receipt.get("selection_rule", {}).get("folds_executed") != len(folds):
        raise MetaSynScreeningStudyError("fit_cv_fold_count_mismatch")
    if [fold.get("fold") for fold in folds] != list(range(len(folds))):
        raise MetaSynScreeningStudyError("fit_cv_fold_indices_invalid")
    validation_components: list[str] = []
    for fold in folds:
        training = set(fold.get("training_component_ids", []))
        validation = set(fold.get("validation_component_ids", []))
        if not training or not validation or training & validation:
            raise MetaSynScreeningStudyError("fit_cv_component_boundary_invalid")
        if fold.get("component_overlap") != 0:
            raise MetaSynScreeningStudyError("fit_cv_component_overlap_nonzero")
        validation_components.extend(validation)
    if len(validation_components) != len(set(validation_components)):
        raise MetaSynScreeningStudyError("fit_cv_component_validated_more_than_once")
    oof_path = _work_artifact(
        work_dir,
        receipt.get("development_oof_rankings_path"),
        field="development_oof_rankings_path",
    )
    if sha256_file(oof_path) != receipt.get("development_oof_rankings_sha256"):
        raise MetaSynScreeningStudyError("development_oof_rankings_hash_mismatch")
    oof_rows = _load_jsonl(oof_path)
    if len(oof_rows) != prepare["questions_by_split"]["development"]:
        raise MetaSynScreeningStudyError("development_oof_ranking_row_count_mismatch")
    seen_questions: set[int] = set()
    for row in oof_rows:
        if set(row) != {"review_id", "candidate_rankings"}:
            raise MetaSynScreeningStudyError("development_oof_ranking_schema_invalid")
        review_id = int(row["review_id"])
        if review_id in seen_questions:
            raise MetaSynScreeningStudyError("development_oof_ranking_duplicate_question")
        seen_questions.add(review_id)
        candidate_rankings = row["candidate_rankings"]
        if not isinstance(candidate_rankings, dict) or set(candidate_rankings) != set(
            CANDIDATE_IDS
        ):
            raise MetaSynScreeningStudyError("development_oof_candidate_family_mismatch")
        baseline_set = set(candidate_rankings[BASELINE_ID])
        if len(baseline_set) != RETRIEVAL_DEPTH:
            raise MetaSynScreeningStudyError("development_oof_baseline_depth_invalid")
        if any(
            len(ranking) != RETRIEVAL_DEPTH
            or len(set(ranking)) != RETRIEVAL_DEPTH
            or set(ranking) != baseline_set
            for ranking in candidate_rankings.values()
        ):
            raise MetaSynScreeningStudyError("development_oof_candidate_ranking_invalid")
    winner = _load_winner_rankings(work_dir=work_dir, fit=receipt)
    feature_path = _work_artifact(
        work_dir, prepare.get("feature_rows_path"), field="feature_rows_path"
    )
    original_from_features: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(dict)
    for row in _load_jsonl(feature_path):
        split = str(row["split"])
        review_id = int(row["review_id"])
        original_from_features[split].setdefault(review_id, []).append(
            (int(row["original_rrf_rank"]), int(row["corpus_id"]))
        )
    for kind in ("selected", "baseline"):
        if len(winner[kind]["development"]) != prepare["questions_by_split"]["development"]:
            raise MetaSynScreeningStudyError("winner_development_question_count_mismatch")
        if len(winner[kind]["calibration"]) != prepare["questions_by_split"]["calibration"]:
            raise MetaSynScreeningStudyError("winner_calibration_question_count_mismatch")
    for split, by_review in original_from_features.items():
        for review_id, ranked_pairs in by_review.items():
            expected_original = [corpus_id for _, corpus_id in sorted(ranked_pairs)]
            if winner["baseline"][split][review_id] != expected_original:
                raise MetaSynScreeningStudyError("winner_baseline_not_original_rrf_order")
    return receipt


def _point_metrics(
    *,
    ranking: Mapping[int, Sequence[int]],
    baseline: Mapping[int, Sequence[int]],
    labels: Sequence[_ScreeningLabel],
    depth: int,
) -> dict[str, Any]:
    absolute_recalls: list[float] = []
    conditional_recalls: list[float] = []
    absolute_hits = absolute_denominator = 0
    conditional_hits = conditional_denominator = 0
    zero_retained = full_inclusion = conditional_full = 0
    zero_retrievable = 0
    for label in labels:
        gold = set(label.gold_corpus_ids)
        retrievable = gold & set(baseline[label.review_id])
        hits = len(gold & set(ranking[label.review_id][:depth]))
        absolute_hits += hits
        absolute_denominator += len(gold)
        absolute_recalls.append(hits / len(gold))
        zero_retained += int(hits == 0)
        full_inclusion += int(hits == len(gold))
        if retrievable:
            conditional_hits += hits
            conditional_denominator += len(retrievable)
            conditional_recalls.append(hits / len(retrievable))
            conditional_full += int(hits == len(retrievable))
        else:
            zero_retrievable += 1
            if hits != 0:
                raise MetaSynScreeningStudyError("hit_outside_frozen_candidate_set")
    question_count = len(labels)
    conditional_questions = len(conditional_recalls)
    return {
        "documents_screened_per_review": depth,
        "total_document_review_pairs_screened": question_count * depth,
        "matched_references_retained": absolute_hits,
        "matched_references_total": absolute_denominator,
        "question_macro_absolute_recall": float(np.mean(absolute_recalls)),
        "micro_absolute_recall": float(absolute_hits / absolute_denominator),
        "questions_with_zero_retained": zero_retained,
        "questions_with_zero_retained_rate": float(zero_retained / question_count),
        "questions_with_full_inclusion": full_inclusion,
        "full_inclusion_rate": float(full_inclusion / question_count),
        "conditional_retrievable_references_retained": conditional_hits,
        "conditional_retrievable_references_total": conditional_denominator,
        "conditional_questions": conditional_questions,
        "questions_with_zero_retrievable_matched_references": zero_retrievable,
        "question_macro_conditional_survival": (
            float(np.mean(conditional_recalls)) if conditional_recalls else None
        ),
        "micro_conditional_survival": (
            float(conditional_hits / conditional_denominator)
            if conditional_denominator
            else None
        ),
        "conditional_full_survival_rate": (
            float(conditional_full / conditional_questions) if conditional_questions else None
        ),
    }


def _cluster_summaries(
    *,
    ranking: Mapping[int, Sequence[int]],
    baseline: Mapping[int, Sequence[int]],
    labels: Sequence[_ScreeningLabel],
    depth: int,
    cluster_ids: Sequence[str],
) -> np.ndarray:
    cluster_index = {component: index for index, component in enumerate(cluster_ids)}
    # q, abs recall sum, hits, gold, conditional q, conditional recall sum,
    # conditional hits, conditional gold, zero retained, full inclusion
    values = np.zeros((len(cluster_ids), 10), dtype=np.float64)
    for label in labels:
        index = cluster_index[label.component_id]
        gold = set(label.gold_corpus_ids)
        retrievable = gold & set(baseline[label.review_id])
        hits = len(gold & set(ranking[label.review_id][:depth]))
        values[index, 0] += 1
        values[index, 1] += hits / len(gold)
        values[index, 2] += hits
        values[index, 3] += len(gold)
        values[index, 8] += int(hits == 0)
        values[index, 9] += int(hits == len(gold))
        if retrievable:
            values[index, 4] += 1
            values[index, 5] += hits / len(retrievable)
            values[index, 6] += hits
            values[index, 7] += len(retrievable)
    return values


def _bootstrap_metrics(values: np.ndarray, sampled: np.ndarray) -> dict[str, np.ndarray]:
    totals = values[sampled].sum(axis=1)
    return {
        "question_macro_absolute_recall": totals[:, 1] / totals[:, 0],
        "micro_absolute_recall": totals[:, 2] / totals[:, 3],
        "question_macro_conditional_survival": np.divide(
            totals[:, 5],
            totals[:, 4],
            out=np.full(len(totals), np.nan),
            where=totals[:, 4] > 0,
        ),
        "micro_conditional_survival": np.divide(
            totals[:, 6],
            totals[:, 7],
            out=np.full(len(totals), np.nan),
            where=totals[:, 7] > 0,
        ),
        "questions_with_zero_retained_rate": totals[:, 8] / totals[:, 0],
        "full_inclusion_rate": totals[:, 9] / totals[:, 0],
    }


def _cluster_bootstrap(
    *,
    selected: Mapping[int, Sequence[int]],
    baseline: Mapping[int, Sequence[int]],
    labels: Sequence[_ScreeningLabel],
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if replicates < 10_000:
        raise MetaSynScreeningStudyError("bootstrap_requires_at_least_10000_replicates")
    cluster_ids = sorted({label.component_id for label in labels})
    if not cluster_ids:
        raise MetaSynScreeningStudyError("bootstrap_components_empty")
    rng = np.random.default_rng(seed)
    samples_by_method_depth: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    # Reuse exactly the same sampled components for all methods/depths.
    sampled_chunks: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        sampled_chunks.append(
            rng.integers(0, len(cluster_ids), size=(size, len(cluster_ids)))
        )
    for method, ranking in (("selected", selected), ("rrf_baseline", baseline)):
        for depth in EVALUATION_DEPTHS:
            cluster_values = _cluster_summaries(
                ranking=ranking,
                baseline=baseline,
                labels=labels,
                depth=depth,
                cluster_ids=cluster_ids,
            )
            metric_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
            for sampled in sampled_chunks:
                for metric, values in _bootstrap_metrics(cluster_values, sampled).items():
                    metric_chunks[metric].append(values)
            samples_by_method_depth[(method, depth)] = {
                metric: np.concatenate(chunks) for metric, chunks in metric_chunks.items()
            }

    intervals: dict[str, Any] = {"selected": {}, "rrf_baseline": {}}
    deltas: dict[str, Any] = {}
    for depth in EVALUATION_DEPTHS:
        for method in ("selected", "rrf_baseline"):
            intervals[method][str(depth)] = {
                metric: [
                    float(value)
                    for value in np.nanquantile(samples, [0.025, 0.975])
                ]
                for metric, samples in samples_by_method_depth[(method, depth)].items()
            }
        selected_samples = samples_by_method_depth[("selected", depth)]
        baseline_samples = samples_by_method_depth[("rrf_baseline", depth)]
        deltas[str(depth)] = {
            metric: [
                float(value)
                for value in np.nanquantile(
                    selected_samples[metric] - baseline_samples[metric], [0.025, 0.975]
                )
            ]
            for metric in selected_samples
        }
    uncertainty = {
        "method": "paired_nonparametric_percentile_component_cluster_bootstrap",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "clusters": len(cluster_ids),
        "questions": len(labels),
        "singleton_clusters": sum(
            sum(label.component_id == component for label in labels) == 1
            for component in cluster_ids
        ),
        "maximum_questions_per_cluster": max(
            sum(label.component_id == component for label in labels)
            for component in cluster_ids
        ),
    }
    return {"intervals_95": intervals, "delta_intervals_95": deltas}, uncertainty


def evaluate_frozen_winner(
    *,
    benchmark_manifest_path: Path,
    review_cache_dir: Path,
    repository_root: Path,
    work_dir: Path,
    public_summary_path: Path,
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate the development-frozen winner once on calibration."""

    evaluation_path = work_dir / "calibration_evaluation.json"
    existing = [
        path.as_posix()
        for path in (evaluation_path, public_summary_path)
        if path.exists()
    ]
    if existing and not force:
        raise MetaSynScreeningStudyError(f"evaluation_outputs_exist:{existing}")
    fit = validate_fit(work_dir=work_dir)
    prepare = validate_prepare(work_dir=work_dir)
    evaluation_config = prepare["calibration_evaluation_config"]
    if bootstrap_seed != evaluation_config["bootstrap_seed"]:
        raise MetaSynScreeningStudyError("bootstrap_seed_differs_from_prepare_freeze")
    if bootstrap_replicates < evaluation_config["minimum_bootstrap_replicates"]:
        raise MetaSynScreeningStudyError("bootstrap_replicates_below_prepare_minimum")
    rankings = _load_winner_rankings(work_dir=work_dir, fit=fit)
    selected = rankings["selected"]["calibration"]
    baseline = rankings["baseline"]["calibration"]
    labels, label_access = _load_split_labels_from_train(
        benchmark_manifest_path=benchmark_manifest_path,
        review_cache_dir=review_cache_dir,
        split="calibration",
    )
    expected_ids = {label.review_id for label in labels}
    if set(selected) != expected_ids or set(baseline) != expected_ids:
        raise MetaSynScreeningStudyError("calibration_ranking_label_universe_mismatch")
    point_results = {
        method: {
            str(depth): _point_metrics(
                ranking=ranking,
                baseline=baseline,
                labels=labels,
                depth=depth,
            )
            for depth in EVALUATION_DEPTHS
        }
        for method, ranking in (("selected", selected), ("rrf_baseline", baseline))
    }
    paired_deltas: dict[str, dict[str, float | None]] = {}
    delta_metrics = (
        "question_macro_absolute_recall",
        "micro_absolute_recall",
        "question_macro_conditional_survival",
        "micro_conditional_survival",
        "questions_with_zero_retained_rate",
        "full_inclusion_rate",
    )
    for depth in EVALUATION_DEPTHS:
        key = str(depth)
        paired_deltas[key] = {}
        for metric in delta_metrics:
            left = point_results["selected"][key][metric]
            right = point_results["rrf_baseline"][key][metric]
            paired_deltas[key][metric] = (
                None if left is None or right is None else float(left - right)
            )
    bootstrap, uncertainty = _cluster_bootstrap(
        selected=selected,
        baseline=baseline,
        labels=labels,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    label_identity = [
        {
            "review_id": label.review_id,
            "component_id": label.component_id,
            "gold_corpus_ids": list(label.gold_corpus_ids),
        }
        for label in labels
    ]
    payload = {
        "metasyn_screening_calibration_evaluation_version": STUDY_VERSION,
        "stage": "single_frozen_winner_calibration_evaluation",
        "fit_payload_sha256": fit["fit_payload_sha256"],
        "fit_receipt_sha256": sha256_file(work_dir / "fit_receipt.json"),
        "winner_rankings_sha256": fit["winner_rankings_sha256"],
        "selected_candidate": fit["selected_candidate"],
        "calibration_label_identity_sha256": hash_canonical(label_identity),
        "calibration_label_access": label_access,
        "evaluation_depths": list(EVALUATION_DEPTHS),
        "point_results": point_results,
        "selected_minus_rrf_paired_deltas": paired_deltas,
        "cluster_bootstrap": bootstrap,
        "uncertainty": uncertainty,
        "denominator_contract": {
            "absolute": (
                "all_released_matched_identifiers_per_question_including_identifiers_"
                "absent_from_the_RRF_top_200"
            ),
            "conditional": (
                "released_matched_identifiers_present_anywhere_in_the_frozen_RRF_top_200"
            ),
            "zero_retrieval_questions_in_absolute_macro": "included_with_zero_recall",
            "zero_retrieval_questions_in_conditional_macro": (
                "reported_separately_and_excluded_only_because_conditional_denominator_is_zero"
            ),
        },
        "access_boundary": {
            "winner_frozen_before_calibration_label_materialization": True,
            "calibration_selected_candidate_evaluations": 1,
            "calibration_alternative_model_selection_candidates_scored": False,
            "prespecified_rrf_baseline_comparison_computed": True,
            "calibration_feedback_allowed": False,
            "official_test_inputs_opened": False,
            "official_test_labels_opened": False,
            "official_test_evaluated": False,
            "labels_historically_opened_before_protocol": True,
            "pristine_holdout_eligible": False,
        },
        "source_code_sha256s": _source_code_hashes(),
    }
    evaluation = _attach_hash(payload, field="calibration_evaluation_payload_sha256")
    atomic_write_json(evaluation_path, evaluation, force=force)
    summary = build_public_summary(
        repository_root=repository_root,
        work_dir=work_dir,
        output_path=public_summary_path,
        force=force,
    )
    return summary


def validate_evaluation(*, work_dir: Path) -> dict[str, Any]:
    fit = validate_fit(work_dir=work_dir)
    prepare = validate_prepare(work_dir=work_dir)
    evaluation = _load_json(work_dir / "calibration_evaluation.json")
    _verify_hash(evaluation, field="calibration_evaluation_payload_sha256")
    if evaluation.get("metasyn_screening_calibration_evaluation_version") != STUDY_VERSION:
        raise MetaSynScreeningStudyError("evaluation_version_unsupported")
    if evaluation.get("fit_payload_sha256") != fit.get("fit_payload_sha256"):
        raise MetaSynScreeningStudyError("evaluation_fit_mismatch")
    if evaluation.get("fit_receipt_sha256") != sha256_file(work_dir / "fit_receipt.json"):
        raise MetaSynScreeningStudyError("evaluation_fit_receipt_hash_mismatch")
    if evaluation.get("winner_rankings_sha256") != fit.get("winner_rankings_sha256"):
        raise MetaSynScreeningStudyError("evaluation_winner_rankings_mismatch")
    if evaluation.get("selected_candidate") != fit.get("selected_candidate"):
        raise MetaSynScreeningStudyError("evaluation_selected_candidate_mismatch")
    if evaluation.get("source_code_sha256s") != _source_code_hashes():
        raise MetaSynScreeningStudyError("evaluation_source_code_drift")
    access = evaluation.get("access_boundary", {})
    if access.get("calibration_alternative_model_selection_candidates_scored") is not False:
        raise MetaSynScreeningStudyError("evaluation_scored_selection_candidates")
    if access.get("calibration_feedback_allowed") is not False:
        raise MetaSynScreeningStudyError("evaluation_allows_calibration_feedback")
    if any(
        access.get(field) is not False
        for field in (
            "official_test_inputs_opened",
            "official_test_labels_opened",
            "official_test_evaluated",
        )
    ):
        raise MetaSynScreeningStudyError("evaluation_official_test_accessed")
    if evaluation.get("evaluation_depths") != list(EVALUATION_DEPTHS):
        raise MetaSynScreeningStudyError("evaluation_depth_contract_changed")
    uncertainty = evaluation.get("uncertainty", {})
    evaluation_config = prepare["calibration_evaluation_config"]
    if uncertainty.get("method") != evaluation_config["bootstrap"]:
        raise MetaSynScreeningStudyError("evaluation_bootstrap_method_mismatch")
    if uncertainty.get("seed") != evaluation_config["bootstrap_seed"]:
        raise MetaSynScreeningStudyError("evaluation_bootstrap_seed_mismatch")
    if (
        not isinstance(uncertainty.get("replicates"), int)
        or uncertainty["replicates"] < evaluation_config["minimum_bootstrap_replicates"]
    ):
        raise MetaSynScreeningStudyError("evaluation_bootstrap_replicates_invalid")
    if access.get("winner_frozen_before_calibration_label_materialization") is not True:
        raise MetaSynScreeningStudyError("evaluation_winner_not_frozen_before_calibration")
    if access.get("prespecified_rrf_baseline_comparison_computed") is not True:
        raise MetaSynScreeningStudyError("evaluation_prespecified_baseline_missing")
    return evaluation


def _validate_public_redaction(value: Any, *, repository_root: Path) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS:
                raise MetaSynScreeningStudyError(f"public_summary_forbidden_key:{key}")
            _validate_public_redaction(child, repository_root=repository_root)
        return
    if isinstance(value, list):
        for child in value:
            _validate_public_redaction(child, repository_root=repository_root)
        return
    if isinstance(value, str):
        if repository_root.as_posix() in value or value.startswith("/Users/"):
            raise MetaSynScreeningStudyError("public_summary_contains_absolute_path")
        if re.search(r"metasyn-review-[0-9]+", value):
            raise MetaSynScreeningStudyError("public_summary_contains_question_identifier")


def _public_summary_payload(
    *,
    repository_root: Path,
    prepare: Mapping[str, Any],
    fit: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "metasyn_screening_public_summary_version": STUDY_VERSION,
        "status": "complete_retrospective_nonpristine",
        "network_calls": 0,
        "provider_calls": 0,
        "task": "protocol_aware_reranking_within_frozen_RRF_top_200",
        "data_scope": {
            "corpus_source_revision": prepare["corpus_source_revision"],
            "corpus_articles": prepare["corpus_rows"],
            "development_questions": prepare["questions_by_split"]["development"],
            "calibration_questions": prepare["questions_by_split"]["calibration"],
            "retrieval_candidate_depth": RETRIEVAL_DEPTH,
            "unique_candidate_articles_across_splits": prepare[
                "unique_candidate_documents"
            ],
        },
        "lineage": {
            "retrieval_freeze_payload_sha256": prepare[
                "retrieval_freeze_payload_sha256"
            ],
            "prepare_payload_sha256": prepare["prepare_payload_sha256"],
            "fit_payload_sha256": fit["fit_payload_sha256"],
            "calibration_evaluation_payload_sha256": evaluation[
                "calibration_evaluation_payload_sha256"
            ],
            "benchmark_manifest_sha256": prepare["benchmark_manifest_sha256"],
            "corpus_manifest_sha256": prepare["corpus_manifest_sha256"],
            "feature_config_sha256": prepare["feature_config_sha256"],
            "development_selection_config_sha256": prepare[
                "development_selection_config_sha256"
            ],
            "calibration_evaluation_config_sha256": prepare[
                "calibration_evaluation_config_sha256"
            ],
            "winner_rankings_sha256": fit["winner_rankings_sha256"],
            "source_train_parquet_sha256": evaluation["calibration_label_access"][
                "source_train_parquet_sha256"
            ],
            "source_code_sha256s": evaluation["source_code_sha256s"],
        },
        "protocol": {
            "features_frozen_before_development_label_materialization": True,
            "candidate_family_frozen": list(CANDIDATE_IDS),
            "development_selection": fit["selection_rule"],
            "selected_candidate": fit["selected_candidate"],
            "winner_calibration_order_frozen_before_calibration_label_materialization": True,
            "calibration_selected_candidate_evaluations": 1,
            "calibration_selection_feedback_allowed": False,
            "official_test_inputs_opened_by_this_study": False,
            "official_test_labels_opened_by_this_study": False,
            "official_test_evaluated": False,
        },
        "development_component_disjoint_cross_validation": {
            "candidates": fit["development_oof_results"],
            "folds": fit["selection_rule"]["folds_executed"],
            "components": len(
                {
                    component
                    for fold in fit["cv_folds"]
                    for component in fold["validation_component_ids"]
                }
            ),
            "all_fold_component_overlaps": 0,
        },
        "calibration": {
            "evaluation_depths": evaluation["evaluation_depths"],
            "point_results": evaluation["point_results"],
            "selected_minus_rrf_paired_deltas": evaluation[
                "selected_minus_rrf_paired_deltas"
            ],
            "cluster_bootstrap": evaluation["cluster_bootstrap"],
            "uncertainty": evaluation["uncertainty"],
            "denominator_contract": evaluation["denominator_contract"],
        },
        "interpretation_limits": {
            "development_and_calibration_labels_historically_opened": True,
            "pristine_holdout_eligible": False,
            "split_access_is_logical_not_storage_level": (
                "development_and_calibration_rows_share_the_official_train_Parquet;_"
                "Arrow_row_predicates_materialize_only_the_named_split"
            ),
            "official_test_never_opened_or_scored_by_this_study": True,
            "official_test_labels_historically_opened_elsewhere_in_repository": True,
            "reranking_cannot_recover_articles_absent_from_RRF_top_200": True,
            "absolute_recall_denominator": (
                "all_released_MetaSyn_matched_identifiers_per_calibration_question"
            ),
            "conditional_survival_denominator": (
                "released_matched_identifiers_already_present_in_RRF_top_200"
            ),
            "nonincluded_candidates_treated_as_implicit_negatives": True,
            "implicit_negatives_may_include_unannotated_eligible_articles": True,
            "released_matching_is_not_exhaustive_eligibility_gold": True,
            "result_does_not_establish_protocol_screening_accuracy_or_scientific_truth": True,
        },
        "public_redaction": {
            "contains_question_or_component_identifiers": False,
            "contains_article_identifiers": False,
            "contains_titles_abstracts_or_protocol_text": False,
            "contains_labels_or_per_question_results": False,
            "contains_absolute_paths": False,
        },
    }
    _validate_public_redaction(payload, repository_root=repository_root)
    return payload


def build_public_summary(
    *, repository_root: Path, work_dir: Path, output_path: Path, force: bool = False
) -> dict[str, Any]:
    """Write a self-hashed aggregate containing no text or row identifiers."""

    prepare = validate_prepare(work_dir=work_dir)
    fit = validate_fit(work_dir=work_dir)
    evaluation = validate_evaluation(work_dir=work_dir)
    payload = _public_summary_payload(
        repository_root=repository_root,
        prepare=prepare,
        fit=fit,
        evaluation=evaluation,
    )
    summary = _attach_hash(payload, field="public_summary_payload_sha256")
    _validate_public_redaction(summary, repository_root=repository_root)
    atomic_write_json(output_path, summary, force=force)
    return summary


def validate_public_summary(
    *, repository_root: Path, work_dir: Path, public_summary_path: Path
) -> dict[str, Any]:
    prepare = validate_prepare(work_dir=work_dir)
    fit = validate_fit(work_dir=work_dir)
    evaluation = validate_evaluation(work_dir=work_dir)
    summary = _load_json(public_summary_path)
    _verify_hash(summary, field="public_summary_payload_sha256")
    if summary.get("metasyn_screening_public_summary_version") != STUDY_VERSION:
        raise MetaSynScreeningStudyError("public_summary_version_unsupported")
    if summary.get("lineage", {}).get(
        "calibration_evaluation_payload_sha256"
    ) != evaluation.get("calibration_evaluation_payload_sha256"):
        raise MetaSynScreeningStudyError("public_summary_evaluation_mismatch")
    expected = _public_summary_payload(
        repository_root=repository_root,
        prepare=prepare,
        fit=fit,
        evaluation=evaluation,
    )
    observed = {
        key: value
        for key, value in summary.items()
        if key != "public_summary_payload_sha256"
    }
    if observed != expected:
        raise MetaSynScreeningStudyError("public_summary_content_mismatch")
    _validate_public_redaction(summary, repository_root=repository_root)
    return summary


def run_screening_study(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    review_cache_dir: Path,
    retrieval_work_dir: Path,
    work_dir: Path,
    public_summary_path: Path,
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run all three explicit stages in order."""

    prepare_label_blind_features(
        benchmark_manifest_path=benchmark_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        repository_root=repository_root,
        retrieval_work_dir=retrieval_work_dir,
        work_dir=work_dir,
        force=force,
    )
    fit_and_freeze_winner(
        benchmark_manifest_path=benchmark_manifest_path,
        review_cache_dir=review_cache_dir,
        work_dir=work_dir,
        force=force,
    )
    return evaluate_frozen_winner(
        benchmark_manifest_path=benchmark_manifest_path,
        review_cache_dir=review_cache_dir,
        repository_root=repository_root,
        work_dir=work_dir,
        public_summary_path=public_summary_path,
        force=force,
        bootstrap_replicates=bootstrap_replicates,
    )
