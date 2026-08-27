"""Deterministic, label-blind lexical retrieval for the local MetaSyn corpus.

The implementation is deliberately streaming: it makes two bounded passes over the
Parquet shards and retains only query vocabulary statistics and a top-k heap per
review.  Gold matched-paper identifiers are never opened by the retrieval function.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import sklearn
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer

from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_bytes,
    sha256_file,
)
from literature_multiverse.metasyn_benchmark import (
    BenchmarkSplit,
    MetaSynPrediction,
    MetaSynQuestionInput,
    load_metasyn_inputs,
    load_metasyn_manifest,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"[a-z0-9]+")
_CORPUS_COLUMNS = ("ID", "title", "abstract")
_QUERY_FIELDS = (
    "research_question",
    "population",
    "intervention",
    "exposure",
    "comparison",
    "outcome",
)

# Frozen rather than imported from a changing NLP package.  Domain words are retained;
# only ordinary English function words and review boilerplate are removed.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    yours yourself yourselves study studies review reviews systematic meta analysis
    investigate investigates investigated examining examines examined evaluate evaluates
    evaluated assess assesses assessed determine determines determined aim aims aimed
    relationship association associations effect effects impact impacts among using use
    used compared comparing comparison group groups patients participants individuals
    """.split()  # noqa: SIM905 - readable frozen vocabulary is preferable to 180 literals.
)


class MetaSynCorpusError(ValueError):
    """The pinned corpus or lexical baseline contract is invalid."""


class CorpusShard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    rows: int = Field(ge=1)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.name != value:
            raise ValueError("corpus_shard_path_must_be_a_basename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("invalid_corpus_shard_sha256")
        return value


class LicenseNotice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    status: Literal["local_evaluation_only_third_party_terms_apply"]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("license_path_must_be_repository_relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("invalid_license_notice_sha256")
        return value


class MetaSynCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_manifest_version: Literal["1"] = "1"
    dataset: Literal["MetaSyn corpus"]
    source_repository: str
    source_revision: str
    local_root: str
    license_notice: LicenseNotice
    shards: list[CorpusShard] = Field(min_length=1)
    total_rows: int = Field(ge=1)

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("source_revision_must_be_a_git_commit")
        return value

    @field_validator("local_root")
    @classmethod
    def validate_local_root(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("local_root_must_be_repository_relative")
        return value

    @model_validator(mode="after")
    def validate_shards(self) -> MetaSynCorpusManifest:
        names = [shard.path for shard in self.shards]
        if names != sorted(set(names)):
            raise ValueError("corpus_shards_must_be_sorted_unique")
        if sum(shard.rows for shard in self.shards) != self.total_rows:
            raise ValueError("corpus_shard_rows_do_not_sum_to_total")
        return self


def load_corpus_manifest(path: Path) -> MetaSynCorpusManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return MetaSynCorpusManifest.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MetaSynCorpusError(f"corpus_manifest_invalid:{path}") from exc


def verify_corpus_manifest(
    manifest_path: Path, *, repository_root: Path
) -> tuple[MetaSynCorpusManifest, list[Path]]:
    """Verify the revision-pinned shard set, schema, row counts, and license notice."""

    manifest = load_corpus_manifest(manifest_path)
    corpus_root = repository_root / manifest.local_root
    license_path = repository_root / manifest.license_notice.path
    if not license_path.is_file():
        raise MetaSynCorpusError(f"license_notice_missing:{license_path}")
    observed_license_hash = sha256_file(license_path)
    if observed_license_hash != manifest.license_notice.sha256:
        raise MetaSynCorpusError(
            "license_notice_hash_mismatch:"
            f"expected={manifest.license_notice.sha256}:observed={observed_license_hash}"
        )

    paths: list[Path] = []
    for shard in manifest.shards:
        path = corpus_root / shard.path
        if not path.is_file():
            raise MetaSynCorpusError(
                "corpus_shard_missing:"
                f"{path}:obtain revision {manifest.source_revision} of "
                f"{manifest.source_repository} without redistributing the local cache"
            )
        observed_hash = sha256_file(path)
        if observed_hash != shard.sha256:
            raise MetaSynCorpusError(
                f"corpus_shard_hash_mismatch:{shard.path}:"
                f"expected={shard.sha256}:observed={observed_hash}"
            )
        try:
            parquet = pq.ParquetFile(path)
        except Exception as exc:
            raise MetaSynCorpusError(f"corpus_shard_unreadable:{shard.path}") from exc
        if parquet.metadata.num_rows != shard.rows:
            raise MetaSynCorpusError(
                f"corpus_shard_row_count_mismatch:{shard.path}:"
                f"expected={shard.rows}:observed={parquet.metadata.num_rows}"
            )
        missing = sorted(set(_CORPUS_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise MetaSynCorpusError(
                f"corpus_shard_columns_missing:{shard.path}:{','.join(missing)}"
            )
        paths.append(path)
    return manifest, paths


def inspect_corpus_coverage(
    shard_paths: Sequence[Path], *, required_corpus_ids: set[int | str]
) -> dict[str, Any]:
    """Return metadata-only payload and identifier coverage for verified shards."""

    try:
        normalized_required_ids = {int(item) for item in required_corpus_ids}
    except (TypeError, ValueError) as exc:
        raise MetaSynCorpusError("required_corpus_id_not_integer_like") from exc
    seen: set[int] = set()
    rows_with_title = 0
    rows_with_abstract = 0
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_path)
        try:
            batches = parquet.iter_batches(batch_size=8192, columns=["ID", "title", "abstract"])
            for batch in batches:
                for row in batch.to_pylist():
                    document_id = int(row["ID"])
                    if document_id in seen:
                        raise MetaSynCorpusError(f"corpus_document_id_duplicate:{document_id}")
                    seen.add(document_id)
                    rows_with_title += int(bool(str(row["title"] or "").strip()))
                    rows_with_abstract += int(bool(str(row["abstract"] or "").strip()))
        except MetaSynCorpusError:
            raise
        except Exception as exc:
            raise MetaSynCorpusError(f"corpus_shard_inventory_failed:{shard_path.name}") from exc
    missing = sorted(normalized_required_ids - seen)
    return {
        "rows": len(seen),
        "rows_with_title": rows_with_title,
        "rows_with_abstract": rows_with_abstract,
        "required_gold_ids": len(normalized_required_ids),
        "required_gold_ids_present": len(normalized_required_ids) - len(missing),
        "required_gold_ids_missing": len(missing),
        "required_gold_ids_complete": not missing,
    }


def _terms(value: str | None) -> list[str]:
    tokens = [
        token
        for token in _TOKEN.findall((value or "").casefold())
        if len(token) > 1 and not token.isdecimal() and token not in _STOPWORDS
    ]
    return [*tokens, *(f"bi:{left}_{right}" for left, right in pairwise(tokens))]


def _query_counter(question: MetaSynQuestionInput) -> Counter[str]:
    counter: Counter[str] = Counter()
    counter.update(_terms(question.research_question))
    for field in _QUERY_FIELDS[1:]:
        # PI/ECO fields are more compact than the prose research question and receive
        # a fixed, label-independent boost.
        terms = _terms(getattr(question, field))
        counter.update(terms)
        counter.update(terms)
    return counter


def _document_counter(title: str | None, abstract: str | None) -> Counter[str]:
    counter = Counter(_terms(abstract))
    title_terms = _terms(title)
    counter.update(title_terms)
    counter.update(title_terms)
    return counter


def _iter_documents(shard_paths: Sequence[Path]) -> Iterable[tuple[int, Counter[str]]]:
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_path)
        try:
            batches = parquet.iter_batches(batch_size=2048, columns=list(_CORPUS_COLUMNS))
            for batch in batches:
                for row in batch.to_pylist():
                    yield int(row["ID"]), _document_counter(row["title"], row["abstract"])
        except Exception as exc:
            raise MetaSynCorpusError(f"corpus_shard_scan_failed:{shard_path.name}") from exc


def _load_tfidf_documents(shard_paths: Sequence[Path]) -> tuple[np.ndarray, list[str]]:
    corpus_ids: list[int] = []
    documents: list[str] = []
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_path)
        try:
            batches = parquet.iter_batches(batch_size=8192, columns=["ID", "title", "abstract"])
            for batch in batches:
                for row in batch.to_pylist():
                    corpus_ids.append(int(row["ID"]))
                    documents.append(f"{row['title'] or ''!s} {row['abstract'] or ''!s}")
        except Exception as exc:
            raise MetaSynCorpusError(f"corpus_shard_tfidf_scan_failed:{shard_path.name}") from exc
    if len(corpus_ids) != len(set(corpus_ids)):
        raise MetaSynCorpusError("corpus_document_ids_not_unique")
    return np.asarray(corpus_ids, dtype=np.int64), documents


def _question_text(question: MetaSynQuestionInput) -> str:
    # The exact field order and one-space join are part of the frozen baseline.
    return " ".join(str(getattr(question, field) or "") for field in _QUERY_FIELDS)


def _stable_top_k(
    scores: np.ndarray,
    *,
    corpus_ids: np.ndarray,
    excluded_ids: set[int],
    top_k: int,
) -> list[int]:
    eligible = np.ones(len(corpus_ids), dtype=bool)
    if excluded_ids:
        eligible &= ~np.isin(corpus_ids, np.fromiter(excluded_ids, dtype=np.int64))
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < top_k:
        raise MetaSynCorpusError("fewer_eligible_documents_than_retrieval_depth")
    eligible_scores = scores[eligible_indices]
    # Find the exact score boundary in linear time, then resolve all boundary ties by
    # ascending corpus ID.  This is stable even when many documents have score zero.
    boundary_offset = len(eligible_indices) - top_k
    threshold = np.partition(eligible_scores, boundary_offset)[boundary_offset]
    above = eligible_indices[eligible_scores > threshold]
    tied = eligible_indices[eligible_scores == threshold]
    tied = tied[np.argsort(corpus_ids[tied], kind="stable")]
    selected = np.concatenate((above, tied[: top_k - len(above)]))
    order = np.lexsort((corpus_ids[selected], -scores[selected]))
    return [int(value) for value in corpus_ids[selected[order]]]


def freeze_tfidf_retrieval_baseline(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    review_cache_dir: Path,
    output_dir: Path,
    top_k: int = 200,
    force: bool = False,
) -> tuple[Path, Path]:
    """Freeze the corpus-only-fit TF-IDF baseline on development + calibration.

    Test is intentionally not an option: its labels and aggregate distribution were
    already opened, so evaluating this retrospectively chosen baseline on it would not
    create a pristine final holdout.
    """

    predictions_path = output_dir / "predictions.jsonl"
    receipt_path = output_dir / "freeze_receipt.json"
    existing = [path.as_posix() for path in (predictions_path, receipt_path) if path.exists()]
    if existing and not force:
        raise MetaSynCorpusError(f"retrieval_outputs_exist:{existing}")
    if top_k < 1:
        raise ValueError("retrieval_top_k_must_be_positive")

    development = load_metasyn_inputs(benchmark_manifest_path, split="development")
    calibration = load_metasyn_inputs(benchmark_manifest_path, split="calibration")
    questions = [*development, *calibration]
    benchmark = load_metasyn_manifest(benchmark_manifest_path)
    corpus, shard_paths = verify_corpus_manifest(
        corpus_manifest_path, repository_root=repository_root
    )
    exclusions, review_source_hashes = load_source_review_exclusions(
        benchmark_manifest_path=benchmark_manifest_path,
        split="development",
        review_cache_dir=review_cache_dir,
        expected_review_ids={question.review_id for question in questions},
    )
    corpus_ids, documents = _load_tfidf_documents(shard_paths)
    query_texts = [_question_text(question) for question in questions]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    # Fit only on corpus documents.  The exploratory prototype fit on documents plus
    # all unlabeled queries; that transductive variant is deliberately not reproduced
    # as the primary frozen baseline.
    document_matrix = vectorizer.fit_transform(documents)
    query_matrix = vectorizer.transform(query_texts)
    rankings: dict[int, list[int]] = {}
    for index, question in enumerate(questions):
        scores = (query_matrix[index] @ document_matrix.T).toarray().ravel()
        rankings[question.review_id] = _stable_top_k(
            scores,
            corpus_ids=corpus_ids,
            excluded_ids=exclusions[question.review_id],
            top_k=top_k,
        )

    predictions = [
        MetaSynPrediction(
            review_id=question.review_id,
            # MetaSynPrediction is a set-valued retrieval contract; rank is hash-bound
            # separately in the receipt because Recall@k does not require rank order.
            retrieved_corpus_ids=sorted(rankings[question.review_id]),
        ).model_dump(mode="json", exclude_none=True)
        for question in sorted(questions, key=lambda item: item.review_id)
    ]
    config = {
        "algorithm": "sklearn_tfidf_title_abstract_v1",
        "document_join": "title + one ASCII space + abstract",
        "dtype": "float32",
        "fit_population": "corpus_documents_only",
        "lowercase": True,
        "max_features": 200_000,
        "min_df": 2,
        "ngram_range": [1, 2],
        "norm": "l2",
        "query_field_order": list(_QUERY_FIELDS),
        "query_join": "one ASCII space with null mapped to empty string",
        "stable_tie_break": "descending_score_then_ascending_corpus_id",
        "stop_words": "sklearn_english",
        "sublinear_tf": True,
        "top_k": top_k,
    }
    atomic_write_jsonl(predictions_path, predictions, force=force)
    vocabulary = {
        term: int(index) for term, index in sorted(vectorizer.vocabulary_.items())
    }
    receipt = {
        "metasyn_retrieval_freeze_version": "1",
        "scientific_role": "retrospective_local_baseline_not_pristine_holdout",
        "splits": ["development", "calibration"],
        "rows_by_split": {
            "development": len(development),
            "calibration": len(calibration),
        },
        "labels_read_by_freeze_function": False,
        "labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "test_split_evaluated": False,
        "transductive_query_fit": False,
        "model_fields_used": list(_QUERY_FIELDS),
        "evaluator_fields_used": ["source_review_corpus_ids_for_exclusion_only"],
        "gold_matched_corpus_ids_used_for_ranking": False,
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_source_revision": corpus.source_revision,
        "corpus_shard_sha256s": {shard.path: shard.sha256 for shard in corpus.shards},
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "model_input_sha256s": {
            "development": benchmark.development.sha256,
            "calibration": benchmark.calibration.sha256,
        },
        "review_source_sha256s": review_source_hashes,
        "config": config,
        "config_sha256": hash_canonical(config),
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "corpus_rows": len(corpus_ids),
        "vocabulary_terms": len(vocabulary),
        "vocabulary_sha256": hash_canonical(vocabulary),
        "idf_float64_bytes_sha256": sha256_bytes(
            np.asarray(vectorizer.idf_, dtype=np.float64).tobytes(order="C")
        ),
        "source_review_exclusions": sum(len(values) for values in exclusions.values()),
        "ranking_sha256": hash_canonical(rankings),
        "predictions_sha256": sha256_file(predictions_path),
    }
    atomic_write_json(receipt_path, receipt, force=force)
    return predictions_path, receipt_path


def _integer_values(value: Any, *, review_id: int) -> set[int]:
    if value is None:
        return set()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        raise MetaSynCorpusError(f"source_review_ids_invalid:review_id={review_id}")
    try:
        result = {int(item) for item in value}
    except (TypeError, ValueError) as exc:
        raise MetaSynCorpusError(f"source_review_ids_invalid:review_id={review_id}") from exc
    if any(item < 0 for item in result):
        raise MetaSynCorpusError(f"source_review_ids_invalid:review_id={review_id}")
    return result


def load_source_review_exclusions(
    *,
    benchmark_manifest_path: Path,
    split: BenchmarkSplit,
    review_cache_dir: Path,
    expected_review_ids: set[int],
) -> tuple[dict[int, set[int]], dict[str, str]]:
    """Read only review identity and source-review exclusions, never outcome labels."""

    benchmark = load_metasyn_manifest(benchmark_manifest_path)
    source = benchmark.source_test if split == "test" else benchmark.source_train
    source_path = review_cache_dir / source.filename
    if not source_path.is_file():
        raise MetaSynCorpusError(f"review_source_missing:{source_path}")
    observed_hash = sha256_file(source_path)
    if observed_hash != source.sha256:
        raise MetaSynCorpusError(
            f"review_source_hash_mismatch:{source.filename}:"
            f"expected={source.sha256}:observed={observed_hash}"
        )
    try:
        table = pq.read_table(source_path, columns=["ID", "source_review_corpus_ids"])
    except Exception as exc:
        raise MetaSynCorpusError(
            f"review_source_exclusion_columns_unreadable:{source_path}"
        ) from exc
    exclusions: dict[int, set[int]] = {}
    for row in table.to_pylist():
        review_id = int(row["ID"])
        if review_id in expected_review_ids:
            if review_id in exclusions:
                raise MetaSynCorpusError(f"duplicate_review_source_id:{review_id}")
            exclusions[review_id] = _integer_values(
                row["source_review_corpus_ids"], review_id=review_id
            )
    missing = sorted(expected_review_ids - set(exclusions))
    if missing:
        raise MetaSynCorpusError(f"review_source_ids_missing:{missing}")
    return exclusions, {source.filename: observed_hash}


def _offer(heap: list[tuple[float, int]], *, score: float, document_id: int, top_k: int) -> None:
    # Higher score is better; at equal score the lower corpus ID is better.  Negating
    # the ID makes heap[0] the worst retained tie.
    candidate = (score, -document_id)
    if len(heap) < top_k:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def bm25_rank(
    *,
    questions: Sequence[MetaSynQuestionInput],
    shard_paths: Sequence[Path],
    exclusions: Mapping[int, set[int]],
    top_k: int = 200,
    k1: float = 1.2,
    b: float = 0.75,
    k3: float = 8.0,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    """Rank corpus IDs using fixed BM25 over title and abstract only."""

    if top_k < 1 or not math.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("invalid_bm25_configuration")
    if not math.isfinite(k3) or k3 <= 0:
        raise ValueError("invalid_bm25_query_saturation")
    if not questions:
        raise MetaSynCorpusError("retrieval_questions_empty")

    query_counters = [_query_counter(question) for question in questions]
    if any(not counter for counter in query_counters):
        empty = [
            question.review_id
            for question, counter in zip(questions, query_counters, strict=True)
            if not counter
        ]
        raise MetaSynCorpusError(f"retrieval_query_has_no_terms:{empty}")
    vocabulary = set().union(*(set(counter) for counter in query_counters))

    document_frequency: Counter[str] = Counter()
    corpus_ids: list[int] = []
    total_document_length = 0
    for document_id, counter in _iter_documents(shard_paths):
        corpus_ids.append(document_id)
        total_document_length += sum(counter.values())
        document_frequency.update(set(counter) & vocabulary)
    if len(corpus_ids) != len(set(corpus_ids)):
        raise MetaSynCorpusError("corpus_document_ids_not_unique")
    document_count = len(corpus_ids)
    if document_count < top_k:
        raise MetaSynCorpusError("corpus_smaller_than_retrieval_depth")
    average_length = total_document_length / document_count
    if average_length <= 0:
        raise MetaSynCorpusError("corpus_has_no_lexical_content")

    term_queries: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for query_index, counter in enumerate(query_counters):
        for term, frequency in counter.items():
            query_weight = frequency * (k3 + 1.0) / (frequency + k3)
            term_queries[term].append((query_index, query_weight))
    idf = {
        term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }

    heaps: list[list[tuple[float, int]]] = [[] for _ in questions]
    nonzero_candidates = [0 for _ in questions]
    for document_id, counter in _iter_documents(shard_paths):
        document_length = sum(counter.values())
        normalization = k1 * (1.0 - b + b * document_length / average_length)
        scores: dict[int, float] = defaultdict(float)
        for term, term_frequency in counter.items():
            if term not in term_queries:
                continue
            document_weight = term_frequency * (k1 + 1.0) / (term_frequency + normalization)
            for query_index, query_weight in term_queries[term]:
                scores[query_index] += idf[term] * document_weight * query_weight
        for query_index, score in scores.items():
            review_id = questions[query_index].review_id
            if document_id in exclusions[review_id]:
                continue
            nonzero_candidates[query_index] += 1
            _offer(heaps[query_index], score=score, document_id=document_id, top_k=top_k)

    corpus_ids.sort()
    rankings: dict[int, list[int]] = {}
    for query_index, question in enumerate(questions):
        ranked = sorted(
            ((score, -negative_id) for score, negative_id in heaps[query_index]),
            key=lambda item: (-item[0], item[1]),
        )
        selected = [document_id for _, document_id in ranked]
        selected_set = set(selected)
        if len(selected) < top_k:
            for document_id in corpus_ids:
                if (
                    document_id not in selected_set
                    and document_id not in exclusions[question.review_id]
                ):
                    selected.append(document_id)
                    selected_set.add(document_id)
                    if len(selected) == top_k:
                        break
        rankings[question.review_id] = selected

    diagnostics = {
        "documents": document_count,
        "average_weighted_document_length": average_length,
        "query_vocabulary_terms": len(vocabulary),
        "document_fields": ["title", "abstract"],
        "title_term_boost": 2,
        "pico_term_boost": 2,
        "unigrams_and_adjacent_bigrams": True,
        "nonzero_candidate_counts": {
            str(question.review_id): nonzero_candidates[index]
            for index, question in enumerate(questions)
        },
    }
    return rankings, diagnostics


def freeze_bm25_retrieval_baseline(
    *,
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    repository_root: Path,
    review_cache_dir: Path,
    split: BenchmarkSplit,
    output_dir: Path,
    top_k: int = 200,
) -> tuple[Path, Path]:
    """Freeze predictions before evaluator labels are opened by a separate step."""

    predictions_path = output_dir / "predictions.jsonl"
    receipt_path = output_dir / "freeze_receipt.json"
    existing = [path.as_posix() for path in (predictions_path, receipt_path) if path.exists()]
    if existing:
        raise MetaSynCorpusError(f"retrieval_outputs_exist:{existing}")

    questions = load_metasyn_inputs(benchmark_manifest_path, split=split)
    benchmark = load_metasyn_manifest(benchmark_manifest_path)
    corpus, shard_paths = verify_corpus_manifest(
        corpus_manifest_path, repository_root=repository_root
    )
    exclusions, review_source_hashes = load_source_review_exclusions(
        benchmark_manifest_path=benchmark_manifest_path,
        split=split,
        review_cache_dir=review_cache_dir,
        expected_review_ids={question.review_id for question in questions},
    )
    rankings, diagnostics = bm25_rank(
        questions=questions,
        shard_paths=shard_paths,
        exclusions=exclusions,
        top_k=top_k,
    )
    predictions = [
        MetaSynPrediction(
            review_id=question.review_id,
            retrieved_corpus_ids=sorted(rankings[question.review_id]),
        ).model_dump(mode="json", exclude_none=True)
        for question in sorted(questions, key=lambda item: item.review_id)
    ]
    config = {
        "algorithm": "streaming_bm25_title_abstract_unigram_bigram_v1",
        "b": 0.75,
        "k1": 1.2,
        "k3": 8.0,
        "split": split,
        "top_k": top_k,
    }
    atomic_write_jsonl(predictions_path, predictions)
    receipt = {
        "metasyn_retrieval_freeze_version": "1",
        "scientific_role": (
            "primary_local_baseline"
            if split in {"development", "calibration"}
            else "diagnostic_only_labels_previously_opened"
        ),
        "split": split,
        "labels_read_by_freeze_function": False,
        "labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "model_fields_used": list(_QUERY_FIELDS),
        "evaluator_fields_used": ["source_review_corpus_ids_for_exclusion_only"],
        "gold_matched_corpus_ids_used_for_ranking": False,
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_source_revision": corpus.source_revision,
        "corpus_shard_sha256s": {shard.path: shard.sha256 for shard in corpus.shards},
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "model_input_sha256": getattr(benchmark, split).sha256,
        "review_source_sha256s": review_source_hashes,
        "config": config,
        "config_sha256": hash_canonical(config),
        "rows": len(predictions),
        "source_review_exclusions": sum(len(values) for values in exclusions.values()),
        "ranking_sha256": hash_canonical(rankings),
        "predictions_sha256": sha256_file(predictions_path),
        "diagnostics": diagnostics,
    }
    atomic_write_json(receipt_path, receipt)
    return predictions_path, receipt_path


__all__ = [
    "MetaSynCorpusError",
    "MetaSynCorpusManifest",
    "bm25_rank",
    "freeze_bm25_retrieval_baseline",
    "freeze_tfidf_retrieval_baseline",
    "inspect_corpus_coverage",
    "load_corpus_manifest",
    "load_source_review_exclusions",
    "verify_corpus_manifest",
]
