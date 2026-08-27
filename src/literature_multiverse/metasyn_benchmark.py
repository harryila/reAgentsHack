"""Leakage-safe, offline adapter and evaluator for the MetaSyn review benchmark.

MetaSyn's official train/test boundary is preserved: every official test review remains
in the held-out test split.  Because some included-paper identifiers occur on both
sides of that boundary, any connected official-train review is quarantined.  Remaining
official-train review components are assigned deterministically to development or
calibration without consulting outcome labels.

Only question/PICO fields are written to model-facing files.  Review conclusions,
effect directions, heterogeneity, statistics, abstracts, included-study titles, and
gold corpus identifiers remain in a separate hash-locked evaluator file.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.calibration import (
    LabelSource,
    RiskExample,
    validate_split_integrity,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel

BenchmarkSplit = Literal["development", "calibration", "test"]
EvaluationSplit = Literal["development", "calibration", "test", "all"]
OfficialSplit = Literal["train", "test"]
GoldDirection = Literal["Positive", "Negative", "Mixed", "NR"]
PredictedDirection = Literal["Positive", "Negative", "Mixed", "NR", "Abstain"]
FixedDirection = Literal["Positive", "Negative", "Mixed"]

DIRECTION_CLASSES: tuple[str, ...] = ("Positive", "Negative", "Mixed")
FORBIDDEN_MODEL_COLUMNS: tuple[str, ...] = (
    "Abstract",
    "Conclusion_Summary",
    "Effect_Direction",
    "Effect_Size_Category",
    "Effect_Size_Type",
    "Effect_Size_Value",
    "Heterogeneity_Level",
    "I2_Value",
    "Key_Insights",
    "P_Value",
    "Q_Value",
    "Statistical_Significance",
    "Tau2_Value",
    "conclusion_paragraph",
    "extracted_titles",
    "matched_corpus_ids",
    "raw_titles",
)
_REQUIRED_COLUMNS = frozenset(
    {
        "ID",
        "Title",
        "Population",
        "Intervention",
        "Exposure",
        "Comparison",
        "Outcome",
        "Effect_Direction",
        "Research_Question",
        "matched_corpus_ids",
        "matched_ref_count",
        "study_count",
        "source_review_corpus_ids",
    }
)
_SAFE_QUESTION_ID = re.compile(r"^metasyn-review-[0-9]{6}$")


class MetaSynBenchmarkError(ValueError):
    """The source data, split, prediction, or artifact contract is invalid."""


class MetaSynQuestionInput(ContractModel):
    """One model-facing review question with no evaluator-only fields."""

    benchmark_input_version: Literal["1"] = "1"
    question_id: str
    review_id: int = Field(ge=0)
    research_question: str
    population: str | None
    intervention: str | None
    exposure: str | None
    comparison: str | None
    outcome: str | None

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        if not _SAFE_QUESTION_ID.fullmatch(value):
            raise ValueError("invalid_metasyn_question_id")
        return value

    @field_validator("research_question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value:
            raise ValueError("research_question_missing")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> MetaSynQuestionInput:
        if self.question_id != f"metasyn-review-{self.review_id:06d}":
            raise ValueError("metasyn_question_id_review_id_mismatch")
        return self


class MetaSynEvaluatorLabel(ContractModel):
    """Private evaluator-side labels; this record must never be rendered to a model."""

    evaluator_label_version: Literal["1"] = "1"
    question_id: str
    review_id: int = Field(ge=0)
    official_split: OfficialSplit
    split: BenchmarkSplit
    component_id: str
    gold_direction: GoldDirection
    gold_matched_corpus_ids: list[int]
    matched_reference_count: int = Field(ge=1)
    review_reported_study_count: int | None = Field(default=None, ge=0)

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        if not _SAFE_QUESTION_ID.fullmatch(value):
            raise ValueError("invalid_metasyn_question_id")
        return value

    @field_validator("gold_matched_corpus_ids")
    @classmethod
    def validate_corpus_ids(cls, value: list[int]) -> list[int]:
        if not value or any(item < 0 for item in value):
            raise ValueError("gold_corpus_ids_must_be_nonempty_nonnegative")
        if value != sorted(set(value)):
            raise ValueError("gold_corpus_ids_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_counts_and_boundary(self) -> MetaSynEvaluatorLabel:
        if self.question_id != f"metasyn-review-{self.review_id:06d}":
            raise ValueError("metasyn_question_id_review_id_mismatch")
        if self.matched_reference_count != len(self.gold_matched_corpus_ids):
            raise ValueError("matched_reference_count_disagrees_with_corpus_ids")
        if self.official_split == "test" and self.split != "test":
            raise ValueError("official_test_review_moved_out_of_test")
        if self.official_split == "train" and self.split == "test":
            raise ValueError("official_train_review_moved_into_test")
        return self


class MetaSynPrediction(ContractModel):
    """One system output.  Missing fields are distinct from explicit NR/empty output."""

    prediction_version: Literal["1"] = "1"
    review_id: int = Field(ge=0)
    predicted_direction: PredictedDirection | None = None
    retrieved_corpus_ids: list[int] | None = None
    risk_features: dict[str, float] | None = None

    @field_validator("retrieved_corpus_ids")
    @classmethod
    def validate_retrieved_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(item < 0 for item in value):
            raise ValueError("retrieved_corpus_ids_must_be_nonnegative")
        if value != sorted(set(value)):
            raise ValueError("retrieved_corpus_ids_must_be_sorted_unique")
        return value

    @field_validator("risk_features")
    @classmethod
    def validate_risk_features(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("risk_features_must_be_nonempty_when_supplied")
        if any(not key or not math.isfinite(number) for key, number in value.items()):
            raise ValueError("risk_features_must_be_named_and_finite")
        return dict(sorted(value.items()))


class FixedDirectionBaselineReceipt(ContractModel):
    """Immutable receipt for a label-blind constant-direction prediction file."""

    fixed_direction_baseline_version: Literal["1"] = "1"
    baseline_kind: Literal["trivial_question_only_constant_direction_control"] = (
        "trivial_question_only_constant_direction_control"
    )
    split: BenchmarkSplit
    predicted_class: FixedDirection
    selection_note: str
    rows: int = Field(ge=1)
    manifest_sha256: str
    model_input_artifact_sha256: str
    model_inputs_canonical_sha256: str
    config: dict[str, str]
    config_sha256: str
    predictions_path: Literal["predictions.jsonl"] = "predictions.jsonl"
    predictions_file_sha256: str
    predictions_canonical_sha256: str
    model_fields_used: list[Literal["review_id_for_join_only"]] = Field(
        default_factory=lambda: ["review_id_for_join_only"]
    )
    labels_opened: Literal[False] = False
    retrieval_ids_emitted: Literal[False] = False
    risk_features_emitted: Literal[False] = False
    calibrated_system_evidence: Literal[False] = False

    @field_validator(
        "manifest_sha256",
        "model_input_artifact_sha256",
        "model_inputs_canonical_sha256",
        "config_sha256",
        "predictions_file_sha256",
        "predictions_canonical_sha256",
    )
    @classmethod
    def validate_receipt_hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_fixed_direction_receipt_sha256")
        return value

    @field_validator("selection_note")
    @classmethod
    def validate_selection_note(cls, value: str) -> str:
        if not value:
            raise ValueError("fixed_direction_selection_note_required")
        return value

    @model_validator(mode="after")
    def validate_config_identity(self) -> FixedDirectionBaselineReceipt:
        if self.config.get("split") != self.split:
            raise ValueError("fixed_direction_config_split_mismatch")
        if self.config.get("predicted_class") != self.predicted_class:
            raise ValueError("fixed_direction_config_class_mismatch")
        if hash_canonical(self.config) != self.config_sha256:
            raise ValueError("fixed_direction_config_hash_mismatch")
        return self


class BenchmarkFile(ContractModel):
    path: str
    sha256: str
    rows: int = Field(ge=1)
    review_ids: list[int]
    component_ids: list[str]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("benchmark_file_path_must_be_manifest_relative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_benchmark_file_sha256")
        return value

    @model_validator(mode="after")
    def validate_metadata(self) -> BenchmarkFile:
        if self.review_ids != sorted(set(self.review_ids)) or self.rows != len(self.review_ids):
            raise ValueError("benchmark_file_review_metadata_invalid")
        if self.component_ids != sorted(set(self.component_ids)):
            raise ValueError("benchmark_file_component_metadata_invalid")
        return self


class SourceParquet(ContractModel):
    filename: str
    sha256: str
    rows: int = Field(ge=1)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if Path(value).name != value or not value:
            raise ValueError("source_parquet_filename_must_be_basename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_source_parquet_sha256")
        return value


class QuarantinedReview(ContractModel):
    review_id: int = Field(ge=0)
    question_id: str
    component_id: str
    reason: Literal["component_linked_to_official_test"]
    linked_test_review_ids: list[int]

    @field_validator("linked_test_review_ids")
    @classmethod
    def validate_test_ids(cls, value: list[int]) -> list[int]:
        if not value or value != sorted(set(value)):
            raise ValueError("linked_test_review_ids_must_be_nonempty_sorted_unique")
        return value


class MetaSynBenchmarkManifest(ContractModel):
    """Immutable manifest for separately stored public inputs and private labels."""

    benchmark_manifest_version: Literal["1"] = "1"
    benchmark_name: Literal["MetaSyn review-level direction and retrieval"] = (
        "MetaSyn review-level direction and retrieval"
    )
    split_algorithm: Literal["official-test-component-quarantine-hash-v1"] = (
        "official-test-component-quarantine-hash-v1"
    )
    seed: int
    calibration_fraction: float = Field(gt=0, lt=1)
    source_train: SourceParquet
    source_test: SourceParquet
    development: BenchmarkFile
    calibration: BenchmarkFile
    test: BenchmarkFile
    evaluator_labels: BenchmarkFile
    quarantined_official_train: list[QuarantinedReview]
    forbidden_model_columns: list[str]
    benchmark_scope: Literal["review_level_not_study_level_meta_analysis"] = (
        "review_level_not_study_level_meta_analysis"
    )

    @model_validator(mode="after")
    def validate_split_contract(self) -> MetaSynBenchmarkManifest:
        if self.forbidden_model_columns != sorted(FORBIDDEN_MODEL_COLUMNS):
            raise ValueError("forbidden_model_columns_contract_changed")
        files = (self.development, self.calibration, self.test)
        for index, left in enumerate(files):
            for right in files[index + 1 :]:
                if set(left.review_ids) & set(right.review_ids):
                    raise ValueError("review_crosses_benchmark_split")
                if set(left.component_ids) & set(right.component_ids):
                    raise ValueError("component_crosses_benchmark_split")
        included_ids = sorted(item for file in files for item in file.review_ids)
        if included_ids != self.evaluator_labels.review_ids:
            raise ValueError("evaluator_label_ids_disagree_with_model_inputs")
        if self.test.rows != self.source_test.rows:
            raise ValueError("official_test_boundary_not_preserved")
        quarantine_ids = [row.review_id for row in self.quarantined_official_train]
        if quarantine_ids != sorted(set(quarantine_ids)):
            raise ValueError("quarantined_review_ids_must_be_sorted_unique")
        accounted_train = self.development.rows + self.calibration.rows + len(quarantine_ids)
        if accounted_train != self.source_train.rows:
            raise ValueError("official_train_accounting_mismatch")
        if self.evaluator_labels.rows != sum(file.rows for file in files):
            raise ValueError("evaluator_label_row_count_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class _RawReview:
    review_id: int
    official_split: OfficialSplit
    title_key: str
    question_key: str
    research_question: str
    population: str | None
    intervention: str | None
    exposure: str | None
    comparison: str | None
    outcome: str | None
    gold_direction: GoldDirection
    matched_corpus_ids: tuple[int, ...]
    matched_reference_count: int
    study_count: int | None
    source_review_corpus_ids: tuple[int, ...]


class _UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    rendered = str(value).strip()
    return rendered or None


def _required_text(value: Any, *, field: str, review_id: int) -> str:
    rendered = _optional_text(value)
    if rendered is None:
        raise MetaSynBenchmarkError(f"missing_{field}:review_id={review_id}")
    return rendered


def _normalized_link_text(value: Any) -> str:
    rendered = _optional_text(value)
    if rendered is None:
        return ""
    return " ".join(rendered.casefold().split())


def _integer_list(value: Any, *, field: str, review_id: int) -> tuple[int, ...]:
    if value is None:
        values: list[Any] = []
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        values = converted if isinstance(converted, list) else [converted]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise MetaSynBenchmarkError(f"invalid_{field}:review_id={review_id}")
    try:
        integers = sorted({int(item) for item in values})
    except (TypeError, ValueError) as exc:
        raise MetaSynBenchmarkError(f"invalid_{field}:review_id={review_id}") from exc
    if any(item < 0 for item in integers):
        raise MetaSynBenchmarkError(f"negative_{field}:review_id={review_id}")
    return tuple(integers)


def _study_count(value: Any, *, review_id: int) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    number = float(value)
    if not number.is_integer() or number < 0:
        raise MetaSynBenchmarkError(f"invalid_study_count:review_id={review_id}")
    return int(number)


def _load_source(path: Path, *, official_split: OfficialSplit) -> list[_RawReview]:
    if not path.is_file():
        raise MetaSynBenchmarkError(f"source_parquet_missing:{path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise MetaSynBenchmarkError(f"source_parquet_unreadable:{path}") from exc
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise MetaSynBenchmarkError(f"source_columns_missing:{missing}")
    if frame.empty:
        raise MetaSynBenchmarkError(f"source_parquet_empty:{path}")

    records: list[_RawReview] = []
    for row in frame.to_dict(orient="records"):
        try:
            review_id = int(row["ID"])
        except (TypeError, ValueError) as exc:
            raise MetaSynBenchmarkError("invalid_review_id") from exc
        matched = _integer_list(
            row["matched_corpus_ids"], field="matched_corpus_ids", review_id=review_id
        )
        if not matched:
            raise MetaSynBenchmarkError(f"empty_matched_corpus_ids:review_id={review_id}")
        try:
            matched_count = int(row["matched_ref_count"])
        except (TypeError, ValueError) as exc:
            raise MetaSynBenchmarkError(f"invalid_matched_ref_count:review_id={review_id}") from exc
        if matched_count != len(matched):
            raise MetaSynBenchmarkError(
                f"matched_ref_count_disagrees:review_id={review_id}:"
                f"count={matched_count}:unique_ids={len(matched)}"
            )
        direction = _required_text(
            row["Effect_Direction"], field="effect_direction", review_id=review_id
        )
        if direction not in {*DIRECTION_CLASSES, "NR"}:
            raise MetaSynBenchmarkError(
                f"invalid_effect_direction:review_id={review_id}:value={direction}"
            )
        records.append(
            _RawReview(
                review_id=review_id,
                official_split=official_split,
                title_key=_normalized_link_text(row["Title"]),
                question_key=_normalized_link_text(row["Research_Question"]),
                research_question=_required_text(
                    row["Research_Question"],
                    field="research_question",
                    review_id=review_id,
                ),
                population=_optional_text(row["Population"]),
                intervention=_optional_text(row["Intervention"]),
                exposure=_optional_text(row["Exposure"]),
                comparison=_optional_text(row["Comparison"]),
                outcome=_optional_text(row["Outcome"]),
                gold_direction=direction,  # type: ignore[arg-type]
                matched_corpus_ids=matched,
                matched_reference_count=matched_count,
                study_count=_study_count(row["study_count"], review_id=review_id),
                source_review_corpus_ids=_integer_list(
                    row["source_review_corpus_ids"],
                    field="source_review_corpus_ids",
                    review_id=review_id,
                ),
            )
        )
    ids = [record.review_id for record in records]
    if len(ids) != len(set(ids)):
        raise MetaSynBenchmarkError(f"duplicate_review_id_within_{official_split}")
    return sorted(records, key=lambda record: record.review_id)


def _components(records: Sequence[_RawReview]) -> list[list[_RawReview]]:
    union_find = _UnionFind(record.review_id for record in records)
    owner: dict[str, int] = {}
    for record in records:
        tokens = [f"paper:{item}" for item in record.matched_corpus_ids]
        tokens.extend(f"source-review:{item}" for item in record.source_review_corpus_ids)
        if record.title_key:
            tokens.append(f"title:{record.title_key}")
        if record.question_key:
            tokens.append(f"question:{record.question_key}")
        for token in tokens:
            prior = owner.setdefault(token, record.review_id)
            union_find.union(record.review_id, prior)
    grouped: dict[int, list[_RawReview]] = defaultdict(list)
    for record in records:
        grouped[union_find.find(record.review_id)].append(record)
    return sorted(
        (sorted(rows, key=lambda row: row.review_id) for rows in grouped.values()),
        key=lambda rows: rows[0].review_id,
    )


def _component_id(component: Sequence[_RawReview]) -> str:
    identity = [record.review_id for record in component]
    return f"metasyn-component-{hash_canonical(identity)[:20]}"


def _calibration_assignment(
    component: Sequence[_RawReview], *, seed: int, calibration_fraction: float
) -> bool:
    identity = {"seed": seed, "review_ids": [record.review_id for record in component]}
    score = int(hash_canonical(identity), 16) / float(2**256)
    return score < calibration_fraction


def _question_id(review_id: int) -> str:
    return f"metasyn-review-{review_id:06d}"


def _file_metadata(
    path: Path,
    *,
    base: Path,
    rows: int,
    review_ids: Iterable[int],
    component_ids: Iterable[str],
) -> BenchmarkFile:
    return BenchmarkFile(
        path=path.relative_to(base).as_posix(),
        sha256=sha256_file(path),
        rows=rows,
        review_ids=sorted(review_ids),
        component_ids=sorted(set(component_ids)),
    )


def prepare_metasyn_benchmark(
    *,
    train_parquet: Path,
    test_parquet: Path,
    output_dir: Path,
    seed: int = 20260826,
    calibration_fraction: float = 0.5,
    force: bool = False,
) -> MetaSynBenchmarkManifest:
    """Create hash-locked model inputs, private labels, and a split manifest."""

    if not 0 < calibration_fraction < 1:
        raise MetaSynBenchmarkError("calibration_fraction_must_be_between_zero_and_one")
    train = _load_source(train_parquet, official_split="train")
    test = _load_source(test_parquet, official_split="test")
    train_ids = {record.review_id for record in train}
    test_ids = {record.review_id for record in test}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise MetaSynBenchmarkError(f"review_id_crosses_official_split:{overlap}")

    components = _components([*train, *test])
    component_lookup = {
        record.review_id: (_component_id(component), component)
        for component in components
        for record in component
    }
    quarantined: list[QuarantinedReview] = []
    assignments: dict[int, BenchmarkSplit] = {}
    train_only_components: list[list[_RawReview]] = []
    for component in components:
        test_rows = [record for record in component if record.official_split == "test"]
        train_rows = [record for record in component if record.official_split == "train"]
        component_id = _component_id(component)
        if test_rows:
            for record in test_rows:
                assignments[record.review_id] = "test"
            linked_test_ids = sorted(record.review_id for record in test_rows)
            quarantined.extend(
                QuarantinedReview(
                    review_id=record.review_id,
                    question_id=_question_id(record.review_id),
                    component_id=component_id,
                    reason="component_linked_to_official_test",
                    linked_test_review_ids=linked_test_ids,
                )
                for record in train_rows
            )
        elif train_rows:
            train_only_components.append(train_rows)

    for component in train_only_components:
        split: BenchmarkSplit = (
            "calibration"
            if _calibration_assignment(
                component, seed=seed, calibration_fraction=calibration_fraction
            )
            else "development"
        )
        for record in component:
            assignments[record.review_id] = split

    assigned_train = [record for record in train if record.review_id in assignments]
    has_development = any(assignments[row.review_id] == "development" for row in assigned_train)
    if assigned_train and not has_development:
        raise MetaSynBenchmarkError("development_split_empty_choose_different_seed")
    has_calibration = any(assignments[row.review_id] == "calibration" for row in assigned_train)
    if assigned_train and not has_calibration:
        raise MetaSynBenchmarkError("calibration_split_empty_choose_different_seed")

    inputs: dict[BenchmarkSplit, list[MetaSynQuestionInput]] = {
        "development": [],
        "calibration": [],
        "test": [],
    }
    labels: list[MetaSynEvaluatorLabel] = []
    for record in sorted([*train, *test], key=lambda row: row.review_id):
        split = assignments.get(record.review_id)
        if split is None:
            continue
        component_id, _ = component_lookup[record.review_id]
        question_id = _question_id(record.review_id)
        inputs[split].append(
            MetaSynQuestionInput(
                question_id=question_id,
                review_id=record.review_id,
                research_question=record.research_question,
                population=record.population,
                intervention=record.intervention,
                exposure=record.exposure,
                comparison=record.comparison,
                outcome=record.outcome,
            )
        )
        labels.append(
            MetaSynEvaluatorLabel(
                question_id=question_id,
                review_id=record.review_id,
                official_split=record.official_split,
                split=split,
                component_id=component_id,
                gold_direction=record.gold_direction,
                gold_matched_corpus_ids=list(record.matched_corpus_ids),
                matched_reference_count=record.matched_reference_count,
                review_reported_study_count=record.study_count,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "model_inputs"
    paths = {
        split: input_dir / f"{split}.jsonl" for split in ("development", "calibration", "test")
    }
    labels_path = output_dir / "evaluator_labels.private.jsonl"
    manifest_path = output_dir / "manifest.json"
    targets = [*paths.values(), labels_path, manifest_path]
    if not force:
        existing = [path.as_posix() for path in targets if path.exists()]
        if existing:
            raise MetaSynBenchmarkError(f"benchmark_outputs_exist:{existing}")
    for split, path in paths.items():
        atomic_write_jsonl(path, inputs[split], force=force)
    labels = sorted(labels, key=lambda row: (row.split, row.review_id))
    atomic_write_jsonl(labels_path, labels, force=force)
    labels_by_split = {
        split: [label for label in labels if label.split == split]
        for split in ("development", "calibration", "test")
    }

    manifest = MetaSynBenchmarkManifest(
        seed=seed,
        calibration_fraction=calibration_fraction,
        source_train=SourceParquet(
            filename=train_parquet.name,
            sha256=sha256_file(train_parquet),
            rows=len(train),
        ),
        source_test=SourceParquet(
            filename=test_parquet.name,
            sha256=sha256_file(test_parquet),
            rows=len(test),
        ),
        development=_file_metadata(
            paths["development"],
            base=output_dir,
            rows=len(inputs["development"]),
            review_ids=(row.review_id for row in inputs["development"]),
            component_ids=(row.component_id for row in labels_by_split["development"]),
        ),
        calibration=_file_metadata(
            paths["calibration"],
            base=output_dir,
            rows=len(inputs["calibration"]),
            review_ids=(row.review_id for row in inputs["calibration"]),
            component_ids=(row.component_id for row in labels_by_split["calibration"]),
        ),
        test=_file_metadata(
            paths["test"],
            base=output_dir,
            rows=len(inputs["test"]),
            review_ids=(row.review_id for row in inputs["test"]),
            component_ids=(row.component_id for row in labels_by_split["test"]),
        ),
        evaluator_labels=_file_metadata(
            labels_path,
            base=output_dir,
            rows=len(labels),
            review_ids=(row.review_id for row in labels),
            component_ids=(row.component_id for row in labels),
        ),
        quarantined_official_train=sorted(quarantined, key=lambda row: row.review_id),
        forbidden_model_columns=sorted(FORBIDDEN_MODEL_COLUMNS),
    )
    _assert_label_paper_disjoint(labels)
    atomic_write_json(manifest_path, manifest, force=force)
    return manifest


def _artifact_path(manifest_path: Path, artifact: BenchmarkFile) -> Path:
    path = manifest_path.parent / artifact.path
    if not path.is_file():
        raise MetaSynBenchmarkError(f"benchmark_artifact_missing:{artifact.path}")
    observed = sha256_file(path)
    if observed != artifact.sha256:
        raise MetaSynBenchmarkError(
            f"benchmark_artifact_hash_mismatch:{artifact.path}:"
            f"expected={artifact.sha256}:observed={observed}"
        )
    return path


def _parse_metasyn_manifest(path: Path) -> MetaSynBenchmarkManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return MetaSynBenchmarkManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MetaSynBenchmarkError(f"benchmark_manifest_invalid:{path}") from exc


def load_metasyn_manifest(path: Path) -> MetaSynBenchmarkManifest:
    """Load and verify the complete evaluator-side benchmark bundle."""

    manifest = _parse_metasyn_manifest(path)
    for artifact in (
        manifest.development,
        manifest.calibration,
        manifest.test,
        manifest.evaluator_labels,
    ):
        _artifact_path(path, artifact)
    _validate_model_inputs(path, manifest)
    return manifest


def _read_jsonl(path: Path) -> list[Any]:
    payloads: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MetaSynBenchmarkError(f"jsonl_unreadable:{path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise MetaSynBenchmarkError(f"jsonl_invalid:{path}:line={line_number}") from exc
    return payloads


def _load_one_input_split(
    *,
    manifest_path: Path,
    manifest: MetaSynBenchmarkManifest,
    split: BenchmarkSplit,
) -> list[MetaSynQuestionInput]:
    artifact: BenchmarkFile = getattr(manifest, split)
    path = _artifact_path(manifest_path, artifact)
    try:
        rows = TypeAdapter(list[MetaSynQuestionInput]).validate_python(_read_jsonl(path))
    except ValueError as exc:
        raise MetaSynBenchmarkError(f"model_inputs_invalid:split={split}") from exc
    review_ids = sorted(row.review_id for row in rows)
    if review_ids != artifact.review_ids or len(rows) != artifact.rows:
        raise MetaSynBenchmarkError(f"model_input_review_metadata_changed:{split}")
    return rows


def load_metasyn_inputs(
    manifest_path: Path, *, split: BenchmarkSplit
) -> list[MetaSynQuestionInput]:
    """Open exactly one model-input split without touching labels or other splits.

    This is the only loader an optimizer or model runner should use.  It parses the
    inline manifest metadata, then verifies and opens only the named JSONL artifact.
    In particular it never hashes or opens the evaluator-label file or the held-out
    test file when development/calibration is requested.
    """

    if split not in {"development", "calibration", "test"}:
        raise MetaSynBenchmarkError(f"invalid_model_input_split:{split}")
    manifest = _parse_metasyn_manifest(manifest_path)
    return _load_one_input_split(
        manifest_path=manifest_path,
        manifest=manifest,
        split=split,
    )


def freeze_fixed_direction_baseline(
    *,
    manifest_path: Path,
    split: BenchmarkSplit,
    direction: FixedDirection,
    selection_note: str,
    output_dir: Path,
) -> FixedDirectionBaselineReceipt:
    """Freeze a constant direction without opening labels or producing retrievals.

    Scientific content is intentionally unused: ``review_id`` is read only to join the
    prediction back to the evaluator.  The named input split is opened exclusively via
    :func:`load_metasyn_inputs`; the evaluator-label artifact and other split files are
    never hashed or opened by this path.
    """

    if direction not in DIRECTION_CLASSES:
        raise MetaSynBenchmarkError(f"invalid_fixed_direction_class:{direction}")
    note = selection_note.strip()
    if not note:
        raise MetaSynBenchmarkError("fixed_direction_selection_note_required")
    predictions_path = output_dir / "predictions.jsonl"
    receipt_path = output_dir / "freeze_receipt.json"
    existing = [path.as_posix() for path in (predictions_path, receipt_path) if path.exists()]
    if existing:
        raise MetaSynBenchmarkError(f"fixed_direction_outputs_exist:{existing}")

    inputs = load_metasyn_inputs(manifest_path, split=split)
    manifest = _parse_metasyn_manifest(manifest_path)
    input_artifact: BenchmarkFile = getattr(manifest, split)
    prediction_payloads = [
        MetaSynPrediction(
            review_id=row.review_id,
            predicted_direction=direction,
        ).model_dump(mode="json", exclude_none=True)
        for row in sorted(inputs, key=lambda item: item.review_id)
    ]
    config = {
        "baseline_kind": "trivial_question_only_constant_direction_control",
        "predicted_class": direction,
        "selection_note": note,
        "split": split,
    }
    atomic_write_jsonl(predictions_path, prediction_payloads)
    receipt = FixedDirectionBaselineReceipt(
        split=split,
        predicted_class=direction,
        selection_note=note,
        rows=len(prediction_payloads),
        manifest_sha256=sha256_file(manifest_path),
        model_input_artifact_sha256=input_artifact.sha256,
        model_inputs_canonical_sha256=hash_canonical(inputs),
        config=config,
        config_sha256=hash_canonical(config),
        predictions_file_sha256=sha256_file(predictions_path),
        predictions_canonical_sha256=hash_canonical(prediction_payloads),
    )
    atomic_write_json(receipt_path, receipt)
    return receipt


def _validate_model_inputs(manifest_path: Path, manifest: MetaSynBenchmarkManifest) -> None:
    for split in ("development", "calibration", "test"):
        _load_one_input_split(
            manifest_path=manifest_path,
            manifest=manifest,
            split=split,
        )


def load_metasyn_labels(
    manifest_path: Path, manifest: MetaSynBenchmarkManifest | None = None
) -> list[MetaSynEvaluatorLabel]:
    manifest = manifest or load_metasyn_manifest(manifest_path)
    label_path = _artifact_path(manifest_path, manifest.evaluator_labels)
    try:
        labels = TypeAdapter(list[MetaSynEvaluatorLabel]).validate_python(_read_jsonl(label_path))
    except ValueError as exc:
        raise MetaSynBenchmarkError("evaluator_labels_invalid") from exc
    if len(labels) != manifest.evaluator_labels.rows:
        raise MetaSynBenchmarkError("evaluator_label_row_count_changed")
    if sorted(label.review_id for label in labels) != manifest.evaluator_labels.review_ids:
        raise MetaSynBenchmarkError("evaluator_label_review_metadata_changed")
    if sorted({label.component_id for label in labels}) != manifest.evaluator_labels.component_ids:
        raise MetaSynBenchmarkError("evaluator_label_component_metadata_changed")
    expected_by_split = {
        split: getattr(manifest, split).review_ids
        for split in ("development", "calibration", "test")
    }
    observed_by_split = {
        split: sorted(label.review_id for label in labels if label.split == split)
        for split in ("development", "calibration", "test")
    }
    if observed_by_split != expected_by_split:
        raise MetaSynBenchmarkError("evaluator_label_split_metadata_changed")
    _assert_label_paper_disjoint(labels)
    return labels


def load_metasyn_predictions(path: Path) -> list[MetaSynPrediction]:
    try:
        predictions = TypeAdapter(list[MetaSynPrediction]).validate_python(_read_jsonl(path))
    except ValueError as exc:
        raise MetaSynBenchmarkError("predictions_invalid") from exc
    ids = [prediction.review_id for prediction in predictions]
    if len(ids) != len(set(ids)):
        raise MetaSynBenchmarkError("duplicate_prediction_review_id")
    return sorted(predictions, key=lambda row: row.review_id)


def _assert_label_paper_disjoint(labels: Sequence[MetaSynEvaluatorLabel]) -> None:
    paper_splits: dict[int, BenchmarkSplit] = {}
    for label in labels:
        for paper_id in label.gold_matched_corpus_ids:
            prior = paper_splits.setdefault(paper_id, label.split)
            if prior != label.split:
                raise MetaSynBenchmarkError(
                    f"gold_paper_crosses_benchmark_split:{paper_id}:{prior}:{label.split}"
                )


def _validate_prediction_universe(
    labels: Sequence[MetaSynEvaluatorLabel], predictions: Sequence[MetaSynPrediction]
) -> dict[int, MetaSynPrediction]:
    known = {label.review_id for label in labels}
    by_id: dict[int, MetaSynPrediction] = {}
    for prediction in predictions:
        if prediction.review_id not in known:
            raise MetaSynBenchmarkError(
                f"prediction_review_id_not_in_manifest:{prediction.review_id}"
            )
        if prediction.review_id in by_id:
            raise MetaSynBenchmarkError(f"duplicate_prediction_review_id:{prediction.review_id}")
        by_id[prediction.review_id] = prediction
    return by_id


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_metasyn_predictions(
    *,
    manifest_path: Path,
    predictions: Sequence[MetaSynPrediction],
    evaluation_split: EvaluationSplit = "test",
) -> dict[str, Any]:
    """Score frozen outputs, defaulting to test reviews held out from model optimization."""

    manifest = load_metasyn_manifest(manifest_path)
    all_labels = load_metasyn_labels(manifest_path, manifest)
    prediction_by_id = _validate_prediction_universe(all_labels, predictions)
    if evaluation_split not in {"development", "calibration", "test", "all"}:
        raise MetaSynBenchmarkError(f"invalid_evaluation_split:{evaluation_split}")
    labels = (
        all_labels
        if evaluation_split == "all"
        else [label for label in all_labels if label.split == evaluation_split]
    )

    gold_counts = {label: 0 for label in DIRECTION_CLASSES}
    answered_counts = {label: 0 for label in DIRECTION_CLASSES}
    predicted_counts = {label: 0 for label in DIRECTION_CLASSES}
    true_positive = {label: 0 for label in DIRECTION_CLASSES}
    confusion = {
        gold: {predicted: 0 for predicted in DIRECTION_CLASSES} for gold in DIRECTION_CLASSES
    }
    eligible = correct = answered = 0
    direction_missing = direction_nr = direction_abstained = gold_nr = 0
    retrieval_supplied = retrieval_missing = retrieval_empty = 0
    total_gold_retrieval = total_retrieved = total_hits = 0
    retrieval_recalls_all: list[float] = []
    retrieval_recalls_supplied: list[float] = []
    retrieval_precisions_nonempty: list[float] = []
    per_review: list[dict[str, Any]] = []

    for label in sorted(labels, key=lambda row: row.review_id):
        prediction = prediction_by_id.get(label.review_id)
        predicted_direction = None if prediction is None else prediction.predicted_direction
        direction_status: str
        direction_correct: bool | None = None
        if label.gold_direction == "NR":
            gold_nr += 1
            direction_status = "gold_nr_excluded"
        else:
            eligible += 1
            gold_counts[label.gold_direction] += 1
            if predicted_direction is None:
                direction_missing += 1
                direction_status = "missing"
            elif predicted_direction == "NR":
                direction_nr += 1
                direction_status = "predicted_nr"
            elif predicted_direction == "Abstain":
                direction_abstained += 1
                direction_status = "abstained"
            else:
                answered += 1
                answered_counts[label.gold_direction] += 1
                predicted_counts[predicted_direction] += 1
                confusion[label.gold_direction][predicted_direction] += 1
                direction_correct = predicted_direction == label.gold_direction
                correct += int(direction_correct)
                true_positive[label.gold_direction] += int(direction_correct)
                direction_status = "correct" if direction_correct else "incorrect"

        gold_ids = set(label.gold_matched_corpus_ids)
        total_gold_retrieval += len(gold_ids)
        retrieved = None if prediction is None else prediction.retrieved_corpus_ids
        if retrieved is None:
            retrieval_missing += 1
            retrieval_status = "missing"
            hits = 0
            recall = 0.0
            precision = None
        else:
            retrieval_supplied += 1
            retrieved_ids = set(retrieved)
            hits = len(gold_ids & retrieved_ids)
            total_hits += hits
            total_retrieved += len(retrieved_ids)
            recall = hits / len(gold_ids)
            retrieval_recalls_supplied.append(recall)
            if retrieved_ids:
                retrieval_status = "supplied"
                precision = hits / len(retrieved_ids)
                retrieval_precisions_nonempty.append(precision)
            else:
                retrieval_empty += 1
                retrieval_status = "empty"
                precision = None
        retrieval_recalls_all.append(recall)
        per_review.append(
            {
                "question_id": label.question_id,
                "review_id": label.review_id,
                "split": label.split,
                "gold_direction": label.gold_direction,
                "predicted_direction": predicted_direction,
                "direction_status": direction_status,
                "direction_correct": direction_correct,
                "retrieval_status": retrieval_status,
                "gold_retrieval_count": len(gold_ids),
                "retrieved_count": None if retrieved is None else len(retrieved),
                "retrieval_hits": hits,
                "retrieval_recall": recall,
                "retrieval_precision": precision,
            }
        )

    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for direction in DIRECTION_CLASSES:
        precision = _safe_ratio(true_positive[direction], predicted_counts[direction])
        recall = _safe_ratio(true_positive[direction], gold_counts[direction])
        f1 = (
            0.0
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        if gold_counts[direction]:
            f1_values.append(f1)
        per_class[direction] = {
            "gold": gold_counts[direction],
            "answered": answered_counts[direction],
            "predicted": predicted_counts[direction],
            "precision": precision,
            "recall_with_unanswered_as_errors": recall,
            "f1_with_unanswered_as_errors": f1,
        }

    return {
        "metasyn_evaluation_version": "1",
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": hash_canonical(list(predictions)),
        "evaluated_split": evaluation_split,
        "benchmark_scope": "review_level_not_study_level_meta_analysis",
        "loss_interpretation": (
            "Direction error means disagreement with the frozen MetaSyn review-level "
            "benchmark annotation; it is not an error label for scientific truth."
        ),
        "direction": {
            "eligible_gold": eligible,
            "gold_nr_excluded": gold_nr,
            "answered": answered,
            "coverage": _safe_ratio(answered, eligible),
            "missing": direction_missing,
            "predicted_nr": direction_nr,
            "abstained": direction_abstained,
            "correct": correct,
            "accuracy_on_answered": _safe_ratio(correct, answered),
            "strict_accuracy_with_unanswered_as_errors": _safe_ratio(correct, eligible),
            "macro_f1_with_unanswered_as_errors": (
                sum(f1_values) / len(f1_values) if f1_values else None
            ),
            "confusion_on_answered": confusion,
            "per_class": per_class,
        },
        "retrieval": {
            "eligible_reviews": len(labels),
            "supplied": retrieval_supplied,
            "coverage": _safe_ratio(retrieval_supplied, len(labels)),
            "missing": retrieval_missing,
            "explicit_empty": retrieval_empty,
            "macro_recall_missing_as_zero": (
                sum(retrieval_recalls_all) / len(retrieval_recalls_all)
                if retrieval_recalls_all
                else None
            ),
            "macro_recall_on_supplied": (
                sum(retrieval_recalls_supplied) / len(retrieval_recalls_supplied)
                if retrieval_recalls_supplied
                else None
            ),
            "micro_recall_missing_as_zero": _safe_ratio(total_hits, total_gold_retrieval),
            "micro_precision_on_retrieved": _safe_ratio(total_hits, total_retrieved),
            "macro_precision_on_nonempty": (
                sum(retrieval_precisions_nonempty) / len(retrieval_precisions_nonempty)
                if retrieval_precisions_nonempty
                else None
            ),
        },
        "per_review": per_review,
    }


def build_metasyn_risk_examples(
    *,
    manifest_path: Path,
    predictions: Sequence[MetaSynPrediction],
    pipeline_sha256: str,
    population_id: str = "metasyn-review-direction-v1",
    label_source: LabelSource = "benchmark_annotation",
) -> list[RiskExample]:
    """Convert answered, feature-bearing predictions into real benchmark losses.

    No system output is imputed.  Rows with a missing/NR benchmark label, missing
    direction, explicit abstention/NR, an empty or missing retrieved corpus, or missing
    features are omitted.  ``paper_ids`` are the system's actual retrieved corpus—not
    gold papers—and split integrity is checked before return.  The binary loss is
    disagreement with the frozen review-level direction annotation, not a claim about
    scientific truth.
    """

    if not SHA256_RE.fullmatch(pipeline_sha256):
        raise MetaSynBenchmarkError("invalid_pipeline_sha256")
    manifest = load_metasyn_manifest(manifest_path)
    labels = load_metasyn_labels(manifest_path, manifest)
    prediction_by_id = _validate_prediction_universe(labels, predictions)
    examples: list[RiskExample] = []
    for label in sorted(labels, key=lambda row: row.question_id):
        prediction = prediction_by_id.get(label.review_id)
        if (
            label.gold_direction not in DIRECTION_CLASSES
            or prediction is None
            or prediction.predicted_direction not in DIRECTION_CLASSES
            or not prediction.retrieved_corpus_ids
            or prediction.risk_features is None
        ):
            continue
        examples.append(
            RiskExample(
                question_id=label.question_id,
                split=label.split,
                population_id=population_id,
                domain="metasyn_systematic_reviews",
                pipeline_sha256=pipeline_sha256,
                paper_ids=sorted(
                    {f"metasyn-corpus:{paper_id}" for paper_id in prediction.retrieved_corpus_ids}
                ),
                features=prediction.risk_features,
                unsupported_claim=prediction.predicted_direction != label.gold_direction,
                label_source=label_source,
            )
        )
    if examples:
        validate_split_integrity(examples)
    return examples


__all__ = [
    "DIRECTION_CLASSES",
    "FORBIDDEN_MODEL_COLUMNS",
    "FixedDirectionBaselineReceipt",
    "MetaSynBenchmarkError",
    "MetaSynBenchmarkManifest",
    "MetaSynEvaluatorLabel",
    "MetaSynPrediction",
    "MetaSynQuestionInput",
    "build_metasyn_risk_examples",
    "evaluate_metasyn_predictions",
    "freeze_fixed_direction_baseline",
    "load_metasyn_inputs",
    "load_metasyn_labels",
    "load_metasyn_manifest",
    "load_metasyn_predictions",
    "prepare_metasyn_benchmark",
]
