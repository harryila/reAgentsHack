"""Staged, label-firewalled EvidenceBench grounding diagnostic.

The public EvidenceBench test set is accessible rather than secret.  This module does
not pretend otherwise.  It makes the narrower, auditable claim that this project's
method roster and development-set choice were frozen before the test labels were
materialized, predictions read only a label-free projection, and the public result is
aggregate-only.

The four stages deliberately have disjoint inputs:

``prepare``
    Reads development data and an expected test-file digest, but has no test path.
``materialize``
    Verifies the frozen plan and splits the pinned test file into visible inputs and
    private gold.
``predict``
    Reads only the visible projection and emits rankings for every frozen method.
``score``
    Joins rankings to private gold and emits an aggregate-only, self-hashed summary.

This is an evidence-retrieval diagnostic.  It does not validate effect extraction,
meta-analysis, claim correctness, or calibrated release.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer

from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel

EVIDENCEBENCH_PROTOCOL_VERSION = "evidencebench-grounding-diagnostic-v1"
EVIDENCEBENCH_UPSTREAM_COMMIT = "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
EVIDENCEBENCH_TRAIN_SHA256 = (
    "a89bb2db83cbe68f5df7af6a430899bdb63af0da731268746d89a2f54c059865"
)
EVIDENCEBENCH_DEV_SHA256 = (
    "0bd63b7f1156d509abe8757ce27f38d31c08059f750c51a6756ec076b3d96a59"
)
EVIDENCEBENCH_TEST_SHA256 = (
    "771cc068e3e79764c2eb523e7d0d9fb276d00254a99212c73c1137674b49d18b"
)
EVIDENCEBENCH_TEST_ROWS = 293
EVIDENCEBENCH_DEV_ROWS = 37
DEFAULT_BOOTSTRAP_SEED = 20260828
DEFAULT_BOOTSTRAP_REPLICATES = 5000
_PROTOCOL_CONFIG_RELATIVE_PATH = Path(
    "configs/benchmarks/evidencebench-grounding-v1.json"
)

type EvidenceBenchMethodId = Literal[
    "bm25-v1",
    "tfidf-word-v1",
    "tfidf-char-v1",
    "rrf-word-char-v1",
    "first-sentences-control-v1",
    "deterministic-random-control-v1",
]

_SELECTION_METHODS: tuple[EvidenceBenchMethodId, ...] = (
    "bm25-v1",
    "tfidf-word-v1",
    "tfidf-char-v1",
    "rrf-word-char-v1",
)
_CONTROL_METHODS: tuple[EvidenceBenchMethodId, ...] = (
    "first-sentences-control-v1",
    "deterministic-random-control-v1",
)
_METHOD_ROSTER = _SELECTION_METHODS + _CONTROL_METHODS
_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_-]+\b")
_QUESTION_ID_RE = re.compile(r"^evidencebench_(?:train|dev|test)_id_[0-9]+$")
_MACHINE_REASON_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9_.:-]*[a-z0-9])?$")


class EvidenceBenchDiagnosticError(ValueError):
    """A benchmark stage violated its frozen or label-firewalled contract."""


def _validate_sha256(value: str, name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"evidencebench_invalid_sha256:{name}")
    return value


def _self_hash(model: ContractModel, field: str) -> str:
    return hash_canonical(model.model_dump(mode="json", exclude={field}))


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBenchDiagnosticError(
            f"evidencebench_json_unreadable:{path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceBenchDiagnosticError(
            f"evidencebench_dataset_not_object:{path.name}"
        )
    return payload


def _assert_exact_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise EvidenceBenchDiagnosticError(
            f"evidencebench_{label}_hash_mismatch:expected={expected}:observed={observed}"
        )


def _implementation_sha256() -> str:
    module_path = Path(__file__).resolve()
    root = module_path.parents[2]
    sources = [
        root / "src" / "literature_multiverse" / "__init__.py",
        module_path,
        root / "src" / "literature_multiverse" / "lineage.py",
        root / "src" / "literature_multiverse" / "models.py",
        root / "src" / "literature_multiverse" / "paths.py",
        root / "scripts" / "run_evidencebench_diagnostic.py",
        root / _PROTOCOL_CONFIG_RELATIVE_PATH,
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    missing = [path.as_posix() for path in sources if not path.is_file()]
    if missing:
        raise EvidenceBenchDiagnosticError(
            "evidencebench_implementation_source_missing:" + ",".join(missing)
        )
    return hash_canonical(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sources
        ]
    )


def _protocol_config_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / _PROTOCOL_CONFIG_RELATIVE_PATH
    config = _load_json_object(path)
    files = config.get("files")
    if not isinstance(files, Mapping):
        raise EvidenceBenchDiagnosticError("evidencebench_protocol_files_invalid")
    expected = {
        "benchmark_id": "evidencebench-original-grounding-v1",
        "upstream_repository": "EvidenceBench/EvidenceBench",
        "upstream_commit": EVIDENCEBENCH_UPSTREAM_COMMIT,
        "selectable_methods": sorted(_SELECTION_METHODS),
        "controls": sorted(_CONTROL_METHODS),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise EvidenceBenchDiagnosticError(
                f"evidencebench_protocol_config_mismatch:{key}"
            )
    expected_files = {
        "train": (EVIDENCEBENCH_TRAIN_SHA256, 96),
        "development": (EVIDENCEBENCH_DEV_SHA256, EVIDENCEBENCH_DEV_ROWS),
        "test": (EVIDENCEBENCH_TEST_SHA256, EVIDENCEBENCH_TEST_ROWS),
    }
    for split, (expected_hash, expected_rows) in expected_files.items():
        value = files.get(split)
        if (
            not isinstance(value, Mapping)
            or value.get("sha256") != expected_hash
            or value.get("rows") != expected_rows
        ):
            raise EvidenceBenchDiagnosticError(
                f"evidencebench_protocol_config_mismatch:{split}"
            )
    return sha256_file(path)


def _runtime_versions() -> dict[str, str]:
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pydantic", "scikit-learn", "scipy")
    }
    return {"python": platform.python_version(), **versions}


def _assert_current_plan_environment(plan: EvidenceBenchFrozenPlanV1) -> None:
    if plan.protocol_config_sha256 != _protocol_config_sha256():
        raise EvidenceBenchDiagnosticError("evidencebench_protocol_config_drift")
    if plan.implementation_sha256 != _implementation_sha256():
        raise EvidenceBenchDiagnosticError("evidencebench_implementation_drift")
    if plan.runtime_versions != _runtime_versions():
        raise EvidenceBenchDiagnosticError("evidencebench_runtime_drift")


class EvidenceBenchMetricV1(ContractModel):
    all_aspect_recall_at_10: Annotated[float, Field(ge=0.0, le=1.0)]
    results_aspect_recall_at_5: Annotated[float, Field(ge=0.0, le=1.0)] | None


class EvidenceBenchDevelopmentResultV1(ContractModel):
    method_id: EvidenceBenchMethodId
    question_count: Annotated[int, Field(gt=0)]
    results_metric_question_count: Annotated[int, Field(gt=0)]
    mean: EvidenceBenchMetricV1

    @model_validator(mode="after")
    def validate_counts(self) -> EvidenceBenchDevelopmentResultV1:
        if self.results_metric_question_count > self.question_count:
            raise ValueError("evidencebench_development_results_count_invalid")
        if self.mean.results_aspect_recall_at_5 is None:
            raise ValueError("evidencebench_development_results_mean_missing")
        return self


class EvidenceBenchFrozenPlanV1(ContractModel):
    plan_version: Literal["evidencebench-frozen-plan-v1"] = (
        "evidencebench-frozen-plan-v1"
    )
    protocol_version: Literal["evidencebench-grounding-diagnostic-v1"] = (
        EVIDENCEBENCH_PROTOCOL_VERSION
    )
    upstream_repository: Literal["EvidenceBench/EvidenceBench"] = (
        "EvidenceBench/EvidenceBench"
    )
    upstream_commit: Literal[
        "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
    ] = EVIDENCEBENCH_UPSTREAM_COMMIT
    development_sha256: str
    expected_test_sha256: str
    expected_test_rows: Annotated[int, Field(gt=0)]
    method_roster: list[EvidenceBenchMethodId]
    selectable_method_roster: list[EvidenceBenchMethodId]
    controls: list[EvidenceBenchMethodId]
    selection_rule: Literal[
        "maximize development mean all-aspect recall@10; then results-aspect "
        "recall@5; then ascending method id"
    ] = (
        "maximize development mean all-aspect recall@10; then results-aspect "
        "recall@5; then ascending method id"
    )
    primary_test_metric: Literal["mean all-aspect recall@10"] = (
        "mean all-aspect recall@10"
    )
    secondary_test_metric: Literal["mean results-aspect recall@5"] = (
        "mean results-aspect recall@5"
    )
    development_results: list[EvidenceBenchDevelopmentResultV1]
    selected_method_id: EvidenceBenchMethodId
    bootstrap_seed: Annotated[int, Field(ge=0)]
    bootstrap_replicates: Annotated[int, Field(ge=1000)]
    protocol_config_sha256: str
    implementation_sha256: str
    runtime_versions: dict[str, str]
    test_labels_opened_by_prepare: Literal[False] = False
    scope_limitations: list[str]
    plan_sha256: str

    @field_validator(
        "development_sha256",
        "expected_test_sha256",
        "protocol_config_sha256",
        "implementation_sha256",
        "plan_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("method_roster", "selectable_method_roster", "controls")
    @classmethod
    def validate_method_list(
        cls, value: list[EvidenceBenchMethodId], info: Any
    ) -> list[EvidenceBenchMethodId]:
        if value != sorted(set(value)):
            raise ValueError(f"evidencebench_methods_not_sorted:{info.field_name}")
        return value

    @field_validator("scope_limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            _MACHINE_REASON_RE.fullmatch(item) is None for item in value
        ):
            raise ValueError("evidencebench_scope_limitations_invalid")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceBenchFrozenPlanV1:
        if self.development_sha256 != EVIDENCEBENCH_DEV_SHA256:
            raise ValueError("evidencebench_development_pin_mismatch")
        if self.expected_test_sha256 != EVIDENCEBENCH_TEST_SHA256:
            raise ValueError("evidencebench_test_pin_mismatch")
        if self.expected_test_rows != EVIDENCEBENCH_TEST_ROWS:
            raise ValueError("evidencebench_test_row_pin_mismatch")
        if self.method_roster != sorted(_METHOD_ROSTER):
            raise ValueError("evidencebench_method_roster_mismatch")
        if self.selectable_method_roster != sorted(_SELECTION_METHODS):
            raise ValueError("evidencebench_selectable_roster_mismatch")
        if self.controls != sorted(_CONTROL_METHODS):
            raise ValueError("evidencebench_control_roster_mismatch")
        results = {row.method_id: row for row in self.development_results}
        result_ids = [row.method_id for row in self.development_results]
        if (
            len(results) != len(self.development_results)
            or set(results) != set(_SELECTION_METHODS)
            or result_ids != sorted(result_ids)
        ):
            raise ValueError("evidencebench_development_results_incomplete")
        if any(row.question_count != EVIDENCEBENCH_DEV_ROWS for row in results.values()):
            raise ValueError("evidencebench_development_question_count_mismatch")
        eligible_counts = {
            row.results_metric_question_count for row in results.values()
        }
        if len(eligible_counts) != 1:
            raise ValueError("evidencebench_development_results_denominator_mismatch")
        ordered = sorted(
            results.values(),
            key=lambda row: (
                -row.mean.all_aspect_recall_at_10,
                -row.mean.results_aspect_recall_at_5,
                row.method_id,
            ),
        )
        if self.selected_method_id != ordered[0].method_id:
            raise ValueError("evidencebench_selected_method_mismatch")
        if set(self.runtime_versions) != {
            "numpy",
            "pydantic",
            "python",
            "scikit-learn",
            "scipy",
        }:
            raise ValueError("evidencebench_runtime_versions_incomplete")
        if self.plan_sha256 != _self_hash(self, "plan_sha256"):
            raise ValueError("evidencebench_plan_hash_mismatch")
        return self


class EvidenceBenchVisibleQuestionV1(ContractModel):
    question_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    hypothesis: Annotated[str, Field(min_length=1)]
    candidate_sentences: Annotated[list[str], Field(min_length=1)]
    sentence_types: list[Literal["section_name", "abstract", "normal_paragraph"]]

    @model_validator(mode="after")
    def validate_question(self) -> EvidenceBenchVisibleQuestionV1:
        if _QUESTION_ID_RE.fullmatch(self.question_id) is None:
            raise ValueError("evidencebench_question_id_invalid")
        if len(self.candidate_sentences) != len(self.sentence_types):
            raise ValueError("evidencebench_visible_sentence_type_length_mismatch")
        return self


class EvidenceBenchPrivateGoldV1(ContractModel):
    question_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    sentence_count: Annotated[int, Field(gt=0)]
    all_aspect_ids: Annotated[list[str], Field(min_length=1)]
    results_aspect_ids: list[str]
    sentence_index_to_aspects: dict[str, list[str]]
    all_oracle_covered_aspect_ids_at_10: list[str]
    results_oracle_covered_aspect_ids_at_5: list[str]

    @field_validator(
        "all_aspect_ids",
        "results_aspect_ids",
        "all_oracle_covered_aspect_ids_at_10",
        "results_oracle_covered_aspect_ids_at_5",
    )
    @classmethod
    def validate_sorted_ids(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"evidencebench_aspect_ids_not_sorted:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_gold(self) -> EvidenceBenchPrivateGoldV1:
        if _QUESTION_ID_RE.fullmatch(self.question_id) is None:
            raise ValueError("evidencebench_gold_question_id_invalid")
        all_aspects = set(self.all_aspect_ids)
        results_aspects = set(self.results_aspect_ids)
        if not results_aspects <= all_aspects:
            raise ValueError("evidencebench_results_aspects_not_subset")
        if not set(self.all_oracle_covered_aspect_ids_at_10) <= all_aspects:
            raise ValueError("evidencebench_all_oracle_aspects_not_subset")
        if not set(self.results_oracle_covered_aspect_ids_at_5) <= results_aspects:
            raise ValueError("evidencebench_results_oracle_aspects_not_subset")
        observed_aspects: set[str] = set()
        for raw_index, aspect_ids in self.sentence_index_to_aspects.items():
            if not raw_index.isdigit() or not 0 <= int(raw_index) < self.sentence_count:
                raise ValueError("evidencebench_gold_sentence_index_invalid")
            if aspect_ids != sorted(set(aspect_ids)) or not set(aspect_ids) <= all_aspects:
                raise ValueError("evidencebench_gold_sentence_aspects_invalid")
            observed_aspects.update(aspect_ids)
        if observed_aspects != all_aspects:
            raise ValueError("evidencebench_gold_aspect_mapping_incomplete")
        return self


class EvidenceBenchMaterializationReceiptV1(ContractModel):
    receipt_version: Literal["evidencebench-materialization-receipt-v1"] = (
        "evidencebench-materialization-receipt-v1"
    )
    plan_sha256: str
    raw_test_sha256: str
    visible_projection_sha256: str
    private_gold_sha256: str
    question_count: Annotated[int, Field(gt=0)]
    unique_paper_count: Annotated[int, Field(gt=0)]
    visible_projection_contains_gold_labels: Literal[False] = False
    private_gold_contains_candidate_text: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "plan_sha256",
        "raw_test_sha256",
        "visible_projection_sha256",
        "private_gold_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceBenchMaterializationReceiptV1:
        if self.raw_test_sha256 != EVIDENCEBENCH_TEST_SHA256:
            raise ValueError("evidencebench_materialization_test_pin_mismatch")
        if self.question_count != EVIDENCEBENCH_TEST_ROWS:
            raise ValueError("evidencebench_materialization_row_count_mismatch")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("evidencebench_materialization_receipt_hash_mismatch")
        return self


class EvidenceBenchPredictionRowV1(ContractModel):
    question_id: Annotated[str, Field(min_length=1)]
    method_rankings: dict[EvidenceBenchMethodId, list[int]]

    @model_validator(mode="after")
    def validate_prediction(self) -> EvidenceBenchPredictionRowV1:
        if set(self.method_rankings) != set(_METHOD_ROSTER):
            raise ValueError("evidencebench_prediction_methods_incomplete")
        for ranking in self.method_rankings.values():
            if len(ranking) > 10 or len(ranking) != len(set(ranking)):
                raise ValueError("evidencebench_prediction_ranking_invalid")
            if any(index < 0 for index in ranking):
                raise ValueError("evidencebench_prediction_index_negative")
        return self


class EvidenceBenchPredictionReceiptV1(ContractModel):
    receipt_version: Literal["evidencebench-prediction-receipt-v1"] = (
        "evidencebench-prediction-receipt-v1"
    )
    plan_sha256: str
    materialization_receipt_sha256: str
    visible_projection_sha256: str
    predictions_sha256: str
    question_count: Annotated[int, Field(gt=0)]
    method_roster: list[EvidenceBenchMethodId]
    private_gold_opened_by_prediction: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "plan_sha256",
        "materialization_receipt_sha256",
        "visible_projection_sha256",
        "predictions_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceBenchPredictionReceiptV1:
        if self.question_count != EVIDENCEBENCH_TEST_ROWS:
            raise ValueError("evidencebench_prediction_row_count_mismatch")
        if self.method_roster != sorted(_METHOD_ROSTER):
            raise ValueError("evidencebench_prediction_roster_mismatch")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("evidencebench_prediction_receipt_hash_mismatch")
        return self


class EvidenceBenchIntervalV1(ContractModel):
    estimate: Annotated[float, Field(ge=0.0, le=1.0)]
    lower_95: Annotated[float, Field(ge=0.0, le=1.0)]
    upper_95: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_interval(self) -> EvidenceBenchIntervalV1:
        if not self.lower_95 <= self.estimate <= self.upper_95:
            raise ValueError("evidencebench_interval_order_invalid")
        return self


class EvidenceBenchSignedIntervalV1(ContractModel):
    estimate: Annotated[float, Field(ge=-1.0, le=1.0)]
    lower_95: Annotated[float, Field(ge=-1.0, le=1.0)]
    upper_95: Annotated[float, Field(ge=-1.0, le=1.0)]

    @model_validator(mode="after")
    def validate_interval(self) -> EvidenceBenchSignedIntervalV1:
        if not self.lower_95 <= self.estimate <= self.upper_95:
            raise ValueError("evidencebench_signed_interval_order_invalid")
        return self


class EvidenceBenchMethodTestResultV1(ContractModel):
    method_id: EvidenceBenchMethodId
    selected_on_development: bool
    control: bool
    results_metric_question_count: Annotated[int, Field(gt=0)]
    all_aspect_recall_at_10: EvidenceBenchIntervalV1
    results_aspect_recall_at_5: EvidenceBenchIntervalV1


class EvidenceBenchPairedDeltaV1(ContractModel):
    selected_method_id: EvidenceBenchMethodId
    comparator_method_id: EvidenceBenchMethodId
    comparator_is_control: bool
    question_count: Annotated[int, Field(gt=0)]
    results_metric_question_count: Annotated[int, Field(gt=0)]
    all_aspect_recall_at_10_delta: EvidenceBenchSignedIntervalV1
    results_aspect_recall_at_5_delta: EvidenceBenchSignedIntervalV1

    @model_validator(mode="after")
    def validate_comparison(self) -> EvidenceBenchPairedDeltaV1:
        if self.selected_method_id == self.comparator_method_id:
            raise ValueError("evidencebench_paired_delta_self_comparison")
        if self.comparator_is_control != (
            self.comparator_method_id in _CONTROL_METHODS
        ):
            raise ValueError("evidencebench_paired_delta_control_flag_mismatch")
        if self.results_metric_question_count > self.question_count:
            raise ValueError("evidencebench_paired_delta_results_count_invalid")
        return self


class EvidenceBenchPublicSummaryV1(ContractModel):
    summary_version: Literal["evidencebench-public-summary-v1"] = (
        "evidencebench-public-summary-v1"
    )
    protocol_version: Literal["evidencebench-grounding-diagnostic-v1"] = (
        EVIDENCEBENCH_PROTOCOL_VERSION
    )
    evidence_status: Literal["retrospective_public_test_diagnostic"] = (
        "retrospective_public_test_diagnostic"
    )
    upstream_repository: Literal["EvidenceBench/EvidenceBench"] = (
        "EvidenceBench/EvidenceBench"
    )
    upstream_commit: Literal[
        "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
    ] = EVIDENCEBENCH_UPSTREAM_COMMIT
    test_license: Literal["CC-BY"] = "CC-BY"
    plan_sha256: str
    materialization_receipt_sha256: str
    prediction_receipt_sha256: str
    raw_test_sha256: str
    private_gold_sha256: str
    predictions_sha256: str
    protocol_config_sha256: str
    implementation_sha256: str
    selected_method_id: EvidenceBenchMethodId
    question_count: Annotated[int, Field(gt=0)]
    results_metric_question_count: Annotated[int, Field(gt=0)]
    unique_paper_count: Annotated[int, Field(gt=0)]
    candidate_sentence_count: Annotated[int, Field(gt=0)]
    median_candidate_sentences_per_question: Annotated[float, Field(gt=0)]
    bootstrap_seed: Annotated[int, Field(ge=0)]
    bootstrap_replicates: Annotated[int, Field(ge=1000)]
    test_results: list[EvidenceBenchMethodTestResultV1]
    selected_method_paired_deltas: list[EvidenceBenchPairedDeltaV1]
    oracle_upper_bounds: EvidenceBenchMetricV1
    output_is_aggregate_only: Literal[True] = True
    licenses_scientific_scope: list[str]
    summary_sha256: str

    @field_validator(
        "plan_sha256",
        "materialization_receipt_sha256",
        "prediction_receipt_sha256",
        "raw_test_sha256",
        "private_gold_sha256",
        "predictions_sha256",
        "protocol_config_sha256",
        "implementation_sha256",
        "summary_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("licenses_scientific_scope")
    @classmethod
    def validate_scope(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            _MACHINE_REASON_RE.fullmatch(item) is None for item in value
        ):
            raise ValueError("evidencebench_public_scope_invalid")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> EvidenceBenchPublicSummaryV1:
        if self.raw_test_sha256 != EVIDENCEBENCH_TEST_SHA256:
            raise ValueError("evidencebench_public_test_pin_mismatch")
        if self.question_count != EVIDENCEBENCH_TEST_ROWS:
            raise ValueError("evidencebench_public_row_count_mismatch")
        if self.results_metric_question_count > self.question_count:
            raise ValueError("evidencebench_public_results_count_invalid")
        rows = {row.method_id: row for row in self.test_results}
        result_ids = [row.method_id for row in self.test_results]
        if (
            len(rows) != len(self.test_results)
            or set(rows) != set(_METHOD_ROSTER)
            or result_ids != sorted(result_ids)
        ):
            raise ValueError("evidencebench_public_method_results_incomplete")
        selected = [row.method_id for row in rows.values() if row.selected_on_development]
        if selected != [self.selected_method_id]:
            raise ValueError("evidencebench_public_selected_method_mismatch")
        for method_id, row in rows.items():
            if row.control != (method_id in _CONTROL_METHODS):
                raise ValueError("evidencebench_public_control_flag_mismatch")
            if row.results_metric_question_count != self.results_metric_question_count:
                raise ValueError("evidencebench_public_results_denominator_mismatch")
        comparators = [
            row.comparator_method_id for row in self.selected_method_paired_deltas
        ]
        expected_comparators = sorted(set(_METHOD_ROSTER) - {self.selected_method_id})
        if comparators != expected_comparators or len(comparators) != len(set(comparators)):
            raise ValueError("evidencebench_public_paired_deltas_incomplete")
        for row in self.selected_method_paired_deltas:
            if (
                row.selected_method_id != self.selected_method_id
                or row.question_count != self.question_count
                or row.results_metric_question_count
                != self.results_metric_question_count
            ):
                raise ValueError("evidencebench_public_paired_delta_binding_mismatch")
        if self.summary_sha256 != _self_hash(self, "summary_sha256"):
            raise ValueError("evidencebench_public_summary_hash_mismatch")
        return self


class EvidenceBenchReplayAuditReceiptV1(ContractModel):
    receipt_version: Literal["evidencebench-replay-audit-receipt-v1"] = (
        "evidencebench-replay-audit-receipt-v1"
    )
    plan_sha256: str
    materialization_receipt_sha256: str
    prediction_receipt_sha256: str
    summary_sha256: str
    protocol_config_sha256: str
    implementation_sha256: str
    runtime_versions: dict[str, str]
    exact_replay_status: Literal["passed"] = "passed"
    test_gold_opened_only_after_prediction_freeze_validation: Literal[True] = True
    public_artifacts_contain_private_rows: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "plan_sha256",
        "materialization_receipt_sha256",
        "prediction_receipt_sha256",
        "summary_sha256",
        "protocol_config_sha256",
        "implementation_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceBenchReplayAuditReceiptV1:
        if set(self.runtime_versions) != {
            "numpy",
            "pydantic",
            "python",
            "scikit-learn",
            "scipy",
        }:
            raise ValueError("evidencebench_audit_runtime_versions_incomplete")
        if self.receipt_sha256 != _self_hash(self, "receipt_sha256"):
            raise ValueError("evidencebench_audit_receipt_hash_mismatch")
        return self


def _validate_dataset_row(
    question_id: str, raw: Any
) -> tuple[EvidenceBenchVisibleQuestionV1, EvidenceBenchPrivateGoldV1]:
    if not isinstance(raw, Mapping):
        raise EvidenceBenchDiagnosticError("evidencebench_question_not_object")
    hypothesis = raw.get("hypothesis")
    sentences = raw.get("paper_as_candidate_pool")
    sentence_types = raw.get("sentence_types_in_candidate_pool")
    paper_id = raw.get("paper_id")
    all_aspects = raw.get("aspect_list_ids")
    results_aspects = raw.get("results_aspect_list_ids")
    forward = raw.get("aspect2sentence_indices")
    inverse = raw.get("sentence_index2aspects")
    all_eval = raw.get("evidence_retrieval_at_10_evaluation")
    results_eval = raw.get("results_evidence_retrieval_at_5_evaluation")
    # The pinned upstream release represents questions without Results aspects with
    # JSON null in both fields.  Normalize that documented absence to empty sets;
    # such questions are excluded from the secondary metric denominator.
    if results_aspects is None and results_eval is None:
        results_aspects = []
        results_eval = {
            "one_selection_of_sentences": [],
            "covered_aspects": [],
        }
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise EvidenceBenchDiagnosticError("evidencebench_hypothesis_invalid")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise EvidenceBenchDiagnosticError("evidencebench_paper_id_invalid")
    if not isinstance(sentences, list) or not sentences or not all(
        isinstance(value, str) and value.strip() for value in sentences
    ):
        raise EvidenceBenchDiagnosticError("evidencebench_candidate_sentences_invalid")
    if not isinstance(sentence_types, list) or len(sentence_types) != len(sentences):
        raise EvidenceBenchDiagnosticError("evidencebench_sentence_types_invalid")
    if not isinstance(all_aspects, list) or not all_aspects:
        raise EvidenceBenchDiagnosticError("evidencebench_all_aspects_invalid")
    if not isinstance(results_aspects, list):
        raise EvidenceBenchDiagnosticError("evidencebench_results_aspects_invalid")
    if not isinstance(forward, Mapping) or not isinstance(inverse, Mapping):
        raise EvidenceBenchDiagnosticError("evidencebench_aspect_mapping_invalid")
    if not isinstance(all_eval, Mapping) or not isinstance(results_eval, Mapping):
        raise EvidenceBenchDiagnosticError("evidencebench_evaluation_payload_invalid")
    all_covered = all_eval.get("covered_aspects")
    results_covered = results_eval.get("covered_aspects")
    if not isinstance(all_covered, list) or not isinstance(results_covered, list):
        raise EvidenceBenchDiagnosticError("evidencebench_oracle_aspects_invalid")
    all_aspect_set = set(all_aspects)
    results_aspect_set = set(results_aspects)
    if len(all_aspect_set) != len(all_aspects) or not results_aspect_set <= all_aspect_set:
        raise EvidenceBenchDiagnosticError("evidencebench_aspect_roster_invalid")
    normalized_forward: dict[str, set[int]] = {}
    for aspect_id, indices in forward.items():
        if aspect_id not in all_aspect_set or not isinstance(indices, list):
            raise EvidenceBenchDiagnosticError("evidencebench_forward_mapping_invalid")
        if any(
            not isinstance(index, int) or not 0 <= index < len(sentences)
            for index in indices
        ):
            raise EvidenceBenchDiagnosticError("evidencebench_forward_index_invalid")
        normalized_forward[str(aspect_id)] = set(indices)
    if set(normalized_forward) != all_aspect_set:
        raise EvidenceBenchDiagnosticError("evidencebench_forward_mapping_incomplete")
    derived_forward: dict[str, set[int]] = {
        str(aspect_id): set() for aspect_id in all_aspects
    }
    for raw_index, aspect_ids in inverse.items():
        if (
            not isinstance(raw_index, str)
            or not raw_index.isdigit()
            or not 0 <= int(raw_index) < len(sentences)
            or not isinstance(aspect_ids, list)
            or any(aspect_id not in all_aspect_set for aspect_id in aspect_ids)
        ):
            raise EvidenceBenchDiagnosticError("evidencebench_inverse_mapping_invalid")
        for aspect_id in aspect_ids:
            derived_forward[str(aspect_id)].add(int(raw_index))
    if derived_forward != normalized_forward:
        raise EvidenceBenchDiagnosticError("evidencebench_aspect_mapping_disagreement")

    def validate_oracle(
        evaluation: Mapping[str, Any],
        *,
        limit: int,
        target_aspects: set[str],
        declared_covered: Sequence[Any],
    ) -> None:
        selection = evaluation.get("one_selection_of_sentences")
        if (
            not isinstance(selection, list)
            or len(selection) > limit
            or len(selection) != len(set(selection))
            or any(
                not isinstance(index, int) or not 0 <= index < len(sentences)
                for index in selection
            )
        ):
            raise EvidenceBenchDiagnosticError("evidencebench_oracle_selection_invalid")
        covered = {
            aspect_id
            for index in selection
            for aspect_id in inverse.get(str(index), [])
            if aspect_id in target_aspects
        }
        if set(declared_covered) != covered:
            raise EvidenceBenchDiagnosticError("evidencebench_oracle_coverage_mismatch")

    validate_oracle(
        all_eval,
        limit=10,
        target_aspects=all_aspect_set,
        declared_covered=all_covered,
    )
    validate_oracle(
        results_eval,
        limit=5,
        target_aspects=results_aspect_set,
        declared_covered=results_covered,
    )
    visible = EvidenceBenchVisibleQuestionV1(
        question_id=question_id,
        paper_id=paper_id,
        hypothesis=hypothesis,
        candidate_sentences=sentences,
        sentence_types=sentence_types,
    )
    gold = EvidenceBenchPrivateGoldV1(
        question_id=question_id,
        paper_id=paper_id,
        sentence_count=len(sentences),
        all_aspect_ids=sorted(set(all_aspects)),
        results_aspect_ids=sorted(set(results_aspects)),
        sentence_index_to_aspects={
            str(index): sorted(set(aspect_ids))
            for index, aspect_ids in inverse.items()
        },
        all_oracle_covered_aspect_ids_at_10=sorted(set(all_covered)),
        results_oracle_covered_aspect_ids_at_5=sorted(set(results_covered)),
    )
    return visible, gold


def _split_dataset(
    payload: Mapping[str, Any],
) -> tuple[list[EvidenceBenchVisibleQuestionV1], list[EvidenceBenchPrivateGoldV1]]:
    visible: list[EvidenceBenchVisibleQuestionV1] = []
    gold: list[EvidenceBenchPrivateGoldV1] = []
    for question_id in sorted(payload):
        visible_row, gold_row = _validate_dataset_row(question_id, payload[question_id])
        visible.append(visible_row)
        gold.append(gold_row)
    if len({row.question_id for row in visible}) != len(visible):
        raise EvidenceBenchDiagnosticError("evidencebench_duplicate_question_id")
    return visible, gold


def _stable_order(scores: Sequence[float]) -> list[int]:
    if any(not math.isfinite(value) for value in scores):
        raise EvidenceBenchDiagnosticError("evidencebench_nonfinite_retrieval_score")
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _tfidf_order(
    hypothesis: str,
    sentences: Sequence[str],
    *,
    analyzer: Literal["word", "char_wb"],
) -> list[int]:
    if analyzer == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            token_pattern=r"(?u)\b\w\w+\b",
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            min_df=1,
        )
    try:
        matrix = vectorizer.fit_transform([hypothesis, *sentences])
    except ValueError:
        return list(range(len(sentences)))
    similarities = (matrix[1:] @ matrix[0].T).toarray().reshape(-1)
    return _stable_order([float(value) for value in similarities])


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(value)]


def _bm25_order(hypothesis: str, sentences: Sequence[str]) -> list[int]:
    documents = [_tokens(sentence) for sentence in sentences]
    query_counts = Counter(_tokens(hypothesis))
    document_frequencies: Counter[str] = Counter()
    for document in documents:
        document_frequencies.update(set(document))
    average_length = mean([len(document) for document in documents]) or 1.0
    document_count = len(documents)
    k1 = 1.2
    b = 0.75
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        denominator_length = k1 * (1.0 - b + b * len(document) / average_length)
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * frequency
                * (k1 + 1.0)
                / (frequency + denominator_length)
            )
        scores.append(score)
    return _stable_order(scores)


def _rrf_order(rankings: Sequence[Sequence[int]], *, size: int) -> list[int]:
    scores = [0.0] * size
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (60.0 + rank)
    return _stable_order(scores)


def _random_order(question_id: str, size: int) -> list[int]:
    seed = int.from_bytes(
        hashlib.sha256(
            f"evidencebench-random-control-v1\0{question_id}".encode()
        ).digest()[:8],
        "big",
    )
    ranking = list(range(size))
    random.Random(seed).shuffle(ranking)
    return ranking


def rank_question(
    question: EvidenceBenchVisibleQuestionV1,
) -> dict[EvidenceBenchMethodId, list[int]]:
    """Return deterministic rankings from label-free question fields only."""

    size = len(question.candidate_sentences)
    word = _tfidf_order(
        question.hypothesis, question.candidate_sentences, analyzer="word"
    )
    char = _tfidf_order(
        question.hypothesis, question.candidate_sentences, analyzer="char_wb"
    )
    bm25 = _bm25_order(question.hypothesis, question.candidate_sentences)
    rankings: dict[EvidenceBenchMethodId, list[int]] = {
        "bm25-v1": bm25,
        "tfidf-word-v1": word,
        "tfidf-char-v1": char,
        "rrf-word-char-v1": _rrf_order([word, char], size=size),
        "first-sentences-control-v1": list(range(size)),
        "deterministic-random-control-v1": _random_order(question.question_id, size),
    }
    return {method_id: rankings[method_id][:10] for method_id in _METHOD_ROSTER}


def _metric_for_ranking(
    gold: EvidenceBenchPrivateGoldV1, ranking: Sequence[int]
) -> EvidenceBenchMetricV1:
    if len(ranking) > 10 or len(ranking) != len(set(ranking)):
        raise EvidenceBenchDiagnosticError("evidencebench_score_ranking_invalid")
    if any(not 0 <= index < gold.sentence_count for index in ranking):
        raise EvidenceBenchDiagnosticError("evidencebench_score_index_out_of_range")

    def coverage(indices: Sequence[int], target: set[str]) -> float:
        covered: set[str] = set()
        for index in indices:
            covered.update(gold.sentence_index_to_aspects.get(str(index), []))
        return len(covered & target) / len(target)

    return EvidenceBenchMetricV1(
        all_aspect_recall_at_10=coverage(ranking[:10], set(gold.all_aspect_ids)),
        results_aspect_recall_at_5=(
            coverage(ranking[:5], set(gold.results_aspect_ids))
            if gold.results_aspect_ids
            else None
        ),
    )


def _development_results(
    visible: Sequence[EvidenceBenchVisibleQuestionV1],
    gold: Sequence[EvidenceBenchPrivateGoldV1],
) -> list[EvidenceBenchDevelopmentResultV1]:
    gold_by_id = {row.question_id: row for row in gold}
    metrics: dict[EvidenceBenchMethodId, list[EvidenceBenchMetricV1]] = defaultdict(list)
    for question in visible:
        question_gold = gold_by_id[question.question_id]
        rankings = rank_question(question)
        for method_id in _SELECTION_METHODS:
            metrics[method_id].append(
                _metric_for_ranking(question_gold, rankings[method_id])
            )
    return [
        EvidenceBenchDevelopmentResultV1(
            method_id=method_id,
            question_count=len(values),
            results_metric_question_count=sum(
                value.results_aspect_recall_at_5 is not None for value in values
            ),
            mean=EvidenceBenchMetricV1(
                all_aspect_recall_at_10=mean(
                    value.all_aspect_recall_at_10 for value in values
                ),
                results_aspect_recall_at_5=mean(
                    value.results_aspect_recall_at_5
                    for value in values
                    if value.results_aspect_recall_at_5 is not None
                ),
            ),
        )
        for method_id, values in sorted(metrics.items())
    ]


def prepare_evidencebench_plan(
    *,
    development_path: Path,
    expected_test_sha256: str = EVIDENCEBENCH_TEST_SHA256,
    expected_test_rows: int = EVIDENCEBENCH_TEST_ROWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> EvidenceBenchFrozenPlanV1:
    """Freeze development selection without accepting or opening a test path."""

    _assert_exact_hash(development_path, EVIDENCEBENCH_DEV_SHA256, "development")
    development = _load_json_object(development_path)
    visible, gold = _split_dataset(development)
    if len(visible) != EVIDENCEBENCH_DEV_ROWS:
        raise EvidenceBenchDiagnosticError("evidencebench_development_row_count_mismatch")
    results = _development_results(visible, gold)
    selected = sorted(
        results,
        key=lambda row: (
            -row.mean.all_aspect_recall_at_10,
            -row.mean.results_aspect_recall_at_5,
            row.method_id,
        ),
    )[0].method_id
    payload: dict[str, Any] = {
        "plan_version": "evidencebench-frozen-plan-v1",
        "protocol_version": EVIDENCEBENCH_PROTOCOL_VERSION,
        "upstream_repository": "EvidenceBench/EvidenceBench",
        "upstream_commit": EVIDENCEBENCH_UPSTREAM_COMMIT,
        "development_sha256": sha256_file(development_path),
        "expected_test_sha256": expected_test_sha256,
        "expected_test_rows": expected_test_rows,
        "method_roster": sorted(_METHOD_ROSTER),
        "selectable_method_roster": sorted(_SELECTION_METHODS),
        "controls": sorted(_CONTROL_METHODS),
        "selection_rule": (
            "maximize development mean all-aspect recall@10; then results-aspect "
            "recall@5; then ascending method id"
        ),
        "primary_test_metric": "mean all-aspect recall@10",
        "secondary_test_metric": "mean results-aspect recall@5",
        "development_results": results,
        "selected_method_id": selected,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "protocol_config_sha256": _protocol_config_sha256(),
        "implementation_sha256": _implementation_sha256(),
        "runtime_versions": _runtime_versions(),
        "test_labels_opened_by_prepare": False,
        "scope_limitations": sorted(
            [
                "does_not_validate_claim_correctness",
                "does_not_validate_effect_extraction",
                "does_not_validate_meta_analysis",
                "does_not_validate_release_calibration",
                "chronology_not_externally_preregistered",
                "public_test_accessible_not_secret",
                "retrospective_test_diagnostic",
            ]
        ),
    }
    return EvidenceBenchFrozenPlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def materialize_evidencebench_test(
    *,
    plan: EvidenceBenchFrozenPlanV1,
    raw_test_path: Path,
) -> tuple[
    list[EvidenceBenchVisibleQuestionV1],
    list[EvidenceBenchPrivateGoldV1],
    EvidenceBenchMaterializationReceiptV1,
]:
    """Split the pinned test corpus only after a valid frozen plan exists."""

    plan = EvidenceBenchFrozenPlanV1.model_validate(plan.model_dump(mode="json"))
    _assert_current_plan_environment(plan)
    _assert_exact_hash(raw_test_path, plan.expected_test_sha256, "test")
    visible, gold = _split_dataset(_load_json_object(raw_test_path))
    if len(visible) != plan.expected_test_rows:
        raise EvidenceBenchDiagnosticError("evidencebench_test_row_count_mismatch")
    visible_payload = [row.model_dump(mode="json") for row in visible]
    gold_payload = [row.model_dump(mode="json") for row in gold]
    receipt_payload: dict[str, Any] = {
        "receipt_version": "evidencebench-materialization-receipt-v1",
        "plan_sha256": plan.plan_sha256,
        "raw_test_sha256": sha256_file(raw_test_path),
        "visible_projection_sha256": hash_canonical(visible_payload),
        "private_gold_sha256": hash_canonical(gold_payload),
        "question_count": len(visible),
        "unique_paper_count": len({row.paper_id for row in visible}),
        "visible_projection_contains_gold_labels": False,
        "private_gold_contains_candidate_text": False,
    }
    receipt = EvidenceBenchMaterializationReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    return visible, gold, receipt


def predict_evidencebench_test(
    *,
    plan: EvidenceBenchFrozenPlanV1,
    visible: Sequence[EvidenceBenchVisibleQuestionV1],
    materialization_receipt: EvidenceBenchMaterializationReceiptV1,
) -> tuple[list[EvidenceBenchPredictionRowV1], EvidenceBenchPredictionReceiptV1]:
    """Rank sentences using only the label-free visible projection."""

    plan = EvidenceBenchFrozenPlanV1.model_validate(plan.model_dump(mode="json"))
    _assert_current_plan_environment(plan)
    materialization_receipt = EvidenceBenchMaterializationReceiptV1.model_validate(
        materialization_receipt.model_dump(mode="json")
    )
    visible_rows = [
        EvidenceBenchVisibleQuestionV1.model_validate(row.model_dump(mode="json"))
        for row in visible
    ]
    visible_payload = [row.model_dump(mode="json") for row in visible_rows]
    if materialization_receipt.plan_sha256 != plan.plan_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_prediction_plan_mismatch")
    if (
        hash_canonical(visible_payload)
        != materialization_receipt.visible_projection_sha256
    ):
        raise EvidenceBenchDiagnosticError("evidencebench_visible_projection_mismatch")
    if len(visible_rows) != plan.expected_test_rows:
        raise EvidenceBenchDiagnosticError("evidencebench_prediction_rows_mismatch")
    predictions = [
        EvidenceBenchPredictionRowV1(
            question_id=question.question_id,
            method_rankings=rank_question(question),
        )
        for question in visible_rows
    ]
    predictions_payload = [row.model_dump(mode="json") for row in predictions]
    receipt_payload: dict[str, Any] = {
        "receipt_version": "evidencebench-prediction-receipt-v1",
        "plan_sha256": plan.plan_sha256,
        "materialization_receipt_sha256": materialization_receipt.receipt_sha256,
        "visible_projection_sha256": materialization_receipt.visible_projection_sha256,
        "predictions_sha256": hash_canonical(predictions_payload),
        "question_count": len(predictions),
        "method_roster": sorted(_METHOD_ROSTER),
        "private_gold_opened_by_prediction": False,
    }
    receipt = EvidenceBenchPredictionReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    return predictions, receipt


def validate_evidencebench_prediction_freeze(
    *,
    plan: EvidenceBenchFrozenPlanV1,
    materialization_receipt: EvidenceBenchMaterializationReceiptV1,
    predictions: Sequence[EvidenceBenchPredictionRowV1],
    prediction_receipt: EvidenceBenchPredictionReceiptV1,
) -> tuple[
    EvidenceBenchFrozenPlanV1,
    EvidenceBenchMaterializationReceiptV1,
    list[EvidenceBenchPredictionRowV1],
    EvidenceBenchPredictionReceiptV1,
]:
    """Validate the complete prediction freeze before any private gold is opened."""

    validated_plan = EvidenceBenchFrozenPlanV1.model_validate(
        plan.model_dump(mode="json")
    )
    _assert_current_plan_environment(validated_plan)
    validated_materialization = EvidenceBenchMaterializationReceiptV1.model_validate(
        materialization_receipt.model_dump(mode="json")
    )
    validated_receipt = EvidenceBenchPredictionReceiptV1.model_validate(
        prediction_receipt.model_dump(mode="json")
    )
    validated_predictions = [
        EvidenceBenchPredictionRowV1.model_validate(row.model_dump(mode="json"))
        for row in predictions
    ]
    prediction_payload = [
        row.model_dump(mode="json") for row in validated_predictions
    ]
    if validated_materialization.plan_sha256 != validated_plan.plan_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_scoring_plan_mismatch")
    if validated_receipt.plan_sha256 != validated_plan.plan_sha256:
        raise EvidenceBenchDiagnosticError(
            "evidencebench_scoring_prediction_plan_mismatch"
        )
    if (
        validated_receipt.materialization_receipt_sha256
        != validated_materialization.receipt_sha256
    ):
        raise EvidenceBenchDiagnosticError("evidencebench_scoring_receipt_mismatch")
    if validated_receipt.visible_projection_sha256 != (
        validated_materialization.visible_projection_sha256
    ):
        raise EvidenceBenchDiagnosticError(
            "evidencebench_scoring_visible_projection_mismatch"
        )
    if hash_canonical(prediction_payload) != validated_receipt.predictions_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_predictions_mismatch")
    if len(validated_predictions) != validated_plan.expected_test_rows:
        raise EvidenceBenchDiagnosticError(
            "evidencebench_scoring_prediction_row_count_mismatch"
        )
    return (
        validated_plan,
        validated_materialization,
        validated_predictions,
        validated_receipt,
    )


def _cluster_bootstrap_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> EvidenceBenchIntervalV1:
    if len(values) != len(clusters) or not values:
        raise EvidenceBenchDiagnosticError("evidencebench_bootstrap_inputs_invalid")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        by_cluster[cluster].append(value)
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(replicates):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        replicate_values = [
            value for cluster_id in selected for value in by_cluster[str(cluster_id)]
        ]
        sampled.append(mean(replicate_values))
    estimate = mean(values)
    lower, upper = np.quantile(sampled, [0.025, 0.975], method="linear")
    # Finite Monte Carlo tails can very occasionally miss the point estimate.  The
    # interval contract remains conservative by including it explicitly.
    return EvidenceBenchIntervalV1(
        estimate=estimate,
        lower_95=min(float(lower), estimate),
        upper_95=max(float(upper), estimate),
    )


def _cluster_bootstrap_signed_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> EvidenceBenchSignedIntervalV1:
    if len(values) != len(clusters) or not values:
        raise EvidenceBenchDiagnosticError(
            "evidencebench_signed_bootstrap_inputs_invalid"
        )
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        if not -1.0 <= value <= 1.0:
            raise EvidenceBenchDiagnosticError(
                "evidencebench_signed_bootstrap_value_invalid"
            )
        by_cluster[cluster].append(value)
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(replicates):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        replicate_values = [
            value for cluster_id in selected for value in by_cluster[str(cluster_id)]
        ]
        sampled.append(mean(replicate_values))
    estimate = mean(values)
    lower, upper = np.quantile(sampled, [0.025, 0.975], method="linear")
    return EvidenceBenchSignedIntervalV1(
        estimate=estimate,
        lower_95=min(float(lower), estimate),
        upper_95=max(float(upper), estimate),
    )


def score_evidencebench_test(
    *,
    plan: EvidenceBenchFrozenPlanV1,
    gold: Sequence[EvidenceBenchPrivateGoldV1],
    materialization_receipt: EvidenceBenchMaterializationReceiptV1,
    predictions: Sequence[EvidenceBenchPredictionRowV1],
    prediction_receipt: EvidenceBenchPredictionReceiptV1,
) -> EvidenceBenchPublicSummaryV1:
    """Score frozen predictions and return an aggregate-only public artifact."""

    (
        plan,
        materialization_receipt,
        prediction_rows,
        prediction_receipt,
    ) = validate_evidencebench_prediction_freeze(
        plan=plan,
        materialization_receipt=materialization_receipt,
        predictions=predictions,
        prediction_receipt=prediction_receipt,
    )
    gold_rows = [
        EvidenceBenchPrivateGoldV1.model_validate(row.model_dump(mode="json"))
        for row in gold
    ]
    gold_payload = [row.model_dump(mode="json") for row in gold_rows]
    if hash_canonical(gold_payload) != materialization_receipt.private_gold_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_private_gold_mismatch")
    if len(gold_rows) != plan.expected_test_rows or len(prediction_rows) != len(gold_rows):
        raise EvidenceBenchDiagnosticError("evidencebench_scoring_row_count_mismatch")
    gold_by_id = {row.question_id: row for row in gold_rows}
    prediction_by_id = {row.question_id: row for row in prediction_rows}
    if len(gold_by_id) != len(gold_rows) or set(gold_by_id) != set(prediction_by_id):
        raise EvidenceBenchDiagnosticError("evidencebench_scoring_question_join_mismatch")

    metrics: dict[EvidenceBenchMethodId, list[EvidenceBenchMetricV1]] = defaultdict(list)
    clusters: list[str] = []
    sentence_counts: list[int] = []
    oracle_all: list[float] = []
    oracle_results: list[float] = []
    for question_id in sorted(gold_by_id):
        question_gold = gold_by_id[question_id]
        question_prediction = prediction_by_id[question_id]
        clusters.append(question_gold.paper_id)
        sentence_counts.append(question_gold.sentence_count)
        oracle_all.append(
            len(question_gold.all_oracle_covered_aspect_ids_at_10)
            / len(question_gold.all_aspect_ids)
        )
        if question_gold.results_aspect_ids:
            oracle_results.append(
                len(question_gold.results_oracle_covered_aspect_ids_at_5)
                / len(question_gold.results_aspect_ids)
            )
        for method_id in _METHOD_ROSTER:
            metrics[method_id].append(
                _metric_for_ranking(
                    question_gold,
                    question_prediction.method_rankings[method_id],
                )
            )

    results: list[EvidenceBenchMethodTestResultV1] = []
    for method_index, method_id in enumerate(sorted(_METHOD_ROSTER)):
        method_metrics = metrics[method_id]
        results_metric_rows = [
            (value.results_aspect_recall_at_5, cluster)
            for value, cluster in zip(method_metrics, clusters, strict=True)
            if value.results_aspect_recall_at_5 is not None
        ]
        results.append(
            EvidenceBenchMethodTestResultV1(
                method_id=method_id,
                selected_on_development=method_id == plan.selected_method_id,
                control=method_id in _CONTROL_METHODS,
                results_metric_question_count=len(results_metric_rows),
                all_aspect_recall_at_10=_cluster_bootstrap_interval(
                    [value.all_aspect_recall_at_10 for value in method_metrics],
                    clusters,
                    seed=plan.bootstrap_seed + method_index * 2,
                    replicates=plan.bootstrap_replicates,
                ),
                results_aspect_recall_at_5=_cluster_bootstrap_interval(
                    [float(value) for value, _cluster in results_metric_rows],
                    [cluster for _value, cluster in results_metric_rows],
                    seed=plan.bootstrap_seed + method_index * 2 + 1,
                    replicates=plan.bootstrap_replicates,
                ),
            )
        )
    paired_deltas: list[EvidenceBenchPairedDeltaV1] = []
    selected_metrics = metrics[plan.selected_method_id]
    for comparator_index, comparator_method_id in enumerate(
        sorted(set(_METHOD_ROSTER) - {plan.selected_method_id})
    ):
        comparator_metrics = metrics[comparator_method_id]
        all_deltas = [
            selected.all_aspect_recall_at_10 - comparator.all_aspect_recall_at_10
            for selected, comparator in zip(
                selected_metrics, comparator_metrics, strict=True
            )
        ]
        results_delta_rows: list[tuple[float, str]] = []
        for selected, comparator, cluster in zip(
            selected_metrics, comparator_metrics, clusters, strict=True
        ):
            if (selected.results_aspect_recall_at_5 is None) != (
                comparator.results_aspect_recall_at_5 is None
            ):
                raise EvidenceBenchDiagnosticError(
                    "evidencebench_paired_results_eligibility_mismatch"
                )
            if selected.results_aspect_recall_at_5 is not None:
                results_delta_rows.append(
                    (
                        selected.results_aspect_recall_at_5
                        - float(comparator.results_aspect_recall_at_5),
                        cluster,
                    )
                )
        paired_deltas.append(
            EvidenceBenchPairedDeltaV1(
                selected_method_id=plan.selected_method_id,
                comparator_method_id=comparator_method_id,
                comparator_is_control=comparator_method_id in _CONTROL_METHODS,
                question_count=len(all_deltas),
                results_metric_question_count=len(results_delta_rows),
                all_aspect_recall_at_10_delta=_cluster_bootstrap_signed_interval(
                    all_deltas,
                    clusters,
                    seed=plan.bootstrap_seed + 100 + comparator_index * 2,
                    replicates=plan.bootstrap_replicates,
                ),
                results_aspect_recall_at_5_delta=(
                    _cluster_bootstrap_signed_interval(
                        [value for value, _cluster in results_delta_rows],
                        [cluster for _value, cluster in results_delta_rows],
                        seed=plan.bootstrap_seed + 100 + comparator_index * 2 + 1,
                        replicates=plan.bootstrap_replicates,
                    )
                ),
            )
        )
    summary_payload: dict[str, Any] = {
        "summary_version": "evidencebench-public-summary-v1",
        "protocol_version": EVIDENCEBENCH_PROTOCOL_VERSION,
        "evidence_status": "retrospective_public_test_diagnostic",
        "upstream_repository": "EvidenceBench/EvidenceBench",
        "upstream_commit": EVIDENCEBENCH_UPSTREAM_COMMIT,
        "test_license": "CC-BY",
        "plan_sha256": plan.plan_sha256,
        "materialization_receipt_sha256": materialization_receipt.receipt_sha256,
        "prediction_receipt_sha256": prediction_receipt.receipt_sha256,
        "raw_test_sha256": materialization_receipt.raw_test_sha256,
        "private_gold_sha256": materialization_receipt.private_gold_sha256,
        "predictions_sha256": prediction_receipt.predictions_sha256,
        "protocol_config_sha256": plan.protocol_config_sha256,
        "implementation_sha256": plan.implementation_sha256,
        "selected_method_id": plan.selected_method_id,
        "question_count": len(gold_rows),
        "results_metric_question_count": len(oracle_results),
        "unique_paper_count": len(set(clusters)),
        "candidate_sentence_count": sum(sentence_counts),
        "median_candidate_sentences_per_question": float(median(sentence_counts)),
        "bootstrap_seed": plan.bootstrap_seed,
        "bootstrap_replicates": plan.bootstrap_replicates,
        "test_results": results,
        "selected_method_paired_deltas": paired_deltas,
        "oracle_upper_bounds": {
            "all_aspect_recall_at_10": mean(oracle_all),
            "results_aspect_recall_at_5": mean(oracle_results),
        },
        "output_is_aggregate_only": True,
        "licenses_scientific_scope": sorted(
            [
                "does_not_validate_claim_correctness",
                "does_not_validate_effect_extraction",
                "does_not_validate_meta_analysis",
                "does_not_validate_release_calibration",
                "chronology_not_externally_preregistered",
                "public_test_accessible_not_secret",
                "retrospective_public_test_diagnostic",
            ]
        ),
    }
    return EvidenceBenchPublicSummaryV1.model_validate(
        {**summary_payload, "summary_sha256": hash_canonical(summary_payload)}
    )


def _assert_exact_replay(name: str, observed: Any, expected: Any) -> None:
    def canonical(value: Any) -> Any:
        if isinstance(value, ContractModel):
            return value.model_dump(mode="json")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [canonical(item) for item in value]
        return value

    if canonical(observed) != canonical(expected):
        raise EvidenceBenchDiagnosticError(
            f"evidencebench_audit_replay_mismatch:{name}"
        )


def audit_evidencebench_run(
    *,
    development_path: Path,
    raw_test_path: Path,
    plan: EvidenceBenchFrozenPlanV1,
    visible: Sequence[EvidenceBenchVisibleQuestionV1],
    gold: Sequence[EvidenceBenchPrivateGoldV1],
    materialization_receipt: EvidenceBenchMaterializationReceiptV1,
    predictions: Sequence[EvidenceBenchPredictionRowV1],
    prediction_receipt: EvidenceBenchPredictionReceiptV1,
    summary: EvidenceBenchPublicSummaryV1,
) -> EvidenceBenchReplayAuditReceiptV1:
    """Replay every stage exactly and issue a content-silent audit receipt."""

    validated_plan, validated_materialization, validated_predictions, validated_prediction = (
        validate_evidencebench_prediction_freeze(
            plan=plan,
            materialization_receipt=materialization_receipt,
            predictions=predictions,
            prediction_receipt=prediction_receipt,
        )
    )
    validated_summary = EvidenceBenchPublicSummaryV1.model_validate(
        summary.model_dump(mode="json")
    )
    replayed_plan = prepare_evidencebench_plan(
        development_path=development_path,
        expected_test_sha256=validated_plan.expected_test_sha256,
        expected_test_rows=validated_plan.expected_test_rows,
        bootstrap_seed=validated_plan.bootstrap_seed,
        bootstrap_replicates=validated_plan.bootstrap_replicates,
    )
    _assert_exact_replay("plan", replayed_plan, validated_plan)
    replayed_visible, replayed_gold, replayed_materialization = (
        materialize_evidencebench_test(
            plan=validated_plan,
            raw_test_path=raw_test_path,
        )
    )
    _assert_exact_replay("visible", replayed_visible, visible)
    _assert_exact_replay("gold", replayed_gold, gold)
    _assert_exact_replay(
        "materialization_receipt",
        replayed_materialization,
        validated_materialization,
    )
    replayed_predictions, replayed_prediction = predict_evidencebench_test(
        plan=validated_plan,
        visible=replayed_visible,
        materialization_receipt=replayed_materialization,
    )
    _assert_exact_replay("predictions", replayed_predictions, validated_predictions)
    _assert_exact_replay(
        "prediction_receipt", replayed_prediction, validated_prediction
    )
    replayed_summary = score_evidencebench_test(
        plan=validated_plan,
        gold=replayed_gold,
        materialization_receipt=replayed_materialization,
        predictions=replayed_predictions,
        prediction_receipt=replayed_prediction,
    )
    _assert_exact_replay("summary", replayed_summary, validated_summary)
    payload: dict[str, Any] = {
        "receipt_version": "evidencebench-replay-audit-receipt-v1",
        "plan_sha256": validated_plan.plan_sha256,
        "materialization_receipt_sha256": validated_materialization.receipt_sha256,
        "prediction_receipt_sha256": validated_prediction.receipt_sha256,
        "summary_sha256": validated_summary.summary_sha256,
        "protocol_config_sha256": validated_plan.protocol_config_sha256,
        "implementation_sha256": validated_plan.implementation_sha256,
        "runtime_versions": validated_plan.runtime_versions,
        "exact_replay_status": "passed",
        "test_gold_opened_only_after_prediction_freeze_validation": True,
        "public_artifacts_contain_private_rows": False,
    }
    return EvidenceBenchReplayAuditReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_evidencebench_public_bundle(
    *,
    summary: Mapping[str, Any] | EvidenceBenchPublicSummaryV1,
    audit_receipt: Mapping[str, Any] | EvidenceBenchReplayAuditReceiptV1,
    require_current_environment: bool = True,
) -> tuple[EvidenceBenchPublicSummaryV1, EvidenceBenchReplayAuditReceiptV1]:
    """Validate the two aggregate-only public artifacts and all cross-bindings."""

    validated_summary = (
        EvidenceBenchPublicSummaryV1.model_validate(summary)
        if isinstance(summary, Mapping)
        else EvidenceBenchPublicSummaryV1.model_validate(
            summary.model_dump(mode="json")
        )
    )
    validated_audit = (
        EvidenceBenchReplayAuditReceiptV1.model_validate(audit_receipt)
        if isinstance(audit_receipt, Mapping)
        else EvidenceBenchReplayAuditReceiptV1.model_validate(
            audit_receipt.model_dump(mode="json")
        )
    )
    expected_bindings = {
        "plan_sha256": validated_summary.plan_sha256,
        "materialization_receipt_sha256": (
            validated_summary.materialization_receipt_sha256
        ),
        "prediction_receipt_sha256": validated_summary.prediction_receipt_sha256,
        "summary_sha256": validated_summary.summary_sha256,
        "protocol_config_sha256": validated_summary.protocol_config_sha256,
        "implementation_sha256": validated_summary.implementation_sha256,
    }
    for field, expected in expected_bindings.items():
        if getattr(validated_audit, field) != expected:
            raise EvidenceBenchDiagnosticError(
                f"evidencebench_public_bundle_binding_mismatch:{field}"
            )
    if require_current_environment:
        if validated_summary.protocol_config_sha256 != _protocol_config_sha256():
            raise EvidenceBenchDiagnosticError(
                "evidencebench_public_bundle_protocol_config_drift"
            )
        if validated_summary.implementation_sha256 != _implementation_sha256():
            raise EvidenceBenchDiagnosticError(
                "evidencebench_public_bundle_implementation_drift"
            )
        if validated_audit.runtime_versions != _runtime_versions():
            raise EvidenceBenchDiagnosticError(
                "evidencebench_public_bundle_runtime_drift"
            )
    return validated_summary, validated_audit


def write_evidencebench_plan(
    path: Path, plan: EvidenceBenchFrozenPlanV1
) -> None:
    atomic_write_json(path, plan.model_dump(mode="json"))


def _preflight_output_paths(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise EvidenceBenchDiagnosticError("evidencebench_output_paths_alias")
    existing = [path.as_posix() for path in paths if path.exists()]
    if existing:
        raise OutputExistsError(",".join(sorted(existing)))


def write_evidencebench_materialization(
    *,
    visible_path: Path,
    gold_path: Path,
    receipt_path: Path,
    visible: Sequence[EvidenceBenchVisibleQuestionV1],
    gold: Sequence[EvidenceBenchPrivateGoldV1],
    receipt: EvidenceBenchMaterializationReceiptV1,
) -> None:
    """Write all custody outputs only after computing the complete valid bundle."""

    visible_payload = [row.model_dump(mode="json") for row in visible]
    gold_payload = [row.model_dump(mode="json") for row in gold]
    if hash_canonical(visible_payload) != receipt.visible_projection_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_visible_write_hash_mismatch")
    if hash_canonical(gold_payload) != receipt.private_gold_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_gold_write_hash_mismatch")
    _preflight_output_paths([visible_path, gold_path, receipt_path])
    for path, payload in (
        (visible_path, visible_payload),
        (gold_path, gold_payload),
        (receipt_path, receipt.model_dump(mode="json")),
    ):
        atomic_write_json(path, payload)


def write_evidencebench_predictions(
    *,
    predictions_path: Path,
    receipt_path: Path,
    predictions: Sequence[EvidenceBenchPredictionRowV1],
    receipt: EvidenceBenchPredictionReceiptV1,
) -> None:
    prediction_payload = [row.model_dump(mode="json") for row in predictions]
    if hash_canonical(prediction_payload) != receipt.predictions_sha256:
        raise EvidenceBenchDiagnosticError("evidencebench_prediction_write_hash_mismatch")
    _preflight_output_paths([predictions_path, receipt_path])
    atomic_write_json(predictions_path, prediction_payload)
    atomic_write_json(receipt_path, receipt.model_dump(mode="json"))


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EVIDENCEBENCH_DEV_ROWS",
    "EVIDENCEBENCH_DEV_SHA256",
    "EVIDENCEBENCH_PROTOCOL_VERSION",
    "EVIDENCEBENCH_TEST_ROWS",
    "EVIDENCEBENCH_TEST_SHA256",
    "EVIDENCEBENCH_TRAIN_SHA256",
    "EVIDENCEBENCH_UPSTREAM_COMMIT",
    "EvidenceBenchDiagnosticError",
    "EvidenceBenchFrozenPlanV1",
    "EvidenceBenchMaterializationReceiptV1",
    "EvidenceBenchPairedDeltaV1",
    "EvidenceBenchPredictionReceiptV1",
    "EvidenceBenchPredictionRowV1",
    "EvidenceBenchPrivateGoldV1",
    "EvidenceBenchPublicSummaryV1",
    "EvidenceBenchReplayAuditReceiptV1",
    "EvidenceBenchSignedIntervalV1",
    "EvidenceBenchVisibleQuestionV1",
    "audit_evidencebench_run",
    "materialize_evidencebench_test",
    "predict_evidencebench_test",
    "prepare_evidencebench_plan",
    "rank_question",
    "score_evidencebench_test",
    "validate_evidencebench_prediction_freeze",
    "validate_evidencebench_public_bundle",
    "write_evidencebench_materialization",
    "write_evidencebench_plan",
    "write_evidencebench_predictions",
]
