"""Leakage-explicit local-Ollama GEPA study for Evidence Inference 2.0.

The module deliberately separates optimization and test evaluation into different
entry points and artifacts.  Optimization may open only the paper-disjoint train and
development payloads.  A hash-bound winner is frozen before the paired test entry point
is allowed to open the test JSONL.  Model calls and all row-level material are stored
under an ignored private run directory; the trackable summary contains aggregates and
hashes only.

This remains a non-pristine diagnostic: every benchmark split in this checkout has been
opened historically, and the pinned 1.2B local model is not a frontier-model evaluation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from literature_multiverse.evidence_inference_ollama import (
    DEFAULT_RETRIEVAL_CONFIG,
    RETRIEVAL_ALGORITHM,
    project_results_passages,
)
from literature_multiverse.grounding import GroundingContractError, ground_evidence
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaClientProtocol,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.prompt_optimization import (
    OptimizationEvaluationBatch,
    OptimizationExample,
    OptimizationSplitManifest,
    load_manifest_split,
    load_split_manifest,
)
from literature_multiverse.prompting import PromptContractError, render_prompt_text

STUDY_VERSION = "evidence-inference-local-ollama-gepa-study-v1"
PLAN_VERSION = "evidence-inference-local-ollama-gepa-plan-v1"
RECEIPT_VERSION = "evidence-inference-local-ollama-gepa-receipt-v1"
TRACE_VERSION = "evidence-inference-local-ollama-gepa-trace-v1"
WINNER_VERSION = "evidence-inference-local-ollama-gepa-winner-v1"
PAIRED_REPORT_VERSION = "evidence-inference-local-ollama-gepa-paired-report-v1"
PUBLIC_SUMMARY_VERSION = "evidence-inference-local-ollama-gepa-public-summary-v1"
OFFICIAL_GEPA_VERSION = "0.1.4"
COMPONENT = "extraction_prompt"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_PROMPT_TOKEN = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")
_REQUIRED_TOKENS = frozenset({"INTERVENTION", "COMPARATOR", "OUTCOME"})
GENERATION_SCHEMA_ALGORITHM = "row-line-id-enumerated-no-regex-json-schema-v1"
_OBJECTIVES = (
    "direction_accuracy",
    "direction_distribution_fidelity",
    "formal_grounding_validity",
    "structured_output_validity",
    "generation_success",
    "token_efficiency",
    "latency_sla_success",
)
_PAIRED_METRICS = (
    *(item for item in _OBJECTIVES if item != "direction_distribution_fidelity"),
    "direction_macro_recall",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "article_id",
        "candidate",
        "candidates",
        "content_lines",
        "example_id",
        "expected_output",
        "group_id",
        "label_paths",
        "paper_id",
        "parsed_output",
        "prompt",
        "prompts",
        "query",
        "raw_prediction",
        "response_text",
        "test_example_ids",
        "trajectory",
        "trajectories",
    }
)
_FORBIDDEN_PUBLIC_VALUES = (
    re.compile(r"\bei2-prompt-[0-9]+\b", re.I),
    re.compile(r"\bPMC[0-9]+\b", re.I),
    re.compile(r"(?:^|\s)/(?:Users|home|private|tmp|var)/"),
)
_OLLAMA_TASK_SCHEMA_BASE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "eligible": {"type": "boolean"},
        "findings": {
            "type": "array",
            "minItems": 0,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["increase", "no_effect", "decrease"],
                    },
                    "evidence_quote": {"type": "string", "minLength": 1},
                    "evidence_lines": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["direction", "evidence_quote", "evidence_lines"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["eligible", "findings"],
    "additionalProperties": False,
}
OLLAMA_TASK_SCHEMA_BASE_SHA256 = hash_canonical(_OLLAMA_TASK_SCHEMA_BASE)


class OllamaGEPAStudyError(ValueError):
    """A study stage, lineage artifact, model call, or public summary is invalid."""


class _NoResultsPassageError(OllamaGEPAStudyError):
    """The frozen label-blind projection exposed no task evidence."""


class OptimizationSettings(BaseModel):
    """Frozen official-GEPA search settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subset_seed: int
    train_target_rows: int = Field(ge=24)
    dev_target_rows: int = Field(ge=24)
    max_metric_calls: int = Field(ge=100)
    reflection_minibatch_size: int = Field(ge=2)
    candidate_selection_strategy: Literal["pareto"] = "pareto"
    frontier_type: Literal["hybrid"] = "hybrid"
    batch_sampler: Literal["epoch_shuffled"] = "epoch_shuffled"
    module_selector: Literal["round_robin"] = "round_robin"
    acceptance_criterion: Literal["improvement_or_equal"] = "improvement_or_equal"
    skip_perfect_score: Literal[False] = False
    use_merge: Literal[False] = False
    cache_evaluation: Literal[False] = False
    track_best_outputs: Literal[False] = False
    gepa_seed: int
    minimum_reflection_proposals: int = Field(ge=2)


class MetricSettings(BaseModel):
    """Prespecified scalar metric and paired uncertainty settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_accuracy_weight: float = Field(ge=0.0, le=1.0)
    direction_distribution_fidelity_weight: float = Field(ge=0.0, le=1.0)
    formal_grounding_validity_weight: float = Field(ge=0.0, le=1.0)
    structured_output_validity_weight: float = Field(ge=0.0, le=1.0)
    generation_success_weight: float = Field(ge=0.0, le=1.0)
    token_efficiency_weight: float = Field(ge=0.0, le=1.0)
    latency_sla_success_weight: float = Field(ge=0.0, le=1.0)
    latency_sla_seconds: float = Field(gt=0.0)
    bootstrap_seed: int
    bootstrap_replicates: int = Field(ge=1000)

    @model_validator(mode="after")
    def validate_weights(self) -> MetricSettings:
        weights = self.objective_weights
        if not all(math.isfinite(value) for value in weights.values()):
            raise ValueError("metric weights must be finite")
        if not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("metric weights must sum to one")
        if not math.isfinite(self.latency_sla_seconds):
            raise ValueError("latency SLA must be finite")
        return self

    @property
    def objective_weights(self) -> dict[str, float]:
        return {
            "direction_accuracy": self.direction_accuracy_weight,
            "direction_distribution_fidelity": self.direction_distribution_fidelity_weight,
            "formal_grounding_validity": self.formal_grounding_validity_weight,
            "structured_output_validity": self.structured_output_validity_weight,
            "generation_success": self.generation_success_weight,
            "token_efficiency": self.token_efficiency_weight,
            "latency_sla_success": self.latency_sla_success_weight,
        }


class OllamaGEPAStudyConfig(BaseModel):
    """One completely frozen local study specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_version: Literal[STUDY_VERSION] = STUDY_VERSION
    manifest_path: str
    seed_prompt_path: str
    private_run_dir: str
    public_summary_path: str
    generation: OllamaGenerationConfig
    optimization: OptimizationSettings
    metrics: MetricSettings

    @field_validator("manifest_path", "seed_prompt_path", "private_run_dir", "public_summary_path")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        if not value or "\0" in value or "\r" in value or "\n" in value:
            raise ValueError("study paths must be nonempty single-line text")
        return value


@dataclass(frozen=True, slots=True)
class StudyPaths:
    config: Path
    manifest: Path
    seed_prompt: Path
    private_run_dir: Path
    public_summary: Path
    plan: Path
    gepa_run_dir: Path
    trace: Path
    winner_prompt: Path
    winner: Path
    paired_report: Path
    receipts: Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repository_root() / path


def load_study_config(path: str | Path) -> OllamaGEPAStudyConfig:
    source = Path(path)
    try:
        return OllamaGEPAStudyConfig.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OllamaGEPAStudyError(f"invalid Ollama-GEPA study configuration: {source}") from exc


def study_paths(config_path: str | Path, config: OllamaGEPAStudyConfig | None = None) -> StudyPaths:
    source = Path(config_path).resolve()
    loaded = config or load_study_config(source)
    private = _resolve_configured_path(loaded.private_run_dir)
    return StudyPaths(
        config=source,
        manifest=_resolve_configured_path(loaded.manifest_path),
        seed_prompt=_resolve_configured_path(loaded.seed_prompt_path),
        private_run_dir=private,
        public_summary=_resolve_configured_path(loaded.public_summary_path),
        plan=private / "optimization-plan.json",
        gepa_run_dir=private / "gepa-run",
        trace=private / "gepa-result.json",
        winner_prompt=private / "frozen-winner.md",
        winner=private / "frozen-winner.json",
        paired_report=private / "paired-test-report.json",
        receipts=private / "receipts",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OllamaGEPAStudyError(f"cannot read study JSON object: {path}") from exc
    if not isinstance(value, Mapping):
        raise OllamaGEPAStudyError(f"study JSON root must be an object: {path}")
    return dict(value)


def _add_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    if field in result:
        raise OllamaGEPAStudyError(f"self-hash field already exists: {field}")
    result[field] = hash_canonical(result)
    return result


def _validate_self_hash(payload: Mapping[str, Any], field: str, artifact: str) -> None:
    snapshot = deepcopy(dict(payload))
    observed = snapshot.pop(field, None)
    if not isinstance(observed, str) or hash_canonical(snapshot) != observed:
        raise OllamaGEPAStudyError(f"{artifact} self-hash mismatch")


def _source_code_hashes() -> dict[str, str]:
    root = _repository_root()
    labels = (
        "pyproject.toml",
        "scripts/run_ollama_gepa_study.py",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/evidence_inference.py",
        "src/literature_multiverse/evidence_inference_diagnostic.py",
        "src/literature_multiverse/evidence_inference_ollama.py",
        "src/literature_multiverse/grounding.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/local_ollama.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/ollama_gepa_study.py",
        "src/literature_multiverse/paths.py",
        "src/literature_multiverse/prompt_optimization.py",
        "src/literature_multiverse/prompting.py",
        "src/literature_multiverse/providers.py",
        "uv.lock",
    )
    missing = [label for label in labels if not (root / label).is_file()]
    if missing:
        raise OllamaGEPAStudyError(f"study source files are missing: {missing}")
    return {label: sha256_file(root / label) for label in labels}


def _python_runtime_identity() -> dict[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def _official_gepa_version() -> str:
    try:
        return importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError as exc:
        raise OllamaGEPAStudyError(
            "official GEPA is unavailable; install the project GEPA extra"
        ) from exc


def _load_official_gepa_optimize() -> Callable[..., Any]:
    if _official_gepa_version() != OFFICIAL_GEPA_VERSION:
        raise OllamaGEPAStudyError(f"study requires official GEPA {OFFICIAL_GEPA_VERSION} exactly")
    try:
        module = importlib.import_module("gepa")
    except ImportError as exc:
        raise OllamaGEPAStudyError("official GEPA cannot be imported") from exc
    optimize = getattr(module, "optimize", None)
    if not callable(optimize):
        raise OllamaGEPAStudyError("installed GEPA does not expose gepa.optimize")
    return optimize


def _runtime_snapshot(start_wall_ns: int, start_cpu_ns: int) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw_rss = int(usage.ru_maxrss)
    peak_rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    return {
        "wall_time_seconds": (time.perf_counter_ns() - start_wall_ns) / 1_000_000_000.0,
        "process_cpu_seconds": (time.process_time_ns() - start_cpu_ns) / 1_000_000_000.0,
        "process_peak_rss_bytes": peak_rss_bytes,
        "scope": "current_stage_process_final_invocation",
    }


def _validate_seed_prompt(text: str) -> None:
    if set(_PROMPT_TOKEN.findall(text)) != _REQUIRED_TOKENS:
        raise OllamaGEPAStudyError("seed prompt must contain exactly the three PICO tokens")
    try:
        render_prompt_text(
            text,
            {"INTERVENTION": "I", "COMPARATOR": "C", "OUTCOME": "O"},
        )
    except PromptContractError as exc:
        raise OllamaGEPAStudyError("seed prompt violates the repository prompt contract") from exc


def _paper_subset(
    examples: Sequence[OptimizationExample],
    *,
    split_name: str,
    target_rows: int,
    seed: int,
) -> list[OptimizationExample]:
    by_paper: dict[str, list[OptimizationExample]] = defaultdict(list)
    for example in examples:
        by_paper[example.paper_id].append(example)
    ranked_papers = sorted(
        by_paper,
        key=lambda paper_id: (
            hashlib.sha256(f"{seed}:{split_name}:{paper_id}".encode()).hexdigest(),
            paper_id,
        ),
    )
    selected: list[OptimizationExample] = []
    for paper_id in ranked_papers:
        selected.extend(sorted(by_paper[paper_id], key=lambda item: item.example_id))
        if len(selected) >= target_rows:
            break
    if len(selected) < target_rows:
        raise OllamaGEPAStudyError(f"{split_name} has fewer examples than the frozen target")
    return selected


def _selection_metadata(examples: Sequence[OptimizationExample]) -> dict[str, Any]:
    serialized = [example.model_dump(mode="json") for example in examples]
    projections = [
        project_results_passages(example.model_dump(mode="json")) for example in examples
    ]
    return {
        "example_ids": [example.example_id for example in examples],
        "paper_ids": sorted({example.paper_id for example in examples}),
        "group_ids": sorted({example.group_id for example in examples}),
        "examples": len(examples),
        "papers": len({example.paper_id for example in examples}),
        "groups": len({example.group_id for example in examples}),
        "selected_payload_sha256": hash_canonical(serialized),
        "projected_task_inputs_sha256": hash_canonical(projections),
        "projected_passages": sum(len(passages) for passages in projections),
        "examples_without_projected_results_passages": sum(
            not passages for passages in projections
        ),
        "maximum_projected_characters": max(
            (sum(len(str(passage["text"])) for passage in passages) for passages in projections),
            default=0,
        ),
        "selection_identity_sha256": hash_canonical(
            [
                {
                    "example_id": example.example_id,
                    "paper_id": example.paper_id,
                    "group_id": example.group_id,
                }
                for example in examples
            ]
        ),
    }


def _load_selected_optimization_examples(
    paths: StudyPaths,
    config: OllamaGEPAStudyConfig,
) -> tuple[OptimizationSplitManifest, list[OptimizationExample], list[OptimizationExample]]:
    manifest = load_split_manifest(paths.manifest)
    # These two split-scoped calls are the only payload access in this function.  In
    # particular, the test artifact is not opened or hashed here.
    train = load_manifest_split(paths.manifest, "train")
    dev = load_manifest_split(paths.manifest, "dev")
    selected_train = _paper_subset(
        train,
        split_name="train",
        target_rows=config.optimization.train_target_rows,
        seed=config.optimization.subset_seed,
    )
    selected_dev = _paper_subset(
        dev,
        split_name="dev",
        target_rows=config.optimization.dev_target_rows,
        seed=config.optimization.subset_seed,
    )
    return manifest, selected_train, selected_dev


def prepare_optimization_plan(
    *,
    config_path: str | Path,
    client: OllamaClientProtocol,
) -> Path:
    """Freeze label-blind paper selections and runtime identity without opening test JSONL."""

    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    if paths.plan.exists():
        validate_optimization_plan(config_path=config_path, client=client)
        return paths.plan
    if paths.winner.exists() or paths.paired_report.exists() or paths.public_summary.exists():
        raise OllamaGEPAStudyError("later-stage artifacts exist without an optimization plan")
    try:
        seed_prompt = paths.seed_prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OllamaGEPAStudyError("seed prompt cannot be read") from exc
    _validate_seed_prompt(seed_prompt)
    manifest, train, dev = _load_selected_optimization_examples(paths, config)
    identity = client.inspect_identity(config.generation)
    if (
        identity.model != config.generation.model
        or identity.model_digest != config.generation.model_digest
    ):
        raise OllamaGEPAStudyError("observed local model identity differs from configuration")
    gepa_version = _official_gepa_version()
    if gepa_version != OFFICIAL_GEPA_VERSION:
        raise OllamaGEPAStudyError(f"study requires official GEPA {OFFICIAL_GEPA_VERSION} exactly")
    payload = {
        "plan_version": PLAN_VERSION,
        "study_version": STUDY_VERSION,
        "stage": "optimization_inputs_frozen_before_test_payload_access",
        "config_file_sha256": sha256_file(paths.config),
        "manifest_file_sha256": sha256_file(paths.manifest),
        "manifest_declared_source_examples_sha256": manifest.source_examples_sha256,
        "manifest_algorithm": manifest.algorithm,
        "train_split_jsonl_sha256": manifest.train.sha256,
        "dev_split_jsonl_sha256": manifest.dev.sha256,
        "test_split_declared_sha256_not_recomputed": manifest.test.sha256,
        "test_membership_metadata_present_in_manifest": True,
        "test_payload_opened": False,
        "test_labels_scored": False,
        "all_labels_historically_opened": True,
        "test_is_non_pristine": True,
        "selection_policy": {
            "algorithm": "sha256_ranked_whole_papers_until_row_target-v1",
            "uses_labels": False,
            "subset_seed": config.optimization.subset_seed,
            "train_target_rows": config.optimization.train_target_rows,
            "dev_target_rows": config.optimization.dev_target_rows,
        },
        "train_selection": _selection_metadata(train),
        "dev_selection": _selection_metadata(dev),
        "seed_prompt_sha256": hashlib.sha256(seed_prompt.encode()).hexdigest(),
        "generation_config": config.generation.model_dump(mode="json"),
        "generation_config_sha256": config.generation.config_sha256,
        "observed_ollama_identity": identity.model_dump(mode="json"),
        "observed_ollama_identity_sha256": identity.identity_sha256,
        "optimizer": {
            "implementation": "official_gepa.optimize",
            "gepa_version": gepa_version,
            **config.optimization.model_dump(mode="json"),
        },
        "python_runtime_identity": _python_runtime_identity(),
        "objective_weights": config.metrics.objective_weights,
        "metric_settings": config.metrics.model_dump(mode="json"),
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_schema_base_sha256": OLLAMA_TASK_SCHEMA_BASE_SHA256,
        "evaluation_schema_sha256": hash_canonical(train[0].output_schema),
        "passage_projection_algorithm": RETRIEVAL_ALGORITHM,
        "passage_projection_config": deepcopy(DEFAULT_RETRIEVAL_CONFIG),
        "passage_projection_config_sha256": hash_canonical(DEFAULT_RETRIEVAL_CONFIG),
        "source_code_sha256s": _source_code_hashes(),
    }
    plan = _add_self_hash(payload, "plan_sha256")
    paths.private_run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.plan, plan)
    return paths.plan


def validate_optimization_plan(
    *,
    config_path: str | Path,
    client: OllamaClientProtocol,
) -> dict[str, Any]:
    """Validate every optimization input and current implementation, still without test JSONL."""

    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    plan = _read_json_object(paths.plan)
    _validate_self_hash(plan, "plan_sha256", "optimization plan")
    if (
        plan.get("plan_version") != PLAN_VERSION
        or plan.get("study_version") != STUDY_VERSION
        or plan.get("test_payload_opened") is not False
        or plan.get("test_labels_scored") is not False
        or plan.get("all_labels_historically_opened") is not True
        or plan.get("test_is_non_pristine") is not True
        or plan.get("generation_schema_algorithm") != GENERATION_SCHEMA_ALGORITHM
        or plan.get("generation_schema_base_sha256") != OLLAMA_TASK_SCHEMA_BASE_SHA256
        or plan.get("passage_projection_algorithm") != RETRIEVAL_ALGORITHM
        or plan.get("passage_projection_config") != DEFAULT_RETRIEVAL_CONFIG
        or plan.get("passage_projection_config_sha256") != hash_canonical(DEFAULT_RETRIEVAL_CONFIG)
        or plan.get("python_runtime_identity") != _python_runtime_identity()
    ):
        raise OllamaGEPAStudyError("optimization plan stage/scope is invalid")
    if plan.get("config_file_sha256") != sha256_file(paths.config):
        raise OllamaGEPAStudyError("study configuration changed after plan freeze")
    if plan.get("manifest_file_sha256") != sha256_file(paths.manifest):
        raise OllamaGEPAStudyError("split manifest changed after plan freeze")
    try:
        seed = paths.seed_prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OllamaGEPAStudyError("seed prompt cannot be read") from exc
    _validate_seed_prompt(seed)
    if plan.get("seed_prompt_sha256") != hashlib.sha256(seed.encode()).hexdigest():
        raise OllamaGEPAStudyError("seed prompt changed after plan freeze")
    manifest, train, dev = _load_selected_optimization_examples(paths, config)
    if (
        plan.get("train_split_jsonl_sha256") != manifest.train.sha256
        or plan.get("dev_split_jsonl_sha256") != manifest.dev.sha256
        or plan.get("test_split_declared_sha256_not_recomputed") != manifest.test.sha256
        or plan.get("train_selection") != _selection_metadata(train)
        or plan.get("dev_selection") != _selection_metadata(dev)
        or plan.get("evaluation_schema_sha256") != hash_canonical(train[0].output_schema)
        or any(
            hash_canonical(example.output_schema) != plan.get("evaluation_schema_sha256")
            for example in [*train, *dev]
        )
    ):
        raise OllamaGEPAStudyError("optimization selections no longer reproduce")
    if plan.get("source_code_sha256s") != _source_code_hashes():
        raise OllamaGEPAStudyError("study implementation changed after plan freeze")
    if plan.get("generation_config_sha256") != config.generation.config_sha256:
        raise OllamaGEPAStudyError("generation configuration hash mismatch")
    if plan.get("optimizer", {}).get("gepa_version") != _official_gepa_version():
        raise OllamaGEPAStudyError("official GEPA version changed after plan freeze")
    identity = client.inspect_identity(config.generation)
    if (
        plan.get("observed_ollama_identity") != identity.model_dump(mode="json")
        or plan.get("observed_ollama_identity_sha256") != identity.identity_sha256
    ):
        raise OllamaGEPAStudyError("local Ollama identity changed after plan freeze")
    return plan


def _candidate_sha256(candidate: Mapping[str, str]) -> str:
    if set(candidate) != {COMPONENT} or not isinstance(candidate.get(COMPONENT), str):
        raise OllamaGEPAStudyError("candidate must contain only extraction_prompt")
    return hash_canonical(dict(candidate))


def _task_prompt(
    example: OptimizationExample, template: str
) -> tuple[str, str, list[dict[str, Any]]]:
    try:
        rendered, prompt_version = render_prompt_text(template, example.replacements)
    except PromptContractError as exc:
        raise OllamaGEPAStudyError(f"candidate prompt contract failed: {exc}") from exc
    passages = project_results_passages(example.model_dump(mode="json"))
    if not passages:
        raise _NoResultsPassageError("no Results passage survived the frozen projection")
    rendered_passages: list[str] = []
    for passage in passages:
        line_id = passage.get("line_id")
        text = passage.get("text")
        if not isinstance(line_id, str) or not isinstance(text, str):
            raise OllamaGEPAStudyError("label-blind Results projection returned an invalid passage")
        rendered_passages.append(
            f"LINE_ID: {line_id}\nBEGIN_EXACT_SOURCE_TEXT\n{text}\nEND_EXACT_SOURCE_TEXT"
        )
    supplied = (
        "\n\n".join(rendered_passages)
        if rendered_passages
        else "[NO_RESULTS_PASSAGE] No Results excerpt survived the frozen projection."
    )
    return (
        f"{rendered.rstrip()}\n\n"
        "## Label-blind Results passages supplied by the local evaluation harness\n\n"
        f"{supplied}\n",
        prompt_version,
        passages,
    )


def _generation_schema_for_passages(passages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use row-specific line enums and no regex in Ollama's native schema parser.

    Ollama 0.15.1 was observed to terminate natively on the official Evidence
    Inference citation regex.  The official example schema remains the post-generation
    evaluator; this narrower schema only constrains local generation.
    """

    line_ids = sorted(
        {
            str(passage["line_id"])
            for passage in passages
            if isinstance(passage, Mapping)
            and isinstance(passage.get("line_id"), str)
            and re.fullmatch(r"L[1-9][0-9]*", str(passage["line_id"]))
        }
    )
    if not line_ids:
        line_ids = ["L1"]
    schema = deepcopy(_OLLAMA_TASK_SCHEMA_BASE)
    schema["properties"]["findings"]["items"]["properties"]["evidence_lines"]["items"]["enum"] = (
        line_ids
    )
    return schema


def _receipt_path(receipt_root: Path, kind: str, request_sha256: str) -> Path:
    if not _SHA256.fullmatch(request_sha256):
        raise OllamaGEPAStudyError("receipt request key must be SHA-256")
    return receipt_root / kind / request_sha256[:2] / f"{request_sha256}.json"


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    request_payload: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    _validate_self_hash(snapshot, "receipt_sha256", "model-call receipt")
    if (
        snapshot.get("receipt_version") != RECEIPT_VERSION
        or snapshot.get("request_sha256") != request_sha256
        or snapshot.get("request") != dict(request_payload)
        or snapshot.get("attempted_model_call") is not True
        or snapshot.get("success") not in {True, False}
    ):
        raise OllamaGEPAStudyError("model-call receipt contract mismatch")
    result = snapshot.get("result")
    if snapshot["success"] is True:
        try:
            OllamaGenerationResult.model_validate(result)
        except ValueError as exc:
            raise OllamaGEPAStudyError("successful receipt has invalid model result") from exc
    elif result is not None or not isinstance(snapshot.get("error_category"), str):
        raise OllamaGEPAStudyError("failed receipt has invalid failure payload")
    return dict(receipt)


def _generate_with_receipt(
    *,
    client: OllamaClientProtocol,
    identity: OllamaIdentity,
    config: OllamaGenerationConfig,
    receipt_root: Path,
    kind: Literal["task", "reflection"],
    namespace: str,
    prompt: str,
    output_schema: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not _SAFE_NAMESPACE.fullmatch(namespace):
        raise OllamaGEPAStudyError("model-call namespace is invalid")
    request = {
        "kind": kind,
        "namespace": namespace,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output_schema_sha256": hash_canonical(output_schema),
        "generation_config_sha256": config.config_sha256,
        "ollama_identity_sha256": identity.identity_sha256,
    }
    request_sha = hash_canonical(request)
    path = _receipt_path(receipt_root, kind, request_sha)
    if path.exists():
        return (
            _validate_receipt(
                _read_json_object(path), request_payload=request, request_sha256=request_sha
            ),
            True,
        )
    try:
        result = client.generate(prompt=prompt, output_schema=output_schema, config=config)
        payload: dict[str, Any] = {
            "receipt_version": RECEIPT_VERSION,
            "request_sha256": request_sha,
            "request": request,
            "attempted_model_call": True,
            "success": True,
            "error_category": None,
            "result": result.model_dump(mode="json"),
        }
    except LocalOllamaError:
        payload = {
            "receipt_version": RECEIPT_VERSION,
            "request_sha256": request_sha,
            "request": request,
            "attempted_model_call": True,
            "success": False,
            "error_category": "local_ollama_error",
            "result": None,
        }
    receipt = _add_self_hash(payload, "receipt_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, receipt)
    return receipt, False


def _schema_error(output: Any, schema: Mapping[str, Any]) -> str | None:
    try:
        copied = deepcopy(dict(schema))
        validator = validator_for(copied)
        validator.check_schema(copied)
        validator(copied).validate(output)
    except (SchemaError, JSONSchemaValidationError) as exc:
        return type(exc).__name__
    return None


def _expected_direction(example: OptimizationExample) -> str:
    try:
        value = example.expected_output["findings"][0]["direction"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OllamaGEPAStudyError("Evidence Inference example lacks a direction label") from exc
    if value not in {"increase", "decrease", "no_effect"}:
        raise OllamaGEPAStudyError("Evidence Inference direction label is invalid")
    return str(value)


def _score_parsed_output(
    *,
    example: OptimizationExample,
    parsed: Mapping[str, Any],
    result: OllamaGenerationResult,
    metrics: MetricSettings,
    num_predict: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    schema_error = _schema_error(parsed, example.output_schema)
    findings = parsed.get("findings")
    finding = findings[0] if isinstance(findings, list) and len(findings) == 1 else None
    predicted_direction = finding.get("direction") if isinstance(finding, Mapping) else None
    task_shape = parsed.get("eligible") is True and isinstance(finding, Mapping)
    direction = float(
        schema_error is None and task_shape and predicted_direction == _expected_direction(example)
    )
    grounding_result: dict[str, Any] | None = None
    grounding = 0.0
    if isinstance(finding, Mapping):
        try:
            raw_grounding = ground_evidence(
                finding.get("evidence_quote"),
                finding.get("evidence_lines"),
                example.content_lines,
                line_sections=example.line_sections,
                source_accessible=example.source_accessible,
            )
            grounding_result = dict(raw_grounding)
            grounding = float(
                schema_error is None
                and task_shape
                and raw_grounding.get("grounding_status") == "exact"
                and raw_grounding.get("section_flagged") is False
                and "relocated_from_line_numbers" not in raw_grounding
            )
        except (GroundingContractError, TypeError, ValueError):
            grounding_result = {"grounding_status": "invalid"}
    token_count = result.eval_count
    token_efficiency = (
        0.0
        if token_count is None
        else max(0.0, 1.0 - min(token_count, num_predict) / float(num_predict))
    )
    duration = result.total_duration_ns
    latency_limit = metrics.latency_sla_seconds * 1_000_000_000.0
    latency_success = float(duration is not None and duration <= latency_limit)
    objectives = {
        "direction_accuracy": direction,
        "direction_distribution_fidelity": 0.0,
        "formal_grounding_validity": grounding,
        "structured_output_validity": float(schema_error is None),
        "generation_success": 1.0,
        "token_efficiency": token_efficiency,
        "latency_sla_success": latency_success,
    }
    details = {
        "expected_direction": _expected_direction(example),
        "predicted_direction": predicted_direction,
        "schema_error": schema_error,
        "grounding_result": grounding_result,
        "eval_count": result.eval_count,
        "prompt_eval_count": result.prompt_eval_count,
        "total_duration_ns": result.total_duration_ns,
        "done_reason": result.done_reason,
    }
    return objectives, details


def _failed_objectives() -> dict[str, float]:
    return {objective: 0.0 for objective in _OBJECTIVES}


def _weighted_score(objectives: Mapping[str, float], metrics: MetricSettings) -> float:
    return math.fsum(
        float(objectives[name]) * metrics.objective_weights[name] for name in _OBJECTIVES
    )


def _direction_distribution_fidelity(
    expected: Sequence[str], predicted: Sequence[str | None]
) -> float:
    if not expected or len(expected) != len(predicted):
        raise OllamaGEPAStudyError("direction-distribution inputs are empty or misaligned")
    categories = ("increase", "decrease", "no_effect", "missing")
    expected_counts = Counter(expected)
    predicted_counts = Counter(
        value if value in categories[:-1] else "missing" for value in predicted
    )
    count = len(expected)
    total_variation = 0.5 * math.fsum(
        abs(expected_counts[category] / count - predicted_counts[category] / count)
        for category in categories
    )
    return max(0.0, 1.0 - total_variation)


_REFLECTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"new_component": {"type": "string", "minLength": 1}},
    "required": ["new_component"],
    "additionalProperties": False,
}


class OllamaGEPAAdapter:
    """Official-GEPA adapter using only the exact local Ollama boundary."""

    def __init__(
        self,
        *,
        client: OllamaClientProtocol,
        identity: OllamaIdentity,
        config: OllamaGenerationConfig,
        metrics: MetricSettings,
        receipt_root: Path,
        namespace: str,
        distribution_fidelity_enabled: bool = True,
    ) -> None:
        if not _SAFE_NAMESPACE.fullmatch(namespace):
            raise OllamaGEPAStudyError("adapter namespace is invalid")
        self.client = client
        self.identity = identity
        self.config = config
        self.metrics = metrics
        self.receipt_root = receipt_root
        self.namespace = namespace
        self.distribution_fidelity_enabled = distribution_fidelity_enabled
        self._state: dict[str, int] = {
            "task_model_calls": 0,
            "task_receipt_replays": 0,
            "reflection_model_calls": 0,
            "reflection_receipt_replays": 0,
            "reflection_proposals": 0,
            "contract_wrapped_proposals": 0,
            "unchanged_proposals": 0,
            "rejected_training_text_proposals": 0,
        }

    def get_adapter_state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        if not state:
            self._state = {key: 0 for key in self._state}
            return
        if set(state) != set(self._state) or any(
            not isinstance(state[key], int) or state[key] < 0 for key in self._state
        ):
            raise OllamaGEPAStudyError("persisted Ollama-GEPA adapter state is invalid")
        self._state = deepcopy(state)

    def evaluate(
        self,
        batch: list[OptimizationExample | Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> OptimizationEvaluationBatch:
        candidate_sha = _candidate_sha256(candidate)
        template = candidate[COMPONENT]
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] = []
        metric_calls = 0
        for raw in batch:
            example = (
                raw
                if isinstance(raw, OptimizationExample)
                else OptimizationExample.model_validate(raw)
            )
            base = {
                "example_id": example.example_id,
                "paper_id": example.paper_id,
                "group_id": example.group_id,
                "candidate_sha256": candidate_sha,
                "expected_direction": _expected_direction(example),
                "replacements": deepcopy(example.replacements),
            }
            try:
                prompt, prompt_version, passages = _task_prompt(example, template)
            except _NoResultsPassageError as exc:
                objectives = _failed_objectives()
                output = {"error_category": "no_results_passage"}
                trajectory = {
                    **base,
                    "error_category": "no_results_passage",
                    "error_detail": str(exc),
                    "parsed_output": None,
                    "objective_scores": objectives,
                    "passage_count": 0,
                }
                metric_calls += 1
            except OllamaGEPAStudyError as exc:
                objectives = _failed_objectives()
                output = {"error_category": "prompt_contract"}
                trajectory = {
                    **base,
                    "error_category": "prompt_contract",
                    "error_detail": str(exc),
                    "parsed_output": None,
                    "objective_scores": objectives,
                }
            else:
                generation_schema = _generation_schema_for_passages(passages)
                receipt, replayed = _generate_with_receipt(
                    client=self.client,
                    identity=self.identity,
                    config=self.config,
                    receipt_root=self.receipt_root,
                    kind="task",
                    namespace=self.namespace,
                    prompt=prompt,
                    output_schema=generation_schema,
                )
                if replayed:
                    self._state["task_receipt_replays"] += 1
                else:
                    self._state["task_model_calls"] += 1
                # GEPA budgets logical metric evaluations. A crash-safe replay is
                # still that same logical evaluation, while physical calls are
                # reported separately in adapter and receipt telemetry.
                metric_calls += 1
                if receipt["success"] is not True:
                    objectives = _failed_objectives()
                    output = {"error_category": receipt["error_category"]}
                    trajectory = {
                        **base,
                        "prompt_version": prompt_version,
                        "request_sha256": receipt["request_sha256"],
                        "error_category": receipt["error_category"],
                        "parsed_output": None,
                        "objective_scores": objectives,
                    }
                else:
                    result = OllamaGenerationResult.model_validate(receipt["result"])
                    try:
                        parsed_raw = json.loads(result.response_text)
                    except json.JSONDecodeError:
                        parsed_raw = None
                    if not isinstance(parsed_raw, Mapping):
                        objectives = _failed_objectives()
                        output = {"error_category": "invalid_json_object"}
                        trajectory = {
                            **base,
                            "prompt_version": prompt_version,
                            "request_sha256": receipt["request_sha256"],
                            "error_category": "invalid_json_object",
                            "parsed_output": None,
                            "objective_scores": objectives,
                        }
                    else:
                        parsed = dict(parsed_raw)
                        objectives, details = _score_parsed_output(
                            example=example,
                            parsed=parsed,
                            result=result,
                            metrics=self.metrics,
                            num_predict=self.config.num_predict,
                        )
                        output = {
                            "parsed_output": parsed,
                            "request_sha256": receipt["request_sha256"],
                            "telemetry": {
                                "eval_count": result.eval_count,
                                "prompt_eval_count": result.prompt_eval_count,
                                "total_duration_ns": result.total_duration_ns,
                            },
                            "generation_schema_sha256": hash_canonical(generation_schema),
                            "evaluation_schema_sha256": hash_canonical(example.output_schema),
                            "passage_projection_sha256": hash_canonical(passages),
                            "passage_count": len(passages),
                        }
                        trajectory = {
                            **base,
                            "prompt_version": prompt_version,
                            "request_sha256": receipt["request_sha256"],
                            "error_category": None,
                            "parsed_output": parsed,
                            "details": details,
                            "objective_scores": objectives,
                            "generation_schema_sha256": hash_canonical(generation_schema),
                            "evaluation_schema_sha256": hash_canonical(example.output_schema),
                            "passage_projection_sha256": hash_canonical(passages),
                            "passage_count": len(passages),
                        }
            score = _weighted_score(objectives, self.metrics)
            output["objective_scores"] = objectives
            output["scalar_score"] = score
            trajectory["scalar_score"] = score
            outputs.append(output)
            scores.append(score)
            objective_scores.append(objectives)
            trajectories.append(trajectory)
        distribution_fidelity = 0.0
        if self.distribution_fidelity_enabled:
            distribution_fidelity = _direction_distribution_fidelity(
                [str(trajectory["expected_direction"]) for trajectory in trajectories],
                [
                    (
                        trajectory["details"].get("predicted_direction")
                        if isinstance(trajectory.get("details"), Mapping)
                        else None
                    )
                    for trajectory in trajectories
                ],
            )
        for index, objectives in enumerate(objective_scores):
            objectives["direction_distribution_fidelity"] = distribution_fidelity
            score = _weighted_score(objectives, self.metrics)
            scores[index] = score
            outputs[index]["objective_scores"] = objectives
            outputs[index]["scalar_score"] = score
            trajectories[index]["objective_scores"] = objectives
            trajectories[index]["scalar_score"] = score
        return OptimizationEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
            num_metric_calls=metric_calls,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: OptimizationEvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        _candidate_sha256(candidate)
        if components_to_update != [COMPONENT]:
            raise OllamaGEPAStudyError("reflection may update only extraction_prompt")
        if eval_batch.trajectories is None or not eval_batch.trajectories:
            raise OllamaGEPAStudyError("GEPA requested reflection without trajectories")
        records: list[dict[str, Any]] = []
        for trajectory in eval_batch.trajectories:
            details = trajectory.get("details")
            records.append(
                {
                    "Inputs": deepcopy(trajectory["replacements"]),
                    "Generated Outputs": deepcopy(trajectory.get("parsed_output")),
                    "Feedback": {
                        "expected_direction": trajectory["expected_direction"],
                        "predicted_direction": (
                            details.get("predicted_direction")
                            if isinstance(details, Mapping)
                            else None
                        ),
                        "schema_error": (
                            details.get("schema_error") if isinstance(details, Mapping) else None
                        ),
                        "grounding_status": (
                            details.get("grounding_result")
                            if isinstance(details, Mapping)
                            else None
                        ),
                        "error_category": trajectory.get("error_category"),
                        "objective_scores": deepcopy(trajectory["objective_scores"]),
                        "scalar_score": trajectory["scalar_score"],
                    },
                }
            )
        return {COMPONENT: records}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        _candidate_sha256(candidate)
        if components_to_update != [COMPONENT] or set(reflective_dataset) != {COMPONENT}:
            raise OllamaGEPAStudyError("proposal request has an invalid component set")
        side_info = json.dumps(
            list(reflective_dataset[COMPONENT]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt = (
            "You are GEPA's local reflection model. Improve the extraction prompt below using the "
            "training feedback. Return one JSON object with key new_component. Preserve exactly "
            "one Prompt version declaration and exactly the tokens [[INTERVENTION]], "
            "[[COMPARATOR]], and [[OUTCOME]]. Do not include article-specific answers, labels, or "
            "evidence. The result must "
            "remain a reusable Evidence Inference direction-and-grounding instruction.\n\n"
            f"CURRENT COMPONENT:\n{candidate[COMPONENT]}\n\n"
            f"TRAINING FEEDBACK:\n{side_info}\n"
        )
        receipt, replayed = _generate_with_receipt(
            client=self.client,
            identity=self.identity,
            config=self.config,
            receipt_root=self.receipt_root,
            kind="reflection",
            namespace=self.namespace,
            prompt=prompt,
            output_schema=_REFLECTION_SCHEMA,
        )
        if replayed:
            self._state["reflection_receipt_replays"] += 1
        else:
            self._state["reflection_model_calls"] += 1
        self._state["reflection_proposals"] += 1
        raw = ""
        if receipt["success"] is True:
            result = OllamaGenerationResult.model_validate(receipt["result"])
            try:
                parsed = json.loads(result.response_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping) and isinstance(parsed.get("new_component"), str):
                raw = parsed["new_component"].strip()
        if raw and _contains_reflective_training_text(raw, reflective_dataset[COMPONENT]):
            self._state["rejected_training_text_proposals"] += 1
            raw = ""
        if raw:
            proposed, wrapped = _contract_bound_proposal(raw)
        else:
            proposed, wrapped = candidate[COMPONENT], False
        if wrapped:
            self._state["contract_wrapped_proposals"] += 1
        if proposed == candidate[COMPONENT]:
            self._state["unchanged_proposals"] += 1
        return {COMPONENT: proposed}


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _all_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _all_strings(item)]
    return [value] if isinstance(value, str) else []


def _contains_reflective_training_text(
    proposal: str, reflective_records: Sequence[Mapping[str, Any]]
) -> bool:
    """Reject exact copying of long row-specific strings into a reusable prompt."""

    normalized = " ".join(proposal.casefold().split())
    protected = {
        " ".join(text.casefold().split())
        for text in _all_strings(list(reflective_records))
        if len(" ".join(text.split())) >= 12
    }
    return any(text and text in normalized for text in protected)


def _contract_bound_proposal(raw: str) -> tuple[str, bool]:
    if raw:
        try:
            _validate_seed_prompt(raw)
        except OllamaGEPAStudyError:
            pass
        else:
            return raw, False
    body = raw.strip() or (
        "Apply the current scientific extraction contract conservatively. Resolve direction only "
        "from explicit intervention-versus-comparator Results evidence, and cite an exact source "
        "span."
    )
    body = re.sub(r"^Prompt version:.*$", "", body, flags=re.MULTILINE).strip()
    body = body.replace("[[", "[ [").replace("]]", "] ]")
    wrapped = (
        "# GEPA-refined Evidence Inference extraction prompt\n\n"
        "Prompt version: `evidence-inference-ollama-gepa-v1`\n\n"
        f"{body}\n\n"
        "Answer one locked clinical-trial comparison using only supplied BODY.RESULTS source "
        "lines. Return only the supplied JSON schema.\n\n"
        "- Outcome: `[[OUTCOME]]`\n"
        "- Intervention: `[[INTERVENTION]]`\n"
        "- Comparator: `[[COMPARATOR]]`\n\n"
        "Use increase or decrease only for an explicitly reported significant direction; use "
        "no_effect only for an explicitly reported null comparison. If the comparison is "
        "unsupported, return eligible=false and findings=[]. Otherwise return one finding with the "
        "shortest exact "
        "Results quote and its exact L-number or inclusive L-range."
    )
    _validate_seed_prompt(wrapped)
    return wrapped, True


def _result_to_dict(result: Any) -> dict[str, Any]:
    converter = getattr(result, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return deepcopy(dict(converted))
    required = (
        "candidates",
        "parents",
        "val_aggregate_scores",
        "val_subscores",
        "per_val_instance_best_candidates",
        "discovery_eval_counts",
        "total_metric_calls",
        "num_full_val_evals",
    )
    if any(not hasattr(result, name) for name in required):
        raise OllamaGEPAStudyError("official GEPA result is missing required fields")
    return {name: deepcopy(getattr(result, name)) for name in required} | {
        "best_idx": int(result.best_idx),
        "val_aggregate_subscores": deepcopy(getattr(result, "val_aggregate_subscores", None)),
        "per_objective_best_candidates": deepcopy(
            getattr(result, "per_objective_best_candidates", None)
        ),
        "objective_pareto_front": deepcopy(getattr(result, "objective_pareto_front", None)),
    }


def run_optimization(
    *,
    config_path: str | Path,
    client: OllamaClientProtocol,
    optimize_fn: Callable[..., Any] | None = None,
) -> Path:
    """Run/resume official GEPA on train/dev and freeze its winner before test access."""

    start_wall_ns = time.perf_counter_ns()
    start_cpu_ns = time.process_time_ns()
    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    if paths.winner.exists():
        validate_frozen_winner(config_path=config_path, client=client)
        return paths.winner
    plan = validate_optimization_plan(config_path=config_path, client=client)
    _, train, dev = _load_selected_optimization_examples(paths, config)
    try:
        seed_text = paths.seed_prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OllamaGEPAStudyError("seed prompt cannot be read") from exc
    identity = OllamaIdentity.model_validate(plan["observed_ollama_identity"])
    adapter = OllamaGEPAAdapter(
        client=client,
        identity=identity,
        config=config.generation,
        metrics=config.metrics,
        receipt_root=paths.receipts,
        namespace="optimization",
    )
    optimizer = optimize_fn or _load_official_gepa_optimize()
    paths.gepa_run_dir.mkdir(parents=True, exist_ok=True)
    result = optimizer(
        seed_candidate={COMPONENT: seed_text},
        trainset=train,
        valset=dev,
        adapter=adapter,
        task_lm=None,
        evaluator=None,
        reflection_lm=None,
        reflection_lm_kwargs=None,
        candidate_selection_strategy=config.optimization.candidate_selection_strategy,
        frontier_type=config.optimization.frontier_type,
        skip_perfect_score=config.optimization.skip_perfect_score,
        batch_sampler=config.optimization.batch_sampler,
        reflection_minibatch_size=config.optimization.reflection_minibatch_size,
        custom_candidate_proposer=None,
        module_selector=config.optimization.module_selector,
        use_merge=config.optimization.use_merge,
        max_metric_calls=config.optimization.max_metric_calls,
        run_dir=str(paths.gepa_run_dir),
        track_best_outputs=config.optimization.track_best_outputs,
        display_progress_bar=False,
        cache_evaluation=config.optimization.cache_evaluation,
        seed=config.optimization.gepa_seed,
        raise_on_exception=True,
        val_evaluation_policy="full_eval",
        acceptance_criterion=config.optimization.acceptance_criterion,
    )
    result_payload = _result_to_dict(result)
    total_calls = result_payload.get("total_metric_calls")
    if not isinstance(total_calls, int) or total_calls < config.optimization.max_metric_calls:
        raise OllamaGEPAStudyError(
            "GEPA returned before exhausting the prespecified metric-call budget; resume instead "
            "of freezing"
        )
    candidates = result_payload.get("candidates")
    best_idx = result_payload.get("best_idx")
    scores = result_payload.get("val_aggregate_scores")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(best_idx, int)
        or best_idx < 0
        or best_idx >= len(candidates)
        or not isinstance(scores, list)
        or len(scores) != len(candidates)
    ):
        raise OllamaGEPAStudyError("official GEPA returned an invalid candidate trace")
    validation_objectives = result_payload.get("val_aggregate_subscores")
    if (
        not isinstance(validation_objectives, list)
        or len(validation_objectives) != len(candidates)
        or any(
            not isinstance(values, Mapping) or set(values) != set(_OBJECTIVES)
            for values in validation_objectives
        )
    ):
        raise OllamaGEPAStudyError("official GEPA omitted the prespecified objective trace")
    winner_candidate = candidates[best_idx]
    winner_sha = _candidate_sha256(winner_candidate)
    winner_text = winner_candidate[COMPONENT]
    _validate_seed_prompt(winner_text)
    trace_payload = _add_self_hash(
        {
            "trace_version": TRACE_VERSION,
            "study_version": STUDY_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "official_gepa_result": result_payload,
            "adapter_state_after_run": adapter.get_adapter_state(),
        },
        "trace_sha256",
    )
    atomic_write_text(paths.winner_prompt, winner_text)
    atomic_write_json(paths.trace, trace_payload)
    state_path = paths.gepa_run_dir / "gepa_state.bin"
    state_sha = sha256_file(state_path) if state_path.is_file() else None
    adapter_state = adapter.get_adapter_state()
    receipt_counts = validate_private_receipts(paths.receipts)
    receipt_telemetry = summarize_private_receipts(paths.receipts, namespaces={"optimization"})
    invocation_runtime = _runtime_snapshot(start_wall_ns, start_cpu_ns)
    winner_payload = _add_self_hash(
        {
            "winner_version": WINNER_VERSION,
            "study_version": STUDY_VERSION,
            "stage": "winner_frozen_before_test_payload_access",
            "plan_file_sha256": sha256_file(paths.plan),
            "plan_sha256": plan["plan_sha256"],
            "trace_file_sha256": sha256_file(paths.trace),
            "trace_sha256": trace_payload["trace_sha256"],
            "gepa_state_file_sha256": state_sha,
            "source_code_sha256s": _source_code_hashes(),
            "config_file_sha256": sha256_file(paths.config),
            "manifest_file_sha256": sha256_file(paths.manifest),
            "generation_config_sha256": config.generation.config_sha256,
            "ollama_identity_sha256": identity.identity_sha256,
            "gepa_version": _official_gepa_version(),
            "winner_index": best_idx,
            "winner_candidate_sha256": winner_sha,
            "winner_prompt_sha256": hashlib.sha256(winner_text.encode()).hexdigest(),
            "winner_prompt_file_sha256": sha256_file(paths.winner_prompt),
            "winner_prompt": winner_text,
            "seed_prompt_sha256": plan["seed_prompt_sha256"],
            "seed_retained": winner_text == seed_text,
            "validation_score": float(scores[best_idx]),
            "seed_validation_score": float(scores[0]),
            "winner_development_objectives": {
                objective: float(validation_objectives[best_idx][objective])
                for objective in _OBJECTIVES
            },
            "seed_development_objectives": {
                objective: float(validation_objectives[0][objective]) for objective in _OBJECTIVES
            },
            "candidate_count": len(candidates),
            "total_metric_calls": total_calls,
            "full_validation_evaluations": result_payload.get("num_full_val_evals"),
            "reflection_proposals": adapter_state["reflection_proposals"],
            "unique_task_receipts": receipt_counts["task"],
            "unique_reflection_receipts": receipt_counts["reflection"],
            "optimization_receipt_telemetry": receipt_telemetry,
            "optimization_final_invocation_runtime": invocation_runtime,
            "optimizer_exploration_sufficient": adapter_state["reflection_proposals"]
            >= config.optimization.minimum_reflection_proposals,
            "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
            "generation_schema_base_sha256": OLLAMA_TASK_SCHEMA_BASE_SHA256,
            "evaluation_schema_sha256": plan["evaluation_schema_sha256"],
            "passage_projection_algorithm": plan["passage_projection_algorithm"],
            "passage_projection_config_sha256": plan["passage_projection_config_sha256"],
            "test_membership_metadata_present_in_manifest": True,
            "test_payload_opened": False,
            "test_labels_scored": False,
            "all_labels_historically_opened": True,
            "test_is_non_pristine": True,
        },
        "winner_sha256",
    )
    atomic_write_json(paths.winner, winner_payload)
    return paths.winner


def validate_frozen_winner(
    *,
    config_path: str | Path,
    client: OllamaClientProtocol,
) -> dict[str, Any]:
    """Validate a complete winner freeze without opening or hashing the test JSONL."""

    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    plan = validate_optimization_plan(config_path=config_path, client=client)
    winner = _read_json_object(paths.winner)
    _validate_self_hash(winner, "winner_sha256", "frozen winner")
    if (
        winner.get("winner_version") != WINNER_VERSION
        or winner.get("study_version") != STUDY_VERSION
        or winner.get("stage") != "winner_frozen_before_test_payload_access"
        or winner.get("test_payload_opened") is not False
        or winner.get("test_labels_scored") is not False
        or winner.get("all_labels_historically_opened") is not True
        or winner.get("test_is_non_pristine") is not True
    ):
        raise OllamaGEPAStudyError("frozen winner stage/scope is invalid")
    expected = {
        "plan_file_sha256": sha256_file(paths.plan),
        "plan_sha256": plan["plan_sha256"],
        "trace_file_sha256": sha256_file(paths.trace),
        "source_code_sha256s": _source_code_hashes(),
        "config_file_sha256": sha256_file(paths.config),
        "manifest_file_sha256": sha256_file(paths.manifest),
        "generation_config_sha256": config.generation.config_sha256,
        "ollama_identity_sha256": plan["observed_ollama_identity_sha256"],
        "gepa_version": _official_gepa_version(),
        "winner_prompt_file_sha256": sha256_file(paths.winner_prompt),
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_schema_base_sha256": OLLAMA_TASK_SCHEMA_BASE_SHA256,
        "evaluation_schema_sha256": plan["evaluation_schema_sha256"],
        "passage_projection_algorithm": RETRIEVAL_ALGORITHM,
        "passage_projection_config_sha256": hash_canonical(DEFAULT_RETRIEVAL_CONFIG),
    }
    if any(winner.get(key) != value for key, value in expected.items()):
        raise OllamaGEPAStudyError("frozen winner lineage no longer matches current bytes")
    state_sha = winner.get("gepa_state_file_sha256")
    state_path = paths.gepa_run_dir / "gepa_state.bin"
    if state_sha is not None and (not state_path.is_file() or sha256_file(state_path) != state_sha):
        raise OllamaGEPAStudyError("official GEPA checkpoint changed after winner freeze")
    trace = _read_json_object(paths.trace)
    _validate_self_hash(trace, "trace_sha256", "GEPA result trace")
    if trace.get("trace_sha256") != winner.get("trace_sha256"):
        raise OllamaGEPAStudyError("frozen winner and GEPA trace are unbound")
    try:
        prompt = paths.winner_prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OllamaGEPAStudyError("frozen winner prompt is unreadable") from exc
    if (
        winner.get("winner_prompt") != prompt
        or winner.get("winner_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
        or winner.get("winner_candidate_sha256") != hash_canonical({COMPONENT: prompt})
    ):
        raise OllamaGEPAStudyError("frozen winner prompt bytes do not match its bundle")
    _validate_seed_prompt(prompt)
    return winner


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise OllamaGEPAStudyError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _clustered_interval(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        paper_id = row.get("paper_id")
        value = row.get(value_key)
        if not isinstance(paper_id, str) or not isinstance(value, (int, float)):
            raise OllamaGEPAStudyError(f"invalid clustered value: {value_key}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise OllamaGEPAStudyError(f"nonfinite clustered value: {value_key}")
        clusters[paper_id].append(numeric)
    if not clusters:
        raise OllamaGEPAStudyError("clustered interval requires rows")
    count = sum(len(values) for values in clusters.values())
    point = math.fsum(value for values in clusters.values() for value in values) / count
    paper_ids = sorted(clusters)
    metric_seed = int.from_bytes(hashlib.sha256(f"{seed}:{value_key}".encode()).digest()[:8], "big")
    generator = random.Random(metric_seed)
    samples: list[float] = []
    for _ in range(replicates):
        selected = [paper_ids[generator.randrange(len(paper_ids))] for _ in paper_ids]
        numerator = math.fsum(value for paper in selected for value in clusters[paper])
        denominator = sum(len(clusters[paper]) for paper in selected)
        samples.append(numerator / denominator)
    return {
        "estimate": point,
        "lower": _quantile(samples, 0.025),
        "upper": _quantile(samples, 0.975),
        "examples": count,
        "articles": len(clusters),
        "replicates": replicates,
        "method": "prompt_weighted_article_cluster_percentile_bootstrap_95",
    }


def _paired_summary(rows: Sequence[Mapping[str, Any]], metrics: MetricSettings) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for objective in (item for item in _OBJECTIVES if item != "direction_distribution_fidelity"):
        flattened: list[dict[str, Any]] = []
        for row in rows:
            seed_value = float(row["seed"]["objective_scores"][objective])
            winner_value = float(row["winner"]["objective_scores"][objective])
            flattened.append(
                {
                    "paper_id": row["paper_id"],
                    "seed": seed_value,
                    "winner": winner_value,
                    "paired_delta": winner_value - seed_value,
                }
            )
        summary[objective] = {
            "seed": _clustered_interval(
                flattened,
                "seed",
                seed=metrics.bootstrap_seed,
                replicates=metrics.bootstrap_replicates,
            ),
            "winner": _clustered_interval(
                flattened,
                "winner",
                seed=metrics.bootstrap_seed,
                replicates=metrics.bootstrap_replicates,
            ),
            "paired_delta_winner_minus_seed": _clustered_interval(
                flattened,
                "paired_delta",
                seed=metrics.bootstrap_seed,
                replicates=metrics.bootstrap_replicates,
            ),
        }
    summary["direction_macro_recall"] = _nonlinear_paired_direction_interval(
        rows,
        metric="macro_recall",
        seed=metrics.bootstrap_seed,
        replicates=metrics.bootstrap_replicates,
    )
    return summary


def _direction_statistic(rows: Sequence[Mapping[str, Any]], arm: str, metric: str) -> float | None:
    expected = [str(row["expected_direction"]) for row in rows]
    predicted = [row[arm].get("predicted_direction") for row in rows]
    if metric == "distribution_fidelity":
        return _direction_distribution_fidelity(expected, predicted)
    if metric != "macro_recall":
        raise OllamaGEPAStudyError("unknown nonlinear direction metric")
    classes = ("increase", "decrease", "no_effect")
    recalls: list[float] = []
    for direction in classes:
        indices = [index for index, value in enumerate(expected) if value == direction]
        if not indices:
            return None
        recalls.append(sum(predicted[index] == direction for index in indices) / len(indices))
    return math.fsum(recalls) / len(recalls)


def _nonlinear_paired_direction_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str):
            raise OllamaGEPAStudyError("paired direction row lacks article identity")
        clusters[paper_id].append(row)
    seed_point = _direction_statistic(rows, "seed", metric)
    winner_point = _direction_statistic(rows, "winner", metric)
    if seed_point is None or winner_point is None:
        raise OllamaGEPAStudyError("paired direction population lacks an expected class")
    paper_ids = sorted(clusters)
    generator = random.Random(
        int.from_bytes(hashlib.sha256(f"{seed}:{metric}".encode()).digest()[:8], "big")
    )
    seed_samples: list[float] = []
    winner_samples: list[float] = []
    attempts = 0
    while len(seed_samples) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled_papers = [paper_ids[generator.randrange(len(paper_ids))] for _ in paper_ids]
        sampled_rows = [row for paper_id in sampled_papers for row in clusters[paper_id]]
        seed_value = _direction_statistic(sampled_rows, "seed", metric)
        winner_value = _direction_statistic(sampled_rows, "winner", metric)
        if seed_value is None or winner_value is None:
            continue
        seed_samples.append(seed_value)
        winner_samples.append(winner_value)
    if len(seed_samples) != replicates:
        raise OllamaGEPAStudyError("cluster bootstrap could not retain all direction classes")
    deltas = [
        winner - seed_value for seed_value, winner in zip(seed_samples, winner_samples, strict=True)
    ]
    common = {
        "examples": len(rows),
        "articles": len(clusters),
        "replicates": replicates,
        "method": "paired_article_cluster_percentile_bootstrap_95_nonlinear_metric",
    }
    return {
        "seed": {
            "estimate": seed_point,
            "lower": _quantile(seed_samples, 0.025),
            "upper": _quantile(seed_samples, 0.975),
            **common,
        },
        "winner": {
            "estimate": winner_point,
            "lower": _quantile(winner_samples, 0.025),
            "upper": _quantile(winner_samples, 0.975),
            **common,
        },
        "paired_delta_winner_minus_seed": {
            "estimate": winner_point - seed_point,
            "lower": _quantile(deltas, 0.025),
            "upper": _quantile(deltas, 0.975),
            **common,
        },
    }


def _one_evaluation(
    adapter: OllamaGEPAAdapter,
    example: OptimizationExample,
    prompt: str,
) -> dict[str, Any]:
    evaluated = adapter.evaluate([example], {COMPONENT: prompt}, capture_traces=False)
    if (
        len(evaluated.outputs) != 1
        or len(evaluated.scores) != 1
        or evaluated.objective_scores is None
        or len(evaluated.objective_scores) != 1
    ):
        raise OllamaGEPAStudyError("paired adapter returned an invalid one-row evaluation")
    return {
        "scalar_score": evaluated.scores[0],
        "objective_scores": evaluated.objective_scores[0],
        "output": evaluated.outputs[0],
        "predicted_direction": _predicted_direction(evaluated.outputs[0]),
    }


def _predicted_direction(output: Mapping[str, Any]) -> str | None:
    parsed = output.get("parsed_output")
    findings = parsed.get("findings") if isinstance(parsed, Mapping) else None
    finding = findings[0] if isinstance(findings, list) and len(findings) == 1 else None
    direction = finding.get("direction") if isinstance(finding, Mapping) else None
    return direction if direction in {"increase", "decrease", "no_effect"} else None


def run_paired_test(
    *,
    config_path: str | Path,
    client: OllamaClientProtocol,
) -> tuple[Path, Path]:
    """Compare frozen winner and exact seed once on test after all lineage checks pass."""

    start_wall_ns = time.perf_counter_ns()
    start_cpu_ns = time.process_time_ns()
    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    winner = validate_frozen_winner(config_path=config_path, client=client)
    if paths.public_summary.exists():
        summary = validate_public_summary(_read_json_object(paths.public_summary))
        report = _read_json_object(paths.paired_report)
        _validate_self_hash(report, "paired_report_sha256", "paired test report")
        if (
            summary["lineage"]["private_paired_report_sha256"] != report["paired_report_sha256"]
            or summary["lineage"]["winner_sha256"] != winner["winner_sha256"]
        ):
            raise OllamaGEPAStudyError("public and private paired reports are unbound")
        return paths.paired_report, paths.public_summary
    plan = _read_json_object(paths.plan)
    if paths.paired_report.exists():
        report = _read_json_object(paths.paired_report)
        _validate_self_hash(report, "paired_report_sha256", "paired test report")
        if report.get("winner_bundle_sha256") != winner["winner_sha256"]:
            raise OllamaGEPAStudyError("existing paired report is bound to another winner")
        public = build_public_summary(
            config_path=config_path,
            winner=winner,
            paired_report=report,
        )
        validate_public_summary(public)
        paths.public_summary.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths.public_summary, public)
        return paths.paired_report, paths.public_summary
    try:
        seed_prompt = paths.seed_prompt.read_text(encoding="utf-8")
        winner_prompt = paths.winner_prompt.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OllamaGEPAStudyError("paired arm prompt is unreadable") from exc
    if hashlib.sha256(seed_prompt.encode()).hexdigest() != winner["seed_prompt_sha256"]:
        raise OllamaGEPAStudyError("paired seed prompt differs from optimization seed")

    # This is intentionally the first test-payload access in the optimization/test path.
    manifest = load_split_manifest(paths.manifest)
    test_path = paths.manifest.parent / manifest.test.path
    test_file_sha = sha256_file(test_path)
    if test_file_sha != manifest.test.sha256:
        raise OllamaGEPAStudyError("test split hash differs from the frozen manifest")
    test_examples = load_manifest_split(paths.manifest, "test")
    if any(
        hash_canonical(example.output_schema) != plan["evaluation_schema_sha256"]
        for example in test_examples
    ):
        raise OllamaGEPAStudyError("test evaluation schema differs from optimization freeze")
    identity = OllamaIdentity.model_validate(plan["observed_ollama_identity"])
    winner_adapter = OllamaGEPAAdapter(
        client=client,
        identity=identity,
        config=config.generation,
        metrics=config.metrics,
        receipt_root=paths.receipts,
        namespace="paired-test-winner",
        distribution_fidelity_enabled=False,
    )
    seed_adapter = OllamaGEPAAdapter(
        client=client,
        identity=identity,
        config=config.generation,
        metrics=config.metrics,
        receipt_root=paths.receipts,
        namespace="paired-test-seed",
        distribution_fidelity_enabled=False,
    )
    rows: list[dict[str, Any]] = []
    order_counts: Counter[str] = Counter()
    for example in sorted(test_examples, key=lambda item: item.example_id):
        winner_first = (
            int.from_bytes(
                hashlib.sha256(f"{example.paper_id}:{example.example_id}".encode()).digest()[:8],
                "big",
            )
            % 2
            == 0
        )
        if winner_first:
            winner_result = _one_evaluation(winner_adapter, example, winner_prompt)
            seed_result = _one_evaluation(seed_adapter, example, seed_prompt)
            order_counts["winner_then_seed"] += 1
        else:
            seed_result = _one_evaluation(seed_adapter, example, seed_prompt)
            winner_result = _one_evaluation(winner_adapter, example, winner_prompt)
            order_counts["seed_then_winner"] += 1
        rows.append(
            {
                "example_id": example.example_id,
                "paper_id": example.paper_id,
                "group_id": example.group_id,
                "expected_direction": _expected_direction(example),
                "seed": seed_result,
                "winner": winner_result,
            }
        )
    metric_summary = _paired_summary(rows, config.metrics)
    primary = metric_summary["direction_accuracy"]["paired_delta_winner_minus_seed"]
    prompt_changed = winner_prompt != seed_prompt
    observed_improvement = bool(prompt_changed and primary["estimate"] > 0 and primary["lower"] > 0)
    if not prompt_changed:
        disposition = "seed_retained_no_improvement_claim"
    elif observed_improvement:
        disposition = "observed_improvement_in_nonpristine_diagnostic_only"
    else:
        disposition = "no_improvement_claim"
    validate_private_receipts(paths.receipts)
    paired_receipt_telemetry = summarize_private_receipts(
        paths.receipts,
        namespaces={"paired-test-seed", "paired-test-winner"},
    )
    paired_runtime = _runtime_snapshot(start_wall_ns, start_cpu_ns)
    report_payload = _add_self_hash(
        {
            "paired_report_version": PAIRED_REPORT_VERSION,
            "study_version": STUDY_VERSION,
            "stage": "single_paired_test_after_frozen_winner",
            "winner_bundle_sha256": winner["winner_sha256"],
            "winner_bundle_file_sha256": sha256_file(paths.winner),
            "plan_sha256": plan["plan_sha256"],
            "manifest_file_sha256": sha256_file(paths.manifest),
            "test_split_jsonl_sha256": test_file_sha,
            "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
            "generation_schema_base_sha256": OLLAMA_TASK_SCHEMA_BASE_SHA256,
            "evaluation_schema_sha256": plan["evaluation_schema_sha256"],
            "seed_prompt_sha256": hashlib.sha256(seed_prompt.encode()).hexdigest(),
            "winner_prompt_sha256": hashlib.sha256(winner_prompt.encode()).hexdigest(),
            "seed_retained": not prompt_changed,
            "arm_order_policy": "sha256_parity_alternation_by_paired_example-v1",
            "arm_order_counts": dict(sorted(order_counts.items())),
            "examples": len(rows),
            "articles": len({row["paper_id"] for row in rows}),
            "rows": rows,
            "metrics": metric_summary,
            "primary_metric": "direction_accuracy_paired_delta_winner_minus_seed",
            "improvement_rule": (
                "winner_changed_and_point_delta_positive_and_95pct_lower_bound_positive"
            ),
            "observed_improvement_rule_satisfied": observed_improvement,
            "disposition": disposition,
            "winner_adapter_state": winner_adapter.get_adapter_state(),
            "seed_adapter_state": seed_adapter.get_adapter_state(),
            "paired_receipt_telemetry": paired_receipt_telemetry,
            "paired_test_invocation_runtime": paired_runtime,
            "all_labels_historically_opened": True,
            "test_is_non_pristine": True,
            "confirmatory_claim_allowed": False,
        },
        "paired_report_sha256",
    )
    atomic_write_json(paths.paired_report, report_payload)
    public = build_public_summary(
        config_path=config_path,
        winner=winner,
        paired_report=report_payload,
    )
    validate_public_summary(public)
    paths.public_summary.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.public_summary, public)
    return paths.paired_report, paths.public_summary


def _public_interval(interval: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: interval[key]
        for key in ("estimate", "lower", "upper", "examples", "articles", "replicates", "method")
    }


def build_public_summary(
    *,
    config_path: str | Path,
    winner: Mapping[str, Any],
    paired_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a private paired report to aggregate, identifier-free public metadata."""

    config = load_study_config(config_path)
    paths = study_paths(config_path, config)
    plan = _read_json_object(paths.plan)
    historical_sources = plan.get("source_code_sha256s")
    winner_sources = winner.get("source_code_sha256s")
    if (
        not isinstance(historical_sources, Mapping)
        or historical_sources != winner_sources
        or not isinstance(historical_sources.get("uv.lock"), str)
    ):
        raise OllamaGEPAStudyError(
            "historical execution source lineage is missing or inconsistent"
        )
    metrics = paired_report["metrics"]
    public_metrics = {
        objective: {arm: _public_interval(interval) for arm, interval in objective_payload.items()}
        for objective, objective_payload in metrics.items()
    }
    payload = {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "study_version": STUDY_VERSION,
        "status": paired_report["disposition"],
        "scientific_scope": "nonpristine_local_1.2B_evidence_inference_diagnostic",
        "confirmatory_claim_allowed": False,
        "all_labels_historically_opened": True,
        "test_is_non_pristine": True,
        "official_test_scored_once_after_freeze_in_this_staged_run": True,
        "optimizer": {
            "implementation": "official_gepa.optimize",
            "gepa_version": winner["gepa_version"],
            "metric_call_budget": config.optimization.max_metric_calls,
            "actual_metric_calls": winner["total_metric_calls"],
            "accepted_candidate_count": winner["candidate_count"],
            "reflection_proposal_count": winner["reflection_proposals"],
            "exploration_sufficient_by_prespecified_minimum": winner[
                "optimizer_exploration_sufficient"
            ],
            "candidate_selection_strategy": config.optimization.candidate_selection_strategy,
            "frontier_type": config.optimization.frontier_type,
            "within_run_test_tuning": False,
            "development_only_objectives": winner["winner_development_objectives"],
            "direction_distribution_fidelity_is_diagnostic_not_correctness": True,
        },
        "optimization_population": {
            "selection_algorithm": plan["selection_policy"]["algorithm"],
            "selection_used_labels": False,
            "train_examples": plan["train_selection"]["examples"],
            "train_articles": plan["train_selection"]["papers"],
            "development_examples": plan["dev_selection"]["examples"],
            "development_articles": plan["dev_selection"]["papers"],
            "train_selected_payload_sha256": plan["train_selection"]["selected_payload_sha256"],
            "development_selected_payload_sha256": plan["dev_selection"]["selected_payload_sha256"],
            "train_projected_task_inputs_sha256": plan["train_selection"][
                "projected_task_inputs_sha256"
            ],
            "development_projected_task_inputs_sha256": plan["dev_selection"][
                "projected_task_inputs_sha256"
            ],
            "train_examples_without_results_passages": plan["train_selection"][
                "examples_without_projected_results_passages"
            ],
            "development_examples_without_results_passages": plan["dev_selection"][
                "examples_without_projected_results_passages"
            ],
        },
        "paired_test_population": {
            "examples": paired_report["examples"],
            "articles": paired_report["articles"],
            "arm_order_policy": paired_report["arm_order_policy"],
        },
        "model": {
            "name": config.generation.model,
            "digest": config.generation.model_digest,
            "ollama_version": config.generation.expected_ollama_version,
            "parameter_size": plan["observed_ollama_identity"]["parameter_size"],
            "generation_config_sha256": config.generation.config_sha256,
            "identity_sha256": plan["observed_ollama_identity_sha256"],
            "same_exact_model_for_task_and_reflection": True,
        },
        "runtime_environment": {
            "python": plan["python_runtime_identity"],
            "dependency_lock_sha256": historical_sources["uv.lock"],
            "exact_hardware_identity_recorded": False,
        },
        "seed_retained": winner["seed_retained"],
        "seed_prompt_sha256": winner["seed_prompt_sha256"],
        "winner_prompt_sha256": winner["winner_prompt_sha256"],
        "primary_metric": paired_report["primary_metric"],
        "improvement_rule": paired_report["improvement_rule"],
        "observed_improvement_rule_satisfied": paired_report["observed_improvement_rule_satisfied"],
        "metrics": public_metrics,
        "objective_weights": config.metrics.objective_weights,
        "resource_and_cost": {
            "optimization_final_invocation_runtime": winner[
                "optimization_final_invocation_runtime"
            ],
            "optimization_local_model_receipts": winner["optimization_receipt_telemetry"],
            "paired_test_invocation_runtime": paired_report["paired_test_invocation_runtime"],
            "paired_test_local_model_receipts": paired_report["paired_receipt_telemetry"],
            "external_provider_calls": 0,
            "external_provider_cost_usd": 0.0,
            "local_hardware_energy_or_monetary_cost_measured": False,
        },
        "schemas": {
            "generation_algorithm": GENERATION_SCHEMA_ALGORITHM,
            "generation_base_sha256": OLLAMA_TASK_SCHEMA_BASE_SHA256,
            "evaluation_sha256": plan["evaluation_schema_sha256"],
            "generation_schema_avoids_regex_for_ollama_0_15_1": True,
        },
        "label_blind_passage_projection": {
            "algorithm": plan["passage_projection_algorithm"],
            "config_sha256": plan["passage_projection_config_sha256"],
            "maximum_passages": DEFAULT_RETRIEVAL_CONFIG["max_passages"],
            "maximum_total_characters": DEFAULT_RETRIEVAL_CONFIG["max_total_characters"],
            "uses_labels": False,
        },
        "lineage": {
            "config_file_sha256": sha256_file(paths.config),
            "manifest_file_sha256": sha256_file(paths.manifest),
            "test_split_jsonl_sha256": paired_report["test_split_jsonl_sha256"],
            "plan_file_sha256": sha256_file(paths.plan),
            "plan_sha256": plan["plan_sha256"],
            "winner_bundle_file_sha256": sha256_file(paths.winner),
            "winner_sha256": winner["winner_sha256"],
            "private_paired_report_sha256": paired_report["paired_report_sha256"],
            "private_paired_report_file_sha256": sha256_file(paths.paired_report),
            # This is execution provenance, not a claim that the historical
            # model calls were rerun under the current checkout.
            "source_code_sha256s": dict(sorted(historical_sources.items())),
        },
        "artifact_boundaries": {
            "contains_article_or_question_text": False,
            "contains_article_or_question_identifiers": False,
            "contains_per_example_labels": False,
            "contains_per_example_predictions": False,
            "contains_candidate_text": False,
            "contains_absolute_paths": False,
            "private_row_level_material_is_gitignored": True,
        },
        "required_caveats": [
            (
                "All Evidence Inference labels in this checkout were historically opened; this is "
                "not a pristine or confirmatory test."
            ),
            (
                "The exact official test payload was scored once only after the winner freeze in "
                "this staged run, but historical opening still precludes a held-out claim."
            ),
            (
                "The pinned llama3.2:1b runtime has 1.2B parameters; results are a local-model "
                "diagnostic and not evidence about frontier systems."
            ),
            (
                "Formal grounding checks exact quote-to-line containment and allowed section, not "
                "semantic entailment or clinical validity."
            ),
            (
                "The benchmark evaluates structured direction extraction and provenance on a "
                "conservative Evidence Inference subset, not retrieval, end-to-end claim "
                "verification, or numerical meta-analysis."
            ),
            (
                "A frozen label-blind Results passage projection bounds context length and may "
                "omit answer-bearing evidence; failures combine projection and extraction."
            ),
            (
                "GEPA used only train/development examples during optimization; test-time "
                "thresholds and scientific conclusions were not optimized."
            ),
            (
                "No improvement is claimed when the seed is retained or the paired "
                "article-clustered 95% interval includes zero."
            ),
        ],
    }
    return _add_self_hash(payload, "public_summary_sha256")


def _forbidden_public_path(value: Any, prefix: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                return path
            nested = _forbidden_public_path(item, path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _forbidden_public_path(item, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _FORBIDDEN_PUBLIC_VALUES
    ):
        return prefix
    return None


def validate_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(summary))
    _validate_self_hash(snapshot, "public_summary_sha256", "public study summary")
    if (
        snapshot.get("public_summary_version") != PUBLIC_SUMMARY_VERSION
        or snapshot.get("study_version") != STUDY_VERSION
        or snapshot.get("confirmatory_claim_allowed") is not False
        or snapshot.get("all_labels_historically_opened") is not True
        or snapshot.get("test_is_non_pristine") is not True
    ):
        raise OllamaGEPAStudyError("public study summary scope is invalid")
    boundaries = snapshot.get("artifact_boundaries")
    if not isinstance(boundaries, Mapping) or any(
        boundaries.get(key) is not False
        for key in (
            "contains_article_or_question_text",
            "contains_article_or_question_identifiers",
            "contains_per_example_labels",
            "contains_per_example_predictions",
            "contains_candidate_text",
            "contains_absolute_paths",
        )
    ):
        raise OllamaGEPAStudyError("public summary boundary declarations are invalid")
    forbidden = _forbidden_public_path(snapshot)
    if forbidden is not None:
        raise OllamaGEPAStudyError(f"public study summary leaks protected material at {forbidden}")
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(_PAIRED_METRICS):
        raise OllamaGEPAStudyError("public study summary metric set is invalid")
    return dict(summary)


def validate_private_receipts(receipt_root: Path) -> dict[str, int]:
    """Tamper-audit every private task/reflection receipt without exposing contents."""

    counts: Counter[str] = Counter()
    if not receipt_root.exists():
        return {"task": 0, "reflection": 0}
    for path in sorted(receipt_root.glob("*/*/*.json")):
        receipt = _read_json_object(path)
        request = receipt.get("request")
        if not isinstance(request, Mapping):
            raise OllamaGEPAStudyError("receipt request payload is invalid")
        kind = request.get("kind")
        if kind not in {"task", "reflection"}:
            raise OllamaGEPAStudyError("receipt kind is invalid")
        request_sha = hash_canonical(request)
        if path.name != f"{request_sha}.json":
            raise OllamaGEPAStudyError("receipt filename does not match request hash")
        _validate_receipt(receipt, request_payload=request, request_sha256=request_sha)
        counts[str(kind)] += 1
    return {"task": counts["task"], "reflection": counts["reflection"]}


def summarize_private_receipts(
    receipt_root: Path, *, namespaces: set[str] | None = None
) -> dict[str, Any]:
    """Aggregate model-reported physical-call telemetry without row content or IDs."""

    by_namespace: dict[str, Counter[str]] = defaultdict(Counter)
    if receipt_root.exists():
        for path in sorted(receipt_root.glob("*/*/*.json")):
            receipt = _read_json_object(path)
            request = receipt.get("request")
            if not isinstance(request, Mapping):
                raise OllamaGEPAStudyError("receipt request payload is invalid")
            request_sha = hash_canonical(request)
            if path.name != f"{request_sha}.json":
                raise OllamaGEPAStudyError("receipt filename does not match request hash")
            validated = _validate_receipt(
                receipt,
                request_payload=request,
                request_sha256=request_sha,
            )
            namespace = request.get("namespace")
            kind = request.get("kind")
            if not isinstance(namespace, str) or kind not in {"task", "reflection"}:
                raise OllamaGEPAStudyError("receipt telemetry identity is invalid")
            if namespaces is not None and namespace not in namespaces:
                continue
            counters = by_namespace[namespace]
            counters["physical_calls"] += 1
            counters[f"{kind}_physical_calls"] += 1
            if validated["success"] is True:
                counters["successful_calls"] += 1
                result = OllamaGenerationResult.model_validate(validated["result"])
                for field in (
                    "total_duration_ns",
                    "load_duration_ns",
                    "prompt_eval_count",
                    "prompt_eval_duration_ns",
                    "eval_count",
                    "eval_duration_ns",
                ):
                    value = getattr(result, field)
                    if value is not None:
                        counters[field] += int(value)
            else:
                counters["failed_calls"] += 1
    projected = {
        namespace: {
            key: int(counters[key])
            for key in (
                "physical_calls",
                "task_physical_calls",
                "reflection_physical_calls",
                "successful_calls",
                "failed_calls",
                "total_duration_ns",
                "load_duration_ns",
                "prompt_eval_count",
                "prompt_eval_duration_ns",
                "eval_count",
                "eval_duration_ns",
            )
        }
        for namespace, counters in sorted(by_namespace.items())
    }
    return {
        "namespaces": projected,
        "physical_calls": sum(item["physical_calls"] for item in projected.values()),
        "successful_calls": sum(item["successful_calls"] for item in projected.values()),
        "failed_calls": sum(item["failed_calls"] for item in projected.values()),
        "total_duration_ns": sum(item["total_duration_ns"] for item in projected.values()),
        "prompt_eval_count": sum(item["prompt_eval_count"] for item in projected.values()),
        "eval_count": sum(item["eval_count"] for item in projected.values()),
        "source": "self_hashed_local_ollama_receipts",
    }


__all__ = [
    "MetricSettings",
    "OllamaGEPAAdapter",
    "OllamaGEPAStudyConfig",
    "OllamaGEPAStudyError",
    "OptimizationSettings",
    "build_public_summary",
    "load_study_config",
    "prepare_optimization_plan",
    "run_optimization",
    "run_paired_test",
    "study_paths",
    "summarize_private_receipts",
    "validate_frozen_winner",
    "validate_optimization_plan",
    "validate_private_receipts",
    "validate_public_summary",
]
