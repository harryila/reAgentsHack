"""Leakage-safe GEPA prompt optimization for extraction and verification.

The scientific labels in an optimization benchmark are never rendered into a task
prompt.  They are used only by this module's evaluator.  Optimization opens the train
and development files named by a split manifest; the test file is opened only by the
separate held-out evaluation entry point.

GEPA is deliberately optional.  Importing this module does not import GEPA, initialize
an LM, read credentials, or open a network connection.  :func:`optimize_prompts` loads
the official ``gepa.optimize`` function only after all local contracts have passed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from literature_multiverse.grounding import GroundingContractError, ground_evidence
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.prompting import PromptContractError, render_prompt_text
from literature_multiverse.providers import ProviderError, ProviderProtocol

PromptKind = Literal["extraction", "quote_verification"]
SplitName = Literal["train", "dev", "test"]

PROMPT_COMPONENTS: dict[PromptKind, str] = {
    "extraction": "extraction_prompt",
    "quote_verification": "quote_verification_prompt",
}
COMPONENT_PROMPTS: dict[str, PromptKind] = {
    component: prompt_kind for prompt_kind, component in PROMPT_COMPONENTS.items()
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOKEN_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBJECTIVE_NAMES = (
    "extraction_correctness",
    "grounding_schema_validity",
    "cost_efficiency",
)
_PAIRED_WINNER_NAMESPACE = "heldout-winner"
_PAIRED_SEED_NAMESPACE = "heldout-seed"
_PAIRED_TIE_TOLERANCE = 1e-12
_SAFE_RUNTIME_METADATA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+<>-]{0,199}$")


class OptimizationContractError(ValueError):
    """Raised when a benchmark, split, candidate, or frozen winner is invalid."""


class GEPAUnavailable(RuntimeError):
    """Raised when the optional official GEPA package/API is unavailable."""


class OptimizationExample(BaseModel):
    """One labeled prompt-optimization example.

    ``label_paths`` are RFC 6901 JSON pointers into ``expected_output``.  This keeps
    scoring restricted to pre-specified scientific labels; generated rationales and
    other free text cannot silently become test targets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimization_example_version: Literal["1"] = "1"
    example_id: str
    paper_id: str
    group_id: str
    prompt_kind: PromptKind
    replacements: dict[str, str]
    expected_output: dict[str, Any]
    label_paths: list[str] = Field(min_length=1)
    output_schema: dict[str, Any]
    content_lines: dict[str, Any] | list[Any] | None = None
    line_sections: dict[str, str | None] | None = None
    source_accessible: bool = True

    @field_validator("example_id", "paper_id", "group_id")
    @classmethod
    def validate_safe_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("optimization IDs must be filesystem-safe ASCII slugs")
        return value

    @field_validator("replacements")
    @classmethod
    def validate_replacements(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("optimization replacements must not be empty")
        if any(not _TOKEN_NAME.fullmatch(key) for key in value):
            raise ValueError("optimization replacement keys must be prompt token names")
        if any(not isinstance(item, str) for item in value.values()):
            raise ValueError("optimization replacement values must be strings")
        return value

    @field_validator("label_paths")
    @classmethod
    def validate_label_paths(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("optimization label paths must be unique")
        for pointer in value:
            if pointer != "" and not pointer.startswith("/"):
                raise ValueError("optimization label paths must be RFC 6901 JSON pointers")
        return value

    @model_validator(mode="after")
    def validate_labels_and_schema(self) -> OptimizationExample:
        if self.prompt_kind == "extraction" and self.content_lines is None:
            raise ValueError("extraction optimization examples require source content lines")
        for pointer in self.label_paths:
            try:
                _json_pointer_get(self.expected_output, pointer)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"expected output is missing label path {pointer!r}") from exc
        try:
            schema_validator = validator_for(self.output_schema)
            schema_validator.check_schema(self.output_schema)
            schema_validator(self.output_schema).validate(self.expected_output)
        except (SchemaError, ValidationError) as exc:
            raise ValueError(f"expected output violates its declared schema: {exc}") from exc
        return self


class SplitArtifact(BaseModel):
    """Hash-locked metadata for one separately stored split file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    rows: int = Field(ge=1)
    example_ids: list[str] = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=1)
    group_ids: list[str] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("split paths must be normalized paths relative to the manifest")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("split sha256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def validate_sorted_unique_metadata(self) -> SplitArtifact:
        for field_name in ("example_ids", "paper_ids", "group_ids"):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(f"{field_name} must be sorted and unique")
        if self.rows != len(self.example_ids):
            raise ValueError("split row count must equal its number of example IDs")
        return self


class OptimizationSplitManifest(BaseModel):
    """Paper- and group-disjoint train/development/test manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_manifest_version: Literal["1"] = "1"
    algorithm: Literal["paper-group-components-v1", "official-paper-groups-v1"] = (
        "paper-group-components-v1"
    )
    seed: int
    train_fraction: float = Field(gt=0, lt=1)
    dev_fraction: float = Field(gt=0, lt=1)
    source_examples_sha256: str
    train: SplitArtifact
    dev: SplitArtifact
    test: SplitArtifact

    @field_validator("source_examples_sha256")
    @classmethod
    def validate_source_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("source_examples_sha256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def validate_fractions_and_leakage(self) -> OptimizationSplitManifest:
        if self.train_fraction + self.dev_fraction >= 1:
            raise ValueError("train_fraction + dev_fraction must be less than one")
        _assert_manifest_disjoint(self)
        return self


@dataclass(frozen=True, slots=True)
class OptimizationEvaluationBatch:
    """Structural implementation of GEPA's official ``EvaluationBatch`` contract."""

    outputs: list[dict[str, Any]]
    scores: list[float]
    trajectories: list[dict[str, Any]] | None = None
    objective_scores: list[dict[str, float]] | None = None
    num_metric_calls: int | None = None


@dataclass(frozen=True, slots=True)
class PromptOptimizationRun:
    """Paths and hashes for a completed, frozen optimization run."""

    run_dir: Path
    trace_path: Path
    winner_path: Path
    prompt_paths: dict[PromptKind, Path]
    winner_sha256: str


def _json_pointer_get(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if part == "-" or not part.isdigit():
                raise IndexError(part)
            current = current[int(part)]
        else:
            raise TypeError(f"cannot traverse {part!r}")
    return current


def _label_values(document: Any, label_paths: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for pointer in label_paths:
        try:
            values[pointer] = deepcopy(_json_pointer_get(document, pointer))
        except (KeyError, IndexError, TypeError, ValueError):
            values[pointer] = {"__missing__": True}
    return values


def _correctness_score(expected: Any, predicted: Any, label_paths: Sequence[str]) -> float:
    expected_labels = _label_values(expected, label_paths)
    predicted_labels = _label_values(predicted, label_paths)
    correct = sum(
        canonical_json_bytes(expected_labels[path]) == canonical_json_bytes(predicted_labels[path])
        for path in label_paths
    )
    return correct / len(label_paths)


def _validate_output_schema(output: Any, schema: Mapping[str, Any]) -> str | None:
    try:
        schema_dict = deepcopy(dict(schema))
        schema_validator = validator_for(schema_dict)
        schema_validator.check_schema(schema_dict)
        schema_validator(schema_dict).validate(output)
    except (SchemaError, ValidationError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _grounding_score(
    example: OptimizationExample, output: Mapping[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    if example.prompt_kind != "extraction":
        return 1.0, []
    raw_findings = output.get("findings")
    if not isinstance(raw_findings, list):
        return 0.0, [{"grounding_status": "invalid_findings"}]
    if not raw_findings:
        return 1.0, []
    results: list[dict[str, Any]] = []
    for finding in raw_findings:
        if not isinstance(finding, Mapping):
            results.append({"grounding_status": "invalid_finding"})
            continue
        try:
            result = ground_evidence(
                finding.get("evidence_quote"),
                finding.get("evidence_lines"),
                example.content_lines,
                line_sections=example.line_sections,
                source_accessible=example.source_accessible,
            )
        except (GroundingContractError, TypeError, ValueError) as exc:
            result = {"grounding_status": "invalid", "error": str(exc)}
        results.append(result)
    exact = sum(
        result.get("grounding_status") == "exact" and result.get("section_flagged") is False
        for result in results
    )
    return exact / len(results), results


def _parse_provider_output(parsed_json: Any, text: str) -> dict[str, Any]:
    output = deepcopy(parsed_json)
    if output is None:
        output = json.loads(text)
    if not isinstance(output, dict):
        raise OptimizationContractError("provider output root must be a JSON object")
    return output


def _candidate_sha256(candidate: Mapping[str, str]) -> str:
    if not candidate or any(not isinstance(value, str) for value in candidate.values()):
        raise OptimizationContractError("GEPA candidate must be a non-empty string mapping")
    return hash_canonical(dict(candidate))


def provider_request_key(
    example: OptimizationExample,
    candidate: Mapping[str, str],
    *,
    request_namespace: str | None = None,
) -> str:
    """Return the immutable provider request key used for an example/candidate pair.

    A namespace is required when two experimental arms could contain byte-identical
    prompts.  It prevents one-shot provider archives from colliding across arms while
    preserving the historical key format for ordinary GEPA optimization.
    """

    prefix = ""
    if request_namespace is not None:
        if not _SAFE_ID.fullmatch(request_namespace):
            raise OptimizationContractError("provider request namespace must be a safe ID")
        prefix = f"{request_namespace}-"
    return f"{prefix}{example.example_id}-{_candidate_sha256(candidate)[:16]}"


def _provider_prompt(example: OptimizationExample, rendered_template: str) -> str:
    """Attach the per-paper task input outside the optimized repository template.

    Paperclip supplies a mapped document separately from ``prompts/extraction.md`` in the
    production extractor.  The direct provider evaluator mirrors that boundary by appending
    the benchmark's source-line object; it never adds labels or expected output.
    """

    if example.prompt_kind != "extraction":
        return rendered_template
    source_json = json.dumps(
        example.content_lines,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        f"{rendered_template.rstrip()}\n\n"
        "## Paper source lines supplied by the evaluation harness\n\n"
        f"```json\n{source_json}\n```\n"
    )


class GEPAPromptAdapter:
    """Official-GEPA-compatible adapter around the repository provider boundary."""

    propose_new_texts = None

    def __init__(
        self,
        provider: ProviderProtocol,
        *,
        cost_cap_usd: float = 0.02,
        request_namespace: str | None = None,
    ) -> None:
        if not math.isfinite(cost_cap_usd) or cost_cap_usd <= 0:
            raise ValueError("cost_cap_usd must be positive and finite")
        if request_namespace is not None and not _SAFE_ID.fullmatch(request_namespace):
            raise ValueError("request_namespace must be a safe ID")
        self.provider = provider
        self.cost_cap_usd = float(cost_cap_usd)
        self.request_namespace = request_namespace
        self._cache: dict[
            tuple[str, str],
            tuple[dict[str, Any], float, dict[str, float], dict[str, Any]],
        ] = {}

    def evaluate(
        self,
        batch: list[OptimizationExample | Mapping[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> OptimizationEvaluationBatch:
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        objectives: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] = []
        candidate_sha = _candidate_sha256(candidate)
        metric_calls = 0
        for raw_example in batch:
            example = (
                raw_example
                if isinstance(raw_example, OptimizationExample)
                else OptimizationExample.model_validate(raw_example)
            )
            cache_key = (candidate_sha, example.example_id)
            cached = self._cache.get(cache_key)
            if cached is None:
                evaluated = self._evaluate_one(example, candidate, candidate_sha)
                self._cache[cache_key] = deepcopy(evaluated)
                metric_calls += 1
            else:
                evaluated = deepcopy(cached)
            output, score, objective, trajectory = evaluated
            outputs.append(output)
            scores.append(score)
            objectives.append(objective)
            trajectories.append(trajectory)
        return OptimizationEvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objectives,
            num_metric_calls=metric_calls,
        )

    def _evaluate_one(
        self,
        example: OptimizationExample,
        candidate: Mapping[str, str],
        candidate_sha: str,
    ) -> tuple[dict[str, Any], float, dict[str, float], dict[str, Any]]:
        component = PROMPT_COMPONENTS[example.prompt_kind]
        template = candidate.get(component)
        base_trajectory: dict[str, Any] = {
            "example_id": example.example_id,
            "paper_id": example.paper_id,
            "group_id": example.group_id,
            "prompt_kind": example.prompt_kind,
            "component": component,
            "candidate_sha256": candidate_sha,
            "inputs": deepcopy(example.replacements),
            "expected_labels": _label_values(example.expected_output, example.label_paths),
        }
        if not isinstance(template, str):
            return self._failed_evaluation(base_trajectory, "candidate component is missing")
        try:
            rendered, prompt_version = render_prompt_text(template, example.replacements)
        except PromptContractError as exc:
            return self._failed_evaluation(base_trajectory, f"prompt contract: {exc}")

        prompt_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        provider_prompt = _provider_prompt(example, rendered)
        request_prompt_sha = hashlib.sha256(provider_prompt.encode("utf-8")).hexdigest()
        request_key = provider_request_key(
            example,
            candidate,
            request_namespace=self.request_namespace,
        )
        base_trajectory.update(
            {
                "prompt_version": prompt_version,
                "rendered_prompt_sha256": prompt_sha,
                "request_prompt_sha256": request_prompt_sha,
                "provider_request_key": request_key,
                "request_namespace": self.request_namespace,
            }
        )
        try:
            result = self.provider.generate(
                operation=f"gepa-{example.prompt_kind}",
                request_key=request_key,
                prompt=provider_prompt,
                output_schema=deepcopy(example.output_schema),
            )
        except (ProviderError, OSError, ValueError) as exc:
            return self._failed_evaluation(base_trajectory, f"provider: {exc}")
        try:
            parsed = _parse_provider_output(result.parsed_json, result.text)
        except (json.JSONDecodeError, OptimizationContractError) as exc:
            failed = dict(base_trajectory)
            failed.update(
                {
                    "provider": result.provider,
                    "model": result.model,
                    "provider_response_id": result.response_id,
                    "estimated_cost_usd": result.estimated_cost_usd,
                }
            )
            return self._failed_evaluation(failed, f"parse: {exc}")

        correctness = _correctness_score(example.expected_output, parsed, example.label_paths)
        schema_error = _validate_output_schema(parsed, example.output_schema)
        grounding, grounding_results = _grounding_score(example, parsed)
        schema_score = float(schema_error is None)
        grounding_schema = schema_score * (0.5 + 0.5 * grounding)
        cost_efficiency = max(0.0, 1.0 - float(result.estimated_cost_usd) / self.cost_cap_usd)
        objective = {
            "extraction_correctness": correctness,
            "grounding_schema_validity": grounding_schema,
            "cost_efficiency": cost_efficiency,
        }
        scalar_score = 0.70 * correctness + 0.25 * grounding_schema + 0.05 * cost_efficiency
        output = {
            "example_id": example.example_id,
            "parsed_output": deepcopy(parsed),
            "rendered_prompt_sha256": prompt_sha,
            "request_prompt_sha256": request_prompt_sha,
            "provider_request_key": request_key,
            "request_namespace": self.request_namespace,
            "provider": result.provider,
            "model": result.model,
            "estimated_cost_usd": float(result.estimated_cost_usd),
            "schema_error": schema_error,
            "grounding_results": grounding_results,
        }
        trajectory = dict(base_trajectory)
        trajectory.update(
            {
                "generated_output": deepcopy(parsed),
                "predicted_labels": _label_values(parsed, example.label_paths),
                "schema_error": schema_error,
                "grounding_results": grounding_results,
                "objective_scores": objective,
                "scalar_score": scalar_score,
                "estimated_cost_usd": float(result.estimated_cost_usd),
            }
        )
        return output, scalar_score, objective, trajectory

    @staticmethod
    def _failed_evaluation(
        trajectory: dict[str, Any], error: str
    ) -> tuple[dict[str, Any], float, dict[str, float], dict[str, Any]]:
        objective = {
            "extraction_correctness": 0.0,
            "grounding_schema_validity": 0.0,
            "cost_efficiency": 0.0,
        }
        failed_trajectory = dict(trajectory)
        failed_trajectory.update(
            {
                "error": error,
                "generated_output": None,
                "predicted_labels": {},
                "objective_scores": objective,
                "scalar_score": 0.0,
            }
        )
        return (
            {"example_id": trajectory["example_id"], "error": error},
            0.0,
            objective,
            failed_trajectory,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: OptimizationEvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        del candidate
        unknown = sorted(set(components_to_update) - set(COMPONENT_PROMPTS))
        if unknown:
            raise OptimizationContractError(f"unknown GEPA prompt components: {unknown}")
        if eval_batch.trajectories is None:
            raise OptimizationContractError("GEPA requested reflection without captured traces")
        reflective: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            records: list[dict[str, Any]] = []
            for trajectory in eval_batch.trajectories:
                if trajectory.get("component") != component:
                    continue
                feedback = {
                    "expected_labels": trajectory.get("expected_labels"),
                    "predicted_labels": trajectory.get("predicted_labels"),
                    "schema_error": trajectory.get("schema_error"),
                    "grounding_results": trajectory.get("grounding_results", []),
                    "objective_scores": trajectory.get("objective_scores"),
                    "scalar_score": trajectory.get("scalar_score"),
                    "error": trajectory.get("error"),
                }
                records.append(
                    {
                        "Inputs": {
                            "example_id": trajectory["example_id"],
                            "prompt_kind": trajectory["prompt_kind"],
                            "replacements": trajectory["inputs"],
                        },
                        "Generated Outputs": trajectory.get("generated_output"),
                        "Feedback": json.dumps(
                            feedback,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    }
                )
            if not records:
                raise OptimizationContractError(
                    f"reflection minibatch has no examples for component {component!r}"
                )
            reflective[component] = records
        return reflective


def load_examples(path: str | Path) -> list[OptimizationExample]:
    """Load and validate a benchmark JSONL file without mutating its labels."""

    source = Path(path)
    examples: list[OptimizationExample] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise OptimizationContractError(f"cannot read optimization examples: {source}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            examples.append(OptimizationExample.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise OptimizationContractError(
                f"invalid optimization example at {source}:{line_number}"
            ) from exc
    if not examples:
        raise OptimizationContractError("optimization example file is empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise OptimizationContractError("optimization example IDs must be unique")
    _assert_uniform_prompt_contracts(examples)
    return examples


def _assert_uniform_prompt_contracts(examples: Sequence[OptimizationExample]) -> None:
    token_sets: dict[PromptKind, frozenset[str]] = {}
    for example in examples:
        observed = frozenset(example.replacements)
        expected = token_sets.setdefault(example.prompt_kind, observed)
        if observed != expected:
            raise OptimizationContractError(
                f"replacement token drift within {example.prompt_kind}: "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )


def _assert_manifest_disjoint(manifest: OptimizationSplitManifest) -> None:
    splits = {name: getattr(manifest, name) for name in ("train", "dev", "test")}
    for field_name in ("example_ids", "paper_ids", "group_ids"):
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = set(getattr(splits[left], field_name)) & set(
                getattr(splits[right], field_name)
            )
            if overlap:
                raise ValueError(
                    f"split leakage: {field_name} overlap between {left}/{right}: {sorted(overlap)}"
                )


def _connected_guard_units(examples: Sequence[OptimizationExample]) -> list[list[str]]:
    parents = {example.example_id: example.example_id for example in examples}

    def find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_paper: dict[str, str] = {}
    by_group: dict[str, str] = {}
    for example in examples:
        previous_paper = by_paper.setdefault(example.paper_id, example.example_id)
        previous_group = by_group.setdefault(example.group_id, example.example_id)
        union(example.example_id, previous_paper)
        union(example.example_id, previous_group)
    units: dict[str, list[str]] = {}
    for example in examples:
        units.setdefault(find(example.example_id), []).append(example.example_id)
    return [sorted(ids) for ids in units.values()]


def _assign_guard_units(
    examples: Sequence[OptimizationExample],
    *,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
) -> dict[SplitName, list[OptimizationExample]]:
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1:
        raise OptimizationContractError("split fractions must be between zero and one")
    if train_fraction + dev_fraction >= 1:
        raise OptimizationContractError("train_fraction + dev_fraction must be less than one")
    units = _connected_guard_units(examples)
    if len(units) < 3:
        raise OptimizationContractError(
            "at least three paper/group-disjoint connected components are required"
        )
    rng = random.Random(seed)
    tie_breakers = {tuple(unit): rng.random() for unit in sorted(units)}
    units.sort(key=lambda unit: (-len(unit), tie_breakers[tuple(unit)], unit))
    fractions: dict[SplitName, float] = {
        "train": train_fraction,
        "dev": dev_fraction,
        "test": 1.0 - train_fraction - dev_fraction,
    }
    assignments: dict[SplitName, list[str]] = {"train": [], "dev": [], "test": []}
    initial_order = sorted(fractions, key=lambda name: (-fractions[name], name))
    for split_name, unit in zip(initial_order, units[:3], strict=True):
        assignments[split_name].extend(unit)
    total = len(examples)
    for unit in units[3:]:
        split_name = max(
            ("train", "dev", "test"),
            key=lambda name: (
                fractions[name] * total - len(assignments[name]),
                fractions[name],
                name,
            ),
        )
        assignments[split_name].extend(unit)
    lookup = {example.example_id: example for example in examples}
    return {
        name: [lookup[example_id] for example_id in sorted(ids)]
        for name, ids in assignments.items()
    }


def _split_artifact(path: Path, examples: Sequence[OptimizationExample]) -> SplitArtifact:
    return SplitArtifact(
        path=path.name,
        sha256=sha256_file(path),
        rows=len(examples),
        example_ids=sorted(example.example_id for example in examples),
        paper_ids=sorted({example.paper_id for example in examples}),
        group_ids=sorted({example.group_id for example in examples}),
    )


def create_split_bundle(
    source_examples_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> Path:
    """Create immutable, separately stored train/dev/test JSONL files and manifest."""

    source = Path(source_examples_path)
    destination = Path(output_dir)
    if destination.exists():
        raise OptimizationContractError(f"split output directory already exists: {destination}")
    examples = load_examples(source)
    assigned = _assign_guard_units(
        examples,
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )
    destination.mkdir(parents=True, exist_ok=False)
    split_paths: dict[SplitName, Path] = {}
    for split_name in ("train", "dev", "test"):
        split_path = destination / f"{split_name}.jsonl"
        atomic_write_jsonl(
            split_path,
            [example.model_dump(mode="json") for example in assigned[split_name]],
        )
        split_paths[split_name] = split_path
    manifest = OptimizationSplitManifest(
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        source_examples_sha256=sha256_file(source),
        train=_split_artifact(split_paths["train"], assigned["train"]),
        dev=_split_artifact(split_paths["dev"], assigned["dev"]),
        test=_split_artifact(split_paths["test"], assigned["test"]),
    )
    manifest_path = destination / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def load_split_manifest(path: str | Path) -> OptimizationSplitManifest:
    source = Path(path)
    try:
        return OptimizationSplitManifest.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OptimizationContractError(f"invalid optimization split manifest: {source}") from exc


def load_manifest_split(
    manifest_path: str | Path, split_name: SplitName
) -> list[OptimizationExample]:
    """Open exactly one split and reconcile it to the hash-locked manifest metadata."""

    manifest_source = Path(manifest_path)
    manifest = load_split_manifest(manifest_source)
    artifact = getattr(manifest, split_name)
    split_path = manifest_source.parent / artifact.path
    if sha256_file(split_path) != artifact.sha256:
        raise OptimizationContractError(f"{split_name} split hash mismatch")
    examples = load_examples(split_path)
    observed = SplitArtifact(
        path=artifact.path,
        sha256=artifact.sha256,
        rows=len(examples),
        example_ids=sorted(example.example_id for example in examples),
        paper_ids=sorted({example.paper_id for example in examples}),
        group_ids=sorted({example.group_id for example in examples}),
    )
    if observed != artifact:
        raise OptimizationContractError(f"{split_name} split metadata mismatch")
    return examples


def load_optimization_examples(
    manifest_path: str | Path,
) -> tuple[list[OptimizationExample], list[OptimizationExample]]:
    """Load train and dev only.  This function never opens or hashes the test file."""

    manifest = load_split_manifest(manifest_path)
    _assert_manifest_disjoint(manifest)
    train = load_manifest_split(manifest_path, "train")
    dev = load_manifest_split(manifest_path, "dev")
    _assert_uniform_prompt_contracts([*train, *dev])
    return train, dev


def _load_official_gepa_optimize() -> tuple[Callable[..., Any], str]:
    try:
        gepa = importlib.import_module("gepa")
    except ImportError as exc:
        raise GEPAUnavailable(
            "official GEPA is not installed; install the project GEPA extra or `pip install gepa`"
        ) from exc
    optimize = getattr(gepa, "optimize", None)
    if not callable(optimize):
        raise GEPAUnavailable("installed GEPA does not expose the official `gepa.optimize` API")
    try:
        version = importlib.metadata.version("gepa")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return optimize, version


def _load_official_gepa_lm_types() -> tuple[type[Any], type[Any]]:
    """Load GEPA's own tracked LM wrappers without constructing or calling an LM."""

    try:
        lm_module = importlib.import_module("gepa.lm")
    except ImportError as exc:
        raise GEPAUnavailable(
            "official GEPA is not installed; install the project GEPA extra or `pip install gepa`"
        ) from exc
    lm_type = getattr(lm_module, "LM", None)
    tracking_lm_type = getattr(lm_module, "TrackingLM", None)
    if not isinstance(lm_type, type) or not isinstance(tracking_lm_type, type):
        raise GEPAUnavailable("installed GEPA does not expose its tracked LM API")
    return lm_type, tracking_lm_type


def _prepare_official_reflection_lm(
    reflection_lm: str | Callable[..., Any],
    reflection_lm_kwargs: Mapping[str, Any] | None,
) -> Any:
    """Create one retained, tracked reflection-LM scope for a single prompt kind."""

    lm_type, tracking_lm_type = _load_official_gepa_lm_types()
    if isinstance(reflection_lm, str):
        return lm_type(reflection_lm, **deepcopy(dict(reflection_lm_kwargs or {})))
    if all(
        hasattr(reflection_lm, attribute)
        for attribute in ("total_cost", "total_tokens_in", "total_tokens_out")
    ):
        return _ReflectionLMUsageWindow(reflection_lm)
    return tracking_lm_type(reflection_lm)


class _ReflectionLMUsageWindow:
    """Expose per-component deltas from a caller-owned cumulative tracked LM."""

    def __init__(self, reflection_lm: Callable[..., Any]) -> None:
        self._reflection_lm = reflection_lm
        baseline = _reflection_lm_usage(reflection_lm)
        if baseline["status"] != "available":
            raise ValueError("tracked reflection callable has invalid usage counters")
        self._baseline_cost = float(baseline["total_cost_usd"])
        self._baseline_tokens_in = int(baseline["total_input_tokens"])
        self._baseline_tokens_out = int(baseline["total_output_tokens"])

    @property
    def total_cost(self) -> float:
        return float(self._reflection_lm.total_cost) - self._baseline_cost

    @property
    def total_tokens_in(self) -> int:
        return int(self._reflection_lm.total_tokens_in) - self._baseline_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return int(self._reflection_lm.total_tokens_out) - self._baseline_tokens_out

    def __call__(self, prompt: Any) -> Any:
        return self._reflection_lm(prompt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reflection_lm, name)


def _stable_type_qualname(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        value_type = type(value)
        module = value_type.__module__
        qualname = value_type.__qualname__
    label = f"{module}.{qualname}"
    if not _SAFE_RUNTIME_METADATA.fullmatch(label):
        raise ValueError("runtime identity label contains unsupported characters")
    return label


def _reflection_lm_identity(
    reflection_lm: str | Callable[..., Any],
) -> dict[str, str]:
    if isinstance(reflection_lm, str):
        if not _SAFE_RUNTIME_METADATA.fullmatch(reflection_lm):
            raise ValueError("reflection LM model identifier is not safe metadata")
        return {"kind": "model", "model": reflection_lm}
    return {
        "kind": "callable",
        "type_qualname": _stable_type_qualname(reflection_lm),
    }


def _task_provider_identity(
    provider: ProviderProtocol,
    supplied: Mapping[str, Any] | None,
    *,
    cost_cap_usd: float,
) -> dict[str, Any]:
    if supplied is None:
        return {
            "status": "not_supplied",
            "provider_type_qualname": _stable_type_qualname(provider),
            "cost_efficiency_cap_usd": cost_cap_usd,
        }
    required = {"provider", "model", "effort", "max_tokens"}
    optional = {"max_budget_usd"}
    if set(supplied) - required - optional or required - set(supplied):
        raise ValueError("task_provider_identity_has_invalid_fields")
    normalized: dict[str, Any] = {"status": "supplied"}
    for field_name in ("provider", "model", "effort"):
        value = supplied[field_name]
        if not isinstance(value, str) or not _SAFE_RUNTIME_METADATA.fullmatch(value):
            raise ValueError(f"task_provider_identity_{field_name}_is_invalid")
        normalized[field_name] = value
    normalized["max_tokens"] = supplied["max_tokens"]
    if (
        isinstance(normalized["max_tokens"], bool)
        or not isinstance(normalized["max_tokens"], int)
        or normalized["max_tokens"] < 1
    ):
        raise ValueError("task_provider_identity_max_tokens_is_invalid")
    if "max_budget_usd" in supplied:
        max_budget_usd = supplied["max_budget_usd"]
        if (
            isinstance(max_budget_usd, bool)
            or not isinstance(max_budget_usd, (int, float))
            or not math.isfinite(float(max_budget_usd))
            or float(max_budget_usd) <= 0
        ):
            raise ValueError("task_provider_identity_max_budget_usd_is_invalid")
        normalized["max_budget_usd"] = float(max_budget_usd)
    normalized["cost_efficiency_cap_usd"] = cost_cap_usd
    return normalized


def _combined_budget_preflight(
    supplied: Mapping[str, Any] | None,
) -> dict[str, float | int] | None:
    if supplied is None:
        return None
    expected = {
        "planning_ceiling_usd",
        "existing_archived_provider_ceiling_usd",
        "task_rollout_ceiling_usd",
        "active_prompt_kinds",
        "reflection_stop_ceiling_usd",
        "reflection_batch_headroom_usd",
        "projected_combined_ceiling_usd",
        "task_provider_global_limit_usd",
    }
    if set(supplied) != expected:
        raise ValueError("combined_budget_preflight_has_invalid_fields")
    normalized: dict[str, float | int] = {}
    for field_name in sorted(expected):
        value = supplied[field_name]
        if field_name == "active_prompt_kinds":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("combined_budget_preflight_active_kinds_is_invalid")
            normalized[field_name] = value
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"combined_budget_preflight_{field_name}_is_invalid")
            normalized[field_name] = float(value)
    return normalized


def _reflection_lm_usage(reflection_lm: Any | None) -> dict[str, Any]:
    """Return strict cumulative telemetry without estimating missing historical data."""

    unavailable = {
        "reflection_lm_usage_version": "1",
        "status": "unavailable",
        "total_cost_usd": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
    }
    if reflection_lm is None:
        return unavailable
    try:
        cost = reflection_lm.total_cost
        input_tokens = reflection_lm.total_tokens_in
        output_tokens = reflection_lm.total_tokens_out
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return unavailable
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        return unavailable
    return {
        "reflection_lm_usage_version": "1",
        "status": "available",
        "total_cost_usd": float(cost),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "tracker": f"{type(reflection_lm).__module__}.{type(reflection_lm).__qualname__}",
    }


def _aggregate_reflection_lm_usage(
    usage_by_prompt_kind: Mapping[PromptKind, Mapping[str, Any]],
) -> dict[str, Any]:
    by_prompt_kind = {
        prompt_kind: deepcopy(dict(usage_by_prompt_kind[prompt_kind]))
        for prompt_kind in sorted(usage_by_prompt_kind)
    }
    unavailable = {
        "reflection_lm_usage_version": "1",
        "status": "unavailable",
        "total_cost_usd": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "by_prompt_kind": by_prompt_kind,
    }
    if not by_prompt_kind or any(
        usage["status"] != "available" for usage in by_prompt_kind.values()
    ):
        return unavailable
    return {
        "reflection_lm_usage_version": "1",
        "status": "available",
        "total_cost_usd": math.fsum(
            float(usage["total_cost_usd"]) for usage in by_prompt_kind.values()
        ),
        "total_input_tokens": sum(
            int(usage["total_input_tokens"]) for usage in by_prompt_kind.values()
        ),
        "total_output_tokens": sum(
            int(usage["total_output_tokens"]) for usage in by_prompt_kind.values()
        ),
        "by_prompt_kind": by_prompt_kind,
    }


def _result_trace(result: Any, component: str) -> tuple[str, dict[str, Any]]:
    best_candidate = getattr(result, "best_candidate", None)
    if not isinstance(best_candidate, Mapping) or not isinstance(
        best_candidate.get(component), str
    ):
        raise OptimizationContractError("GEPA result has no valid best prompt candidate")
    candidates = getattr(result, "candidates", None)
    parents = getattr(result, "parents", None)
    val_scores = getattr(result, "val_aggregate_scores", None)
    if (
        not isinstance(candidates, list)
        or not isinstance(parents, list)
        or not isinstance(val_scores, list)
    ):
        raise OptimizationContractError("GEPA result is missing immutable candidate trace fields")
    best_idx = int(result.best_idx)
    trace = {
        "component": component,
        "best_idx": best_idx,
        "best_score": float(val_scores[best_idx]),
        "best_candidate_sha256": hash_canonical(dict(best_candidate)),
        "candidates": [deepcopy(dict(candidate)) for candidate in candidates],
        "candidate_sha256s": [hash_canonical(dict(candidate)) for candidate in candidates],
        "parents": deepcopy(parents),
        "val_aggregate_scores": deepcopy(val_scores),
        "val_aggregate_subscores": deepcopy(getattr(result, "val_aggregate_subscores", None)),
        "per_objective_best_candidates": _json_safe(
            getattr(result, "per_objective_best_candidates", None)
        ),
        "objective_pareto_front": _json_safe(getattr(result, "objective_pareto_front", None)),
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "seed": getattr(result, "seed", None),
    }
    return str(best_candidate[component]), trace


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def optimize_prompts(
    *,
    manifest_path: str | Path,
    seed_templates: Mapping[PromptKind, str | Path],
    provider: ProviderProtocol,
    run_dir: str | Path,
    reflection_lm: str | Callable[..., Any],
    max_metric_calls_per_prompt: int,
    max_reflection_cost_usd_per_prompt: float,
    seed: int,
    cost_cap_usd: float = 0.02,
    reflection_minibatch_size: int = 3,
    reflection_lm_kwargs: Mapping[str, Any] | None = None,
    task_provider_identity: Mapping[str, Any] | None = None,
    combined_budget_preflight: Mapping[str, Any] | None = None,
    optimize_fn: Callable[..., Any] | None = None,
) -> PromptOptimizationRun:
    """Optimize each prompt kind with the official GEPA API and freeze both winners.

    The optional ``optimize_fn`` is a narrow dependency-injection seam for contract tests.
    Production callers leave it unset, which dynamically loads ``gepa.optimize``.  No
    fallback or home-grown optimizer exists.
    """

    if max_metric_calls_per_prompt < 1:
        raise ValueError("max_metric_calls_per_prompt must be positive")
    if (
        not math.isfinite(max_reflection_cost_usd_per_prompt)
        or max_reflection_cost_usd_per_prompt <= 0
    ):
        raise ValueError("max_reflection_cost_usd_per_prompt must be positive and finite")
    if reflection_minibatch_size < 1:
        raise ValueError("reflection_minibatch_size must be positive")
    if not math.isfinite(cost_cap_usd) or cost_cap_usd <= 0:
        raise ValueError("cost_cap_usd must be positive and finite")
    if reflection_lm_kwargs is not None:
        if not isinstance(reflection_lm, str):
            raise ValueError("reflection_lm_kwargs_require_string_model")
        unknown_kwargs = sorted(
            set(reflection_lm_kwargs) - {"max_tokens", "num_retries", "temperature"}
        )
        if unknown_kwargs:
            raise ValueError(f"unsupported_reflection_lm_kwargs:{unknown_kwargs}")
        max_tokens = reflection_lm_kwargs.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("reflection_max_tokens_must_be_positive_integer")
        if reflection_lm_kwargs.get("num_retries") != 0:
            raise ValueError("reflection_lm_num_retries_must_be_zero")
        temperature = reflection_lm_kwargs.get("temperature")
        if temperature is not None and (
            not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
        ):
            raise ValueError("reflection_temperature_must_be_between_zero_and_two")
    destination = Path(run_dir)
    if destination.exists():
        raise OptimizationContractError(f"optimization run directory already exists: {destination}")
    manifest_source = Path(manifest_path)
    manifest = load_split_manifest(manifest_source)
    train, dev = load_optimization_examples(manifest_source)
    active_kinds = sorted({example.prompt_kind for example in [*train, *dev]})
    missing_templates = sorted(set(active_kinds) - set(seed_templates))
    if missing_templates:
        raise OptimizationContractError(f"seed templates are missing: {missing_templates}")
    template_texts: dict[PromptKind, str] = {}
    for prompt_kind in active_kinds:
        template_texts[prompt_kind] = Path(seed_templates[prompt_kind]).read_text(encoding="utf-8")
    seed_prompt_hashes = {
        kind: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for kind, text in template_texts.items()
    }
    for example in [*train, *dev]:
        try:
            render_prompt_text(template_texts[example.prompt_kind], example.replacements)
        except PromptContractError as exc:
            raise OptimizationContractError(
                f"seed template does not render for {example.example_id}: {exc}"
            ) from exc

    for prompt_kind in active_kinds:
        train_kind = [example for example in train if example.prompt_kind == prompt_kind]
        dev_kind = [example for example in dev if example.prompt_kind == prompt_kind]
        if not train_kind or not dev_kind:
            raise OptimizationContractError(
                f"{prompt_kind} must have at least one train and one dev example"
            )

    reflection_identity = _reflection_lm_identity(reflection_lm)
    normalized_task_provider_identity = _task_provider_identity(
        provider,
        task_provider_identity,
        cost_cap_usd=cost_cap_usd,
    )
    task_provider_identity_sha256 = hash_canonical(normalized_task_provider_identity)
    normalized_budget_preflight = _combined_budget_preflight(combined_budget_preflight)
    retained_reflection_lms: dict[PromptKind, Any] = {}
    reflection_usage_by_kind: dict[PromptKind, dict[str, Any]] = {}
    if optimize_fn is None:
        optimize_callable, gepa_version = _load_official_gepa_optimize()
        uses_official_runtime = True
    else:
        optimize_callable, gepa_version = optimize_fn, "injected-contract-test"
        uses_official_runtime = False

    destination.mkdir(parents=True, exist_ok=False)
    adapter = GEPAPromptAdapter(provider, cost_cap_usd=cost_cap_usd)
    winning_prompts = dict(template_texts)
    component_traces: dict[str, Any] = {}
    for index, prompt_kind in enumerate(active_kinds):
        component = PROMPT_COMPONENTS[prompt_kind]
        train_kind = [example for example in train if example.prompt_kind == prompt_kind]
        dev_kind = [example for example in dev if example.prompt_kind == prompt_kind]
        if uses_official_runtime:
            reflection_lm_for_gepa = _prepare_official_reflection_lm(
                reflection_lm, reflection_lm_kwargs
            )
            retained_reflection_lms[prompt_kind] = reflection_lm_for_gepa
            reflection_kwargs_for_gepa = None
        else:
            reflection_lm_for_gepa = reflection_lm
            reflection_kwargs_for_gepa = (
                deepcopy(dict(reflection_lm_kwargs))
                if isinstance(reflection_lm, str) and reflection_lm_kwargs is not None
                else None
            )
        result = optimize_callable(
            seed_candidate={component: template_texts[prompt_kind]},
            trainset=train_kind,
            valset=dev_kind,
            adapter=adapter,
            reflection_lm=reflection_lm_for_gepa,
            reflection_lm_kwargs=reflection_kwargs_for_gepa,
            max_metric_calls=max_metric_calls_per_prompt,
            max_reflection_cost=max_reflection_cost_usd_per_prompt,
            reflection_minibatch_size=min(reflection_minibatch_size, len(train_kind)),
            seed=seed + index,
            run_dir=str(destination / prompt_kind),
            use_merge=True,
            # GEPA 0.1.4 assigns independent integer IDs starting at zero to train
            # and validation loaders, while its cache key omits the split.  Enabling
            # the library cache can therefore reuse a train score as a validation
            # score.  The adapter already caches safely by candidate hash plus the
            # globally unique example_id, so the GEPA-level cache must stay disabled.
            cache_evaluation=False,
            track_best_outputs=True,
            display_progress_bar=False,
        )
        if uses_official_runtime:
            reflection_usage_by_kind[prompt_kind] = _reflection_lm_usage(reflection_lm_for_gepa)
        winner, result_trace = _result_trace(result, component)
        winning_prompts[prompt_kind] = winner
        component_traces[prompt_kind] = result_trace

    if uses_official_runtime and (
        set(retained_reflection_lms) != set(active_kinds)
        or set(reflection_usage_by_kind) != set(active_kinds)
    ):
        raise OptimizationContractError("reflection LM telemetry is incomplete")

    prompt_paths: dict[PromptKind, Path] = {}
    prompt_hashes: dict[PromptKind, str] = {}
    for prompt_kind, prompt_text in winning_prompts.items():
        path = destination / f"frozen_{prompt_kind}.md"
        atomic_write_text(path, prompt_text)
        prompt_paths[prompt_kind] = path
        prompt_hashes[prompt_kind] = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    manifest_sha = sha256_file(manifest_source)
    run_identity = hash_canonical(
        {
            "manifest_sha256": manifest_sha,
            "seed_prompt_sha256s": seed_prompt_hashes,
            "optimizer_seed": seed,
            "max_metric_calls_per_prompt": max_metric_calls_per_prompt,
            "max_reflection_cost_usd_per_prompt": max_reflection_cost_usd_per_prompt,
            "reflection_minibatch_size": reflection_minibatch_size,
            "reflection_lm_kwargs": (
                dict(reflection_lm_kwargs)
                if isinstance(reflection_lm, str) and reflection_lm_kwargs is not None
                else None
            ),
            "reflection_lm_identity": reflection_identity,
            "task_provider_identity_sha256": task_provider_identity_sha256,
            "cost_cap_usd": cost_cap_usd,
            "gepa_cache_evaluation": False,
        }
    )
    trace = {
        "optimization_trace_version": "1",
        "run_id": f"gepa-{run_identity[:20]}",
        "optimizer": "gepa.optimize",
        "gepa_version": gepa_version,
        "manifest_path": manifest_source.as_posix(),
        "manifest_sha256": manifest_sha,
        "manifest_seed": manifest.seed,
        "optimizer_seed": seed,
        "optimization_splits": ["train", "dev"],
        "test_split_opened": False,
        "test_evaluated": False,
        "objective_weights": {
            "extraction_correctness": 0.70,
            "grounding_schema_validity": 0.25,
            "cost_efficiency": 0.05,
        },
        "cost_cap_usd": cost_cap_usd,
        "gepa_cache_evaluation": False,
        "evaluation_cache_key_basis": "adapter_candidate_sha256_plus_example_id",
        "max_metric_calls_per_prompt": max_metric_calls_per_prompt,
        "max_reflection_cost_usd_per_prompt": max_reflection_cost_usd_per_prompt,
        "reflection_minibatch_size": reflection_minibatch_size,
        "reflection_lm_kwargs": (
            dict(reflection_lm_kwargs)
            if isinstance(reflection_lm, str) and reflection_lm_kwargs is not None
            else None
        ),
        "reflection_lm_identity": reflection_identity,
        "task_provider_identity": normalized_task_provider_identity,
        "task_provider_identity_sha256": task_provider_identity_sha256,
        "combined_budget_preflight": normalized_budget_preflight,
        "reflection_lm_usage": _aggregate_reflection_lm_usage(reflection_usage_by_kind),
        "train_example_ids": manifest.train.example_ids,
        "dev_example_ids": manifest.dev.example_ids,
        "component_traces": component_traces,
        "seed_prompt_sha256s": seed_prompt_hashes,
        "winning_prompt_sha256s": prompt_hashes,
    }
    trace_path = destination / "optimization_trace.json"
    atomic_write_json(trace_path, trace)
    winner = {
        "frozen_prompt_bundle_version": "1",
        "run_id": trace["run_id"],
        "optimizer": "gepa.optimize",
        "gepa_version": gepa_version,
        "manifest_sha256": manifest_sha,
        "optimization_trace_sha256": sha256_file(trace_path),
        "seed_prompt_sha256s": seed_prompt_hashes,
        "test_evaluated_at_freeze": False,
        "prompts": {
            prompt_kind: {
                "path": path.name,
                "sha256": prompt_hashes[prompt_kind],
            }
            for prompt_kind, path in prompt_paths.items()
        },
    }
    winner_path = destination / "frozen_winner.json"
    atomic_write_json(winner_path, winner)
    return PromptOptimizationRun(
        run_dir=destination,
        trace_path=trace_path,
        winner_path=winner_path,
        prompt_paths=prompt_paths,
        winner_sha256=sha256_file(winner_path),
    )


def load_frozen_prompts(winner_path: str | Path) -> dict[PromptKind, str]:
    """Load a frozen winner only when every prompt hash still matches."""

    source = Path(winner_path)
    try:
        winner = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OptimizationContractError(f"invalid frozen prompt bundle: {source}") from exc
    if winner.get("frozen_prompt_bundle_version") != "1":
        raise OptimizationContractError("unsupported frozen prompt bundle version")
    raw_prompts = winner.get("prompts")
    if not isinstance(raw_prompts, Mapping) or not raw_prompts:
        raise OptimizationContractError("frozen prompt bundle has no prompts")
    prompts: dict[PromptKind, str] = {}
    for raw_kind, metadata in raw_prompts.items():
        if raw_kind not in PROMPT_COMPONENTS or not isinstance(metadata, Mapping):
            raise OptimizationContractError(f"invalid frozen prompt entry: {raw_kind!r}")
        path_value = metadata.get("path")
        expected_sha = metadata.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha, str):
            raise OptimizationContractError("frozen prompt path/hash pair is invalid")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise OptimizationContractError(
                "frozen prompt path must remain inside its run directory"
            )
        prompt_path = source.parent / relative
        prompt_text = prompt_path.read_text(encoding="utf-8")
        observed_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if observed_sha != expected_sha:
            raise OptimizationContractError(f"frozen prompt hash mismatch: {raw_kind}")
        prompts[raw_kind] = prompt_text
    return prompts


def evaluate_frozen_test(
    *,
    manifest_path: str | Path,
    winner_path: str | Path,
    provider: ProviderProtocol,
    output_path: str | Path,
    cost_cap_usd: float = 0.02,
) -> Path:
    """Evaluate a frozen winner once on the test split held out from optimization."""

    target = Path(output_path)
    if target.exists():
        raise OptimizationContractError(f"held-out report already exists: {target}")
    manifest_source = Path(manifest_path)
    manifest_sha = sha256_file(manifest_source)
    winner_source = Path(winner_path)
    try:
        winner = json.loads(winner_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OptimizationContractError(f"invalid frozen prompt bundle: {winner_source}") from exc
    if not isinstance(winner, Mapping):
        raise OptimizationContractError(f"invalid frozen prompt bundle: {winner_source}")
    if winner.get("manifest_sha256") != manifest_sha:
        raise OptimizationContractError("frozen winner was selected under a different manifest")
    if winner.get("test_evaluated_at_freeze") is not False:
        raise OptimizationContractError(
            "winner does not certify that test labels were unopened at freeze"
        )
    prompts = load_frozen_prompts(winner_source)
    test_examples = load_manifest_split(manifest_source, "test")
    missing = sorted({example.prompt_kind for example in test_examples} - set(prompts))
    if missing:
        raise OptimizationContractError(f"frozen winner is missing test prompt kinds: {missing}")
    adapter = GEPAPromptAdapter(provider, cost_cap_usd=cost_cap_usd)
    results: dict[str, Any] = {}
    for prompt_kind in sorted({example.prompt_kind for example in test_examples}):
        component = PROMPT_COMPONENTS[prompt_kind]
        batch = [example for example in test_examples if example.prompt_kind == prompt_kind]
        evaluation = adapter.evaluate(
            batch, {component: prompts[prompt_kind]}, capture_traces=False
        )
        objective_means = {
            objective: sum(scores[objective] for scores in evaluation.objective_scores or [])
            / len(batch)
            for objective in (
                "extraction_correctness",
                "grounding_schema_validity",
                "cost_efficiency",
            )
        }
        results[prompt_kind] = {
            "rows": len(batch),
            "mean_scalar_score": sum(evaluation.scores) / len(batch),
            "mean_objective_scores": objective_means,
            "outputs": evaluation.outputs,
        }
    report = {
        "heldout_evaluation_version": "1",
        "winner_sha256": sha256_file(winner_source),
        "manifest_sha256": manifest_sha,
        "split": "test",
        "test_example_ids": sorted(example.example_id for example in test_examples),
        "results": results,
    }
    atomic_write_json(target, report)
    return target


def _load_explicit_seed_templates(
    seed_templates: Mapping[PromptKind, str | Path],
    *,
    expected_hashes: Mapping[str, Any],
) -> tuple[dict[PromptKind, str], dict[PromptKind, dict[str, str]]]:
    if not seed_templates:
        raise OptimizationContractError("paired test requires explicit seed templates")
    unknown = sorted(set(seed_templates) - set(PROMPT_COMPONENTS))
    if unknown:
        raise OptimizationContractError(f"unknown seed prompt kinds: {unknown}")
    expected_kinds = set(expected_hashes)
    if any(
        kind not in PROMPT_COMPONENTS
        or not isinstance(expected_hashes[kind], str)
        or not _SHA256.fullmatch(expected_hashes[kind])
        for kind in expected_kinds
    ):
        raise OptimizationContractError("frozen winner seed prompt hashes are invalid")
    if set(seed_templates) != expected_kinds:
        missing = sorted(expected_kinds - set(seed_templates))
        extra = sorted(set(seed_templates) - expected_kinds)
        raise OptimizationContractError(
            f"explicit seed template set mismatch: missing={missing}:extra={extra}"
        )

    texts: dict[PromptKind, str] = {}
    metadata: dict[PromptKind, dict[str, str]] = {}
    for prompt_kind in sorted(seed_templates):
        source = Path(seed_templates[prompt_kind])
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise OptimizationContractError(
                f"explicit seed template is unreadable: {prompt_kind}:{source}"
            ) from exc
        observed_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if observed_sha != expected_hashes[prompt_kind]:
            raise OptimizationContractError(
                f"explicit seed template hash does not match frozen optimization: {prompt_kind}"
            )
        texts[prompt_kind] = text
        metadata[prompt_kind] = {
            "path": source.as_posix(),
            "sha256": observed_sha,
        }
    return texts, metadata


def _paired_row(
    *,
    example: OptimizationExample,
    winner_score: float,
    seed_score: float,
    winner_objectives: Mapping[str, float],
    seed_objectives: Mapping[str, float],
    winner_output: Mapping[str, Any],
    seed_output: Mapping[str, Any],
) -> dict[str, Any]:
    scalar_delta = float(winner_score - seed_score)
    if scalar_delta > _PAIRED_TIE_TOLERANCE:
        outcome = "win"
    elif scalar_delta < -_PAIRED_TIE_TOLERANCE:
        outcome = "loss"
    else:
        outcome = "tie"
    objective_delta = {
        objective: float(winner_objectives[objective] - seed_objectives[objective])
        for objective in _OBJECTIVE_NAMES
    }
    return {
        "example_id": example.example_id,
        "paper_id": example.paper_id,
        "group_id": example.group_id,
        "prompt_kind": example.prompt_kind,
        "winner": {
            "scalar_score": float(winner_score),
            "objective_scores": dict(winner_objectives),
            "output": deepcopy(dict(winner_output)),
        },
        "seed": {
            "scalar_score": float(seed_score),
            "objective_scores": dict(seed_objectives),
            "output": deepcopy(dict(seed_output)),
        },
        "paired_delta_winner_minus_seed": {
            "scalar_score": scalar_delta,
            "objective_scores": objective_delta,
        },
        "winner_outcome": outcome,
    }


def _summarize_paired_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise OptimizationContractError("paired held-out summary cannot be empty")
    count = len(rows)
    winner_scalars = [float(row["winner"]["scalar_score"]) for row in rows]
    seed_scalars = [float(row["seed"]["scalar_score"]) for row in rows]
    scalar_deltas = [float(row["paired_delta_winner_minus_seed"]["scalar_score"]) for row in rows]
    objectives: dict[str, dict[str, float]] = {}
    for objective in _OBJECTIVE_NAMES:
        winner_values = [float(row["winner"]["objective_scores"][objective]) for row in rows]
        seed_values = [float(row["seed"]["objective_scores"][objective]) for row in rows]
        deltas = [
            float(row["paired_delta_winner_minus_seed"]["objective_scores"][objective])
            for row in rows
        ]
        objectives[objective] = {
            "winner_mean": sum(winner_values) / count,
            "seed_mean": sum(seed_values) / count,
            "mean_paired_delta_winner_minus_seed": sum(deltas) / count,
        }
    return {
        "rows": count,
        "scalar_score": {
            "winner_mean": sum(winner_scalars) / count,
            "seed_mean": sum(seed_scalars) / count,
            "mean_paired_delta_winner_minus_seed": sum(scalar_deltas) / count,
        },
        "objective_scores": objectives,
        "winner_win_tie_loss": {
            outcome: sum(row["winner_outcome"] == outcome for row in rows)
            for outcome in ("win", "tie", "loss")
        },
        "tie_tolerance": _PAIRED_TIE_TOLERANCE,
    }


def compare_frozen_test_to_seed(
    *,
    manifest_path: str | Path,
    winner_path: str | Path,
    seed_templates: Mapping[PromptKind, str | Path],
    provider: ProviderProtocol,
    output_path: str | Path,
    cost_cap_usd: float = 0.02,
) -> Path:
    """Paired held-out evaluation of a frozen GEPA winner versus its exact seed.

    The winner and explicitly supplied original seed templates are hash-validated
    before this function opens the test split.  Each arm receives the same examples
    and schemas.  Arm-specific provider request namespaces prevent archive collisions
    even when the winner is byte-identical to the seed.
    """

    target = Path(output_path)
    if target.exists():
        raise OptimizationContractError(f"paired held-out report already exists: {target}")
    manifest_source = Path(manifest_path)
    manifest = load_split_manifest(manifest_source)
    manifest_sha = sha256_file(manifest_source)
    winner_source = Path(winner_path)
    try:
        winner_metadata = json.loads(winner_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OptimizationContractError(f"invalid frozen prompt bundle: {winner_source}") from exc
    if not isinstance(winner_metadata, Mapping):
        raise OptimizationContractError(f"invalid frozen prompt bundle: {winner_source}")
    if winner_metadata.get("manifest_sha256") != manifest_sha:
        raise OptimizationContractError("frozen winner was selected under a different manifest")
    if winner_metadata.get("test_evaluated_at_freeze") is not False:
        raise OptimizationContractError(
            "winner does not certify that test labels were unopened at freeze"
        )
    raw_seed_hashes = winner_metadata.get("seed_prompt_sha256s")
    if not isinstance(raw_seed_hashes, Mapping) or not raw_seed_hashes:
        raise OptimizationContractError("frozen winner does not record original seed prompt hashes")
    winner_prompts = load_frozen_prompts(winner_source)
    seed_prompts, seed_metadata = _load_explicit_seed_templates(
        seed_templates,
        expected_hashes=raw_seed_hashes,
    )
    winner_adapter = GEPAPromptAdapter(
        provider,
        cost_cap_usd=cost_cap_usd,
        request_namespace=_PAIRED_WINNER_NAMESPACE,
    )
    seed_adapter = GEPAPromptAdapter(
        provider,
        cost_cap_usd=cost_cap_usd,
        request_namespace=_PAIRED_SEED_NAMESPACE,
    )

    # This is deliberately the first test-artifact access in the comparison path.
    test_examples = load_manifest_split(manifest_source, "test")
    active_kinds = sorted({example.prompt_kind for example in test_examples})
    missing_winner = sorted(set(active_kinds) - set(winner_prompts))
    missing_seed = sorted(set(active_kinds) - set(seed_prompts))
    if missing_winner:
        raise OptimizationContractError(
            f"frozen winner is missing test prompt kinds: {missing_winner}"
        )
    if missing_seed:
        raise OptimizationContractError(
            f"explicit seed is missing test prompt kinds: {missing_seed}"
        )
    for example in test_examples:
        for arm, template in (
            ("winner", winner_prompts[example.prompt_kind]),
            ("seed", seed_prompts[example.prompt_kind]),
        ):
            try:
                render_prompt_text(template, example.replacements)
            except PromptContractError as exc:
                raise OptimizationContractError(
                    f"{arm} template does not render for test example {example.example_id}: {exc}"
                ) from exc

    rows_by_kind: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for prompt_kind in active_kinds:
        component = PROMPT_COMPONENTS[prompt_kind]
        batch = [example for example in test_examples if example.prompt_kind == prompt_kind]
        winner_evaluation = winner_adapter.evaluate(
            batch,
            {component: winner_prompts[prompt_kind]},
            capture_traces=False,
        )
        seed_evaluation = seed_adapter.evaluate(
            batch,
            {component: seed_prompts[prompt_kind]},
            capture_traces=False,
        )
        winner_objectives = winner_evaluation.objective_scores
        seed_objectives = seed_evaluation.objective_scores
        if winner_objectives is None or seed_objectives is None:
            raise OptimizationContractError("paired evaluator omitted objective scores")
        if not (
            len(batch)
            == len(winner_evaluation.scores)
            == len(seed_evaluation.scores)
            == len(winner_evaluation.outputs)
            == len(seed_evaluation.outputs)
            == len(winner_objectives)
            == len(seed_objectives)
        ):
            raise OptimizationContractError("paired evaluator result length mismatch")
        paired_rows = [
            _paired_row(
                example=example,
                winner_score=winner_score,
                seed_score=seed_score,
                winner_objectives=winner_objective,
                seed_objectives=seed_objective,
                winner_output=winner_output,
                seed_output=seed_output,
            )
            for (
                example,
                winner_score,
                seed_score,
                winner_objective,
                seed_objective,
                winner_output,
                seed_output,
            ) in zip(
                batch,
                winner_evaluation.scores,
                seed_evaluation.scores,
                winner_objectives,
                seed_objectives,
                winner_evaluation.outputs,
                seed_evaluation.outputs,
                strict=True,
            )
        ]
        rows_by_kind[prompt_kind] = paired_rows
        all_rows.extend(paired_rows)

    report = {
        "paired_heldout_evaluation_version": "1",
        "comparison": "frozen_gepa_winner_vs_explicit_original_seed",
        "split": "test",
        "manifest_sha256": manifest_sha,
        "test_split_sha256": manifest.test.sha256,
        "winner_bundle_sha256": sha256_file(winner_source),
        "winner_prompt_sha256s": {
            kind: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for kind, text in winner_prompts.items()
        },
        "seed_templates": seed_metadata,
        "seed_prompt_bundle_sha256": hash_canonical(
            {kind: metadata["sha256"] for kind, metadata in seed_metadata.items()}
        ),
        "provider_request_namespaces": {
            "winner": _PAIRED_WINNER_NAMESPACE,
            "seed": _PAIRED_SEED_NAMESPACE,
        },
        "objective_weights": {
            "extraction_correctness": 0.70,
            "grounding_schema_validity": 0.25,
            "cost_efficiency": 0.05,
        },
        "test_example_ids": sorted(example.example_id for example in test_examples),
        "overall": _summarize_paired_rows(all_rows),
        "by_prompt_kind": {
            prompt_kind: {
                "winner_prompt_sha256": hashlib.sha256(
                    winner_prompts[prompt_kind].encode("utf-8")
                ).hexdigest(),
                "seed_prompt_sha256": seed_metadata[prompt_kind]["sha256"],
                "summary": _summarize_paired_rows(rows),
                "per_example": rows,
            }
            for prompt_kind, rows in rows_by_kind.items()
        },
    }
    atomic_write_json(target, report)
    return target


__all__ = [
    "GEPAPromptAdapter",
    "GEPAUnavailable",
    "OptimizationContractError",
    "OptimizationEvaluationBatch",
    "OptimizationExample",
    "OptimizationSplitManifest",
    "PromptOptimizationRun",
    "SplitArtifact",
    "compare_frozen_test_to_seed",
    "create_split_bundle",
    "evaluate_frozen_test",
    "load_examples",
    "load_frozen_prompts",
    "load_manifest_split",
    "load_optimization_examples",
    "load_split_manifest",
    "optimize_prompts",
    "provider_request_key",
]
