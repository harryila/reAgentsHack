"""Retrospective item-risk calibration over the frozen Evidence Inference paired run.

This module deliberately does not attach historical GEPA predictions to the current
Literature Multiverse verifier.  It constructs a standalone diagnostic pipeline whose
settings bind the exact prediction-time plan, model, prompts, schemas, source-code
snapshot, and paired-report hashes.  The resulting ItemRiskCalibrationBundle retains
``release_probability_authority=False`` and is useful only for aggregate scheduling-
cell UCL mechanics on a historically opened benchmark.
"""

from __future__ import annotations

import ast
import json
import math
import platform
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.item_risk_artifacts import (
    FixedRiskBinsReceipt,
    ItemRiskCalibrationRunReceipt,
    RiskBinDefinitionArtifact,
)
from literature_multiverse.item_risk_calibration import (
    DomainRiskBinCalibration,
    ItemRiskCalibrationUnit,
    make_fixed_risk_bin_family,
    seal_item_risk_calibration_unit,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_bytes,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.ollama_gepa_study import (
    validate_public_summary as validate_gepa_public_summary,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintError,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
)

STUDY_VERSION = "evidence-inference-item-risk-retrospective-v1"
PUBLIC_SUMMARY_VERSION = "evidence-inference-item-risk-public-summary-v1"
DESIGN_VERSION = "evidence-inference-item-risk-design-v1"
MATERIALIZATION_VERSION = "evidence-inference-item-risk-materialization-v1"
FEATURE_VERSION = "evidence-inference-paired-label-free-risk-features-v1"
REPRESENTATIVE_ALGORITHM = "minimum-namespaced-sha256-example-per-paper-v1"
SPLIT_ALGORITHM = "namespaced-sha256-paper-modulo-5-buckets-0-or-1-development-v1"
SCORE_FORMULA = "mean_of_five_binary_label_free_failure_flags"
FEATURE_NAMES = [
    "seed_winner_direction_disagreement",
    "seed_structured_output_invalid",
    "winner_structured_output_invalid",
    "seed_exact_grounding_invalid",
    "winner_exact_grounding_invalid",
]
BIN_EDGES = [0.0, 0.1, 0.3, 1.0]
ERROR_EVENT_DEFINITION = (
    "winner predicted direction differs from the official Evidence Inference direction "
    "for the deterministically selected paper-level example"
)
SHIFT_DETECTOR_ID = "not-assessed-retrospective-diagnostic-v1"
POPULATION_ID = "evidence-inference-official-test-paper-sample-nonpristine-v1"
DOMAIN = "biomedical-evidence-inference"

_PIPELINE_DEPENDENCY_ENTRYPOINTS = (
    "scripts/calibrate_item_risk.py",
    "scripts/run_evidence_inference_item_risk_diagnostic.py",
    "src/literature_multiverse/evidence_inference_item_risk.py",
)
_PIPELINE_NONPYTHON_PATHS = (
    "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json",
    "configs/benchmarks/evidence-inference-item-risk-v1.json",
    "pyproject.toml",
    "uv.lock",
)
_PUBLIC_FORBIDDEN_KEYS = {
    "article_id",
    "example_id",
    "expected_direction",
    "paper_id",
    "predicted_direction",
    "question_id",
    "reference",
    "rows",
}
_PUBLIC_FORBIDDEN_STRING_PATTERNS = (
    re.compile(r"\bPMC[0-9]+\b", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])Users[\\/]"),
    re.compile(r"^/"),
)
_REQUIRED_CAVEATS = [
    (
        "All Evidence Inference labels in this checkout were historically opened; this "
        "retrospective diagnostic is neither pristine nor confirmatory."
    ),
    (
        "Predictions and official directions are co-located in the frozen paired report. "
        "This rerun froze the score and bins before opening that report and delayed logical "
        "reference-field access until after feature projection, but it does not recreate "
        "physical or historical label blinding."
    ),
    (
        "The paired predictions came from the pinned local llama3.2:1b runtime with 1.2B "
        "parameters; this does not establish performance for frontier systems."
    ),
    (
        "One example was selected deterministically per unique paper/question group, so the "
        "result does not estimate prompt-level prevalence across all 524 benchmark rows."
    ),
    (
        "The risk score uses only arm disagreement, structured-output validity, and exact "
        "quote/line grounding validity; it uses no reference direction, accuracy, distribution "
        "fidelity, scalar objective, confidence, or human label as a feature."
    ),
    (
        "The calibrated values are simultaneous group-average domain-by-bin error-rate upper "
        "confidence limits for scheduling or blocking, not individual error probabilities."
    ),
    (
        "The calibration bundle has release_probability_authority=false and cannot authorize "
        "a claim release, a scientific-truth statement, or an end-to-end verifier claim."
    ),
    (
        "No prospective distribution-shift assessment was run, so no deployment or shift "
        "robustness result is claimed."
    ),
]

_DESIGN_ACCESS_ORDER = [
    "config_opened_and_validated",
    "identifier_free_gepa_public_summary_opened_and_validated",
    "historical_prediction_source_lineage_frozen",
    "standalone_diagnostic_pipeline_fingerprinted",
    "label_free_score_model_frozen",
    "prespecified_bins_written_and_sealed",
    "design_receipt_sealed_before_private_paired_report_access",
]
_MATERIALIZATION_ACCESS_ORDER = [
    "config_opened_and_validated",
    "design_receipt_opened_and_external_hash_matched",
    "standalone_diagnostic_pipeline_recomputed_and_matched",
    "identifier_free_gepa_public_summary_opened_and_lineage_matched",
    "prediction_plan_winner_and_manifest_opened_and_matched",
    "private_paired_report_opened_after_design_freeze",
    "one_representative_selected_per_paper_without_reference_labels",
    "label_free_features_and_hash_partition_materialized",
    "official_direction_labels_joined_after_feature_materialization",
    "development_and_calibration_units_atomically_written",
    "materialization_receipt_sealed",
]


class EvidenceInferenceItemRiskError(ValueError):
    """The retrospective diagnostic contract or source lineage failed."""


def _sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("diagnostic_path_must_use_posix_separators")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
    ):
        raise ValueError("diagnostic_path_must_be_normalized_repository_relative")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceInferenceItemRiskError(f"{label}_symlink_forbidden")
    try:
        if not path.is_file():
            raise EvidenceInferenceItemRiskError(f"{label}_missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceItemRiskError(f"{label}_invalid_json") from exc
    if not isinstance(payload, dict):
        raise EvidenceInferenceItemRiskError(f"{label}_must_be_object")
    return payload


def _validate_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
        raise EvidenceInferenceItemRiskError(f"{label}_self_hash_missing")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if hash_canonical(unsigned) != observed:
        raise EvidenceInferenceItemRiskError(f"{label}_self_hash_mismatch")


def _tagged_digest(namespace: str, value: str) -> str:
    return sha256_bytes(f"{namespace}\0{value}".encode())


class EvidenceInferenceItemRiskConfig(ContractModel):
    config_version: Literal["evidence-inference-item-risk-config-v1"] = (
        "evidence-inference-item-risk-config-v1"
    )
    study_version: Literal["evidence-inference-item-risk-retrospective-v1"] = STUDY_VERSION
    gepa_public_summary_path: str
    gepa_private_report_path: str
    gepa_plan_path: str
    gepa_winner_path: str
    gepa_manifest_path: str
    private_run_dir: str
    public_summary_path: str
    population_id: Literal["evidence-inference-official-test-paper-sample-nonpristine-v1"] = (
        POPULATION_ID
    )
    domain: Literal["biomedical-evidence-inference"] = DOMAIN
    representative_algorithm: Literal["minimum-namespaced-sha256-example-per-paper-v1"] = (
        REPRESENTATIVE_ALGORITHM
    )
    split_algorithm: Literal["namespaced-sha256-paper-modulo-5-buckets-0-or-1-development-v1"] = (
        SPLIT_ALGORITHM
    )
    development_modulo: Literal[5] = 5
    development_buckets: list[int]
    feature_version: Literal["evidence-inference-paired-label-free-risk-features-v1"] = (
        FEATURE_VERSION
    )
    feature_names: list[str]
    score_formula: Literal["mean_of_five_binary_label_free_failure_flags"] = SCORE_FORMULA
    bin_edges: list[float]
    familywise_delta: Annotated[float, Field(gt=0, lt=1)]
    error_event_definition: Literal[
        "winner predicted direction differs from the official Evidence Inference direction "
        "for the deterministically selected paper-level example"
    ] = ERROR_EVENT_DEFINITION
    shift_detector_id: Literal["not-assessed-retrospective-diagnostic-v1"] = SHIFT_DETECTOR_ID
    all_labels_historically_opened: Literal[True] = True
    test_is_non_pristine: Literal[True] = True
    confirmatory_claim_allowed: Literal[False] = False
    config_sha256: str

    @field_validator(
        "gepa_public_summary_path",
        "gepa_private_report_path",
        "gepa_plan_path",
        "gepa_winner_path",
        "gepa_manifest_path",
        "private_run_dir",
        "public_summary_path",
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "item_risk_config")

    @model_validator(mode="after")
    def validate_config(self) -> EvidenceInferenceItemRiskConfig:
        if self.development_buckets != [0, 1]:
            raise ValueError("item_risk_development_buckets_changed")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("item_risk_feature_family_changed")
        if self.bin_edges != BIN_EDGES:
            raise ValueError("item_risk_bin_edges_changed")
        if not math.isclose(self.familywise_delta, 0.05, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("item_risk_familywise_delta_changed")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if hash_canonical(payload) != self.config_sha256:
            raise ValueError("item_risk_config_hash_mismatch")
        return self


def load_config(path: Path) -> EvidenceInferenceItemRiskConfig:
    try:
        return EvidenceInferenceItemRiskConfig.model_validate(
            _read_json_object(path, label="item_risk_config")
        )
    except ValueError as exc:
        raise EvidenceInferenceItemRiskError("item_risk_config_invalid") from exc


def _prediction_source_lineage(summary: Mapping[str, Any]) -> dict[str, Any]:
    lineage = summary.get("lineage")
    model = summary.get("model")
    schemas = summary.get("schemas")
    population = summary.get("paired_test_population")
    if not all(isinstance(value, Mapping) for value in (lineage, model, schemas, population)):
        raise EvidenceInferenceItemRiskError("gepa_public_summary_lineage_missing")
    assert isinstance(lineage, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(schemas, Mapping)
    assert isinstance(population, Mapping)
    source_hashes = lineage.get("source_code_sha256s")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise EvidenceInferenceItemRiskError("gepa_prediction_source_code_lineage_missing")
    for path, digest in source_hashes.items():
        _relative_path(str(path))
        _sha256(str(digest), "gepa_prediction_source_code")
    payload = {
        "lineage_version": "evidence-inference-gepa-prediction-source-v1",
        "gepa_public_summary_sha256": summary["public_summary_sha256"],
        "gepa_study_version": summary["study_version"],
        "plan_file_sha256": lineage["plan_file_sha256"],
        "plan_sha256": lineage["plan_sha256"],
        "winner_bundle_file_sha256": lineage["winner_bundle_file_sha256"],
        "winner_sha256": lineage["winner_sha256"],
        "private_paired_report_file_sha256": lineage["private_paired_report_file_sha256"],
        "private_paired_report_sha256": lineage["private_paired_report_sha256"],
        "manifest_file_sha256": lineage["manifest_file_sha256"],
        "test_split_jsonl_sha256": lineage["test_split_jsonl_sha256"],
        "seed_prompt_sha256": summary["seed_prompt_sha256"],
        "winner_prompt_sha256": summary["winner_prompt_sha256"],
        "model_name": model["name"],
        "model_digest": model["digest"],
        "model_parameter_size": model["parameter_size"],
        "ollama_version": model["ollama_version"],
        "generation_config_sha256": model["generation_config_sha256"],
        "model_identity_sha256": model["identity_sha256"],
        "evaluation_schema_sha256": schemas["evaluation_sha256"],
        "generation_schema_base_sha256": schemas["generation_base_sha256"],
        "generation_schema_algorithm": schemas["generation_algorithm"],
        "paired_examples": population["examples"],
        "paired_articles": population["articles"],
        "arm_order_policy": population["arm_order_policy"],
        "historical_source_code_sha256s": dict(sorted(source_hashes.items())),
    }
    for key, value in payload.items():
        if key.endswith("sha256"):
            _sha256(str(value), key)
    return {
        **payload,
        "prediction_source_lineage_sha256": hash_canonical(payload),
    }


def _load_gepa_public_summary(path: Path) -> dict[str, Any]:
    try:
        return validate_gepa_public_summary(_read_json_object(path, label="gepa_public_summary"))
    except ValueError as exc:
        raise EvidenceInferenceItemRiskError("gepa_public_summary_invalid") from exc


def _resolve_local_python_import(
    *,
    repository_root: Path,
    current_path: str,
    module: str,
    level: int,
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _diagnostic_python_dependency_closure(repository_root: Path) -> list[str]:
    """Mechanically bind every local Python dependency used by the diagnostic."""

    pending = list(_PIPELINE_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if not source_path.is_file():
            raise EvidenceInferenceItemRiskError(
                f"diagnostic_pipeline_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise EvidenceInferenceItemRiskError(
                f"diagnostic_pipeline_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_python_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def compute_diagnostic_pipeline_fingerprint(
    *,
    repository_root: Path,
    config: EvidenceInferenceItemRiskConfig,
    gepa_public_summary: Mapping[str, Any],
) -> tuple[PipelineFingerprint, dict[str, Any]]:
    """Bind the standalone projector plus exact historical prediction lineage."""

    source_lineage = _prediction_source_lineage(gepa_public_summary)
    python_closure = _diagnostic_python_dependency_closure(repository_root)
    component = PipelineComponentSpec(
        component_id="evidence-inference-retrospective-item-risk",
        component_version="2",
        file_paths=sorted({*_PIPELINE_NONPYTHON_PATHS, *python_closure}),
        settings={
            "scientific_scope": "standalone_retrospective_nonpristine_diagnostic",
            "current_verifier_pipeline_compatible": False,
            "release_probability_authority": False,
            "feature_version": config.feature_version,
            "feature_names": config.feature_names,
            "score_formula": config.score_formula,
            "representative_algorithm": config.representative_algorithm,
            "split_algorithm": config.split_algorithm,
            "prediction_source_lineage": source_lineage,
            "dependency_closure_entrypoints": list(_PIPELINE_DEPENDENCY_ENTRYPOINTS),
            "in_repository_dependency_closure_bound": True,
            "python_version": platform.python_version(),
            "numpy_version": distribution_version("numpy"),
            "pydantic_version": distribution_version("pydantic"),
            "scipy_version": distribution_version("scipy"),
        },
    )
    try:
        fingerprint = compute_pipeline_fingerprint(
            root=repository_root,
            components=[component],
        )
    except (OSError, ValueError) as exc:
        raise EvidenceInferenceItemRiskError("diagnostic_pipeline_fingerprint_failed") from exc
    return fingerprint, source_lineage


class EvidenceInferenceItemRiskDesignReceipt(ContractModel):
    receipt_version: Literal["evidence-inference-item-risk-design-v1"] = DESIGN_VERSION
    freeze_state: Literal[
        "score_bins_and_protocol_frozen_before_paired_report_opened_in_this_run"
    ] = "score_bins_and_protocol_frozen_before_paired_report_opened_in_this_run"
    config_file_sha256: str
    config_sha256: str
    gepa_public_summary_file_sha256: str
    gepa_public_summary_sha256: str
    prediction_source_lineage_sha256: str
    diagnostic_pipeline_fingerprint: PipelineFingerprint
    diagnostic_pipeline_sha256: str
    score_model_sha256: str
    bin_definition_file_sha256: str
    fixed_bins_file_sha256: str
    fixed_bins: FixedRiskBinsReceipt
    sampling_protocol_sha256: str
    adjudication_protocol_sha256: str
    shift_detector_sha256: str
    paired_report_opened: Literal[False] = False
    private_row_labels_opened_in_this_stage: Literal[False] = False
    labels_historically_opened_before_protocol: Literal[True] = True
    access_order: list[str]
    receipt_sha256: str

    @field_validator(
        "config_file_sha256",
        "config_sha256",
        "gepa_public_summary_file_sha256",
        "gepa_public_summary_sha256",
        "prediction_source_lineage_sha256",
        "diagnostic_pipeline_sha256",
        "score_model_sha256",
        "bin_definition_file_sha256",
        "fixed_bins_file_sha256",
        "sampling_protocol_sha256",
        "adjudication_protocol_sha256",
        "shift_detector_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceItemRiskDesignReceipt:
        if self.access_order != _DESIGN_ACCESS_ORDER:
            raise ValueError("item_risk_design_access_order_invalid")
        if (
            self.diagnostic_pipeline_fingerprint.pipeline_sha256 != self.diagnostic_pipeline_sha256
            or self.fixed_bins.bin_family.score_model_sha256 != self.score_model_sha256
            or self.fixed_bins.definition_file_sha256 != self.bin_definition_file_sha256
            or self.fixed_bins_file_sha256 == self.bin_definition_file_sha256
        ):
            raise ValueError("item_risk_design_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("item_risk_design_receipt_hash_mismatch")
        return self


class EvidenceInferenceItemRiskMaterializationReceipt(ContractModel):
    receipt_version: Literal["evidence-inference-item-risk-materialization-v1"] = (
        MATERIALIZATION_VERSION
    )
    design_receipt_file_sha256: str
    design_receipt_sha256: str
    pipeline_verification_sha256: str
    prediction_source_lineage_sha256: str
    gepa_public_summary_file_sha256: str
    gepa_public_summary_sha256: str
    gepa_plan_file_sha256: str
    gepa_plan_sha256: str
    gepa_winner_file_sha256: str
    gepa_winner_sha256: str
    gepa_manifest_file_sha256: str
    test_split_jsonl_sha256: str
    paired_report_file_sha256: str
    paired_report_sha256: str
    paired_example_count: Annotated[int, Field(gt=0)]
    paired_paper_count: Annotated[int, Field(gt=0)]
    representative_unit_count: Annotated[int, Field(gt=0)]
    development_unit_count: Annotated[int, Field(gt=0)]
    calibration_unit_count: Annotated[int, Field(gt=0)]
    development_error_count: Annotated[int, Field(ge=0)]
    calibration_error_count: Annotated[int, Field(ge=0)]
    representative_membership_sha256: str
    development_membership_sha256: str
    calibration_membership_sha256: str
    development_units_file_sha256: str
    calibration_units_file_sha256: str
    development_unit_sha256s: list[str]
    calibration_unit_sha256s: list[str]
    feature_names: list[str]
    feature_labels_used: Literal[False] = False
    official_labels_opened_after_design_freeze: Literal[True] = True
    labels_historically_opened_before_protocol: Literal[True] = True
    access_order: list[str]
    receipt_sha256: str

    @field_validator(
        "design_receipt_file_sha256",
        "design_receipt_sha256",
        "pipeline_verification_sha256",
        "prediction_source_lineage_sha256",
        "gepa_public_summary_file_sha256",
        "gepa_public_summary_sha256",
        "gepa_plan_file_sha256",
        "gepa_plan_sha256",
        "gepa_winner_file_sha256",
        "gepa_winner_sha256",
        "gepa_manifest_file_sha256",
        "test_split_jsonl_sha256",
        "paired_report_file_sha256",
        "paired_report_sha256",
        "representative_membership_sha256",
        "development_membership_sha256",
        "calibration_membership_sha256",
        "development_units_file_sha256",
        "calibration_units_file_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("development_unit_sha256s", "calibration_unit_sha256s")
    @classmethod
    def validate_hashes(cls, value: list[str], info: Any) -> list[str]:
        if not value or value != sorted(set(value)):
            raise ValueError(f"{info.field_name}_must_be_sorted_unique")
        for digest in value:
            _sha256(digest, info.field_name)
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceItemRiskMaterializationReceipt:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("item_risk_materialization_feature_family_changed")
        if (
            self.representative_unit_count
            != self.development_unit_count + self.calibration_unit_count
            or self.development_error_count > self.development_unit_count
            or self.calibration_error_count > self.calibration_unit_count
            or self.development_unit_count != len(self.development_unit_sha256s)
            or self.calibration_unit_count != len(self.calibration_unit_sha256s)
        ):
            raise ValueError("item_risk_materialization_count_mismatch")
        if self.access_order != _MATERIALIZATION_ACCESS_ORDER:
            raise ValueError("item_risk_materialization_access_order_invalid")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("item_risk_materialization_receipt_hash_mismatch")
        return self


def _design_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "pipeline": work_dir / "diagnostic-pipeline-fingerprint.json",
        "definition": work_dir / "risk-bin-definition.json",
        "bins": work_dir / "fixed-risk-bins.json",
        "design": work_dir / "design-receipt.json",
    }


def _materialization_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "development": work_dir / "development-units.jsonl",
        "calibration": work_dir / "calibration-units.jsonl",
        "materialization": work_dir / "materialization-receipt.json",
    }


def _preflight_outputs(paths: Sequence[Path], *, force: bool) -> None:
    for path in paths:
        if path.is_symlink():
            raise EvidenceInferenceItemRiskError(f"output_symlink_forbidden:{path.name}")
        if path.exists() and not force:
            raise EvidenceInferenceItemRiskError(f"output_exists:{path.name}")


def _require_configured_work_dir(
    *,
    repository_root: Path,
    config: EvidenceInferenceItemRiskConfig,
    work_dir: Path,
) -> None:
    expected = (repository_root / config.private_run_dir).resolve(strict=False)
    if work_dir.resolve(strict=False) != expected:
        raise EvidenceInferenceItemRiskError("item_risk_work_dir_config_mismatch")


def _require_configured_public_output(
    *,
    repository_root: Path,
    config: EvidenceInferenceItemRiskConfig,
    output_path: Path,
) -> None:
    expected = (repository_root / config.public_summary_path).resolve(strict=False)
    if output_path.resolve(strict=False) != expected:
        raise EvidenceInferenceItemRiskError("item_risk_public_output_config_mismatch")


def _require_calibration_run_path(*, work_dir: Path, calibration_run_path: Path) -> None:
    expected = (work_dir / "calibration-run.json").resolve(strict=False)
    if calibration_run_path.resolve(strict=False) != expected:
        raise EvidenceInferenceItemRiskError("item_risk_calibration_run_path_mismatch")


def freeze_design(
    *,
    repository_root: Path,
    config_path: Path,
    work_dir: Path,
    force: bool = False,
) -> EvidenceInferenceItemRiskDesignReceipt:
    """Freeze label-free score/bins without opening the private paired report."""

    outputs = _design_paths(work_dir)
    _preflight_outputs(list(outputs.values()), force=force)
    access_order = ["config_opened_and_validated"]
    config = load_config(config_path)
    _require_configured_work_dir(
        repository_root=repository_root,
        config=config,
        work_dir=work_dir,
    )
    gepa_summary_path = repository_root / config.gepa_public_summary_path
    gepa_summary = _load_gepa_public_summary(gepa_summary_path)
    access_order.append("identifier_free_gepa_public_summary_opened_and_validated")
    fingerprint, source_lineage = compute_diagnostic_pipeline_fingerprint(
        repository_root=repository_root,
        config=config,
        gepa_public_summary=gepa_summary,
    )
    access_order.append("historical_prediction_source_lineage_frozen")
    atomic_write_json(outputs["pipeline"], fingerprint, force=force)
    access_order.append("standalone_diagnostic_pipeline_fingerprinted")
    score_model_payload = {
        "score_model_version": FEATURE_VERSION,
        "diagnostic_pipeline_sha256": fingerprint.pipeline_sha256,
        "prediction_source_lineage_sha256": source_lineage["prediction_source_lineage_sha256"],
        "feature_names": FEATURE_NAMES,
        "score_formula": SCORE_FORMULA,
        "reference_or_accuracy_features_used": False,
    }
    score_model_sha256 = hash_canonical(score_model_payload)
    access_order.append("label_free_score_model_frozen")
    definition = RiskBinDefinitionArtifact(
        definition_source="prespecified",
        source_split="none",
        labels_used=False,
        label_source=None,
        simulation=False,
        score_name="paired_label_free_failure_flag_mean",
        score_model_sha256=score_model_sha256,
        edges=BIN_EDGES,
    )
    atomic_write_json(outputs["definition"], definition, force=force)
    definition_file_sha256 = sha256_file(outputs["definition"])
    family = make_fixed_risk_bin_family(
        edges=definition.edges,
        score_name=definition.score_name,
        score_model_sha256=score_model_sha256,
        definition_source="prespecified",
        definition_artifact_sha256=definition_file_sha256,
    )
    bins_payload: dict[str, Any] = {
        "receipt_version": "fixed-item-risk-bins-receipt-v1",
        "definition_file_sha256": definition_file_sha256,
        "definition": definition,
        "bin_family": family,
        "access_order": ["definition_opened", "fixed_bins_sealed"],
    }
    bins = FixedRiskBinsReceipt.model_validate(
        {**bins_payload, "receipt_sha256": hash_canonical(bins_payload)}
    )
    atomic_write_json(outputs["bins"], bins, force=force)
    access_order.append("prespecified_bins_written_and_sealed")
    sampling_protocol_sha256 = hash_canonical(
        {
            "protocol_version": "evidence-inference-paper-sampling-v1",
            "representative_algorithm": REPRESENTATIVE_ALGORITHM,
            "split_algorithm": SPLIT_ALGORITHM,
            "one_unit_per_unique_question_and_paper": True,
            "selection_and_split_use_labels": False,
        }
    )
    adjudication_protocol_sha256 = hash_canonical(
        {
            "protocol_version": "evidence-inference-official-direction-error-v1",
            "test_split_jsonl_sha256": source_lineage["test_split_jsonl_sha256"],
            "error_event_definition": ERROR_EVENT_DEFINITION,
            "label_source": "benchmark_annotation",
            "human_reannotation_performed": False,
        }
    )
    shift_detector_sha256 = hash_canonical(
        {
            "detector_id": SHIFT_DETECTOR_ID,
            "status": "not_assessed",
            "prospective_inputs_opened": False,
        }
    )
    access_order.append("design_receipt_sealed_before_private_paired_report_access")
    payload: dict[str, Any] = {
        "receipt_version": DESIGN_VERSION,
        "freeze_state": ("score_bins_and_protocol_frozen_before_paired_report_opened_in_this_run"),
        "config_file_sha256": sha256_file(config_path),
        "config_sha256": config.config_sha256,
        "gepa_public_summary_file_sha256": sha256_file(gepa_summary_path),
        "gepa_public_summary_sha256": gepa_summary["public_summary_sha256"],
        "prediction_source_lineage_sha256": source_lineage["prediction_source_lineage_sha256"],
        "diagnostic_pipeline_fingerprint": fingerprint,
        "diagnostic_pipeline_sha256": fingerprint.pipeline_sha256,
        "score_model_sha256": score_model_sha256,
        "bin_definition_file_sha256": definition_file_sha256,
        "fixed_bins_file_sha256": sha256_file(outputs["bins"]),
        "fixed_bins": bins,
        "sampling_protocol_sha256": sampling_protocol_sha256,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "shift_detector_sha256": shift_detector_sha256,
        "paired_report_opened": False,
        "private_row_labels_opened_in_this_stage": False,
        "labels_historically_opened_before_protocol": True,
        "access_order": access_order,
    }
    receipt = EvidenceInferenceItemRiskDesignReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )
    atomic_write_json(outputs["design"], receipt, force=force)
    return receipt


def _arm_feature_flags(arm: Mapping[str, Any], *, arm_name: str) -> dict[str, bool]:
    objectives = arm.get("objective_scores")
    if not isinstance(objectives, Mapping):
        raise EvidenceInferenceItemRiskError(f"paired_{arm_name}_objectives_missing")
    structured = objectives.get("structured_output_validity")
    grounding = objectives.get("formal_grounding_validity")
    if structured not in {0, 1} or grounding not in {0, 1}:
        raise EvidenceInferenceItemRiskError(f"paired_{arm_name}_label_free_metric_invalid")
    predicted = arm.get("predicted_direction")
    if predicted not in {None, "increase", "decrease", "no_effect"}:
        raise EvidenceInferenceItemRiskError(f"paired_{arm_name}_direction_invalid")
    return {
        f"{arm_name}_structured_output_invalid": float(structured) != 1.0,
        f"{arm_name}_exact_grounding_invalid": float(grounding) != 1.0,
    }


def label_free_feature_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project only frozen prediction/grounding fields; never read the reference label."""

    seed = row.get("seed")
    winner = row.get("winner")
    if not isinstance(seed, Mapping) or not isinstance(winner, Mapping):
        raise EvidenceInferenceItemRiskError("paired_arm_payload_missing")
    seed_flags = _arm_feature_flags(seed, arm_name="seed")
    winner_flags = _arm_feature_flags(winner, arm_name="winner")
    flags = {
        "seed_winner_direction_disagreement": (
            seed.get("predicted_direction") != winner.get("predicted_direction")
        ),
        **seed_flags,
        **winner_flags,
    }
    ordered = {name: bool(flags[name]) for name in FEATURE_NAMES}
    risk_score = sum(ordered.values()) / float(len(FEATURE_NAMES))
    output_lineage: dict[str, Any] = {}
    for arm_name, arm in (("seed", seed), ("winner", winner)):
        output = arm.get("output")
        if not isinstance(output, Mapping):
            raise EvidenceInferenceItemRiskError(f"paired_{arm_name}_output_missing")
        output_lineage[arm_name] = {
            "request_sha256": output.get("request_sha256"),
            "passage_projection_sha256": output.get("passage_projection_sha256"),
            "generation_schema_sha256": output.get("generation_schema_sha256"),
            "evaluation_schema_sha256": output.get("evaluation_schema_sha256"),
        }
        for field, digest in output_lineage[arm_name].items():
            if digest is not None:
                _sha256(str(digest), f"{arm_name}_{field}")
    return {
        "feature_version": FEATURE_VERSION,
        "flags": ordered,
        "risk_score": risk_score,
        "output_lineage": output_lineage,
    }


def _representatives(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    example_ids: set[str] = set()
    for row in rows:
        example_id = row.get("example_id")
        paper_id = row.get("paper_id")
        group_id = row.get("group_id")
        if not all(isinstance(value, str) and value for value in (example_id, paper_id, group_id)):
            raise EvidenceInferenceItemRiskError("paired_row_identity_invalid")
        if example_id in example_ids:
            raise EvidenceInferenceItemRiskError("paired_example_id_duplicate")
        example_ids.add(example_id)
        if paper_id != group_id:
            raise EvidenceInferenceItemRiskError("paired_paper_group_identity_mismatch")
        grouped[(paper_id, group_id)].append(row)
    representatives = [
        min(
            group_rows,
            key=lambda row: _tagged_digest(
                "evidence-inference-item-risk-representative-v1",
                str(row["example_id"]),
            ),
        )
        for _, group_rows in sorted(grouped.items())
    ]
    return sorted(
        representatives,
        key=lambda row: _tagged_digest(
            "evidence-inference-item-risk-paper-order-v1",
            str(row["paper_id"]),
        ),
    )


def _split_for_paper(paper_id: str) -> Literal["development", "calibration"]:
    digest = _tagged_digest("evidence-inference-item-risk-split-v1", paper_id)
    return "development" if int(digest, 16) % 5 in {0, 1} else "calibration"


def _pre_paired_source_crosschecks(
    *,
    config: EvidenceInferenceItemRiskConfig,
    repository_root: Path,
    gepa_summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    winner: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source = _prediction_source_lineage(gepa_summary)
    paths = {
        "plan": repository_root / config.gepa_plan_path,
        "winner": repository_root / config.gepa_winner_path,
        "manifest": repository_root / config.gepa_manifest_path,
    }
    _validate_self_hash(plan, field="plan_sha256", label="gepa_plan")
    _validate_self_hash(winner, field="winner_sha256", label="gepa_winner")
    checks = {
        "plan_file_sha256": sha256_file(paths["plan"]),
        "plan_sha256": plan.get("plan_sha256"),
        "winner_bundle_file_sha256": sha256_file(paths["winner"]),
        "winner_sha256": winner.get("winner_sha256"),
        "manifest_file_sha256": sha256_file(paths["manifest"]),
        "test_split_jsonl_sha256": manifest.get("test", {}).get("sha256"),
    }
    for key, observed in checks.items():
        if observed != source.get(key):
            raise EvidenceInferenceItemRiskError(f"prediction_source_lineage_mismatch:{key}")
    if (
        winner.get("source_code_sha256s") != source["historical_source_code_sha256s"]
        or winner.get("winner_prompt_sha256") != source["winner_prompt_sha256"]
        or winner.get("seed_prompt_sha256") != source["seed_prompt_sha256"]
        or plan.get("source_code_sha256s") != source["historical_source_code_sha256s"]
    ):
        raise EvidenceInferenceItemRiskError("prediction_source_semantic_lineage_mismatch")
    return source


def _paired_source_crosschecks(
    *,
    config: EvidenceInferenceItemRiskConfig,
    repository_root: Path,
    paired: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    paired_path = repository_root / config.gepa_private_report_path
    _validate_self_hash(paired, field="paired_report_sha256", label="paired_report")
    checks = {
        "private_paired_report_file_sha256": sha256_file(paired_path),
        "private_paired_report_sha256": paired.get("paired_report_sha256"),
        "seed_prompt_sha256": paired.get("seed_prompt_sha256"),
        "winner_prompt_sha256": paired.get("winner_prompt_sha256"),
        "evaluation_schema_sha256": paired.get("evaluation_schema_sha256"),
    }
    for key, observed in checks.items():
        if observed != source.get(key):
            raise EvidenceInferenceItemRiskError(f"prediction_source_lineage_mismatch:{key}")
    if (
        paired.get("plan_sha256") != source["plan_sha256"]
        or paired.get("test_split_jsonl_sha256") != source["test_split_jsonl_sha256"]
        or paired.get("all_labels_historically_opened") is not True
        or paired.get("test_is_non_pristine") is not True
        or paired.get("confirmatory_claim_allowed") is not False
    ):
        raise EvidenceInferenceItemRiskError("prediction_paired_source_semantic_mismatch")


def materialize_units(
    *,
    repository_root: Path,
    config_path: Path,
    work_dir: Path,
    expected_design_receipt_sha256: str,
    force: bool = False,
) -> EvidenceInferenceItemRiskMaterializationReceipt:
    """Join official labels only after the externally anchored design is verified."""

    outputs = _materialization_paths(work_dir)
    _preflight_outputs(list(outputs.values()), force=force)
    access_order = ["config_opened_and_validated"]
    config = load_config(config_path)
    _require_configured_work_dir(
        repository_root=repository_root,
        config=config,
        work_dir=work_dir,
    )
    design_path = _design_paths(work_dir)["design"]
    try:
        design = EvidenceInferenceItemRiskDesignReceipt.model_validate(
            _read_json_object(design_path, label="item_risk_design_receipt")
        )
    except ValueError as exc:
        raise EvidenceInferenceItemRiskError("item_risk_design_receipt_invalid") from exc
    if design.receipt_sha256 != expected_design_receipt_sha256:
        raise EvidenceInferenceItemRiskError("item_risk_expected_design_hash_mismatch")
    access_order.append("design_receipt_opened_and_external_hash_matched")
    try:
        verification = require_pipeline_fingerprint_match(
            expected=design.diagnostic_pipeline_fingerprint,
            root=repository_root,
        )
    except PipelineFingerprintError as exc:
        raise EvidenceInferenceItemRiskError("diagnostic_pipeline_changed") from exc
    access_order.append("standalone_diagnostic_pipeline_recomputed_and_matched")
    gepa_summary_path = repository_root / config.gepa_public_summary_path
    gepa_summary = _load_gepa_public_summary(gepa_summary_path)
    if (
        sha256_file(gepa_summary_path) != design.gepa_public_summary_file_sha256
        or gepa_summary["public_summary_sha256"] != design.gepa_public_summary_sha256
    ):
        raise EvidenceInferenceItemRiskError("gepa_public_summary_changed_after_design")
    access_order.append("identifier_free_gepa_public_summary_opened_and_lineage_matched")
    plan = _read_json_object(repository_root / config.gepa_plan_path, label="gepa_plan")
    winner_artifact = _read_json_object(
        repository_root / config.gepa_winner_path,
        label="gepa_winner",
    )
    manifest = _read_json_object(
        repository_root / config.gepa_manifest_path,
        label="gepa_manifest",
    )
    source_lineage = _pre_paired_source_crosschecks(
        config=config,
        repository_root=repository_root,
        gepa_summary=gepa_summary,
        plan=plan,
        winner=winner_artifact,
        manifest=manifest,
    )
    if source_lineage["prediction_source_lineage_sha256"] != (
        design.prediction_source_lineage_sha256
    ):
        raise EvidenceInferenceItemRiskError("prediction_source_lineage_changed_after_design")
    access_order.append("prediction_plan_winner_and_manifest_opened_and_matched")
    paired = _read_json_object(
        repository_root / config.gepa_private_report_path,
        label="gepa_paired_report",
    )
    access_order.append("private_paired_report_opened_after_design_freeze")
    _paired_source_crosschecks(
        config=config,
        repository_root=repository_root,
        paired=paired,
        source=source_lineage,
    )
    raw_rows = paired.get("rows")
    if not isinstance(raw_rows, list) or any(not isinstance(row, Mapping) for row in raw_rows):
        raise EvidenceInferenceItemRiskError("paired_report_rows_invalid")
    if (
        len(raw_rows) != paired.get("examples")
        or paired.get("examples") != source_lineage["paired_examples"]
    ):
        raise EvidenceInferenceItemRiskError("paired_report_example_count_mismatch")
    representatives = _representatives(raw_rows)
    if (
        len(representatives) != paired.get("articles")
        or paired.get("articles") != (source_lineage["paired_articles"])
    ):
        raise EvidenceInferenceItemRiskError("paired_report_paper_count_mismatch")
    access_order.append("one_representative_selected_per_paper_without_reference_labels")
    projected: list[tuple[Mapping[str, Any], dict[str, Any], str]] = []
    for row in representatives:
        features = label_free_feature_projection(row)
        split = _split_for_paper(str(row["paper_id"]))
        projected.append((row, features, split))
    access_order.append("label_free_features_and_hash_partition_materialized")
    units: list[ItemRiskCalibrationUnit] = []
    membership: list[dict[str, str]] = []
    for row, features, split in projected:
        expected = row.get("expected_direction")
        winner_arm = row["winner"]
        assert isinstance(winner_arm, Mapping)
        if expected not in {"increase", "decrease", "no_effect"}:
            raise EvidenceInferenceItemRiskError("paired_reference_direction_invalid")
        paper_digest = _tagged_digest("evidence-inference-paper-v1", str(row["paper_id"]))
        question_digest = _tagged_digest(
            "evidence-inference-question-v1",
            str(row["example_id"]),
        )
        item_digest = _tagged_digest(
            "evidence-inference-item-v1",
            str(row["example_id"]),
        )
        item_id = f"ei-item-{item_digest}"
        question_id = f"ei-question-{question_digest}"
        paper_id = f"ei-paper-{paper_digest}"
        score_input = {
            "score_model_sha256": design.score_model_sha256,
            "item_identity_sha256": item_digest,
            "feature_projection": features,
        }
        observed_error = winner_arm.get("predicted_direction") != expected
        adjudication_artifact_sha256 = hash_canonical(
            {
                "test_split_jsonl_sha256": source_lineage["test_split_jsonl_sha256"],
                "item_identity_sha256": item_digest,
                "official_direction": expected,
            }
        )
        unit = seal_item_risk_calibration_unit(
            split=split,
            item_id=item_id,
            question_id=question_id,
            paper_id=paper_id,
            population_id=POPULATION_ID,
            domain=DOMAIN,
            pipeline_sha256=design.diagnostic_pipeline_sha256,
            score_model_sha256=design.score_model_sha256,
            score_input_sha256=hash_canonical(score_input),
            risk_score=float(features["risk_score"]),
            observed_error=bool(observed_error),
            label_source="benchmark_annotation",
            adjudication_protocol_sha256=design.adjudication_protocol_sha256,
            adjudication_artifact_sha256=adjudication_artifact_sha256,
        )
        units.append(unit)
        membership.append(
            {
                "split": split,
                "item_sha256": item_digest,
                "question_sha256": question_digest,
                "paper_sha256": paper_digest,
            }
        )
    access_order.append("official_direction_labels_joined_after_feature_materialization")
    development = sorted(
        (unit for unit in units if unit.split == "development"),
        key=lambda unit: (unit.question_id, unit.item_id),
    )
    calibration = sorted(
        (unit for unit in units if unit.split == "calibration"),
        key=lambda unit: (unit.question_id, unit.item_id),
    )
    if not development or not calibration:
        raise EvidenceInferenceItemRiskError("item_risk_hash_partition_empty")
    if len({unit.paper_id for unit in units}) != len(units) or len(
        {unit.question_id for unit in units}
    ) != len(units):
        raise EvidenceInferenceItemRiskError("item_risk_units_not_independent")
    atomic_write_jsonl(outputs["development"], development, force=force)
    atomic_write_jsonl(outputs["calibration"], calibration, force=force)
    access_order.append("development_and_calibration_units_atomically_written")
    access_order.append("materialization_receipt_sealed")
    development_membership = sorted(
        (row for row in membership if row["split"] == "development"),
        key=lambda row: row["item_sha256"],
    )
    calibration_membership = sorted(
        (row for row in membership if row["split"] == "calibration"),
        key=lambda row: row["item_sha256"],
    )
    payload: dict[str, Any] = {
        "receipt_version": MATERIALIZATION_VERSION,
        "design_receipt_file_sha256": sha256_file(design_path),
        "design_receipt_sha256": design.receipt_sha256,
        "pipeline_verification_sha256": verification.verification_sha256,
        "prediction_source_lineage_sha256": source_lineage["prediction_source_lineage_sha256"],
        "gepa_public_summary_file_sha256": sha256_file(gepa_summary_path),
        "gepa_public_summary_sha256": gepa_summary["public_summary_sha256"],
        "gepa_plan_file_sha256": sha256_file(repository_root / config.gepa_plan_path),
        "gepa_plan_sha256": plan["plan_sha256"],
        "gepa_winner_file_sha256": sha256_file(repository_root / config.gepa_winner_path),
        "gepa_winner_sha256": winner_artifact["winner_sha256"],
        "gepa_manifest_file_sha256": sha256_file(repository_root / config.gepa_manifest_path),
        "test_split_jsonl_sha256": source_lineage["test_split_jsonl_sha256"],
        "paired_report_file_sha256": sha256_file(repository_root / config.gepa_private_report_path),
        "paired_report_sha256": paired["paired_report_sha256"],
        "paired_example_count": len(raw_rows),
        "paired_paper_count": int(paired["articles"]),
        "representative_unit_count": len(units),
        "development_unit_count": len(development),
        "calibration_unit_count": len(calibration),
        "development_error_count": sum(unit.observed_error for unit in development),
        "calibration_error_count": sum(unit.observed_error for unit in calibration),
        "representative_membership_sha256": hash_canonical(sorted(membership, key=str)),
        "development_membership_sha256": hash_canonical(development_membership),
        "calibration_membership_sha256": hash_canonical(calibration_membership),
        "development_units_file_sha256": sha256_file(outputs["development"]),
        "calibration_units_file_sha256": sha256_file(outputs["calibration"]),
        "development_unit_sha256s": sorted(unit.unit_sha256 for unit in development),
        "calibration_unit_sha256s": sorted(unit.unit_sha256 for unit in calibration),
        "feature_names": FEATURE_NAMES,
        "feature_labels_used": False,
        "official_labels_opened_after_design_freeze": True,
        "labels_historically_opened_before_protocol": True,
        "access_order": access_order,
    }
    receipt = EvidenceInferenceItemRiskMaterializationReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )
    atomic_write_json(outputs["materialization"], receipt, force=force)
    return receipt


def _public_redaction_check(
    value: Any,
    *,
    path: str = "$",
    repository_root: Path | None = None,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _PUBLIC_FORBIDDEN_KEYS:
                raise EvidenceInferenceItemRiskError(f"public_summary_forbidden_key:{path}.{key}")
            _public_redaction_check(
                child,
                path=f"{path}.{key}",
                repository_root=repository_root,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _public_redaction_check(
                child,
                path=f"{path}[{index}]",
                repository_root=repository_root,
            )
        return
    if isinstance(value, str):
        if repository_root is not None and repository_root.as_posix() in value:
            raise EvidenceInferenceItemRiskError("public_summary_absolute_path")
        if any(pattern.search(value) for pattern in _PUBLIC_FORBIDDEN_STRING_PATTERNS):
            raise EvidenceInferenceItemRiskError(f"public_summary_forbidden_value:{path}")


def _validate_public_hash_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            location = f"{path}.{name}"
            if name.endswith("_sha256"):
                if not isinstance(child, str) or not SHA256_RE.fullmatch(child):
                    raise EvidenceInferenceItemRiskError(
                        f"item_risk_public_hash_invalid:{location}"
                    )
            elif name.endswith("_sha256s"):
                if not isinstance(child, Mapping) or not child:
                    raise EvidenceInferenceItemRiskError(
                        f"item_risk_public_hash_map_invalid:{location}"
                    )
                for source_path, digest in child.items():
                    _relative_path(str(source_path))
                    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                        raise EvidenceInferenceItemRiskError(
                            f"item_risk_public_hash_map_invalid:{location}"
                        )
            _validate_public_hash_fields(child, path=location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_hash_fields(child, path=f"{path}[{index}]")


def _public_payload(
    *,
    design: EvidenceInferenceItemRiskDesignReceipt,
    materialization: EvidenceInferenceItemRiskMaterializationReceipt,
    calibration_run: ItemRiskCalibrationRunReceipt,
    gepa_summary: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = calibration_run.bundle
    calibration_errors = materialization.calibration_error_count
    calibration_count = materialization.calibration_unit_count
    prediction_source = _prediction_source_lineage(gepa_summary)
    return {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "study_version": STUDY_VERSION,
        "status": "complete_retrospective_nonpristine_diagnostic",
        "scientific_scope": "standalone_real_benchmark_item_scheduling_diagnostic",
        "real_benchmark_annotations": True,
        "all_labels_historically_opened": True,
        "test_is_non_pristine": True,
        "confirmatory_claim_allowed": False,
        "current_verifier_pipeline_compatible": False,
        "prediction_model": {
            "name": gepa_summary["model"]["name"],
            "digest": gepa_summary["model"]["digest"],
            "parameter_size": gepa_summary["model"]["parameter_size"],
            "ollama_version": gepa_summary["model"]["ollama_version"],
            "local_model_only": True,
        },
        "prediction_source": prediction_source,
        "population": {
            "source_paired_examples": materialization.paired_example_count,
            "source_unique_papers": materialization.paired_paper_count,
            "representative_units": materialization.representative_unit_count,
            "development_units": materialization.development_unit_count,
            "calibration_units": calibration_count,
            "development_calibration_question_overlap": 0,
            "development_calibration_paper_overlap": 0,
            "representative_algorithm": REPRESENTATIVE_ALGORITHM,
            "split_algorithm": SPLIT_ALGORITHM,
        },
        "label_free_risk_score": {
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
            "formula": SCORE_FORMULA,
            "score_model_sha256": design.score_model_sha256,
            "bin_definition_source": "prespecified",
            "bin_edges": BIN_EDGES,
            "reference_labels_used_as_features": False,
            "accuracy_or_distribution_metrics_used_as_features": False,
            "model_scalar_score_used_as_feature": False,
            "fabricated_confidence_used": False,
        },
        "calibration": {
            "bundle_version": bundle.bundle_version,
            "bundle_sha256": bundle.bundle_sha256,
            "population_id": bundle.population_id,
            "domain_count": len(bundle.calibration_domains),
            "familywise_delta": bundle.familywise_delta,
            "correction": bundle.correction,
            "cell_rate_estimand": bundle.cell_rate_estimand,
            "release_probability_authority": bundle.release_probability_authority,
            "calibration_observed_errors": calibration_errors,
            "calibration_empirical_error_rate": calibration_errors / calibration_count,
            "bounds": [bound.model_dump(mode="json") for bound in bundle.bounds],
            "risk_score_monotonicity_claimed": False,
        },
        "protocol": {
            "bins_frozen_before_private_paired_report_opened_in_this_rerun": True,
            "historical_label_blinding_claimed": False,
            "one_unit_per_unique_question_and_paper": True,
            "selection_split_and_features_are_label_free": True,
            "paired_predictions_and_labels_co_located_in_source": True,
            "logical_reference_field_access_after_feature_projection": True,
            "physical_label_access_firewall_claimed": False,
            "item_risk_bundle_semantics_reused_without_release_authority": True,
            "design_access_order": design.access_order,
            "materialization_access_order": materialization.access_order,
        },
        "shift_assessment": {
            "status": "not_assessed",
            "prospective_inputs_evaluated": False,
            "no_shift_detected_claimed": False,
            "deployment_scoring_performed": False,
        },
        "lineage": {
            "config_file_sha256": design.config_file_sha256,
            "config_sha256": design.config_sha256,
            "gepa_public_summary_sha256": design.gepa_public_summary_sha256,
            "gepa_private_paired_report_sha256": materialization.paired_report_sha256,
            "gepa_private_paired_report_file_sha256": (materialization.paired_report_file_sha256),
            "prediction_source_lineage_sha256": design.prediction_source_lineage_sha256,
            "diagnostic_pipeline_sha256": design.diagnostic_pipeline_sha256,
            "design_receipt_sha256": design.receipt_sha256,
            "materialization_receipt_sha256": materialization.receipt_sha256,
            "calibration_run_receipt_sha256": calibration_run.receipt_sha256,
            "fixed_bins_receipt_sha256": design.fixed_bins.receipt_sha256,
            "sampling_protocol_sha256": design.sampling_protocol_sha256,
            "adjudication_protocol_sha256": design.adjudication_protocol_sha256,
            "representative_membership_sha256": (materialization.representative_membership_sha256),
            "development_membership_sha256": materialization.development_membership_sha256,
            "calibration_membership_sha256": materialization.calibration_membership_sha256,
        },
        "artifact_boundaries": {
            "contains_article_or_question_text": False,
            "contains_article_question_or_example_identifiers": False,
            "contains_per_example_labels_or_predictions": False,
            "contains_prompt_or_candidate_text": False,
            "contains_absolute_paths": False,
            "private_row_level_material_is_gitignored": True,
        },
        "required_caveats": _REQUIRED_CAVEATS,
    }


def validate_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate aggregate semantics without requiring private row-level caches."""

    snapshot = deepcopy(dict(summary))
    _validate_self_hash(
        snapshot,
        field="public_summary_sha256",
        label="item_risk_public_summary",
    )
    expected_top_level = {
        "all_labels_historically_opened",
        "artifact_boundaries",
        "calibration",
        "confirmatory_claim_allowed",
        "current_verifier_pipeline_compatible",
        "label_free_risk_score",
        "lineage",
        "population",
        "prediction_model",
        "prediction_source",
        "protocol",
        "public_summary_sha256",
        "public_summary_version",
        "real_benchmark_annotations",
        "required_caveats",
        "scientific_scope",
        "shift_assessment",
        "status",
        "study_version",
        "test_is_non_pristine",
    }
    if (
        set(snapshot) != expected_top_level
        or snapshot.get("public_summary_version") != PUBLIC_SUMMARY_VERSION
        or snapshot.get("study_version") != STUDY_VERSION
        or snapshot.get("status") != "complete_retrospective_nonpristine_diagnostic"
        or snapshot.get("scientific_scope")
        != "standalone_real_benchmark_item_scheduling_diagnostic"
        or snapshot.get("real_benchmark_annotations") is not True
        or snapshot.get("all_labels_historically_opened") is not True
        or snapshot.get("test_is_non_pristine") is not True
        or snapshot.get("confirmatory_claim_allowed") is not False
        or snapshot.get("current_verifier_pipeline_compatible") is not False
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_scope_invalid")
    prediction_model = snapshot.get("prediction_model")
    prediction_source = snapshot.get("prediction_source")
    population = snapshot.get("population")
    score = snapshot.get("label_free_risk_score")
    calibration = snapshot.get("calibration")
    protocol = snapshot.get("protocol")
    shift = snapshot.get("shift_assessment")
    boundaries = snapshot.get("artifact_boundaries")
    lineage = snapshot.get("lineage")
    if not all(
        isinstance(value, Mapping)
        for value in (
            prediction_model,
            prediction_source,
            population,
            score,
            calibration,
            protocol,
            shift,
            boundaries,
            lineage,
        )
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_sections_missing")
    assert isinstance(prediction_model, Mapping)
    assert isinstance(prediction_source, Mapping)
    assert isinstance(population, Mapping)
    assert isinstance(score, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(shift, Mapping)
    assert isinstance(boundaries, Mapping)
    assert isinstance(lineage, Mapping)
    if (
        set(prediction_model)
        != {"digest", "local_model_only", "name", "ollama_version", "parameter_size"}
        or prediction_model.get("name") != "llama3.2:1b"
        or prediction_model.get("parameter_size") != "1.2B"
        or prediction_model.get("local_model_only") is not True
        or not isinstance(prediction_model.get("digest"), str)
        or not SHA256_RE.fullmatch(str(prediction_model["digest"]))
        or not isinstance(prediction_model.get("ollama_version"), str)
        or not prediction_model["ollama_version"]
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_prediction_model_invalid")
    prediction_source_fields = {
        "arm_order_policy",
        "evaluation_schema_sha256",
        "generation_config_sha256",
        "generation_schema_algorithm",
        "generation_schema_base_sha256",
        "gepa_public_summary_sha256",
        "gepa_study_version",
        "historical_source_code_sha256s",
        "lineage_version",
        "manifest_file_sha256",
        "model_digest",
        "model_identity_sha256",
        "model_name",
        "model_parameter_size",
        "ollama_version",
        "paired_articles",
        "paired_examples",
        "plan_file_sha256",
        "plan_sha256",
        "prediction_source_lineage_sha256",
        "private_paired_report_file_sha256",
        "private_paired_report_sha256",
        "seed_prompt_sha256",
        "test_split_jsonl_sha256",
        "winner_bundle_file_sha256",
        "winner_prompt_sha256",
        "winner_sha256",
    }
    source_payload = {
        key: value
        for key, value in prediction_source.items()
        if key != "prediction_source_lineage_sha256"
    }
    if (
        set(prediction_source) != prediction_source_fields
        or prediction_source.get("lineage_version")
        != "evidence-inference-gepa-prediction-source-v1"
        or hash_canonical(source_payload)
        != prediction_source.get("prediction_source_lineage_sha256")
        or prediction_source.get("model_name") != prediction_model["name"]
        or prediction_source.get("model_digest") != prediction_model["digest"]
        or prediction_source.get("model_parameter_size")
        != prediction_model["parameter_size"]
        or prediction_source.get("ollama_version") != prediction_model["ollama_version"]
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_prediction_source_invalid")
    historical_sources = prediction_source.get("historical_source_code_sha256s")
    if not isinstance(historical_sources, Mapping) or not historical_sources:
        raise EvidenceInferenceItemRiskError("item_risk_public_prediction_source_invalid")
    population_count_fields = (
        "source_paired_examples",
        "source_unique_papers",
        "representative_units",
        "development_units",
        "calibration_units",
    )
    if (
        set(population)
        != {
            *population_count_fields,
            "development_calibration_paper_overlap",
            "development_calibration_question_overlap",
            "representative_algorithm",
            "split_algorithm",
        }
        or any(
            isinstance(population.get(key), bool)
            or not isinstance(population.get(key), int)
            or int(population[key]) <= 0
            for key in population_count_fields
        )
        or population.get("source_paired_examples")
        != prediction_source.get("paired_examples")
        or population.get("source_unique_papers")
        != prediction_source.get("paired_articles")
        or population.get("representative_units")
        != population.get("development_units", 0) + population.get("calibration_units", 0)
        or population.get("source_paired_examples", 0)
        < population.get("representative_units", 0)
        or population.get("representative_units") != population.get("source_unique_papers")
        or population.get("development_calibration_question_overlap") != 0
        or population.get("development_calibration_paper_overlap") != 0
        or population.get("representative_algorithm") != REPRESENTATIVE_ALGORITHM
        or population.get("split_algorithm") != SPLIT_ALGORITHM
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_population_invalid")
    if (
        set(score)
        != {
            "accuracy_or_distribution_metrics_used_as_features",
            "bin_definition_source",
            "bin_edges",
            "fabricated_confidence_used",
            "feature_names",
            "feature_version",
            "formula",
            "model_scalar_score_used_as_feature",
            "reference_labels_used_as_features",
            "score_model_sha256",
        }
        or score.get("feature_version") != FEATURE_VERSION
        or score.get("feature_names") != FEATURE_NAMES
        or score.get("formula") != SCORE_FORMULA
        or score.get("bin_edges") != BIN_EDGES
        or any(
            score.get(key) is not False
            for key in (
                "reference_labels_used_as_features",
                "accuracy_or_distribution_metrics_used_as_features",
                "model_scalar_score_used_as_feature",
                "fabricated_confidence_used",
            )
        )
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_score_contract_invalid")
    if (
        set(calibration)
        != {
            "bounds",
            "bundle_sha256",
            "bundle_version",
            "calibration_empirical_error_rate",
            "calibration_observed_errors",
            "cell_rate_estimand",
            "correction",
            "domain_count",
            "familywise_delta",
            "population_id",
            "release_probability_authority",
            "risk_score_monotonicity_claimed",
        }
        or calibration.get("bundle_version") != "item-risk-calibration-v2"
        or calibration.get("population_id") != POPULATION_ID
        or calibration.get("domain_count") != 1
        or not math.isclose(
            float(calibration.get("familywise_delta", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or calibration.get("release_probability_authority") is not False
        or calibration.get("risk_score_monotonicity_claimed") is not False
        or calibration.get("cell_rate_estimand")
        != "group_average_item_error_rate_within_domain_score_bin"
        or calibration.get("correction") != "bonferroni-clopper-pearson"
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_calibration_scope_invalid")
    raw_bounds = calibration.get("bounds")
    if not isinstance(raw_bounds, list) or not raw_bounds:
        raise EvidenceInferenceItemRiskError("item_risk_public_bounds_missing")
    try:
        bounds = [DomainRiskBinCalibration.model_validate(value) for value in raw_bounds]
    except ValueError as exc:
        raise EvidenceInferenceItemRiskError("item_risk_public_bound_invalid") from exc
    if (
        [bound.bin_id for bound in bounds]
        != ["risk-bin-000", "risk-bin-001", "risk-bin-002"]
        or any(
            bound.domain != DOMAIN
            or bound.family_cell_count != len(bounds)
            or not math.isclose(
                bound.familywise_delta,
                float(calibration["familywise_delta"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for bound in bounds
        )
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_bound_family_invalid")
    calibration_units = int(population["calibration_units"])
    calibration_errors = int(calibration["calibration_observed_errors"])
    if (
        sum(bound.cell_calibration_units for bound in bounds) != calibration_units
        or sum(bound.cell_observed_errors for bound in bounds) != calibration_errors
        or not math.isclose(
            float(calibration["calibration_empirical_error_rate"]),
            calibration_errors / calibration_units,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_bound_totals_invalid")
    if (
        set(protocol)
        != {
            "bins_frozen_before_private_paired_report_opened_in_this_rerun",
            "design_access_order",
            "historical_label_blinding_claimed",
            "item_risk_bundle_semantics_reused_without_release_authority",
            "logical_reference_field_access_after_feature_projection",
            "materialization_access_order",
            "one_unit_per_unique_question_and_paper",
            "paired_predictions_and_labels_co_located_in_source",
            "physical_label_access_firewall_claimed",
            "selection_split_and_features_are_label_free",
        }
        or protocol.get("bins_frozen_before_private_paired_report_opened_in_this_rerun") is not True
        or protocol.get("historical_label_blinding_claimed") is not False
        or protocol.get("one_unit_per_unique_question_and_paper") is not True
        or protocol.get("selection_split_and_features_are_label_free") is not True
        or protocol.get("paired_predictions_and_labels_co_located_in_source") is not True
        or protocol.get("logical_reference_field_access_after_feature_projection") is not True
        or protocol.get("physical_label_access_firewall_claimed") is not False
        or protocol.get("item_risk_bundle_semantics_reused_without_release_authority") is not True
        or protocol.get("design_access_order") != _DESIGN_ACCESS_ORDER
        or protocol.get("materialization_access_order") != _MATERIALIZATION_ACCESS_ORDER
        or shift
        != {
            "status": "not_assessed",
            "prospective_inputs_evaluated": False,
            "no_shift_detected_claimed": False,
            "deployment_scoring_performed": False,
        }
        or snapshot.get("required_caveats") != _REQUIRED_CAVEATS
        or any(
            value is not False
            for key, value in boundaries.items()
            if key != "private_row_level_material_is_gitignored"
        )
        or boundaries.get("private_row_level_material_is_gitignored") is not True
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_caveat_contract_invalid")
    expected_lineage_fields = {
        "adjudication_protocol_sha256",
        "calibration_membership_sha256",
        "calibration_run_receipt_sha256",
        "config_file_sha256",
        "config_sha256",
        "design_receipt_sha256",
        "development_membership_sha256",
        "diagnostic_pipeline_sha256",
        "fixed_bins_receipt_sha256",
        "gepa_private_paired_report_file_sha256",
        "gepa_private_paired_report_sha256",
        "gepa_public_summary_sha256",
        "materialization_receipt_sha256",
        "prediction_source_lineage_sha256",
        "representative_membership_sha256",
        "sampling_protocol_sha256",
    }
    if (
        set(lineage) != expected_lineage_fields
        or lineage.get("prediction_source_lineage_sha256")
        != prediction_source.get("prediction_source_lineage_sha256")
        or lineage.get("gepa_public_summary_sha256")
        != prediction_source.get("gepa_public_summary_sha256")
        or lineage.get("gepa_private_paired_report_sha256")
        != prediction_source.get("private_paired_report_sha256")
        or lineage.get("gepa_private_paired_report_file_sha256")
        != prediction_source.get("private_paired_report_file_sha256")
    ):
        raise EvidenceInferenceItemRiskError("item_risk_public_lineage_invalid")
    _validate_public_hash_fields(snapshot)
    _public_redaction_check(snapshot)
    return dict(summary)


def build_public_summary(
    *,
    repository_root: Path,
    config_path: Path,
    work_dir: Path,
    calibration_run_path: Path,
    expected_design_receipt_sha256: str,
    expected_materialization_receipt_sha256: str,
    expected_calibration_run_receipt_sha256: str,
    output_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Project fully validated private artifacts to a content-silent aggregate."""

    _preflight_outputs([output_path], force=force)
    config = load_config(config_path)
    _require_configured_work_dir(
        repository_root=repository_root,
        config=config,
        work_dir=work_dir,
    )
    _require_configured_public_output(
        repository_root=repository_root,
        config=config,
        output_path=output_path,
    )
    _require_calibration_run_path(
        work_dir=work_dir,
        calibration_run_path=calibration_run_path,
    )
    try:
        design = EvidenceInferenceItemRiskDesignReceipt.model_validate(
            _read_json_object(_design_paths(work_dir)["design"], label="design_receipt")
        )
        materialization = EvidenceInferenceItemRiskMaterializationReceipt.model_validate(
            _read_json_object(
                _materialization_paths(work_dir)["materialization"],
                label="materialization_receipt",
            )
        )
        calibration_run = ItemRiskCalibrationRunReceipt.model_validate(
            _read_json_object(calibration_run_path, label="item_risk_calibration_run")
        )
    except ValueError as exc:
        raise EvidenceInferenceItemRiskError("item_risk_private_artifact_invalid") from exc
    if (
        design.receipt_sha256 != expected_design_receipt_sha256
        or materialization.receipt_sha256 != expected_materialization_receipt_sha256
        or calibration_run.receipt_sha256 != expected_calibration_run_receipt_sha256
    ):
        raise EvidenceInferenceItemRiskError("item_risk_external_receipt_hash_mismatch")
    gepa_summary_path = repository_root / config.gepa_public_summary_path
    gepa_summary = _load_gepa_public_summary(gepa_summary_path)
    try:
        require_pipeline_fingerprint_match(
            expected=design.diagnostic_pipeline_fingerprint,
            root=repository_root,
        )
    except PipelineFingerprintError as exc:
        raise EvidenceInferenceItemRiskError("diagnostic_pipeline_changed") from exc
    design_paths = _design_paths(work_dir)
    materialization_paths = _materialization_paths(work_dir)
    if (
        design.config_sha256 != config.config_sha256
        or design.config_file_sha256 != sha256_file(config_path)
        or design.gepa_public_summary_sha256 != gepa_summary["public_summary_sha256"]
        or design.gepa_public_summary_file_sha256 != sha256_file(gepa_summary_path)
        or materialization.design_receipt_sha256 != design.receipt_sha256
        or materialization.design_receipt_file_sha256 != sha256_file(design_paths["design"])
        or materialization.prediction_source_lineage_sha256
        != design.prediction_source_lineage_sha256
        or materialization.gepa_public_summary_sha256 != design.gepa_public_summary_sha256
        or materialization.gepa_public_summary_file_sha256
        != design.gepa_public_summary_file_sha256
        or materialization.development_units_file_sha256
        != sha256_file(materialization_paths["development"])
        or materialization.calibration_units_file_sha256
        != sha256_file(materialization_paths["calibration"])
        or calibration_run.expected_pipeline_file_sha256
        != sha256_file(design_paths["pipeline"])
        or calibration_run.fixed_bins_file_sha256 != sha256_file(design_paths["bins"])
        or design.fixed_bins_file_sha256 != calibration_run.fixed_bins_file_sha256
        or design.fixed_bins.receipt_sha256 != calibration_run.fixed_bins_receipt_sha256
        or calibration_run.development_units_file_sha256
        != materialization.development_units_file_sha256
        or calibration_run.calibration_units_file_sha256
        != materialization.calibration_units_file_sha256
        or calibration_run.bundle.pipeline_sha256 != design.diagnostic_pipeline_sha256
        or calibration_run.bundle.score_model_sha256 != design.score_model_sha256
        or calibration_run.bundle.sampling_protocol_sha256 != design.sampling_protocol_sha256
        or calibration_run.bundle.adjudication_protocol_sha256
        != design.adjudication_protocol_sha256
        or calibration_run.bundle.shift_detector_id != SHIFT_DETECTOR_ID
        or calibration_run.bundle.shift_detector_sha256 != design.shift_detector_sha256
        or calibration_run.bundle.population_id != POPULATION_ID
        or calibration_run.bundle.release_probability_authority is not False
        or calibration_run.development_unit_count != materialization.development_unit_count
        or calibration_run.calibration_unit_count != materialization.calibration_unit_count
    ):
        raise EvidenceInferenceItemRiskError("item_risk_private_lineage_mismatch")
    payload = _public_payload(
        design=design,
        materialization=materialization,
        calibration_run=calibration_run,
        gepa_summary=gepa_summary,
    )
    summary = {**payload, "public_summary_sha256": hash_canonical(payload)}
    validate_public_summary(summary)
    _public_redaction_check(summary, repository_root=repository_root)
    atomic_write_json(output_path, summary, force=force)
    return summary


def validate_public_summary_file(path: Path) -> dict[str, Any]:
    return validate_public_summary(_read_json_object(path, label="item_risk_public_summary"))


__all__ = [
    "BIN_EDGES",
    "DOMAIN",
    "ERROR_EVENT_DEFINITION",
    "FEATURE_NAMES",
    "POPULATION_ID",
    "PUBLIC_SUMMARY_VERSION",
    "SHIFT_DETECTOR_ID",
    "STUDY_VERSION",
    "EvidenceInferenceItemRiskConfig",
    "EvidenceInferenceItemRiskDesignReceipt",
    "EvidenceInferenceItemRiskError",
    "EvidenceInferenceItemRiskMaterializationReceipt",
    "build_public_summary",
    "compute_diagnostic_pipeline_fingerprint",
    "freeze_design",
    "label_free_feature_projection",
    "load_config",
    "materialize_units",
    "validate_public_summary",
    "validate_public_summary_file",
]
