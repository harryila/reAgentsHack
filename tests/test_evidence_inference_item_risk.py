from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import literature_multiverse.evidence_inference_item_risk as item_risk_module
import literature_multiverse.public_artifacts as public_artifacts_module
from literature_multiverse.evidence_inference_item_risk import (
    EvidenceInferenceItemRiskError,
    compute_diagnostic_pipeline_fingerprint,
    freeze_design,
    label_free_feature_projection,
    load_config,
    validate_public_summary,
)
from literature_multiverse.item_risk_calibration import (
    _one_sided_clopper_pearson_upper,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.public_artifacts import (
    PUBLIC_RESULT_REGISTRY,
    PublicArtifactValidationError,
    _validate_evidence_inference_item_risk,
)

_CONFIG = Path("configs/benchmarks/evidence-inference-item-risk-v1.json")
_SUMMARY = Path(
    "artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json"
)


def _load_summary(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / _SUMMARY).read_text(encoding="utf-8"))


def _rehash_summary(summary: dict[str, Any]) -> None:
    summary["public_summary_sha256"] = hash_canonical(
        {key: value for key, value in summary.items() if key != "public_summary_sha256"}
    )


def _synthetic_row() -> dict[str, Any]:
    arm = {
        "predicted_direction": "increase",
        "objective_scores": {
            "structured_output_validity": 1.0,
            "formal_grounding_validity": 1.0,
            "direction_accuracy": 0.0,
            "scalar_objective": 0.0,
        },
        "confidence": 0.99,
        "output": {},
    }
    return {
        "example_id": "fixture-example",
        "paper_id": "fixture-paper",
        "group_id": "fixture-paper",
        "expected_direction": "decrease",
        "seed": deepcopy(arm),
        "winner": deepcopy(arm),
    }


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
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
        candidates = [Path(*module.split(".")).with_suffix(".py")]
    else:
        return None
    return next(
        (
            candidate.as_posix()
            for candidate in candidates
            if (repository_root / candidate).is_file()
        ),
        None,
    )


def _independent_python_dependency_closure(repository_root: Path) -> set[str]:
    pending = list(item_risk_module._PIPELINE_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse((repository_root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def test_checked_in_evidence_inference_item_risk_summary_is_publicly_replayable(
    repo_root: Path,
) -> None:
    summary = _load_summary(repo_root)
    validated = validate_public_summary(summary)
    _validate_evidence_inference_item_risk(summary, root=repo_root)

    assert validated["population"] == {
        "source_paired_examples": 524,
        "source_unique_papers": 191,
        "representative_units": 191,
        "development_units": 82,
        "calibration_units": 109,
        "development_calibration_question_overlap": 0,
        "development_calibration_paper_overlap": 0,
        "representative_algorithm": (
            "minimum-namespaced-sha256-example-per-paper-v1"
        ),
        "split_algorithm": (
            "namespaced-sha256-paper-modulo-5-buckets-0-or-1-development-v1"
        ),
    }
    assert validated["calibration"]["calibration_observed_errors"] == 75
    assert validated["calibration"]["release_probability_authority"] is False
    assert validated["calibration"]["risk_score_monotonicity_claimed"] is False
    assert validated["shift_assessment"]["status"] == "not_assessed"
    assert validated["current_verifier_pipeline_compatible"] is False
    assert validated["confirmatory_claim_allowed"] is False
    spec = next(spec for spec in PUBLIC_RESULT_REGISTRY if spec.path == _SUMMARY.as_posix())
    assert spec.semantic_validator == "evidence_inference_item_risk"
    assert spec.result_recomputed_from_public_inputs is False


def test_public_registry_replay_never_opens_ignored_cache(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _load_summary(repo_root)
    opened: list[Path] = []
    original_item_risk_loader = item_risk_module._read_json_object
    original_registry_loader = public_artifacts_module._load_json_object

    def guarded_item_risk_loader(path: Path, *, label: str) -> dict[str, Any]:
        opened.append(path)
        assert "data/cache" not in path.as_posix()
        return original_item_risk_loader(path, label=label)

    def guarded_registry_loader(path: Path) -> dict[str, Any]:
        opened.append(path)
        assert "data/cache" not in path.as_posix()
        return original_registry_loader(path)

    monkeypatch.setattr(item_risk_module, "_read_json_object", guarded_item_risk_loader)
    monkeypatch.setattr(public_artifacts_module, "_load_json_object", guarded_registry_loader)
    _validate_evidence_inference_item_risk(summary, root=repo_root)

    assert opened
    bound_paths = {
        *item_risk_module._PIPELINE_NONPYTHON_PATHS,
        *item_risk_module._diagnostic_python_dependency_closure(repo_root),
    }
    assert all(not path.startswith("data/cache/") for path in bound_paths)


def test_standalone_pipeline_binds_exact_local_dependency_closure(repo_root: Path) -> None:
    config = load_config(repo_root / _CONFIG)
    gepa_summary = json.loads(
        (repo_root / config.gepa_public_summary_path).read_text(encoding="utf-8")
    )
    fingerprint, _ = compute_diagnostic_pipeline_fingerprint(
        repository_root=repo_root,
        config=config,
        gepa_public_summary=gepa_summary,
    )
    component = fingerprint.components[0]
    expected_python = _independent_python_dependency_closure(repo_root)
    expected_files = {*item_risk_module._PIPELINE_NONPYTHON_PATHS, *expected_python}

    assert component.component_version == "2"
    assert {record.path for record in component.files} == expected_files
    assert component.settings["in_repository_dependency_closure_bound"] is True
    assert component.settings["dependency_closure_entrypoints"] == list(
        item_risk_module._PIPELINE_DEPENDENCY_ENTRYPOINTS
    )


@pytest.mark.parametrize("tamper", ["population", "upper_bound", "prediction_source"])
def test_public_validator_rejects_coherently_rehashed_semantic_tamper(
    repo_root: Path,
    tamper: str,
) -> None:
    summary = deepcopy(_load_summary(repo_root))
    if tamper == "population":
        summary["population"]["development_units"] += 1
    elif tamper == "upper_bound":
        summary["calibration"]["bounds"][0]["upper_cell_error_rate"] -= 0.01
    else:
        source = summary["prediction_source"]
        source["model_name"] = "invented-model"
        source["prediction_source_lineage_sha256"] = hash_canonical(
            {
                key: value
                for key, value in source.items()
                if key != "prediction_source_lineage_sha256"
            }
        )
        summary["lineage"]["prediction_source_lineage_sha256"] = source[
            "prediction_source_lineage_sha256"
        ]
    _rehash_summary(summary)

    with pytest.raises(EvidenceInferenceItemRiskError):
        validate_public_summary(summary)


def test_public_registry_rejects_coherently_rehashed_registered_result_tamper(
    repo_root: Path,
) -> None:
    summary = deepcopy(_load_summary(repo_root))
    first, second = summary["calibration"]["bounds"][:2]
    first["cell_calibration_units"] -= 1
    first["cell_observed_errors"] -= 1
    second["cell_calibration_units"] += 1
    second["cell_observed_errors"] += 1
    for bound in (first, second):
        bound["empirical_cell_error_rate"] = (
            bound["cell_observed_errors"] / bound["cell_calibration_units"]
        )
        bound["upper_cell_error_rate"] = _one_sided_clopper_pearson_upper(
            bound["cell_observed_errors"],
            bound["cell_calibration_units"],
            delta=bound["cellwise_delta"],
        )
    _rehash_summary(summary)

    validate_public_summary(summary)
    with pytest.raises(
        ValueError,
        match="evidence_inference_item_risk_registered_result_mismatch",
    ):
        _validate_evidence_inference_item_risk(summary, root=repo_root)


def test_label_free_projection_is_reference_and_confidence_invariant() -> None:
    row = _synthetic_row()
    baseline = label_free_feature_projection(row)
    row["expected_direction"] = "no_effect"
    row["seed"]["confidence"] = 0.01
    row["winner"]["objective_scores"]["direction_accuracy"] = 1.0
    row["winner"]["objective_scores"]["scalar_objective"] = 1.0

    assert label_free_feature_projection(row) == baseline
    assert baseline["risk_score"] == 0.0


def test_label_free_projection_uses_only_declared_failure_flags() -> None:
    row = _synthetic_row()
    row["winner"]["objective_scores"]["structured_output_validity"] = 0.0
    projected = label_free_feature_projection(row)

    assert projected["flags"] == {
        "seed_winner_direction_disagreement": False,
        "seed_structured_output_invalid": False,
        "winner_structured_output_invalid": True,
        "seed_exact_grounding_invalid": False,
        "winner_exact_grounding_invalid": False,
    }
    assert projected["risk_score"] == 0.2


def test_config_and_freeze_reject_unregistered_work_directory(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    config_path = repo_root / _CONFIG
    config = load_config(config_path)
    assert config.private_run_dir.endswith("-final-v5")

    with pytest.raises(EvidenceInferenceItemRiskError, match="work_dir_config_mismatch"):
        freeze_design(
            repository_root=repo_root,
            config_path=config_path,
            work_dir=tmp_path / "invented-run",
        )


def test_public_summary_contains_no_row_identifiers_or_absolute_paths(
    repo_root: Path,
) -> None:
    summary = _load_summary(repo_root)
    serialized = json.dumps(summary, sort_keys=True)

    assert "PMC" not in serialized
    assert repo_root.as_posix() not in serialized
    assert '"example_id"' not in serialized
    assert '"paper_id"' not in serialized
    assert '"question_id"' not in serialized
    assert '"expected_direction"' not in serialized
    assert '"predicted_direction"' not in serialized


_HISTORICAL_FINGERPRINT = Path(
    "artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json"
)


def _stand_in_fingerprint(**overrides: Any):
    def factory(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        fingerprint, source = _REAL_COMPUTE(**kwargs)
        component = fingerprint.components[0]
        return (
            SimpleNamespace(
                pipeline_sha256=overrides.get("pipeline_sha256", fingerprint.pipeline_sha256),
                components=[
                    SimpleNamespace(
                        component_id=component.component_id,
                        component_version=component.component_version,
                        files=overrides.get("files", component.files),
                    )
                ],
            ),
            source,
        )

    return factory


_REAL_COMPUTE = public_artifacts_module.compute_ei_item_risk_pipeline_fingerprint


def test_item_risk_public_validation_binds_historical_fingerprint_not_current_tree(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        public_artifacts_module,
        "compute_ei_item_risk_pipeline_fingerprint",
        _stand_in_fingerprint(pipeline_sha256="0" * 64),
    )
    _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_closure_definition_drift_fails_closed(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, _ = _REAL_COMPUTE(
        repository_root=repo_root,
        config=load_config(repo_root / _CONFIG),
        gepa_public_summary=json.loads(
            (repo_root / load_config(repo_root / _CONFIG).gepa_public_summary_path).read_text(
                encoding="utf-8"
            )
        ),
    )
    monkeypatch.setattr(
        public_artifacts_module,
        "compute_ei_item_risk_pipeline_fingerprint",
        _stand_in_fingerprint(files=list(reversed(real.components[0].files))),
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_historical_closure_definition_mismatch",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_historical_fingerprint_manifest_tamper_fails_closed(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = public_artifacts_module._load_json_object
    target = (repo_root / _HISTORICAL_FINGERPRINT).resolve()

    def tampered(path: Path) -> dict[str, Any]:
        value = original(path)
        if Path(path).resolve() == target:
            value = json.loads(json.dumps(value))
            value["components"][0]["files"][0]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(public_artifacts_module, "_load_json_object", tampered)
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_historical_fingerprint_invalid",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_coherently_rehashed_manifest_cannot_replace_the_pin(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = public_artifacts_module._load_json_object
    target = (repo_root / _HISTORICAL_FINGERPRINT).resolve()

    def rehashed(path: Path) -> dict[str, Any]:
        value = original(path)
        if Path(path).resolve() == target:
            value = json.loads(json.dumps(value))
            component = value["components"][0]
            component["files"][0]["sha256"] = "1" * 64
            component["component_sha256"] = hash_canonical(
                {k: v for k, v in component.items() if k != "component_sha256"}
            )
            value["pipeline_sha256"] = hash_canonical(
                {k: v for k, v in value.items() if k != "pipeline_sha256"}
            )
        return value

    monkeypatch.setattr(public_artifacts_module, "_load_json_object", rehashed)
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_current_lineage_mismatch",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)
