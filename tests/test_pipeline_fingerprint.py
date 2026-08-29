from __future__ import annotations

import ast
import shutil
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import literature_multiverse.verifier as verifier_module
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprintError,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
    validate_pipeline_fingerprint_integrity,
    verify_pipeline_fingerprint,
)
from literature_multiverse.verifier import (
    compute_verifier_pipeline_fingerprint,
    verifier_pipeline_components,
)

_EXPECTED_NATIVE_COMPONENT_FILES = {
    "configs/benchmarks/native-antiox-bounded-v1.json",
    "prompts/native_candidate_inventory.md",
    "prompts/native_candidate_packet.md",
    "prompts/native_extraction.md",
    "scripts/build_hosted_native_grounding_package.py",
    "scripts/build_native_source_manifest.py",
    "scripts/build_typed_evidence_corpus.py",
    "scripts/reconcile_native_cohorts.py",
    "scripts/run_native_ollama_diagnostic.py",
    "scripts/run_native_bounded_ollama_diagnostic.py",
    "scripts/s3_extract_typed.py",
    "src/literature_multiverse/acquisition.py",
    "src/literature_multiverse/cohort_reconciliation.py",
    "src/literature_multiverse/extract.py",
    "src/literature_multiverse/grounding.py",
    "src/literature_multiverse/harvester/archive.py",
    "src/literature_multiverse/harvester/contracts.py",
    "src/literature_multiverse/harvester/http.py",
    "src/literature_multiverse/harvester/pipeline.py",
    "src/literature_multiverse/harvester/sources.py",
    "src/literature_multiverse/hosted_native_extraction_contract.py",
    "src/literature_multiverse/hosted_native_grounding_bridge.py",
    "src/literature_multiverse/live.py",
    "src/literature_multiverse/local_ollama.py",
    "src/literature_multiverse/metasyn_benchmark.py",
    "src/literature_multiverse/metasyn_retrieval.py",
    "src/literature_multiverse/native_extraction.py",
    "src/literature_multiverse/native_bounded_generation.py",
    "src/literature_multiverse/native_bounded_ollama_diagnostic.py",
    "src/literature_multiverse/native_grounding.py",
    "src/literature_multiverse/native_ollama_diagnostic.py",
    "src/literature_multiverse/normalize.py",
    "src/literature_multiverse/paperclip_cli.py",
    "src/literature_multiverse/prompting.py",
    "src/literature_multiverse/screen.py",
    "src/literature_multiverse/search.py",
    "src/literature_multiverse/source_manifest_bridge.py",
    "src/literature_multiverse/typed_extraction.py",
}

_EXPECTED_NATIVE_PYTHON_CLOSURE = {
    "scripts/build_hosted_native_grounding_package.py",
    "scripts/build_native_source_manifest.py",
    "scripts/build_typed_evidence_corpus.py",
    "scripts/reconcile_native_cohorts.py",
    "scripts/run_native_ollama_diagnostic.py",
    "scripts/run_native_bounded_ollama_diagnostic.py",
    "scripts/s3_extract_typed.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/acquisition.py",
    "src/literature_multiverse/adaptive_calibration.py",
    "src/literature_multiverse/audit_session.py",
    "src/literature_multiverse/budgeted_verification.py",
    "src/literature_multiverse/calibration.py",
    "src/literature_multiverse/certificate.py",
    "src/literature_multiverse/claim_release.py",
    "src/literature_multiverse/claim_semantics.py",
    "src/literature_multiverse/cohort_reconciliation.py",
    "src/literature_multiverse/condition_confirmation.py",
    "src/literature_multiverse/config.py",
    "src/literature_multiverse/effects.py",
    "src/literature_multiverse/evidence_graph.py",
    "src/literature_multiverse/extract.py",
    "src/literature_multiverse/grounding.py",
    "src/literature_multiverse/harvester/archive.py",
    "src/literature_multiverse/harvester/contracts.py",
    "src/literature_multiverse/harvester/http.py",
    "src/literature_multiverse/harvester/pipeline.py",
    "src/literature_multiverse/harvester/sources.py",
    "src/literature_multiverse/hosted_native_extraction_contract.py",
    "src/literature_multiverse/hosted_native_grounding_bridge.py",
    "src/literature_multiverse/independence_identity.py",
    "src/literature_multiverse/item_risk_artifacts.py",
    "src/literature_multiverse/item_risk_calibration.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/live.py",
    "src/literature_multiverse/local_ollama.py",
    "src/literature_multiverse/meta_analysis.py",
    "src/literature_multiverse/metasyn_benchmark.py",
    "src/literature_multiverse/metasyn_retrieval.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/native_extraction.py",
    "src/literature_multiverse/native_bounded_generation.py",
    "src/literature_multiverse/native_bounded_ollama_diagnostic.py",
    "src/literature_multiverse/native_grounding.py",
    "src/literature_multiverse/native_ollama_diagnostic.py",
    "src/literature_multiverse/normalize.py",
    "src/literature_multiverse/paperclip_cli.py",
    "src/literature_multiverse/paths.py",
    "src/literature_multiverse/pipeline_fingerprint.py",
    "src/literature_multiverse/production_policy.py",
    "src/literature_multiverse/prompting.py",
    "src/literature_multiverse/records.py",
    "src/literature_multiverse/schemas.py",
    "src/literature_multiverse/screen.py",
    "src/literature_multiverse/search.py",
    "src/literature_multiverse/sequential_verification.py",
    "src/literature_multiverse/source_manifest_bridge.py",
    "src/literature_multiverse/typed_extraction.py",
    "src/literature_multiverse/verifier.py",
}

_EXPECTED_VERIFICATION_RELEASE_FILES = {
    "scripts/build_condition_calibration_trajectory.py",
    "scripts/build_question_replay_state.py",
    "scripts/calibrate_adaptive_release.py",
    "scripts/calibrate_item_risk.py",
    "scripts/evaluate_question_benchmark.py",
    "src/literature_multiverse/adaptive_calibration.py",
    "src/literature_multiverse/audit_session.py",
    "src/literature_multiverse/budgeted_verification.py",
    "src/literature_multiverse/calibration.py",
    "src/literature_multiverse/certificate.py",
    "src/literature_multiverse/claim_release.py",
    "src/literature_multiverse/cli.py",
    "src/literature_multiverse/condition_trajectory_builder.py",
    "src/literature_multiverse/item_risk_artifacts.py",
    "src/literature_multiverse/item_risk_calibration.py",
    "src/literature_multiverse/pipeline_fingerprint.py",
    "src/literature_multiverse/production_policy.py",
    "src/literature_multiverse/question_evaluation.py",
    "src/literature_multiverse/sequential_verification.py",
    "src/literature_multiverse/verifier.py",
}


def _resolve_local_module(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    """Resolve one AST import to its repository-local Python source, if any."""

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


def _native_python_dependency_closure(repository_root: Path) -> set[str]:
    """Mechanically walk imports from every executable native pipeline entry point."""

    pending = [path for path in _EXPECTED_NATIVE_COMPONENT_FILES if path.endswith(".py")]
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (repository_root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_module(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def _verifier_python_dependency_closure(repository_root: Path) -> set[str]:
    """Walk every executable Python file declared by the full verifier manifest."""

    declared_paths = {
        path for component in verifier_pipeline_components() for path in component.file_paths
    }
    pending = [path for path in declared_paths if path.endswith(".py")]
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (repository_root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_module(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def _components() -> list[PipelineComponentSpec]:
    return [
        PipelineComponentSpec(
            component_id="extractor",
            component_version="1",
            file_paths=["config.json", "src/extract.py"],
            settings={"model": "frozen-model", "temperature": 0},
        )
    ]


def _write_pipeline(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src/extract.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "config.json").write_text('{"threshold": 0.5}\n', encoding="utf-8")


def test_computed_pipeline_fingerprint_matches_exact_files(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    fingerprint = compute_pipeline_fingerprint(root=tmp_path, components=_components())

    verification = verify_pipeline_fingerprint(expected=fingerprint, root=tmp_path)

    assert verification.status == "matched"
    assert verification.issues == []
    assert verification.computed_pipeline_sha256 == fingerprint.pipeline_sha256
    assert (
        require_pipeline_fingerprint_match(expected=fingerprint, root=tmp_path).verification_sha256
        == verification.verification_sha256
    )


def test_pipeline_verification_detects_changed_bytes(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    expected = compute_pipeline_fingerprint(root=tmp_path, components=_components())
    (tmp_path / "src/extract.py").write_text("VALUE = 2\n", encoding="utf-8")

    verification = verify_pipeline_fingerprint(expected=expected, root=tmp_path)

    assert verification.status == "mismatch"
    assert "file_sha256_mismatch:src/extract.py" in verification.issues
    assert "component_sha256_mismatch:extractor" in verification.issues
    assert "pipeline_sha256_mismatch" in verification.issues
    with pytest.raises(PipelineFingerprintError, match="pipeline_fingerprint_not_matched"):
        require_pipeline_fingerprint_match(expected=expected, root=tmp_path)


def test_pipeline_verification_fails_closed_for_missing_file(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    expected = compute_pipeline_fingerprint(root=tmp_path, components=_components())
    (tmp_path / "config.json").unlink()

    verification = verify_pipeline_fingerprint(expected=expected, root=tmp_path)

    assert verification.status == "unverifiable"
    assert verification.computed is None
    assert verification.issues == ["pipeline_file_missing:config.json"]


def test_nested_fingerprint_mutation_cannot_retain_old_hash(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    fingerprint = compute_pipeline_fingerprint(root=tmp_path, components=_components())
    fingerprint.components.reverse()
    fingerprint.components[0].files.reverse()

    with pytest.raises(PipelineFingerprintError, match="integrity_changed"):
        validate_pipeline_fingerprint_integrity(fingerprint)


def test_pipeline_manifest_rejects_traversal_and_duplicate_file_ownership(
    tmp_path: Path,
) -> None:
    _write_pipeline(tmp_path)
    with pytest.raises(ValueError, match="normalized_repository_relative"):
        PipelineComponentSpec(
            component_id="bad",
            component_version="1",
            file_paths=["../secret"],
        )

    components = [
        PipelineComponentSpec(
            component_id="one", component_version="1", file_paths=["config.json"]
        ),
        PipelineComponentSpec(
            component_id="two", component_version="1", file_paths=["config.json"]
        ),
    ]
    with pytest.raises(PipelineFingerprintError, match="multiple_components"):
        compute_pipeline_fingerprint(root=tmp_path, components=components)


@pytest.mark.parametrize("changed_path", sorted(_EXPECTED_NATIVE_COMPONENT_FILES))
def test_native_dependency_bytes_are_bound_to_verifier_fingerprint(
    tmp_path: Path,
    changed_path: str,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    native_spec = next(
        component
        for component in verifier_pipeline_components()
        if component.component_id == "native-extraction"
    )
    assert native_spec.component_version == "13"
    assert set(native_spec.file_paths) == _EXPECTED_NATIVE_COMPONENT_FILES
    assert changed_path in native_spec.file_paths
    assert native_spec.settings["cross_publication_reconciliation_receipt_required"] is True
    assert native_spec.settings["exact_extraction_execution_context_required"] is True
    assert native_spec.settings["contract"] == "publication-fragment-v3"
    assert (
        native_spec.settings["grounding_package_contract"] == "typed-evidence-grounding-package-v4"
    )
    assert native_spec.settings["in_repository_dependency_closure_bound"] is True
    assert native_spec.settings["bounded_pre_call_intent_required"] is True
    assert native_spec.settings["hosted_native_extraction_run_contract"] == (
        "hosted-native-extraction-run-v1"
    )
    assert native_spec.settings["hosted_native_execution_mode"] == "hosted_exact_once"
    assert (
        "scripts/build_hosted_native_grounding_package.py"
        in native_spec.settings["native_extraction_entry_points"]
    )
    assert native_spec.settings["frozen_acquisition_replay"] == (
        "exact-query-membership-to-screen-to-native-package"
    )
    assert native_spec.settings["protocol_free_text_screening_without_external_authority"] == (
        "blocking"
    )
    assert native_spec.settings["bounded_exact_source_projection_quote_grounding_required"] is True
    for relative in native_spec.file_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative, destination)
    expected = compute_pipeline_fingerprint(root=tmp_path, components=[native_spec])
    target = tmp_path / changed_path
    target.write_text(target.read_text() + "\n# simulated byte drift\n", encoding="utf-8")

    verification = verify_pipeline_fingerprint(expected=expected, root=tmp_path)

    assert verification.status == "mismatch"
    assert f"file_sha256_mismatch:{changed_path}" in verification.issues
    assert "component_sha256_mismatch:native-extraction" in verification.issues


def test_stale_native_v12_manifest_cannot_match_current_v13_component(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    current = next(
        component
        for component in verifier_pipeline_components()
        if component.component_id == "native-extraction"
    )
    hosted_v13_paths = {
        "scripts/build_hosted_native_grounding_package.py",
        "src/literature_multiverse/hosted_native_extraction_contract.py",
        "src/literature_multiverse/hosted_native_grounding_bridge.py",
    }
    for relative in current.file_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative, destination)
    stale_settings = dict(current.settings)
    stale_settings.pop("hosted_native_extraction_run_contract")
    stale_settings.pop("hosted_native_execution_mode")
    stale_settings["native_extraction_entry_points"] = [
        path
        for path in current.settings["native_extraction_entry_points"]
        if path != "scripts/build_hosted_native_grounding_package.py"
    ]
    stale = PipelineComponentSpec(
        component_id="native-extraction",
        component_version="12",
        file_paths=sorted(set(current.file_paths) - hosted_v13_paths),
        settings=stale_settings,
    )
    expected = compute_pipeline_fingerprint(root=tmp_path, components=[stale])

    verification = verify_pipeline_fingerprint(
        expected=expected,
        root=tmp_path,
        current_components=[current],
    )

    assert verification.status == "mismatch"
    assert "component_version_mismatch:native-extraction" in verification.issues
    assert "component_settings_mismatch:native-extraction" in verification.issues
    for path in hosted_v13_paths:
        assert f"file_added_to_current_manifest:{path}" in verification.issues
    assert "pipeline_sha256_mismatch" in verification.issues


def test_native_pipeline_python_dependency_closure_is_exact_and_fingerprinted() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    components = verifier_pipeline_components()
    observed_closure = _native_python_dependency_closure(repository_root)
    fingerprinted_paths = {path for component in components for path in component.file_paths}

    assert observed_closure == _EXPECTED_NATIVE_PYTHON_CLOSURE
    assert observed_closure <= fingerprinted_paths


def test_full_verifier_python_dependency_closure_is_fingerprinted() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    components = verifier_pipeline_components()
    observed_closure = _verifier_python_dependency_closure(repository_root)
    fingerprinted_paths = {path for component in components for path in component.file_paths}

    assert observed_closure <= fingerprinted_paths


def test_verifier_fingerprint_binds_runtime_and_shared_contract_dependencies() -> None:
    runtime_spec = next(
        component
        for component in verifier_pipeline_components()
        if component.component_id == "runtime-contract"
    )

    assert {
        "pyproject.toml",
        "uv.lock",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/records.py",
    } <= set(runtime_spec.file_paths)
    assert runtime_spec.component_version == "4"
    assert runtime_spec.settings["dependency_lock_bound"] is True
    assert runtime_spec.settings["shared_contract_helpers_bound"] is True
    assert set(runtime_spec.settings["installed_dependency_versions"]) == {
        "PyYAML",
        "httpx",
        "jsonschema",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "scikit-learn",
        "scipy",
    }
    assert runtime_spec.settings["python_version"]
    assert runtime_spec.settings["platform_machine"]


def test_verifier_fingerprint_detects_installed_httpx_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected = compute_verifier_pipeline_fingerprint(root=repository_root)
    installed_version = verifier_module.distribution_version

    def drifted_version(name: str) -> str:
        if name == "httpx":
            return "999.0-test-drift"
        return installed_version(name)

    monkeypatch.setattr(verifier_module, "distribution_version", drifted_version)
    verification = verify_pipeline_fingerprint(
        expected=expected,
        root=repository_root,
        current_components=verifier_pipeline_components(),
    )

    assert verification.status == "mismatch"
    assert "component_settings_mismatch:runtime-contract" in verification.issues
    assert "component_sha256_mismatch:runtime-contract" in verification.issues
    assert "pipeline_sha256_mismatch" in verification.issues


def test_verifier_fingerprint_fails_closed_when_httpx_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_version = verifier_module.distribution_version

    def missing_version(name: str) -> str:
        if name == "httpx":
            raise PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr(verifier_module, "distribution_version", missing_version)
    with pytest.raises(
        PipelineFingerprintError,
        match="pipeline_runtime_dependency_missing:httpx",
    ):
        verifier_pipeline_components()


def test_verification_release_component_binds_every_public_entrypoint() -> None:
    release_spec = next(
        component
        for component in verifier_pipeline_components()
        if component.component_id == "verification-release"
    )

    assert release_spec.component_version == "9"
    assert set(release_spec.file_paths) == _EXPECTED_VERIFICATION_RELEASE_FILES
    assert release_spec.settings["in_repository_dependency_closure_bound"] is True
    assert release_spec.settings["condition_calibration_outcome_opening"] == (
        "frozen-source-roster-membership-before-assessment"
    )
    assert (
        "prebundle-collection-source-external-replay"
        in release_spec.settings["condition_calibration_contract"]
    )
    assert release_spec.settings["condition_multi_arm_trajectory_construction"] == (
        "independent-single-arm-pass-then-canonical-builder-then-shared-trajectory-pass"
    )
