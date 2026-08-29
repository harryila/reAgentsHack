from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.metasyn_v5_source_surface as source_surface_module
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    MetaSynHostedExecutionBundleV1,
    load_current_metasyn_hosted_execution_bundle,
)
from literature_multiverse.metasyn_v5_source_surface import (
    CONSUMED_V5_RUNTIME_ARTIFACT_KINDS,
    EXPECTED_V5_ADAPTER_BUNDLE_SHA256,
    EXPECTED_V5_ADAPTER_PIPELINE_SHA256,
    EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256,
    EXPECTED_V5_EXECUTION_BUNDLE_SHA256,
    EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256,
    EXPECTED_V5_ROW_MEMBERSHIP_SHA256,
    EXPECTED_V5_RUNTIME_PIPELINE_SHA256,
    FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS,
    PROJECTED_ROW_CONTEXT_FIELDS,
    MetaSynV5SourceSurfaceError,
    MetaSynV5SourceSurfaceV1,
    _pinned_v5_workspace,
    _rehash_source_artifact,
    _source_surface_python_dependency_closure,
    compute_metasyn_v5_source_surface_pipeline_fingerprint,
    freeze_metasyn_v5_source_surface,
    validate_metasyn_v5_source_surface,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    verify_pipeline_fingerprint,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V5_WORKSPACE = REPOSITORY_ROOT / "data/cache/metasyn/bounded-anthropic-yield-v5"


@pytest.fixture(scope="session")
def v5_source_surface() -> MetaSynV5SourceSurfaceV1:
    # This is the normative integration exercise: it invokes the real v5 loader with
    # external_replay=True and rehashes the actual source artifacts.
    return freeze_metasyn_v5_source_surface(repository_root=REPOSITORY_ROOT)


@pytest.fixture(scope="session")
def v5_execution_bundle() -> MetaSynHostedExecutionBundleV1:
    # The session's real external replay is performed by v5_source_surface.  This cheap
    # load supplies the already self-hashed row contexts for exact projection assertions.
    _, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=V5_WORKSPACE,
        repository_root=REPOSITORY_ROOT,
        external_replay=False,
    )
    return bundle


def _rehash_surface_payload(payload: dict[str, Any]) -> None:
    payload["source_surface_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "source_surface_sha256"}
    )


def _self_consistent_tampered_fingerprint(
    fingerprint: PipelineFingerprint,
) -> PipelineFingerprint:
    payload = fingerprint.model_dump(mode="json")
    component = payload["components"][0]
    component["files"][0]["sha256"] = "0" * 64
    component["component_sha256"] = hash_canonical(
        {key: value for key, value in component.items() if key != "component_sha256"}
    )
    payload["pipeline_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "pipeline_sha256"}
    )
    return PipelineFingerprint.model_validate(payload)


def test_real_v5_replay_retains_exact_full_roster_and_fixed_anchors(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    surface = v5_source_surface
    assert surface.v5_execution_bundle_sha256 == EXPECTED_V5_EXECUTION_BUNDLE_SHA256
    assert surface.v5_adapter_bundle_sha256 == EXPECTED_V5_ADAPTER_BUNDLE_SHA256
    assert surface.v5_adapter_pipeline_sha256 == EXPECTED_V5_ADAPTER_PIPELINE_SHA256
    assert surface.v5_runtime_pipeline_sha256 == EXPECTED_V5_RUNTIME_PIPELINE_SHA256
    assert surface.v5_question_membership_sha256 == EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256
    assert surface.v5_component_membership_sha256 == EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256
    assert surface.v5_row_membership_sha256 == EXPECTED_V5_ROW_MEMBERSHIP_SHA256
    assert (surface.question_count, surface.component_count, surface.publication_count) == (
        10,
        10,
        32,
    )
    assert len(surface.rows) == 32
    assert [row.row_ordinal for row in surface.rows] == list(range(32))
    assert [row.row_key for row in surface.rows] == sorted({row.row_key for row in surface.rows})
    assert surface.reference_fields_unopened is True
    assert surface.official_test_labels_opened is False
    assert surface.directional_accuracy_authority is False
    assert surface.scientific_effectiveness_authority is False
    assert surface.claim_release_authority is False


def test_projection_is_exactly_the_whitelisted_question_source_row_lineage(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
    v5_execution_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    surface = v5_source_surface
    contexts = v5_execution_bundle.adapter_bundle.row_contexts
    assert surface.projected_row_context_fields == list(PROJECTED_ROW_CONTEXT_FIELDS)
    assert len(contexts) == len(surface.rows) == 32

    excluded_context_fields = {
        "allowed_moderators",
        "allowed_outcomes",
        "allowed_sections",
        "inventory_prompt",
        "inventory_prompt_sha256",
        "inventory_prompt_version",
        "inventory_schema",
        "inventory_schema_sha256",
        "outcome_positive_directions",
        "packet_base_prompt",
        "packet_base_prompt_sha256",
        "packet_base_prompt_version",
        "source_locator",
    }
    projected_model_fields = set(source_surface_module.MetaSynV5SourceSurfaceRowV1.model_fields)
    assert projected_model_fields.isdisjoint(excluded_context_fields)

    for context, row in zip(contexts, surface.rows, strict=True):
        assert row.row_key == context.row_key
        assert row.upstream_row_context_sha256 == context.row_context_sha256
        assert row.question_bundle_sha256 == context.question_bundle_sha256
        assert row.question_spec == context.question_spec
        assert row.question_spec_sha256 == context.question_spec_sha256
        assert row.source_row == context.source_row
        assert row.source_row_sha256 == context.source_row_sha256
        assert row.projection_sha256 == context.projection_sha256
        assert row.component_binding.independence_component_id == (
            context.independence_component_id
        )
        assert row.component_binding.independence_component_review_ids == (
            context.independence_component_review_ids
        )
        assert row.component_binding.independence_component_membership_sha256 == (
            context.independence_component_membership_sha256
        )


def test_v5_receipts_results_and_provider_outputs_are_explicitly_not_consumed(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    surface = v5_source_surface
    assert surface.projection_input_contract == (
        "externally_replayed_v5_execution_bundle_embedded_adapter_rows_only"
    )
    assert surface.consumed_v5_runtime_artifact_kinds == list(CONSUMED_V5_RUNTIME_ARTIFACT_KINDS)
    assert surface.forbidden_v5_runtime_artifact_kinds == list(FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS)
    assert surface.upstream_v5_execution_bundle_consumed is True
    assert surface.upstream_v5_call_receipts_consumed is False
    assert surface.upstream_v5_row_results_consumed is False
    assert surface.upstream_v5_provider_outputs_consumed is False
    assert "call_receipts" in surface.forbidden_v5_runtime_artifact_kinds
    assert "row_results" in surface.forbidden_v5_runtime_artifact_kinds
    assert "provider_outputs" in surface.forbidden_v5_runtime_artifact_kinds


def test_external_replay_flag_is_mandatory_for_projection(
    monkeypatch: pytest.MonkeyPatch,
    v5_execution_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    observed: dict[str, Any] = {}

    def replay_spy(**kwargs: Any) -> tuple[Path, MetaSynHostedExecutionBundleV1]:
        observed.update(kwargs)
        return V5_WORKSPACE.resolve(strict=True), v5_execution_bundle

    monkeypatch.setattr(
        source_surface_module,
        "load_current_metasyn_hosted_execution_bundle",
        replay_spy,
    )
    replayed = freeze_metasyn_v5_source_surface(repository_root=REPOSITORY_ROOT)
    assert observed["external_replay"] is True
    assert observed["workspace"] == V5_WORKSPACE.resolve(strict=True)
    assert replayed.v5_execution_bundle_sha256 == EXPECTED_V5_EXECUTION_BUNDLE_SHA256


def test_every_row_binds_rehashed_actual_source_artifact_bytes(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    unique_paths: set[str] = set()
    independently_observed: dict[str, tuple[str, int]] = {}
    for row in v5_source_surface.rows:
        binding = row.artifact_binding
        unique_paths.add(binding.artifact_path)
        assert binding.source_document_sha256 == (
            row.source_row.source_record.source_document.sha256
        )
        assert binding.projection_artifact_sha256 == (row.source_row.projection.artifact_sha256)
        if binding.artifact_path not in independently_observed:
            independently_observed[binding.artifact_path] = _rehash_source_artifact(
                repository_root=REPOSITORY_ROOT,
                artifact_path=binding.artifact_path,
                expected_sha256=binding.source_document_sha256,
            )
        observed_sha256, observed_bytes = independently_observed[binding.artifact_path]
        assert observed_sha256 == binding.observed_artifact_sha256
        assert observed_bytes == binding.observed_artifact_bytes
    assert len(unique_paths) == 3


def test_ast_dependency_closure_is_computed_and_verifies_current_bytes(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    closure = _source_surface_python_dependency_closure(REPOSITORY_ROOT)
    assert closure == sorted(set(closure))
    assert "src/literature_multiverse/metasyn_v5_source_surface.py" in closure
    assert "src/literature_multiverse/metasyn_bounded_hosted_runtime.py" in closure
    assert "src/literature_multiverse/metasyn_bounded_adapter.py" in closure
    assert "src/literature_multiverse/metasyn_typed_pilot.py" in closure
    assert "src/literature_multiverse/pipeline_fingerprint.py" in closure
    assert "src/literature_multiverse/lineage.py" in closure

    recomputed = compute_metasyn_v5_source_surface_pipeline_fingerprint(root=REPOSITORY_ROOT)
    assert recomputed == v5_source_surface.source_surface_pipeline_fingerprint
    verification = verify_pipeline_fingerprint(expected=recomputed, root=REPOSITORY_ROOT)
    assert verification.status == "matched"
    assert verification.issues == []


def test_self_consistent_pipeline_file_hash_tamper_fails_byte_verification(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    tampered = _self_consistent_tampered_fingerprint(
        v5_source_surface.source_surface_pipeline_fingerprint
    )
    verification = verify_pipeline_fingerprint(expected=tampered, root=REPOSITORY_ROOT)
    assert verification.status == "mismatch"
    assert any(issue.startswith("file_sha256_mismatch:") for issue in verification.issues)


def test_fixed_v5_anchor_tamper_fails_even_with_rehashed_surface(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    payload = v5_source_surface.model_dump(mode="json")
    payload["v5_execution_bundle_sha256"] = "0" * 64
    _rehash_surface_payload(payload)
    with pytest.raises(ValueError, match="metasyn_v5_source_surface_anchor_mismatch"):
        MetaSynV5SourceSurfaceV1.model_validate(payload)


def test_per_row_projection_hash_tamper_fails_even_with_rehashed_row_and_surface(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    payload = v5_source_surface.model_dump(mode="json")
    row = payload["rows"][0]
    row["projection_sha256"] = "0" * 64
    row["source_surface_row_sha256"] = hash_canonical(
        {key: value for key, value in row.items() if key != "source_surface_row_sha256"}
    )
    payload["projected_row_hash_membership_sha256"] = hash_canonical(
        [item["source_surface_row_sha256"] for item in payload["rows"]]
    )
    _rehash_surface_payload(payload)
    with pytest.raises(
        ValueError,
        match="metasyn_v5_source_surface_projection_hash_alias_mismatch",
    ):
        MetaSynV5SourceSurfaceV1.model_validate(payload)


def test_non_consumption_and_label_declarations_cannot_be_flipped(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    for field in (
        "upstream_v5_call_receipts_consumed",
        "upstream_v5_row_results_consumed",
        "upstream_v5_provider_outputs_consumed",
        "official_test_labels_opened",
    ):
        payload = v5_source_surface.model_dump(mode="json")
        payload[field] = True
        _rehash_surface_payload(payload)
        with pytest.raises(ValueError):
            MetaSynV5SourceSurfaceV1.model_validate(payload)

    payload = v5_source_surface.model_dump(mode="json")
    payload["reference_fields_unopened"] = False
    _rehash_surface_payload(payload)
    with pytest.raises(ValueError):
        MetaSynV5SourceSurfaceV1.model_validate(payload)


def test_missing_row_fails_full_roster_contract(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    payload = v5_source_surface.model_dump(mode="json")
    payload["rows"].pop()
    payload["projected_row_hash_membership_sha256"] = hash_canonical(
        [item["source_surface_row_sha256"] for item in payload["rows"]]
    )
    _rehash_surface_payload(payload)
    with pytest.raises(ValueError):
        MetaSynV5SourceSurfaceV1.model_validate(payload)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../source.bin",
        "/absolute/source.bin",
        "./source.bin",
        "nested\\source.bin",
        "nested//source.bin",
    ),
)
def test_source_artifact_paths_fail_closed(tmp_path: Path, unsafe_path: str) -> None:
    expected_sha256 = hashlib.sha256(b"source").hexdigest()
    with pytest.raises(MetaSynV5SourceSurfaceError):
        _rehash_source_artifact(
            repository_root=tmp_path,
            artifact_path=unsafe_path,
            expected_sha256=expected_sha256,
        )


def test_source_artifact_symlink_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "source.bin"
    target.write_bytes(b"source")
    link = tmp_path / "source-link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(
        MetaSynV5SourceSurfaceError,
        match="metasyn_v5_source_surface_path_symlink_forbidden",
    ):
        _rehash_source_artifact(
            repository_root=tmp_path,
            artifact_path=link.name,
            expected_sha256=hashlib.sha256(b"source").hexdigest(),
        )


def test_source_artifact_byte_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original source")
    expected_sha256 = hashlib.sha256(b"original source").hexdigest()
    assert _rehash_source_artifact(
        repository_root=tmp_path,
        artifact_path=source.name,
        expected_sha256=expected_sha256,
    ) == (expected_sha256, len(b"original source"))

    source.write_bytes(b"tampered source")
    with pytest.raises(
        MetaSynV5SourceSurfaceError,
        match="metasyn_v5_source_surface_artifact_sha256_mismatch",
    ):
        _rehash_source_artifact(
            repository_root=tmp_path,
            artifact_path=source.name,
            expected_sha256=expected_sha256,
        )


def test_alternate_or_symlinked_v5_workspace_fails_before_replay(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MetaSynV5SourceSurfaceError,
        match="metasyn_v5_source_surface_execution_workspace_not_pinned_v5",
    ):
        _pinned_v5_workspace(
            repository_root=REPOSITORY_ROOT,
            execution_workspace=tmp_path,
        )

    linked_workspace = tmp_path / "linked-v5"
    try:
        linked_workspace.symlink_to(V5_WORKSPACE, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(
        MetaSynV5SourceSurfaceError,
        match="metasyn_v5_source_surface_execution_workspace_symlink_forbidden",
    ):
        _pinned_v5_workspace(
            repository_root=REPOSITORY_ROOT,
            execution_workspace=linked_workspace,
        )


def test_surface_validation_can_reparse_without_runtime_artifact_access(
    v5_source_surface: MetaSynV5SourceSurfaceV1,
) -> None:
    reparsed = validate_metasyn_v5_source_surface(
        source_surface=v5_source_surface.model_dump(mode="json"),
        repository_root=REPOSITORY_ROOT,
        external_replay=False,
    )
    assert reparsed == v5_source_surface
