from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import literature_multiverse.public_artifacts as public_artifacts
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.public_artifacts import (
    _LOCAL_SUITE_SOURCE_PATHS,
    PUBLIC_RESULT_REGISTRY,
    PublicArtifactValidationError,
    PublicSourceMapBinding,
    _load_json_object,
    _resolve_registered_artifact_path,
    _semantic_validate,
    _validate_bound_source_maps,
    _validate_current_source_maps,
    _validate_evidencebench_grounding,
    _validate_legacy_antiox_bundles,
    _validate_local_suite,
    _validate_metasyn_fixed_positive,
    _validate_metasyn_retrieval,
    _validate_metasyn_screening,
    _validate_metasyn_synthesis_yield,
    _validate_metasyn_synthesis_yield_v2,
    _validate_planted_simulation,
    _validate_self_hash,
    _validate_source_bridge,
)


def test_public_validator_cli_runs_outside_repository_working_directory(
    repo_root: Path, tmp_path: Path
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            (repo_root / "scripts/validate_public_artifacts.py").as_posix(),
            "--repository-root",
            repo_root.as_posix(),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "public_artifact_integrity_valid_with_scoped_semantics"
    assert result["registered_artifacts"] == len(PUBLIC_RESULT_REGISTRY)
    records = {item["path"]: item for item in result["artifacts"]}
    bounded = records[
        "artifacts/diagnostics/native-antiox-bounded-ollama/summary.json"
    ]
    assert bounded["payload_sha256"] == (
        "af6680f7f2d94ad5bea6bb59e6fcddc46e140ee8377447c96b55d7413ae788dc"
    )
    assert bounded["semantic_validator"] == "native_bounded_ollama_v1_historical"
    assert bounded["result_recomputed_from_public_inputs"] is False
    gepa = records[
        "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
    ]
    assert gepa["current_source_maps_rehashed"] == 0
    assert gepa["historical_source_maps_hash_bound"] == 1
    local = records[
        "artifacts/diagnostics/evidence-inference-ollama/summary.json"
    ]
    assert local["current_source_maps_rehashed"] == 1
    assert local["historical_source_maps_hash_bound"] == 0


def test_every_public_diagnostic_json_is_registered_or_a_bound_companion(
    repo_root: Path,
) -> None:
    registered = {spec.path for spec in PUBLIC_RESULT_REGISTRY}
    diagnostics_root = repo_root / "artifacts/diagnostics"
    observed = {
        path.relative_to(repo_root).as_posix()
        for path in diagnostics_root.rglob("*.json")
        if path.is_file() and not path.is_symlink()
    }
    bound_companions = {
        "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json",
    }

    assert observed - registered == bound_companions


def test_metasyn_synthesis_yield_public_summary_is_immutable_and_public_only(
    repo_root: Path,
) -> None:
    relative = "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json"
    summary = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    spec = next(item for item in PUBLIC_RESULT_REGISTRY if item.path == relative)

    assert spec.self_hash_field == "summary_sha256"
    assert spec.semantic_validator == "metasyn_synthesis_yield"
    assert spec.result_recomputed_from_public_inputs is False
    assert spec.limitation is not None
    _validate_self_hash(summary, field="summary_sha256", artifact_path=relative)
    _validate_metasyn_synthesis_yield(summary)

    rehashed_count_tamper = json.loads(json.dumps(summary))
    rehashed_count_tamper["blocker_counts"]["no_estimable_graph"] = 9
    rehashed_count_tamper["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in rehashed_count_tamper.items()
            if key != "summary_sha256"
        }
    )
    _validate_self_hash(
        rehashed_count_tamper,
        field="summary_sha256",
        artifact_path=relative,
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="registered_zero_yield_mismatch:blocker_counts",
    ):
        _validate_metasyn_synthesis_yield(rehashed_count_tamper)

    rehashed_lineage_tamper = json.loads(json.dumps(summary))
    rehashed_lineage_tamper["synthesis_private_report_sha256"] = "0" * 64
    rehashed_lineage_tamper["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in rehashed_lineage_tamper.items()
            if key != "summary_sha256"
        }
    )
    _validate_self_hash(
        rehashed_lineage_tamper,
        field="summary_sha256",
        artifact_path=relative,
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="registered_lineage_mismatch:synthesis_private_report_sha256",
    ):
        _validate_metasyn_synthesis_yield(rehashed_lineage_tamper)

    forbidden_identifier = json.loads(json.dumps(summary))
    forbidden_identifier["question_id"] = "must-not-be-public"
    forbidden_identifier["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in forbidden_identifier.items()
            if key != "summary_sha256"
        }
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="public_contract_invalid",
    ):
        _validate_metasyn_synthesis_yield(forbidden_identifier)


def test_metasyn_synthesis_yield_v2_public_summary_is_immutable_and_public_only(
    repo_root: Path,
) -> None:
    relative = "artifacts/diagnostics/metasyn-synthesis-yield-v2/summary.json"
    summary = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    spec = next(item for item in PUBLIC_RESULT_REGISTRY if item.path == relative)

    assert spec.self_hash_field == "summary_sha256"
    assert spec.semantic_validator == "metasyn_synthesis_yield_v2"
    assert spec.result_recomputed_from_public_inputs is False
    assert spec.limitation is not None
    _validate_self_hash(summary, field="summary_sha256", artifact_path=relative)
    _validate_metasyn_synthesis_yield_v2(summary)

    rehashed_count_tamper = json.loads(json.dumps(summary))
    rehashed_count_tamper["blocker_counts"]["no_estimable_graph"] = 9
    rehashed_count_tamper["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in rehashed_count_tamper.items()
            if key != "summary_sha256"
        }
    )
    _validate_self_hash(
        rehashed_count_tamper,
        field="summary_sha256",
        artifact_path=relative,
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="registered_zero_yield_mismatch:blocker_counts",
    ):
        _validate_metasyn_synthesis_yield_v2(rehashed_count_tamper)

    rehashed_lineage_tamper = json.loads(json.dumps(summary))
    rehashed_lineage_tamper["hosted_execution_bundle_sha256"] = "0" * 64
    rehashed_lineage_tamper["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in rehashed_lineage_tamper.items()
            if key != "summary_sha256"
        }
    )
    _validate_self_hash(
        rehashed_lineage_tamper,
        field="summary_sha256",
        artifact_path=relative,
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="registered_lineage_mismatch:hosted_execution_bundle_sha256",
    ):
        _validate_metasyn_synthesis_yield_v2(rehashed_lineage_tamper)

    rehashed_authority_tamper = json.loads(json.dumps(summary))
    rehashed_authority_tamper["claim_release_authority"] = True
    rehashed_authority_tamper["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in rehashed_authority_tamper.items()
            if key != "summary_sha256"
        }
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="public_contract_invalid",
    ):
        _validate_metasyn_synthesis_yield_v2(rehashed_authority_tamper)

    forbidden_identifier = json.loads(json.dumps(summary))
    forbidden_identifier["publication_id"] = "must-not-be-public"
    forbidden_identifier["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in forbidden_identifier.items()
            if key != "summary_sha256"
        }
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="public_contract_invalid",
    ):
        _validate_metasyn_synthesis_yield_v2(forbidden_identifier)


def test_unknown_public_semantic_validator_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        PublicArtifactValidationError,
        match="semantic_validator_unknown",
    ):
        _semantic_validate(  # type: ignore[arg-type]
            "unregistered_validator",
            {},
            root=tmp_path,
            artifact_path="fixture.json",
        )


def test_bounded_native_v1_public_aggregate_has_immutable_semantic_validation(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    relative = "artifacts/diagnostics/native-antiox-bounded-ollama/summary.json"
    summary = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    spec = next(item for item in PUBLIC_RESULT_REGISTRY if item.path == relative)

    assert spec.self_hash_field == "summary_sha256"
    assert spec.semantic_validator == "native_bounded_ollama_v1_historical"
    assert spec.result_recomputed_from_public_inputs is False
    assert spec.limitation is not None
    assert "private" in spec.limitation
    _validate_self_hash(summary, field="summary_sha256", artifact_path=relative)
    _semantic_validate(
        "native_bounded_ollama_v1_historical",
        summary,
        root=tmp_path,
        artifact_path=relative,
    )

    tampered = json.loads(json.dumps(summary))
    tampered["packet_status_counts"]["packet_contract_invalid"] = 32
    tampered["summary_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "summary_sha256"}
    )
    _validate_self_hash(tampered, field="summary_sha256", artifact_path=relative)
    with pytest.raises(ValueError, match="frozen_lineage_mismatch:summary_sha256"):
        _semantic_validate(
            "native_bounded_ollama_v1_historical",
            tampered,
            root=tmp_path,
            artifact_path=relative,
        )


def test_evidencebench_public_bundle_crossbinds_replay_and_current_environment(
    repo_root: Path, tmp_path: Path
) -> None:
    summary_path = (
        repo_root / "artifacts/diagnostics/evidencebench-grounding-v1/summary.json"
    )
    audit_path = (
        repo_root
        / "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_evidencebench_grounding(summary, root=repo_root)

    rehashed_summary = json.loads(json.dumps(summary))
    rehashed_summary["candidate_sentence_count"] += 1
    summary_payload = {
        key: value
        for key, value in rehashed_summary.items()
        if key != "summary_sha256"
    }
    rehashed_summary["summary_sha256"] = hash_canonical(summary_payload)
    with pytest.raises(ValueError, match="bundle_binding_mismatch:summary_sha256"):
        _validate_evidencebench_grounding(rehashed_summary, root=repo_root)

    private_field = json.loads(json.dumps(summary))
    private_field["question_id"] = "must-not-be-public"
    private_payload = {
        key: value for key, value in private_field.items() if key != "summary_sha256"
    }
    private_field["summary_sha256"] = hash_canonical(private_payload)
    with pytest.raises(ValueError):
        _validate_evidencebench_grounding(private_field, root=repo_root)

    stale_runtime = json.loads(json.dumps(audit))
    stale_runtime["runtime_versions"]["python"] = "0.0.0"
    receipt_payload = {
        key: value
        for key, value in stale_runtime.items()
        if key != "receipt_sha256"
    }
    stale_runtime["receipt_sha256"] = hash_canonical(receipt_payload)
    temporary_audit = (
        tmp_path
        / "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json"
    )
    temporary_audit.parent.mkdir(parents=True)
    temporary_audit.write_text(json.dumps(stale_runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="public_bundle_runtime_drift"):
        _validate_evidencebench_grounding(summary, root=tmp_path)


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("limitations", "evidencebench_mandatory_limitations_mismatch"),
        ("paired_estimate", "evidencebench_paired_estimate_mismatch"),
    ],
)
def test_evidencebench_public_release_guard_rejects_semantic_rehashes(
    repo_root: Path,
    tmp_path: Path,
    tamper: str,
    error: str,
) -> None:
    summary = json.loads(
        (
            repo_root
            / "artifacts/diagnostics/evidencebench-grounding-v1/summary.json"
        ).read_text(encoding="utf-8")
    )
    audit = json.loads(
        (
            repo_root
            / "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json"
        ).read_text(encoding="utf-8")
    )
    if tamper == "limitations":
        summary["licenses_scientific_scope"] = sorted(
            summary["licenses_scientific_scope"]
            + ["invented_non_mandatory_limitation"]
        )
    else:
        summary["selected_method_paired_deltas"][0][
            "all_aspect_recall_at_10_delta"
        ]["estimate"] += 0.01

    summary["summary_sha256"] = hash_canonical(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    audit["summary_sha256"] = summary["summary_sha256"]
    audit["receipt_sha256"] = hash_canonical(
        {key: value for key, value in audit.items() if key != "receipt_sha256"}
    )
    audit_path = (
        tmp_path
        / "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(PublicArtifactValidationError, match=error):
        _validate_evidencebench_grounding(summary, root=tmp_path)


def test_generic_public_artifact_hash_and_source_lineage(repo_root: Path) -> None:
    relative = "src/literature_multiverse/lineage.py"
    payload = {
        "source_code_sha256s": {relative: sha256_file(repo_root / relative)},
        "result": {"estimate": 0.5},
    }
    artifact = {**payload, "artifact_payload_sha256": hash_canonical(payload)}
    _validate_self_hash(
        artifact,
        field="artifact_payload_sha256",
        artifact_path="fixture.json",
    )
    assert (
        _validate_current_source_maps(
            artifact,
            root=repo_root,
            artifact_path="fixture.json",
        )
        == 1
    )


def test_generic_public_artifact_rejects_tamper_and_stale_source(
    repo_root: Path,
) -> None:
    relative = "src/literature_multiverse/lineage.py"
    payload = {"source_files_sha256": {relative: "0" * 64}, "result": 1}
    artifact = {**payload, "artifact_payload_sha256": hash_canonical(payload)}
    with pytest.raises(PublicArtifactValidationError, match="source_lineage_stale"):
        _validate_current_source_maps(
            artifact,
            root=repo_root,
            artifact_path="fixture.json",
        )

    tampered = json.loads(json.dumps(artifact))
    tampered["result"] = 2
    with pytest.raises(PublicArtifactValidationError, match="self_hash_mismatch"):
        _validate_self_hash(
            tampered,
            field="artifact_payload_sha256",
            artifact_path="fixture.json",
        )


def test_historical_gepa_source_map_is_pinned_not_relabelled(
    repo_root: Path,
) -> None:
    relative = "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
    artifact = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    binding = PublicSourceMapBinding(
        "$.lineage.source_code_sha256s",
        "historical_execution",
        "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3",
    )
    assert _validate_bound_source_maps(
        artifact,
        root=repo_root,
        artifact_path=relative,
        bindings=(binding,),
    ) == (0, 1)

    artifact["lineage"]["source_code_sha256s"][
        "src/literature_multiverse/providers.py"
    ] = sha256_file(repo_root / "src/literature_multiverse/providers.py")
    payload = {
        key: value
        for key, value in artifact.items()
        if key != "public_summary_sha256"
    }
    artifact["public_summary_sha256"] = hash_canonical(payload)
    with pytest.raises(
        PublicArtifactValidationError, match="historical_source_bundle_mismatch"
    ):
        _validate_bound_source_maps(
            artifact,
            root=repo_root,
            artifact_path=relative,
            bindings=(binding,),
        )


def test_historical_source_map_cannot_hide_at_an_unclassified_location(
    repo_root: Path,
) -> None:
    relative = "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
    artifact = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    artifact["undeclared_lineage"] = {
        "historical_source_code_sha256s": dict(
            artifact["lineage"]["source_code_sha256s"]
        )
    }
    binding = PublicSourceMapBinding(
        "$.lineage.source_code_sha256s",
        "historical_execution",
        "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3",
    )

    with pytest.raises(PublicArtifactValidationError, match="source_map_unclassified"):
        _validate_bound_source_maps(
            artifact,
            root=repo_root,
            artifact_path=relative,
            bindings=(binding,),
        )


def test_current_source_map_cannot_resolve_outside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("external bytes\n", encoding="utf-8")
    (repository_root / "linked.py").symlink_to(outside)
    artifact = {
        "source_code_sha256s": {"linked.py": sha256_file(outside)},
    }

    with pytest.raises(
        PublicArtifactValidationError,
        match="source_path_outside_repository",
    ):
        _validate_current_source_maps(
            artifact,
            root=repository_root,
            artifact_path="fixture.json",
        )


@pytest.mark.parametrize("malformed", [[], None, "not-a-map"])
def test_current_source_map_declaration_must_be_an_object(
    repo_root: Path,
    malformed: object,
) -> None:
    with pytest.raises(PublicArtifactValidationError, match="source_binding_not_map"):
        _validate_current_source_maps(
            {"source_code_sha256s": malformed},
            root=repo_root,
            artifact_path="fixture.json",
        )


def test_malformed_nested_historical_source_map_cannot_be_unclassified(
    repo_root: Path,
) -> None:
    relative = "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
    artifact = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    artifact["undeclared_lineage"] = {"historical_source_code_sha256s": None}
    binding = PublicSourceMapBinding(
        "$.lineage.source_code_sha256s",
        "historical_execution",
        "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3",
    )

    with pytest.raises(PublicArtifactValidationError, match="source_map_unclassified"):
        _validate_bound_source_maps(
            artifact,
            root=repo_root,
            artifact_path=relative,
            bindings=(binding,),
        )


def test_explicit_binding_must_target_a_reserved_source_map_key(
    repo_root: Path,
) -> None:
    artifact = {"lineage": {"arbitrary_hashes": {"pyproject.toml": "0" * 64}}}
    binding = PublicSourceMapBinding(
        "$.lineage.arbitrary_hashes",
        "historical_execution",
        hash_canonical(artifact["lineage"]["arbitrary_hashes"]),
    )

    with pytest.raises(PublicArtifactValidationError, match="not_reserved_map"):
        _validate_bound_source_maps(
            artifact,
            root=repo_root,
            artifact_path="fixture.json",
            bindings=(binding,),
        )


def test_rendered_source_map_location_collision_fails_closed(
    repo_root: Path,
) -> None:
    source_map = {"pyproject.toml": sha256_file(repo_root / "pyproject.toml")}
    artifact = {
        "ambiguous.path": {"source_code_sha256s": source_map},
        "ambiguous": {"path": {"source_code_sha256s": source_map}},
    }

    with pytest.raises(PublicArtifactValidationError, match="location_collision"):
        _validate_current_source_maps(
            artifact,
            root=repo_root,
            artifact_path="fixture.json",
        )


def test_public_json_loader_rejects_duplicate_keys_at_any_depth(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "duplicate.json"
    artifact.write_text('{"lineage":{"sha":"first","sha":"second"}}', encoding="utf-8")

    with pytest.raises(PublicArtifactValidationError, match="duplicate_json_key"):
        _load_json_object(artifact)


def test_registered_artifact_cannot_resolve_outside_repository(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (repository_root / "artifact.json").symlink_to(outside)

    with pytest.raises(
        PublicArtifactValidationError,
        match="registry_path_outside_repository",
    ):
        _resolve_registered_artifact_path(
            root=repository_root,
            artifact_path="artifact.json",
        )


def test_source_map_rejects_cross_platform_backslash_path(repo_root: Path) -> None:
    with pytest.raises(PublicArtifactValidationError, match="source_path_unsafe"):
        _validate_current_source_maps(
            {"source_code_sha256s": {"src\\outside.py": "0" * 64}},
            root=repo_root,
            artifact_path="fixture.json",
        )


def test_item_risk_rejects_difference_from_recomputed_prediction_source(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = json.loads(
        (
            repo_root
            / "artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json"
        ).read_text(encoding="utf-8")
    )
    original_compute = public_artifacts.compute_ei_item_risk_pipeline_fingerprint
    fingerprint, prediction_source = original_compute(
        repository_root=repo_root,
        config=public_artifacts.load_ei_item_risk_config(
            repo_root / "configs/benchmarks/evidence-inference-item-risk-v1.json"
        ),
        gepa_public_summary=json.loads(
            (
                repo_root
                / "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
            ).read_text(encoding="utf-8")
        ),
    )
    altered_prediction_source = dict(prediction_source)
    altered_prediction_source["manifest_file_sha256"] = "0" * 64
    monkeypatch.setattr(
        public_artifacts,
        "compute_ei_item_risk_pipeline_fingerprint",
        lambda **_kwargs: (fingerprint, altered_prediction_source),
    )

    with pytest.raises(
        PublicArtifactValidationError,
        match="item_risk_current_lineage_mismatch",
    ):
        public_artifacts._validate_evidence_inference_item_risk(
            summary,
            root=repo_root,
        )


def test_fixed_positive_bundle_is_exactly_recomputed_from_public_inputs(
    repo_root: Path,
) -> None:
    receipt = json.loads(
        (
            repo_root
            / "artifacts/paper/metasyn-fixed-positive-test/freeze_receipt.json"
        ).read_text(encoding="utf-8")
    )

    _validate_metasyn_fixed_positive(receipt, root=repo_root)

    tampered = {**receipt, "rows": 85}
    with pytest.raises(PublicArtifactValidationError, match="receipt_registry_mismatch"):
        _validate_metasyn_fixed_positive(tampered, root=repo_root)


def test_fixed_positive_bundle_rejects_per_review_label_tamper(
    repo_root: Path, tmp_path: Path
) -> None:
    for relative in (
        "artifacts/paper/metasyn-benchmark",
        "artifacts/paper/metasyn-fixed-positive-test",
    ):
        source = repo_root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    audit_source = repo_root / "artifacts/paper/closed-corpus-local-audit.json"
    audit_destination = tmp_path / "artifacts/paper/closed-corpus-local-audit.json"
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_source, audit_destination)

    evaluation_path = (
        tmp_path / "artifacts/paper/metasyn-fixed-positive-test/evaluation.json"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    row = evaluation["per_review"][0]
    row["gold_direction"] = "Negative" if row["gold_direction"] != "Negative" else "Positive"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    receipt = json.loads(
        (
            tmp_path
            / "artifacts/paper/metasyn-fixed-positive-test/freeze_receipt.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(PublicArtifactValidationError, match="recompute_mismatch"):
        _validate_metasyn_fixed_positive(receipt, root=tmp_path)


def test_metasyn_public_aggregates_have_public_only_semantic_validation(
    repo_root: Path,
) -> None:
    retrieval = json.loads(
        (repo_root / "artifacts/diagnostics/metasyn-retrieval-study-v1.json").read_text(
            encoding="utf-8"
        )
    )
    screening = json.loads(
        (repo_root / "artifacts/diagnostics/metasyn-screening-study-v1.json").read_text(
            encoding="utf-8"
        )
    )

    _validate_metasyn_retrieval(retrieval, root=repo_root)
    _validate_metasyn_screening(screening, root=repo_root)

    leaked = {**retrieval, "question_id": "metasyn-review-000001"}
    with pytest.raises(PublicArtifactValidationError, match="forbidden_key"):
        _validate_metasyn_retrieval(leaked, root=repo_root)
    networked = {**screening, "network_calls": 1}
    with pytest.raises(PublicArtifactValidationError, match="network_invariant"):
        _validate_metasyn_screening(networked, root=repo_root)

    retrieval_arithmetic_tamper = json.loads(json.dumps(retrieval))
    retrieval_arithmetic_tamper["selected_calibration_result"]["candidates"][
        "rrf-tfidf-bm25-fixed-v1"
    ]["micro_recall_at_200"] = 0.99
    with pytest.raises(PublicArtifactValidationError, match="candidate_arithmetic_invalid"):
        _validate_metasyn_retrieval(retrieval_arithmetic_tamper, root=repo_root)

    screening_delta_tamper = json.loads(json.dumps(screening))
    screening_delta_tamper["calibration"]["selected_minus_rrf_paired_deltas"]["50"][
        "question_macro_absolute_recall"
    ] = 0.99
    with pytest.raises(PublicArtifactValidationError, match="delta_arithmetic_invalid"):
        _validate_metasyn_screening(screening_delta_tamper, root=repo_root)


def test_local_suite_binds_current_suite_and_complete_source_inventory(
    repo_root: Path,
) -> None:
    base = {
        "local_benchmark_report_version": "2",
        "status": "complete",
        "network_calls": 0,
        "suite_sha256": "0" * 64,
        "source_code_sha256s": {},
    }
    with pytest.raises(PublicArtifactValidationError, match="config_hash_mismatch"):
        _validate_local_suite(base, root=repo_root)

    config = repo_root / "configs/benchmarks/local-suite-v1.json"
    wrong_inventory = {**base, "suite_sha256": sha256_file(config)}
    with pytest.raises(PublicArtifactValidationError, match="source_inventory_invalid"):
        _validate_local_suite(wrong_inventory, root=repo_root)


def test_local_suite_recomputes_results_and_scientific_payload(repo_root: Path) -> None:
    retrieval_path = "artifacts/diagnostics/metasyn-retrieval-study-v1.json"
    screening_path = "artifacts/diagnostics/metasyn-screening-study-v1.json"
    retrieval = json.loads((repo_root / retrieval_path).read_text(encoding="utf-8"))
    screening = json.loads((repo_root / screening_path).read_text(encoding="utf-8"))
    freeze_payload = screening["lineage"]["retrieval_freeze_payload_sha256"]
    integrity = {
        "metasyn_retrieval_summary": {
            "path": retrieval_path,
            "file_sha256": sha256_file(repo_root / retrieval_path),
            "payload_sha256": retrieval["public_summary_payload_sha256"],
            "freeze_receipt_sha256": retrieval["lineage"]["freeze_receipt_sha256"],
            "freeze_payload_sha256": freeze_payload,
        },
        "metasyn_screening_summary": {
            "path": screening_path,
            "file_sha256": sha256_file(repo_root / screening_path),
            "payload_sha256": screening["public_summary_payload_sha256"],
            "retrieval_freeze_payload_sha256": freeze_payload,
        },
    }
    results = {
        "metasyn_retrieval_development_selection_calibration": {
            "scientific_role": "retrospective_nonpristine",
            "selected_candidate": retrieval["selection_protocol"]["selected_candidate"],
            "development": retrieval["development_results"],
            "calibration": retrieval["selected_calibration_result"],
            "official_test_evaluated": False,
        },
        "metasyn_protocol_aware_screening_reranking": {
            "scientific_role": "retrospective_nonpristine_matched_subset_survival",
            "selected_candidate": screening["protocol"]["selected_candidate"],
            "development": screening[
                "development_component_disjoint_cross_validation"
            ],
            "calibration": screening["calibration"],
            "official_test_evaluated": False,
        },
    }
    suite_sha256 = sha256_file(repo_root / "configs/benchmarks/local-suite-v1.json")
    report = {
        "local_benchmark_report_version": "2",
        "status": "complete",
        "network_calls": 0,
        "suite_sha256": suite_sha256,
        "source_code_sha256s": {
            relative: sha256_file(repo_root / relative)
            for relative in _LOCAL_SUITE_SOURCE_PATHS
        },
        "artifacts": {"integrity": integrity},
        "results": results,
        "reproducibility": {
            "scientific_payload_sha256": hash_canonical(
                {
                    "suite_sha256": suite_sha256,
                    "artifact_integrity": integrity,
                    "results": results,
                }
            ),
            "timestamps_in_scientific_payload": False,
        },
    }
    _validate_local_suite(report, root=repo_root)

    tampered = json.loads(json.dumps(report))
    tampered["results"]["metasyn_retrieval_development_selection_calibration"][
        "selected_candidate"
    ] = "tampered"
    with pytest.raises(PublicArtifactValidationError, match="results_content_mismatch"):
        _validate_local_suite(tampered, root=repo_root)

    tampered_hash = json.loads(json.dumps(report))
    tampered_hash["reproducibility"]["scientific_payload_sha256"] = "0" * 64
    with pytest.raises(PublicArtifactValidationError, match="scientific_payload_mismatch"):
        _validate_local_suite(tampered_hash, root=repo_root)


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/diagnostics/native-source/antiox-eligible-source-bridge.json",
        "artifacts/diagnostics/native-source/antiox-source-bridge.json",
        "artifacts/diagnostics/native-source/metasyn-boundary-source-bridge.json",
    ],
)
def test_source_bridges_enforce_scope_counts_and_public_bindings(
    repo_root: Path, relative: str
) -> None:
    bridge = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    _validate_source_bridge(bridge, root=repo_root, artifact_path=relative)

    tampered = json.loads(json.dumps(bridge))
    tampered["pristine_final_holdout_eligible"] = True
    payload = {key: value for key, value in tampered.items() if key != "run_sha256"}
    tampered["run_sha256"] = hash_canonical(payload)
    _validate_self_hash(tampered, field="run_sha256", artifact_path=relative)
    with pytest.raises(PublicArtifactValidationError, match="source_bridge_scope_invalid"):
        _validate_source_bridge(tampered, root=repo_root, artifact_path=relative)


def test_eligible_source_bridge_is_cross_bound_to_native_config_and_summary(
    repo_root: Path,
) -> None:
    relative = (
        "artifacts/diagnostics/native-source/antiox-eligible-source-bridge.json"
    )
    bridge = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(bridge))
    tampered["native_source_manifest_content_sha256"] = "f" * 64
    payload = {key: value for key, value in tampered.items() if key != "run_sha256"}
    tampered["run_sha256"] = hash_canonical(payload)

    _validate_self_hash(tampered, field="run_sha256", artifact_path=relative)
    with pytest.raises(PublicArtifactValidationError, match="native_config_mismatch"):
        _validate_source_bridge(tampered, root=repo_root, artifact_path=relative)


def test_metasyn_source_bridge_is_bound_to_tracked_corpus_manifest(
    repo_root: Path,
) -> None:
    relative = (
        "artifacts/diagnostics/native-source/metasyn-boundary-source-bridge.json"
    )
    bridge = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(bridge))
    tampered["verified_source_artifacts"][-1]["sha256"] = "f" * 64
    payload = {key: value for key, value in tampered.items() if key != "run_sha256"}
    tampered["run_sha256"] = hash_canonical(payload)

    _validate_self_hash(tampered, field="run_sha256", artifact_path=relative)
    with pytest.raises(PublicArtifactValidationError, match="artifact_binding_mismatch"):
        _validate_source_bridge(tampered, root=repo_root, artifact_path=relative)


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/paper/budgeted-verification-simulation-200.json",
        "artifacts/paper/calibration-simulation-100.json",
        "artifacts/paper/meta-simulation-200.json",
    ],
)
def test_planted_simulation_artifacts_exactly_replay_and_reject_rehashed_results(
    repo_root: Path, relative: str
) -> None:
    artifact = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    _validate_planted_simulation(
        artifact,
        root=repo_root,
        artifact_path=relative,
    )

    tampered = json.loads(json.dumps(artifact))
    tampered["summary"]["interpretation"] = "coherently rehashed fabricated result"
    payload = {
        key: value
        for key, value in tampered.items()
        if key != "artifact_payload_sha256"
    }
    tampered["artifact_payload_sha256"] = hash_canonical(payload)
    _validate_self_hash(
        tampered,
        field="artifact_payload_sha256",
        artifact_path=relative,
    )
    with pytest.raises(PublicArtifactValidationError, match="replay_mismatch"):
        _validate_planted_simulation(
            tampered,
            root=repo_root,
            artifact_path=relative,
        )


def test_all_legacy_antiox_bundles_load_through_offline_app_boundary(
    repo_root: Path,
) -> None:
    _validate_legacy_antiox_bundles(root=repo_root)
