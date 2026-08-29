from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from literature_multiverse.corpus_pipeline_composition import (
    CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION,
    PIPELINE_COMPOSITION_CONTEXT_VERSION,
    CorpusPipelineCompositionError,
    CorpusPipelineCompositionJoinV1,
    build_composed_calibration_pipeline_verification_v1,
    build_corpus_ingress_projection_v1,
    build_corpus_pipeline_composition_join_v1,
    compute_pipeline_composition_context_sha256_v1,
    validate_corpus_pipeline_composition_join_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    verify_pipeline_fingerprint,
)


def _sha(label: str) -> str:
    return hash_canonical({"test-fixture": label})


def _proof(tmp_path: Path, role: str) -> PipelineFingerprintVerification:
    path = tmp_path / f"{role}.txt"
    path.write_text(f"{role}\n", encoding="utf-8")
    spec = PipelineComponentSpec(
        component_id=role,
        component_version="1",
        file_paths=[path.name],
        settings={"role": role},
    )
    fingerprint = compute_pipeline_fingerprint(root=tmp_path, components=[spec])
    proof = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=tmp_path,
        current_components=[spec],
    )
    assert proof.status == "matched"
    return proof


def _spec_from_proof(proof: PipelineFingerprintVerification) -> PipelineComponentSpec:
    assert proof.computed is not None
    assert len(proof.computed.components) == 1
    component = proof.computed.components[0]
    return PipelineComponentSpec(
        component_id=component.component_id,
        component_version=component.component_version,
        file_paths=[item.path for item in component.files],
        settings=component.settings,
    )


def _ingress(
    *,
    label: str,
    extraction: PipelineFingerprintVerification,
):
    reconciled_graph_sha256 = _sha(f"{label}:reconciled-graph")
    return build_corpus_ingress_projection_v1(
        ingress_interface="hosted-exact-once-native-grounding-v4",
        corpus_id=f"corpus-{label}",
        question_id=f"question-{label}",
        corpus_cutoff="2026-08-01T00:00:00Z",
        corpus_source_sha256=_sha(f"{label}:source"),
        grounding_package_sha256=_sha(f"{label}:package"),
        typed_corpus_sha256=_sha(f"{label}:typed-corpus"),
        extraction_pipeline_sha256=extraction.expected_pipeline_sha256,
        extraction_pipeline_verification_sha256=extraction.verification_sha256,
        source_manifest_sha256=_sha(f"{label}:manifest"),
        source_membership_sha256=_sha(f"{label}:source-membership"),
        question_config_sha256=_sha(f"{label}:question-config"),
        extraction_context_sha256=_sha(f"{label}:context"),
        extraction_context_receipt_sha256=_sha(f"{label}:context-receipt"),
        hosted_run_sha256=_sha(f"{label}:hosted-run"),
        terminal_call_membership_sha256=_sha(f"{label}:terminal-calls"),
        terminal_fragment_membership_sha256=_sha(f"{label}:terminal-fragments"),
        grounding_validation_sha256=_sha(f"{label}:grounding-validation"),
        grounding_replay_sha256=_sha(f"{label}:grounding-replay"),
        cohort_reconciliation_receipt_sha256=_sha(f"{label}:cohort-reconciliation"),
        reconciled_graph_sha256=reconciled_graph_sha256,
        effective_graph_sha256=reconciled_graph_sha256,
        hosted_bridge_receipt_sha256=_sha(f"{label}:bridge-receipt"),
    )


def test_composition_binds_three_matched_proofs_and_exact_ingress(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    ingress = _ingress(label="a", extraction=extraction)

    join = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
        corpus_ingress=ingress,
    )

    expected_context_sha256 = hash_canonical(
        {
            "composition_version": PIPELINE_COMPOSITION_CONTEXT_VERSION,
            "extraction_pipeline_verification": extraction.model_dump(mode="json"),
            "verifier_core_pipeline_verification": verifier_core.model_dump(mode="json"),
            "join_policy_pipeline_verification": join_policy.model_dump(mode="json"),
        }
    )
    assert join.join_version == CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION
    assert join.extraction_pipeline_sha256 == extraction.expected_pipeline_sha256
    assert join.verifier_core_pipeline_sha256 == verifier_core.expected_pipeline_sha256
    assert join.join_policy_pipeline_sha256 == join_policy.expected_pipeline_sha256
    assert join.composition_context_sha256 == expected_context_sha256
    assert join.composition_context_sha256 == compute_pipeline_composition_context_sha256_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )
    assert join.corpus_ingress_projection_sha256 == ingress.ingress_projection_sha256
    assert join.extraction_pipeline_sha256 != join.composition_context_sha256
    assert join.compatibility_decision == "portable-integrity-only"
    assert join.external_replay_required is True
    assert join.external_replay_completed is False
    assert join.endpoint_identity_equality_required is False
    assert join.extraction_accuracy_authority is False
    assert join.scientific_synthesis_authority is False
    assert join.scientific_claim_truth_authority is False
    assert join.calibration_authority is False
    assert join.claim_release_authority is False
    assert join.release_authorizing is False
    assert join.join_sha256 == hash_canonical(join.model_dump(mode="json", exclude={"join_sha256"}))
    assert validate_corpus_pipeline_composition_join_v1(join) == join


def test_portable_integrity_object_does_not_assert_endpoint_inequality(
    tmp_path: Path,
) -> None:
    shared = _proof(tmp_path, "shared")
    join_policy = _proof(tmp_path, "join-policy")
    ingress = _ingress(label="shared", extraction=shared)

    join = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=shared,
        verifier_core_pipeline_verification=shared,
        join_policy_pipeline_verification=join_policy,
        corpus_ingress=ingress,
    )

    assert join.extraction_pipeline_sha256 == join.verifier_core_pipeline_sha256
    assert join.endpoint_identity_equality_required is False


def test_standard_composed_pipeline_is_extraction_specific_and_corpus_independent(
    tmp_path: Path,
) -> None:
    extraction_a = _proof(tmp_path, "extraction-a")
    extraction_b = _proof(tmp_path, "extraction-b")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    core_components = [_spec_from_proof(verifier_core)]

    proof_a = build_composed_calibration_pipeline_verification_v1(
        repository_root=tmp_path,
        verifier_core_components=core_components,
        join_policy_components=[_spec_from_proof(join_policy)],
        ingress_interface="hosted-exact-once-native-grounding-v4",
        extraction_pipeline_verification=extraction_a,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )
    repeated_a = build_composed_calibration_pipeline_verification_v1(
        repository_root=tmp_path,
        verifier_core_components=core_components,
        join_policy_components=[_spec_from_proof(join_policy)],
        ingress_interface="hosted-exact-once-native-grounding-v4",
        extraction_pipeline_verification=extraction_a,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )
    proof_b = build_composed_calibration_pipeline_verification_v1(
        repository_root=tmp_path,
        verifier_core_components=core_components,
        join_policy_components=[_spec_from_proof(join_policy)],
        ingress_interface="hosted-exact-once-native-grounding-v4",
        extraction_pipeline_verification=extraction_b,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )

    assert proof_a.status == "matched"
    assert proof_a == repeated_a
    assert proof_a.computed_pipeline_sha256 != proof_b.computed_pipeline_sha256
    assert proof_a.computed is not None
    settings_component = next(
        item
        for item in proof_a.computed.components
        if item.component_id == "frozen-corpus-extraction"
    )
    assert settings_component.files == []
    assert settings_component.settings["extraction_pipeline_sha256"] == (
        extraction_a.expected_pipeline_sha256
    )
    assert "corpus_id" not in settings_component.settings
    assert "grounding_package_sha256" not in settings_component.settings


def test_standard_composed_pipeline_requires_current_external_bytes(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    (tmp_path / "extraction.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        CorpusPipelineCompositionError,
        match="extraction_external_replay_failed",
    ):
        build_composed_calibration_pipeline_verification_v1(
            repository_root=tmp_path,
            verifier_core_components=[_spec_from_proof(verifier_core)],
            join_policy_components=[_spec_from_proof(join_policy)],
            ingress_interface="hosted-exact-once-native-grounding-v4",
            extraction_pipeline_verification=extraction,
            verifier_core_pipeline_verification=verifier_core,
            join_policy_pipeline_verification=join_policy,
        )


def test_authoritative_composed_replay_rejects_reauthored_extraction_setting(
    tmp_path: Path,
) -> None:
    extraction_a = _proof(tmp_path, "extraction-a")
    extraction_b = _proof(tmp_path, "extraction-b")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    core_components = [_spec_from_proof(verifier_core)]
    proof_b = build_composed_calibration_pipeline_verification_v1(
        repository_root=tmp_path,
        verifier_core_components=core_components,
        join_policy_components=[_spec_from_proof(join_policy)],
        ingress_interface="hosted-exact-once-native-grounding-v4",
        extraction_pipeline_verification=extraction_b,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )
    assert proof_b.computed is not None

    # Generic replay could trust the settings embedded in ``proof_b``.  The
    # authoritative helper rebuilds current components from the independently
    # replayed extraction-a proof and must therefore reject the stale/authored
    # extraction-b setting.
    with pytest.raises(
        CorpusPipelineCompositionError,
        match="composed_calibration_pipeline_external_replay_failed",
    ):
        build_composed_calibration_pipeline_verification_v1(
            repository_root=tmp_path,
            verifier_core_components=core_components,
            join_policy_components=[_spec_from_proof(join_policy)],
            ingress_interface="hosted-exact-once-native-grounding-v4",
            extraction_pipeline_verification=extraction_a,
            verifier_core_pipeline_verification=verifier_core,
            join_policy_pipeline_verification=join_policy,
            expected=proof_b.computed,
        )


def test_portable_integrity_rejects_unrehashed_substitution_and_alias_mismatch(
    tmp_path: Path,
) -> None:
    extraction_a = _proof(tmp_path, "extraction-a")
    extraction_b = _proof(tmp_path, "extraction-b")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    ingress_a = _ingress(label="a", extraction=extraction_a)
    ingress_b = _ingress(label="b", extraction=extraction_b)
    join_a = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=extraction_a,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
        corpus_ingress=ingress_a,
    )

    transplanted_ingress = join_a.model_dump(mode="json")
    transplanted_ingress["corpus_ingress"] = ingress_b.model_dump(mode="json")
    transplanted_ingress["corpus_ingress_projection_sha256"] = ingress_b.ingress_projection_sha256
    with pytest.raises(ValidationError, match=r"extraction_ingress_mismatch|self_hash_mismatch"):
        CorpusPipelineCompositionJoinV1.model_validate(transplanted_ingress)

    # Even a newly re-hashed join cannot attach a different extraction proof to an
    # ingress whose externally replayed extraction identity is different.
    with pytest.raises(ValueError, match="extraction_ingress_mismatch"):
        build_corpus_pipeline_composition_join_v1(
            extraction_pipeline_verification=extraction_b,
            verifier_core_pipeline_verification=verifier_core,
            join_policy_pipeline_verification=join_policy,
            corpus_ingress=ingress_a,
        )


def test_composition_rejects_unmatched_proof_and_nested_tamper(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    ingress = _ingress(label="a", extraction=extraction)
    join = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
        corpus_ingress=ingress,
    )

    changed = tmp_path / "extraction.txt"
    changed.write_text("changed\n", encoding="utf-8")
    expected = extraction.computed
    assert expected is not None
    mismatch = verify_pipeline_fingerprint(expected=expected, root=tmp_path)
    assert mismatch.status == "mismatch"
    with pytest.raises(ValueError, match="proof_not_matched"):
        build_corpus_pipeline_composition_join_v1(
            extraction_pipeline_verification=mismatch,
            verifier_core_pipeline_verification=verifier_core,
            join_policy_pipeline_verification=join_policy,
            corpus_ingress=ingress,
        )

    # Lists nested in a frozen model can still be mutated in place.  Explicit replay
    # reparses every nested proof and fails closed on that class of mutation.
    join.extraction_pipeline_verification.computed.components[0].files.append(
        join.extraction_pipeline_verification.computed.components[0].files[0]
    )
    with pytest.raises(CorpusPipelineCompositionError):
        validate_corpus_pipeline_composition_join_v1(join)


def test_hosted_ingress_requires_complete_hosted_lineage(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    kwargs = _ingress(label="complete", extraction=extraction).model_dump(mode="json")
    for field in (
        "projection_version",
        "ingress_input_membership_sha256",
        "source_membership_projection_sha256",
        "extraction_execution_projection_sha256",
        "grounding_projection_sha256",
        "reconciliation_projection_sha256",
        "ingress_projection_sha256",
    ):
        kwargs.pop(field)
    kwargs["hosted_run_sha256"] = None

    with pytest.raises(ValidationError, match="lineage_incomplete"):
        build_corpus_ingress_projection_v1(**kwargs)
