from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.adaptive_calibration import freeze_complete_corpus_identity
from literature_multiverse.composed_corpus_identity import (
    COMPLETE_CORPUS_IDENTITY_V2_VERSION,
    MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION,
    CompleteCorpusIdentityV2,
    ComposedCorpusIdentityError,
    ManifestCorpusPolicyBindingV2,
    freeze_complete_corpus_identity_v2,
    freeze_manifest_corpus_policy_binding_v2,
    validate_complete_corpus_identity_v2,
    validate_manifest_corpus_policy_binding_v2,
)
from literature_multiverse.corpus_pipeline_composition import (
    CorpusPipelineCompositionJoinV1,
    build_composed_calibration_pipeline_verification_v1,
    build_corpus_ingress_projection_v1,
    build_corpus_pipeline_composition_join_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    verify_pipeline_fingerprint,
)


def _sha(label: str) -> str:
    return hash_canonical({"composed-corpus-test": label})


def _proof(tmp_path: Path, role: str) -> PipelineFingerprintVerification:
    path = tmp_path / f"{role}.txt"
    path.write_text(f"{role}\n", encoding="utf-8")
    component = PipelineComponentSpec(
        component_id=role,
        component_version="1",
        file_paths=[path.name],
        settings={"role": role},
    )
    fingerprint = compute_pipeline_fingerprint(root=tmp_path, components=[component])
    proof = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=tmp_path,
        current_components=[component],
    )
    assert proof.status == "matched"
    return proof


def _composed_proof(
    *,
    tmp_path: Path,
    extraction: PipelineFingerprintVerification,
    verifier_core: PipelineFingerprintVerification,
    join_policy: PipelineFingerprintVerification,
) -> PipelineFingerprintVerification:
    verifier_core_component = PipelineComponentSpec(
        component_id="verifier-core",
        component_version="1",
        file_paths=["verifier-core.txt"],
        settings={"role": "verifier-core"},
    )
    return build_composed_calibration_pipeline_verification_v1(
        repository_root=tmp_path,
        verifier_core_components=[verifier_core_component],
        join_policy_components=[
            PipelineComponentSpec(
                component_id="join-policy",
                component_version="1",
                file_paths=["join-policy.txt"],
                settings={"role": "join-policy"},
            )
        ],
        ingress_interface="hosted-exact-once-native-grounding-v4",
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
    )


def _join(
    *,
    extraction: PipelineFingerprintVerification,
    verifier_core: PipelineFingerprintVerification,
    join_policy: PipelineFingerprintVerification,
    corpus_label: str,
    question_label: str,
    graph_sha256: str,
) -> CorpusPipelineCompositionJoinV1:
    source_manifest_sha256 = _sha(f"{corpus_label}:source-manifest")
    ingress = build_corpus_ingress_projection_v1(
        ingress_interface="hosted-exact-once-native-grounding-v4",
        corpus_id=f"corpus-{corpus_label}",
        question_id=f"question-{question_label}",
        corpus_cutoff="2026-08-01T00:00:00Z",
        corpus_source_sha256=_sha(f"{corpus_label}:source"),
        grounding_package_sha256=_sha(f"{corpus_label}:{question_label}:package"),
        typed_corpus_sha256=_sha(f"{corpus_label}:{question_label}:typed-corpus"),
        extraction_pipeline_sha256=extraction.expected_pipeline_sha256,
        extraction_pipeline_verification_sha256=extraction.verification_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_membership_sha256=_sha(f"{corpus_label}:source-membership"),
        question_config_sha256=_sha(f"{question_label}:question-config"),
        extraction_context_sha256=_sha(f"{question_label}:context"),
        extraction_context_receipt_sha256=_sha(f"{question_label}:context-receipt"),
        hosted_run_sha256=_sha(f"{corpus_label}:{question_label}:hosted-run"),
        terminal_call_membership_sha256=_sha(
            f"{corpus_label}:{question_label}:terminal-calls"
        ),
        terminal_fragment_membership_sha256=_sha(
            f"{corpus_label}:{question_label}:terminal-fragments"
        ),
        grounding_validation_sha256=_sha(
            f"{corpus_label}:{question_label}:grounding-validation"
        ),
        grounding_replay_sha256=_sha(
            f"{corpus_label}:{question_label}:grounding-replay"
        ),
        cohort_reconciliation_receipt_sha256=_sha(
            f"{corpus_label}:{question_label}:cohort-reconciliation"
        ),
        reconciled_graph_sha256=graph_sha256,
        effective_graph_sha256=graph_sha256,
        hosted_bridge_receipt_sha256=_sha(
            f"{corpus_label}:{question_label}:bridge-receipt"
        ),
    )
    return build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core,
        join_policy_pipeline_verification=join_policy,
        corpus_ingress=ingress,
    )


def _v1_membership(join: CorpusPipelineCompositionJoinV1):
    ingress = join.corpus_ingress
    return freeze_complete_corpus_identity(
        corpus_id=ingress.corpus_id,
        corpus_source_sha256=ingress.corpus_source_sha256,
        corpus_cutoff=ingress.corpus_cutoff,
        publication_ids=["publication-1", "publication-2"],
        source_manifest_sha256=ingress.source_manifest_sha256,
    )


def _rehash(payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    return {**unhashed, hash_field: hash_canonical(unhashed)}


def test_complete_corpus_v2_retains_v1_and_binds_full_join(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    join = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q-a",
        graph_sha256=_sha("shared-graph"),
    )
    membership_v1 = _v1_membership(join)

    complete = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=membership_v1,
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
    )

    assert complete.identity_version == COMPLETE_CORPUS_IDENTITY_V2_VERSION
    assert complete.complete_corpus_membership_v1 == membership_v1
    assert complete.corpus_id == membership_v1.corpus_id
    assert complete.corpus_source_sha256 == membership_v1.corpus_source_sha256
    assert complete.corpus_cutoff == membership_v1.corpus_cutoff
    assert complete.membership_basis == membership_v1.membership_basis
    assert complete.publication_ids == membership_v1.publication_ids
    assert complete.source_manifest_sha256 == membership_v1.source_manifest_sha256
    assert complete.membership_sha256 == membership_v1.membership_sha256
    assert complete.extraction_pipeline_sha256 == join.extraction_pipeline_sha256
    assert complete.composed_pipeline_verification == composed
    assert complete.composed_pipeline_verification_sha256 == composed.verification_sha256
    assert complete.calibration_pipeline_sha256 == composed.computed_pipeline_sha256
    assert complete.corpus_pipeline_join_sha256 == join.join_sha256
    assert (
        complete.corpus_ingress_projection_sha256
        == join.corpus_ingress_projection_sha256
    )
    assert complete.membership_composition_sha256 != complete.membership_sha256
    assert complete.membership_composition_sha256 == hash_canonical(
        complete.model_dump(mode="json", exclude={"membership_composition_sha256"})
    )
    assert complete.extraction_accuracy_authority is False
    assert complete.scientific_synthesis_authority is False
    assert complete.scientific_claim_truth_authority is False
    assert complete.calibration_authority is False
    assert complete.claim_release_authority is False
    assert complete.release_authorizing is False
    assert validate_complete_corpus_identity_v2(complete) == complete


def test_calibration_pipeline_spans_questions_but_not_extraction_pipelines(
    tmp_path: Path,
) -> None:
    extraction_a = _proof(tmp_path, "extraction-a")
    extraction_b = _proof(tmp_path, "extraction-b")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    composed_a = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction_a,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    composed_b = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction_b,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    graph_sha256 = _sha("same-effective-graph")

    join_q1 = _join(
        extraction=extraction_a,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="shared",
        question_label="q1",
        graph_sha256=graph_sha256,
    )
    join_q2 = _join(
        extraction=extraction_a,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="shared",
        question_label="q2",
        graph_sha256=graph_sha256,
    )
    assert join_q1.join_sha256 != join_q2.join_sha256
    assert (
        join_q1.corpus_ingress_projection_sha256
        != join_q2.corpus_ingress_projection_sha256
    )
    assert composed_a.computed_pipeline_sha256 is not None

    complete_q1 = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join_q1),
        corpus_pipeline_join=join_q1,
        composed_pipeline_verification=composed_a,
    )
    complete_q2 = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join_q2),
        corpus_pipeline_join=join_q2,
        composed_pipeline_verification=composed_a,
    )
    assert complete_q1.membership_sha256 == complete_q2.membership_sha256
    assert (
        complete_q1.membership_composition_sha256
        != complete_q2.membership_composition_sha256
    )
    assert complete_q1.calibration_pipeline_sha256 == complete_q2.calibration_pipeline_sha256

    join_other_extraction = _join(
        extraction=extraction_b,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="shared",
        question_label="q1",
        graph_sha256=graph_sha256,
    )
    assert composed_b.computed_pipeline_sha256 != composed_a.computed_pipeline_sha256
    complete_other_extraction = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join_other_extraction),
        corpus_pipeline_join=join_other_extraction,
        composed_pipeline_verification=composed_b,
    )
    assert (
        complete_other_extraction.calibration_pipeline_sha256
        != complete_q1.calibration_pipeline_sha256
    )


def test_manifest_policy_binding_is_one_self_hashed_ledger_identity(
    tmp_path: Path,
) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    join = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q-a",
        graph_sha256=_sha("graph"),
    )
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    complete = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join),
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
    )

    binding = freeze_manifest_corpus_policy_binding_v2(
        claim_manifest_sha256=_sha("claim-manifest"),
        complete_corpus_identity_v2=complete,
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
        policy_sha256=_sha("deployed-policy"),
    )

    assert binding.binding_version == MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION
    assert binding.complete_corpus_identity_v2 == complete
    assert (
        binding.complete_corpus_membership_v2_sha256
        == complete.membership_composition_sha256
    )
    assert binding.corpus_pipeline_join_sha256 == join.join_sha256
    assert binding.corpus_ingress_projection_sha256 == join.corpus_ingress_projection_sha256
    assert binding.extraction_pipeline_sha256 == join.extraction_pipeline_sha256
    assert binding.calibration_pipeline_sha256 == composed.computed_pipeline_sha256
    assert binding.manifest_corpus_policy_binding_sha256 == hash_canonical(
        binding.model_dump(
            mode="json", exclude={"manifest_corpus_policy_binding_sha256"}
        )
    )
    assert binding.calibration_authority is False
    assert binding.claim_release_authority is False
    assert binding.release_authorizing is False
    assert validate_manifest_corpus_policy_binding_v2(binding) == binding


def test_complete_v2_rejects_coherently_rehashed_cross_corpus_join_transplant(
    tmp_path: Path,
) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    graph_sha256 = _sha("identical-effective-graph")
    join_a = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q",
        graph_sha256=graph_sha256,
    )
    join_b = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="b",
        question_label="q",
        graph_sha256=graph_sha256,
    )
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    complete_a = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join_a),
        corpus_pipeline_join=join_a,
        composed_pipeline_verification=composed,
    )

    transplanted = complete_a.model_dump(mode="json")
    transplanted.update(
        {
            "corpus_pipeline_join": join_b.model_dump(mode="json"),
            "extraction_pipeline_sha256": join_b.extraction_pipeline_sha256,
            "corpus_pipeline_join_sha256": join_b.join_sha256,
            "corpus_ingress_projection_sha256": (
                join_b.corpus_ingress_projection_sha256
            ),
        }
    )
    transplanted = _rehash(transplanted, "membership_composition_sha256")
    with pytest.raises(ValidationError, match="corpus_id_join_alias_mismatch"):
        CompleteCorpusIdentityV2.model_validate(transplanted)

    with pytest.raises(ValueError, match="corpus_id_join_alias_mismatch"):
        freeze_complete_corpus_identity_v2(
            complete_corpus_membership_v1=_v1_membership(join_a),
            corpus_pipeline_join=join_b,
            composed_pipeline_verification=composed,
        )


def test_policy_binding_rejects_join_transplant_even_when_graph_matches(
    tmp_path: Path,
) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    graph_sha256 = _sha("identical-effective-graph")
    join_a = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q",
        graph_sha256=graph_sha256,
    )
    join_b = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="b",
        question_label="q",
        graph_sha256=graph_sha256,
    )
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    complete_a = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join_a),
        corpus_pipeline_join=join_a,
        composed_pipeline_verification=composed,
    )
    binding_a = freeze_manifest_corpus_policy_binding_v2(
        claim_manifest_sha256=_sha("claim"),
        complete_corpus_identity_v2=complete_a,
        corpus_pipeline_join=join_a,
        composed_pipeline_verification=composed,
        policy_sha256=_sha("policy"),
    )

    transplanted = binding_a.model_dump(mode="json")
    transplanted.update(
        {
            "corpus_pipeline_join": join_b.model_dump(mode="json"),
            "corpus_pipeline_join_sha256": join_b.join_sha256,
            "corpus_ingress_projection_sha256": (
                join_b.corpus_ingress_projection_sha256
            ),
            "extraction_pipeline_sha256": join_b.extraction_pipeline_sha256,
        }
    )
    transplanted = _rehash(
        transplanted, "manifest_corpus_policy_binding_sha256"
    )
    with pytest.raises(ValidationError, match="complete_join_object_mismatch"):
        ManifestCorpusPolicyBindingV2.model_validate(transplanted)

    with pytest.raises(
        ComposedCorpusIdentityError, match="complete_join_object_mismatch"
    ):
        freeze_manifest_corpus_policy_binding_v2(
            claim_manifest_sha256=_sha("claim"),
            complete_corpus_identity_v2=complete_a,
            corpus_pipeline_join=join_b,
            composed_pipeline_verification=composed,
            policy_sha256=_sha("policy"),
        )


def test_nested_tamper_and_authority_escalation_fail_closed(tmp_path: Path) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    join = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q",
        graph_sha256=_sha("graph"),
    )
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    complete = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join),
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
    )
    binding = freeze_manifest_corpus_policy_binding_v2(
        claim_manifest_sha256=_sha("claim"),
        complete_corpus_identity_v2=complete,
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
        policy_sha256=_sha("policy"),
    )

    escalated = complete.model_dump(mode="json")
    escalated["calibration_authority"] = True
    escalated = _rehash(escalated, "membership_composition_sha256")
    with pytest.raises(ValidationError):
        CompleteCorpusIdentityV2.model_validate(escalated)

    escalated_binding = binding.model_dump(mode="json")
    escalated_binding["claim_release_authority"] = True
    escalated_binding = _rehash(
        escalated_binding, "manifest_corpus_policy_binding_sha256"
    )
    with pytest.raises(ValidationError):
        ManifestCorpusPolicyBindingV2.model_validate(escalated_binding)

    # The outer contract is frozen, but a nested v1 publication list is mutable.
    # Explicit integrity replay must still detect that mutation.
    complete.complete_corpus_membership_v1.publication_ids.append("transplanted")
    with pytest.raises(ComposedCorpusIdentityError, match="integrity_changed"):
        validate_complete_corpus_identity_v2(complete)


def test_policy_binding_hash_covers_manifest_policy_and_complete_corpus(
    tmp_path: Path,
) -> None:
    extraction = _proof(tmp_path, "extraction")
    verifier_core = _proof(tmp_path, "verifier-core")
    join_policy = _proof(tmp_path, "join-policy")
    join = _join(
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
        corpus_label="a",
        question_label="q",
        graph_sha256=_sha("graph"),
    )
    composed = _composed_proof(
        tmp_path=tmp_path,
        extraction=extraction,
        verifier_core=verifier_core,
        join_policy=join_policy,
    )
    complete = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=_v1_membership(join),
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
    )
    binding = freeze_manifest_corpus_policy_binding_v2(
        claim_manifest_sha256=_sha("claim"),
        complete_corpus_identity_v2=complete,
        corpus_pipeline_join=join,
        composed_pipeline_verification=composed,
        policy_sha256=_sha("policy"),
    )

    for field in ("claim_manifest_sha256", "policy_sha256"):
        changed = binding.model_dump(mode="json")
        changed[field] = _sha(f"changed-{field}")
        with pytest.raises(ValidationError, match="binding_hash_mismatch"):
            ManifestCorpusPolicyBindingV2.model_validate(changed)
