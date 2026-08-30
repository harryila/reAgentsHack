from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.private_cache_support import require_private_cache

from literature_multiverse.adaptive_calibration import freeze_complete_corpus_identity
from literature_multiverse.composed_corpus_identity import (
    COMPLETE_CORPUS_IDENTITY_V3_VERSION,
    MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION,
    CompleteCorpusIdentityV3,
    ComposedCorpusIdentityError,
    freeze_complete_corpus_identity_v3,
    freeze_manifest_corpus_policy_binding_v3,
    validate_complete_corpus_identity_v3,
    validate_complete_corpus_identity_v3_external_replay,
    validate_manifest_corpus_policy_binding_v3,
)
from literature_multiverse.corpus_pipeline_composition_runtime import (
    CorpusPipelineCompositionExternalReplayReceiptV1,
    build_corpus_pipeline_composition_external_replay_receipt_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.verifier import LegacyAdapterConfig, build_offline_fixture, load_corpus

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "data/cache/hosted-native-numeric-yield-pilot-v5-grounding"
    / "typed_evidence_grounding_package.json"
)
BRIDGE = (
    ROOT
    / "data/cache/hosted-native-numeric-yield-pilot-v5-grounding"
    / "hosted_native_grounding_bridge_receipt.json"
)
pytestmark = pytest.mark.private_cache


@pytest.fixture(scope="module")
def real_inputs() -> tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1]:
    require_private_cache(
        "data/cache/hosted-native-numeric-yield-pilot-v5-grounding/typed_evidence_grounding_package.json",
        "data/cache/hosted-native-numeric-yield-pilot-v5-grounding/hosted_native_grounding_bridge_receipt.json",
    )
    corpus = load_corpus(
        PACKAGE,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=ROOT,
    )
    receipt = build_corpus_pipeline_composition_external_replay_receipt_v1(
        repository_root=ROOT,
        grounding_package_path=PACKAGE,
        hosted_bridge_receipt_path=BRIDGE,
    )
    return corpus, receipt


def _membership(corpus: Any):
    return freeze_complete_corpus_identity(
        corpus_id=corpus.corpus_id,
        corpus_source_sha256=corpus.source_sha256,
        corpus_cutoff=corpus.metadata["native_corpus_cutoff"],
        publication_ids=sorted(
            publication.publication_id for publication in corpus.graph.publications
        ),
        source_manifest_sha256=corpus.metadata["source_manifest_sha256"],
    )


def _rehash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != field}
    return {**unsigned, field: hash_canonical(unsigned)}


def test_v3_binds_runtime_receipt_and_replays_current_artifacts(
    real_inputs: tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    corpus, receipt = real_inputs
    identity = freeze_complete_corpus_identity_v3(
        complete_corpus_membership_v1=_membership(corpus),
        external_replay_receipt=receipt,
        loaded_corpus=corpus,
    )

    assert identity.identity_version == COMPLETE_CORPUS_IDENTITY_V3_VERSION
    assert identity.external_replay_receipt == receipt
    assert identity.external_replay_receipt_sha256 == receipt.receipt_sha256
    assert identity.calibration_pipeline_sha256 == receipt.composed_pipeline_sha256
    assert identity.release_pipeline_sha256 == receipt.composed_pipeline_sha256
    assert identity.membership_composition_sha256 not in {
        identity.membership_sha256,
        identity.membership_composition_v2_sha256,
        identity.external_replay_receipt_sha256,
    }
    assert identity.calibration_authority is False
    assert identity.claim_release_authority is False
    assert validate_complete_corpus_identity_v3(identity) == identity
    assert (
        validate_complete_corpus_identity_v3_external_replay(
            identity=identity,
            loaded_corpus=corpus,
            repository_root=ROOT,
            grounding_package_path=PACKAGE,
            hosted_bridge_receipt_path=BRIDGE,
        )
        == identity
    )


def test_v3_rejects_loaded_corpus_substitution(
    real_inputs: tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    _, receipt = real_inputs
    _, fixture_corpus = build_offline_fixture()
    with pytest.raises(
        ComposedCorpusIdentityError,
        match="loaded_corpus_replay_mismatch",
    ):
        freeze_complete_corpus_identity_v3(
            complete_corpus_membership_v1=_membership(real_inputs[0]),
            external_replay_receipt=receipt,
            loaded_corpus=fixture_corpus,
        )


def test_v3_rejects_coherently_rehashed_receipt_alias_tamper(
    real_inputs: tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    corpus, receipt = real_inputs
    identity = freeze_complete_corpus_identity_v3(
        complete_corpus_membership_v1=_membership(corpus),
        external_replay_receipt=receipt,
        loaded_corpus=corpus,
    )
    payload = identity.model_dump(mode="json")
    payload["external_replay_receipt_sha256"] = "0" * 64
    payload = _rehash(payload, "membership_composition_sha256")
    with pytest.raises(ValidationError, match="external_replay_receipt_sha256_alias_mismatch"):
        CompleteCorpusIdentityV3.model_validate(payload)


def test_v3_manifest_policy_binding_carries_exact_receipt(
    real_inputs: tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    corpus, receipt = real_inputs
    identity = freeze_complete_corpus_identity_v3(
        complete_corpus_membership_v1=_membership(corpus),
        external_replay_receipt=receipt,
        loaded_corpus=corpus,
    )
    binding = freeze_manifest_corpus_policy_binding_v3(
        claim_manifest_sha256=hash_canonical({"claim": "v3-test"}),
        complete_corpus_identity_v3=identity,
        policy_sha256=hash_canonical({"policy": "v3-test"}),
    )

    assert binding.binding_version == MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION
    assert binding.complete_corpus_identity_v3 == identity
    assert binding.external_replay_receipt_sha256 == receipt.receipt_sha256
    assert binding.complete_corpus_membership_v3_sha256 == identity.membership_composition_sha256
    assert binding.release_authorizing is False
    assert validate_manifest_corpus_policy_binding_v3(binding) == binding


def test_v3_external_replay_rejects_wrong_bridge_artifact(
    real_inputs: tuple[Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    corpus, receipt = real_inputs
    identity = freeze_complete_corpus_identity_v3(
        complete_corpus_membership_v1=_membership(corpus),
        external_replay_receipt=receipt,
        loaded_corpus=corpus,
    )
    with pytest.raises(
        ComposedCorpusIdentityError,
        match="external_replay_failed",
    ):
        validate_complete_corpus_identity_v3_external_replay(
            identity=identity,
            loaded_corpus=corpus,
            repository_root=ROOT,
            grounding_package_path=PACKAGE,
            hosted_bridge_receipt_path=PACKAGE,
        )
