from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from literature_multiverse.corpus_pipeline_composition_runtime import (
    CorpusPipelineCompositionExternalReplayReceiptV1,
    build_corpus_pipeline_composition_external_replay_receipt_v1,
    require_external_replay_receipt_matches_corpus_load_result_v1,
    validate_corpus_pipeline_composition_external_replay_receipt_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.verifier import LegacyAdapterConfig, load_corpus

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
HAS_REAL_V5 = PACKAGE.is_file() and BRIDGE.is_file()


@pytest.fixture(scope="module")
def replayed_receipt() -> CorpusPipelineCompositionExternalReplayReceiptV1:
    if not HAS_REAL_V5:
        pytest.skip("local immutable hosted-v5 grounding artifacts are unavailable")
    return build_corpus_pipeline_composition_external_replay_receipt_v1(
        repository_root=ROOT,
        grounding_package_path=PACKAGE,
        hosted_bridge_receipt_path=BRIDGE,
    )


def test_real_v5_package_builds_and_replays_non_authorizing_composition(
    replayed_receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> None:
    receipt = replayed_receipt
    assert receipt.external_replay_completed is True
    assert receipt.composition_join.external_replay_completed is False
    assert receipt.extraction_pipeline_verification_sha256 == (
        receipt.composition_join.extraction_pipeline_verification.verification_sha256
    )
    assert receipt.composed_pipeline_sha256 == receipt.calibration_pipeline_sha256
    assert receipt.composed_pipeline_sha256 == receipt.release_pipeline_sha256
    assert receipt.composed_pipeline_sha256 != (
        receipt.composition_join.extraction_pipeline_sha256
    )
    assert receipt.scientific_authority is False
    assert receipt.calibration_authority is False
    assert receipt.claim_release_authority is False

    assert (
        validate_corpus_pipeline_composition_external_replay_receipt_v1(
            receipt=receipt,
            repository_root=ROOT,
            grounding_package_path=PACKAGE,
            hosted_bridge_receipt_path=BRIDGE,
        )
        == receipt
    )
    loaded = load_corpus(
        PACKAGE,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=ROOT,
    )
    assert (
        require_external_replay_receipt_matches_corpus_load_result_v1(
            receipt=receipt,
            corpus=loaded,
        )
        == receipt
    )


def test_runtime_receipt_rejects_coherently_rehashed_pipeline_alias_tamper(
    replayed_receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> None:
    payload = replayed_receipt.model_dump(mode="json")
    payload["calibration_pipeline_sha256"] = "0" * 64
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = hash_canonical(unsigned)

    with pytest.raises(ValidationError, match="composed_pipeline_alias_mismatch"):
        CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(payload)


def test_legacy_v5_certificate_bytes_remain_unchanged() -> None:
    certificate_path = (
        ROOT
        / "data/cache/hosted-native-numeric-yield-pilot-v5-verifier/output-v4"
        / "verification-certificate.json"
    )
    if not certificate_path.is_file():
        pytest.skip("local immutable hosted-v5 certificate is unavailable")
    observed = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
    assert observed == "2766057c480ddec7bf0ff4cd7f991d9ead3ad409cfbb2d6d0a1b3a92e43beae9"
