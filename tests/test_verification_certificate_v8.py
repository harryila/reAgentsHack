from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.private_cache_support import require_private_cache

from literature_multiverse.certificate import (
    VerificationCertificate,
    VerificationCertificateV8,
)
from literature_multiverse.cli import main as cli_main
from literature_multiverse.corpus_pipeline_composition_runtime import (
    CorpusPipelineCompositionExternalReplayReceiptV1,
    build_corpus_pipeline_composition_external_replay_receipt_v1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.verifier import (
    VerificationContractError,
    compute_verifier_pipeline_fingerprint,
    load_claim_manifest,
    load_corpus,
    run_verification,
)

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
CLAIM = ROOT / "data/cache/hosted-native-numeric-yield-pilot-v5-verifier/claim.yaml"
LEGACY_CERTIFICATE = (
    ROOT
    / "data/cache/hosted-native-numeric-yield-pilot-v5-verifier/output-v4"
    / "verification-certificate.json"
)

# Every test in this module reaches the private local hosted-v5 cache transitively
# through `v8_inputs` (or `v8_certificate`, which depends on it), so the whole module
# is marked rather than repeating the marker on each test.
pytestmark = pytest.mark.private_cache


@pytest.fixture(scope="module")
def v8_inputs() -> tuple[Any, Any, CorpusPipelineCompositionExternalReplayReceiptV1]:
    require_private_cache(
        "data/cache/hosted-native-numeric-yield-pilot-v5-grounding/typed_evidence_grounding_package.json",
        "data/cache/hosted-native-numeric-yield-pilot-v5-grounding/hosted_native_grounding_bridge_receipt.json",
        "data/cache/hosted-native-numeric-yield-pilot-v5-verifier/claim.yaml",
    )
    manifest = load_claim_manifest(CLAIM)
    corpus = load_corpus(
        PACKAGE,
        legacy_settings=manifest.legacy_adapter,
        repository_root=ROOT,
    )
    receipt = build_corpus_pipeline_composition_external_replay_receipt_v1(
        repository_root=ROOT,
        grounding_package_path=PACKAGE,
        hosted_bridge_receipt_path=BRIDGE,
    )
    return manifest, replace(
        corpus,
        composition_external_replay_receipt=receipt,
    ), receipt


@pytest.fixture(scope="module")
def v8_certificate(
    v8_inputs: tuple[Any, Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> VerificationCertificateV8:
    manifest, corpus, _ = v8_inputs
    result = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=60,
        pipeline_root=ROOT,
        composition_grounding_package_path=PACKAGE,
        composition_hosted_bridge_receipt_path=BRIDGE,
        generated_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
    )
    assert isinstance(result, VerificationCertificateV8)
    return result


def test_real_v5_corpus_runs_through_join_aware_ordinary_v8(
    v8_certificate: VerificationCertificateV8,
) -> None:
    certificate = v8_certificate
    assert certificate.certificate_version == "literature-multiverse-verification-v8"
    assert certificate.status == "abstained"
    assert "adapter:corpus_pipeline_identity_mismatch" not in certificate.reasons
    assert all(
        issue["code"] != "corpus_pipeline_identity_mismatch"
        for issue in certificate.adapter_issues
    )
    assert certificate.pipeline_verification == (
        certificate.composition_external_replay_receipt.composed_pipeline_verification
    )
    assert certificate.release_assessment.pipeline_sha256 == (
        certificate.composition_external_replay_receipt.release_pipeline_sha256
    )
    assert certificate.complete_corpus_identity_v3.external_replay_receipt == (
        certificate.composition_external_replay_receipt
    )
    assert VerificationCertificateV8.model_validate(
        certificate.model_dump(mode="json")
    ) == certificate


def test_v8_clears_only_legacy_pipeline_blocker(
    v8_certificate: VerificationCertificateV8,
) -> None:
    require_private_cache(
        "data/cache/hosted-native-numeric-yield-pilot-v5-verifier/output-v4/verification-certificate.json"
    )
    legacy = VerificationCertificate.model_validate_json(
        LEGACY_CERTIFICATE.read_bytes()
    )
    expected = sorted(
        set(legacy.reasons) - {"adapter:corpus_pipeline_identity_mismatch"}
    )
    assert v8_certificate.reasons == expected
    assert hashlib.sha256(LEGACY_CERTIFICATE.read_bytes()).hexdigest() == (
        "2766057c480ddec7bf0ff4cd7f991d9ead3ad409cfbb2d6d0a1b3a92e43beae9"
    )


def test_v8_rejects_coherently_rehashed_receipt_alias_tamper(
    v8_certificate: VerificationCertificateV8,
) -> None:
    payload = v8_certificate.model_dump(mode="json")
    payload["composition_external_replay_receipt_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="composition_alias_mismatch"):
        VerificationCertificateV8.model_validate(payload)


def test_v8_rejects_manifest_corpus_policy_binding_alias_tamper(
    v8_certificate: VerificationCertificateV8,
) -> None:
    payload = v8_certificate.model_dump(mode="json")
    payload["manifest_corpus_policy_binding_v3_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="composition_alias_mismatch"):
        VerificationCertificateV8.model_validate(payload)


def test_v8_rejects_legacy_pipeline_blocker_reintroduction(
    v8_certificate: VerificationCertificateV8,
) -> None:
    payload = v8_certificate.model_dump(mode="json")
    payload["adapter_issues"] = [
        *payload["adapter_issues"],
        {
            "severity": "blocking",
            "code": "corpus_pipeline_identity_mismatch",
            "detail": "forged downgrade",
            "paper_id": None,
            "finding_id": None,
        },
    ]
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="legacy_pipeline_blocker_not_cleared"):
        VerificationCertificateV8.model_validate(payload)


def test_v8_cannot_be_parsed_as_legacy_v5(
    v8_certificate: VerificationCertificateV8,
) -> None:
    with pytest.raises(ValidationError):
        VerificationCertificate.model_validate(
            v8_certificate.model_dump(mode="json")
        )


def test_composition_rejects_uncomposed_expected_pipeline(
    v8_inputs: tuple[Any, Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    manifest, corpus, _ = v8_inputs
    with pytest.raises(
        VerificationContractError,
        match="expected_pipeline_fingerprint_not_composed_pipeline",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=60,
            expected_pipeline_fingerprint=compute_verifier_pipeline_fingerprint(
                root=ROOT
            ),
            pipeline_root=ROOT,
            composition_grounding_package_path=PACKAGE,
            composition_hosted_bridge_receipt_path=BRIDGE,
            generated_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        )


def test_composition_receipt_requires_full_replay_paths(
    v8_inputs: tuple[Any, Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    manifest, corpus, _ = v8_inputs
    with pytest.raises(
        VerificationContractError,
        match="requires_package_and_bridge_paths",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=60,
            pipeline_root=ROOT,
            generated_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        )


def test_v8_normative_json_contains_no_shadow_corpus_rewrite(
    v8_certificate: VerificationCertificateV8,
) -> None:
    raw = json.loads(v8_certificate.model_dump_json())
    assert raw["corpus"]["metadata"]["pipeline_fingerprint_sha256"] == (
        v8_certificate.composition_external_replay_receipt.composition_join
        .extraction_pipeline_sha256
    )
    assert raw["corpus"]["metadata"]["pipeline_fingerprint_sha256"] != (
        v8_certificate.pipeline_verification.computed_pipeline_sha256
    )


def test_public_cli_replays_composed_fingerprint_and_writes_v8(
    tmp_path: Path,
    v8_inputs: tuple[Any, Any, CorpusPipelineCompositionExternalReplayReceiptV1],
) -> None:
    _, _, receipt = v8_inputs
    receipt_path = tmp_path / "composition-receipt.json"
    fingerprint_path = tmp_path / "composed-pipeline.json"
    output_dir = tmp_path / "verification"
    atomic_write_json(receipt_path, receipt, force=False)

    assert (
        cli_main(
            [
                "fingerprint",
                "--corpus",
                PACKAGE.as_posix(),
                "--composition-receipt",
                receipt_path.as_posix(),
                "--composition-hosted-bridge-receipt",
                BRIDGE.as_posix(),
                "--pipeline-root",
                ROOT.as_posix(),
                "--output",
                fingerprint_path.as_posix(),
            ]
        )
        == 0
    )
    frozen = json.loads(fingerprint_path.read_text())
    assert frozen["pipeline_sha256"] == receipt.composed_pipeline_sha256

    assert (
        cli_main(
            [
                "verify",
                "--claim",
                CLAIM.as_posix(),
                "--corpus",
                PACKAGE.as_posix(),
                "--composition-receipt",
                receipt_path.as_posix(),
                "--composition-hosted-bridge-receipt",
                BRIDGE.as_posix(),
                "--pipeline-root",
                ROOT.as_posix(),
                "--pipeline-fingerprint",
                fingerprint_path.as_posix(),
                "--budget-minutes",
                "60",
                "--output-dir",
                output_dir.as_posix(),
            ]
        )
        == 0
    )
    certificate = VerificationCertificateV8.model_validate_json(
        (output_dir / "verification-certificate.json").read_bytes()
    )
    assert certificate.status == "abstained"
    assert "adapter:corpus_pipeline_identity_mismatch" not in certificate.reasons
