from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.evidence_inference_gepa_scaled_readiness_v1 as readiness
from literature_multiverse.evidence_inference_gepa_scaled_readiness_v1 import (
    EvidenceInferenceGEPAScaledReadinessConfigV1,
    EvidenceInferenceGEPAScaledReadinessError,
    EvidenceInferenceGEPAScaledReadinessReceiptV1,
)
from literature_multiverse.lineage import hash_canonical, sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> EvidenceInferenceGEPAScaledReadinessReceiptV1:
    return readiness.freeze_evidence_inference_gepa_scaled_readiness_v1(repository_root=ROOT)


def test_real_metadata_only_readiness_is_blocked_and_externally_replayable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original = Path.read_text

    def guarded(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.suffix == ".jsonl" or path.name == "paired-test-report.json":
            raise AssertionError(f"row-level or private payload opened: {path}")
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    frozen = readiness.freeze_evidence_inference_gepa_scaled_readiness_v1(repository_root=ROOT)
    replayed = readiness.validate_evidence_inference_gepa_scaled_readiness_v1(
        receipt=frozen,
        repository_root=ROOT,
        external_replay=True,
    )

    assert replayed == frozen
    assert frozen.status == (
        "blocked_no_unopened_question_and_paper_disjoint_evaluation_population"
    )
    assert frozen.benchmark_row_payloads_opened_by_preflight is False
    assert frozen.private_reports_opened_by_preflight is False
    assert frozen.hidden_or_reference_labels_opened_by_preflight is False
    assert frozen.provider_calls_made_by_preflight == 0
    assert frozen.live_optimizer_or_evaluation_calls_authorized is False
    assert frozen.accuracy_claim_authority is False
    assert frozen.gepa_improvement_claim_authority is False
    assert frozen.calibration_claim_authority is False
    assert frozen.claim_release_authority is False
    assert not any(path.suffix == ".jsonl" for path in opened)


def test_exact_local_population_and_prior_exposure_are_frozen(
    receipt: EvidenceInferenceGEPAScaledReadinessReceiptV1,
) -> None:
    counts = {
        item.variant: {split.split: (split.rows, split.papers) for split in item.splits}
        for item in receipt.local_variants
    }
    assert counts == {
        "full": {
            "train": (4371, 1477),
            "dev": (522, 192),
            "test": (524, 191),
        },
        "low_budget_12": {
            "train": (12, 12),
            "dev": (12, 12),
            "test": (12, 12),
        },
        "pilot30": {
            "train": (30, 4),
            "dev": (30, 16),
            "test": (30, 7),
        },
    }
    prior = receipt.prior_study
    assert prior.all_labels_historically_opened is True
    assert prior.confirmatory_claim_allowed is False
    assert (prior.train_examples, prior.train_articles) == (256, 93)
    assert (prior.development_examples, prior.development_articles) == (96, 34)
    assert (prior.paired_test_examples, prior.paired_test_articles) == (524, 191)
    assert prior.provider_call_unseen_but_label_opened_rows == 482
    assert prior.provider_call_unseen_but_label_opened_articles == 179
    assert prior.accepted_candidate_count == 7
    assert prior.actual_metric_calls == 864
    assert prior.observed_improvement_rule_satisfied is False
    assert receipt.qualified_examples_all_expected_eligible is True
    assert receipt.eligibility_negative_example_count == 0


def test_external_data_contract_is_exact_and_separates_objectives(
    receipt: EvidenceInferenceGEPAScaledReadinessReceiptV1,
) -> None:
    assert receipt.required_separate_evaluation_objectives == [
        "eligibility_screening_recall_and_specificity",
        "extraction_target_correctness",
        "formal_exact_source_grounding_validity",
        "structured_output_reliability",
        "provider_usage_and_cost",
    ]
    requirements = set(receipt.external_evaluation_requirements)
    assert "eligibility_positive_and_negative_examples_are_prespecified" in requirements
    assert (
        "evaluation_labels_are_held_by_an_external_evaluator_and_unavailable_locally"
        in requirements
    )
    assert (
        "structured_output_failures_grounding_failures_and_task_errors_are_separate" in requirements
    )
    assert "usage_and_cost_receipts_are_complete_for_every_attempt" in requirements
    assert "seed_and_gepa_winner_receive_equal_paired_evaluation_call_budgets" in requirements
    assert "calibration_units_are_complete_independent_review_questions" in requirements


def test_coherently_rehashed_blocker_or_authority_forgery_fails_intrinsically(
    receipt: EvidenceInferenceGEPAScaledReadinessReceiptV1,
) -> None:
    missing_blocker = receipt.model_dump(mode="json")
    missing_blocker["blocker_codes"] = missing_blocker["blocker_codes"][:-1]
    missing_blocker["receipt_sha256"] = hash_canonical(
        {key: value for key, value in missing_blocker.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="receipt_alias_mismatch"):
        EvidenceInferenceGEPAScaledReadinessReceiptV1.model_validate(missing_blocker)

    authority = receipt.model_dump(mode="json")
    authority["accuracy_claim_authority"] = True
    authority["receipt_sha256"] = hash_canonical(
        {key: value for key, value in authority.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError):
        EvidenceInferenceGEPAScaledReadinessReceiptV1.model_validate(authority)


def test_coherently_rehashed_config_cannot_relax_external_requirements() -> None:
    path = ROOT / readiness.DEFAULT_CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    value["external_evaluation_requirements"] = value["external_evaluation_requirements"][:-1]
    value["config_sha256"] = hash_canonical(
        {key: item for key, item in value.items() if key != "config_sha256"}
    )
    with pytest.raises(ValueError, match="config_contract_changed"):
        EvidenceInferenceGEPAScaledReadinessConfigV1.model_validate(value)


def test_converter_task_shape_proof_rejects_eligibility_negative_claim(
    tmp_path: Path,
) -> None:
    source = tmp_path / "converter.py"
    source.write_text(
        "def _optimization_example():\n"
        "    return OptimizationExample(expected_output={'eligible': False})\n",
        encoding="utf-8",
    )
    with pytest.raises(
        EvidenceInferenceGEPAScaledReadinessError,
        match="eligibility_task_shape_changed",
    ):
        readiness._prove_all_qualified_examples_expected_eligible(source)


def test_manifest_paper_leakage_fails_closed_before_any_row_payload_read(
    tmp_path: Path,
) -> None:
    source_manifest = ROOT / "data/cache/evidence-inference-gepa/manifest.json"
    source_report = ROOT / "data/cache/evidence-inference-gepa/conversion_report.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    report = json.loads(source_report.read_text(encoding="utf-8"))
    manifest["dev"]["paper_ids"][0] = manifest["train"]["paper_ids"][0]
    manifest["dev"]["group_ids"][0] = manifest["train"]["group_ids"][0]
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "conversion_report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report["manifest_sha256"] = sha256_file(manifest_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    for split in ("train", "dev", "test"):
        (tmp_path / manifest[split]["path"]).touch()

    with pytest.raises(
        EvidenceInferenceGEPAScaledReadinessError,
        match="manifest_membership_leakage",
    ):
        readiness._variant_facts(
            variant="full",
            manifest_path=manifest_path,
            report_path=report_path,
        )


def test_external_replay_rejects_coherent_receipt_substitution(
    receipt: EvidenceInferenceGEPAScaledReadinessReceiptV1,
) -> None:
    payload = receipt.model_dump(mode="json")
    payload["component_source_sha256"] = "b" * 64
    payload["receipt_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    forged = EvidenceInferenceGEPAScaledReadinessReceiptV1.model_validate(payload)
    with pytest.raises(
        EvidenceInferenceGEPAScaledReadinessError,
        match="external_replay_mismatch",
    ):
        readiness.validate_evidence_inference_gepa_scaled_readiness_v1(
            receipt=forged,
            repository_root=ROOT,
            external_replay=True,
        )


def test_symlinked_config_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "config.json"
    link.symlink_to(ROOT / readiness.DEFAULT_CONFIG_PATH)
    with pytest.raises(
        EvidenceInferenceGEPAScaledReadinessError,
        match=r"config_outside_repository|config_unsafe",
    ):
        readiness.freeze_evidence_inference_gepa_scaled_readiness_v1(
            repository_root=ROOT,
            config_path=link,
        )
