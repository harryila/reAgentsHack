from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from literature_multiverse.evidence_inference_ollama_reporting import (
    SOURCE_CODE_PATHS,
    EvidenceInferenceOllamaReportingError,
    source_code_hashes,
    validate_augmented_public_summary,
)
from literature_multiverse.lineage import hash_canonical, sha256_file


def test_source_inventory_binds_current_declared_runtime_surface(repo_root: Path) -> None:
    observed = source_code_hashes(repo_root)
    assert set(observed) == set(SOURCE_CODE_PATHS)
    for relative, digest in observed.items():
        assert digest == sha256_file(repo_root / relative)


def test_augmented_summary_rejects_stale_source_lineage(
    repo_root: Path,
) -> None:
    sources = source_code_hashes(repo_root)
    supplement_payload = {
        "reporting_supplement_version": (
            "evidence-inference-local-ollama-reproducibility-supplement-v1"
        ),
        "base_public_summary_sha256": "a" * 64,
        "full_private_report_sha256": "b" * 64,
        "input_bundle_sha256": "c" * 64,
        "prediction_ledger_sha256": "d" * 64,
        "source_code_sha256s": sources,
        "source_code_bundle_sha256": hash_canonical(sources),
        "receipt_telemetry": {
            "receipt_rows": 1,
            "done_reason_counts": {"stop": 1},
            "configured_num_predict": 384,
            "execution_failure_rows_with_length_stop_at_num_predict": 0,
        },
        "hash_security_boundary": (
            "unkeyed reproducibility and tamper-evidence hashes; not signatures, "
            "authorship proof, freshness proof, or rollback protection"
        ),
    }
    supplement = {
        **supplement_payload,
        "reporting_supplement_sha256": hash_canonical(supplement_payload),
    }
    payload = {
        "public_summary_version": "evidence-inference-local-ollama-public-summary-v1",
        "diagnostic_version": "evidence-inference-local-ollama-diagnostic-v1",
        "status": "metadata_only_non_pristine_offline_local_model_diagnostic",
        "contains_article_text": False,
        "contains_evidence_quotes": False,
        "contains_gold_labels": False,
        "contains_per_example_labels": False,
        "contains_paper_or_example_ids": False,
        "contains_raw_predictions": False,
        "contains_absolute_paths": False,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "full_private_report_sha256": "b" * 64,
        "input_bundle_sha256": "c" * 64,
        "input_bundle_file_sha256": "e" * 64,
        "prediction_ledger_sha256": "d" * 64,
        "prediction_ledger_file_sha256": "f" * 64,
        "population": {"rows": 1, "articles": 1},
        "generation_config": {"num_predict": 384},
        "local_ollama": {"prediction_output_distribution": {"increase": 1}},
        "fixed_lexical": {"prediction_output_distribution": {"increase": 1}},
        "reproducibility_supplement": supplement,
    }
    summary = {**payload, "public_summary_sha256": hash_canonical(payload)}
    assert validate_augmented_public_summary(summary, repository_root=repo_root) == summary

    tampered = deepcopy(summary)
    tampered_supplement = tampered["reproducibility_supplement"]
    tampered_supplement["source_code_sha256s"][SOURCE_CODE_PATHS[0]] = "0" * 64
    tampered_supplement["source_code_bundle_sha256"] = hash_canonical(
        tampered_supplement["source_code_sha256s"]
    )
    inner = {
        key: value
        for key, value in tampered_supplement.items()
        if key != "reporting_supplement_sha256"
    }
    tampered_supplement["reporting_supplement_sha256"] = hash_canonical(inner)
    outer = {
        key: value for key, value in tampered.items() if key != "public_summary_sha256"
    }
    tampered["public_summary_sha256"] = hash_canonical(outer)
    with pytest.raises(EvidenceInferenceOllamaReportingError, match="source lineage is stale"):
        validate_augmented_public_summary(
            tampered,
            repository_root=repo_root,
            require_current_sources=True,
        )
