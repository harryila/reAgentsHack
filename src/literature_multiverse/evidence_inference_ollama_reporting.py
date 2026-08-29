"""Versioned public reproducibility supplement for the frozen Ollama diagnostic.

This reporting layer is deliberately separate from the run-producing diagnostic module.
The official GEPA study binds the exact bytes of that module, so additive reporting
hardening must not silently mutate its historical execution identity.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_ollama import (
    validate_frozen_predictions_before_label_access,
    validate_private_report,
    validate_public_summary,
)
from literature_multiverse.lineage import hash_canonical, sha256_file

REPORTING_SUPPLEMENT_VERSION = (
    "evidence-inference-local-ollama-reproducibility-supplement-v1"
)
SOURCE_CODE_PATHS = (
    "pyproject.toml",
    "scripts/evaluate_evidence_inference_ollama.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/evidence_inference.py",
    "src/literature_multiverse/evidence_inference_diagnostic.py",
    "src/literature_multiverse/evidence_inference_ollama.py",
    "src/literature_multiverse/evidence_inference_ollama_reporting.py",
    "src/literature_multiverse/grounding.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/local_ollama.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "src/literature_multiverse/prompt_optimization.py",
    "src/literature_multiverse/prompting.py",
    "src/literature_multiverse/providers.py",
    "uv.lock",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EvidenceInferenceOllamaReportingError(ValueError):
    """The reporting supplement or its frozen receipt lineage is invalid."""


def source_code_hashes(repository_root: Path | None = None) -> dict[str, str]:
    """Hash the declared direct and transitive reporting/evaluator runtime surface."""

    root = (
        repository_root.resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    missing = [relative for relative in SOURCE_CODE_PATHS if not (root / relative).is_file()]
    if missing:
        raise EvidenceInferenceOllamaReportingError(
            f"diagnostic reporting source files are missing: {missing}"
        )
    return {relative: sha256_file(root / relative) for relative in SOURCE_CODE_PATHS}


def _receipt_telemetry(
    receipts: Mapping[str, Mapping[str, Any]], *, num_predict: int
) -> dict[str, Any]:
    done_reasons: Counter[str] = Counter()
    output_cap_execution_failures = 0
    for receipt in receipts.values():
        telemetry = receipt.get("telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        done_reason = telemetry.get("done_reason")
        if isinstance(done_reason, str) and done_reason:
            done_reasons[done_reason] += 1
        if (
            receipt.get("status") == "execution_failure"
            and done_reason == "length"
            and telemetry.get("eval_count") == num_predict
        ):
            output_cap_execution_failures += 1
    return {
        "receipt_rows": len(receipts),
        "done_reason_counts": dict(sorted(done_reasons.items())),
        "configured_num_predict": num_predict,
        "execution_failure_rows_with_length_stop_at_num_predict": (
            output_cap_execution_failures
        ),
    }


def augment_public_summary(
    *,
    report: Mapping[str, Any],
    public_summary: Mapping[str, Any],
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    receipts_dir: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Attach source and truncation aggregates after independent frozen-ledger replay."""

    validated_report = validate_private_report(report)
    validated_public = validate_public_summary(public_summary)
    bundle, ledger, receipts = validate_frozen_predictions_before_label_access(
        input_bundle=input_bundle,
        prediction_ledger=prediction_ledger,
        receipts_dir=receipts_dir,
    )
    execution_inputs = validated_report["execution_inputs"]
    if (
        execution_inputs.get("input_bundle_sha256") != bundle["input_bundle_sha256"]
        or execution_inputs.get("prediction_ledger_sha256")
        != ledger["prediction_ledger_sha256"]
        or validated_public.get("full_private_report_sha256")
        != validated_report["report_sha256"]
    ):
        raise EvidenceInferenceOllamaReportingError(
            "reporting supplement inputs differ from frozen scored artifacts"
        )
    sources = source_code_hashes(repository_root)
    base_public_hash = validated_public["public_summary_sha256"]
    num_predict = int(ledger["generation_config"]["num_predict"])
    supplement_payload = {
        "reporting_supplement_version": REPORTING_SUPPLEMENT_VERSION,
        "base_public_summary_sha256": base_public_hash,
        "full_private_report_sha256": validated_report["report_sha256"],
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "source_code_sha256s": sources,
        "source_code_bundle_sha256": hash_canonical(sources),
        "receipt_telemetry": _receipt_telemetry(receipts, num_predict=num_predict),
        "hash_security_boundary": (
            "unkeyed reproducibility and tamper-evidence hashes; not signatures, "
            "authorship proof, freshness proof, or rollback protection"
        ),
    }
    supplement = {
        **supplement_payload,
        "reporting_supplement_sha256": hash_canonical(supplement_payload),
    }
    payload = deepcopy(dict(validated_public))
    payload.pop("public_summary_sha256", None)
    payload["reproducibility_supplement"] = supplement
    augmented = {**payload, "public_summary_sha256": hash_canonical(payload)}
    return validate_augmented_public_summary(
        augmented,
        repository_root=repository_root,
        require_current_sources=True,
    )


def validate_augmented_public_summary(
    summary: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    require_current_sources: bool = False,
) -> dict[str, Any]:
    """Validate supplement structure, self-hash, and optionally current source bytes."""

    validated = validate_public_summary(summary)
    supplement = validated.get("reproducibility_supplement")
    if not isinstance(supplement, Mapping):
        raise EvidenceInferenceOllamaReportingError("reporting supplement is missing")
    supplement_payload = {
        key: value
        for key, value in supplement.items()
        if key != "reporting_supplement_sha256"
    }
    sources = supplement.get("source_code_sha256s")
    telemetry = supplement.get("receipt_telemetry")
    generation_config = validated.get("generation_config")
    if (
        supplement.get("reporting_supplement_version")
        != REPORTING_SUPPLEMENT_VERSION
        or supplement.get("reporting_supplement_sha256")
        != hash_canonical(supplement_payload)
        or supplement.get("full_private_report_sha256")
        != validated.get("full_private_report_sha256")
        or supplement.get("input_bundle_sha256")
        != validated.get("input_bundle_sha256")
        or supplement.get("prediction_ledger_sha256")
        != validated.get("prediction_ledger_sha256")
        or not isinstance(sources, Mapping)
        or set(sources) != set(SOURCE_CODE_PATHS)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in sources.values()
        )
        or supplement.get("source_code_bundle_sha256") != hash_canonical(sources)
        or not isinstance(telemetry, Mapping)
        or not isinstance(generation_config, Mapping)
        or telemetry.get("configured_num_predict") != generation_config.get("num_predict")
        or not isinstance(telemetry.get("receipt_rows"), int)
        or telemetry.get("receipt_rows") != validated.get("population", {}).get("rows")
        or not isinstance(telemetry.get("done_reason_counts"), Mapping)
        or sum(telemetry["done_reason_counts"].values()) != telemetry["receipt_rows"]
        or not isinstance(
            telemetry.get("execution_failure_rows_with_length_stop_at_num_predict"),
            int,
        )
    ):
        raise EvidenceInferenceOllamaReportingError(
            "reporting supplement contract mismatch"
        )
    if require_current_sources and dict(sources) != source_code_hashes(repository_root):
        raise EvidenceInferenceOllamaReportingError(
            "reporting supplement source lineage is stale"
        )
    return dict(summary)


__all__ = [
    "REPORTING_SUPPLEMENT_VERSION",
    "SOURCE_CODE_PATHS",
    "EvidenceInferenceOllamaReportingError",
    "augment_public_summary",
    "source_code_hashes",
    "validate_augmented_public_summary",
]
