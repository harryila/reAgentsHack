"""Public-only validation registry for headline scientific result artifacts."""

from __future__ import annotations

import json
import math
import re
import runpy
from collections.abc import Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Literal

from literature_multiverse.adaptive_stress_study import (
    validate_adaptive_stress_study_artifact,
)
from literature_multiverse.evidence_inference_diagnostic import (
    validate_public_diagnostic_summary,
)
from literature_multiverse.evidence_inference_item_risk import (
    compute_diagnostic_pipeline_fingerprint as compute_ei_item_risk_pipeline_fingerprint,
)
from literature_multiverse.evidence_inference_item_risk import (
    load_config as load_ei_item_risk_config,
)
from literature_multiverse.evidence_inference_item_risk import (
    validate_public_summary as validate_ei_item_risk_summary,
)
from literature_multiverse.evidence_inference_ollama_reporting import (
    validate_augmented_public_summary,
)
from literature_multiverse.evidencebench_diagnostic import (
    validate_evidencebench_public_bundle,
)
from literature_multiverse.harvester.validation import (
    HarvesterValidationError,
    validate_harvester_validation_summary,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.local_corpus_audit import validate_local_corpus_audit
from literature_multiverse.metasyn_benchmark import (
    FixedDirectionBaselineReceipt,
    MetaSynEvaluatorLabel,
    evaluate_metasyn_predictions,
    load_metasyn_inputs,
    load_metasyn_manifest_metadata,
    load_metasyn_predictions,
)
from literature_multiverse.metasyn_synthesis_yield import (
    MetaSynSynthesisYieldPublicSummaryV1,
)
from literature_multiverse.metasyn_synthesis_yield_v2 import (
    MetaSynSynthesisYieldPublicSummaryV2,
)
from literature_multiverse.native_bounded_ollama_diagnostic import (
    validate_bounded_public_summary,
)
from literature_multiverse.native_ollama_diagnostic import (
    validate_public_summary as validate_native_public_summary,
)
from literature_multiverse.ollama_gepa_study import (
    validate_public_summary as validate_gepa_public_summary,
)
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint

SemanticValidator = Literal[
    "adaptive_stress",
    "closed_corpus",
    "decisive_readiness_blocked",
    "evidence_boundary_ledger",
    "evidence_inference_provider_free",
    "evidence_inference_local_ollama",
    "evidence_inference_ollama_gepa",
    "evidence_inference_item_risk",
    "evidencebench_grounding",
    "fable_public_paired_summary",
    "fable_public_union_evaluation_v2",
    "generic",
    "harvester",
    "historical_verification_certificate_v5",
    "legacy_antiox_bundles",
    "local_suite",
    "metasyn_fixed_positive",
    "metasyn_offline_audit_model_only",
    "metasyn_retrieval",
    "metasyn_screening",
    "metasyn_synthesis_yield",
    "metasyn_synthesis_yield_v2",
    "native_bounded_ollama",
    "native_bounded_ollama_v1_historical",
    "native_ollama",
    "planted_simulation",
    "source_bridge",
]


class PublicArtifactValidationError(ValueError):
    """A registered public result is missing, stale, malformed, or tampered."""


class _DuplicateJSONKeyError(ValueError):
    """Internal signal for an ambiguous JSON object."""


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKeyError(key)
        value[key] = item
    return value


SourceMapRole = Literal["current_checkout_replay", "historical_execution"]


@dataclass(frozen=True, slots=True)
class PublicSourceMapBinding:
    location: str
    role: SourceMapRole
    expected_bundle_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PublicArtifactSpec:
    path: str
    self_hash_field: str | None
    semantic_validator: SemanticValidator = "generic"
    limitation: str | None = None
    result_recomputed_from_public_inputs: bool = False
    source_map_bindings: tuple[PublicSourceMapBinding, ...] | None = None


PUBLIC_RESULT_REGISTRY = (
    PublicArtifactSpec(
        "artifacts/diagnostics/adaptive-stress-study-v1.json",
        "artifact_sha256",
        "adaptive_stress",
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/summary.json",
        "public_summary_sha256",
        "evidence_inference_provider_free",
        "historical public report binds execution/private artifacts but no current source map",
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference-ollama/summary.json",
        "public_summary_sha256",
        "evidence_inference_local_ollama",
        source_map_bindings=(
            PublicSourceMapBinding(
                "$.reproducibility_supplement.source_code_sha256s",
                "current_checkout_replay",
            ),
        ),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json",
        "public_summary_sha256",
        "evidence_inference_ollama_gepa",
        "historical execution artifact; current CI validates its pinned execution "
        "source-map bundle without relabeling the frozen model outputs",
        False,
        (
            PublicSourceMapBinding(
                "$.lineage.source_code_sha256s",
                "historical_execution",
                "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3",
            ),
        ),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json",
        "public_summary_sha256",
        "evidence_inference_item_risk",
        "historical diagnostic: its frozen 23-file pipeline fingerprint (2949fde1...) is "
        "bound by the tracked companion manifest and no longer equals the current tree "
        "(pipeline_fingerprint.py changed 2026-08-29). Before 2026-08-29 public CI recomputed "
        "this fingerprint from current bytes and required equality; it no longer does, and "
        "checks only manifest self-consistency, the closure definition, aggregate semantics, "
        "and public GEPA lineage. Ignored row-level paired predictions/labels are unavailable "
        "for metric replay, and this historically opened diagnostic has no claim-release "
        "authority",
        False,
        (
            PublicSourceMapBinding(
                "$.prediction_source.historical_source_code_sha256s",
                "historical_execution",
                "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3",
            ),
        ),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidencebench-grounding-v1/summary.json",
        "summary_sha256",
        "evidencebench_grounding",
        "public CI validates the aggregate summary, exact-replay receipt, and current "
        "source/config/runtime bindings but cannot rerun metrics because licensed "
        "development/test rows and private predictions are intentionally untracked",
        False,
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-retrieval-study-v1.json",
        "public_summary_payload_sha256",
        "metasyn_retrieval",
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-screening-study-v1.json",
        "public_summary_payload_sha256",
        "metasyn_screening",
        "zero-call fields are artifact declarations; the integrated suite supplies "
        "runtime socket-denial evidence",
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json",
        "summary_sha256",
        "metasyn_synthesis_yield",
        "public CI validates the identifier-free aggregate contract, immutable "
        "lineage and counts without loading the ignored private runtime or "
        "synthesis reports; this yield-only diagnostic reports no extraction "
        "accuracy, calibration, or claim-release authority",
        False,
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-synthesis-yield-v2/summary.json",
        "summary_sha256",
        "metasyn_synthesis_yield_v2",
        "public CI validates the identifier-free hosted aggregate contract, exact "
        "runtime/evaluation lineage, and zero-yield counts without loading the "
        "ignored private hosted runtime or synthesis reports; this label-blind "
        "yield diagnostic reports no extraction accuracy, calibration, truth, or "
        "claim-release authority",
        False,
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/native-antiox-ollama/summary.json",
        "public_summary_sha256",
        "native_ollama",
        "historical one-stage local-model diagnostic retained only for provenance; "
        "it is superseded by the bounded two-stage diagnostic and has no current "
        "extraction, calibration, or claim-release authority",
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/native-antiox-bounded-ollama/summary.json",
        "summary_sha256",
        "native_bounded_ollama_v1_historical",
        "immutable historical v1 negative diagnostic bound to its exact frozen "
        "code/config/model/pipeline identities and aggregate self-hash; public CI "
        "does not reinterpret it under later bounded-schema versions, and empirical "
        "counts require the ignored private v1 intent, receipt, ledger, and report "
        "replay; it reports no extraction accuracy, calibration, or claim-release "
        "authority",
        False,
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/native-source/antiox-eligible-source-bridge.json",
        "run_sha256",
        "source_bridge",
        limitation=(
            "tracked archived corpus bytes are rehashed; scientific labels and an "
            "independent gold extraction are unavailable"
        ),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/native-source/antiox-source-bridge.json",
        "run_sha256",
        "source_bridge",
        limitation=(
            "tracked archived corpus bytes are rehashed; scientific labels and an "
            "independent gold extraction are unavailable"
        ),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/native-source/metasyn-boundary-source-bridge.json",
        "run_sha256",
        "source_bridge",
        limitation=(
            "third-party article shards are not public; their declarations are checked "
            "against the tracked revision-pinned corpus manifest"
        ),
    ),
    PublicArtifactSpec(
        "artifacts/benchmarks/local-suite-v1/benchmark-report.json",
        "report_payload_sha256",
        "local_suite",
    ),
    PublicArtifactSpec(
        "artifacts/antiox-training/demo/manifest.json",
        None,
        "legacy_antiox_bundles",
        "semantic validator loads the demo and every frozen release through the "
        "legacy offline app boundary; this is bundle consistency, not result recomputation",
    ),
    PublicArtifactSpec(
        "artifacts/paper/budgeted-verification-simulation-200.json",
        "artifact_payload_sha256",
        "planted_simulation",
        result_recomputed_from_public_inputs=True,
    ),
    PublicArtifactSpec(
        "artifacts/paper/calibration-simulation-100.json",
        "artifact_payload_sha256",
        "planted_simulation",
        result_recomputed_from_public_inputs=True,
    ),
    PublicArtifactSpec(
        "artifacts/paper/closed-corpus-local-audit.json",
        "audit_payload_sha256",
        "closed_corpus",
    ),
    PublicArtifactSpec(
        "artifacts/paper/harvester/validation_summary.json",
        "artifact_payload_sha256",
        "harvester",
        "historical single-record OpenAlex live/replay transport probe of "
        "2026-08-26; harvester/sources.py changed on 2026-08-29 and the probe was "
        "not re-run; the frozen source map is hash-pinned, not recomputed; "
        "transport/provenance validation only, never retrieval-recall evidence",
        False,
        (
            PublicSourceMapBinding(
                "$.reproducibility.source_files_sha256",
                "historical_execution",
                "b663b0ea80c9cfd1c11d1ad59f90e94d6227ba65232e69e72eee972b2824bacd",
            ),
        ),
    ),
    PublicArtifactSpec(
        "artifacts/paper/meta-simulation-200.json",
        "artifact_payload_sha256",
        "planted_simulation",
        result_recomputed_from_public_inputs=True,
    ),
    PublicArtifactSpec(
        "artifacts/paper/metasyn-fixed-positive-test/freeze_receipt.json",
        None,
        "metasyn_fixed_positive",
        "three-file fixed-direction control bundle has no aggregate self-hash; "
        "every file and recomputed metric is cross-bound instead",
        True,
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/"
        "verification-certificate.json",
        "certificate_sha256",
        "historical_verification_certificate_v5",
        "historical abstained v5 certificate from the single-row MetaSyn post-live "
        "recovery diagnostic; its embedded pipeline identity predates the "
        "2026-08-29 fingerprint bump and is not recomputed; embeds MetaSyn source "
        "quotes and titles (rights policy: release-blocking); no release or "
        "effectiveness authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/"
        "sequential-audit-state.json",
        "state_sha256",
        "generic",
        "historical sequential state bound byte-for-byte inside the sibling v5 "
        "certificate; embeds MetaSyn quotes/titles; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/"
        "external-validation.json",
        "validation_sha256",
        "generic",
        "hash/flag-only validation receipt that byte-binds the three sibling "
        "files; embeds no source text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-boundary-ledger-v1.json",
        "ledger_sha256",
        "evidence_boundary_ledger",
        "implementation identity current (checked against the tree); embedded "
        "question-evaluation pipeline identity historical (drifted after "
        "2026-08-29); every authority boundary in the ledger is false; no "
        "authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/decisive-claim-evaluation-v1-real-readiness-blocked.json",
        "readiness_sha256",
        "decisive_readiness_blocked",
        "current blocked readiness receipt with zero development/calibration/"
        "evaluation questions; no run, no pipeline identity; "
        "`real_scored_run_candidate=false`",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json",
        "plan_sha256",
        "generic",
        "frozen plan/roster; no execution identity; embeds public Evidence "
        "Inference PMC article and benchmark example identifiers, no article "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json",
        "plan_sha256",
        "generic",
        "frozen plan/roster; no execution identity; embeds public Evidence "
        "Inference PMC article and benchmark example identifiers, no article "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/"
        "fable-retrospective-pilot30-recovery-v2-plan-v1.json",
        "plan_sha256",
        "generic",
        "frozen plan/roster; no execution identity; embeds public Evidence "
        "Inference PMC article and benchmark example identifiers, no article "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/"
        "fable-retrospective-pilot-recovery-v2-exclusions.json",
        "exclusion_ledger_sha256",
        "generic",
        "frozen plan/roster; no execution identity; embeds public Evidence "
        "Inference PMC article and benchmark example identifiers, no article "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/gepa-candidate-search-plan-v1.json",
        "plan_sha256",
        "generic",
        "frozen plan/roster; no execution identity; embeds public Evidence "
        "Inference PMC article and benchmark example identifiers, no article "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/"
        "fable-retrospective-pilot-recovery-v2-execution-policy.json",
        "policy_sha256",
        "generic",
        "frozen execution policy; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json",
        "public_summary_sha256",
        "fable_public_paired_summary",
        "public aggregate of a retrospective cross-model transfer on "
        "historically opened labels; every authority flag typed False; "
        "exploratory only",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/"
        "fable-retrospective-pilot30-recovery-v2-summary-v1.json",
        "public_summary_sha256",
        "fable_public_paired_summary",
        "public aggregate of a retrospective cross-model transfer on "
        "historically opened labels; every authority flag typed False; "
        "exploratory only",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/evidence-inference/"
        "fable-retrospective-full-union-evaluation-v2.json",
        "evaluation_sha256",
        "fable_public_union_evaluation_v2",
        "public aggregate of a retrospective cross-model transfer on "
        "historically opened labels; every authority flag typed False; "
        "exploratory only",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/contextual-grounding-offline-feasibility-suite-v3.json",
        "suite_sha256",
        "generic",
        "historical offline diagnostic; embedded pipeline identity drifted "
        "after 2026-08-29 and is not recomputed; embeds MetaSyn quotes/titles; "
        "no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-passage-offline-feasibility-audit-v1.json",
        "audit_sha256",
        "metasyn_offline_audit_model_only",
        "historical offline diagnostic; embedded pipeline identity drifted "
        "after 2026-08-29 and is not recomputed; embeds MetaSyn quotes; "
        "no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/metasyn-contextual-frontier-v1-failure-audit-v1.json",
        "audit_sha256",
        "metasyn_offline_audit_model_only",
        "historical offline diagnostic; embedded pipeline identity drifted "
        "after 2026-08-29 and is not recomputed; no embedded MetaSyn source "
        "text; no authority",
        False,
        (),
    ),
    PublicArtifactSpec(
        "artifacts/diagnostics/postlive-recovery-v4-join-v1.json",
        "artifact_sha256",
        "generic",
        "historical offline diagnostic; embedded pipeline identity drifted "
        "after 2026-08-29 and is not recomputed; embeds MetaSyn quotes/titles; "
        "no authority",
        False,
        (),
    ),
)

_SOURCE_MAP_KEYS = frozenset(
    {
        "historical_source_code_sha256s",
        "source_code_sha256s",
        "source_files_sha256",
    }
)
_METASYN_BENCHMARK_MANIFEST = "artifacts/paper/metasyn-benchmark/manifest.json"
_METASYN_CORPUS_MANIFEST = "configs/benchmarks/metasyn-corpus-c8fa07d.json"
_METASYN_CLOSED_AUDIT = "artifacts/paper/closed-corpus-local-audit.json"
_METASYN_SYNTHESIS_YIELD_REGISTERED = {
    "summary_sha256": (
        "29ab78fa11a3c2393431f32507b04a2241e52031098a55ebab58278dceb9afbb"
    ),
    "evaluation_pipeline_sha256": (
        "6d4910284ee1ace54f4922bfec29021bd8a3ba7d186cffc7886115672f5f3127"
    ),
    "runtime_private_report_sha256": (
        "527e28be0711eeff1690507f3124ba9c7a6d74004bcb2c9f5a0b4bc2513b732b"
    ),
    "synthesis_private_report_sha256": (
        "9423772efbf1a23c0b6821a0ee201ae656247c0bbe0efcc2adc1299484252bce"
    ),
    "adapter_bundle_sha256": (
        "15226fe7d6838b27886447ae152d7ef69e733c8b031baae1007ba945041220e8"
    ),
    "prepare_bundle_sha256": (
        "e29122c792642b102f6f3941a52ea307cf588597ec11b846e2e82a9ecdeffc11"
    ),
    "downstream_verifier_pipeline_sha256": (
        "1f332b803be247af6ee7c8e25e9eb0320dbe32203e3d36a0dc9e366a68dac475"
    ),
}
_METASYN_SYNTHESIS_YIELD_V2_REGISTERED = {
    "summary_sha256": (
        "343373f59bc4e4299a45ed8e8880dbd55fc758a17d288608b9813f6ae0e3d858"
    ),
    "evaluation_pipeline_sha256": (
        "92dce71d75d311b7d1f4ae3618613b30d6638287553013c1ec66b37df7823ded"
    ),
    "runtime_private_report_sha256": (
        "3e1a99a3a3e2dc124d11539046008982b9a2c050485427e58a275440bd86ff67"
    ),
    "hosted_runtime_private_report_sha256": (
        "3e1a99a3a3e2dc124d11539046008982b9a2c050485427e58a275440bd86ff67"
    ),
    "hosted_runtime_pipeline_sha256": (
        "d813267fc90717a3842fc70baf3b6f62ace5b12c582595b03365b33141973dea"
    ),
    "hosted_execution_bundle_sha256": (
        "d53fedfb58ab4937fe314d10d1612d300d573f60eb0a968de0e861b67b5c3aa7"
    ),
    "row_view_membership_sha256": (
        "fd561394d1baac15a5d863cfa46bbeaca8ba64c9bf9fc75c9e0069e04e2785b9"
    ),
    "synthesis_private_report_sha256": (
        "cf9199e559b2abead031030309dbf228f6443797266fb80e083ad42deedab21a"
    ),
    "adapter_bundle_sha256": (
        "15226fe7d6838b27886447ae152d7ef69e733c8b031baae1007ba945041220e8"
    ),
    "prepare_bundle_sha256": (
        "e29122c792642b102f6f3941a52ea307cf588597ec11b846e2e82a9ecdeffc11"
    ),
    "downstream_verifier_pipeline_sha256": (
        "1f332b803be247af6ee7c8e25e9eb0320dbe32203e3d36a0dc9e366a68dac475"
    ),
}
_FIXED_POSITIVE_DIRECTORY = "artifacts/paper/metasyn-fixed-positive-test"
_LOCAL_SUITE_CONFIG = "configs/benchmarks/local-suite-v1.json"
_NATIVE_CONFIG = "configs/benchmarks/native-antiox-ollama-v1.json"
_NATIVE_PUBLIC_SUMMARY = "artifacts/diagnostics/native-antiox-ollama/summary.json"
_NATIVE_BOUNDED_V1_LINEAGE = {
    "config_file_sha256": "26a86e5800ee8d727be3f374619fedd1c3e5c27be75d9a18973655c1b8b6344e",
    "config_sha256": "f42384e399602e4cd42a0cf9f07f56b0220560c61acfdf32c6e5b1e8d4fea746",
    "diagnostic_execution_sha256": (
        "938c790b77e144e6195d3e9b6c5efdb86fff4fd9f7ceea6bc3910e3d26d80f9b"
    ),
    "downstream_verifier_pipeline_sha256": (
        "1f332b803be247af6ee7c8e25e9eb0320dbe32203e3d36a0dc9e366a68dac475"
    ),
    "input_bundle_sha256": "f48876719dbfd95308bcf0d405add1a78ee7882a68636243654f8e5b9afa24dd",
    "official_schema_sha256": "8913bfa2846c6f45cb27789c3ab47199c38322ae0954c847f77bcb10750a3d65",
    "prediction_ledger_sha256": (
        "38872854af5eb720442ce95e5486b3286237552d2b140522e721c9f3a9a1cda2"
    ),
    "private_report_sha256": "632284469d7828b8fe6d9823e74665cf6d1108ead39882144b021f1e0e7d2dbc",
    "source_adapter_sha256": "690ac9ea937ebdcb33121ea087970a092d715a5a05711c03eac029d58001a7cc",
    "summary_sha256": "af6680f7f2d94ad5bea6bb59e6fcddc46e140ee8377447c96b55d7413ae788dc",
}
_EVIDENCEBENCH_AUDIT_RECEIPT = (
    "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json"
)
_EVIDENCE_INFERENCE_ITEM_RISK_CONFIG = (
    "configs/benchmarks/evidence-inference-item-risk-v1.json"
)
_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_FINGERPRINT = (
    "artifacts/diagnostics/evidence-inference/"
    "item-risk-calibration-v1-pipeline-fingerprint.json"
)
_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_PIPELINE_SHA256 = (
    "2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15"
)
_EVIDENCE_INFERENCE_ITEM_RISK_REGISTERED_POPULATION = {
    "source_paired_examples": 524,
    "source_unique_papers": 191,
    "representative_units": 191,
    "development_units": 82,
    "calibration_units": 109,
}
_EVIDENCE_INFERENCE_ITEM_RISK_REGISTERED_CELLS = [
    ("risk-bin-000", 64, 42),
    ("risk-bin-001", 27, 21),
    ("risk-bin-002", 18, 12),
]
_EVIDENCEBENCH_REQUIRED_LIMITATIONS = frozenset(
    {
        "chronology_not_externally_preregistered",
        "does_not_validate_claim_correctness",
        "does_not_validate_effect_extraction",
        "does_not_validate_meta_analysis",
        "does_not_validate_release_calibration",
        "public_test_accessible_not_secret",
        "retrospective_public_test_diagnostic",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_BRIDGE_TOP_LEVEL_KEYS = frozenset(
    {
        "content_scope_counts",
        "corpus_kind",
        "dataset_version",
        "diagnostic_only",
        "diagnostic_source_ledger_content_sha256",
        "diagnostic_source_ledger_file_sha256",
        "labels_previously_opened",
        "license_status",
        "manifest_excluded_records",
        "native_manifest_records",
        "native_source_manifest_content_sha256",
        "native_source_manifest_file_sha256",
        "pristine_final_holdout_eligible",
        "question_id",
        "records",
        "run_sha256",
        "selection_scope",
        "source_absent_records",
        "source_available_records",
        "source_manifest_bridge_run_version",
        "source_revision",
        "verified_source_artifacts",
    }
)
_SOURCE_BRIDGE_CONTRACTS: dict[str, dict[str, Any]] = {
    "artifacts/diagnostics/native-source/antiox-eligible-source-bridge.json": {
        "corpus_kind": "antiox",
        "question_id": "antiox-training",
        "dataset_version": (
            "local-antiox-archive@papers-a5ac0466860a+source-lines-5e78d47a10cb"
        ),
        "source_revision": None,
        "license_status": "local_archived_inputs_redistribution_not_assessed",
        "selection_scope": "legacy_eligible",
        "records": 648,
        "native_manifest_records": 19,
        "source_available_records": 647,
        "source_absent_records": 1,
        "manifest_excluded_records": 629,
        "content_scope_counts": {"numbered_source_lines": 647, "unavailable": 1},
    },
    "artifacts/diagnostics/native-source/antiox-source-bridge.json": {
        "corpus_kind": "antiox",
        "question_id": "antiox-training",
        "dataset_version": (
            "local-antiox-archive@papers-a5ac0466860a+source-lines-5e78d47a10cb"
        ),
        "source_revision": None,
        "license_status": "local_archived_inputs_redistribution_not_assessed",
        "selection_scope": "successful_screened_in",
        "records": 648,
        "native_manifest_records": 646,
        "source_available_records": 647,
        "source_absent_records": 1,
        "manifest_excluded_records": 2,
        "content_scope_counts": {"numbered_source_lines": 647, "unavailable": 1},
    },
    "artifacts/diagnostics/native-source/metasyn-boundary-source-bridge.json": {
        "corpus_kind": "metasyn",
        "question_id": "metasyn-diagnostic",
        "dataset_version": (
            "THUIR/MetaSyn@c8fa07d89c44093d623f9a213c6bf070f40ab960"
        ),
        "source_revision": "c8fa07d89c44093d623f9a213c6bf070f40ab960",
        "license_status": "local_evaluation_only_third_party_terms_apply",
        "selection_scope": "explicit_corpus_id_subset",
        "records": 12,
        "native_manifest_records": 12,
        "source_available_records": 12,
        "source_absent_records": 0,
        "manifest_excluded_records": 0,
        "content_scope_counts": {"full_text_sections": 4, "title_abstract": 8},
    },
}
_PLANTED_SIMULATION_RUNNERS = {
    "artifacts/paper/budgeted-verification-simulation-200.json": (
        "scripts/simulate_budgeted_verification.py"
    ),
    "artifacts/paper/calibration-simulation-100.json": (
        "scripts/simulate_risk_calibration.py"
    ),
    "artifacts/paper/meta-simulation-200.json": "scripts/simulate_meta_analysis.py",
}
_PLANTED_SIMULATION_PATHS = frozenset(_PLANTED_SIMULATION_RUNNERS)
_LOCAL_SUITE_SOURCE_PATHS = frozenset(
    {
        "pyproject.toml",
        "scripts/run_local_benchmarks.py",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/calibration.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/metasyn_benchmark.py",
        "src/literature_multiverse/metasyn_retrieval.py",
        "src/literature_multiverse/metasyn_retrieval_study.py",
        "src/literature_multiverse/metasyn_screening_study.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/paths.py",
        "uv.lock",
    }
)
_METASYN_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "abstract",
        "article_id",
        "component_id",
        "corpus_id",
        "gold_matched_corpus_ids",
        "ordered_corpus_ids",
        "outcome",
        "population",
        "question_id",
        "research_question",
        "retrieved_corpus_ids",
        "review_id",
        "title",
    }
)
_METASYN_IDENTIFIER_RE = re.compile(r"metasyn-(?:review|component)-[0-9a-f]+", re.I)
_METASYN_RETRIEVAL_TOP_LEVEL_KEYS = frozenset(
    {
        "access_boundary",
        "candidate_configs",
        "contains_article_text",
        "contains_per_question_or_per_article_identifiers",
        "contains_question_text",
        "dataset_boundary",
        "development_results",
        "lineage",
        "metasyn_retrieval_public_summary_version",
        "network_calls",
        "provider_calls",
        "public_summary_payload_sha256",
        "runtime_versions",
        "selected_calibration_result",
        "selection_protocol",
        "source_review_exclusions",
        "status",
        "task",
        "timestamps_in_scientific_payload",
    }
)
_METASYN_SCREENING_TOP_LEVEL_KEYS = frozenset(
    {
        "calibration",
        "data_scope",
        "development_component_disjoint_cross_validation",
        "interpretation_limits",
        "lineage",
        "metasyn_screening_public_summary_version",
        "network_calls",
        "protocol",
        "provider_calls",
        "public_redaction",
        "public_summary_payload_sha256",
        "status",
        "task",
    }
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except _DuplicateJSONKeyError as exc:
        raise PublicArtifactValidationError(
            f"public_artifact_duplicate_json_key:{path}:{exc}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicArtifactValidationError(f"public_artifact_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise PublicArtifactValidationError(f"public_artifact_root_not_object:{path}")
    return value


def _resolve_registered_artifact_path(*, root: Path, artifact_path: str) -> Path:
    relative = PurePosixPath(artifact_path)
    if (
        not artifact_path
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in artifact_path
        or relative.as_posix() != artifact_path
    ):
        raise PublicArtifactValidationError(
            f"public_artifact_registry_path_unsafe:{artifact_path}"
        )
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = (resolved_root / artifact_path).resolve(strict=True)
    except OSError as exc:
        raise PublicArtifactValidationError(
            f"public_artifact_registry_path_unreadable:{artifact_path}"
        ) from exc
    if not resolved_path.is_relative_to(resolved_root) or not resolved_path.is_file():
        raise PublicArtifactValidationError(
            f"public_artifact_registry_path_outside_repository:{artifact_path}"
        )
    return resolved_path


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, artifact_path: str
) -> None:
    observed = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if not isinstance(observed, str) or observed != hash_canonical(payload):
        raise PublicArtifactValidationError(
            f"public_artifact_self_hash_mismatch:{artifact_path}:{field}"
        )


def _source_maps(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            location = f"{prefix}.{key}"
            if key in _SOURCE_MAP_KEYS:
                found.append((location, item))
                continue
            found.extend(_source_maps(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_source_maps(item, f"{prefix}[{index}]"))
    return found


def _validate_current_source_maps(
    value: Mapping[str, Any], *, root: Path, artifact_path: str
) -> int:
    current, historical = _validate_bound_source_maps(
        value,
        root=root,
        artifact_path=artifact_path,
        bindings=None,
    )
    if historical:
        raise PublicArtifactValidationError(
            f"public_artifact_unexpected_historical_source_map:{artifact_path}"
        )
    return current


def _source_map_at_location(
    value: Mapping[str, Any], *, location: str, artifact_path: str
) -> Mapping[str, Any]:
    if not location.startswith("$.") or "[" in location or "]" in location:
        raise PublicArtifactValidationError(
            f"public_artifact_source_binding_location_invalid:{artifact_path}:{location}"
        )
    current: Any = value
    for part in location[2:].split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            raise PublicArtifactValidationError(
                f"public_artifact_source_binding_missing:{artifact_path}:{location}"
            )
        current = current[part]
    if not isinstance(current, Mapping):
        raise PublicArtifactValidationError(
            f"public_artifact_source_binding_not_map:{artifact_path}:{location}"
        )
    return current


def _validate_bound_source_maps(
    value: Mapping[str, Any],
    *,
    root: Path,
    artifact_path: str,
    bindings: tuple[PublicSourceMapBinding, ...] | None,
) -> tuple[int, int]:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise PublicArtifactValidationError(
            f"public_artifact_repository_root_unreadable:{artifact_path}"
        ) from exc
    detected_items = _source_maps(value)
    detected_locations = [location for location, _item in detected_items]
    if len(detected_locations) != len(set(detected_locations)):
        raise PublicArtifactValidationError(
            f"public_artifact_source_map_location_collision:{artifact_path}"
        )
    detected = dict(detected_items)
    if bindings is None:
        effective = tuple(
            PublicSourceMapBinding(location, "current_checkout_replay")
            for location in sorted(detected)
        )
    else:
        locations = [binding.location for binding in bindings]
        if len(locations) != len(set(locations)):
            raise PublicArtifactValidationError(
                f"public_artifact_source_binding_duplicate:{artifact_path}"
            )
        non_source_bindings = sorted(set(locations) - set(detected))
        if non_source_bindings:
            raise PublicArtifactValidationError(
                f"public_artifact_source_binding_not_reserved_map:{artifact_path}:"
                f"{non_source_bindings[0]}"
            )
        unclassified = sorted(set(detected) - set(locations))
        if unclassified:
            raise PublicArtifactValidationError(
                f"public_artifact_source_map_unclassified:{artifact_path}:"
                f"{unclassified[0]}"
            )
        effective = bindings

    current_count = 0
    historical_count = 0
    for binding in effective:
        source_map = _source_map_at_location(
            value,
            location=binding.location,
            artifact_path=artifact_path,
        )
        if not source_map:
            raise PublicArtifactValidationError(
                f"public_artifact_source_map_empty:{artifact_path}:{binding.location}"
            )
        for relative, expected in source_map.items():
            if (
                not isinstance(relative, str)
                or not isinstance(expected, str)
                or _SHA256_RE.fullmatch(expected) is None
            ):
                raise PublicArtifactValidationError(
                    f"public_artifact_source_map_invalid:{artifact_path}:"
                    f"{binding.location}"
                )
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in relative
                or path.as_posix() != relative
            ):
                raise PublicArtifactValidationError(
                    f"public_artifact_source_path_unsafe:{artifact_path}:{relative}"
                )
            if binding.role == "current_checkout_replay":
                source = root / relative
                try:
                    resolved_source = source.resolve(strict=True)
                except OSError as exc:
                    raise PublicArtifactValidationError(
                        f"public_artifact_source_lineage_stale:"
                        f"{artifact_path}:{relative}"
                    ) from exc
                if not resolved_source.is_relative_to(resolved_root):
                    raise PublicArtifactValidationError(
                        f"public_artifact_source_path_outside_repository:"
                        f"{artifact_path}:{relative}"
                    )
                if not resolved_source.is_file() or sha256_file(resolved_source) != expected:
                    raise PublicArtifactValidationError(
                        f"public_artifact_source_lineage_stale:{artifact_path}:{relative}"
                    )
        if binding.role == "current_checkout_replay":
            if binding.expected_bundle_sha256 is not None:
                raise PublicArtifactValidationError(
                    f"public_artifact_current_source_binding_has_historical_pin:"
                    f"{artifact_path}:{binding.location}"
                )
            current_count += 1
        elif binding.role == "historical_execution":
            if (
                binding.expected_bundle_sha256 is None
                or _SHA256_RE.fullmatch(binding.expected_bundle_sha256) is None
                or hash_canonical(dict(source_map))
                != binding.expected_bundle_sha256
            ):
                raise PublicArtifactValidationError(
                    f"public_artifact_historical_source_bundle_mismatch:"
                    f"{artifact_path}:{binding.location}"
                )
            historical_count += 1
        else:
            raise PublicArtifactValidationError(
                f"public_artifact_source_binding_role_unknown:{artifact_path}:"
                f"{binding.location}"
            )
    return current_count, historical_count


def _validate_source_artifact_records(
    value: Mapping[str, Any], *, artifact_path: str
) -> list[dict[str, Any]]:
    records = value.get("verified_source_artifacts")
    if not isinstance(records, list) or not records:
        raise PublicArtifactValidationError(
            f"source_bridge_artifact_inventory_invalid:{artifact_path}"
        )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping) or set(item) != {
            "artifact_path",
            "media_type",
            "role",
            "rows",
            "sha256",
        }:
            raise PublicArtifactValidationError(
                f"source_bridge_artifact_record_invalid:{artifact_path}:{index}"
            )
        relative = item.get("artifact_path")
        role = item.get("role")
        media_type = item.get("media_type")
        rows = item.get("rows")
        digest = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(role, str)
            or not role
            or not isinstance(media_type, str)
            or not media_type
            or (rows is not None and (isinstance(rows, bool) or not isinstance(rows, int)))
            or (isinstance(rows, int) and rows < 0)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise PublicArtifactValidationError(
                f"source_bridge_artifact_record_invalid:{artifact_path}:{index}"
            )
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise PublicArtifactValidationError(
                f"source_bridge_artifact_path_unsafe:{artifact_path}:{index}"
            )
        normalized.append(dict(item))
    keys = [(str(item["role"]), str(item["artifact_path"])) for item in normalized]
    if keys != sorted(set(keys)):
        raise PublicArtifactValidationError(
            f"source_bridge_artifact_inventory_unsorted_or_duplicate:{artifact_path}"
        )
    return normalized


def _expected_antiox_source_artifacts(*, root: Path) -> list[dict[str, Any]]:
    records = (
        (
            "publication_metadata",
            "data/processed/antiox-training/papers.parquet",
            "application/vnd.apache.parquet",
            648,
        ),
        (
            "source_payload",
            "data/raw/map/antiox-training/source_lines.json",
            "application/json",
            647,
        ),
    )
    expected: list[dict[str, Any]] = []
    for role, relative, media_type, rows in records:
        source = root / relative
        if not source.is_file():
            raise PublicArtifactValidationError(
                f"source_bridge_tracked_artifact_missing:{relative}"
            )
        expected.append(
            {
                "artifact_path": relative,
                "media_type": media_type,
                "role": role,
                "rows": rows,
                "sha256": sha256_file(source),
            }
        )
    return expected


def _expected_metasyn_source_artifacts(*, root: Path) -> list[dict[str, Any]]:
    manifest_path = root / _METASYN_CORPUS_MANIFEST
    manifest = _load_json_object(manifest_path)
    license_notice = manifest.get("license_notice")
    shards = manifest.get("shards")
    if (
        manifest.get("corpus_manifest_version") != "1"
        or manifest.get("source_revision")
        != "c8fa07d89c44093d623f9a213c6bf070f40ab960"
        or manifest.get("local_root") != "data/cache/metasyn/corpus"
        or not isinstance(license_notice, Mapping)
        or not isinstance(shards, list)
        or len(shards) != 6
    ):
        raise PublicArtifactValidationError("source_bridge_metasyn_manifest_invalid")
    total_rows = 0
    expected = [
        {
            "artifact_path": _METASYN_CORPUS_MANIFEST,
            "media_type": "application/json",
            "role": "corpus_manifest",
            "rows": None,
            "sha256": sha256_file(manifest_path),
        },
        {
            "artifact_path": str(license_notice.get("path")),
            "media_type": "text/plain",
            "role": "license_notice",
            "rows": None,
            "sha256": str(license_notice.get("sha256")),
        },
    ]
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise PublicArtifactValidationError(
                f"source_bridge_metasyn_shard_invalid:{index}"
            )
        relative = shard.get("path")
        rows = shard.get("rows")
        digest = shard.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 1
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise PublicArtifactValidationError(
                f"source_bridge_metasyn_shard_invalid:{index}"
            )
        total_rows += rows
        expected.append(
            {
                "artifact_path": f"{manifest['local_root']}/{relative}",
                "media_type": "application/vnd.apache.parquet",
                "role": "source_shard",
                "rows": rows,
                "sha256": digest,
            }
        )
    if total_rows != manifest.get("total_rows") or total_rows != 140_585:
        raise PublicArtifactValidationError("source_bridge_metasyn_row_total_invalid")
    tracked_license = root / "artifacts/paper/metasyn-benchmark/METASYN_LICENSE.txt"
    if (
        license_notice.get("path") != "data/cache/metasyn/LICENSE"
        or license_notice.get("status")
        != "local_evaluation_only_third_party_terms_apply"
        or _SHA256_RE.fullmatch(str(license_notice.get("sha256"))) is None
        or not tracked_license.is_file()
        or sha256_file(tracked_license) != license_notice.get("sha256")
    ):
        raise PublicArtifactValidationError("source_bridge_metasyn_license_invalid")
    return expected


def _validate_eligible_bridge_consumers(
    value: Mapping[str, Any], *, root: Path
) -> None:
    config = _load_json_object(root / _NATIVE_CONFIG)
    if (
        config.get("config_version") != "native-antiox-ollama-config-v1"
        or config.get("question_id") != value.get("question_id")
        or config.get("bridge_run_path")
        != "data/cache/native-source-v1/antiox-eligible/source_manifest_bridge_run.json"
        or config.get("bridge_run_sha256") != value.get("run_sha256")
    ):
        raise PublicArtifactValidationError("source_bridge_native_config_mismatch")
    summary = _load_json_object(root / _NATIVE_PUBLIC_SUMMARY)
    _validate_self_hash(
        summary,
        field="public_summary_sha256",
        artifact_path=_NATIVE_PUBLIC_SUMMARY,
    )
    expected = {
        "source_bridge_run_sha256": value.get("run_sha256"),
        "source_manifest_content_sha256": value.get(
            "native_source_manifest_content_sha256"
        ),
        "source_manifest_file_sha256": value.get("native_source_manifest_file_sha256"),
        "population_count": value.get("native_manifest_records"),
        "selection_scope": value.get("selection_scope"),
    }
    if any(summary.get(field) != expected_value for field, expected_value in expected.items()):
        raise PublicArtifactValidationError("source_bridge_native_summary_mismatch")


def _validate_source_bridge(
    value: Mapping[str, Any], *, root: Path, artifact_path: str
) -> None:
    contract = _SOURCE_BRIDGE_CONTRACTS.get(artifact_path)
    if contract is None:
        raise PublicArtifactValidationError(f"source_bridge_path_unregistered:{artifact_path}")
    _validate_self_hash(value, field="run_sha256", artifact_path=artifact_path)
    if (
        set(value) != _SOURCE_BRIDGE_TOP_LEVEL_KEYS
        or value.get("source_manifest_bridge_run_version") != "2"
        or value.get("diagnostic_only") is not True
        or value.get("labels_previously_opened") is not True
        or value.get("pristine_final_holdout_eligible") is not False
        or any(value.get(field) != expected for field, expected in contract.items())
    ):
        raise PublicArtifactValidationError(f"source_bridge_scope_invalid:{artifact_path}")
    hash_fields = (
        "diagnostic_source_ledger_content_sha256",
        "diagnostic_source_ledger_file_sha256",
        "native_source_manifest_content_sha256",
        "native_source_manifest_file_sha256",
        "run_sha256",
    )
    if any(
        not isinstance(value.get(field), str)
        or _SHA256_RE.fullmatch(str(value[field])) is None
        for field in hash_fields
    ):
        raise PublicArtifactValidationError(f"source_bridge_hash_invalid:{artifact_path}")
    records = value["records"]
    available = value["source_available_records"]
    absent = value["source_absent_records"]
    native = value["native_manifest_records"]
    excluded = value["manifest_excluded_records"]
    scopes = value["content_scope_counts"]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (
            records,
            available,
            absent,
            native,
            excluded,
        ))
        or records != available + absent
        or records != native + excluded
        or not isinstance(scopes, Mapping)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in scopes.items()
        )
        or sum(scopes.values()) != records
    ):
        raise PublicArtifactValidationError(f"source_bridge_count_invalid:{artifact_path}")
    artifacts = _validate_source_artifact_records(value, artifact_path=artifact_path)
    expected_artifacts = (
        _expected_antiox_source_artifacts(root=root)
        if value["corpus_kind"] == "antiox"
        else _expected_metasyn_source_artifacts(root=root)
    )
    if artifacts != expected_artifacts:
        raise PublicArtifactValidationError(
            f"source_bridge_artifact_binding_mismatch:{artifact_path}"
        )
    if artifact_path.endswith("antiox-eligible-source-bridge.json"):
        _validate_eligible_bridge_consumers(value, root=root)


@lru_cache(maxsize=3)
def _replay_planted_simulation(artifact_path: str) -> dict[str, Any]:
    if artifact_path not in _PLANTED_SIMULATION_PATHS:
        raise PublicArtifactValidationError(
            f"planted_simulation_path_unregistered:{artifact_path}"
        )
    implementation_root = Path(__file__).resolve().parents[2]
    runner_path = implementation_root / _PLANTED_SIMULATION_RUNNERS[artifact_path]
    try:
        namespace = runpy.run_path(
            runner_path.as_posix(),
            run_name=f"_literature_multiverse_public_replay_{runner_path.stem}",
        )
    except (OSError, ImportError) as exc:
        raise PublicArtifactValidationError(
            f"planted_simulation_runner_unavailable:{artifact_path}"
        ) from exc
    runner = namespace.get("main")
    if not callable(runner):
        raise PublicArtifactValidationError(
            f"planted_simulation_runner_missing_main:{artifact_path}"
        )
    with TemporaryDirectory(prefix="lm-public-simulation-replay-") as temporary:
        output = Path(temporary) / "artifact.json"
        with redirect_stdout(StringIO()):
            status = runner(["--output", output.as_posix()])
        if status != 0:
            raise PublicArtifactValidationError(
                f"planted_simulation_replay_failed:{artifact_path}:{status}"
            )
        return _load_json_object(output)


def _validate_planted_simulation(
    value: Mapping[str, Any], *, root: Path, artifact_path: str
) -> None:
    implementation_root = Path(__file__).resolve().parents[2]
    if root != implementation_root:
        raise PublicArtifactValidationError("planted_simulation_repository_root_mismatch")
    replayed = _replay_planted_simulation(artifact_path)
    if dict(value) != replayed:
        raise PublicArtifactValidationError(
            f"planted_simulation_replay_mismatch:{artifact_path}"
        )


def _validate_evidence_inference_local_ollama(
    value: Mapping[str, Any], *, root: Path
) -> None:
    validated = validate_augmented_public_summary(
        value,
        repository_root=root,
        require_current_sources=True,
    )
    supplement = validated.get("reproducibility_supplement")
    if not isinstance(supplement, Mapping):
        raise PublicArtifactValidationError(
            "evidence_inference_ollama_reproducibility_supplement_missing"
        )
    historical_base = dict(validated)
    historical_base.pop("public_summary_sha256", None)
    historical_base.pop("reproducibility_supplement", None)
    if (
        hash_canonical(historical_base)
        != "7eb48a1b275185cb7d3a54577def0f0cf6320ee108962fe583e421ca1e3b19ea"
        or supplement.get("base_public_summary_sha256")
        != "7eb48a1b275185cb7d3a54577def0f0cf6320ee108962fe583e421ca1e3b19ea"
        or validated.get("execution_fingerprint_sha256")
        != "001547bd324518e883b67851bdee909f3a2a437ea8c17b185fdc4fca59bbfec3"
        or validated.get("full_private_report_sha256")
        != "c0bf040fef322b7ce4c449bdcef85c5e35b7808bd6c5abeb869f3cb499d602ad"
        or validated.get("prediction_ledger_sha256")
        != "ae054dab1c75c5689ee29b5e1768f03fa391ea3c963745859a4c31cb5dfc63d0"
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_ollama_historical_base_mismatch"
        )


def _validate_evidence_inference_ollama_gepa(value: Mapping[str, Any]) -> None:
    validated = validate_gepa_public_summary(value)
    lineage = validated.get("lineage")
    if (
        validated.get("public_summary_sha256")
        != "1039156083798863e85761ecf94b76578c74066af2ef7b7691fd4d724f4967ce"
        or not isinstance(lineage, Mapping)
        or lineage.get("plan_sha256")
        != "8a91fcf9ba93bb5512299557eb514fcf814eee6fcacc572a2463855bd949a89a"
        or lineage.get("winner_sha256")
        != "a82b503e07bdc07e94e9fe773742739d6c0066b20c91f399d33d7f7d13a46c05"
        or lineage.get("private_paired_report_sha256")
        != "30aa756d15a0104f869647a3d3b2b44f6e4ce7b3c7c087e45f2e5ddd2bc872e0"
        or not isinstance(lineage.get("source_code_sha256s"), Mapping)
        or hash_canonical(dict(lineage["source_code_sha256s"]))
        != "ffce448ed9d13148a909e5d9e3c6e3beec7cdc2cd7d3cb93eef30b31eb23ead3"
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_gepa_historical_lineage_mismatch"
        )


def _validate_evidence_inference_item_risk(
    value: Mapping[str, Any], *, root: Path
) -> None:
    """Validate the public aggregate against its frozen historical pipeline manifest and
    the public GEPA lineage. Before 2026-08-29 this recomputed the diagnostic fingerprint
    from current bytes and required equality; it now consults the current tree only for
    the closure definition."""

    try:
        validated = validate_ei_item_risk_summary(value)
        config_path = root / _EVIDENCE_INFERENCE_ITEM_RISK_CONFIG
        config = load_ei_item_risk_config(config_path)
        gepa_summary_path = root / config.gepa_public_summary_path
        gepa_summary = validate_gepa_public_summary(_load_json_object(gepa_summary_path))
        fingerprint, recomputed_prediction_source = (
            compute_ei_item_risk_pipeline_fingerprint(
                repository_root=root,
                config=config,
                gepa_public_summary=gepa_summary,
            )
        )
    except (OSError, ValueError) as exc:
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_public_replay_invalid"
        ) from exc
    try:
        historical = PipelineFingerprint.model_validate(
            _load_json_object(root / _EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_FINGERPRINT)
        )
    except (OSError, ValueError) as exc:
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_historical_fingerprint_invalid"
        ) from exc
    current_component = fingerprint.components[0]
    if (
        len(historical.components) != 1
        or historical.components[0].component_id != current_component.component_id
        or historical.components[0].component_version != current_component.component_version
        or [record.path for record in historical.components[0].files]
        != [record.path for record in current_component.files]
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_historical_closure_definition_mismatch"
        )
    lineage = validated.get("lineage")
    population = validated.get("population")
    calibration = validated.get("calibration")
    prediction_source = validated.get("prediction_source")
    if not all(
        isinstance(item, Mapping)
        for item in (lineage, population, calibration, prediction_source)
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_lineage_missing"
        )
    assert isinstance(lineage, Mapping)
    assert isinstance(population, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(prediction_source, Mapping)
    bounds = calibration.get("bounds")
    if (
        any(
            population.get(key) != expected
            for key, expected in _EVIDENCE_INFERENCE_ITEM_RISK_REGISTERED_POPULATION.items()
        )
        or not isinstance(bounds, list)
        or [
            (
                item.get("bin_id"),
                item.get("cell_calibration_units"),
                item.get("cell_observed_errors"),
            )
            for item in bounds
            if isinstance(item, Mapping)
        ]
        != _EVIDENCE_INFERENCE_ITEM_RISK_REGISTERED_CELLS
        or len(bounds) != len(_EVIDENCE_INFERENCE_ITEM_RISK_REGISTERED_CELLS)
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_registered_result_mismatch"
        )
    if (
        validated.get("public_summary_sha256")
        != "d1c7191d68cc7fb40ed92bddd25bfb9a1d8f625306d10850afde9a96cd9aebe1"
        or lineage.get("config_file_sha256") != sha256_file(config_path)
        or lineage.get("config_sha256") != config.config_sha256
        or lineage.get("diagnostic_pipeline_sha256") != historical.pipeline_sha256
        or historical.pipeline_sha256
        != _EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_PIPELINE_SHA256
        or lineage.get("prediction_source_lineage_sha256")
        != recomputed_prediction_source["prediction_source_lineage_sha256"]
        or validated.get("prediction_source") != recomputed_prediction_source
        or lineage.get("gepa_public_summary_sha256")
        != gepa_summary.get("public_summary_sha256")
        or prediction_source.get("prediction_source_lineage_sha256")
        != "d4fa141b5b767a78943460a9e1eb2e41854d2667f518e9dd542e761d1bbc6808"
        or prediction_source.get("historical_source_code_sha256s")
        != gepa_summary.get("lineage", {}).get("source_code_sha256s")
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_current_lineage_mismatch"
        )


def _validate_metasyn_public_redaction(
    value: Any,
    *,
    root: Path,
    artifact_kind: str,
) -> None:
    """Reject row identifiers, scientific text, host paths, and remote payloads."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _METASYN_PUBLIC_FORBIDDEN_KEYS:
                raise PublicArtifactValidationError(
                    f"{artifact_kind}_public_summary_forbidden_key:{key}"
                )
            _validate_metasyn_public_redaction(
                item,
                root=root,
                artifact_kind=artifact_kind,
            )
        return
    if isinstance(value, list):
        for item in value:
            _validate_metasyn_public_redaction(
                item,
                root=root,
                artifact_kind=artifact_kind,
            )
        return
    if not isinstance(value, str):
        return
    if (
        root.as_posix() in value
        or value.startswith(("/Users/", "/home/", "file://"))
        or _METASYN_IDENTIFIER_RE.search(value)
    ):
        raise PublicArtifactValidationError(
            f"{artifact_kind}_public_summary_privacy_violation"
        )


def _validate_metasyn_input_lineage(
    value: Mapping[str, Any], *, root: Path, artifact_kind: str
) -> None:
    lineage = value.get("lineage")
    if not isinstance(lineage, Mapping):
        raise PublicArtifactValidationError(f"{artifact_kind}_lineage_missing")
    expected = {
        "benchmark_manifest_sha256": sha256_file(root / _METASYN_BENCHMARK_MANIFEST),
        "corpus_manifest_sha256": sha256_file(root / _METASYN_CORPUS_MANIFEST),
    }
    for field, digest in expected.items():
        if lineage.get(field) != digest:
            raise PublicArtifactValidationError(
                f"{artifact_kind}_tracked_input_hash_mismatch:{field}"
            )
    benchmark = _load_json_object(root / _METASYN_BENCHMARK_MANIFEST)
    corpus = _load_json_object(root / _METASYN_CORPUS_MANIFEST)
    expected_shards = {
        str(record["path"]): str(record["sha256"])
        for record in corpus.get("shards", [])
        if isinstance(record, Mapping)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("sha256"), str)
    }
    if not expected_shards or sum(
        isinstance(record, Mapping) for record in corpus.get("shards", [])
    ) != len(expected_shards):
        raise PublicArtifactValidationError("metasyn_corpus_manifest_shards_invalid")
    if "corpus_shard_sha256s" in lineage and lineage.get(
        "corpus_shard_sha256s"
    ) != expected_shards:
        raise PublicArtifactValidationError(
            f"{artifact_kind}_declared_corpus_shards_mismatch"
        )
    evaluator = benchmark.get("evaluator_labels")
    source_train = benchmark.get("source_train")
    if not isinstance(evaluator, Mapping) or not isinstance(source_train, Mapping):
        raise PublicArtifactValidationError("metasyn_benchmark_manifest_lineage_invalid")
    if "evaluator_labels_sha256" in lineage and lineage.get(
        "evaluator_labels_sha256"
    ) != evaluator.get("sha256"):
        raise PublicArtifactValidationError(
            f"{artifact_kind}_declared_evaluator_hash_mismatch"
        )
    if "source_train_parquet_sha256" in lineage and lineage.get(
        "source_train_parquet_sha256"
    ) != source_train.get("sha256"):
        raise PublicArtifactValidationError(
            f"{artifact_kind}_declared_source_train_hash_mismatch"
        )
    source_reviews = lineage.get("source_review_sha256s")
    if source_reviews is not None and source_reviews != {
        str(source_train.get("filename")): str(source_train.get("sha256"))
    }:
        raise PublicArtifactValidationError(
            f"{artifact_kind}_declared_source_reviews_mismatch"
        )


def _finite_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_retrieval_aggregate_block(
    block: Any,
    *,
    expected_questions: int,
    expected_candidates: set[str],
    artifact_kind: str,
) -> None:
    if not isinstance(block, Mapping):
        raise PublicArtifactValidationError(f"{artifact_kind}_result_block_invalid")
    candidates = block.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != expected_candidates:
        raise PublicArtifactValidationError(f"{artifact_kind}_candidate_inventory_invalid")
    for candidate_id, row in candidates.items():
        if not isinstance(row, Mapping):
            raise PublicArtifactValidationError(
                f"{artifact_kind}_candidate_result_invalid:{candidate_id}"
            )
        total = row.get("matched_references")
        retrieved = row.get("matched_references_retrieved")
        if (
            row.get("questions") != expected_questions
            or row.get("retrieval_depth") != 200
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or not isinstance(retrieved, int)
            or isinstance(retrieved, bool)
            or not 0 <= retrieved <= total
            or not _finite_probability(row.get("macro_recall_at_200"))
            or not _finite_probability(row.get("micro_recall_at_200"))
            or not math.isclose(
                float(row["micro_recall_at_200"]),
                retrieved / total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not isinstance(row.get("questions_with_full_recall"), int)
            or not 0 <= row["questions_with_full_recall"] <= expected_questions
            or not isinstance(row.get("questions_with_zero_recall"), int)
            or not 0 <= row["questions_with_zero_recall"] <= expected_questions
        ):
            raise PublicArtifactValidationError(
                f"{artifact_kind}_candidate_arithmetic_invalid:{candidate_id}"
            )
        intervals = row.get("cluster_bootstrap_interval_95")
        if not isinstance(intervals, Mapping):
            raise PublicArtifactValidationError(
                f"{artifact_kind}_candidate_interval_invalid:{candidate_id}"
            )
        for metric in ("macro_recall_at_200", "micro_recall_at_200"):
            interval = intervals.get(metric)
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(_finite_probability(item) for item in interval)
                or interval[0] > row[metric]
                or row[metric] > interval[1]
            ):
                raise PublicArtifactValidationError(
                    f"{artifact_kind}_candidate_interval_invalid:{candidate_id}:{metric}"
                )
    contrasts = block.get("paired_cluster_bootstrap_differences")
    if not isinstance(contrasts, Mapping):
        raise PublicArtifactValidationError(f"{artifact_kind}_contrasts_invalid")
    for contrast_id, contrast in contrasts.items():
        if not isinstance(contrast_id, str) or "_minus_" not in contrast_id:
            raise PublicArtifactValidationError(f"{artifact_kind}_contrast_id_invalid")
        left, right = contrast_id.split("_minus_", maxsplit=1)
        if left not in candidates or right not in candidates or not isinstance(
            contrast, Mapping
        ):
            raise PublicArtifactValidationError(
                f"{artifact_kind}_contrast_membership_invalid:{contrast_id}"
            )
        for metric in ("macro_recall_at_200", "micro_recall_at_200"):
            field = f"{metric}_difference"
            expected = float(candidates[left][metric]) - float(candidates[right][metric])
            observed = contrast.get(field)
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not math.isclose(
                    float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise PublicArtifactValidationError(
                    f"{artifact_kind}_contrast_arithmetic_invalid:{contrast_id}:{field}"
                )


_SCREENING_RATE_FIELDS = (
    "full_inclusion_rate",
    "micro_absolute_recall",
    "micro_conditional_survival",
    "question_macro_absolute_recall",
    "question_macro_conditional_survival",
    "questions_with_zero_retained_rate",
)


def _validate_screening_aggregate(value: Mapping[str, Any]) -> None:
    protocol = value["protocol"]
    development = value.get("development_component_disjoint_cross_validation")
    calibration = value.get("calibration")
    questions = value.get("data_scope", {}).get("calibration_questions")
    if (
        not isinstance(development, Mapping)
        or not isinstance(calibration, Mapping)
        or not isinstance(questions, int)
        or isinstance(questions, bool)
        or questions <= 0
    ):
        raise PublicArtifactValidationError("metasyn_screening_aggregate_shape_invalid")
    development_candidates = development.get("candidates")
    expected_candidate_ids = set(protocol["candidate_family_frozen"])
    if (
        not isinstance(development_candidates, Mapping)
        or set(development_candidates) != expected_candidate_ids
    ):
        raise PublicArtifactValidationError(
            "metasyn_screening_development_candidate_inventory_invalid"
        )
    selection_rows: list[tuple[float, str]] = []
    for candidate_id, candidate in development_candidates.items():
        if not isinstance(candidate, Mapping) or not isinstance(
            candidate.get("depths"), Mapping
        ):
            raise PublicArtifactValidationError(
                f"metasyn_screening_development_candidate_invalid:{candidate_id}"
            )
        depth_rows = candidate["depths"]
        try:
            expected_score = math.fsum(
                float(depth_rows[str(depth)]["question_macro_absolute_recall"])
                for depth in (10, 20, 50, 100)
            ) / 4.0
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicArtifactValidationError(
                f"metasyn_screening_development_score_invalid:{candidate_id}"
            ) from exc
        score = candidate.get("selection_score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isclose(
                float(score), expected_score, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise PublicArtifactValidationError(
                f"metasyn_screening_development_score_invalid:{candidate_id}"
            )
        selection_rows.append((float(score), str(candidate_id)))
    expected_winner = sorted(selection_rows, key=lambda item: (-item[0], item[1]))[0][1]
    if protocol.get("selected_candidate") != expected_winner:
        raise PublicArtifactValidationError("metasyn_screening_selected_candidate_invalid")

    depths = calibration.get("evaluation_depths")
    point_results = calibration.get("point_results")
    if (
        depths != [10, 20, 50, 100, 200]
        or not isinstance(point_results, Mapping)
        or set(point_results) != {"rrf_baseline", "selected"}
    ):
        raise PublicArtifactValidationError("metasyn_screening_point_results_invalid")
    expected_depth_keys = {str(depth) for depth in depths}
    for arm_id, arm in point_results.items():
        if not isinstance(arm, Mapping) or set(arm) != expected_depth_keys:
            raise PublicArtifactValidationError(
                f"metasyn_screening_depth_inventory_invalid:{arm_id}"
            )
        for depth in depths:
            row = arm[str(depth)]
            if not isinstance(row, Mapping):
                raise PublicArtifactValidationError(
                    f"metasyn_screening_point_row_invalid:{arm_id}:{depth}"
                )
            matched_total = row.get("matched_references_total")
            retrievable_total = row.get("conditional_retrievable_references_total")
            retained = row.get("matched_references_retained")
            zero_retrievable = row.get(
                "questions_with_zero_retrievable_matched_references"
            )
            if (
                row.get("documents_screened_per_review") != depth
                or row.get("total_document_review_pairs_screened") != questions * depth
                or not isinstance(matched_total, int)
                or matched_total <= 0
                or not isinstance(retrievable_total, int)
                or retrievable_total <= 0
                or retrievable_total > matched_total
                or not isinstance(retained, int)
                or not 0 <= retained <= retrievable_total
                or row.get("conditional_retrievable_references_retained") != retained
                or not isinstance(zero_retrievable, int)
                or not 0 <= zero_retrievable < questions
                or row.get("conditional_questions") != questions - zero_retrievable
                or not all(_finite_probability(row.get(field)) for field in _SCREENING_RATE_FIELDS)
                or not math.isclose(
                    float(row["micro_absolute_recall"]),
                    retained / matched_total,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["micro_conditional_survival"]),
                    retained / retrievable_total,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or row.get("questions_with_full_inclusion", -1) not in range(questions + 1)
                or row.get("questions_with_zero_retained", -1) not in range(questions + 1)
                or not math.isclose(
                    float(row["full_inclusion_rate"]),
                    row["questions_with_full_inclusion"] / questions,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["questions_with_zero_retained_rate"]),
                    row["questions_with_zero_retained"] / questions,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise PublicArtifactValidationError(
                    f"metasyn_screening_point_arithmetic_invalid:{arm_id}:{depth}"
                )
    deltas = calibration.get("selected_minus_rrf_paired_deltas")
    if not isinstance(deltas, Mapping) or set(deltas) != expected_depth_keys:
        raise PublicArtifactValidationError("metasyn_screening_delta_inventory_invalid")
    for depth in depths:
        selected = point_results["selected"][str(depth)]
        baseline = point_results["rrf_baseline"][str(depth)]
        delta = deltas[str(depth)]
        if not isinstance(delta, Mapping) or set(delta) != set(_SCREENING_RATE_FIELDS):
            raise PublicArtifactValidationError(
                f"metasyn_screening_delta_shape_invalid:{depth}"
            )
        for field in _SCREENING_RATE_FIELDS:
            expected = float(selected[field]) - float(baseline[field])
            observed = delta[field]
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not math.isclose(
                    float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise PublicArtifactValidationError(
                    f"metasyn_screening_delta_arithmetic_invalid:{depth}:{field}"
                )


def _validate_metasyn_retrieval(
    value: Mapping[str, Any], *, root: Path
) -> None:
    _validate_metasyn_public_redaction(
        value,
        root=root,
        artifact_kind="metasyn_retrieval",
    )
    protocol = value.get("selection_protocol")
    access = value.get("access_boundary")
    if (
        set(value) != _METASYN_RETRIEVAL_TOP_LEVEL_KEYS
        or value.get("metasyn_retrieval_public_summary_version") != "1"
        or value.get("status") != "complete_retrospective_nonpristine"
        or value.get("network_calls") != 0
        or value.get("provider_calls") != 0
        or value.get("contains_question_text") is not False
        or value.get("contains_article_text") is not False
        or value.get("contains_per_question_or_per_article_identifiers") is not False
        or value.get("timestamps_in_scientific_payload") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("official_test_evaluated") is not False
        or not isinstance(access, Mapping)
        or access.get("official_test_gold_not_scored") is not True
        or access.get("pristine_final_holdout_eligible") is not False
    ):
        raise PublicArtifactValidationError("metasyn_retrieval_public_contract_invalid")
    _validate_metasyn_input_lineage(value, root=root, artifact_kind="metasyn_retrieval")
    manifest = _load_json_object(root / _METASYN_BENCHMARK_MANIFEST)
    boundary = value["dataset_boundary"]
    configs = value.get("candidate_configs")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("development_reviews") != manifest["development"]["rows"]
        or boundary.get("calibration_reviews") != manifest["calibration"]["rows"]
        or boundary.get("official_test_source_reviews_not_evaluated")
        != manifest["test"]["rows"]
        or not isinstance(configs, Mapping)
        or set(configs) != set(protocol["development_compared_candidates"])
        or any(
            not isinstance(record, Mapping)
            or record.get("config_sha256") != hash_canonical(record.get("config"))
            for record in configs.values()
        )
    ):
        raise PublicArtifactValidationError("metasyn_retrieval_aggregate_contract_invalid")
    development_candidates = set(protocol["development_compared_candidates"])
    selected = protocol.get("selected_candidate")
    if (
        not isinstance(selected, str)
        or selected not in development_candidates
        or protocol.get("calibration_scored_candidates") != [selected]
    ):
        raise PublicArtifactValidationError("metasyn_retrieval_selection_invalid")
    _validate_retrieval_aggregate_block(
        value.get("development_results"),
        expected_questions=int(boundary["development_reviews"]),
        expected_candidates=development_candidates,
        artifact_kind="metasyn_retrieval_development",
    )
    development_rows = value["development_results"]["candidates"]
    expected_selected = sorted(
        development_candidates,
        key=lambda candidate: (
            -float(development_rows[candidate]["macro_recall_at_200"]),
            candidate,
        ),
    )[0]
    if selected != expected_selected:
        raise PublicArtifactValidationError("metasyn_retrieval_selection_invalid")
    _validate_retrieval_aggregate_block(
        value.get("selected_calibration_result"),
        expected_questions=int(boundary["calibration_reviews"]),
        expected_candidates={selected},
        artifact_kind="metasyn_retrieval_calibration",
    )


def _validate_metasyn_screening(
    value: Mapping[str, Any], *, root: Path
) -> None:
    _validate_metasyn_public_redaction(
        value,
        root=root,
        artifact_kind="metasyn_screening",
    )
    protocol = value.get("protocol")
    limits = value.get("interpretation_limits")
    redaction = value.get("public_redaction")
    if (
        set(value) != _METASYN_SCREENING_TOP_LEVEL_KEYS
        or value.get("metasyn_screening_public_summary_version") != "1"
        or value.get("status") != "complete_retrospective_nonpristine"
        or not isinstance(protocol, Mapping)
        or protocol.get("official_test_inputs_opened_by_this_study") is not False
        or protocol.get("official_test_labels_opened_by_this_study") is not False
        or protocol.get("official_test_evaluated") is not False
        or not isinstance(limits, Mapping)
        or limits.get("official_test_never_opened_or_scored_by_this_study") is not True
        or limits.get("pristine_holdout_eligible") is not False
        or redaction
        != {
            "contains_question_or_component_identifiers": False,
            "contains_article_identifiers": False,
            "contains_titles_abstracts_or_protocol_text": False,
            "contains_labels_or_per_question_results": False,
            "contains_absolute_paths": False,
        }
    ):
        raise PublicArtifactValidationError("metasyn_screening_public_contract_invalid")
    for field in ("network_calls", "provider_calls"):
        if value.get(field) != 0:
            raise PublicArtifactValidationError(
                f"metasyn_screening_network_invariant_invalid:{field}"
            )
    _validate_metasyn_input_lineage(value, root=root, artifact_kind="metasyn_screening")
    _validate_screening_aggregate(value)


def _validate_metasyn_synthesis_yield(value: Mapping[str, Any]) -> None:
    """Validate the frozen public-only yield boundary without private inputs."""

    try:
        canonical = MetaSynSynthesisYieldPublicSummaryV1.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_public_contract_invalid"
        ) from exc
    if dict(value) != canonical.model_dump(mode="json"):
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_public_contract_not_canonical"
        )

    for field, expected in _METASYN_SYNTHESIS_YIELD_REGISTERED.items():
        if field == "summary_sha256":
            continue
        if getattr(canonical, field) != expected:
            raise PublicArtifactValidationError(
                f"metasyn_synthesis_yield_registered_lineage_mismatch:{field}"
            )

    expected_zero_yield: dict[str, Any] = {
        "runtime_contract_typed_publication_count": 0,
        "diagnostic_only_typed_publication_count": 0,
        "original_source_grounding_attempt_count": 0,
        "original_source_grounding_authorized_count": 0,
        "release_grade_estimable_publication_count": 0,
        "questions_with_estimable_graph": 0,
        "graph_estimate_count": 0,
        "compatibility_group_count": 0,
        "synthesis_input_group_count": 0,
        "synthesis_attempted_group_count": 0,
        "synthesis_completed_group_count": 0,
        "questions_with_completed_synthesis": 0,
        "publication_stage_counts": {"runtime_terminal_fragment_excluded": 32},
        "synthesis_group_stage_counts": {},
        "synthesis_completion_mode_counts": {},
        "blocker_counts": {
            "no_estimable_graph": 10,
            "runtime_terminal_exclusion": 10,
        },
        "residual_conflict_counts": {},
    }
    for field, expected in expected_zero_yield.items():
        if getattr(canonical, field) != expected:
            raise PublicArtifactValidationError(
                f"metasyn_synthesis_yield_registered_zero_yield_mismatch:{field}"
            )
    if canonical.summary_sha256 != _METASYN_SYNTHESIS_YIELD_REGISTERED[
        "summary_sha256"
    ]:
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_registered_lineage_mismatch:summary_sha256"
        )


def _validate_metasyn_synthesis_yield_v2(value: Mapping[str, Any]) -> None:
    """Validate the frozen hosted aggregate without private source-bearing inputs."""

    try:
        canonical = MetaSynSynthesisYieldPublicSummaryV2.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_v2_public_contract_invalid"
        ) from exc
    if dict(value) != canonical.model_dump(mode="json"):
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_v2_public_contract_not_canonical"
        )

    for field, expected in _METASYN_SYNTHESIS_YIELD_V2_REGISTERED.items():
        if field == "summary_sha256":
            continue
        if getattr(canonical, field) != expected:
            raise PublicArtifactValidationError(
                f"metasyn_synthesis_yield_v2_registered_lineage_mismatch:{field}"
            )

    expected_zero_yield: dict[str, Any] = {
        "question_count": 10,
        "component_count": 10,
        "publication_count": 32,
        "runtime_contract_typed_publication_count": 0,
        "diagnostic_only_typed_publication_count": 0,
        "original_source_grounding_attempt_count": 0,
        "original_source_grounding_authorized_count": 0,
        "release_grade_estimable_publication_count": 0,
        "terminal_fragment_count": 32,
        "graph_construction_completed_question_count": 10,
        "questions_with_estimable_graph": 0,
        "graph_estimate_count": 0,
        "compatibility_group_count": 0,
        "synthesis_unit_authorization_receipt_count": 0,
        "authorized_synthesis_input_group_count": 0,
        "synthesis_input_group_count": 0,
        "synthesis_attempted_group_count": 0,
        "synthesis_completed_group_count": 0,
        "questions_with_completed_synthesis": 0,
        "publication_stage_counts": {"runtime_terminal_fragment_excluded": 32},
        "synthesis_group_stage_counts": {},
        "synthesis_completion_mode_counts": {},
        "blocker_counts": {
            "no_estimable_graph": 10,
            "runtime_terminal_exclusion": 10,
        },
        "residual_conflict_counts": {},
        "provider_neutral_yield_report_present": False,
        "provider_neutral_yield_report_sha256": None,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "truth_or_clinical_benefit_reported": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    for field, expected in expected_zero_yield.items():
        if getattr(canonical, field) != expected:
            raise PublicArtifactValidationError(
                f"metasyn_synthesis_yield_v2_registered_zero_yield_mismatch:{field}"
            )
    if canonical.summary_sha256 != _METASYN_SYNTHESIS_YIELD_V2_REGISTERED[
        "summary_sha256"
    ]:
        raise PublicArtifactValidationError(
            "metasyn_synthesis_yield_v2_registered_lineage_mismatch:summary_sha256"
        )


def _synthetic_public_evaluator_labels(
    *,
    manifest: Any,
    inputs_by_split: Mapping[str, list[Any]],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct sufficient labels from tracked aggregates for evaluator replay.

    Development/calibration labels are never scored in this replay. Test labels and
    their matched-set sizes come from the tracked per-review evaluation rows. Synthetic
    paper identifiers are split-disjoint and only stand in for the undisclosed IDs;
    the fixed control emits no retrieval IDs, so only their counts affect the result.
    """

    raw_rows = evaluation.get("per_review")
    if not isinstance(raw_rows, list):
        raise PublicArtifactValidationError("fixed_positive_per_review_rows_missing")
    test_rows: dict[int, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("review_id"), int):
            raise PublicArtifactValidationError("fixed_positive_per_review_row_invalid")
        review_id = int(row["review_id"])
        if review_id in test_rows:
            raise PublicArtifactValidationError("fixed_positive_per_review_id_duplicate")
        test_rows[review_id] = row

    labels: list[dict[str, Any]] = []
    split_offsets = {
        "development": 1_000_000_000,
        "calibration": 2_000_000_000,
        "test": 3_000_000_000,
    }
    for split in ("development", "calibration", "test"):
        inputs = inputs_by_split[split]
        artifact = getattr(manifest, split)
        components = artifact.component_ids
        if not components or len(components) > len(inputs):
            raise PublicArtifactValidationError(
                f"fixed_positive_component_inventory_invalid:{split}"
            )
        for index, question in enumerate(inputs):
            component = components[index] if index < len(components) else components[0]
            if split == "test":
                row = test_rows.get(question.review_id)
                if (
                    row is None
                    or row.get("question_id") != question.question_id
                    or row.get("split") != "test"
                    or row.get("gold_direction") not in {"Positive", "Negative", "Mixed", "NR"}
                    or not isinstance(row.get("gold_retrieval_count"), int)
                    or int(row["gold_retrieval_count"]) < 1
                ):
                    raise PublicArtifactValidationError(
                        f"fixed_positive_test_label_invalid:{question.review_id}"
                    )
                direction = str(row["gold_direction"])
                count = int(row["gold_retrieval_count"])
            else:
                direction = "NR"
                count = 1
            base = split_offsets[split] + index * 10_000
            label = MetaSynEvaluatorLabel(
                question_id=question.question_id,
                review_id=question.review_id,
                official_split="test" if split == "test" else "train",
                split=split,
                component_id=component,
                gold_direction=direction,
                gold_matched_corpus_ids=list(range(base, base + count)),
                matched_reference_count=count,
            )
            labels.append(label.model_dump(mode="json", exclude_none=True))
    if set(test_rows) != {item.review_id for item in inputs_by_split["test"]}:
        raise PublicArtifactValidationError("fixed_positive_test_label_universe_mismatch")
    return labels


def _recompute_fixed_positive_evaluation(
    *,
    root: Path,
    manifest_path: Path,
    manifest: Any,
    inputs_by_split: Mapping[str, list[Any]],
    predictions: list[Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    labels = _synthetic_public_evaluator_labels(
        manifest=manifest,
        inputs_by_split=inputs_by_split,
        evaluation=evaluation,
    )
    with TemporaryDirectory(prefix="metasyn-public-replay-") as temporary:
        replay_root = Path(temporary)
        manifest_payload = _load_json_object(manifest_path)
        for split in ("development", "calibration", "test"):
            relative = PurePosixPath(getattr(manifest, split).path)
            target = replay_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((manifest_path.parent / relative).read_bytes())
        labels_relative = PurePosixPath(manifest.evaluator_labels.path)
        labels_path = replay_root / labels_relative
        atomic_write_jsonl(labels_path, labels)
        manifest_payload["evaluator_labels"]["sha256"] = sha256_file(labels_path)
        replay_manifest = replay_root / "manifest.json"
        atomic_write_json(replay_manifest, manifest_payload)
        recomputed = evaluate_metasyn_predictions(
            manifest_path=replay_manifest,
            predictions=predictions,
            evaluation_split="test",
        )
    # The replay manifest differs only in the private-label file digest. Bind the
    # published result to the physical tracked manifest independently, then normalize
    # this one evaluator-generated identity field for exact result comparison.
    recomputed["manifest_sha256"] = sha256_file(root / _METASYN_BENCHMARK_MANIFEST)
    return recomputed


def _validate_metasyn_fixed_positive(
    value: Mapping[str, Any], *, root: Path
) -> None:
    bundle = root / _FIXED_POSITIVE_DIRECTORY
    receipt_path = bundle / "freeze_receipt.json"
    predictions_path = bundle / "predictions.jsonl"
    evaluation_path = bundle / "evaluation.json"
    if value != _load_json_object(receipt_path):
        raise PublicArtifactValidationError("fixed_positive_receipt_registry_mismatch")
    try:
        receipt = FixedDirectionBaselineReceipt.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("fixed_positive_receipt_invalid") from exc
    manifest_path = root / _METASYN_BENCHMARK_MANIFEST
    manifest = load_metasyn_manifest_metadata(manifest_path)
    inputs_by_split = {
        split: load_metasyn_inputs(manifest_path, split=split)
        for split in ("development", "calibration", "test")
    }
    predictions = load_metasyn_predictions(predictions_path)
    if (
        receipt.split != "test"
        or receipt.predicted_class != "Positive"
        or receipt.manifest_sha256 != sha256_file(manifest_path)
        or receipt.model_input_artifact_sha256 != manifest.test.sha256
        or receipt.model_inputs_canonical_sha256 != hash_canonical(inputs_by_split["test"])
        or receipt.predictions_file_sha256 != sha256_file(predictions_path)
        or receipt.predictions_canonical_sha256
        != hash_canonical(
            [item.model_dump(mode="json", exclude_none=True) for item in predictions]
        )
        or receipt.rows != len(predictions)
        or [item.review_id for item in predictions]
        != [item.review_id for item in inputs_by_split["test"]]
        or any(
            item.predicted_direction != "Positive"
            or item.retrieved_corpus_ids is not None
            or item.risk_features is not None
            for item in predictions
        )
    ):
        raise PublicArtifactValidationError("fixed_positive_freeze_lineage_invalid")

    evaluation = _load_json_object(evaluation_path)
    recomputed = _recompute_fixed_positive_evaluation(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        inputs_by_split=inputs_by_split,
        predictions=predictions,
        evaluation=evaluation,
    )
    if recomputed != evaluation:
        raise PublicArtifactValidationError("fixed_positive_evaluation_recompute_mismatch")
    direction = evaluation.get("direction")
    retrieval = evaluation.get("retrieval")
    if (
        evaluation.get("evaluated_split") != "test"
        or evaluation.get("predictions_sha256") != hash_canonical(predictions)
        or not isinstance(direction, Mapping)
        or direction.get("eligible_gold") != 86
        or direction.get("correct") != 42
        or not isinstance(retrieval, Mapping)
        or retrieval.get("eligible_reviews") != 86
        or retrieval.get("supplied") != 0
        or retrieval.get("coverage") != 0.0
    ):
        raise PublicArtifactValidationError("fixed_positive_control_boundary_invalid")

    audit = _load_json_object(root / _METASYN_CLOSED_AUDIT)
    _validate_self_hash(
        audit,
        field="audit_payload_sha256",
        artifact_path=_METASYN_CLOSED_AUDIT,
    )
    audit_inputs = audit.get("cached_local_baseline", {}).get("input_hashes", {})
    metasyn_inputs = audit.get("corpora", {}).get("metasyn", {}).get("input_hashes", {})
    if (
        audit_inputs.get("existing_metasyn_evaluation") != sha256_file(evaluation_path)
        or audit_inputs.get("predictions") != sha256_file(predictions_path)
        or metasyn_inputs.get("manifest") != sha256_file(manifest_path)
        or metasyn_inputs.get("private_evaluator_labels")
        != manifest.evaluator_labels.sha256
    ):
        raise PublicArtifactValidationError("fixed_positive_closed_audit_lineage_invalid")


def _validate_local_suite(value: Mapping[str, Any], *, root: Path) -> None:
    if (
        value.get("local_benchmark_report_version") != "2"
        or value.get("status") not in {"complete", "complete_with_release_license_blocker"}
        or value.get("network_calls") != 0
    ):
        raise PublicArtifactValidationError("local_suite_public_report_incomplete_or_stale")
    suite_path = root / _LOCAL_SUITE_CONFIG
    if value.get("suite_sha256") != sha256_file(suite_path):
        raise PublicArtifactValidationError("local_suite_config_hash_mismatch")
    source_map = value.get("source_code_sha256s")
    if not isinstance(source_map, Mapping) or set(source_map) != _LOCAL_SUITE_SOURCE_PATHS:
        raise PublicArtifactValidationError("local_suite_source_inventory_invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PublicArtifactValidationError("local_suite_artifact_inventory_invalid")
    integrity = artifacts.get("integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "metasyn_retrieval_summary",
        "metasyn_screening_summary",
    }:
        raise PublicArtifactValidationError("local_suite_integrity_inventory_invalid")
    expected_paths = {
        "metasyn_retrieval_summary": (
            "artifacts/diagnostics/metasyn-retrieval-study-v1.json"
        ),
        "metasyn_screening_summary": (
            "artifacts/diagnostics/metasyn-screening-study-v1.json"
        ),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, record in integrity.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise PublicArtifactValidationError(f"local_suite_integrity_record_invalid:{name}")
        relative = str(record["path"])
        if relative != expected_paths[name]:
            raise PublicArtifactValidationError(f"local_suite_integrity_path_invalid:{name}")
        path = root / relative
        if not path.is_file() or record.get("file_sha256") != sha256_file(path):
            raise PublicArtifactValidationError(f"local_suite_summary_file_mismatch:{name}")
        summary = _load_json_object(path)
        summaries[name] = summary
        if record.get("payload_sha256") != summary.get("public_summary_payload_sha256"):
            raise PublicArtifactValidationError(f"local_suite_summary_payload_mismatch:{name}")
    retrieval = summaries["metasyn_retrieval_summary"]
    screening = summaries["metasyn_screening_summary"]
    retrieval_freeze_payload = screening.get("lineage", {}).get(
        "retrieval_freeze_payload_sha256"
    )
    expected_integrity = {
        "metasyn_retrieval_summary": {
            "path": expected_paths["metasyn_retrieval_summary"],
            "file_sha256": sha256_file(
                root / expected_paths["metasyn_retrieval_summary"]
            ),
            "payload_sha256": retrieval["public_summary_payload_sha256"],
            "freeze_receipt_sha256": retrieval["lineage"]["freeze_receipt_sha256"],
            "freeze_payload_sha256": retrieval_freeze_payload,
        },
        "metasyn_screening_summary": {
            "path": expected_paths["metasyn_screening_summary"],
            "file_sha256": sha256_file(
                root / expected_paths["metasyn_screening_summary"]
            ),
            "payload_sha256": screening["public_summary_payload_sha256"],
            "retrieval_freeze_payload_sha256": retrieval_freeze_payload,
        },
    }
    if dict(integrity) != expected_integrity:
        raise PublicArtifactValidationError("local_suite_integrity_content_mismatch")
    expected_results = {
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
    if value.get("results") != expected_results:
        raise PublicArtifactValidationError("local_suite_results_content_mismatch")
    scientific_payload = {
        "suite_sha256": value["suite_sha256"],
        "artifact_integrity": expected_integrity,
        "results": expected_results,
    }
    reproducibility = value.get("reproducibility")
    if (
        not isinstance(reproducibility, Mapping)
        or reproducibility.get("scientific_payload_sha256")
        != hash_canonical(scientific_payload)
        or reproducibility.get("timestamps_in_scientific_payload") is not False
    ):
        raise PublicArtifactValidationError("local_suite_scientific_payload_mismatch")


def _validate_legacy_antiox_bundles(*, root: Path) -> None:
    try:
        app_namespace = runpy.run_path(
            (root / "app/streamlit_app.py").as_posix(),
            run_name="_literature_multiverse_public_streamlit_app",
        )
    except (OSError, ImportError) as exc:
        raise PublicArtifactValidationError("legacy_antiox_loader_unavailable") from exc
    load_demo_bundle = app_namespace.get("load_demo_bundle")
    if not callable(load_demo_bundle):
        raise PublicArtifactValidationError("legacy_antiox_loader_missing")
    antiox_root = root / "artifacts/antiox-training"
    release_root = antiox_root / "releases"
    try:
        release_directories = sorted(
            path
            for path in release_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    except OSError as exc:
        raise PublicArtifactValidationError("legacy_antiox_release_root_missing") from exc
    bundle_directories = [antiox_root / "demo", *release_directories]
    if len(release_directories) != 3:
        raise PublicArtifactValidationError("legacy_antiox_release_inventory_changed")
    for directory in bundle_directories:
        try:
            load_demo_bundle(directory)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise PublicArtifactValidationError(
                f"legacy_antiox_bundle_invalid:{directory.relative_to(root).as_posix()}"
            ) from exc


def _validate_evidencebench_grounding(
    value: Mapping[str, Any], *, root: Path
) -> None:
    audit_receipt = _load_json_object(root / _EVIDENCEBENCH_AUDIT_RECEIPT)
    validate_evidencebench_public_bundle(
        summary=value,
        audit_receipt=audit_receipt,
        require_current_environment=True,
    )

    limitations = value.get("licenses_scientific_scope")
    if not isinstance(limitations, list) or set(limitations) != (
        _EVIDENCEBENCH_REQUIRED_LIMITATIONS
    ):
        raise PublicArtifactValidationError(
            "evidencebench_mandatory_limitations_mismatch"
        )

    selected_method_id = value.get("selected_method_id")
    test_results = value.get("test_results")
    paired_deltas = value.get("selected_method_paired_deltas")
    if not isinstance(selected_method_id, str) or not isinstance(test_results, list):
        raise PublicArtifactValidationError("evidencebench_selected_method_invalid")
    if not isinstance(paired_deltas, list):
        raise PublicArtifactValidationError("evidencebench_paired_deltas_invalid")

    result_by_method: dict[str, Mapping[str, Any]] = {}
    selected_rows: list[Mapping[str, Any]] = []
    for row in test_results:
        if not isinstance(row, Mapping) or not isinstance(row.get("method_id"), str):
            raise PublicArtifactValidationError("evidencebench_method_row_invalid")
        method_id = row["method_id"]
        if method_id in result_by_method:
            raise PublicArtifactValidationError("evidencebench_duplicate_method")
        result_by_method[method_id] = row
        if row.get("selected_on_development") is True:
            selected_rows.append(row)
    if (
        len(selected_rows) != 1
        or selected_rows[0]["method_id"] != selected_method_id
        or selected_rows[0].get("control") is not False
    ):
        raise PublicArtifactValidationError(
            "evidencebench_selected_method_membership_mismatch"
        )

    paired_by_comparator: dict[str, Mapping[str, Any]] = {}
    for row in paired_deltas:
        comparator = row.get("comparator_method_id") if isinstance(row, Mapping) else None
        if not isinstance(comparator, str) or comparator in paired_by_comparator:
            raise PublicArtifactValidationError(
                "evidencebench_paired_comparator_invalid"
            )
        paired_by_comparator[comparator] = row
    expected_comparators = set(result_by_method) - {selected_method_id}
    if set(paired_by_comparator) != expected_comparators:
        raise PublicArtifactValidationError(
            "evidencebench_paired_comparator_roster_mismatch"
        )

    selected = result_by_method[selected_method_id]
    for comparator_id, delta_row in paired_by_comparator.items():
        comparator = result_by_method[comparator_id]
        if (
            delta_row.get("selected_method_id") != selected_method_id
            or delta_row.get("comparator_is_control") is not comparator.get("control")
        ):
            raise PublicArtifactValidationError(
                "evidencebench_paired_method_binding_mismatch"
            )
        for metric in (
            "all_aspect_recall_at_10",
            "results_aspect_recall_at_5",
        ):
            delta = delta_row.get(f"{metric}_delta")
            selected_metric = selected.get(metric)
            comparator_metric = comparator.get(metric)
            if not all(
                isinstance(item, Mapping)
                for item in (delta, selected_metric, comparator_metric)
            ):
                raise PublicArtifactValidationError(
                    "evidencebench_paired_metric_invalid"
                )
            expected = float(selected_metric["estimate"]) - float(
                comparator_metric["estimate"]
            )
            if not math.isclose(
                float(delta["estimate"]),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise PublicArtifactValidationError(
                    "evidencebench_paired_estimate_mismatch"
                )


def _validate_native_bounded_ollama_v1_historical(
    value: Mapping[str, Any],
) -> None:
    """Validate the immutable v1 aggregate without projecting later code onto it."""

    for field, expected in _NATIVE_BOUNDED_V1_LINEAGE.items():
        if value.get(field) != expected:
            raise PublicArtifactValidationError(
                f"native_bounded_v1_frozen_lineage_mismatch:{field}"
            )
    if (
        value.get("public_summary_version") != "native-bounded-public-summary-v1"
        or value.get("diagnostic_version")
        != "native-antiox-bounded-ollama-diagnostic-v1"
        or value.get("status") != "complete_retrospective_development_diagnostic"
        or value.get("artifact_scope") != "aggregate_only_content_silent"
        or value.get("scientific_role")
        != "diagnostic_only_no_accuracy_calibration_or_release_authority"
    ):
        raise PublicArtifactValidationError("native_bounded_v1_scope_mismatch")
    if value.get("inventory_status_counts") != {
        "inventory_below_cap": 7,
        "inventory_contract_invalid": 2,
        "inventory_no_candidate_non_authorizing": 10,
    }:
        raise PublicArtifactValidationError(
            "native_bounded_v1_inventory_counts_mismatch"
        )
    if value.get("packet_status_counts") != {"packet_contract_invalid": 33}:
        raise PublicArtifactValidationError("native_bounded_v1_packet_counts_mismatch")
    if value.get("publication_status_counts") != {
        "inventory_contract_invalid": 2,
        "inventory_no_candidate_non_authorizing": 10,
        "packet_set_non_authorizing:packet_contract_invalid": 7,
    }:
        raise PublicArtifactValidationError(
            "native_bounded_v1_publication_counts_mismatch"
        )
    generation = value.get("generation")
    if (
        not isinstance(generation, Mapping)
        or generation.get("inventory_generation_calls") != 19
        or generation.get("packet_generation_calls") != 33
        or generation.get("paper_generation_calls") != 52
        or generation.get("synthetic_preflight_calls") != 6
        or generation.get("generation_retries") != 0
    ):
        raise PublicArtifactValidationError("native_bounded_v1_call_counts_mismatch")
    if (
        value.get("official_native_v1_estimable_publications") != 0
        or value.get("official_native_v1_findings") != 0
        or value.get("partial_packet_salvage_count") != 0
        or value.get("semantic_entailment_verified") is not False
        or value.get("extraction_accuracy_reported") is not False
        or value.get("release_probability_authority") is not False
        or value.get("claim_release_authority") is not False
        or value.get("empirical_counts_require_private_receipt_replay") is not True
    ):
        raise PublicArtifactValidationError("native_bounded_v1_authority_mismatch")


_HARVESTER_VALIDATION_HISTORICAL_SOURCE_BUNDLE_SHA256 = (
    "b663b0ea80c9cfd1c11d1ad59f90e94d6227ba65232e69e72eee972b2824bacd"
)


def _validate_harvester(value: Mapping[str, Any], *, root: Path) -> None:
    """Validate the pinned historical harvester validation summary.

    Before 2026-08-29 this required the artifact's embedded source map to equal a
    live rehash of the current checkout. `harvester/sources.py` changed on
    2026-08-29 without a probe re-run, so currency is now a pinned historical-bundle
    equality check instead of a live rehash; every receipt/identity/self-hash
    structural check performed by `validate_harvester_validation_summary` is
    unchanged.
    """

    try:
        summary = validate_harvester_validation_summary(
            value,
            repository_root=root,
            require_current_sources=False,
        )
    except HarvesterValidationError as exc:
        raise PublicArtifactValidationError(f"harvester_validation_invalid:{exc}") from exc
    if (
        hash_canonical(dict(summary.reproducibility.source_files_sha256))
        != _HARVESTER_VALIDATION_HISTORICAL_SOURCE_BUNDLE_SHA256
    ):
        raise PublicArtifactValidationError(
            "harvester_validation_source_lineage_historical_mismatch"
        )


def _validate_historical_verification_certificate_v5(value: Mapping[str, Any]) -> None:
    from literature_multiverse.certificate import VerificationCertificate

    try:
        certificate = VerificationCertificate.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "historical_verification_certificate_v5_invalid"
        ) from exc
    if certificate.certificate_version != "literature-multiverse-verification-v5":
        raise PublicArtifactValidationError(
            "historical_verification_certificate_version_mismatch"
        )
    if certificate.status != "abstained":
        raise PublicArtifactValidationError(
            "historical_verification_certificate_status_mismatch"
        )


def _validate_evidence_boundary_ledger_public(
    value: Mapping[str, Any], *, root: Path
) -> None:
    from literature_multiverse.evidence_boundary_ledger_v1 import (
        _implementation_hashes,
        validate_evidence_boundary_ledger,
    )

    try:
        ledger = validate_evidence_boundary_ledger(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("evidence_boundary_ledger_invalid") from exc
    if ledger.ledger_implementation_file_sha256s != _implementation_hashes(root):
        raise PublicArtifactValidationError("evidence_boundary_ledger_implementation_stale")


def _validate_decisive_readiness_blocked(value: Mapping[str, Any]) -> None:
    from literature_multiverse.decisive_claim_evaluation_v1 import (
        DecisiveEvaluationReadinessV1,
    )

    try:
        readiness = DecisiveEvaluationReadinessV1.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("decisive_readiness_invalid") from exc
    if readiness.status != "blocked" or readiness.real_scored_run_candidate is not False:
        raise PublicArtifactValidationError("decisive_readiness_registered_state_mismatch")


def _validate_metasyn_offline_audit_model_only(
    value: Mapping[str, Any], *, artifact_path: str
) -> None:
    if artifact_path.endswith("metasyn-passage-offline-feasibility-audit-v1.json"):
        from literature_multiverse.metasyn_passage_offline_feasibility_audit_v1 import (
            MetaSynPassageOfflineFeasibilityAuditV1 as model,
        )
    elif artifact_path.endswith("metasyn-contextual-frontier-v1-failure-audit-v1.json"):
        from literature_multiverse.metasyn_contextual_frontier_v1_failure_audit import (
            MetaSynContextualFrontierV1FailureAudit as model,
        )
    else:
        raise PublicArtifactValidationError(
            f"metasyn_offline_audit_unknown_artifact:{artifact_path}"
        )
    try:
        model.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("metasyn_offline_audit_invalid") from exc


def _validate_fable_public_paired_summary(value: Mapping[str, Any]) -> None:
    from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
        PublicPairedSummaryV1,
    )

    try:
        PublicPairedSummaryV1.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("fable_public_paired_summary_invalid") from exc
    if value.get("public_summary_version") != "evidence-inference-fable-public-paired-summary-v1":
        raise PublicArtifactValidationError("fable_public_paired_summary_version_mismatch")


def _validate_fable_public_union_evaluation_v2(value: Mapping[str, Any]) -> None:
    from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
        EvidenceInferenceFableFullUnionPublicEvaluationV2,
    )

    try:
        EvidenceInferenceFableFullUnionPublicEvaluationV2.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "fable_public_union_evaluation_v2_invalid"
        ) from exc


def _semantic_validate(
    kind: SemanticValidator,
    value: Mapping[str, Any],
    *,
    root: Path,
    artifact_path: str,
) -> None:
    if kind == "adaptive_stress":
        validate_adaptive_stress_study_artifact(value)
    elif kind == "closed_corpus":
        validate_local_corpus_audit(dict(value), require_current_sources=True)
    elif kind == "evidence_inference_provider_free":
        validate_public_diagnostic_summary(value)
    elif kind == "evidence_inference_local_ollama":
        _validate_evidence_inference_local_ollama(value, root=root)
    elif kind == "evidence_inference_ollama_gepa":
        _validate_evidence_inference_ollama_gepa(value)
    elif kind == "evidence_inference_item_risk":
        _validate_evidence_inference_item_risk(value, root=root)
    elif kind == "evidencebench_grounding":
        _validate_evidencebench_grounding(value, root=root)
    elif kind == "harvester":
        _validate_harvester(value, root=root)
    elif kind == "local_suite":
        _validate_local_suite(value, root=root)
    elif kind == "legacy_antiox_bundles":
        _validate_legacy_antiox_bundles(root=root)
    elif kind == "metasyn_fixed_positive":
        _validate_metasyn_fixed_positive(value, root=root)
    elif kind == "metasyn_retrieval":
        _validate_metasyn_retrieval(value, root=root)
    elif kind == "metasyn_screening":
        _validate_metasyn_screening(value, root=root)
    elif kind == "metasyn_synthesis_yield":
        _validate_metasyn_synthesis_yield(value)
    elif kind == "metasyn_synthesis_yield_v2":
        _validate_metasyn_synthesis_yield_v2(value)
    elif kind == "native_bounded_ollama":
        validate_bounded_public_summary(value, repository_root=root)
    elif kind == "native_bounded_ollama_v1_historical":
        _validate_native_bounded_ollama_v1_historical(value)
    elif kind == "native_ollama":
        validate_native_public_summary(value)
    elif kind == "planted_simulation":
        _validate_planted_simulation(
            value,
            root=root,
            artifact_path=artifact_path,
        )
    elif kind == "source_bridge":
        _validate_source_bridge(
            value,
            root=root,
            artifact_path=artifact_path,
        )
    elif kind == "historical_verification_certificate_v5":
        _validate_historical_verification_certificate_v5(value)
    elif kind == "evidence_boundary_ledger":
        _validate_evidence_boundary_ledger_public(value, root=root)
    elif kind == "decisive_readiness_blocked":
        _validate_decisive_readiness_blocked(value)
    elif kind == "metasyn_offline_audit_model_only":
        _validate_metasyn_offline_audit_model_only(value, artifact_path=artifact_path)
    elif kind == "fable_public_paired_summary":
        _validate_fable_public_paired_summary(value)
    elif kind == "fable_public_union_evaluation_v2":
        _validate_fable_public_union_evaluation_v2(value)
    elif kind == "generic":
        pass
    else:
        raise PublicArtifactValidationError(
            f"public_artifact_semantic_validator_unknown:{kind}"
        )


def validate_public_result_registry(*, repository_root: Path) -> dict[str, Any]:
    """Validate every registered result using only files intended for public CI."""

    root = repository_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    paths = [spec.path for spec in PUBLIC_RESULT_REGISTRY]
    if len(paths) != len(set(paths)):
        raise PublicArtifactValidationError("public_artifact_registry_path_duplicate")
    for spec in PUBLIC_RESULT_REGISTRY:
        path = _resolve_registered_artifact_path(
            root=root,
            artifact_path=spec.path,
        )
        value = _load_json_object(path)
        if spec.self_hash_field is not None:
            _validate_self_hash(value, field=spec.self_hash_field, artifact_path=spec.path)
        current_source_map_count, historical_source_map_count = (
            _validate_bound_source_maps(
                value,
                root=root,
                artifact_path=spec.path,
                bindings=spec.source_map_bindings,
            )
        )
        try:
            _semantic_validate(
                spec.semantic_validator,
                value,
                root=root,
                artifact_path=spec.path,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise PublicArtifactValidationError(
                f"public_artifact_semantic_validation_failed:{spec.path}:{exc}"
            ) from exc
        records.append(
            {
                "path": spec.path,
                "file_sha256": sha256_file(path),
                "payload_sha256": (
                    None
                    if spec.self_hash_field is None
                    else value[spec.self_hash_field]
                ),
                "semantic_validator": spec.semantic_validator,
                "declared_source_maps_validated": (
                    current_source_map_count + historical_source_map_count
                ),
                "current_source_maps_rehashed": current_source_map_count,
                "historical_source_maps_hash_bound": historical_source_map_count,
                "validation_access_policy": "public_checkout_files_only",
                "result_recomputed_from_public_inputs": (
                    spec.result_recomputed_from_public_inputs
                ),
                "limitation": spec.limitation,
            }
        )
    return {
        "status": "public_artifact_integrity_valid_with_scoped_semantics",
        "validation_scope": (
            "current registered headline artifacts and legacy Antiox release bundles: "
            "public integrity, role-declared current-replay and pinned historical "
            "source maps, and registered semantic "
            "invariants; not every tracked artifact and not blanket source-data or "
            "scientific-claim recomputation"
        ),
        "access_policy": {
            "public_checkout_files_only": True,
            "ignored_cache_required": False,
            "network_access_required": False,
        },
        "registered_artifacts": len(records),
        "artifacts": records,
    }


__all__ = [
    "PUBLIC_RESULT_REGISTRY",
    "PublicArtifactSpec",
    "PublicArtifactValidationError",
    "PublicSourceMapBinding",
    "validate_public_result_registry",
]
