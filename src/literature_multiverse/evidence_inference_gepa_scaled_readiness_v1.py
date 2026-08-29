"""Fail-closed readiness audit for a scaled Evidence Inference GEPA study.

This module deliberately does not load benchmark JSONL payloads, private paired-test
reports, model responses, or credentials.  It replays only manifests, aggregate public
summaries, conversion metadata, and source-code task-shape evidence.  The current local
checkout has no pristine evaluation population, so this version can only freeze a
blocked readiness receipt; it cannot authorize optimization, evaluation, or an accuracy
claim.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical, sha256_file

READINESS_VERSION = "evidence-inference-gepa-scaled-readiness-v1"
CONFIG_VERSION = "evidence-inference-gepa-scaled-readiness-config-v1"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/evidence-inference-gepa-scaled-readiness-v1.json")
_EXPECTED_CONFIG_SHA256 = "57d7671d8a9b2d361c19285efe726a22297b9293f3b52b73e763dcab97fe0d67"
_EXPECTED_CONFIG_FILE_SHA256 = "671068c0f9791b4e62802173eaccf23fac688f81d09fc0466868b1e3e3a390a0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("train", "dev", "test")
_EXPECTED_ARTIFACTS = {
    "converter_source": (
        "src/literature_multiverse/evidence_inference.py",
        "90bc352a055ee2a48e31b215559df57ca4273f8321472ecf8fa77e60e09f9040",
    ),
    "full_conversion_report": (
        "data/cache/evidence-inference-gepa/conversion_report.json",
        "e22d1038bb244fd83a4fcd4d1dc93fb89b050b4ac289b04d3c35a1e4136ad086",
    ),
    "full_manifest": (
        "data/cache/evidence-inference-gepa/manifest.json",
        "77b145436e1c3902ebf87c0d0bf0e82475b8d3090057a7852e5c5fd9cfa63be2",
    ),
    "legacy_ollama_config": (
        "configs/benchmarks/evidence-inference-ollama-gepa-v1.json",
        "4d2ca82d12682cad3a2967e70a158a00731af0e347d4e7b61085b430ab4178c2",
    ),
    "low_budget_conversion_report": (
        "data/cache/evidence-inference-gepa-low-budget/conversion_report.json",
        "a58c6a8b1674d5409a6d2b774225b48f2f7b172c57dfbbfeab7167e36cdc3ef5",
    ),
    "low_budget_manifest": (
        "data/cache/evidence-inference-gepa-low-budget/manifest.json",
        "ad85e50afe44ddee9f033a5a5d27dddd34daaf1b27dfa28e7901725146e8a199",
    ),
    "opened_diagnostic_summary": (
        "artifacts/diagnostics/evidence-inference/summary.json",
        "4ff447c91a4742a33fc95c3dff6bc3a02b492227df849074cc8ea784fa423282",
    ),
    "pilot30_conversion_report": (
        "data/cache/evidence-inference-gepa-pilot30/conversion_report.json",
        "5fdcf78f7b60e9519f6429746c2aa0afb09e134a8ad27a7bb89f7e3c64bc667c",
    ),
    "pilot30_manifest": (
        "data/cache/evidence-inference-gepa-pilot30/manifest.json",
        "4868e76d0e6e5466ad824856f8a2715fcff294b0df5aeeb4938a3aefaeafd52b",
    ),
    "public_ollama_gepa_summary": (
        "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json",
        "2b7843ec918c874b61db6281457e965b5ae0aec32c32ca76c7777f63ccb67bf5",
    ),
}
_EXPECTED_VARIANTS = {
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
_EXPECTED_SPLIT_HASHES = {
    "full": {
        "train": "828688ee695ee64a979a362710e4186ce4148f7f22cd59a9aa5ffe3ab9de7268",
        "dev": "be465b52776938700d426edef8d2017f0ca2032ce82dd6b40219d43ad4e1e001",
        "test": "77143667614f58e049f746992cdf7440394a90bb22036037f07bacb1e10054cd",
    },
    "low_budget_12": {
        "train": "cebfe1f5187aaf8d8f676b194c6e6f62fd86589590a9238003995df625922eee",
        "dev": "4883ec45a1d02e870eed232959ebe02971f0e7f955ee068d5c92e0896fdea3d3",
        "test": "2ac9f1101379817db8b0e1293fafb35a58143f443042ed224b45acd3a6d9f8fc",
    },
    "pilot30": {
        "train": "c38b386d3114df9791e7ac0d8c5fc02dd3198fcaf038e17f92207c3377867e43",
        "dev": "79bc8803a2c112979bc9e0b11cd87d55591d726cb78a1df6f5aa9a2a65cf1996",
        "test": "732658dcb52873854454690b69c5a4775898e3a97a37fc61ffe2ff29c4e941b4",
    },
}
_EXPECTED_BLOCKERS = (
    "all_local_evidence_inference_labels_historically_opened",
    "complete_official_test_split_already_scored",
    "eligibility_negative_class_absent",
    "hidden_external_evaluation_labels_absent",
    "local_hardware_energy_or_monetary_cost_not_measured",
    "no_separate_complete_question_calibration_population",
    "no_unopened_paper_disjoint_evaluation_population",
    "no_unopened_question_disjoint_evaluation_population",
)
_SEPARATE_EVALUATION_OBJECTIVES = (
    "eligibility_screening_recall_and_specificity",
    "extraction_target_correctness",
    "formal_exact_source_grounding_validity",
    "structured_output_reliability",
    "provider_usage_and_cost",
)
_EXTERNAL_REQUIREMENTS = (
    "calibration_units_are_complete_independent_review_questions",
    "development_calibration_and_evaluation_questions_are_disjoint",
    "development_calibration_and_evaluation_papers_are_disjoint",
    "eligibility_positive_and_negative_examples_are_prespecified",
    "evaluation_labels_are_held_by_an_external_evaluator_and_unavailable_locally",
    "exact_source_spans_and_extraction_targets_are_expert_adjudicated",
    "gepa_candidate_budget_and_handwritten_seed_are_frozen_before_evaluation",
    "hidden_evaluation_is_scored_once_after_winner_and_threshold_freeze",
    "provider_identity_generation_contract_and_price_table_are_frozen",
    "seed_and_gepa_winner_receive_equal_paired_evaluation_call_budgets",
    "structured_output_failures_grounding_failures_and_task_errors_are_separate",
    "usage_and_cost_receipts_are_complete_for_every_attempt",
)


class EvidenceInferenceGEPAScaledReadinessError(ValueError):
    """Raised when metadata cannot support even a blocked readiness receipt."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_sha256(value: str, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"ei_gepa_readiness_v1_hash_invalid:{field}")
    return value


class MetadataArtifactBindingV1(_FrozenModel):
    role: str
    relative_path: str
    file_sha256: str

    @field_validator("file_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "file_sha256")

    @model_validator(mode="after")
    def validate_binding(self) -> MetadataArtifactBindingV1:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.relative_path:
            raise ValueError("ei_gepa_readiness_v1_artifact_path_unsafe")
        return self


class SplitCountSpecV1(_FrozenModel):
    rows: Annotated[int, Field(ge=0)]
    papers: Annotated[int, Field(ge=0)]


class VariantCountSpecV1(_FrozenModel):
    train: SplitCountSpecV1
    dev: SplitCountSpecV1
    test: SplitCountSpecV1


class EvidenceInferenceGEPAScaledReadinessConfigV1(_FrozenModel):
    config_version: Literal["evidence-inference-gepa-scaled-readiness-config-v1"] = CONFIG_VERSION
    artifact_bindings: Annotated[list[MetadataArtifactBindingV1], Field(min_length=10)]
    expected_variants: dict[str, VariantCountSpecV1]
    expected_public_gepa_summary_sha256: str
    expected_opened_diagnostic_summary_sha256: str
    separate_evaluation_objectives: list[str]
    external_evaluation_requirements: list[str]
    config_sha256: str

    @field_validator(
        "expected_public_gepa_summary_sha256",
        "expected_opened_diagnostic_summary_sha256",
        "config_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_config(self) -> EvidenceInferenceGEPAScaledReadinessConfigV1:
        observed = {
            item.role: (item.relative_path, item.file_sha256) for item in self.artifact_bindings
        }
        variant_payload = {
            name: {
                split: (getattr(spec, split).rows, getattr(spec, split).papers) for split in _SPLITS
            }
            for name, spec in self.expected_variants.items()
        }
        if (
            self.artifact_bindings != sorted(self.artifact_bindings, key=lambda item: item.role)
            or len(observed) != len(self.artifact_bindings)
            or observed != _EXPECTED_ARTIFACTS
            or variant_payload != _EXPECTED_VARIANTS
            or tuple(self.separate_evaluation_objectives) != _SEPARATE_EVALUATION_OBJECTIVES
            or tuple(self.external_evaluation_requirements) != _EXTERNAL_REQUIREMENTS
            or self.expected_public_gepa_summary_sha256
            != "1039156083798863e85761ecf94b76578c74066af2ef7b7691fd4d724f4967ce"
            or self.expected_opened_diagnostic_summary_sha256
            != "fe989ba03c606e6cacbcd4bea3517a08d5a0ac05b40b751ed4533dbceccef7d9"
        ):
            raise ValueError("ei_gepa_readiness_v1_config_contract_changed")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if self.config_sha256 != hash_canonical(payload):
            raise ValueError("ei_gepa_readiness_v1_config_hash_mismatch")
        return self


class LocalSplitFactsV1(_FrozenModel):
    split: Literal["train", "dev", "test"]
    rows: Annotated[int, Field(ge=0)]
    papers: Annotated[int, Field(ge=0)]
    groups: Annotated[int, Field(ge=0)]
    advertised_jsonl_sha256: str
    row_payload_present_but_unopened: Literal[True] = True

    @field_validator("advertised_jsonl_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "advertised_jsonl_sha256")


class LocalVariantFactsV1(_FrozenModel):
    variant: Literal["full", "low_budget_12", "pilot30"]
    manifest_file_sha256: str
    conversion_report_file_sha256: str
    splits: Annotated[list[LocalSplitFactsV1], Field(min_length=3, max_length=3)]
    example_membership_disjoint: Literal[True] = True
    paper_membership_disjoint: Literal[True] = True
    group_membership_disjoint: Literal[True] = True

    @field_validator("manifest_file_sha256", "conversion_report_file_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_variant(self) -> LocalVariantFactsV1:
        if [item.split for item in self.splits] != list(_SPLITS):
            raise ValueError("ei_gepa_readiness_v1_split_order_invalid")
        expected = _EXPECTED_VARIANTS[self.variant]
        if any(
            (item.rows, item.papers) != expected[item.split]
            or item.groups != item.papers
            or item.advertised_jsonl_sha256 != _EXPECTED_SPLIT_HASHES[self.variant][item.split]
            for item in self.splits
        ):
            raise ValueError("ei_gepa_readiness_v1_variant_count_mismatch")
        return self


class PriorStudyFactsV1(_FrozenModel):
    all_labels_historically_opened: Literal[True] = True
    confirmatory_claim_allowed: Literal[False] = False
    train_examples: Literal[256]
    train_articles: Literal[93]
    development_examples: Literal[96]
    development_articles: Literal[34]
    accepted_candidate_count: Literal[7]
    actual_metric_calls: Literal[864]
    paired_test_examples: Literal[524]
    paired_test_articles: Literal[191]
    provider_touched_test_rows: Literal[12]
    provider_touched_test_articles: Literal[12]
    provider_call_unseen_but_label_opened_rows: Literal[482]
    provider_call_unseen_but_label_opened_articles: Literal[179]
    observed_improvement_rule_satisfied: Literal[False] = False
    local_hardware_energy_or_monetary_cost_measured: Literal[False] = False


class EvidenceInferenceGEPAScaledReadinessReceiptV1(_FrozenModel):
    readiness_version: Literal["evidence-inference-gepa-scaled-readiness-v1"] = READINESS_VERSION
    status: Literal["blocked_no_unopened_question_and_paper_disjoint_evaluation_population"]
    config_file_sha256: str
    config_sha256: str
    component_source_sha256: str
    metadata_artifacts: Annotated[list[MetadataArtifactBindingV1], Field(min_length=10)]
    metadata_artifact_membership_sha256: str
    local_variants: Annotated[list[LocalVariantFactsV1], Field(min_length=3, max_length=3)]
    local_variant_membership_sha256: str
    prior_study: PriorStudyFactsV1
    qualified_examples_all_expected_eligible: Literal[True] = True
    eligibility_negative_example_count: Literal[0] = 0
    separate_calibration_split_present: Literal[False] = False
    unopened_question_disjoint_evaluation_population_exists: Literal[False] = False
    unopened_paper_disjoint_evaluation_population_exists: Literal[False] = False
    provider_call_unseen_is_not_label_unopened: Literal[True] = True
    benchmark_row_payloads_opened_by_preflight: Literal[False] = False
    private_reports_opened_by_preflight: Literal[False] = False
    hidden_or_reference_labels_opened_by_preflight: Literal[False] = False
    provider_calls_made_by_preflight: Literal[0] = 0
    development_only_diagnostic_possible: Literal[True] = True
    claim_bearing_scaled_optimizer_evaluation_ready: Literal[False] = False
    live_optimizer_or_evaluation_calls_authorized: Literal[False] = False
    blocker_codes: list[str]
    required_separate_evaluation_objectives: list[str]
    external_evaluation_requirements: list[str]
    accuracy_claim_authority: Literal[False] = False
    gepa_improvement_claim_authority: Literal[False] = False
    calibration_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "config_file_sha256",
        "config_sha256",
        "component_source_sha256",
        "metadata_artifact_membership_sha256",
        "local_variant_membership_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceGEPAScaledReadinessReceiptV1:
        observed_artifacts = {
            item.role: (item.relative_path, item.file_sha256) for item in self.metadata_artifacts
        }
        variant_hashes = {
            item.variant: (
                item.manifest_file_sha256,
                item.conversion_report_file_sha256,
            )
            for item in self.local_variants
        }
        expected_variant_hashes = {
            "full": (
                _EXPECTED_ARTIFACTS["full_manifest"][1],
                _EXPECTED_ARTIFACTS["full_conversion_report"][1],
            ),
            "low_budget_12": (
                _EXPECTED_ARTIFACTS["low_budget_manifest"][1],
                _EXPECTED_ARTIFACTS["low_budget_conversion_report"][1],
            ),
            "pilot30": (
                _EXPECTED_ARTIFACTS["pilot30_manifest"][1],
                _EXPECTED_ARTIFACTS["pilot30_conversion_report"][1],
            ),
        }
        if (
            self.metadata_artifacts != sorted(self.metadata_artifacts, key=lambda item: item.role)
            or observed_artifacts != _EXPECTED_ARTIFACTS
            or self.metadata_artifact_membership_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.metadata_artifacts])
            or [item.variant for item in self.local_variants]
            != ["full", "low_budget_12", "pilot30"]
            or variant_hashes != expected_variant_hashes
            or self.local_variant_membership_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.local_variants])
            or tuple(self.blocker_codes) != _EXPECTED_BLOCKERS
            or tuple(self.required_separate_evaluation_objectives)
            != _SEPARATE_EVALUATION_OBJECTIVES
            or tuple(self.external_evaluation_requirements) != _EXTERNAL_REQUIREMENTS
            or self.config_sha256 != _EXPECTED_CONFIG_SHA256
            or self.config_file_sha256 != _EXPECTED_CONFIG_FILE_SHA256
        ):
            raise ValueError("ei_gepa_readiness_v1_receipt_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("ei_gepa_readiness_v1_receipt_hash_mismatch")
        return self


def _repository_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    if candidate.is_symlink() or not candidate.is_file():
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_artifact_missing_or_unsafe"
        )
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_artifact_outside_repository"
        ) from exc
    return candidate


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_metadata_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceGEPAScaledReadinessError("ei_gepa_readiness_v1_metadata_not_object")
    return value


def _validate_public_self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = deepcopy(dict(value))
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or hash_canonical(payload) != observed:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_public_summary_hash_mismatch"
        )
    return observed


def _manifest_split_facts(
    *, manifest: Mapping[str, Any], report: Mapping[str, Any], split: str, base: Path
) -> LocalSplitFactsV1:
    raw = manifest.get(split)
    report_splits = report.get("splits")
    if not isinstance(raw, Mapping) or not isinstance(report_splits, Mapping):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_split_metadata_missing"
        )
    reported = report_splits.get(split)
    if not isinstance(reported, Mapping):
        raise EvidenceInferenceGEPAScaledReadinessError("ei_gepa_readiness_v1_report_split_missing")
    rows = raw.get("rows")
    paper_ids = raw.get("paper_ids")
    group_ids = raw.get("group_ids")
    example_ids = raw.get("example_ids")
    advertised_hash = raw.get("sha256")
    payload_name = raw.get("path")
    if (
        not isinstance(rows, int)
        or not isinstance(paper_ids, list)
        or not isinstance(group_ids, list)
        or not isinstance(example_ids, list)
        or not isinstance(advertised_hash, str)
        or not isinstance(payload_name, str)
        or len(example_ids) != rows
        or len(set(example_ids)) != rows
        or len(set(paper_ids)) != len(paper_ids)
        or len(set(group_ids)) != len(group_ids)
        or reported.get("rows") != rows
        or reported.get("papers") != len(paper_ids)
        or reported.get("jsonl_sha256") != advertised_hash
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_split_metadata_incoherent"
        )
    payload_path = base / payload_name
    if (
        Path(payload_name).is_absolute()
        or ".." in Path(payload_name).parts
        or payload_path.is_symlink()
        or not payload_path.is_file()
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_row_payload_missing_or_unsafe"
        )
    return LocalSplitFactsV1(
        split=split,
        rows=rows,
        papers=len(paper_ids),
        groups=len(group_ids),
        advertised_jsonl_sha256=advertised_hash,
        row_payload_present_but_unopened=True,
    )


def _assert_disjoint(manifest: Mapping[str, Any], field: str) -> None:
    memberships: dict[str, set[str]] = {}
    for split in _SPLITS:
        raw = manifest.get(split)
        values = raw.get(field) if isinstance(raw, Mapping) else None
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise EvidenceInferenceGEPAScaledReadinessError(
                "ei_gepa_readiness_v1_manifest_membership_invalid"
            )
        memberships[split] = set(values)
    if any(
        memberships[left].intersection(memberships[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_manifest_membership_leakage"
        )


def _variant_facts(
    *,
    variant: Literal["full", "low_budget_12", "pilot30"],
    manifest_path: Path,
    report_path: Path,
) -> LocalVariantFactsV1:
    manifest = _read_json_object(manifest_path)
    report = _read_json_object(report_path)
    if (
        report.get("manifest_sha256") != sha256_file(manifest_path)
        or report.get("split_policy")
        != "preserve official article-level train/validation/test membership"
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_conversion_lineage_mismatch"
        )
    for field in ("example_ids", "paper_ids", "group_ids"):
        _assert_disjoint(manifest, field)
    splits = [
        _manifest_split_facts(
            manifest=manifest,
            report=report,
            split=split,
            base=manifest_path.parent,
        )
        for split in _SPLITS
    ]
    return LocalVariantFactsV1(
        variant=variant,
        manifest_file_sha256=sha256_file(manifest_path),
        conversion_report_file_sha256=sha256_file(report_path),
        splits=splits,
        example_membership_disjoint=True,
        paper_membership_disjoint=True,
        group_membership_disjoint=True,
    )


def _prove_all_qualified_examples_expected_eligible(source_path: Path) -> None:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_converter_source_invalid"
        ) from exc
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_optimization_example"
    ]
    calls = (
        [
            node
            for node in ast.walk(functions[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OptimizationExample"
        ]
        if len(functions) == 1
        else []
    )
    eligible_values: list[object] = []
    for call in calls:
        expected = next(
            (item.value for item in call.keywords if item.arg == "expected_output"),
            None,
        )
        if isinstance(expected, ast.Dict):
            for key, value in zip(expected.keys, expected.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "eligible":
                    eligible_values.append(
                        value.value if isinstance(value, ast.Constant) else object()
                    )
    if len(calls) != 1 or eligible_values != [True]:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_eligibility_task_shape_changed"
        )


def _load_config(path: Path) -> EvidenceInferenceGEPAScaledReadinessConfigV1:
    try:
        return EvidenceInferenceGEPAScaledReadinessConfigV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_config_invalid"
        ) from exc


def freeze_evidence_inference_gepa_scaled_readiness_v1(
    *,
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> EvidenceInferenceGEPAScaledReadinessReceiptV1:
    """Freeze the metadata-only blocked readiness receipt; never open row payloads."""

    try:
        root = repository_root.resolve(strict=True)
    except OSError as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_repository_missing"
        ) from exc
    if root.is_symlink() or not root.is_dir():
        raise EvidenceInferenceGEPAScaledReadinessError("ei_gepa_readiness_v1_repository_unsafe")
    resolved_config = config_path if config_path.is_absolute() else root / config_path
    try:
        resolved_config.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_config_outside_repository"
        ) from exc
    if resolved_config.is_symlink() or not resolved_config.is_file():
        raise EvidenceInferenceGEPAScaledReadinessError("ei_gepa_readiness_v1_config_unsafe")
    config = _load_config(resolved_config)
    paths: dict[str, Path] = {}
    for binding in config.artifact_bindings:
        artifact = _repository_file(root, binding.relative_path)
        if sha256_file(artifact) != binding.file_sha256:
            raise EvidenceInferenceGEPAScaledReadinessError(
                "ei_gepa_readiness_v1_artifact_hash_mismatch"
            )
        paths[binding.role] = artifact

    _prove_all_qualified_examples_expected_eligible(paths["converter_source"])
    variants = [
        _variant_facts(
            variant="full",
            manifest_path=paths["full_manifest"],
            report_path=paths["full_conversion_report"],
        ),
        _variant_facts(
            variant="low_budget_12",
            manifest_path=paths["low_budget_manifest"],
            report_path=paths["low_budget_conversion_report"],
        ),
        _variant_facts(
            variant="pilot30",
            manifest_path=paths["pilot30_manifest"],
            report_path=paths["pilot30_conversion_report"],
        ),
    ]

    public = _read_json_object(paths["public_ollama_gepa_summary"])
    diagnostic = _read_json_object(paths["opened_diagnostic_summary"])
    if (
        _validate_public_self_hash(public, "public_summary_sha256")
        != config.expected_public_gepa_summary_sha256
        or _validate_public_self_hash(diagnostic, "public_summary_sha256")
        != config.expected_opened_diagnostic_summary_sha256
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_public_summary_anchor_mismatch"
        )
    optimization = public.get("optimization_population")
    optimizer = public.get("optimizer")
    paired = public.get("paired_test_population")
    resource = public.get("resource_and_cost")
    population = diagnostic.get("population")
    if not all(
        isinstance(item, Mapping)
        for item in (optimization, optimizer, paired, resource, population)
    ):
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_public_summary_shape_invalid"
        )
    prior = PriorStudyFactsV1(
        all_labels_historically_opened=public.get("all_labels_historically_opened"),
        confirmatory_claim_allowed=public.get("confirmatory_claim_allowed"),
        train_examples=optimization.get("train_examples"),
        train_articles=optimization.get("train_articles"),
        development_examples=optimization.get("development_examples"),
        development_articles=optimization.get("development_articles"),
        accepted_candidate_count=optimizer.get("accepted_candidate_count"),
        actual_metric_calls=optimizer.get("actual_metric_calls"),
        paired_test_examples=paired.get("examples"),
        paired_test_articles=paired.get("articles"),
        provider_touched_test_rows=population.get("registered_previous_provider_test_attempt_rows"),
        provider_touched_test_articles=population.get("provider_touched_test_articles"),
        provider_call_unseen_but_label_opened_rows=population.get(
            "provider_call_unseen_paper_diagnostic_rows"
        ),
        provider_call_unseen_but_label_opened_articles=population.get(
            "provider_call_unseen_paper_diagnostic_articles"
        ),
        observed_improvement_rule_satisfied=public.get("observed_improvement_rule_satisfied"),
        local_hardware_energy_or_monetary_cost_measured=resource.get(
            "local_hardware_energy_or_monetary_cost_measured"
        ),
    )
    artifacts = sorted(config.artifact_bindings, key=lambda item: item.role)
    payload: dict[str, Any] = {
        "readiness_version": READINESS_VERSION,
        "status": ("blocked_no_unopened_question_and_paper_disjoint_evaluation_population"),
        "config_file_sha256": sha256_file(resolved_config),
        "config_sha256": config.config_sha256,
        "component_source_sha256": sha256_file(Path(__file__)),
        "metadata_artifacts": artifacts,
        "metadata_artifact_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in artifacts]
        ),
        "local_variants": variants,
        "local_variant_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in variants]
        ),
        "prior_study": prior,
        "qualified_examples_all_expected_eligible": True,
        "eligibility_negative_example_count": 0,
        "separate_calibration_split_present": False,
        "unopened_question_disjoint_evaluation_population_exists": False,
        "unopened_paper_disjoint_evaluation_population_exists": False,
        "provider_call_unseen_is_not_label_unopened": True,
        "benchmark_row_payloads_opened_by_preflight": False,
        "private_reports_opened_by_preflight": False,
        "hidden_or_reference_labels_opened_by_preflight": False,
        "provider_calls_made_by_preflight": 0,
        "development_only_diagnostic_possible": True,
        "claim_bearing_scaled_optimizer_evaluation_ready": False,
        "live_optimizer_or_evaluation_calls_authorized": False,
        "blocker_codes": list(_EXPECTED_BLOCKERS),
        "required_separate_evaluation_objectives": list(_SEPARATE_EVALUATION_OBJECTIVES),
        "external_evaluation_requirements": list(_EXTERNAL_REQUIREMENTS),
        "accuracy_claim_authority": False,
        "gepa_improvement_claim_authority": False,
        "calibration_claim_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceGEPAScaledReadinessReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_evidence_inference_gepa_scaled_readiness_v1(
    *,
    receipt: EvidenceInferenceGEPAScaledReadinessReceiptV1 | Mapping[str, Any],
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    external_replay: bool = True,
) -> EvidenceInferenceGEPAScaledReadinessReceiptV1:
    """Validate intrinsically and, by default, replay all metadata-only inputs."""

    try:
        canonical = EvidenceInferenceGEPAScaledReadinessReceiptV1.model_validate(
            receipt.model_dump(mode="json")
            if isinstance(receipt, EvidenceInferenceGEPAScaledReadinessReceiptV1)
            else receipt
        )
    except ValueError as exc:
        raise EvidenceInferenceGEPAScaledReadinessError(
            "ei_gepa_readiness_v1_receipt_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_evidence_inference_gepa_scaled_readiness_v1(
            repository_root=repository_root,
            config_path=config_path,
        )
        if replayed != canonical:
            raise EvidenceInferenceGEPAScaledReadinessError(
                "ei_gepa_readiness_v1_external_replay_mismatch"
            )
    return canonical


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_CONFIG_PATH",
    "READINESS_VERSION",
    "EvidenceInferenceGEPAScaledReadinessConfigV1",
    "EvidenceInferenceGEPAScaledReadinessError",
    "EvidenceInferenceGEPAScaledReadinessReceiptV1",
    "LocalSplitFactsV1",
    "LocalVariantFactsV1",
    "MetadataArtifactBindingV1",
    "PriorStudyFactsV1",
    "freeze_evidence_inference_gepa_scaled_readiness_v1",
    "validate_evidence_inference_gepa_scaled_readiness_v1",
]
