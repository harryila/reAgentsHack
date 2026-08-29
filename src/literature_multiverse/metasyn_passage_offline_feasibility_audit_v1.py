"""Label-blind exhaustive feasibility audit of the immutable v2 packet roster.

The audit asks a deliberately narrower question than extraction accuracy: can a
previously-unattempted candidate produce a completed typed effect when the model
is constrained to one *entire* exact candidate passage and the existing v2
grounder/assembler are left unchanged?  A candidate is reachable only if the
quote itself makes the endpoint, contrast, arms, and all required values explicit;
every emitted numeric token is unique under v2; grounding completes; and assembly
completes.  No substring quote, cross-passage join, inferred statistic, parser
relaxation, hidden label, or provider call is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_extraction_inputs_v2 import MetaSynSourceStrengthSurfaceV2
from literature_multiverse.metasyn_passage_packet_rescue_v3 import (
    EXPECTED_V2_EXECUTION_BUNDLE_SHA256,
    EXPECTED_V2_FAILED_SMOKE_SHA256,
    EXPECTED_V2_INVENTORY_LEDGER_SHA256,
    EXPECTED_V2_PACKET_ROSTER_SHA256,
    EXPECTED_V2_PROVIDER_RECEIPT_COUNT,
    _python_dependency_closure,
    _replay_v2_base,
    _scientific_request_signature,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import EffectKind
from literature_multiverse.native_packet_grounding_v2 import (
    NativePacketGroundingV2Error,
    freeze_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

CONFIG_VERSION = "metasyn-passage-offline-feasibility-audit-config-v1"
CANDIDATE_AUDIT_VERSION = "metasyn-passage-offline-candidate-feasibility-v1"
AUDIT_VERSION = "metasyn-passage-offline-feasibility-audit-v1"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-passage-offline-feasibility-audit-v1.json")
DEFAULT_V2_WORKSPACE = Path("data/cache/metasyn/passage-hosted-yield-v2")
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/diagnostics/metasyn-passage-offline-feasibility-audit-v1.json"
)

_NEW_FINGERPRINT_FILES = (
    "src/literature_multiverse/metasyn_passage_offline_feasibility_audit_v1.py",
    "scripts/run_metasyn_passage_offline_feasibility_audit_v1.py",
    DEFAULT_CONFIG_PATH.as_posix(),
)

_CONTINUOUS_REQUIRED = (
    "effect.control_mean",
    "effect.control_n",
    "effect.control_sd",
    "effect.treatment_mean",
    "effect.treatment_n",
    "effect.treatment_sd",
)
_BINARY_REQUIRED = (
    "effect.control_events",
    "effect.control_total",
    "effect.treatment_events",
    "effect.treatment_total",
)
_DIRECT_CI_REQUIRED = (
    "effect.ci_level",
    "effect.ci_lower",
    "effect.ci_upper",
    "effect.estimate",
)

_FAMILIES: dict[str, tuple[str, ...]] = {
    "continuous_core_incomplete": ("03:02", "03:03", "03:04", "10:01", "16:02"),
    "binary_core_or_contrast_incomplete": (
        "14:01",
        "15:01",
        "15:02",
        "15:03",
        "15:04",
        "16:03",
        "16:04",
        "16:05",
        "17:01",
    ),
    "numeric_token_ambiguity": ("17:02", "17:03"),
    "direct_ci_contract_unreachable": (
        "16:01",
        "20:01",
        "20:02",
        "20:03",
        "20:04",
        "22:01",
        "23:01",
    ),
    "misrouted_non_ci": ("28:01", "28:02", "28:03"),
}
_COORDINATE_TO_FAMILY = {
    coordinate: family for family, coordinates in _FAMILIES.items() for coordinate in coordinates
}

_MISSING_FIELDS: dict[str, tuple[str, ...]] = {
    "03:02": (
        "effect.control_n",
        "effect.control_sd",
        "effect.treatment_n",
        "effect.treatment_sd",
    ),
    "03:03": ("effect.control_n", "effect.treatment_n"),
    "03:04": ("effect.control_n", "effect.treatment_n"),
    "10:01": _CONTINUOUS_REQUIRED,
    "16:02": ("effect.control_sd", "effect.treatment_sd"),
    **{
        coordinate: _BINARY_REQUIRED
        for coordinate in _FAMILIES["binary_core_or_contrast_incomplete"]
    },
    "23:01": ("effect.ci_level",),
    **{coordinate: _DIRECT_CI_REQUIRED for coordinate in _FAMILIES["misrouted_non_ci"]},
}

_BLOCKER_CODES: dict[str, tuple[str, ...]] = {
    "03:02": ("per_arm_sample_sizes_absent", "per_arm_standard_deviations_absent"),
    "03:03": ("per_arm_sample_sizes_absent",),
    "03:04": ("per_arm_sample_sizes_absent",),
    "10:01": ("no_explicit_treatment_comparator_group_statistics",),
    "16:02": ("per_arm_standard_deviations_absent",),
    "14:01": ("sensitivity_specificity_percentages_not_2x2_counts",),
    "15:01": ("single_concordance_percentage_not_2x2_counts",),
    "15:02": ("single_concordance_percentage_not_2x2_counts",),
    "15:03": ("concordance_percentages_not_2x2_counts",),
    "15:04": ("concordance_percentages_not_2x2_counts",),
    "16:03": ("single_group_events_no_comparator_arm",),
    "16:04": ("arm_percentages_without_integer_totals",),
    "16:05": ("events_each_group_without_group_totals_or_named_arm_mapping",),
    "17:01": ("adverse_event_named_without_arm_counts",),
    "17:02": ("required_treatment_total_token_nonunique",),
    "17:03": ("required_control_total_token_nonunique", "control_event_token_nonunique"),
    "16:01": ("ci_upper_rejected_at_en_dash_numeric_boundary",),
    "20:01": ("exact_effect_format_rate_ratio_unsupported", "ci_upper_en_dash_boundary"),
    "20:02": ("exact_effect_format_rr_unsupported", "ci_upper_en_dash_boundary"),
    "20:03": ("exact_effect_format_rr_unsupported", "ci_upper_en_dash_boundary"),
    "20:04": ("exact_effect_format_rr_unsupported", "ci_upper_en_dash_boundary"),
    "22:01": (
        "exact_effect_format_hazard_ratio_unsupported",
        "candidate_composite_endpoint_effect_incomplete",
    ),
    "23:01": (
        "effect_format_not_stated_in_candidate_quote",
        "ci_level_not_stated",
        "contrast_not_self_contained",
    ),
    "28:01": ("icer_threshold_not_direct_effect_with_ci",),
    "28:02": ("icer_point_value_not_direct_effect_with_ci",),
    "28:03": ("no_numeric_effect_or_ci",),
}

_EXPLICIT_MAPPING = frozenset(
    {
        "03:02",
        "03:03",
        "03:04",
        "16:01",
        "16:02",
        "16:04",
        "17:02",
        "17:03",
        "20:04",
    }
)

_WITHDRAWN_PARTIAL_WITNESSES: dict[str, dict[str, str]] = {
    "17:02": {
        "observed_partial_grounding_receipt_sha256": (
            "8b398ba440655daa26457870605229d4c24e11d9816a8f0c4bd9b2178fc255c1"
        ),
        "observed_partial_assembly_receipt_sha256": (
            "6082a9b1a9609f073b671cf869b1f2951d855d41a2a0ba130bd091621ce5ea02"
        ),
        "full_source_passage_text_sha256": (
            "625172b190c130f778efe1bbe398a48ca7e838e6f63b1d702ccca961971dd8bb"
        ),
        "withdrawal_reason": (
            "partial_quote_omitted_the_33_of_91_400_mg_arm_while_retaining_a_"
            "three_arm_respectively_clause_so_arm_value_mapping_was_not_self_contained"
        ),
    },
    "17:03": {
        "observed_partial_grounding_receipt_sha256": (
            "a9a0eeae718f83b8ddbfed47f76f448060a476fe3251c93d3522470455a83ffa"
        ),
        "observed_partial_assembly_receipt_sha256": (
            "268e135da4f344611a0117e10710ca6aeef8b3e35b74eb51adfc211c831fa553"
        ),
        "full_source_passage_text_sha256": (
            "edd2191f8f2cefe305497f05ca929b5e2e05439abc4c6eb1ea069c262bc53a25"
        ),
        "withdrawal_reason": (
            "partial_quote_omitted_the_35_of_96_400_mg_arm_while_retaining_the_"
            "three_arm_sentence_so_arm_value_mapping_was_not_self_contained"
        ),
    },
}

_PROBE_REQUIRED_FAILURE_FRAGMENT = {
    "16:01": "packet_grounding_v2_numeric_token_absent:effect.ci_upper",
    "17:02": "packet_grounding_v2_numeric_token_not_unique:effect.treatment_total",
    "17:03": "packet_grounding_v2_numeric_token_not_unique:effect.control_total",
    "20:01": "packet_grounding_v2_effect_format_alias_unsupported",
    "20:02": "packet_grounding_v2_effect_format_alias_unsupported",
    "20:03": "packet_grounding_v2_effect_format_alias_unsupported",
    "20:04": "packet_grounding_v2_effect_format_alias_unsupported",
    "22:01": "packet_grounding_v2_effect_format_alias_unsupported",
}

_PROBE_SPECS: dict[str, tuple[str | None, str, str, str, str]] = {
    "16:01": ("odds ratio", "134.4", "18.0", "1005", "95%"),
    "20:01": ("rate ratio", "0.75", "0.62", "0.92", "95%"),
    "20:02": ("RR", "0.94", "0.65", "1.34", "95%"),
    "20:03": ("RR", "0.60", "0.44", "0.81", "95%"),
    "20:04": ("RR", "0.62", "0.46", "0.80", "95%"),
    "22:01": ("hazard ratio", "0.67", "0.57", "0.79", "95%"),
}

_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?|\.\d+)%?")


class MetaSynPassageOfflineFeasibilityAuditV1Error(ValueError):
    """The offline roster or a frozen audit artifact is unsafe."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_self_hash(model: _FrozenExactModel, field_name: str) -> None:
    if getattr(model, field_name) != hash_canonical(
        model.model_dump(mode="json", exclude={field_name})
    ):
        raise ValueError(f"offline_feasibility_v1_self_hash_mismatch:{field_name}")


def _coordinate(row_ordinal: int, candidate_index: int) -> str:
    return f"{row_ordinal:02d}:{candidate_index:02d}"


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode) or not resolved.is_dir():
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_repository_root_unsafe"
        )
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassageOfflineFeasibilityAuditV1Error("offline_feasibility_v1_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_artifact_not_object"
        )
    return value


class MetaSynPassageOfflineFeasibilityConfigV1(_FrozenExactModel):
    config_version: Literal["metasyn-passage-offline-feasibility-audit-config-v1"] = CONFIG_VERSION
    audit_scope: Literal[
        "label_blind_exhaustive_previously_unattempted_packet_roster_full_quote_feasibility"
    ]
    expected_v2_execution_bundle_sha256: Sha256
    expected_v2_inventory_ledger_sha256: Sha256
    expected_v2_packet_roster_sha256: Sha256
    expected_v2_failed_smoke_sha256: Sha256
    expected_v2_provider_receipt_count: Literal[43]
    expected_v2_attempted_packet_count: Literal[3]
    expected_unattempted_candidate_count: Literal[26]
    full_quote_policy: Literal[
        "entire_exact_candidate_passage_no_substring_truncation_or_cross_passage_join"
    ]
    continuous_core_incomplete: list[str]
    binary_core_or_contrast_incomplete: list[str]
    numeric_token_ambiguity: list[str]
    direct_ci_contract_unreachable: list[str]
    misrouted_non_ci: list[str]
    provider_calls_permitted: Literal[False]
    application_retries_permitted: Literal[0]
    sdk_retries_permitted: Literal[0]
    reference_fields_unopened: Literal[True]
    official_test_labels_opened: Literal[False]
    inventory_normalization_permitted: Literal[False]
    quote_truncation_permitted: Literal[False]
    parser_relaxation_permitted: Literal[False]
    zero_yield_scope: Literal[
        "unchanged_v2_grounder_and_assembler_with_one_entire_exact_candidate_passage"
    ]
    zero_reachable_implies_zero_extractable_in_general: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    config_sha256: Sha256

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynPassageOfflineFeasibilityConfigV1:
        observed = {family: tuple(getattr(self, family)) for family in _FAMILIES}
        if observed != _FAMILIES:
            raise ValueError("offline_feasibility_v1_family_roster_mismatch")
        flattened = [value for values in observed.values() for value in values]
        if len(flattened) != 26 or len(set(flattened)) != 26:
            raise ValueError("offline_feasibility_v1_family_partition_invalid")
        if (
            self.expected_v2_execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256
            or self.expected_v2_inventory_ledger_sha256 != EXPECTED_V2_INVENTORY_LEDGER_SHA256
            or self.expected_v2_packet_roster_sha256 != EXPECTED_V2_PACKET_ROSTER_SHA256
            or self.expected_v2_failed_smoke_sha256 != EXPECTED_V2_FAILED_SMOKE_SHA256
            or self.expected_v2_provider_receipt_count != EXPECTED_V2_PROVIDER_RECEIPT_COUNT
        ):
            raise ValueError("offline_feasibility_v1_config_v2_anchor_mismatch")
        _validate_self_hash(self, "config_sha256")
        return self


def load_metasyn_passage_offline_feasibility_config_v1(
    *, repository_root: Path
) -> tuple[MetaSynPassageOfflineFeasibilityConfigV1, str]:
    root = _canonical_root(repository_root)
    path = root / DEFAULT_CONFIG_PATH
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_config_missing"
        ) from exc
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise MetaSynPassageOfflineFeasibilityAuditV1Error("offline_feasibility_v1_config_unsafe")
    return (
        MetaSynPassageOfflineFeasibilityConfigV1.model_validate(_read_object(path)),
        sha256_file(path),
    )


BlockerFamilyV1 = Literal[
    "continuous_core_incomplete",
    "binary_core_or_contrast_incomplete",
    "numeric_token_ambiguity",
    "direct_ci_contract_unreachable",
    "misrouted_non_ci",
]


class MetaSynCandidateOfflineFeasibilityV1(_FrozenExactModel):
    candidate_audit_version: Literal["metasyn-passage-offline-candidate-feasibility-v1"] = (
        CANDIDATE_AUDIT_VERSION
    )
    coordinate: str
    request_key: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    row_key: str
    effect_kind: EffectKind
    canonical_outcome_id: str
    outcome_concept_quote: str
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    packet_input_sha256: Sha256
    packet_request_sha256: Sha256
    scientific_request_signature_sha256: Sha256
    question_surface_sha256: Sha256
    row_protocol_orientation_sha256: Sha256
    assembly_analysis_policy_sha256: Sha256
    previously_attempted_in_v2: Literal[False]
    source_strength: MetaSynSourceStrengthSurfaceV2
    source_strength_surface_sha256: Sha256
    candidate_passage_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    full_exact_candidate_passage_texts: Annotated[list[str], Field(min_length=1, max_length=4)]
    full_exact_candidate_passage_text_sha256s: Annotated[
        list[Sha256], Field(min_length=1, max_length=4)
    ]
    full_exact_candidate_passage_lineage_sha256s: Annotated[
        list[Sha256], Field(min_length=1, max_length=4)
    ]
    full_quote_policy_applied: Literal[True]
    substring_quote_used: Literal[False]
    cross_passage_join_used: Literal[False]
    visible_numeric_lexemes: list[str]
    required_numeric_field_paths: list[str]
    missing_required_numeric_field_paths: list[str]
    explicit_arm_value_mapping_within_quote: bool
    candidate_endpoint_and_contrast_held_fixed: Literal[True]
    candidate_target_mutated: Literal[False]
    full_self_contained_completed_quote_available: Literal[False]
    all_required_numeric_tokens_unique_under_unchanged_v2: Literal[False]
    probe_model_outcome: dict[str, Any] | None
    probe_model_outcome_sha256: Sha256 | None
    probe_failure_chain: list[str]
    blocker_family: BlockerFamilyV1
    blocker_codes: Annotated[list[str], Field(min_length=1)]
    unchanged_v2_grounding_completed: Literal[False]
    unchanged_v2_assembly_completed: Literal[False]
    reachable_typed_effect: Literal[False]
    provider_calls_made: Literal[0]
    reference_fields_unopened: Literal[True]
    official_test_labels_opened: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    candidate_audit_sha256: Sha256

    @model_validator(mode="after")
    def validate_candidate(self) -> MetaSynCandidateOfflineFeasibilityV1:
        if (
            self.coordinate != _coordinate(self.row_ordinal, self.candidate_index)
            or self.source_strength_surface_sha256
            != self.source_strength.source_strength_surface_sha256
            or len(self.candidate_passage_ids)
            != len(self.full_exact_candidate_passage_texts)
            != len(self.full_exact_candidate_passage_text_sha256s)
            != len(self.full_exact_candidate_passage_lineage_sha256s)
            or self.full_exact_candidate_passage_text_sha256s
            != [_sha256_text(text) for text in self.full_exact_candidate_passage_texts]
            or self.probe_model_outcome_sha256
            != (
                hash_canonical(self.probe_model_outcome)
                if self.probe_model_outcome is not None
                else None
            )
            or self.blocker_family != _COORDINATE_TO_FAMILY.get(self.coordinate)
            or tuple(self.blocker_codes) != _BLOCKER_CODES[self.coordinate]
            or (
                self.coordinate in _PROBE_REQUIRED_FAILURE_FRAGMENT
                and (
                    self.probe_model_outcome is None
                    or not self.probe_failure_chain
                    or not any(
                        _PROBE_REQUIRED_FAILURE_FRAGMENT[self.coordinate] in item
                        for item in self.probe_failure_chain
                    )
                )
            )
            or (
                self.coordinate not in _PROBE_REQUIRED_FAILURE_FRAGMENT
                and (
                    self.probe_model_outcome is not None
                    or self.probe_model_outcome_sha256 is not None
                    or self.probe_failure_chain
                )
            )
        ):
            raise ValueError("offline_feasibility_v1_candidate_alias_mismatch")
        _validate_self_hash(self, "candidate_audit_sha256")
        return self


class MetaSynWithdrawnPartialWitnessV1(_FrozenExactModel):
    witness_version: Literal["metasyn-withdrawn-partial-feasibility-witness-v1"] = (
        "metasyn-withdrawn-partial-feasibility-witness-v1"
    )
    coordinate: Literal["17:02", "17:03"]
    prior_observation_status: Literal["partial_quote_grounding_and_assembly_observed_ephemerally"]
    withdrawal_status: Literal["withdrawn_after_full_source_sentence_self_containment_review"]
    observed_partial_grounding_receipt_sha256: Sha256
    observed_partial_assembly_receipt_sha256: Sha256
    observed_partial_receipt_bytes_retained: Literal[False]
    observed_partial_receipts_revalidation_authority: Literal[False]
    discarded_partial_quote_not_reused: Literal[True]
    full_source_passage_text_sha256: Sha256
    full_quote_probe_failure_chain: Annotated[list[str], Field(min_length=1)]
    withdrawal_reason: str
    source_feasibility_authority: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    withdrawn_witness_sha256: Sha256

    @model_validator(mode="after")
    def validate_witness(self) -> MetaSynWithdrawnPartialWitnessV1:
        expected = _WITHDRAWN_PARTIAL_WITNESSES[self.coordinate]
        if (
            self.observed_partial_grounding_receipt_sha256
            != expected["observed_partial_grounding_receipt_sha256"]
            or self.observed_partial_assembly_receipt_sha256
            != expected["observed_partial_assembly_receipt_sha256"]
            or self.full_source_passage_text_sha256 != expected["full_source_passage_text_sha256"]
            or self.withdrawal_reason != expected["withdrawal_reason"]
            or not any(
                _PROBE_REQUIRED_FAILURE_FRAGMENT[self.coordinate] in item
                for item in self.full_quote_probe_failure_chain
            )
        ):
            raise ValueError("offline_feasibility_v1_withdrawn_witness_alias_mismatch")
        _validate_self_hash(self, "withdrawn_witness_sha256")
        return self


class MetaSynPassageOfflineFeasibilityAuditV1(_FrozenExactModel):
    audit_version: Literal["metasyn-passage-offline-feasibility-audit-v1"] = AUDIT_VERSION
    status: Literal[
        "formal_zero_yield_blocker_no_full_quote_candidate_reaches_unchanged_v2_assembly"
    ]
    config: MetaSynPassageOfflineFeasibilityConfigV1
    config_sha256: Sha256
    config_file_sha256: Sha256
    v2_replay_snapshot_sha256: Sha256
    v2_execution_bundle_sha256: Sha256
    v2_inventory_ledger_sha256: Sha256
    v2_packet_roster_sha256: Sha256
    v2_failed_smoke_sha256: Sha256
    v2_provider_receipt_count: Literal[43]
    v2_attempted_packet_count: Literal[3]
    audited_unattempted_candidate_count: Literal[26]
    candidate_audits: Annotated[
        list[MetaSynCandidateOfflineFeasibilityV1], Field(min_length=26, max_length=26)
    ]
    candidate_audit_membership_sha256: Sha256
    withdrawn_partial_witnesses: Annotated[
        list[MetaSynWithdrawnPartialWitnessV1], Field(min_length=2, max_length=2)
    ]
    withdrawn_witness_membership_sha256: Sha256
    blocker_family_counts: dict[str, int]
    reachable_candidate_count: Literal[0]
    ranked_reachable_candidates: list[str]
    zero_yield_scope: Literal[
        "unchanged_v2_grounder_and_assembler_with_one_entire_exact_candidate_passage"
    ]
    general_extractability_evaluated: Literal[False]
    zero_reachable_does_not_imply_zero_extractable_in_general: Literal[True]
    full_quote_policy_satisfied_for_every_audit: Literal[True]
    v2_failed_gate_preserved: Literal[True]
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: Sha256
    provider_calls_made: Literal[0]
    hidden_or_reference_labels_opened: Literal[False]
    official_test_labels_opened: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    audit_sha256: Sha256

    @model_validator(mode="after")
    def validate_audit(self) -> MetaSynPassageOfflineFeasibilityAuditV1:
        expected_coordinates = sorted(_COORDINATE_TO_FAMILY)
        if (
            self.config_sha256 != self.config.config_sha256
            or [item.coordinate for item in self.candidate_audits] != expected_coordinates
            or self.candidate_audit_membership_sha256
            != hash_canonical([item.candidate_audit_sha256 for item in self.candidate_audits])
            or self.blocker_family_counts
            != {family: len(values) for family, values in _FAMILIES.items()}
            or [item.coordinate for item in self.withdrawn_partial_witnesses] != ["17:02", "17:03"]
            or self.withdrawn_witness_membership_sha256
            != hash_canonical(
                [item.withdrawn_witness_sha256 for item in self.withdrawn_partial_witnesses]
            )
            or self.ranked_reachable_candidates
            or any(item.reachable_typed_effect for item in self.candidate_audits)
            or self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256
        ):
            raise ValueError("offline_feasibility_v1_audit_alias_mismatch")
        _validate_self_hash(self, "audit_sha256")
        return self


def _required_fields(effect_kind: EffectKind) -> tuple[str, ...]:
    if effect_kind == "continuous_group_statistics":
        return _CONTINUOUS_REQUIRED
    if effect_kind == "binary_group_statistics":
        return _BINARY_REQUIRED
    return _DIRECT_CI_REQUIRED


def _exception_chain(exc: BaseException) -> list[str]:
    output: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        output.append(f"{type(current).__name__}:{current!s}")
        current = current.__cause__ or current.__context__
    return output


def _full_passage_probe(
    *, coordinate: str, request: Any, row: Any, full_quote: str
) -> tuple[dict[str, Any] | None, list[str]]:
    if coordinate in {"17:02", "17:03"}:
        values = (
            {
                "effect.control_events": ("6", "identity"),
                "effect.control_total": ("85", "identity"),
                "effect.treatment_events": ("31", "identity"),
                "effect.treatment_total": ("91", "identity"),
            }
            if coordinate == "17:02"
            else {
                "effect.control_events": ("1", "identity"),
                "effect.control_total": ("96", "identity"),
                "effect.treatment_events": ("39", "identity"),
                "effect.treatment_total": ("97", "identity"),
            }
        )
        effect_format_token = None
    elif coordinate in _PROBE_SPECS:
        effect_format_token, estimate, lower, upper, level = _PROBE_SPECS[coordinate]
        values = {
            "effect.ci_level": (level, "percent_to_proportion"),
            "effect.ci_lower": (lower, "identity"),
            "effect.ci_upper": (upper, "identity"),
            "effect.estimate": (estimate, "identity"),
        }
    else:
        return None, []
    raw: dict[str, Any] = {
        "outcome_version": "native-packet-grounding-model-outcome-v2",
        "packet_status": "completed",
        "candidate_binding_sha256": request.packet_input.candidate_binding_sha256,
        "evidence_quote": full_quote,
        "effect_format_token": effect_format_token,
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": field_path,
                "verbatim_numeric_token": token,
                "normalization": normalization,
            }
            for field_path, (token, normalization) in sorted(values.items())
        ],
        "identity_claims": [],
        "timepoint": {"kind": "not_reported"},
    }
    try:
        freeze_passage_packet_grounding_receipt_v2(
            model_outcome=raw,
            candidate=request.packet_input.candidate,
            projection=row.projection_v2,
        )
    except (NativePacketGroundingV2Error, ValueError) as exc:
        return raw, _exception_chain(exc)
    raise MetaSynPassageOfflineFeasibilityAuditV1Error(
        f"offline_feasibility_v1_probe_unexpectedly_grounded:{coordinate}"
    )


def _pipeline_fingerprint(
    *, repository_root: Path, config_sha256: str, snapshot_sha256: str
) -> PipelineFingerprint:
    files = sorted(set(_python_dependency_closure(repository_root)) | set(_NEW_FINGERPRINT_FILES))
    component = PipelineComponentSpec(
        component_id="metasyn-passage-offline-feasibility-audit-v1",
        component_version="1",
        file_paths=files,
        settings={
            "config_sha256": config_sha256,
            "v2_replay_snapshot_sha256": snapshot_sha256,
            "full_quote_policy": (
                "entire_exact_candidate_passage_no_substring_truncation_or_cross_passage_join"
            ),
            "expected_unattempted_candidates": 26,
            "provider_calls_permitted": False,
            "reference_fields_unopened": True,
            "official_test_labels_opened": False,
            "unchanged_v2_grounder": True,
            "unchanged_v2_assembler": True,
            "withdrawn_partial_witness_count": 2,
            "zero_yield_scope": (
                "unchanged_v2_grounder_and_assembler_with_one_entire_exact_candidate_passage"
            ),
            "zero_reachable_implies_zero_extractable_in_general": False,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        },
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


def freeze_metasyn_passage_offline_feasibility_audit_v1(
    *, repository_root: Path, v2_workspace: Path = DEFAULT_V2_WORKSPACE
) -> MetaSynPassageOfflineFeasibilityAuditV1:
    root = _canonical_root(repository_root)
    config, config_file_sha256 = load_metasyn_passage_offline_feasibility_config_v1(
        repository_root=root
    )
    context = _replay_v2_base(repository_root=root, v2_workspace=v2_workspace)
    snapshot = context.snapshot
    if (
        snapshot.execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256
        or snapshot.inventory_ledger_sha256 != EXPECTED_V2_INVENTORY_LEDGER_SHA256
        or snapshot.packet_roster_sha256 != EXPECTED_V2_PACKET_ROSTER_SHA256
        or snapshot.failed_smoke_sha256 != EXPECTED_V2_FAILED_SMOKE_SHA256
        or snapshot.provider_receipt_count != EXPECTED_V2_PROVIDER_RECEIPT_COUNT
        or context.smoke.status != "failed_gate"
        or context.smoke.remaining_packet_calls_permitted
    ):
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_v2_anchor_mismatch"
        )
    attempted = {
        (item.row_ordinal, item.candidate_index) for item in snapshot.attempted_packet_requests
    }
    unattempted = [
        request
        for request in context.packet_roster.requests
        if (request.row_ordinal, request.candidate_index) not in attempted
    ]
    observed_coordinates = {
        _coordinate(item.row_ordinal, item.candidate_index) for item in unattempted
    }
    if observed_coordinates != set(_COORDINATE_TO_FAMILY) or len(unattempted) != 26:
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_unattempted_roster_changed"
        )
    audits: list[MetaSynCandidateOfflineFeasibilityV1] = []
    for request in sorted(unattempted, key=lambda item: (item.row_ordinal, item.candidate_index)):
        coordinate = _coordinate(request.row_ordinal, request.candidate_index)
        row = context.bundle.extraction_inputs.rows[request.row_ordinal]
        row_orientation = context.bundle.protocol_orientations[request.row_ordinal]
        passage_by_id = {passage.passage_anchor: passage for passage in row.projection_v2.passages}
        passages = [
            passage_by_id[passage_id] for passage_id in request.packet_input.candidate.passage_ids
        ]
        texts = [passage.text for passage in passages]
        # A v2 evidence quote cannot cross passages.  Probes are possible only for
        # the single-passage roster members whose complete core is source-visible.
        probe, failure_chain = (
            _full_passage_probe(
                coordinate=coordinate,
                request=request,
                row=row,
                full_quote=texts[0],
            )
            if len(texts) == 1
            else (None, [])
        )
        payload: dict[str, Any] = {
            "candidate_audit_version": CANDIDATE_AUDIT_VERSION,
            "coordinate": coordinate,
            "request_key": request.request.request_key,
            "row_ordinal": request.row_ordinal,
            "candidate_index": request.candidate_index,
            "row_key": request.row_key,
            "effect_kind": request.packet_input.candidate.effect_kind,
            "canonical_outcome_id": (request.packet_input.candidate.canonical_outcome_id),
            "outcome_concept_quote": (request.packet_input.candidate.outcome_concept_quote),
            "candidate_descriptor_sha256": (request.packet_input.candidate_descriptor_sha256),
            "candidate_binding_sha256": (request.packet_input.candidate_binding_sha256),
            "packet_input_sha256": request.packet_input_sha256,
            "packet_request_sha256": request.packet_request_sha256,
            "scientific_request_signature_sha256": (_scientific_request_signature(request)),
            "question_surface_sha256": row.question_surface_sha256,
            "row_protocol_orientation_sha256": (row_orientation.protocol_orientation_sha256),
            "assembly_analysis_policy_sha256": (context.bundle.assembly_analysis_policy_sha256),
            "previously_attempted_in_v2": False,
            "source_strength": row.source_strength,
            "source_strength_surface_sha256": (row.source_strength.source_strength_surface_sha256),
            "candidate_passage_ids": request.packet_input.candidate.passage_ids,
            "full_exact_candidate_passage_texts": texts,
            "full_exact_candidate_passage_text_sha256s": [
                passage.text_sha256 for passage in passages
            ],
            "full_exact_candidate_passage_lineage_sha256s": [
                passage.passage_lineage_sha256 for passage in passages
            ],
            "full_quote_policy_applied": True,
            "substring_quote_used": False,
            "cross_passage_join_used": False,
            "visible_numeric_lexemes": sorted(
                {match.group(0) for text in texts for match in _NUMERIC_RE.finditer(text)}
            ),
            "required_numeric_field_paths": list(
                _required_fields(request.packet_input.candidate.effect_kind)
            ),
            "missing_required_numeric_field_paths": list(_MISSING_FIELDS.get(coordinate, ())),
            "explicit_arm_value_mapping_within_quote": (coordinate in _EXPLICIT_MAPPING),
            "candidate_endpoint_and_contrast_held_fixed": True,
            "candidate_target_mutated": False,
            "full_self_contained_completed_quote_available": False,
            "all_required_numeric_tokens_unique_under_unchanged_v2": False,
            "probe_model_outcome": probe,
            "probe_model_outcome_sha256": hash_canonical(probe) if probe else None,
            "probe_failure_chain": failure_chain,
            "blocker_family": _COORDINATE_TO_FAMILY[coordinate],
            "blocker_codes": list(_BLOCKER_CODES[coordinate]),
            "unchanged_v2_grounding_completed": False,
            "unchanged_v2_assembly_completed": False,
            "reachable_typed_effect": False,
            "provider_calls_made": 0,
            "reference_fields_unopened": True,
            "official_test_labels_opened": False,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        audits.append(
            MetaSynCandidateOfflineFeasibilityV1.model_validate(
                {
                    **payload,
                    "candidate_audit_sha256": hash_canonical(payload),
                }
            )
        )
    audit_by_coordinate = {item.coordinate: item for item in audits}
    withdrawn_witnesses: list[MetaSynWithdrawnPartialWitnessV1] = []
    for coordinate in sorted(_WITHDRAWN_PARTIAL_WITNESSES):
        prior = _WITHDRAWN_PARTIAL_WITNESSES[coordinate]
        candidate_audit = audit_by_coordinate[coordinate]
        witness_payload = {
            "witness_version": "metasyn-withdrawn-partial-feasibility-witness-v1",
            "coordinate": coordinate,
            "prior_observation_status": (
                "partial_quote_grounding_and_assembly_observed_ephemerally"
            ),
            "withdrawal_status": ("withdrawn_after_full_source_sentence_self_containment_review"),
            "observed_partial_grounding_receipt_sha256": (
                prior["observed_partial_grounding_receipt_sha256"]
            ),
            "observed_partial_assembly_receipt_sha256": (
                prior["observed_partial_assembly_receipt_sha256"]
            ),
            "observed_partial_receipt_bytes_retained": False,
            "observed_partial_receipts_revalidation_authority": False,
            "discarded_partial_quote_not_reused": True,
            "full_source_passage_text_sha256": (prior["full_source_passage_text_sha256"]),
            "full_quote_probe_failure_chain": candidate_audit.probe_failure_chain,
            "withdrawal_reason": prior["withdrawal_reason"],
            "source_feasibility_authority": False,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        withdrawn_witnesses.append(
            MetaSynWithdrawnPartialWitnessV1.model_validate(
                {
                    **witness_payload,
                    "withdrawn_witness_sha256": hash_canonical(witness_payload),
                }
            )
        )
    pipeline = _pipeline_fingerprint(
        repository_root=root,
        config_sha256=config.config_sha256,
        snapshot_sha256=snapshot.snapshot_sha256,
    )
    payload = {
        "audit_version": AUDIT_VERSION,
        "status": (
            "formal_zero_yield_blocker_no_full_quote_candidate_reaches_unchanged_v2_assembly"
        ),
        "config": config,
        "config_sha256": config.config_sha256,
        "config_file_sha256": config_file_sha256,
        "v2_replay_snapshot_sha256": snapshot.snapshot_sha256,
        "v2_execution_bundle_sha256": snapshot.execution_bundle_sha256,
        "v2_inventory_ledger_sha256": snapshot.inventory_ledger_sha256,
        "v2_packet_roster_sha256": snapshot.packet_roster_sha256,
        "v2_failed_smoke_sha256": snapshot.failed_smoke_sha256,
        "v2_provider_receipt_count": snapshot.provider_receipt_count,
        "v2_attempted_packet_count": len(snapshot.attempted_packet_requests),
        "audited_unattempted_candidate_count": len(audits),
        "candidate_audits": audits,
        "candidate_audit_membership_sha256": hash_canonical(
            [item.candidate_audit_sha256 for item in audits]
        ),
        "withdrawn_partial_witnesses": withdrawn_witnesses,
        "withdrawn_witness_membership_sha256": hash_canonical(
            [item.withdrawn_witness_sha256 for item in withdrawn_witnesses]
        ),
        "blocker_family_counts": {family: len(values) for family, values in _FAMILIES.items()},
        "reachable_candidate_count": 0,
        "ranked_reachable_candidates": [],
        "zero_yield_scope": (
            "unchanged_v2_grounder_and_assembler_with_one_entire_exact_candidate_passage"
        ),
        "general_extractability_evaluated": False,
        "zero_reachable_does_not_imply_zero_extractable_in_general": True,
        "full_quote_policy_satisfied_for_every_audit": True,
        "v2_failed_gate_preserved": True,
        "pipeline_fingerprint": pipeline,
        "pipeline_sha256": pipeline.pipeline_sha256,
        "provider_calls_made": 0,
        "hidden_or_reference_labels_opened": False,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageOfflineFeasibilityAuditV1.model_validate(
        {**payload, "audit_sha256": hash_canonical(payload)}
    )


def validate_metasyn_passage_offline_feasibility_audit_v1(
    *,
    audit: MetaSynPassageOfflineFeasibilityAuditV1 | Mapping[str, Any],
    repository_root: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    external_replay: bool = True,
) -> MetaSynPassageOfflineFeasibilityAuditV1:
    try:
        canonical = MetaSynPassageOfflineFeasibilityAuditV1.model_validate(audit)
    except ValueError as exc:
        raise MetaSynPassageOfflineFeasibilityAuditV1Error(
            "offline_feasibility_v1_saved_audit_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_passage_offline_feasibility_audit_v1(
            repository_root=repository_root, v2_workspace=v2_workspace
        )
        if replayed != canonical:
            raise MetaSynPassageOfflineFeasibilityAuditV1Error(
                "offline_feasibility_v1_external_replay_mismatch"
            )
    return canonical


def write_metasyn_passage_offline_feasibility_audit_v1(
    *,
    audit: MetaSynPassageOfflineFeasibilityAuditV1,
    output_path: Path,
) -> Path:
    path = Path(os.path.abspath(output_path))
    if path.exists() or path.is_symlink():
        saved = MetaSynPassageOfflineFeasibilityAuditV1.model_validate(_read_object(path))
        if saved != audit:
            raise MetaSynPassageOfflineFeasibilityAuditV1Error(
                "offline_feasibility_v1_output_replay_mismatch"
            )
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, audit)
    return path.resolve(strict=True)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_V2_WORKSPACE",
    "MetaSynCandidateOfflineFeasibilityV1",
    "MetaSynPassageOfflineFeasibilityAuditV1",
    "MetaSynPassageOfflineFeasibilityAuditV1Error",
    "freeze_metasyn_passage_offline_feasibility_audit_v1",
    "validate_metasyn_passage_offline_feasibility_audit_v1",
    "write_metasyn_passage_offline_feasibility_audit_v1",
]
