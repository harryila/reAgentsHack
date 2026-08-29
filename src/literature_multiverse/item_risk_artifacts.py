"""Artifact contracts for the production item-risk calibration command line.

These records bind the pure calibration contracts to the exact files consumed by a
run.  They deliberately keep bin definition, development labels, calibration labels,
shift detection, and prospective scoring in physically separate artifacts.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.item_risk_calibration import (
    FixedRiskBinFamily,
    ItemRiskCalibrationBundle,
    ItemRiskCandidate,
    RiskBound,
    ShiftAssessment,
    score_item_risk_bound,
    validate_item_risk_calibration_bundle_integrity,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import PipelineFingerprintVerification


class ItemRiskArtifactError(ValueError):
    """An on-disk item-risk artifact violated the production contract."""


def _sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _strict_false(value: object, field_name: str) -> object:
    if value is not False:
        raise ValueError(f"{field_name}_must_be_false")
    return value


class RiskBinDefinitionArtifact(ContractModel):
    """Human-authored, label-scope-explicit definition of fixed score bins."""

    definition_version: Literal["item-risk-bin-definition-v1"] = (
        "item-risk-bin-definition-v1"
    )
    definition_source: Literal["prespecified", "development_only"]
    source_split: Literal["none", "development"]
    labels_used: bool
    label_source: Literal["benchmark_annotation", "expert_adjudication"] | None
    simulation: Literal[False]
    score_name: Annotated[str, Field(min_length=1)]
    score_model_sha256: str
    edges: list[Annotated[float, Field(ge=0, le=1)]]

    @field_validator("score_model_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "risk_bin_score_model")

    @field_validator("labels_used", mode="before")
    @classmethod
    def validate_labels_used_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("labels_used_must_be_boolean")
        return value

    @field_validator("simulation", mode="before")
    @classmethod
    def validate_not_simulation(cls, value: object) -> object:
        return _strict_false(value, "risk_bin_definition_simulation")

    @model_validator(mode="after")
    def validate_definition_scope(self) -> RiskBinDefinitionArtifact:
        if self.definition_source == "prespecified":
            if self.source_split != "none" or self.labels_used or self.label_source is not None:
                raise ValueError("prespecified_bins_must_not_use_data_labels")
        elif self.source_split != "development":
            raise ValueError("development_bins_require_development_source_split")
        elif self.labels_used != (self.label_source is not None):
            raise ValueError("development_bin_label_source_mismatch")
        return self


class FixedRiskBinsReceipt(ContractModel):
    """Self-hashed receipt binding fixed bins to their exact definition file."""

    receipt_version: Literal["fixed-item-risk-bins-receipt-v1"] = (
        "fixed-item-risk-bins-receipt-v1"
    )
    definition_file_sha256: str
    definition: RiskBinDefinitionArtifact
    bin_family: FixedRiskBinFamily
    access_order: list[str]
    receipt_sha256: str

    @field_validator("definition_file_sha256", "receipt_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "fixed_risk_bins_receipt")

    @model_validator(mode="after")
    def validate_receipt(self) -> FixedRiskBinsReceipt:
        if self.access_order != ["definition_opened", "fixed_bins_sealed"]:
            raise ValueError("fixed_risk_bins_access_order_invalid")
        family = self.bin_family
        definition = self.definition
        if (
            family.definition_artifact_sha256 != self.definition_file_sha256
            or family.definition_source != definition.definition_source
            or family.score_name != definition.score_name
            or family.score_model_sha256 != definition.score_model_sha256
        ):
            raise ValueError("fixed_risk_bins_definition_lineage_mismatch")
        observed_edges = [family.bins[0].lower, *(risk_bin.upper for risk_bin in family.bins)]
        if observed_edges != definition.edges:
            raise ValueError("fixed_risk_bins_edges_definition_mismatch")
        if hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"})) != (
            self.receipt_sha256
        ):
            raise ValueError("fixed_risk_bins_receipt_hash_mismatch")
        return self


class ItemRiskCalibrationRunReceipt(ContractModel):
    """Self-hashed calibration run with file and access-order evidence."""

    receipt_version: Literal["item-risk-calibration-run-v2"] = (
        "item-risk-calibration-run-v2"
    )
    expected_pipeline_file_sha256: str
    fixed_bins_file_sha256: str
    fixed_bins_receipt_sha256: str
    development_units_file_sha256: str
    calibration_units_file_sha256: str
    development_unit_count: Annotated[int, Field(gt=0)]
    calibration_unit_count: Annotated[int, Field(gt=0)]
    pipeline_verification: PipelineFingerprintVerification
    bundle: ItemRiskCalibrationBundle
    access_order: list[str]
    receipt_sha256: str

    @field_validator(
        "expected_pipeline_file_sha256",
        "fixed_bins_file_sha256",
        "fixed_bins_receipt_sha256",
        "development_units_file_sha256",
        "calibration_units_file_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "item_risk_calibration_run")

    @model_validator(mode="after")
    def validate_receipt(self) -> ItemRiskCalibrationRunReceipt:
        expected_order = [
            "expected_pipeline_fingerprint_opened",
            "pipeline_fingerprint_recomputed_and_matched",
            "fixed_bins_receipt_opened",
            "development_units_opened",
            "calibration_units_opened",
            "calibration_bundle_sealed",
        ]
        if self.access_order != expected_order:
            raise ValueError("item_risk_calibration_access_order_invalid")
        if (
            self.pipeline_verification.status != "matched"
            or self.bundle.pipeline_verification_sha256
            != self.pipeline_verification.verification_sha256
            or self.bundle.pipeline_sha256
            != self.pipeline_verification.expected_pipeline_sha256
            or self.development_unit_count != len(self.bundle.development.item_ids)
            or self.calibration_unit_count != len(self.bundle.calibration.item_ids)
        ):
            raise ValueError("item_risk_calibration_receipt_lineage_mismatch")
        if hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"})) != (
            self.receipt_sha256
        ):
            raise ValueError("item_risk_calibration_receipt_hash_mismatch")
        return self


class ExternalShiftDetectorReceipt(ContractModel):
    """Self-hashed decision emitted by an external, frozen shift detector."""

    receipt_version: Literal["external-item-risk-shift-v1"] = (
        "external-item-risk-shift-v1"
    )
    calibration_bundle_sha256: str
    detector_id: Annotated[str, Field(min_length=1)]
    detector_sha256: str
    candidate_population_id: Annotated[str, Field(min_length=1)]
    candidate_domain: Annotated[str, Field(min_length=1)]
    candidate_input_file_sha256: str
    status: Literal["no_shift_detected", "shift_detected"]
    detector_artifact_sha256: str
    source_split: Literal["prospective"]
    labels_opened: Literal[False]
    simulation: Literal[False]
    receipt_sha256: str

    @field_validator(
        "calibration_bundle_sha256",
        "detector_sha256",
        "candidate_input_file_sha256",
        "detector_artifact_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "external_shift_detector_receipt")

    @field_validator("labels_opened", "simulation", mode="before")
    @classmethod
    def validate_false_flags(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", "external_shift_flag")
        return _strict_false(value, field_name)

    @model_validator(mode="after")
    def validate_receipt_hash(self) -> ExternalShiftDetectorReceipt:
        if hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"})) != (
            self.receipt_sha256
        ):
            raise ValueError("external_shift_detector_receipt_hash_mismatch")
        return self


class ShiftAssessmentRunReceipt(ContractModel):
    """Self-hashed bridge from an external detector receipt to a shift assessment."""

    receipt_version: Literal["item-risk-shift-run-v1"] = "item-risk-shift-run-v1"
    calibration_run_file_sha256: str
    calibration_run_receipt_sha256: str
    detector_receipt_file_sha256: str
    detector_receipt_sha256: str
    detector_artifact_file_sha256: str
    assessment: ShiftAssessment
    access_order: list[str]
    receipt_sha256: str

    @field_validator(
        "calibration_run_file_sha256",
        "calibration_run_receipt_sha256",
        "detector_receipt_file_sha256",
        "detector_receipt_sha256",
        "detector_artifact_file_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "shift_assessment_run")

    @model_validator(mode="after")
    def validate_receipt(self) -> ShiftAssessmentRunReceipt:
        if self.access_order != [
            "calibration_run_receipt_opened",
            "external_detector_receipt_opened",
            "external_detector_artifact_opened_and_verified",
            "shift_assessment_sealed",
        ]:
            raise ValueError("shift_assessment_access_order_invalid")
        if self.assessment.assessment_artifact_sha256 != self.detector_artifact_file_sha256:
            raise ValueError("shift_assessment_detector_artifact_mismatch")
        if hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"})) != (
            self.receipt_sha256
        ):
            raise ValueError("shift_assessment_run_receipt_hash_mismatch")
        return self


class ProspectiveItemRiskInput(ContractModel):
    """One prospective raw score with no outcome or probability field."""

    input_version: Literal["prospective-item-risk-input-v1"] = (
        "prospective-item-risk-input-v1"
    )
    source_split: Literal["prospective"]
    simulation: Literal[False]
    item_id: Annotated[str, Field(min_length=1)]
    question_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    score_model_sha256: str
    score_input_sha256: str
    risk_score: Annotated[float, Field(ge=0, le=1)]

    @field_validator("pipeline_sha256", "score_model_sha256", "score_input_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "prospective_item_risk_input")

    @field_validator("simulation", mode="before")
    @classmethod
    def validate_not_simulation(cls, value: object) -> object:
        return _strict_false(value, "prospective_item_risk_simulation")


class ItemRiskScoringRunReceipt(ContractModel):
    """Self-contained v2 scoring receipt with recomputable candidate/bound lineage."""

    receipt_version: Literal["item-risk-scoring-run-v2"] = "item-risk-scoring-run-v2"
    calibration_run_file_sha256: str
    calibration_run_receipt_sha256: str
    calibration_bundle_sha256: str
    calibration_bundle: ItemRiskCalibrationBundle
    expected_pipeline_file_sha256: str
    shift_run_file_sha256: str
    shift_run_receipt_sha256: str
    candidate_input_file_sha256: str
    candidate_count: Annotated[int, Field(gt=0)]
    pipeline_verification: PipelineFingerprintVerification
    candidates: list[ItemRiskCandidate]
    candidate_sha256s: list[str]
    bounds: list[RiskBound]
    access_order: list[str]
    receipt_sha256: str

    @field_validator(
        "calibration_run_file_sha256",
        "calibration_run_receipt_sha256",
        "calibration_bundle_sha256",
        "expected_pipeline_file_sha256",
        "shift_run_file_sha256",
        "shift_run_receipt_sha256",
        "candidate_input_file_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "item_risk_scoring_run")

    @field_validator("candidate_sha256s")
    @classmethod
    def validate_candidate_hashes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("item_risk_scoring_candidates_empty")
        for candidate_sha256 in value:
            _sha256(candidate_sha256, "scored_candidate")
        if len(value) != len(set(value)):
            raise ValueError("item_risk_scoring_candidate_hashes_duplicate")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> ItemRiskScoringRunReceipt:
        if self.access_order != [
            "calibration_run_receipt_opened",
            "expected_pipeline_fingerprint_opened",
            "pipeline_fingerprint_recomputed_and_matched",
            "shift_assessment_receipt_opened",
            "prospective_candidates_opened",
            "risk_bounds_scored",
        ]:
            raise ValueError("item_risk_scoring_access_order_invalid")
        if (
            self.candidate_count != len(self.candidate_sha256s)
            or self.candidate_count != len(self.candidates)
            or self.candidate_count != len(self.bounds)
            or [candidate.candidate_sha256 for candidate in self.candidates]
            != self.candidate_sha256s
            or [bound.candidate_sha256 for bound in self.bounds]
            != self.candidate_sha256s
        ):
            raise ValueError("item_risk_scoring_receipt_count_mismatch")
        item_ids = [candidate.item_id for candidate in self.candidates]
        if item_ids != sorted(set(item_ids)):
            raise ValueError("item_risk_scoring_candidates_not_canonically_sorted")
        if [bound.item_id for bound in self.bounds] != item_ids:
            raise ValueError("item_risk_scoring_candidate_bound_order_mismatch")
        bundle = validate_item_risk_calibration_bundle_integrity(self.calibration_bundle)
        if any(
            bound.pipeline_verification_sha256
            != self.pipeline_verification.verification_sha256
            for bound in self.bounds
        ) or any(
            bound.calibration_bundle_sha256 != bundle.bundle_sha256
            for bound in self.bounds
        ):
            raise ValueError("item_risk_scoring_receipt_lineage_mismatch")
        if self.calibration_bundle_sha256 != bundle.bundle_sha256:
            raise ValueError("item_risk_scoring_calibration_bundle_hash_mismatch")
        for candidate, bound in zip(self.candidates, self.bounds, strict=True):
            expected = score_item_risk_bound(
                candidate=candidate,
                bundle=bundle,
                pipeline_verification=self.pipeline_verification,
            )
            if expected != bound:
                raise ValueError(
                    f"item_risk_scoring_bound_recomputation_mismatch:{candidate.item_id}"
                )
        if hash_canonical(self.model_dump(mode="json", exclude={"receipt_sha256"})) != (
            self.receipt_sha256
        ):
            raise ValueError("item_risk_scoring_run_receipt_hash_mismatch")
        return self


class LegacyItemRiskScoringRunReceiptV1(ContractModel):
    """Parse-only legacy receipt.

    Version one omitted the sealed candidates and calibration bundle, so its bounds
    cannot be recomputed or used as release/scheduling authority.  Keeping this model
    separate prevents accidental promotion through the production v2 type.
    """

    receipt_version: Literal["item-risk-scoring-run-v1"]
    calibration_run_file_sha256: str
    calibration_run_receipt_sha256: str
    expected_pipeline_file_sha256: str
    shift_run_file_sha256: str
    shift_run_receipt_sha256: str
    candidate_input_file_sha256: str
    candidate_count: Annotated[int, Field(gt=0)]
    pipeline_verification: PipelineFingerprintVerification
    candidate_sha256s: list[str]
    bounds: list[dict[str, Any]]
    access_order: list[str]
    receipt_sha256: str
    diagnostic_only: Literal[True] = True

    @field_validator(
        "calibration_run_file_sha256",
        "calibration_run_receipt_sha256",
        "expected_pipeline_file_sha256",
        "shift_run_file_sha256",
        "shift_run_receipt_sha256",
        "candidate_input_file_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "legacy_item_risk_scoring_run")

    @model_validator(mode="after")
    def validate_diagnostic_receipt(self) -> LegacyItemRiskScoringRunReceiptV1:
        if (
            self.candidate_count != len(self.candidate_sha256s)
            or self.candidate_count != len(self.bounds)
        ):
            raise ValueError("legacy_item_risk_scoring_receipt_count_mismatch")
        for candidate_sha256 in self.candidate_sha256s:
            _sha256(candidate_sha256, "legacy_scored_candidate")
        payload = self.model_dump(
            mode="json", exclude={"receipt_sha256", "diagnostic_only"}
        )
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("legacy_item_risk_scoring_receipt_hash_mismatch")
        return self


__all__ = [
    "ExternalShiftDetectorReceipt",
    "FixedRiskBinsReceipt",
    "ItemRiskArtifactError",
    "ItemRiskCalibrationRunReceipt",
    "ItemRiskScoringRunReceipt",
    "LegacyItemRiskScoringRunReceiptV1",
    "ProspectiveItemRiskInput",
    "RiskBinDefinitionArtifact",
    "ShiftAssessmentRunReceipt",
]
