"""Label-blind preparation for a bounded MetaSyn typed-synthesis pilot.

The pilot is an explicit oracle-corpus feasibility study.  Released matched-paper
membership is opened to define the evaluator-provided corpus, while review direction,
conclusion, aggregate effect, significance, and official-test fields remain unopened.
Raw protocol text, identifiers, source projections, and source locators stay in the
ignored private workspace; a later stage may emit aggregate diagnostics only.
"""

from __future__ import annotations

import ast
import importlib
import json
import platform
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

import pyarrow.parquet as pq
from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import (
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_screening_study import validate_fit
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import (
    NativeSourceManifest,
    NativeSourceRecord,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import resolve_native_source_document
from literature_multiverse.native_question_projection import (
    CanonicalOutcomeV1,
    FrozenSourceProjectionV1,
    QuestionProjectionSpecV1,
    freeze_canonical_outcomes,
    freeze_question_projection_spec,
    project_resolved_source_for_question,
    validate_frozen_source_projection_external_replay,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.source_manifest_bridge import (
    DiagnosticSourceRecord,
    SourceContentScope,
    build_metasyn_native_source_bridge,
)

PILOT_VERSION = "metasyn-typed-oracle-pilot-v1"
PREPARE_BUNDLE_VERSION = "metasyn-typed-oracle-prepare-bundle-v1"
PREPARE_RECEIPT_VERSION = "metasyn-typed-oracle-prepare-receipt-v1"
QUESTION_SPEC_VERSION = "metasyn-typed-oracle-question-spec-v1"
QUESTION_BUNDLE_VERSION = "metasyn-typed-oracle-question-bundle-v1"
SOURCE_ROW_VERSION = "metasyn-typed-oracle-source-row-v1"
PILOT_PIPELINE_COMPONENT_VERSION = "1"

_PILOT_DEPENDENCY_ENTRYPOINTS = (
    "scripts/run_metasyn_typed_pilot.py",
    "src/literature_multiverse/metasyn_typed_pilot.py",
    "src/literature_multiverse/native_bounded_generation.py",
    "src/literature_multiverse/native_question_projection.py",
)
_PILOT_NON_PYTHON_INPUTS = (
    "configs/benchmarks/metasyn-corpus-c8fa07d.json",
    "prompts/metasyn_candidate_inventory.md",
    "prompts/metasyn_candidate_packet.md",
    "pyproject.toml",
    "uv.lock",
)

EXPECTED_REVIEWS_TRAIN_SHA256 = (
    "9582b29191c52a4ebf695b976b42f900c9003728e3baac9fec38f19cf73d4e5b"
)
EXPECTED_CALIBRATION_QUESTIONS = 161
EXPECTED_SELECTED_QUESTIONS = 10
EXPECTED_SELECTED_COMPONENTS = 10
EXPECTED_SELECTED_PAPERS = 32
MIN_MATCHED_PAPERS = 2
MAX_MATCHED_PAPERS = 4
MIN_FULL_TEXT_PAPERS = 2
MAX_PROTOCOL_TEXT_CHARACTERS = 4_096
MAX_CRITERIA_CHARACTERS = 16_384
MAX_SEARCH_DATE_CHARACTERS = 128
MAX_REPOSITORY_PATH_CHARACTERS = 2_048
MAX_SOURCE_LOCATOR_CHARACTERS = 2_048

PREPARE_BUNDLE_FILENAME = "prepare-bundle.private.json"
PREPARE_RECEIPT_FILENAME = "prepare-receipt.json"

MATERIALIZED_REVIEW_COLUMNS: tuple[str, ...] = (
    "ID",
    "Title",
    "Research_Question",
    "Population",
    "Intervention",
    "Exposure",
    "Comparison",
    "Outcome",
    "inclusion_criteria",
    "exclusion_criteria",
    "search_end_date",
    "matched_corpus_ids",
    "matched_ref_count",
    "source_review_corpus_ids",
)
FORBIDDEN_REFERENCE_COLUMNS: tuple[str, ...] = (
    "Abstract",
    "CI_Lower",
    "CI_Upper",
    "Conclusion_Summary",
    "Effect_Direction",
    "Effect_Size_Category",
    "Effect_Size_Type",
    "Effect_Size_Value",
    "Heterogeneity_Level",
    "I2_Value",
    "Key_Insights",
    "P_Value",
    "Q_Value",
    "Statistical_Significance",
    "Tau2_Value",
    "conclusion_paragraph",
)


class MetaSynTypedPilotError(ValueError):
    """A source, split, selection, or private artifact violated the pilot contract."""


def _resolve_pilot_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _pilot_python_dependency_closure(repository_root: Path) -> list[str]:
    """Walk every direct/transitive in-repository import from public entry points."""

    pending = list(_PILOT_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if not source_path.is_file():
            raise MetaSynTypedPilotError(
                f"metasyn_pilot_pipeline_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynTypedPilotError(
                f"metasyn_pilot_pipeline_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_pilot_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def _downstream_verifier_pipeline_fingerprint(
    repository_root: Path,
) -> PipelineFingerprint:
    # Dynamic import intentionally keeps the pilot's mechanically walked code closure
    # distinct from the separately computed, nested verifier identity.
    module = importlib.import_module("literature_multiverse.verifier")
    compute = getattr(module, "compute_verifier_pipeline_fingerprint", None)
    if not callable(compute):
        raise MetaSynTypedPilotError("metasyn_pilot_verifier_fingerprint_api_missing")
    value = compute(root=repository_root)
    return PipelineFingerprint.model_validate(value)


def compute_metasyn_typed_pilot_pipeline_fingerprint(
    *, root: Path | None = None
) -> PipelineFingerprint:
    """Compute the dependency-closed preparation/generation/verifier identity."""

    repository_root = (root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    python_closure = _pilot_python_dependency_closure(repository_root)
    downstream = _downstream_verifier_pipeline_fingerprint(repository_root)
    official_schema_sha256 = hash_canonical(native_publication_extraction_json_schema())
    component = PipelineComponentSpec(
        component_id="metasyn-typed-oracle-pilot",
        component_version=PILOT_PIPELINE_COMPONENT_VERSION,
        file_paths=sorted({*python_closure, *_PILOT_NON_PYTHON_INPUTS}),
        settings={
            "allowed_scientific_output": (
                "typed_extraction_grounding_and_synthesis_yield_only"
            ),
            "dependency_closure_entrypoints": list(_PILOT_DEPENDENCY_ENTRYPOINTS),
            "downstream_verifier_pipeline_sha256": downstream.pipeline_sha256,
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name)
                for name in ("jsonschema", "pyarrow", "pydantic")
            },
            "official_native_extraction_schema_sha256": official_schema_sha256,
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "reference_fields_unopened_during_prepare_and_prediction": True,
            "selection_rule": (
                "calibration_component_independent_10_questions_32_oracle_papers"
            ),
        },
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


class MetaSynPilotAccessStateV1(ContractModel):
    access_state_version: Literal["metasyn-typed-pilot-access-v1"] = (
        "metasyn-typed-pilot-access-v1"
    )
    logical_split: Literal["calibration"] = "calibration"
    scientific_role: Literal[
        "retrospective_nonpristine_oracle_corpus_extraction_synthesis_feasibility"
    ] = "retrospective_nonpristine_oracle_corpus_extraction_synthesis_feasibility"
    released_matched_membership_opened_for_oracle_corpus: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    review_direction_values_opened: Literal[False] = False
    review_conclusion_values_opened: Literal[False] = False
    review_aggregate_effect_values_opened: Literal[False] = False
    official_test_inputs_opened: Literal[False] = False
    official_test_labels_opened: Literal[False] = False
    shared_all_split_evaluator_opened: Literal[False] = False
    strict_storage_level_unopened_claim_possible: Literal[False] = False
    model_prediction_opened_during_prepare: Literal[False] = False
    materialized_review_columns: list[str]
    forbidden_reference_columns: list[str]

    @model_validator(mode="after")
    def validate_access(self) -> MetaSynPilotAccessStateV1:
        if self.materialized_review_columns != list(MATERIALIZED_REVIEW_COLUMNS):
            raise ValueError("metasyn_pilot_materialized_columns_mismatch")
        if self.forbidden_reference_columns != sorted(FORBIDDEN_REFERENCE_COLUMNS):
            raise ValueError("metasyn_pilot_forbidden_columns_mismatch")
        if set(self.materialized_review_columns) & set(self.forbidden_reference_columns):
            raise ValueError("metasyn_pilot_reference_column_materialized")
        return self


class MetaSynPilotSelectionConfigV1(ContractModel):
    selection_config_version: Literal["metasyn-typed-pilot-selection-v1"] = (
        "metasyn-typed-pilot-selection-v1"
    )
    pilot_version: Literal["metasyn-typed-oracle-pilot-v1"] = PILOT_VERSION
    universe: Literal[
        "frozen_metasyn_screening_calibration_questions"
    ] = "frozen_metasyn_screening_calibration_questions"
    arm: Literal["evaluator_oracle_corpus"] = "evaluator_oracle_corpus"
    matched_papers_minimum: Literal[2] = MIN_MATCHED_PAPERS
    matched_papers_maximum: Literal[4] = MAX_MATCHED_PAPERS
    full_text_papers_minimum: Literal[2] = MIN_FULL_TEXT_PAPERS
    require_complete_population: Literal[True] = True
    require_exactly_one_intervention_or_exposure: Literal[True] = True
    require_complete_comparison: Literal[True] = True
    require_complete_outcome: Literal[True] = True
    require_complete_research_question: Literal[True] = True
    require_every_oracle_source_available: Literal[True] = True
    require_component_independence: Literal[True] = True
    selection_rationale: Literal[
        "deterministic_census_of_source_adequate_small_oracle_corpora"
    ] = "deterministic_census_of_source_adequate_small_oracle_corpora"
    representativeness_claim: Literal[
        "none_source_adequate_oracle_stratum_only"
    ] = "none_source_adequate_oracle_stratum_only"
    component_algorithm: Literal[
        "connected_matched_paper_source_review_title_question_v1"
    ] = "connected_matched_paper_source_review_title_question_v1"
    expected_calibration_questions: Literal[161] = EXPECTED_CALIBRATION_QUESTIONS
    expected_selected_questions: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    expected_selected_components: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    expected_selected_papers: Literal[32] = EXPECTED_SELECTED_PAPERS
    selection_config_sha256: str

    @field_validator("selection_config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_pilot_selection_config_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynPilotSelectionConfigV1:
        payload = self.model_dump(mode="json", exclude={"selection_config_sha256"})
        if hash_canonical(payload) != self.selection_config_sha256:
            raise ValueError("metasyn_pilot_selection_config_hash_mismatch")
        return self


def freeze_metasyn_pilot_selection_config() -> MetaSynPilotSelectionConfigV1:
    payload: dict[str, Any] = {
        "selection_config_version": "metasyn-typed-pilot-selection-v1",
        "pilot_version": PILOT_VERSION,
        "universe": "frozen_metasyn_screening_calibration_questions",
        "arm": "evaluator_oracle_corpus",
        "matched_papers_minimum": MIN_MATCHED_PAPERS,
        "matched_papers_maximum": MAX_MATCHED_PAPERS,
        "full_text_papers_minimum": MIN_FULL_TEXT_PAPERS,
        "require_complete_population": True,
        "require_exactly_one_intervention_or_exposure": True,
        "require_complete_comparison": True,
        "require_complete_outcome": True,
        "require_complete_research_question": True,
        "require_every_oracle_source_available": True,
        "require_component_independence": True,
        "selection_rationale": (
            "deterministic_census_of_source_adequate_small_oracle_corpora"
        ),
        "representativeness_claim": "none_source_adequate_oracle_stratum_only",
        "component_algorithm": "connected_matched_paper_source_review_title_question_v1",
        "expected_calibration_questions": EXPECTED_CALIBRATION_QUESTIONS,
        "expected_selected_questions": EXPECTED_SELECTED_QUESTIONS,
        "expected_selected_components": EXPECTED_SELECTED_COMPONENTS,
        "expected_selected_papers": EXPECTED_SELECTED_PAPERS,
    }
    return MetaSynPilotSelectionConfigV1.model_validate(
        {**payload, "selection_config_sha256": hash_canonical(payload)}
    )


class MetaSynPilotQuestionSpecV1(ContractModel):
    question_spec_version: Literal["metasyn-typed-oracle-question-spec-v1"] = (
        QUESTION_SPEC_VERSION
    )
    question_id: Annotated[str, Field(pattern=r"^metasyn-review-[0-9]{6}$")]
    review_id: Annotated[int, Field(ge=0)]
    research_question: Annotated[
        str, Field(min_length=1, max_length=MAX_PROTOCOL_TEXT_CHARACTERS)
    ]
    population: Annotated[
        str, Field(min_length=1, max_length=MAX_PROTOCOL_TEXT_CHARACTERS)
    ]
    relation_kind: Literal["intervention", "exposure"]
    intervention_or_exposure: Annotated[
        str, Field(min_length=1, max_length=MAX_PROTOCOL_TEXT_CHARACTERS)
    ]
    comparison: Annotated[
        str, Field(min_length=1, max_length=MAX_PROTOCOL_TEXT_CHARACTERS)
    ]
    treatment_role: Literal["intervention_or_exposure"] = "intervention_or_exposure"
    comparator_role: Literal["comparator"] = "comparator"
    contrast_orientation: Literal[
        "intervention_or_exposure_minus_comparator"
    ] = "intervention_or_exposure_minus_comparator"
    contrast_estimand: Literal[
        "between_group_effect_intervention_or_exposure_vs_comparator_on_reported_measure"
    ] = (
        "between_group_effect_intervention_or_exposure_vs_comparator_on_reported_measure"
    )
    canonical_outcomes: Annotated[
        list[CanonicalOutcomeV1], Field(min_length=1, max_length=1)
    ]
    outcome_id_to_text: dict[str, Annotated[str, Field(min_length=1)]]
    positive_direction_means_by_outcome_id: dict[
        str, Annotated[str, Field(min_length=1)]
    ]
    clinical_benefit_direction_by_outcome_id: dict[
        str, Annotated[str, Field(min_length=1)]
    ]
    effect_measure_harmonization_status: Literal[
        "not_prespecified_in_metasyn_protocol_metadata"
    ] = "not_prespecified_in_metasyn_protocol_metadata"
    inclusion_criteria: Annotated[str, Field(max_length=MAX_CRITERIA_CHARACTERS)] | None
    exclusion_criteria: Annotated[str, Field(max_length=MAX_CRITERIA_CHARACTERS)] | None
    search_end_date: Annotated[str, Field(max_length=MAX_SEARCH_DATE_CHARACTERS)] | None
    allowed_outcomes: Annotated[list[str], Field(min_length=1, max_length=1)]
    allowed_moderators: list[str]
    claim_direction_interpretation: Literal[
        "not_opened_or_inferred_during_prepare"
    ] = "not_opened_or_inferred_during_prepare"
    directional_evaluation_eligible: Literal[False] = False
    directional_evaluation_blocker: Literal[
        "clinical_benefit_polarity_and_harmonized_effect_measure_not_prespecified_in_protocol_metadata"
    ] = (
        "clinical_benefit_polarity_and_harmonized_effect_measure_not_prespecified_in_protocol_metadata"
    )
    protocol_row_sha256: str
    question_spec_sha256: str

    @field_validator("protocol_row_sha256", "question_spec_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_pilot_question_sha256_invalid:{info.field_name}")
        return value

    @field_validator("inclusion_criteria", "exclusion_criteria", "search_end_date")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "outcome_id_to_text",
        "positive_direction_means_by_outcome_id",
        "clinical_benefit_direction_by_outcome_id",
    )
    @classmethod
    def validate_outcome_maps(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())):
            raise ValueError("metasyn_pilot_outcome_map_not_sorted")
        return value

    @model_validator(mode="after")
    def validate_question(self) -> MetaSynPilotQuestionSpecV1:
        if self.question_id != f"metasyn-review-{self.review_id:06d}":
            raise ValueError("metasyn_pilot_question_review_identity_mismatch")
        expected_ids = [item.outcome_id for item in self.canonical_outcomes]
        if self.allowed_outcomes != expected_ids:
            raise ValueError("metasyn_pilot_allowed_outcome_mismatch")
        expected_text = {
            item.outcome_id: item.outcome_text for item in self.canonical_outcomes
        }
        if self.outcome_id_to_text != expected_text:
            raise ValueError("metasyn_pilot_outcome_text_map_mismatch")
        expected_direction = {
            item.outcome_id: item.positive_direction_means
            for item in self.canonical_outcomes
        }
        if self.positive_direction_means_by_outcome_id != expected_direction:
            raise ValueError("metasyn_pilot_outcome_direction_map_mismatch")
        if set(expected_direction.values()) != {
            "higher_reported_outcome_value_or_event_frequency_in_intervention_or_exposure_than_comparator"
        }:
            raise ValueError("metasyn_pilot_outcome_raw_direction_semantics_mismatch")
        expected_benefit = {
            outcome_id: "not_prespecified_from_protocol_metadata"
            for outcome_id in expected_ids
        }
        if self.clinical_benefit_direction_by_outcome_id != expected_benefit:
            raise ValueError("metasyn_pilot_outcome_benefit_direction_mismatch")
        if self.allowed_moderators:
            raise ValueError("metasyn_pilot_moderators_must_be_empty")
        payload = self.model_dump(mode="json", exclude={"question_spec_sha256"})
        if hash_canonical(payload) != self.question_spec_sha256:
            raise ValueError("metasyn_pilot_question_spec_hash_mismatch")
        return self


class MetaSynPilotSourceProjectionRowV1(ContractModel):
    source_row_version: Literal["metasyn-typed-oracle-source-row-v1"] = SOURCE_ROW_VERSION
    question_id: Annotated[str, Field(min_length=1)]
    corpus_id: Annotated[int, Field(ge=0)]
    doc_id: Annotated[str, Field(min_length=1)]
    source_record: NativeSourceRecord
    diagnostic_source_record_sha256: str
    source_content_scope: Literal["full_text_sections", "title_abstract"]
    oracle_selection_full_text_scope: bool
    source_projection_strength: Literal[
        "full_text_textual_grounding",
        "diagnostic_title_abstract_grounding",
        "diagnostic_unrecognized_sections",
        "no_eligible_source_passage",
    ]
    release_grade_source_grounding_eligible: bool
    source_strength_blockers: list[str]
    projection: FrozenSourceProjectionV1
    projection_sha256: str
    source_row_sha256: str

    @field_validator(
        "diagnostic_source_record_sha256",
        "projection_sha256",
        "source_row_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_pilot_source_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> MetaSynPilotSourceProjectionRowV1:
        expected_doc_id = f"metasyn-corpus:{self.corpus_id}"
        if self.doc_id != expected_doc_id or self.source_record.doc_id != expected_doc_id:
            raise ValueError("metasyn_pilot_source_doc_identity_mismatch")
        if self.projection.row_id != self.doc_id:
            raise ValueError("metasyn_pilot_source_projection_row_mismatch")
        if self.projection.question_id != self.question_id:
            raise ValueError("metasyn_pilot_source_projection_question_mismatch")
        if self.projection.source_locator != self.source_record.source_document.source_locator:
            raise ValueError("metasyn_pilot_source_projection_locator_mismatch")
        if self.projection.artifact_sha256 != self.source_record.source_document.sha256:
            raise ValueError("metasyn_pilot_source_projection_artifact_mismatch")
        if self.source_content_scope == "full_text_sections":
            expected_modalities = {
                "full_text_recognized_sections",
                "full_text_unrecognized_sections",
            }
        else:
            expected_modalities = {"title_abstract", "abstract_only", "title_only"}
        if self.projection.source_modality not in expected_modalities:
            raise ValueError("metasyn_pilot_source_modality_mismatch")
        if self.projection_sha256 != self.projection.projection_sha256:
            raise ValueError("metasyn_pilot_projection_hash_alias_mismatch")
        if self.oracle_selection_full_text_scope != (
            self.source_content_scope == "full_text_sections"
        ):
            raise ValueError("metasyn_pilot_source_completeness_flag_mismatch")
        if self.source_projection_strength != self.projection.source_strength:
            raise ValueError("metasyn_pilot_source_strength_alias_mismatch")
        if (
            self.release_grade_source_grounding_eligible
            != self.projection.release_grade_source_grounding_eligible
        ):
            raise ValueError("metasyn_pilot_release_grade_source_alias_mismatch")
        if self.source_strength_blockers != self.projection.source_strength_blockers:
            raise ValueError("metasyn_pilot_source_blockers_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"source_row_sha256"})
        if hash_canonical(payload) != self.source_row_sha256:
            raise ValueError("metasyn_pilot_source_row_hash_mismatch")
        return self


class MetaSynPilotQuestionBundleV1(ContractModel):
    question_bundle_version: Literal["metasyn-typed-oracle-question-bundle-v1"] = (
        QUESTION_BUNDLE_VERSION
    )
    question_spec: MetaSynPilotQuestionSpecV1
    question_spec_sha256: str
    projection_spec: QuestionProjectionSpecV1
    projection_spec_sha256: str
    independence_component_id: Annotated[str, Field(min_length=1)]
    independence_component_review_ids: Annotated[list[int], Field(min_length=1)]
    independence_component_membership_sha256: str
    oracle_corpus_ids: Annotated[list[int], Field(min_length=MIN_MATCHED_PAPERS)]
    oracle_roster_membership_sha256: str
    source_manifest: NativeSourceManifest
    source_manifest_sha256: str
    source_rows: Annotated[list[MetaSynPilotSourceProjectionRowV1], Field(min_length=1)]
    question_bundle_sha256: str

    @field_validator(
        "question_spec_sha256",
        "projection_spec_sha256",
        "independence_component_membership_sha256",
        "oracle_roster_membership_sha256",
        "source_manifest_sha256",
        "question_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_pilot_bundle_sha256_invalid:{info.field_name}")
        return value

    @field_validator("oracle_corpus_ids")
    @classmethod
    def validate_corpus_ids(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)) or any(item < 0 for item in value):
            raise ValueError("metasyn_pilot_oracle_ids_not_sorted_unique")
        if not MIN_MATCHED_PAPERS <= len(value) <= MAX_MATCHED_PAPERS:
            raise ValueError("metasyn_pilot_oracle_id_count_outside_selection")
        return value

    @field_validator("independence_component_review_ids")
    @classmethod
    def validate_component_review_ids(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)) or any(item < 0 for item in value):
            raise ValueError("metasyn_pilot_component_review_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynPilotQuestionBundleV1:
        question_id = self.question_spec.question_id
        if self.question_spec_sha256 != self.question_spec.question_spec_sha256:
            raise ValueError("metasyn_pilot_question_hash_alias_mismatch")
        if self.projection_spec_sha256 != self.projection_spec.projection_spec_sha256:
            raise ValueError("metasyn_pilot_projection_spec_hash_alias_mismatch")
        if self.projection_spec.question_id != question_id:
            raise ValueError("metasyn_pilot_projection_spec_question_mismatch")
        if self.projection_spec.allowed_outcomes != self.question_spec.allowed_outcomes:
            raise ValueError("metasyn_pilot_projection_question_outcome_mismatch")
        if self.projection_spec.question_fields.outcomes != self.question_spec.canonical_outcomes:
            raise ValueError("metasyn_pilot_projection_question_outcome_map_mismatch")
        if (
            self.projection_spec.question_fields.treatment_role
            != self.question_spec.treatment_role
            or self.projection_spec.question_fields.comparator_role
            != self.question_spec.comparator_role
            or self.projection_spec.question_fields.contrast_estimand
            != self.question_spec.contrast_estimand
        ):
            raise ValueError("metasyn_pilot_projection_estimand_semantics_mismatch")
        expected_component_hash = hash_canonical(self.independence_component_review_ids)
        if self.independence_component_membership_sha256 != expected_component_hash:
            raise ValueError("metasyn_pilot_component_membership_hash_mismatch")
        if self.question_spec.review_id not in self.independence_component_review_ids:
            raise ValueError("metasyn_pilot_question_outside_independence_component")
        if self.source_manifest.question_id != question_id:
            raise ValueError("metasyn_pilot_source_manifest_question_mismatch")
        if self.source_manifest_sha256 != hash_canonical(self.source_manifest):
            raise ValueError("metasyn_pilot_source_manifest_hash_mismatch")
        expected_doc_ids = [f"metasyn-corpus:{item}" for item in self.oracle_corpus_ids]
        if [record.doc_id for record in self.source_manifest.records] != sorted(
            expected_doc_ids
        ):
            raise ValueError("metasyn_pilot_source_manifest_oracle_roster_mismatch")
        if [row.corpus_id for row in self.source_rows] != self.oracle_corpus_ids:
            raise ValueError("metasyn_pilot_source_rows_oracle_roster_mismatch")
        if any(row.question_id != question_id for row in self.source_rows):
            raise ValueError("metasyn_pilot_source_row_question_mismatch")
        expected_roster_hash = hash_canonical(
            {"question_id": question_id, "oracle_corpus_ids": self.oracle_corpus_ids}
        )
        if self.oracle_roster_membership_sha256 != expected_roster_hash:
            raise ValueError("metasyn_pilot_oracle_roster_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"question_bundle_sha256"})
        if hash_canonical(payload) != self.question_bundle_sha256:
            raise ValueError("metasyn_pilot_question_bundle_hash_mismatch")
        return self


class MetaSynTypedPilotPrepareBundleV1(ContractModel):
    prepare_bundle_version: Literal["metasyn-typed-oracle-prepare-bundle-v1"] = (
        PREPARE_BUNDLE_VERSION
    )
    pilot_version: Literal["metasyn-typed-oracle-pilot-v1"] = PILOT_VERSION
    status: Literal["prepared_predictions_and_reference_fields_unopened"] = (
        "prepared_predictions_and_reference_fields_unopened"
    )
    pilot_pipeline_fingerprint: PipelineFingerprint
    pilot_pipeline_sha256: str
    downstream_verifier_pipeline_sha256: str
    official_native_extraction_schema_sha256: str
    selection_config: MetaSynPilotSelectionConfigV1
    selection_config_sha256: str
    access_state: MetaSynPilotAccessStateV1
    repository_inputs: dict[
        str, Annotated[str, Field(min_length=1, max_length=MAX_REPOSITORY_PATH_CHARACTERS)]
    ]
    repository_input_sha256s: dict[str, str]
    screening_fit_payload_sha256: str
    screening_winner_rankings_sha256: str
    corpus_source_revision: Annotated[str, Field(min_length=1)]
    calibration_source_inventory_sha256: str
    calibration_question_count: Literal[161] = EXPECTED_CALIBRATION_QUESTIONS
    calibration_component_count: Annotated[int, Field(ge=1)]
    selected_question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    selected_component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    selected_paper_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    selected_unique_paper_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    source_modality_counts: dict[str, Annotated[int, Field(ge=0)]]
    source_strength_counts: dict[str, Annotated[int, Field(ge=0)]]
    release_grade_source_grounding_count: Annotated[int, Field(ge=0)]
    selected_question_membership_sha256: str
    selected_component_membership_sha256: str
    selected_oracle_roster_membership_sha256: str
    questions: Annotated[list[MetaSynPilotQuestionBundleV1], Field(min_length=1)]
    prepare_bundle_sha256: str

    @field_validator(
        "selection_config_sha256",
        "pilot_pipeline_sha256",
        "downstream_verifier_pipeline_sha256",
        "official_native_extraction_schema_sha256",
        "screening_fit_payload_sha256",
        "screening_winner_rankings_sha256",
        "calibration_source_inventory_sha256",
        "selected_question_membership_sha256",
        "selected_component_membership_sha256",
        "selected_oracle_roster_membership_sha256",
        "prepare_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_pilot_prepare_sha256_invalid:{info.field_name}")
        return value

    @field_validator("repository_input_sha256s")
    @classmethod
    def validate_input_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())) or any(
            not SHA256_RE.fullmatch(item) for item in value.values()
        ):
            raise ValueError("metasyn_pilot_repository_hashes_invalid")
        return value

    @field_validator("repository_inputs")
    @classmethod
    def validate_repository_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        expected = {
            "corpus_manifest",
            "reviews_train",
            "screening_fit_receipt",
            "screening_winner_rankings",
        }
        if set(value) != expected or value != dict(sorted(value.items())):
            raise ValueError("metasyn_pilot_repository_inputs_invalid")
        for raw in value.values():
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
                raise ValueError("metasyn_pilot_repository_input_path_unsafe")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynTypedPilotPrepareBundleV1:
        if self.pilot_pipeline_sha256 != self.pilot_pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_pilot_pipeline_hash_alias_mismatch")
        if len(self.pilot_pipeline_fingerprint.components) != 1:
            raise ValueError("metasyn_pilot_pipeline_component_count_mismatch")
        component = self.pilot_pipeline_fingerprint.components[0]
        if component.component_id != "metasyn-typed-oracle-pilot":
            raise ValueError("metasyn_pilot_pipeline_component_identity_mismatch")
        if (
            component.settings.get("downstream_verifier_pipeline_sha256")
            != self.downstream_verifier_pipeline_sha256
        ):
            raise ValueError("metasyn_pilot_downstream_verifier_hash_mismatch")
        if (
            component.settings.get("official_native_extraction_schema_sha256")
            != self.official_native_extraction_schema_sha256
        ):
            raise ValueError("metasyn_pilot_official_schema_hash_mismatch")
        if self.selection_config_sha256 != self.selection_config.selection_config_sha256:
            raise ValueError("metasyn_pilot_selection_hash_alias_mismatch")
        question_ids = [row.question_spec.question_id for row in self.questions]
        if question_ids != sorted(set(question_ids)):
            raise ValueError("metasyn_pilot_questions_not_sorted_unique")
        components = [row.independence_component_id for row in self.questions]
        if len(components) != len(set(components)):
            raise ValueError("metasyn_pilot_selected_components_overlap")
        corpus_ids = [item for row in self.questions for item in row.oracle_corpus_ids]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError("metasyn_pilot_selected_papers_overlap")
        expected_counts = {
            "selected_question_count": len(self.questions),
            "selected_component_count": len(set(components)),
            "selected_paper_count": len(corpus_ids),
            "selected_unique_paper_count": len(set(corpus_ids)),
        }
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"metasyn_pilot_prepare_count_mismatch:{name}")
        modality = Counter(
            row.source_content_scope
            for question in self.questions
            for row in question.source_rows
        )
        if self.source_modality_counts != dict(sorted(modality.items())):
            raise ValueError("metasyn_pilot_source_modality_counts_mismatch")
        strengths = Counter(
            row.source_projection_strength
            for question in self.questions
            for row in question.source_rows
        )
        if self.source_strength_counts != dict(sorted(strengths.items())):
            raise ValueError("metasyn_pilot_source_strength_counts_mismatch")
        release_grade_count = sum(
            row.release_grade_source_grounding_eligible
            for question in self.questions
            for row in question.source_rows
        )
        if self.release_grade_source_grounding_count != release_grade_count:
            raise ValueError("metasyn_pilot_release_grade_source_count_mismatch")
        if self.selected_question_membership_sha256 != hash_canonical(question_ids):
            raise ValueError("metasyn_pilot_question_membership_hash_mismatch")
        if self.selected_component_membership_sha256 != hash_canonical(sorted(components)):
            raise ValueError("metasyn_pilot_component_membership_hash_mismatch")
        oracle_hashes = sorted(row.oracle_roster_membership_sha256 for row in self.questions)
        if self.selected_oracle_roster_membership_sha256 != hash_canonical(oracle_hashes):
            raise ValueError("metasyn_pilot_oracle_membership_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"prepare_bundle_sha256"})
        if hash_canonical(payload) != self.prepare_bundle_sha256:
            raise ValueError("metasyn_pilot_prepare_bundle_hash_mismatch")
        return self


class MetaSynTypedPilotPrepareReceiptV1(ContractModel):
    prepare_receipt_version: Literal["metasyn-typed-oracle-prepare-receipt-v1"] = (
        PREPARE_RECEIPT_VERSION
    )
    pilot_version: Literal["metasyn-typed-oracle-pilot-v1"] = PILOT_VERSION
    status: Literal["prepared_predictions_and_reference_fields_unopened"] = (
        "prepared_predictions_and_reference_fields_unopened"
    )
    scientific_role: Literal[
        "retrospective_nonpristine_oracle_corpus_extraction_synthesis_feasibility"
    ] = "retrospective_nonpristine_oracle_corpus_extraction_synthesis_feasibility"
    pilot_pipeline_sha256: str
    downstream_verifier_pipeline_sha256: str
    official_native_extraction_schema_sha256: str
    prepare_bundle_path: Literal["prepare-bundle.private.json"] = PREPARE_BUNDLE_FILENAME
    prepare_bundle_file_sha256: str
    prepare_bundle_sha256: str
    selection_config_sha256: str
    selected_question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    selected_component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    selected_paper_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    source_modality_counts: dict[str, Annotated[int, Field(ge=0)]]
    source_strength_counts: dict[str, Annotated[int, Field(ge=0)]]
    release_grade_source_grounding_count: Annotated[int, Field(ge=0)]
    selected_question_membership_sha256: str
    selected_component_membership_sha256: str
    selected_oracle_roster_membership_sha256: str
    reference_fields_unopened: Literal[True] = True
    model_predictions_opened: Literal[False] = False
    official_test_opened: Literal[False] = False
    pristine_holdout_eligible: Literal[False] = False
    directional_agreement_evaluation_eligible: Literal[False] = False
    permitted_scientific_outputs: Literal[
        "typed_extraction_grounding_and_synthesis_yield_only"
    ] = "typed_extraction_grounding_and_synthesis_yield_only"
    prepare_receipt_sha256: str

    @field_validator(
        "prepare_bundle_file_sha256",
        "prepare_bundle_sha256",
        "selection_config_sha256",
        "pilot_pipeline_sha256",
        "downstream_verifier_pipeline_sha256",
        "official_native_extraction_schema_sha256",
        "selected_question_membership_sha256",
        "selected_component_membership_sha256",
        "selected_oracle_roster_membership_sha256",
        "prepare_receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_pilot_receipt_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynTypedPilotPrepareReceiptV1:
        payload = self.model_dump(mode="json", exclude={"prepare_receipt_sha256"})
        if hash_canonical(payload) != self.prepare_receipt_sha256:
            raise ValueError("metasyn_pilot_prepare_receipt_hash_mismatch")
        return self


class _UnionFind:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def _safe_repository_file(path: Path, *, repository_root: Path) -> tuple[Path, str]:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise MetaSynTypedPilotError("metasyn_pilot_input_symlink_forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_input_outside_repository") from exc
    if not resolved.is_file():
        raise MetaSynTypedPilotError("metasyn_pilot_input_not_file")
    return resolved, relative


def _private_workspace(path: Path, *, repository_root: Path) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.exists() and candidate.is_symlink():
        raise MetaSynTypedPilotError("metasyn_pilot_workspace_symlink_forbidden")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root / "data" / "cache")
    except ValueError as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_workspace_must_be_private_cache") from exc
    return resolved


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynTypedPilotError(f"metasyn_pilot_json_invalid:{path.name}") from exc
    if not isinstance(value, dict):
        raise MetaSynTypedPilotError(f"metasyn_pilot_json_not_object:{path.name}")
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_rankings_unreadable") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise MetaSynTypedPilotError(f"metasyn_pilot_rankings_blank_row:{number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetaSynTypedPilotError(
                f"metasyn_pilot_rankings_invalid_json:{number}"
            ) from exc
        if not isinstance(value, dict):
            raise MetaSynTypedPilotError(f"metasyn_pilot_rankings_row_not_object:{number}")
        rows.append(value)
    return rows


def _load_calibration_review_ids(
    *, screening_work_dir: Path
) -> tuple[list[int], dict[str, Any], Path]:
    fit = validate_fit(work_dir=screening_work_dir)
    relative = fit.get("winner_rankings_path")
    if not isinstance(relative, str):
        raise MetaSynTypedPilotError("metasyn_pilot_winner_rankings_path_invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MetaSynTypedPilotError("metasyn_pilot_winner_rankings_path_unsafe")
    rankings_path = screening_work_dir / relative_path
    if rankings_path.is_symlink() or sha256_file(rankings_path) != fit.get(
        "winner_rankings_sha256"
    ):
        raise MetaSynTypedPilotError("metasyn_pilot_winner_rankings_hash_mismatch")
    rows = _jsonl_objects(rankings_path)
    selected_candidate = fit.get("selected_candidate")
    review_ids: list[int] = []
    for row in rows:
        if row.get("split") != "calibration":
            continue
        if set(row) != {
            "split",
            "review_id",
            "selected_candidate",
            "ordered_corpus_ids",
            "original_rrf_ordered_corpus_ids",
        }:
            raise MetaSynTypedPilotError("metasyn_pilot_winner_ranking_schema_invalid")
        if row.get("selected_candidate") != selected_candidate:
            raise MetaSynTypedPilotError("metasyn_pilot_winner_candidate_mismatch")
        review_id = row.get("review_id")
        if not isinstance(review_id, int) or isinstance(review_id, bool) or review_id < 0:
            raise MetaSynTypedPilotError("metasyn_pilot_winner_review_id_invalid")
        review_ids.append(review_id)
    if review_ids != sorted(set(review_ids)):
        raise MetaSynTypedPilotError("metasyn_pilot_calibration_review_ids_not_sorted_unique")
    if len(review_ids) != EXPECTED_CALIBRATION_QUESTIONS:
        raise MetaSynTypedPilotError("metasyn_pilot_calibration_question_count_mismatch")
    return review_ids, fit, rankings_path


def _optional_text(value: Any, *, field: str, review_id: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_review_text_invalid:{field}:review={review_id}"
        )
    normalized = value.strip()
    maximum = (
        MAX_CRITERIA_CHARACTERS
        if field in {"inclusion_criteria", "exclusion_criteria"}
        else MAX_SEARCH_DATE_CHARACTERS
        if field == "search_end_date"
        else MAX_PROTOCOL_TEXT_CHARACTERS
    )
    if len(normalized) > maximum:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_review_text_exceeds_cap:{field}:review={review_id}"
        )
    return normalized or None


def _integer_ids(value: Any, *, field: str, review_id: int) -> list[int]:
    if value is None:
        raw: list[Any] = []
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        raw = converted if isinstance(converted, list) else [converted]
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_review_ids_invalid:{field}:review={review_id}"
        )
    try:
        ids = sorted({int(item) for item in raw})
    except (TypeError, ValueError) as exc:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_review_ids_invalid:{field}:review={review_id}"
        ) from exc
    if any(item < 0 for item in ids):
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_review_ids_negative:{field}:review={review_id}"
        )
    return ids


def _load_protocol_rows(
    *,
    reviews_train_path: Path,
    review_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Materialize the exact allowlisted calibration columns and no reference values."""

    if reviews_train_path.name != "reviews-train.parquet":
        raise MetaSynTypedPilotError("metasyn_pilot_nontraining_review_table_forbidden")
    try:
        table = pq.read_table(
            reviews_train_path,
            columns=list(MATERIALIZED_REVIEW_COLUMNS),
            filters=[("ID", "in", list(review_ids))],
        )
        raw_rows = table.to_pylist()
    except Exception as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_protocol_rows_unreadable") from exc
    observed_ids = [row.get("ID") for row in raw_rows]
    if (
        any(not isinstance(item, int) or isinstance(item, bool) for item in observed_ids)
        or sorted(observed_ids) != list(review_ids)
        or len(observed_ids) != len(set(observed_ids))
    ):
        raise MetaSynTypedPilotError("metasyn_pilot_protocol_row_universe_mismatch")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if set(raw) != set(MATERIALIZED_REVIEW_COLUMNS):
            raise MetaSynTypedPilotError("metasyn_pilot_protocol_row_columns_mismatch")
        review_id = int(raw["ID"])
        matched = _integer_ids(
            raw["matched_corpus_ids"],
            field="matched_corpus_ids",
            review_id=review_id,
        )
        matched_count = raw["matched_ref_count"]
        if (
            not isinstance(matched_count, int)
            or isinstance(matched_count, bool)
            or matched_count != len(matched)
            or not matched
        ):
            raise MetaSynTypedPilotError(
                f"metasyn_pilot_matched_count_mismatch:review={review_id}"
            )
        normalized = {
            "ID": review_id,
            "Title": _optional_text(raw["Title"], field="Title", review_id=review_id),
            "Research_Question": _optional_text(
                raw["Research_Question"], field="Research_Question", review_id=review_id
            ),
            "Population": _optional_text(
                raw["Population"], field="Population", review_id=review_id
            ),
            "Intervention": _optional_text(
                raw["Intervention"], field="Intervention", review_id=review_id
            ),
            "Exposure": _optional_text(
                raw["Exposure"], field="Exposure", review_id=review_id
            ),
            "Comparison": _optional_text(
                raw["Comparison"], field="Comparison", review_id=review_id
            ),
            "Outcome": _optional_text(raw["Outcome"], field="Outcome", review_id=review_id),
            "inclusion_criteria": _optional_text(
                raw["inclusion_criteria"], field="inclusion_criteria", review_id=review_id
            ),
            "exclusion_criteria": _optional_text(
                raw["exclusion_criteria"], field="exclusion_criteria", review_id=review_id
            ),
            "search_end_date": _optional_text(
                raw["search_end_date"], field="search_end_date", review_id=review_id
            ),
            "matched_corpus_ids": matched,
            "matched_ref_count": len(matched),
            "source_review_corpus_ids": _integer_ids(
                raw["source_review_corpus_ids"],
                field="source_review_corpus_ids",
                review_id=review_id,
            ),
        }
        rows.append(normalized)
    return sorted(rows, key=lambda row: int(row["ID"]))


def _normalized_link_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _component_assignments(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[str, list[int], str]]:
    review_ids = [int(row["ID"]) for row in rows]
    union_find = _UnionFind(review_ids)
    owner: dict[str, int] = {}
    for row in rows:
        review_id = int(row["ID"])
        tokens = [f"paper:{item}" for item in row["matched_corpus_ids"]]
        tokens.extend(f"source-review:{item}" for item in row["source_review_corpus_ids"])
        for prefix, field in (("title", "Title"), ("question", "Research_Question")):
            normalized = _normalized_link_text(row[field])
            if normalized:
                tokens.append(f"{prefix}:{normalized}")
        for token in tokens:
            prior = owner.setdefault(token, review_id)
            union_find.union(review_id, prior)
    grouped: dict[int, list[int]] = defaultdict(list)
    for review_id in review_ids:
        grouped[union_find.find(review_id)].append(review_id)
    output: dict[int, tuple[str, list[int], str]] = {}
    for members in grouped.values():
        members = sorted(members)
        membership_sha256 = hash_canonical(members)
        component_id = f"metasyn-component-{membership_sha256[:20]}"
        for review_id in members:
            output[review_id] = (component_id, members, membership_sha256)
    return output


def _select_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_scope_by_corpus_id: Mapping[int, SourceContentScope],
    config: MetaSynPilotSelectionConfigV1,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        matched = list(row["matched_corpus_ids"])
        if not config.matched_papers_minimum <= len(matched) <= config.matched_papers_maximum:
            continue
        scopes = [source_scope_by_corpus_id.get(item) for item in matched]
        if any(scope is None or scope is SourceContentScope.UNAVAILABLE for scope in scopes):
            continue
        if sum(scope is SourceContentScope.FULL_TEXT_SECTIONS for scope in scopes) < (
            config.full_text_papers_minimum
        ):
            continue
        if not row.get("Research_Question") or not row.get("Population"):
            continue
        intervention = bool(row.get("Intervention"))
        exposure = bool(row.get("Exposure"))
        if intervention == exposure:
            continue
        if not row.get("Comparison") or not row.get("Outcome"):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: int(row["ID"]))


def _question_spec(row: Mapping[str, Any]) -> MetaSynPilotQuestionSpecV1:
    review_id = int(row["ID"])
    intervention = row.get("Intervention")
    exposure = row.get("Exposure")
    relation_kind = "intervention" if intervention else "exposure"
    relation_value = intervention or exposure
    assert isinstance(relation_value, str)
    outcome_text = row["Outcome"]
    assert isinstance(outcome_text, str)
    canonical_outcomes = freeze_canonical_outcomes(
        outcome_texts=[outcome_text],
        positive_direction_means_by_outcome={
            outcome_text: (
                "higher_reported_outcome_value_or_event_frequency_in_"
                "intervention_or_exposure_than_comparator"
            )
        },
    )
    protocol_payload = {column: row[column] for column in MATERIALIZED_REVIEW_COLUMNS}
    payload: dict[str, Any] = {
        "question_spec_version": QUESTION_SPEC_VERSION,
        "question_id": f"metasyn-review-{review_id:06d}",
        "review_id": review_id,
        "research_question": row["Research_Question"],
        "population": row["Population"],
        "relation_kind": relation_kind,
        "intervention_or_exposure": relation_value,
        "comparison": row["Comparison"],
        "treatment_role": "intervention_or_exposure",
        "comparator_role": "comparator",
        "contrast_orientation": "intervention_or_exposure_minus_comparator",
        "contrast_estimand": (
            "between_group_effect_intervention_or_exposure_vs_comparator_on_"
            "reported_measure"
        ),
        "canonical_outcomes": canonical_outcomes,
        "outcome_id_to_text": {
            item.outcome_id: item.outcome_text for item in canonical_outcomes
        },
        "positive_direction_means_by_outcome_id": {
            item.outcome_id: item.positive_direction_means
            for item in canonical_outcomes
        },
        "clinical_benefit_direction_by_outcome_id": {
            item.outcome_id: "not_prespecified_from_protocol_metadata"
            for item in canonical_outcomes
        },
        "effect_measure_harmonization_status": (
            "not_prespecified_in_metasyn_protocol_metadata"
        ),
        "inclusion_criteria": row["inclusion_criteria"],
        "exclusion_criteria": row["exclusion_criteria"],
        "search_end_date": row["search_end_date"],
        "allowed_outcomes": [item.outcome_id for item in canonical_outcomes],
        "allowed_moderators": [],
        "claim_direction_interpretation": "not_opened_or_inferred_during_prepare",
        "directional_evaluation_eligible": False,
        "directional_evaluation_blocker": (
            "clinical_benefit_polarity_and_harmonized_effect_measure_not_"
            "prespecified_in_protocol_metadata"
        ),
        "protocol_row_sha256": hash_canonical(protocol_payload),
    }
    return MetaSynPilotQuestionSpecV1.model_validate(
        {**payload, "question_spec_sha256": hash_canonical(payload)}
    )


def _freeze_source_row(
    *,
    question_id: str,
    corpus_id: int,
    source_record: NativeSourceRecord,
    diagnostic_record: DiagnosticSourceRecord,
    projection: FrozenSourceProjectionV1,
) -> MetaSynPilotSourceProjectionRowV1:
    scope = diagnostic_record.content_scope.value
    if scope not in {"full_text_sections", "title_abstract"}:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_selected_source_scope_unsupported:{scope}"
        )
    payload: dict[str, Any] = {
        "source_row_version": SOURCE_ROW_VERSION,
        "question_id": question_id,
        "corpus_id": corpus_id,
        "doc_id": source_record.doc_id,
        "source_record": source_record,
        "diagnostic_source_record_sha256": diagnostic_record.record_sha256,
        "source_content_scope": scope,
        "oracle_selection_full_text_scope": scope == "full_text_sections",
        "source_projection_strength": projection.source_strength,
        "release_grade_source_grounding_eligible": (
            projection.release_grade_source_grounding_eligible
        ),
        "source_strength_blockers": projection.source_strength_blockers,
        "projection": projection,
        "projection_sha256": projection.projection_sha256,
    }
    return MetaSynPilotSourceProjectionRowV1.model_validate(
        {**payload, "source_row_sha256": hash_canonical(payload)}
    )


def _build_prepare_bundle(
    *,
    repository_root: Path,
    screening_work_dir: Path,
    reviews_train_path: Path,
    corpus_manifest_path: Path,
) -> MetaSynTypedPilotPrepareBundleV1:
    root = repository_root.resolve(strict=True)
    pilot_pipeline = compute_metasyn_typed_pilot_pipeline_fingerprint(root=root)
    pipeline_settings = pilot_pipeline.components[0].settings
    downstream_verifier_sha256 = pipeline_settings.get(
        "downstream_verifier_pipeline_sha256"
    )
    official_schema_sha256 = pipeline_settings.get(
        "official_native_extraction_schema_sha256"
    )
    if not isinstance(downstream_verifier_sha256, str) or not isinstance(
        official_schema_sha256, str
    ):
        raise MetaSynTypedPilotError("metasyn_pilot_pipeline_settings_missing")
    screening_dir = (
        screening_work_dir
        if screening_work_dir.is_absolute()
        else root / screening_work_dir
    ).resolve(strict=True)
    reviews_file, reviews_relative = _safe_repository_file(
        reviews_train_path, repository_root=root
    )
    corpus_manifest_file, corpus_manifest_relative = _safe_repository_file(
        corpus_manifest_path, repository_root=root
    )
    observed_reviews_sha256 = sha256_file(reviews_file)
    if observed_reviews_sha256 != EXPECTED_REVIEWS_TRAIN_SHA256:
        raise MetaSynTypedPilotError("metasyn_pilot_reviews_train_hash_mismatch")
    review_ids, fit, rankings_path = _load_calibration_review_ids(
        screening_work_dir=screening_dir
    )
    rows = _load_protocol_rows(
        reviews_train_path=reviews_file,
        review_ids=review_ids,
    )
    components = _component_assignments(rows)
    all_matched_ids = {
        item for row in rows for item in row["matched_corpus_ids"]
    }
    inventory_manifest, inventory_ledger = build_metasyn_native_source_bridge(
        question_id="metasyn-typed-pilot",
        corpus_manifest_path=corpus_manifest_file,
        repository_root=root,
        corpus_ids=all_matched_ids,
    )
    manifest_by_id = {
        int(record.doc_id.removeprefix("metasyn-corpus:")): record
        for record in inventory_manifest.records
    }
    ledger_by_id = {
        int(record.doc_id.removeprefix("metasyn-corpus:")): record
        for record in inventory_ledger.records
    }
    if set(ledger_by_id) != all_matched_ids:
        raise MetaSynTypedPilotError("metasyn_pilot_source_inventory_membership_mismatch")
    source_scopes = {item: record.content_scope for item, record in ledger_by_id.items()}
    config = freeze_metasyn_pilot_selection_config()
    selected = _select_rows(
        rows=rows,
        source_scope_by_corpus_id=source_scopes,
        config=config,
    )
    selected_components = [components[int(row["ID"])][0] for row in selected]
    selected_papers = [item for row in selected for item in row["matched_corpus_ids"]]
    if len(rows) != config.expected_calibration_questions:
        raise MetaSynTypedPilotError("metasyn_pilot_calibration_rows_mismatch")
    if len(selected) != config.expected_selected_questions:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_selected_question_count_mismatch:{len(selected)}"
        )
    if len(set(selected_components)) != config.expected_selected_components:
        raise MetaSynTypedPilotError("metasyn_pilot_selected_component_count_mismatch")
    if len(selected_papers) != config.expected_selected_papers:
        raise MetaSynTypedPilotError(
            f"metasyn_pilot_selected_paper_count_mismatch:{len(selected_papers)}"
        )
    if len(set(selected_papers)) != len(selected_papers):
        raise MetaSynTypedPilotError("metasyn_pilot_selected_paper_overlap")

    question_bundles: list[MetaSynPilotQuestionBundleV1] = []
    for row in selected:
        spec = _question_spec(row)
        projection_spec = freeze_question_projection_spec(
            question_id=spec.question_id,
            population=spec.population,
            intervention_or_exposure=spec.intervention_or_exposure,
            comparison=spec.comparison,
            outcome_texts=[item.outcome_text for item in spec.canonical_outcomes],
            treatment_role=spec.treatment_role,
            comparator_role=spec.comparator_role,
            contrast_estimand=spec.contrast_estimand,
            positive_direction_means_by_outcome={
                item.outcome_text: item.positive_direction_means
                for item in spec.canonical_outcomes
            },
            allowed_moderators=spec.allowed_moderators,
        )
        corpus_ids = list(row["matched_corpus_ids"])
        records = sorted(
            (manifest_by_id[item] for item in corpus_ids),
            key=lambda item: item.doc_id,
        )
        source_manifest = NativeSourceManifest(
            question_id=spec.question_id,
            records=records,
        )
        source_rows: list[MetaSynPilotSourceProjectionRowV1] = []
        for corpus_id in corpus_ids:
            source_record = manifest_by_id[corpus_id]
            resolved = resolve_native_source_document(
                repository_root=root,
                source_document=source_record.source_document,
            )
            projection = project_resolved_source_for_question(
                row_id=source_record.doc_id,
                source=resolved,
                spec=projection_spec,
            )
            validate_frozen_source_projection_external_replay(
                projection=projection,
                source=resolved,
                spec=projection_spec,
            )
            source_rows.append(
                _freeze_source_row(
                    question_id=spec.question_id,
                    corpus_id=corpus_id,
                    source_record=source_record,
                    diagnostic_record=ledger_by_id[corpus_id],
                    projection=projection,
                )
            )
        (
            component_id,
            component_review_ids,
            component_membership_sha256,
        ) = components[spec.review_id]
        roster_hash = hash_canonical(
            {"question_id": spec.question_id, "oracle_corpus_ids": corpus_ids}
        )
        question_payload: dict[str, Any] = {
            "question_bundle_version": QUESTION_BUNDLE_VERSION,
            "question_spec": spec,
            "question_spec_sha256": spec.question_spec_sha256,
            "projection_spec": projection_spec,
            "projection_spec_sha256": projection_spec.projection_spec_sha256,
            "independence_component_id": component_id,
            "independence_component_review_ids": component_review_ids,
            "independence_component_membership_sha256": component_membership_sha256,
            "oracle_corpus_ids": corpus_ids,
            "oracle_roster_membership_sha256": roster_hash,
            "source_manifest": source_manifest,
            "source_manifest_sha256": hash_canonical(source_manifest),
            "source_rows": source_rows,
        }
        question_bundles.append(
            MetaSynPilotQuestionBundleV1.model_validate(
                {
                    **question_payload,
                    "question_bundle_sha256": hash_canonical(question_payload),
                }
            )
        )
    question_bundles.sort(key=lambda item: item.question_spec.question_id)
    question_ids = [item.question_spec.question_id for item in question_bundles]
    component_ids = [item.independence_component_id for item in question_bundles]
    oracle_hashes = sorted(
        item.oracle_roster_membership_sha256 for item in question_bundles
    )
    modality_counts = dict(
        sorted(
            Counter(
                source_row.source_content_scope
                for question in question_bundles
                for source_row in question.source_rows
            ).items()
        )
    )
    strength_counts = dict(
        sorted(
            Counter(
                source_row.source_projection_strength
                for question in question_bundles
                for source_row in question.source_rows
            ).items()
        )
    )
    release_grade_source_count = sum(
        source_row.release_grade_source_grounding_eligible
        for question in question_bundles
        for source_row in question.source_rows
    )
    try:
        screening_relative = screening_dir.relative_to(root).as_posix()
        rankings_relative = rankings_path.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_screening_input_outside_repository") from exc
    repository_inputs = {
        "corpus_manifest": corpus_manifest_relative,
        "reviews_train": reviews_relative,
        "screening_fit_receipt": f"{screening_relative}/fit_receipt.json",
        "screening_winner_rankings": rankings_relative,
    }
    repository_hashes = {
        "corpus_manifest": sha256_file(corpus_manifest_file),
        "reviews_train": observed_reviews_sha256,
        "screening_fit_receipt": sha256_file(screening_dir / "fit_receipt.json"),
        "screening_winner_rankings": sha256_file(rankings_path),
    }
    access = MetaSynPilotAccessStateV1(
        materialized_review_columns=list(MATERIALIZED_REVIEW_COLUMNS),
        forbidden_reference_columns=sorted(FORBIDDEN_REFERENCE_COLUMNS),
    )
    bundle_payload: dict[str, Any] = {
        "prepare_bundle_version": PREPARE_BUNDLE_VERSION,
        "pilot_version": PILOT_VERSION,
        "status": "prepared_predictions_and_reference_fields_unopened",
        "pilot_pipeline_fingerprint": pilot_pipeline,
        "pilot_pipeline_sha256": pilot_pipeline.pipeline_sha256,
        "downstream_verifier_pipeline_sha256": downstream_verifier_sha256,
        "official_native_extraction_schema_sha256": official_schema_sha256,
        "selection_config": config,
        "selection_config_sha256": config.selection_config_sha256,
        "access_state": access,
        "repository_inputs": dict(sorted(repository_inputs.items())),
        "repository_input_sha256s": dict(sorted(repository_hashes.items())),
        "screening_fit_payload_sha256": fit["fit_payload_sha256"],
        "screening_winner_rankings_sha256": fit["winner_rankings_sha256"],
        "corpus_source_revision": inventory_ledger.source_revision,
        "calibration_source_inventory_sha256": inventory_ledger.ledger_sha256,
        "calibration_question_count": len(rows),
        "calibration_component_count": len(set(value[0] for value in components.values())),
        "selected_question_count": len(question_bundles),
        "selected_component_count": len(set(component_ids)),
        "selected_paper_count": len(selected_papers),
        "selected_unique_paper_count": len(set(selected_papers)),
        "source_modality_counts": modality_counts,
        "source_strength_counts": strength_counts,
        "release_grade_source_grounding_count": release_grade_source_count,
        "selected_question_membership_sha256": hash_canonical(question_ids),
        "selected_component_membership_sha256": hash_canonical(sorted(component_ids)),
        "selected_oracle_roster_membership_sha256": hash_canonical(oracle_hashes),
        "questions": question_bundles,
    }
    return MetaSynTypedPilotPrepareBundleV1.model_validate(
        {**bundle_payload, "prepare_bundle_sha256": hash_canonical(bundle_payload)}
    )


def _freeze_receipt(
    *, bundle: MetaSynTypedPilotPrepareBundleV1, bundle_path: Path
) -> MetaSynTypedPilotPrepareReceiptV1:
    payload: dict[str, Any] = {
        "prepare_receipt_version": PREPARE_RECEIPT_VERSION,
        "pilot_version": PILOT_VERSION,
        "status": bundle.status,
        "scientific_role": bundle.access_state.scientific_role,
        "pilot_pipeline_sha256": bundle.pilot_pipeline_sha256,
        "downstream_verifier_pipeline_sha256": (
            bundle.downstream_verifier_pipeline_sha256
        ),
        "official_native_extraction_schema_sha256": (
            bundle.official_native_extraction_schema_sha256
        ),
        "prepare_bundle_path": PREPARE_BUNDLE_FILENAME,
        "prepare_bundle_file_sha256": sha256_file(bundle_path),
        "prepare_bundle_sha256": bundle.prepare_bundle_sha256,
        "selection_config_sha256": bundle.selection_config_sha256,
        "selected_question_count": bundle.selected_question_count,
        "selected_component_count": bundle.selected_component_count,
        "selected_paper_count": bundle.selected_paper_count,
        "source_modality_counts": bundle.source_modality_counts,
        "source_strength_counts": bundle.source_strength_counts,
        "release_grade_source_grounding_count": (
            bundle.release_grade_source_grounding_count
        ),
        "selected_question_membership_sha256": (
            bundle.selected_question_membership_sha256
        ),
        "selected_component_membership_sha256": (
            bundle.selected_component_membership_sha256
        ),
        "selected_oracle_roster_membership_sha256": (
            bundle.selected_oracle_roster_membership_sha256
        ),
        "reference_fields_unopened": True,
        "model_predictions_opened": False,
        "official_test_opened": False,
        "pristine_holdout_eligible": False,
        "directional_agreement_evaluation_eligible": False,
        "permitted_scientific_outputs": (
            "typed_extraction_grounding_and_synthesis_yield_only"
        ),
    }
    return MetaSynTypedPilotPrepareReceiptV1.model_validate(
        {**payload, "prepare_receipt_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_typed_pilot(
    *,
    repository_root: Path,
    screening_work_dir: Path,
    reviews_train_path: Path,
    corpus_manifest_path: Path,
    workspace: Path,
    force: bool = False,
) -> MetaSynTypedPilotPrepareReceiptV1:
    """Freeze the exact private 10-question/32-paper outcome-label-blind population."""

    root = repository_root.resolve(strict=True)
    private = _private_workspace(workspace, repository_root=root)
    bundle_path = private / PREPARE_BUNDLE_FILENAME
    receipt_path = private / PREPARE_RECEIPT_FILENAME
    existing = [path.as_posix() for path in (bundle_path, receipt_path) if path.exists()]
    if existing and not force:
        raise MetaSynTypedPilotError(f"metasyn_pilot_prepare_outputs_exist:{existing}")
    bundle = _build_prepare_bundle(
        repository_root=root,
        screening_work_dir=screening_work_dir,
        reviews_train_path=reviews_train_path,
        corpus_manifest_path=corpus_manifest_path,
    )
    atomic_write_json(bundle_path, bundle, force=force)
    receipt = _freeze_receipt(bundle=bundle, bundle_path=bundle_path)
    atomic_write_json(receipt_path, receipt, force=force)
    return receipt


def validate_metasyn_typed_pilot_prepare(
    *,
    repository_root: Path,
    workspace: Path,
) -> MetaSynTypedPilotPrepareReceiptV1:
    """Rebuild selection/projections from current inputs and compare exact identities."""

    root = repository_root.resolve(strict=True)
    private = _private_workspace(workspace, repository_root=root)
    bundle_path = private / PREPARE_BUNDLE_FILENAME
    receipt_path = private / PREPARE_RECEIPT_FILENAME
    bundle = MetaSynTypedPilotPrepareBundleV1.model_validate(_json_object(bundle_path))
    receipt = MetaSynTypedPilotPrepareReceiptV1.model_validate(_json_object(receipt_path))
    if receipt.prepare_bundle_file_sha256 != sha256_file(bundle_path):
        raise MetaSynTypedPilotError("metasyn_pilot_prepare_bundle_file_hash_mismatch")
    if receipt.prepare_bundle_sha256 != bundle.prepare_bundle_sha256:
        raise MetaSynTypedPilotError("metasyn_pilot_prepare_bundle_receipt_mismatch")
    expected_receipt = _freeze_receipt(bundle=bundle, bundle_path=bundle_path)
    if expected_receipt != receipt:
        raise MetaSynTypedPilotError("metasyn_pilot_prepare_receipt_projection_mismatch")
    inputs = bundle.repository_inputs
    try:
        screening_dir = (root / Path(inputs["screening_fit_receipt"])).parent
        reviews_path = root / Path(inputs["reviews_train"])
        corpus_manifest_path = root / Path(inputs["corpus_manifest"])
    except KeyError as exc:
        raise MetaSynTypedPilotError("metasyn_pilot_prepare_input_binding_missing") from exc
    rebuilt = _build_prepare_bundle(
        repository_root=root,
        screening_work_dir=screening_dir,
        reviews_train_path=reviews_path,
        corpus_manifest_path=corpus_manifest_path,
    )
    if rebuilt.prepare_bundle_sha256 != bundle.prepare_bundle_sha256 or rebuilt != bundle:
        raise MetaSynTypedPilotError("metasyn_pilot_prepare_external_replay_mismatch")
    return receipt


__all__ = [
    "EXPECTED_CALIBRATION_QUESTIONS",
    "EXPECTED_SELECTED_COMPONENTS",
    "EXPECTED_SELECTED_PAPERS",
    "EXPECTED_SELECTED_QUESTIONS",
    "FORBIDDEN_REFERENCE_COLUMNS",
    "MATERIALIZED_REVIEW_COLUMNS",
    "MetaSynPilotAccessStateV1",
    "MetaSynPilotQuestionBundleV1",
    "MetaSynPilotQuestionSpecV1",
    "MetaSynPilotSelectionConfigV1",
    "MetaSynPilotSourceProjectionRowV1",
    "MetaSynTypedPilotError",
    "MetaSynTypedPilotPrepareBundleV1",
    "MetaSynTypedPilotPrepareReceiptV1",
    "freeze_metasyn_pilot_selection_config",
    "prepare_metasyn_typed_pilot",
    "validate_metasyn_typed_pilot_prepare",
]
