"""Artifact-backed item-risk calibration for the grounded MetaSyn v2 pipeline.

This module is an additive, label-firewalled bridge between externally replayable
grounded terminal artifacts and the repository's conservative item-risk bounds.
It deliberately does *not* accept caller-supplied scores.  A fixed, prespecified
heuristic is recomputed from terminal, grounding, assembly, source-coverage, and
inventory receipts.  Its output is only a scheduling score; it is not an item
error probability and it has no synthesis or claim-release authority.

The full eligible roster retains at most one completed effect per
question/publication.  A hash-salted question split is frozen before any labels
are opened.  Calibration labels live in a physically separate complete-question
sidecar; evaluation labels are neither accepted nor opened.  The existing
Bonferroni/Clopper--Pearson implementation is reused for the final group-average
cell-rate upper confidence limits.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictBool, field_validator, model_validator

from literature_multiverse.item_risk_calibration import (
    AdjudicatedLabelSource,
    DomainRiskBinCalibration,
    FixedRiskBinFamily,
    ItemRiskCalibrationBundle,
    RiskBinSpec,
    calibrate_item_risk_bounds,
    make_fixed_risk_bin_family,
    seal_item_risk_calibration_unit,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_grounded_analysis_v2 import (
    MetaSynGroundedAnalysisV2,
    validate_metasyn_grounded_analysis_v2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
)

PREPARATION_VERSION = "metasyn-artifact-item-risk-preparation-v1"
SPLIT_VERSION = "metasyn-artifact-item-risk-question-split-v1"
ITEM_VERSION = "metasyn-artifact-item-risk-item-v1"
REPRESENTATIVE_VERSION = "metasyn-artifact-item-risk-representative-v1"
SCORE_POLICY_VERSION = "metasyn-terminal-risk-score-policy-v1"
FEATURE_ROW_VERSION = "metasyn-terminal-risk-feature-row-v1"
FEATURE_SET_VERSION = "metasyn-terminal-risk-feature-set-v1"
SHIFT_VERSION = "metasyn-terminal-risk-shift-assessment-v1"
SIDECAR_MANIFEST_VERSION = "metasyn-adjudication-sidecar-manifest-v1"
SIDECAR_QUESTION_FILE_VERSION = "metasyn-question-adjudication-file-v1"
SIDECAR_VERSION = "metasyn-complete-question-adjudication-sidecar-v1"
CALIBRATION_RUN_VERSION = "metasyn-artifact-item-risk-calibration-run-v1"
ASSIGNMENT_VERSION = "metasyn-artifact-item-risk-assignment-v1"

MODULE_PATH = "src/literature_multiverse/metasyn_item_risk_calibration_v1.py"
CLI_PATH = "scripts/run_metasyn_item_risk_calibration_v1.py"
COMPONENT_ID = "metasyn-artifact-item-risk-calibration-v1"
COMPONENT_VERSION = "1"

MIN_CALIBRATION_QUESTIONS = 5
MIN_BOUND_QUESTIONS = 4
MIN_EVALUATION_QUESTIONS = 4
LEGACY_DEVELOPMENT_QUESTION_COUNT = 1
MAX_SIDECAR_BYTES = 16 * 1024 * 1024
FAMILYWISE_DELTA = 0.05
SHIFT_MAX_FEATURE_MEAN_DIFFERENCE = 0.35
RISK_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
POPULATION_ID = "metasyn-grounded-hosted-v2-terminal-effects"
PRESPECIFIED_SPLIT_SALT = "literature-multiverse:metasyn-item-risk:calibration-evaluation-split:v1"
ERROR_EVENT_DEFINITION = (
    "expert adjudication finds any prespecified scientific field, numerical value, "
    "effect representation, arm assignment, outcome binding, or exact source support "
    "in the selected grounded typed effect materially incorrect"
)

FEATURE_NAMES = (
    "candidate_rank_fraction",
    "computed_effect_indicator",
    "coverage_gap_fraction",
    "inventory_load_fraction",
    "passage_complexity_fraction",
    "projection_omission_fraction",
    "quote_brevity_fraction",
    "source_scope_risk",
)
FEATURE_WEIGHTS = {
    "candidate_rank_fraction": 0.08,
    "computed_effect_indicator": 0.15,
    "coverage_gap_fraction": 0.15,
    "inventory_load_fraction": 0.10,
    "passage_complexity_fraction": 0.10,
    "projection_omission_fraction": 0.12,
    "quote_brevity_fraction": 0.12,
    "source_scope_risk": 0.18,
}
SCORE_ASSUMPTIONS = [
    "fixed_prespecified_weights_use_no_adjudication_labels",
    "score_is_a_scheduling_rank_not_an_error_probability",
    "one_completed_terminal_selected_per_question_publication",
    "question_split_is_hash_salted_and_frozen_before_label_access",
    "evaluation_labels_are_never_opened_by_calibration",
    "label_blind_shift_check_does_not_prove_exchangeability",
]


class MetaSynItemRiskCalibrationV1Error(ValueError):
    """The artifact-backed calibration contract failed closed."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    if getattr(model, field_name) != hash_canonical(
        model.model_dump(mode="json", exclude={field_name})
    ):
        raise ValueError(f"metasyn_item_risk_v1_self_hash_mismatch:{field_name}")


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_repository_root_symlink")
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_repository_root_invalid"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_repository_root_not_directory"
        )
    return resolved


def _installed_version(name: str) -> str:
    try:
        return distribution_version(name)
    except PackageNotFoundError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            f"metasyn_item_risk_v1_dependency_missing:{name}"
        ) from exc


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_relative_import_invalid:{current_path}:{module}"
            )
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
    raise MetaSynItemRiskCalibrationV1Error(
        f"metasyn_item_risk_v1_local_dependency_missing:{current_path}:{module}"
    )


def _python_dependency_closure(repository_root: Path) -> list[str]:
    pending = [MODULE_PATH, CLI_PATH]
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source = repository_root / relative
        if not source.is_file():
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_dependency_missing:{relative}"
            )
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    non_python = [
        path for path in ("pyproject.toml", "uv.lock") if (repository_root / path).is_file()
    ]
    return sorted(observed | set(non_python))


def _salted_key(*, salt_sha256: str, purpose: str, value: str) -> str:
    return hash_canonical({"salt_sha256": salt_sha256, "purpose": purpose, "value": value})


def _split_question_ids(
    question_ids: Sequence[str], *, split_salt_sha256: str
) -> tuple[list[str], list[str]]:
    ranked = sorted(
        set(question_ids),
        key=lambda item: (
            _salted_key(
                salt_sha256=split_salt_sha256,
                purpose="calibration-evaluation-question-rank",
                value=item,
            ),
            item,
        ),
    )
    calibration_count = (len(ranked) + 1) // 2
    return sorted(ranked[:calibration_count]), sorted(ranked[calibration_count:])


class FrozenQuestionSplitV1(_FrozenExactModel):
    split_version: Literal["metasyn-artifact-item-risk-question-split-v1"] = SPLIT_VERSION
    split_salt: Literal[
        "literature-multiverse:metasyn-item-risk:calibration-evaluation-split:v1"
    ] = PRESPECIFIED_SPLIT_SALT
    split_salt_sha256: Sha256
    assignment_algorithm: Literal["sha256_salted_rank_balanced_calibration_first_v1"] = (
        "sha256_salted_rank_balanced_calibration_first_v1"
    )
    eligible_question_ids: list[str]
    calibration_question_ids: list[str]
    evaluation_question_ids: list[str]
    eligible_question_count: Annotated[int, Field(ge=0)]
    calibration_question_count: Annotated[int, Field(ge=0)]
    evaluation_question_count: Annotated[int, Field(ge=0)]
    question_disjoint: Literal[True] = True
    labels_used_for_assignment: Literal[False] = False
    split_sha256: Sha256

    @field_validator("eligible_question_ids", "calibration_question_ids", "evaluation_question_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("metasyn_item_risk_v1_split_ids_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_split(self) -> FrozenQuestionSplitV1:
        expected_salt = hash_canonical({"split_salt": self.split_salt})
        if self.split_salt_sha256 != expected_salt:
            raise ValueError("metasyn_item_risk_v1_split_salt_hash_mismatch")
        calibration, evaluation = _split_question_ids(
            self.eligible_question_ids, split_salt_sha256=self.split_salt_sha256
        )
        if (
            self.calibration_question_ids != calibration
            or self.evaluation_question_ids != evaluation
            or set(calibration) & set(evaluation)
            or sorted(calibration + evaluation) != self.eligible_question_ids
        ):
            raise ValueError("metasyn_item_risk_v1_split_assignment_mismatch")
        if (
            self.eligible_question_count,
            self.calibration_question_count,
            self.evaluation_question_count,
        ) != (len(self.eligible_question_ids), len(calibration), len(evaluation)):
            raise ValueError("metasyn_item_risk_v1_split_count_mismatch")
        _self_hash(self, "split_sha256")
        return self


class FixedTerminalRiskScorePolicyV1(_FrozenExactModel):
    policy_version: Literal["metasyn-terminal-risk-score-policy-v1"] = SCORE_POLICY_VERSION
    score_name: Literal["metasyn_terminal_artifact_risk_rank_v1"] = (
        "metasyn_terminal_artifact_risk_rank_v1"
    )
    definition_source: Literal["prespecified"] = "prespecified"
    learned_weights_enabled: Literal[False] = False
    feature_names: list[str]
    feature_weights: dict[str, Annotated[float, Field(ge=0, le=1)]]
    score_range: Literal["unit_interval"] = "unit_interval"
    risk_bin_edges: list[Annotated[float, Field(ge=0, le=1)]]
    assumptions: list[str]
    score_model_sha256: Sha256

    @model_validator(mode="after")
    def validate_policy(self) -> FixedTerminalRiskScorePolicyV1:
        if (
            self.feature_names != list(FEATURE_NAMES)
            or self.feature_weights != FEATURE_WEIGHTS
            or not math.isclose(sum(self.feature_weights.values()), 1.0)
            or self.risk_bin_edges != list(RISK_BIN_EDGES)
            or self.assumptions != SCORE_ASSUMPTIONS
        ):
            raise ValueError("metasyn_item_risk_v1_score_policy_not_prespecified")
        _self_hash(self, "score_model_sha256")
        return self


def freeze_fixed_terminal_risk_score_policy_v1() -> FixedTerminalRiskScorePolicyV1:
    payload = {
        "policy_version": SCORE_POLICY_VERSION,
        "score_name": "metasyn_terminal_artifact_risk_rank_v1",
        "definition_source": "prespecified",
        "learned_weights_enabled": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": dict(FEATURE_WEIGHTS),
        "score_range": "unit_interval",
        "risk_bin_edges": list(RISK_BIN_EDGES),
        "assumptions": SCORE_ASSUMPTIONS,
    }
    return FixedTerminalRiskScorePolicyV1.model_validate(
        {**payload, "score_model_sha256": hash_canonical(payload)}
    )


class FrozenEligibleRiskItemV1(_FrozenExactModel):
    item_version: Literal["metasyn-artifact-item-risk-item-v1"] = ITEM_VERSION
    item_id: str
    question_id: str
    publication_id: str
    paper_id: str
    domain: Literal["metasyn-exposure", "metasyn-intervention"]
    row_ordinal: Annotated[int, Field(ge=0)]
    row_key: str
    publication_join_sha256: Sha256
    source_strength_surface_sha256: Sha256
    inventory_receipt_sha256: Sha256
    candidate_index: Annotated[int, Field(ge=1)]
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    terminal_sha256: Sha256
    grounding_receipt_sha256: Sha256
    assembly_receipt_sha256: Sha256
    typed_effect_sha256: Sha256
    compatibility_sha256: Sha256
    item_sha256: Sha256

    @model_validator(mode="after")
    def validate_item(self) -> FrozenEligibleRiskItemV1:
        identity = self.model_dump(mode="json", exclude={"item_id", "item_sha256"})
        expected_id = f"metasyn-risk-item-{hash_canonical({'item_identity': identity})[:40]}"
        if self.item_id != expected_id:
            raise ValueError("metasyn_item_risk_v1_item_id_mismatch")
        _self_hash(self, "item_sha256")
        return self


class CalibrationRepresentativeV1(_FrozenExactModel):
    representative_version: Literal["metasyn-artifact-item-risk-representative-v1"] = (
        REPRESENTATIVE_VERSION
    )
    question_id: str
    item_id: str
    paper_id: str
    core_split: Literal["development", "calibration"]
    purpose: Literal[
        "structural_legacy_development_no_fitting",
        "clopper_pearson_bound_input",
    ]
    representative_sha256: Sha256

    @model_validator(mode="after")
    def validate_representative(self) -> CalibrationRepresentativeV1:
        if (self.core_split == "development") != (
            self.purpose == "structural_legacy_development_no_fitting"
        ):
            raise ValueError("metasyn_item_risk_v1_representative_role_mismatch")
        _self_hash(self, "representative_sha256")
        return self


def _representative_assignment(
    *,
    items: Sequence[FrozenEligibleRiskItemV1],
    calibration_question_ids: Sequence[str],
    split_salt_sha256: str,
) -> list[CalibrationRepresentativeV1]:
    if not calibration_question_ids:
        return []
    by_question: dict[str, list[FrozenEligibleRiskItemV1]] = defaultdict(list)
    for item in items:
        if item.question_id in set(calibration_question_ids):
            by_question[item.question_id].append(item)
    question_order = sorted(
        calibration_question_ids,
        key=lambda question_id: (
            _salted_key(
                salt_sha256=split_salt_sha256,
                purpose="unique-paper-representative-question-order",
                value=question_id,
            ),
            question_id,
        ),
    )
    candidates = {
        question_id: sorted(
            by_question.get(question_id, []),
            key=lambda item: (
                _salted_key(
                    salt_sha256=split_salt_sha256,
                    purpose="unique-paper-representative-item-order",
                    value=f"{question_id}:{item.item_id}",
                ),
                item.item_id,
            ),
        )
        for question_id in question_order
    }
    assignment: dict[str, FrozenEligibleRiskItemV1] = {}

    def search(index: int, used_papers: set[str]) -> bool:
        if index == len(question_order):
            return True
        question_id = question_order[index]
        for item in candidates[question_id]:
            if item.paper_id in used_papers:
                continue
            assignment[question_id] = item
            if search(index + 1, used_papers | {item.paper_id}):
                return True
            assignment.pop(question_id, None)
        return False

    if not search(0, set()):
        return []
    legacy_question = min(
        calibration_question_ids,
        key=lambda question_id: (
            _salted_key(
                salt_sha256=split_salt_sha256,
                purpose="structural-legacy-development-question",
                value=question_id,
            ),
            question_id,
        ),
    )
    result: list[CalibrationRepresentativeV1] = []
    for question_id in sorted(assignment):
        item = assignment[question_id]
        is_legacy = question_id == legacy_question
        payload = {
            "representative_version": REPRESENTATIVE_VERSION,
            "question_id": question_id,
            "item_id": item.item_id,
            "paper_id": item.paper_id,
            "core_split": "development" if is_legacy else "calibration",
            "purpose": (
                "structural_legacy_development_no_fitting"
                if is_legacy
                else "clopper_pearson_bound_input"
            ),
        }
        result.append(
            CalibrationRepresentativeV1.model_validate(
                {**payload, "representative_sha256": hash_canonical(payload)}
            )
        )
    return result


def _preparation_blockers(
    *,
    items: Sequence[FrozenEligibleRiskItemV1],
    split: FrozenQuestionSplitV1,
    representatives: Sequence[CalibrationRepresentativeV1],
) -> list[str]:
    blockers: list[str] = []
    if not items:
        blockers.append("no_completed_terminal_artifacts")
    if split.calibration_question_count < MIN_CALIBRATION_QUESTIONS:
        blockers.append(f"calibration_complete_questions_below_{MIN_CALIBRATION_QUESTIONS}")
    if split.evaluation_question_count < MIN_EVALUATION_QUESTIONS:
        blockers.append(f"evaluation_complete_questions_below_{MIN_EVALUATION_QUESTIONS}")
    bound_count = sum(item.core_split == "calibration" for item in representatives)
    if representatives and len(representatives) != split.calibration_question_count:
        blockers.append("unique_paper_representative_roster_incomplete")
    elif split.calibration_question_ids and not representatives:
        blockers.append("unique_paper_representative_assignment_unavailable")
    if bound_count < MIN_BOUND_QUESTIONS:
        blockers.append(f"bound_input_questions_below_{MIN_BOUND_QUESTIONS}")
    if sum(item.core_split == "development" for item in representatives) != (
        LEGACY_DEVELOPMENT_QUESTION_COUNT if representatives else 0
    ):
        blockers.append("structural_legacy_development_question_count_invalid")
    calibration_questions = set(split.calibration_question_ids)
    evaluation_questions = set(split.evaluation_question_ids)
    calibration_papers = {
        item.paper_id for item in items if item.question_id in calibration_questions
    }
    evaluation_papers = {
        item.paper_id for item in items if item.question_id in evaluation_questions
    }
    for paper_id in sorted(calibration_papers & evaluation_papers):
        blockers.append(f"cross_split_paper_reuse:{paper_id}")
    return sorted(set(blockers))


class MetaSynItemRiskPreparationV1(_FrozenExactModel):
    preparation_version: Literal["metasyn-artifact-item-risk-preparation-v1"] = PREPARATION_VERSION
    status: Literal["ready_for_label_blind_materialization", "insufficient_real_yield"]
    source_analysis_sha256: Sha256
    source_bridge_sha256: Sha256
    source_terminal_membership_sha256: Sha256
    source_publication_join_membership_sha256: Sha256
    score_policy: FixedTerminalRiskScorePolicyV1
    score_model_sha256: Sha256
    eligible_items: list[FrozenEligibleRiskItemV1]
    eligible_item_membership_sha256: Sha256
    eligible_item_count: Annotated[int, Field(ge=0)]
    split: FrozenQuestionSplitV1
    split_sha256: Sha256
    calibration_representatives: list[CalibrationRepresentativeV1]
    representative_membership_sha256: Sha256
    structural_legacy_development_question_id: str | None
    bound_input_question_ids: list[str]
    preparation_blockers: list[str]
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: Sha256
    source_external_replay_validated: Literal[True] = True
    selected_question_publication_roster_complete: Literal[True] = True
    all_completed_terminals_scored: Literal[False] = False
    labels_opened: Literal[False] = False
    evaluation_labels_opened: Literal[False] = False
    learned_weights_enabled: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    accuracy_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    access_order: list[str]
    preparation_sha256: Sha256

    @field_validator("eligible_items")
    @classmethod
    def validate_item_roster(
        cls, value: list[FrozenEligibleRiskItemV1]
    ) -> list[FrozenEligibleRiskItemV1]:
        if value != sorted(value, key=lambda item: (item.question_id, item.publication_id)):
            raise ValueError("metasyn_item_risk_v1_item_roster_not_canonical")
        pairs = [(item.question_id, item.publication_id) for item in value]
        if len(pairs) != len(set(pairs)):
            raise ValueError("metasyn_item_risk_v1_question_publication_duplicate")
        if len({item.item_id for item in value}) != len(value):
            raise ValueError("metasyn_item_risk_v1_item_id_duplicate")
        return value

    @field_validator("bound_input_question_ids", "preparation_blockers")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_item_risk_v1_preparation_list_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_preparation(self) -> MetaSynItemRiskPreparationV1:
        if self.score_model_sha256 != self.score_policy.score_model_sha256:
            raise ValueError("metasyn_item_risk_v1_score_model_alias_mismatch")
        if self.eligible_item_count != len(self.eligible_items) or (
            self.eligible_item_membership_sha256
            != hash_canonical([item.item_sha256 for item in self.eligible_items])
        ):
            raise ValueError("metasyn_item_risk_v1_item_membership_mismatch")
        if self.split_sha256 != self.split.split_sha256 or (
            self.split.eligible_question_ids
            != sorted({item.question_id for item in self.eligible_items})
        ):
            raise ValueError("metasyn_item_risk_v1_split_alias_mismatch")
        expected_representatives = _representative_assignment(
            items=self.eligible_items,
            calibration_question_ids=self.split.calibration_question_ids,
            split_salt_sha256=self.split.split_salt_sha256,
        )
        if self.calibration_representatives != expected_representatives or (
            self.representative_membership_sha256
            != hash_canonical(
                [item.representative_sha256 for item in self.calibration_representatives]
            )
        ):
            raise ValueError("metasyn_item_risk_v1_representative_membership_mismatch")
        legacy = [
            item.question_id
            for item in self.calibration_representatives
            if item.core_split == "development"
        ]
        expected_legacy = legacy[0] if len(legacy) == 1 else None
        expected_bound = sorted(
            item.question_id
            for item in self.calibration_representatives
            if item.core_split == "calibration"
        )
        if (
            self.structural_legacy_development_question_id != expected_legacy
            or self.bound_input_question_ids != expected_bound
        ):
            raise ValueError("metasyn_item_risk_v1_representative_alias_mismatch")
        blockers = _preparation_blockers(
            items=self.eligible_items,
            split=self.split,
            representatives=self.calibration_representatives,
        )
        if self.preparation_blockers != blockers or self.status != (
            "ready_for_label_blind_materialization" if not blockers else "insufficient_real_yield"
        ):
            raise ValueError("metasyn_item_risk_v1_preparation_status_mismatch")
        if self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_item_risk_v1_pipeline_alias_mismatch")
        if self.access_order != [
            "source_analysis_externally_replayed",
            "eligible_question_publication_roster_frozen",
            "question_split_frozen_without_labels",
            "representative_roster_frozen_without_labels",
            "computed_pipeline_identity_frozen",
        ]:
            raise ValueError("metasyn_item_risk_v1_preparation_access_order_invalid")
        _self_hash(self, "preparation_sha256")
        return self


def _freeze_split(*, question_ids: Sequence[str], split_salt: str) -> FrozenQuestionSplitV1:
    if split_salt != PRESPECIFIED_SPLIT_SALT:
        raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_split_salt_not_prespecified")
    eligible = sorted(set(question_ids))
    salt_sha = hash_canonical({"split_salt": split_salt})
    calibration, evaluation = _split_question_ids(eligible, split_salt_sha256=salt_sha)
    payload = {
        "split_version": SPLIT_VERSION,
        "split_salt": split_salt,
        "split_salt_sha256": salt_sha,
        "assignment_algorithm": "sha256_salted_rank_balanced_calibration_first_v1",
        "eligible_question_ids": eligible,
        "calibration_question_ids": calibration,
        "evaluation_question_ids": evaluation,
        "eligible_question_count": len(eligible),
        "calibration_question_count": len(calibration),
        "evaluation_question_count": len(evaluation),
        "question_disjoint": True,
        "labels_used_for_assignment": False,
    }
    return FrozenQuestionSplitV1.model_validate(
        {**payload, "split_sha256": hash_canonical(payload)}
    )


def _derive_eligible_item_roster(
    analysis: MetaSynGroundedAnalysisV2,
) -> list[FrozenEligibleRiskItemV1]:
    extraction_rows = {
        row.row_key: row for row in analysis.bridge.execution_bundle.extraction_inputs.rows
    }
    items: list[FrozenEligibleRiskItemV1] = []
    for join in analysis.bridge.publication_joins:
        completed = sorted(
            (terminal for terminal in join.candidate_terminals if terminal.authorizes_typed_effect),
            key=lambda terminal: (terminal.candidate_index, terminal.terminal_sha256),
        )
        if not completed:
            continue
        terminal = completed[0]
        effects = [
            effect
            for effect in join.compatibility_effects
            if effect.candidate_descriptor_sha256 == terminal.candidate_descriptor_sha256
        ]
        if len(effects) != 1:
            raise MetaSynItemRiskCalibrationV1Error(
                "metasyn_item_risk_v1_terminal_compatibility_join_ambiguous"
            )
        effect = effects[0]
        try:
            extraction_row = extraction_rows[join.row_key]
            relation_kind = extraction_row.question_surface.relation_kind
        except KeyError as exc:
            raise MetaSynItemRiskCalibrationV1Error(
                "metasyn_item_risk_v1_extraction_row_missing"
            ) from exc
        domain = "metasyn-intervention" if relation_kind == "intervention" else "metasyn-exposure"
        identity = {
            "item_version": ITEM_VERSION,
            "question_id": join.question_id,
            "publication_id": join.publication.publication_id,
            "paper_id": join.publication.paper_id,
            "domain": domain,
            "row_ordinal": join.row_ordinal,
            "row_key": join.row_key,
            "publication_join_sha256": join.publication_join_sha256,
            "source_strength_surface_sha256": join.source_strength_surface_sha256,
            "inventory_receipt_sha256": join.inventory_receipt_sha256,
            "candidate_index": terminal.candidate_index,
            "candidate_descriptor_sha256": terminal.candidate_descriptor_sha256,
            "candidate_binding_sha256": terminal.candidate_binding_sha256,
            "terminal_sha256": terminal.terminal_sha256,
            "grounding_receipt_sha256": terminal.grounding_receipt_sha256,
            "assembly_receipt_sha256": terminal.assembly_receipt_sha256,
            "typed_effect_sha256": terminal.assembly_receipt.typed_effect_sha256,
            "compatibility_sha256": effect.compatibility_sha256,
        }
        item_id = f"metasyn-risk-item-{hash_canonical({'item_identity': identity})[:40]}"
        payload = {**identity, "item_id": item_id}
        items.append(
            FrozenEligibleRiskItemV1.model_validate(
                {**payload, "item_sha256": hash_canonical(payload)}
            )
        )
    return sorted(items, key=lambda item: (item.question_id, item.publication_id))


def _pipeline_settings(
    *,
    analysis: MetaSynGroundedAnalysisV2,
    item_membership_sha256: str,
    split_sha256: str,
    representative_membership_sha256: str,
    score_model_sha256: str,
) -> dict[str, Any]:
    return {
        "source_analysis_sha256": analysis.analysis_sha256,
        "source_bridge_sha256": analysis.bridge_sha256,
        "source_terminal_membership_sha256": analysis.terminal_membership_sha256,
        "eligible_item_membership_sha256": item_membership_sha256,
        "split_sha256": split_sha256,
        "representative_membership_sha256": representative_membership_sha256,
        "score_model_sha256": score_model_sha256,
        "minimum_calibration_questions": MIN_CALIBRATION_QUESTIONS,
        "minimum_bound_questions": MIN_BOUND_QUESTIONS,
        "minimum_evaluation_questions": MIN_EVALUATION_QUESTIONS,
        "legacy_development_question_count": LEGACY_DEVELOPMENT_QUESTION_COUNT,
        "risk_bin_edges": list(RISK_BIN_EDGES),
        "familywise_delta": FAMILYWISE_DELTA,
        "shift_max_feature_mean_difference": SHIFT_MAX_FEATURE_MEAN_DIFFERENCE,
        "python_version": platform.python_version(),
        "installed_dependencies": {
            name: _installed_version(name) for name in ("pydantic", "scipy")
        },
        "labels_used_for_score_or_split": False,
        "split_salt_source": "code_owned_prespecified",
        "evaluation_labels_opened": False,
        "cross_split_paper_reuse_permitted": False,
        "learned_weights_enabled": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }


def compute_metasyn_item_risk_pipeline_fingerprint_v1(
    *,
    repository_root: Path,
    analysis: MetaSynGroundedAnalysisV2,
    item_membership_sha256: str,
    split_sha256: str,
    representative_membership_sha256: str,
    score_model_sha256: str,
) -> PipelineFingerprint:
    root = _canonical_root(repository_root)
    component = PipelineComponentSpec(
        component_id=COMPONENT_ID,
        component_version=COMPONENT_VERSION,
        file_paths=_python_dependency_closure(root),
        settings=_pipeline_settings(
            analysis=analysis,
            item_membership_sha256=item_membership_sha256,
            split_sha256=split_sha256,
            representative_membership_sha256=representative_membership_sha256,
            score_model_sha256=score_model_sha256,
        ),
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


def _prepare_from_replayed_analysis(
    *,
    source: MetaSynGroundedAnalysisV2,
    repository_root: Path,
    split_salt: str = PRESPECIFIED_SPLIT_SALT,
) -> MetaSynItemRiskPreparationV1:
    root = _canonical_root(repository_root)
    items = _derive_eligible_item_roster(source)
    split = _freeze_split(question_ids=[item.question_id for item in items], split_salt=split_salt)
    policy = freeze_fixed_terminal_risk_score_policy_v1()
    item_membership = hash_canonical([item.item_sha256 for item in items])
    representatives = _representative_assignment(
        items=items,
        calibration_question_ids=split.calibration_question_ids,
        split_salt_sha256=split.split_salt_sha256,
    )
    representative_membership = hash_canonical(
        [item.representative_sha256 for item in representatives]
    )
    blockers = _preparation_blockers(items=items, split=split, representatives=representatives)
    fingerprint = compute_metasyn_item_risk_pipeline_fingerprint_v1(
        repository_root=root,
        analysis=source,
        item_membership_sha256=item_membership,
        split_sha256=split.split_sha256,
        representative_membership_sha256=representative_membership,
        score_model_sha256=policy.score_model_sha256,
    )
    legacy = [item.question_id for item in representatives if item.core_split == "development"]
    payload = {
        "preparation_version": PREPARATION_VERSION,
        "status": (
            "ready_for_label_blind_materialization" if not blockers else "insufficient_real_yield"
        ),
        "source_analysis_sha256": source.analysis_sha256,
        "source_bridge_sha256": source.bridge_sha256,
        "source_terminal_membership_sha256": source.terminal_membership_sha256,
        "source_publication_join_membership_sha256": (source.publication_join_membership_sha256),
        "score_policy": policy,
        "score_model_sha256": policy.score_model_sha256,
        "eligible_items": items,
        "eligible_item_membership_sha256": item_membership,
        "eligible_item_count": len(items),
        "split": split,
        "split_sha256": split.split_sha256,
        "calibration_representatives": representatives,
        "representative_membership_sha256": representative_membership,
        "structural_legacy_development_question_id": (legacy[0] if len(legacy) == 1 else None),
        "bound_input_question_ids": sorted(
            item.question_id for item in representatives if item.core_split == "calibration"
        ),
        "preparation_blockers": blockers,
        "pipeline_fingerprint": fingerprint,
        "pipeline_sha256": fingerprint.pipeline_sha256,
        "source_external_replay_validated": True,
        "selected_question_publication_roster_complete": True,
        "all_completed_terminals_scored": False,
        "labels_opened": False,
        "evaluation_labels_opened": False,
        "learned_weights_enabled": False,
        "synthesis_input_authority": False,
        "accuracy_claim_authority": False,
        "claim_release_authority": False,
        "access_order": [
            "source_analysis_externally_replayed",
            "eligible_question_publication_roster_frozen",
            "question_split_frozen_without_labels",
            "representative_roster_frozen_without_labels",
            "computed_pipeline_identity_frozen",
        ],
    }
    return MetaSynItemRiskPreparationV1.model_validate(
        {**payload, "preparation_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_item_risk_calibration_v1(
    *,
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    split_salt: str = PRESPECIFIED_SPLIT_SALT,
) -> MetaSynItemRiskPreparationV1:
    """Externally replay the source, then freeze roster, split, and pipeline."""

    root = _canonical_root(repository_root)
    source = validate_metasyn_grounded_analysis_v2(
        analysis=analysis, repository_root=root, external_replay=True
    )
    return _prepare_from_replayed_analysis(
        source=source,
        repository_root=root,
        split_salt=split_salt,
    )


def _validate_preparation_and_source(
    *,
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
) -> tuple[MetaSynItemRiskPreparationV1, MetaSynGroundedAnalysisV2]:
    root = _canonical_root(repository_root)
    try:
        canonical = MetaSynItemRiskPreparationV1.model_validate(
            preparation.model_dump(mode="json")
            if isinstance(preparation, MetaSynItemRiskPreparationV1)
            else preparation
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_preparation_contract_invalid"
        ) from exc
    source = validate_metasyn_grounded_analysis_v2(
        analysis=analysis, repository_root=root, external_replay=True
    )
    replayed = _prepare_from_replayed_analysis(
        source=source,
        repository_root=root,
        split_salt=canonical.split.split_salt,
    )
    if replayed != canonical:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_preparation_external_replay_mismatch"
        )
    return canonical, source


def validate_metasyn_item_risk_preparation_v1(
    *,
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
) -> MetaSynItemRiskPreparationV1:
    canonical, _source = _validate_preparation_and_source(
        preparation=preparation,
        analysis=analysis,
        repository_root=repository_root,
    )
    return canonical


def _bounded(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 12)


def _score_features(features: Mapping[str, float], policy: FixedTerminalRiskScorePolicyV1) -> float:
    if sorted(features) != list(FEATURE_NAMES):
        raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_feature_roster_invalid")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in features.values()):
        raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_feature_value_invalid")
    return _bounded(sum(features[name] * policy.feature_weights[name] for name in FEATURE_NAMES))


class TerminalRiskFeatureRowV1(_FrozenExactModel):
    feature_row_version: Literal["metasyn-terminal-risk-feature-row-v1"] = FEATURE_ROW_VERSION
    item: FrozenEligibleRiskItemV1
    item_sha256: Sha256
    source_analysis_sha256: Sha256
    score_model_sha256: Sha256
    split: Literal["calibration", "evaluation"]
    features: dict[str, Annotated[float, Field(ge=0, le=1)]]
    score_input_sha256: Sha256
    risk_score: Annotated[float, Field(ge=0, le=1)]
    score_semantics: Literal["scheduling_rank_not_error_probability"] = (
        "scheduling_rank_not_error_probability"
    )
    labels_opened: Literal[False] = False
    observed_error_present: Literal[False] = False
    feature_row_sha256: Sha256

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if list(value) != list(FEATURE_NAMES):
            raise ValueError("metasyn_item_risk_v1_feature_keys_not_canonical")
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("metasyn_item_risk_v1_feature_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> TerminalRiskFeatureRowV1:
        if self.item_sha256 != self.item.item_sha256:
            raise ValueError("metasyn_item_risk_v1_feature_item_alias_mismatch")
        expected_input = hash_canonical(
            {
                "item_sha256": self.item_sha256,
                "source_analysis_sha256": self.source_analysis_sha256,
                "score_model_sha256": self.score_model_sha256,
                "features": self.features,
                "score_model_input_version": FEATURE_ROW_VERSION,
            }
        )
        if self.score_input_sha256 != expected_input:
            raise ValueError("metasyn_item_risk_v1_score_input_hash_mismatch")
        _self_hash(self, "feature_row_sha256")
        return self


class DomainFeatureShiftV1(_FrozenExactModel):
    domain: str
    calibration_question_count: Annotated[int, Field(ge=0)]
    evaluation_question_count: Annotated[int, Field(ge=0)]
    maximum_absolute_feature_mean_difference: Annotated[float | None, Field(ge=0, le=1)]
    status: Literal["no_shift_detected", "shift_detected"]
    domain_shift_sha256: Sha256

    @model_validator(mode="after")
    def validate_domain_shift(self) -> DomainFeatureShiftV1:
        if self.maximum_absolute_feature_mean_difference is None:
            if self.calibration_question_count and self.evaluation_question_count:
                raise ValueError("metasyn_item_risk_v1_shift_difference_missing")
            if self.status != "shift_detected":
                raise ValueError("metasyn_item_risk_v1_missing_domain_must_shift")
        else:
            expected = (
                "no_shift_detected"
                if self.maximum_absolute_feature_mean_difference
                <= SHIFT_MAX_FEATURE_MEAN_DIFFERENCE
                else "shift_detected"
            )
            if self.status != expected:
                raise ValueError("metasyn_item_risk_v1_domain_shift_status_mismatch")
        _self_hash(self, "domain_shift_sha256")
        return self


class LabelBlindShiftAssessmentV1(_FrozenExactModel):
    shift_version: Literal["metasyn-terminal-risk-shift-assessment-v1"] = SHIFT_VERSION
    detector_id: Literal["prespecified-question-centroid-max-difference-v1"] = (
        "prespecified-question-centroid-max-difference-v1"
    )
    detector_sha256: Sha256
    maximum_allowed_feature_mean_difference: Literal[0.35] = SHIFT_MAX_FEATURE_MEAN_DIFFERENCE
    calibration_domains: list[str]
    evaluation_domains: list[str]
    domains: list[DomainFeatureShiftV1]
    status: Literal["no_shift_detected", "shift_detected", "insufficient_split"]
    shift_blockers: list[str]
    labels_opened: Literal[False] = False
    exchangeability_proved: Literal[False] = False
    shift_assessment_sha256: Sha256

    @field_validator("calibration_domains", "evaluation_domains", "shift_blockers")
    @classmethod
    def validate_canonical_lists(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_item_risk_v1_shift_list_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_shift(self) -> LabelBlindShiftAssessmentV1:
        detector_payload = {
            "detector_id": self.detector_id,
            "feature_names": list(FEATURE_NAMES),
            "maximum_allowed_feature_mean_difference": (
                self.maximum_allowed_feature_mean_difference
            ),
            "question_level_centroids": True,
            "labels_opened": False,
        }
        if self.detector_sha256 != hash_canonical(detector_payload):
            raise ValueError("metasyn_item_risk_v1_shift_detector_hash_mismatch")
        if self.domains != sorted(self.domains, key=lambda item: item.domain):
            raise ValueError("metasyn_item_risk_v1_shift_domains_not_canonical")
        _self_hash(self, "shift_assessment_sha256")
        return self


def _question_feature_centroids(
    rows: Sequence[TerminalRiskFeatureRowV1],
) -> dict[str, tuple[str, dict[str, float]]]:
    by_question: dict[str, list[TerminalRiskFeatureRowV1]] = defaultdict(list)
    for row in rows:
        by_question[row.item.question_id].append(row)
    result: dict[str, tuple[str, dict[str, float]]] = {}
    for question_id, question_rows in by_question.items():
        domains = {row.item.domain for row in question_rows}
        if len(domains) != 1:
            raise MetaSynItemRiskCalibrationV1Error(
                "metasyn_item_risk_v1_question_domain_ambiguous"
            )
        result[question_id] = (
            next(iter(domains)),
            {
                name: sum(row.features[name] for row in question_rows) / len(question_rows)
                for name in FEATURE_NAMES
            },
        )
    return result


def _freeze_shift_assessment(
    *, rows: Sequence[TerminalRiskFeatureRowV1], split: FrozenQuestionSplitV1
) -> LabelBlindShiftAssessmentV1:
    centroids = _question_feature_centroids(rows)
    calibration_domains = sorted(
        {centroids[item][0] for item in split.calibration_question_ids if item in centroids}
    )
    evaluation_domains = sorted(
        {centroids[item][0] for item in split.evaluation_question_ids if item in centroids}
    )
    domain_rows: list[DomainFeatureShiftV1] = []
    blockers: list[str] = []
    all_domains = sorted(set(calibration_domains) | set(evaluation_domains))
    for domain in all_domains:
        cal = [
            values
            for question_id, (observed_domain, values) in centroids.items()
            if observed_domain == domain and question_id in set(split.calibration_question_ids)
        ]
        evaluation = [
            values
            for question_id, (observed_domain, values) in centroids.items()
            if observed_domain == domain and question_id in set(split.evaluation_question_ids)
        ]
        difference: float | None
        if not cal or not evaluation:
            difference = None
            blockers.append(f"domain_roster_mismatch:{domain}")
            status = "shift_detected"
        else:
            difference = _bounded(
                max(
                    abs(
                        sum(item[name] for item in cal) / len(cal)
                        - sum(item[name] for item in evaluation) / len(evaluation)
                    )
                    for name in FEATURE_NAMES
                )
            )
            status = (
                "no_shift_detected"
                if difference <= SHIFT_MAX_FEATURE_MEAN_DIFFERENCE
                else "shift_detected"
            )
            if status == "shift_detected":
                blockers.append(f"feature_mean_shift_detected:{domain}")
        payload = {
            "domain": domain,
            "calibration_question_count": len(cal),
            "evaluation_question_count": len(evaluation),
            "maximum_absolute_feature_mean_difference": difference,
            "status": status,
        }
        domain_rows.append(
            DomainFeatureShiftV1.model_validate(
                {**payload, "domain_shift_sha256": hash_canonical(payload)}
            )
        )
    insufficient = (
        split.calibration_question_count < MIN_CALIBRATION_QUESTIONS
        or split.evaluation_question_count < MIN_EVALUATION_QUESTIONS
    )
    if insufficient:
        blockers.append("minimum_question_split_not_met")
    status: Literal["no_shift_detected", "shift_detected", "insufficient_split"]
    if insufficient:
        status = "insufficient_split"
    elif blockers:
        status = "shift_detected"
    else:
        status = "no_shift_detected"
    detector_payload = {
        "detector_id": "prespecified-question-centroid-max-difference-v1",
        "feature_names": list(FEATURE_NAMES),
        "maximum_allowed_feature_mean_difference": (SHIFT_MAX_FEATURE_MEAN_DIFFERENCE),
        "question_level_centroids": True,
        "labels_opened": False,
    }
    payload = {
        "shift_version": SHIFT_VERSION,
        "detector_id": "prespecified-question-centroid-max-difference-v1",
        "detector_sha256": hash_canonical(detector_payload),
        "maximum_allowed_feature_mean_difference": (SHIFT_MAX_FEATURE_MEAN_DIFFERENCE),
        "calibration_domains": calibration_domains,
        "evaluation_domains": evaluation_domains,
        "domains": domain_rows,
        "status": status,
        "shift_blockers": sorted(set(blockers)),
        "labels_opened": False,
        "exchangeability_proved": False,
    }
    return LabelBlindShiftAssessmentV1.model_validate(
        {**payload, "shift_assessment_sha256": hash_canonical(payload)}
    )


class MetaSynTerminalRiskFeatureSetV1(_FrozenExactModel):
    feature_set_version: Literal["metasyn-terminal-risk-feature-set-v1"] = FEATURE_SET_VERSION
    preparation: MetaSynItemRiskPreparationV1
    preparation_sha256: Sha256
    pipeline_verification: PipelineFingerprintVerification
    pipeline_verification_sha256: Sha256
    score_model_sha256: Sha256
    rows: list[TerminalRiskFeatureRowV1]
    feature_row_membership_sha256: Sha256
    feature_row_count: Annotated[int, Field(ge=0)]
    calibration_feature_row_count: Annotated[int, Field(ge=0)]
    evaluation_feature_row_count: Annotated[int, Field(ge=0)]
    shift_assessment: LabelBlindShiftAssessmentV1
    shift_assessment_sha256: Sha256
    source_external_replay_validated: Literal[True] = True
    scores_computed_not_supplied: Literal[True] = True
    labels_opened: Literal[False] = False
    evaluation_labels_opened: Literal[False] = False
    score_is_error_probability: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    access_order: list[str]
    feature_set_sha256: Sha256

    @model_validator(mode="after")
    def validate_feature_set(self) -> MetaSynTerminalRiskFeatureSetV1:
        if self.preparation_sha256 != self.preparation.preparation_sha256:
            raise ValueError("metasyn_item_risk_v1_feature_preparation_alias_mismatch")
        if (
            self.pipeline_verification.status != "matched"
            or self.pipeline_verification.expected_pipeline_sha256
            != self.preparation.pipeline_sha256
            or self.pipeline_verification.computed_pipeline_sha256
            != self.preparation.pipeline_sha256
            or self.pipeline_verification_sha256 != self.pipeline_verification.verification_sha256
        ):
            raise ValueError("metasyn_item_risk_v1_feature_pipeline_mismatch")
        if self.score_model_sha256 != self.preparation.score_model_sha256:
            raise ValueError("metasyn_item_risk_v1_feature_score_model_mismatch")
        expected_order = sorted(
            self.rows, key=lambda row: (row.item.question_id, row.item.publication_id)
        )
        if self.rows != expected_order:
            raise ValueError("metasyn_item_risk_v1_feature_rows_not_canonical")
        if [row.item for row in self.rows] != self.preparation.eligible_items:
            raise ValueError("metasyn_item_risk_v1_feature_item_roster_mismatch")
        for row in self.rows:
            expected_split = (
                "calibration"
                if row.item.question_id in set(self.preparation.split.calibration_question_ids)
                else "evaluation"
            )
            if (
                row.split != expected_split
                or row.source_analysis_sha256 != self.preparation.source_analysis_sha256
                or row.score_model_sha256 != self.preparation.score_model_sha256
                or row.risk_score != _score_features(row.features, self.preparation.score_policy)
            ):
                raise ValueError("metasyn_item_risk_v1_feature_score_replay_mismatch")
        if (
            self.feature_row_count,
            self.calibration_feature_row_count,
            self.evaluation_feature_row_count,
        ) != (
            len(self.rows),
            sum(row.split == "calibration" for row in self.rows),
            sum(row.split == "evaluation" for row in self.rows),
        ):
            raise ValueError("metasyn_item_risk_v1_feature_count_mismatch")
        if self.feature_row_membership_sha256 != hash_canonical(
            [row.feature_row_sha256 for row in self.rows]
        ):
            raise ValueError("metasyn_item_risk_v1_feature_membership_mismatch")
        expected_shift = _freeze_shift_assessment(rows=self.rows, split=self.preparation.split)
        if (
            self.shift_assessment != expected_shift
            or self.shift_assessment_sha256 != self.shift_assessment.shift_assessment_sha256
        ):
            raise ValueError("metasyn_item_risk_v1_shift_replay_mismatch")
        if self.access_order != [
            "preparation_and_source_externally_replayed",
            "computed_pipeline_identity_recomputed_and_matched",
            "terminal_features_materialized_without_labels",
            "label_blind_shift_assessment_frozen",
        ]:
            raise ValueError("metasyn_item_risk_v1_feature_access_order_invalid")
        _self_hash(self, "feature_set_sha256")
        return self


def _terminal_for_item(analysis: MetaSynGroundedAnalysisV2, item: FrozenEligibleRiskItemV1):
    joins = [
        join
        for join in analysis.bridge.publication_joins
        if join.publication_join_sha256 == item.publication_join_sha256
    ]
    if len(joins) != 1:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_feature_publication_join_missing"
        )
    join = joins[0]
    terminals = [
        terminal
        for terminal in join.candidate_terminals
        if terminal.terminal_sha256 == item.terminal_sha256
    ]
    effects = [
        effect
        for effect in join.compatibility_effects
        if effect.compatibility_sha256 == item.compatibility_sha256
    ]
    if len(terminals) != 1 or len(effects) != 1:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_feature_terminal_join_missing"
        )
    return join, terminals[0], effects[0]


def _materialize_feature_row(
    *,
    analysis: MetaSynGroundedAnalysisV2,
    item: FrozenEligibleRiskItemV1,
    split: Literal["calibration", "evaluation"],
    policy: FixedTerminalRiskScorePolicyV1,
    source_analysis_sha256: str,
) -> TerminalRiskFeatureRowV1:
    join, terminal, effect = _terminal_for_item(analysis, item)
    passage_count = len(terminal.packet_input.candidate.passage_ids)
    omitted = terminal.packet_input.projection_surface.omitted_passage_count
    quote_length = len(effect.exact_evidence_quote)
    extraction_method = terminal.assembly_receipt.typed_effect.extraction_method
    features = {
        "candidate_rank_fraction": _bounded((terminal.candidate_index - 1) / 7),
        "computed_effect_indicator": 1.0 if extraction_method != "reported" else 0.0,
        "coverage_gap_fraction": _bounded(len(effect.coverage.coverage_blockers) / 5),
        "inventory_load_fraction": _bounded(join.inventoried_candidate_count / 8),
        "passage_complexity_fraction": _bounded((passage_count - 1) / 3),
        "projection_omission_fraction": _bounded(omitted / 8),
        "quote_brevity_fraction": _bounded(1 - min(quote_length / 128, 1.0)),
        "source_scope_risk": (
            0.0
            if terminal.packet_input.projection_surface.source_strength.source_content_scope
            == "full_text_sections"
            else 1.0
        ),
    }
    score_input = hash_canonical(
        {
            "item_sha256": item.item_sha256,
            "source_analysis_sha256": source_analysis_sha256,
            "score_model_sha256": policy.score_model_sha256,
            "features": features,
            "score_model_input_version": FEATURE_ROW_VERSION,
        }
    )
    payload = {
        "feature_row_version": FEATURE_ROW_VERSION,
        "item": item,
        "item_sha256": item.item_sha256,
        "source_analysis_sha256": source_analysis_sha256,
        "score_model_sha256": policy.score_model_sha256,
        "split": split,
        "features": features,
        "score_input_sha256": score_input,
        "risk_score": _score_features(features, policy),
        "score_semantics": "scheduling_rank_not_error_probability",
        "labels_opened": False,
        "observed_error_present": False,
    }
    return TerminalRiskFeatureRowV1.model_validate(
        {**payload, "feature_row_sha256": hash_canonical(payload)}
    )


def materialize_metasyn_terminal_risk_features_v1(
    *,
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
) -> MetaSynTerminalRiskFeatureSetV1:
    """Replay all pre-label artifacts and deterministically compute scores."""

    canonical, source = _validate_preparation_and_source(
        preparation=preparation,
        analysis=analysis,
        repository_root=repository_root,
    )
    verification = require_pipeline_fingerprint_match(
        expected=canonical.pipeline_fingerprint,
        root=_canonical_root(repository_root),
    )
    calibration_ids = set(canonical.split.calibration_question_ids)
    rows = [
        _materialize_feature_row(
            analysis=source,
            item=item,
            split=("calibration" if item.question_id in calibration_ids else "evaluation"),
            policy=canonical.score_policy,
            source_analysis_sha256=source.analysis_sha256,
        )
        for item in canonical.eligible_items
    ]
    rows.sort(key=lambda row: (row.item.question_id, row.item.publication_id))
    shift = _freeze_shift_assessment(rows=rows, split=canonical.split)
    payload = {
        "feature_set_version": FEATURE_SET_VERSION,
        "preparation": canonical,
        "preparation_sha256": canonical.preparation_sha256,
        "pipeline_verification": verification,
        "pipeline_verification_sha256": verification.verification_sha256,
        "score_model_sha256": canonical.score_model_sha256,
        "rows": rows,
        "feature_row_membership_sha256": hash_canonical([row.feature_row_sha256 for row in rows]),
        "feature_row_count": len(rows),
        "calibration_feature_row_count": sum(row.split == "calibration" for row in rows),
        "evaluation_feature_row_count": sum(row.split == "evaluation" for row in rows),
        "shift_assessment": shift,
        "shift_assessment_sha256": shift.shift_assessment_sha256,
        "source_external_replay_validated": True,
        "scores_computed_not_supplied": True,
        "labels_opened": False,
        "evaluation_labels_opened": False,
        "score_is_error_probability": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
        "access_order": [
            "preparation_and_source_externally_replayed",
            "computed_pipeline_identity_recomputed_and_matched",
            "terminal_features_materialized_without_labels",
            "label_blind_shift_assessment_frozen",
        ],
    }
    return MetaSynTerminalRiskFeatureSetV1.model_validate(
        {**payload, "feature_set_sha256": hash_canonical(payload)}
    )


def validate_metasyn_terminal_risk_features_v1(
    *,
    feature_set: MetaSynTerminalRiskFeatureSetV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
) -> MetaSynTerminalRiskFeatureSetV1:
    try:
        canonical = MetaSynTerminalRiskFeatureSetV1.model_validate(
            feature_set.model_dump(mode="json")
            if isinstance(feature_set, MetaSynTerminalRiskFeatureSetV1)
            else feature_set
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_feature_set_contract_invalid"
        ) from exc
    replayed = materialize_metasyn_terminal_risk_features_v1(
        preparation=canonical.preparation,
        analysis=analysis,
        repository_root=repository_root,
    )
    if replayed != canonical:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_feature_set_external_replay_mismatch"
        )
    return canonical


class ItemAdjudicationV1(_FrozenExactModel):
    item_id: str
    observed_error: StrictBool
    adjudication_artifact_sha256: Sha256


class AdjudicationQuestionFileV1(_FrozenExactModel):
    question_file_version: Literal["metasyn-question-adjudication-file-v1"] = (
        SIDECAR_QUESTION_FILE_VERSION
    )
    preparation_sha256: Sha256
    feature_set_sha256: Sha256
    split_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    question_id: str
    question_adjudication_artifact_sha256: Sha256
    complete_question: Literal[True] = True
    label_source: AdjudicatedLabelSource
    adjudication_protocol_sha256: Sha256
    items: list[ItemAdjudicationV1]
    simulation: Literal[False] = False
    question_file_sha256: Sha256

    @model_validator(mode="after")
    def validate_question(self) -> AdjudicationQuestionFileV1:
        if self.items != sorted(self.items, key=lambda item: item.item_id) or len(
            {item.item_id for item in self.items}
        ) != len(self.items):
            raise ValueError("metasyn_item_risk_v1_sidecar_items_not_canonical")
        _self_hash(self, "question_file_sha256")
        return self


def _validate_relative_sidecar_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
        or value == "manifest.json"
    ):
        raise ValueError("metasyn_item_risk_v1_sidecar_relative_path_invalid")
    return value


class AdjudicationManifestEntryV1(_FrozenExactModel):
    question_id: str
    relative_path: str
    file_sha256: Sha256
    file_bytes: Annotated[int, Field(ge=1, le=MAX_SIDECAR_BYTES)]
    entry_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_sidecar_path(value)

    @model_validator(mode="after")
    def validate_entry(self) -> AdjudicationManifestEntryV1:
        _self_hash(self, "entry_sha256")
        return self


class AdjudicationSidecarManifestV1(_FrozenExactModel):
    manifest_version: Literal["metasyn-adjudication-sidecar-manifest-v1"] = SIDECAR_MANIFEST_VERSION
    preparation_sha256: Sha256
    feature_set_sha256: Sha256
    split_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    split_salt_sha256: Sha256
    question_files: Annotated[list[AdjudicationManifestEntryV1], Field(min_length=1)]
    question_ids: list[str]
    question_file_membership_sha256: Sha256
    label_values_present: Literal[False] = False
    observed_error_fields_present: Literal[False] = False
    simulation: Literal[False] = False
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> AdjudicationSidecarManifestV1:
        if self.question_files != sorted(self.question_files, key=lambda item: item.question_id):
            raise ValueError("metasyn_item_risk_v1_manifest_entries_not_canonical")
        question_ids = [item.question_id for item in self.question_files]
        paths = [item.relative_path for item in self.question_files]
        if (
            len(question_ids) != len(set(question_ids))
            or len(paths) != len(set(paths))
            or self.question_ids != question_ids
            or self.question_file_membership_sha256
            != hash_canonical([item.entry_sha256 for item in self.question_files])
        ):
            raise ValueError("metasyn_item_risk_v1_manifest_roster_invalid")
        _self_hash(self, "manifest_sha256")
        return self


class OpenedCalibrationQuestionFileV1(_FrozenExactModel):
    manifest_entry: AdjudicationManifestEntryV1
    entry_sha256: Sha256
    question: AdjudicationQuestionFileV1
    question_file_sha256: Sha256
    opened_file_sha256: Sha256
    opened_file_bytes: Annotated[int, Field(ge=1, le=MAX_SIDECAR_BYTES)]
    opened_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_opened(self) -> OpenedCalibrationQuestionFileV1:
        if (
            self.entry_sha256 != self.manifest_entry.entry_sha256
            or self.question_file_sha256 != self.question.question_file_sha256
            or self.question.question_id != self.manifest_entry.question_id
            or self.opened_file_sha256 != self.manifest_entry.file_sha256
            or self.opened_file_bytes != self.manifest_entry.file_bytes
        ):
            raise ValueError("metasyn_item_risk_v1_opened_question_alias_mismatch")
        _self_hash(self, "opened_receipt_sha256")
        return self


class CompleteQuestionAdjudicationSidecarV1(_FrozenExactModel):
    sidecar_version: Literal["metasyn-complete-question-adjudication-sidecar-v1"] = SIDECAR_VERSION
    manifest_file_sha256: Sha256
    manifest_file_bytes: Annotated[int, Field(ge=1, le=MAX_SIDECAR_BYTES)]
    manifest: AdjudicationSidecarManifestV1
    manifest_sha256: Sha256
    preparation_sha256: Sha256
    feature_set_sha256: Sha256
    split_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    opened_calibration_files: list[OpenedCalibrationQuestionFileV1]
    opened_calibration_membership_sha256: Sha256
    questions: list[AdjudicationQuestionFileV1]
    calibration_question_ids: list[str]
    evaluation_question_ids: list[str]
    calibration_item_ids: list[str]
    complete_question_count: Annotated[int, Field(ge=1)]
    adjudicated_item_count: Annotated[int, Field(ge=1)]
    label_source: AdjudicatedLabelSource
    adjudication_protocol_sha256: Sha256
    opened_relative_paths: list[str]
    evaluation_relative_paths: list[str]
    feature_pipeline_split_frozen_before_manifest_open: Literal[True] = True
    manifest_is_label_free: Literal[True] = True
    evaluation_files_opened: Literal[False] = False
    evaluation_file_hashes_verified: Literal[False] = False
    simulation: Literal[False] = False
    sidecar_sha256: Sha256

    @model_validator(mode="after")
    def validate_sidecar(self) -> CompleteQuestionAdjudicationSidecarV1:
        if self.manifest_sha256 != self.manifest.manifest_sha256:
            raise ValueError("metasyn_item_risk_v1_manifest_hash_alias_mismatch")
        if self.opened_calibration_files != sorted(
            self.opened_calibration_files,
            key=lambda item: item.manifest_entry.question_id,
        ):
            raise ValueError("metasyn_item_risk_v1_opened_files_not_canonical")
        questions = [item.question for item in self.opened_calibration_files]
        question_ids = [item.question_id for item in questions]
        item_ids = sorted(item.item_id for row in questions for item in row.items)
        if (
            self.questions != questions
            or self.calibration_question_ids != question_ids
            or self.calibration_item_ids != item_ids
            or self.complete_question_count != len(question_ids)
            or self.adjudicated_item_count != len(item_ids)
            or self.opened_calibration_membership_sha256
            != hash_canonical(
                [item.opened_receipt_sha256 for item in self.opened_calibration_files]
            )
            or self.opened_relative_paths
            != [
                "manifest.json",
                *(item.manifest_entry.relative_path for item in self.opened_calibration_files),
            ]
        ):
            raise ValueError("metasyn_item_risk_v1_sidecar_count_alias_mismatch")
        if any(path in set(self.opened_relative_paths) for path in self.evaluation_relative_paths):
            raise ValueError("metasyn_item_risk_v1_evaluation_path_opened")
        _self_hash(self, "sidecar_sha256")
        return self


def _regular_json_object(path: Path) -> tuple[dict[str, Any], str, int]:
    source = Path(os.path.abspath(path))
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_file_missing"
        ) from exc
    try:
        if not stat.S_ISREG(metadata.st_mode):
            raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_sidecar_file_not_regular")
        if metadata.st_nlink != 1:
            raise MetaSynItemRiskCalibrationV1Error(
                "metasyn_item_risk_v1_sidecar_hardlink_forbidden"
            )
        if metadata.st_size > MAX_SIDECAR_BYTES:
            raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_sidecar_file_too_large")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_SIDECAR_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_SIDECAR_BYTES:
                raise MetaSynItemRiskCalibrationV1Error(
                    "metasyn_item_risk_v1_sidecar_file_too_large"
                )
        content = b"".join(chunks)
        value = json.loads(content.decode("utf-8"))
    except MetaSynItemRiskCalibrationV1Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_json_invalid"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_sidecar_json_not_object")
    return value, hashlib.sha256(content).hexdigest(), len(content)


def _canonical_sidecar_directory(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise MetaSynItemRiskCalibrationV1Error(
                "metasyn_item_risk_v1_sidecar_directory_symlink"
            )
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_directory_missing"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynItemRiskCalibrationV1Error("metasyn_item_risk_v1_sidecar_path_not_directory")
    return resolved


def _checked_sidecar_file(root: Path, relative_path: str) -> Path:
    _validate_relative_sidecar_path(relative_path)
    current = root
    for part in PurePosixPath(relative_path).parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_sidecar_manifest_file_missing:{relative_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_sidecar_symlink_forbidden:{relative_path}"
            )
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise MetaSynItemRiskCalibrationV1Error(
            f"metasyn_item_risk_v1_sidecar_file_invalid:{relative_path}"
        )
    return resolved


def _inventory_sidecar_tree(root: Path) -> tuple[list[str], list[str]]:
    observed_files: list[str] = []
    observed_directories: list[str] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            child = current_path / name
            if child.is_symlink():
                raise MetaSynItemRiskCalibrationV1Error(
                    "metasyn_item_risk_v1_sidecar_symlink_forbidden"
                )
            if not child.is_dir():
                raise MetaSynItemRiskCalibrationV1Error(
                    "metasyn_item_risk_v1_sidecar_nonregular_directory"
                )
            observed_directories.append(child.relative_to(root).as_posix())
        for name in sorted(file_names):
            child = current_path / name
            if child.is_symlink():
                raise MetaSynItemRiskCalibrationV1Error(
                    "metasyn_item_risk_v1_sidecar_symlink_forbidden"
                )
            if not child.is_file():
                raise MetaSynItemRiskCalibrationV1Error(
                    "metasyn_item_risk_v1_sidecar_nonregular_file"
                )
            observed_files.append(child.relative_to(root).as_posix())
    return sorted(observed_files), sorted(observed_directories)


def _open_calibration_sidecar_directory(
    *,
    sidecar_directory: Path,
    preparation: MetaSynItemRiskPreparationV1,
    features: MetaSynTerminalRiskFeatureSetV1,
) -> tuple[CompleteQuestionAdjudicationSidecarV1 | None, list[str], str]:
    root = _canonical_sidecar_directory(sidecar_directory)
    manifest_path = root / "manifest.json"
    raw_manifest, manifest_file_sha256, manifest_file_bytes = _regular_json_object(manifest_path)
    try:
        manifest = AdjudicationSidecarManifestV1.model_validate(raw_manifest)
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_manifest_contract_invalid"
        ) from exc
    expected_lineage = (
        preparation.preparation_sha256,
        features.feature_set_sha256,
        preparation.split_sha256,
        preparation.pipeline_sha256,
        preparation.score_model_sha256,
        preparation.split.split_salt_sha256,
    )
    observed_lineage = (
        manifest.preparation_sha256,
        manifest.feature_set_sha256,
        manifest.split_sha256,
        manifest.pipeline_sha256,
        manifest.score_model_sha256,
        manifest.split_salt_sha256,
    )
    if observed_lineage != expected_lineage:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_frozen_lineage_mismatch"
        )
    if manifest.question_ids != preparation.split.eligible_question_ids:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_manifest_question_roster_mismatch"
        )
    calibration_ids, evaluation_ids = _split_question_ids(
        manifest.question_ids,
        split_salt_sha256=hash_canonical({"split_salt": PRESPECIFIED_SPLIT_SALT}),
    )
    if (
        calibration_ids != preparation.split.calibration_question_ids
        or evaluation_ids != preparation.split.evaluation_question_ids
    ):
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_manifest_split_replay_mismatch"
        )
    entries = {item.question_id: item for item in manifest.question_files}
    expected_tree = sorted(
        ["manifest.json", *(item.relative_path for item in manifest.question_files)]
    )
    expected_directories: set[str] = set()
    for entry in manifest.question_files:
        parent = PurePosixPath(entry.relative_path).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_tree, observed_directories = _inventory_sidecar_tree(root)
    if observed_tree != expected_tree or observed_directories != sorted(expected_directories):
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_sidecar_directory_roster_mismatch"
        )
    for entry in manifest.question_files:
        candidate = _checked_sidecar_file(root, entry.relative_path)
        metadata = candidate.lstat()
        if metadata.st_nlink != 1:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_sidecar_hardlink_forbidden:{entry.question_id}"
            )
        if metadata.st_size != entry.file_bytes:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_sidecar_file_size_mismatch:{entry.question_id}"
            )

    expected_items_by_question: dict[str, set[str]] = defaultdict(set)
    for row in features.rows:
        if row.split == "calibration":
            expected_items_by_question[row.item.question_id].add(row.item.item_id)
    opened: list[OpenedCalibrationQuestionFileV1] = []
    blockers: list[str] = []
    for question_id in calibration_ids:
        entry = entries[question_id]
        question_path = _checked_sidecar_file(root, entry.relative_path)
        raw_question, opened_sha256, opened_bytes = _regular_json_object(question_path)
        if opened_sha256 != entry.file_sha256 or opened_bytes != entry.file_bytes:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_calibration_file_integrity_mismatch:{question_id}"
            )
        try:
            question = AdjudicationQuestionFileV1.model_validate(raw_question)
        except ValueError as exc:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_calibration_question_contract_invalid:{question_id}"
            ) from exc
        if question.question_id != question_id:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_mixed_question_file:{question_id}"
            )
        question_lineage = (
            question.preparation_sha256,
            question.feature_set_sha256,
            question.split_sha256,
            question.pipeline_sha256,
            question.score_model_sha256,
        )
        if question_lineage != expected_lineage[:5]:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_question_file_lineage_mismatch:{question_id}"
            )
        observed_items = {item.item_id for item in question.items}
        expected_items = expected_items_by_question[question_id]
        if not observed_items <= expected_items:
            raise MetaSynItemRiskCalibrationV1Error(
                f"metasyn_item_risk_v1_sidecar_item_out_of_scope:{question_id}"
            )
        if observed_items != expected_items:
            blockers.append(f"question_item_roster_incomplete:{question_id}")
        receipt_payload = {
            "manifest_entry": entry,
            "entry_sha256": entry.entry_sha256,
            "question": question,
            "question_file_sha256": question.question_file_sha256,
            "opened_file_sha256": opened_sha256,
            "opened_file_bytes": opened_bytes,
        }
        opened.append(
            OpenedCalibrationQuestionFileV1.model_validate(
                {
                    **receipt_payload,
                    "opened_receipt_sha256": hash_canonical(receipt_payload),
                }
            )
        )
    label_sources = {item.question.label_source for item in opened}
    protocols = {item.question.adjudication_protocol_sha256 for item in opened}
    if len(label_sources) != 1 or len(protocols) != 1:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_calibration_question_protocol_mixed"
        )
    if blockers:
        return None, sorted(set(blockers)), manifest_file_sha256
    evaluation_paths = [entries[item].relative_path for item in evaluation_ids]
    opened_paths = ["manifest.json", *(item.manifest_entry.relative_path for item in opened)]
    questions = [item.question for item in opened]
    payload = {
        "sidecar_version": SIDECAR_VERSION,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_file_bytes": manifest_file_bytes,
        "manifest": manifest,
        "manifest_sha256": manifest.manifest_sha256,
        "preparation_sha256": preparation.preparation_sha256,
        "feature_set_sha256": features.feature_set_sha256,
        "split_sha256": preparation.split_sha256,
        "pipeline_sha256": preparation.pipeline_sha256,
        "score_model_sha256": preparation.score_model_sha256,
        "opened_calibration_files": opened,
        "opened_calibration_membership_sha256": hash_canonical(
            [item.opened_receipt_sha256 for item in opened]
        ),
        "questions": questions,
        "calibration_question_ids": calibration_ids,
        "evaluation_question_ids": evaluation_ids,
        "calibration_item_ids": sorted(
            item.item_id for question in questions for item in question.items
        ),
        "complete_question_count": len(questions),
        "adjudicated_item_count": sum(len(item.items) for item in questions),
        "label_source": next(iter(label_sources)),
        "adjudication_protocol_sha256": next(iter(protocols)),
        "opened_relative_paths": opened_paths,
        "evaluation_relative_paths": evaluation_paths,
        "feature_pipeline_split_frozen_before_manifest_open": True,
        "manifest_is_label_free": True,
        "evaluation_files_opened": False,
        "evaluation_file_hashes_verified": False,
        "simulation": False,
    }
    return (
        CompleteQuestionAdjudicationSidecarV1.model_validate(
            {**payload, "sidecar_sha256": hash_canonical(payload)}
        ),
        [],
        manifest_file_sha256,
    )


CalibrationRunStatus = Literal[
    "calibrated_scheduling_only",
    "abstained_no_eligible_terminal_artifacts",
    "abstained_too_few_complete_questions",
    "abstained_split_independence_violation",
    "abstained_domain_shift",
    "abstained_no_calibration_labels",
    "abstained_incomplete_question_sidecar",
]


class ArtifactBackedItemRiskBoundsReceiptV1(_FrozenExactModel):
    receipt_version: Literal["artifact-backed-item-risk-bounds-receipt-v1"] = (
        "artifact-backed-item-risk-bounds-receipt-v1"
    )
    core_bundle_sha256: Sha256
    preparation_sha256: Sha256
    feature_set_sha256: Sha256
    sidecar_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    population_id: str
    calibration_domains: list[str]
    supported_deployment_domains: list[str]
    bin_family: FixedRiskBinFamily
    bin_family_sha256: Sha256
    bounds: list[DomainRiskBinCalibration]
    bounds_membership_sha256: Sha256
    core_unit_membership_sha256: Sha256
    core_unit_count: Annotated[int, Field(ge=MIN_CALIBRATION_QUESTIONS)]
    bound_input_question_count: Annotated[int, Field(ge=MIN_BOUND_QUESTIONS)]
    familywise_delta: Literal[0.05] = FAMILYWISE_DELTA
    correction: Literal["bonferroni-clopper-pearson"] = "bonferroni-clopper-pearson"
    executable_score_semantics: Literal[
        "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
    ] = "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
    accepts_caller_supplied_scores: Literal[False] = False
    generic_core_bundle_exported: Literal[False] = False
    group_average_bounds_only: Literal[True] = True
    individual_item_probability_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> ArtifactBackedItemRiskBoundsReceiptV1:
        expected_cells = [
            (domain, risk_bin.bin_id)
            for domain in self.calibration_domains
            for risk_bin in self.bin_family.bins
        ]
        observed_cells = [(item.domain, item.bin_id) for item in self.bounds]
        if (
            self.bin_family_sha256 != self.bin_family.family_sha256
            or self.bin_family.score_model_sha256 != self.score_model_sha256
            or self.bounds_membership_sha256 != hash_canonical(self.bounds)
            or not self.calibration_domains
            or self.calibration_domains != sorted(set(self.calibration_domains))
            or not self.supported_deployment_domains
            or self.supported_deployment_domains != sorted(set(self.supported_deployment_domains))
            or not set(self.supported_deployment_domains) <= set(self.calibration_domains)
            or observed_cells != expected_cells
            or sum(item.cell_calibration_units for item in self.bounds)
            != self.bound_input_question_count
            or self.core_unit_count
            != self.bound_input_question_count + LEGACY_DEVELOPMENT_QUESTION_COUNT
            or any(item.familywise_delta != self.familywise_delta for item in self.bounds)
        ):
            raise ValueError("metasyn_item_risk_v1_bounds_receipt_alias_mismatch")
        _self_hash(self, "receipt_sha256")
        return self


def _sanitize_core_bundle(
    *,
    bundle: ItemRiskCalibrationBundle,
    preparation: MetaSynItemRiskPreparationV1,
    features: MetaSynTerminalRiskFeatureSetV1,
    sidecar: CompleteQuestionAdjudicationSidecarV1,
) -> ArtifactBackedItemRiskBoundsReceiptV1:
    payload = {
        "receipt_version": "artifact-backed-item-risk-bounds-receipt-v1",
        "core_bundle_sha256": bundle.bundle_sha256,
        "preparation_sha256": preparation.preparation_sha256,
        "feature_set_sha256": features.feature_set_sha256,
        "sidecar_sha256": sidecar.sidecar_sha256,
        "pipeline_sha256": preparation.pipeline_sha256,
        "score_model_sha256": preparation.score_model_sha256,
        "population_id": bundle.population_id,
        "calibration_domains": bundle.calibration_domains,
        "supported_deployment_domains": bundle.supported_deployment_domains,
        "bin_family": bundle.bin_family,
        "bin_family_sha256": bundle.bin_family_sha256,
        "bounds": bundle.bounds,
        "bounds_membership_sha256": hash_canonical(bundle.bounds),
        "core_unit_membership_sha256": hash_canonical([item.unit_sha256 for item in bundle.units]),
        "core_unit_count": len(bundle.units),
        "bound_input_question_count": len(preparation.bound_input_question_ids),
        "familywise_delta": FAMILYWISE_DELTA,
        "correction": "bonferroni-clopper-pearson",
        "executable_score_semantics": (
            "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
        ),
        "accepts_caller_supplied_scores": False,
        "generic_core_bundle_exported": False,
        "group_average_bounds_only": True,
        "individual_item_probability_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return ArtifactBackedItemRiskBoundsReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class MetaSynItemRiskCalibrationRunV1(_FrozenExactModel):
    calibration_run_version: Literal["metasyn-artifact-item-risk-calibration-run-v1"] = (
        CALIBRATION_RUN_VERSION
    )
    status: CalibrationRunStatus
    preparation_sha256: Sha256
    feature_set_sha256: Sha256
    split_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    pipeline_verification: PipelineFingerprintVerification
    pipeline_verification_sha256: Sha256
    shift_assessment_sha256: Sha256
    sidecar_manifest_file_sha256: Sha256 | None
    calibration_sidecar: CompleteQuestionAdjudicationSidecarV1 | None
    calibration_sidecar_sha256: Sha256 | None
    bounds_receipt: ArtifactBackedItemRiskBoundsReceiptV1 | None
    bounds_receipt_sha256: Sha256 | None
    structural_legacy_development_question_id: str | None
    structural_legacy_development_label_used_for_fitting: Literal[False] = False
    bound_input_question_ids: list[str]
    bound_input_question_count: Annotated[int, Field(ge=0)]
    evaluation_question_ids: list[str]
    labels_opened: bool
    evaluation_labels_opened: Literal[False] = False
    evaluation_label_file_path_accepted_by_api: Literal[False] = False
    executable_score_semantics: Literal[
        "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
    ] = "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
    generic_core_bundle_exported: Literal[False] = False
    score_is_error_probability: Literal[False] = False
    group_average_bounds_only: Literal[True] = True
    scheduling_authority: bool
    synthesis_input_authority: Literal[False] = False
    accuracy_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    blockers: list[str]
    access_order: list[str]
    calibration_run_sha256: Sha256

    @field_validator("bound_input_question_ids", "evaluation_question_ids", "blockers")
    @classmethod
    def validate_lists(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_item_risk_v1_calibration_list_not_canonical")
        return value

    @field_validator("labels_opened", "scheduling_authority", mode="before")
    @classmethod
    def validate_strict_bools(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("metasyn_item_risk_v1_calibration_boolean_not_strict")
        return value

    @model_validator(mode="after")
    def validate_run(self) -> MetaSynItemRiskCalibrationRunV1:
        if (
            self.pipeline_verification.status != "matched"
            or self.pipeline_verification.expected_pipeline_sha256 != self.pipeline_sha256
            or self.pipeline_verification.computed_pipeline_sha256 != self.pipeline_sha256
            or self.pipeline_verification_sha256 != self.pipeline_verification.verification_sha256
        ):
            raise ValueError("metasyn_item_risk_v1_run_pipeline_mismatch")
        if self.bound_input_question_count != len(self.bound_input_question_ids):
            raise ValueError("metasyn_item_risk_v1_bound_question_count_mismatch")
        calibrated = self.status == "calibrated_scheduling_only"
        if calibrated != (
            self.calibration_sidecar is not None
            and self.bounds_receipt is not None
            and self.labels_opened
            and self.scheduling_authority
            and not self.blockers
        ):
            raise ValueError("metasyn_item_risk_v1_calibrated_shape_mismatch")
        if self.calibration_sidecar is None:
            if self.calibration_sidecar_sha256 is not None:
                raise ValueError("metasyn_item_risk_v1_sidecar_hash_without_sidecar")
        elif self.calibration_sidecar_sha256 != self.calibration_sidecar.sidecar_sha256:
            raise ValueError("metasyn_item_risk_v1_sidecar_hash_alias_mismatch")
        if self.bounds_receipt is None:
            if self.bounds_receipt_sha256 is not None:
                raise ValueError("metasyn_item_risk_v1_bounds_hash_without_receipt")
        elif self.bounds_receipt_sha256 != self.bounds_receipt.receipt_sha256:
            raise ValueError("metasyn_item_risk_v1_bounds_hash_alias_mismatch")
        if self.labels_opened != (self.sidecar_manifest_file_sha256 is not None):
            raise ValueError("metasyn_item_risk_v1_label_access_alias_mismatch")
        if self.calibration_sidecar is not None and (
            self.sidecar_manifest_file_sha256 != self.calibration_sidecar.manifest_file_sha256
            or self.preparation_sha256 != self.calibration_sidecar.preparation_sha256
            or self.feature_set_sha256 != self.calibration_sidecar.feature_set_sha256
            or self.split_sha256 != self.calibration_sidecar.split_sha256
            or self.pipeline_sha256 != self.calibration_sidecar.pipeline_sha256
            or self.score_model_sha256 != self.calibration_sidecar.score_model_sha256
            or self.evaluation_question_ids != self.calibration_sidecar.evaluation_question_ids
        ):
            raise ValueError("metasyn_item_risk_v1_sidecar_run_lineage_mismatch")
        if self.bounds_receipt is not None and (
            self.preparation_sha256 != self.bounds_receipt.preparation_sha256
            or self.feature_set_sha256 != self.bounds_receipt.feature_set_sha256
            or self.pipeline_sha256 != self.bounds_receipt.pipeline_sha256
            or self.score_model_sha256 != self.bounds_receipt.score_model_sha256
            or self.calibration_sidecar_sha256 != self.bounds_receipt.sidecar_sha256
            or self.bound_input_question_count != self.bounds_receipt.bound_input_question_count
        ):
            raise ValueError("metasyn_item_risk_v1_bounds_run_lineage_mismatch")
        _self_hash(self, "calibration_run_sha256")
        return self


class ArtifactBackedItemRiskAssignmentV1(_FrozenExactModel):
    """A replay-bound group UCL assignment for one computed feature row.

    The nested row and sanitized receipt make the score and selected cell
    intrinsically replayable.  External membership in the frozen feature set is
    enforced by :func:`assign_artifact_backed_item_risk_v1` and its validator.
    """

    assignment_version: Literal["metasyn-artifact-item-risk-assignment-v1"] = ASSIGNMENT_VERSION
    feature_row: TerminalRiskFeatureRowV1
    feature_row_sha256: Sha256
    feature_set_sha256: Sha256
    preparation_sha256: Sha256
    calibration_run_sha256: Sha256
    bounds_receipt: ArtifactBackedItemRiskBoundsReceiptV1
    bounds_receipt_sha256: Sha256
    pipeline_sha256: Sha256
    score_model_sha256: Sha256
    item_id: str
    question_id: str
    publication_id: str
    paper_id: str
    domain: str
    score_input_sha256: Sha256
    computed_risk_score: Annotated[float, Field(ge=0, le=1)]
    bin_id: str
    bin_lower: Annotated[float, Field(ge=0, le=1)]
    bin_upper: Annotated[float, Field(gt=0, le=1)]
    bin_upper_inclusive: bool
    bound_cell_sha256: Sha256
    cell_calibration_units: Annotated[int, Field(gt=0)]
    cell_observed_errors: Annotated[int, Field(ge=0)]
    conservative_group_upper_error_rate: Annotated[float, Field(ge=0, le=1)]
    estimand: Literal["group_average_item_error_rate_within_domain_score_bin"] = (
        "group_average_item_error_rate_within_domain_score_bin"
    )
    score_source: Literal["externally_replayed_terminal_feature_row"] = (
        "externally_replayed_terminal_feature_row"
    )
    accepts_caller_supplied_score: Literal[False] = False
    accepts_caller_supplied_domain: Literal[False] = False
    accepts_caller_supplied_bin: Literal[False] = False
    individual_item_probability_authority: Literal[False] = False
    scheduling_authority: Literal[True] = True
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    assignment_sha256: Sha256

    @field_validator("bin_upper_inclusive", mode="before")
    @classmethod
    def validate_strict_upper_inclusive(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("metasyn_item_risk_v1_assignment_boolean_not_strict")
        return value

    @model_validator(mode="after")
    def validate_assignment(self) -> ArtifactBackedItemRiskAssignmentV1:
        row = self.feature_row
        receipt = self.bounds_receipt
        if (
            self.feature_row_sha256 != row.feature_row_sha256
            or self.bounds_receipt_sha256 != receipt.receipt_sha256
            or self.preparation_sha256 != receipt.preparation_sha256
            or self.feature_set_sha256 != receipt.feature_set_sha256
            or self.pipeline_sha256 != receipt.pipeline_sha256
            or self.score_model_sha256 != receipt.score_model_sha256
            or self.score_model_sha256 != row.score_model_sha256
            or self.item_id != row.item.item_id
            or self.question_id != row.item.question_id
            or self.publication_id != row.item.publication_id
            or self.paper_id != row.item.paper_id
            or self.domain != row.item.domain
            or self.score_input_sha256 != row.score_input_sha256
            or self.computed_risk_score != row.risk_score
        ):
            raise ValueError("metasyn_item_risk_v1_assignment_row_alias_mismatch")
        if self.domain not in receipt.supported_deployment_domains:
            raise ValueError("metasyn_item_risk_v1_assignment_domain_unsupported")
        risk_bin = receipt.bin_family.bin_for_score(row.risk_score)
        if (
            self.bin_id != risk_bin.bin_id
            or self.bin_lower != risk_bin.lower
            or self.bin_upper != risk_bin.upper
            or self.bin_upper_inclusive != risk_bin.upper_inclusive
        ):
            raise ValueError("metasyn_item_risk_v1_assignment_bin_mismatch")
        matching_cells = [
            cell
            for cell in receipt.bounds
            if cell.domain == self.domain and cell.bin_id == self.bin_id
        ]
        if len(matching_cells) != 1:
            raise ValueError("metasyn_item_risk_v1_assignment_bound_cell_ambiguous")
        cell = matching_cells[0]
        if cell.status != "calibrated" or cell.upper_cell_error_rate is None:
            raise ValueError("metasyn_item_risk_v1_assignment_bound_cell_empty")
        if (
            self.bound_cell_sha256 != hash_canonical(cell)
            or self.cell_calibration_units != cell.cell_calibration_units
            or self.cell_observed_errors != cell.cell_observed_errors
            or self.conservative_group_upper_error_rate != cell.upper_cell_error_rate
            or self.estimand != cell.estimand
        ):
            raise ValueError("metasyn_item_risk_v1_assignment_bound_alias_mismatch")
        _self_hash(self, "assignment_sha256")
        return self


def _calibration_run(
    *,
    status: CalibrationRunStatus,
    preparation: MetaSynItemRiskPreparationV1,
    features: MetaSynTerminalRiskFeatureSetV1,
    verification: PipelineFingerprintVerification,
    blockers: Sequence[str],
    labels_opened: bool,
    manifest_file_sha256: str | None = None,
    sidecar: CompleteQuestionAdjudicationSidecarV1 | None = None,
    bounds_receipt: ArtifactBackedItemRiskBoundsReceiptV1 | None = None,
) -> MetaSynItemRiskCalibrationRunV1:
    access_order = [
        "preparation_source_features_externally_replayed",
        "computed_pipeline_identity_recomputed_and_matched",
        "split_size_and_label_blind_shift_checked",
    ]
    if labels_opened:
        access_order.extend(
            [
                "label_free_sidecar_manifest_opened",
                "calibration_question_files_opened_without_evaluation_files",
                "complete_question_roster_checked",
            ]
        )
    if bounds_receipt is not None:
        access_order.append("conservative_group_average_bounds_sealed")
    payload = {
        "calibration_run_version": CALIBRATION_RUN_VERSION,
        "status": status,
        "preparation_sha256": preparation.preparation_sha256,
        "feature_set_sha256": features.feature_set_sha256,
        "split_sha256": preparation.split_sha256,
        "pipeline_sha256": preparation.pipeline_sha256,
        "score_model_sha256": preparation.score_model_sha256,
        "pipeline_verification": verification,
        "pipeline_verification_sha256": verification.verification_sha256,
        "shift_assessment_sha256": features.shift_assessment_sha256,
        "sidecar_manifest_file_sha256": manifest_file_sha256,
        "calibration_sidecar": sidecar,
        "calibration_sidecar_sha256": None if sidecar is None else sidecar.sidecar_sha256,
        "bounds_receipt": bounds_receipt,
        "bounds_receipt_sha256": (
            None if bounds_receipt is None else bounds_receipt.receipt_sha256
        ),
        "structural_legacy_development_question_id": (
            preparation.structural_legacy_development_question_id
        ),
        "structural_legacy_development_label_used_for_fitting": False,
        "bound_input_question_ids": preparation.bound_input_question_ids,
        "bound_input_question_count": len(preparation.bound_input_question_ids),
        "evaluation_question_ids": preparation.split.evaluation_question_ids,
        "labels_opened": labels_opened,
        "evaluation_labels_opened": False,
        "evaluation_label_file_path_accepted_by_api": False,
        "executable_score_semantics": (
            "terminal_artifact_features_recomputed_by_frozen_pipeline_for_scheduling_only"
        ),
        "generic_core_bundle_exported": False,
        "score_is_error_probability": False,
        "group_average_bounds_only": True,
        "scheduling_authority": bounds_receipt is not None,
        "synthesis_input_authority": False,
        "accuracy_claim_authority": False,
        "claim_release_authority": False,
        "blockers": sorted(set(blockers)),
        "access_order": access_order,
    }
    return MetaSynItemRiskCalibrationRunV1.model_validate(
        {**payload, "calibration_run_sha256": hash_canonical(payload)}
    )


def calibrate_metasyn_item_risk_v1(
    *,
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    feature_set: MetaSynTerminalRiskFeatureSetV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    adjudication_sidecar_directory: Path | None,
) -> MetaSynItemRiskCalibrationRunV1:
    """Calibrate without ever accepting or opening evaluation labels."""

    canonical_features = validate_metasyn_terminal_risk_features_v1(
        feature_set=feature_set,
        analysis=analysis,
        repository_root=repository_root,
    )
    try:
        canonical_preparation = MetaSynItemRiskPreparationV1.model_validate(
            preparation.model_dump(mode="json")
            if isinstance(preparation, MetaSynItemRiskPreparationV1)
            else preparation
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_preparation_contract_invalid"
        ) from exc
    if canonical_features.preparation != canonical_preparation:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_calibration_preparation_feature_mismatch"
        )
    verification = require_pipeline_fingerprint_match(
        expected=canonical_preparation.pipeline_fingerprint,
        root=_canonical_root(repository_root),
    )
    if "no_completed_terminal_artifacts" in canonical_preparation.preparation_blockers:
        return _calibration_run(
            status="abstained_no_eligible_terminal_artifacts",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=canonical_preparation.preparation_blockers,
            labels_opened=False,
        )
    split_independence_blockers = [
        item
        for item in canonical_preparation.preparation_blockers
        if item.startswith("cross_split_paper_reuse:")
    ]
    if split_independence_blockers:
        return _calibration_run(
            status="abstained_split_independence_violation",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=split_independence_blockers,
            labels_opened=False,
        )
    if canonical_preparation.preparation_blockers:
        return _calibration_run(
            status="abstained_too_few_complete_questions",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=canonical_preparation.preparation_blockers,
            labels_opened=False,
        )
    if canonical_features.shift_assessment.status != "no_shift_detected":
        return _calibration_run(
            status="abstained_domain_shift",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=canonical_features.shift_assessment.shift_blockers,
            labels_opened=False,
        )
    if adjudication_sidecar_directory is None:
        return _calibration_run(
            status="abstained_no_calibration_labels",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=["complete_question_calibration_sidecar_not_supplied"],
            labels_opened=False,
        )

    sidecar, completeness_blockers, manifest_file_sha256 = _open_calibration_sidecar_directory(
        sidecar_directory=adjudication_sidecar_directory,
        preparation=canonical_preparation,
        features=canonical_features,
    )
    if sidecar is None:
        return _calibration_run(
            status="abstained_incomplete_question_sidecar",
            preparation=canonical_preparation,
            features=canonical_features,
            verification=verification,
            blockers=completeness_blockers,
            labels_opened=True,
            manifest_file_sha256=manifest_file_sha256,
        )

    labels = {item.item_id: item for question in sidecar.questions for item in question.items}
    features_by_item = {row.item.item_id: row for row in canonical_features.rows}
    units = []
    for representative in canonical_preparation.calibration_representatives:
        feature = features_by_item[representative.item_id]
        label = labels[representative.item_id]
        units.append(
            seal_item_risk_calibration_unit(
                split=representative.core_split,
                item_id=representative.item_id,
                question_id=representative.question_id,
                paper_id=representative.paper_id,
                population_id=POPULATION_ID,
                domain=feature.item.domain,
                pipeline_sha256=canonical_preparation.pipeline_sha256,
                score_model_sha256=canonical_preparation.score_model_sha256,
                score_input_sha256=feature.score_input_sha256,
                risk_score=feature.risk_score,
                observed_error=label.observed_error,
                label_source=sidecar.label_source,
                adjudication_protocol_sha256=sidecar.adjudication_protocol_sha256,
                adjudication_artifact_sha256=label.adjudication_artifact_sha256,
            )
        )
    bin_family = make_fixed_risk_bin_family(
        edges=RISK_BIN_EDGES,
        score_name=canonical_preparation.score_policy.score_name,
        score_model_sha256=canonical_preparation.score_model_sha256,
        definition_source="prespecified",
        definition_artifact_sha256=canonical_preparation.score_model_sha256,
    )
    sampling_protocol_sha256 = hash_canonical(
        {
            "protocol_version": "metasyn-complete-question-calibration-sampling-v1",
            "split_sha256": canonical_preparation.split_sha256,
            "representative_membership_sha256": (
                canonical_preparation.representative_membership_sha256
            ),
            "one_item_per_question_and_unique_paper": True,
            "structural_legacy_development_label_used_for_fitting": False,
            "minimum_bound_questions": MIN_BOUND_QUESTIONS,
            "evaluation_labels_opened": False,
        }
    )
    core_bundle = calibrate_item_risk_bounds(
        units,
        pipeline_verification=verification,
        bin_family=bin_family,
        familywise_delta=FAMILYWISE_DELTA,
        sampling_protocol_sha256=sampling_protocol_sha256,
        error_event_definition=ERROR_EVENT_DEFINITION,
        shift_detector_id=canonical_features.shift_assessment.detector_id,
        shift_detector_sha256=canonical_features.shift_assessment.detector_sha256,
        supported_deployment_domains=sorted(
            {unit.domain for unit in units if unit.split == "calibration"}
        ),
    )
    bounds_receipt = _sanitize_core_bundle(
        bundle=core_bundle,
        preparation=canonical_preparation,
        features=canonical_features,
        sidecar=sidecar,
    )
    return _calibration_run(
        status="calibrated_scheduling_only",
        preparation=canonical_preparation,
        features=canonical_features,
        verification=verification,
        blockers=[],
        labels_opened=True,
        manifest_file_sha256=manifest_file_sha256,
        sidecar=sidecar,
        bounds_receipt=bounds_receipt,
    )


def validate_metasyn_item_risk_calibration_run_v1(
    *,
    calibration_run: MetaSynItemRiskCalibrationRunV1 | Mapping[str, Any],
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    feature_set: MetaSynTerminalRiskFeatureSetV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    adjudication_sidecar_directory: Path | None,
) -> MetaSynItemRiskCalibrationRunV1:
    try:
        canonical = MetaSynItemRiskCalibrationRunV1.model_validate(
            calibration_run.model_dump(mode="json")
            if isinstance(calibration_run, MetaSynItemRiskCalibrationRunV1)
            else calibration_run
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_calibration_run_contract_invalid"
        ) from exc
    replayed = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=feature_set,
        analysis=analysis,
        repository_root=repository_root,
        adjudication_sidecar_directory=adjudication_sidecar_directory,
    )
    if replayed != canonical:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_calibration_run_external_replay_mismatch"
        )
    return canonical


def _bound_cell_for_replayed_feature_row(
    *,
    feature_row: TerminalRiskFeatureRowV1,
    receipt: ArtifactBackedItemRiskBoundsReceiptV1,
) -> tuple[RiskBinSpec, DomainRiskBinCalibration]:
    domain = feature_row.item.domain
    if domain not in receipt.supported_deployment_domains:
        raise MetaSynItemRiskCalibrationV1Error(
            f"metasyn_item_risk_v1_assignment_domain_unsupported:{domain}"
        )
    risk_bin = receipt.bin_family.bin_for_score(feature_row.risk_score)
    cells = [
        cell for cell in receipt.bounds if cell.domain == domain and cell.bin_id == risk_bin.bin_id
    ]
    if len(cells) != 1:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_bound_cell_ambiguous"
        )
    cell = cells[0]
    if cell.status != "calibrated" or cell.upper_cell_error_rate is None:
        raise MetaSynItemRiskCalibrationV1Error(
            f"metasyn_item_risk_v1_assignment_bound_cell_empty:{domain}:{risk_bin.bin_id}"
        )
    return risk_bin, cell


def assign_artifact_backed_item_risk_v1(
    *,
    feature_row: TerminalRiskFeatureRowV1,
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    feature_set: MetaSynTerminalRiskFeatureSetV1 | Mapping[str, Any],
    calibration_run: MetaSynItemRiskCalibrationRunV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    adjudication_sidecar_directory: Path,
) -> ArtifactBackedItemRiskAssignmentV1:
    """Assign a conservative scheduling UCL to one replayed feature row.

    There are intentionally no score, domain, or bin override parameters.  The
    score is recomputed by external feature-set replay, and the selected domain
    and bin come only from the exact frozen row.
    """

    if not isinstance(feature_row, TerminalRiskFeatureRowV1):
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_requires_typed_feature_row"
        )
    try:
        canonical_row = TerminalRiskFeatureRowV1.model_validate(feature_row.model_dump(mode="json"))
        canonical_preparation = MetaSynItemRiskPreparationV1.model_validate(
            preparation.model_dump(mode="json")
            if isinstance(preparation, MetaSynItemRiskPreparationV1)
            else preparation
        )
        canonical_features = MetaSynTerminalRiskFeatureSetV1.model_validate(
            feature_set.model_dump(mode="json")
            if isinstance(feature_set, MetaSynTerminalRiskFeatureSetV1)
            else feature_set
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_input_contract_invalid"
        ) from exc
    canonical_run = validate_metasyn_item_risk_calibration_run_v1(
        calibration_run=calibration_run,
        preparation=canonical_preparation,
        feature_set=canonical_features,
        analysis=analysis,
        repository_root=repository_root,
        adjudication_sidecar_directory=adjudication_sidecar_directory,
    )
    if (
        canonical_run.status != "calibrated_scheduling_only"
        or not canonical_run.scheduling_authority
        or canonical_run.bounds_receipt is None
    ):
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_requires_calibrated_scheduling_receipt"
        )
    if (
        canonical_features.preparation != canonical_preparation
        or canonical_run.preparation_sha256 != canonical_preparation.preparation_sha256
        or canonical_run.feature_set_sha256 != canonical_features.feature_set_sha256
    ):
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_artifact_lineage_mismatch"
        )
    matching_rows = [
        row
        for row in canonical_features.rows
        if row.feature_row_sha256 == canonical_row.feature_row_sha256
    ]
    if len(matching_rows) != 1 or matching_rows[0] != canonical_row:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_feature_row_not_member"
        )
    receipt = canonical_run.bounds_receipt
    domain = canonical_row.item.domain
    risk_bin, cell = _bound_cell_for_replayed_feature_row(
        feature_row=canonical_row,
        receipt=receipt,
    )
    payload = {
        "assignment_version": ASSIGNMENT_VERSION,
        "feature_row": canonical_row,
        "feature_row_sha256": canonical_row.feature_row_sha256,
        "feature_set_sha256": canonical_features.feature_set_sha256,
        "preparation_sha256": canonical_preparation.preparation_sha256,
        "calibration_run_sha256": canonical_run.calibration_run_sha256,
        "bounds_receipt": receipt,
        "bounds_receipt_sha256": receipt.receipt_sha256,
        "pipeline_sha256": canonical_preparation.pipeline_sha256,
        "score_model_sha256": canonical_preparation.score_model_sha256,
        "item_id": canonical_row.item.item_id,
        "question_id": canonical_row.item.question_id,
        "publication_id": canonical_row.item.publication_id,
        "paper_id": canonical_row.item.paper_id,
        "domain": domain,
        "score_input_sha256": canonical_row.score_input_sha256,
        "computed_risk_score": canonical_row.risk_score,
        "bin_id": risk_bin.bin_id,
        "bin_lower": risk_bin.lower,
        "bin_upper": risk_bin.upper,
        "bin_upper_inclusive": risk_bin.upper_inclusive,
        "bound_cell_sha256": hash_canonical(cell),
        "cell_calibration_units": cell.cell_calibration_units,
        "cell_observed_errors": cell.cell_observed_errors,
        "conservative_group_upper_error_rate": cell.upper_cell_error_rate,
        "estimand": cell.estimand,
        "score_source": "externally_replayed_terminal_feature_row",
        "accepts_caller_supplied_score": False,
        "accepts_caller_supplied_domain": False,
        "accepts_caller_supplied_bin": False,
        "individual_item_probability_authority": False,
        "scheduling_authority": True,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return ArtifactBackedItemRiskAssignmentV1.model_validate(
        {**payload, "assignment_sha256": hash_canonical(payload)}
    )


def validate_artifact_backed_item_risk_assignment_v1(
    *,
    assignment: ArtifactBackedItemRiskAssignmentV1 | Mapping[str, Any],
    preparation: MetaSynItemRiskPreparationV1 | Mapping[str, Any],
    feature_set: MetaSynTerminalRiskFeatureSetV1 | Mapping[str, Any],
    calibration_run: MetaSynItemRiskCalibrationRunV1 | Mapping[str, Any],
    analysis: MetaSynGroundedAnalysisV2 | Mapping[str, Any],
    repository_root: Path,
    adjudication_sidecar_directory: Path,
) -> ArtifactBackedItemRiskAssignmentV1:
    """Externally replay an artifact-backed risk assignment."""

    try:
        canonical = ArtifactBackedItemRiskAssignmentV1.model_validate(
            assignment.model_dump(mode="json")
            if isinstance(assignment, ArtifactBackedItemRiskAssignmentV1)
            else assignment
        )
    except ValueError as exc:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_contract_invalid"
        ) from exc
    replayed = assign_artifact_backed_item_risk_v1(
        feature_row=canonical.feature_row,
        preparation=preparation,
        feature_set=feature_set,
        calibration_run=calibration_run,
        analysis=analysis,
        repository_root=repository_root,
        adjudication_sidecar_directory=adjudication_sidecar_directory,
    )
    if replayed != canonical:
        raise MetaSynItemRiskCalibrationV1Error(
            "metasyn_item_risk_v1_assignment_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "ASSIGNMENT_VERSION",
    "CALIBRATION_RUN_VERSION",
    "CLI_PATH",
    "FEATURE_SET_VERSION",
    "MODULE_PATH",
    "AdjudicationQuestionFileV1",
    "AdjudicationSidecarManifestV1",
    "ArtifactBackedItemRiskAssignmentV1",
    "ArtifactBackedItemRiskBoundsReceiptV1",
    "CompleteQuestionAdjudicationSidecarV1",
    "FixedTerminalRiskScorePolicyV1",
    "FrozenEligibleRiskItemV1",
    "FrozenQuestionSplitV1",
    "MetaSynItemRiskCalibrationRunV1",
    "MetaSynItemRiskCalibrationV1Error",
    "MetaSynItemRiskPreparationV1",
    "MetaSynTerminalRiskFeatureSetV1",
    "TerminalRiskFeatureRowV1",
    "assign_artifact_backed_item_risk_v1",
    "calibrate_metasyn_item_risk_v1",
    "compute_metasyn_item_risk_pipeline_fingerprint_v1",
    "freeze_fixed_terminal_risk_score_policy_v1",
    "materialize_metasyn_terminal_risk_features_v1",
    "prepare_metasyn_item_risk_calibration_v1",
    "validate_artifact_backed_item_risk_assignment_v1",
    "validate_metasyn_item_risk_calibration_run_v1",
    "validate_metasyn_item_risk_preparation_v1",
    "validate_metasyn_terminal_risk_features_v1",
]
