"""Validated question configuration and stage authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from literature_multiverse.models import CanonicalDirection, ContractModel
from literature_multiverse.paths import PATHS

FIXTURE_QUESTION_IDS: frozenset[str] = frozenset(
    {"fixture-a", "fixture-b-story", "fixture-b-m4", "fixture-b-incomplete"}
)
PRODUCTION_STAGES: frozenset[str] = frozenset({"s3", "s4", "s5", "s6", "s7"})
TRIAGE_STAGES: frozenset[str] = frozenset({"s1", "s2", "triage_probe"})
ALL_STAGES: frozenset[str] = frozenset({"s0", *TRIAGE_STAGES, *PRODUCTION_STAGES})
_SLUG = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


class ConfigAuthorizationError(ValueError):
    """A valid config is not authorized for the requested runtime context."""


class TargetRelation(ContractModel):
    exposure: str
    comparator: str
    outcome: str
    increase_definition: str
    decrease_definition: str
    no_effect_definition: str
    desirability_claims_allowed: Literal[False]
    # Case-insensitive substrings, at least one of which must appear in an accepted
    # finding's intervention or intervention_class text.  Rows describing a non-exposure
    # arm (e.g. the plain-exercise arm of a four-arm factorial trial) are quarantined
    # deterministically (2026-08-16 census-audit remediation).  Empty list disables.
    exposure_terms: list[str] = Field(default_factory=list)


class AuditPaperExclusion(ContractModel):
    """A paper excluded by the human full-text audit, PRISMA-style, with provenance."""

    paper_id: str
    reason: str
    audited_at: str


class Eligibility(ContractModel):
    include: Annotated[list[str], Field(min_length=1)]
    exclude: list[str]
    article_types: Annotated[list[str], Field(min_length=1)]
    # Confirmed article types removed at retrieval time (`--exclude-article-type`); a
    # document whose type the index cannot confirm is still retrieved, and extraction
    # remains the authoritative eligibility judgment.
    exclude_article_types: list[str] = Field(default_factory=lambda: ["review-article"])


class QueryFamily(ContractModel):
    id: str
    queries: Annotated[list[str], Field(min_length=1)]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError("invalid_query_family_id")
        return value


class SearchConfig(ContractModel):
    sources: Annotated[list[str], Field(min_length=1)]
    query_families: Annotated[list[QueryFamily], Field(min_length=1)]
    use_all: Literal[True]
    # The CLI's search default is 100; every retrieval is bounded and recorded.
    per_query_limit: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def validate_query_family_ids(self) -> SearchConfig:
        ids = [family.id for family in self.query_families]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_query_family_id")
        return self


class DirectionOverride(ContractModel):
    increase_definition: str
    decrease_definition: str
    no_effect_definition: str


class OutcomesConfig(ContractModel):
    primary_family: str | None
    family_map: dict[str, str]
    included_primary_endpoints: list[str]
    endpoint_direction_overrides: dict[str, DirectionOverride]
    # Canonicalizes free-text extracted outcome names onto the locked endpoint registry
    # (exact case-insensitive keys, or explicit ``re:`` patterns — same resolution rules
    # as family_map).  Values must be included primary endpoints; unmatched names keep
    # the extractor's wording and resolve through family_map only.
    endpoint_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_mapping(self) -> OutcomesConfig:
        if len(self.included_primary_endpoints) != len(set(self.included_primary_endpoints)):
            raise ValueError("duplicate_primary_endpoint")
        unknown_overrides = set(self.endpoint_direction_overrides) - set(
            self.included_primary_endpoints
        )
        if unknown_overrides:
            raise ValueError(
                f"direction_override_for_unknown_endpoint:{','.join(sorted(unknown_overrides))}"
            )
        unknown_endpoint_targets = set(self.endpoint_map.values()) - set(
            self.included_primary_endpoints
        )
        if unknown_endpoint_targets:
            raise ValueError(
                "endpoint_map_target_not_included:"
                + ",".join(sorted(unknown_endpoint_targets))
            )
        return self


class NumericBin(ContractModel):
    """A named, non-overlapping numeric analysis interval."""

    label: str
    lower: float | None
    upper: float | None
    lower_inclusive: bool = True
    upper_inclusive: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> NumericBin:
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("numeric_bin_lower_must_be_less_than_upper")
        return self


ModeratorType = Literal["categorical", "float", "int", "bool"]
ModeratorSource = Literal["fixed", "topic"]
ModeratorRole = Literal["tested", "descriptive"]
ModeratorKind = Literal["paper_constant", "within_paper"]
PermutationKind = Literal["paper", "paper_summary", "none"]


class ModeratorSpec(ContractModel):
    name: str
    type: ModeratorType
    source: ModeratorSource
    role: ModeratorRole
    kind: ModeratorKind
    permutation: PermutationKind
    paper_summary: str | None
    display_name: str
    allowed_values: list[str | int | float | bool] | None
    bins: list[NumericBin] | None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError("invalid_moderator_name")
        return value

    @model_validator(mode="after")
    def validate_representation_and_permutation(self) -> ModeratorSpec:
        if self.kind == "paper_constant":
            if self.permutation != "paper" or self.paper_summary is not None:
                raise ValueError("paper_constant_requires_paper_permutation_without_summary")
        elif self.role == "tested" and (
            self.permutation != "paper_summary" or self.paper_summary is None
        ):
            raise ValueError("tested_within_paper_requires_summary_permutation")
        elif self.permutation == "paper_summary" and self.paper_summary is None:
            raise ValueError("paper_summary_permutation_requires_summary")
        elif self.permutation == "paper" and self.kind != "paper_constant":
            raise ValueError("paper_permutation_requires_paper_constant")
        if self.role == "tested" and self.permutation == "none":
            raise ValueError("tested_moderator_requires_permutation")

        if self.type in {"categorical", "bool"}:
            if not self.allowed_values or self.bins is not None:
                raise ValueError("categorical_or_bool_requires_allowed_values_only")
            serialized = [json.dumps(value, sort_keys=True) for value in self.allowed_values]
            if len(serialized) != len(set(serialized)):
                raise ValueError("duplicate_moderator_allowed_value")
            if self.type == "categorical" and any(
                not isinstance(value, str) for value in self.allowed_values
            ):
                raise ValueError("categorical_allowed_values_must_be_strings")
            if self.type == "bool" and (
                len(self.allowed_values) != 2
                or {type(value) is bool and value for value in self.allowed_values}
                != {False, True}
            ):
                raise ValueError("bool_allowed_values_must_be_false_true")
        else:
            if self.allowed_values is not None or not self.bins or len(self.bins) < 2:
                raise ValueError("numeric_moderator_requires_two_or_more_bins_only")
            labels = [item.label for item in self.bins]
            if len(labels) != len(set(labels)):
                raise ValueError("duplicate_numeric_bin_label")
            if self.bins[0].lower is not None or self.bins[-1].upper is not None:
                raise ValueError("numeric_bins_must_cover_full_range")
            for left, right in zip(self.bins[:-1], self.bins[1:], strict=True):
                if left.upper != right.lower:
                    raise ValueError("numeric_bins_must_be_contiguous")
                if left.upper_inclusive == right.lower_inclusive:
                    raise ValueError("numeric_bin_boundary_must_belong_to_exactly_one_bin")
        return self

    @property
    def declared_levels(self) -> list[str | int | float | bool]:
        if self.allowed_values is not None:
            return list(self.allowed_values)
        assert self.bins is not None
        return [item.label for item in self.bins]


class AnalysisConfig(ContractModel):
    seed: int
    labels: list[CanonicalDirection]
    max_missingness_headline: Annotated[float, Field(ge=0, le=1)]
    min_papers_per_level: Annotated[int, Field(ge=1)]
    cv_max_folds: Annotated[int, Field(ge=2)]
    permutation_count: Annotated[int, Field(ge=1)]
    bootstrap_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_registered_constants(self) -> AnalysisConfig:
        if self.labels != [
            CanonicalDirection.INCREASE,
            CanonicalDirection.NO_EFFECT,
            CanonicalDirection.DECREASE,
        ]:
            raise ValueError("analysis_labels_must_be_canonical_three_class_order")
        return self


class VariantBConfig(ContractModel):
    axes: Annotated[list[str], Field(min_length=1)]
    primary_endpoints: Annotated[list[str], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_values(self) -> VariantBConfig:
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("duplicate_variant_b_axis")
        if len(self.primary_endpoints) != len(set(self.primary_endpoints)):
            raise ValueError("duplicate_variant_b_endpoint")
        return self


class TriageConfig(ContractModel):
    candidate_topic: str
    score_record: str
    # Probe-sample relevance prefilter: a paper qualifies for the 10-paper probe only when
    # its title+abstract matches at least one term from EVERY group (case-insensitive
    # substring).  This spends probe slots on plausibly-eligible papers; the production
    # corpus itself stays recall-heavy, and extraction remains the eligibility judgment.
    probe_keyword_groups: list[list[str]] = Field(default_factory=list)
    # Title-level probe exclusions (case-insensitive substring), e.g. obvious reviews.
    probe_exclude_terms: list[str] = Field(default_factory=list)


class AnchorPaper(ContractModel):
    paper_id: str
    expected_eligible: bool
    expected_finding_count: Annotated[int, Field(ge=0)]
    expected_directions: list[CanonicalDirection]
    notes: str | None = None


class RecoveryCheck(ContractModel):
    kind: Literal["published", "narrative", "none"]
    statement: str
    source: str | None

    @model_validator(mode="after")
    def validate_source(self) -> RecoveryCheck:
        if self.kind == "published" and self.source is None:
            raise ValueError("published_recovery_check_requires_source")
        if self.kind == "none" and self.source is not None:
            raise ValueError("none_recovery_check_source_must_be_null")
        return self


class DemoConfig(ContractModel):
    hook: str
    spoken_question: str
    moderator_display_names: dict[str, str]
    corpus_qualifier: Literal["our retrieved corpus"]
    fixture_mode: bool = False

    @field_validator("spoken_question")
    @classmethod
    def validate_spoken_question(cls, value: str) -> str:
        if len(value.split()) > 20:
            raise ValueError("spoken_question_exceeds_20_words")
        return value


class QuestionConfig(ContractModel):
    """The entire canonical triage or locked topic contract."""

    schema_version: Literal["1"]
    status: Literal["triage", "locked"]
    question_id: str
    research_question: str
    target_relation: TargetRelation
    eligibility: Eligibility
    search: SearchConfig
    outcomes: OutcomesConfig
    moderators: list[ModeratorSpec]
    analysis: AnalysisConfig
    variant_b: VariantBConfig | None = None
    triage: TriageConfig
    anchor_papers: list[AnchorPaper] | None = None
    recovery_check: RecoveryCheck | None = None
    demo: DemoConfig | None = None
    # Papers the frozen human audit excluded at full-text level, with recorded reasons.
    # Applied at s2 as a deterministic screen rule; the config hash change makes the
    # exclusion visible in every downstream lineage record.
    audit_paper_exclusions: list[AuditPaperExclusion] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("invalid_question_id")
        return value

    @model_validator(mode="after")
    def validate_state_and_scientific_contract(self) -> QuestionConfig:
        names = [moderator.name for moderator in self.moderators]
        if len(names) != len(set(names)):
            raise ValueError("duplicate_moderator_name")
        if sum(moderator.role == "tested" for moderator in self.moderators) > 6:
            raise ValueError("more_than_six_tested_moderators")
        if sum(moderator.role == "descriptive" for moderator in self.moderators) > 2:
            raise ValueError("more_than_two_descriptive_moderators")
        if any(
            moderator.role == "tested"
            and moderator.name in {"outcome_family", self.outcomes.primary_family}
            for moderator in self.moderators
        ):
            raise ValueError("outcome_family_cannot_be_tested_moderator")

        if self.status == "triage":
            if self.question_id in FIXTURE_QUESTION_IDS:
                raise ValueError("fixture_question_must_be_locked")
            if self.demo is not None and self.demo.fixture_mode:
                raise ValueError("triage_config_cannot_enable_fixture_mode")
            return self

        required = {
            "outcomes.primary_family": self.outcomes.primary_family,
            "outcomes.included_primary_endpoints": self.outcomes.included_primary_endpoints,
            "variant_b": self.variant_b,
            "anchor_papers": self.anchor_papers,
            "recovery_check": self.recovery_check,
            "demo": self.demo,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"locked_config_missing:{','.join(missing)}")
        assert self.outcomes.primary_family is not None
        assert self.variant_b is not None
        assert self.demo is not None

        endpoints = set(self.outcomes.included_primary_endpoints)
        incorrectly_mapped = sorted(
            endpoint
            for endpoint in endpoints
            if self.outcomes.family_map.get(endpoint) != self.outcomes.primary_family
        )
        if incorrectly_mapped:
            raise ValueError(
                f"primary_endpoint_not_mapped_to_primary_family:{','.join(incorrectly_mapped)}"
            )
        unknown_endpoints = set(self.variant_b.primary_endpoints) - endpoints
        if unknown_endpoints:
            raise ValueError(
                f"variant_b_unknown_primary_endpoint:{','.join(sorted(unknown_endpoints))}"
            )
        moderator_by_name = {moderator.name: moderator for moderator in self.moderators}
        unknown_axes = set(self.variant_b.axes) - set(moderator_by_name)
        if unknown_axes:
            raise ValueError(f"variant_b_unknown_axis:{','.join(sorted(unknown_axes))}")
        for axis in self.variant_b.axes:
            if not moderator_by_name[axis].declared_levels:
                raise ValueError(f"variant_b_axis_without_complete_levels:{axis}")
        unknown_display_names = set(self.demo.moderator_display_names) - set(moderator_by_name)
        if unknown_display_names:
            raise ValueError(
                f"demo_display_name_for_unknown_moderator:{','.join(sorted(unknown_display_names))}"
            )
        if self.demo.fixture_mode != (self.question_id in FIXTURE_QUESTION_IDS):
            raise ValueError("fixture_mode_question_id_mismatch")
        return self

    def authorize_stage(
        self,
        stage: str,
        *,
        explicit_fixture: bool = False,
        live_provider: bool = False,
        triage_paper_count: int | None = None,
    ) -> None:
        authorize_stage(
            self,
            stage,
            explicit_fixture=explicit_fixture,
            live_provider=live_provider,
            triage_paper_count=triage_paper_count,
        )


def canonical_config_dict(config: QuestionConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=False)


def config_sha256(config: QuestionConfig) -> str:
    encoded = json.dumps(
        canonical_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_question_config(path: str | Path, *, require_locked: bool = False) -> QuestionConfig:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid_question_config_yaml:{config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"question_config_root_must_be_mapping:{config_path}")
    config = QuestionConfig.model_validate(data)
    if require_locked and config.status != "locked":
        raise ConfigAuthorizationError("locked_config_required")
    if config_path.stem != config.question_id:
        raise ValueError(
            f"question_config_filename_mismatch:expected={config.question_id}.yaml"
        )
    return config


def load_config_for_question(
    question_id: str, *, root: Path | None = None, require_locked: bool = False
) -> QuestionConfig:
    path = (
        PATHS.config_path(question_id)
        if root is None
        else root / "configs" / "questions" / f"{question_id}.yaml"
    )
    return load_question_config(path, require_locked=require_locked)


def authorize_stage(
    config: QuestionConfig,
    stage: str,
    *,
    explicit_fixture: bool = False,
    live_provider: bool = False,
    triage_paper_count: int | None = None,
) -> None:
    """Apply config-state, triage-isolation, and fixture runtime boundaries."""

    if stage not in ALL_STAGES:
        raise ConfigAuthorizationError(f"unknown_stage:{stage}")
    if config.status == "triage":
        if stage not in TRIAGE_STAGES:
            raise ConfigAuthorizationError("triage_config_not_authorized_for_production_stage")
        if stage == "triage_probe" and (
            triage_paper_count is None or not 1 <= triage_paper_count <= 10
        ):
            raise ConfigAuthorizationError("triage_probe_paper_count_must_be_1_to_10")
        return
    if stage == "triage_probe":
        raise ConfigAuthorizationError("triage_probe_requires_triage_config")
    if stage in PRODUCTION_STAGES and config.status != "locked":
        raise ConfigAuthorizationError("locked_config_required")

    assert config.demo is not None
    if config.demo.fixture_mode:
        if not explicit_fixture:
            raise ConfigAuthorizationError("fixture_config_requires_explicit_fixture_flag")
        if live_provider:
            raise ConfigAuthorizationError("fixture_mode_forbids_live_provider")
    elif explicit_fixture:
        raise ConfigAuthorizationError("fixture_flag_forbidden_for_non_fixture_config")


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Literature Multiverse question config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--require-locked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    config = load_question_config(args.path, require_locked=args.require_locked)
    print(f"valid {config.question_id} {config.status} sha256={config_sha256(config)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke usage
    raise SystemExit(main())
