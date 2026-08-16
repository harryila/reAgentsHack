from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.config import (
    FIXTURE_QUESTION_IDS,
    ConfigAuthorizationError,
    QuestionConfig,
    config_sha256,
)


def as_payload(config: QuestionConfig) -> dict:
    return deepcopy(config.model_dump(mode="json", exclude_none=False))


def test_all_four_named_fixture_configs_are_locked_and_hash_distinct(
    fixture_configs: dict[str, QuestionConfig],
) -> None:
    assert set(fixture_configs) == FIXTURE_QUESTION_IDS
    assert all(config.status == "locked" for config in fixture_configs.values())
    hashes = [config_sha256(config) for config in fixture_configs.values()]
    assert len(set(hashes)) == 4
    assert all(len(value) == 64 for value in hashes)


def test_fixture_runtime_requires_both_config_and_explicit_flag(
    fixture_config: QuestionConfig,
) -> None:
    fixture_config.authorize_stage("s3", explicit_fixture=True)
    with pytest.raises(ConfigAuthorizationError, match="requires_explicit_fixture"):
        fixture_config.authorize_stage("s3")
    with pytest.raises(ConfigAuthorizationError, match="forbids_live_provider"):
        fixture_config.authorize_stage("s3", explicit_fixture=True, live_provider=True)

    payload = as_payload(fixture_config)
    payload["question_id"] = "selected-question"
    payload["demo"]["fixture_mode"] = False
    production = QuestionConfig.model_validate(payload)
    production.authorize_stage("s3")
    with pytest.raises(ConfigAuthorizationError, match="fixture_flag_forbidden"):
        production.authorize_stage("s3", explicit_fixture=True)


def test_triage_is_isolated_and_probe_hard_caps_at_ten(
    fixture_config: QuestionConfig,
) -> None:
    payload = as_payload(fixture_config)
    payload.update(
        {
            "status": "triage",
            "question_id": "triage-c",
            "variant_b": None,
            "anchor_papers": None,
            "recovery_check": None,
            "demo": None,
        }
    )
    triage = QuestionConfig.model_validate(payload)
    triage.authorize_stage("s1")
    triage.authorize_stage("s2")
    triage.authorize_stage("triage_probe", triage_paper_count=10)
    with pytest.raises(ConfigAuthorizationError, match="paper_count_must_be_1_to_10"):
        triage.authorize_stage("triage_probe", triage_paper_count=11)
    with pytest.raises(ConfigAuthorizationError, match="not_authorized_for_production"):
        triage.authorize_stage("s3")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant_b", None),
        ("anchor_papers", None),
        ("recovery_check", None),
        ("demo", None),
    ],
)
def test_locked_config_cannot_omit_final_contract_fields(
    fixture_config: QuestionConfig, field: str, value: object
) -> None:
    payload = as_payload(fixture_config)
    payload[field] = value
    with pytest.raises(ValidationError, match="locked_config_missing"):
        QuestionConfig.model_validate(payload)


def test_locked_moderator_family_caps_and_outcome_family_ban(
    fixture_config: QuestionConfig,
) -> None:
    payload = as_payload(fixture_config)
    template = deepcopy(payload["moderators"][0])
    payload["moderators"].extend(
        [{**deepcopy(template), "name": f"extra_{index}"} for index in range(5)]
    )
    with pytest.raises(ValidationError, match="more_than_six_tested"):
        QuestionConfig.model_validate(payload)

    payload = as_payload(fixture_config)
    payload["moderators"][0]["name"] = "outcome_family"
    payload["variant_b"]["axes"][0] = "outcome_family"
    payload["demo"]["moderator_display_names"]["outcome_family"] = payload["demo"][
        "moderator_display_names"
    ].pop("dose_regime")
    with pytest.raises(ValidationError, match="outcome_family_cannot_be_tested"):
        QuestionConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source": "remap"}, "source"),
        ({"kind": "paper_constant", "permutation": "none"}, "paper_constant_requires"),
        (
            {"kind": "within_paper", "permutation": "paper_summary", "paper_summary": None},
            "tested_within_paper_requires",
        ),
    ],
)
def test_locked_moderator_source_and_permutation_pairs_are_strict(
    fixture_config: QuestionConfig, updates: dict[str, object], message: str
) -> None:
    payload = as_payload(fixture_config)
    payload["moderators"][0].update(updates)
    with pytest.raises(ValidationError, match=message):
        QuestionConfig.model_validate(payload)


def test_numeric_bins_are_named_complete_contiguous_intervals(
    fixture_config: QuestionConfig,
) -> None:
    payload = as_payload(fixture_config)
    numeric = payload["moderators"][0]
    numeric.update(
        {
            "name": "dose_value",
            "type": "float",
            "allowed_values": None,
            "bins": [
                {
                    "label": "low",
                    "lower": None,
                    "upper": 50.0,
                    "lower_inclusive": True,
                    "upper_inclusive": False,
                },
                {
                    "label": "high",
                    "lower": 50.0,
                    "upper": None,
                    "lower_inclusive": True,
                    "upper_inclusive": False,
                },
            ],
        }
    )
    payload["variant_b"]["axes"][0] = "dose_value"
    display = payload["demo"]["moderator_display_names"].pop("dose_regime")
    payload["demo"]["moderator_display_names"]["dose_value"] = display
    assert QuestionConfig.model_validate(payload).moderators[0].declared_levels == ["low", "high"]

    payload["moderators"][0]["bins"][1]["lower"] = 60.0
    with pytest.raises(ValidationError, match="numeric_bins_must_be_contiguous"):
        QuestionConfig.model_validate(payload)


def test_spoken_question_is_capped_at_twenty_words(fixture_config: QuestionConfig) -> None:
    payload = as_payload(fixture_config)
    payload["demo"]["spoken_question"] = " ".join(f"word{index}" for index in range(21))
    with pytest.raises(ValidationError, match="spoken_question_exceeds_20_words"):
        QuestionConfig.model_validate(payload)


def test_entire_canonical_config_hash_changes_for_scientific_edits(
    fixture_config: QuestionConfig,
) -> None:
    baseline = config_sha256(fixture_config)
    mutations = []

    eligibility = as_payload(fixture_config)
    eligibility["eligibility"]["exclude"].append("new exclusion")
    mutations.append(eligibility)

    outcome = as_payload(fixture_config)
    outcome["outcomes"]["endpoint_direction_overrides"]["peak_power"] = {
        "increase_definition": "higher",
        "decrease_definition": "lower",
        "no_effect_definition": "unchanged",
    }
    mutations.append(outcome)

    moderator = as_payload(fixture_config)
    moderator["moderators"][0]["allowed_values"].append("medium")
    mutations.append(moderator)

    observed = {config_sha256(QuestionConfig.model_validate(payload)) for payload in mutations}
    assert baseline not in observed
    assert len(observed) == len(mutations)
