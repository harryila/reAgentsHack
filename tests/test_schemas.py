from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.config import QuestionConfig, config_sha256
from literature_multiverse.schemas import (
    assert_closed_object_schema,
    generate_extraction_schema,
    schema_sha256,
    validate_extraction_payload,
    validate_finding_row,
)


def extraction_finding(finding_payload: dict) -> dict:
    pipeline_keys = {
        "finding_id",
        "paper_id",
        "doc_id",
        "map_result_id",
        "array_position",
        "prompt_version",
        "schema_version",
        "cfghash",
        "grounding_status",
        "evidence_section",
        "section_flagged",
        "normalization_warnings",
    }
    return {
        key: deepcopy(value)
        for key, value in finding_payload.items()
        if key not in pipeline_keys
    }


def test_generated_schema_is_closed_recursively_and_has_no_model_identity(
    fixture_config: QuestionConfig,
) -> None:
    schema = generate_extraction_schema(fixture_config)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["x-question-config-sha256"] == config_sha256(fixture_config)
    assert_closed_object_schema(schema)
    serialized = json.dumps(schema, sort_keys=True)
    for forbidden in ("paper_id", "doc_id", "finding_id", "map_result_id"):
        assert f'"{forbidden}"' not in serialized


def test_extraction_requires_every_nullable_key_and_canonical_non_null_direction(
    fixture_config: QuestionConfig, finding_payload: dict
) -> None:
    finding = extraction_finding(finding_payload)
    envelope = {"eligible": True, "exclusion_reason": None, "findings": [finding]}
    assert validate_extraction_payload(envelope, fixture_config).eligible is True

    missing = deepcopy(envelope)
    del missing["findings"][0]["study_type"]
    with pytest.raises(ValidationError, match="study_type"):
        validate_extraction_payload(missing, fixture_config)

    null_direction = deepcopy(envelope)
    null_direction["findings"][0]["effect_direction"] = None
    with pytest.raises(ValidationError, match="effect_direction"):
        validate_extraction_payload(null_direction, fixture_config)

    legacy_alias = deepcopy(envelope)
    legacy_alias["findings"][0]["effect_direction"] = "positive"
    with pytest.raises(ValidationError, match="effect_direction"):
        validate_extraction_payload(legacy_alias, fixture_config)


def test_ineligible_extraction_must_have_zero_findings(
    fixture_config: QuestionConfig, finding_payload: dict
) -> None:
    envelope = {
        "eligible": False,
        "exclusion_reason": "wrong intervention",
        "findings": [extraction_finding(finding_payload)],
    }
    with pytest.raises(ValidationError, match="ineligible_extraction_must_have_zero"):
        validate_extraction_payload(envelope, fixture_config)


def test_topic_moderator_keys_and_types_come_only_from_config(
    fixture_config: QuestionConfig, finding_payload: dict
) -> None:
    finding = extraction_finding(finding_payload)
    envelope = {"eligible": True, "exclusion_reason": None, "findings": [finding]}

    extra = deepcopy(envelope)
    extra["findings"][0]["moderators"]["model_invented"] = "value"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        validate_extraction_payload(extra, fixture_config)

    missing = deepcopy(envelope)
    del missing["findings"][0]["moderators"]["dose_regime"]
    with pytest.raises(ValidationError, match="dose_regime"):
        validate_extraction_payload(missing, fixture_config)

    wrong_value = deepcopy(envelope)
    wrong_value["findings"][0]["moderators"]["dose_regime"] = "invented"
    with pytest.raises(ValidationError, match="dose_regime"):
        validate_extraction_payload(wrong_value, fixture_config)


def test_normalized_topic_row_accepts_closed_alias_after_pre_normalization_only(
    fixture_config: QuestionConfig, finding_payload: dict
) -> None:
    aliased = deepcopy(finding_payload)
    aliased["effect_direction"] = "positive"
    aliased["finding_id"] = aliased["finding_id"].replace(
        aliased["finding_id"].rsplit(":", maxsplit=1)[-1],
        "placeholder",
    )
    from literature_multiverse.models import make_finding_id

    aliased["finding_id"] = make_finding_id(
        paper_id=aliased["paper_id"],
        map_result_id=aliased["map_result_id"],
        array_position=aliased["array_position"],
        outcome_name=aliased["outcome_name"],
        timepoint_raw=aliased["timepoint_raw"],
        dose_raw=aliased["dose_raw"],
        effect_direction="positive",
    )
    assert validate_finding_row(aliased, fixture_config).effect_direction.value == "increase"

    extra = deepcopy(finding_payload)
    extra["moderators"]["surprise"] = "x"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        validate_finding_row(extra, fixture_config)


def test_schema_hash_changes_when_topic_schema_changes(fixture_config: QuestionConfig) -> None:
    baseline = schema_sha256(fixture_config)
    payload = fixture_config.model_dump(mode="json", exclude_none=False)
    payload["moderators"][0]["allowed_values"].append("medium")
    changed = QuestionConfig.model_validate(payload)
    assert schema_sha256(changed) != baseline


def test_committed_fixture_schema_snapshots_match_generator(
    repo_root, fixture_configs: dict[str, QuestionConfig]
) -> None:
    for question_id, config in fixture_configs.items():
        path = repo_root / "schemas" / f"extraction.{question_id}.schema.json"
        assert json.loads(path.read_text(encoding="utf-8")) == generate_extraction_schema(config)
