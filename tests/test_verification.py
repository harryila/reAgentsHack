from __future__ import annotations

import json
from pathlib import Path

import pytest

from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.providers import FixtureProvider
from literature_multiverse.verification import (
    VerificationContractError,
    fixture_verification,
    reconcile_model_decisions,
    requested_grounded_rows,
    verification_output_schema,
    verify_with_provider,
)


def _findings() -> list[dict[str, object]]:
    return [
        {
            "finding_id": "f1",
            "grounding_status": "exact",
            "effect_direction": "increase",
            "outcome_name": "power",
            "timepoint_raw": "post",
            "comparator": "control",
            "evidence_quote": "Power increased.",
            "evidence_lines": ["L10"],
        },
        {
            "finding_id": "f2",
            "grounding_status": "mismatch",
            "effect_direction": "decrease",
        },
        {
            "finding_id": "f3",
            "grounding_status": "exact",
            "effect_direction": "no_effect",
            "outcome_name": "fatigue",
            "timepoint_raw": None,
            "comparator": "control",
            "evidence_quote": "There was no difference.",
            "evidence_lines": ["L20"],
        },
    ]


def _rendered(repo_root: Path):
    return render_prompt_file(
        repo_root / "prompts" / "quote_verification.md",
        {
            "DIRECTION_DEFINITIONS_JSON": json.dumps(
                {"increase": "higher", "no_effect": "no difference", "decrease": "lower"},
                sort_keys=True,
            ),
            "FINDINGS_JSON": json.dumps(requested_grounded_rows(_findings()), sort_keys=True),
        },
    )


def test_request_selection_and_schema_are_closed() -> None:
    rows = requested_grounded_rows(_findings())
    assert [row["finding_id"] for row in rows] == ["f1", "f3"]
    schema = verification_output_schema(["f1", "f3"])
    assert schema["additionalProperties"] is False
    item = schema["properties"]["decisions"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["finding_id"]["enum"] == ["f1", "f3"]
    # Provider structured outputs reject minItems > 1 and maxItems entirely; the exact
    # count is enforced post-hoc by reconcile_verification.
    assert schema["properties"]["decisions"]["minItems"] == 1
    assert "maxItems" not in schema["properties"]["decisions"]


def test_provider_verification_calls_once_and_reconciles_order(repo_root: Path) -> None:
    provider = FixtureProvider(
        {
            (
                "quote_verification",
                "cohort-1",
            ): {
                "decisions": [
                    {"finding_id": "f3", "model_status": "disagree", "rationale": "wrong"},
                    {"finding_id": "f1", "model_status": "agree", "rationale": "direct"},
                ]
            }
        }
    )
    record = verify_with_provider(
        findings=_findings(),
        rendered_prompt=_rendered(repo_root),
        provider=provider,
        request_key="cohort-1",
    )
    assert provider.calls == [("quote_verification", "cohort-1")]
    assert record.requested_finding_ids == ["f1", "f3"]
    assert [decision.finding_id for decision in record.decisions] == ["f1", "f3"]
    assert record.decisions[1].model_status == "disagree"
    assert all(decision.adjudication == "none" for decision in record.decisions)


def test_fixture_verifier_never_needs_provider_and_supports_pinned_override(
    repo_root: Path,
) -> None:
    record = fixture_verification(
        findings=_findings(),
        rendered_prompt=_rendered(repo_root),
        status_overrides={"f3": "unverifiable"},
    )
    assert record.provider == "fixture"
    assert [decision.model_status for decision in record.decisions] == [
        "agree",
        "unverifiable",
    ]


@pytest.mark.parametrize(
    "raw,match",
    [
        (
            [
                {"finding_id": "f1", "model_status": "agree"},
                {"finding_id": "f1", "model_status": "agree"},
            ],
            "duplicate_verification_response",
        ),
        ([{"finding_id": "other", "model_status": "agree"}], "unknown_verification_response"),
        ([{"finding_id": "f1", "model_status": "agree"}], "missing_verification_responses"),
        (
            [
                {"finding_id": "f1", "model_status": "invented"},
                {"finding_id": "f3", "model_status": "agree"},
            ],
            "invalid_verification_response",
        ),
    ],
)
def test_reconciliation_rejects_duplicate_unknown_missing_and_invalid(
    raw: list[dict[str, str]], match: str
) -> None:
    with pytest.raises(VerificationContractError, match=match):
        reconcile_model_decisions(["f1", "f3"], raw)
