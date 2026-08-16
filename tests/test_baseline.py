from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from literature_multiverse.baseline import (
    BaselineContractError,
    create_baseline,
    select_primary_rows,
    write_baseline_once,
)
from literature_multiverse.providers import FixtureProvider, ProviderError

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
COHORT_HASH = "c" * 64


def _rows() -> list[dict[str, str]]:
    return [
        {"paper_id": "p1", "effect_direction": "increase"},
        {"paper_id": "p1", "effect_direction": "increase"},
        {"paper_id": "p1", "effect_direction": "increase"},
        {"paper_id": "p2", "effect_direction": "decrease"},
        {"paper_id": "p3", "effect_direction": "increase"},
    ]


def test_fixture_baseline_is_deterministic_and_never_calls_provider(repo_root: Path) -> None:
    class BombProvider:
        def generate(self, **_: object) -> None:
            raise AssertionError("fixture baseline must never call a provider")

    first = create_baseline(
        cohort_hash=COHORT_HASH,
        research_question="Does the intervention change the outcome?",
        primary_rows=_rows(),
        prompt_path=repo_root / "prompts" / "baseline_consensus.md",
        attempted_at=NOW,
        fixture_mode=True,
        provider=BombProvider(),
    )
    second = create_baseline(
        cohort_hash=COHORT_HASH,
        research_question="Does the intervention change the outcome?",
        primary_rows=_rows(),
        prompt_path=repo_root / "prompts" / "baseline_consensus.md",
        attempted_at=NOW,
        fixture_mode=True,
    )
    assert first == second
    assert first.source == "fixture_stub"
    assert first.status == "complete"
    assert first.majority.direction == "increase"
    # Paper p1 has many rows but still contributes only one paper of mass.
    assert first.majority.agreement == pytest.approx(2 / 3)
    assert first.llm is not None and "ungrounded" in first.llm.paragraph


def test_live_baseline_calls_exactly_once_and_records_raw_hash(repo_root: Path) -> None:
    provider = FixtureProvider(
        {
            (
                "baseline_consensus",
                COHORT_HASH,
            ): "  A cautious aggregate answer.\nIt remains uncertain.  "
        }
    )
    artifact = create_baseline(
        cohort_hash=COHORT_HASH,
        research_question="Does the intervention change the outcome?",
        primary_rows=_rows(),
        prompt_path=repo_root / "prompts" / "baseline_consensus.md",
        attempted_at=NOW,
        fixture_mode=False,
        provider=provider,
    )
    assert provider.calls == [("baseline_consensus", COHORT_HASH)]
    assert artifact.status == "complete"
    assert artifact.source == "live_llm"
    assert artifact.llm is not None
    assert artifact.llm.paragraph == "A cautious aggregate answer. It remains uncertain."
    assert artifact.llm.raw_response_sha256 != artifact.llm.prompt_sha256


def test_provider_failure_is_visible_and_never_retried(repo_root: Path) -> None:
    class FailingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_: object) -> None:
            self.calls += 1
            raise ProviderError("archived failure")

    provider = FailingProvider()
    artifact = create_baseline(
        cohort_hash=COHORT_HASH,
        research_question="Does the intervention change the outcome?",
        primary_rows=_rows(),
        prompt_path=repo_root / "prompts" / "baseline_consensus.md",
        attempted_at=NOW,
        fixture_mode=False,
        provider=provider,
    )
    assert provider.calls == 1
    assert artifact.status == "unavailable"
    assert artifact.failure_code == "BASELINE_PROVIDER_FAILED"
    assert artifact.llm is None


def test_baseline_write_is_atomic_and_immutable(tmp_path: Path, repo_root: Path) -> None:
    artifact = create_baseline(
        cohort_hash=COHORT_HASH,
        research_question="Does the intervention change the outcome?",
        primary_rows=_rows(),
        prompt_path=repo_root / "prompts" / "baseline_consensus.md",
        attempted_at=NOW,
        fixture_mode=True,
    )
    path = write_baseline_once(tmp_path / "baseline.json", artifact)
    assert json.loads(path.read_text(encoding="utf-8"))["cohort_hash"] == COHORT_HASH
    with pytest.raises(BaselineContractError, match="baseline_already_exists"):
        write_baseline_once(path, artifact)


def test_baseline_rejects_empty_primary_cohort(repo_root: Path) -> None:
    with pytest.raises(BaselineContractError, match="requires_primary_rows"):
        create_baseline(
            cohort_hash=COHORT_HASH,
            research_question="Does the intervention change the outcome?",
            primary_rows=[],
            prompt_path=repo_root / "prompts" / "baseline_consensus.md",
            attempted_at=NOW,
            fixture_mode=True,
        )


def test_primary_selection_reconciles_every_exact_grounded_request() -> None:
    findings = [
        {
            "finding_id": "f1",
            "paper_id": "p1",
            "effect_direction": "increase",
            "outcome_family": "primary",
            "grounding_status": "exact",
            "section_flagged": False,
        },
        {
            "finding_id": "f2",
            "paper_id": "p2",
            "effect_direction": "decrease",
            "outcome_family": "primary",
            "grounding_status": "exact",
            "section_flagged": False,
        },
        {
            "finding_id": "f3",
            "paper_id": "p3",
            "effect_direction": "increase",
            "outcome_family": "other",
            "grounding_status": "exact",
            "section_flagged": False,
        },
    ]
    verification = {
        "requested_finding_ids": ["f1", "f2", "f3"],
        "decisions": [
            {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
            {"finding_id": "f2", "model_status": "disagree", "adjudication": "accept"},
            {"finding_id": "f3", "model_status": "agree", "adjudication": "none"},
        ],
    }
    selected = select_primary_rows(findings, verification, primary_family="primary")
    assert [row["finding_id"] for row in selected] == ["f1", "f2"]
    verification["decisions"] = verification["decisions"][:-1]
    with pytest.raises(BaselineContractError, match="verification_decision_set_mismatch"):
        select_primary_rows(findings, verification, primary_family="primary")
