"""Immutable majority-plus-LLM comparison baseline construction.

The baseline is intentionally downstream of the scientific analysis.  Its language-model
paragraph is visibly ungrounded, is never fed back into statistics, and receives one archived
attempt for an exact cohort hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.disagreement import paper_balanced_finding_summary
from literature_multiverse.grounding import is_primary_headline_row
from literature_multiverse.models import ContractModel
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.providers import ProviderError, ProviderProtocol

SHA256_RE = r"^[0-9a-f]{64}$"


class BaselineContractError(ValueError):
    """A baseline input or immutable-output contract was violated."""


class BaselineMajority(ContractModel):
    direction: Literal["increase", "no_effect", "decrease", "mixed"]
    agreement: float = Field(ge=0.0, le=1.0)


class BaselineLlmOutput(ContractModel):
    provider: str
    model: str
    prompt_sha256: str
    raw_response_sha256: str
    paragraph: str

    @field_validator("prompt_sha256", "raw_response_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid_sha256")
        return value

    @field_validator("paragraph")
    @classmethod
    def validate_paragraph(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("baseline_paragraph_empty")
        return collapsed


class BaselineArtifact(ContractModel):
    cohort_hash: str
    status: Literal["complete", "unavailable"]
    source: Literal["live_llm", "fixture_stub"]
    majority: BaselineMajority
    llm: BaselineLlmOutput | None
    attempted_at: datetime
    failure_code: Literal[
        "BASELINE_PROVIDER_FAILED",
        "BASELINE_INVALID_RESPONSE",
    ] | None

    @field_validator("cohort_hash")
    @classmethod
    def validate_cohort_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid_cohort_hash")
        return value

    @field_validator("attempted_at")
    @classmethod
    def validate_attempted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("attempted_at_timezone_required")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> BaselineArtifact:
        if self.status == "complete":
            if self.llm is None or self.failure_code is not None:
                raise ValueError("complete_baseline_requires_llm_without_failure")
        elif self.llm is not None or self.failure_code is None:
            raise ValueError("unavailable_baseline_requires_failure_without_llm")
        if self.source == "fixture_stub" and self.status != "complete":
            raise ValueError("fixture_baseline_must_be_complete")
        return self


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def derive_majority(rows: Sequence[Mapping[str, Any]]) -> BaselineMajority:
    """Derive the paper-balanced primary majority, preserving an honest tied state."""

    summary = paper_balanced_finding_summary(rows)
    majority = summary["majority"]
    direction = majority["modal_direction"] if majority["unique"] else "mixed"
    return BaselineMajority(direction=direction, agreement=majority["agreement"])


def select_primary_rows(
    findings: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    *,
    primary_family: str,
) -> list[dict[str, Any]]:
    """Reconcile verification and select the exact primary-headline cohort."""

    requested = verification.get("requested_finding_ids")
    decisions = verification.get("decisions")
    if not isinstance(requested, list) or not isinstance(decisions, list):
        raise BaselineContractError("verification_shape_invalid")
    expected = {
        str(row["finding_id"])
        for row in findings
        if row.get("grounding_status") == "exact"
    }
    requested_ids = [str(value) for value in requested]
    if len(requested_ids) != len(set(requested_ids)) or set(requested_ids) != expected:
        raise BaselineContractError("verification_request_set_mismatch")
    by_id: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("finding_id"), str):
            raise BaselineContractError("verification_decision_invalid")
        finding_id = str(decision["finding_id"])
        if finding_id in by_id or finding_id not in expected:
            raise BaselineContractError("verification_decision_id_mismatch")
        if decision.get("model_status") not in {"agree", "disagree", "unverifiable"}:
            raise BaselineContractError("verification_model_status_invalid")
        if decision.get("adjudication") not in {"none", "accept", "reject"}:
            raise BaselineContractError("verification_adjudication_invalid")
        by_id[finding_id] = decision
    if set(by_id) != expected:
        raise BaselineContractError("verification_decision_set_mismatch")
    return [
        dict(row)
        for row in findings
        if is_primary_headline_row(
            row,
            by_id.get(str(row["finding_id"])),
            primary_family=primary_family,
        )
    ]


def _aggregate_summary(majority: BaselineMajority, row_count: int, paper_count: int) -> str:
    value = {
        "majority_direction": majority.direction,
        "paper_balanced_agreement": majority.agreement,
        "primary_grounded_findings": row_count,
        "primary_grounded_papers": paper_count,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fixture_paragraph(majority: BaselineMajority) -> str:
    return (
        "A generic, ungrounded expectation would describe the overall direction as "
        f"{majority.direction}, while noting that the aggregate agreement is "
        f"{majority.agreement:.3f} and does not establish a universal or causal conclusion."
    )


def create_baseline(
    *,
    cohort_hash: str,
    research_question: str,
    primary_rows: Sequence[Mapping[str, Any]],
    prompt_path: str | Path,
    attempted_at: datetime,
    fixture_mode: bool,
    provider: ProviderProtocol | None = None,
) -> BaselineArtifact:
    """Construct one typed baseline; provider failures become a visible unavailable state."""

    if not primary_rows:
        raise BaselineContractError("baseline_requires_primary_rows")
    if attempted_at.tzinfo is None or attempted_at.utcoffset() is None:
        raise BaselineContractError("baseline_attempted_at_timezone_required")
    majority = derive_majority(primary_rows)
    rendered = render_prompt_file(
        prompt_path,
        {
            "RESEARCH_QUESTION": research_question,
            "AGGREGATE_SUMMARY_JSON": _aggregate_summary(
                majority,
                len(primary_rows),
                len({str(row["paper_id"]) for row in primary_rows}),
            ),
        },
    )
    attempted_utc = attempted_at.astimezone(UTC)

    if fixture_mode:
        paragraph = _fixture_paragraph(majority)
        return BaselineArtifact(
            cohort_hash=cohort_hash,
            status="complete",
            source="fixture_stub",
            majority=majority,
            llm=BaselineLlmOutput(
                provider="fixture",
                model="fixture-stub",
                prompt_sha256=rendered.sha256,
                raw_response_sha256=hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
                paragraph=paragraph,
            ),
            attempted_at=attempted_utc,
            failure_code=None,
        )

    if provider is None:
        raise BaselineContractError("production_baseline_requires_provider")
    try:
        result = provider.generate(
            operation="baseline_consensus",
            request_key=cohort_hash,
            prompt=rendered.text,
            system=(
                "Return one plain-text paragraph. This is an explicitly ungrounded comparison, "
                "not a literature extraction or scientific result."
            ),
        )
    except ProviderError:
        return BaselineArtifact(
            cohort_hash=cohort_hash,
            status="unavailable",
            source="live_llm",
            majority=majority,
            llm=None,
            attempted_at=attempted_utc,
            failure_code="BASELINE_PROVIDER_FAILED",
        )
    paragraph = " ".join(result.text.split())
    if not paragraph:
        return BaselineArtifact(
            cohort_hash=cohort_hash,
            status="unavailable",
            source="live_llm",
            majority=majority,
            llm=None,
            attempted_at=attempted_utc,
            failure_code="BASELINE_INVALID_RESPONSE",
        )
    return BaselineArtifact(
        cohort_hash=cohort_hash,
        status="complete",
        source="live_llm",
        majority=majority,
        llm=BaselineLlmOutput(
            provider=result.provider,
            model=result.model,
            prompt_sha256=rendered.sha256,
            raw_response_sha256=hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
            paragraph=paragraph,
        ),
        attempted_at=attempted_utc,
        failure_code=None,
    )


def write_baseline_once(path: str | Path, artifact: BaselineArtifact) -> Path:
    """Atomically persist the immutable artifact and refuse every overwrite attempt."""

    target = Path(path)
    if target.exists():
        raise BaselineContractError(f"baseline_already_exists:{target}")
    _atomic_json(target, artifact.model_dump(mode="json"))
    return target


__all__ = [
    "BaselineArtifact",
    "BaselineContractError",
    "BaselineLlmOutput",
    "BaselineMajority",
    "create_baseline",
    "derive_majority",
    "select_primary_rows",
    "write_baseline_once",
]
