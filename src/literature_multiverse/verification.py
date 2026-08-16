"""Independent quote-to-direction verification with exact request reconciliation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from literature_multiverse.models import VerificationDecision, VerificationRecord
from literature_multiverse.prompting import RenderedPrompt
from literature_multiverse.providers import ProviderProtocol


class VerificationContractError(ValueError):
    """A verifier request or response failed the exact echo-back contract."""


def requested_grounded_rows(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select every exact-grounded accepted finding in stable ledger order."""

    requested: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in findings:
        if row.get("grounding_status") != "exact":
            continue
        finding_id = row.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise VerificationContractError("exact_grounded_row_missing_finding_id")
        if finding_id in seen:
            raise VerificationContractError(f"duplicate_finding_id:{finding_id}")
        seen.add(finding_id)
        requested.append(
            {
                "finding_id": finding_id,
                "proposed_direction": row.get("effect_direction"),
                "outcome_name": row.get("outcome_name"),
                "timepoint": row.get("timepoint_raw"),
                "comparator": row.get("comparator"),
                "evidence_quote": row.get("evidence_quote"),
                "evidence_lines": row.get("evidence_lines"),
            }
        )
    if not requested:
        raise VerificationContractError("verification_requires_exact_grounded_rows")
    return requested


def verification_output_schema(finding_ids: Sequence[str]) -> dict[str, Any]:
    """Build a closed JSON Schema that can echo each supplied ID exactly once."""

    identifiers = list(finding_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise VerificationContractError("verification_schema_ids_must_be_nonempty_unique")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                # The provider's structured-output validator rejects minItems values other
                # than 0/1 and maxItems entirely (live 400s, 2026-08-16), so the
                # one-decision-per-request count contract is enforced post-hoc by
                # reconcile_verification instead.
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "finding_id": {"type": "string", "enum": identifiers},
                        "model_status": {
                            "type": "string",
                            "enum": ["agree", "disagree", "unverifiable"],
                        },
                        "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
                    },
                    "required": ["finding_id", "model_status", "rationale"],
                },
            }
        },
        "required": ["decisions"],
    }


def reconcile_model_decisions(
    finding_ids: Sequence[str], raw_decisions: Sequence[Mapping[str, Any]]
) -> list[VerificationDecision]:
    """Require exactly one valid decision for every request, preserving request order."""

    requested = list(finding_ids)
    if len(requested) != len(set(requested)):
        raise VerificationContractError("duplicate_verification_request_id")
    by_id: dict[str, VerificationDecision] = {}
    for raw in raw_decisions:
        finding_id = raw.get("finding_id")
        if not isinstance(finding_id, str):
            raise VerificationContractError("verification_response_missing_finding_id")
        if finding_id in by_id:
            raise VerificationContractError(f"duplicate_verification_response:{finding_id}")
        if finding_id not in requested:
            raise VerificationContractError(f"unknown_verification_response:{finding_id}")
        try:
            decision = VerificationDecision(
                finding_id=finding_id,
                model_status=raw.get("model_status"),
                adjudication="none",
            )
        except ValueError as exc:
            raise VerificationContractError(
                f"invalid_verification_response:{finding_id}"
            ) from exc
        by_id[finding_id] = decision
    missing = set(requested) - set(by_id)
    if missing:
        raise VerificationContractError(
            "missing_verification_responses:" + ",".join(sorted(missing))
        )
    return [by_id[finding_id] for finding_id in requested]


def verify_with_provider(
    *,
    findings: Sequence[Mapping[str, Any]],
    rendered_prompt: RenderedPrompt,
    provider: ProviderProtocol,
    request_key: str,
) -> VerificationRecord:
    """Make one structured provider call and produce the strict immutable model record."""

    rows = requested_grounded_rows(findings)
    finding_ids = [str(row["finding_id"]) for row in rows]
    result = provider.generate(
        operation="quote_verification",
        request_key=request_key,
        prompt=rendered_prompt.text,
        system=(
            "Independently verify only quote-to-direction support. Return the supplied strict "
            "schema and never make eligibility or desirability judgments."
        ),
        output_schema=verification_output_schema(finding_ids),
    )
    payload = result.parsed_json
    if payload is None:
        try:
            payload = json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise VerificationContractError("verification_response_not_json") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"decisions"}:
        raise VerificationContractError("verification_response_root_invalid")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list) or any(
        not isinstance(decision, Mapping) for decision in raw_decisions
    ):
        raise VerificationContractError("verification_decisions_array_invalid")
    decisions = reconcile_model_decisions(finding_ids, raw_decisions)
    return VerificationRecord(
        verification_version="1",
        provider=result.provider,
        model=result.model,
        prompt_version=rendered_prompt.prompt_version,
        prompt_sha256=rendered_prompt.sha256,
        requested_finding_ids=finding_ids,
        decisions=decisions,
    )


def fixture_verification(
    *,
    findings: Sequence[Mapping[str, Any]],
    rendered_prompt: RenderedPrompt,
    status_overrides: Mapping[str, str] | None = None,
) -> VerificationRecord:
    """Create deterministic fixture decisions without invoking a provider."""

    rows = requested_grounded_rows(findings)
    overrides = dict(status_overrides or {})
    identifiers = [str(row["finding_id"]) for row in rows]
    unknown = set(overrides) - set(identifiers)
    if unknown:
        raise VerificationContractError(
            "fixture_verification_override_unknown:" + ",".join(sorted(unknown))
        )
    decisions = reconcile_model_decisions(
        identifiers,
        [
            {
                "finding_id": finding_id,
                "model_status": overrides.get(finding_id, "agree"),
            }
            for finding_id in identifiers
        ],
    )
    return VerificationRecord(
        verification_version="1",
        provider="fixture",
        model="fixture-verifier",
        prompt_version=rendered_prompt.prompt_version,
        prompt_sha256=rendered_prompt.sha256,
        requested_finding_ids=identifiers,
        decisions=decisions,
    )


__all__ = [
    "VerificationContractError",
    "fixture_verification",
    "reconcile_model_decisions",
    "requested_grounded_rows",
    "verification_output_schema",
    "verify_with_provider",
]
