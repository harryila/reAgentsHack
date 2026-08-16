"""Paperclip map-envelope parsing and lossless extraction-ledger assembly."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import QuestionConfig, config_sha256
from .models import FindingRow, PaperRecord

_MAP_ID_RE = re.compile(r"^Map results\s+\[(?P<map_id>[^\]]+)\]\s*$", re.MULTILINE)
_ENTRY_RE = re.compile(
    # Titles can wrap across lines in the results export (observed live on the 309-paper
    # corpus: entry 178's title spanned two lines), so the title group is DOTALL-lazy and
    # terminates at the first line ending in ``---``.
    r"^---\s+\[(?P<position>\d+)\]\s+\[(?P<status>[^\]]+)\]\s+"
    r"(?P<title>.*?)\s+---\s*$",
    re.MULTILINE | re.DOTALL,
)
_DOC_ID_RE = re.compile(r"^doc_id:\s*(?P<doc_id>\S+)\s*$", re.MULTILINE)
_ERROR_PATH_TITLE_RE = re.compile(r"/papers/(?P<doc_id>[^/\s]+)/?")


class MapParseError(ValueError):
    """A raw map artifact violated the pinned envelope format."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class LedgerReconciliationError(ValueError):
    """Extraction outputs cannot be reconciled to the screened include set."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class MapEnvelope:
    """One authoritative result envelope from Paperclip map stdout."""

    map_result_id: str
    position: int
    status: str
    title: str
    doc_id: str
    payload: Mapping[str, Any] | None
    provider_message: str | None

    @property
    def successful(self) -> bool:
        return self.status.casefold() == "success"

    @property
    def raw_finding_count(self) -> int:
        if not self.successful or self.payload is None:
            return 0
        findings = self.payload.get("findings")
        return len(findings) if isinstance(findings, list) else 0


@dataclass(frozen=True, slots=True)
class RawFinding:
    """A model finding paired with identity injected from its envelope."""

    paper_id: str
    doc_id: str
    map_result_id: str
    array_position: int
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionLedgers:
    """The complete, reconciled s3 scientific data boundary."""

    papers: tuple[PaperRecord, ...]
    findings: tuple[FindingRow, ...]
    quarantine: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, int]


def _decode_single_json_object(raw: str, *, position: int) -> Mapping[str, Any]:
    stripped = raw.strip()
    if not stripped:
        raise MapParseError("MAP_SUCCESS_PAYLOAD_MISSING", f"entry {position} has no payload")
    try:
        payload, consumed = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise MapParseError(
            "MAP_SUCCESS_PAYLOAD_INVALID_JSON",
            f"entry {position}: {exc.msg} at offset {exc.pos}",
        ) from exc
    if stripped[consumed:].strip():
        raise MapParseError(
            "MAP_SUCCESS_PAYLOAD_TRAILING_TEXT",
            f"entry {position} has text after its JSON object",
        )
    if not isinstance(payload, dict):
        raise MapParseError(
            "MAP_SUCCESS_PAYLOAD_NOT_OBJECT", f"entry {position} payload must be an object"
        )
    return payload


def parse_map_text(raw: str | bytes) -> list[MapEnvelope]:
    """Parse pinned human-readable ``paperclip map`` output.

    The envelope's ``doc_id`` is authoritative.  Any identity-looking fields in model JSON
    remain untrusted payload and are overwritten when findings are materialized.
    """

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MapParseError("MAP_OUTPUT_NOT_UTF8", str(exc)) from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("raw map output must be str or bytes")

    map_match = _MAP_ID_RE.search(text)
    if map_match is None:
        raise MapParseError("MAP_RESULT_ID_MISSING", "missing 'Map results [id]' header")
    map_result_id = map_match.group("map_id").strip()
    if not map_result_id:
        raise MapParseError("MAP_RESULT_ID_MISSING", "empty map result id")

    matches = list(_ENTRY_RE.finditer(text))
    if not matches:
        raise MapParseError("MAP_ENVELOPES_MISSING", "no result envelopes found")

    envelopes: list[MapEnvelope] = []
    seen_positions: set[int] = set()
    seen_doc_ids: set[str] = set()
    for index, match in enumerate(matches):
        position = int(match.group("position"))
        if position in seen_positions:
            raise MapParseError("MAP_DUPLICATE_POSITION", f"duplicate entry position {position}")
        seen_positions.add(position)

        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        status_field = match.group("status").strip().casefold()
        title_field = match.group("title").strip()
        doc_match = _DOC_ID_RE.search(body)
        if doc_match is None:
            # Worker-error envelopes may carry only a paper path in the title slot with an
            # empty body ("--- [151] [error] /papers/PMC9692807/ ---", live 2026-08-15).
            path_match = _ERROR_PATH_TITLE_RE.fullmatch(title_field)
            if status_field != "success" and path_match:
                doc_id = path_match.group("doc_id")
            else:
                raise MapParseError("MAP_DOC_ID_MISSING", f"entry {position} has no doc_id")
        else:
            doc_id = doc_match.group("doc_id").strip()
        if doc_id in seen_doc_ids:
            raise MapParseError("MAP_DUPLICATE_DOC_ID", f"duplicate envelope doc_id {doc_id}")
        seen_doc_ids.add(doc_id)

        payload_text = (body[doc_match.end() :] if doc_match else body).strip()
        status = status_field
        if status == "success":
            payload = _decode_single_json_object(payload_text, position=position)
            provider_message = None
        else:
            payload = None
            provider_message = payload_text or None

        envelopes.append(
            MapEnvelope(
                map_result_id=map_result_id,
                position=position,
                status=status,
                title=title_field,
                doc_id=doc_id,
                payload=payload,
                provider_message=provider_message,
            )
        )
    return envelopes


def parse_map_file(path: str | Path) -> list[MapEnvelope]:
    """Parse a locally archived map artifact without invoking a provider."""

    return parse_map_text(Path(path).read_bytes())


def validate_envelope_payload(envelope: MapEnvelope) -> None:
    """Validate top-level map semantics before row-level normalization."""

    if not envelope.successful:
        return
    assert envelope.payload is not None
    payload = envelope.payload
    if set(payload) - {"eligible", "exclusion_reason", "findings"}:
        extras = sorted(set(payload) - {"eligible", "exclusion_reason", "findings"})
        raise MapParseError(
            "MAP_PAYLOAD_EXTRA_FIELDS",
            f"entry {envelope.position} has forbidden fields {extras}",
        )
    if not isinstance(payload.get("eligible"), bool):
        raise MapParseError(
            "MAP_ELIGIBLE_NOT_BOOLEAN", f"entry {envelope.position} eligible must be boolean"
        )
    exclusion_reason = payload.get("exclusion_reason")
    if exclusion_reason is not None and not isinstance(exclusion_reason, str):
        raise MapParseError(
            "MAP_EXCLUSION_REASON_INVALID",
            f"entry {envelope.position} exclusion_reason must be string or null",
        )
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise MapParseError(
            "MAP_FINDINGS_NOT_ARRAY", f"entry {envelope.position} findings must be an array"
        )
    if payload["eligible"] is False and findings:
        raise MapParseError(
            "MAP_INELIGIBLE_HAS_FINDINGS",
            f"entry {envelope.position} is ineligible but returned findings",
        )
    if payload["eligible"] is False and not exclusion_reason:
        raise MapParseError(
            "MAP_INELIGIBLE_REASON_MISSING",
            f"entry {envelope.position} is ineligible without an exclusion reason",
        )
    for array_position, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise MapParseError(
                "MAP_FINDING_NOT_OBJECT",
                f"entry {envelope.position} finding {array_position} must be an object",
            )


def iter_raw_findings(envelope: MapEnvelope, *, paper_id: str) -> Iterator[RawFinding]:
    """Yield findings with locally authoritative identity attached."""

    validate_envelope_payload(envelope)
    if not envelope.successful:
        return
    assert envelope.payload is not None
    for array_position, source_payload in enumerate(envelope.payload["findings"]):
        # Identity-looking model fields never cross the trust boundary.
        payload = {
            key: value
            for key, value in source_payload.items()
            if key not in {"paper_id", "doc_id", "map_result_id", "array_position", "finding_id"}
        }
        yield RawFinding(
            paper_id=paper_id,
            doc_id=envelope.doc_id,
            map_result_id=envelope.map_result_id,
            array_position=array_position,
            payload=payload,
        )


def reconcile_envelopes(
    envelope_batches: Iterable[Sequence[MapEnvelope]],
    *,
    expected_doc_ids: Iterable[str],
) -> list[MapEnvelope]:
    """Reconcile initial/resumed map artifacts to exactly one terminal envelope per input.

    A later success may replace an earlier failed/nonterminal result for the same document.
    Conflicting successes are rejected, which prevents duplicate terminal papers/findings.
    """

    requested = tuple(expected_doc_ids)
    expected = tuple(dict.fromkeys(requested))
    if len(expected) != len(requested):
        raise LedgerReconciliationError(
            "MAP_EXPECTED_DOC_IDS_DUPLICATE", "screened include doc IDs must be unique"
        )
    expected_set = set(expected)
    terminal: dict[str, MapEnvelope] = {}
    for batch in envelope_batches:
        for envelope in batch:
            if envelope.doc_id not in expected_set:
                raise LedgerReconciliationError(
                    "MAP_UNKNOWN_DOC_ID", f"map returned unexpected doc_id {envelope.doc_id}"
                )
            existing = terminal.get(envelope.doc_id)
            if existing is None:
                terminal[envelope.doc_id] = envelope
                continue
            if existing.successful and envelope.successful:
                if existing.payload != envelope.payload:
                    raise LedgerReconciliationError(
                        "MAP_CONFLICTING_SUCCESS",
                        f"multiple successful payloads for {envelope.doc_id}",
                    )
                continue
            if envelope.successful or not existing.successful:
                terminal[envelope.doc_id] = envelope

    missing = [doc_id for doc_id in expected if doc_id not in terminal]
    if missing:
        raise LedgerReconciliationError(
            "MAP_MISSING_TERMINAL_RECORDS", f"no terminal result for doc IDs {missing}"
        )
    return [terminal[doc_id] for doc_id in expected]


def reconcile_ledger_counts(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    quarantine: Sequence[Mapping[str, Any]],
    *,
    s2_paper_ids: Iterable[str],
) -> dict[str, int]:
    """Enforce the lossless s2→s3 ledger count identities."""

    expected_papers = set(s2_paper_ids)
    actual_ids = [str(paper["paper_id"]) for paper in papers]
    if len(actual_ids) != len(set(actual_ids)):
        raise LedgerReconciliationError("PAPER_LEDGER_DUPLICATE_ID", "paper_id must be unique")
    if set(actual_ids) != expected_papers:
        raise LedgerReconciliationError(
            "PAPER_LEDGER_SET_MISMATCH",
            f"paper ledger differs from s2: missing={sorted(expected_papers - set(actual_ids))}, "
            f"extra={sorted(set(actual_ids) - expected_papers)}",
        )

    finding_papers = {str(row["paper_id"]) for row in findings}
    if not finding_papers <= expected_papers:
        raise LedgerReconciliationError(
            "ORPHAN_FINDING",
            f"unknown finding paper IDs {sorted(finding_papers - expected_papers)}",
        )

    accepted_by_paper: dict[str, int] = {}
    for row in findings:
        paper_id = str(row["paper_id"])
        accepted_by_paper[paper_id] = accepted_by_paper.get(paper_id, 0) + 1
    quarantine_by_paper: dict[str, int] = {}
    for row in quarantine:
        paper_id = str(row["paper_id"])
        if paper_id not in expected_papers:
            raise LedgerReconciliationError(
                "ORPHAN_QUARANTINE", f"unknown quarantine paper ID {paper_id}"
            )
        quarantine_by_paper[paper_id] = quarantine_by_paper.get(paper_id, 0) + 1

    for paper in papers:
        paper_id = str(paper["paper_id"])
        raw_count = int(paper["raw_finding_count"])
        accepted_count = int(paper["accepted_finding_count"])
        quarantined_count = int(paper["quarantined_finding_count"])
        if accepted_count != accepted_by_paper.get(paper_id, 0):
            raise LedgerReconciliationError(
                "PAPER_ACCEPTED_COUNT_MISMATCH", f"accepted count mismatch for {paper_id}"
            )
        if quarantined_count != quarantine_by_paper.get(paper_id, 0):
            raise LedgerReconciliationError(
                "PAPER_QUARANTINE_COUNT_MISMATCH", f"quarantine count mismatch for {paper_id}"
            )
        if paper.get("map_status") == "success" and raw_count != accepted_count + quarantined_count:
            raise LedgerReconciliationError(
                "PAPER_RAW_COUNT_MISMATCH", f"raw count mismatch for {paper_id}"
            )

    return {
        "papers": len(papers),
        "raw_findings": sum(int(paper["raw_finding_count"]) for paper in papers),
        "accepted_findings": len(findings),
        "quarantined_findings": len(quarantine),
    }


def _paper_dict(record: PaperRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, PaperRecord):
        return record.model_dump(mode="python")
    return dict(record)


def _stable_map_failure_code(status: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", status.upper()).strip("_") or "UNKNOWN"
    return f"PAPERCLIP_MAP_{normalized}"


def extraction_prompt_replacements(config: QuestionConfig) -> dict[str, str]:
    """Return the exact replacement set for ``prompts/extraction.md``.

    A triage config deliberately has no locked outcome family or endpoint list; rendering
    Python's ``None``/``[]`` literally confuses the extractor into treating every outcome
    as out of family.  Triage renders an explicit exploratory clause instead.
    """

    if config.outcomes.primary_family is None:
        outcome_family = (
            "exploratory (triage): no outcome family is locked yet — any outcome the "
            "paper's own results directly attribute to the target relation qualifies"
        )
    else:
        outcome_family = config.outcomes.primary_family
    if config.outcomes.included_primary_endpoints:
        endpoints = json.dumps(config.outcomes.included_primary_endpoints)
    else:
        endpoints = (
            '"exploratory (triage): report each qualifying outcome under the paper\'s '
            'own outcome name"'
        )
    return {
        "RESEARCH_QUESTION": config.research_question,
        "EXPOSURE": config.target_relation.exposure,
        "COMPARATOR": config.target_relation.comparator,
        "PRIMARY_OUTCOME_FAMILY": outcome_family,
        "PRIMARY_ENDPOINTS_JSON": endpoints,
        "INCREASE_DEFINITION": config.target_relation.increase_definition,
        "DECREASE_DEFINITION": config.target_relation.decrease_definition,
        "NO_EFFECT_DEFINITION": config.target_relation.no_effect_definition,
        "ELIGIBILITY_RULES_JSON": json.dumps(
            config.eligibility.model_dump(mode="json"), sort_keys=True
        ),
        "MODERATOR_RULES_JSON": json.dumps(
            [moderator.model_dump(mode="json") for moderator in config.moderators],
            sort_keys=True,
        ),
    }


def assemble_extraction_ledgers(
    screened_papers: Sequence[PaperRecord | Mapping[str, Any]],
    envelopes: Sequence[MapEnvelope],
    *,
    config: QuestionConfig,
    prompt_version: str,
    cfghash: str,
    raw_artifact_path: str,
    raw_artifact_path_by_doc: Mapping[str, str] | None = None,
    source_lines_by_doc: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    allow_triage: bool = False,
) -> ExtractionLedgers:
    """Build exactly one terminal paper row per s2 paper and quarantine bad findings.

    This is the first boundary at which included papers become strict ``PaperRecord``
    instances.  s2's included/not-mapped rows are intentionally only shaped dictionaries;
    the terminal contract forbids that intermediate state.
    """

    if config.status != "locked" and not (allow_triage and config.status == "triage"):
        raise LedgerReconciliationError("S3_LOCKED_CONFIG_REQUIRED", config.question_id)
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise LedgerReconciliationError("S3_TIMESTAMP_NAIVE", "created_at must include timezone")
    config_hash = config_sha256(config)
    source_lookup = source_lines_by_doc or {}
    artifact_lookup = raw_artifact_path_by_doc or {}
    screened = [_paper_dict(record) for record in screened_papers]
    paper_ids = [str(record.get("paper_id", "")) for record in screened]
    doc_ids = [str(record.get("doc_id", "")) for record in screened]
    if any(not value for value in paper_ids) or len(paper_ids) != len(set(paper_ids)):
        raise LedgerReconciliationError(
            "S2_PAPER_IDS_INVALID", "paper IDs must be non-empty/unique"
        )
    if any(not value for value in doc_ids) or len(doc_ids) != len(set(doc_ids)):
        raise LedgerReconciliationError("S2_DOC_IDS_INVALID", "doc IDs must be non-empty/unique")
    for record in screened:
        if record.get("config_sha256") != config_hash:
            raise LedgerReconciliationError("S2_CONFIG_HASH_MISMATCH", str(record.get("paper_id")))

    included = [record for record in screened if record.get("screen_status") == "included"]
    excluded = [record for record in screened if record.get("screen_status") == "excluded"]
    if len(included) + len(excluded) != len(screened):
        raise LedgerReconciliationError(
            "S2_SCREEN_STATUS_INVALID", "screen_status must be included or excluded"
        )
    # An archived map may cover papers the current screen no longer includes — either a
    # deterministic audit exclusion or provider retrieval drift between reruns of the
    # recency-weighted searches (observed 2026-08-16: 23 of 660 mapped docs absent from
    # a same-night rerun, all ineligible).  Their envelopes are simply unused; the
    # reconciliation still rejects duplicate or conflicting envelopes for real papers.
    included_doc_ids = {str(record["doc_id"]) for record in included}
    unused_envelope_docs = sorted(
        {envelope.doc_id for envelope in envelopes if envelope.doc_id not in included_doc_ids}
    )
    terminal_envelopes = reconcile_envelopes(
        [[envelope for envelope in envelopes if envelope.doc_id in included_doc_ids]],
        expected_doc_ids=sorted(included_doc_ids),
    )
    envelope_by_doc = {envelope.doc_id: envelope for envelope in terminal_envelopes}

    # Import here to keep the raw parser usable as a very small standalone boundary.
    from .normalize import normalize_raw_finding
    from .schemas import validate_finding_row

    papers: list[PaperRecord] = []
    findings: list[FindingRow] = []
    quarantine: list[dict[str, Any]] = []
    for source_paper in screened:
        paper = dict(source_paper)
        paper["config_sha256"] = config_hash
        paper["schema_version"] = config.schema_version
        paper["created_at"] = timestamp
        if paper["screen_status"] == "excluded":
            paper.update(
                {
                    "map_status": "not_mapped",
                    "eligible": None,
                    "exclusion_reason": None,
                    "map_result_id": None,
                    "raw_artifact_path": None,
                    "raw_finding_count": 0,
                    "accepted_finding_count": 0,
                    "quarantined_finding_count": 0,
                    "failure_code": None,
                    "prompt_version": None,
                    "cfghash": None,
                }
            )
            papers.append(PaperRecord.model_validate(paper))
            continue

        envelope = envelope_by_doc[str(paper["doc_id"])]
        if not envelope.successful:
            paper.update(
                {
                    "map_status": "failed",
                    "eligible": None,
                    "exclusion_reason": None,
                    "map_result_id": envelope.map_result_id,
                    "raw_artifact_path": artifact_lookup.get(envelope.doc_id, raw_artifact_path),
                    "raw_finding_count": 0,
                    "accepted_finding_count": 0,
                    "quarantined_finding_count": 0,
                    "failure_code": _stable_map_failure_code(envelope.status),
                    "prompt_version": prompt_version,
                    "cfghash": cfghash,
                }
            )
            papers.append(PaperRecord.model_validate(paper))
            continue

        validate_envelope_payload(envelope)
        assert envelope.payload is not None
        accepted_for_paper: list[FindingRow] = []
        quarantined_for_paper: list[dict[str, Any]] = []
        for raw_finding in iter_raw_findings(envelope, paper_id=str(paper["paper_id"])):
            normalized, rejected = normalize_raw_finding(
                raw_finding,
                prompt_version=prompt_version,
                schema_version=config.schema_version,
                cfghash=cfghash,
                moderator_specs=config.moderators,
                outcome_family_map=config.outcomes.family_map,
                outcome_endpoint_map=config.outcomes.endpoint_map,
                source_lines=source_lookup.get(envelope.doc_id),
                require_all_keys=True,
                exposure_terms=config.target_relation.exposure_terms,
            )
            if rejected is not None:
                quarantined_for_paper.append(rejected.model_dump())
                continue
            assert normalized is not None
            try:
                accepted_for_paper.append(validate_finding_row(normalized, config))
            except ValidationError as exc:
                quarantined_for_paper.append(
                    {
                        "paper_id": raw_finding.paper_id,
                        "doc_id": raw_finding.doc_id,
                        "map_result_id": raw_finding.map_result_id,
                        "array_position": raw_finding.array_position,
                        "reason_code": "FINDING_SCHEMA_INVALID",
                        "detail": exc.errors(include_url=False, include_input=False),
                        "raw_finding": dict(raw_finding.payload),
                    }
                )

        raw_count = envelope.raw_finding_count
        paper.update(
            {
                "map_status": "success",
                "eligible": bool(envelope.payload["eligible"]),
                "exclusion_reason": envelope.payload["exclusion_reason"],
                "map_result_id": envelope.map_result_id,
                "raw_artifact_path": artifact_lookup.get(envelope.doc_id, raw_artifact_path),
                "raw_finding_count": raw_count,
                "accepted_finding_count": len(accepted_for_paper),
                "quarantined_finding_count": len(quarantined_for_paper),
                "failure_code": None,
                "prompt_version": prompt_version,
                "cfghash": cfghash,
            }
        )
        papers.append(PaperRecord.model_validate(paper))
        findings.extend(accepted_for_paper)
        quarantine.extend(quarantined_for_paper)

    paper_payloads = [record.model_dump(mode="json") for record in papers]
    finding_payloads = [record.model_dump(mode="json") for record in findings]
    counts = reconcile_ledger_counts(
        paper_payloads,
        finding_payloads,
        quarantine,
        s2_paper_ids=paper_ids,
    )
    counts.update(
        {
            "s2_included": len(included),
            "s2_excluded": len(excluded),
            "map_success": sum(record.map_status.value == "success" for record in papers),
            "map_failure": sum(record.map_status.value == "failed" for record in papers),
            "not_mapped": sum(record.map_status.value == "not_mapped" for record in papers),
            "unused_map_envelopes": len(unused_envelope_docs),
        }
    )
    if counts["map_success"] + counts["map_failure"] + counts["not_mapped"] != len(papers):
        raise LedgerReconciliationError(
            "S3_TERMINAL_STATUS_COUNT_MISMATCH", "terminal paper statuses do not reconcile"
        )
    return ExtractionLedgers(
        papers=tuple(papers),
        findings=tuple(findings),
        quarantine=tuple(quarantine),
        counts=counts,
    )
