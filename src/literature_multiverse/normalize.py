"""Deterministic row normalization, grounding, patching, and s4 reconciliation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from .extract import RawFinding
from .grounding import quote_content_contained, relocate_quote, strip_line_reference_suffix
from .models import (
    CanonicalDirection,
    DirectionNormalizationError,
    FindingRow,
    PaperRecord,
)
from .models import (
    make_finding_id as contract_make_finding_id,
)
from .models import (
    normalize_direction as contract_normalize_direction,
)

CANONICAL_DIRECTIONS = frozenset(direction.value for direction in CanonicalDirection)
PRIMARY_DIRECTIONS = frozenset({"increase", "no_effect", "decrease"})
BANNED_EVIDENCE_SECTIONS = frozenset(
    {"abstract", "discussion", "conclusion", "conclusions", "references", "unknown"}
)

FIXED_EXTRACTION_FIELDS = (
    "study_type",
    "species",
    "model",
    "population_state",
    "intervention",
    "intervention_class",
    "comparator",
    "dose_raw",
    "duration_raw",
    "timing_context",
    "outcome_name",
    "outcome_family",
    "timepoint_raw",
    "effect_direction",
    "effect_size_raw",
    "p_value",
    "significant",
    "sample_size",
    "evidence_quote",
    "evidence_lines",
    "confidence",
)


class NormalizationError(ValueError):
    """A deterministic normalizer or reconciliation contract failed."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class SourceLine:
    line_id: str
    text: str
    section: str | None = None


@dataclass(frozen=True, slots=True)
class GroundingResult:
    status: str
    evidence_section: str | None
    section_flagged: bool
    cited_text: str | None
    relocated_line_ids: tuple[str, ...] | None = None
    refined_line_ids: tuple[str, ...] | None = None
    dropped_line_tokens: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class QuarantinedFinding:
    paper_id: str
    doc_id: str
    map_result_id: str
    array_position: int
    reason_code: str
    detail: str
    raw_finding: Mapping[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: str) -> str:
    """Normalize Unicode and collapse whitespace for exact quote grounding."""

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def normalize_direction(value: Any) -> str:
    """Normalize only the design's closed direction vocabulary and legacy aliases."""

    try:
        return contract_normalize_direction(value).value
    except DirectionNormalizationError as exc:
        error = str(exc)
        code = {
            "direction_null": "FINDING_DIRECTION_JSON_NULL",
            "direction_not_string": "FINDING_DIRECTION_INVALID_TYPE",
        }.get(error, "FINDING_DIRECTION_UNKNOWN")
        raise NormalizationError(
            code, f"effect_direction {value!r} violates the closed direction contract"
        ) from exc


def _line_number(line_id: str) -> int | None:
    match = re.fullmatch(r"L(\d+)", line_id.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def expand_evidence_lines(references: Sequence[str]) -> list[str]:
    """Expand ``L10-L12`` while preserving cited line order and rejecting bad tokens.

    A strictly increasing two-single-line citation within a bounded span (``["L58",
    "L60"]``) is read as an inclusive range — the live extractor's dominant convention
    for contiguous passages, pinned on the 2026-08-15 full-corpus run.
    """

    references = [
        strip_line_reference_suffix(item) if isinstance(item, str) else item
        for item in references
    ]
    if len(references) == 2 and all(isinstance(item, str) for item in references):
        pair = [re.fullmatch(r"L(\d+)", item.strip().upper()) for item in references]
        if all(pair):
            start, end = int(pair[0].group(1)), int(pair[1].group(1))
            if start < end and end - start <= 30:
                references = [f"L{start}-L{end}"]
    expanded: list[str] = []
    for raw_reference in references:
        if not isinstance(raw_reference, str):
            raise NormalizationError(
                "EVIDENCE_LINE_INVALID", "every evidence line reference must be a string"
            )
        reference = raw_reference.strip().upper()
        single = re.fullmatch(r"L(\d+)", reference)
        if single:
            expanded.append(f"L{int(single.group(1))}")
            continue
        range_match = re.fullmatch(r"L(\d+)\s*-\s*L?(\d+)", reference)
        if not range_match:
            raise NormalizationError(
                "EVIDENCE_LINE_INVALID", f"invalid evidence line reference {raw_reference!r}"
            )
        start, end = (int(range_match.group(1)), int(range_match.group(2)))
        if end < start or end - start > 500:
            raise NormalizationError(
                "EVIDENCE_LINE_RANGE_INVALID", f"invalid evidence line range {raw_reference!r}"
            )
        expanded.extend(f"L{number}" for number in range(start, end + 1))
    # Repeated references should not duplicate source text.
    return list(dict.fromkeys(expanded))


def coerce_source_lines(
    source_lines: Mapping[str, Any] | Sequence[Mapping[str, Any] | SourceLine] | None,
) -> dict[str, SourceLine] | None:
    if source_lines is None:
        return None
    coerced: dict[str, SourceLine] = {}
    if isinstance(source_lines, Mapping):
        items: Iterable[tuple[str, Any]] = source_lines.items()
        for key, value in items:
            if isinstance(value, SourceLine):
                line = value
            elif isinstance(value, str):
                line = SourceLine(str(key).upper(), value, None)
            elif isinstance(value, Mapping):
                line = SourceLine(
                    str(value.get("line_id", key)).upper(),
                    str(value.get("text", "")),
                    None if value.get("section") is None else str(value["section"]),
                )
            else:
                raise NormalizationError(
                    "SOURCE_LINE_INVALID", f"unsupported source line value for {key}"
                )
            coerced[line.line_id.upper()] = line
        return coerced

    for value in source_lines:
        if isinstance(value, SourceLine):
            line = value
        elif isinstance(value, Mapping):
            raw_id = value.get("line_id", value.get("id", value.get("line")))
            if raw_id is None:
                raise NormalizationError("SOURCE_LINE_ID_MISSING", "source line lacks an id")
            line = SourceLine(
                str(raw_id).upper(),
                str(value.get("text", value.get("content", ""))),
                None if value.get("section") is None else str(value["section"]),
            )
        else:
            raise NormalizationError("SOURCE_LINE_INVALID", "unsupported source line record")
        coerced[line.line_id.upper()] = line
    return coerced


def ground_evidence(
    evidence_quote: Any,
    evidence_lines: Any,
    source_lines: Mapping[str, Any] | Sequence[Mapping[str, Any] | SourceLine] | None,
) -> GroundingResult:
    """Mechanically ground a quote against authoritative cited content lines."""

    if not isinstance(evidence_quote, str) or not evidence_quote.strip():
        return GroundingResult("missing", None, True, None)
    if not isinstance(evidence_lines, list) or not evidence_lines:
        return GroundingResult("missing", None, True, None)
    dropped_tokens: tuple[str, ...] = ()
    try:
        cited_ids = expand_evidence_lines(evidence_lines)
    except NormalizationError:
        # Extractors occasionally append a non-line locator ("Table 3") to otherwise
        # valid citations (observed 2026-08-15).  Grounding proceeds on the valid line
        # references alone; the dropped tokens are surfaced as a normalization warning.
        valid, dropped = [], []
        for item in evidence_lines:
            try:
                expand_evidence_lines([item])
            except NormalizationError:
                dropped.append(str(item))
            else:
                valid.append(item)
        if not valid or not dropped:
            return GroundingResult("unverifiable", None, True, None)
        try:
            cited_ids = expand_evidence_lines(valid)
        except NormalizationError:
            return GroundingResult("unverifiable", None, True, None)
        dropped_tokens = tuple(dropped)
    authoritative = coerce_source_lines(source_lines)
    if authoritative is None:
        return GroundingResult("unverifiable", None, True, None)
    if any(line_id not in authoritative for line_id in cited_ids):
        return GroundingResult("unverifiable", None, True, None)

    def _resolve(line_ids: list[str]) -> GroundingResult:
        cited = [authoritative[line_id] for line_id in line_ids]
        cited.sort(
            key=lambda line: (_line_number(line.line_id) is None, _line_number(line.line_id))
        )
        cited_text = " ".join(line.text for line in cited)
        sections = list(
            dict.fromkeys((line.section.strip() if line.section else "unknown") for line in cited)
        )
        evidence_section = sections[0] if len(sections) == 1 else " / ".join(sections)
        section_flagged = any(
            section.casefold() in BANNED_EVIDENCE_SECTIONS for section in sections
        )
        # Shared whitespace-insensitive, ellipsis-aware verbatim containment (the s3 and s4
        # grounding implementations should eventually be consolidated).
        status = (
            "exact"
            if quote_content_contained(normalize_text(evidence_quote), normalize_text(cited_text))
            else "mismatch"
        )
        return GroundingResult(status, evidence_section, section_flagged, cited_text)

    def _line_clean(line_id: str) -> bool:
        section = authoritative[line_id].section
        return (
            isinstance(section, str)
            and bool(section.strip())
            and section.strip().casefold() not in BANNED_EVIDENCE_SECTIONS
        )

    result = _resolve(cited_ids)
    if result.status == "mismatch":
        line_index: dict[int, tuple[str, str | None]] = {}
        for line in authoritative.values():
            number = _line_number(line.line_id)
            if number is not None:
                line_index[number] = (line.text, line.section)
        relocated = relocate_quote(normalize_text(evidence_quote), line_index)
        if relocated is not None:
            relocated_ids = [f"L{number}" for number in relocated]
            resolved = _resolve(relocated_ids)
            result = GroundingResult(
                resolved.status,
                resolved.evidence_section,
                resolved.section_flagged,
                resolved.cited_text,
                relocated_line_ids=tuple(relocated_ids),
            )
    elif result.section_flagged and len(cited_ids) > 1:
        # Citation-subset refinement — see grounding.ground_evidence for the rule.
        clean_subset = [line_id for line_id in cited_ids if _line_clean(line_id)]
        if clean_subset and len(clean_subset) < len(cited_ids):
            refined = _resolve(clean_subset)
            if refined.status == "exact" and not refined.section_flagged:
                result = GroundingResult(
                    refined.status,
                    refined.evidence_section,
                    refined.section_flagged,
                    refined.cited_text,
                    refined_line_ids=tuple(clean_subset),
                )
    if dropped_tokens:
        result = dataclass_replace(result, dropped_line_tokens=dropped_tokens)
    return result


def make_finding_id(
    *,
    paper_id: str,
    map_result_id: str,
    array_position: int,
    outcome_name: str,
    timepoint_raw: str | None,
    dose_raw: str | None,
    effect_direction: str,
) -> str:
    return contract_make_finding_id(
        paper_id=paper_id,
        map_result_id=map_result_id,
        array_position=array_position,
        outcome_name=outcome_name,
        timepoint_raw=timepoint_raw,
        dose_raw=dose_raw,
        effect_direction=effect_direction,
    )


def map_outcome_family(outcome_name: str, family_map: Mapping[str, str]) -> str | None:
    """Resolve a configured outcome-family mapping deterministically.

    Exact case-insensitive keys win.  A key written as ``re:<pattern>`` is an explicit regex;
    implicit fuzzy or substring mapping is intentionally forbidden.
    """

    normalized_name = normalize_text(outcome_name).casefold()
    exact = {
        normalize_text(str(raw_name)).casefold(): family
        for raw_name, family in family_map.items()
        if not str(raw_name).startswith("re:")
    }
    if normalized_name in exact:
        return str(exact[normalized_name])
    for raw_pattern, family in family_map.items():
        if not str(raw_pattern).startswith("re:"):
            continue
        try:
            if re.search(str(raw_pattern)[3:], outcome_name, flags=re.IGNORECASE):
                return str(family)
        except re.error as exc:
            raise NormalizationError(
                "OUTCOME_FAMILY_PATTERN_INVALID", f"invalid configured regex {raw_pattern!r}"
            ) from exc
    return None


def _moderator_name(spec: Any) -> str:
    if isinstance(spec, Mapping):
        return str(spec["name"])
    return str(spec.name)


def normalize_raw_finding(
    raw: RawFinding,
    *,
    prompt_version: str,
    schema_version: str,
    cfghash: str,
    moderator_specs: Sequence[Any] = (),
    outcome_family_map: Mapping[str, str] | None = None,
    outcome_endpoint_map: Mapping[str, str] | None = None,
    source_lines: Mapping[str, Any] | Sequence[Mapping[str, Any] | SourceLine] | None = None,
    require_all_keys: bool = False,
    exposure_terms: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, QuarantinedFinding | None]:
    """Normalize one raw finding or return a stable quarantine record."""

    payload = dict(raw.payload)
    warnings: list[str] = []
    if not require_all_keys and "population" in payload and "population_state" not in payload:
        payload["population_state"] = payload.pop("population")
        warnings.append("legacy_population_field")
    try:
        moderator_names = [_moderator_name(spec) for spec in moderator_specs]
        allowed_fields = set(FIXED_EXTRACTION_FIELDS) | {"moderators"}
        if not require_all_keys:
            allowed_fields |= {"population", *moderator_names}
        extra_fields = set(payload) - allowed_fields
        if extra_fields:
            raise NormalizationError(
                "FINDING_EXTRA_FIELDS", f"unrecognized fields {sorted(extra_fields)}"
            )
        if require_all_keys:
            missing_fields = set(FIXED_EXTRACTION_FIELDS) - set(payload)
            if missing_fields:
                raise NormalizationError(
                    "FINDING_REQUIRED_KEYS_MISSING",
                    f"missing nullable keys {sorted(missing_fields)}",
                )
            if "moderators" not in payload:
                raise NormalizationError(
                    "FINDING_MODERATORS_MISSING", "strict extraction requires moderators"
                )
        direction = normalize_direction(payload.get("effect_direction"))
        if exposure_terms:
            # A finding row must describe the locked exposure's arm.  Multi-arm factorial
            # trials produce rows for non-exposure arms (e.g. placebo+exercise labeled
            # "exercise_only", 2026-08-16 census audit); when the row's own intervention
            # text names an arm without any exposure term, it is not a finding of the
            # target relation.  Rows with no intervention text at all pass through.
            intervention_text = " ".join(
                str(payload.get(field) or "")
                for field in ("intervention", "intervention_class")
            ).casefold()
            if intervention_text.strip() and not any(
                term.casefold() in intervention_text for term in exposure_terms
            ):
                raise NormalizationError(
                    "FINDING_INTERVENTION_LACKS_EXPOSURE",
                    f"intervention {payload.get('intervention')!r} names no exposure term",
                )
        outcome_name = payload.get("outcome_name")
        if not isinstance(outcome_name, str) or not outcome_name.strip():
            raise NormalizationError(
                "FINDING_OUTCOME_NAME_MISSING", "outcome_name must be a non-empty string"
            )
        if outcome_endpoint_map:
            canonical_endpoint = map_outcome_family(outcome_name, outcome_endpoint_map)
            if canonical_endpoint is not None and canonical_endpoint != outcome_name:
                warnings.append(f"outcome_name_canonicalized:{outcome_name}")
                outcome_name = canonical_endpoint
        evidence_lines = payload.get("evidence_lines")
        if evidence_lines is not None and not isinstance(evidence_lines, list):
            raise NormalizationError(
                "FINDING_EVIDENCE_LINES_INVALID", "evidence_lines must be an array or null"
            )
        if evidence_lines is None:
            evidence_lines = []

        family_map = outcome_family_map or {}
        configured_family = map_outcome_family(outcome_name, family_map) if family_map else None
        extracted_family = payload.get("outcome_family")
        if configured_family is not None:
            if extracted_family not in (None, configured_family):
                warnings.append("outcome_family_overridden_by_config")
            outcome_family = configured_family
        else:
            outcome_family = extracted_family

        grounding = ground_evidence(payload.get("evidence_quote"), evidence_lines, source_lines)
        if grounding.relocated_line_ids is not None:
            original_refs = ",".join(str(item) for item in evidence_lines)
            relocated_refs = ",".join(grounding.relocated_line_ids)
            warnings.append(f"evidence_lines_relocated:{original_refs}->{relocated_refs}")
            evidence_lines = list(grounding.relocated_line_ids)
        elif grounding.refined_line_ids is not None:
            original_refs = ",".join(str(item) for item in evidence_lines)
            refined_refs = ",".join(grounding.refined_line_ids)
            warnings.append(f"evidence_lines_flag_refined:{original_refs}->{refined_refs}")
            evidence_lines = list(grounding.refined_line_ids)
        if grounding.dropped_line_tokens is not None:
            dropped_refs = ",".join(grounding.dropped_line_tokens)
            warnings.append(f"evidence_line_tokens_dropped:{dropped_refs}")
            evidence_lines = [
                item
                for item in evidence_lines
                if str(item) not in grounding.dropped_line_tokens
            ]
        moderators_payload = payload.get("moderators")
        if moderators_payload is None:
            moderators_payload = {}
        if not isinstance(moderators_payload, Mapping):
            raise NormalizationError("FINDING_MODERATORS_INVALID", "moderators must be an object")
        extra_moderators = set(moderators_payload) - set(moderator_names)
        if extra_moderators:
            raise NormalizationError(
                "FINDING_MODERATOR_EXTRA",
                f"unconfigured moderator fields {sorted(extra_moderators)}",
            )
        moderators: dict[str, Any] = {}
        if require_all_keys:
            missing_moderators = set(moderator_names) - set(moderators_payload)
            if missing_moderators:
                raise NormalizationError(
                    "FINDING_MODERATOR_KEYS_MISSING",
                    f"missing moderator keys {sorted(missing_moderators)}",
                )
        for name in moderator_names:
            if name in moderators_payload:
                moderators[name] = moderators_payload[name]
            elif name in payload:
                moderators[name] = payload[name]
            else:
                moderators[name] = None

        record = {field: payload.get(field) for field in FIXED_EXTRACTION_FIELDS}
        record.update(
            {
                "paper_id": raw.paper_id,
                "doc_id": raw.doc_id,
                "map_result_id": raw.map_result_id,
                "array_position": raw.array_position,
                "prompt_version": prompt_version,
                "schema_version": schema_version,
                "cfghash": cfghash,
                "effect_direction": direction,
                "outcome_name": outcome_name.strip(),
                "outcome_family": outcome_family,
                "evidence_lines": evidence_lines,
                "grounding_status": grounding.status,
                "evidence_section": grounding.evidence_section,
                "section_flagged": grounding.section_flagged,
                "normalization_warnings": sorted(set(warnings)),
                "moderators": moderators,
            }
        )
        record["finding_id"] = make_finding_id(
            paper_id=raw.paper_id,
            map_result_id=raw.map_result_id,
            array_position=raw.array_position,
            outcome_name=record["outcome_name"],
            timepoint_raw=record["timepoint_raw"],
            dose_raw=record["dose_raw"],
            effect_direction=direction,
        )
        return record, None
    except NormalizationError as exc:
        return None, QuarantinedFinding(
            paper_id=raw.paper_id,
            doc_id=raw.doc_id,
            map_result_id=raw.map_result_id,
            array_position=raw.array_position,
            reason_code=exc.code,
            detail=exc.detail,
            raw_finding=deepcopy(raw.payload),
        )


def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="python")
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"unsupported record type {type(record)!r}")


def apply_patches(
    rows: Sequence[Mapping[str, Any]], patches: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply human patches with exact selector/old-value/reason safeguards."""

    updated = [deepcopy(dict(row)) for row in rows]
    audit: list[dict[str, Any]] = []
    for patch_index, patch in enumerate(patches):
        selector = patch.get("selector")
        field = patch.get("field")
        reason = patch.get("reason")
        has_new_value = "value" in patch or "new_value" in patch
        if not isinstance(selector, Mapping) or not selector:
            raise NormalizationError(
                "PATCH_SELECTOR_INVALID", f"patch {patch_index} needs a non-empty selector"
            )
        if not isinstance(field, str) or not field:
            raise NormalizationError("PATCH_FIELD_INVALID", f"patch {patch_index} needs a field")
        if "expected_old_value" not in patch:
            raise NormalizationError(
                "PATCH_EXPECTED_OLD_MISSING", f"patch {patch_index} needs expected_old_value"
            )
        if not has_new_value:
            raise NormalizationError("PATCH_NEW_VALUE_MISSING", f"patch {patch_index} needs value")
        if not isinstance(reason, str) or not reason.strip():
            raise NormalizationError("PATCH_REASON_MISSING", f"patch {patch_index} needs a reason")

        immutable_fields = {
            "finding_id",
            "paper_id",
            "doc_id",
            "map_result_id",
            "array_position",
            "prompt_version",
            "schema_version",
            "cfghash",
        }
        if field in immutable_fields:
            raise NormalizationError(
                "PATCH_IMMUTABLE_FIELD_FORBIDDEN",
                f"patch {patch_index} cannot alter immutable field {field}",
            )
        matches = [
            index
            for index, row in enumerate(updated)
            if all(row.get(key) == expected for key, expected in selector.items())
        ]
        if len(matches) != 1:
            raise NormalizationError(
                "PATCH_SELECTOR_CARDINALITY",
                f"patch {patch_index} matched {len(matches)} rows; exactly one required",
            )
        row_index = matches[0]
        expected_old = patch["expected_old_value"]
        if updated[row_index].get(field) != expected_old:
            raise NormalizationError(
                "PATCH_OLD_VALUE_MISMATCH",
                f"patch {patch_index} expected {field}={expected_old!r}, found "
                f"{updated[row_index].get(field)!r}",
            )
        new_value = patch.get("value", patch.get("new_value"))
        old_finding_id = updated[row_index].get("finding_id")
        if field == "effect_direction":
            new_value = normalize_direction(new_value)
        updated[row_index][field] = new_value
        if field in {"outcome_name", "timepoint_raw", "dose_raw", "effect_direction"}:
            updated[row_index]["finding_id"] = make_finding_id(
                paper_id=str(updated[row_index]["paper_id"]),
                map_result_id=str(updated[row_index]["map_result_id"]),
                array_position=int(updated[row_index]["array_position"]),
                outcome_name=str(updated[row_index]["outcome_name"]),
                timepoint_raw=updated[row_index].get("timepoint_raw"),
                dose_raw=updated[row_index].get("dose_raw"),
                effect_direction=str(updated[row_index]["effect_direction"]),
            )
        audit.append(
            {
                "patch_index": patch_index,
                "selector": dict(selector),
                "field": field,
                "old_value": expected_old,
                "new_value": new_value,
                "reason": reason.strip(),
                "old_finding_id": old_finding_id,
                "new_finding_id": updated[row_index].get("finding_id"),
            }
        )
    return updated, audit


def reconcile_verification(
    findings: Sequence[Mapping[str, Any]], verification: Mapping[str, Any] | Any
) -> dict[str, Mapping[str, Any]]:
    """Require exactly one verifier decision per exact-grounded accepted finding."""

    requested = {
        str(row["finding_id"]) for row in findings if row.get("grounding_status") == "exact"
    }
    verification_payload = _as_dict(verification)
    declared_requests = verification_payload.get("requested_finding_ids")
    if declared_requests is not None and set(map(str, declared_requests)) != requested:
        raise NormalizationError(
            "VERIFICATION_REQUEST_SET_MISMATCH",
            "declared request IDs must equal every exact-grounded accepted finding",
        )
    raw_decisions = verification_payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise NormalizationError(
            "VERIFICATION_DECISIONS_INVALID", "verification decisions must be an array"
        )
    decisions: dict[str, Mapping[str, Any]] = {}
    for raw_decision in raw_decisions:
        if not isinstance(raw_decision, Mapping) and not hasattr(raw_decision, "model_dump"):
            raise NormalizationError(
                "VERIFICATION_DECISION_INVALID", "every decision must be an object"
            )
        decision = _as_dict(raw_decision)
        if "finding_id" not in decision:
            raise NormalizationError(
                "VERIFICATION_DECISION_INVALID", "every decision needs a finding_id"
            )
        finding_id = str(decision["finding_id"])
        if finding_id in decisions:
            raise NormalizationError(
                "VERIFICATION_DUPLICATE_ID", f"duplicate verifier decision for {finding_id}"
            )
        if finding_id not in requested:
            raise NormalizationError(
                "VERIFICATION_UNKNOWN_ID", f"unexpected verifier decision for {finding_id}"
            )
        if decision.get("model_status") not in {"agree", "disagree", "unverifiable"}:
            raise NormalizationError(
                "VERIFICATION_MODEL_STATUS_INVALID", f"invalid model status for {finding_id}"
            )
        if decision.get("adjudication") not in {"none", "accept", "reject"}:
            raise NormalizationError(
                "VERIFICATION_ADJUDICATION_INVALID", f"invalid adjudication for {finding_id}"
            )
        decisions[finding_id] = decision
    missing = requested - set(decisions)
    if missing:
        raise NormalizationError(
            "VERIFICATION_MISSING_IDS", f"missing verifier decisions for {sorted(missing)}"
        )
    return decisions


def primary_cohort(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    *,
    primary_family: str,
) -> list[dict[str, Any]]:
    decisions = reconcile_verification(findings, verification)
    eligible_papers = {
        str(paper["paper_id"])
        for paper in papers
        if paper.get("screen_status") == "included" and paper.get("eligible") is True
    }
    selected: list[dict[str, Any]] = []
    for finding in findings:
        if str(finding["paper_id"]) not in eligible_papers:
            continue
        if finding.get("outcome_family") != primary_family:
            continue
        if finding.get("grounding_status") != "exact" or bool(finding.get("section_flagged")):
            continue
        if finding.get("effect_direction") not in PRIMARY_DIRECTIONS:
            continue
        decision = decisions[str(finding["finding_id"])]
        if decision["model_status"] != "agree" and decision["adjudication"] != "accept":
            continue
        selected.append(dict(finding))
    return selected


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_quality_metrics(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    *,
    primary_family: str,
) -> dict[str, Any]:
    """Recompute the exact design §4.5 quality/exclusion denominators."""

    decisions = reconcile_verification(findings, verification)
    paper_by_id = {str(paper["paper_id"]): paper for paper in papers}
    accepted_primary = [
        row
        for row in findings
        if row.get("outcome_family") == primary_family
        and (paper := paper_by_id.get(str(row["paper_id"]))) is not None
        and paper.get("screen_status") == "included"
        and paper.get("eligible") is True
    ]
    grounded_n = sum(row.get("grounding_status") == "exact" for row in accepted_primary)
    mixed_n = sum(row.get("effect_direction") in {"mixed", "unclear"} for row in accepted_primary)
    section_n = sum(bool(row.get("section_flagged")) for row in accepted_primary)

    successful_included = [
        paper
        for paper in papers
        if paper.get("screen_status") == "included" and paper.get("map_status") == "success"
    ]
    accepted_all = sum(int(paper.get("accepted_finding_count", 0)) for paper in successful_included)
    quarantined_all = sum(
        int(paper.get("quarantined_finding_count", 0)) for paper in successful_included
    )
    quarantine_denominator = accepted_all + quarantined_all

    requested = [row for row in findings if row.get("grounding_status") == "exact"]
    agreement_n = sum(
        decisions[str(row["finding_id"])]["model_status"] == "agree" for row in requested
    )

    verification_eligible = [
        row
        for row in accepted_primary
        if row.get("grounding_status") == "exact"
        and not bool(row.get("section_flagged"))
        and row.get("effect_direction") in PRIMARY_DIRECTIONS
    ]
    verification_excluded_n = sum(
        decisions[str(row["finding_id"])]["model_status"] != "agree"
        and decisions[str(row["finding_id"])]["adjudication"] != "accept"
        for row in verification_eligible
    )

    return {
        "grounded": {
            "numerator": grounded_n,
            "denominator": len(accepted_primary),
            "fraction": _fraction(grounded_n, len(accepted_primary)),
        },
        "quarantine": {
            "numerator": quarantined_all,
            "denominator": quarantine_denominator,
            "fraction": _fraction(quarantined_all, quarantine_denominator),
        },
        "cross_model_agreement": {
            "numerator": agreement_n,
            "denominator": len(requested),
            "fraction": _fraction(agreement_n, len(requested)),
        },
        "mixed_or_unclear_exclusion": {
            "numerator": mixed_n,
            "denominator": len(accepted_primary),
            "fraction": _fraction(mixed_n, len(accepted_primary)),
        },
        "section_flagged_exclusion": {
            "numerator": section_n,
            "denominator": len(accepted_primary),
            "fraction": _fraction(section_n, len(accepted_primary)),
        },
        "verification_exclusion": {
            "numerator": verification_excluded_n,
            "denominator": len(verification_eligible),
            "fraction": _fraction(verification_excluded_n, len(verification_eligible)),
        },
    }


def normalized_frames(
    papers: Sequence[Any],
    findings: Sequence[Any],
    *,
    allow_mixed: bool = False,
    moderator_names: Sequence[str] = (),
    moderator_types: Mapping[str, str] | None = None,
):
    """Create strictly typed s4 frames and enforce referential/version integrity."""

    import pandas as pd

    paper_rows = [_as_dict(row) for row in papers]
    finding_rows = [_as_dict(row) for row in findings]
    paper_ids = [str(row["paper_id"]) for row in paper_rows]
    if len(paper_ids) != len(set(paper_ids)):
        raise NormalizationError("S4_DUPLICATE_PAPER_ID", "papers must have unique paper_id")
    finding_ids = [str(row["finding_id"]) for row in finding_rows]
    if len(finding_ids) != len(set(finding_ids)):
        raise NormalizationError("S4_DUPLICATE_FINDING_ID", "findings must have unique finding_id")
    orphan_ids = {str(row["paper_id"]) for row in finding_rows} - set(paper_ids)
    if orphan_ids:
        raise NormalizationError("S4_ORPHAN_FINDINGS", f"orphan paper IDs {sorted(orphan_ids)}")
    contract_tuples = {
        (str(row["prompt_version"]), str(row["schema_version"]), str(row["cfghash"]))
        for row in finding_rows
    }
    if len(contract_tuples) > 1 and not allow_mixed:
        raise NormalizationError(
            "S4_MIXED_EXTRACTION_TUPLES", f"found {len(contract_tuples)} extraction tuples"
        )

    flattened_findings: list[dict[str, Any]] = []
    for row in finding_rows:
        flat = dict(row)
        moderators = flat.pop("moderators", {}) or {}
        if not isinstance(moderators, Mapping):
            raise NormalizationError(
                "S4_MODERATORS_INVALID", f"moderators must be an object for {row['finding_id']}"
            )
        for name, value in moderators.items():
            flat[f"mod__{name}"] = value
        flattened_findings.append(flat)

    resolved_moderator_names = set(moderator_names)
    resolved_moderator_names.update(
        key.removeprefix("mod__")
        for row in flattened_findings
        for key in row
        if key.startswith("mod__")
    )
    paper_columns = list(PaperRecord.model_fields)
    finding_columns = [name for name in FindingRow.model_fields if name != "moderators"]
    finding_columns.extend(f"mod__{name}" for name in sorted(resolved_moderator_names))
    papers_frame = pd.DataFrame(paper_rows, columns=paper_columns)
    findings_frame = pd.DataFrame(flattened_findings, columns=finding_columns)
    for column in (
        "pub_year",
        "raw_finding_count",
        "accepted_finding_count",
        "quarantined_finding_count",
    ):
        if column in papers_frame:
            papers_frame[column] = pd.array(papers_frame[column], dtype="Int64")
    for column in ("sample_size", "array_position"):
        if column in findings_frame:
            findings_frame[column] = pd.array(findings_frame[column], dtype="Int64")
    for column in ("dedupe_preferred", "eligible"):
        if column in papers_frame:
            papers_frame[column] = pd.array(papers_frame[column], dtype="boolean")
    for column in ("section_flagged", "significant"):
        if column in findings_frame:
            findings_frame[column] = pd.array(findings_frame[column], dtype="boolean")
    for column in ("p_value", "confidence"):
        if column in findings_frame:
            findings_frame[column] = pd.array(findings_frame[column], dtype="Float64")
    if "created_at" in papers_frame:
        papers_frame["created_at"] = pd.to_datetime(papers_frame["created_at"], utc=True)
    for name, declared_type in (moderator_types or {}).items():
        column = f"mod__{name}"
        if column not in findings_frame:
            continue
        dtype = {
            "categorical": "string",
            "float": "Float64",
            "int": "Int64",
            "bool": "boolean",
        }[declared_type]
        findings_frame[column] = pd.array(findings_frame[column], dtype=dtype)
    return papers_frame, findings_frame


def write_processed_ledgers(
    papers: Sequence[Any],
    findings: Sequence[Any],
    *,
    papers_path: str | Path,
    findings_path: str | Path,
    allow_mixed: bool = False,
    moderator_names: Sequence[str] = (),
    moderator_types: Mapping[str, str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    papers_frame, findings_frame = normalized_frames(
        papers,
        findings,
        allow_mixed=allow_mixed,
        moderator_names=moderator_names,
        moderator_types=moderator_types,
    )
    papers_destination = Path(papers_path)
    findings_destination = Path(findings_path)
    existing = [path for path in (papers_destination, findings_destination) if path.exists()]
    if existing and not force:
        raise NormalizationError(
            "S4_OUTPUT_EXISTS", f"refusing to replace {[path.as_posix() for path in existing]}"
        )
    papers_destination.parent.mkdir(parents=True, exist_ok=True)
    findings_destination.parent.mkdir(parents=True, exist_ok=True)
    papers_temporary = papers_destination.with_name(f".{papers_destination.name}.tmp")
    findings_temporary = findings_destination.with_name(f".{findings_destination.name}.tmp")
    try:
        papers_frame.to_parquet(papers_temporary, index=False)
        findings_frame.to_parquet(findings_temporary, index=False)
        papers_temporary.replace(papers_destination)
        findings_temporary.replace(findings_destination)
    finally:
        papers_temporary.unlink(missing_ok=True)
        findings_temporary.unlink(missing_ok=True)
    return len(papers_frame), len(findings_frame)
