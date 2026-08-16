"""Mechanical evidence grounding and verifier reconciliation.

The functions in this module deliberately accept plain mappings.  Stage code may pass
Pydantic ``model_dump()`` output, parquet records, or fixture dictionaries without the
scientific rules depending on a storage implementation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

GROUNDING_STATUSES = frozenset({"exact", "missing", "mismatch", "unverifiable"})
MODEL_VERIFICATION_STATUSES = frozenset({"agree", "disagree", "unverifiable"})
ADJUDICATION_STATUSES = frozenset({"none", "accept", "reject"})
PRIMARY_DIRECTIONS = frozenset({"increase", "no_effect", "decrease"})
BANNED_EVIDENCE_SECTIONS = frozenset(
    {"abstract", "discussion", "conclusion", "conclusions", "references", "unknown"}
)

_LINE_REF = re.compile(r"^\s*[Ll]?(?P<start>\d+)\s*(?:[-\u2013\u2014]\s*[Ll]?(?P<end>\d+))?\s*$")


class GroundingContractError(ValueError):
    """Raised when grounding or verification input violates a closed contract."""


def normalize_evidence_text(value: str) -> str:
    """Normalize Unicode and collapse all whitespace for exact substring grounding."""

    if not isinstance(value, str):
        raise TypeError("evidence text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


_ELLIPSIS_RE = re.compile(r"\.{3,}|…")


def quote_content_contained(quote: str, cited_text: str) -> bool:
    """Whitespace-insensitive, ellipsis-aware verbatim containment.

    Extractors abbreviate long sentences with ``...`` splices (observed live on the
    2026-08-15 corpus).  Every spliced segment must appear verbatim (ignoring
    whitespace), in order, within the cited text; a quote without ellipses degenerates
    to plain containment.
    """

    cited_compact = cited_text.replace(" ", "")
    position = 0
    for segment in _ELLIPSIS_RE.split(quote):
        compact = segment.replace(" ", "")
        if not compact:
            continue
        found = cited_compact.find(compact, position)
        if found < 0:
            return False
        position = found + len(compact)
    return True


_LINE_REF_PREFIX = re.compile(r"\s*[Ll]?\d+\s*(?:[-\u2013\u2014]\s*[Ll]?\d+\s*)?")


def strip_line_reference_suffix(reference: str | int) -> str:
    """Drop inline quoted text from a line reference.

    The live extractor sometimes cites ``"L18: However, the AS group ..."`` — the line id
    plus the quoted text itself (observed 2026-08-15).  Only a colon directly after a
    well-formed line reference is stripped; anything else passes through unchanged.
    """

    text = str(reference)
    head, separator, _tail = text.partition(":")
    if separator and _LINE_REF_PREFIX.fullmatch(head):
        return head.strip()
    return text


def expand_line_reference(reference: str | int) -> tuple[int, ...]:
    """Expand ``L12`` or ``L12-L14`` into an inclusive tuple of line numbers."""

    text = strip_line_reference_suffix(reference)
    match = _LINE_REF.fullmatch(text)
    if match is None:
        raise GroundingContractError(f"invalid evidence line reference: {text!r}")
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if start < 1 or end < start:
        raise GroundingContractError(f"invalid evidence line range: {text!r}")
    return tuple(range(start, end + 1))


def _line_number(value: Any, fallback: int | None = None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    if fallback is not None:
        return fallback
    raise GroundingContractError(f"line identifier has no integer component: {value!r}")


def _content_line_index(
    content_lines: Mapping[Any, Any] | Sequence[Any],
    line_sections: Mapping[Any, str | None] | None = None,
) -> dict[int, tuple[str, str | None]]:
    """Coerce common archived-line shapes into ``line -> (text, section)``."""

    result: dict[int, tuple[str, str | None]] = {}
    items: Iterable[tuple[Any, Any]]
    if isinstance(content_lines, Mapping):
        items = content_lines.items()
    elif isinstance(content_lines, Sequence) and not isinstance(content_lines, (str, bytes)):
        items = enumerate(content_lines, start=1)
    else:
        raise GroundingContractError("content_lines must be a mapping or sequence")

    for fallback_key, item in items:
        if isinstance(item, Mapping):
            identifier = item.get("line_number", item.get("line", item.get("id", fallback_key)))
            try:
                fallback_number = _line_number(fallback_key)
            except GroundingContractError:
                fallback_number = None
            number = _line_number(identifier, fallback_number)
            text = item.get("text", item.get("content"))
            section = item.get("section")
        else:
            number = _line_number(fallback_key)
            text = item
            section = None
        if not isinstance(text, str):
            raise GroundingContractError(f"line {number} does not contain string text")
        if number in result:
            raise GroundingContractError(f"duplicate content line number: {number}")
        if line_sections is not None:
            section = line_sections.get(number, line_sections.get(f"L{number}", section))
        if section is not None and not isinstance(section, str):
            raise GroundingContractError(f"line {number} section must be a string or null")
        result[number] = (text, section)
    return result


def relocate_quote(
    normalized_quote: str,
    line_index: Mapping[int, tuple[str, str | None]],
    *,
    max_window: int = 3,
) -> list[int] | None:
    """Locate a verbatim quote whose line citation is wrong.

    The live extractor occasionally copies a faithful quote but cites an unrelated line
    (methods text, a section heading — observed 2026-08-15).  When the quote is verbatim-
    contained (whitespace-insensitive, ellipsis-aware) in exactly ONE region of consecutive
    source lines, that region's minimal window is the mechanical relocation target.  Zero
    or multiple regions return ``None`` — ambiguity never relocates.
    """

    if not normalized_quote.strip():
        return None
    numbers = sorted(line_index)
    normalized_lines = {
        number: normalize_evidence_text(line_index[number][0]) for number in numbers
    }
    minimal_windows: list[list[int]] = []
    for start_position in range(len(numbers)):
        for width in range(1, max_window + 1):
            window = numbers[start_position : start_position + width]
            if len(window) < width or window[-1] - window[0] != width - 1:
                break
            text = " ".join(normalized_lines[number] for number in window)
            if quote_content_contained(normalized_quote, text):
                minimal_windows.append(list(window))
                break
    if not minimal_windows:
        return None
    regions: list[list[int]] = []
    for window in minimal_windows:
        if regions and window[0] <= regions[-1][-1]:
            regions[-1] = sorted(set(regions[-1]) | set(window))
        else:
            regions.append(list(window))
    if len(regions) != 1:
        return None
    return min(minimal_windows, key=lambda window: (len(window), window[0]))


def ground_evidence(
    evidence_quote: str | None,
    evidence_lines: Sequence[str | int] | None,
    content_lines: Mapping[Any, Any] | Sequence[Any] | None,
    *,
    line_sections: Mapping[Any, str | None] | None = None,
    source_accessible: bool = True,
) -> dict[str, Any]:
    """Return the exact §4.3 grounding result for one finding.

    Cited ranges are concatenated in citation order. Repeated or overlapping line
    references are included once, at their first occurrence.  A missing quote/reference
    is ``missing``; an inaccessible source or unresolved cited line is ``unverifiable``;
    only a failed normalized substring check is ``mismatch``.
    """

    quote_missing = not isinstance(evidence_quote, str) or not evidence_quote.strip()
    lines_missing = not evidence_lines
    if quote_missing or lines_missing:
        return {
            "grounding_status": "missing",
            "evidence_section": "unknown",
            "section_flagged": True,
            "resolved_line_numbers": [],
            "normalized_quote": "" if quote_missing else normalize_evidence_text(evidence_quote),
            "normalized_cited_text": "",
        }

    normalized_quote = normalize_evidence_text(evidence_quote)
    if not source_accessible or content_lines is None:
        return {
            "grounding_status": "unverifiable",
            "evidence_section": "unknown",
            "section_flagged": True,
            "resolved_line_numbers": [],
            "normalized_quote": normalized_quote,
            "normalized_cited_text": "",
        }

    index = _content_line_index(content_lines, line_sections)
    references: list[str | int] = list(evidence_lines)
    # The live extractor's dominant convention for a contiguous passage is a two-element
    # ["L58", "L60"] list rather than "L58-L60" (observed on the 2026-08-15 full-corpus
    # run, where it produced spurious mismatches by skipping the interior lines).  A
    # strictly increasing two-single-line citation within a bounded span is therefore
    # read as an inclusive range; anything else keeps the literal listed-lines meaning.
    if len(references) == 2:
        expanded_pair = [expand_line_reference(reference) for reference in references]
        if all(len(item) == 1 for item in expanded_pair):
            start, end = expanded_pair[0][0], expanded_pair[1][0]
            if start < end and end - start <= 30:
                references = [f"L{start}-L{end}"]
    line_numbers: list[int] = []
    seen: set[int] = set()
    for reference in references:
        for number in expand_line_reference(reference):
            if number not in seen:
                seen.add(number)
                line_numbers.append(number)

    if any(number not in index for number in line_numbers):
        return {
            "grounding_status": "unverifiable",
            "evidence_section": "unknown",
            "section_flagged": True,
            "resolved_line_numbers": line_numbers,
            "normalized_quote": normalized_quote,
            "normalized_cited_text": "",
        }

    def _resolve(numbers: list[int]) -> dict[str, Any]:
        cited_text = normalize_evidence_text(" ".join(index[number][0] for number in numbers))
        raw_sections = [index[number][1] for number in numbers]
        sections = [
            section.strip()
            for section in raw_sections
            if isinstance(section, str) and section.strip()
        ]
        normalized_sections = [section.casefold() for section in sections]
        if not sections:
            resolved_section = "unknown"
        elif len(dict.fromkeys(normalized_sections)) == 1:
            resolved_section = sections[0]
        else:
            resolved_section = "multiple"
        section_flagged = not sections or any(
            section in BANNED_EVIDENCE_SECTIONS for section in normalized_sections
        )
        # Containment ignores whitespace entirely: provider text spaces out special glyphs
        # ("V ̇ O 2max") that quotes render adjacently ("V ̇O2max"), and either spelling is
        # the same verbatim character content.  The stored normalized fields keep spacing so
        # audits can still see both renderings.
        status = "exact" if quote_content_contained(normalized_quote, cited_text) else "mismatch"
        return {
            "grounding_status": status,
            "evidence_section": resolved_section,
            "section_flagged": section_flagged,
            "resolved_line_numbers": numbers,
            "normalized_quote": normalized_quote,
            "normalized_cited_text": cited_text,
        }

    def _line_clean(number: int) -> bool:
        section = index[number][1]
        return (
            isinstance(section, str)
            and bool(section.strip())
            and section.strip().casefold() not in BANNED_EVIDENCE_SECTIONS
        )

    result = _resolve(line_numbers)
    if result["grounding_status"] == "mismatch":
        relocated = relocate_quote(normalized_quote, index)
        if relocated is not None:
            result = _resolve(relocated)
            result["relocated_from_line_numbers"] = line_numbers
    elif result["section_flagged"] and len(line_numbers) > 1:
        # Citation-subset refinement: extractors sometimes co-cite an abstract line
        # alongside the body line that already contains the full quote (observed
        # 2026-08-15).  If the quote is verbatim-contained in the clean cited lines
        # alone, the banned lines were unnecessary and the citation refines to the
        # clean subset.  A quote that NEEDS a banned line stays flagged.
        clean_subset = [number for number in line_numbers if _line_clean(number)]
        if clean_subset and len(clean_subset) < len(line_numbers):
            refined = _resolve(clean_subset)
            if refined["grounding_status"] == "exact" and not refined["section_flagged"]:
                refined["refined_from_line_numbers"] = line_numbers
                result = refined
    return result


def reconcile_verification(
    requested_finding_ids: Sequence[str], decisions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate the one-decision-per-request verifier contract.

    Missing, duplicate, and unknown IDs are hard failures. Human adjudication controls
    inclusion, but the model-agreement numerator remains immutable.
    """

    requested = list(requested_finding_ids)
    if len(requested) != len(set(requested)):
        raise GroundingContractError("requested finding IDs must be unique")
    requested_set = set(requested)
    by_id: dict[str, dict[str, str]] = {}
    for raw in decisions:
        finding_id = raw.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise GroundingContractError("verification decision requires finding_id")
        if finding_id in by_id:
            raise GroundingContractError(f"duplicate verification decision: {finding_id}")
        if finding_id not in requested_set:
            raise GroundingContractError(f"unknown verification decision: {finding_id}")
        model_status = raw.get("model_status")
        adjudication = raw.get("adjudication")
        if model_status not in MODEL_VERIFICATION_STATUSES:
            raise GroundingContractError(f"invalid model_status for {finding_id}: {model_status!r}")
        if adjudication not in ADJUDICATION_STATUSES:
            raise GroundingContractError(f"invalid adjudication for {finding_id}: {adjudication!r}")
        by_id[finding_id] = {
            "finding_id": finding_id,
            "model_status": model_status,
            "adjudication": adjudication,
        }
    missing = requested_set - by_id.keys()
    if missing:
        raise GroundingContractError(
            "missing verification decisions: " + ", ".join(sorted(missing))
        )

    ordered = [by_id[finding_id] for finding_id in requested]
    agree = sum(item["model_status"] == "agree" for item in ordered)
    included = sum(verification_allows_primary(item) for item in ordered)
    total = len(ordered)
    return {
        "decisions": ordered,
        "requested_exact_grounded": total,
        "model_agree": agree,
        "model_disagree": sum(item["model_status"] == "disagree" for item in ordered),
        "model_unverifiable": sum(item["model_status"] == "unverifiable" for item in ordered),
        "cross_model_agreement": agree / total if total else None,
        "primary_included": included,
        "primary_excluded": total - included,
    }


def verification_allows_primary(decision: Mapping[str, Any]) -> bool:
    """Return whether verification permits a row into the primary cohort."""

    return decision.get("model_status") == "agree" or decision.get("adjudication") == "accept"


def is_primary_headline_row(
    row: Mapping[str, Any],
    verification_decision: Mapping[str, Any] | None,
    *,
    primary_family: str | None = None,
) -> bool:
    """Apply the finding-level primary cohort filters without hiding rejected rows."""

    if row.get("grounding_status") != "exact" or bool(row.get("section_flagged")):
        return False
    if row.get("effect_direction") not in PRIMARY_DIRECTIONS:
        return False
    if primary_family is not None and row.get("outcome_family") != primary_family:
        return False
    if bool(row.get("quarantined", False)):
        return False
    return verification_decision is not None and verification_allows_primary(verification_decision)


__all__ = [
    "ADJUDICATION_STATUSES",
    "BANNED_EVIDENCE_SECTIONS",
    "GROUNDING_STATUSES",
    "MODEL_VERIFICATION_STATUSES",
    "PRIMARY_DIRECTIONS",
    "GroundingContractError",
    "expand_line_reference",
    "ground_evidence",
    "is_primary_headline_row",
    "normalize_evidence_text",
    "reconcile_verification",
    "relocate_quote",
    "strip_line_reference_suffix",
    "verification_allows_primary",
]
