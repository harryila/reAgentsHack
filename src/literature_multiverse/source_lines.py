"""Fetch-side parsing of ``/papers/<id>/content.lines`` into grounded source lines.

The installed CLI exposes full paper text as ``L<number>: <text>`` lines.  Section names
are not delivered per line, so they are derived from the document's own headings with a
conservative heuristic that is deliberately biased toward the *banned* buckets: everything
before the body's first heading is treated as ``Abstract`` (front matter and structured
abstracts are not usable evidence), and unrecognized stretches inherit the most recent
heading rather than silently counting as body text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

_LINE_RE = re.compile(r"^L(?P<number>\d+):\s?(?P<text>.*)$")
_NUMBERED_HEADING_RE = re.compile(r"^(?P<major>\d+)(?:\.\d+)*\.?\s+(?P<title>[A-Z].{0,120})$")
# Figure/table captions serialize wherever the provider drops them (often after the
# Discussion); they present the paper's own data and must not inherit a banned label.
_CAPTION_RE = re.compile(r"^(?:Fig(?:ure)?\.?|Table)\s*\d", re.IGNORECASE)
# A numbered line whose "title" carries a publication-year signature is a reference
# entry, not structure; letting it inherit a body section via its leading number would
# relabel the bibliography as unbanned body text.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_BARE_HEADING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^abstract\b", "Abstract"),
    (r"^background\b", "Abstract"),
    (r"^objectives?\s*:?$", "Abstract"),
    (r"^introduction\b", "Introduction"),
    (r"^(materials?\s+and\s+)?methods?\b", "Methods"),
    (r"^(study\s+design|participants|subjects)\b", "Methods"),
    (r"^results?\b", "Results"),
    (r"^(results?\s+and\s+discussion)\b", "Results"),
    (r"^discussion\b", "Discussion"),
    (r"^conclusions?\b", "Conclusion"),
    (r"^(references|bibliography|literature\s+cited)\b", "References"),
    # All back matter (acknowledgments, contributions, supplements, CONSORT checklists)
    # is conservatively bucketed as References: none of it is usable result evidence,
    # and supplements frequently reprint section names that would otherwise unflag them.
    (r"^(acknowledg|author\s+contributions|funding|conflicts?\s+of\s+interest|"
     r"data\s+availability|supplementary|abbreviations)", "References"),
)
_CANONICAL_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("reference", "References"),
    ("bibliograph", "References"),
    ("method", "Methods"),
    ("result", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
    ("introduction", "Introduction"),
    ("abstract", "Abstract"),
    ("acknowledg", "References"),
)


class SourceLinesParseError(ValueError):
    """The provider content stream violated the pinned line format."""


def _canonical_heading(title: str) -> str | None:
    lowered = title.casefold()
    for keyword, canonical in _CANONICAL_KEYWORDS:
        if keyword in lowered:
            return canonical
    return None


def _bare_heading(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) > 80 or not stripped:
        return None
    lowered = stripped.casefold().rstrip(":").strip()
    for pattern, canonical in _BARE_HEADING_PATTERNS:
        if re.match(pattern, lowered):
            return canonical
    return None


def iter_numbered_lines(raw: str) -> Iterator[tuple[int, str]]:
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _LINE_RE.match(line)
        if match is None:
            # Provider preamble/pagination banners are not L-numbered; skip them.
            continue
        yield int(match.group("number")), match.group("text")


def parse_content_lines(raw: str) -> dict[str, dict[str, str]]:
    """Parse one paper's numbered content into ``{"L<n>": {"section", "text"}}``."""

    numbered = list(iter_numbered_lines(raw))
    if not numbered:
        raise SourceLinesParseError("no L-numbered content lines found")

    current_section = "Abstract"  # Front matter + abstract are never usable evidence.
    seen_body_heading = False
    major_map: dict[str, str] = {}
    result: dict[str, dict[str, str]] = {}
    for number, text in numbered:
        stripped = text.strip()
        if current_section == "References":
            # Terminal: nothing after the back matter starts is usable evidence, and
            # supplements (CONSORT checklists, reprinted protocols) must not re-open
            # body sections by echoing their headings.
            result[f"L{number}"] = {"section": "References", "text": text}
            continue
        if _CAPTION_RE.match(stripped):
            # Result-bearing caption: label it as its own section so it neither inherits
            # a banned bucket nor opens one.  The next ordinary line resumes inheritance.
            result[f"L{number}"] = {"section": "FigureTable", "text": text}
            continue
        numbered_heading = _NUMBERED_HEADING_RE.match(stripped)
        if numbered_heading and len(stripped) <= 120 and not _YEAR_RE.search(stripped):
            major = numbered_heading.group("major")
            canonical = _canonical_heading(numbered_heading.group("title"))
            if canonical is not None:
                major_map[major] = canonical
                current_section = canonical
                seen_body_heading = True
            elif major in major_map:
                current_section = major_map[major]
                seen_body_heading = True
            # An unknown numbered "heading" is NOT structure: numbered affiliations
            # ("1 Department of Kinesiology, ..."), numbered list items, and figure
            # enumerations all match the numbered shape.  Promoting their text to a
            # section label creates unflaggable garbage sections (7,853 distinct labels
            # on the 2026-08-15 corpus) that defeat the banned-section rule, so they
            # inherit the current section instead.
        else:
            bare = _bare_heading(stripped)
            if bare is not None and (seen_body_heading or bare != "Abstract" or number < 60):
                current_section = bare
                if bare not in {"Abstract"}:
                    seen_body_heading = True
        result[f"L{number}"] = {"section": current_section, "text": text}
    return result


def build_source_lines(contents_by_doc: Mapping[str, str]) -> dict[str, dict[str, dict[str, str]]]:
    return {doc_id: parse_content_lines(raw) for doc_id, raw in contents_by_doc.items()}
