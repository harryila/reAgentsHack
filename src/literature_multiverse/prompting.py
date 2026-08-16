"""Strict rendering and hashing for versioned repository prompt templates."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")
VERSION_RE = re.compile(r"^Prompt version: `([^`]+)`$", re.MULTILINE)


class PromptContractError(ValueError):
    """Raised when a template cannot be rendered exactly."""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    template_path: Path
    prompt_version: str
    text: str
    sha256: str


def render_prompt_text(template: str, replacements: Mapping[str, str]) -> tuple[str, str]:
    version_matches = VERSION_RE.findall(template)
    if len(version_matches) != 1:
        raise PromptContractError("prompt template must declare exactly one Prompt version")
    expected = set(TOKEN_RE.findall(template))
    provided = set(replacements)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise PromptContractError(
            f"prompt replacement mismatch: missing={missing}, extra={extra}"
        )
    rendered = template
    for key in sorted(expected):
        value = replacements[key]
        if not isinstance(value, str):
            raise PromptContractError(f"replacement {key} must already be a string")
        rendered = rendered.replace(f"[[{key}]]", value)
    unresolved = sorted(set(TOKEN_RE.findall(rendered)))
    if unresolved:
        raise PromptContractError(f"unresolved prompt tokens: {unresolved}")
    return rendered, version_matches[0]


def render_prompt_file(
    path: str | Path, replacements: Mapping[str, str]
) -> RenderedPrompt:
    source = Path(path)
    template = source.read_text(encoding="utf-8")
    rendered, version = render_prompt_text(template, replacements)
    return RenderedPrompt(
        template_path=source,
        prompt_version=version,
        text=rendered,
        sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )
