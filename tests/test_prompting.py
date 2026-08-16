from __future__ import annotations

import hashlib

import pytest

from literature_multiverse.prompting import PromptContractError, render_prompt_text


def test_prompt_renderer_requires_exact_token_set() -> None:
    template = "Prompt version: `v1`\nQuestion: [[QUESTION]]\n"
    rendered, version = render_prompt_text(template, {"QUESTION": "Does it change?"})
    assert version == "v1"
    assert "[[" not in rendered
    assert rendered.endswith("Does it change?\n")


@pytest.mark.parametrize(
    "replacements",
    [{}, {"QUESTION": "x", "UNUSED": "y"}],
)
def test_prompt_renderer_refuses_missing_or_extra_values(replacements) -> None:
    template = "Prompt version: `v1`\n[[QUESTION]]"
    with pytest.raises(PromptContractError):
        render_prompt_text(template, replacements)


def test_rendered_bytes_have_stable_hash() -> None:
    template = "Prompt version: `v1`\n[[VALUE]]"
    rendered, _ = render_prompt_text(template, {"VALUE": "café"})
    assert hashlib.sha256(rendered.encode()).hexdigest() == hashlib.sha256(
        "Prompt version: `v1`\ncafé".encode()
    ).hexdigest()
