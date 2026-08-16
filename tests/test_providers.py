from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from literature_multiverse.providers import (
    AnthropicProvider,
    FixtureProvider,
    LiveProviderDisabled,
    ProviderAttemptExists,
    ProviderBudgetExceeded,
    ProviderError,
    ProviderUsage,
    estimate_cost_usd,
    load_live_environment,
)


class FakeMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def fake_client(text: str = "ok") -> SimpleNamespace:
    response = SimpleNamespace(
        id="msg_test",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    return SimpleNamespace(messages=FakeMessages(response))


def test_fixture_provider_is_deterministic_and_in_memory() -> None:
    provider = FixtureProvider({("baseline", "cohort-a"): "fixture paragraph"})
    result = provider.generate(
        operation="baseline", request_key="cohort-a", prompt="question"
    )
    assert result.text == "fixture paragraph"
    assert result.estimated_cost_usd == 0
    assert provider.calls == [("baseline", "cohort-a")]


def test_live_provider_requires_explicit_opt_in(tmp_path) -> None:
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=False,
        client=fake_client(),
    )
    with pytest.raises(LiveProviderDisabled):
        provider.generate(operation="smoke", request_key="one", prompt="hello")
    assert list(tmp_path.iterdir()) == []


def test_live_provider_archives_once_and_forwards_structured_output(tmp_path) -> None:
    client = fake_client('{"answer":"yes"}')
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
        client=client,
        clock=lambda: datetime(2026, 8, 15, tzinfo=UTC),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = provider.generate(
        operation="smoke",
        request_key="one",
        prompt="hello",
        output_schema=schema,
    )
    assert result.parsed_json == {"answer": "yes"}
    call = client.messages.calls[0]
    assert call["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": schema},
    }
    archive = json.loads(result.archive_path.read_text())
    assert archive["status"] == "complete"
    assert archive["usage"]["input_tokens"] == 100
    assert "api" not in json.dumps(archive).casefold()
    with pytest.raises(ProviderAttemptExists):
        provider.generate(operation="smoke", request_key="one", prompt="hello")
    assert len(client.messages.calls) == 1


def test_budget_is_checked_before_client_call(tmp_path) -> None:
    client = fake_client()
    provider = AnthropicProvider(
        model="claude-fable-5",
        effort="high",
        max_tokens=100_000,
        archive_dir=tmp_path,
        max_budget_usd=0.01,
        live_enabled=True,
        client=client,
    )
    with pytest.raises(ProviderBudgetExceeded):
        provider.generate(operation="expensive", request_key="one", prompt="hello")
    assert client.messages.calls == []


def test_global_budget_spans_nested_operation_archives(tmp_path) -> None:
    first_client = fake_client()
    first = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path / "question-a" / "baseline",
        max_budget_usd=1,
        live_enabled=True,
        global_budget_dir=tmp_path,
        global_max_budget_usd=0.0013,
        client=first_client,
    )
    first.generate(operation="baseline", request_key="one", prompt="hello")
    assert len(first_client.messages.calls) == 1

    second_client = fake_client()
    second = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path / "question-a" / "verification",
        max_budget_usd=1,
        live_enabled=True,
        global_budget_dir=tmp_path,
        global_max_budget_usd=0.0013,
        client=second_client,
    )
    with pytest.raises(ProviderBudgetExceeded):
        second.generate(operation="verification", request_key="two", prompt="hello")
    assert second_client.messages.calls == []


def test_live_environment_loader_requires_opt_in_and_mode_0600(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=dummy-not-a-real-key\n", encoding="utf-8")
    env_path.chmod(0o644)
    with pytest.raises(LiveProviderDisabled):
        load_live_environment(env_path, live_enabled=False)
    with pytest.raises(ProviderError, match="mode 0600"):
        load_live_environment(env_path, live_enabled=True)


def test_invalid_structured_text_archives_reported_usage_without_retry(tmp_path) -> None:
    client = fake_client("not-json")
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
        client=client,
    )
    with pytest.raises(ProviderError, match="attempt archived"):
        provider.generate(
            operation="verify",
            request_key="batch-one",
            prompt="hello",
            output_schema={"type": "object", "additionalProperties": False},
        )
    archive = json.loads(next(tmp_path.glob("*.provider.json")).read_text())
    assert archive["failure"] == "PROVIDER_INVALID_STRUCTURED_JSON"
    assert archive["cost_basis"] == "reported_usage"
    assert archive["estimated_cost_usd"] > 0
    assert len(client.messages.calls) == 1


def test_cost_estimate_uses_pinned_rates() -> None:
    usage = ProviderUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost_usd("claude-sonnet-5", usage) == pytest.approx(12.0)
