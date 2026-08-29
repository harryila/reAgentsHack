from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jsonschema.validators import validator_for

from literature_multiverse.native_bounded_schema_v2 import (
    synthetic_schema_v2_preflight_specs,
)
from literature_multiverse.providers import (
    AnthropicProvider,
    FixtureProvider,
    LiveProviderDisabled,
    ProviderAttemptExists,
    ProviderBudgetExceeded,
    ProviderError,
    ProviderUsage,
    _prepare_anthropic_schema,
    estimate_cost_usd,
    load_live_environment,
    sha256_json,
)


class FakeMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeBadRequestError(Exception):
    status_code = 400
    request_id = "req_safe_test"


class FailingMessages:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        raise self.error


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
    assert archive["output_schema"] == schema
    assert archive["output_schema_original_sha256"] == sha256_json(schema)
    assert archive["output_schema_provider"] == schema
    assert archive["output_schema_provider_sha256"] == sha256_json(schema)
    assert archive["output_schema_transform"]["name"] == (
        "anthropic-literal-type-compiler-v1+anthropic.transform_schema"
    )
    assert archive["conservative_request_ceiling_usd"] > (
        archive["estimated_cost_usd"]
    )
    assert archive["conservative_ceiling_basis"] == {
        "input_token_upper_bound": (
            "canonical_wire_request_utf8_bytes_plus_fixed_framing"
        ),
        "fixed_framing_tokens": 1024,
        "output_tokens": 100,
    }
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
        global_max_budget_usd=0.0035,
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
        global_max_budget_usd=0.0035,
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


def test_live_provider_transforms_wire_schema_but_archives_strict_original(tmp_path) -> None:
    client = fake_client('{"code":"AB","items":["x"]}')
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
        client=client,
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "pattern": "^[A-Z]+$",
                "minLength": 2,
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 1,
            },
        },
        "required": ["code", "items"],
        "additionalProperties": False,
    }
    original_before = deepcopy(schema)

    result = provider.generate(
        operation="structured",
        request_key="transform",
        prompt="hello",
        output_schema=schema,
    )

    assert schema == original_before
    wire_schema = client.messages.calls[0]["output_config"]["format"]["schema"]

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key for child in value.values() for key in all_keys(child)
            }
        if isinstance(value, list):
            return {key for child in value for key in all_keys(child)}
        return set()

    assert not {"$schema", "pattern", "minLength", "maxItems"} & all_keys(wire_schema)
    assert wire_schema["properties"]["code"]["description"] == (
        "{pattern: ^[A-Z]+$, minLength: 2}"
    )
    archive = json.loads(result.archive_path.read_text(encoding="utf-8"))
    assert archive["output_schema"] == original_before
    assert archive["output_schema_provider"] == wire_schema
    assert archive["output_schema_original_sha256"] == sha256_json(original_before)
    assert archive["output_schema_provider_sha256"] == sha256_json(wire_schema)
    assert archive["request_sha256"] == result.request_sha256


def test_transformed_schema_does_not_weaken_local_validation(tmp_path) -> None:
    client = fake_client('{"code":"lowercase"}')
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
        client=client,
    )
    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "pattern": "^[A-Z]+$", "minLength": 2}
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    with pytest.raises(ProviderError, match="attempt archived"):
        provider.generate(
            operation="structured",
            request_key="strict-local",
            prompt="hello",
            output_schema=schema,
        )

    archive = json.loads(next(tmp_path.glob("*.provider.json")).read_text(encoding="utf-8"))
    assert archive["failure"] == "PROVIDER_STRUCTURED_OUTPUT_SCHEMA_MISMATCH"
    assert archive["failure_detail"]["exception_type"] == "JSONSchemaValidationError"
    assert "does not match" in archive["failure_detail"]["message"]
    assert archive["cost_basis"] == "reported_usage"
    assert len(client.messages.calls) == 1


def test_bad_request_error_detail_is_sanitized_and_not_charged(tmp_path) -> None:
    error = FakeBadRequestError(
        "unsupported schema; api_key=sk-ant-super-secret; "
        "Authorization: Bearer very-secret; token=also-secret"
    )
    messages = FailingMessages(error)
    client = SimpleNamespace(messages=messages)
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
        client=client,
    )

    with pytest.raises(ProviderError, match="no retry"):
        provider.generate(
            operation="structured",
            request_key="bad-request",
            prompt="hello",
            output_schema={"type": "object", "additionalProperties": False},
        )

    archive_path = next(tmp_path.glob("*.provider.json"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    serialized = json.dumps(archive)
    assert archive["failure"] == "FakeBadRequestError"
    assert archive["failure_detail"]["status_code"] == 400
    assert archive["failure_detail"]["request_id"] == "req_safe_test"
    assert archive["estimated_cost_usd"] == 0
    assert archive["cost_basis"] == "known_bad_request_before_generation"
    assert "unsupported schema" in archive["failure_detail"]["message"]
    assert "sk-ant-super-secret" not in serialized
    assert "very-secret" not in serialized
    assert "also-secret" not in serialized
    assert len(messages.calls) == 1


def test_cost_estimate_uses_pinned_rates() -> None:
    usage = ProviderUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost_usd("claude-sonnet-5", usage) == pytest.approx(12.0)


def test_default_anthropic_client_disables_sdk_retries(tmp_path) -> None:
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=100,
        archive_dir=tmp_path,
        max_budget_usd=1,
        live_enabled=True,
    )
    sentinel = object()
    with patch("anthropic.Anthropic", return_value=sentinel) as constructor:
        assert provider._client_or_create() is sentinel
    constructor.assert_called_once_with(max_retries=0)


def test_all_v2_preflight_provider_schemas_transform_for_anthropic() -> None:
    specs = synthetic_schema_v2_preflight_specs()
    assert len(specs) == 8
    for spec in specs:
        original, transformed, sdk_version = _prepare_anthropic_schema(
            spec["provider_schema"]
        )
        assert original == spec["provider_schema"]
        validator = validator_for(transformed)
        validator.check_schema(transformed)
        validator(transformed).validate(spec["valid_example"])
        assert sdk_version == "0.120.2"
