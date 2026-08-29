from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import scripts.run_hosted_native_numeric_canary_v4 as canary_cli
from pydantic import ValidationError

import literature_multiverse.hosted_native_numeric_canary_v4 as canary
from literature_multiverse.hosted_native_numeric_canary_v4 import (
    CANARY_FIXTURE,
    CANARY_HARD_CEILING_USD_MICROS,
    HostedNativeNumericCanaryFailureV4,
    HostedNativeNumericCanaryRawResponseV4,
    HostedNativeNumericCanarySuccessV4,
    HostedNativeNumericCanaryV4Error,
    assert_source_free_canary_payload_v4,
    execute_hosted_native_numeric_canary_v4,
    freeze_hosted_native_numeric_canary_authorization_v4,
    freeze_hosted_native_numeric_canary_intent_v4,
    freeze_hosted_native_numeric_canary_plan_v4,
    load_successful_hosted_native_numeric_canary_v4,
    preflight_hosted_native_numeric_canary_execution_v4,
    prepare_hosted_native_numeric_canary_v4,
    require_hosted_native_numeric_canary_binding_v4,
    validate_hosted_native_numeric_canary_terminal_v4,
)
from literature_multiverse.lineage import hash_canonical

_FAKE_API_KEY = "sk-ant-test-source-free-canary-never-send-0123456789"


class _Http400(Exception):
    status_code = 400
    request_id = "req_canary_http_400"


def _raw(**changes: Any) -> HostedNativeNumericCanaryRawResponseV4:
    response = HostedNativeNumericCanaryRawResponseV4(
        response_id="msg_source_free_canary_v4",
        response_model="claude-fable-5",
        stop_reason="end_turn",
        content_block_count=1,
        content_text=json.dumps(CANARY_FIXTURE, separators=(",", ":")),
        input_tokens=100,
        output_tokens=10,
    )
    return replace(response, **changes)


class _Client:
    def __init__(
        self,
        *,
        behavior: str = "success",
        workspace: Path | None = None,
        entered: Event | None = None,
        release: Event | None = None,
    ) -> None:
        self.behavior = behavior
        self.workspace = workspace
        self.entered = entered
        self.release = release
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def generate(self, wire_request: dict[str, Any]) -> HostedNativeNumericCanaryRawResponseV4:
        if self.workspace is not None:
            assert (self.workspace / "01-authorization.json").is_file()
            assert (self.workspace / "02-intent.json").is_file()
        self.calls += 1
        self.requests.append(wire_request)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        if self.behavior == "http_400":
            raise _Http400("provider detail must not be archived")
        if self.behavior == "exception":
            raise RuntimeError("transport detail must not be archived")
        if self.behavior == "refusal":
            return _raw(stop_reason="refusal")
        if self.behavior == "max_tokens":
            return _raw(stop_reason="max_tokens")
        if self.behavior == "wrong_model":
            return _raw(response_model="different-model")
        if self.behavior == "missing_id":
            return _raw(response_id=None)
        if self.behavior == "source_id":
            return _raw(response_id="PMC2427034")
        if self.behavior == "missing_usage":
            return _raw(input_tokens=None, output_tokens=None)
        if self.behavior == "invalid_json":
            return _raw(content_text="not-json")
        if self.behavior == "invalid_schema":
            return _raw(content_text='{"canary":"WRONG","ordinal":4}')
        if self.behavior == "multiple_blocks":
            return _raw(content_block_count=2)
        if self.behavior == "usage_over_ceiling":
            return _raw(input_tokens=1_000_000)
        return _raw()


def _prepare(tmp_path: Path) -> tuple[Path, Any, Any]:
    workspace = tmp_path.resolve() / "canary-v4"
    plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
    return workspace, plan, authorization


def _execute(workspace: Path, plan: Any, authorization: Any, client: Any) -> Any:
    return execute_hosted_native_numeric_canary_v4(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )


def test_plan_is_unique_source_free_prompt_json_and_within_cost_cap() -> None:
    first = freeze_hosted_native_numeric_canary_plan_v4()
    second = freeze_hosted_native_numeric_canary_plan_v4()
    first_auth = freeze_hosted_native_numeric_canary_authorization_v4(first)
    second_auth = freeze_hosted_native_numeric_canary_authorization_v4(second)
    first_intent = freeze_hosted_native_numeric_canary_intent_v4(
        plan=first, authorization=first_auth
    )
    second_intent = freeze_hosted_native_numeric_canary_intent_v4(
        plan=second, authorization=second_auth
    )

    assert first.execution_id != second.execution_id
    assert first.request_key != second.request_key
    assert first.request_sha256 != second.request_sha256
    assert first_intent.attempt_id != second_intent.attempt_id
    assert first.provider_config.model == "claude-fable-5"
    assert first.provider_config.effort == "high"
    assert first.provider_config.service_tier == "standard_only"
    assert first.provider_config.max_output_tokens == 10_240
    assert first.provider_config.transport_mode == "prompt_json_schema"
    assert first.provider_config.structured_grammar_enforced_by_provider is False
    assert first.provider_config.output_format_present_in_call is False
    assert first.wire_request["output_config"] == {"effort": "high"}
    assert first.certified_request_liability_usd_micros == 532_210
    assert first.certified_request_liability_usd_micros <= CANARY_HARD_CEILING_USD_MICROS
    assert first.project_after_v4_reservation_usd_micros == 61_031_869
    assert first_auth.certified_request_liability_usd_micros == 532_210
    assert first.delivered_schema_sha256 == hash_canonical(canary.CANARY_SCHEMA)
    assert '"const"' in first.model_system


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"content": "PMC2427034"}]},
        {"schema": ("2afd321025e677af36ddef3d26a03af1e7197cdac5798996c2415324c436c049")},
        {"path": "data/cache/evidence-inference-2.0/txt_files/PMC3104134.txt"},
        {"api_key": "redacted"},
        {"message": _FAKE_API_KEY},
    ],
)
def test_source_free_scanner_rejects_sources_and_credential_surfaces(payload: Any) -> None:
    with pytest.raises(HostedNativeNumericCanaryV4Error):
        assert_source_free_canary_payload_v4(payload)


def test_prepare_is_fresh_private_and_authorizes_before_intent(tmp_path: Path) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    assert stat_mode(workspace) == 0o700
    assert {path.name for path in workspace.iterdir()} == {
        ".lock",
        "00-plan.json",
        "01-authorization.json",
    }
    assert all(stat_mode(path) == 0o600 for path in workspace.iterdir() if path.is_file())
    assert not (workspace / "02-intent.json").exists()
    assert preflight_hosted_native_numeric_canary_execution_v4(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_authorization_sha256=authorization.authorization_sha256,
    )
    with pytest.raises(
        HostedNativeNumericCanaryV4Error,
        match="workspace_must_be_fresh",
    ):
        prepare_hosted_native_numeric_canary_v4(workspace=workspace)


def stat_mode(path: Path) -> int:
    return path.stat(follow_symlinks=False).st_mode & 0o777


def test_success_is_durable_strict_replayable_and_public_safe(tmp_path: Path) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    client = _Client(workspace=workspace)
    first = _execute(workspace, plan, authorization, client)
    second = _execute(workspace, plan, authorization, client)

    assert isinstance(first, HostedNativeNumericCanarySuccessV4)
    assert second == first
    assert client.calls == 1
    assert first.parsed_fixture == CANARY_FIXTURE
    assert first.fixture_exact is True
    assert first.wire_schema_validated is True
    assert first.full_acceptance_schema_validated is True
    assert first.provider_attempts_observed == 1
    assert first.application_retries == first.sdk_retries == 0
    assert first.observed_cost_usd_micros == 1_500
    assert first.charged_cost_upper_bound_usd_micros == 532_210
    assert validate_hosted_native_numeric_canary_terminal_v4(workspace=workspace) == first

    binding = load_successful_hosted_native_numeric_canary_v4(
        workspace=workspace,
        expected_terminal_sha256=first.terminal_sha256,
    )
    assert binding.terminal_sha256 == first.terminal_sha256
    assert binding.provider_result_sha256 == first.provider_result_sha256
    assert binding.certified_request_liability_usd_micros == 532_210
    assert binding.delivered_schema_sha256 == first.delivered_schema_sha256
    assert len(binding.terminal_artifact_sha256) == 64
    assert (
        require_hosted_native_numeric_canary_binding_v4(
            workspace=workspace,
            expected_binding=binding,
        )
        == binding
    )
    terminal_text = (workspace / "03-terminal.json").read_text(encoding="utf-8")
    assert "PMC2427034" not in terminal_text
    assert "PMC3104134" not in terminal_text
    assert _FAKE_API_KEY not in terminal_text
    assert "scientific_authority" in terminal_text


@pytest.mark.parametrize(
    ("behavior", "failure_code", "ambiguous"),
    [
        ("http_400", "provider_http_failure", False),
        ("exception", "provider_call_ambiguous_exception", True),
        ("refusal", "response_stop_reason_refusal", False),
        ("max_tokens", "response_stop_reason_max_tokens", False),
        ("wrong_model", "response_model_mismatch", False),
        ("missing_id", "response_id_invalid", False),
        ("source_id", "response_id_invalid", False),
        ("missing_usage", "response_usage_invalid", False),
        ("invalid_json", "response_json_invalid", False),
        ("invalid_schema", "response_schema_invalid", False),
        ("multiple_blocks", "response_content_invalid", False),
        ("usage_over_ceiling", "response_usage_invalid", False),
    ],
)
def test_failure_matrix_is_terminal_once_and_keeps_full_liability(
    tmp_path: Path,
    behavior: str,
    failure_code: str,
    ambiguous: bool,
) -> None:
    workspace = tmp_path.resolve() / behavior
    plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
    client = _Client(behavior=behavior, workspace=workspace)
    first = _execute(workspace, plan, authorization, client)
    second = _execute(workspace, plan, authorization, client)

    assert isinstance(first, HostedNativeNumericCanaryFailureV4)
    assert second == first
    assert client.calls == 1
    assert first.failure_code == failure_code
    assert (first.status == "terminal_ambiguous_attempt_poison") is ambiguous
    assert first.retry_permitted is False
    assert first.charged_cost_upper_bound_usd_micros == 532_210
    assert "detail must not be archived" not in (workspace / "03-terminal.json").read_text()
    with pytest.raises(HostedNativeNumericCanaryV4Error, match="success_terminal_required"):
        load_successful_hosted_native_numeric_canary_v4(workspace=workspace)


def test_orphan_intent_is_poisoned_without_transport(tmp_path: Path) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    intent = freeze_hosted_native_numeric_canary_intent_v4(plan=plan, authorization=authorization)
    canary._atomic_write_json_0600(workspace / "02-intent.json", intent)
    client = _Client()

    assert (
        preflight_hosted_native_numeric_canary_execution_v4(
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_authorization_sha256=authorization.authorization_sha256,
        )
        is False
    )
    terminal = _execute(workspace, plan, authorization, client)
    assert isinstance(terminal, HostedNativeNumericCanaryFailureV4)
    assert terminal.failure_code == "orphan_intent_on_resume"
    assert terminal.provider_attempt_observation == "unknown_after_orphaned_intent"
    assert client.calls == 0


def test_concurrent_callers_share_one_attempt(tmp_path: Path) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    entered = Event()
    release = Event()
    client = _Client(workspace=workspace, entered=entered, release=release)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_execute, workspace, plan, authorization, client)
        assert entered.wait(timeout=5)
        second = executor.submit(_execute, workspace, plan, authorization, client)
        release.set()
        first_terminal = first.result(timeout=5)
        second_terminal = second.result(timeout=5)
    assert first_terminal == second_terminal
    assert client.calls == 1


def test_workspace_mode_symlink_hardlink_and_unexpected_file_fail_closed(
    tmp_path: Path,
) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    workspace.chmod(0o755)
    with pytest.raises(HostedNativeNumericCanaryV4Error, match="workspace_missing_or_unsafe"):
        preflight_hosted_native_numeric_canary_execution_v4(
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_authorization_sha256=authorization.authorization_sha256,
        )
    workspace.chmod(0o700)
    (workspace / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HostedNativeNumericCanaryV4Error, match="unexpected_workspace_artifact"):
        preflight_hosted_native_numeric_canary_execution_v4(
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_authorization_sha256=authorization.authorization_sha256,
        )
    (workspace / "unexpected.json").unlink()
    plan_path = workspace / "00-plan.json"
    hardlink = tmp_path.resolve() / "plan-hardlink.json"
    os.link(plan_path, hardlink)
    with pytest.raises(HostedNativeNumericCanaryV4Error, match="artifact_missing_or_unsafe"):
        preflight_hosted_native_numeric_canary_execution_v4(
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_authorization_sha256=authorization.authorization_sha256,
        )
    hardlink.unlink()
    target = tmp_path.resolve() / "symlink-target"
    target.mkdir(mode=0o700)
    symlink = tmp_path.resolve() / "workspace-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(HostedNativeNumericCanaryV4Error):
        prepare_hosted_native_numeric_canary_v4(workspace=symlink)


def test_budget_and_terminal_tampering_fail_closed(tmp_path: Path) -> None:
    plan = freeze_hosted_native_numeric_canary_plan_v4()
    authorization = freeze_hosted_native_numeric_canary_authorization_v4(plan)
    changed = authorization.model_dump(mode="json")
    changed["certified_request_liability_usd_micros"] = 600_001
    changed["authorization_sha256"] = hash_canonical(
        {key: value for key, value in changed.items() if key != "authorization_sha256"}
    )
    with pytest.raises(ValidationError):
        canary.HostedNativeNumericCanaryAuthorizationV4.model_validate(changed)

    workspace, plan, authorization = _prepare(tmp_path)
    terminal = _execute(workspace, plan, authorization, _Client())
    path = workspace / "03-terminal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response_id"] = "msg_tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises((ValidationError, HostedNativeNumericCanaryV4Error)):
        load_successful_hosted_native_numeric_canary_v4(
            workspace=workspace,
            expected_terminal_sha256=terminal.terminal_sha256,
        )


def _write_env(path: Path, key: str = _FAKE_API_KEY) -> None:
    path.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    path.chmod(0o600)


def test_cli_opens_env_only_after_offline_preflight_and_never_persists_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    workspace = root / "cli-canary"
    plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
    env = root / ".env"
    _write_env(env)
    observed_keys: list[str] = []
    client = _Client(workspace=workspace)

    def factory(api_key: str) -> _Client:
        observed_keys.append(api_key)
        return client

    assert (
        canary_cli.main(
            [
                "execute",
                "--repository-root",
                str(root),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(env),
                "--live",
            ],
            client_factory=factory,
        )
        == 0
    )
    assert observed_keys == [_FAKE_API_KEY]
    assert client.calls == 1
    captured = capsys.readouterr()
    assert _FAKE_API_KEY not in captured.out
    assert _FAKE_API_KEY not in captured.err
    for path in workspace.iterdir():
        if path.is_file():
            assert _FAKE_API_KEY.encode() not in path.read_bytes()

    env.unlink()
    assert (
        canary_cli.main(
            [
                "execute",
                "--repository-root",
                str(root),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(env),
                "--live",
            ],
            client_factory=factory,
        )
        == 0
    )
    assert observed_keys == [_FAKE_API_KEY]
    assert client.calls == 1


def test_cli_anchor_failure_precedes_missing_env_and_client_factory(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    workspace = root / "cli-anchor"
    plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
    factory_calls = 0

    def factory(api_key: str) -> _Client:
        nonlocal factory_calls
        del api_key
        factory_calls += 1
        return _Client()

    with pytest.raises(
        HostedNativeNumericCanaryV4Error,
        match="execution_anchor_mismatch",
    ):
        canary_cli.main(
            [
                "execute",
                "--repository-root",
                str(root),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                "0" * 64,
                "--expected-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(root / "missing.env"),
                "--live",
            ],
            client_factory=factory,
        )
    assert plan.plan_sha256 != "0" * 64
    assert factory_calls == 0
    assert not (workspace / "02-intent.json").exists()


def test_success_loader_rejects_wrong_expected_hash(tmp_path: Path) -> None:
    workspace, plan, authorization = _prepare(tmp_path)
    _execute(workspace, plan, authorization, _Client())
    with pytest.raises(
        HostedNativeNumericCanaryV4Error,
        match="expected_terminal_hash_mismatch",
    ):
        load_successful_hosted_native_numeric_canary_v4(
            workspace=workspace,
            expected_terminal_sha256="0" * 64,
        )
