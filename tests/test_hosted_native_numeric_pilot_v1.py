from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import scripts.run_hosted_native_numeric_pilot_v1 as pilot_cli

import literature_multiverse.hosted_native_numeric_pilot_v1 as pilot_runtime
from literature_multiverse.hosted_native_grounding_bridge import (
    build_hosted_native_grounding_package_v1,
)
from literature_multiverse.hosted_native_numeric_canary_v4 import (
    CANARY_FIXTURE,
    HostedNativeNumericCanaryRawResponseV4,
    execute_hosted_native_numeric_canary_v4,
    prepare_hosted_native_numeric_canary_v4,
)
from literature_multiverse.hosted_native_numeric_pilot_v1 import (
    CANARY_LIABILITY_ALLOCATION_USD_MICROS,
    COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE,
    MAXIMUM_PROVIDER_CALLS,
    NEW_LIABILITY_HARD_CEILING_USD_MICROS,
    PROVIDER_GRAMMAR_ENABLED,
    TRANSPORT_MODE,
    AnthropicHostedNativeNumericGenerationClientV1,
    HostedNativeNumericPilotError,
    HostedNativeNumericPilotPlanV1,
    HostedNativeNumericPreparedSurfaceV1,
    HostedNativeNumericRawResponseV1,
    _expected_extraction_payload,
    authorize_hosted_native_numeric_pilot_v1,
    count_hosted_native_numeric_pilot_tokens_v1,
    execute_hosted_native_numeric_pilot_v1,
    freeze_hosted_native_numeric_pilot_plan_v1,
    freeze_hosted_native_numeric_reservation_v1,
    load_hosted_native_numeric_pilot_plan_v1,
    preflight_hosted_native_numeric_execution_v1,
    prepare_hosted_native_numeric_pilot_v1,
    require_hosted_native_prompt_json_schema_guard_v1,
    reserve_hosted_native_numeric_pilot_v1,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical

ROOT = Path(__file__).resolve().parents[1]
_FAKE_API_KEY = "sk-ant-test-offline-fixture-never-send-0123456789"
_TEST_CANARY_WORKSPACE: Path | None = None
_TEST_CANARY_TERMINAL_SHA256: str | None = None


class _OfflineCanaryClient:
    def generate(self, wire_request: dict[str, object]) -> HostedNativeNumericCanaryRawResponseV4:
        assert wire_request["output_config"] == {"effort": "high"}
        return HostedNativeNumericCanaryRawResponseV4(
            response_id="msg_scientific_test_canary_v4",
            response_model="claude-fable-5",
            stop_reason="end_turn",
            content_block_count=1,
            content_text=json.dumps(CANARY_FIXTURE, separators=(",", ":")),
            input_tokens=100,
            output_tokens=10,
        )


@pytest.fixture(scope="session", autouse=True)
def validated_source_free_canary() -> Iterator[None]:
    global _TEST_CANARY_TERMINAL_SHA256, _TEST_CANARY_WORKSPACE
    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="numeric-v4-test-canary-", dir=cache) as directory:
        workspace = Path(directory) / "canary"
        plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
        terminal = execute_hosted_native_numeric_canary_v4(
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_authorization_sha256=authorization.authorization_sha256,
            client=_OfflineCanaryClient(),
        )
        assert terminal.status == "passed_source_free_prompt_json_canary"
        _TEST_CANARY_WORKSPACE = workspace
        _TEST_CANARY_TERMINAL_SHA256 = terminal.terminal_sha256
        yield
    _TEST_CANARY_WORKSPACE = None
    _TEST_CANARY_TERMINAL_SHA256 = None


def _canary_kwargs() -> dict[str, object]:
    assert _TEST_CANARY_WORKSPACE is not None
    assert _TEST_CANARY_TERMINAL_SHA256 is not None
    return {
        "canary_workspace": _TEST_CANARY_WORKSPACE,
        "expected_canary_terminal_sha256": _TEST_CANARY_TERMINAL_SHA256,
    }


def _prepare_successful_canary(parent: Path) -> tuple[Path, str]:
    workspace = parent / "source-free-canary"
    plan, authorization = prepare_hosted_native_numeric_canary_v4(workspace=workspace)
    terminal = execute_hosted_native_numeric_canary_v4(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_authorization_sha256=authorization.authorization_sha256,
        client=_OfflineCanaryClient(),
    )
    assert terminal.status == "passed_source_free_prompt_json_canary"
    return workspace, terminal.terminal_sha256


def _freeze_plan() -> object:
    return freeze_hosted_native_numeric_pilot_plan_v1(
        repository_root=ROOT,
        **_canary_kwargs(),
    )


def _prepare_plan(workspace: Path) -> object:
    return prepare_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        **_canary_kwargs(),
    )


@pytest.fixture
def repo_cache_sandbox() -> Iterator[Path]:
    cache = ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hosted-native-numeric-env-test-",
        dir=cache,
    ) as directory:
        yield Path(directory)


def _write_env(path: Path, *, mode: int = 0o600, include_key: bool = True) -> None:
    value = (
        f"ANTHROPIC_API_KEY={_FAKE_API_KEY}\n"
        if include_key
        else "UNRELATED_OFFLINE_TEST_VALUE=1\n"
    )
    path.write_text(value, encoding="utf-8")
    path.chmod(mode)


def _assert_key_not_persisted(workspace: Path, captured: object) -> None:
    assert _FAKE_API_KEY not in captured.out
    assert _FAKE_API_KEY not in captured.err
    encoded = _FAKE_API_KEY.encode("utf-8")
    for path in workspace.rglob("*"):
        if path.is_file():
            assert encoded not in path.read_bytes()


class _Counter:
    def __init__(self, count: int = 5000) -> None:
        self.count = count
        self.calls: list[dict[str, object]] = []

    def count_tokens(self, count_request: dict[str, object]) -> int:
        self.calls.append(count_request)
        return self.count


class _IntentCheckingCounter(_Counter):
    def __init__(self, *, workspace: Path, plan: object) -> None:
        super().__init__()
        self.workspace = workspace
        self.plan = plan

    def count_tokens(self, count_request: dict[str, object]) -> int:
        request = json.dumps(count_request, ensure_ascii=False)
        surface = next(item for item in self.plan.surfaces if item.roster_record.doc_id in request)
        assert (self.workspace / "count-intents" / f"{surface.intent.request_key}.json").is_file()
        return super().count_tokens(count_request)


_QUOTES = {
    "PMC2427034": "51.1% (157/307) vs. 83.9% (260/310)",
    "PMC3104134": (
        "In the eradication group 13% (20/152, 95% CI 9\u201320%) and in the placebo "
        "group 79% (123/155, 95% CI 72\u201385%)"
    ),
}
_COUNTS = {
    "PMC2427034": (157, 307, 260, 310),
    "PMC3104134": (20, 152, 123, 155),
}


class _GenerationClient:
    def __init__(self, plan: object, *, corrupt_first: bool = False) -> None:
        self.plan = plan
        self.corrupt_first = corrupt_first
        self.calls = 0

    def generate(self, wire_request: dict[str, object]) -> HostedNativeNumericRawResponseV1:
        self.calls += 1
        prompt = json.dumps(wire_request["messages"], ensure_ascii=False)
        surface = next(item for item in self.plan.surfaces if item.roster_record.doc_id in prompt)
        doc_id = surface.roster_record.doc_id
        counts = list(_COUNTS[doc_id])
        if self.corrupt_first and self.calls == 1:
            counts[0] += 1
        payload = _expected_extraction_payload(
            roster=surface.roster_record,
            treatment_events=counts[0],
            treatment_total=counts[1],
            control_events=counts[2],
            control_total=counts[3],
            quote=_QUOTES[doc_id],
        )
        return HostedNativeNumericRawResponseV1(
            response_id=f"msg_test_{self.calls}",
            response_model="claude-fable-5",
            stop_reason="end_turn",
            content_block_count=1,
            content_text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            input_tokens=5000,
            output_tokens=1000,
        )


class _Http400Error(Exception):
    status_code = 400
    request_id = "req_test_http_400"


class _TerminalGenerationClient(_GenerationClient):
    def __init__(self, plan: object, *, workspace: Path, behavior: str) -> None:
        super().__init__(plan)
        self.workspace = workspace
        self.behavior = behavior

    def generate(self, wire_request: dict[str, object]) -> HostedNativeNumericRawResponseV1:
        prompt = json.dumps(wire_request["messages"], ensure_ascii=False)
        surface = next(item for item in self.plan.surfaces if item.roster_record.doc_id in prompt)
        assert (
            self.workspace / "generation-intents" / f"{surface.intent.request_key}.json"
        ).is_file()
        if self.behavior == "http_400":
            self.calls += 1
            raise _Http400Error("invalid request")
        if self.behavior == "exception":
            self.calls += 1
            raise RuntimeError("ambiguous transport failure")
        response = super().generate(wire_request)
        if self.behavior == "malformed_json":
            return replace(response, content_text="{not-json")
        if self.behavior == "fenced_json":
            return replace(response, content_text=f"```json\n{response.content_text}\n```")
        if self.behavior == "leading_prose":
            return replace(response, content_text=f"Result: {response.content_text}")
        return replace(response, stop_reason=self.behavior)


def _prepare_authorized(tmp_path: Path) -> tuple[Path, object, object, object]:
    workspace = tmp_path / "fresh-pilot"
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    certificate = count_hosted_native_numeric_pilot_tokens_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_reservation_sha256=reservation.reservation_sha256,
        counter=_Counter(),
    )
    authorization = authorize_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_reservation_sha256=reservation.reservation_sha256,
        expected_count_certificate_sha256=certificate.certificate_sha256,
        phase_budget_usd_micros=NEW_LIABILITY_HARD_CEILING_USD_MICROS,
    )
    return workspace, plan, certificate, authorization


def test_plan_is_source_only_tiny_and_bounded() -> None:
    plan = _freeze_plan()
    assert plan.maximum_provider_calls == MAXIMUM_PROVIDER_CALLS == 2
    assert DEFAULT_CONFIG_PATH.name == "hosted-native-numeric-yield-pilot-v5.json"
    assert DEFAULT_WORKSPACE.name == "hosted-native-numeric-yield-pilot-v5-live"
    assert plan.config.run_id == "hosted-native-numeric-yield-pilot-fable5-prompt-json-v5"
    assert plan.maximum_new_liability_usd_micros == 2_400_000
    assert plan.canary_liability_allocation_usd_micros == 600_000
    assert plan.combined_v4_phase_hard_ceiling_usd_micros == 3_000_000
    assert plan.transport_mode == TRANSPORT_MODE == "prompt_json_schema"
    assert plan.provider_grammar_enabled is PROVIDER_GRAMMAR_ENABLED is False
    assert plan.maximum_canary_generation_calls == 1
    assert plan.maximum_combined_v4_generation_calls == 3
    assert plan.maximum_combined_v4_provider_contacts == 5
    assert plan.canary_success_binding.terminal_sha256 == plan.canary_terminal_sha256
    assert plan.canary_success_binding.binding_sha256 == plan.canary_success_binding_sha256
    assert plan.provider_identity.runtime_metadata["canary_terminal_sha256"] == (
        plan.canary_terminal_sha256
    )
    assert plan.provider_identity.runtime_metadata["output_format_present_in_call"] is False
    assert plan.config.selection_contract.labels_opened is False
    assert plan.config.selection_contract.private_predictions_opened is False
    assert plan.config.selection_contract.accuracy_authority is False
    assert [item.intent.doc_id for item in plan.surfaces] == [
        "PMC2427034",
        "PMC3104134",
    ]
    assert all(item.compiled_schema.wire_optional_parameter_count == 0 for item in plan.surfaces)
    assert all(item.compiled_schema.wire_union_parameter_count == 0 for item in plan.surfaces)
    assert all(item.offline_known_input_token_ceiling < 68_800 for item in plan.surfaces)
    assert all(
        item.offline_known_request_liability_usd_micros <= 1_200_000 for item in plan.surfaces
    )


def test_plan_requires_exact_successful_canary_before_source_resolution(
    repo_cache_sandbox: Path,
) -> None:
    missing = repo_cache_sandbox / "missing-canary"
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_canary_workspace_outside_repository_or_missing",
    ):
        freeze_hosted_native_numeric_pilot_plan_v1(
            repository_root=ROOT,
            canary_workspace=missing,
            expected_canary_terminal_sha256="0" * 64,
        )

    canary_workspace, terminal_sha256 = _prepare_successful_canary(repo_cache_sandbox)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_successful_source_free_canary_required",
    ):
        freeze_hosted_native_numeric_pilot_plan_v1(
            repository_root=ROOT,
            canary_workspace=canary_workspace,
            expected_canary_terminal_sha256=("0" * 64 if terminal_sha256 != "0" * 64 else "1" * 64),
        )


def test_original_schema_is_delivered_once_without_provider_grammar() -> None:
    plan = _freeze_plan()
    for surface in plan.surfaces:
        request = json.loads(surface.intent.wire_request_utf8)
        output_config = request["output_config"]
        assert output_config == {"effort": "high"}
        assert "format" not in output_config
        assert request["max_tokens"] == 10_240
        assert surface.generation_schema.schema_payload == surface.compiled_schema.original_schema
        assert surface.delivered_schema_sha256 == hash_canonical(
            surface.compiled_schema.original_schema
        )
        metrics = surface.schema_structural_metrics
        assert metrics.node_count == 401
        assert (
            metrics.schema_utf8_bytes
            == {
                "PMC2427034": 6_598,
                "PMC3104134": 6_599,
            }[surface.roster_record.doc_id]
        )
        assert metrics.max_depth == 17
        assert metrics.object_schema_count == 10
        assert metrics.total_object_properties == 80
        assert metrics.array_schema_count == 12
        assert metrics.arrays_with_min_items == 12
        assert metrics.arrays_with_max_items == 12
        assert metrics.const_keyword_count == 59

        canonical_schema = canonical_json_bytes(surface.compiled_schema.original_schema).decode(
            "utf-8"
        )
        model_system = request["system"]
        assert isinstance(model_system, str)
        assert model_system.endswith(canonical_schema)
        assert model_system.count(canonical_schema) == 1
        assert f"WIRE_SCHEMA_SHA256={surface.delivered_schema_sha256}" in model_system
        assert f"WIRE_SCHEMA_UTF8_BYTES={len(canonical_schema.encode('utf-8'))}" in model_system
        assert (
            surface.model_system_sha256 == hashlib.sha256(model_system.encode("utf-8")).hexdigest()
        )
        assert surface.model_system_utf8_bytes == len(model_system.encode("utf-8"))
        assert canonical_schema not in surface.prompt.rendered_prompt


def test_count_surface_is_exact_generation_input_surface() -> None:
    plan = _freeze_plan()
    for surface in plan.surfaces:
        generation = json.loads(surface.intent.wire_request_utf8)
        count = json.loads(surface.count_request_utf8)
        assert count == {
            key: value
            for key, value in generation.items()
            if key not in {"max_tokens", "service_tier"}
        }
        assert set(count) == {"messages", "model", "output_config", "system"}


def test_delivered_schema_guard_rejects_compiled_or_structurally_drifted_schema() -> None:
    plan = _freeze_plan()
    original = plan.surfaces[0].compiled_schema.original_schema
    compiled = plan.surfaces[0].compiled_schema.wire_schema
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_delivered_schema_guard_failed",
    ):
        require_hosted_native_prompt_json_schema_guard_v1(compiled)

    without_max_items = json.loads(canonical_json_bytes(original))
    pending = [without_max_items]
    removed = False
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("type") == "array" and "maxItems" in value and not removed:
                value.pop("maxItems")
                removed = True
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert removed
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_delivered_schema_guard_failed",
    ):
        require_hosted_native_prompt_json_schema_guard_v1(without_max_items)


def test_coherently_rehashed_provider_grammar_drift_is_rejected() -> None:
    plan = _freeze_plan()
    raw = plan.surfaces[0].model_dump(mode="json")
    intent = raw["intent"]
    request = json.loads(intent["wire_request_utf8"])
    request["output_config"]["format"] = {
        "type": "json_schema",
        "schema": raw["compiled_schema"]["wire_schema"],
    }
    intent["wire_request_utf8"] = canonical_json_bytes(request).decode("utf-8")
    intent["wire_request_sha256"] = hashlib.sha256(
        intent["wire_request_utf8"].encode("utf-8")
    ).hexdigest()
    intent_payload = {key: value for key, value in intent.items() if key != "intent_sha256"}
    intent["intent_sha256"] = hash_canonical(intent_payload)
    raw["intent"] = intent
    surface_payload = {key: value for key, value in raw.items() if key != "surface_sha256"}
    raw["surface_sha256"] = hash_canonical(surface_payload)
    with pytest.raises(ValueError, match="hosted_numeric_prepared_surface_mismatch"):
        HostedNativeNumericPreparedSurfaceV1.model_validate(raw)


def test_coherently_rehashed_provider_identity_and_intent_drift_is_rejected() -> None:
    plan = _freeze_plan()
    raw = plan.model_dump(mode="json")
    identity = raw["provider_identity"]
    identity["runtime_metadata"]["unapproved_identity_drift"] = True
    identity["runtime_metadata"] = dict(sorted(identity["runtime_metadata"].items()))
    identity_payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
    identity["identity_sha256"] = hash_canonical(identity_payload)
    raw["provider_identity"] = identity

    for surface in raw["surfaces"]:
        intent = surface["intent"]
        intent["provider_identity_sha256"] = identity["identity_sha256"]
        intent_payload = {key: value for key, value in intent.items() if key != "intent_sha256"}
        intent["intent_sha256"] = hash_canonical(intent_payload)
        surface["intent"] = intent
        surface_payload = {key: value for key, value in surface.items() if key != "surface_sha256"}
        surface["surface_sha256"] = hash_canonical(surface_payload)
    raw["surface_membership_sha256"] = hash_canonical(
        [surface["surface_sha256"] for surface in raw["surfaces"]]
    )
    plan_payload = {key: value for key, value in raw.items() if key != "plan_sha256"}
    raw["plan_sha256"] = hash_canonical(plan_payload)
    with pytest.raises(ValueError, match="hosted_numeric_plan_alias_mismatch"):
        HostedNativeNumericPilotPlanV1.model_validate(raw)


def test_v3_prepared_artifact_is_rejected_before_credentials_or_contact(
    repo_cache_sandbox: Path,
) -> None:
    plan = _freeze_plan()
    workspace = repo_cache_sandbox / "stale-v3-workspace"
    workspace.mkdir(mode=0o700)
    raw = plan.model_dump(mode="json")
    raw["plan_version"] = "hosted-native-numeric-pilot-plan-v3"
    plan_payload = {key: value for key, value in raw.items() if key != "plan_sha256"}
    raw["plan_sha256"] = hash_canonical(plan_payload)
    prepared = workspace / "00-prepared.json"
    prepared.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    prepared.chmod(0o600)
    with pytest.raises(ValueError):
        load_hosted_native_numeric_pilot_plan_v1(workspace=workspace)

    factory_calls = 0

    def factory(api_key: str) -> _Counter:
        nonlocal factory_calls
        del api_key
        factory_calls += 1
        return _Counter()

    with pytest.raises(ValueError):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                raw["plan_sha256"],
                "--expected-reservation-sha256",
                "0" * 64,
                "--env-file",
                str(repo_cache_sandbox / "missing.env"),
                "--live",
            ],
            token_counter_factory=factory,
        )
    assert factory_calls == 0
    assert not (workspace / "count-intents").exists()


def test_project_liability_receipt_precedes_network_and_is_below_100_dollars() -> None:
    plan = _freeze_plan()
    reservation = freeze_hosted_native_numeric_reservation_v1(plan=plan)
    prior = reservation.prior_accounting_receipt
    assert prior.reported_prior_spend_usd_micros == 38_616_150
    assert prior.unknown_prior_liability_usd_micros == 17_482_919
    assert prior.v3_certified_unresolved_upper_bound_usd_micros == 1_932_800
    assert prior.v4_certified_liability_upper_bound_usd_micros == 1_815_360
    assert prior.reconciled_prior_liability_usd_micros == 59_847_229
    assert reservation.new_liability_reserved_usd_micros == 2_400_000
    assert NEW_LIABILITY_HARD_CEILING_USD_MICROS == 2_400_000
    assert reservation.companion_canary_reserved_usd_micros == 600_000
    assert CANARY_LIABILITY_ALLOCATION_USD_MICROS == 600_000
    assert reservation.combined_v4_phase_reserved_usd_micros == 3_000_000
    assert COMBINED_V4_PHASE_HARD_CEILING_USD_MICROS == 3_000_000
    assert reservation.project_liability_after_reservation_usd_micros == 62_847_229
    assert reservation.project_liability_after_reservation_usd_micros < 100_000_000


def test_safe_env_loader_requires_0600_and_returns_only_in_memory_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key-must-not-win")
    observed = pilot_cli._load_anthropic_api_key(
        repository_root=tmp_path,
        env_file=env_file,
    )
    assert observed == _FAKE_API_KEY


def test_live_env_rejects_duplicate_key_and_hardlink(
    repo_cache_sandbox: Path,
) -> None:
    duplicate = repo_cache_sandbox / "duplicate.env"
    duplicate.write_text(
        f"ANTHROPIC_API_KEY={_FAKE_API_KEY}\nANTHROPIC_API_KEY=second-value\n",
        encoding="utf-8",
    )
    duplicate.chmod(0o600)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_anthropic_api_key_missing_or_duplicate",
    ):
        pilot_cli._load_anthropic_api_key(
            repository_root=ROOT,
            env_file=duplicate,
        )

    original = repo_cache_sandbox / "hardlink-source.env"
    linked = repo_cache_sandbox / "hardlink.env"
    _write_env(original)
    os.link(original, linked)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_live_env_not_regular_file",
    ):
        pilot_cli._safe_live_environment_file(
            repository_root=ROOT,
            env_file=linked,
        )


def test_live_env_must_be_repository_contained_and_not_a_symlink(
    repo_cache_sandbox: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / ".env"
    _write_env(outside)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_live_env_outside_repository",
    ):
        pilot_cli._safe_live_environment_file(
            repository_root=ROOT,
            env_file=outside,
        )

    target = repo_cache_sandbox / "credential-target"
    link = repo_cache_sandbox / ".env"
    _write_env(target)
    link.symlink_to(target)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_live_env_symlink_forbidden",
    ):
        pilot_cli._safe_live_environment_file(
            repository_root=ROOT,
            env_file=link,
        )


def test_explicit_key_reaches_anthropic_sdk_factory_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAnthropicSDK:
        @staticmethod
        def DefaultHttpxClient(**kwargs: object) -> object:
            captured["http"] = kwargs
            return object()

        @staticmethod
        def Anthropic(**kwargs: object) -> object:
            captured["client"] = kwargs
            return object()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setattr(pilot_runtime, "_require_sdk", lambda: FakeAnthropicSDK)
    pilot_runtime._anthropic_client(api_key=_FAKE_API_KEY)
    client_kwargs = captured["client"]
    assert isinstance(client_kwargs, dict)
    assert client_kwargs["api_key"] == _FAKE_API_KEY


def test_generation_adapter_accepts_one_text_block_with_reasoning_block() -> None:
    class Block:
        def __init__(self, block_type: str, text: str | None = None) -> None:
            self.type = block_type
            self.text = text

    class Usage:
        input_tokens = 100
        output_tokens = 20
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

    class Response:
        def __init__(self) -> None:
            self.id = "msg_multiblock_test"
            self.model = "claude-fable-5"
            self.stop_reason = "end_turn"
            self.usage = Usage()
            self.content = [Block("thinking"), Block("text", '{"ok":true}')]

    class Messages:
        @staticmethod
        def create(**_: object) -> Response:
            return Response()

    class Client:
        messages = Messages()

    adapter = AnthropicHostedNativeNumericGenerationClientV1(
        api_key=_FAKE_API_KEY,
        client=Client(),
    )
    raw = adapter.generate({})
    assert raw.content_block_count == 2
    assert raw.text_block_count == 1
    assert raw.non_text_block_types == ("thinking",)
    assert raw.content_text == '{"ok":true}'


def test_count_adapter_initialization_failure_precedes_intent(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = repo_cache_sandbox / "count-init-failure"
    env_file = repo_cache_sandbox / ".env"
    _write_env(env_file)
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )

    def fail_client(*, api_key: str) -> object:
        assert api_key == _FAKE_API_KEY
        raise RuntimeError("offline sdk initialization failure")

    monkeypatch.setattr(pilot_runtime, "_anthropic_client", fail_client)
    with pytest.raises(RuntimeError, match="offline sdk initialization failure"):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ]
        )
    assert not (workspace / "count-intents").exists()


def test_generation_adapter_initialization_failure_precedes_intent(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(repo_cache_sandbox)
    env_file = repo_cache_sandbox / ".env"
    _write_env(env_file)

    def fail_client(*, api_key: str) -> object:
        assert api_key == _FAKE_API_KEY
        raise RuntimeError("offline sdk initialization failure")

    monkeypatch.setattr(pilot_runtime, "_anthropic_client", fail_client)
    with pytest.raises(RuntimeError, match="offline sdk initialization failure"):
        pilot_cli.main(
            [
                "execute",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-generation-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ]
        )
    assert not (workspace / "generation-intents").exists()
    assert not (workspace / "call-authorizations").exists()


def test_count_cli_loads_key_after_preflight_without_persisting_it(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = repo_cache_sandbox / "count-workspace"
    env_file = repo_cache_sandbox / ".env"
    _write_env(env_file)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    observed_keys: list[str] = []

    def factory(api_key: str) -> _Counter:
        observed_keys.append(api_key)
        return _Counter()

    assert (
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ],
            token_counter_factory=factory,
        )
        == 0
    )
    assert observed_keys == [_FAKE_API_KEY]
    _assert_key_not_persisted(workspace, capsys.readouterr())


@pytest.mark.parametrize(
    ("env_case", "failure_code"),
    [
        ("missing", "hosted_numeric_live_env_missing_or_unsafe"),
        ("unsafe_mode", "hosted_numeric_live_env_mode_must_be_0600"),
        ("missing_key", "hosted_numeric_anthropic_api_key_missing_or_duplicate"),
        (
            "custom_transport",
            "hosted_numeric_transport_environment_override_forbidden",
        ),
    ],
)
def test_count_env_failure_precedes_intent_and_transport(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_case: str,
    failure_code: str,
) -> None:
    workspace = repo_cache_sandbox / "count-workspace"
    env_file = repo_cache_sandbox / ".env"
    if env_case == "unsafe_mode":
        _write_env(env_file, mode=0o644)
    elif env_case == "missing_key":
        _write_env(env_file, include_key=False)
    elif env_case == "custom_transport":
        _write_env(env_file)
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    factory_calls = 0

    def factory(api_key: str) -> _Counter:
        nonlocal factory_calls
        del api_key
        factory_calls += 1
        return _Counter()

    with pytest.raises(HostedNativeNumericPilotError, match=failure_code):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ],
            token_counter_factory=factory,
        )
    assert factory_calls == 0
    assert not (workspace / "count-intents").exists()


def test_count_anchor_gate_precedes_env_validation(repo_cache_sandbox: Path) -> None:
    workspace = repo_cache_sandbox / "count-workspace"
    missing_env = repo_cache_sandbox / "missing.env"
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_count_anchor_mismatch",
    ):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                "0" * 64,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(missing_env),
                "--live",
            ],
        )
    assert not (workspace / "count-intents").exists()


@pytest.mark.parametrize(
    ("unsafe_target", "failure_code"),
    [
        ("workspace", "hosted_numeric_workspace_missing_or_unsafe"),
        ("prepared", "hosted_numeric_prepared_plan_invalid"),
        ("lock", "hosted_numeric_workspace_lock_mode_invalid"),
    ],
)
def test_count_path_and_mode_gates_precede_env_opening(
    repo_cache_sandbox: Path,
    unsafe_target: str,
    failure_code: str,
) -> None:
    workspace = repo_cache_sandbox / "count-workspace"
    missing_env = repo_cache_sandbox / "missing.env"
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    target = {
        "workspace": workspace,
        "prepared": workspace / "00-prepared.json",
        "lock": workspace / ".lock",
    }[unsafe_target]
    target.chmod(0o755 if unsafe_target == "workspace" else 0o644)
    with pytest.raises(HostedNativeNumericPilotError, match=failure_code):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(missing_env),
                "--live",
            ]
        )
    assert not (workspace / "count-intents").exists()


def test_count_intent_is_durable_before_each_transport_call(tmp_path: Path) -> None:
    workspace = tmp_path / "fresh-pilot"
    plan = _prepare_plan(workspace)
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    counter = _IntentCheckingCounter(workspace=workspace, plan=plan)
    count_hosted_native_numeric_pilot_tokens_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_reservation_sha256=reservation.reservation_sha256,
        counter=counter,
    )
    assert len(counter.calls) == 2


def test_canary_artifact_drift_blocks_direct_and_cli_count_before_contact(
    repo_cache_sandbox: Path,
) -> None:
    canary_workspace, terminal_sha256 = _prepare_successful_canary(repo_cache_sandbox)
    workspace = repo_cache_sandbox / "science"
    plan = prepare_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        canary_workspace=canary_workspace,
        expected_canary_terminal_sha256=terminal_sha256,
    )
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    terminal_path = canary_workspace / "03-terminal.json"
    terminal_path.write_text(terminal_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    terminal_path.chmod(0o600)

    counter = _Counter()
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_canary_prerequisite_drift",
    ):
        count_hosted_native_numeric_pilot_tokens_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_reservation_sha256=reservation.reservation_sha256,
            counter=counter,
        )
    assert counter.calls == []
    assert not (workspace / "count-intents").exists()

    factory_calls = 0

    def factory(api_key: str) -> _Counter:
        nonlocal factory_calls
        del api_key
        factory_calls += 1
        return _Counter()

    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_canary_prerequisite_drift",
    ):
        pilot_cli.main(
            [
                "count",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-reservation-sha256",
                reservation.reservation_sha256,
                "--env-file",
                str(repo_cache_sandbox / "missing.env"),
                "--live",
            ],
            token_counter_factory=factory,
        )
    assert factory_calls == 0
    assert not (workspace / "count-intents").exists()


def test_canary_drift_blocks_authorization_after_count(
    repo_cache_sandbox: Path,
) -> None:
    canary_workspace, terminal_sha256 = _prepare_successful_canary(repo_cache_sandbox)
    workspace = repo_cache_sandbox / "science"
    plan = prepare_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        canary_workspace=canary_workspace,
        expected_canary_terminal_sha256=terminal_sha256,
    )
    reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
    )
    certificate = count_hosted_native_numeric_pilot_tokens_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_reservation_sha256=reservation.reservation_sha256,
        counter=_Counter(),
    )
    terminal_path = canary_workspace / "03-terminal.json"
    terminal_path.write_text(terminal_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    terminal_path.chmod(0o600)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_canary_prerequisite_drift",
    ):
        authorize_hosted_native_numeric_pilot_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_reservation_sha256=reservation.reservation_sha256,
            expected_count_certificate_sha256=certificate.certificate_sha256,
            phase_budget_usd_micros=NEW_LIABILITY_HARD_CEILING_USD_MICROS,
        )
    assert not (workspace / "03-generation-authorization.json").exists()


def test_scientific_cost_boundary_accepts_exact_cap_and_rejects_one_token_over(
    tmp_path: Path,
) -> None:
    accepted_workspace = tmp_path / "accepted"
    accepted_plan = _prepare_plan(accepted_workspace)
    accepted_reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=accepted_workspace,
        expected_plan_sha256=accepted_plan.plan_sha256,
    )
    accepted_counter = _Counter(count=67_776)
    certificate = count_hosted_native_numeric_pilot_tokens_v1(
        repository_root=ROOT,
        workspace=accepted_workspace,
        expected_plan_sha256=accepted_plan.plan_sha256,
        expected_reservation_sha256=accepted_reservation.reservation_sha256,
        counter=accepted_counter,
    )
    assert {item.certified_input_token_limit for item in certificate.receipts} == {68_800}
    assert certificate.certified_total_liability_usd_micros == 2_400_000
    authorization = authorize_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=accepted_workspace,
        expected_plan_sha256=accepted_plan.plan_sha256,
        expected_reservation_sha256=accepted_reservation.reservation_sha256,
        expected_count_certificate_sha256=certificate.certificate_sha256,
        phase_budget_usd_micros=2_400_000,
    )
    assert authorization.combined_v4_certified_liability_usd_micros == 2_932_210
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_generation_authorization_rejected",
    ):
        authorize_hosted_native_numeric_pilot_v1(
            repository_root=ROOT,
            workspace=accepted_workspace,
            expected_plan_sha256=accepted_plan.plan_sha256,
            expected_reservation_sha256=accepted_reservation.reservation_sha256,
            expected_count_certificate_sha256=certificate.certificate_sha256,
            phase_budget_usd_micros=2_400_001,
        )

    rejected_workspace = tmp_path / "rejected"
    rejected_plan = _prepare_plan(rejected_workspace)
    rejected_reservation = reserve_hosted_native_numeric_pilot_v1(
        workspace=rejected_workspace,
        expected_plan_sha256=rejected_plan.plan_sha256,
    )
    rejected_counter = _Counter(count=67_777)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_count_exceeds_cap",
    ):
        count_hosted_native_numeric_pilot_tokens_v1(
            repository_root=ROOT,
            workspace=rejected_workspace,
            expected_plan_sha256=rejected_plan.plan_sha256,
            expected_reservation_sha256=rejected_reservation.reservation_sha256,
            counter=rejected_counter,
        )
    assert len(rejected_counter.calls) == 1
    assert not (rejected_workspace / "02-token-count-certificate.json").exists()


def test_two_valid_calls_build_bridge_v4_without_reinterpretation(tmp_path: Path) -> None:
    workspace, plan, certificate, authorization = _prepare_authorized(tmp_path)
    client = _GenerationClient(plan)
    output = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert output.run.completed_extraction_count == 2
    assert output.terminal.release_grade_native_numeric_yield == 2
    assert output.terminal.certified_maximum_liability_usd_micros == (
        certificate.certified_total_liability_usd_micros
    )
    assert output.terminal.canary_success_binding_sha256 == plan.canary_success_binding_sha256
    assert output.terminal.canary_terminal_sha256 == plan.canary_terminal_sha256
    assert output.terminal.combined_v4_call_records_terminally_closed == 3
    assert output.terminal.companion_canary_reserved_usd_micros == 600_000
    assert output.terminal.combined_v4_phase_reserved_usd_micros == 3_000_000
    assert output.terminal.combined_v4_certified_liability_usd_micros == (
        certificate.certified_total_liability_usd_micros
        + plan.canary_success_binding.charged_cost_upper_bound_usd_micros
    )
    bridged = build_hosted_native_grounding_package_v1(
        run=output.run,
        repository_root=ROOT,
    )
    assert bridged.receipt.completed_extraction_count == 2
    assert bridged.receipt.estimable_fragment_count == 2
    assert len(bridged.corpus.graph.outcome_estimates) == 2
    assert bridged.package.package_version == "typed-evidence-grounding-package-v4"


def test_completed_run_repairs_missing_summary_terminal_without_transport(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    client = _GenerationClient(plan)
    completed = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    terminal_path = workspace / "05-terminal.json"
    terminal_path.unlink()

    assert (
        preflight_hosted_native_numeric_execution_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_generation_authorization_sha256=authorization.authorization_sha256,
        )
        is False
    )
    assert not terminal_path.exists()

    recovered = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert terminal_path.exists()
    assert recovered == completed


def test_execute_cli_loads_key_after_preflight_without_persisting_it(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(repo_cache_sandbox)
    env_file = repo_cache_sandbox / ".env"
    _write_env(env_file)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    observed_keys: list[str] = []
    client = _GenerationClient(plan)

    def factory(api_key: str) -> _GenerationClient:
        observed_keys.append(api_key)
        return client

    assert (
        pilot_cli.main(
            [
                "execute",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-generation-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ],
            generation_client_factory=factory,
        )
        == 0
    )
    assert observed_keys == [_FAKE_API_KEY]
    assert client.calls == 2
    _assert_key_not_persisted(workspace, capsys.readouterr())


@pytest.mark.parametrize(
    ("env_case", "failure_code"),
    [
        ("missing", "hosted_numeric_live_env_missing_or_unsafe"),
        ("unsafe_mode", "hosted_numeric_live_env_mode_must_be_0600"),
        ("missing_key", "hosted_numeric_anthropic_api_key_missing_or_duplicate"),
        (
            "custom_transport",
            "hosted_numeric_transport_environment_override_forbidden",
        ),
    ],
)
def test_execute_env_failure_precedes_intent_and_transport(
    repo_cache_sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_case: str,
    failure_code: str,
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(repo_cache_sandbox)
    env_file = repo_cache_sandbox / ".env"
    if env_case == "unsafe_mode":
        _write_env(env_file, mode=0o644)
    elif env_case == "missing_key":
        _write_env(env_file, include_key=False)
    elif env_case == "custom_transport":
        _write_env(env_file)
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", '{"x-test":"forbidden"}')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    factory_calls = 0

    def factory(api_key: str) -> _GenerationClient:
        nonlocal factory_calls
        del api_key
        factory_calls += 1
        return _GenerationClient(plan)

    with pytest.raises(HostedNativeNumericPilotError, match=failure_code):
        pilot_cli.main(
            [
                "execute",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-generation-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(env_file),
                "--live",
            ],
            generation_client_factory=factory,
        )
    assert factory_calls == 0
    assert not (workspace / "generation-intents").exists()
    assert not (workspace / "call-authorizations").exists()


def test_execute_anchor_gate_precedes_env_validation(repo_cache_sandbox: Path) -> None:
    workspace, _, _, authorization = _prepare_authorized(repo_cache_sandbox)
    missing_env = repo_cache_sandbox / "missing.env"
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_execution_anchor_mismatch",
    ):
        pilot_cli.main(
            [
                "execute",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                "0" * 64,
                "--expected-generation-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(missing_env),
                "--live",
            ],
        )
    assert not (workspace / "generation-intents").exists()
    assert not (workspace / "call-authorizations").exists()


def test_execute_workspace_mode_gate_precedes_env_opening(
    repo_cache_sandbox: Path,
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(repo_cache_sandbox)
    missing_env = repo_cache_sandbox / "missing.env"
    workspace.chmod(0o755)
    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_workspace_missing_or_unsafe",
    ):
        pilot_cli.main(
            [
                "execute",
                "--repository-root",
                str(ROOT),
                "--workspace",
                str(workspace),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-generation-authorization-sha256",
                authorization.authorization_sha256,
                "--env-file",
                str(missing_env),
                "--live",
            ]
        )
    assert not (workspace / "generation-intents").exists()
    assert not (workspace / "call-authorizations").exists()


def test_wrong_source_association_fails_terminally_without_retry(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    client = _GenerationClient(plan, corrupt_first=True)
    output = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert output.run.completed_extraction_count == 1
    assert output.run.failed_or_ambiguous_count == 1
    failures = [call for call in output.run.calls if call.terminal.outcome != "completed"]
    assert failures[0].terminal.failure_code == "response_target_association_invalid"
    assert failures[0].intent.application_retries == 0
    assert failures[0].intent.sdk_retries == 0


@pytest.mark.parametrize(
    ("behavior", "outcome", "failure_code"),
    [
        ("fenced_json", "provider_failed", "response_json_invalid"),
        ("leading_prose", "provider_failed", "response_json_invalid"),
        ("malformed_json", "provider_failed", "response_json_invalid"),
        ("refusal", "provider_failed", "response_stop_reason_refusal"),
        ("max_tokens", "provider_failed", "response_stop_reason_max_tokens"),
        ("http_400", "provider_failed", "provider_http_400"),
        (
            "exception",
            "ambiguous_attempt_poison",
            "provider_call_ambiguous_exception",
        ),
    ],
)
def test_terminal_provider_failures_are_durable_bridgeable_and_never_retried(
    tmp_path: Path,
    behavior: str,
    outcome: str,
    failure_code: str,
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    client = _TerminalGenerationClient(plan, workspace=workspace, behavior=behavior)
    output = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert output.run.completed_extraction_count == 0
    assert output.run.failed_or_ambiguous_count == 2
    assert {call.terminal.outcome for call in output.run.calls} == {outcome}
    assert {call.terminal.failure_code for call in output.run.calls} == {failure_code}
    assert all(call.intent.application_retries == 0 for call in output.run.calls)
    assert all(call.intent.sdk_retries == 0 for call in output.run.calls)

    replay = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert replay == output

    bridged = build_hosted_native_grounding_package_v1(
        run=output.run,
        repository_root=ROOT,
    )
    assert bridged.receipt.completed_extraction_count == 0
    assert bridged.receipt.failed_or_ambiguous_count == 2
    assert bridged.receipt.estimable_fragment_count == 0
    assert bridged.receipt.non_estimable_fragment_count == 2
    assert bridged.receipt.extraction_accuracy_benchmark_authority is False
    assert bridged.receipt.scientific_claim_truth_authority is False
    assert bridged.receipt.claim_release_authority is False


def test_direct_executor_rejects_downstream_state_without_intent(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    key = plan.surfaces[0].intent.request_key
    receipt_dir = workspace / "provider-receipts"
    receipt_dir.mkdir(mode=0o700)
    foreign = receipt_dir / f"{key}.json"
    foreign.write_text("{}\n", encoding="utf-8")
    foreign.chmod(0o600)
    client = _GenerationClient(plan)

    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_generation_state_without_intent",
    ):
        execute_hosted_native_numeric_pilot_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_generation_authorization_sha256=authorization.authorization_sha256,
            client=client,
        )
    assert client.calls == 0


def test_preflight_rejects_terminal_without_hosted_run(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    terminal = workspace / "05-terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    terminal.chmod(0o600)

    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_terminal_without_hosted_run",
    ):
        preflight_hosted_native_numeric_execution_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_generation_authorization_sha256=authorization.authorization_sha256,
        )


def test_preflight_revalidates_source_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)

    def reject_source(**_: object) -> object:
        raise HostedNativeNumericPilotError("forced_source_drift")

    monkeypatch.setattr(pilot_runtime, "resolve_native_source_document", reject_source)
    with pytest.raises(HostedNativeNumericPilotError, match="forced_source_drift"):
        preflight_hosted_native_numeric_execution_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_generation_authorization_sha256=authorization.authorization_sha256,
        )
    assert not (workspace / "generation-intents").exists()


def test_provider_secret_key_shape_is_redacted_before_persistence(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)

    class SecretShapeClient(_GenerationClient):
        def generate(self, wire_request: dict[str, object]) -> HostedNativeNumericRawResponseV1:
            response = super().generate(wire_request)
            return replace(response, content_text='{"api_key":"not-a-real-secret"}')

    client = SecretShapeClient(plan)
    output = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    assert client.calls == 2
    assert {call.terminal.failure_code for call in output.run.calls} == {
        "response_content_secret_rejected"
    }
    for directory in ("provider-receipts", "call-terminals"):
        for path in (workspace / directory).iterdir():
            assert b'"api_key"' not in path.read_bytes()
            assert b"not-a-real-secret" not in path.read_bytes()


def test_observed_request_budget_breach_invalidates_yield(tmp_path: Path) -> None:
    workspace, plan, certificate, authorization = _prepare_authorized(tmp_path)

    class OverBudgetClient(_GenerationClient):
        def generate(self, wire_request: dict[str, object]) -> HostedNativeNumericRawResponseV1:
            response = super().generate(wire_request)
            return replace(response, input_tokens=70_000)

    client = OverBudgetClient(plan)
    output = execute_hosted_native_numeric_pilot_v1(
        repository_root=ROOT,
        workspace=workspace,
        expected_plan_sha256=plan.plan_sha256,
        expected_generation_authorization_sha256=authorization.authorization_sha256,
        client=client,
    )
    terminal = output.terminal
    assert terminal.status == "terminal_scientific_budget_breach"
    assert terminal.release_grade_native_numeric_yield == 0
    assert terminal.request_budget_breach_count == 2
    assert terminal.provider_usage_missing_count == 0
    assert terminal.observed_generation_cost_usd_micros == 1_500_000
    assert terminal.generation_liability_accounted_usd_micros == 1_500_000
    assert terminal.generation_liability_accounted_usd_micros > (
        certificate.certified_total_liability_usd_micros
    )
    assert terminal.scientific_budget_breach_detected is True
    assert terminal.certified_budget_claim_valid is False


def test_old_or_foreign_request_identity_is_not_ignored(tmp_path: Path) -> None:
    workspace, plan, _, authorization = _prepare_authorized(tmp_path)
    intent_dir = workspace / "generation-intents"
    intent_dir.mkdir(mode=0o700)
    stale = intent_dir / "extract-old-v1-request.json"
    stale.write_text("{}\n", encoding="utf-8")
    stale.chmod(0o600)

    with pytest.raises(
        HostedNativeNumericPilotError,
        match="hosted_numeric_workspace_foreign_artifact",
    ):
        preflight_hosted_native_numeric_execution_v1(
            repository_root=ROOT,
            workspace=workspace,
            expected_plan_sha256=plan.plan_sha256,
            expected_generation_authorization_sha256=authorization.authorization_sha256,
        )
