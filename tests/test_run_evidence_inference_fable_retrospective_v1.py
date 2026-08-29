from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import scripts.run_evidence_inference_fable_retrospective_v1 as harness
from scripts.run_evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableHarnessError,
    _atomic_publish_score_payloads,
    _safe_private_target,
    _safe_public_target,
    main,
)

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFablePreparedRuntimeV1,
    execute_evidence_inference_fable_paired_v1,
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    EvidenceInferenceFableScoringError,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    DEFAULT_FULL_PLAN_PATH,
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

ROOT = Path(__file__).resolve().parents[1]


class _NoCallClient:
    def generate(self, _surface: Any) -> Any:  # pragma: no cover - gate must stop first
        raise AssertionError("provider call crossed a pre-call budget gate")


class _FakeTokenCounter:
    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, _wire_kwargs: object) -> int:
        self.calls += 1
        return 20_000


def _shared(workspace: Path, mode: str = "pilot30_paired") -> list[str]:
    return [
        "--repository-root",
        str(ROOT),
        "--mode",
        mode,
        "--workspace",
        str(workspace),
    ]


def test_prepare_authorize_budget_terminal_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    assert main(["prepare", *shared]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["artifact"] == "EvidenceInferenceFablePreparedRuntimeV1"

    assert main(["authorize", *shared, "--budget-usd-micros", "1"]) == 0
    authorization = json.loads(capsys.readouterr().out)
    assert authorization["artifact"] == "EvidenceInferenceFableBudgetAuthorizationV1"
    assert authorization["liability_basis"] == "full_context_fallback"

    plan, _ = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace,
        plan=plan,
        client=_NoCallClient(),
    )
    assert terminal.status == "clean_budget_exhaustion_before_next_pair"
    assert terminal.completed_request_count == 0

    assert main(["validate", *shared]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["terminal_sha256"] == terminal.terminal_sha256
    assert validated["full_population_score_permitted"] is False


def test_incomplete_runtime_stops_before_private_label_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    main(["prepare", *shared])
    capsys.readouterr()
    main(["authorize", *shared, "--budget-usd-micros", "1"])
    capsys.readouterr()
    plan, _ = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    execute_evidence_inference_fable_paired_v1(
        workspace=workspace,
        plan=plan,
        client=_NoCallClient(),
    )

    def forbidden_label_open(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("private label split opened before complete terminal roster")

    monkeypatch.setattr(
        "scripts.run_evidence_inference_fable_retrospective_v1.load_manifest_split",
        forbidden_label_open,
    )
    with pytest.raises(
        EvidenceInferenceFableScoringError,
        match="completed_runtime_required_for_full_population_score",
    ):
        main(["score", *shared])


def test_full_authorization_and_run_each_require_pilot_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "full-runtime"
    shared = _shared(workspace, mode="full_paired")
    assert main(["prepare", *shared]) == 0
    capsys.readouterr()
    plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        json.loads((ROOT / DEFAULT_FULL_PLAN_PATH).read_text(encoding="utf-8"))
    )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        json.loads((workspace / "00-prepared.json").read_text(encoding="utf-8"))
    )
    monkeypatch.setattr(harness, "_frozen_plan", lambda **_kwargs: (plan, prepared))
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="requires_pilot_workspace_and_certificate",
    ):
        main(["authorize", *shared, "--budget-usd-micros", "1"])
    assert not (workspace / "01-authorization.json").exists()

    gate_calls: list[str] = []

    def record_gate(**_kwargs: Any) -> object:
        gate_calls.append("gate")
        return object()

    monkeypatch.setattr(harness, "_require_full_gate", record_gate)
    assert main(["authorize", *shared, "--budget-usd-micros", "1"]) == 0
    authorization = json.loads(capsys.readouterr().out)
    assert gate_calls == ["gate"]

    def stop_before_environment(*_args: Any, **_kwargs: Any) -> None:
        assert gate_calls == ["gate", "gate"]
        raise RuntimeError("environment boundary reached after second gate")

    monkeypatch.setattr(harness, "load_live_environment", stop_before_environment)
    with pytest.raises(RuntimeError, match="after second gate"):
        main(
            [
                "run",
                *shared,
                "--live",
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-authorization-sha256",
                authorization["authorization_sha256"],
            ]
        )
    assert gate_calls == ["gate", "gate"]


def test_live_command_requires_explicit_flag_before_environment_access(tmp_path: Path) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    main(["prepare", *shared])
    main(["authorize", *shared, "--budget-usd-micros", "1"])
    with pytest.raises(EvidenceInferenceFableHarnessError, match="live_flag_required"):
        main(
            [
                "run",
                *shared,
                "--expected-plan-sha256",
                "0" * 64,
                "--expected-authorization-sha256",
                "0" * 64,
            ]
        )


def test_full_score_requires_reuse_guard_before_private_label_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "full-runtime"
    workspace.mkdir()
    plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        json.loads((ROOT / DEFAULT_FULL_PLAN_PATH).read_text(encoding="utf-8"))
    )
    _, prepared = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="full_paired",
    )
    monkeypatch.setattr(harness, "_frozen_plan", lambda **_kwargs: (plan, prepared))

    def stop_at_reuse_guard(**_kwargs: Any) -> None:
        raise RuntimeError("reuse scoring guard reached before labels")

    def forbidden_label_loader(**_kwargs: Any) -> Any:
        raise AssertionError("private label loader constructed before reuse guard")

    monkeypatch.setattr(harness, "_require_full_reuse_scoring_guard", stop_at_reuse_guard)
    monkeypatch.setattr(harness, "_private_label_loader", forbidden_label_loader)
    with pytest.raises(RuntimeError, match="reuse scoring guard reached before labels"):
        main(["score", *_shared(workspace, mode="full_paired")])


def test_token_count_cli_certifies_validates_and_authorizes_sequential_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "pilot-runtime"
    count_workspace = tmp_path / "pilot-token-counts"
    shared = _shared(workspace)
    main(["prepare", *shared])
    capsys.readouterr()
    plan, prepared_runtime = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    counter = _FakeTokenCounter()

    class FakeCounterFactory:
        @classmethod
        def from_anthropic_sdk(cls) -> _FakeTokenCounter:
            return counter

    environment_accesses: list[Path] = []

    def fake_environment(path: Path, *, live_enabled: bool) -> None:
        assert live_enabled is True
        environment_accesses.append(path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-placeholder")

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setattr(harness, "load_live_environment", fake_environment)
    monkeypatch.setattr(harness, "AnthropicFableTokenCounterV1", FakeCounterFactory)
    count_args = [
        *shared,
        "--token-count-workspace",
        str(count_workspace),
        "--expected-plan-sha256",
        plan.plan_sha256,
    ]
    assert main(["count-tokens", *count_args, "--live"]) == 0
    counted = json.loads(capsys.readouterr().out)
    assert counted["artifact"] == "EvidenceInferenceFableCountTerminalV1"
    assert counted["status"] == "completed_certified"
    assert counted["certified_total_liability_usd_micros"] < 12_000_000
    assert counter.calls == 14
    assert len(environment_accesses) == 1

    assert main(["validate-counts", *count_args]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated == counted

    def forbidden_environment(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("completed count replay must not access credentials")

    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    assert main(["count-tokens", *count_args, "--live"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed == counted
    assert counter.calls == 14

    certified_total = counted["certified_total_liability_usd_micros"]
    archived_count_terminal = json.loads((count_workspace / "terminal.json").read_text())
    certified_liabilities = archived_count_terminal[
        "certified_request_liabilities_usd_micros"
    ]
    pair_liabilities = [
        sum(
            certified_liabilities[surface.request_key]
            for surface in prepared_runtime.surfaces[index : index + 2]
        )
        for index in range(0, len(prepared_runtime.surfaces), 2)
    ]
    largest_pair_liability = max(pair_liabilities)
    assert largest_pair_liability < certified_total
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="budget_below_certified_largest_pair_liability",
    ):
        main(
            [
                "authorize",
                *shared,
                "--budget-usd-micros",
                str(largest_pair_liability - 1),
                "--token-count-workspace",
                str(count_workspace),
            ]
        )
    assert not (workspace / "01-authorization.json").exists()
    sequential_budget = largest_pair_liability + (
        certified_total - largest_pair_liability
    ) // 2
    assert largest_pair_liability <= sequential_budget < certified_total
    assert main(
        [
            "authorize",
            *shared,
            "--budget-usd-micros",
            str(sequential_budget),
            "--token-count-workspace",
            str(count_workspace),
        ]
    ) == 0
    authorization = json.loads(capsys.readouterr().out)
    assert authorization["liability_basis"] == "certified_provider_token_count"
    archived = json.loads((workspace / "01-authorization.json").read_text())
    assert archived["certified_count_terminal_sha256"] == counted["terminal_sha256"]
    assert archived["configured_total_budget_usd_micros"] == sequential_budget
    assert sum(archived["certified_request_liabilities_usd_micros"].values()) == (
        certified_total
    )


def test_token_count_requires_live_flag_before_environment_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    main(["prepare", *shared])
    plan, _ = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )

    def forbidden_environment(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("missing --live must stop before environment access")

    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="token_count_live_flag_required",
    ):
        main(
            [
                "count-tokens",
                *shared,
                "--token-count-workspace",
                str(tmp_path / "counts"),
                "--expected-plan-sha256",
                plan.plan_sha256,
            ]
        )


def test_token_count_rejects_prepared_identity_tamper_before_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    main(["prepare", *shared])
    plan, _ = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    prepared_path = workspace / "00-prepared.json"
    tampered = json.loads(prepared_path.read_text())
    tampered["retrospective_plan_sha256"] = "0" * 64
    tampered["prepared_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "prepared_sha256"}
    )
    atomic_write_json(prepared_path, tampered, force=True)

    def forbidden_environment(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("identity mismatch must stop before environment access")

    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="token_count_live_identity_anchor_mismatch",
    ):
        main(
            [
                "count-tokens",
                *shared,
                "--token-count-workspace",
                str(tmp_path / "counts"),
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--live",
            ]
        )
    assert not (tmp_path / "counts").exists()


def test_live_identity_rejects_coherent_prepared_tamper_before_environment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "pilot-runtime"
    shared = _shared(workspace)
    main(["prepare", *shared])
    capsys.readouterr()
    main(["authorize", *shared, "--budget-usd-micros", "1"])
    authorization = json.loads(capsys.readouterr().out)
    prepared_path = workspace / "00-prepared.json"
    tampered = json.loads(prepared_path.read_text(encoding="utf-8"))
    tampered["retrospective_plan_sha256"] = "0" * 64
    tampered["prepared_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "prepared_sha256"}
    )
    atomic_write_json(prepared_path, tampered, force=True)

    def forbidden_environment(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("identity mismatch must stop before environment access")

    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    plan, _ = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="live_identity_anchor_mismatch",
    ):
        main(
            [
                "run",
                *shared,
                "--live",
                "--expected-plan-sha256",
                plan.plan_sha256,
                "--expected-authorization-sha256",
                authorization["authorization_sha256"],
            ]
        )


@pytest.mark.parametrize(
    "relative",
    [
        Path("paper/fable-summary.json"),
        Path("Formatting_Instructions_For_NeurIPS_2026 (2)/fable-summary.json"),
        Path("artifacts/submission/fable-summary.json"),
        Path(".git/fable-summary.json"),
    ],
)
def test_public_target_rejects_manuscript_and_control_namespaces(relative: Path) -> None:
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="sensitive_namespace_output_forbidden",
    ):
        _safe_public_target(relative, ROOT)


def test_private_outputs_cannot_contaminate_runtime_ledger(tmp_path: Path) -> None:
    workspace = tmp_path / "runtime"
    (workspace / "private").mkdir(parents=True)
    (workspace / "receipts").mkdir()
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="runtime_private_namespace",
    ):
        _safe_private_target(Path("receipts/scored-report.json"), workspace)
    assert _safe_private_target(Path("private/scored-report.json"), workspace) == (
        workspace / "private" / "scored-report.json"
    )


def test_score_rejects_preexisting_private_namespace_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "runtime"
    escape = tmp_path / "escape"
    workspace.mkdir()
    escape.mkdir()
    (workspace / "private").symlink_to(escape, target_is_directory=True)
    shared = _shared(workspace)
    with pytest.raises(
        EvidenceInferenceFableHarnessError,
        match="runtime_private_namespace_unsafe",
    ):
        main(["score", *shared])
    assert list(escape.iterdir()) == []


def test_score_bundle_publishes_public_last_and_rolls_back_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "private" / "rows.json"
    certificate_path = tmp_path / "private" / "certificate.json"
    public_path = tmp_path / "public" / "summary.json"
    private_path.parent.mkdir()
    public_path.parent.mkdir()
    observed: list[Path] = []
    real_write = atomic_write_json

    def fail_public(path: Path, value: Any, **kwargs: Any) -> None:
        observed.append(path)
        if path == public_path:
            raise OSError("planted public write failure")
        real_write(path, value, **kwargs)

    monkeypatch.setattr(harness, "atomic_write_json", fail_public)
    with pytest.raises(OSError, match="planted public"):
        _atomic_publish_score_payloads(
            payloads=(
                (private_path, {"private": "fixture"}),
                (certificate_path, {"certificate": "fixture"}),
                (public_path, {"public": "fixture"}),
            )
        )
    assert observed == [private_path, certificate_path, public_path]
    assert not private_path.exists()
    assert not certificate_path.exists()
    assert not public_path.exists()


def test_union_score_lineage_is_persisted_before_public_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "private" / "rows.json"
    certificate_path = tmp_path / "private" / "certificate.json"
    lineage_path = tmp_path / "private" / "union-lineage.json"
    public_path = tmp_path / "public" / "summary.json"
    private_path.parent.mkdir()
    public_path.parent.mkdir()
    observed: list[Path] = []
    real_write = atomic_write_json

    def observe(path: Path, value: Any, **kwargs: Any) -> None:
        observed.append(path)
        real_write(path, value, **kwargs)

    monkeypatch.setattr(harness, "atomic_write_json", observe)
    _atomic_publish_score_payloads(
        payloads=(
            (private_path, {"private": "fixture"}),
            (certificate_path, {"certificate": "fixture"}),
            (lineage_path, {"union_lineage": "fixture"}),
            (public_path, {"public": "fixture"}),
        )
    )
    assert observed == [
        private_path,
        certificate_path,
        lineage_path,
        public_path,
    ]


def test_score_bundle_preserves_dependencies_if_public_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "private" / "rows.json"
    certificate_path = tmp_path / "private" / "certificate.json"
    public_path = tmp_path / "public" / "summary.json"
    private_path.parent.mkdir()
    public_path.parent.mkdir()
    real_write = atomic_write_json
    real_unlink = Path.unlink

    def fail_after_public_write(path: Path, value: Any, **kwargs: Any) -> None:
        real_write(path, value, **kwargs)
        if path == public_path:
            raise OSError("planted post-public fsync failure")

    def fail_public_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == public_path:
            raise OSError("planted public cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(harness, "atomic_write_json", fail_after_public_write)
    monkeypatch.setattr(Path, "unlink", fail_public_unlink)
    with pytest.raises(OSError, match="post-public"):
        _atomic_publish_score_payloads(
            payloads=(
                (private_path, {"private": "fixture"}),
                (certificate_path, {"certificate": "fixture"}),
                (public_path, {"public": "fixture"}),
            )
        )
    assert private_path.is_file()
    assert certificate_path.is_file()
    assert public_path.is_file()
