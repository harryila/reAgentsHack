from __future__ import annotations

import json
from pathlib import Path

import pytest

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFablePairedRuntimeError,
    freeze_evidence_inference_fable_budget_authorization_v1,
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
)
from literature_multiverse.evidence_inference_fable_token_count_v1 import (
    EvidenceInferenceFableTokenCountError,
    execute_evidence_inference_fable_token_count_v1,
    freeze_evidence_inference_fable_count_authorization_v1,
    validate_evidence_inference_fable_token_count_v1,
)
from literature_multiverse.lineage import hash_canonical

ROOT = Path(__file__).resolve().parents[1]


class FakeCounter:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.calls = 0

    def count_tokens(self, wire_kwargs: object) -> int:
        self.calls += 1
        if self.raises:
            raise RuntimeError("offline fake")
        return 20_000


def _prepared():
    return reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT, mode="pilot30_paired"
    )[1]


def test_certified_pilot_liability_is_under_twelve_dollars_and_replays(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    auth = freeze_evidence_inference_fable_count_authorization_v1(prepared)
    counter = FakeCounter()
    terminal = execute_evidence_inference_fable_token_count_v1(
        workspace=tmp_path / "counts",
        prepared=prepared,
        authorization=auth,
        counter=counter,
    )
    assert terminal.status == "completed_certified"
    assert terminal.certified_total_liability_usd_micros < 12_000_000
    execution_auth = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=12_000_000,
        certified_count_terminal=terminal.model_dump(mode="json"),
    )
    assert execution_auth.liability_basis == "certified_provider_token_count"
    assert sum(execution_auth.certified_request_liabilities_usd_micros.values()) < 12_000_000
    assert counter.calls == 14
    assert terminal == validate_evidence_inference_fable_token_count_v1(
        workspace=tmp_path / "counts", prepared=prepared
    )
    tampered = terminal.model_dump(mode="json")
    first_key = prepared.surfaces[0].request_key
    tampered["certified_request_liabilities_usd_micros"][first_key] += 1
    with pytest.raises(
        EvidenceInferenceFablePairedRuntimeError,
        match="authorization_count_terminal_invalid",
    ):
        freeze_evidence_inference_fable_budget_authorization_v1(
            prepared=prepared,
            configured_total_budget_usd_micros=12_000_000,
            certified_count_terminal=tampered,
        )


def test_count_exception_is_poisoned_and_never_retried(tmp_path: Path) -> None:
    prepared = _prepared()
    auth = freeze_evidence_inference_fable_count_authorization_v1(prepared)
    counter = FakeCounter(raises=True)
    first = execute_evidence_inference_fable_token_count_v1(
        workspace=tmp_path / "counts",
        prepared=prepared,
        authorization=auth,
        counter=counter,
    )
    second = execute_evidence_inference_fable_token_count_v1(
        workspace=tmp_path / "counts",
        prepared=prepared,
        authorization=auth,
        counter=counter,
    )
    assert first == second
    assert first.status == "terminal_ambiguous_count_poison"
    assert counter.calls == 1


def test_resume_replays_complete_receipt_prefix_before_next_count_call(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    auth = freeze_evidence_inference_fable_count_authorization_v1(prepared)
    workspace = tmp_path / "counts"
    execute_evidence_inference_fable_token_count_v1(
        workspace=workspace,
        prepared=prepared,
        authorization=auth,
        counter=FakeCounter(),
    )
    (workspace / "terminal.json").unlink()
    first_key = prepared.surfaces[0].request_key
    for directory_name in ("intents", "receipts"):
        for path in (workspace / directory_name).glob("*.json"):
            if path.stem != first_key:
                path.unlink()
    receipt_path = workspace / "receipts" / f"{first_key}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["tightened_request_liability_usd_micros"] -= 1
    receipt["receipt_sha256"] = hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt))
    counter = FakeCounter()
    with pytest.raises(
        EvidenceInferenceFableTokenCountError,
        match="receipt_binding_mismatch",
    ):
        execute_evidence_inference_fable_token_count_v1(
            workspace=workspace,
            prepared=prepared,
            authorization=auth,
            counter=counter,
        )
    assert counter.calls == 0


def test_resume_malformed_json_fails_with_normalized_artifact_error(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    auth = freeze_evidence_inference_fable_count_authorization_v1(prepared)
    workspace = tmp_path / "counts"
    execute_evidence_inference_fable_token_count_v1(
        workspace=workspace,
        prepared=prepared,
        authorization=auth,
        counter=FakeCounter(),
    )
    (workspace / "terminal.json").unlink()
    first_key = prepared.surfaces[0].request_key
    for directory_name in ("intents", "receipts"):
        for path in (workspace / directory_name).glob("*.json"):
            if path.stem != first_key:
                path.unlink()
    (workspace / "receipts" / f"{first_key}.json").write_text("{")
    counter = FakeCounter()
    with pytest.raises(
        EvidenceInferenceFableTokenCountError,
        match="artifact_invalid",
    ):
        execute_evidence_inference_fable_token_count_v1(
            workspace=workspace,
            prepared=prepared,
            authorization=auth,
            counter=counter,
        )
    assert counter.calls == 0
