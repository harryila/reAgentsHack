from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.prepare_evidence_inference_fable_retrospective_v1 import main

from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectiveError,
    validate_evidence_inference_fable_retrospective_plan_v1,
    write_evidence_inference_fable_retrospective_plan_v1,
)

ROOT = Path(__file__).resolve().parents[1]


def _workspace_test_path(tmp_path: Path, name: str) -> Path:
    return Path("artifacts/diagnostics/evidence-inference") / (
        f".test-{tmp_path.name}-{name}.json"
    )


@pytest.mark.parametrize(
    "mode",
    ["pilot30_paired", "pilot30_recovery_v2_paired", "full_paired"],
)
def test_write_and_external_replay_are_exact(tmp_path: Path, mode: str) -> None:
    relative = _workspace_test_path(tmp_path, mode)
    try:
        plan = write_evidence_inference_fable_retrospective_plan_v1(
            repository_root=ROOT,
            mode=mode,  # type: ignore[arg-type]
            output_path=relative,
        )
        raw = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        replayed = validate_evidence_inference_fable_retrospective_plan_v1(
            repository_root=ROOT,
            plan=raw,
        )
        assert replayed == plan
    finally:
        (ROOT / relative).unlink(missing_ok=True)


def test_cli_build_validate_and_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    relative = _workspace_test_path(tmp_path, "pilot")
    shared = [
        "--repository-root",
        str(ROOT),
        "--mode",
        "pilot30_paired",
        "--plan",
        str(relative),
    ]
    try:
        assert main(["build", *shared]) == 0
        built = json.loads(capsys.readouterr().out)
        assert built["provider_calls_made"] == 0
        assert built["confirmatory_claim_authority"] is False
        assert main(["validate", *shared]) == 0
        validated = json.loads(capsys.readouterr().out)
        assert validated["plan_sha256"] == built["plan_sha256"]
        assert validated["validation"] == "external_label_safe_input_replay_exact"
        assert main(["status", *shared]) == 0
        status = json.loads(capsys.readouterr().out)
        assert status["validation"] == "serialized_contract_and_self_hash_only"
    finally:
        (ROOT / relative).unlink(missing_ok=True)


def test_replay_rejects_semantic_rehash(tmp_path: Path) -> None:
    relative = _workspace_test_path(tmp_path, "pilot-rehash")
    try:
        plan = write_evidence_inference_fable_retrospective_plan_v1(
            repository_root=ROOT,
            mode="pilot30_paired",
            output_path=relative,
        )
        raw = plan.model_dump(mode="json")
        raw["claim_release_authority"] = True
        with pytest.raises(EvidenceInferenceFableRetrospectiveError):
            validate_evidence_inference_fable_retrospective_plan_v1(
                repository_root=ROOT,
                plan=raw,
            )
    finally:
        (ROOT / relative).unlink(missing_ok=True)
