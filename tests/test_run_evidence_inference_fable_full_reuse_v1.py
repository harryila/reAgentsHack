from __future__ import annotations

from pathlib import Path

import pytest
import scripts.run_evidence_inference_fable_full_reuse_v1 as harness
from tests.test_evidence_inference_fable_full_reuse_v1 import _context

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    freeze_evidence_inference_fable_full_reuse_plan_v1,
)


def test_live_cli_stops_before_environment_without_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_plan, prepared, authorization, sources = _context()
    adoption_plan = freeze_evidence_inference_fable_full_reuse_plan_v1(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    monkeypatch.setattr(
        harness,
        "_context",
        lambda _args: (
            Path("unused"),
            full_plan,
            prepared,
            authorization,
            sources,
            adoption_plan,
        ),
    )

    def forbidden_environment(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("environment opened before explicit live flag")

    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    with pytest.raises(
        harness.EvidenceInferenceFableFullReuseHarnessError,
        match="live_flag_required",
    ):
        harness.main(
            [
                "run",
                "--expected-full-plan-sha256",
                full_plan.plan_sha256,
                "--expected-authorization-sha256",
                authorization.authorization_sha256,
                "--expected-reuse-plan-sha256",
                adoption_plan.plan_sha256,
            ]
        )
