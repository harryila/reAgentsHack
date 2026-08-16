"""The HTTP API serves the frozen bundle faithfully and validates real configs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

BUNDLE = Path("artifacts/antiox-training/demo")

pytestmark = pytest.mark.skipif(
    not BUNDLE.is_dir(), reason="frozen demo bundle not present"
)


@pytest.fixture(scope="module")
def client():
    from app.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_service_info_lists_question(client) -> None:
    payload = client.get("/").json()
    assert payload["service"] == "papertrail"
    assert "antiox-training" in payload["questions"]
    assert payload["release"]["disposition"] == "v1_frozen"


def test_claims_are_claim_grade_with_full_provenance(client) -> None:
    payload = client.get("/questions/antiox-training/claims").json()
    # 19 claim-grade rows across every outcome family; 11 in the primary release family.
    assert payload["count"] == 19
    primary = client.get(
        "/questions/antiox-training/claims", params={"family": "functional_adaptation"}
    ).json()
    assert primary["count"] == 11
    for row in payload["claims"]:
        assert row["claim_grade"] is True
        prov = row["provenance"]
        assert prov["grounding_status"] == "exact"
        assert prov["evidence_quote"]
        assert prov["evidence_lines"]
        assert row["verification"] is not None


def test_single_claim_resolves_with_quote_and_lines(client) -> None:
    listing = client.get("/questions/antiox-training/claims").json()
    finding_id = listing["claims"][0]["finding_id"]
    row = client.get(f"/questions/antiox-training/claims/{finding_id}").json()
    assert row["finding_id"] == finding_id
    assert row["provenance"]["evidence_quote"]


def test_direction_filter(client) -> None:
    params = {"family": "functional_adaptation"}
    increases = client.get(
        "/questions/antiox-training/claims", params={**params, "direction": "increase"}
    ).json()
    nulls = client.get(
        "/questions/antiox-training/claims", params={**params, "direction": "no_effect"}
    ).json()
    assert increases["count"] == 5
    assert nulls["count"] == 6


def test_gates_and_funnel_and_multiverse(client) -> None:
    gates = client.get("/questions/antiox-training/gates").json()
    assert gates["trust_passed"] is True
    funnel = client.get("/questions/antiox-training/funnel").json()
    assert funnel["funnel"]["primary_grounded_findings"] == 11
    multiverse = client.get("/questions/antiox-training/multiverse").json()
    assert multiverse["claim_grade_by_family"]["functional_adaptation"]["increase"] == 5


def test_unknown_question_404(client) -> None:
    assert client.get("/questions/nope").status_code == 404


def test_validate_accepts_the_real_config_and_rejects_garbage(client) -> None:
    real = Path("configs/questions/antiox-training.yaml").read_text()
    ok = client.post("/questions/validate", json={"config_yaml": real}).json()
    assert ok["valid"] is True
    assert ok["question_id"] == "antiox-training"
    assert ok["queries"] >= 14

    bad = client.post(
        "/questions/validate", json={"config_yaml": "question_id: broken"}
    ).json()
    assert bad["valid"] is False
    assert bad["errors"]
