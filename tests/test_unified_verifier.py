from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from literature_multiverse.certificate import VerificationCertificate
from literature_multiverse.cli import main as cli_main
from literature_multiverse.models import FindingRow, make_finding_id
from literature_multiverse.verifier import (
    ClaimManifest,
    LegacyAdapterConfig,
    ScientificClaim,
    VerificationProtocol,
    adapt_legacy_findings,
    build_offline_fixture,
    load_corpus,
    run_verification,
)


def test_offline_fixture_runs_complete_fail_closed_path_and_freezes_hashes() -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert certificate.release_assessment.calibration.status == "not_run"
    assert len(certificate.counterfactual_reruns) == 3
    assert len(certificate.release_assessment.audit.ranking) == 3
    assert all(
        row["scenario"] == "leave_one_out_actual_synthesis_rerun"
        for row in certificate.counterfactual_reruns
    )
    assert certificate.evidence_graph_sha256 == (
        certificate.release_assessment.evidence_graph_sha256
    )

    tampered = certificate.model_dump(mode="json")
    tampered["corpus"]["metadata"]["purpose"] = "tampered"
    with pytest.raises(ValidationError, match="verification_certificate_hash_mismatch"):
        VerificationCertificate.model_validate(tampered)


def test_cli_fixture_writes_self_contained_json_and_html(tmp_path, capsys) -> None:
    output = tmp_path / "certificate"
    result = cli_main(
        [
            "verify",
            "--fixture",
            "--budget-minutes",
            "30",
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads((output / "verification-certificate.json").read_text())
    html = (output / "verification-certificate.html").read_text()
    certificate = VerificationCertificate.model_validate(payload)
    assert summary["certificate_sha256"] == certificate.certificate_sha256
    assert certificate.run_id in html
    assert certificate.certificate_sha256 in html
    assert "Complete normative JSON payload" in html
    assert "<script" not in html


def _legacy_finding() -> FindingRow:
    finding_id = make_finding_id(
        paper_id="doc:legacy-1",
        map_result_id="map-legacy-1",
        array_position=0,
        outcome_name="performance",
        timepoint_raw="4 weeks",
        dose_raw=None,
        effect_direction="increase",
    )
    return FindingRow(
        finding_id=finding_id,
        paper_id="doc:legacy-1",
        doc_id="legacy-1",
        map_result_id="map-legacy-1",
        array_position=0,
        prompt_version="legacy-test-v1",
        schema_version="1",
        cfghash="a" * 64,
        grounding_status="exact",
        evidence_section="Results",
        section_flagged=False,
        normalization_warnings=[],
        study_type="randomized trial",
        species="human",
        model=None,
        population_state="healthy",
        intervention="intervention",
        intervention_class=None,
        comparator="control",
        dose_raw=None,
        duration_raw="4 weeks",
        timing_context="post intervention",
        outcome_name="performance",
        outcome_family="performance",
        timepoint_raw="4 weeks",
        effect_direction="increase",
        effect_size_raw=None,
        p_value=None,
        significant=True,
        sample_size=20,
        evidence_quote="Performance increased after four weeks.",
        evidence_lines=["L1"],
        confidence=0.8,
        moderators={},
    )


def test_legacy_findings_are_connected_but_cannot_silently_release() -> None:
    graph, issues = adapt_legacy_findings(
        [_legacy_finding()], settings=LegacyAdapterConfig()
    )
    manifest = ClaimManifest(
        question_id="legacy-verification-test",
        population_id="legacy-test-population",
        domain="synthetic",
        claim=ScientificClaim(
            statement="The intervention increases performance.",
            direction="increase",
            outcome_name="performance",
        ),
        protocol=VerificationProtocol(corpus_cutoff="legacy-fixture-v1"),
    )
    _, fixture_corpus = build_offline_fixture()
    corpus = type(fixture_corpus)(
        corpus_id="legacy-fixture",
        source_label="embedded:legacy-fixture",
        source_format="legacy_findings_test",
        source_sha256="b" * 64,
        graph=graph,
        eligibility=(),
        adapter_issues=issues,
        metadata={},
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=10,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert certificate.release_assessment.evidence.classification == "not_evaluable"
    assert "adapter:legacy_effect_not_quantitatively_interpretable" in certificate.reasons
    assert "adapter:unresolved_cohort_identity" in certificate.reasons


def test_typed_graph_json_is_a_supported_corpus_input(tmp_path) -> None:
    manifest, fixture = build_offline_fixture()
    graph_path = tmp_path / "evidence_graph.json"
    graph_path.write_text(fixture.graph.model_dump_json(indent=2))

    loaded = load_corpus(graph_path, legacy_settings=manifest.legacy_adapter)

    assert loaded.source_format == "evidence_graph_json"
    assert loaded.graph == fixture.graph
    assert all(item.status == "included" for item in loaded.eligibility)
