from __future__ import annotations

import json
from pathlib import Path

from literature_multiverse.cli import main as cli_main
from literature_multiverse.postlive_recovery_v4_public_verify_v1 import (
    REQUIRED_ADAPTER_ISSUE_CODES,
    REQUIRED_CERTIFICATE_REASONS,
    freeze_postlive_recovery_v4_public_verify_inputs_v1,
    validate_postlive_recovery_v4_public_verify_output_v1,
    write_postlive_recovery_v4_public_verify_inputs_v1,
)
from literature_multiverse.verifier import load_claim_manifest, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_inputs_bind_actual_joined_claim_and_blocking_issues() -> None:
    preparation, claim, bundle = freeze_postlive_recovery_v4_public_verify_inputs_v1(
        repository_root=ROOT
    )
    graph = bundle["graph"]
    assert claim["claim"]["direction"] == "increase"
    assert claim["claim"]["outcome_name"] == "spleen response"
    assert claim["claim"]["contrast_id"] == graph["contrasts"][0]["contrast_id"]
    assert {arm["label"] for arm in graph["arms"]} == {"500-mg", "placebo group"}
    assert graph["outcome_estimates"][0]["timepoint"]["raw_label"] == "week 24"
    assert tuple(item["code"] for item in bundle["adapter_issues"]) == (
        REQUIRED_ADAPTER_ISSUE_CODES
    )
    assert all(item["severity"] == "blocking" for item in bundle["adapter_issues"])
    assert bundle["metadata"]["source_lineage_sha256"] == (
        preparation.source_lineage_sha256
    )
    assert preparation.source_join_external_replay_performed
    assert preparation.source_posthoc_external_replay_performed
    assert "row17-candidate3" in claim["protocol"]["corpus_cutoff"]
    assert "source-id20" in claim["protocol"]["corpus_cutoff"]
    assert bundle["metadata"]["source_lineage"]["recovery_witness_identity"] == (
        "metasyn-row17-candidate3"
    )
    assert bundle["metadata"]["source_lineage"]["source_record_identity"] == (
        "metasyn-source-id20"
    )
    assert not preparation.claim_release_authority


def test_public_loader_preserves_every_required_blocker(tmp_path: Path) -> None:
    workspace = tmp_path / "inputs"
    write_postlive_recovery_v4_public_verify_inputs_v1(
        repository_root=ROOT, workspace=workspace
    )
    manifest = load_claim_manifest(workspace / "claim.json")
    corpus = load_corpus(
        workspace / "corpus-bundle.json",
        legacy_settings=manifest.legacy_adapter,
        repository_root=ROOT,
    )
    issues = {item.code: item.severity.value for item in corpus.adapter_issues}
    assert all(issues[code] == "blocking" for code in REQUIRED_ADAPTER_ISSUE_CODES)
    assert issues["unverified_source_provenance"] == "blocking"
    assert corpus.metadata["source_lineage_sha256"]
    assert not corpus.provenance_release_eligible()


def test_public_cli_abstains_and_external_replay_validates(
    tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "inputs"
    output = tmp_path / "output"
    write_postlive_recovery_v4_public_verify_inputs_v1(
        repository_root=ROOT, workspace=workspace
    )
    assert (
        cli_main(
            [
                "verify",
                "--claim",
                str(workspace / "claim.json"),
                "--corpus",
                str(workspace / "corpus-bundle.json"),
                "--budget-minutes",
                "5",
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    report = validate_postlive_recovery_v4_public_verify_output_v1(
        repository_root=ROOT,
        workspace=workspace,
        output_dir=output,
        write_report=False,
    )
    certificate = json.loads((output / "verification-certificate.json").read_text())
    assert summary["status"] == "abstained"
    assert report.release_status == "abstained"
    assert report.selected_audit_item_id is not None
    assert set(REQUIRED_CERTIFICATE_REASONS).issubset(certificate["reasons"])
    assert certificate["status"] == "abstained"
