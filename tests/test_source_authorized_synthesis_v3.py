from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.run_source_authorized_synthesis_v4 import main as v4_cli_main
from tests.test_cohort_reconciliation import _cohort_payload
from tests.test_synthesis_unit_authorization import (
    _assertion,
    _citation,
    _rehash_assertion,
)

from literature_multiverse.cohort_reconciliation import reconcile_native_cohorts
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    freeze_native_publication_extraction,
)
from literature_multiverse.native_grounding import (
    freeze_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.source_authorized_synthesis_v4 import (
    SourceAuthorizedSynthesisReportV4,
    SourceAuthorizedSynthesisV4Error,
    compute_source_authorized_synthesis_v4_fingerprint,
    reverify_source_authorized_synthesis_v4_report,
    run_source_authorized_synthesis_v4,
)
from literature_multiverse.synthesis_unit_authorization import (
    SynthesisUnitAuthorizationReceiptV1,
    authorize_synthesis_unit,
    reverify_synthesis_unit_authorization,
)
from literature_multiverse.typed_extraction import (
    SourceDocumentArtifact,
    assemble_typed_evidence_corpus,
)


def _case(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "doc-1": {
                    "L1": {
                        "section": "Results",
                        "text": "Registry NCT-ONE. Effect for cohort-a was 0.2.",
                    }
                },
                "doc-2": {
                    "L1": {
                        "section": "Results",
                        "text": "Registry NCT-TWO. Effect for cohort-b was 0.3.",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fragments, grounding_receipts = [], []
    for number, key, identifier, estimate in (
        (1, "cohort-a", "NCT-ONE", 0.2),
        (2, "cohort-b", "NCT-TWO", 0.3),
    ):
        locator = f"json:sources.json#/doc-{number}"
        cohort = _cohort_payload(key, registry_ids=[identifier], dataset_ids=[], estimate=estimate)
        cohort["findings"][0]["evidence"].update(
            {
                "source_locator": locator,
                "quote": f"Effect for {key} was {estimate}.",
                "line_ids": ["L1"],
            }
        )
        extraction = NativePublicationExtraction.model_validate(
            {
                "status": "estimable",
                "studies": [
                    {
                        "key": "study",
                        "source_label": f"Report {number}",
                        "registration_ids": [identifier],
                        "cohorts": [cohort],
                    }
                ],
            }
        )
        publication = PublicationIdentity(
            publication_id=f"publication-{number}",
            paper_id=f"paper-{number}",
            doc_id=f"doc-{number}",
        )
        source = SourceDocumentArtifact(
            artifact_path="sources.json",
            sha256=sha256_file(path),
            media_type="application/json",
            source_locator=locator,
        )
        grounding = verify_native_publication_grounding(
            repository_root=tmp_path, source_document=source, extraction=extraction
        )
        grounding_receipts.append(grounding)
        fragments.append(
            freeze_native_publication_extraction(
                payload=extraction,
                question_id="reconciliation-question",
                publication=publication,
                pipeline_fingerprint_sha256="a" * 64,
                source_document=source,
                grounding_receipt_sha256=grounding.receipt_sha256,
            )
        )
    corpus = assemble_typed_evidence_corpus(fragments)
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus, grounding_receipts=grounding_receipts
    )
    estimates = sorted(row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates)
    assertion = _assertion(
        "independent_cohorts",
        sorted(row.canonical_id for row in reconciliation.cohort_groups),
        [
            _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    positive = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimates,
        assertions=[assertion],
        repository_root=tmp_path,
    )
    return corpus, reconciliation, package, estimates, positive


def test_reverify_and_v3_run_existing_cohort_aware_synthesis(tmp_path) -> None:
    corpus, reconciliation, package, estimates, positive = _case(tmp_path)
    assert (
        reverify_synthesis_unit_authorization(
            corpus=corpus,
            reconciliation=reconciliation,
            receipt=positive,
            repository_root=tmp_path,
        )
        == positive
    )
    report = run_source_authorized_synthesis_v4(
        corpus=corpus,
        grounding_package=package,
        reconciliation=reconciliation,
        receipts=[positive],
        requested_estimate_ids=estimates,
        authorized_receipt_sha256s=[positive.receipt_sha256],
        repository_root=tmp_path,
    )
    assert report.units[0].status == "synthesized"
    assert report.units[0].synthesis["quantitative"]["n_cohorts"] == 2
    assert report.supports_accuracy_claim is False
    assert report.supports_release_claim is False
    assert report.verifier_replay_status.startswith("not_constructed")
    assert report.publication_source_content_visible is True
    assert report.benchmark_reference_labels_accessed is False
    assert report.benchmark_review_verdicts_accessed is False
    assert (
        reverify_source_authorized_synthesis_v4_report(
            report=report,
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=[positive],
            repository_root=tmp_path,
        )
        == report
    )

    tampered = deepcopy(report.model_dump(mode="json"))
    tampered["runtime"] = {"python": "forged"}
    tampered.pop("report_sha256")
    tampered["report_sha256"] = hash_canonical(tampered)
    with pytest.raises(ValueError, match="runtime_invalid"):
        SourceAuthorizedSynthesisReportV4.model_validate(tampered)


def test_v4_never_calls_legacy_runner_and_rejects_v1(tmp_path, monkeypatch) -> None:
    corpus, reconciliation, package, estimates, positive = _case(tmp_path)

    def forbidden_legacy_call(**_kwargs):
        raise AssertionError("v4_called_legacy_v3_runner")

    monkeypatch.setattr(
        "literature_multiverse.source_authorized_synthesis_v3.run_source_authorized_synthesis_v3",
        forbidden_legacy_call,
    )
    report = run_source_authorized_synthesis_v4(
        corpus=corpus,
        grounding_package=package,
        reconciliation=reconciliation,
        receipts=[positive],
        requested_estimate_ids=estimates,
        authorized_receipt_sha256s=[positive.receipt_sha256],
        repository_root=tmp_path,
    )
    assert report.report_version == "source-authorized-synthesis-report-v4"

    legacy_payload = {
        "receipt_version": "source-backed-synthesis-authorization-v1",
        "input_corpus_sha256": positive.input_corpus_sha256,
        "reconciliation_receipt_sha256": positive.reconciliation_receipt_sha256,
        "reconciled_graph_sha256": positive.reconciled_graph_sha256,
        "estimate_ids": positive.estimate_ids,
        "canonical_cohort_ids": positive.canonical_cohort_ids,
        "assertions": positive.assertions,
        "unresolved_overlap_pairs": positive.unresolved_overlap_pairs,
        "reference_labels_accessed": False,
        "review_conclusions_accessed": False,
        "authorizes_synthesis_input": positive.authorizes_synthesis_input,
        "authorization_basis": positive.authorization_basis,
    }
    legacy = SynthesisUnitAuthorizationReceiptV1.model_validate(
        {**legacy_payload, "receipt_sha256": hash_canonical(legacy_payload)}
    )
    with pytest.raises(ValueError, match="source-backed-synthesis-authorization-v2"):
        run_source_authorized_synthesis_v4(
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=[legacy],  # type: ignore[list-item]
            requested_estimate_ids=estimates,
            authorized_receipt_sha256s=[legacy.receipt_sha256],
            repository_root=tmp_path,
        )


def test_v4_source_has_no_v3_dependency() -> None:
    source = (
        Path(__file__).parents[1] / "src/literature_multiverse/source_authorized_synthesis_v4.py"
    ).read_text(encoding="utf-8")
    assert "source_authorized_synthesis_v3" not in source
    assert "SynthesisV3" not in source
    assert "UnitV3" not in source


def test_v4_fingerprint_binds_transitive_closure_and_byte_drift(tmp_path) -> None:
    repository_root = Path(__file__).parents[1]
    original = compute_source_authorized_synthesis_v4_fingerprint(root=repository_root)
    paths = {item.path for component in original.components for item in component.files}
    assert "src/literature_multiverse/models.py" in paths
    assert "src/literature_multiverse/budgeted_verification.py" in paths

    clone = tmp_path / "fingerprint-root"
    for relative in paths:
        destination = clone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative, destination)
    copied = compute_source_authorized_synthesis_v4_fingerprint(root=clone)
    assert copied.pipeline_sha256 == original.pipeline_sha256

    transitive = clone / "src/literature_multiverse/budgeted_verification.py"
    transitive.write_text(
        transitive.read_text(encoding="utf-8") + "\n# dependency-drift\n",
        encoding="utf-8",
    )
    drifted = compute_source_authorized_synthesis_v4_fingerprint(root=clone)
    assert drifted.pipeline_sha256 != copied.pipeline_sha256

    (clone / "src/literature_multiverse/models.py").unlink()
    with pytest.raises(SourceAuthorizedSynthesisV4Error, match="local_dependency_missing"):
        compute_source_authorized_synthesis_v4_fingerprint(root=clone)


def test_v3_rejects_overlap_unknown_coverage_and_abstention_in_authorized_set(tmp_path) -> None:
    corpus, reconciliation, package, estimates, positive = _case(tmp_path)
    alternate_assertion = _rehash_assertion(
        positive.assertions[0], rationale="A second complete source review rationale."
    )
    overlapping = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimates,
        assertions=[alternate_assertion],
        repository_root=tmp_path,
    )
    with pytest.raises(SourceAuthorizedSynthesisV4Error, match="overlapping"):
        run_source_authorized_synthesis_v4(
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=sorted([positive, overlapping], key=lambda row: row.receipt_sha256),
            requested_estimate_ids=estimates,
            authorized_receipt_sha256s=sorted(
                [positive.receipt_sha256, overlapping.receipt_sha256]
            ),
            repository_root=tmp_path,
        )

    abstaining = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimates,
        assertions=[],
        repository_root=tmp_path,
    )
    with pytest.raises(SourceAuthorizedSynthesisV4Error, match="disagrees"):
        run_source_authorized_synthesis_v4(
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=[abstaining],
            requested_estimate_ids=estimates,
            authorized_receipt_sha256s=[abstaining.receipt_sha256],
            repository_root=tmp_path,
        )
    with pytest.raises(SourceAuthorizedSynthesisV4Error, match="coverage"):
        run_source_authorized_synthesis_v4(
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=[positive],
            requested_estimate_ids=[*estimates, "unknown-estimate"],
            authorized_receipt_sha256s=[positive.receipt_sha256],
            repository_root=tmp_path,
        )


def test_v3_rejects_source_drift_before_synthesis(tmp_path) -> None:
    corpus, reconciliation, package, estimates, positive = _case(tmp_path)
    path = tmp_path / corpus.fragments[0].source_document.artifact_path
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native_grounding_receipt_replay_mismatch"):
        run_source_authorized_synthesis_v4(
            corpus=corpus,
            grounding_package=package,
            reconciliation=reconciliation,
            receipts=[positive],
            requested_estimate_ids=estimates,
            authorized_receipt_sha256s=[positive.receipt_sha256],
            repository_root=tmp_path,
        )


def test_v3_cli_replays_exact_bound_inputs(tmp_path, capsys) -> None:
    corpus, reconciliation, package, estimates, positive = _case(tmp_path)
    paths = {
        "corpus": tmp_path / "corpus.json",
        "reconciliation": tmp_path / "reconciliation.json",
        "package": tmp_path / "grounding-package.json",
        "authorization": tmp_path / "authorization.json",
        "request": tmp_path / "request.json",
        "output": tmp_path / "report.json",
    }
    for key, value in (
        ("corpus", corpus),
        ("reconciliation", reconciliation),
        ("package", package),
        ("authorization", positive),
    ):
        paths[key].write_text(value.model_dump_json(), encoding="utf-8")
    paths["request"].write_text(
        json.dumps(
            {
                "request_version": "source-authorized-synthesis-v4-request-v1",
                "requested_estimate_ids": estimates,
                "authorized_receipt_sha256s": [positive.receipt_sha256],
            }
        ),
        encoding="utf-8",
    )
    assert (
        v4_cli_main(
            [
                "--corpus",
                str(paths["corpus"]),
                "--grounding-package",
                str(paths["package"]),
                "--reconciliation",
                str(paths["reconciliation"]),
                "--authorization",
                str(paths["authorization"]),
                "--request",
                str(paths["request"]),
                "--repository-root",
                str(tmp_path),
                "--output",
                str(paths["output"]),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    report = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert result["report_sha256"] == report["report_sha256"]
    assert result["synthesized_units"] == 1
