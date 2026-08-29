"""Fail-closed contracts for the integrated local benchmark runner."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_local_benchmarks as local_benchmark_runner
from scripts.run_local_benchmarks import (
    _SOURCE_CODE_PATHS,
    BenchmarkSuiteError,
    _source_code_hashes,
    main,
    offline_network_guard,
    preflight_run_paths,
    validate_study_artifacts,
    verify_pinned_files,
    write_self_hashed_report,
)

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file


def test_public_local_suite_git_index_contains_only_aggregate_report() -> None:
    """Prevent identifier-bearing staged outputs from remaining in the Git index."""
    indexed = subprocess.run(
        ["git", "ls-files", "artifacts/benchmarks/local-suite-v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert indexed == [
        "artifacts/benchmarks/local-suite-v1/benchmark-report.json"
    ], "local_suite_git_index_contains_identifier_bearing_artifacts"


def _self_hashed(payload: dict[str, object], *, field: str) -> dict[str, object]:
    return {**payload, field: hash_canonical(payload)}


def test_contract_only_opens_no_payloads_or_runs_studies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--suite",
            Path("configs/benchmarks/local-suite-v1.json").resolve().as_posix(),
            "--contract-only",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "contract_valid"
    assert output["local_payload_files_opened"] is False
    assert output["studies_executed"] is False
    assert set(output["source_code_sha256s"]) == set(_SOURCE_CODE_PATHS)


def test_runner_source_lineage_is_complete_and_current(repo_root: Path) -> None:
    source_map = _source_code_hashes(repository_root=repo_root)

    assert list(source_map) == list(_SOURCE_CODE_PATHS)
    assert source_map == {
        relative: sha256_file(repo_root / relative) for relative in _SOURCE_CODE_PATHS
    }


def test_public_local_suite_documented_hashes_match_current_aggregate(
    repo_root: Path,
) -> None:
    """Keep the reproducibility guide bound to the exact checked-in aggregates."""

    report_path = repo_root / "artifacts/benchmarks/local-suite-v1/benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    documentation = (repo_root / "docs/local-benchmark-suite.md").read_text(
        encoding="utf-8"
    )
    integrity = report["artifacts"]["integrity"]
    expected = {
        sha256_file(report_path),
        report["report_payload_sha256"],
        report["reproducibility"]["scientific_payload_sha256"],
        integrity["metasyn_retrieval_summary"]["file_sha256"],
        integrity["metasyn_screening_summary"]["file_sha256"],
        integrity["metasyn_screening_summary"]["payload_sha256"],
    }

    assert all(digest in documentation for digest in expected)


def test_offline_guard_rejects_socket_connections() -> None:
    with offline_network_guard():
        with pytest.raises(
            BenchmarkSuiteError, match="network_call_attempted_during_offline_suite"
        ):
            socket.create_connection(("127.0.0.1", 9))
        guarded_socket = socket.socket()
        try:
            with pytest.raises(
                BenchmarkSuiteError,
                match="network_call_attempted_during_offline_suite",
            ):
                guarded_socket.connect(("127.0.0.1", 9))
        finally:
            guarded_socket.close()


def test_preflight_requires_force_for_any_prior_artifact_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    retrieval_work = root / "private" / "retrieval"
    screening_work = root / "private" / "screening"
    retrieval_summary = root / "public" / "retrieval.json"
    screening_summary = root / "public" / "screening.json"
    report = root / "reports" / "report.json"
    retrieval_work.mkdir(parents=True)

    with pytest.raises(
        BenchmarkSuiteError, match="prior_run_artifacts_require_explicit_force"
    ):
        preflight_run_paths(
            repository_root=root,
            report_path=report,
            retrieval_work_dir=retrieval_work,
            screening_work_dir=screening_work,
            retrieval_summary_path=retrieval_summary,
            screening_summary_path=screening_summary,
            force=False,
        )

    preflight_run_paths(
        repository_root=root,
        report_path=report,
        retrieval_work_dir=retrieval_work,
        screening_work_dir=screening_work,
        retrieval_summary_path=retrieval_summary,
        screening_summary_path=screening_summary,
        force=True,
    )
    with pytest.raises(
        BenchmarkSuiteError, match="retrieval_and_screening_work_dirs_overlap"
    ):
        preflight_run_paths(
            repository_root=root,
            report_path=report,
            retrieval_work_dir=retrieval_work,
            screening_work_dir=retrieval_work / "nested",
            retrieval_summary_path=retrieval_summary,
            screening_summary_path=screening_summary,
            force=True,
        )


def test_pinned_file_symlink_cannot_escape_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = outside / "payload.bin"
    payload.write_bytes(b"outside")
    (repository / "link").symlink_to(outside, target_is_directory=True)
    suite = {
        "pinned_files": [
            {
                "path": "link/payload.bin",
                "sha256": sha256_file(payload),
                "required_for": "fixture",
            }
        ]
    }

    with pytest.raises(
        BenchmarkSuiteError, match="pinned_file_resolves_outside_repository"
    ):
        verify_pinned_files(suite, repository_root=repository)


def test_report_is_canonically_self_hashed(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = write_self_hashed_report(path, {"status": "complete", "results": {}})

    assert report["report_payload_sha256"] == hash_canonical(
        {key: value for key, value in report.items() if key != "report_payload_sha256"}
    )
    assert json.loads(path.read_text(encoding="utf-8")) == report


def _configure_blocked_local_suite_run(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], Path, list[dict[str, str]]]:
    root = tmp_path / "repository"
    root.mkdir()
    suite_path = root / "suite.json"
    suite_path.write_text("{}\n", encoding="utf-8")
    report_path = root / "public" / "benchmark-report.json"
    blockers = [
        {
            "code": "pinned_file_missing",
            "path": "private/missing-input.json",
            "action": "restore the pinned local input",
        }
    ]
    suite = {
        "corpora": {
            "metasyn": {"corpus_manifest": "private/corpus-manifest.json"}
        }
    }
    monkeypatch.setattr(
        local_benchmark_runner,
        "PATHS",
        SimpleNamespace(root=root),
    )
    monkeypatch.setattr(local_benchmark_runner, "load_suite", lambda _path: suite)
    monkeypatch.setattr(
        local_benchmark_runner,
        "_source_code_hashes",
        lambda *, repository_root: {"runner.py": "0" * 64},
    )
    monkeypatch.setattr(
        local_benchmark_runner,
        "verify_pinned_files",
        lambda _suite, *, repository_root: ([], blockers),
    )
    monkeypatch.setattr(
        local_benchmark_runner,
        "license_audit",
        lambda *, repository_root: {"repository_public_release_ready": True},
    )
    argv = [
        "--suite",
        suite_path.as_posix(),
        "--output-dir",
        report_path.parent.as_posix(),
    ]
    return argv, report_path, blockers


def test_blocked_input_does_not_overwrite_existing_report_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, report_path, _blockers = _configure_blocked_local_suite_run(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    report_path.parent.mkdir(parents=True)
    prior_report = b'{"status":"complete","sentinel":"preserve-me"}\n'
    report_path.write_bytes(prior_report)

    with pytest.raises(
        BenchmarkSuiteError, match="prior_run_artifacts_require_explicit_force"
    ):
        main(argv)

    assert report_path.read_bytes() == prior_report


def test_blocked_input_overwrites_existing_report_only_with_explicit_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, report_path, blockers = _configure_blocked_local_suite_run(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        '{"status":"complete","sentinel":"replace-me"}\n',
        encoding="utf-8",
    )

    assert main([*argv, "--force"]) == 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["blockers"] == blockers
    assert report["report_payload_sha256"] == hash_canonical(
        {
            key: value
            for key, value in report.items()
            if key != "report_payload_sha256"
        }
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "report": report_path.as_posix(),
    }


def _artifact_fixture(tmp_path: Path) -> dict[str, object]:
    suite = json.loads(
        Path("configs/benchmarks/local-suite-v1.json").read_text(encoding="utf-8")
    )
    benchmark = tmp_path / "benchmark.json"
    corpus = tmp_path / "corpus.json"
    benchmark.write_text("benchmark fixture\n", encoding="utf-8")
    corpus.write_text("corpus fixture\n", encoding="utf-8")
    retrieval_work = tmp_path / "retrieval"
    retrieval_work.mkdir()
    freeze_payload = {
        "candidates": {
            candidate: {}
            for candidate in suite["corpora"]["metasyn"]["retrieval_study"][
                "candidate_ids"
            ]
        }
    }
    freeze = _self_hashed(freeze_payload, field="freeze_payload_sha256")
    freeze_path = retrieval_work / "freeze_receipt.json"
    atomic_write_json(freeze_path, freeze)
    retrieval_payload = {
        "metasyn_retrieval_public_summary_version": "1",
        "selection_protocol": {
            "development_compared_candidates": suite["corpora"]["metasyn"][
                "retrieval_study"
            ]["candidate_ids"],
            "official_test_evaluated": False,
        },
        "access_boundary": {"official_test_gold_not_scored": True},
        "network_calls": 0,
        "provider_calls": 0,
        "lineage": {
            "benchmark_manifest_sha256": sha256_file(benchmark),
            "corpus_manifest_sha256": sha256_file(corpus),
            "freeze_receipt_sha256": sha256_file(freeze_path),
        },
    }
    retrieval = _self_hashed(
        retrieval_payload, field="public_summary_payload_sha256"
    )
    retrieval_path = tmp_path / "retrieval-summary.json"
    atomic_write_json(retrieval_path, retrieval)
    screening_payload = {
        "metasyn_screening_public_summary_version": "1",
        "protocol": {
            "candidate_family_frozen": suite["corpora"]["metasyn"][
                "screening_study"
            ]["candidate_ids"],
            "official_test_inputs_opened_by_this_study": False,
            "official_test_labels_opened_by_this_study": False,
            "official_test_evaluated": False,
        },
        "data_scope": {"retrieval_candidate_depth": 200},
        "interpretation_limits": {
            "official_test_labels_historically_opened_elsewhere_in_repository": True
        },
        "public_redaction": {
            "contains_question_or_component_identifiers": False,
            "contains_article_identifiers": False,
            "contains_titles_abstracts_or_protocol_text": False,
            "contains_labels_or_per_question_results": False,
            "contains_absolute_paths": False,
        },
        "lineage": {
            "benchmark_manifest_sha256": sha256_file(benchmark),
            "corpus_manifest_sha256": sha256_file(corpus),
            "retrieval_freeze_payload_sha256": freeze["freeze_payload_sha256"],
        },
    }
    screening = _self_hashed(
        screening_payload, field="public_summary_payload_sha256"
    )
    screening_path = tmp_path / "screening-summary.json"
    atomic_write_json(screening_path, screening)
    return {
        "suite": suite,
        "benchmark": benchmark,
        "corpus": corpus,
        "retrieval_work": retrieval_work,
        "retrieval": retrieval,
        "retrieval_path": retrieval_path,
        "screening": screening,
        "screening_path": screening_path,
    }


def test_integrated_artifact_validation_binds_screening_to_retrieval_freeze(
    tmp_path: Path,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    integrity = validate_study_artifacts(
        suite=fixture["suite"],
        benchmark_manifest_path=fixture["benchmark"],
        corpus_manifest_path=fixture["corpus"],
        retrieval_work_dir=fixture["retrieval_work"],
        retrieval_summary_path=fixture["retrieval_path"],
        returned_retrieval_summary=fixture["retrieval"],
        screening_summary_path=fixture["screening_path"],
        returned_screening_summary=fixture["screening"],
    )
    assert (
        integrity["metasyn_screening_summary"]["retrieval_freeze_payload_sha256"]
        == integrity["metasyn_retrieval_summary"]["freeze_payload_sha256"]
    )

    screening_payload = {
        key: value
        for key, value in fixture["screening"].items()
        if key != "public_summary_payload_sha256"
    }
    screening_payload["lineage"] = {
        **screening_payload["lineage"],
        "retrieval_freeze_payload_sha256": "0" * 64,
    }
    tampered = _self_hashed(
        screening_payload, field="public_summary_payload_sha256"
    )
    atomic_write_json(fixture["screening_path"], tampered, force=True)
    with pytest.raises(
        BenchmarkSuiteError, match="metasyn_retrieval_screening_lineage_mismatch"
    ):
        validate_study_artifacts(
            suite=fixture["suite"],
            benchmark_manifest_path=fixture["benchmark"],
            corpus_manifest_path=fixture["corpus"],
            retrieval_work_dir=fixture["retrieval_work"],
            retrieval_summary_path=fixture["retrieval_path"],
            returned_retrieval_summary=fixture["retrieval"],
            screening_summary_path=fixture["screening_path"],
            returned_screening_summary=tampered,
        )
