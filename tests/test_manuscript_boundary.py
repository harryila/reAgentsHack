from __future__ import annotations

import subprocess
from pathlib import Path

PRIVATE_ROOTS = (
    "paper",
    "Formatting_Instructions_For_NeurIPS_2026 (2)",
    "artifacts/submission",
)

EVALUATION_ASSETS = {
    "docs/paper/harvester-validation.md",
    "docs/paper/metasyn-benchmark.md",
    "docs/paper/neurips26-evaluation-protocol.md",
    "docs/paper/task-evaluation-contract.md",
    "artifacts/paper/budgeted-verification-simulation-200.json",
    "artifacts/paper/calibration-simulation-100.json",
    "artifacts/paper/closed-corpus-local-audit.json",
    "artifacts/paper/evidence-inference-2/failed-raw-schema-pilot30-summary.json",
    "artifacts/paper/evidence-inference-benchmark-summary.json",
    "artifacts/paper/evidence-inference-gepa-pilot-summary.json",
    "artifacts/paper/evidence-inference-low-budget-summary.json",
    "artifacts/paper/harvester/validation_summary.json",
    "artifacts/paper/meta-simulation-200.json",
    "artifacts/paper/metasyn-benchmark/METASYN_LICENSE.txt",
    "artifacts/paper/metasyn-benchmark/manifest.json",
    "artifacts/paper/metasyn-benchmark/model_inputs/calibration.jsonl",
    "artifacts/paper/metasyn-benchmark/model_inputs/development.jsonl",
    "artifacts/paper/metasyn-benchmark/model_inputs/test.jsonl",
    "artifacts/paper/metasyn-fixed-positive-test/README.md",
    "artifacts/paper/metasyn-fixed-positive-test/evaluation.json",
    "artifacts/paper/metasyn-fixed-positive-test/freeze_receipt.json",
    "artifacts/paper/metasyn-fixed-positive-test/predictions.jsonl",
}


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", repo_root.as_posix(), *args],
        check=False,
        capture_output=True,
    )


def _indexed_paths(repo_root: Path, *pathspecs: str) -> set[str]:
    completed = _git(repo_root, "ls-files", "-z", "--", *pathspecs)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return {
        value.decode("utf-8")
        for value in completed.stdout.split(b"\0")
        if value
    }


def test_private_manuscript_roots_are_ignored_and_unindexed(repo_root: Path) -> None:
    ignore_lines = set(
        (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    )
    assert {
        "/paper/",
        "/Formatting_Instructions_For_NeurIPS_2026 (2)/",
        "/artifacts/submission/",
    } <= ignore_lines

    for root in PRIVATE_ROOTS:
        ignored = _git(
            repo_root,
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            f"{root}/.manuscript-boundary-sentinel",
        )
        assert ignored.returncode == 0

    assert _indexed_paths(
        repo_root,
        "paper/**",
        "Formatting_Instructions_For_NeurIPS_2026 (2)/**",
        "artifacts/submission/**",
    ) == set()


def test_paper_named_evaluation_namespaces_are_explicit_regular_files(
    repo_root: Path,
) -> None:
    for root in ("docs/paper", "artifacts/paper"):
        ignored = _git(
            repo_root,
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            f"{root}/.evaluation-boundary-sentinel",
        )
        assert ignored.returncode == 1

    assert _indexed_paths(
        repo_root,
        "docs/paper/**",
        "artifacts/paper/**",
    ) == EVALUATION_ASSETS

    modes = _git(
        repo_root,
        "ls-files",
        "-s",
        "-z",
        "--",
        "docs/paper/**",
        "artifacts/paper/**",
    )
    assert modes.returncode == 0, modes.stderr.decode(errors="replace")
    records = [record for record in modes.stdout.split(b"\0") if record]
    assert len(records) == len(EVALUATION_ASSETS)
    assert all(record.startswith(b"100644 ") for record in records)
