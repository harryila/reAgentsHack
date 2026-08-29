"""Lineage and self-hash contracts for the three public planted studies."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from scripts.simulate_budgeted_verification import main as budget_main
from scripts.simulate_meta_analysis import main as meta_main
from scripts.simulate_risk_calibration import main as calibration_main
from scripts.verify_paper_results import _validate_json_artifact

from literature_multiverse.lineage import hash_canonical, sha256_file


@pytest.mark.parametrize(
    ("runner", "arguments", "required_sources"),
    [
        (
            budget_main,
            ["--replicates", "2", "--item-count", "8", "--budgets", "5"],
            {
                "scripts/simulate_budgeted_verification.py",
                "src/literature_multiverse/__init__.py",
                "src/literature_multiverse/budgeted_verification_simulation.py",
                "src/literature_multiverse/budgeted_verification.py",
                "src/literature_multiverse/lineage.py",
                "src/literature_multiverse/models.py",
                "src/literature_multiverse/paths.py",
                "pyproject.toml",
                "uv.lock",
            },
        ),
        (
            calibration_main,
            [
                "--replicates",
                "1",
                "--development-count",
                "20",
                "--calibration-count",
                "30",
                "--test-count",
                "30",
                "--candidate-thresholds",
                "0.1",
            ],
            {
                "scripts/simulate_risk_calibration.py",
                "src/literature_multiverse/__init__.py",
                "src/literature_multiverse/calibration_simulation.py",
                "src/literature_multiverse/calibration.py",
                "src/literature_multiverse/lineage.py",
                "src/literature_multiverse/models.py",
                "src/literature_multiverse/paths.py",
                "pyproject.toml",
                "uv.lock",
            },
        ),
        (
            meta_main,
            [
                "--replicates",
                "2",
                "--papers-per-level",
                "4",
                "--heldout-papers-per-level",
                "4",
            ],
            {
                "scripts/simulate_meta_analysis.py",
                "src/literature_multiverse/__init__.py",
                "src/literature_multiverse/meta_simulation.py",
                "src/literature_multiverse/meta_analysis.py",
                "src/literature_multiverse/budgeted_verification.py",
                "src/literature_multiverse/claim_semantics.py",
                "src/literature_multiverse/effects.py",
                "src/literature_multiverse/evidence_graph.py",
                "src/literature_multiverse/lineage.py",
                "src/literature_multiverse/models.py",
                "src/literature_multiverse/paths.py",
                "pyproject.toml",
                "uv.lock",
            },
        ),
    ],
)
def test_public_planted_artifact_has_complete_lineage_and_self_hash(
    tmp_path: Path,
    runner: Callable[[list[str] | None], int],
    arguments: list[str],
    required_sources: set[str],
) -> None:
    output = tmp_path / "artifact.json"
    assert runner(["--output", output.as_posix(), *arguments]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    payload = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_payload_sha256"
    }

    assert artifact["artifact_payload_sha256"] == hash_canonical(payload)
    source_hashes = artifact["run_config"]["source_files_sha256"]
    assert set(source_hashes) == required_sources
    for relative, expected in source_hashes.items():
        assert sha256_file(Path(relative)) == expected


def test_paper_result_validator_rejects_full_payload_tampering() -> None:
    relative = "artifacts/paper/meta-simulation-200.json"
    artifact = json.loads(Path(relative).read_text(encoding="utf-8"))
    _validate_json_artifact(relative, artifact)
    artifact["summary"]["alpha"] = 0.10

    with pytest.raises(ValueError, match="paper_result_payload_hash_mismatch"):
        _validate_json_artifact(relative, artifact)
