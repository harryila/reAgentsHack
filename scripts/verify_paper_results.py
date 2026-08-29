#!/usr/bin/env python3
"""Verify that paper result macros are exact transcriptions of frozen JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import hash_canonical, sha256_file

_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_LINEAGE_BY_ARTIFACT = {
    "artifacts/paper/budgeted-verification-simulation-200.json": {
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
    "artifacts/paper/calibration-simulation-100.json": {
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
    "artifacts/paper/meta-simulation-200.json": {
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
}


def _validate_json_artifact(relative: str, value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"paper_result_artifact_root_not_object:{relative}")
    if value.get("run_config_sha256") != hash_canonical(value.get("run_config")):
        raise ValueError(f"paper_result_run_config_hash_mismatch:{relative}")
    payload = {
        key: item for key, item in value.items() if key != "artifact_payload_sha256"
    }
    if value.get("artifact_payload_sha256") != hash_canonical(payload):
        raise ValueError(f"paper_result_payload_hash_mismatch:{relative}")
    source_hashes = value["run_config"]["source_files_sha256"]
    if set(source_hashes) != _REQUIRED_LINEAGE_BY_ARTIFACT[relative]:
        raise ValueError(f"paper_result_source_lineage_incomplete:{relative}")
    for source, expected in source_hashes.items():
        if sha256_file(_ROOT / source) != expected:
            raise ValueError(f"paper_result_source_hash_mismatch:{relative}:{source}")


def _load_json(relative: str) -> dict[str, Any]:
    value = json.loads((_ROOT / relative).read_text(encoding="utf-8"))
    _validate_json_artifact(relative, value)
    return value


def _load_macros(relative: str) -> tuple[dict[str, str], str]:
    text = (_ROOT / relative).read_text(encoding="utf-8")
    macros: dict[str, str] = {}
    for line in text.splitlines():
        prefix = "\\newcommand{\\"
        if not line.startswith(prefix):
            continue
        name, separator, value = line[len(prefix) :].partition("}{")
        if not separator or not value.endswith("}"):
            raise ValueError(f"paper_result_macro_invalid:{relative}:{line}")
        macros[name] = value[:-1]
    return macros, text


def _percentage(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}\\%"


def _number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _assert_macros(
    *, relative: str, actual: dict[str, str], expected: dict[str, str]
) -> None:
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"paper_result_macro_mismatch:{relative}:"
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _verify_budget() -> dict[str, str]:
    artifact_relative = "artifacts/paper/budgeted-verification-simulation-200.json"
    tex_relative = "paper/results/budgeted_verification_simulation_200.tex"
    artifact = _load_json(artifact_relative)
    macros, tex = _load_macros(tex_relative)
    summary = artifact["summary"]
    expected: dict[str, str] = {}
    policies = {
        "Random": "random",
        "Cost": "cost_only",
        "Risk": "risk_only",
        "Disagreement": "disagreement",
        "Influence": "influence_only",
        "RiskInfluence": "risk_x_influence",
        "RiskCost": "risk_per_cost",
        "InfluenceCost": "influence_per_cost",
        "Proposed": "risk_x_influence_per_cost",
    }
    for prefix, policy in policies.items():
        for budget_name, budget in (("Five", "5.0"), ("Ten", "10.0")):
            row = summary["policies"][policy]["budgets"][budget]
            expected[f"Budget{prefix}Loss{budget_name}"] = _percentage(
                row["mean_claim_loss_recovery_fraction"]
            )
            expected[f"Budget{prefix}Repair{budget_name}"] = _percentage(
                row["claim_repair_rate"]
            )
    comparators = {
        "Risk": "risk_only",
        "Random": "random",
        "RiskInfluence": "risk_x_influence",
        "RiskCost": "risk_per_cost",
        "InfluenceCost": "influence_per_cost",
    }
    rows = summary["paired_contrasts"]["5.0"]["comparators"]
    for prefix, comparator in comparators.items():
        loss = rows[comparator]["claim_loss_recovery_fraction"]
        repair = rows[comparator]["claim_repair_rate"]
        expected[f"BudgetProposed{prefix}LossDiffFive"] = _number(
            100.0 * loss["proposed_minus_comparator_mean_difference"]
        )
        expected[f"BudgetProposed{prefix}LossDiffFiveLow"] = _number(
            100.0 * loss["confidence_interval_95"][0]
        )
        expected[f"BudgetProposed{prefix}LossDiffFiveHigh"] = _number(
            100.0 * loss["confidence_interval_95"][1]
        )
        expected[f"BudgetProposed{prefix}RepairDiffFive"] = _number(
            100.0 * repair["proposed_minus_comparator_rate_difference"]
        )
        expected[f"BudgetProposed{prefix}RepairDiffFiveLow"] = _number(
            100.0 * repair["confidence_interval_95"][0]
        )
        expected[f"BudgetProposed{prefix}RepairDiffFiveHigh"] = _number(
            100.0 * repair["confidence_interval_95"][1]
        )
    _assert_macros(relative=tex_relative, actual=macros, expected=expected)
    artifact_sha256 = sha256_file(_ROOT / artifact_relative)
    if artifact_sha256 not in tex:
        raise ValueError("paper_result_budget_artifact_hash_missing")
    return {
        "artifact_sha256": artifact_sha256,
        "run_config_sha256": artifact["run_config_sha256"],
    }


def _verify_calibration() -> dict[str, str]:
    artifact_relative = "artifacts/paper/calibration-simulation-100.json"
    tex_relative = "paper/results/calibration_simulation_100.tex"
    artifact = _load_json(artifact_relative)
    macros, _ = _load_macros(tex_relative)
    policies = artifact["summary"]["policies"]
    expected = {
        "CalRunConfigHash": f"\\nolinkurl{{{artifact['run_config_sha256']}}}",
    }
    specifications = {
        "Calibrated": ("calibrated", (1, 1, 2, 2)),
        "Raw": ("uncalibrated_score_at_alpha", (2, 2, 2, 2)),
        "Bootstrap": ("bootstrap_instability_only", (1, 2, 1, 1)),
        "Fixed": ("fixed_at_least_five_papers", (1, 2, 1, 1)),
    }
    for prefix, (policy, digits) in specifications.items():
        row = policies[policy]
        expected[f"{prefix}Coverage"] = _percentage(row["mean_coverage"], digits[0])
        expected[f"{prefix}CoverageSD"] = _percentage(row["sd_coverage"], digits[1])
        expected[f"{prefix}EmpiricalRisk"] = _percentage(
            row["mean_empirical_risk"], digits[2]
        )
        expected[f"{prefix}TrueRisk"] = _percentage(
            row["mean_true_selective_risk"], digits[3]
        )
        expected[f"{prefix}Nonempty"] = str(row["nonempty_replicates"])
        expected[f"{prefix}Violations"] = (
            f"{row['true_risk_violation_count']}/{row['nonempty_replicates']}"
        )
    _assert_macros(relative=tex_relative, actual=macros, expected=expected)
    return {
        "artifact_sha256": sha256_file(_ROOT / artifact_relative),
        "run_config_sha256": artifact["run_config_sha256"],
    }


def _verify_meta() -> dict[str, str]:
    artifact_relative = "artifacts/paper/meta-simulation-200.json"
    tex_relative = "paper/results/meta_simulation_200.tex"
    artifact = _load_json(artifact_relative)
    macros, _ = _load_macros(tex_relative)
    null = artifact["summary"]["null_moderator"]
    planted = artifact["summary"]["planted_moderator"]
    expected = {
        "MetaRunConfigHash": f"\\nolinkurl{{{artifact['run_config_sha256']}}}",
        "MetaNullDetection": f"{null['meta_detection_rate']:.3f}",
        "VoteNullDetection": f"{null['significance_vote_detection_rate']:.3f}",
        "MetaNullBrier": f"{null['meta_mean_heldout_brier']:.6f}",
        "VoteNullBrier": f"{null['significance_vote_mean_heldout_brier']:.6f}",
        "MetaPlantedDetection": f"{planted['meta_detection_rate']:.3f}",
        "VotePlantedDetection": f"{planted['significance_vote_detection_rate']:.3f}",
        "MetaPlantedBrier": f"{planted['meta_mean_heldout_brier']:.6f}",
        "VotePlantedBrier": f"{planted['significance_vote_mean_heldout_brier']:.6f}",
    }
    _assert_macros(relative=tex_relative, actual=macros, expected=expected)
    return {
        "artifact_sha256": sha256_file(_ROOT / artifact_relative),
        "run_config_sha256": artifact["run_config_sha256"],
    }


def main() -> int:
    result = {
        "status": "verified",
        "budget": _verify_budget(),
        "calibration": _verify_calibration(),
        "meta_analysis": _verify_meta(),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
