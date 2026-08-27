#!/usr/bin/env python3
"""Run or verify the version-pinned, fully local real-data benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_benchmark import (
    evaluate_metasyn_predictions,
    load_metasyn_predictions,
)
from literature_multiverse.metasyn_retrieval import (
    MetaSynCorpusError,
    freeze_tfidf_retrieval_baseline,
    verify_corpus_manifest,
)
from literature_multiverse.paths import PATHS


class BenchmarkSuiteError(ValueError):
    """The suite contract, local inputs, or frozen output is invalid."""


def load_suite(path: Path) -> dict[str, Any]:
    try:
        suite = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkSuiteError(f"suite_manifest_invalid:{path}") from exc
    if not isinstance(suite, dict) or suite.get("benchmark_suite_version") != "1":
        raise BenchmarkSuiteError("suite_manifest_version_unsupported")
    if suite.get("network_calls") != 0:
        raise BenchmarkSuiteError("local_suite_must_forbid_network_calls")
    corpora = suite.get("corpora")
    if not isinstance(corpora, dict) or not corpora:
        raise BenchmarkSuiteError("suite_corpora_missing")
    for corpus_name, corpus in corpora.items():
        access = corpus.get("access_state") if isinstance(corpus, dict) else None
        if not isinstance(access, dict) or not access:
            raise BenchmarkSuiteError(f"access_state_missing:{corpus_name}")
        for split, state in access.items():
            if not isinstance(state, dict):
                raise BenchmarkSuiteError(f"access_state_invalid:{corpus_name}:{split}")
            opened = state.get("labels_previously_opened")
            pristine = state.get("pristine_final_holdout_eligible")
            if not isinstance(opened, bool) or not isinstance(pristine, bool):
                raise BenchmarkSuiteError(f"access_state_flags_invalid:{corpus_name}:{split}")
            if opened and pristine:
                raise BenchmarkSuiteError(f"opened_labels_cannot_be_pristine:{corpus_name}:{split}")
    metasyn_test = corpora["metasyn"]["access_state"]["test"]
    if (
        metasyn_test["labels_previously_opened"] is not True
        or metasyn_test["pristine_final_holdout_eligible"] is not False
    ):
        raise BenchmarkSuiteError("metasyn_test_access_state_must_remain_opened")
    return suite


def verify_pinned_files(
    suite: dict[str, Any], *, repository_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    pinned = suite.get("pinned_files")
    if not isinstance(pinned, list) or not pinned:
        raise BenchmarkSuiteError("suite_pinned_files_missing")
    for record in pinned:
        if not isinstance(record, dict):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        path = repository_root / relative
        if not path.is_file():
            blockers.append(
                {
                    "code": "required_local_input_missing",
                    "path": relative,
                    "action": (
                        "restore the exact locally licensed/cache artifact; the runner "
                        "does not download or fabricate benchmark payloads"
                    ),
                }
            )
            checks.append({"path": relative, "status": "missing"})
            continue
        observed = sha256_file(path)
        status = "verified" if observed == expected else "hash_mismatch"
        checks.append(
            {
                "path": relative,
                "status": status,
                "expected_sha256": expected,
                "observed_sha256": observed,
            }
        )
        if observed != expected:
            blockers.append(
                {
                    "code": "required_local_input_hash_mismatch",
                    "path": relative,
                    "action": "restore the pinned version or create a new suite version",
                }
            )
    return checks, blockers


def license_audit(*, repository_root: Path) -> dict[str, Any]:
    project_licenses = sorted(
        path.name
        for path in repository_root.iterdir()
        if path.is_file() and path.name.casefold().startswith("license")
    )
    return {
        "repository_code_license_files": project_licenses,
        "repository_public_release_ready": bool(project_licenses),
        "release_blocker": None if project_licenses else "repository_license_not_declared",
        "metasyn_payload_policy": "local_evaluation_only_third_party_terms_apply",
        "evidence_inference_payload_policy": (
            "local_only_until_bundled_article_redistribution_rights_are_confirmed"
        ),
    }


def _aggregate_retrieval(evaluation: dict[str, Any]) -> dict[str, Any]:
    retrieval = evaluation["retrieval"]
    return {
        "eligible_reviews": retrieval["eligible_reviews"],
        "retrieval_depth": 200,
        "macro_recall_at_200": retrieval["macro_recall_missing_as_zero"],
        "micro_recall_at_200": retrieval["micro_recall_missing_as_zero"],
        "coverage": retrieval["coverage"],
    }


def _combined_retrieval(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for evaluation in evaluations for row in evaluation["per_review"]]
    hits = sum(int(row["retrieval_hits"]) for row in rows)
    gold = sum(int(row["gold_retrieval_count"]) for row in rows)
    recalls = [float(row["retrieval_recall"]) for row in rows]
    return {
        "eligible_reviews": len(rows),
        "retrieval_depth": 200,
        "macro_recall_at_200": sum(recalls) / len(recalls),
        "micro_recall_at_200": hits / gold,
        "coverage": sum(row["retrieval_status"] == "supplied" for row in rows) / len(rows),
    }


def _verify_existing(output_dir: Path) -> tuple[Path, Path]:
    predictions = output_dir / "predictions.jsonl"
    receipt_path = output_dir / "freeze_receipt.json"
    if not predictions.is_file() or not receipt_path.is_file():
        raise BenchmarkSuiteError("partial_retrieval_output_requires_--force")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkSuiteError("retrieval_receipt_invalid") from exc
    if receipt.get("predictions_sha256") != sha256_file(predictions):
        raise BenchmarkSuiteError("retrieval_predictions_hash_mismatch")
    if receipt.get("config_sha256") != hash_canonical(receipt.get("config")):
        raise BenchmarkSuiteError("retrieval_config_hash_mismatch")
    if receipt.get("test_split_evaluated") is not False:
        raise BenchmarkSuiteError("primary_retrieval_must_not_evaluate_test")
    return predictions, receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/benchmarks/local-suite-v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmarks/local-suite-v1"),
    )
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="validate the static access/leakage contract without requiring local caches",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = PATHS.root
    suite_path = args.suite if args.suite.is_absolute() else root / args.suite
    suite = load_suite(suite_path)
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "contract_valid",
                    "suite_sha256": sha256_file(suite_path),
                    "all_opened_splits_ineligible_as_pristine_holdout": True,
                },
                sort_keys=True,
            )
        )
        return 0

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    report_path = output_dir / "benchmark-report.json"
    checks, blockers = verify_pinned_files(suite, repository_root=root)
    licenses = license_audit(repository_root=root)
    corpus_manifest_path = root / suite["corpora"]["metasyn"]["corpus_manifest"]
    if not blockers:
        try:
            verify_corpus_manifest(corpus_manifest_path, repository_root=root)
        except MetaSynCorpusError as exc:
            blockers.append(
                {
                    "code": "metasyn_corpus_invalid",
                    "path": suite["corpora"]["metasyn"]["corpus_manifest"],
                    "action": str(exc),
                }
            )
    if blockers:
        report = {
            "local_benchmark_report_version": "1",
            "status": "blocked",
            "suite_sha256": sha256_file(suite_path),
            "input_checks": checks,
            "blockers": blockers,
            "license_audit": licenses,
            "results": {},
        }
        atomic_write_json(report_path, report, force=True)
        print(json.dumps({"status": "blocked", "report": report_path.as_posix()}))
        return 2

    benchmark_manifest = root / suite["corpora"]["metasyn"]["benchmark_manifest"]
    predictions_path = output_dir / "predictions.jsonl"
    receipt_path = output_dir / "freeze_receipt.json"
    if args.force or (not predictions_path.exists() and not receipt_path.exists()):
        predictions_path, receipt_path = freeze_tfidf_retrieval_baseline(
            benchmark_manifest_path=benchmark_manifest,
            corpus_manifest_path=corpus_manifest_path,
            repository_root=root,
            review_cache_dir=root / suite["corpora"]["metasyn"]["review_cache_dir"],
            output_dir=output_dir,
            top_k=200,
            force=args.force,
        )
    else:
        predictions_path, receipt_path = _verify_existing(output_dir)

    predictions = load_metasyn_predictions(predictions_path)
    development = evaluate_metasyn_predictions(
        manifest_path=benchmark_manifest,
        predictions=predictions,
        evaluation_split="development",
    )
    calibration = evaluate_metasyn_predictions(
        manifest_path=benchmark_manifest,
        predictions=predictions,
        evaluation_split="calibration",
    )
    results = {
        "metasyn_tfidf_retrieval": {
            "scientific_role": "retrospective_local_baseline_not_pristine_holdout",
            "test_evaluated": False,
            "development": _aggregate_retrieval(development),
            "calibration": _aggregate_retrieval(calibration),
            "combined_development_calibration": _combined_retrieval([development, calibration]),
        }
    }
    scientific_payload = {
        "suite_sha256": sha256_file(suite_path),
        "freeze_receipt_sha256": sha256_file(receipt_path),
        "predictions_sha256": sha256_file(predictions_path),
        "results": results,
    }
    report = {
        "local_benchmark_report_version": "1",
        "status": "complete_with_release_license_blocker",
        "suite_sha256": scientific_payload["suite_sha256"],
        "network_calls": 0,
        "input_checks": checks,
        "label_access": suite["corpora"],
        "license_audit": licenses,
        "artifacts": {
            "freeze_receipt": receipt_path.relative_to(root).as_posix(),
            "predictions": predictions_path.relative_to(root).as_posix(),
        },
        "results": results,
        "reproducibility": {
            "scientific_payload_sha256": hash_canonical(scientific_payload),
            "timestamps_in_scientific_payload": False,
            "stable_tie_break": "descending_score_then_ascending_corpus_id",
        },
        "claim_boundary": (
            "Recall@200 is measured against MetaSyn's matched-paper subset on previously "
            "opened development/calibration labels. It is neither a pristine holdout "
            "result nor recall against an exhaustive scientifically eligible corpus."
        ),
    }
    atomic_write_json(report_path, report, force=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report_path.as_posix(),
                "report_sha256": sha256_file(report_path),
                "results": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
