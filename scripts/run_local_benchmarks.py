#!/usr/bin/env python3
"""Run or verify the version-pinned, fully local real-data benchmark suite."""

from __future__ import annotations

import argparse
import json
import re
import socket
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_retrieval import (
    MetaSynCorpusError,
    verify_corpus_manifest,
)
from literature_multiverse.metasyn_retrieval_study import run_retrieval_study
from literature_multiverse.metasyn_screening_study import run_screening_study
from literature_multiverse.paths import PATHS


class BenchmarkSuiteError(ValueError):
    """The suite contract, local inputs, or frozen output is invalid."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_CORPORA = frozenset({"antiox_training", "evidence_inference_2", "metasyn"})
_EXPECTED_PINNED_PATHS = frozenset(
    {
        "artifacts/paper/metasyn-benchmark/manifest.json",
        "configs/benchmarks/metasyn-corpus-c8fa07d.json",
        "data/cache/evidence-inference-gepa/conversion_report.json",
        "data/cache/evidence-inference-gepa/manifest.json",
        "data/cache/metasyn/LICENSE",
        "data/cache/metasyn/reviews-test.parquet",
        "data/cache/metasyn/reviews-train.parquet",
        "data/processed/antiox-training/findings.parquet",
        "data/processed/antiox-training/papers.parquet",
        "data/raw/map/antiox-training/source_lines.json",
    }
)
_SOURCE_CODE_PATHS = (
    "pyproject.toml",
    "scripts/run_local_benchmarks.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/calibration.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/metasyn_benchmark.py",
    "src/literature_multiverse/metasyn_retrieval.py",
    "src/literature_multiverse/metasyn_retrieval_study.py",
    "src/literature_multiverse/metasyn_screening_study.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "uv.lock",
)


def _source_code_hashes(*, repository_root: Path) -> dict[str, str]:
    """Hash the complete in-repository dependency closure of this runner."""

    missing = [
        relative
        for relative in _SOURCE_CODE_PATHS
        if not (repository_root / relative).is_file()
    ]
    if missing:
        raise BenchmarkSuiteError(f"local_suite_source_file_missing:{missing}")
    return {
        relative: sha256_file(repository_root / relative)
        for relative in _SOURCE_CODE_PATHS
    }


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkSuiteError(f"suite_path_invalid:{field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise BenchmarkSuiteError(f"suite_path_unsafe:{field}")
    return path


def _load_json_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkSuiteError(f"{code}:{path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkSuiteError(f"{code}:{path}")
    return value


def _attach_self_hash(payload: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in payload:
        raise BenchmarkSuiteError(f"self_hash_field_already_present:{field}")
    return {**payload, field: hash_canonical(payload)}


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, code: str) -> None:
    observed = payload.get(field)
    unhashed = {key: value for key, value in payload.items() if key != field}
    if not isinstance(observed, str) or observed != hash_canonical(unhashed):
        raise BenchmarkSuiteError(code)


def load_suite(path: Path) -> dict[str, Any]:
    try:
        suite = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkSuiteError(f"suite_manifest_invalid:{path}") from exc
    if not isinstance(suite, dict) or suite.get("benchmark_suite_version") != "1":
        raise BenchmarkSuiteError("suite_manifest_version_unsupported")
    if suite.get("suite_id") != "local-real-data-evaluation-v1":
        raise BenchmarkSuiteError("suite_id_changed_without_version_bump")
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
    if set(corpora) != _EXPECTED_CORPORA:
        raise BenchmarkSuiteError("suite_corpus_inventory_changed_without_version_bump")
    expected_splits = {
        "antiox_training": {"training"},
        "evidence_inference_2": {"dev", "test", "train"},
        "metasyn": {"calibration", "development", "test"},
    }
    for corpus_name, splits in expected_splits.items():
        access = corpora[corpus_name]["access_state"]
        if set(access) != splits:
            raise BenchmarkSuiteError(f"access_state_split_inventory_changed:{corpus_name}")
        for split in splits:
            state = access[split]
            if (
                state["labels_previously_opened"] is not True
                or state["pristine_final_holdout_eligible"] is not False
                or not isinstance(state.get("scientific_role"), str)
                or not state["scientific_role"]
            ):
                raise BenchmarkSuiteError(
                    f"historical_access_state_contract_changed:{corpus_name}:{split}"
                )
    metasyn = corpora["metasyn"]
    expected_metasyn_identity = {
        "dataset_version": (
            "THUIR/MetaSyn@c8fa07d89c44093d623f9a213c6bf070f40ab960"
        ),
        "corpus_manifest": "configs/benchmarks/metasyn-corpus-c8fa07d.json",
        "benchmark_manifest": "artifacts/paper/metasyn-benchmark/manifest.json",
        "review_cache_dir": "data/cache/metasyn",
    }
    if any(metasyn.get(key) != value for key, value in expected_metasyn_identity.items()):
        raise BenchmarkSuiteError("metasyn_identity_contract_changed")
    for field in ("corpus_manifest", "benchmark_manifest", "review_cache_dir"):
        _safe_relative_path(metasyn[field], field=f"corpora.metasyn.{field}")
    retrieval_study = metasyn.get("retrieval_study")
    expected_study = {
        "protocol_version": "1",
        "candidate_ids": [
            "bm25-fixed-v1",
            "rrf-tfidf-bm25-fixed-v1",
            "tfidf-fixed-v1",
        ],
        "selection_split": "development",
        "single_evaluation_split": "calibration",
        "official_test_evaluated": False,
        "primary_metric": "question_weighted_macro_matched_subset_recall_at_200",
    }
    if retrieval_study != expected_study:
        raise BenchmarkSuiteError("metasyn_retrieval_study_contract_changed")
    screening_study = metasyn.get("screening_study")
    expected_screening_study = {
        "protocol_version": "1",
        "candidate_ids": [
            "logistic-l2-balanced-v1",
            "monotonic-hist-gradient-balanced-v1",
            "rrf-passthrough-v1",
        ],
        "selection_split": "development",
        "single_evaluation_split": "calibration",
        "retrieval_source_candidate_id": "rrf-tfidf-bm25-fixed-v1",
        "retrieval_candidate_depth": 200,
        "official_test_evaluated": False,
        "primary_metric": (
            "unweighted_mean_question_macro_absolute_recall_at_10_20_50_100"
        ),
    }
    if screening_study != expected_screening_study:
        raise BenchmarkSuiteError("metasyn_screening_study_contract_changed")
    evidence_inference = corpora["evidence_inference_2"]
    if (
        evidence_inference.get("dataset_version") != "Evidence Inference 2.0"
        or evidence_inference.get("manifest")
        != "data/cache/evidence-inference-gepa/manifest.json"
        or evidence_inference.get("conversion_report")
        != "data/cache/evidence-inference-gepa/conversion_report.json"
        or evidence_inference.get("license_status")
        != "local_only_redistribution_rights_unconfirmed"
    ):
        raise BenchmarkSuiteError("evidence_inference_inventory_contract_changed")
    for field in ("manifest", "conversion_report"):
        _safe_relative_path(
            evidence_inference[field], field=f"corpora.evidence_inference_2.{field}"
        )
    pinned = suite.get("pinned_files")
    if not isinstance(pinned, list) or not pinned:
        raise BenchmarkSuiteError("suite_pinned_files_missing")
    pinned_paths: list[str] = []
    for index, record in enumerate(pinned):
        if not isinstance(record, dict):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        relative = _safe_relative_path(
            record.get("path"), field=f"pinned_files[{index}].path"
        ).as_posix()
        expected = record.get("sha256")
        required_for = record.get("required_for")
        if (
            not isinstance(expected, str)
            or _SHA256_RE.fullmatch(expected) is None
            or not isinstance(required_for, str)
            or not required_for
        ):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        pinned_paths.append(relative)
    if len(pinned_paths) != len(set(pinned_paths)):
        raise BenchmarkSuiteError("suite_pinned_file_duplicate")
    if set(pinned_paths) != _EXPECTED_PINNED_PATHS:
        raise BenchmarkSuiteError("suite_pinned_file_inventory_changed_without_version_bump")
    if suite.get("license_policy") != {
        "repository_license_required_for_public_release": True,
        "metasyn_article_payloads_local_only": True,
        "evidence_inference_text_bearing_artifacts_local_only": True,
    }:
        raise BenchmarkSuiteError("suite_license_policy_changed_without_version_bump")
    return suite


def verify_pinned_files(
    suite: dict[str, Any], *, repository_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    pinned = suite.get("pinned_files")
    if not isinstance(pinned, list) or not pinned:
        raise BenchmarkSuiteError("suite_pinned_files_missing")
    repository_root_resolved = repository_root.resolve()
    seen: set[str] = set()
    for index, record in enumerate(pinned):
        if not isinstance(record, dict):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        relative_path = _safe_relative_path(
            record.get("path"), field=f"pinned_files[{index}].path"
        )
        relative = relative_path.as_posix()
        expected = record.get("sha256")
        if (
            relative in seen
            or not isinstance(expected, str)
            or _SHA256_RE.fullmatch(expected) is None
        ):
            raise BenchmarkSuiteError("suite_pinned_file_invalid")
        seen.add(relative)
        path = repository_root / relative_path
        try:
            path.resolve(strict=False).relative_to(repository_root_resolved)
        except ValueError as exc:
            raise BenchmarkSuiteError(
                f"pinned_file_resolves_outside_repository:{relative}"
            ) from exc
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
    project_license_paths = sorted(
        path
        for path in repository_root.iterdir()
        if path.is_file() and path.name.casefold().startswith("license")
    )
    project_licenses = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in project_license_paths
    ]
    declared_license: str | None = None
    pyproject_path = repository_root / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            pyproject = {}
        raw_license = pyproject.get("project", {}).get("license")
        if isinstance(raw_license, str) and raw_license.strip():
            declared_license = raw_license.strip()
    release_ready = bool(project_licenses) and all(
        record["bytes"] > 0 for record in project_licenses
    ) and bool(declared_license)
    return {
        "repository_code_license_files": project_licenses,
        "repository_declared_license": declared_license,
        "repository_public_release_ready": release_ready,
        "release_blocker": None if release_ready else "repository_license_not_declared",
        "metasyn_payload_policy": "local_evaluation_only_third_party_terms_apply",
        "evidence_inference_payload_policy": (
            "local_only_until_bundled_article_redistribution_rights_are_confirmed"
        ),
    }


class _OfflineSocket(socket.socket):
    def connect(self, address: Any) -> None:
        del address
        raise BenchmarkSuiteError("network_call_attempted_during_offline_suite")

    def connect_ex(self, address: Any) -> int:
        del address
        raise BenchmarkSuiteError("network_call_attempted_during_offline_suite")


def _deny_create_connection(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise BenchmarkSuiteError("network_call_attempted_during_offline_suite")


@contextmanager
def offline_network_guard() -> Iterator[None]:
    """Fail the run if Python code attempts an outbound socket connection."""

    with (
        patch.object(socket, "socket", _OfflineSocket),
        patch.object(socket, "create_connection", _deny_create_connection),
    ):
        yield


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def preflight_run_paths(
    *,
    repository_root: Path,
    report_path: Path,
    retrieval_work_dir: Path,
    screening_work_dir: Path,
    retrieval_summary_path: Path,
    screening_summary_path: Path,
    force: bool,
) -> None:
    """Reject ambiguous paths and require explicit authority to replace prior runs."""

    root = repository_root.resolve(strict=False)
    work_dirs = {
        "retrieval_work_dir": retrieval_work_dir,
        "screening_work_dir": screening_work_dir,
    }
    for name, path in work_dirs.items():
        if path.resolve(strict=False) == root:
            raise BenchmarkSuiteError(f"work_dir_cannot_be_repository_root:{name}")
    if _paths_overlap(retrieval_work_dir, screening_work_dir):
        raise BenchmarkSuiteError("retrieval_and_screening_work_dirs_overlap")
    outputs = {
        "report": report_path,
        "retrieval_summary": retrieval_summary_path,
        "screening_summary": screening_summary_path,
    }
    if len({path.resolve(strict=False) for path in outputs.values()}) != len(outputs):
        raise BenchmarkSuiteError("public_output_paths_overlap")
    for output_name, output_path in outputs.items():
        for work_name, work_path in work_dirs.items():
            if _paths_overlap(output_path, work_path):
                raise BenchmarkSuiteError(
                    f"public_output_overlaps_work_dir:{output_name}:{work_name}"
                )
    if force:
        return
    existing = sorted(
        path.as_posix()
        for path in [*work_dirs.values(), *outputs.values()]
        if path.exists()
    )
    if existing:
        raise BenchmarkSuiteError(
            "prior_run_artifacts_require_explicit_force:" + ",".join(existing)
        )


def validate_study_artifacts(
    *,
    suite: Mapping[str, Any],
    benchmark_manifest_path: Path,
    corpus_manifest_path: Path,
    retrieval_work_dir: Path,
    retrieval_summary_path: Path,
    returned_retrieval_summary: Mapping[str, Any],
    screening_summary_path: Path,
    returned_screening_summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Recheck public self-hashes and the cross-study retrieval lineage."""

    retrieval = _load_json_object(
        retrieval_summary_path, code="metasyn_retrieval_summary_invalid"
    )
    screening = _load_json_object(
        screening_summary_path, code="metasyn_screening_summary_invalid"
    )
    if retrieval != dict(returned_retrieval_summary):
        raise BenchmarkSuiteError("metasyn_retrieval_returned_summary_file_mismatch")
    if screening != dict(returned_screening_summary):
        raise BenchmarkSuiteError("metasyn_screening_returned_summary_file_mismatch")
    _verify_self_hash(
        retrieval,
        field="public_summary_payload_sha256",
        code="metasyn_retrieval_public_summary_hash_mismatch",
    )
    _verify_self_hash(
        screening,
        field="public_summary_payload_sha256",
        code="metasyn_screening_public_summary_hash_mismatch",
    )
    metasyn_contract = suite["corpora"]["metasyn"]
    retrieval_contract = metasyn_contract["retrieval_study"]
    screening_contract = metasyn_contract["screening_study"]
    if (
        retrieval.get("metasyn_retrieval_public_summary_version") != "1"
        or retrieval.get("selection_protocol", {}).get(
            "development_compared_candidates"
        )
        != retrieval_contract["candidate_ids"]
        or retrieval.get("selection_protocol", {}).get("official_test_evaluated")
        is not False
        or retrieval.get("access_boundary", {}).get("official_test_gold_not_scored")
        is not True
        or retrieval.get("network_calls") != 0
        or retrieval.get("provider_calls") != 0
    ):
        raise BenchmarkSuiteError("metasyn_retrieval_public_contract_mismatch")
    screening_protocol = screening.get("protocol", {})
    if (
        screening.get("metasyn_screening_public_summary_version") != "1"
        or screening_protocol.get("candidate_family_frozen")
        != screening_contract["candidate_ids"]
        or screening_protocol.get("official_test_inputs_opened_by_this_study")
        is not False
        or screening_protocol.get("official_test_labels_opened_by_this_study")
        is not False
        or screening_protocol.get("official_test_evaluated") is not False
        or screening.get("data_scope", {}).get("retrieval_candidate_depth")
        != screening_contract["retrieval_candidate_depth"]
        or screening.get("interpretation_limits", {}).get(
            "official_test_labels_historically_opened_elsewhere_in_repository"
        )
        is not True
        or screening.get("public_redaction")
        != {
            "contains_question_or_component_identifiers": False,
            "contains_article_identifiers": False,
            "contains_titles_abstracts_or_protocol_text": False,
            "contains_labels_or_per_question_results": False,
            "contains_absolute_paths": False,
        }
    ):
        raise BenchmarkSuiteError("metasyn_screening_public_contract_mismatch")
    expected_benchmark_sha256 = sha256_file(benchmark_manifest_path)
    expected_corpus_sha256 = sha256_file(corpus_manifest_path)
    retrieval_lineage = retrieval.get("lineage", {})
    screening_lineage = screening.get("lineage", {})
    for lineage in (retrieval_lineage, screening_lineage):
        if (
            lineage.get("benchmark_manifest_sha256") != expected_benchmark_sha256
            or lineage.get("corpus_manifest_sha256") != expected_corpus_sha256
        ):
            raise BenchmarkSuiteError("metasyn_public_summary_input_lineage_mismatch")
    freeze_path = retrieval_work_dir / "freeze_receipt.json"
    freeze = _load_json_object(freeze_path, code="metasyn_retrieval_freeze_invalid")
    _verify_self_hash(
        freeze,
        field="freeze_payload_sha256",
        code="metasyn_retrieval_freeze_payload_hash_mismatch",
    )
    if (
        retrieval_lineage.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or screening_lineage.get("retrieval_freeze_payload_sha256")
        != freeze.get("freeze_payload_sha256")
        or screening_contract["retrieval_source_candidate_id"]
        not in freeze.get("candidates", {})
    ):
        raise BenchmarkSuiteError("metasyn_retrieval_screening_lineage_mismatch")
    return {
        "metasyn_retrieval_summary": {
            "path": retrieval_summary_path.as_posix(),
            "file_sha256": sha256_file(retrieval_summary_path),
            "payload_sha256": retrieval["public_summary_payload_sha256"],
            "freeze_receipt_sha256": sha256_file(freeze_path),
            "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        },
        "metasyn_screening_summary": {
            "path": screening_summary_path.as_posix(),
            "file_sha256": sha256_file(screening_summary_path),
            "payload_sha256": screening["public_summary_payload_sha256"],
            "retrieval_freeze_payload_sha256": screening_lineage[
                "retrieval_freeze_payload_sha256"
            ],
        },
    }


def write_self_hashed_report(
    path: Path,
    payload: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    report = _attach_self_hash(payload, field="report_payload_sha256")
    atomic_write_json(path, report, force=force)
    persisted = _load_json_object(path, code="local_benchmark_report_invalid")
    if persisted != report:
        raise BenchmarkSuiteError("local_benchmark_report_write_mismatch")
    _verify_self_hash(
        persisted,
        field="report_payload_sha256",
        code="local_benchmark_report_payload_hash_mismatch",
    )
    return report


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
        "--metasyn-work-dir",
        type=Path,
        default=Path("data/cache/metasyn/retrieval-study-v1"),
        help="Ignored local directory for identifier-bearing retrieval artifacts.",
    )
    parser.add_argument(
        "--metasyn-screening-work-dir",
        type=Path,
        default=Path("data/cache/metasyn/screening-study-v1"),
        help="Ignored local directory for identifier-bearing screening artifacts.",
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
    source_code_sha256s = _source_code_hashes(repository_root=root)
    if args.contract_only:
        print(
            json.dumps(
                {
                    "status": "contract_valid",
                    "suite_sha256": sha256_file(suite_path),
                    "source_code_sha256s": source_code_sha256s,
                    "all_opened_splits_ineligible_as_pristine_holdout": True,
                    "local_payload_files_opened": False,
                    "studies_executed": False,
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
            with offline_network_guard():
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
        if report_path.exists() and not args.force:
            raise BenchmarkSuiteError(
                "prior_run_artifacts_require_explicit_force:"
                + report_path.as_posix()
            )
        write_self_hashed_report(
            report_path,
            {
                "local_benchmark_report_version": "2",
                "status": "blocked",
                "suite_sha256": sha256_file(suite_path),
                "source_code_sha256s": source_code_sha256s,
                "input_checks": checks,
                "blockers": blockers,
                "license_audit": licenses,
                "results": {},
            },
            force=args.force,
        )
        print(json.dumps({"status": "blocked", "report": report_path.as_posix()}))
        return 2

    benchmark_manifest = root / suite["corpora"]["metasyn"]["benchmark_manifest"]
    metasyn_work_dir = (
        args.metasyn_work_dir
        if args.metasyn_work_dir.is_absolute()
        else root / args.metasyn_work_dir
    )
    metasyn_summary_path = root / "artifacts/diagnostics/metasyn-retrieval-study-v1.json"
    metasyn_screening_work_dir = (
        args.metasyn_screening_work_dir
        if args.metasyn_screening_work_dir.is_absolute()
        else root / args.metasyn_screening_work_dir
    )
    metasyn_screening_summary_path = (
        root / "artifacts/diagnostics/metasyn-screening-study-v1.json"
    )
    preflight_run_paths(
        repository_root=root,
        report_path=report_path,
        retrieval_work_dir=metasyn_work_dir,
        screening_work_dir=metasyn_screening_work_dir,
        retrieval_summary_path=metasyn_summary_path,
        screening_summary_path=metasyn_screening_summary_path,
        force=args.force,
    )
    write_self_hashed_report(
        report_path,
        {
            "local_benchmark_report_version": "2",
            "status": "running_unverified",
            "suite_sha256": sha256_file(suite_path),
            "source_code_sha256s": source_code_sha256s,
            "input_checks": checks,
            "results": {},
        },
        force=args.force,
    )
    with offline_network_guard():
        metasyn_summary = run_retrieval_study(
            benchmark_manifest_path=benchmark_manifest,
            corpus_manifest_path=corpus_manifest_path,
            repository_root=root,
            review_cache_dir=root / suite["corpora"]["metasyn"]["review_cache_dir"],
            work_dir=metasyn_work_dir,
            public_summary_path=metasyn_summary_path,
            force=args.force,
        )
        metasyn_screening_summary = run_screening_study(
            benchmark_manifest_path=benchmark_manifest,
            corpus_manifest_path=corpus_manifest_path,
            repository_root=root,
            review_cache_dir=root / suite["corpora"]["metasyn"]["review_cache_dir"],
            retrieval_work_dir=metasyn_work_dir,
            work_dir=metasyn_screening_work_dir,
            public_summary_path=metasyn_screening_summary_path,
            force=args.force,
        )
        artifact_integrity = validate_study_artifacts(
            suite=suite,
            benchmark_manifest_path=benchmark_manifest,
            corpus_manifest_path=corpus_manifest_path,
            retrieval_work_dir=metasyn_work_dir,
            retrieval_summary_path=metasyn_summary_path,
            returned_retrieval_summary=metasyn_summary,
            screening_summary_path=metasyn_screening_summary_path,
            returned_screening_summary=metasyn_screening_summary,
        )
    for record in artifact_integrity.values():
        record["path"] = Path(record["path"]).relative_to(root).as_posix()
    results = {
        "metasyn_retrieval_development_selection_calibration": {
            "scientific_role": "retrospective_nonpristine",
            "selected_candidate": metasyn_summary["selection_protocol"]["selected_candidate"],
            "development": metasyn_summary["development_results"],
            "calibration": metasyn_summary["selected_calibration_result"],
            "official_test_evaluated": False,
        },
        "metasyn_protocol_aware_screening_reranking": {
            "scientific_role": "retrospective_nonpristine_matched_subset_survival",
            "selected_candidate": metasyn_screening_summary["protocol"][
                "selected_candidate"
            ],
            "development": metasyn_screening_summary[
                "development_component_disjoint_cross_validation"
            ],
            "calibration": metasyn_screening_summary["calibration"],
            "official_test_evaluated": False,
        },
    }
    scientific_payload = {
        "suite_sha256": sha256_file(suite_path),
        "artifact_integrity": artifact_integrity,
        "results": results,
    }
    report = write_self_hashed_report(
        report_path,
        {
            "local_benchmark_report_version": "2",
            "status": (
                "complete"
                if licenses["repository_public_release_ready"]
                else "complete_with_release_license_blocker"
            ),
            "suite_sha256": scientific_payload["suite_sha256"],
            "source_code_sha256s": source_code_sha256s,
            "network_calls": 0,
            "network_enforcement": {
                "python_socket_connect_disabled_during_full_run": True,
                "successful_completion_requires_zero_attempted_python_socket_connections": (
                    True
                ),
            },
            "input_checks": checks,
            "label_access": suite["corpora"],
            "license_audit": licenses,
            "execution_scope": {
                "executed_studies": [
                    "metasyn_retrieval_development_selection_calibration",
                    "metasyn_protocol_aware_screening_reranking",
                ],
                "inventory_only_no_metric_in_this_runner": [
                    "antiox_training",
                    "evidence_inference_2",
                ],
            },
            "artifacts": {
                "metasyn_retrieval_summary": (
                    metasyn_summary_path.relative_to(root).as_posix()
                ),
                "metasyn_screening_summary": (
                    metasyn_screening_summary_path.relative_to(root).as_posix()
                ),
                "identifier_bearing_predictions_tracked": False,
                "integrity": artifact_integrity,
            },
            "results": results,
            "reproducibility": {
                "scientific_payload_sha256": hash_canonical(scientific_payload),
                "timestamps_in_scientific_payload": False,
                "selection_sequence": (
                    "freeze three fixed candidates; select once on development; "
                    "evaluate only the selected candidate once on calibration; then freeze "
                    "three within-top-200 screening rerankers, select one by "
                    "component-disjoint development cross-validation, and evaluate it "
                    "once on calibration"
                ),
            },
            "claim_boundary": (
                "Retrieval and screening survival are measured against MetaSyn's "
                "matched-paper subset on previously opened development/calibration "
                "labels. Neither is a pristine holdout result, protocol screening "
                "accuracy, or recall against an exhaustive scientifically eligible corpus."
            ),
        },
        force=True,
    )
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
