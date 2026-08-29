#!/usr/bin/env python3
"""Build hash-bound native source manifests from local archived corpora only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import (
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.paths import PATHS
from literature_multiverse.source_manifest_bridge import (
    DiagnosticSourceLedger,
    build_antiox_native_source_bridge,
    build_metasyn_native_source_bridge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=PATHS.root,
        help="repository root used to resolve and constrain artifact paths",
    )
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--public-run-output",
        type=Path,
        help="optional metadata-only copy of the self-hashed run receipt",
    )
    parser.add_argument("--force", action="store_true")
    subparsers = parser.add_subparsers(dest="corpus_kind", required=True)

    antiox = subparsers.add_parser("antiox")
    antiox.add_argument(
        "--papers",
        type=Path,
        default=Path("data/processed/antiox-training/papers.parquet"),
    )
    antiox.add_argument(
        "--source-lines",
        type=Path,
        default=Path("data/raw/map/antiox-training/source_lines.json"),
    )
    antiox.add_argument(
        "--scope",
        choices=(
            "successful_screened_in",
            "legacy_eligible",
            "all_source_available",
        ),
        default="successful_screened_in",
        help=(
            "Explicit diagnostic subset. legacy_eligible uses previously opened "
            "eligibility outputs and is never a pristine evaluation corpus."
        ),
    )
    antiox.add_argument("--expected-papers-sha256")
    antiox.add_argument("--expected-source-lines-sha256")

    metasyn = subparsers.add_parser("metasyn")
    metasyn.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("configs/benchmarks/metasyn-corpus-c8fa07d.json"),
    )
    metasyn.add_argument(
        "--corpus-id",
        type=int,
        action="append",
        default=[],
        help="repeat for each selected MetaSyn corpus ID",
    )
    metasyn.add_argument(
        "--corpus-ids-file",
        type=Path,
        help="JSON integer array or newline-delimited integer IDs",
    )
    metasyn.add_argument(
        "--all-corpus-rows",
        action="store_true",
        help="explicitly permit a manifest containing every revision-pinned row",
    )
    return parser


def _corpus_ids_file(path: Path) -> set[int]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"metasyn_corpus_ids_file_unreadable:{path}") from exc
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(parsed, list):
        raise ValueError("metasyn_corpus_ids_file_requires_array_or_lines")
    ids: set[int] = set()
    for value in parsed:
        if isinstance(value, bool):
            raise ValueError("metasyn_corpus_ids_file_value_invalid")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("metasyn_corpus_ids_file_value_invalid") from exc
        if normalized < 0 or str(value).strip() != str(normalized):
            raise ValueError("metasyn_corpus_ids_file_value_invalid")
        ids.add(normalized)
    return ids


def _selected_metasyn_ids(args: argparse.Namespace) -> set[int] | None:
    selected = set(args.corpus_id)
    if args.corpus_ids_file is not None:
        selected.update(_corpus_ids_file(args.corpus_ids_file))
    if args.all_corpus_rows:
        if selected:
            raise ValueError("--all-corpus-rows cannot be combined with ID filters")
        return None
    if not selected:
        raise ValueError(
            "MetaSyn requires --corpus-id/--corpus-ids-file or explicit --all-corpus-rows"
        )
    return selected


def _write_outputs(
    *,
    output_dir: Path,
    manifest: Any,
    ledger: DiagnosticSourceLedger,
    force: bool,
    public_run_output: Path | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "native_source_manifest.json"
    ledger_path = output_dir / "diagnostic_source_ledger.json"
    run_path = output_dir / "source_manifest_bridge_run.json"
    existing = [path.as_posix() for path in (manifest_path, ledger_path, run_path) if path.exists()]
    if existing and not force:
        raise ValueError(f"source_bridge_outputs_exist:{existing}")
    atomic_write_json(manifest_path, manifest, force=force)
    atomic_write_json(ledger_path, ledger, force=force)
    run_payload = {
        "source_manifest_bridge_run_version": "2",
        "corpus_kind": ledger.corpus_kind,
        "question_id": ledger.question_id,
        "dataset_version": ledger.dataset_version,
        "source_revision": ledger.source_revision,
        "license_status": ledger.license_status,
        "selection_scope": ledger.selection_scope,
        "diagnostic_only": True,
        "labels_previously_opened": True,
        "pristine_final_holdout_eligible": False,
        "native_source_manifest_content_sha256": hash_canonical(manifest),
        "native_source_manifest_file_sha256": sha256_file(manifest_path),
        "diagnostic_source_ledger_content_sha256": ledger.ledger_sha256,
        "diagnostic_source_ledger_file_sha256": sha256_file(ledger_path),
        "records": ledger.source_records,
        "native_manifest_records": ledger.native_manifest_records,
        "source_available_records": ledger.source_available_records,
        "source_absent_records": ledger.source_absent_records,
        "manifest_excluded_records": ledger.manifest_excluded_records,
        "content_scope_counts": ledger.content_scope_counts,
        "verified_source_artifacts": [
            artifact.model_dump(mode="json") for artifact in ledger.artifacts
        ],
    }
    run = {**run_payload, "run_sha256": hash_canonical(run_payload)}
    atomic_write_json(run_path, run, force=force)
    if public_run_output is not None:
        atomic_write_json(public_run_output, run, force=force)
    return {
        "manifest": manifest_path.as_posix(),
        "ledger": ledger_path.as_posix(),
        "run": run_path.as_posix(),
        "public_run": (public_run_output.as_posix() if public_run_output is not None else None),
        "ledger_sha256": ledger.ledger_sha256,
        "run_sha256": run["run_sha256"],
        "records": ledger.source_records,
        "native_manifest_records": ledger.native_manifest_records,
        "diagnostic_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    if args.corpus_kind == "antiox":
        manifest, ledger = build_antiox_native_source_bridge(
            question_id=args.question_id,
            papers_path=args.papers,
            source_lines_path=args.source_lines,
            repository_root=repository_root,
            scope=args.scope,
            expected_papers_sha256=args.expected_papers_sha256,
            expected_source_lines_sha256=args.expected_source_lines_sha256,
        )
    else:
        manifest, ledger = build_metasyn_native_source_bridge(
            question_id=args.question_id,
            corpus_manifest_path=args.corpus_manifest,
            repository_root=repository_root,
            corpus_ids=_selected_metasyn_ids(args),
        )
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    public_run_output = args.public_run_output
    if public_run_output is not None and not public_run_output.is_absolute():
        public_run_output = repository_root / public_run_output
    result = _write_outputs(
        output_dir=output_dir,
        manifest=manifest,
        ledger=ledger,
        force=args.force,
        public_run_output=public_run_output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
