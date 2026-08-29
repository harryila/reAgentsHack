#!/usr/bin/env python3
"""Replay finalized MetaSyn outputs and materialize synthesis-yield receipts.

This command is provider-neutral and makes no model calls.  It externally replays the
complete finalized bounded runtime before deriving a private typed-graph/synthesis-yield
report and its identifier-free aggregate public summary.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import (
    OutputExistsError,
    canonical_json_bytes,
)
from literature_multiverse.metasyn_synthesis_yield import (
    evaluate_current_metasyn_synthesis_yield,
    validate_metasyn_synthesis_yield_public_summary,
)
from literature_multiverse.models import SHA256_RE

PRIVATE_OUTPUT_RELATIVE = Path(
    "data/cache/metasyn/synthesis-yield-v1/private-report.json"
)
PUBLIC_OUTPUT_RELATIVE = Path(
    "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json"
)


def _sha256_argument(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _rooted(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _canonical_output_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _fixed_repository_output(
    *,
    requested: Path,
    repository_root: Path,
    expected_relative: Path,
    output_kind: str,
) -> Path:
    """Resolve one fixed output without following a repository-internal symlink."""

    if requested.is_absolute() or requested != expected_relative:
        raise ValueError(
            "metasyn_synthesis_"
            f"{output_kind}_output_must_equal_repository_relative_path:"
            f"{expected_relative.as_posix()}"
        )

    destination = repository_root / expected_relative
    current = repository_root
    for part in expected_relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"metasyn_synthesis_{output_kind}_output_symlink_forbidden"
            )
        if current != destination and os.path.lexists(current) and not current.is_dir():
            raise ValueError(
                f"metasyn_synthesis_{output_kind}_output_ancestor_not_directory"
            )

    try:
        destination.resolve(strict=False).relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"metasyn_synthesis_{output_kind}_output_escapes_repository"
        ) from exc
    return destination


def _preflight_outputs(*, private_output: Path, public_output: Path) -> None:
    private_canonical = _canonical_output_path(private_output)
    public_canonical = _canonical_output_path(public_output)
    if private_canonical == public_canonical:
        raise ValueError("metasyn_synthesis_private_and_public_outputs_must_differ")
    existing = [
        path
        for path in (private_output, public_output)
        if os.path.lexists(path)
    ]
    if existing:
        raise OutputExistsError(
            ",".join(sorted(path.as_posix() for path in existing))
        )


def _stage_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _install_staged_no_overwrite(*, temporary: Path, destination: Path) -> None:
    try:
        # A same-directory hard link atomically publishes the fully fsynced temporary
        # file and fails if any file or symlink already occupies the destination.
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise OutputExistsError(destination.as_posix()) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_output_pair_no_overwrite(
    *, private_output: Path, private_value: Any, public_output: Path, public_value: Any
) -> None:
    """Atomically install each output and roll back this pair on partial failure."""

    _preflight_outputs(
        private_output=private_output,
        public_output=public_output,
    )
    staged: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, int, int]] = []
    try:
        staged.append((_stage_json(private_output, private_value), private_output))
        staged.append((_stage_json(public_output, public_value), public_output))
        for temporary, destination in staged:
            _install_staged_no_overwrite(
                temporary=temporary,
                destination=destination,
            )
            stat = destination.stat()
            installed.append((destination, stat.st_dev, stat.st_ino))
        for parent in sorted({path.parent for _, path in staged}, key=str):
            _fsync_directory(parent)
    except BaseException:
        # Delete only a path that is still the exact inode installed by this call.
        for destination, device, inode in reversed(installed):
            try:
                current = destination.stat()
            except FileNotFoundError:
                continue
            if current.st_dev == device and current.st_ino == inode:
                destination.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--runtime-workspace",
        type=Path,
        required=True,
        help="Workspace containing the finalized 10-question/32-row private runtime.",
    )
    parser.add_argument(
        "--pilot-workspace",
        type=Path,
        required=True,
        help="Workspace containing the externally replayable adapter/prepare bundle.",
    )
    parser.add_argument(
        "--expected-execution-bundle-sha256",
        type=_sha256_argument,
        required=True,
        help="Out-of-band identity anchor for the frozen bounded execution bundle.",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PRIVATE_OUTPUT_RELATIVE,
        help=(
            "Fixed repository-relative destination for the self-hashed private "
            f"yield report (must be {PRIVATE_OUTPUT_RELATIVE.as_posix()})."
        ),
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=PUBLIC_OUTPUT_RELATIVE,
        help=(
            "Fixed repository-relative destination for the identifier-free "
            f"aggregate summary (must be {PUBLIC_OUTPUT_RELATIVE.as_posix()})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    private_output = _fixed_repository_output(
        requested=args.private_output,
        repository_root=root,
        expected_relative=PRIVATE_OUTPUT_RELATIVE,
        output_kind="private",
    )
    public_output = _fixed_repository_output(
        requested=args.public_output,
        repository_root=root,
        expected_relative=PUBLIC_OUTPUT_RELATIVE,
        output_kind="public",
    )
    _preflight_outputs(
        private_output=private_output,
        public_output=public_output,
    )
    report, public = evaluate_current_metasyn_synthesis_yield(
        repository_root=root,
        runtime_workspace=_rooted(args.runtime_workspace, root),
        pilot_workspace=_rooted(args.pilot_workspace, root),
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    validate_metasyn_synthesis_yield_public_summary(summary=public, report=report)
    # Re-check after the potentially long read-only replay so a newly inserted
    # repository-internal symlink cannot redirect either materialization path.
    _fixed_repository_output(
        requested=args.private_output,
        repository_root=root,
        expected_relative=PRIVATE_OUTPUT_RELATIVE,
        output_kind="private",
    )
    _fixed_repository_output(
        requested=args.public_output,
        repository_root=root,
        expected_relative=PUBLIC_OUTPUT_RELATIVE,
        output_kind="public",
    )
    _write_output_pair_no_overwrite(
        private_output=private_output,
        private_value=report,
        public_output=public_output,
        public_value=public,
    )
    print(
        json.dumps(
            {
                "stage": "metasyn-typed-synthesis-yield",
                "status": report.status,
                "private_output": private_output.as_posix(),
                "private_report_sha256": report.report_sha256,
                "public_output": public_output.as_posix(),
                "public_summary_sha256": public.summary_sha256,
                "provider_calls_made": False,
                "reference_fields_opened": False,
                "accuracy_or_direction_reported": False,
                "claim_release_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
