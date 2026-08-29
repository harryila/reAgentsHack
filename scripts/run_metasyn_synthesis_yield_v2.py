#!/usr/bin/env python3
"""Replay a finalized hosted MetaSyn run and write v2 yield-only artifacts.

This command makes no provider calls.  It externally validates the hosted runtime,
provider-neutral adapter/report when present, and frozen prepare bundle before replaying
original-source grounding, cohort reconciliation, compatibility grouping, and synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import OutputExistsError, canonical_json_bytes
from literature_multiverse.metasyn_synthesis_yield_v2 import (
    evaluate_current_metasyn_synthesis_yield_v2,
    validate_metasyn_synthesis_yield_v2_public_summary,
)
from literature_multiverse.models import SHA256_RE

PRIVATE_OUTPUT_RELATIVE = Path(
    "data/cache/metasyn/synthesis-yield-v2/private-report.json"
)
PUBLIC_OUTPUT_RELATIVE = Path(
    "artifacts/diagnostics/metasyn-synthesis-yield-v2/summary.json"
)


def _sha256_argument(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _rooted(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _fixed_repository_output(
    *,
    requested: Path,
    repository_root: Path,
    expected_relative: Path,
    output_kind: str,
) -> Path:
    if requested.is_absolute() or requested != expected_relative:
        raise ValueError(
            "metasyn_synthesis_v2_"
            f"{output_kind}_output_must_equal_repository_relative_path:"
            f"{expected_relative.as_posix()}"
        )
    destination = repository_root / expected_relative
    current = repository_root
    for part in expected_relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"metasyn_synthesis_v2_{output_kind}_output_symlink_forbidden"
            )
        if current != destination and os.path.lexists(current) and not current.is_dir():
            raise ValueError(
                f"metasyn_synthesis_v2_{output_kind}_ancestor_not_directory"
            )
    try:
        destination.resolve(strict=False).relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_output_escapes_repository"
        ) from exc
    return destination


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_repository_root(repository_root: Path) -> int:
    try:
        return os.open(repository_root, _directory_open_flags())
    except OSError as exc:
        raise ValueError(
            "metasyn_synthesis_v2_repository_root_open_failed"
        ) from exc


def _raise_unsafe_ancestor(
    *, parent_fd: int, name: str, output_kind: str, cause: OSError
) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_ancestor_changed"
        ) from cause
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_output_symlink_forbidden"
        ) from cause
    raise ValueError(
        f"metasyn_synthesis_v2_{output_kind}_ancestor_not_directory"
    ) from cause


def _open_output_parent_at(
    *,
    repository_root_fd: int,
    relative: Path,
    output_kind: str,
    create: bool,
) -> int | None:
    """Open an output parent beneath a pinned repository root without symlinks."""

    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_output_escapes_repository"
        )
    current_fd = os.dup(repository_root_fd)
    try:
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if not create:
                    os.close(current_fd)
                    return None
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o755, dir_fd=current_fd)
                try:
                    next_fd = os.open(
                        part,
                        _directory_open_flags(),
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    _raise_unsafe_ancestor(
                        parent_fd=current_fd,
                        name=part,
                        output_kind=output_kind,
                        cause=exc,
                    )
            except OSError as exc:
                _raise_unsafe_ancestor(
                    parent_fd=current_fd,
                    name=part,
                    output_kind=output_kind,
                    cause=exc,
                )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        with suppress(OSError):
            os.close(current_fd)
        raise


def _destination_exists_at(*, parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _preflight_outputs_at(
    *,
    repository_root_fd: int,
    outputs: tuple[tuple[Path, Path, str], ...],
) -> None:
    existing: list[str] = []
    for relative, display_path, output_kind in outputs:
        parent_fd = _open_output_parent_at(
            repository_root_fd=repository_root_fd,
            relative=relative,
            output_kind=output_kind,
            create=False,
        )
        if parent_fd is None:
            continue
        try:
            if _destination_exists_at(
                parent_fd=parent_fd, name=relative.name
            ):
                existing.append(display_path.as_posix())
        finally:
            os.close(parent_fd)
    if existing:
        raise OutputExistsError(",".join(sorted(existing)))


def _stage_json_at(*, parent_fd: int, destination_name: str, value: Any) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = -1
    temporary_name = ""
    for _ in range(128):
        temporary_name = (
            f".{destination_name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_fd,
            )
            break
        except FileExistsError:
            continue
    if descriptor < 0:
        raise RuntimeError("metasyn_synthesis_v2_temporary_name_exhausted")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        raise
    return temporary_name


def _assert_parent_still_repository_bound(
    *,
    repository_root_fd: int,
    relative: Path,
    output_kind: str,
    expected_parent_fd: int,
) -> int:
    """Reopen a parent from the root and require the staged inode is still bound."""

    current_parent_fd = _open_output_parent_at(
        repository_root_fd=repository_root_fd,
        relative=relative,
        output_kind=output_kind,
        create=False,
    )
    if current_parent_fd is None:
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_ancestor_changed"
        )
    expected = os.fstat(expected_parent_fd)
    current = os.fstat(current_parent_fd)
    if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
        os.close(current_parent_fd)
        raise ValueError(
            f"metasyn_synthesis_v2_{output_kind}_ancestor_changed"
        )
    return current_parent_fd


def _install_staged_no_overwrite_at(
    *,
    repository_root_fd: int,
    parent_fd: int,
    temporary_name: str,
    relative: Path,
    display_path: Path,
    output_kind: str,
) -> os.stat_result:
    current_parent_fd = _assert_parent_still_repository_bound(
        repository_root_fd=repository_root_fd,
        relative=relative,
        output_kind=output_kind,
        expected_parent_fd=parent_fd,
    )
    try:
        staged_metadata = os.stat(
            temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        try:
            os.link(
                temporary_name,
                relative.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=current_parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise OutputExistsError(display_path.as_posix()) from exc
        return staged_metadata
    finally:
        os.close(current_parent_fd)


def _unlink_if_same_inode_at(
    *, parent_fd: int, name: str, device: int, inode: int
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == (device, inode):
        os.unlink(name, dir_fd=parent_fd)


def _unlink_missing_ok_at(*, parent_fd: int, name: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(name, dir_fd=parent_fd)


def _write_output_pair_no_overwrite(
    *,
    repository_root_fd: int,
    private_output: Path,
    private_value: Any,
    public_output: Path,
    public_value: Any,
) -> None:
    if PRIVATE_OUTPUT_RELATIVE == PUBLIC_OUTPUT_RELATIVE:
        raise ValueError("metasyn_synthesis_v2_outputs_must_differ")
    outputs = (
        (
            PRIVATE_OUTPUT_RELATIVE,
            private_output,
            "v2_private",
            private_value,
        ),
        (
            PUBLIC_OUTPUT_RELATIVE,
            public_output,
            "v2_public",
            public_value,
        ),
    )
    _preflight_outputs_at(
        repository_root_fd=repository_root_fd,
        outputs=tuple((relative, display, kind) for relative, display, kind, _ in outputs),
    )
    staged: list[tuple[int, str, Path, Path, str]] = []
    installed: list[tuple[int, str, int, int]] = []
    parent_fds: list[int] = []
    try:
        for relative, display, output_kind, value in outputs:
            parent_fd = _open_output_parent_at(
                repository_root_fd=repository_root_fd,
                relative=relative,
                output_kind=output_kind,
                create=True,
            )
            assert parent_fd is not None
            parent_fds.append(parent_fd)
            temporary_name = _stage_json_at(
                parent_fd=parent_fd,
                destination_name=relative.name,
                value=value,
            )
            staged.append(
                (parent_fd, temporary_name, relative, display, output_kind)
            )
        for parent_fd, temporary_name, relative, display, output_kind in staged:
            metadata = _install_staged_no_overwrite_at(
                repository_root_fd=repository_root_fd,
                parent_fd=parent_fd,
                temporary_name=temporary_name,
                relative=relative,
                display_path=display,
                output_kind=output_kind,
            )
            installed.append(
                (parent_fd, relative.name, metadata.st_dev, metadata.st_ino)
            )
        for parent_fd in parent_fds:
            os.fsync(parent_fd)
    except BaseException:
        for parent_fd, name, device, inode in reversed(installed):
            with suppress(OSError):
                _unlink_if_same_inode_at(
                    parent_fd=parent_fd,
                    name=name,
                    device=device,
                    inode=inode,
                )
        for parent_fd in parent_fds:
            with suppress(OSError):
                os.fsync(parent_fd)
        raise
    finally:
        for parent_fd, temporary_name, _, _, _ in staged:
            with suppress(OSError):
                _unlink_missing_ok_at(
                    parent_fd=parent_fd, name=temporary_name
                )
        for parent_fd in parent_fds:
            with suppress(OSError):
                os.fsync(parent_fd)
            os.close(parent_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--hosted-runtime-workspace",
        type=Path,
        required=True,
        help="Workspace containing the finalized hosted 10-question/32-row runtime.",
    )
    parser.add_argument(
        "--pilot-workspace",
        type=Path,
        required=True,
        help="Workspace containing the externally replayable prepare bundle.",
    )
    parser.add_argument(
        "--expected-execution-bundle-sha256",
        type=_sha256_argument,
        required=True,
        help="Out-of-band SHA-256 anchor for the hosted execution bundle.",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=PRIVATE_OUTPUT_RELATIVE,
        help=(
            "Fixed repository-relative destination for the identifier-bearing report "
            f"(must be {PRIVATE_OUTPUT_RELATIVE.as_posix()})."
        ),
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=PUBLIC_OUTPUT_RELATIVE,
        help=(
            "Fixed repository-relative destination for the aggregate-only summary "
            f"(must be {PUBLIC_OUTPUT_RELATIVE.as_posix()})."
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
        output_kind="v2_private",
    )
    public_output = _fixed_repository_output(
        requested=args.public_output,
        repository_root=root,
        expected_relative=PUBLIC_OUTPUT_RELATIVE,
        output_kind="v2_public",
    )
    expected_root = root.stat()
    repository_root_fd = _open_repository_root(root)
    opened_root = os.fstat(repository_root_fd)
    if (expected_root.st_dev, expected_root.st_ino) != (
        opened_root.st_dev,
        opened_root.st_ino,
    ):
        os.close(repository_root_fd)
        raise ValueError("metasyn_synthesis_v2_repository_root_changed")
    try:
        _preflight_outputs_at(
            repository_root_fd=repository_root_fd,
            outputs=(
                (PRIVATE_OUTPUT_RELATIVE, private_output, "v2_private"),
                (PUBLIC_OUTPUT_RELATIVE, public_output, "v2_public"),
            ),
        )
        report, public = evaluate_current_metasyn_synthesis_yield_v2(
            repository_root=root,
            hosted_runtime_workspace=_rooted(
                args.hosted_runtime_workspace, root
            ),
            pilot_workspace=_rooted(args.pilot_workspace, root),
            expected_execution_bundle_sha256=(
                args.expected_execution_bundle_sha256
            ),
        )
        validate_metasyn_synthesis_yield_v2_public_summary(
            summary=public, report=report
        )
        # Keep the descriptive pathname checks, then perform the transaction itself
        # solely through no-follow descriptors rooted at the pinned repository inode.
        _fixed_repository_output(
            requested=args.private_output,
            repository_root=root,
            expected_relative=PRIVATE_OUTPUT_RELATIVE,
            output_kind="v2_private",
        )
        _fixed_repository_output(
            requested=args.public_output,
            repository_root=root,
            expected_relative=PUBLIC_OUTPUT_RELATIVE,
            output_kind="v2_public",
        )
        _write_output_pair_no_overwrite(
            repository_root_fd=repository_root_fd,
            private_output=private_output,
            private_value=report,
            public_output=public_output,
            public_value=public,
        )
    finally:
        os.close(repository_root_fd)
    print(
        json.dumps(
            {
                "stage": "metasyn-hosted-synthesis-yield-v2",
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
