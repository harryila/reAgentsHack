"""Live Paperclip invocation conventions verified against the installed CLI build.

The subprocess boundary in ``paperclip_cli`` stays schema-agnostic; this module owns the
observed output conventions instead: search prints ``Found N papers  [s_...]`` to stdout,
``map`` prints a live progress stream containing ``m_...``, and the only pinned
machine-readable artifacts are the ``paperclip results <id> --save`` exports (CSV for a
search, the ``Map results`` text format for a map).  The CLI build in use has no ``--json``
flag on ``search`` or ``map``; G1b live probes proved these paths on 2026-08-15.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from literature_multiverse.paperclip_cli import (
    PaperclipBoundaryError,
    require_success,
    run_paperclip,
)

_SEARCH_ID_RE = re.compile(
    r"(?:Found\s+\d+\s+papers?|(?:Search\s+)?results?)\s*\[(?P<result_id>s_[0-9A-Za-z]+)\]"
)
_MAP_ID_RE = re.compile(r"\b(?P<map_id>m_[0-9a-f]{6,})\b")


def paperclip_executable() -> str:
    """Locate the Paperclip CLI without ever falling back to a shell."""

    found = shutil.which("paperclip")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/paperclip")
    if os.path.exists(fallback):
        return fallback
    raise PaperclipBoundaryError("paperclip executable not found on PATH or ~/.local/bin")


def search_id_from_stdout(stdout: bytes | str) -> str:
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
    match = _SEARCH_ID_RE.search(text)
    if match is None:
        raise PaperclipBoundaryError("search stdout contained no saved s_ result id")
    return match.group("result_id")


def map_id_from_stdout(stdout: bytes | str) -> str:
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
    match = _MAP_ID_RE.search(text)
    if match is None:
        raise PaperclipBoundaryError("map output contained no m_ result id")
    return match.group("map_id")


@dataclass(frozen=True, slots=True)
class LiveSearch:
    """One archived live search plus its machine-readable CSV export."""

    result_id: str
    csv_path: Path
    csv_bytes: bytes
    artifacts: tuple[Path, ...]


def live_search_to_csv(
    query: str,
    *,
    sources: tuple[str, ...] | list[str],
    archive_dir: Path,
    archive_stem: str,
    limit: int | None = None,
    use_all: bool = True,
    exclude_article_types: tuple[str, ...] | list[str] = (),
    exact: bool = False,
    force: bool = False,
) -> LiveSearch:
    """Run one search, then export its saved result set as the canonical CSV artifact."""

    executable = paperclip_executable()
    argv = [executable, "search", "-s", ",".join(sources)]
    if use_all:
        argv.append("--all")
    if limit is not None:
        argv.extend(["-n", str(limit)])
    for article_type in exclude_article_types:
        argv.extend(["--exclude-article-type", article_type])
    if exact:
        argv.append("-e")
    argv.append(query)
    search_run = run_paperclip(
        argv,
        archive_dir=archive_dir,
        archive_stem=archive_stem,
        timeout_seconds=300,
        force=force,
    )
    search_attempt = require_success(search_run)
    result_id = search_id_from_stdout(search_attempt.stdout)
    csv_path = archive_dir / f"{archive_stem}.results.csv"
    if csv_path.exists() and not force:
        raise FileExistsError(f"refusing to replace archived search export: {csv_path}")
    export_run = run_paperclip(
        [executable, "results", result_id, "--save", str(csv_path)],
        archive_dir=archive_dir,
        archive_stem=f"{archive_stem}.export",
        timeout_seconds=300,
        force=force,
    )
    export_attempt = require_success(export_run)
    if not csv_path.exists():
        raise PaperclipBoundaryError(f"results export did not produce {csv_path}")
    return LiveSearch(
        result_id=result_id,
        csv_path=csv_path,
        csv_bytes=csv_path.read_bytes(),
        artifacts=(
            search_attempt.stdout_path,
            search_attempt.stderr_path,
            search_attempt.metadata_path,
            export_attempt.stdout_path,
            export_attempt.stderr_path,
            export_attempt.metadata_path,
            csv_path,
        ),
    )


@dataclass(frozen=True, slots=True)
class LiveMap:
    """One archived live map plus its pinned-format results export."""

    map_id: str
    results_path: Path
    artifacts: tuple[Path, ...]


def live_map_to_results_file(
    *,
    archive_dir: Path,
    archive_stem: str,
    from_result: str | None = None,
    schema_json: str | None = None,
    prompt: str | None = None,
    concurrency: int | None = None,
    resume_map_id: str | None = None,
    retry_failed: bool = False,
    timeout_seconds: float | None = None,
    force: bool = False,
) -> LiveMap:
    """Run (or resume) one map, then export the terminal set in the pinned parser format."""

    if bool(from_result) == bool(resume_map_id):
        raise PaperclipBoundaryError("exactly one of from_result or resume_map_id is required")
    executable = paperclip_executable()
    if resume_map_id is not None:
        argv = [executable, "map", "--resume", resume_map_id]
        if retry_failed:
            argv.append("--retry-failed")
    else:
        if not schema_json or not prompt:
            raise PaperclipBoundaryError("a fresh map requires schema_json and prompt")
        argv = [executable, "map", "--from", str(from_result)]
        # NOTE: `-j/--max-concurrent` (like the named workers) is gated to GXL testers on
        # this build — "Parallel map workers are currently limited to GXL testers."  The
        # server still parallelizes the default worker on its own.  Only pass an explicit
        # concurrency once tester access is confirmed.
        if concurrency is not None:
            argv.extend(["-j", str(concurrency)])
        argv.extend(["--output-schema", schema_json, prompt])
    map_run = run_paperclip(
        argv,
        archive_dir=archive_dir,
        archive_stem=archive_stem,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    map_attempt = require_success(map_run)
    map_id = (
        resume_map_id
        if resume_map_id is not None
        else map_id_from_stdout(map_attempt.stdout)
    )
    results_path = archive_dir / f"{archive_stem}.results.txt"
    if results_path.exists() and not force:
        raise FileExistsError(f"refusing to replace archived map export: {results_path}")
    export_run = run_paperclip(
        [executable, "results", map_id, "--save", str(results_path)],
        archive_dir=archive_dir,
        archive_stem=f"{archive_stem}.export",
        timeout_seconds=600,
        force=force,
    )
    export_attempt = require_success(export_run)
    if not results_path.exists():
        raise PaperclipBoundaryError(f"results export did not produce {results_path}")
    return LiveMap(
        map_id=map_id,
        results_path=results_path,
        artifacts=(
            map_attempt.stdout_path,
            map_attempt.stderr_path,
            map_attempt.metadata_path,
            export_attempt.stdout_path,
            export_attempt.stderr_path,
            export_attempt.metadata_path,
            results_path,
        ),
    )
