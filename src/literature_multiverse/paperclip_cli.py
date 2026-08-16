"""Safe, auditable subprocess boundary for the Paperclip CLI.

This module deliberately knows nothing about Paperclip's response schema.  It runs an
argv vector (never a shell command), archives the provider bytes before a caller parses
them, and classifies only a small allowlist of transport failures as retryable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_KEY_VALUE_RE = re.compile(
    rb'(?i)((?:"?(?:api[_-]?key|authorization|token|secret)"?)\s*[:=]\s*"?)'
    rb'([^"\s,;}]+)'
)
_BEARER_RE = re.compile(rb"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_ANTHROPIC_KEY_RE = re.compile(rb"sk-ant-[A-Za-z0-9_-]+")
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_RATE_LIMIT_PATTERNS = (
    b"rate limit",
    b"rate_limit",
    b"too many requests",
    b"http 429",
    b"status 429",
)
_TRANSIENT_PATTERNS = (
    b"temporarily unavailable",
    b"service unavailable",
    b"connection reset",
    b"connection timed out",
    b"gateway timeout",
    b"http 502",
    b"http 503",
    b"http 504",
    b"status 502",
    b"status 503",
    b"status 504",
)


class PaperclipBoundaryError(RuntimeError):
    """Raised when the local subprocess boundary itself is used unsafely."""


@dataclass(frozen=True, slots=True)
class PaperclipAttempt:
    """One archived Paperclip invocation."""

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    started_at: str
    completed_at: str
    elapsed_seconds: float
    failure_code: str | None
    retryable: bool
    stdout_path: Path
    stderr_path: Path
    metadata_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.failure_code is None


@dataclass(frozen=True, slots=True)
class PaperclipRun:
    """A complete invocation, including any explicitly allowed retries."""

    attempts: tuple[PaperclipAttempt, ...]

    @property
    def final(self) -> PaperclipAttempt:
        return self.attempts[-1]

    @property
    def ok(self) -> bool:
        return self.final.ok


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of strings, never a shell command")
    materialized = tuple(argv)
    if not materialized:
        raise ValueError("argv must not be empty")
    for argument in materialized:
        if not isinstance(argument, str):
            raise TypeError("every argv item must be a string")
        if "\x00" in argument:
            raise ValueError("argv items must not contain NUL bytes")
    return materialized


def redact_bytes(value: bytes) -> bytes:
    """Redact recognizable credential shapes without consulting the environment."""

    value = _BEARER_RE.sub(rb"\1[REDACTED]", value)
    value = _KEY_VALUE_RE.sub(rb"\1[REDACTED]", value)
    return _ANTHROPIC_KEY_RE.sub(b"[REDACTED]", value)


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Return a log-safe argv copy.

    Secret values following conventional key/token flags are hidden.  The actual argv
    remains unchanged and is passed directly to ``subprocess.run``.
    """

    hidden_next = False
    redacted: list[str] = []
    for item in argv:
        lowered = item.casefold()
        if hidden_next:
            redacted.append("[REDACTED]")
            hidden_next = False
            continue
        if lowered in {"--api-key", "--token", "--secret", "--authorization"}:
            redacted.append(item)
            hidden_next = True
            continue
        if any(marker in lowered for marker in ("api_key=", "api-key=", "token=", "secret=")):
            prefix = item.split("=", 1)[0]
            redacted.append(f"{prefix}=[REDACTED]")
            continue
        redacted.append(redact_bytes(item.encode("utf-8", "replace")).decode("utf-8"))
    return redacted


_ANSI_SGR_RE = re.compile(rb"\x1b\[[0-9;]*m")
_INBAND_ERROR_RE = re.compile(rb"(?m)^[\s\x00-\x1f]*(?:ERR:|\[error\])")


def classify_failure(
    returncode: int, stderr: bytes, stdout: bytes = b""
) -> tuple[str | None, bool]:
    """Map an execution result to a stable failure code and retry decision.

    The installed CLI build reports many failures **in band**: it prints an ``ERR:`` (or
    ``[error]``) line plus a literal ``[exit N]`` marker on stdout and then exits 0.  The
    live G1b probes pinned this on 2026-08-15 (bogus result IDs and invalid schemas both
    returned exit code 0).  A zero exit therefore only counts as success when no in-band
    error marker leads the output.
    """

    stdout = _ANSI_SGR_RE.sub(b"", stdout)
    stderr = _ANSI_SGR_RE.sub(b"", stderr)
    lowered_err = stderr.lower()
    combined = stdout + b"\n" + stderr
    if returncode == 0:
        inband = _INBAND_ERROR_RE.search(stdout) or _INBAND_ERROR_RE.search(stderr)
        if not inband:
            return None, False
        lowered_all = combined.lower()
        if any(pattern in lowered_all for pattern in _RATE_LIMIT_PATTERNS):
            return "PAPERCLIP_RATE_LIMIT", True
        if any(pattern in lowered_all for pattern in _TRANSIENT_PATTERNS):
            return "PAPERCLIP_TRANSIENT", True
        return "PAPERCLIP_INBAND_ERROR", False
    if any(pattern in lowered_err for pattern in _RATE_LIMIT_PATTERNS):
        return "PAPERCLIP_RATE_LIMIT", True
    if any(pattern in lowered_err for pattern in _TRANSIENT_PATTERNS):
        return "PAPERCLIP_TRANSIENT", True
    return "PAPERCLIP_NONZERO_EXIT", False


def _write_new(path: Path, content: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace archived Paperclip output: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _archive_attempt(
    *,
    archive_dir: Path,
    archive_stem: str,
    attempt_number: int,
    argv: tuple[str, ...],
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    failure_code: str | None,
    retryable: bool,
    force: bool,
) -> tuple[Path, Path, Path]:
    safe_stem = _SAFE_STEM_RE.sub("-", archive_stem).strip(".-") or "paperclip"
    prefix = f"{safe_stem}.attempt-{attempt_number:02d}"
    stdout_path = archive_dir / f"{prefix}.stdout"
    stderr_path = archive_dir / f"{prefix}.stderr"
    metadata_path = archive_dir / f"{prefix}.json"

    # Provider stdout is the scientific source artifact.  Redaction happens before disk;
    # ordinary map/search output is otherwise byte-for-byte unchanged.
    archived_stdout = redact_bytes(stdout)
    archived_stderr = redact_bytes(stderr)
    _write_new(stdout_path, archived_stdout, force=force)
    _write_new(stderr_path, archived_stderr, force=force)
    metadata = {
        "argv": redact_argv(argv),
        "returncode": returncode,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "failure_code": failure_code,
        "retryable": retryable,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "stdout_bytes": len(archived_stdout),
        "stderr_bytes": len(archived_stderr),
    }
    _write_new(
        metadata_path,
        (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(),
        force=force,
    )
    return stdout_path, stderr_path, metadata_path


def run_paperclip(
    argv: Sequence[str],
    *,
    archive_dir: str | Path,
    archive_stem: str,
    timeout_seconds: float | None = None,
    max_retries: int = 0,
    retry_delay_seconds: float = 1.0,
    force: bool = False,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> PaperclipRun:
    """Run and archive Paperclip using an argv-only subprocess.

    ``max_retries`` applies only to the explicit rate-limit/transient allowlist above.
    A successful command whose output later fails parsing is never retried by this layer.
    ``runner`` exists for offline unit tests and must follow ``subprocess.run`` semantics.
    """

    safe_argv = _validate_argv(argv)
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    destination = Path(archive_dir)
    attempts: list[PaperclipAttempt] = []

    for attempt_number in range(1, max_retries + 2):
        started_at = _iso_now()
        monotonic_start = time.monotonic()
        try:
            completed = runner(
                list(safe_argv),
                shell=False,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=None if env is None else dict(env),
            )
            stdout = bytes(completed.stdout or b"")
            stderr = bytes(completed.stderr or b"")
            returncode = int(completed.returncode)
            failure_code, retryable = classify_failure(returncode, stderr, stdout)
        except FileNotFoundError as exc:
            stdout = b""
            stderr = str(exc).encode("utf-8", "replace")
            returncode = 127
            failure_code, retryable = "PAPERCLIP_EXECUTABLE_NOT_FOUND", False
        except subprocess.TimeoutExpired as exc:
            stdout = bytes(exc.stdout or b"")
            stderr = bytes(exc.stderr or b"")
            returncode = 124
            failure_code, retryable = "PAPERCLIP_TIMEOUT", False

        elapsed_seconds = time.monotonic() - monotonic_start
        completed_at = _iso_now()
        stdout_path, stderr_path, metadata_path = _archive_attempt(
            archive_dir=destination,
            archive_stem=archive_stem,
            attempt_number=attempt_number,
            argv=safe_argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed_seconds,
            failure_code=failure_code,
            retryable=retryable,
            force=force,
        )
        attempt = PaperclipAttempt(
            argv=safe_argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed_seconds,
            failure_code=failure_code,
            retryable=retryable,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=metadata_path,
        )
        attempts.append(attempt)
        if attempt.ok or not retryable or attempt_number > max_retries:
            break
        if retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)

    return PaperclipRun(tuple(attempts))


def require_success(run: PaperclipRun) -> PaperclipAttempt:
    """Return the final attempt or raise with its stable failure code."""

    final = run.final
    if not final.ok:
        raise PaperclipBoundaryError(
            f"Paperclip failed with {final.failure_code} (exit {final.returncode}); "
            f"see {final.stderr_path}"
        )
    return final
