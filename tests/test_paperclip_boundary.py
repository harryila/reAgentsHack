"""Failure-classification contracts pinned by the live G1b probes on 2026-08-15."""

from __future__ import annotations

from literature_multiverse.paperclip_cli import classify_failure

# Observed verbatim: the CLI reports these failures on stdout and then EXITS ZERO.
BOGUS_RESULT_STDOUT = (
    b"ERR: map: Results not found: s_00000000\n"
    b"[exit 1]\n"
    b"Try: paperclip search <query>  or  paperclip lookup doi <doi>\n"
)
INVALID_SCHEMA_STDOUT = (
    b"ERR: map: invalid output schema: invalid Draft 2020-12 JSON Schema at "
    b"allOf.3.properties.type.anyOf: 'bogus_type' is not valid under any of the "
    b"given schemas\n[exit 1]\n"
)
GATED_WORKER_STDOUT = (
    b"\x1b[91m[error] Parallel map workers are currently limited to GXL testers."
    b"\x1b[0m\r\n"
)


def test_zero_exit_with_inband_err_is_a_failure() -> None:
    code, retryable = classify_failure(0, b"", BOGUS_RESULT_STDOUT)
    assert code == "PAPERCLIP_INBAND_ERROR"
    assert retryable is False


def test_zero_exit_invalid_schema_is_a_failure() -> None:
    code, retryable = classify_failure(0, b"", INVALID_SCHEMA_STDOUT)
    assert code == "PAPERCLIP_INBAND_ERROR"
    assert retryable is False


def test_ansi_wrapped_error_marker_is_detected() -> None:
    code, retryable = classify_failure(0, b"", GATED_WORKER_STDOUT)
    assert code == "PAPERCLIP_INBAND_ERROR"
    assert retryable is False


def test_clean_zero_exit_stays_success() -> None:
    stdout = b"Found 10 papers  [s_1304e91f]\n\n  1. Some Title\n"
    assert classify_failure(0, b"", stdout) == (None, False)


def test_map_success_output_with_err_free_payload_stays_success() -> None:
    stdout = (
        b"Map complete: 4/4 papers\nResults ID: m_2bc51e4b\n\n"
        b'  \xe2\x9c\x93 Some paper title\n    {"eligible": true}\n'
    )
    assert classify_failure(0, b"", stdout) == (None, False)


def test_inband_rate_limit_is_retryable() -> None:
    stdout = b"ERR: map: rate limit exceeded, try again shortly\n[exit 1]\n"
    code, retryable = classify_failure(0, b"", stdout)
    assert code == "PAPERCLIP_RATE_LIMIT"
    assert retryable is True


def test_nonzero_exit_still_classifies_from_stderr() -> None:
    code, retryable = classify_failure(2, b"connection reset by peer", b"")
    assert code == "PAPERCLIP_TRANSIENT"
    assert retryable is True
