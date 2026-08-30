"""Helpers that keep private-cache integration tests out of the public offline contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TYPED_PILOT_STALE_CODES = frozenset({"metasyn_pilot_prepare_external_replay_mismatch"})
HOSTED_ADAPTER_STALE_CODES = frozenset({"metasyn_hosted_adapter_upstream_stale"})


def require_private_cache(*relative: str) -> Path:
    """Skip unless every named ignored path exists; never fabricate a substitute."""

    missing = [item for item in relative if not (REPOSITORY_ROOT / item).exists()]
    if missing:
        pytest.skip(f"private local cache artifact unavailable in this checkout: {missing[0]}")
    return REPOSITORY_ROOT


def skip_when_historical_replay_is_stale[T](
    build: Callable[[], T],
    *,
    stale_errors: tuple[type[BaseException], ...],
    stale_codes: frozenset[str],
) -> T:
    """Run a frozen-bundle replay; convert only the documented stale codes into a skip."""

    try:
        return build()
    except stale_errors as exc:
        if str(exc) in stale_codes:
            pytest.skip(
                "historical private bundle is stale under the current pipeline "
                f"({exc}); the identity-only staleness test pins this"
            )
        raise
