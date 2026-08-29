"""Shared, label-free production stopping semantics.

The verifier and retrospective question replay must not carry subtly different
definitions of a releasable prefix.  This small contract is intentionally ignorant
of audit outcomes and reference labels: callers first freeze their complete release
decision, then ask whether that state is the first state at which production stops.
"""

from __future__ import annotations

from collections.abc import Sequence

PRODUCTION_STOPPING_RULE = "stop_at_first_full_frozen_release_eligible_state"


def should_stop_at_full_frozen_release(
    *,
    release_status: str,
    blocking_reasons: Sequence[str] = (),
    active_action: bool = False,
) -> bool:
    """Return whether a fully joined frozen state is production-releasable.

    ``release_status`` is the output of all scientific, audit, and calibration gates.
    Adapter/corpus blockers are supplied separately because they are joined at the
    certificate boundary.  An in-progress human action always forces abstention.
    """

    if release_status not in {"released", "abstained"}:
        raise ValueError("production_stop_release_status_invalid")
    if any(not isinstance(reason, str) or not reason for reason in blocking_reasons):
        raise ValueError("production_stop_blocking_reason_invalid")
    return release_status == "released" and not blocking_reasons and not active_action


__all__ = [
    "PRODUCTION_STOPPING_RULE",
    "should_stop_at_full_frozen_release",
]
