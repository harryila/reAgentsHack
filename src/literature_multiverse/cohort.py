"""Representation-independent canonical identity for a primary finding cohort."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from typing import Any

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import FindingRow


class CohortContractError(ValueError):
    """Primary rows cannot be converted to one lossless canonical representation."""


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CohortContractError("cohort_datetime_requires_timezone")
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CohortContractError("cohort_forbids_nonfinite_float")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    # PyArrow-backed pandas cells may expose ndarray/to-list or NumPy scalar APIs.
    if hasattr(value, "tolist") and callable(value.tolist):
        return _value(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _value(value.item())
        except (TypeError, ValueError):
            pass
    try:
        missing = bool(value != value)
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    raise CohortContractError(f"cohort_value_not_jsonable:{type(value).__name__}")


def _moderators(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("moderators")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CohortContractError("cohort_moderators_json_invalid") from exc
    if raw is None:
        result: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        result = {str(key): _value(value) for key, value in raw.items()}
    else:
        raise CohortContractError("cohort_moderators_invalid")
    for key, value in row.items():
        if not str(key).startswith("mod__"):
            continue
        name = str(key).removeprefix("mod__")
        # Paper-summary/remap columns are derived analysis inputs, not base FindingRow fields.
        if "__" in name:
            continue
        normalized = _value(value)
        if name in result and result[name] != normalized:
            raise CohortContractError(f"cohort_moderator_representation_conflict:{name}")
        result[name] = normalized
    return dict(sorted(result.items()))


def canonical_primary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize nested/flattened ledger rows and sort by immutable finding identity."""

    canonical: list[dict[str, Any]] = []
    field_names = tuple(FindingRow.model_fields)
    for source in rows:
        finding_id = source.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise CohortContractError("cohort_row_missing_finding_id")
        row = {
            name: (_moderators(source) if name == "moderators" else _value(source.get(name)))
            for name in field_names
        }
        canonical.append(row)
    canonical.sort(key=lambda row: str(row["finding_id"]))
    identifiers = [str(row["finding_id"]) for row in canonical]
    if len(identifiers) != len(set(identifiers)):
        raise CohortContractError("cohort_finding_ids_not_unique")
    return canonical


def cohort_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete canonical primary cohort, independent of parquet representation."""

    return hash_canonical(canonical_primary_rows(rows))


__all__ = [
    "CohortContractError",
    "canonical_primary_rows",
    "cohort_sha256",
]
