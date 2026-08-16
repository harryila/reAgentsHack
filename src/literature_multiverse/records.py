"""Plain-Python views of parquet artifacts.

``pandas.read_parquet(...).to_dict(orient="records")`` returns numpy scalars and
``ndarray`` list columns, which are not JSON-serializable and break canonical hashing.
Every script that round-trips parquet rows into JSON artifacts must read through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def plain_value(value: Any) -> Any:
    """Convert pandas/numpy scalars and arrays into JSON-serializable Python values."""

    if isinstance(value, np.ndarray):
        return [plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value


def read_parquet_records(path: Path | str) -> list[dict[str, Any]]:
    return [
        {key: plain_value(value) for key, value in record.items()}
        for record in pd.read_parquet(path).to_dict(orient="records")
    ]


__all__ = ["plain_value", "read_parquet_records"]
