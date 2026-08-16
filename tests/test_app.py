from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
APP_SPEC = importlib.util.spec_from_file_location("literature_multiverse_streamlit_app", APP_PATH)
assert APP_SPEC is not None and APP_SPEC.loader is not None
APP_MODULE = importlib.util.module_from_spec(APP_SPEC)
sys.modules[APP_SPEC.name] = APP_MODULE
APP_SPEC.loader.exec_module(APP_MODULE)

ALL_DEMO_PATHS = APP_MODULE.ALL_DEMO_PATHS
BUNDLED_PATHS = APP_MODULE.BUNDLED_PATHS
BundleValidationError = APP_MODULE.BundleValidationError
_question_from_argv = APP_MODULE._question_from_argv
_validate_artifact_inventory = APP_MODULE._validate_artifact_inventory


def _write_inventory(root: Path) -> dict[str, object]:
    artifact_rows: list[dict[str, object]] = []
    for relative in sorted(ALL_DEMO_PATHS):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = b"{}\n" if target.suffix == ".json" else f"fixture:{relative}\n".encode()
        target.write_bytes(payload)
        if relative in BUNDLED_PATHS:
            artifact_rows.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "rows": None,
                }
            )
    return {"artifacts": artifact_rows}


def test_app_inventory_validation_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    manifest = _write_inventory(tmp_path)
    _validate_artifact_inventory(tmp_path, manifest)

    target = tmp_path / "analysis" / "headline.json"
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(BundleValidationError, match=r"artifact_(size|hash)_mismatch"):
        _validate_artifact_inventory(tmp_path, manifest)

    manifest = _write_inventory(tmp_path)
    (tmp_path / "unexpected.txt").write_text("not allowlisted", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="bundle_inventory_mismatch"):
        _validate_artifact_inventory(tmp_path, manifest)


def test_app_question_argument_is_strict() -> None:
    assert _question_from_argv(["--question", "fixture-a"]) == "fixture-a"
    assert _question_from_argv(["--question=fixture-b-m4"]) == "fixture-b-m4"
    assert _question_from_argv([]) is None
    with pytest.raises(BundleValidationError, match="invalid --question"):
        _question_from_argv(["--question", "../../escape"])
    with pytest.raises(BundleValidationError, match="requires a value"):
        _question_from_argv(["--question"])


def test_app_module_has_no_network_or_provider_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("requests", "urllib", "httpx", "anthropic", "paperclip", "providers")
    for name in forbidden:
        assert f"import {name}" not in source
        assert f"from {name}" not in source
