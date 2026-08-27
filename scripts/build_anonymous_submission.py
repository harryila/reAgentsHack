"""Build a deterministic, scrubbed source artifact for anonymous review.

The bundle is intentionally assembled from an allowlist.  It excludes repository
history, local corpora, caches, provider transcripts, credentials, private labels,
and planning notes.  The command fails closed on symlinks, identity markers,
host-specific paths, e-mail addresses, or high-confidence credential patterns.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "submission"
ARCHIVE_NAME = "literature-multiverse-anonymous-source.zip"
MANIFEST_NAME = "anonymous-source-manifest.json"
ARCHIVE_ROOT = "literature-multiverse-anonymous-source"

ALLOWLIST_FILES = (
    ".python-version",
    "README.md",
    "configs/questions/fixture-a.yaml",
    "pyproject.toml",
    "uv.lock",
    "scripts/assess_claim_release.py",
    "scripts/audit_local_corpora.py",
    "scripts/build_anonymous_submission.py",
    "scripts/calibrate_risk_gate.py",
    "scripts/evaluate_closed_corpus.py",
    "scripts/metasyn_benchmark.py",
    "scripts/optimize_prompts.py",
    "scripts/prepare_human_review_packet.py",
    "scripts/simulate_budgeted_verification.py",
    "scripts/simulate_meta_analysis.py",
    "scripts/simulate_risk_calibration.py",
    "scripts/summarize_gepa_pilot.py",
    "scripts/validate_harvester.py",
    "scripts/verify_paper_results.py",
    "docs/budgeted-verification.md",
    "docs/calibration.md",
    "docs/claim-release.md",
    "docs/closed-corpus-evaluation.md",
    "docs/evidence-graph.md",
    "docs/evidence-inference-benchmark.md",
    "docs/meta-analysis.md",
    "docs/paper/harvester-validation.md",
    "docs/paper/metasyn-benchmark.md",
    "docs/paper/neurips26-evaluation-protocol.md",
    "docs/paper/task-evaluation-contract.md",
    "paper/EXECUTION_CHECKLIST.md",
    "paper/README.md",
    "paper/STYLE_SOURCE.md",
    "paper/main.pdf",
    "paper/main.tex",
    "paper/neurips_2026.sty",
    "paper/references.bib",
    "paper/results/budgeted_verification_simulation_200.tex",
    "paper/results/calibration_simulation_100.tex",
    "paper/results/meta_simulation_200.tex",
    "prompts/evidence_inference_extraction.md",
    "prompts/extraction.md",
    "prompts/moderator_proposal.md",
    "prompts/quote_verification.md",
    "prompts/targeted_remap.md",
    "artifacts/paper/budgeted-verification-simulation-200.json",
    "artifacts/paper/calibration-simulation-100.json",
    "artifacts/paper/closed-corpus-local-audit.json",
    "artifacts/paper/evidence-inference-2/failed-raw-schema-pilot30-summary.json",
    "artifacts/paper/evidence-inference-benchmark-summary.json",
    "artifacts/paper/evidence-inference-gepa-pilot-summary.json",
    "artifacts/paper/evidence-inference-low-budget-summary.json",
    "artifacts/paper/harvester/validation_summary.json",
    "artifacts/paper/meta-simulation-200.json",
    "artifacts/paper/metasyn-benchmark/METASYN_LICENSE.txt",
    "artifacts/paper/metasyn-fixed-positive-test/README.md",
    "artifacts/paper/metasyn-fixed-positive-test/evaluation.json",
    "artifacts/paper/metasyn-fixed-positive-test/freeze_receipt.json",
    "artifacts/paper/metasyn-fixed-positive-test/predictions.jsonl",
    "tests/conftest.py",
    "tests/test_budgeted_verification.py",
    "tests/test_budgeted_verification_simulation.py",
    "tests/test_calibrate_risk_gate_cli.py",
    "tests/test_calibration.py",
    "tests/test_claim_release.py",
    "tests/test_closed_corpus.py",
    "tests/test_evidence_graph.py",
    "tests/test_gepa_pilot_summary.py",
    "tests/test_harvester_validation.py",
    "tests/test_local_corpus_audit.py",
    "tests/test_meta_analysis.py",
    "tests/test_meta_simulation.py",
    "tests/test_metasyn_benchmark.py",
    "tests/test_prompt_optimization.py",
    "tests/fixtures/harvester/frozen/article-1.xml",
    "tests/fixtures/harvester/frozen/corpus.json",
    "tests/fixtures/harvester/openalex_page.json",
)

# Every Python module in the package is code-owned and included. Interpreter caches
# are explicitly skipped; other non-Python files under this tree are rejected.
ALLOWLIST_TREES = (("src/literature_multiverse", frozenset({".py"})),)

EXCLUDED_SCOPES = (
    ".env and all credentials",
    ".git and repository history",
    "data and all raw/local corpora or provider transcripts",
    "artifacts/antiox-training and human-review packets",
    "MetaSyn evaluator_labels.private.jsonl and model_inputs",
    "docs/planning, docs/superpowers, project context, and personal notes",
    "applications, screenshots, slide decks, caches, virtual environments, and tmp",
)

# The byte-for-byte official workshop style contains one public upstream
# maintainer address in a comment. No other file or address is exempted.
ALLOWED_EMAILS_BY_FILE = {
    "paper/neurips_2026.sty": frozenset({"garnett" + "@" + "wustl.edu"})
}

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
}

IDENTITY_PATTERNS = {
    "personal_name": re.compile(r"(?i)\b" + "har" + r"ry\b"),
    "macos_home_path": re.compile(r"(?i)/" + "Users" + r"/[^\s\"'<>]+"),
    "linux_home_path": re.compile(r"(?i)/" + "home" + r"/[^\s\"'<>]+"),
    "windows_home_path": re.compile(r"(?i)[A-Z]:\\\\Users\\\\[^\s\"'<>]+"),
    "email_address": re.compile(
        r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9.-])"
    ),
}

SECRET_PATTERNS = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(?:gh[opusr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "model_provider_key": re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gxl_[A-Za-z0-9_-]{16,})\b"
    ),
    "long_bearer_token": re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{24,}={0,2}\b"
    ),
    "embedded_url_credentials": re.compile(
        r"(?i)https?://[^\s/:@]+:[^\s/@]+@"
    ),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _relative_file(path_text: str) -> Path:
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe allowlist path: {path_text}")
    path = ROOT.joinpath(*relative.parts)
    if not path.exists():
        raise FileNotFoundError(f"allowlisted file is missing: {path_text}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"allowlisted path is not a regular file: {path_text}")
    return path


def _source_paths() -> list[tuple[str, Path]]:
    selected = {name: _relative_file(name) for name in ALLOWLIST_FILES}
    for tree_name, suffixes in ALLOWLIST_TREES:
        tree = ROOT / tree_name
        if tree.is_symlink() or not tree.is_dir():
            raise ValueError(f"allowlisted tree is not a regular directory: {tree_name}")
        for path in sorted(tree.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                relative = path.relative_to(ROOT)
                raise ValueError(f"symlink rejected in allowlisted tree: {relative}")
            if path.is_dir():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if path.suffix not in suffixes:
                raise ValueError(f"unexpected file type in allowlisted tree: {relative}")
            selected[relative] = path
    return sorted(selected.items())


def _scan_text(relative: str, payload: bytes) -> None:
    suffix = Path(relative).suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"allowlisted text file is not UTF-8: {relative}") from error
    findings: list[str] = []
    for label, pattern in {**IDENTITY_PATTERNS, **SECRET_PATTERNS}.items():
        scanned_text = text
        if label == "email_address":
            for allowed in ALLOWED_EMAILS_BY_FILE.get(relative, ()):
                scanned_text = scanned_text.replace(allowed, "[UPSTREAM_EMAIL]")
        if pattern.search(scanned_text):
            findings.append(label)
    if findings:
        raise ValueError(f"sensitive marker(s) in {relative}: {', '.join(findings)}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build() -> dict[str, object]:
    payloads: dict[str, bytes] = {}
    entries: list[dict[str, object]] = []
    for relative, path in _source_paths():
        payload = path.read_bytes()
        _scan_text(relative, payload)
        payloads[relative] = payload
        entries.append(
            {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
        )

    content_manifest = {
        "anonymous_artifact_manifest_version": "1",
        "archive_root": ARCHIVE_ROOT,
        "deterministic_zip_timestamp": "1980-01-01T00:00:00",
        "files": entries,
        "excluded_scopes": list(EXCLUDED_SCOPES),
        "scan_exceptions": [
            {
                "path": "paper/neurips_2026.sty",
                "reason": (
                    "One public upstream maintainer address is retained because the "
                    "official workshop style is distributed byte-for-byte."
                ),
            }
        ],
        "scope_note": (
            "This is an anonymous review artifact for code, contract tests, and public "
            "simulation outputs. It intentionally cannot reproduce restricted-corpus or "
            "provider-backed runs without separately authorized inputs and credentials."
        ),
    }
    internal_manifest = _canonical_json(content_manifest)
    internal_name = "ANONYMOUS_ARTIFACT_MANIFEST.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, mode="w", allowZip64=True) as archive:
        for relative, payload in sorted(payloads.items()):
            member = f"{ARCHIVE_ROOT}/{relative}"
            archive.writestr(_zip_info(member), payload)
        archive.writestr(
            _zip_info(f"{ARCHIVE_ROOT}/{internal_name}"), internal_manifest
        )

    archive_payload = archive_path.read_bytes()
    external_manifest = {
        **content_manifest,
        "archive": {
            "path": f"artifacts/submission/{ARCHIVE_NAME}",
            "bytes": len(archive_payload),
            "sha256": _sha256(archive_payload),
        },
        "archive_member_count": len(entries) + 1,
        "internal_manifest": {
            "path": internal_name,
            "bytes": len(internal_manifest),
            "sha256": _sha256(internal_manifest),
        },
    }
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    manifest_path.write_bytes(_canonical_json(external_manifest))
    return external_manifest


def main() -> int:
    manifest = build()
    archive = manifest["archive"]
    assert isinstance(archive, dict)
    print(
        json.dumps(
            {
                "archive": archive,
                "archive_member_count": manifest["archive_member_count"],
                "source_file_count": len(manifest["files"]),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
