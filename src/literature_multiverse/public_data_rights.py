"""Content-silent audit of tracked research-data redistribution declarations.

This module deliberately treats a rights declaration as an inventory contract, not
as legal advice.  It reads bytes only to hash them.  Reports contain aggregate counts,
sizes, path-pattern declarations, and hashes; they never contain corpus values.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from literature_multiverse.lineage import hash_canonical, sha256_file


class PublicDataRightsAuditError(ValueError):
    """The tracked-data rights policy or Git inventory is invalid."""


_RIGHTS_STATUSES = frozenset(
    {
        "project_authored",
        "redistribution_established",
        "redistribution_not_established",
    }
)
_RELEASE_ALLOWED_BY_STATUS = {
    "project_authored": True,
    "redistribution_established": True,
    "redistribution_not_established": False,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PMC_PATH_RE = re.compile(r"(?i)(?<![a-z0-9])pmc[0-9]+(?![a-z0-9])")
_HASH_PATH_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


@dataclass(frozen=True)
class _IndexEntry:
    path: str
    mode: str
    object_id: str


@dataclass(frozen=True)
class _BlobSummary:
    path: str
    mode: str
    object_id: str
    size_bytes: int
    content_sha256: str
    differs_from_worktree: bool


def _run_git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicDataRightsAuditError("git_inventory_unavailable") from exc
    return completed.stdout


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicDataRightsAuditError(f"policy_path_invalid:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise PublicDataRightsAuditError(f"policy_path_unsafe:{field}")
    return value


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicDataRightsAuditError("rights_policy_invalid_json") from exc
    if not isinstance(policy, dict) or policy.get("public_data_rights_policy_version") != "1":
        raise PublicDataRightsAuditError("rights_policy_version_unsupported")
    if policy.get("policy_id") != "literature-multiverse-public-data-rights-v1":
        raise PublicDataRightsAuditError("rights_policy_id_invalid")

    prefixes = policy.get("monitored_path_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise PublicDataRightsAuditError("rights_policy_prefixes_missing")
    normalized_prefixes = [
        _safe_relative(value, field=f"monitored_path_prefixes[{index}]")
        for index, value in enumerate(prefixes)
    ]
    if normalized_prefixes != sorted(set(normalized_prefixes)):
        raise PublicDataRightsAuditError("rights_policy_prefixes_not_sorted_unique")

    suffixes = policy.get("deny_by_default_suffixes")
    if (
        not isinstance(suffixes, list)
        or not suffixes
        or any(
            not isinstance(value, str)
            or not value.startswith(".")
            or value != value.casefold()
            for value in suffixes
        )
        or suffixes != sorted(set(suffixes))
    ):
        raise PublicDataRightsAuditError("rights_policy_suffixes_invalid")

    tokens = policy.get("deny_by_default_path_tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(value, str) or not value or value != value.casefold()
            for value in tokens
        )
        or tokens != sorted(set(tokens))
    ):
        raise PublicDataRightsAuditError("rights_policy_path_tokens_invalid")

    forbidden_fields = policy.get("established_rights_forbidden_field_names")
    if (
        not isinstance(forbidden_fields, list)
        or not forbidden_fields
        or any(
            not isinstance(value, str) or not value or value != value.casefold()
            for value in forbidden_fields
        )
        or forbidden_fields != sorted(set(forbidden_fields))
    ):
        raise PublicDataRightsAuditError("rights_policy_forbidden_fields_invalid")

    collections = policy.get("collections")
    if not isinstance(collections, list) or not collections:
        raise PublicDataRightsAuditError("rights_policy_collections_missing")
    collection_ids: list[str] = []
    for index, collection in enumerate(collections):
        if not isinstance(collection, dict):
            raise PublicDataRightsAuditError("rights_policy_collection_invalid")
        collection_id = collection.get("collection_id")
        if not isinstance(collection_id, str) or not collection_id:
            raise PublicDataRightsAuditError("rights_policy_collection_id_invalid")
        collection_ids.append(collection_id)
        patterns = collection.get("path_globs")
        if not isinstance(patterns, list) or not patterns:
            raise PublicDataRightsAuditError(
                f"rights_policy_collection_globs_missing:{collection_id}"
            )
        normalized_patterns = [
            _safe_relative(value, field=f"collections[{index}].path_globs[{pattern_index}]")
            for pattern_index, value in enumerate(patterns)
        ]
        if normalized_patterns != sorted(set(normalized_patterns)):
            raise PublicDataRightsAuditError(
                f"rights_policy_collection_globs_not_sorted_unique:{collection_id}"
            )
        content_class = collection.get("content_class")
        if not isinstance(content_class, str) or not content_class:
            raise PublicDataRightsAuditError(
                f"rights_policy_content_class_invalid:{collection_id}"
            )
        rights_status = collection.get("rights_status")
        if rights_status not in _RIGHTS_STATUSES:
            raise PublicDataRightsAuditError(
                f"rights_policy_status_invalid:{collection_id}"
            )
        if collection.get("public_release_allowed") is not _RELEASE_ALLOWED_BY_STATUS[
            rights_status
        ]:
            raise PublicDataRightsAuditError(
                f"rights_policy_release_flag_inconsistent:{collection_id}"
            )
        rationale = collection.get("rationale")
        if not isinstance(rationale, str) or not rationale:
            raise PublicDataRightsAuditError(
                f"rights_policy_rationale_missing:{collection_id}"
            )
        evidence = collection.get("license_evidence", [])
        if not isinstance(evidence, list):
            raise PublicDataRightsAuditError(
                f"rights_policy_license_evidence_invalid:{collection_id}"
            )
        if rights_status == "redistribution_established" and not evidence:
            raise PublicDataRightsAuditError(
                f"rights_policy_license_evidence_missing:{collection_id}"
            )
        for evidence_index, record in enumerate(evidence):
            if not isinstance(record, dict):
                raise PublicDataRightsAuditError(
                    f"rights_policy_license_evidence_invalid:{collection_id}"
                )
            _safe_relative(
                record.get("path"),
                field=(
                    f"collections[{index}].license_evidence[{evidence_index}].path"
                ),
            )
            expected = record.get("sha256")
            if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
                raise PublicDataRightsAuditError(
                    f"rights_policy_license_hash_invalid:{collection_id}"
                )
        if not isinstance(collection.get("allow_empty", False), bool):
            raise PublicDataRightsAuditError(
                f"rights_policy_allow_empty_invalid:{collection_id}"
            )
    if collection_ids != sorted(set(collection_ids)):
        raise PublicDataRightsAuditError("rights_policy_collection_ids_not_sorted_unique")
    return policy


def _git_index(root: Path) -> list[_IndexEntry]:
    raw = _run_git(root, "ls-files", "--stage", "-z")
    entries: list[_IndexEntry] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, encoded_path = item.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise PublicDataRightsAuditError("git_index_entry_invalid") from exc
        if stage != "0":
            raise PublicDataRightsAuditError("git_index_contains_unmerged_entry")
        _safe_relative(path, field="git_index_path")
        entries.append(_IndexEntry(path=path, mode=mode, object_id=object_id))
    paths = [entry.path for entry in entries]
    if paths != sorted(set(paths)):
        raise PublicDataRightsAuditError("git_index_paths_not_sorted_unique")
    return entries


def _dirty_worktree_paths(root: Path) -> set[str]:
    raw = _run_git(root, "diff-files", "--name-only", "-z")
    try:
        return {item.decode("utf-8") for item in raw.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise PublicDataRightsAuditError("git_worktree_path_invalid") from exc


def _read_index_blob(root: Path, entry: _IndexEntry, *, dirty_paths: set[str]) -> bytes:
    path = root / entry.path
    if entry.path not in dirty_paths and path.is_file() and not path.is_symlink():
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PublicDataRightsAuditError("tracked_blob_unreadable") from exc
    return _run_git(root, "cat-file", "blob", entry.object_id)


def _is_monitored(path: str, policy: dict[str, Any]) -> bool:
    if any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in policy["monitored_path_prefixes"]
    ):
        return True
    relative = PurePosixPath(path)
    name = relative.name.casefold()
    if any(name.endswith(suffix) for suffix in policy["deny_by_default_suffixes"]):
        return True
    if relative.suffix.casefold() in {
        ".css",
        ".html",
        ".js",
        ".lock",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }:
        return False
    folded_path = path.casefold()
    return any(token in folded_path for token in policy["deny_by_default_path_tokens"])


def _matches(path: str, collection: dict[str, Any]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in collection["path_globs"])


def _blob_summary(
    root: Path,
    entry: _IndexEntry,
    *,
    dirty_paths: set[str],
) -> _BlobSummary:
    content = _read_index_blob(root, entry, dirty_paths=dirty_paths)
    return _BlobSummary(
        path=entry.path,
        mode=entry.mode,
        object_id=entry.object_id,
        size_bytes=len(content),
        content_sha256=hashlib.sha256(content).hexdigest(),
        differs_from_worktree=entry.path in dirty_paths,
    )


def _collect_mapping_keys(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            output.add(str(key).casefold())
            _collect_mapping_keys(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_mapping_keys(item, output)


def _structured_field_names(path: str, content: bytes) -> set[str] | None:
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix not in {".json", ".jsonl"}:
        return None
    fields: set[str] = set()
    try:
        if suffix == ".json":
            _collect_mapping_keys(json.loads(content), fields)
        else:
            for line in content.splitlines():
                if line.strip():
                    _collect_mapping_keys(json.loads(line), fields)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicDataRightsAuditError("licensed_structured_payload_invalid") from exc
    return fields


def _collection_summary(
    collection: dict[str, Any],
    files: list[_BlobSummary],
    *,
    structured_fields: set[str],
) -> dict[str, Any]:
    extension_counts: Counter[str] = Counter()
    pmc_identifiers: set[str] = set()
    hash_identifiers: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for record in sorted(files, key=lambda item: item.path):
        suffix = PurePosixPath(record.path).suffix.casefold() or "[no_extension]"
        extension_counts[suffix] += 1
        pmc_identifiers.update(
            match.group(0).casefold() for match in _PMC_PATH_RE.finditer(record.path)
        )
        hash_identifiers.update(
            match.group(0).casefold() for match in _HASH_PATH_RE.finditer(record.path)
        )
        inventory.append(
            {
                "path": record.path,
                "mode": record.mode,
                "git_object_id": record.object_id,
                "bytes": record.size_bytes,
                "sha256": record.content_sha256,
            }
        )
    return {
        "collection_id": collection["collection_id"],
        "content_class": collection["content_class"],
        "rights_status": collection["rights_status"],
        "public_release_allowed": collection["public_release_allowed"],
        "path_globs": collection["path_globs"],
        "file_count": len(files),
        "total_bytes": sum(record.size_bytes for record in files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "path_identifier_counts": {
            "distinct_pmc_identifiers": len(pmc_identifiers),
            "distinct_sha256_like_identifiers": len(hash_identifiers),
        },
        "structured_field_count": len(structured_fields),
        "structured_field_inventory_sha256": hash_canonical(sorted(structured_fields)),
        "index_inventory_sha256": hash_canonical(inventory),
        "worktree_difference_count": sum(record.differs_from_worktree for record in files),
    }


def audit_public_data_rights(
    *,
    repository_root: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    """Audit tracked research-data paths without returning corpus content.

    The Git index is the authoritative public inventory.  A dirty worktree is reported,
    while hashes remain bound to the index bytes that would be committed by default.
    """

    root = repository_root.resolve()
    selected_policy_path = (policy_path or root / "configs/public-data-rights-v1.json").resolve()
    try:
        relative_policy_path = selected_policy_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PublicDataRightsAuditError("rights_policy_outside_repository") from exc
    policy = _load_policy(selected_policy_path)
    index_entries = _git_index(root)
    index_by_path = {entry.path: entry for entry in index_entries}
    dirty_paths = _dirty_worktree_paths(root)
    candidates = [entry for entry in index_entries if _is_monitored(entry.path, policy)]

    matched_by_collection: dict[str, list[_IndexEntry]] = defaultdict(list)
    undeclared: list[str] = []
    ambiguous: list[str] = []
    for entry in candidates:
        matched = [
            collection
            for collection in policy["collections"]
            if _matches(entry.path, collection)
        ]
        if not matched:
            undeclared.append(entry.path)
        elif len(matched) > 1:
            ambiguous.append(entry.path)
        else:
            matched_by_collection[matched[0]["collection_id"]].append(entry)

    policy_blockers: list[dict[str, Any]] = []
    if undeclared:
        policy_blockers.append(
            {
                "code": "tracked_data_rights_undeclared",
                "count": len(undeclared),
                "sample_path_sha256s": [
                    hashlib.sha256(path.encode("utf-8")).hexdigest()
                    for path in undeclared[:20]
                ],
            }
        )
    if ambiguous:
        policy_blockers.append(
            {
                "code": "tracked_data_rights_ambiguous",
                "count": len(ambiguous),
                "sample_path_sha256s": [
                    hashlib.sha256(path.encode("utf-8")).hexdigest()
                    for path in ambiguous[:20]
                ],
            }
        )

    evidence_cache: dict[str, _BlobSummary] = {}
    collection_summaries: list[dict[str, Any]] = []
    release_blockers: list[dict[str, Any]] = []
    classified_records: list[_BlobSummary] = []
    for collection in policy["collections"]:
        collection_id = collection["collection_id"]
        entries = matched_by_collection.get(collection_id, [])
        if not entries and not collection.get("allow_empty", False):
            policy_blockers.append(
                {"code": "rights_policy_collection_empty", "collection_id": collection_id}
            )
        blobs = [
            _blob_summary(root, entry, dirty_paths=dirty_paths) for entry in entries
        ]
        classified_records.extend(blobs)

        structured_fields: set[str] = set()
        if collection["rights_status"] == "redistribution_established":
            try:
                for entry in entries:
                    content = _read_index_blob(root, entry, dirty_paths=dirty_paths)
                    observed = _structured_field_names(entry.path, content)
                    if observed is not None:
                        structured_fields.update(observed)
            except PublicDataRightsAuditError:
                policy_blockers.append(
                    {
                        "code": "licensed_structured_payload_invalid",
                        "collection_id": collection_id,
                    }
                )
            forbidden = structured_fields.intersection(
                policy["established_rights_forbidden_field_names"]
            )
            if forbidden:
                policy_blockers.append(
                    {
                        "code": "licensed_collection_contains_forbidden_fields",
                        "collection_id": collection_id,
                        "field_count": len(forbidden),
                        "field_name_sha256s": sorted(
                            hashlib.sha256(value.encode("utf-8")).hexdigest()
                            for value in forbidden
                        ),
                    }
                )
        collection_summaries.append(
            _collection_summary(
                collection,
                blobs,
                structured_fields=structured_fields,
            )
        )

        for evidence in collection.get("license_evidence", []):
            evidence_path = evidence["path"]
            entry = index_by_path.get(evidence_path)
            if entry is None:
                policy_blockers.append(
                    {
                        "code": "license_evidence_not_tracked",
                        "collection_id": collection_id,
                        "path_sha256": hashlib.sha256(
                            evidence_path.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                continue
            record = evidence_cache.get(evidence_path)
            if record is None:
                record = _blob_summary(root, entry, dirty_paths=dirty_paths)
                evidence_cache[evidence_path] = record
            if record.content_sha256 != evidence["sha256"]:
                policy_blockers.append(
                    {
                        "code": "license_evidence_hash_mismatch",
                        "collection_id": collection_id,
                        "path_sha256": hashlib.sha256(
                            evidence_path.encode("utf-8")
                        ).hexdigest(),
                    }
                )
        if collection["rights_status"] == "redistribution_not_established" and entries:
            release_blockers.append(
                {
                    "code": "redistribution_not_established",
                    "collection_id": collection_id,
                    "file_count": len(entries),
                }
            )

    all_inventory = [
        {
            "path": record.path,
            "mode": record.mode,
            "git_object_id": record.object_id,
            "bytes": record.size_bytes,
            "sha256": record.content_sha256,
        }
        for record in sorted(classified_records, key=lambda item: item.path)
    ]
    rights_counts = Counter(
        summary["rights_status"]
        for summary in collection_summaries
        for _ in range(summary["file_count"])
    )
    policy_complete = not policy_blockers
    release_ready = policy_complete and not release_blockers
    payload: dict[str, Any] = {
        "public_data_rights_audit_version": "1",
        "inventory_source": "git_index",
        "policy": {
            "path": relative_policy_path,
            "file_sha256": sha256_file(selected_policy_path),
            "policy_id": policy["policy_id"],
        },
        "tracked_files_total": len(index_entries),
        "audited_candidate_files": len(candidates),
        "classified_files": len(classified_records),
        "classified_bytes": sum(record.size_bytes for record in classified_records),
        "rights_status_file_counts": dict(sorted(rights_counts.items())),
        "worktree_difference_count": sum(
            record.differs_from_worktree for record in classified_records
        ),
        "collection_summaries": collection_summaries,
        "classified_index_inventory_sha256": hash_canonical(all_inventory),
        "undeclared_file_count": len(undeclared),
        "ambiguous_file_count": len(ambiguous),
        "policy_blockers": policy_blockers,
        "release_blockers": release_blockers,
        "policy_complete": policy_complete,
        "release_ready": release_ready,
        "interpretation": (
            "inventory_and_policy_check_not_legal_advice_or_a_grant_of_rights"
        ),
        "content_disclosure": "aggregate_metadata_and_hashes_only",
    }
    return {**payload, "audit_payload_sha256": hash_canonical(payload)}


def verify_audit_self_hash(report: dict[str, Any]) -> None:
    """Verify the canonical full-payload hash on an audit report."""

    observed = report.get("audit_payload_sha256")
    payload = {key: value for key, value in report.items() if key != "audit_payload_sha256"}
    if not isinstance(observed, str) or observed != hash_canonical(payload):
        raise PublicDataRightsAuditError("rights_audit_self_hash_mismatch")
