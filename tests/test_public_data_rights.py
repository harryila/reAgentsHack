from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from scripts.audit_public_data_rights import main

from literature_multiverse.public_data_rights import (
    PublicDataRightsAuditError,
    _is_monitored,
    _load_policy,
    _matches,
    audit_public_data_rights,
    verify_audit_self_hash,
)
from literature_multiverse.public_data_rights import (
    _structured_field_names as _scan_structured_field_names,
)


def _structured_field_names(path: Path) -> set[str]:
    # Test-local adapter: the real helper takes (path: str, content: bytes), not a Path.
    return _scan_structured_field_names(path.as_posix(), path.read_bytes()) or set()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


def _write_policy(
    root: Path,
    *,
    collections: list[dict[str, object]],
) -> Path:
    policy = {
        "public_data_rights_policy_version": "1",
        "policy_id": "literature-multiverse-public-data-rights-v1",
        "monitored_path_prefixes": ["data"],
        "deny_by_default_suffixes": [".jsonl", ".parquet", ".txt"],
        "deny_by_default_path_tokens": ["abstract", "source_lines"],
        "established_rights_forbidden_field_names": [
            "abstract",
            "article_text",
            "evidence_quote",
            "full_text",
        ],
        "collections": collections,
    }
    path = root / "configs/public-data-rights-v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def _collection(
    *,
    collection_id: str = "research_data",
    path_globs: list[str] | None = None,
    rights_status: str = "redistribution_not_established",
    license_evidence: list[dict[str, str]] | None = None,
    allow_empty: bool = False,
) -> dict[str, object]:
    record: dict[str, object] = {
        "collection_id": collection_id,
        "path_globs": path_globs or ["data/**"],
        "content_class": "test_research_data",
        "rights_status": rights_status,
        "public_release_allowed": rights_status != "redistribution_not_established",
        "rationale": "Test declaration with no corpus values in the policy.",
    }
    if license_evidence is not None:
        record["license_evidence"] = license_evidence
    if allow_empty:
        record["allow_empty"] = True
    return record


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")


def test_audit_is_content_silent_and_binds_git_index(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    payload = tmp_path / "data/raw/PMC123.stdout"
    payload.parent.mkdir(parents=True)
    original = b"ARTICLE SENTENCE THAT MUST NEVER APPEAR IN THE REPORT"
    payload.write_bytes(original)
    _git(tmp_path, "add", payload.relative_to(tmp_path).as_posix())
    _write_policy(tmp_path, collections=[_collection()])

    first = audit_public_data_rights(repository_root=tmp_path)
    verify_audit_self_hash(first)
    assert first["policy_complete"] is True
    assert first["release_ready"] is False
    assert first["classified_files"] == 1
    assert first["classified_bytes"] == len(original)
    assert first["collection_summaries"][0]["path_identifier_counts"] == {
        "distinct_pmc_identifiers": 1,
        "distinct_sha256_like_identifiers": 0,
    }
    serialized = json.dumps(first)
    assert original.decode() not in serialized
    assert "PMC123" not in serialized

    payload.write_bytes(b"UNSTAGED REPLACEMENT THAT ALSO MUST NOT APPEAR")
    dirty = audit_public_data_rights(repository_root=tmp_path)
    assert dirty["classified_bytes"] == len(original)
    assert dirty["classified_index_inventory_sha256"] == first[
        "classified_index_inventory_sha256"
    ]
    assert dirty["worktree_difference_count"] == 1
    assert "UNSTAGED REPLACEMENT" not in json.dumps(dirty)


def test_undeclared_and_ambiguous_tracked_data_fail_closed(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    payload = tmp_path / "data/vendor/input.jsonl"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"opaque":"value"}\n', encoding="utf-8")
    _git(tmp_path, "add", payload.relative_to(tmp_path).as_posix())

    _write_policy(
        tmp_path,
        collections=[
            _collection(
                path_globs=["data/approved/**"],
                allow_empty=True,
            )
        ],
    )
    undeclared = audit_public_data_rights(repository_root=tmp_path)
    assert undeclared["policy_complete"] is False
    assert undeclared["undeclared_file_count"] == 1
    assert undeclared["policy_blockers"][0]["code"] == (
        "tracked_data_rights_undeclared"
    )
    assert payload.relative_to(tmp_path).as_posix() not in json.dumps(undeclared)

    _write_policy(
        tmp_path,
        collections=[
            _collection(collection_id="a", path_globs=["data/**"]),
            _collection(collection_id="b", path_globs=["data/vendor/**"]),
        ],
    )
    ambiguous = audit_public_data_rights(repository_root=tmp_path)
    assert ambiguous["policy_complete"] is False
    assert ambiguous["ambiguous_file_count"] == 1
    assert ambiguous["policy_blockers"][0]["code"] == (
        "tracked_data_rights_ambiguous"
    )


def test_established_rights_require_bound_license_and_forbid_article_fields(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    license_path = tmp_path / "LICENSE.DATA"
    license_path.write_text("fixture redistribution grant", encoding="utf-8")
    input_path = tmp_path / "data/licensed/input.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"research_question":"synthetic"}\n', encoding="utf-8")
    _git(tmp_path, "add", "LICENSE.DATA", "data/licensed/input.jsonl")
    license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
    _write_policy(
        tmp_path,
        collections=[
            _collection(
                rights_status="redistribution_established",
                license_evidence=[{"path": "LICENSE.DATA", "sha256": license_sha}],
            )
        ],
    )

    clean = audit_public_data_rights(repository_root=tmp_path)
    assert clean["policy_complete"] is True
    assert clean["release_ready"] is True
    assert clean["collection_summaries"][0]["structured_field_count"] == 1

    input_path.write_text('{"abstract":"must remain private"}\n', encoding="utf-8")
    _git(tmp_path, "add", "data/licensed/input.jsonl")
    forbidden = audit_public_data_rights(repository_root=tmp_path)
    assert forbidden["policy_complete"] is False
    assert forbidden["policy_blockers"][0]["code"] == (
        "licensed_collection_contains_forbidden_fields"
    )
    assert "must remain private" not in json.dumps(forbidden)

    policy = json.loads((tmp_path / "configs/public-data-rights-v1.json").read_text())
    policy["collections"][0]["license_evidence"][0]["sha256"] = "0" * 64
    (tmp_path / "configs/public-data-rights-v1.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    bad_license = audit_public_data_rights(repository_root=tmp_path)
    assert any(
        blocker["code"] == "license_evidence_hash_mismatch"
        for blocker in bad_license["policy_blockers"]
    )


def test_policy_rejects_unsafe_paths_and_tampered_report(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    payload = tmp_path / "data/input.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("fixture", encoding="utf-8")
    _git(tmp_path, "add", "data/input.txt")
    _write_policy(
        tmp_path,
        collections=[_collection(path_globs=["../private/**"])],
    )
    with pytest.raises(PublicDataRightsAuditError, match="policy_path_unsafe"):
        audit_public_data_rights(repository_root=tmp_path)

    _write_policy(tmp_path, collections=[_collection()])
    report = audit_public_data_rights(repository_root=tmp_path)
    report["classified_bytes"] += 1
    with pytest.raises(PublicDataRightsAuditError, match="self_hash_mismatch"):
        verify_audit_self_hash(report)


def test_cli_distinguishes_policy_completeness_from_release_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_repo(tmp_path)
    payload = tmp_path / "data/input.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("PRIVATE FIXTURE VALUE", encoding="utf-8")
    _git(tmp_path, "add", "data/input.txt")
    _write_policy(tmp_path, collections=[_collection()])

    assert main(["--repository-root", str(tmp_path)]) == 0
    assert "PRIVATE FIXTURE VALUE" not in capsys.readouterr().out
    assert (
        main(["--repository-root", str(tmp_path), "--require-release-ready"])
        == 2
    )
    assert "PRIVATE FIXTURE VALUE" not in capsys.readouterr().out


def test_repository_policy_covers_current_tracked_data_without_values(
    repo_root: Path,
) -> None:
    report = audit_public_data_rights(repository_root=repo_root)
    verify_audit_self_hash(report)
    assert report["release_ready"] is False
    assert report["classified_files"] == report["audited_candidate_files"]
    assert report["undeclared_file_count"] == 0
    assert report["ambiguous_file_count"] == 0
    assert report["rights_status_file_counts"][
        "redistribution_not_established"
    ] > 0
    synthesis = next(
        item
        for item in report["collection_summaries"]
        if item["collection_id"] == "metasyn_synthesis_yield_aggregate"
    )
    expected_local_handoff_blocker = {
        "code": "rights_policy_collection_empty",
        "collection_id": "metasyn_synthesis_yield_aggregate",
    }
    if synthesis["file_count"] == 0:
        assert report["policy_complete"] is False
        assert report["policy_blockers"] == [expected_local_handoff_blocker]
    else:
        assert synthesis["file_count"] == 1
        assert report["policy_complete"] is True
        assert report["policy_blockers"] == []
    serialized = json.dumps(report)
    assert "PMC10034200" not in serialized
    assert "source_lines.prior-run.json" not in serialized


def test_local_suite_identifier_receipts_stay_declared_but_unindexed(repo_root: Path) -> None:
    report = audit_public_data_rights(repository_root=repo_root)
    assert report["release_ready"] is False
    local_suite = next(
        item for item in report["collection_summaries"]
        if item["collection_id"] == "local_suite_identifier_receipts"
    )
    assert local_suite["path_globs"] == [
        "artifacts/benchmarks/local-suite-v1/freeze_receipt.json",
        "artifacts/benchmarks/local-suite-v1/predictions.jsonl",
    ]
    assert local_suite["rights_status"] == "redistribution_not_established"
    assert local_suite["public_release_allowed"] is False
    # .gitignore and the CI aggregate-only step forbid indexing these receipts; the
    # declaration stays (allow_empty) so that indexing them can never be silent.
    assert local_suite["file_count"] == 0
    assert local_suite["extension_counts"] == {}
    assert [
        item for item in report["policy_blockers"]
        if item.get("collection_id") == "local_suite_identifier_receipts"
    ] == []
    serialized = json.dumps(report)
    assert "git_object_id" not in serialized
    assert "ARTICLE SENTENCE" not in serialized


def test_evidencebench_public_rights_scope_is_an_exact_two_file_allowlist(
    repo_root: Path,
) -> None:
    policy = json.loads(
        (repo_root / "configs/public-data-rights-v1.json").read_text(encoding="utf-8")
    )
    collection = next(
        item
        for item in policy["collections"]
        if item["collection_id"] == "evidencebench_grounding_aggregate_and_audit"
    )
    assert collection["path_globs"] == [
        "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json",
        "artifacts/diagnostics/evidencebench-grounding-v1/summary.json",
    ]


def test_metasyn_synthesis_yield_rights_scope_requires_exact_indexed_aggregate(
    repo_root: Path,
) -> None:
    policy = json.loads(
        (repo_root / "configs/public-data-rights-v1.json").read_text(encoding="utf-8")
    )
    prefix = "artifacts/diagnostics/metasyn-synthesis-yield-v1"
    assert policy["monitored_path_prefixes"] == sorted(
        set(policy["monitored_path_prefixes"])
    )
    assert prefix in policy["monitored_path_prefixes"]
    assert [item["collection_id"] for item in policy["collections"]] == sorted(
        {item["collection_id"] for item in policy["collections"]}
    )
    collection = next(
        item
        for item in policy["collections"]
        if item["collection_id"] == "metasyn_synthesis_yield_aggregate"
    )
    assert collection["path_globs"] == [f"{prefix}/summary.json"]
    assert collection["content_class"] == (
        "project_authored_aggregate_yield_counts_hashes_and_caveats"
    )
    assert collection["rights_status"] == "project_authored"
    assert collection["public_release_allowed"] is True
    assert "allow_empty" not in collection
    assert "license_evidence" not in collection
    assert "makes no claim about rights in the private source corpus" in collection[
        "rationale"
    ]

    report = audit_public_data_rights(repository_root=repo_root)
    verify_audit_self_hash(report)
    summary = next(
        item
        for item in report["collection_summaries"]
        if item["collection_id"] == "metasyn_synthesis_yield_aggregate"
    )
    assert summary["path_globs"] == [f"{prefix}/summary.json"]
    assert summary["rights_status"] == "project_authored"
    assert summary["public_release_allowed"] is True
    relevant_blockers = [
        blocker
        for blocker in report["policy_blockers"] + report["release_blockers"]
        if blocker.get("collection_id") == "metasyn_synthesis_yield_aggregate"
    ]
    if summary["file_count"] == 0:
        assert relevant_blockers == [
            {
                "code": "rights_policy_collection_empty",
                "collection_id": "metasyn_synthesis_yield_aggregate",
            }
        ]
        assert summary["extension_counts"] == {}
    else:
        assert summary["file_count"] == 1
        assert summary["extension_counts"] == {".json": 1}
        assert relevant_blockers == []


def _tracked_paths_from_index(repo_root: Path) -> set[str]:
    # Parse .git/index (v2) directly; the worker never runs git.
    import struct

    data = (repo_root / ".git/index").read_bytes()
    signature, version, count = struct.unpack(">4sII", data[:12])
    if signature != b"DIRC" or version != 2:
        pytest.skip(f"unsupported git index format: {signature!r} v{version}")
    position, paths = 12, set()
    for _ in range(count):
        flags = struct.unpack(">H", data[position + 60 : position + 62])[0]
        if flags & 0x4000:
            pytest.skip(
                "git index entry uses extended flags; parser supports v2 basic entries only"
            )
        name_length = flags & 0x0FFF
        assert name_length < 0x0FFF
        paths.add(data[position + 62 : position + 62 + name_length].decode("utf-8"))
        position += ((62 + name_length + 8) // 8) * 8
    return paths


def test_project_authored_prompt_templates_are_declared(repo_root: Path) -> None:
    policy = json.loads(
        (repo_root / "configs/public-data-rights-v1.json").read_text(encoding="utf-8")
    )
    collection = next(
        item for item in policy["collections"]
        if item["collection_id"] == "project_authored_prompt_templates"
    )
    assert collection["path_globs"] == ["prompts/**"]
    assert collection["rights_status"] == "project_authored"
    assert collection["public_release_allowed"] is True
    assert "license_evidence" not in collection
    report = audit_public_data_rights(repository_root=repo_root)
    summary = next(
        item for item in report["collection_summaries"]
        if item["collection_id"] == "project_authored_prompt_templates"
    )
    assert summary["file_count"] >= 3
    assert set(summary["extension_counts"]) <= {".txt"}
    assert report["undeclared_file_count"] == 0


def test_every_monitored_diagnostic_and_prompt_on_disk_matches_exactly_one_collection(
    repo_root: Path,
) -> None:
    policy = _load_policy(repo_root / "configs/public-data-rights-v1.json")
    candidates = [
        path.relative_to(repo_root).as_posix()
        for base in ("artifacts/diagnostics", "prompts")
        for path in (repo_root / base).rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    monitored = [rel for rel in candidates if _is_monitored(rel, policy)]
    assert monitored
    for rel in monitored:
        matches = [c["collection_id"] for c in policy["collections"] if _matches(rel, c)]
        assert len(matches) == 1, (rel, matches)


def test_diagnostic_rights_split_matches_structured_field_content(repo_root: Path) -> None:
    policy = _load_policy(repo_root / "configs/public-data-rights-v1.json")
    forbidden = set(policy["established_rights_forbidden_field_names"])
    by_id = {c["collection_id"]: c for c in policy["collections"]}
    tracked = _tracked_paths_from_index(repo_root)
    text_free = by_id["project_authored_diagnostic_aggregates"]["path_globs"] + by_id[
        "project_authored_evidence_inference_rosters_with_pmc_identifiers"
    ]["path_globs"]
    text_bearing = by_id["metasyn_derived_diagnostics_with_source_text"]["path_globs"]
    for rel in text_free:
        if rel.endswith(".json") and (repo_root / rel).is_file():
            assert not (_structured_field_names(repo_root / rel) & forbidden), rel
    for rel in text_bearing:
        if rel.endswith(".json"):
            assert _structured_field_names(repo_root / rel) & forbidden, rel
    declared_tracked = {rel for rel in text_free + text_bearing if rel in tracked}
    monitored_tracked = {
        rel
        for rel in tracked
        if rel.startswith("artifacts/diagnostics/") and _is_monitored(rel, policy)
    }
    already_declared = {
        "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json",
        "artifacts/diagnostics/evidencebench-grounding-v1/summary.json",
        "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json",
    }
    assert declared_tracked | already_declared == monitored_tracked
