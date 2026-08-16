"""Deterministic, offline-only packaging for the frozen Literature Multiverse demo."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from literature_multiverse.cohort import CohortContractError, cohort_sha256
from literature_multiverse.config import QuestionConfig, config_sha256, load_question_config
from literature_multiverse.contradictions import residual_summary
from literature_multiverse.disagreement import (
    paper_balanced_finding_summary,
    paper_modal_summary,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import (
    AuditRecord,
    EvidenceHashes,
    FindingRow,
    NullableEvidenceHashes,
    NullableStageRunHashes,
    PaperRecord,
    ReleaseSelection,
    RunRecord,
    ScaledAttempt,
    SelectedRelease,
    StageRunHashes,
    render_release_disclosure,
)
from literature_multiverse.normalize import (
    compute_quality_metrics,
    primary_cohort,
    reconcile_verification,
)
from literature_multiverse.paths import ProjectPaths
from literature_multiverse.resampling import (
    build_variant_a_headline,
    build_variant_b_headline,
    validate_checkpoint,
)
from literature_multiverse.resampling import canonical_sha256 as checkpoint_sha256
from literature_multiverse.schemas import validate_finding_row

ANALYSIS_FILES = frozenset(
    {
        "analysis/moderators.parquet",
        "analysis/m4_checkpoint.json",
        "analysis/tree.json",
        "analysis/contradictions.parquet",
        "analysis/evidence_gaps.parquet",
        "analysis/bootstrap.json",
        "analysis/permutation.json",
        "analysis/m4_gate.json",
        "analysis/headline.json",
    }
)
ROOT_SOURCE_FILES = frozenset(
    {
        "papers.parquet",
        "findings.parquet",
        "audit.json",
        "verification.json",
        "g3_gate.json",
        "baseline.json",
        "trace.json",
    }
)
GENERATED_FILES = frozenset({"release_selection.json", "demo_script.md"})
BUNDLED_PATHS = ANALYSIS_FILES | ROOT_SOURCE_FILES | GENERATED_FILES
ALL_DEMO_PATHS = BUNDLED_PATHS | {"manifest.json"}

if len(BUNDLED_PATHS) != 18:  # An executable guard against accidental inventory drift.
    raise RuntimeError("demo bundle must contain exactly 18 non-manifest files")

PARQUET_PATHS = frozenset(path for path in BUNDLED_PATHS if path.endswith(".parquet"))
JSON_PATHS = frozenset(path for path in BUNDLED_PATHS if path.endswith(".json"))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"{{\s*([a-zA-Z0-9_.]+)\s*}}")
UNRESOLVED_TOKEN_RE = re.compile(r"{{.*?}}", flags=re.DOTALL)

VARIANT_TOKENS = {
    "A": frozenset(
        {
            "manifest.spoken_question",
            "manifest.paper_funnel.searched_documents",
            "manifest.paper_funnel.identity_deduped_papers",
            "manifest.paper_funnel.primary_grounded_papers",
            "manifest.paper_funnel.primary_grounded_findings",
            "manifest.release_selection.rendered_disclosure",
            "headline.rendered_sentence",
        }
    ),
    "B": frozenset(
        {
            "manifest.spoken_question",
            "manifest.paper_funnel.searched_documents",
            "manifest.paper_funnel.identity_deduped_papers",
            "manifest.paper_funnel.primary_grounded_papers",
            "manifest.paper_funnel.primary_grounded_findings",
            "manifest.release_selection.rendered_disclosure",
            "headline.rendered_sentence",
            "headline.residuals.rendered_sentence",
        }
    ),
}

FailureKind = Literal["integrity", "reconciliation", "unvalidated", "incomplete"]


class ExportError(ValueError):
    """A release cannot be safely staged, selected, or verified."""

    def __init__(self, code: str, detail: str = "", *, kind: FailureKind = "integrity") -> None:
        self.code = code
        self.detail = detail
        self.kind = kind
        suffix = f":{detail}" if detail else ""
        super().__init__(f"{code}{suffix}")


@dataclass(frozen=True, slots=True)
class ReleaseSource:
    """Canonical scientific inputs for one v1 or scaled candidate."""

    repository_root: Path
    question_id: str
    corpus_role: Literal["v1", "scaled"]
    processed_dir: Path
    analysis_dir: Path
    extracted_dir: Path

    @classmethod
    def from_repository(
        cls,
        repository_root: str | Path,
        question_id: str,
        *,
        corpus_role: Literal["v1", "scaled"] = "v1",
        processed_dir: str | Path | None = None,
        analysis_dir: str | Path | None = None,
        extracted_dir: str | Path | None = None,
    ) -> ReleaseSource:
        root = Path(repository_root).resolve()
        paths = ProjectPaths(root)
        return cls(
            repository_root=root,
            question_id=question_id,
            corpus_role=corpus_role,
            processed_dir=(
                Path(processed_dir).resolve()
                if processed_dir is not None
                else paths.processed_dir(question_id)
            ),
            analysis_dir=(
                Path(analysis_dir).resolve()
                if analysis_dir is not None
                else paths.analysis_dir(question_id)
            ),
            extracted_dir=(
                Path(extracted_dir).resolve()
                if extracted_dir is not None
                else paths.extracted_dir(question_id)
            ),
        )

    def source_path(self, bundled_path: str) -> Path:
        if bundled_path == "papers.parquet":
            return self.processed_dir / "papers.parquet"
        if bundled_path == "findings.parquet":
            return self.processed_dir / "findings.parquet"
        if bundled_path in {"audit.json", "verification.json", "g3_gate.json"}:
            return self.processed_dir / bundled_path
        if bundled_path.startswith("analysis/"):
            return self.analysis_dir / bundled_path.removeprefix("analysis/")
        if bundled_path in {"baseline.json", "trace.json"}:
            return self.analysis_dir / bundled_path
        raise ExportError("source_path_unknown", bundled_path)

    @property
    def stage_run_paths(self) -> dict[str, Path]:
        return {
            "s3": self.extracted_dir / "run.json",
            "s4": self.processed_dir / "run.json",
            "s5": self.analysis_dir / "run.json",
        }


@dataclass(slots=True)
class ValidatedRelease:
    source: ReleaseSource
    config: QuestionConfig
    papers: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    primary_rows: list[dict[str, Any]]
    json_artifacts: dict[str, dict[str, Any]]
    table_frames: dict[str, pd.DataFrame]
    stage_runs: dict[str, RunRecord]
    stage_run_sha256s: StageRunHashes
    evidence_sha256s: EvidenceHashes
    cohort_sha256: str
    release_id: str
    paper_funnel: dict[str, int]
    quality: dict[str, Any]
    exclusions: dict[str, Any]
    narrative_variant: Literal["A", "B"]
    created_at: str


@dataclass(slots=True)
class CandidateProgress:
    signals: set[FailureKind] = field(default_factory=set)
    stage_hashes: dict[str, str | None] = field(
        default_factory=lambda: {"s3": None, "s4": None, "s5": None}
    )
    evidence_hashes: dict[str, str | None] = field(
        default_factory=lambda: {
            "g3_gate": None,
            "audit": None,
            "verification": None,
            "headline": None,
            "baseline": None,
        }
    )
    last_completed_stage: str | None = None
    candidate_release_id: str | None = None
    primary_grounded_papers: int | None = None


def _load_json_object(path: Path, *, code: str = "invalid_json") -> dict[str, Any]:
    if not path.is_file():
        raise ExportError("required_artifact_missing", path.as_posix(), kind="incomplete")
    if path.is_symlink():
        raise ExportError("symlink_artifact_forbidden", path.as_posix())
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(code, path.as_posix()) from exc
    if not isinstance(value, dict):
        raise ExportError(code, f"{path}:root_not_object")
    return value


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ExportError("required_artifact_missing", path.as_posix(), kind="incomplete")
    if path.is_symlink():
        raise ExportError("symlink_artifact_forbidden", path.as_posix())
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ExportError("invalid_parquet", path.as_posix()) from exc


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        return json.loads(frame.to_json(orient="records", date_format="iso"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExportError("parquet_rows_not_json_serializable", str(exc)) from exc


def _relative_under_root(path: Path, root: Path) -> str:
    if path.is_symlink():
        raise ExportError("symlink_artifact_forbidden", path.as_posix())
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ExportError("artifact_outside_repository", path.as_posix()) from exc


def _validate_artifact_reference(reference: Any, root: Path) -> None:
    path = root / reference.path
    _relative_under_root(path, root)
    if not path.is_file():
        raise ExportError("declared_artifact_missing", reference.path)
    if path.stat().st_size != reference.bytes:
        raise ExportError("declared_artifact_size_mismatch", reference.path)
    if sha256_file(path) != reference.sha256:
        raise ExportError("declared_artifact_hash_mismatch", reference.path)


def _load_run_record(path: Path, *, expected_stage: str | None = None) -> RunRecord:
    payload = _load_json_object(path, code="invalid_run_record")
    if payload.get("status") in {"partial", "failed"}:
        raise ExportError(
            "stage_not_complete",
            f"{payload.get('stage')}:{payload.get('status')}",
            kind="incomplete",
        )
    try:
        record = RunRecord.model_validate(payload)
    except ValidationError as exc:
        raise ExportError("invalid_run_record", f"{path}:{exc}") from exc
    if expected_stage is not None and record.stage != expected_stage:
        raise ExportError("stage_run_identity_mismatch", f"expected={expected_stage}")
    if record.status != "complete":
        raise ExportError("stage_not_complete", record.stage, kind="incomplete")
    return record


def _validate_run_tree(
    path: Path,
    *,
    source: ReleaseSource,
    config: QuestionConfig,
    expected_code_version: str | None,
    allow_dirty_demo: bool,
    visited: dict[str, RunRecord],
) -> RunRecord:
    relative = _relative_under_root(path, source.repository_root)
    if relative in visited:
        return visited[relative]
    record = _load_run_record(path)
    if record.question_id != source.question_id:
        raise ExportError("run_question_mismatch", relative)
    expected_config_hash = config_sha256(config)
    if record.config_sha256 != expected_config_hash:
        raise ExportError("run_config_hash_mismatch", relative)
    if expected_code_version is not None and record.code_version != expected_code_version:
        raise ExportError("mixed_code_versions", relative)
    if record.code_version.startswith("dirty:") and not allow_dirty_demo:
        raise ExportError("dirty_lineage_requires_override", relative)
    config_path = source.repository_root / record.config_path
    try:
        declared_config = load_question_config(config_path)
    except (OSError, ValueError) as exc:
        raise ExportError("run_config_path_invalid", record.config_path) from exc
    if config_sha256(declared_config) != expected_config_hash:
        raise ExportError("run_config_file_hash_mismatch", record.config_path)
    for reference in (*record.inputs, *record.outputs):
        _validate_artifact_reference(reference, source.repository_root)
    visited[relative] = record
    for upstream in record.upstream:
        if upstream.stage == "triage_probe":
            raise ExportError("triage_upstream_forbidden", upstream.run_record_path)
        upstream_path = source.repository_root / upstream.run_record_path
        if not upstream_path.is_file():
            raise ExportError("upstream_run_missing", upstream.run_record_path)
        if sha256_file(upstream_path) != upstream.run_record_sha256:
            raise ExportError("upstream_run_hash_mismatch", upstream.run_record_path)
        upstream_record = _validate_run_tree(
            upstream_path,
            source=source,
            config=config,
            expected_code_version=record.code_version,
            allow_dirty_demo=allow_dirty_demo,
            visited=visited,
        )
        if upstream_record.stage != upstream.stage or upstream_record.run_id != upstream.run_id:
            raise ExportError("upstream_run_identity_mismatch", upstream.run_record_path)
    return record


def _load_selected_lineage(
    source: ReleaseSource,
    config: QuestionConfig,
    *,
    allow_dirty_demo: bool,
) -> tuple[dict[str, RunRecord], StageRunHashes]:
    selected: dict[str, RunRecord] = {}
    selected_hashes: dict[str, str] = {}
    visited: dict[str, RunRecord] = {}
    code_version: str | None = None
    previous: tuple[str, RunRecord, str] | None = None
    for stage in ("s3", "s4", "s5"):
        path = source.stage_run_paths[stage]
        record = _validate_run_tree(
            path,
            source=source,
            config=config,
            expected_code_version=code_version,
            allow_dirty_demo=allow_dirty_demo,
            visited=visited,
        )
        if record.stage != stage:
            raise ExportError("stage_run_identity_mismatch", f"{path}:expected={stage}")
        if code_version is None:
            code_version = record.code_version
        run_hash = sha256_file(path)
        if previous is not None:
            previous_stage, previous_record, previous_hash = previous
            matches = [
                item
                for item in record.upstream
                if item.stage == previous_stage
                and item.run_id == previous_record.run_id
                and item.run_record_sha256 == previous_hash
            ]
            if len(matches) != 1:
                raise ExportError("selected_stage_chain_missing", f"{previous_stage}->{stage}")
        selected[stage] = record
        selected_hashes[stage] = run_hash
        previous = (stage, record, run_hash)
    return selected, StageRunHashes.model_validate(selected_hashes)


def _parse_json_cell(value: Any, *, field_name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExportError("invalid_canonical_json_cell", field_name) from exc
    return value


def _validate_ledgers(
    source: ReleaseSource,
    config: QuestionConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    papers_frame = _read_parquet(source.source_path("papers.parquet"))
    findings_frame = _read_parquet(source.source_path("findings.parquet"))
    expected_paper_columns = set(PaperRecord.model_fields)
    if set(papers_frame.columns) != expected_paper_columns:
        raise ExportError(
            "paper_ledger_schema_mismatch",
            f"missing={sorted(expected_paper_columns - set(papers_frame.columns))}:"
            f"extra={sorted(set(papers_frame.columns) - expected_paper_columns)}",
            kind="reconciliation",
        )
    expected_mod_columns = {f"mod__{moderator.name}" for moderator in config.moderators}
    fixed_finding_columns = set(FindingRow.model_fields) - {"moderators"}
    expected_finding_columns = fixed_finding_columns | expected_mod_columns
    if set(findings_frame.columns) != expected_finding_columns:
        raise ExportError(
            "finding_ledger_schema_mismatch",
            f"missing={sorted(expected_finding_columns - set(findings_frame.columns))}:"
            f"extra={sorted(set(findings_frame.columns) - expected_finding_columns)}",
            kind="reconciliation",
        )

    raw_papers = _frame_records(papers_frame)
    raw_findings = _frame_records(findings_frame)
    papers: list[dict[str, Any]] = []
    for raw in raw_papers:
        for field_name in ("alternate_doc_ids", "query_families", "search_result_ids"):
            raw[field_name] = _parse_json_cell(raw.get(field_name), field_name=field_name)
        try:
            paper = PaperRecord.model_validate(raw)
        except ValidationError as exc:
            raise ExportError(
                "paper_ledger_contract_invalid",
                str(exc),
                kind="reconciliation",
            ) from exc
        papers.append(paper.model_dump(mode="json", exclude_none=False))

    findings: list[dict[str, Any]] = []
    for raw in raw_findings:
        for field_name in ("evidence_lines", "normalization_warnings"):
            raw[field_name] = _parse_json_cell(raw.get(field_name), field_name=field_name)
        moderators = {
            moderator.name: raw.pop(f"mod__{moderator.name}") for moderator in config.moderators
        }
        raw["moderators"] = moderators
        try:
            finding = validate_finding_row(raw, config)
        except ValidationError as exc:
            raise ExportError(
                "finding_ledger_contract_invalid",
                str(exc),
                kind="reconciliation",
            ) from exc
        findings.append(finding.model_dump(mode="json", exclude_none=False))

    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    if len(paper_by_id) != len(papers):
        raise ExportError("duplicate_paper_id", kind="reconciliation")
    finding_by_id = {finding["finding_id"]: finding for finding in findings}
    if len(finding_by_id) != len(findings):
        raise ExportError("duplicate_finding_id", kind="reconciliation")
    orphans = sorted({finding["paper_id"] for finding in findings} - set(paper_by_id))
    if orphans:
        raise ExportError("orphan_findings", ",".join(orphans), kind="reconciliation")

    config_hash = config_sha256(config)
    if any(paper["config_sha256"] != config_hash for paper in papers):
        raise ExportError("paper_config_hash_mismatch", kind="reconciliation")
    finding_counts: dict[str, int] = {paper_id: 0 for paper_id in paper_by_id}
    tuples: set[tuple[str, str, str]] = set()
    for finding in findings:
        paper = paper_by_id[finding["paper_id"]]
        if finding["doc_id"] != paper["doc_id"]:
            raise ExportError(
                "finding_paper_doc_id_mismatch",
                finding["finding_id"],
                kind="reconciliation",
            )
        if finding["map_result_id"] != paper["map_result_id"]:
            raise ExportError(
                "finding_paper_map_result_mismatch",
                finding["finding_id"],
                kind="reconciliation",
            )
        if finding["cfghash"] != paper["cfghash"]:
            raise ExportError(
                "finding_paper_cfghash_mismatch",
                finding["finding_id"],
                kind="reconciliation",
            )
        finding_counts[finding["paper_id"]] += 1
        tuples.add(
            (finding["prompt_version"], finding["schema_version"], finding["cfghash"])
        )
    if len(tuples) > 1:
        raise ExportError("mixed_extraction_tuples", kind="reconciliation")
    for paper_id, paper in paper_by_id.items():
        if finding_counts[paper_id] != paper["accepted_finding_count"]:
            raise ExportError(
                "paper_accepted_count_mismatch",
                paper_id,
                kind="reconciliation",
            )
        if (
            paper["map_status"] == "success"
            and paper["accepted_finding_count"] + paper["quarantined_finding_count"]
            != paper["raw_finding_count"]
        ):
            raise ExportError("paper_raw_count_mismatch", paper_id, kind="reconciliation")
    return papers, findings, papers_frame, findings_frame


def _validate_verification(
    findings: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    required_metadata = {"provider", "model", "prompt_version", "prompt_sha256"}
    missing = sorted(required_metadata - set(artifact))
    if missing:
        raise ExportError("verification_metadata_missing", ",".join(missing))
    if not SHA256_RE.fullmatch(str(artifact.get("prompt_sha256", ""))):
        raise ExportError("verification_prompt_hash_invalid")
    try:
        return reconcile_verification(findings, artifact)
    except ValueError as exc:
        raise ExportError(
            "verification_reconciliation_failed",
            str(exc),
            kind="reconciliation",
        ) from exc


def _validate_audit(artifact: Mapping[str, Any]) -> tuple[int, int]:
    legacy_aliases = {"audit_correct", "audit_total"} & set(artifact)
    if legacy_aliases:
        raise ExportError(
            "audit_legacy_count_alias_forbidden",
            ",".join(sorted(legacy_aliases)),
        )
    try:
        audit = AuditRecord.model_validate(artifact)
    except ValidationError as exc:
        raise ExportError(
            "audit_contract_invalid",
            str(exc),
            kind="reconciliation",
        ) from exc
    # Sampling mode audits exactly 20 rows; census mode (2026-08-16 execution amendment)
    # audits the ENTIRE candidate pool when it holds fewer than 20 rows, so the recorded
    # requested size equals the total and both sit below 20.
    census_audit = 0 < audit.total_count < 20 and audit.total_count == audit.requested_sample_size
    if audit.total_count != 20 and not census_audit:
        raise ExportError(
            "audit_must_have_exactly_20_rows",
            str(audit.total_count),
            kind="unvalidated",
        )
    if not audit.anchor_results:
        raise ExportError("audit_anchor_results_missing")
    if not all(audit.anchor_results.values()):
        raise ExportError("audit_anchor_failed", kind="unvalidated")
    return audit.correct_count, audit.total_count


def _paper_balanced_majority(primary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("increase", "no_effect", "decrease")
    by_paper: dict[str, list[Mapping[str, Any]]] = {}
    for row in primary_rows:
        by_paper.setdefault(str(row["paper_id"]), []).append(row)
    mass = {label: 0.0 for label in labels}
    for rows in by_paper.values():
        weight = 1 / len(rows)
        for row in rows:
            mass[str(row["effect_direction"])] += weight
    total = sum(mass.values())
    maximum = max(mass.values()) if mass else 0.0
    winners = [label for label in labels if math.isclose(mass[label], maximum, abs_tol=1e-12)]
    return {
        "direction": winners[0] if len(winners) == 1 else "mixed",
        "agreement": maximum / total if total else None,
    }


def _validate_baseline(
    artifact: Mapping[str, Any],
    *,
    config: QuestionConfig,
    cohort_sha256: str,
    primary_rows: Sequence[Mapping[str, Any]],
) -> None:
    if artifact.get("cohort_hash") != cohort_sha256:
        raise ExportError("baseline_cohort_hash_mismatch")
    status = artifact.get("status")
    source = artifact.get("source")
    if status not in {"complete", "unavailable"} or source not in {"live_llm", "fixture_stub"}:
        raise ExportError("baseline_status_or_source_invalid")
    expected_majority = _paper_balanced_majority(primary_rows)
    majority = artifact.get("majority")
    if not isinstance(majority, Mapping) or dict(majority) != expected_majority:
        raise ExportError("baseline_majority_mismatch")
    fixture = bool(config.demo and config.demo.fixture_mode)
    if source == "fixture_stub" and not fixture:
        raise ExportError("fixture_baseline_forbidden_in_production")
    if fixture and source != "fixture_stub":
        raise ExportError("fixture_baseline_must_be_stub")
    llm = artifact.get("llm")
    failure_code = artifact.get("failure_code")
    if status == "complete":
        if not isinstance(llm, Mapping) or failure_code is not None:
            raise ExportError("complete_baseline_payload_invalid")
        required = {"provider", "model", "prompt_sha256", "raw_response_sha256", "paragraph"}
        if set(llm) != required or not all(
            SHA256_RE.fullmatch(str(llm[field]))
            for field in ("prompt_sha256", "raw_response_sha256")
        ):
            raise ExportError("baseline_llm_provenance_invalid")
    elif llm is not None or not isinstance(failure_code, str) or not failure_code:
        raise ExportError("unavailable_baseline_payload_invalid")
    attempted_at = artifact.get("attempted_at")
    try:
        parsed = datetime.fromisoformat(str(attempted_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExportError("baseline_attempted_at_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExportError("baseline_attempted_at_timezone_missing")


def _validate_moderator_table(frame: pd.DataFrame, config: QuestionConfig) -> None:
    required = {"moderator", "status", "role"}
    if not required.issubset(frame.columns):
        raise ExportError("moderator_table_schema_invalid")
    names = frame["moderator"].tolist()
    expected = [moderator.name for moderator in config.moderators]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ExportError("moderator_table_family_mismatch", kind="reconciliation")
    if any(
        row["role"] != {moderator.name: moderator.role for moderator in config.moderators}[
            row["moderator"]
        ]
        for row in _frame_records(frame)
    ):
        raise ExportError("moderator_table_role_mismatch", kind="reconciliation")


def _validate_contradictions(
    frame: pd.DataFrame,
    *,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required = {
        "pair_id",
        "outcome_family",
        "left_direction",
        "right_direction",
        "shared_context_fields",
        "shared_context_count",
        "distance",
        "distance_components",
        "left_citation",
        "right_citation",
    }
    if set(frame.columns) != required:
        raise ExportError("contradiction_table_schema_invalid")
    paper_ids = {str(paper["paper_id"]) for paper in papers}
    finding_ids = {str(finding["finding_id"]) for finding in findings}
    pairs = _frame_records(frame)
    for row in pairs:
        for field_name in (
            "shared_context_fields",
            "distance_components",
            "left_citation",
            "right_citation",
        ):
            row[field_name] = _parse_json_cell(row[field_name], field_name=field_name)
        if row["left_direction"] == row["right_direction"] or {
            row["left_direction"],
            row["right_direction"],
        } != {"increase", "decrease"}:
            raise ExportError("contradiction_directions_invalid", str(row["pair_id"]))
        if row["shared_context_count"] < 2 or row["shared_context_count"] != len(
            row["shared_context_fields"]
        ):
            raise ExportError("contradiction_shared_context_invalid", str(row["pair_id"]))
        for side in ("left_citation", "right_citation"):
            citation = row[side]
            if not isinstance(citation, Mapping):
                raise ExportError("contradiction_citation_invalid", str(row["pair_id"]))
            if (
                citation.get("paper_id") not in paper_ids
                or citation.get("finding_id") not in finding_ids
            ):
                raise ExportError(
                    "contradiction_citation_orphan",
                    str(row["pair_id"]),
                    kind="reconciliation",
                )
    return pairs


def _validate_evidence_gaps(frame: pd.DataFrame, config: QuestionConfig) -> list[dict[str, Any]]:
    required = {
        "cell_id",
        "primary_endpoint",
        "axis_values",
        "n_papers_total",
        "n_papers_grounded",
        "n_findings",
        "grounded_fraction",
        "classifiable_fraction",
        "paper_entropy",
        "status",
    }
    if set(frame.columns) != required:
        raise ExportError("evidence_gap_table_schema_invalid")
    assert config.variant_b is not None
    by_name = {moderator.name: moderator for moderator in config.moderators}
    expected_count = len(config.variant_b.primary_endpoints)
    for axis in config.variant_b.axes:
        expected_count *= len(by_name[axis].declared_levels)
    if len(frame) != expected_count:
        raise ExportError("evidence_gap_grid_size_mismatch", kind="reconciliation")
    rows = _frame_records(frame)
    identities: set[str] = set()
    for row in rows:
        axis_values = _parse_json_cell(row["axis_values"], field_name="axis_values")
        if not isinstance(axis_values, Mapping) or set(axis_values) != set(config.variant_b.axes):
            raise ExportError("evidence_gap_axis_values_invalid")
        if row["primary_endpoint"] not in config.variant_b.primary_endpoints:
            raise ExportError("evidence_gap_endpoint_invalid")
        for axis, value in axis_values.items():
            if value not in by_name[axis].declared_levels:
                raise ExportError("evidence_gap_axis_level_invalid", axis)
        identity = canonical_json_bytes(
            {"primary_endpoint": row["primary_endpoint"], "axis_values": axis_values}
        ).decode("utf-8")
        if identity in identities:
            raise ExportError("duplicate_evidence_gap_cell", kind="reconciliation")
        identities.add(identity)
        grounded = int(row["n_papers_grounded"])
        expected_status = "empty" if grounded == 0 else "sparse" if grounded < 5 else "supported"
        if row["status"] != expected_status:
            raise ExportError("evidence_gap_status_mismatch", str(row["cell_id"]))
    return rows


def _validate_checkpoint_artifact(artifact: Mapping[str, Any], m4_status: str) -> None:
    if artifact.get("status") == "not_applicable":
        if set(artifact) != {"status", "reason"} or artifact.get("reason") not in {
            "m4_completed",
            "m4_not_run",
        }:
            raise ExportError("m4_checkpoint_placeholder_invalid")
        if m4_status == "incomplete":
            raise ExportError("incomplete_m4_requires_frozen_checkpoint")
        return
    if set(artifact) != {"status", "source_checkpoint_sha256", "checkpoint"}:
        raise ExportError("m4_checkpoint_wrapper_shape_invalid")
    if artifact.get("status") != "frozen_incomplete" or m4_status != "incomplete":
        raise ExportError("m4_checkpoint_wrapper_status_invalid")
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ExportError("m4_checkpoint_source_invalid")
    try:
        validate_checkpoint(checkpoint)
    except ValueError as exc:
        raise ExportError("m4_checkpoint_source_invalid", str(exc)) from exc
    if checkpoint_sha256(checkpoint) != artifact.get("source_checkpoint_sha256"):
        raise ExportError("m4_checkpoint_source_hash_mismatch")


def _validate_typed_resampling_artifacts(
    *,
    variant: str,
    selection_reason: str | None,
    tree: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
) -> None:
    tree_status = tree.get("status")
    if tree_status not in {"supporting", "exploratory", "not_run", "incomplete"}:
        raise ExportError("tree_status_invalid")
    if not isinstance(tree.get("nodes"), list):
        raise ExportError("tree_nodes_invalid")
    if tree_status in {"not_run", "incomplete"} and tree["nodes"]:
        raise ExportError("unfinished_tree_nodes_must_be_empty")
    if variant == "A" and tree_status != "supporting":
        raise ExportError("variant_a_requires_supporting_tree")
    if selection_reason == "g3_story_not_viable" and tree_status != "not_run":
        raise ExportError("g3_story_variant_requires_not_run_tree")
    if selection_reason == "m4_no_moderator" and tree_status != "exploratory":
        raise ExportError("m4_no_moderator_requires_exploratory_tree")
    if selection_reason == "m4_incomplete" and tree_status != "incomplete":
        raise ExportError("m4_incomplete_requires_incomplete_tree")

    if permutation.get("status") not in {"complete", "not_run", "incomplete"}:
        raise ExportError("permutation_status_invalid")
    for name in ("success_count", "attempt_count"):
        if name not in permutation:
            raise ExportError("permutation_counts_missing", name)
    success_count = permutation["success_count"]
    attempt_count = permutation["attempt_count"]
    if permutation["status"] == "complete":
        if not isinstance(success_count, int) or not isinstance(attempt_count, int):
            raise ExportError("permutation_counts_invalid")
        if success_count > 100 or attempt_count > 125 or success_count > attempt_count:
            raise ExportError("permutation_budget_invalid")
    elif success_count is not None and not isinstance(success_count, int):
        raise ExportError("permutation_counts_invalid")

    if set(bootstrap) < {"entropy", "model_stability"}:
        raise ExportError("bootstrap_components_missing")
    entropy = bootstrap["entropy"]
    stability = bootstrap["model_stability"]
    if not isinstance(entropy, Mapping) or entropy.get("status") != "complete":
        raise ExportError("entropy_bootstrap_must_be_complete")
    if not isinstance(stability, Mapping) or stability.get("status") not in {
        "complete",
        "not_run",
        "incomplete",
    }:
        raise ExportError("model_stability_status_invalid")
    expected_stability = {
        "A": "complete",
        "g3_story_not_viable": "not_run",
        "m4_no_moderator": "complete",
        "m4_incomplete": "incomplete",
    }[variant if variant == "A" else str(selection_reason)]
    if stability.get("status") != expected_stability:
        raise ExportError("model_stability_branch_mismatch")


def _validate_g3_gate(
    artifact: Mapping[str, Any],
    *,
    config_hash: str,
    cohort_sha256: str,
    audit_hash: str,
    verification_hash: str,
    audit_correct: int,
    audit_total: int,
    quality_metrics: Mapping[str, Any],
) -> None:
    expected_hashes = {
        "config_sha256": config_hash,
        "cohort_sha256": cohort_sha256,
        "audit_sha256": audit_hash,
        "verification_sha256": verification_hash,
    }
    for name, expected in expected_hashes.items():
        if artifact.get(name) != expected:
            raise ExportError("g3_declared_hash_mismatch", name)
    if not isinstance(artifact.get("trust_passed"), bool) or not isinstance(
        artifact.get("story_passed"), bool
    ):
        raise ExportError("g3_boolean_fields_invalid")
    expected_action = (
        "block_release"
        if not artifact["trust_passed"]
        else "run_m4"
        if artifact["story_passed"]
        else "select_variant_b_story"
    )
    if artifact.get("action") != expected_action:
        raise ExportError("g3_action_mismatch")
    if not artifact["trust_passed"]:
        raise ExportError("g3_trust_failed", kind="unvalidated")
    quarantine = quality_metrics["quarantine"]["fraction"]
    agreement = quality_metrics["cross_model_agreement"]["fraction"]
    trust_rules = {
        str(rule.get("name")): rule for rule in artifact.get("trust_rules", []) or []
    }
    audit_rule_observed = (trust_rules.get("human_audit") or {}).get("observed")
    census_audit = (
        0 < audit_total < 20
        and isinstance(audit_rule_observed, Mapping)
        and audit_rule_observed.get("failed_ids_in_cohort") == []
        and audit_rule_observed.get("total") == audit_total
    )
    # Census mode (2026-08-16 execution amendment): the whole candidate pool was audited
    # and no audit-failed row remains in the release cohort; the ≥0.85 verification bar
    # applies to the release set, with the full exact-grounded rate re-anchored to the
    # bundled ledgers as the transparency figure.
    if census_audit:
        if quarantine is None or quarantine > 0.10:
            raise ExportError("g3_trust_values_fail_recompute", kind="unvalidated")
        cross_rule_observed = (trust_rules.get("cross_model_agreement") or {}).get("observed")
        if not isinstance(cross_rule_observed, Mapping):
            raise ExportError("g3_verification_fails_recompute", kind="unvalidated")
        declared_full = cross_rule_observed.get("all_exact_grounded")
        release_rate = cross_rule_observed.get("release_set")
        if (
            agreement is None
            or declared_full is None
            or abs(float(declared_full) - float(agreement)) > 1e-9
            or release_rate is None
            or float(release_rate) < 0.85
        ):
            raise ExportError("g3_verification_fails_recompute", kind="unvalidated")
        return
    if audit_total != 20 or audit_correct < 17 or quarantine is None or quarantine > 0.10:
        raise ExportError("g3_trust_values_fail_recompute", kind="unvalidated")
    if agreement is None or agreement < 0.85:
        raise ExportError("g3_verification_fails_recompute", kind="unvalidated")


def _require_stage_outputs(
    source: ReleaseSource,
    stage_runs: Mapping[str, RunRecord],
) -> None:
    required = {
        "s4": {"papers.parquet", "findings.parquet"},
        "s5": {
            "moderators.parquet",
            "m4_checkpoint.json",
            "tree.json",
            "contradictions.parquet",
            "evidence_gaps.parquet",
            "bootstrap.json",
            "permutation.json",
            "m4_gate.json",
            "headline.json",
        },
    }
    expected_paths = {
        "s4": {
            source.source_path("papers.parquet").resolve(),
            source.source_path("findings.parquet").resolve(),
        },
        "s5": {source.source_path(f"analysis/{name}").resolve() for name in required["s5"]},
    }
    for stage, targets in expected_paths.items():
        declared = {
            (source.repository_root / reference.path).resolve()
            for reference in stage_runs[stage].outputs
        }
        missing = targets - declared
        if missing:
            raise ExportError(
                "stage_required_outputs_not_declared",
                f"{stage}:{sorted(path.name for path in missing)}",
            )


def _validate_m4_gate(
    artifact: Mapping[str, Any],
    *,
    config: QuestionConfig,
    cohort_sha256: str,
    g3_action: str,
) -> tuple[str, str | None]:
    status = artifact.get("status")
    variant = artifact.get("selected_variant")
    reason = artifact.get("selection_reason")
    if status not in {"complete", "not_run", "incomplete"} or variant not in {"A", "B"}:
        raise ExportError("m4_gate_status_or_variant_invalid")
    if artifact.get("config_sha256") != config_sha256(config):
        raise ExportError("m4_gate_config_hash_mismatch")
    if artifact.get("cohort_hash") != cohort_sha256:
        raise ExportError("m4_gate_cohort_hash_mismatch")
    for key in ("input_hashes", "output_hashes"):
        values = artifact.get(key)
        if not isinstance(values, Mapping) or not values or any(
            not SHA256_RE.fullmatch(str(value)) for value in values.values()
        ):
            raise ExportError("m4_gate_hash_map_invalid", key)
    if artifact.get("seed") != config.analysis.seed:
        raise ExportError("m4_gate_seed_mismatch")

    tested = {moderator.name for moderator in config.moderators if moderator.role == "tested"}
    moderator_results = artifact.get("moderators")
    if not isinstance(moderator_results, list):
        raise ExportError("m4_gate_moderator_results_invalid")
    observed_names = {
        row.get("moderator") for row in moderator_results if isinstance(row, Mapping)
    }
    if observed_names != tested:
        raise ExportError("m4_gate_moderator_family_mismatch")

    if g3_action == "select_variant_b_story":
        if (status, variant, reason) != ("not_run", "B", "g3_story_not_viable"):
            raise ExportError("g3_story_m4_branch_mismatch")
    elif g3_action != "run_m4":
        raise ExportError("blocked_g3_cannot_reach_m4", kind="unvalidated")
    elif status == "incomplete":
        if variant != "B" or reason != "m4_incomplete":
            raise ExportError("incomplete_m4_branch_mismatch")
    elif status == "complete" and variant == "B":
        if reason != "m4_no_moderator" or artifact.get("selected_moderator") is not None:
            raise ExportError("completed_variant_b_m4_branch_mismatch")
        if any(row.get("passed") is True for row in moderator_results):
            raise ExportError("variant_b_contains_passing_moderator")
    elif status == "complete" and variant == "A":
        selected = artifact.get("selected_moderator")
        winners = [row for row in moderator_results if row.get("moderator") == selected]
        if len(winners) != 1 or winners[0].get("passed") is not True:
            raise ExportError("variant_a_selected_moderator_invalid")
        rules = winners[0].get("rules")
        if not isinstance(rules, list) or not rules or any(
            set(rule) != {"name", "observed", "threshold", "passed"}
            for rule in rules
            if isinstance(rule, Mapping)
        ):
            raise ExportError("variant_a_rule_inventory_invalid")
    else:
        raise ExportError("m4_branch_combination_invalid")
    return str(variant), str(reason) if reason is not None else None


def _validate_headline(
    headline: Mapping[str, Any],
    *,
    variant: str,
    selection_reason: str | None,
    m4_gate: Mapping[str, Any],
    config: QuestionConfig,
    primary_rows: Sequence[Mapping[str, Any]],
    contradiction_rows: Sequence[Mapping[str, Any]],
    evidence_gap_rows: Sequence[Mapping[str, Any]],
) -> None:
    if headline.get("narrative_variant") != variant:
        raise ExportError("headline_variant_mismatch")
    if headline.get("cohort_definition") != "primary_grounded_unflagged":
        raise ExportError("headline_cohort_definition_invalid")
    if headline.get("analysis_labels") != ["increase", "no_effect", "decrease"]:
        raise ExportError("headline_analysis_labels_invalid")
    if variant == "A":
        moderator = headline.get("moderator")
        if not isinstance(moderator, Mapping):
            raise ExportError("variant_a_moderator_invalid")
        name = str(moderator.get("name"))
        if name != m4_gate.get("selected_moderator"):
            raise ExportError("headline_m4_selected_moderator_mismatch")
        comparison_subset = headline.get("comparison_subset")
        global_baseline = headline.get("global_baseline")
        within_regime = headline.get("within_regime")
        if not all(
            isinstance(value, Mapping)
            for value in (comparison_subset, global_baseline, within_regime)
        ):
            raise ExportError("variant_a_comparison_fields_invalid")
        result = {
            "moderator": name,
            "k": moderator.get("k"),
            "delta_ll": moderator.get("delta_ll"),
            "positive_folds": moderator.get("positive_folds"),
            "westfall_young_p": moderator.get("westfall_young_p"),
            "comparison": {
                "n_findings": comparison_subset.get("n_findings"),
                "n_papers": comparison_subset.get("n_papers"),
                "coverage_papers": comparison_subset.get("coverage_papers"),
                "global_mode": global_baseline.get("modal_direction"),
                "agreement_q": global_baseline.get("agreement_q"),
                "agreement_p": within_regime.get("agreement_p"),
                "absolute_gain": within_regime.get("absolute_gain"),
                "contrast": headline.get("contrast"),
            },
            "stability": headline.get("stability"),
        }
        display_name = (
            config.demo.moderator_display_names.get(name)
            if config.demo is not None
            else None
        )
        expected = build_variant_a_headline(result, moderator_display_name=display_name)
        if dict(headline) != expected:
            raise ExportError("variant_a_headline_recompute_mismatch")
        if int(comparison_subset["n_papers"]) > len({row["paper_id"] for row in primary_rows}):
            raise ExportError("variant_a_comparison_exceeds_primary_cohort")
        return

    if selection_reason not in {
        "g3_story_not_viable",
        "m4_no_moderator",
        "m4_incomplete",
    }:
        raise ExportError("variant_b_selection_reason_invalid")
    if headline.get("selection_reason") != selection_reason:
        raise ExportError("variant_b_selection_reason_mismatch")
    expected_residual = residual_summary(contradiction_rows)
    if headline.get("residuals") != expected_residual:
        raise ExportError("variant_b_residual_summary_mismatch")
    sparse = sum(row["status"] in {"empty", "sparse"} for row in evidence_gap_rows)
    disagreement = headline.get("disagreement")
    if not isinstance(disagreement, Mapping):
        raise ExportError("variant_b_disagreement_invalid")
    paper_summary = paper_modal_summary(primary_rows)
    if disagreement.get("n_papers") != paper_summary["n_papers_classifiable"]:
        raise ExportError("variant_b_paper_count_mismatch")
    if disagreement.get("n_findings") != len(primary_rows):
        raise ExportError("variant_b_finding_count_mismatch")
    observed_entropy = disagreement.get("paper_entropy")
    expected_entropy = paper_summary["primary"]["normalized_entropy"]
    if observed_entropy is None or expected_entropy is None or not math.isclose(
        float(observed_entropy), float(expected_entropy), abs_tol=1e-12
    ):
        raise ExportError("variant_b_entropy_recompute_mismatch")
    finding_majority = paper_balanced_finding_summary(primary_rows)["majority"]
    expected = build_variant_b_headline(
        selection_reason=selection_reason,
        disagreement=disagreement,
        residuals=expected_residual,
        sparse_or_empty_cells=sparse,
        total_cells=len(evidence_gap_rows),
        m4_failures=headline.get("m4", {}).get("failures", []),
        global_baseline={
            "modal_direction": finding_majority["modal_direction"],
            "agreement_q": finding_majority["agreement"],
        },
    )
    if dict(headline) != expected:
        raise ExportError("variant_b_headline_recompute_mismatch")


def _validate_trace(artifact: Mapping[str, Any], *, variant: str, reason: str | None) -> None:
    status = artifact.get("status")
    if variant == "B":
        expected_reason = {
            "g3_story_not_viable": "g3_story_not_viable",
            "m4_no_moderator": "m4_selected_variant_b",
            "m4_incomplete": "m4_incomplete",
        }[str(reason)]
        if set(artifact) != {"status", "reason"} or (
            status,
            artifact.get("reason"),
        ) != ("not_run", expected_reason):
            raise ExportError("variant_b_trace_invalid")
        return
    allowed = {
        "not_run",
        "proposed",
        "approved",
        "rejected",
        "kept_exploratory",
        "discarded",
        "indeterminate",
    }
    if status not in allowed:
        raise ExportError("variant_a_trace_status_invalid")
    if status == "not_run" and artifact.get("reason") != "human_approval_unavailable":
        raise ExportError("variant_a_not_run_trace_reason_invalid")


def _cohort_sha256(config: QuestionConfig, primary_rows: Sequence[Mapping[str, Any]]) -> str:
    del config  # The release ID binds the configuration independently.
    try:
        return cohort_sha256(primary_rows)
    except CohortContractError as exc:
        raise ExportError(
            "primary_cohort_canonicalization_failed",
            str(exc),
            kind="reconciliation",
        ) from exc


def _derive_release_id(
    *,
    question_id: str,
    corpus_role: str,
    cohort_sha256: str,
    config_sha256_value: str,
    code_version: str,
    stage_hashes: StageRunHashes,
    evidence_hashes: EvidenceHashes,
) -> str:
    digest = hash_canonical(
        {
            "question_id": question_id,
            "corpus_role": corpus_role,
            "cohort_sha256": cohort_sha256,
            "config_sha256": config_sha256_value,
            "code_version": code_version,
            "stage_run_sha256s": stage_hashes.model_dump(),
            "evidence_sha256s": evidence_hashes.model_dump(),
        }
    )
    return f"{question_id}-{corpus_role}-{digest[:20]}"


def validate_release_source(
    source: ReleaseSource,
    config: QuestionConfig,
    *,
    explicit_fixture: bool = False,
    allow_dirty_demo: bool = False,
) -> ValidatedRelease:
    """Validate every scientific, lineage, and branch invariant before staging bytes."""

    if config.status != "locked" or config.question_id != source.question_id:
        raise ExportError("locked_config_or_question_mismatch")
    try:
        config.authorize_stage(
            "s7",
            explicit_fixture=explicit_fixture,
            live_provider=False,
        )
    except ValueError as exc:
        raise ExportError("fixture_runtime_guard_failed", str(exc)) from exc
    for directory in (source.processed_dir, source.analysis_dir, source.extracted_dir):
        _relative_under_root(directory, source.repository_root)

    stage_runs, stage_hashes = _load_selected_lineage(
        source,
        config,
        allow_dirty_demo=allow_dirty_demo,
    )
    _require_stage_outputs(source, stage_runs)
    papers, findings, papers_frame, findings_frame = _validate_ledgers(source, config)

    json_artifacts = {
        path: _load_json_object(source.source_path(path))
        for path in sorted((ANALYSIS_FILES | ROOT_SOURCE_FILES) - PARQUET_PATHS)
    }
    verification = json_artifacts["verification.json"]
    _validate_verification(findings, verification)
    audit = json_artifacts["audit.json"]
    audit_correct, audit_total = _validate_audit(audit)
    try:
        primary_rows = primary_cohort(
            papers,
            findings,
            verification,
            primary_family=str(config.outcomes.primary_family),
        )
    except ValueError as exc:
        raise ExportError(
            "primary_cohort_reconciliation_failed",
            str(exc),
            kind="reconciliation",
        ) from exc
    if not primary_rows:
        raise ExportError("primary_cohort_empty", kind="unvalidated")
    cohort_hash = _cohort_sha256(config, primary_rows)
    try:
        metrics = compute_quality_metrics(
            papers,
            findings,
            verification,
            primary_family=str(config.outcomes.primary_family),
        )
    except ValueError as exc:
        raise ExportError(
            "quality_metric_reconciliation_failed",
            str(exc),
            kind="reconciliation",
        ) from exc

    table_frames = {
        "papers.parquet": papers_frame,
        "findings.parquet": findings_frame,
        **{
            path: _read_parquet(source.source_path(path))
            for path in sorted(ANALYSIS_FILES & PARQUET_PATHS)
        },
    }
    _validate_moderator_table(table_frames["analysis/moderators.parquet"], config)
    contradiction_rows = _validate_contradictions(
        table_frames["analysis/contradictions.parquet"],
        papers=papers,
        findings=findings,
    )
    evidence_gap_rows = _validate_evidence_gaps(
        table_frames["analysis/evidence_gaps.parquet"],
        config,
    )

    audit_hash = sha256_file(source.source_path("audit.json"))
    verification_hash = sha256_file(source.source_path("verification.json"))
    g3 = json_artifacts["g3_gate.json"]
    _validate_g3_gate(
        g3,
        config_hash=config_sha256(config),
        cohort_sha256=cohort_hash,
        audit_hash=audit_hash,
        verification_hash=verification_hash,
        audit_correct=audit_correct,
        audit_total=audit_total,
        quality_metrics=metrics,
    )
    m4_gate = json_artifacts["analysis/m4_gate.json"]
    variant, selection_reason = _validate_m4_gate(
        m4_gate,
        config=config,
        cohort_sha256=cohort_hash,
        g3_action=str(g3["action"]),
    )
    _validate_checkpoint_artifact(
        json_artifacts["analysis/m4_checkpoint.json"],
        str(m4_gate["status"]),
    )
    _validate_typed_resampling_artifacts(
        variant=variant,
        selection_reason=selection_reason,
        tree=json_artifacts["analysis/tree.json"],
        bootstrap=json_artifacts["analysis/bootstrap.json"],
        permutation=json_artifacts["analysis/permutation.json"],
    )
    _validate_headline(
        json_artifacts["analysis/headline.json"],
        variant=variant,
        selection_reason=selection_reason,
        m4_gate=m4_gate,
        config=config,
        primary_rows=primary_rows,
        contradiction_rows=contradiction_rows,
        evidence_gap_rows=evidence_gap_rows,
    )
    _validate_trace(json_artifacts["trace.json"], variant=variant, reason=selection_reason)
    _validate_baseline(
        json_artifacts["baseline.json"],
        config=config,
        cohort_sha256=cohort_hash,
        primary_rows=primary_rows,
    )

    checkpoint = json_artifacts["analysis/m4_checkpoint.json"]
    if selection_reason == "m4_incomplete":
        if stage_runs["s5"].completion_mode != "frozen_incomplete":
            raise ExportError("m4_incomplete_requires_frozen_s5_run")
        if stage_runs["s5"].checkpoint_sha256 != checkpoint.get(
            "source_checkpoint_sha256"
        ):
            raise ExportError("frozen_s5_checkpoint_hash_mismatch")
    elif stage_runs["s5"].completion_mode != "normal":
        raise ExportError("completed_m4_requires_normal_s5_run")

    evidence_hashes = EvidenceHashes(
        g3_gate=sha256_file(source.source_path("g3_gate.json")),
        audit=audit_hash,
        verification=verification_hash,
        headline=sha256_file(source.source_path("analysis/headline.json")),
        baseline=sha256_file(source.source_path("baseline.json")),
    )
    release_id = _derive_release_id(
        question_id=source.question_id,
        corpus_role=source.corpus_role,
        cohort_sha256=cohort_hash,
        config_sha256_value=config_sha256(config),
        code_version=stage_runs["s5"].code_version,
        stage_hashes=stage_hashes,
        evidence_hashes=evidence_hashes,
    )
    searched_documents = {
        identifier
        for paper in papers
        for identifier in [paper["doc_id"], *paper["alternate_doc_ids"]]
    }
    paper_funnel = {
        "searched_documents": len(searched_documents),
        "identity_deduped_papers": len(papers),
        "deterministic_included_papers": sum(
            paper["screen_status"] == "included" for paper in papers
        ),
        "extraction_eligible_papers": sum(
            paper["screen_status"] == "included" and paper["eligible"] is True
            for paper in papers
        ),
        "primary_grounded_papers": len({row["paper_id"] for row in primary_rows}),
        "primary_grounded_findings": len(primary_rows),
    }
    quality = {
        "audit_correct": audit_correct,
        "audit_total": audit_total,
        "grounded_fraction": metrics["grounded"]["fraction"],
        "grounded_numerator": metrics["grounded"]["numerator"],
        "grounded_denominator": metrics["grounded"]["denominator"],
        "quarantine_fraction": metrics["quarantine"]["fraction"],
        "quarantine_numerator": metrics["quarantine"]["numerator"],
        "quarantine_denominator": metrics["quarantine"]["denominator"],
        "cross_model_agreement": metrics["cross_model_agreement"]["fraction"],
        "cross_model_agree": metrics["cross_model_agreement"]["numerator"],
        "cross_model_requested": metrics["cross_model_agreement"]["denominator"],
    }
    exclusions = {
        "mixed_or_unclear_fraction": metrics["mixed_or_unclear_exclusion"]["fraction"],
        "mixed_or_unclear_numerator": metrics["mixed_or_unclear_exclusion"]["numerator"],
        "mixed_or_unclear_denominator": metrics["mixed_or_unclear_exclusion"]["denominator"],
        "section_flagged_fraction": metrics["section_flagged_exclusion"]["fraction"],
        "section_flagged_numerator": metrics["section_flagged_exclusion"]["numerator"],
        "section_flagged_denominator": metrics["section_flagged_exclusion"]["denominator"],
        "verification_excluded_fraction": metrics["verification_exclusion"]["fraction"],
        "verification_excluded_numerator": metrics["verification_exclusion"]["numerator"],
        "verification_excluded_denominator": metrics["verification_exclusion"]["denominator"],
    }
    if any(value is None for value in quality.values()) or any(
        value is None for value in exclusions.values()
    ):
        raise ExportError("manifest_quality_scalar_undefined", kind="unvalidated")
    completed_at = stage_runs["s5"].completed_at
    if completed_at is None:
        raise ExportError("s5_completed_at_missing")
    return ValidatedRelease(
        source=source,
        config=config,
        papers=papers,
        findings=findings,
        primary_rows=primary_rows,
        json_artifacts=json_artifacts,
        table_frames=table_frames,
        stage_runs=stage_runs,
        stage_run_sha256s=stage_hashes,
        evidence_sha256s=evidence_hashes,
        cohort_sha256=cohort_hash,
        release_id=release_id,
        paper_funnel=paper_funnel,
        quality=quality,
        exclusions=exclusions,
        narrative_variant=variant,  # type: ignore[arg-type]
        created_at=completed_at.isoformat(),
    )


def _resolve_token(context: Mapping[str, Any], token: str) -> str:
    parts = token.split(".")
    value: Any = context
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            raise ExportError("template_token_missing", token, kind="unvalidated")
        value = value[part]
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int, float)):
        raise ExportError("template_token_type_invalid", token, kind="unvalidated")
    return str(value)


def spoken_word_count(rendered_script: str) -> int:
    marker = "## Spoken copy"
    if marker not in rendered_script:
        raise ExportError("script_spoken_copy_section_missing", kind="unvalidated")
    spoken = rendered_script.split(marker, maxsplit=1)[1]
    spoken = spoken.split("## Release disclosure", maxsplit=1)[0]
    return len(re.findall(r"\b[\w]+(?:-[\w]+)*\b", spoken, flags=re.UNICODE))


def render_demo_script(
    template: str,
    *,
    variant: Literal["A", "B"],
    manifest: Mapping[str, Any],
    headline: Mapping[str, Any],
) -> str:
    """Render only the variant's allowlisted scalar tokens and enforce the spoken cap."""

    tokens = TOKEN_RE.findall(template)
    if set(tokens) != VARIANT_TOKENS[variant]:
        raise ExportError(
            "template_token_allowlist_mismatch",
            f"missing={sorted(VARIANT_TOKENS[variant] - set(tokens))}:"
            f"extra={sorted(set(tokens) - VARIANT_TOKENS[variant])}",
            kind="unvalidated",
        )
    context = {"manifest": manifest, "headline": headline}
    rendered = TOKEN_RE.sub(lambda match: _resolve_token(context, match.group(1)), template)
    if UNRESOLVED_TOKEN_RE.search(rendered):
        raise ExportError("template_unresolved_token", kind="unvalidated")
    if spoken_word_count(rendered) > 225:
        raise ExportError("script_exceeds_225_spoken_words", kind="unvalidated")
    qualifier = str(manifest["corpus_qualifier"])
    branch_caveat = "not a causal" if variant == "A" else "not proof"
    if qualifier not in rendered or branch_caveat not in rendered:
        raise ExportError("script_required_caveat_missing", kind="unvalidated")
    return rendered


def _selected_release(validated: ValidatedRelease) -> SelectedRelease:
    return SelectedRelease(
        corpus_role=validated.source.corpus_role,
        release_id=validated.release_id,
        primary_grounded_papers=validated.paper_funnel["primary_grounded_papers"],
        stage_run_sha256s=validated.stage_run_sha256s,
        evidence_sha256s=validated.evidence_sha256s,
    )


def _v1_selection(validated: ValidatedRelease) -> ReleaseSelection:
    selected = _selected_release(validated)
    if selected.corpus_role != "v1":
        raise ExportError("initial_release_must_be_v1")
    papers = selected.primary_grounded_papers
    return ReleaseSelection(
        disposition="v1_frozen",
        frozen_v1_primary_papers=papers,
        selected_release=selected,
        scaled_attempt=None,
        rendered_disclosure=render_release_disclosure(
            "v1_frozen",
            frozen_v1_primary_papers=papers,
            selected_primary_papers=papers,
        ),
    )


def _scaled_selection(
    validated: ValidatedRelease,
    *,
    frozen_v1_primary_papers: int,
) -> ReleaseSelection:
    selected = _selected_release(validated)
    if selected.corpus_role != "scaled":
        raise ExportError("scaled_promotion_requires_scaled_role")
    attempt = ScaledAttempt(
        status="selected",
        failure_code=None,
        last_completed_stage="s5",
        candidate_release_id=validated.release_id,
        primary_grounded_papers=selected.primary_grounded_papers,
        stage_run_sha256s=NullableStageRunHashes(
            **validated.stage_run_sha256s.model_dump()
        ),
        evidence_sha256s=NullableEvidenceHashes(
            **validated.evidence_sha256s.model_dump()
        ),
    )
    return ReleaseSelection(
        disposition="scaled_promoted",
        frozen_v1_primary_papers=frozen_v1_primary_papers,
        selected_release=selected,
        scaled_attempt=attempt,
        rendered_disclosure=render_release_disclosure(
            "scaled_promoted",
            frozen_v1_primary_papers=frozen_v1_primary_papers,
            selected_primary_papers=selected.primary_grounded_papers,
        ),
    )


def classify_scaled_failure(
    signals: Sequence[FailureKind] | set[FailureKind],
) -> tuple[str, str, str]:
    """Apply integrity -> reconciliation -> trust/offline -> incomplete precedence."""

    observed = set(signals)
    if "integrity" in observed:
        return (
            "v1_retained_scaled_corrupt",
            "rejected",
            "scaled_artifact_integrity_failed",
        )
    if "reconciliation" in observed:
        return (
            "v1_retained_scaled_unreconciled",
            "rejected",
            "scaled_ledger_reconciliation_failed",
        )
    if "unvalidated" in observed:
        return (
            "v1_retained_scaled_unvalidated",
            "rejected",
            "scaled_trust_or_offline_validation_failed",
        )
    return (
        "v1_retained_scaled_incomplete",
        "incomplete",
        "scaled_incomplete",
    )


def _artifact_rows(bundle_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in sorted(BUNDLED_PATHS):
        path = bundle_root / relative
        row_count: int | None = None
        if relative in PARQUET_PATHS:
            row_count = len(_read_parquet(path))
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": row_count,
            }
        )
    return rows


def _copy_scientific_files(
    sources: Mapping[str, Path],
    staging: Path,
) -> None:
    expected = (ANALYSIS_FILES | ROOT_SOURCE_FILES)
    if set(sources) != expected:
        raise ExportError("scientific_source_inventory_mismatch")
    for relative, source_path in sorted(sources.items()):
        if not source_path.is_file() or source_path.is_symlink():
            raise ExportError("scientific_source_invalid", source_path.as_posix())
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def _manifest_release_selection(selection: ReleaseSelection, record_hash: str) -> dict[str, Any]:
    return {
        "record_sha256": record_hash,
        **selection.model_dump(mode="json", exclude_none=False),
    }


def _base_manifest(
    *,
    config: QuestionConfig,
    variant: str,
    created_at: str,
    code_version: str,
    paper_funnel: Mapping[str, int],
    quality: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    release_selection: Mapping[str, Any],
    lineage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    assert config.demo is not None
    return {
        "manifest_version": "1",
        "schema_version": "1",
        "fixture": config.demo.fixture_mode,
        "question_id": config.question_id,
        "research_question": config.research_question,
        "spoken_question": config.demo.spoken_question,
        "corpus_qualifier": config.demo.corpus_qualifier,
        "narrative_variant": variant,
        "created_at": created_at,
        "config_sha256": config_sha256(config),
        "code_version": code_version,
        "primary_cohort_definition": "primary_grounded_unflagged",
        "paper_funnel": dict(paper_funnel),
        "quality": dict(quality),
        "exclusions": dict(exclusions),
        "release_selection": dict(release_selection),
        "lineage": [dict(row) for row in lineage],
    }


def _stage_bundle(
    *,
    staging: Path,
    scientific_sources: Mapping[str, Path],
    config: QuestionConfig,
    selection: ReleaseSelection,
    variant: Literal["A", "B"],
    template_path: Path,
    created_at: str,
    code_version: str,
    paper_funnel: Mapping[str, int],
    quality: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    lineage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _copy_scientific_files(scientific_sources, staging)
    selection_path = staging / "release_selection.json"
    atomic_write_json(selection_path, selection)
    selection_manifest = _manifest_release_selection(selection, sha256_file(selection_path))
    manifest = _base_manifest(
        config=config,
        variant=variant,
        created_at=created_at,
        code_version=code_version,
        paper_funnel=paper_funnel,
        quality=quality,
        exclusions=exclusions,
        release_selection=selection_manifest,
        lineage=lineage,
    )
    try:
        template = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExportError("script_template_unreadable", template_path.as_posix()) from exc
    headline = _load_json_object(staging / "analysis" / "headline.json")
    rendered = render_demo_script(
        template,
        variant=variant,
        manifest=manifest,
        headline=headline,
    )
    atomic_write_text(staging / "demo_script.md", rendered)
    manifest["artifacts"] = _artifact_rows(staging)
    atomic_write_json(staging / "manifest.json", manifest)
    return manifest


def _atomic_promote(staging: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ExportError("demo_destination_symlink_forbidden")
    if destination.exists() and not force:
        raise ExportError("demo_destination_exists_requires_force")
    backup = destination.with_name(f".{destination.name}.previous-{os.getpid()}")
    if backup.exists():
        raise ExportError("stale_demo_backup_exists", backup.as_posix())
    moved_previous = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(staging, destination)
    except BaseException:
        if moved_previous and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _actual_bundle_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    expected = {
        "manifest_version",
        "schema_version",
        "fixture",
        "question_id",
        "research_question",
        "spoken_question",
        "corpus_qualifier",
        "narrative_variant",
        "created_at",
        "config_sha256",
        "code_version",
        "primary_cohort_definition",
        "paper_funnel",
        "quality",
        "exclusions",
        "release_selection",
        "lineage",
        "artifacts",
    }
    if set(manifest) != expected:
        raise ExportError("manifest_field_inventory_mismatch")
    if manifest.get("manifest_version") != "1" or manifest.get("schema_version") != "1":
        raise ExportError("manifest_version_invalid")
    if manifest.get("narrative_variant") not in {"A", "B"}:
        raise ExportError("manifest_variant_invalid")
    if manifest.get("primary_cohort_definition") != "primary_grounded_unflagged":
        raise ExportError("manifest_cohort_definition_invalid")


def verify_demo_bundle(
    bundle_root: str | Path,
    config: QuestionConfig,
    *,
    explicit_fixture: bool = False,
    allow_dirty_demo: bool = False,
) -> dict[str, Any]:
    """Recompute a frozen bundle entirely offline, without consulting working artifacts."""

    root = Path(bundle_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ExportError("demo_bundle_directory_invalid", root.as_posix())
    actual_files = _actual_bundle_files(root)
    if actual_files != ALL_DEMO_PATHS:
        raise ExportError(
            "bundle_inventory_mismatch",
            f"missing={sorted(ALL_DEMO_PATHS - actual_files)}:"
            f"extra={sorted(actual_files - ALL_DEMO_PATHS)}",
        )
    manifest = _load_json_object(root / "manifest.json")
    _validate_manifest_shape(manifest)
    try:
        config.authorize_stage(
            "s7",
            explicit_fixture=explicit_fixture,
            live_provider=False,
        )
    except ValueError as exc:
        raise ExportError("fixture_runtime_guard_failed", str(exc)) from exc
    assert config.demo is not None
    if manifest["question_id"] != config.question_id:
        raise ExportError("manifest_question_mismatch")
    if manifest["fixture"] is not config.demo.fixture_mode:
        raise ExportError("manifest_fixture_flag_mismatch")
    if manifest["config_sha256"] != config_sha256(config):
        raise ExportError("manifest_config_hash_mismatch")
    if manifest["research_question"] != config.research_question:
        raise ExportError("manifest_research_question_mismatch")
    if manifest["spoken_question"] != config.demo.spoken_question:
        raise ExportError("manifest_spoken_question_mismatch")
    if manifest["corpus_qualifier"] != config.demo.corpus_qualifier:
        raise ExportError("manifest_corpus_qualifier_mismatch")
    if str(manifest["code_version"]).startswith("dirty:") and not allow_dirty_demo:
        raise ExportError("dirty_lineage_requires_override")
    try:
        created_at = datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExportError("manifest_created_at_invalid") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ExportError("manifest_created_at_timezone_missing")

    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list):
        raise ExportError("manifest_artifact_rows_mismatch")
    # Hash/size binding runs first so any bundled-file tampering — including a corrupted
    # parquet — is reported as the manifest binding failure rather than a parse error.
    integrity_keys = ("path", "sha256", "bytes")
    expected_integrity = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(BUNDLED_PATHS)
    ]
    observed_integrity = [
        {key: row.get(key) for key in integrity_keys}
        for row in manifest_artifacts
        if isinstance(row, Mapping)
    ]
    if observed_integrity != expected_integrity:
        raise ExportError("manifest_artifact_rows_mismatch")
    expected_artifacts = _artifact_rows(root)
    if manifest_artifacts != expected_artifacts:
        raise ExportError("manifest_artifact_rows_mismatch")
    selection_payload = _load_json_object(root / "release_selection.json")
    try:
        selection = ReleaseSelection.model_validate(selection_payload)
    except ValidationError as exc:
        raise ExportError("release_selection_contract_invalid", str(exc)) from exc
    manifest_selection = manifest.get("release_selection")
    if not isinstance(manifest_selection, Mapping):
        raise ExportError("manifest_release_selection_invalid")
    expected_manifest_selection = _manifest_release_selection(
        selection,
        sha256_file(root / "release_selection.json"),
    )
    if dict(manifest_selection) != expected_manifest_selection:
        raise ExportError("manifest_release_selection_mismatch")

    bundle_source = ReleaseSource(
        repository_root=root,
        question_id=config.question_id,
        corpus_role=selection.selected_release.corpus_role,
        processed_dir=root,
        analysis_dir=root / "analysis",
        extracted_dir=root,
    )
    papers, findings, _papers_frame, _findings_frame = _validate_ledgers(
        bundle_source,
        config,
    )
    json_artifacts = {
        path: _load_json_object(root / path)
        for path in sorted((ANALYSIS_FILES | ROOT_SOURCE_FILES) - PARQUET_PATHS)
    }
    verification = json_artifacts["verification.json"]
    _validate_verification(findings, verification)
    audit_correct, audit_total = _validate_audit(json_artifacts["audit.json"])
    try:
        primary_rows = primary_cohort(
            papers,
            findings,
            verification,
            primary_family=str(config.outcomes.primary_family),
        )
        metrics = compute_quality_metrics(
            papers,
            findings,
            verification,
            primary_family=str(config.outcomes.primary_family),
        )
    except ValueError as exc:
        raise ExportError(
            "bundle_ledger_reconciliation_failed",
            str(exc),
            kind="reconciliation",
        ) from exc
    if not primary_rows:
        raise ExportError("primary_cohort_empty", kind="unvalidated")
    cohort_hash = _cohort_sha256(config, primary_rows)
    moderators = _read_parquet(root / "analysis" / "moderators.parquet")
    contradictions = _read_parquet(root / "analysis" / "contradictions.parquet")
    evidence_gaps = _read_parquet(root / "analysis" / "evidence_gaps.parquet")
    _validate_moderator_table(moderators, config)
    contradiction_rows = _validate_contradictions(
        contradictions,
        papers=papers,
        findings=findings,
    )
    evidence_gap_rows = _validate_evidence_gaps(evidence_gaps, config)
    g3 = json_artifacts["g3_gate.json"]
    _validate_g3_gate(
        g3,
        config_hash=config_sha256(config),
        cohort_sha256=cohort_hash,
        audit_hash=sha256_file(root / "audit.json"),
        verification_hash=sha256_file(root / "verification.json"),
        audit_correct=audit_correct,
        audit_total=audit_total,
        quality_metrics=metrics,
    )
    m4_gate = json_artifacts["analysis/m4_gate.json"]
    variant, selection_reason = _validate_m4_gate(
        m4_gate,
        config=config,
        cohort_sha256=cohort_hash,
        g3_action=str(g3["action"]),
    )
    if manifest["narrative_variant"] != variant:
        raise ExportError("manifest_scientific_variant_mismatch")
    _validate_checkpoint_artifact(
        json_artifacts["analysis/m4_checkpoint.json"],
        str(m4_gate["status"]),
    )
    _validate_typed_resampling_artifacts(
        variant=variant,
        selection_reason=selection_reason,
        tree=json_artifacts["analysis/tree.json"],
        bootstrap=json_artifacts["analysis/bootstrap.json"],
        permutation=json_artifacts["analysis/permutation.json"],
    )
    _validate_headline(
        json_artifacts["analysis/headline.json"],
        variant=variant,
        selection_reason=selection_reason,
        m4_gate=m4_gate,
        config=config,
        primary_rows=primary_rows,
        contradiction_rows=contradiction_rows,
        evidence_gap_rows=evidence_gap_rows,
    )
    _validate_trace(json_artifacts["trace.json"], variant=variant, reason=selection_reason)
    _validate_baseline(
        json_artifacts["baseline.json"],
        config=config,
        cohort_sha256=cohort_hash,
        primary_rows=primary_rows,
    )

    searched_documents = {
        identifier
        for paper in papers
        for identifier in [paper["doc_id"], *paper["alternate_doc_ids"]]
    }
    expected_funnel = {
        "searched_documents": len(searched_documents),
        "identity_deduped_papers": len(papers),
        "deterministic_included_papers": sum(
            paper["screen_status"] == "included" for paper in papers
        ),
        "extraction_eligible_papers": sum(
            paper["screen_status"] == "included" and paper["eligible"] is True
            for paper in papers
        ),
        "primary_grounded_papers": len({row["paper_id"] for row in primary_rows}),
        "primary_grounded_findings": len(primary_rows),
    }
    expected_quality = {
        "audit_correct": audit_correct,
        "audit_total": audit_total,
        "grounded_fraction": metrics["grounded"]["fraction"],
        "grounded_numerator": metrics["grounded"]["numerator"],
        "grounded_denominator": metrics["grounded"]["denominator"],
        "quarantine_fraction": metrics["quarantine"]["fraction"],
        "quarantine_numerator": metrics["quarantine"]["numerator"],
        "quarantine_denominator": metrics["quarantine"]["denominator"],
        "cross_model_agreement": metrics["cross_model_agreement"]["fraction"],
        "cross_model_agree": metrics["cross_model_agreement"]["numerator"],
        "cross_model_requested": metrics["cross_model_agreement"]["denominator"],
    }
    expected_exclusions = {
        "mixed_or_unclear_fraction": metrics["mixed_or_unclear_exclusion"]["fraction"],
        "mixed_or_unclear_numerator": metrics["mixed_or_unclear_exclusion"]["numerator"],
        "mixed_or_unclear_denominator": metrics["mixed_or_unclear_exclusion"]["denominator"],
        "section_flagged_fraction": metrics["section_flagged_exclusion"]["fraction"],
        "section_flagged_numerator": metrics["section_flagged_exclusion"]["numerator"],
        "section_flagged_denominator": metrics["section_flagged_exclusion"]["denominator"],
        "verification_excluded_fraction": metrics["verification_exclusion"]["fraction"],
        "verification_excluded_numerator": metrics["verification_exclusion"]["numerator"],
        "verification_excluded_denominator": metrics["verification_exclusion"]["denominator"],
    }
    if manifest["paper_funnel"] != expected_funnel:
        raise ExportError("manifest_paper_funnel_mismatch")
    if manifest["quality"] != expected_quality:
        raise ExportError("manifest_quality_mismatch")
    if manifest["exclusions"] != expected_exclusions:
        raise ExportError("manifest_exclusions_mismatch")

    evidence_hashes = EvidenceHashes(
        g3_gate=sha256_file(root / "g3_gate.json"),
        audit=sha256_file(root / "audit.json"),
        verification=sha256_file(root / "verification.json"),
        headline=sha256_file(root / "analysis" / "headline.json"),
        baseline=sha256_file(root / "baseline.json"),
    )
    if selection.selected_release.evidence_sha256s != evidence_hashes:
        raise ExportError("selected_release_evidence_hashes_mismatch")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, list) or [row.get("stage") for row in lineage] != [
        "s3",
        "s4",
        "s5",
    ]:
        raise ExportError("manifest_lineage_invalid")
    lineage_hashes = {str(row["stage"]): row.get("run_sha256") for row in lineage}
    if selection.selected_release.stage_run_sha256s.model_dump() != lineage_hashes:
        raise ExportError("selected_release_stage_hashes_mismatch")
    expected_release_id = _derive_release_id(
        question_id=config.question_id,
        corpus_role=selection.selected_release.corpus_role,
        cohort_sha256=cohort_hash,
        config_sha256_value=config_sha256(config),
        code_version=str(manifest["code_version"]),
        stage_hashes=selection.selected_release.stage_run_sha256s,
        evidence_hashes=evidence_hashes,
    )
    if selection.selected_release.release_id != expected_release_id:
        raise ExportError("selected_release_id_mismatch")
    if selection.selected_release.primary_grounded_papers != expected_funnel[
        "primary_grounded_papers"
    ]:
        raise ExportError("selected_release_primary_count_mismatch")

    script = (root / "demo_script.md").read_text(encoding="utf-8")
    if UNRESOLVED_TOKEN_RE.search(script) or spoken_word_count(script) > 225:
        raise ExportError("rendered_script_invalid", kind="unvalidated")
    headline = json_artifacts["analysis/headline.json"]
    required_script_text = [
        str(headline["rendered_sentence"]),
        selection.rendered_disclosure,
        config.demo.corpus_qualifier,
    ]
    if variant == "B":
        required_script_text.append(str(headline["residuals"]["rendered_sentence"]))
    if any(text not in script for text in required_script_text):
        raise ExportError("rendered_script_content_mismatch", kind="unvalidated")
    return manifest


def _scientific_sources_from_validated(validated: ValidatedRelease) -> dict[str, Path]:
    return {
        relative: validated.source.source_path(relative)
        for relative in ANALYSIS_FILES | ROOT_SOURCE_FILES
    }


def _lineage_from_validated(validated: ValidatedRelease) -> list[dict[str, str]]:
    hashes = validated.stage_run_sha256s.model_dump()
    return [
        {
            "stage": stage,
            "run_id": validated.stage_runs[stage].run_id,
            "run_sha256": hashes[stage],
        }
        for stage in ("s3", "s4", "s5")
    ]


def _default_templates(root: Path) -> dict[str, Path]:
    return {
        "A": root / "docs" / "demo" / "variant_a.md",
        "B": root / "docs" / "demo" / "variant_b.md",
    }


def _inspect_candidate_progress(
    source: ReleaseSource,
    config: QuestionConfig,
    *,
    allow_dirty_demo: bool,
) -> CandidateProgress:
    progress = CandidateProgress()
    code_version: str | None = None
    for stage in ("s3", "s4", "s5"):
        path = source.stage_run_paths[stage]
        if not path.is_file():
            progress.signals.add("incomplete")
            break
        try:
            record = _load_run_record(path, expected_stage=stage)
            if record.config_sha256 != config_sha256(config):
                raise ExportError("run_config_hash_mismatch")
            if code_version is not None and record.code_version != code_version:
                raise ExportError("mixed_code_versions")
            if record.code_version.startswith("dirty:") and not allow_dirty_demo:
                raise ExportError("dirty_lineage_requires_override")
            for reference in (*record.inputs, *record.outputs):
                _validate_artifact_reference(reference, source.repository_root)
            code_version = record.code_version
        except ExportError as exc:
            progress.signals.add(exc.kind)
            break
        progress.stage_hashes[stage] = sha256_file(path)
        progress.last_completed_stage = stage

    papers: list[dict[str, Any]] | None = None
    findings: list[dict[str, Any]] | None = None
    primary_rows: list[dict[str, Any]] | None = None
    cohort_hash: str | None = None
    if progress.stage_hashes["s4"] is not None:
        try:
            papers, findings, _, _ = _validate_ledgers(source, config)
            verification = _load_json_object(source.source_path("verification.json"))
            _validate_verification(findings, verification)
            primary_rows = primary_cohort(
                papers,
                findings,
                verification,
                primary_family=str(config.outcomes.primary_family),
            )
            cohort_hash = _cohort_sha256(config, primary_rows)
            progress.primary_grounded_papers = len(
                {row["paper_id"] for row in primary_rows}
            )
        except (ExportError, ValueError) as exc:
            progress.signals.add(
                exc.kind if isinstance(exc, ExportError) else "reconciliation"
            )

    if progress.stage_hashes["s4"] is not None:
        for key, relative in (
            ("g3_gate", "g3_gate.json"),
            ("audit", "audit.json"),
            ("verification", "verification.json"),
        ):
            try:
                _load_json_object(source.source_path(relative))
                progress.evidence_hashes[key] = sha256_file(source.source_path(relative))
            except ExportError as exc:
                progress.signals.add(exc.kind)
    if progress.stage_hashes["s5"] is not None:
        for key, relative in (
            ("headline", "analysis/headline.json"),
            ("baseline", "baseline.json"),
        ):
            try:
                _load_json_object(source.source_path(relative))
                progress.evidence_hashes[key] = sha256_file(source.source_path(relative))
            except ExportError as exc:
                progress.signals.add(exc.kind)

    if (
        cohort_hash is not None
        and code_version is not None
        and all(progress.stage_hashes.values())
        and all(progress.evidence_hashes.values())
    ):
        stage_hashes = StageRunHashes.model_validate(progress.stage_hashes)
        evidence_hashes = EvidenceHashes.model_validate(progress.evidence_hashes)
        progress.candidate_release_id = _derive_release_id(
            question_id=source.question_id,
            corpus_role="scaled",
            cohort_sha256=cohort_hash,
            config_sha256_value=config_sha256(config),
            code_version=code_version,
            stage_hashes=stage_hashes,
            evidence_hashes=evidence_hashes,
        )
    return progress


def _retained_selection(
    frozen_selection: ReleaseSelection,
    progress: CandidateProgress,
) -> ReleaseSelection:
    if frozen_selection.selected_release.corpus_role != "v1":
        raise ExportError("fallback_selected_release_must_be_v1")
    disposition, attempt_status, failure_code = classify_scaled_failure(progress.signals)
    attempt = ScaledAttempt(
        status=attempt_status,
        failure_code=failure_code,
        last_completed_stage=progress.last_completed_stage,
        candidate_release_id=progress.candidate_release_id,
        primary_grounded_papers=progress.primary_grounded_papers,
        stage_run_sha256s=NullableStageRunHashes.model_validate(progress.stage_hashes),
        evidence_sha256s=NullableEvidenceHashes.model_validate(progress.evidence_hashes),
    )
    frozen_count = frozen_selection.frozen_v1_primary_papers
    return ReleaseSelection(
        disposition=disposition,
        frozen_v1_primary_papers=frozen_count,
        selected_release=frozen_selection.selected_release,
        scaled_attempt=attempt,
        rendered_disclosure=render_release_disclosure(
            disposition,
            frozen_v1_primary_papers=frozen_count,
            selected_primary_papers=frozen_count,
        ),
    )


def _ensure_release_archive(demo_dir: Path, release_id: str) -> Path:
    archive = demo_dir.parent / "releases" / release_id
    if archive.exists():
        if not archive.is_dir() or _actual_bundle_files(archive) != ALL_DEMO_PATHS:
            raise ExportError("frozen_release_archive_invalid", archive.as_posix())
        for relative in ALL_DEMO_PATHS:
            if sha256_file(archive / relative) != sha256_file(demo_dir / relative):
                raise ExportError("frozen_release_archive_hash_mismatch", relative)
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{release_id}.staging-", dir=archive.parent)
    )
    try:
        for relative in sorted(ALL_DEMO_PATHS):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo_dir / relative, target)
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return archive


def export_demo(
    source: ReleaseSource,
    config: QuestionConfig,
    *,
    destination: str | Path | None = None,
    template_paths: Mapping[str, Path] | None = None,
    explicit_fixture: bool = False,
    allow_dirty_demo: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Stage, verify, and promote v1/scaled, retaining v1 on scale failure."""

    paths = ProjectPaths(source.repository_root)
    destination_path = (
        Path(destination).resolve()
        if destination is not None
        else paths.demo_dir(source.question_id)
    )
    templates = dict(template_paths or _default_templates(source.repository_root))
    if set(templates) != {"A", "B"}:
        raise ExportError("template_path_inventory_invalid")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    frozen_manifest: dict[str, Any] | None = None
    frozen_selection: ReleaseSelection | None = None
    if source.corpus_role == "scaled":
        if not destination_path.is_dir():
            raise ExportError("scaled_candidate_requires_frozen_v1_bundle")
        frozen_manifest = verify_demo_bundle(
            destination_path,
            config,
            explicit_fixture=explicit_fixture,
            allow_dirty_demo=allow_dirty_demo,
        )
        try:
            frozen_selection = ReleaseSelection.model_validate(
                _load_json_object(destination_path / "release_selection.json")
            )
        except ValidationError as exc:
            raise ExportError("frozen_v1_selection_invalid", str(exc)) from exc
        if frozen_selection.selected_release.corpus_role != "v1":
            raise ExportError("scaled_candidate_fallback_is_not_v1")
        _ensure_release_archive(
            destination_path,
            frozen_selection.selected_release.release_id,
        )

    try:
        validated = validate_release_source(
            source,
            config,
            explicit_fixture=explicit_fixture,
            allow_dirty_demo=allow_dirty_demo,
        )
    except ExportError as candidate_error:
        if source.corpus_role != "scaled" or frozen_manifest is None or frozen_selection is None:
            raise
        progress = _inspect_candidate_progress(
            source,
            config,
            allow_dirty_demo=allow_dirty_demo,
        )
        progress.signals.add(candidate_error.kind)
        selection = _retained_selection(frozen_selection, progress)
        variant = str(frozen_manifest["narrative_variant"])
        scientific_sources = {
            relative: destination_path / relative
            for relative in ANALYSIS_FILES | ROOT_SOURCE_FILES
        }
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_path.name}.staging-",
                dir=destination_path.parent,
            )
        )
        try:
            manifest = _stage_bundle(
                staging=staging,
                scientific_sources=scientific_sources,
                config=config,
                selection=selection,
                variant=variant,  # type: ignore[arg-type]
                template_path=templates[variant],
                created_at=str(frozen_manifest["created_at"]),
                code_version=str(frozen_manifest["code_version"]),
                paper_funnel=frozen_manifest["paper_funnel"],
                quality=frozen_manifest["quality"],
                exclusions=frozen_manifest["exclusions"],
                lineage=frozen_manifest["lineage"],
            )
            verify_demo_bundle(
                staging,
                config,
                explicit_fixture=explicit_fixture,
                allow_dirty_demo=allow_dirty_demo,
            )
            _atomic_promote(staging, destination_path, force=True)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    selection = (
        _v1_selection(validated)
        if source.corpus_role == "v1"
        else _scaled_selection(
            validated,
            frozen_v1_primary_papers=int(frozen_manifest["paper_funnel"]["primary_grounded_papers"]),
        )
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.staging-", dir=destination_path.parent)
    )
    try:
        manifest = _stage_bundle(
            staging=staging,
            scientific_sources=_scientific_sources_from_validated(validated),
            config=config,
            selection=selection,
            variant=validated.narrative_variant,
            template_path=templates[validated.narrative_variant],
            created_at=validated.created_at,
            code_version=validated.stage_runs["s5"].code_version,
            paper_funnel=validated.paper_funnel,
            quality=validated.quality,
            exclusions=validated.exclusions,
            lineage=_lineage_from_validated(validated),
        )
        verify_demo_bundle(
            staging,
            config,
            explicit_fixture=explicit_fixture,
            allow_dirty_demo=allow_dirty_demo,
        )
        _atomic_promote(staging, destination_path, force=force)
        if source.corpus_role == "v1":
            _ensure_release_archive(destination_path, selection.selected_release.release_id)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "ALL_DEMO_PATHS",
    "BUNDLED_PATHS",
    "ExportError",
    "ReleaseSource",
    "classify_scaled_failure",
    "export_demo",
    "render_demo_script",
    "spoken_word_count",
    "validate_release_source",
    "verify_demo_bundle",
]
