from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from literature_multiverse.analysis import analyze_s5, write_analysis_bundle
from literature_multiverse.baseline import create_baseline
from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import QuestionConfig, config_sha256, load_question_config
from literature_multiverse.export import (
    ALL_DEMO_PATHS,
    BUNDLED_PATHS,
    ExportError,
    ReleaseSource,
    classify_scaled_failure,
    export_demo,
    render_demo_script,
    spoken_word_count,
    validate_release_source,
    verify_demo_bundle,
)
from literature_multiverse.gates import build_g3_artifact
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    sha256_file,
    write_run_record,
)
from literature_multiverse.models import (
    AuditDecision,
    AuditFieldChecks,
    AuditRecord,
    FindingRow,
    PaperRecord,
    RunRecord,
    UpstreamRef,
    VerificationDecision,
    VerificationRecord,
    make_finding_id,
)
from literature_multiverse.normalize import normalized_frames

FIXED_TIME = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
CODE_VERSION = "fixture-code-version"
CONTRADICTORY_FAILURES = ("integrity", "reconciliation", "unvalidated", "incomplete")


@dataclass(frozen=True, slots=True)
class ExportWorkspace:
    root: Path
    source: ReleaseSource
    config: QuestionConfig
    destination: Path
    templates: dict[str, Path]


def _upstream(path: Path, record: RunRecord, root: Path) -> UpstreamRef:
    return UpstreamRef(
        stage=record.stage,
        run_id=record.run_id,
        run_record_path=path.relative_to(root).as_posix(),
        run_record_sha256=sha256_file(path),
    )


def _run_record(
    *,
    stage: str,
    question_id: str,
    config_path: Path,
    config_hash: str,
    root: Path,
    started_at: datetime,
    code_version: str,
    upstream: list[UpstreamRef],
    inputs: list[Any],
    outputs: list[Any],
) -> RunRecord:
    return RunRecord(
        run_id=f"{stage}-fixture-run",
        question_id=question_id,
        stage=stage,
        stage_version="1",
        status="complete",
        completion_mode="normal",
        checkpoint_sha256=None,
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=1),
        code_version=code_version,
        command_argv=[f"scripts/{stage}_fixture.py", "--fixture"],
        config_path=config_path.relative_to(root).as_posix(),
        config_sha256=config_hash,
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=upstream,
        inputs=inputs,
        outputs=outputs,
        external_result_ids={},
        counts={},
        warnings=[],
    )


def _paper(
    *,
    index: int,
    doc_id: str,
    accepted: int,
    config_hash: str,
    extraction_hash: str,
) -> PaperRecord:
    return PaperRecord(
        paper_id=f"doc:{doc_id}",
        doc_id=doc_id,
        alternate_doc_ids=[],
        doi=None,
        pmid=None,
        title=f"Synthetic controlled study {index:02d}",
        first_author=f"Author {index:02d}",
        pub_year=2020 + index % 6,
        source="fixture",
        article_type="research-article",
        query_families=["direct"],
        search_result_ids=[f"result-{index:02d}"],
        content_tier="full_text",
        publication_status="peer_reviewed",
        screen_status="included",
        screen_reason=None,
        dedupe_cluster_id=f"cluster-{index:02d}",
        dedupe_preferred=True,
        map_status="success",
        eligible=True,
        exclusion_reason=None,
        map_result_id=f"map-{index:02d}",
        raw_artifact_path=f"data/raw/map/fixture-b-story/map-{index:02d}.json",
        raw_finding_count=accepted,
        accepted_finding_count=accepted,
        quarantined_finding_count=0,
        failure_code=None,
        dataset_or_cohort_id=f"cohort-{index:02d}",
        prompt_version="fixture-prompt-v1",
        schema_version="1",
        config_sha256=config_hash,
        cfghash=extraction_hash,
        created_at=FIXED_TIME,
    )


def _finding(
    *,
    paper: PaperRecord,
    index: int,
    array_position: int,
    endpoint: str,
    extraction_hash: str,
    grounding_status: str = "exact",
) -> FindingRow:
    identity = {
        "paper_id": paper.paper_id,
        "map_result_id": str(paper.map_result_id),
        "array_position": array_position,
        "outcome_name": endpoint,
        "timepoint_raw": "post intervention",
        "dose_raw": "low" if index % 2 == 0 else "high",
        "effect_direction": "no_effect",
    }
    return FindingRow(
        finding_id=make_finding_id(**identity),
        **identity,
        doc_id=paper.doc_id,
        prompt_version="fixture-prompt-v1",
        schema_version="1",
        cfghash=extraction_hash,
        grounding_status=grounding_status,
        evidence_section="Results",
        section_flagged=False,
        normalization_warnings=[],
        study_type="controlled trial",
        species="human",
        model=None,
        population_state="healthy" if index % 2 == 0 else "clinical",
        intervention="synthetic intervention",
        intervention_class="synthetic",
        comparator="control condition",
        duration_raw="4 weeks",
        timing_context="acute" if index % 2 == 0 else "chronic",
        outcome_family="performance",
        effect_size_raw=None,
        p_value=0.5,
        significant=False,
        sample_size=40 + index,
        evidence_quote="No detectable difference from control was reported.",
        evidence_lines=[f"L{index + 1}"],
        confidence=0.95,
        moderators={
            "dose_regime": "low" if index % 2 == 0 else "high",
            "training_status": "trained" if (index // 2) % 2 == 0 else "untrained",
            "population_state": "healthy" if index % 2 == 0 else "clinical",
            "timing_context": "acute" if index % 2 == 0 else "chronic",
        },
    )


def _build_workspace(
    temporary_root: Path,
    repository_root: Path,
    *,
    code_version: str = CODE_VERSION,
) -> ExportWorkspace:
    root = temporary_root / "repository"
    question_id = "fixture-b-story"
    config_path = root / "configs" / "questions" / f"{question_id}.yaml"
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(repository_root / "configs" / "questions" / f"{question_id}.yaml", config_path)
    config = load_question_config(config_path, require_locked=True)
    config_hash = config_sha256(config)
    extraction_hash = "b" * 64

    papers: list[PaperRecord] = []
    findings: list[FindingRow] = []
    for index in range(20):
        doc_id = "fixture-b-story-anchor" if index == 0 else f"fixture-b-story-{index:02d}"
        accepted = 2 if index in {0, 19} else 1
        paper = _paper(
            index=index,
            doc_id=doc_id,
            accepted=accepted,
            config_hash=config_hash,
            extraction_hash=extraction_hash,
        )
        papers.append(paper)
        findings.append(
            _finding(
                paper=paper,
                index=index,
                array_position=0,
                endpoint="peak_power" if index % 2 == 0 else "fatigue_time",
                extraction_hash=extraction_hash,
            )
        )
        if index == 0:
            findings.append(
                _finding(
                    paper=paper,
                    index=index,
                    array_position=1,
                    endpoint="fatigue_time",
                    extraction_hash=extraction_hash,
                )
            )
        elif index == 19:
            # This accepted row is deliberately outside the 20-row audit and is not grounded.
            # It makes the fixture prove that audit accuracy cannot alter whole-ledger rates.
            findings.append(
                _finding(
                    paper=paper,
                    index=index,
                    array_position=1,
                    endpoint="peak_power",
                    extraction_hash=extraction_hash,
                    grounding_status="mismatch",
                )
            )

    processed = root / "data" / "processed" / question_id
    analysis = root / "artifacts" / question_id / "analysis"
    extracted = root / "data" / "extracted" / question_id
    processed.mkdir(parents=True)
    analysis.mkdir(parents=True)
    extracted.mkdir(parents=True)
    paper_frame, finding_frame = normalized_frames(
        papers,
        findings,
        moderator_names=[moderator.name for moderator in config.moderators],
        moderator_types={moderator.name: moderator.type for moderator in config.moderators},
    )
    paper_frame.to_parquet(processed / "papers.parquet", index=False)
    finding_frame.to_parquet(processed / "findings.parquet", index=False)

    exact_findings = [row for row in findings if row.grounding_status.value == "exact"]
    verification = VerificationRecord(
        provider="fixture",
        model="fixture-verifier",
        prompt_version="fixture-verification-v1",
        prompt_sha256="c" * 64,
        requested_finding_ids=[row.finding_id for row in exact_findings],
        decisions=[
            VerificationDecision(
                finding_id=row.finding_id,
                model_status="agree",
                adjudication="none",
            )
            for row in exact_findings
        ],
    )
    audit_ids = [row.finding_id for row in exact_findings[:20]]
    audit = AuditRecord(
        seed=config.analysis.seed,
        requested_sample_size=20,
        sampled_finding_ids=audit_ids,
        decisions=[
            AuditDecision(
                finding_id=finding_id,
                checks=AuditFieldChecks(
                    eligibility=True,
                    atomicity=True,
                    intervention=True,
                    comparator=True,
                    outcome=True,
                    timepoint=True,
                    direction=True,
                    quote_support=True,
                ),
            )
            for finding_id in audit_ids
        ],
        anchor_results={"doc:fixture-b-story-anchor": True},
        correct_count=20,
        total_count=20,
        wilson_interval=(0.838, 1.0),
        error_taxonomy={},
    )
    atomic_write_json(processed / "verification.json", verification)
    atomic_write_json(processed / "audit.json", audit)

    paper_rows = [paper.model_dump(mode="json", exclude_none=False) for paper in papers]
    finding_rows = [finding.model_dump(mode="json", exclude_none=False) for finding in findings]
    g3 = build_g3_artifact(
        config=config,
        papers=paper_rows,
        findings=finding_rows,
        verification=verification,
        audit=audit,
        g1b_passed=True,
    )
    assert g3["trust_passed"] is True
    assert g3["action"] == "select_variant_b_story"
    atomic_write_json(processed / "g3_gate.json", g3)

    bundle = analyze_s5(
        config=config,
        papers=paper_rows,
        findings=finding_rows,
        verification=verification,
        g3_gate=g3,
    )
    written = write_analysis_bundle(bundle, analysis)
    baseline = create_baseline(
        cohort_hash=cohort_sha256(bundle.primary_rows),
        research_question=config.research_question,
        primary_rows=bundle.primary_rows,
        prompt_path=repository_root / "prompts" / "baseline_consensus.md",
        attempted_at=FIXED_TIME + timedelta(hours=1),
        fixture_mode=True,
    )
    atomic_write_json(analysis / "baseline.json", baseline)

    s3_path = extracted / "run.json"
    s3 = _run_record(
        stage="s3",
        question_id=question_id,
        config_path=config_path,
        config_hash=config_hash,
        root=root,
        started_at=FIXED_TIME,
        code_version=code_version,
        upstream=[],
        inputs=[],
        outputs=[],
    )
    write_run_record(s3_path, s3)

    s4_path = processed / "run.json"
    s4 = _run_record(
        stage="s4",
        question_id=question_id,
        config_path=config_path,
        config_hash=config_hash,
        root=root,
        started_at=FIXED_TIME + timedelta(minutes=2),
        code_version=code_version,
        upstream=[_upstream(s3_path, s3, root)],
        inputs=[],
        outputs=[
            artifact_ref(processed / "papers.parquet", root=root, rows=len(papers)),
            artifact_ref(processed / "findings.parquet", root=root, rows=len(findings)),
        ],
    )
    write_run_record(s4_path, s4)

    s5_path = analysis / "run.json"
    s5_outputs = [
        artifact_ref(
            path,
            root=root,
            rows=len(bundle.table_artifacts[name]) if name.endswith(".parquet") else None,
        )
        for name, path in sorted(written.items())
    ]
    s5 = _run_record(
        stage="s5",
        question_id=question_id,
        config_path=config_path,
        config_hash=config_hash,
        root=root,
        started_at=FIXED_TIME + timedelta(minutes=4),
        code_version=code_version,
        upstream=[_upstream(s4_path, s4, root)],
        inputs=[
            artifact_ref(processed / name, root=root)
            for name in ("papers.parquet", "findings.parquet", "g3_gate.json", "verification.json")
        ],
        outputs=s5_outputs,
    )
    write_run_record(s5_path, s5)

    source = ReleaseSource.from_repository(root, question_id)
    return ExportWorkspace(
        root=root,
        source=source,
        config=config,
        destination=root / "artifacts" / question_id / "demo",
        templates={
            "A": repository_root / "docs" / "demo" / "variant_a.md",
            "B": repository_root / "docs" / "demo" / "variant_b.md",
        },
    )


@pytest.fixture
def export_workspace(tmp_path: Path, repo_root: Path) -> ExportWorkspace:
    return _build_workspace(tmp_path, repo_root)


def _export(workspace: ExportWorkspace, *, force: bool = False) -> dict[str, Any]:
    return export_demo(
        workspace.source,
        workspace.config,
        destination=workspace.destination,
        template_paths=workspace.templates,
        explicit_fixture=True,
        force=force,
    )


def _bundle_hashes(path: Path) -> dict[str, str]:
    return {relative: sha256_file(path / relative) for relative in sorted(ALL_DEMO_PATHS)}


def test_valid_v1_export_has_exact_inventory_and_verifies_offline(
    export_workspace: ExportWorkspace,
) -> None:
    validated = validate_release_source(
        export_workspace.source,
        export_workspace.config,
        explicit_fixture=True,
    )
    manifest = _export(export_workspace)

    assert validated.narrative_variant == "B"
    assert validated.cohort_sha256 == cohort_sha256(validated.primary_rows)
    assert manifest == verify_demo_bundle(
        export_workspace.destination,
        export_workspace.config,
        explicit_fixture=True,
    )
    actual = {
        path.relative_to(export_workspace.destination).as_posix()
        for path in export_workspace.destination.rglob("*")
        if path.is_file()
    }
    assert actual == ALL_DEMO_PATHS
    assert len(manifest["artifacts"]) == 18
    assert {row["path"] for row in manifest["artifacts"]} == BUNDLED_PATHS
    assert all(row["path"] != "manifest.json" for row in manifest["artifacts"])
    assert manifest["created_at"] == (FIXED_TIME + timedelta(minutes=5)).isoformat()
    assert manifest["release_selection"]["disposition"] == "v1_frozen"
    selection_text = (export_workspace.destination / "release_selection.json").read_text()
    assert "manifest" not in selection_text
    assert '"s7"' not in selection_text


def test_whole_ledger_rates_do_not_use_the_unrepresentative_audit(
    export_workspace: ExportWorkspace,
) -> None:
    manifest = _export(export_workspace)

    assert manifest["quality"]["audit_total"] == 20
    assert manifest["quality"]["audit_correct"] == 20
    assert manifest["quality"]["grounded_denominator"] == 22
    assert manifest["quality"]["grounded_numerator"] == 21
    assert manifest["quality"]["cross_model_requested"] == 21
    assert manifest["paper_funnel"]["primary_grounded_findings"] == 21


def test_replay_reproduces_every_bundle_hash(export_workspace: ExportWorkspace) -> None:
    first = _export(export_workspace)
    first_hashes = _bundle_hashes(export_workspace.destination)
    replay = export_workspace.destination.with_name("demo-replay")

    second = export_demo(
        export_workspace.source,
        export_workspace.config,
        destination=replay,
        template_paths=export_workspace.templates,
        explicit_fixture=True,
    )

    assert second == first
    assert _bundle_hashes(replay) == first_hashes


def test_every_non_manifest_file_is_hash_bound(export_workspace: ExportWorkspace) -> None:
    _export(export_workspace)
    for relative in sorted(BUNDLED_PATHS):
        path = export_workspace.destination / relative
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(ExportError, match="manifest_artifact_rows_mismatch"):
            verify_demo_bundle(
                export_workspace.destination,
                export_workspace.config,
                explicit_fixture=True,
            )
        path.write_bytes(original)


def test_missing_and_extra_bundle_files_are_rejected(export_workspace: ExportWorkspace) -> None:
    _export(export_workspace)
    missing = export_workspace.destination / "trace.json"
    original = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ExportError, match="bundle_inventory_mismatch"):
        verify_demo_bundle(
            export_workspace.destination,
            export_workspace.config,
            explicit_fixture=True,
        )
    missing.write_bytes(original)
    extra = export_workspace.destination / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ExportError, match="bundle_inventory_mismatch"):
        verify_demo_bundle(
            export_workspace.destination,
            export_workspace.config,
            explicit_fixture=True,
        )


def test_manifest_scalar_tampering_is_recomputed(export_workspace: ExportWorkspace) -> None:
    _export(export_workspace)
    path = export_workspace.destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["quality"]["grounded_numerator"] -= 1
    atomic_write_json(path, manifest, force=True)

    with pytest.raises(ExportError, match="manifest_quality_mismatch"):
        verify_demo_bundle(
            export_workspace.destination,
            export_workspace.config,
            explicit_fixture=True,
        )


def test_template_contract_rejects_missing_extra_bad_type_and_word_overflow(
    export_workspace: ExportWorkspace,
) -> None:
    manifest = _export(export_workspace)
    template = export_workspace.templates["B"].read_text(encoding="utf-8")
    headline = json.loads(
        (export_workspace.destination / "analysis" / "headline.json").read_text()
    )
    rendered = render_demo_script(
        template,
        variant="B",
        manifest=manifest,
        headline=headline,
    )
    assert spoken_word_count(rendered) <= 225
    assert "not proof" in rendered

    with pytest.raises(ExportError, match="template_token_allowlist_mismatch"):
        render_demo_script(
            template + "\n{{headline.unregistered}}\n",
            variant="B",
            manifest=manifest,
            headline=headline,
        )
    with pytest.raises(ExportError, match="template_token_allowlist_mismatch"):
        render_demo_script(
            template.replace("{{manifest.spoken_question}}", ""),
            variant="B",
            manifest=manifest,
            headline=headline,
        )
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["spoken_question"] = ["not", "scalar"]
    with pytest.raises(ExportError, match="template_token_type_invalid"):
        render_demo_script(
            template,
            variant="B",
            manifest=invalid_manifest,
            headline=headline,
        )
    overflow = template.replace(
        "## Release disclosure",
        ("word " * 230) + "\n\n## Release disclosure",
    )
    with pytest.raises(ExportError, match="script_exceeds_225_spoken_words"):
        render_demo_script(
            overflow,
            variant="B",
            manifest=manifest,
            headline=headline,
        )
    with pytest.raises(ExportError, match="script_required_caveat_missing"):
        render_demo_script(
            template.replace("not proof", "not a guarantee"),
            variant="B",
            manifest=manifest,
            headline=headline,
        )


def test_staging_failure_preserves_last_good_release(export_workspace: ExportWorkspace) -> None:
    _export(export_workspace)
    before = _bundle_hashes(export_workspace.destination)
    bad_template = export_workspace.destination.parent / "bad-template.md"
    bad_template.write_text("## Spoken copy\ninvalid\n", encoding="utf-8")
    templates = {**export_workspace.templates, "B": bad_template}

    with pytest.raises(ExportError, match="template_token_allowlist_mismatch"):
        export_demo(
            export_workspace.source,
            export_workspace.config,
            destination=export_workspace.destination,
            template_paths=templates,
            explicit_fixture=True,
            force=True,
        )
    assert _bundle_hashes(export_workspace.destination) == before


def test_fixture_and_dirty_lineage_guards(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    fixture = _build_workspace(tmp_path / "fixture", repo_root)
    with pytest.raises(ExportError, match="fixture_runtime_guard_failed"):
        export_demo(
            fixture.source,
            fixture.config,
            destination=fixture.destination,
            template_paths=fixture.templates,
        )

    dirty = _build_workspace(tmp_path / "dirty", repo_root, code_version="dirty:" + "d" * 64)
    with pytest.raises(ExportError, match="dirty_lineage_requires_override"):
        export_demo(
            dirty.source,
            dirty.config,
            destination=dirty.destination,
            template_paths=dirty.templates,
            explicit_fixture=True,
        )
    export_demo(
        dirty.source,
        dirty.config,
        destination=dirty.destination,
        template_paths=dirty.templates,
        explicit_fixture=True,
        allow_dirty_demo=True,
    )


def test_valid_scaled_candidate_promotes_with_separate_identity(
    export_workspace: ExportWorkspace,
) -> None:
    first = _export(export_workspace)
    v1_release_id = first["release_selection"]["selected_release"]["release_id"]
    scaled = replace(export_workspace.source, corpus_role="scaled")

    promoted = export_demo(
        scaled,
        export_workspace.config,
        destination=export_workspace.destination,
        template_paths=export_workspace.templates,
        explicit_fixture=True,
        force=True,
    )

    selection = promoted["release_selection"]
    assert selection["disposition"] == "scaled_promoted"
    assert selection["selected_release"]["corpus_role"] == "scaled"
    assert selection["selected_release"]["release_id"] != v1_release_id
    assert selection["scaled_attempt"]["status"] == "selected"
    assert selection["scaled_attempt"]["candidate_release_id"] == selection[
        "selected_release"
    ]["release_id"]


def test_incomplete_scaled_candidate_retains_frozen_v1(export_workspace: ExportWorkspace) -> None:
    first = _export(export_workspace)
    selected_before = first["release_selection"]["selected_release"]
    scientific_before = {
        relative: sha256_file(export_workspace.destination / relative)
        for relative in sorted(BUNDLED_PATHS - {"release_selection.json", "demo_script.md"})
    }
    export_workspace.source.stage_run_paths["s5"].unlink()

    retained = export_demo(
        replace(export_workspace.source, corpus_role="scaled"),
        export_workspace.config,
        destination=export_workspace.destination,
        template_paths=export_workspace.templates,
        explicit_fixture=True,
        force=True,
    )

    selection = retained["release_selection"]
    assert selection["disposition"] == "v1_retained_scaled_incomplete"
    assert selection["selected_release"] == selected_before
    assert selection["scaled_attempt"]["status"] == "incomplete"
    assert selection["scaled_attempt"]["failure_code"] == "scaled_incomplete"
    assert selection["scaled_attempt"]["last_completed_stage"] == "s4"
    assert {
        relative: sha256_file(export_workspace.destination / relative)
        for relative in scientific_before
    } == scientific_before


def test_scaled_failure_precedence_is_total_and_deterministic() -> None:
    expected = {
        "integrity": (
            "v1_retained_scaled_corrupt",
            "rejected",
            "scaled_artifact_integrity_failed",
        ),
        "reconciliation": (
            "v1_retained_scaled_unreconciled",
            "rejected",
            "scaled_ledger_reconciliation_failed",
        ),
        "unvalidated": (
            "v1_retained_scaled_unvalidated",
            "rejected",
            "scaled_trust_or_offline_validation_failed",
        ),
        "incomplete": (
            "v1_retained_scaled_incomplete",
            "incomplete",
            "scaled_incomplete",
        ),
    }
    precedence = list(CONTRADICTORY_FAILURES)
    nonempty_subsets = chain.from_iterable(
        combinations(CONTRADICTORY_FAILURES, size)
        for size in range(1, len(CONTRADICTORY_FAILURES) + 1)
    )
    for signals in nonempty_subsets:
        winner = next(item for item in precedence if item in signals)
        assert classify_scaled_failure(signals) == expected[winner]
    assert classify_scaled_failure(()) == expected["incomplete"]


def test_manifest_has_no_recursive_or_operational_s7_hash(
    export_workspace: ExportWorkspace,
) -> None:
    manifest = _export(export_workspace)
    serialized = json.dumps(manifest, sort_keys=True)

    assert "manifest.json" not in {row["path"] for row in manifest["artifacts"]}
    assert '"stage": "s7"' not in serialized
    assert [row["stage"] for row in manifest["lineage"]] == ["s3", "s4", "s5"]


def test_audit_legacy_count_aliases_are_rejected(export_workspace: ExportWorkspace) -> None:
    from literature_multiverse import export as export_module

    audit = json.loads(export_workspace.source.source_path("audit.json").read_text())
    audit["audit_correct"] = audit["correct_count"]
    with pytest.raises(ExportError, match="audit_legacy_count_alias_forbidden"):
        export_module._validate_audit(audit)


def test_source_row_value_changes_alter_the_cohort_hash(
    export_workspace: ExportWorkspace,
) -> None:
    validated = validate_release_source(
        export_workspace.source,
        export_workspace.config,
        explicit_fixture=True,
    )
    changed = deepcopy(validated.primary_rows)
    changed[0]["evidence_quote"] = "A changed scientific value."
    moderator_changed = deepcopy(validated.primary_rows)
    moderator_changed[0]["moderators"]["dose_regime"] = "high"

    assert cohort_sha256(changed) != validated.cohort_sha256
    assert cohort_sha256(moderator_changed) != validated.cohort_sha256


def test_config_hash_is_independently_bound_into_release_identity(
    export_workspace: ExportWorkspace,
) -> None:
    validated = validate_release_source(
        export_workspace.source,
        export_workspace.config,
        explicit_fixture=True,
    )
    selection = _export(export_workspace)["release_selection"]["selected_release"]

    assert selection["release_id"] == validated.release_id
    assert selection["stage_run_sha256s"] == validated.stage_run_sha256s.model_dump()
    assert selection["evidence_sha256s"] == validated.evidence_sha256s.model_dump()


def test_parquet_row_counts_are_manifest_bound(export_workspace: ExportWorkspace) -> None:
    manifest = _export(export_workspace)
    rows = {row["path"]: row["rows"] for row in manifest["artifacts"]}

    assert rows["papers.parquet"] == 20
    assert rows["findings.parquet"] == 22
    assert rows["analysis/moderators.parquet"] == 4
    assert rows["analysis/contradictions.parquet"] == 0
    assert rows["analysis/evidence_gaps.parquet"] == 8
    assert all(rows[path] is None for path in BUNDLED_PATHS if not path.endswith(".parquet"))


def test_bundle_tables_are_self_contained(export_workspace: ExportWorkspace) -> None:
    _export(export_workspace)
    papers = pd.read_parquet(export_workspace.destination / "papers.parquet")
    findings = pd.read_parquet(export_workspace.destination / "findings.parquet")

    assert set(findings["paper_id"]) <= set(papers["paper_id"])
    assert len(findings["finding_id"]) == len(set(findings["finding_id"]))
    assert not any(path.is_symlink() for path in export_workspace.destination.rglob("*"))
