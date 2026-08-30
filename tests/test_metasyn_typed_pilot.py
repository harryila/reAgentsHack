from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from scripts.run_metasyn_typed_pilot import main as pilot_cli_main
from tests.private_cache_support import require_private_cache

import literature_multiverse.metasyn_typed_pilot as metasyn_typed_pilot
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_typed_pilot import (
    _PILOT_DEPENDENCY_ENTRYPOINTS,
    _PILOT_NON_PYTHON_INPUTS,
    EXPECTED_SELECTED_COMPONENTS,
    EXPECTED_SELECTED_PAPERS,
    EXPECTED_SELECTED_QUESTIONS,
    FORBIDDEN_REFERENCE_COLUMNS,
    MATERIALIZED_REVIEW_COLUMNS,
    PREPARE_BUNDLE_FILENAME,
    MetaSynTypedPilotError,
    MetaSynTypedPilotPrepareBundleV1,
    _build_prepare_bundle,
    _component_assignments,
    _load_protocol_rows,
    _question_spec,
    _select_rows,
    compute_metasyn_typed_pilot_pipeline_fingerprint,
    freeze_metasyn_pilot_selection_config,
    validate_metasyn_typed_pilot_prepare,
)
from literature_multiverse.native_extraction import (
    native_publication_extraction_json_schema,
)
from literature_multiverse.source_manifest_bridge import SourceContentScope
from literature_multiverse.verifier import compute_verifier_pipeline_fingerprint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Four-field identity/fingerprint surface that a rebuild is allowed to move on
# without that movement counting as a real regression (see
# `test_historical_typed_oracle_pilot_v2_diverges_from_current_pipeline_only_in_identity`
# below). Confirmed against `MetaSynTypedPilotPrepareBundleV1`: today the frozen
# `typed-oracle-pilot-v2` bundle and a live rebuild differ in exactly 7 leaves,
# all nested under these 4 top-level fields.
_IDENTITY_FIELDS = {
    "pilot_pipeline_fingerprint",
    "pilot_pipeline_sha256",
    "downstream_verifier_pipeline_sha256",
    "prepare_bundle_sha256",
}


def _row(review_id: int, matched: list[int], **updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ID": review_id,
        "Title": f"Review {review_id}",
        "Research_Question": "Does intervention A improve outcome Z?",
        "Population": "Adults",
        "Intervention": "Intervention A",
        "Exposure": None,
        "Comparison": "Comparator B",
        "Outcome": "A deliberately long verbatim outcome name that exceeds sixty-four chars",
        "inclusion_criteria": "Randomized studies",
        "exclusion_criteria": None,
        "search_end_date": "2024-01-01",
        "matched_corpus_ids": matched,
        "matched_ref_count": len(matched),
        "source_review_corpus_ids": [],
    }
    row.update(updates)
    return row


def _resolve_local_module(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _independent_dependency_closure(repository_root: Path) -> set[str]:
    pending = list(_PILOT_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (repository_root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_module(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def test_question_spec_uses_bounded_ids_and_freezes_unknown_direction_semantics() -> None:
    spec = _question_spec(_row(42, [1, 2]))

    assert spec.allowed_outcomes == ["outcome-01"]
    assert len(spec.allowed_outcomes[0]) <= 64
    assert spec.outcome_id_to_text == {
        "outcome-01": "A deliberately long verbatim outcome name that exceeds sixty-four chars"
    }
    assert spec.positive_direction_means_by_outcome_id == {
        "outcome-01": (
            "higher_reported_outcome_value_or_event_frequency_in_"
            "intervention_or_exposure_than_comparator"
        )
    }
    assert spec.clinical_benefit_direction_by_outcome_id == {
        "outcome-01": "not_prespecified_from_protocol_metadata"
    }
    assert spec.treatment_role == "intervention_or_exposure"
    assert spec.comparator_role == "comparator"
    assert spec.contrast_orientation == "intervention_or_exposure_minus_comparator"
    assert spec.directional_evaluation_eligible is False
    assert "conclusion" not in spec.model_dump(mode="json")


def test_selection_is_deterministic_and_uses_no_reference_direction() -> None:
    rows = [
        _row(2, [20, 21]),
        _row(1, [10, 11], Effect_Direction="Positive"),
        _row(3, [30, 31], Comparison=None),
    ]
    scopes = {
        10: SourceContentScope.FULL_TEXT_SECTIONS,
        11: SourceContentScope.FULL_TEXT_SECTIONS,
        20: SourceContentScope.FULL_TEXT_SECTIONS,
        21: SourceContentScope.TITLE_ABSTRACT,
        30: SourceContentScope.FULL_TEXT_SECTIONS,
        31: SourceContentScope.FULL_TEXT_SECTIONS,
    }
    config = freeze_metasyn_pilot_selection_config()

    first = _select_rows(rows=rows, source_scope_by_corpus_id=scopes, config=config)
    second = _select_rows(
        rows=list(reversed(rows)), source_scope_by_corpus_id=scopes, config=config
    )

    assert [row["ID"] for row in first] == [1]
    assert first == second


def test_connected_components_prevent_review_and_paper_overlap() -> None:
    assignments = _component_assignments(
        [
            _row(1, [10, 11]),
            _row(2, [11, 12]),
            _row(
                3,
                [30, 31],
                Research_Question="Does intervention X improve outcome Y?",
            ),
        ]
    )

    assert assignments[1] == assignments[2]
    assert assignments[1][1] == [1, 2]
    assert assignments[1][2] == hash_canonical([1, 2])
    assert assignments[3][1] == [3]
    assert assignments[1][0] != assignments[3][0]


def test_protocol_loader_materializes_only_allowlist_and_rejects_test_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _row(1, [10, 11])
    captured: dict[str, Any] = {}

    def fake_read_table(path: Path, *, columns: list[str], filters: Any) -> pa.Table:
        captured.update(path=path, columns=columns, filters=filters)
        return pa.Table.from_pylist([raw])

    monkeypatch.setattr("literature_multiverse.metasyn_typed_pilot.pq.read_table", fake_read_table)
    rows = _load_protocol_rows(reviews_train_path=Path("reviews-train.parquet"), review_ids=[1])

    assert rows[0]["ID"] == 1
    assert captured["columns"] == list(MATERIALIZED_REVIEW_COLUMNS)
    assert not set(captured["columns"]).intersection(FORBIDDEN_REFERENCE_COLUMNS)
    assert captured["filters"] == [("ID", "in", [1])]
    with pytest.raises(MetaSynTypedPilotError, match="nontraining_review_table_forbidden"):
        _load_protocol_rows(reviews_train_path=Path("reviews-test.parquet"), review_ids=[1])


def test_pilot_pipeline_fingerprint_binds_exact_local_closure_and_downstream() -> None:
    fingerprint = compute_metasyn_typed_pilot_pipeline_fingerprint(root=REPOSITORY_ROOT)
    component = fingerprint.components[0]
    observed_paths = {item.path for item in component.files}
    expected_paths = {
        *_independent_dependency_closure(REPOSITORY_ROOT),
        *_PILOT_NON_PYTHON_INPUTS,
    }

    assert observed_paths == expected_paths
    assert component.settings["in_repository_dependency_closure_bound"] is True
    assert component.settings["dependency_closure_entrypoints"] == list(
        _PILOT_DEPENDENCY_ENTRYPOINTS
    )
    assert component.settings["official_native_extraction_schema_sha256"] == hash_canonical(
        native_publication_extraction_json_schema()
    )
    assert component.settings["downstream_verifier_pipeline_sha256"] == (
        compute_verifier_pipeline_fingerprint(root=REPOSITORY_ROOT).pipeline_sha256
    )
    assert {
        "prompts/metasyn_candidate_inventory.md",
        "prompts/metasyn_candidate_packet.md",
        "configs/benchmarks/metasyn-corpus-c8fa07d.json",
    }.issubset(observed_paths)


def test_public_cli_help_exposes_only_prepare_and_external_replay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        pilot_cli_main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "prepare" in output
    assert "validate-prepare" in output
    assert "reviews-test" not in output.casefold()
    assert "effect_direction" not in output.casefold()
    assert "reference-label" not in output.casefold()


@pytest.mark.private_cache
def test_real_label_blind_calibration_census_is_exact_10_questions_32_papers() -> None:
    root = require_private_cache(
        "data/cache/metasyn/reviews-train.parquet",
        "data/cache/metasyn/screening-study-v1-final-v2/fit_receipt.json",
    )
    bundle = _build_prepare_bundle(
        repository_root=root,
        screening_work_dir=root / "data/cache/metasyn/screening-study-v1-final-v2",
        reviews_train_path=root / "data/cache/metasyn/reviews-train.parquet",
        corpus_manifest_path=root / "configs/benchmarks/metasyn-corpus-c8fa07d.json",
    )

    assert bundle.selected_question_count == EXPECTED_SELECTED_QUESTIONS == 10
    assert bundle.selected_component_count == EXPECTED_SELECTED_COMPONENTS == 10
    assert bundle.selected_paper_count == EXPECTED_SELECTED_PAPERS == 32
    assert bundle.source_modality_counts == {
        "full_text_sections": 22,
        "title_abstract": 10,
    }
    assert bundle.source_strength_counts == {
        "diagnostic_title_abstract_grounding": 10,
        "full_text_textual_grounding": 22,
    }
    assert bundle.release_grade_source_grounding_count == 22
    assert len({row.independence_component_id for row in bundle.questions}) == 10
    assert (
        len(
            {corpus_id for question in bundle.questions for corpus_id in question.oracle_corpus_ids}
        )
        == 32
    )
    assert bundle.access_state.reference_fields_unopened is True
    assert bundle.access_state.official_test_labels_opened is False
    assert all(
        not question.question_spec.directional_evaluation_eligible for question in bundle.questions
    )


@pytest.mark.private_cache
def test_historical_typed_oracle_pilot_v2_diverges_from_current_pipeline_only_in_identity() -> None:
    root = require_private_cache(
        "data/cache/metasyn/typed-oracle-pilot-v2",
        "data/cache/metasyn/screening-study-v1-final-v2/fit_receipt.json",
        "data/cache/metasyn/reviews-train.parquet",
    )
    workspace = root / "data/cache/metasyn/typed-oracle-pilot-v2"
    with pytest.raises(
        MetaSynTypedPilotError, match="metasyn_pilot_prepare_external_replay_mismatch"
    ):
        validate_metasyn_typed_pilot_prepare(repository_root=root, workspace=workspace)
    private = metasyn_typed_pilot._private_workspace(workspace, repository_root=root)
    bundle = MetaSynTypedPilotPrepareBundleV1.model_validate(
        json.loads((private / PREPARE_BUNDLE_FILENAME).read_text(encoding="utf-8"))
    )
    inputs = bundle.repository_inputs
    rebuilt = _build_prepare_bundle(
        repository_root=root,
        screening_work_dir=(root / inputs["screening_fit_receipt"]).parent,
        reviews_train_path=root / inputs["reviews_train"],
        corpus_manifest_path=root / inputs["corpus_manifest"],
    )
    assert rebuilt.model_dump(mode="json", exclude=_IDENTITY_FIELDS) == bundle.model_dump(
        mode="json", exclude=_IDENTITY_FIELDS
    )
    assert rebuilt.pilot_pipeline_sha256 != bundle.pilot_pipeline_sha256
