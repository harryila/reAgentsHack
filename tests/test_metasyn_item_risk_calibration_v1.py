from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import literature_multiverse.metasyn_item_risk_calibration_v1 as risk_module
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_item_risk_calibration_v1 import (
    AdjudicationQuestionFileV1,
    AdjudicationSidecarManifestV1,
    ArtifactBackedItemRiskAssignmentV1,
    MetaSynItemRiskCalibrationRunV1,
    MetaSynItemRiskCalibrationV1Error,
    MetaSynItemRiskPreparationV1,
    MetaSynTerminalRiskFeatureSetV1,
    TerminalRiskFeatureRowV1,
    assign_artifact_backed_item_risk_v1,
    calibrate_metasyn_item_risk_v1,
    materialize_metasyn_terminal_risk_features_v1,
    prepare_metasyn_item_risk_calibration_v1,
    validate_artifact_backed_item_risk_assignment_v1,
    validate_metasyn_item_risk_calibration_run_v1,
    validate_metasyn_item_risk_preparation_v1,
    validate_metasyn_terminal_risk_features_v1,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SALT = risk_module.PRESPECIFIED_SPLIT_SALT


def _hash(value: Any) -> str:
    return hash_canonical({"test_value": value})


def _terminal(
    *, question_id: str, publication_id: str, candidate_index: int = 1
) -> tuple[SimpleNamespace, SimpleNamespace]:
    descriptor = _hash(f"descriptor:{question_id}:{publication_id}:{candidate_index}")
    binding = _hash(f"binding:{question_id}:{publication_id}:{candidate_index}")
    terminal_sha = _hash(f"terminal:{question_id}:{publication_id}:{candidate_index}")
    typed_effect_sha = _hash(f"typed:{question_id}:{publication_id}:{candidate_index}")
    terminal = SimpleNamespace(
        authorizes_typed_effect=True,
        candidate_index=candidate_index,
        candidate_descriptor_sha256=descriptor,
        candidate_binding_sha256=binding,
        terminal_sha256=terminal_sha,
        grounding_receipt_sha256=_hash(f"grounding:{terminal_sha}"),
        assembly_receipt_sha256=_hash(f"assembly:{terminal_sha}"),
        assembly_receipt=SimpleNamespace(
            typed_effect_sha256=typed_effect_sha,
            typed_effect=SimpleNamespace(
                extraction_method="reported",
                effect=SimpleNamespace(effect_kind="direct_standard_error"),
            ),
        ),
        packet_input=SimpleNamespace(
            candidate=SimpleNamespace(passage_ids=[_hash(f"passage:{terminal_sha}")]),
            projection_surface=SimpleNamespace(
                omitted_passage_count=0,
                source_strength=SimpleNamespace(source_content_scope="full_text_sections"),
            ),
        ),
    )
    effect = SimpleNamespace(
        candidate_descriptor_sha256=descriptor,
        compatibility_sha256=_hash(f"compatibility:{terminal_sha}"),
        exact_evidence_quote=(
            "The prespecified outcome estimate and standard error were reported "
            "with exact source support in this passage."
        ),
        coverage=SimpleNamespace(
            coverage_blockers=[
                "equivalence_conclusion_not_extracted",
                "moderators_not_extracted",
                "reported_significance_conclusion_not_extracted",
            ]
        ),
    )
    return terminal, effect


def _fake_analysis(
    *,
    question_count: int = 10,
    completed: bool = True,
    terminals_per_publication: int = 1,
) -> SimpleNamespace:
    rows = []
    joins = []
    for index in range(question_count):
        question_id = f"question-{index:02d}"
        publication_id = f"publication-{index:02d}"
        row_key = f"row-{index:02d}"
        rows.append(
            SimpleNamespace(
                row_key=row_key,
                question_surface=SimpleNamespace(relation_kind="intervention"),
            )
        )
        terminals: list[SimpleNamespace] = []
        effects: list[SimpleNamespace] = []
        if completed:
            for candidate_index in range(1, terminals_per_publication + 1):
                terminal, effect = _terminal(
                    question_id=question_id,
                    publication_id=publication_id,
                    candidate_index=candidate_index,
                )
                terminals.append(terminal)
                effects.append(effect)
        joins.append(
            SimpleNamespace(
                row_ordinal=index,
                row_key=row_key,
                question_id=question_id,
                publication=SimpleNamespace(
                    publication_id=publication_id,
                    paper_id=f"paper-{index:02d}",
                ),
                publication_join_sha256=_hash(f"join:{index}"),
                source_strength_surface_sha256=_hash(f"strength:{index}"),
                inventory_receipt_sha256=_hash(f"inventory:{index}"),
                inventoried_candidate_count=len(terminals),
                candidate_terminals=terminals,
                compatibility_effects=effects,
            )
        )
    bridge = SimpleNamespace(
        execution_bundle=SimpleNamespace(extraction_inputs=SimpleNamespace(rows=rows)),
        publication_joins=joins,
    )
    return SimpleNamespace(
        analysis_sha256=_hash(f"analysis:{question_count}:{completed}"),
        bridge_sha256=_hash(f"bridge:{question_count}:{completed}"),
        terminal_membership_sha256=_hash(f"terminal-membership:{question_count}:{completed}"),
        publication_join_membership_sha256=_hash(f"join-membership:{question_count}:{completed}"),
        bridge=bridge,
    )


@pytest.fixture
def external_replay_stub(monkeypatch: pytest.MonkeyPatch):
    calls: list[bool] = []

    def validate(*, analysis: Any, repository_root: Path, external_replay: bool):
        assert Path(repository_root).resolve() == ROOT.resolve()
        calls.append(external_replay)
        return analysis

    monkeypatch.setattr(risk_module, "validate_metasyn_grounded_analysis_v2", validate)
    return calls


def _prepare_and_materialize(
    source: SimpleNamespace,
) -> tuple[MetaSynItemRiskPreparationV1, MetaSynTerminalRiskFeatureSetV1]:
    preparation = prepare_metasyn_item_risk_calibration_v1(
        analysis=source,
        repository_root=ROOT,
        split_salt=SPLIT_SALT,
    )
    features = materialize_metasyn_terminal_risk_features_v1(
        preparation=preparation,
        analysis=source,
        repository_root=ROOT,
    )
    return preparation, features


def _write_sidecar_directory(
    root: Path,
    preparation: MetaSynItemRiskPreparationV1,
    features: MetaSynTerminalRiskFeatureSetV1,
    *,
    incomplete_question: str | None = None,
    poison_evaluation_files: bool = True,
    stale_pipeline: bool = False,
) -> Path:
    root.mkdir()
    question_root = root / "questions"
    question_root.mkdir()
    calibration_ids = set(preparation.split.calibration_question_ids)
    entries = []
    for index, question_id in enumerate(preparation.split.eligible_question_ids):
        relative_path = f"questions/question-{index:03d}.json"
        target = root / relative_path
        if question_id not in calibration_ids and poison_evaluation_files:
            content = b"\x80POISONED-EVALUATION-LABEL-FILE-NOT-JSON"
        else:
            items = sorted(
                [
                    {
                        "item_id": row.item.item_id,
                        "observed_error": row.item.row_ordinal % 3 == 0,
                        "adjudication_artifact_sha256": _hash(f"adjudication:{row.item.item_id}"),
                    }
                    for row in features.rows
                    if row.item.question_id == question_id
                ],
                key=lambda item: item["item_id"],
            )
            if question_id == incomplete_question:
                items = items[:-1]
            question_payload = {
                "question_file_version": "metasyn-question-adjudication-file-v1",
                "preparation_sha256": preparation.preparation_sha256,
                "feature_set_sha256": features.feature_set_sha256,
                "split_sha256": preparation.split_sha256,
                "pipeline_sha256": ("f" * 64 if stale_pipeline else preparation.pipeline_sha256),
                "score_model_sha256": preparation.score_model_sha256,
                "question_id": question_id,
                "question_adjudication_artifact_sha256": _hash(
                    f"question-adjudication:{question_id}"
                ),
                "complete_question": True,
                "label_source": "expert_adjudication",
                "adjudication_protocol_sha256": _hash("adjudication-protocol"),
                "items": items,
                "simulation": False,
            }
            question_payload["question_file_sha256"] = hash_canonical(question_payload)
            AdjudicationQuestionFileV1.model_validate(question_payload)
            content = json.dumps(question_payload, sort_keys=True).encode("utf-8")
        target.write_bytes(content)
        entry_payload = {
            "question_id": question_id,
            "relative_path": relative_path,
            "file_sha256": hashlib.sha256(content).hexdigest(),
            "file_bytes": len(content),
        }
        entries.append({**entry_payload, "entry_sha256": hash_canonical(entry_payload)})
    manifest_payload = {
        "manifest_version": "metasyn-adjudication-sidecar-manifest-v1",
        "preparation_sha256": preparation.preparation_sha256,
        "feature_set_sha256": features.feature_set_sha256,
        "split_sha256": preparation.split_sha256,
        "pipeline_sha256": ("f" * 64 if stale_pipeline else preparation.pipeline_sha256),
        "score_model_sha256": preparation.score_model_sha256,
        "split_salt_sha256": preparation.split.split_salt_sha256,
        "question_files": entries,
        "question_ids": preparation.split.eligible_question_ids,
        "question_file_membership_sha256": hash_canonical(
            [item["entry_sha256"] for item in entries]
        ),
        "label_values_present": False,
        "observed_error_fields_present": False,
        "simulation": False,
    }
    manifest_payload["manifest_sha256"] = hash_canonical(manifest_payload)
    AdjudicationSidecarManifestV1.model_validate(manifest_payload)
    (root / "manifest.json").write_text(
        json.dumps(manifest_payload, sort_keys=True), encoding="utf-8"
    )
    return root


def _calibrated_artifacts(
    root: Path,
    source: SimpleNamespace,
) -> tuple[
    MetaSynItemRiskPreparationV1,
    MetaSynTerminalRiskFeatureSetV1,
    Path,
    MetaSynItemRiskCalibrationRunV1,
]:
    preparation, features = _prepare_and_materialize(source)
    sidecar_directory = _write_sidecar_directory(root, preparation, features)
    run = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )
    assert run.status == "calibrated_scheduling_only"
    return preparation, features, sidecar_directory, run


def test_successful_artifact_backed_calibration_is_scheduling_only(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    sidecar_directory = _write_sidecar_directory(
        tmp_path / "adjudication-sidecar",
        preparation,
        features,
    )
    manifest = json.loads((sidecar_directory / "manifest.json").read_text(encoding="utf-8"))
    assert all("observed_error" not in item for item in [manifest, *manifest["question_files"]])
    entries = {item["question_id"]: item for item in manifest["question_files"]}
    for question_id in preparation.split.evaluation_question_ids:
        (sidecar_directory / entries[question_id]["relative_path"]).chmod(0)

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )

    assert preparation.status == "ready_for_label_blind_materialization"
    assert preparation.split.calibration_question_count == 5
    assert preparation.split.evaluation_question_count == 5
    assert set(preparation.split.calibration_question_ids).isdisjoint(
        preparation.split.evaluation_question_ids
    )
    assert features.feature_row_count == 10
    assert features.scores_computed_not_supplied is True
    assert features.shift_assessment.status == "no_shift_detected"
    assert result.status == "calibrated_scheduling_only"
    assert result.bound_input_question_count == 4
    assert result.structural_legacy_development_label_used_for_fitting is False
    assert result.bounds_receipt is not None
    assert result.bounds_receipt.claim_release_authority is False
    assert result.bounds_receipt.generic_core_bundle_exported is False
    assert result.bounds_receipt.accepts_caller_supplied_scores is False
    assert result.scheduling_authority is True
    assert result.evaluation_labels_opened is False
    assert result.calibration_sidecar is not None
    assert set(result.calibration_sidecar.opened_relative_paths).isdisjoint(
        result.calibration_sidecar.evaluation_relative_paths
    )
    assert result.calibration_sidecar.evaluation_files_opened is False
    assert "calibration_bundle" not in result.model_dump(mode="json")
    assert all(external_replay_stub)

    component = preparation.pipeline_fingerprint.components[0]
    fingerprinted_paths = {item.path for item in component.files}
    assert risk_module.MODULE_PATH in fingerprinted_paths
    assert risk_module.CLI_PATH in fingerprinted_paths
    assert component.settings["split_sha256"] == preparation.split_sha256
    assert component.settings["score_model_sha256"] == preparation.score_model_sha256

    replayed = validate_metasyn_item_risk_calibration_run_v1(
        calibration_run=result,
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )
    assert replayed == result

    forged_access = result.model_dump(mode="json")
    forged_access["labels_opened"] = False
    forged_access["sidecar_manifest_file_sha256"] = None
    forged_access["calibration_run_sha256"] = hash_canonical(
        {key: value for key, value in forged_access.items() if key != "calibration_run_sha256"}
    )
    with pytest.raises(ValidationError, match="calibrated_shape_mismatch"):
        MetaSynItemRiskCalibrationRunV1.model_validate(forged_access)


def test_artifact_backed_assignment_replays_membership_and_exports_only_group_ucl(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features, sidecar_directory, run = _calibrated_artifacts(
        tmp_path / "assignment-sidecar",
        source,
    )
    evaluation_ids = set(preparation.split.evaluation_question_ids)
    row = next(item for item in features.rows if item.item.question_id in evaluation_ids)

    assignment = assign_artifact_backed_item_risk_v1(
        feature_row=row,
        preparation=preparation,
        feature_set=features,
        calibration_run=run,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )

    assert assignment.feature_row == row
    assert assignment.computed_risk_score == row.risk_score
    assert assignment.domain == row.item.domain
    assert assignment.bin_lower <= row.risk_score
    assert assignment.conservative_group_upper_error_rate >= 0
    assert assignment.accepts_caller_supplied_score is False
    assert assignment.accepts_caller_supplied_domain is False
    assert assignment.accepts_caller_supplied_bin is False
    assert assignment.individual_item_probability_authority is False
    assert assignment.scheduling_authority is True
    assert assignment.claim_release_authority is False

    replayed = validate_artifact_backed_item_risk_assignment_v1(
        assignment=assignment,
        preparation=preparation,
        feature_set=features,
        calibration_run=run,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )
    assert replayed == assignment
    assert all(external_replay_stub)


def test_assignment_rejects_untyped_nonmember_and_coherent_output_tamper(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features, sidecar_directory, run = _calibrated_artifacts(
        tmp_path / "assignment-tamper-sidecar",
        source,
    )
    row = features.rows[0]

    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="requires_typed_feature_row"):
        assign_artifact_backed_item_risk_v1(
            feature_row=row.model_dump(mode="json"),  # type: ignore[arg-type]
            preparation=preparation,
            feature_set=features,
            calibration_run=run,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=sidecar_directory,
        )

    nonmember_payload = row.model_dump(mode="json")
    nonmember_payload["split"] = "evaluation" if row.split == "calibration" else "calibration"
    nonmember_payload["feature_row_sha256"] = hash_canonical(
        {key: value for key, value in nonmember_payload.items() if key != "feature_row_sha256"}
    )
    nonmember = TerminalRiskFeatureRowV1.model_validate(nonmember_payload)
    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="feature_row_not_member"):
        assign_artifact_backed_item_risk_v1(
            feature_row=nonmember,
            preparation=preparation,
            feature_set=features,
            calibration_run=run,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=sidecar_directory,
        )

    assignment = assign_artifact_backed_item_risk_v1(
        feature_row=row,
        preparation=preparation,
        feature_set=features,
        calibration_run=run,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )
    tampered = assignment.model_dump(mode="json")
    tampered["conservative_group_upper_error_rate"] = 0.0
    tampered["assignment_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "assignment_sha256"}
    )
    with pytest.raises(ValidationError, match="assignment_bound_alias_mismatch"):
        ArtifactBackedItemRiskAssignmentV1.model_validate(tampered)
    assert all(external_replay_stub)


def test_assignment_rejects_unsupported_replayed_row_domain(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features, _sidecar_directory, run = _calibrated_artifacts(
        tmp_path / "unsupported-domain-sidecar",
        source,
    )
    assert run.bounds_receipt is not None
    assert run.bounds_receipt.supported_deployment_domains == ["metasyn-intervention"]

    exposure_source = _fake_analysis()
    for row in exposure_source.bridge.execution_bundle.extraction_inputs.rows:
        row.question_surface.relation_kind = "exposure"
    _exposure_preparation, exposure_features = _prepare_and_materialize(exposure_source)
    exposure_row = exposure_features.rows[0]
    assert exposure_row.item.domain == "metasyn-exposure"

    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="domain_unsupported"):
        risk_module._bound_cell_for_replayed_feature_row(
            feature_row=exposure_row,
            receipt=run.bounds_receipt,
        )
    assert preparation.pipeline_sha256 != _exposure_preparation.pipeline_sha256
    assert features.rows[0].item.domain == "metasyn-intervention"
    assert all(external_replay_stub)


def test_assignment_uses_half_open_prespecified_bin_at_exact_boundary(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis(terminals_per_publication=2)
    for join in source.bridge.publication_joins:
        join.compatibility_effects[0].coverage.coverage_blockers = []
        join.compatibility_effects[0].exact_evidence_quote = "q" * 80
        join.candidate_terminals[
            0
        ].packet_input.projection_surface.source_strength.source_content_scope = "title_abstract"
    preparation, features, sidecar_directory, run = _calibrated_artifacts(
        tmp_path / "boundary-sidecar",
        source,
    )
    assert {row.risk_score for row in features.rows} == {0.25}
    evaluation_ids = set(preparation.split.evaluation_question_ids)
    row = next(item for item in features.rows if item.item.question_id in evaluation_ids)

    assignment = assign_artifact_backed_item_risk_v1(
        feature_row=row,
        preparation=preparation,
        feature_set=features,
        calibration_run=run,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=sidecar_directory,
    )

    assert assignment.computed_risk_score == 0.25
    assert assignment.bin_id == "risk-bin-001"
    assert assignment.bin_lower == 0.25
    assert assignment.bin_upper == 0.5
    assert assignment.bin_upper_inclusive is False
    assert assignment.cell_calibration_units == 4
    assert all(external_replay_stub)


def test_ready_run_without_label_sidecar_abstains_explicitly(
    external_replay_stub: list[bool],
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=None,
    )

    assert result.status == "abstained_no_calibration_labels"
    assert result.labels_opened is False
    assert result.bounds_receipt is None
    assert result.scheduling_authority is False
    assert result.evaluation_labels_opened is False
    assert all(external_replay_stub)


def test_full_roster_selects_at_most_one_terminal_per_question_publication(
    external_replay_stub: list[bool],
) -> None:
    source = _fake_analysis(terminals_per_publication=2)
    preparation, features = _prepare_and_materialize(source)

    assert preparation.eligible_item_count == 10
    assert features.feature_row_count == 10
    assert {item.candidate_index for item in preparation.eligible_items} == {1}
    assert (
        len({(item.question_id, item.publication_id) for item in preparation.eligible_items}) == 10
    )
    assert all(external_replay_stub)


def test_split_tamper_and_coherent_score_tamper_fail_closed(
    external_replay_stub: list[bool],
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)

    split_payload = preparation.model_dump(mode="json")
    (
        split_payload["split"]["calibration_question_ids"][0],
        split_payload["split"]["evaluation_question_ids"][0],
    ) = (
        split_payload["split"]["evaluation_question_ids"][0],
        split_payload["split"]["calibration_question_ids"][0],
    )
    split_payload["split"]["calibration_question_ids"].sort()
    split_payload["split"]["evaluation_question_ids"].sort()
    split_payload["split"]["split_sha256"] = hash_canonical(
        {key: value for key, value in split_payload["split"].items() if key != "split_sha256"}
    )
    with pytest.raises(ValidationError, match="split_assignment_mismatch"):
        MetaSynItemRiskPreparationV1.model_validate(split_payload)

    feature_payload = features.model_dump(mode="json")
    feature_payload["rows"][0]["risk_score"] = 0.999
    feature_payload["rows"][0]["feature_row_sha256"] = hash_canonical(
        {
            key: value
            for key, value in feature_payload["rows"][0].items()
            if key != "feature_row_sha256"
        }
    )
    feature_payload["feature_row_membership_sha256"] = hash_canonical(
        [row["feature_row_sha256"] for row in feature_payload["rows"]]
    )
    feature_payload["feature_set_sha256"] = hash_canonical(
        {key: value for key, value in feature_payload.items() if key != "feature_set_sha256"}
    )
    with pytest.raises(ValidationError, match="feature_score_replay_mismatch"):
        MetaSynTerminalRiskFeatureSetV1.model_validate(feature_payload)
    assert all(external_replay_stub)


def test_insufficient_real_yield_never_opens_label_path(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis(question_count=7)
    preparation, features = _prepare_and_materialize(source)
    missing_path = tmp_path / "must-not-be-opened.json"

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=missing_path,
    )

    assert result.status == "abstained_too_few_complete_questions"
    assert result.labels_opened is False
    assert result.sidecar_manifest_file_sha256 is None
    assert not missing_path.exists()
    assert all(external_replay_stub)


def test_zero_terminal_yield_abstains_without_pseudo_units(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis(completed=False)
    preparation, features = _prepare_and_materialize(source)

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=tmp_path / "must-not-be-opened",
    )

    assert preparation.eligible_items == []
    assert features.rows == []
    assert result.status == "abstained_no_eligible_terminal_artifacts"
    assert result.bounds_receipt is None
    assert result.labels_opened is False
    assert all(external_replay_stub)


def test_domain_shift_abstains_before_sidecar_open(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation = prepare_metasyn_item_risk_calibration_v1(
        analysis=source,
        repository_root=ROOT,
        split_salt=SPLIT_SALT,
    )
    calibration_ids = set(preparation.split.calibration_question_ids)
    for join in source.bridge.publication_joins:
        if join.question_id in calibration_ids:
            join.candidate_terminals[
                0
            ].packet_input.projection_surface.source_strength.source_content_scope = (
                "title_abstract"
            )
    features = materialize_metasyn_terminal_risk_features_v1(
        preparation=preparation,
        analysis=source,
        repository_root=ROOT,
    )
    missing_path = tmp_path / "must-not-be-opened.json"

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=missing_path,
    )

    assert features.shift_assessment.status == "shift_detected"
    assert result.status == "abstained_domain_shift"
    assert result.labels_opened is False
    assert not missing_path.exists()
    assert all(external_replay_stub)


@pytest.mark.parametrize("mutation", ["file", "empty_directory"])
def test_sidecar_directory_rejects_extra_tree_entry_before_question_labels(
    tmp_path: Path, external_replay_stub: list[bool], mutation: str
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    directory = _write_sidecar_directory(
        tmp_path / "sidecar-with-extra-file", preparation, features
    )
    if mutation == "file":
        (directory / "unexpected.json").write_text("{}", encoding="utf-8")
    else:
        (directory / "unexpected-directory").mkdir()

    with pytest.raises(
        MetaSynItemRiskCalibrationV1Error,
        match="sidecar_directory_roster_mismatch",
    ):
        calibrate_metasyn_item_risk_v1(
            preparation=preparation,
            feature_set=features,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=directory,
        )
    assert all(external_replay_stub)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "hardlink"])
def test_sidecar_directory_rejects_missing_or_linked_evaluation_file(
    tmp_path: Path,
    external_replay_stub: list[bool],
    mutation: str,
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    directory = _write_sidecar_directory(tmp_path / f"sidecar-{mutation}", preparation, features)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entries = {item["question_id"]: item for item in manifest["question_files"]}
    evaluation_ids = preparation.split.evaluation_question_ids
    target = directory / entries[evaluation_ids[0]]["relative_path"]
    target.unlink()
    if mutation == "symlink":
        replacement = directory / entries[evaluation_ids[1]]["relative_path"]
        target.symlink_to(replacement)
    elif mutation == "hardlink":
        replacement = directory / entries[evaluation_ids[1]]["relative_path"]
        target.hardlink_to(replacement)

    expected = {
        "missing": "directory_roster",
        "symlink": "sidecar_symlink_forbidden",
        "hardlink": "sidecar_hardlink_forbidden",
    }[mutation]
    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match=expected):
        calibrate_metasyn_item_risk_v1(
            preparation=preparation,
            feature_set=features,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=directory,
        )
    assert all(external_replay_stub)


def test_manifest_rejects_duplicate_and_path_escape_before_label_access() -> None:
    entry_payload = {
        "question_id": "question-00",
        "relative_path": "../escape.json",
        "file_sha256": "a" * 64,
        "file_bytes": 1,
    }
    with pytest.raises(ValidationError, match="relative_path_invalid"):
        risk_module.AdjudicationManifestEntryV1.model_validate(
            {**entry_payload, "entry_sha256": hash_canonical(entry_payload)}
        )

    safe_entry_payload = {
        **entry_payload,
        "relative_path": "questions/question-000.json",
    }
    safe_entry = {
        **safe_entry_payload,
        "entry_sha256": hash_canonical(safe_entry_payload),
    }
    manifest_payload = {
        "manifest_version": "metasyn-adjudication-sidecar-manifest-v1",
        "preparation_sha256": "b" * 64,
        "feature_set_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "pipeline_sha256": "e" * 64,
        "score_model_sha256": "f" * 64,
        "split_salt_sha256": "1" * 64,
        "question_files": [safe_entry, safe_entry],
        "question_ids": ["question-00", "question-00"],
        "question_file_membership_sha256": hash_canonical(
            [safe_entry["entry_sha256"], safe_entry["entry_sha256"]]
        ),
        "label_values_present": False,
        "observed_error_fields_present": False,
        "simulation": False,
    }
    manifest_payload["manifest_sha256"] = hash_canonical(manifest_payload)
    with pytest.raises(ValidationError, match="manifest_roster_invalid"):
        AdjudicationSidecarManifestV1.model_validate(manifest_payload)


def test_mixed_question_calibration_file_fails_closed(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    directory = _write_sidecar_directory(
        tmp_path / "mixed-question-sidecar",
        preparation,
        features,
        poison_evaluation_files=False,
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["question_id"]: item for item in manifest["question_files"]}
    calibration_id = preparation.split.calibration_question_ids[0]
    evaluation_id = preparation.split.evaluation_question_ids[0]
    calibration_entry = entries[calibration_id]
    evaluation_entry = entries[evaluation_id]
    mixed_content = (directory / evaluation_entry["relative_path"]).read_bytes()
    (directory / calibration_entry["relative_path"]).write_bytes(mixed_content)
    calibration_entry["file_sha256"] = hashlib.sha256(mixed_content).hexdigest()
    calibration_entry["file_bytes"] = len(mixed_content)
    calibration_entry["entry_sha256"] = hash_canonical(
        {key: value for key, value in calibration_entry.items() if key != "entry_sha256"}
    )
    manifest["question_file_membership_sha256"] = hash_canonical(
        [item["entry_sha256"] for item in manifest["question_files"]]
    )
    manifest["manifest_sha256"] = hash_canonical(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="mixed_question_file"):
        calibrate_metasyn_item_risk_v1(
            preparation=preparation,
            feature_set=features,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=directory,
        )
    assert all(external_replay_stub)


def test_incomplete_complete_question_sidecar_abstains_explicitly(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    incomplete_question = preparation.split.calibration_question_ids[-1]
    directory = _write_sidecar_directory(
        tmp_path / "incomplete-sidecar",
        preparation,
        features,
        incomplete_question=incomplete_question,
    )

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=directory,
    )

    assert result.status == "abstained_incomplete_question_sidecar"
    assert result.labels_opened is True
    assert result.calibration_sidecar is None
    assert f"question_item_roster_incomplete:{incomplete_question}" in result.blockers
    assert all(external_replay_stub)


def test_cross_split_paper_reuse_abstains_without_opening_labels(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    initial = prepare_metasyn_item_risk_calibration_v1(
        analysis=source,
        repository_root=ROOT,
        split_salt=SPLIT_SALT,
    )
    calibration_question = initial.split.calibration_question_ids[0]
    evaluation_question = initial.split.evaluation_question_ids[0]
    for join in source.bridge.publication_joins:
        if join.question_id in {calibration_question, evaluation_question}:
            join.publication.paper_id = "paper-shared-across-split"
    preparation, features = _prepare_and_materialize(source)
    missing_path = tmp_path / "must-not-be-opened.json"

    result = calibrate_metasyn_item_risk_v1(
        preparation=preparation,
        feature_set=features,
        analysis=source,
        repository_root=ROOT,
        adjudication_sidecar_directory=missing_path,
    )

    assert any(
        item.startswith("cross_split_paper_reuse:") for item in preparation.preparation_blockers
    )
    assert result.status == "abstained_split_independence_violation"
    assert result.labels_opened is False
    assert not missing_path.exists()
    assert all(external_replay_stub)


def test_stale_sidecar_lineage_and_external_replay_mismatch_fail_closed(
    tmp_path: Path, external_replay_stub: list[bool]
) -> None:
    source = _fake_analysis()
    preparation, features = _prepare_and_materialize(source)
    directory = _write_sidecar_directory(
        tmp_path / "stale-sidecar",
        preparation,
        features,
        stale_pipeline=True,
    )

    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="frozen_lineage_mismatch"):
        calibrate_metasyn_item_risk_v1(
            preparation=preparation,
            feature_set=features,
            analysis=source,
            repository_root=ROOT,
            adjudication_sidecar_directory=directory,
        )

    changed = deepcopy(source)
    changed.analysis_sha256 = _hash("changed-analysis")
    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="external_replay_mismatch"):
        validate_metasyn_item_risk_preparation_v1(
            preparation=preparation,
            analysis=changed,
            repository_root=ROOT,
        )
    with pytest.raises(MetaSynItemRiskCalibrationV1Error, match="external_replay_mismatch"):
        validate_metasyn_terminal_risk_features_v1(
            feature_set=features,
            analysis=changed,
            repository_root=ROOT,
        )
    assert all(external_replay_stub)
