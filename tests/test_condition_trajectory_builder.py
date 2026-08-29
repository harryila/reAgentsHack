from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import scripts.build_condition_calibration_trajectory as cli
from tests.test_unified_verifier import _condition_runtime_fixture

from literature_multiverse.adaptive_calibration import (
    PolicyVisibleQuestionTrajectoryV2,
    freeze_condition_calibration_collection_source_roster_v1,
)
from literature_multiverse.condition_trajectory_builder import (
    ConditionTrajectoryBuilderError,
    build_condition_calibration_question_trajectory,
    read_condition_calibration_collection_source,
)
from literature_multiverse.lineage import OutputExistsError, atomic_write_json, hash_canonical
from literature_multiverse.verifier import (
    build_verifier_adaptive_policy_context,
    run_condition_calibration_collection,
)


@pytest.fixture(scope="module")
def collected_sources():
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        _assessment,
        _bundle,
        item_risk_receipt,
    ) = _condition_runtime_fixture()
    repository_root = Path(__file__).resolve().parents[1]

    def collect(arm_id: str, *, split: str = "calibration", visible=None):
        context = build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
            budget_minutes=30,
            policy_arm_id=arm_id,
        )
        source = run_condition_calibration_collection(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            collection_split=split,
            adaptive_policy_context=context,
            condition_plan=plan,
            condition_development_graph=development,
            condition_frozen_model=model,
            policy_visible_question_trajectory=visible,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            item_risk_scoring_receipt=item_risk_receipt,
            generated_at=datetime(2026, 8, 28, 13, tzinfo=UTC),
        )
        assert source.policy_visible_question_trajectory is not None
        return source

    return {
        "collect": collect,
        "pipeline_root": repository_root,
        "calibration": [collect("policy-arm-a"), collect("policy-arm-b")],
    }


def test_builder_is_canonical_and_two_pass_sources_freeze_one_exact_roster(
    collected_sources,
    monkeypatch,
) -> None:
    sources = collected_sources["calibration"]
    repository_root = collected_sources["pipeline_root"]
    forward = build_condition_calibration_question_trajectory(
        sources,
        pipeline_root=repository_root,
    )
    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        lambda source, *, pipeline_root=None: source,
    )
    reverse = build_condition_calibration_question_trajectory(
        list(reversed(sources)),
        pipeline_root=repository_root,
    )
    assert forward == reverse
    assert [arm.base_arm.policy_arm_id for arm in forward.arms] == [
        "policy-arm-a",
        "policy-arm-b",
    ]
    assert forward.base_visible.arms == [arm.base_arm for arm in forward.arms]

    second_pass = [
        collected_sources["collect"](
            source.policy_arm_id,
            visible=forward,
        )
        for source in sources
    ]
    assert all(
        source.policy_visible_question_trajectory == forward for source in second_pass
    )
    roster = freeze_condition_calibration_collection_source_roster_v1(second_pass)
    assert len(roster.collection_sources) == 2
    assert [anchor.policy_arm_id for anchor in roster.source_anchors] == [
        "policy-arm-a",
        "policy-arm-b",
    ]
    assert {anchor.visible_trajectory_sha256 for anchor in roster.source_anchors} == {
        forward.trajectory_sha256
    }


def test_builder_replays_each_source_and_rejects_arm_overlap(
    collected_sources,
    monkeypatch,
) -> None:
    sources = collected_sources["calibration"]
    calls: list[str] = []

    def replay(source, *, pipeline_root=None):
        assert pipeline_root == collected_sources["pipeline_root"]
        calls.append(source.policy_arm_id)
        return source

    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        replay,
    )
    build_condition_calibration_question_trajectory(
        list(reversed(sources)),
        pipeline_root=collected_sources["pipeline_root"],
    )
    assert calls == ["policy-arm-b", "policy-arm-a"]

    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_policy_arm_overlap",
    ):
        build_condition_calibration_question_trajectory(
            [sources[0], sources[0]],
            pipeline_root=collected_sources["pipeline_root"],
        )

    duplicate_context = sources[1].model_copy(
        update={
            "adaptive_policy_context": sources[1].adaptive_policy_context.model_copy(
                update={
                    "policy_context_sha256": (
                        sources[0].adaptive_policy_context.policy_context_sha256
                    )
                }
            )
        }
    )
    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        lambda source, *, pipeline_root=None: (
            duplicate_context if source.policy_arm_id == "policy-arm-b" else source
        ),
    )
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_policy_context_overlap",
    ):
        build_condition_calibration_question_trajectory(
            sources,
            pipeline_root=collected_sources["pipeline_root"],
        )


def _mutate_replayed_source(source, case: str):
    if case == "question":
        return source.model_copy(
            update={
                "condition_calibration_projection": (
                    source.condition_calibration_projection.model_copy(
                        update={"question_id": "different-question"}
                    )
                )
            }
        )
    if case == "split":
        return source.model_copy(update={"collection_split": "development"})
    if case in {"population", "domain"}:
        manifest = dict(source.claim_manifest)
        manifest[f"{case}_id" if case == "population" else "domain"] = f"different-{case}"
        return source.model_copy(update={"claim_manifest": manifest})
    if case == "corpus":
        return source.model_copy(
            update={
                "complete_corpus_identity": source.complete_corpus_identity.model_copy(
                    update={"corpus_id": "different-corpus"}
                )
            }
        )
    if case == "source_graph":
        return source.model_copy(update={"source_evidence_graph_sha256": "f" * 64})
    if case == "target_semantics":
        return source.model_copy(
            update={
                "condition_target_semantics": source.condition_target_semantics.model_copy(
                    update={"target_semantics_sha256": "f" * 64}
                )
            }
        )
    if case == "independence_semantics":
        return source.model_copy(
            update={
                "condition_independence_identity": (
                    source.condition_independence_identity.model_copy(
                        update={"identity_sha256": "f" * 64}
                    )
                )
            }
        )
    if case == "pipeline":
        return source.model_copy(
            update={
                "pipeline_verification": source.pipeline_verification.model_copy(
                    update={"computed_pipeline_sha256": "f" * 64}
                )
            }
        )
    if case == "arm_binding":
        return source.model_copy(
            update={
                "adaptive_policy_context": source.adaptive_policy_context.model_copy(
                    update={"policy_context_sha256": "f" * 64}
                )
            }
        )
    raise AssertionError(f"unknown case: {case}")


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("question", "question_mismatch"),
        ("split", "split_mismatch"),
        ("population", "population_mismatch"),
        ("domain", "domain_mismatch"),
        ("corpus", "corpus_mismatch"),
        ("source_graph", "source_graph_mismatch"),
        ("target_semantics", "target_semantics_mismatch"),
        ("independence_semantics", "independence_semantics_mismatch"),
        ("pipeline", "pipeline_mismatch"),
        ("arm_binding", "source_arm_binding_mismatch"),
    ],
)
def test_builder_rejects_every_cross_arm_family_mismatch(
    collected_sources,
    monkeypatch,
    case: str,
    error: str,
) -> None:
    sources = collected_sources["calibration"]

    def replay(source, *, pipeline_root=None):
        return (
            _mutate_replayed_source(source, case)
            if source.policy_arm_id == "policy-arm-b"
            else source
        )

    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        replay,
    )
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match=f"condition_trajectory_{error}",
    ):
        build_condition_calibration_question_trajectory(
            sources,
            pipeline_root=collected_sources["pipeline_root"],
        )


def test_reader_rejects_tamper_and_outcome_bearing_contracts(
    collected_sources,
    tmp_path: Path,
) -> None:
    source = collected_sources["calibration"][0]
    tampered = source.model_dump(mode="json")
    tampered["collection_source_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    atomic_write_json(tampered_path, tampered)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_source_contract_invalid",
    ):
        read_condition_calibration_collection_source(tampered_path)

    for forbidden_key in (
        "adaptive_calibration_bundle_v2",
        "calibration_gate_result",
        "condition_confirmation_assessment",
        "reference_verdict",
        "release_qualification_proof",
        "terminal_gate_result",
    ):
        outcome_bearing = source.model_dump(mode="json")
        outcome_bearing[forbidden_key] = {
            "status": "confirmed",
            "artifact_sha256": "1" * 64,
        }
        outcome_path = tmp_path / f"outcome-bearing-{forbidden_key}.json"
        atomic_write_json(outcome_path, outcome_bearing)
        with pytest.raises(
            ConditionTrajectoryBuilderError,
            match="condition_trajectory_outcome_bearing_input_forbidden",
        ):
            read_condition_calibration_collection_source(outcome_path)


def test_cli_atomic_no_force_and_symlink_guards(
    collected_sources,
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    sources = collected_sources["calibration"]
    source_paths: list[Path] = []
    for source in sources:
        path = tmp_path / f"{source.policy_arm_id}.json"
        atomic_write_json(path, source)
        source_paths.append(path)
    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        lambda source, *, pipeline_root=None: source,
    )
    output = tmp_path / "visible-v2.json"
    arguments = [
        "--source",
        str(source_paths[1]),
        "--source",
        str(source_paths[0]),
        "--pipeline-root",
        str(collected_sources["pipeline_root"]),
        "--output",
        str(output),
    ]
    assert cli.main(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    trajectory = PolicyVisibleQuestionTrajectoryV2.model_validate_json(output.read_text())
    assert receipt["trajectory_sha256"] == trajectory.trajectory_sha256
    assert receipt["condition_assessments_opened"] is False
    assert receipt["gate_outcomes_opened"] is False
    assert receipt["reference_labels_opened"] is False
    assert receipt["calibration_bundles_opened"] is False
    assert receipt["receipt_sha256"] == hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )

    with monkeypatch.context() as preflight_patch:
        preflight_patch.setattr(
            cli,
            "read_condition_calibration_collection_source",
            lambda _path: pytest.fail("existing output must fail before any source opens"),
        )
        with pytest.raises(OutputExistsError, match="lineage_output_exists"):
            cli.main(arguments)
    monkeypatch.setattr(
        "literature_multiverse.condition_trajectory_builder."
        "validate_condition_calibration_collection_source_external_replay",
        lambda source, *, pipeline_root=None: source,
    )
    assert cli.main([*arguments, "--force"]) == 0
    capsys.readouterr()

    symlink_input = tmp_path / "source-link.json"
    symlink_input.symlink_to(source_paths[0].name)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_source_symlink_forbidden",
    ):
        cli.main(
            [
                "--source",
                str(symlink_input),
                "--source",
                str(source_paths[1]),
                "--output",
                str(tmp_path / "from-link.json"),
            ]
        )

    symlink_target = tmp_path / "unrelated.json"
    atomic_write_json(symlink_target, {"unrelated": True})
    symlink_output = tmp_path / "output-link.json"
    symlink_output.symlink_to(symlink_target.name)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_output_symlink_forbidden",
    ):
        cli.main(
            [
                "--source",
                str(source_paths[0]),
                "--source",
                str(source_paths[1]),
                "--output",
                str(symlink_output),
                "--force",
            ]
        )
    assert json.loads(symlink_target.read_text()) == {"unrelated": True}

    real_output_parent = tmp_path / "real-output-parent"
    real_output_parent.mkdir()
    linked_output_parent = tmp_path / "linked-output-parent"
    linked_output_parent.symlink_to(real_output_parent.name, target_is_directory=True)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_output_symlink_forbidden",
    ):
        cli.main(
            [
                "--source",
                str(source_paths[0]),
                "--source",
                str(source_paths[1]),
                "--output",
                str(linked_output_parent / "visible.json"),
            ]
        )

    pipeline_root_link = tmp_path / "pipeline-root-link"
    pipeline_root_link.symlink_to(
        collected_sources["pipeline_root"],
        target_is_directory=True,
    )
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_pipeline_root_symlink_forbidden",
    ):
        cli.main(
            [
                "--source",
                str(source_paths[0]),
                "--source",
                str(source_paths[1]),
                "--pipeline-root",
                str(pipeline_root_link),
                "--output",
                str(tmp_path / "from-pipeline-link.json"),
            ]
        )


def test_cli_rejects_source_and_output_hardlink_aliases(
    collected_sources,
    tmp_path: Path,
) -> None:
    sources = collected_sources["calibration"]
    source_a = tmp_path / "source-a.json"
    source_b = tmp_path / "source-b.json"
    atomic_write_json(source_a, sources[0])
    atomic_write_json(source_b, sources[1])

    duplicate_source = tmp_path / "duplicate-source.json"
    os.link(source_a, duplicate_source)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_source_file_overlap",
    ):
        cli.main(
            [
                "--source",
                str(source_a),
                "--source",
                str(duplicate_source),
                "--output",
                str(tmp_path / "unused.json"),
            ]
        )

    output_alias = tmp_path / "output-alias.json"
    os.link(source_a, output_alias)
    with pytest.raises(
        ConditionTrajectoryBuilderError,
        match="condition_trajectory_output_must_not_alias_source",
    ):
        cli.main(
            [
                "--source",
                str(source_a),
                "--source",
                str(source_b),
                "--output",
                str(output_alias),
                "--force",
            ]
        )
