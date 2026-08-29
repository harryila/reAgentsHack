from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.decisive_claim_evaluation_v1 import (
    LABEL_ENVELOPE_BYTES,
    PRIMARY_POLICY_ARM_ID,
    DecisiveClaimEvaluationResultV1,
    DecisiveClaimEvaluationV1Error,
    DecisiveEvaluationConfigV1,
    DecisiveEvaluationReadinessV1,
    DecisivePolicyFreezeV1,
    DecisiveSplitManifestV1,
    EnvelopeNonceOrigin,
    EvaluationLabelManifestV1,
    FitStageReceiptV1,
    OpenedEvaluationLabelV1,
    StudySplit,
    TrajectoryBundleV1,
    _score_question_v1,
    assess_decisive_evaluation_readiness_v1,
    build_decisive_mechanics_fixture_v1,
    freeze_decisive_evaluation_config_v1,
    freeze_decisive_policy_trajectories_v1,
    freeze_decisive_split_manifest_v1,
    freeze_evaluation_reference_envelope_v1,
    freeze_question_identity_v1,
    freeze_question_trajectory_v1,
    freeze_trajectory_bundle_v1,
    parse_evaluation_reference_envelope_v1,
    required_policy_roster_v1,
    score_decisive_claim_evaluation_v1,
    validate_decisive_claim_evaluation_result_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.question_evaluation import (
    BenchmarkEvidenceKind,
    ReferenceClaimVerdictValue,
    freeze_reference_claim_verdict,
)

ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 8, 29, tzinfo=UTC)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class FixtureArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.labels = root / "evaluation-labels"
        self.config = DecisiveEvaluationConfigV1.model_validate(_object(root / "config.json"))
        self.split = DecisiveSplitManifestV1.model_validate(_object(root / "split-manifest.json"))
        self.development = FitStageReceiptV1.model_validate(
            _object(root / "development-receipt.json")
        )
        self.calibration = FitStageReceiptV1.model_validate(
            _object(root / "calibration-receipt.json")
        )
        self.trajectories = TrajectoryBundleV1.model_validate(
            _object(root / "trajectory-bundle.json")
        )
        self.label_manifest = EvaluationLabelManifestV1.model_validate(
            _object(root / "label-manifest.json")
        )
        self.custody = DecisiveEvaluationReadinessV1.model_validate(
            _object(root / "readiness.json")
        )
        self.freeze = DecisivePolicyFreezeV1.model_validate(_object(root / "policy-freeze.json"))
        self.result = DecisiveClaimEvaluationResultV1.model_validate(
            _object(root / "evaluation-result.json")
        )


@pytest.fixture(scope="module")
def mechanics(tmp_path_factory: pytest.TempPathFactory) -> FixtureArtifacts:
    root = tmp_path_factory.mktemp("decisive-mechanics") / "fixture"
    config = freeze_decisive_evaluation_config_v1(
        budgets_minutes_per_question=(6.0,),
        bootstrap_draws=100,
    )
    build_decisive_mechanics_fixture_v1(
        output_root=root,
        repository_root=ROOT,
        config=config,
    )
    return FixtureArtifacts(root)


def _copy_fixture(mechanics: FixtureArtifacts, tmp_path: Path) -> FixtureArtifacts:
    target = tmp_path / "fixture-copy"
    shutil.copytree(mechanics.root, target)
    return FixtureArtifacts(target)


def _seal_all(fixture: FixtureArtifacts) -> None:
    for entry in fixture.label_manifest.entries:
        os.chmod(fixture.labels / entry.relative_path, 0)


def _unseal_all(fixture: FixtureArtifacts) -> None:
    for entry in fixture.label_manifest.entries:
        os.chmod(fixture.labels / entry.relative_path, 0o400)


def _fresh_custody(
    fixture: FixtureArtifacts,
    *,
    trajectory_bundle: TrajectoryBundleV1 | None = None,
) -> DecisiveEvaluationReadinessV1:
    return assess_decisive_evaluation_readiness_v1(
        config=fixture.config,
        repository_root=ROOT,
        assessed_at=START + timedelta(hours=1),
        split_manifest=fixture.split,
        development_receipt=fixture.development,
        calibration_receipt=fixture.calibration,
        trajectory_bundle=trajectory_bundle or fixture.trajectories,
        label_manifest=fixture.label_manifest,
        label_root=fixture.labels,
    )


def test_exact_policy_roster_and_prespecified_primary() -> None:
    roster = required_policy_roster_v1()
    assert len(roster) == 20
    assert len({row.arm_id for row in roster}) == 20
    assert PRIMARY_POLICY_ARM_ID in {row.arm_id for row in roster}
    assert {row.adaptation.value for row in roster} >= {
        "static_baseline_scores",
        "adaptive_state_scores",
    }


def test_legacy_blocked_readiness_remains_parseable_but_not_real_candidate() -> None:
    readiness = DecisiveEvaluationReadinessV1.model_validate(
        _object(
            ROOT
            / "artifacts"
            / "diagnostics"
            / "decisive-claim-evaluation-v1-real-readiness-blocked.json"
        )
    )
    assert readiness.status == "blocked"
    assert readiness.compilation_replay_proof is None
    assert readiness.real_scored_run_candidate is False


def test_fixture_is_explicitly_non_empirical(mechanics: FixtureArtifacts) -> None:
    result = mechanics.result
    assert result.scientific_claim_eligible is False
    assert result.released_claim_error_claim_authority is False
    assert result.human_efficiency_claim_authority is False
    assert result.same_realized_cost_claim_authority is False
    assert result.typed_replayed_calibration_artifact_present is False
    assert result.claim_release_authority is False
    assert result.expert_labels_fabricated is False
    assert result.empirical_scope == "planted_simulation_mechanics_only_non_empirical"
    assert all(
        row.reference_verdict.source.value == "planted_simulation" for row in result.opened_labels
    )


def test_score_refuses_self_declared_real_bundle_without_compiler_replay(
    mechanics: FixtureArtifacts,
) -> None:
    forged_bundle = mechanics.freeze.trajectory_bundle.model_copy(
        update={"evidence_kind": BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED}
    )
    forged_freeze = mechanics.freeze.model_copy(update={"trajectory_bundle": forged_bundle})
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="real_compilation_replay_inputs_required",
    ):
        score_decisive_claim_evaluation_v1(
            frozen=forged_freeze,
            custody=mechanics.custody,
            repository_root=ROOT,
            label_root=mechanics.labels,
            scored_at=START + timedelta(hours=3),
        )


def test_adaptive_and_static_paths_diverge(mechanics: FixtureArtifacts) -> None:
    populations = {
        (row.policy_arm.arm_id, row.budget_minutes): row
        for row in mechanics.freeze.policy_populations
    }
    adaptive = populations[(PRIMARY_POLICY_ARM_ID, 6.0)]
    static = populations[("risk_x_influence_per_cost_static", 6.0)]
    assert adaptive.questions[0].selected_item_ids != static.questions[0].selected_item_ids
    scored = {
        (row.policy_arm.arm_id, row.budget_minutes): row
        for row in mechanics.result.scored_policy_populations
    }
    assert scored[(PRIMARY_POLICY_ARM_ID, 6.0)].metrics["released_claim_error"] == 0.0
    assert scored[("risk_x_influence_per_cost_static", 6.0)].metrics["released_claim_error"] == 1.0


def test_every_completed_action_has_exact_rerun_lineage(
    mechanics: FixtureArtifacts,
) -> None:
    for population in mechanics.freeze.policy_populations:
        for question in population.questions:
            for step in question.steps:
                if step.action_outcome == "completed_and_rerun":
                    assert step.post_replay_sha256 is not None
                    assert step.post_audit_sequence == [
                        *step.pre_audit_sequence,
                        step.selected_item_id,
                    ]
                else:
                    assert step.post_replay_sha256 is None
            if question.budget_minutes is not None:
                assert question.total_realized_minutes <= question.budget_minutes + 1e-9


def test_paired_comparisons_use_identical_questions_and_deadlines(
    mechanics: FixtureArtifacts,
) -> None:
    assert mechanics.result.paired_policy_comparisons
    for row in mechanics.result.paired_policy_comparisons:
        assert row.question_ids == mechanics.freeze.evaluation_question_ids
        assert row.identical_question_population is True
        assert row.identical_budget_cap_and_deadline is True
        assert row.realized_cost_matched is False
        assert row.same_realized_cost_claim_authority is False
        assert row.budget_minutes_per_question == 6.0
        uncertainty = row.paired_question_clustered_uncertainty
        intervals = uncertainty["intervals"]
        assert uncertainty["small_sample_authority"] is False
        assert all("not causal" in value["semantics"] for value in intervals.values())


def test_portable_freeze_and_result_exclude_local_custody(
    mechanics: FixtureArtifacts,
) -> None:
    freeze_payload = mechanics.freeze.model_dump(mode="json")
    result_payload = mechanics.result.model_dump(mode="json")
    rendered_freeze = json.dumps(freeze_payload, sort_keys=True)
    rendered_result = json.dumps(result_payload, sort_keys=True)
    for forbidden in ('"device"', '"inode"', '"mode_permissions"', '"readiness"'):
        assert forbidden not in rendered_freeze
        assert forbidden not in rendered_result
    assert mechanics.custody.custody_semantics.startswith("machine_local_nonportable")
    assert mechanics.freeze.freeze_portability.startswith("portable_semantic_artifact")
    assert "score_pipeline_frozen_before_benchmark_labels" not in rendered_freeze
    assert "policy_and_thresholds_frozen_after_permitted_fit_stages" in rendered_freeze


def test_portable_hashes_match_across_distinct_custody_inodes(
    mechanics: FixtureArtifacts, tmp_path: Path
) -> None:
    second_root = tmp_path / "second-fixture"
    build_decisive_mechanics_fixture_v1(
        output_root=second_root,
        repository_root=ROOT,
        config=mechanics.config,
    )
    second = FixtureArtifacts(second_root)
    assert mechanics.freeze.freeze_sha256 == second.freeze.freeze_sha256
    assert mechanics.result.result_sha256 == second.result.result_sha256
    assert mechanics.custody.readiness_sha256 != second.custody.readiness_sha256


def test_external_result_replay(mechanics: FixtureArtifacts) -> None:
    assert (
        validate_decisive_claim_evaluation_result_v1(
            result=mechanics.result,
            custody=mechanics.custody,
            repository_root=ROOT,
            label_root=mechanics.labels,
        )
        == mechanics.result
    )


def test_readiness_does_not_open_or_hash_poisoned_evaluation_labels(
    mechanics: FixtureArtifacts, tmp_path: Path
) -> None:
    fixture = _copy_fixture(mechanics, tmp_path)
    first = fixture.label_manifest.entries[0]
    target = fixture.labels / first.relative_path
    original_size = target.stat().st_size
    os.chmod(target, 0o600)
    target.write_bytes(b"X" * original_size)
    _seal_all(fixture)
    custody = _fresh_custody(fixture)
    assert custody.status == "ready"
    assert custody.evaluation_label_contents_opened is False
    _unseal_all(fixture)
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="label_file_hash_mismatch",
    ):
        score_decisive_claim_evaluation_v1(
            frozen=mechanics.freeze,
            custody=custody,
            repository_root=ROOT,
            label_root=fixture.labels,
            scored_at=START + timedelta(hours=2),
        )


def test_hardlinked_label_is_rejected_before_freeze(
    mechanics: FixtureArtifacts, tmp_path: Path
) -> None:
    fixture = _copy_fixture(mechanics, tmp_path)
    first, second = fixture.label_manifest.entries[:2]
    first_path = fixture.labels / first.relative_path
    second_path = fixture.labels / second.relative_path
    first_path.unlink()
    os.link(second_path, first_path)
    _seal_all(fixture)
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="label_file_hardlinked",
    ):
        _fresh_custody(fixture)


def test_execute_only_mode_is_not_a_valid_seal(mechanics: FixtureArtifacts, tmp_path: Path) -> None:
    fixture = _copy_fixture(mechanics, tmp_path)
    _seal_all(fixture)
    first = fixture.label_manifest.entries[0]
    os.chmod(fixture.labels / first.relative_path, 0o111)
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="evaluation_label_not_sealed",
    ):
        _fresh_custody(fixture)


def test_envelope_hash_is_not_enumerable_from_five_verdicts(
    mechanics: FixtureArtifacts,
) -> None:
    entry = mechanics.label_manifest.entries[0]
    content = (mechanics.labels / entry.relative_path).read_bytes()
    actual = parse_evaluation_reference_envelope_v1(content)
    guessed_hashes: set[str] = set()
    for verdict_value in ReferenceClaimVerdictValue:
        guessed_verdict = freeze_reference_claim_verdict(
            question_id=actual.reference_verdict.question_id,
            claim_id=actual.reference_verdict.claim_id,
            verdict=verdict_value,
            source=actual.reference_verdict.source,
            adjudicator_count=actual.reference_verdict.adjudicator_count,
            protocol_sha256=actual.reference_verdict.protocol_sha256,
            artifact_sha256=actual.reference_verdict.artifact_sha256,
        )
        guessed = freeze_evaluation_reference_envelope_v1(
            reference_verdict=guessed_verdict,
            custodian_nonce_hex="0" * 32,
            nonce_origin=EnvelopeNonceOrigin.PLANTED_SIMULATION_FIXTURE,
            reference_condition_set_artifact_sha256=(
                "f" * 64
                if verdict_value is ReferenceClaimVerdictValue.CONDITION_DEPENDENT
                else None
            ),
        )
        guessed_hashes.add(hashlib.sha256(guessed).hexdigest())
    assert entry.expected_envelope_sha256 not in guessed_hashes
    manifest_text = json.dumps(mechanics.label_manifest.model_dump(mode="json"))
    assert "custodian_nonce" not in manifest_text
    assert mechanics.label_manifest.nonce_values_present is False


def test_all_verdict_envelopes_have_identical_fixed_size(
    mechanics: FixtureArtifacts,
) -> None:
    verdicts = set()
    sizes = set()
    for entry in mechanics.label_manifest.entries:
        content = (mechanics.labels / entry.relative_path).read_bytes()
        sizes.add(len(content))
        verdicts.add(parse_evaluation_reference_envelope_v1(content).reference_verdict.verdict)
    assert len(verdicts) >= 2
    assert sizes == {LABEL_ENVELOPE_BYTES}
    assert {row.fixed_envelope_bytes for row in mechanics.label_manifest.entries} == {
        LABEL_ENVELOPE_BYTES
    }


def test_label_filename_cannot_encode_verdict_or_arbitrary_metadata(
    mechanics: FixtureArtifacts,
) -> None:
    entry = mechanics.label_manifest.entries[0]
    payload = entry.model_dump(mode="json")
    payload["relative_path"] = "supported.json"
    payload["entry_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "entry_sha256"}
    )
    with pytest.raises(
        ValidationError,
        match="label_path_not_public_question_derivation",
    ):
        type(entry).model_validate(payload)


def test_post_custody_path_swap_is_rejected(mechanics: FixtureArtifacts, tmp_path: Path) -> None:
    fixture = _copy_fixture(mechanics, tmp_path)
    _seal_all(fixture)
    custody = _fresh_custody(fixture)
    first = fixture.label_manifest.entries[0]
    target = fixture.labels / first.relative_path
    os.chmod(target, 0o400)
    content = target.read_bytes()
    original_mtime = target.stat().st_mtime_ns
    target.unlink()
    target.write_bytes(content)
    os.utime(target, ns=(original_mtime, original_mtime))
    _unseal_all(fixture)
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="label_file_identity_changed",
    ):
        score_decisive_claim_evaluation_v1(
            frozen=mechanics.freeze,
            custody=custody,
            repository_root=ROOT,
            label_root=fixture.labels,
            scored_at=START + timedelta(hours=2),
        )


def test_missing_post_audit_rerun_fails_closed(mechanics: FixtureArtifacts, tmp_path: Path) -> None:
    fixture = _copy_fixture(mechanics, tmp_path)
    _seal_all(fixture)
    first = fixture.trajectories.trajectories[0]
    b_item = next(row.item_id for row in first.audit_events if row.item_id.endswith("-b"))
    c_item = next(row.item_id for row in first.audit_events if row.item_id.endswith("-c"))
    reduced_states = [row for row in first.replay_states if row.audit_sequence != [b_item, c_item]]
    reduced_first = freeze_question_trajectory_v1(
        question_identity=first.question_identity,
        evidence_kind=first.evidence_kind,
        policy_input_provenance=first.policy_input_provenance,
        audit_events=first.audit_events,
        replay_states=reduced_states,
        condition_set_artifact_sha256_by_replay_sha256={
            row.replay_sha256: row.condition_set_artifact_sha256
            for row in first.condition_set_bindings
            if row.replay_sha256 in {state.replay_sha256 for state in reduced_states}
        },
    )
    modified_bundle = freeze_trajectory_bundle_v1(
        split_manifest=fixture.split,
        evidence_kind=fixture.trajectories.evidence_kind,
        trajectories=[reduced_first, *fixture.trajectories.trajectories[1:]],
    )
    custody = _fresh_custody(fixture, trajectory_bundle=modified_bundle)
    with pytest.raises(
        DecisiveClaimEvaluationV1Error,
        match="post_audit_rerun_missing",
    ):
        freeze_decisive_policy_trajectories_v1(
            config=fixture.config,
            readiness=custody,
            split_manifest=fixture.split,
            development_receipt=fixture.development,
            calibration_receipt=fixture.calibration,
            trajectory_bundle=modified_bundle,
            label_manifest=fixture.label_manifest,
            label_root=fixture.labels,
            repository_root=ROOT,
            frozen_at=START + timedelta(hours=2),
        )


def test_cross_split_paper_overlap_is_rejected() -> None:
    rows = [
        freeze_question_identity_v1(
            split=split,
            question_id=f"q-{split.value}",
            claim_id=f"claim-{split.value}",
            domain="domain",
            population_id=f"population-{split.value}",
            pipeline_sha256="a" * 64,
            corpus_sha256=character * 64,
            paper_ids=["shared-paper"],
            cohort_ids=[f"cohort-{split.value}"],
        )
        for split, character in (
            (StudySplit.DEVELOPMENT, "b"),
            (StudySplit.CALIBRATION, "c"),
            (StudySplit.EVALUATION, "d"),
        )
    ]
    with pytest.raises(ValidationError, match="cross_split_overlap:paper"):
        freeze_decisive_split_manifest_v1(
            identities=rows,
            split_salt_sha256="e" * 64,
        )


def test_paper_and_cohort_may_repeat_within_one_split() -> None:
    rows = []
    for split, count in (
        (StudySplit.DEVELOPMENT, 2),
        (StudySplit.CALIBRATION, 1),
        (StudySplit.EVALUATION, 1),
    ):
        for index in range(count):
            rows.append(
                freeze_question_identity_v1(
                    split=split,
                    question_id=f"q-{split.value}-{index}",
                    claim_id=f"claim-{split.value}-{index}",
                    domain="domain",
                    population_id=f"population-{split.value}-{index}",
                    pipeline_sha256="a" * 64,
                    corpus_sha256=f"{index + 1}" * 64,
                    paper_ids=[f"paper-{split.value}"],
                    cohort_ids=[f"cohort-{split.value}"],
                )
            )
    manifest = freeze_decisive_split_manifest_v1(
        identities=rows,
        split_salt_sha256="e" * 64,
    )
    assert len(manifest.development_question_ids) == 2


def test_condition_dependent_match_requires_exact_condition_set(
    mechanics: FixtureArtifacts,
) -> None:
    label = next(
        row
        for row in mechanics.result.opened_labels
        if row.reference_condition_set_artifact_sha256 is not None
    )
    population = next(
        row
        for row in mechanics.freeze.policy_populations
        if row.policy_arm.arm_id == PRIMARY_POLICY_ARM_ID and row.budget_minutes == 6.0
    )
    frozen_question = next(
        row for row in population.questions if row.question_id == label.question_id
    )
    assert frozen_question.claim_classification == "condition_dependent"
    correct = _score_question_v1(frozen_question, label)
    assert correct.classification_exact_match is True
    assert correct.condition_set_exact_match is True
    assert correct.decision_exact_match is True

    wrong_payload = label.model_dump(mode="json")
    wrong_payload["reference_condition_set_artifact_sha256"] = "f" * 64
    assert wrong_payload["reference_condition_set_artifact_sha256"] != (
        frozen_question.condition_set_artifact_sha256
    )
    wrong_payload["opened_label_sha256"] = hash_canonical(
        {key: value for key, value in wrong_payload.items() if key != "opened_label_sha256"}
    )
    wrong_label = OpenedEvaluationLabelV1.model_validate(wrong_payload)
    wrong = _score_question_v1(frozen_question, wrong_label)
    assert wrong.classification_exact_match is True
    assert wrong.condition_set_exact_match is False
    assert wrong.decision_exact_match is False
    assert wrong.released_claim_error is True


def test_configuration_cannot_shrink_evaluation_below_twenty() -> None:
    with pytest.raises(ValidationError):
        freeze_decisive_evaluation_config_v1(
            minimum_evaluation_questions=19,
        )


def test_result_authority_cannot_be_escalated(mechanics: FixtureArtifacts) -> None:
    payload = mechanics.result.model_dump(mode="json")
    payload["scientific_claim_eligible"] = True
    payload["released_claim_error_claim_authority"] = True
    payload["human_efficiency_claim_authority"] = True
    payload["result_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    with pytest.raises(ValidationError, match="authority_escalation"):
        DecisiveClaimEvaluationResultV1.model_validate(payload)


def test_policy_population_cannot_drop_a_question(mechanics: FixtureArtifacts) -> None:
    payload = mechanics.freeze.model_dump(mode="json")
    population = deepcopy(payload["policy_populations"][0])
    population["questions"] = population["questions"][:-1]
    population["question_ids"] = population["question_ids"][:-1]
    population["population_sha256"] = hash_canonical(
        {key: value for key, value in population.items() if key != "population_sha256"}
    )
    payload["policy_populations"][0] = population
    payload["freeze_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "freeze_sha256"}
    )
    with pytest.raises(ValidationError, match="policy_population_unequal"):
        DecisivePolicyFreezeV1.model_validate(payload)
