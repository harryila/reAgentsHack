from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.confirm_condition_dependence as cli
from tests.test_condition_confirmation import (
    PIPELINE_SHA256,
    ConfirmationCase,
)
from tests.test_condition_confirmation import (
    confirmation_case as core_confirmation_case_fixture,
)

from literature_multiverse.condition_confirmation import (
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationError,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationMaterializationReceiptV1,
    ConditionConfirmationPlanV1,
    LabelFreeGraphRosterV1,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
)


@pytest.fixture(scope="module", name="confirmation_case")
def _confirmation_case() -> ConfirmationCase:
    return core_confirmation_case_fixture.__wrapped__()


def _write_inputs(tmp_path: Path, case: ConfirmationCase) -> dict[str, Path]:
    paths = {
        "target": tmp_path / "target.json",
        "config": tmp_path / "config.json",
        "roster": tmp_path / "label-free-roster.json",
        "materialization_receipt": tmp_path / "materialization-receipt.json",
        "plan": tmp_path / "plan.json",
        "development": tmp_path / "development-graph.private.json",
        "confirmation": tmp_path / "confirmation-graph.private.json",
        "model": tmp_path / "frozen-model.json",
        "full": tmp_path / "full-graph.private.json",
        "assessment": tmp_path / "assessment.json",
    }
    atomic_write_json(paths["target"], case.plan.target)
    atomic_write_json(paths["config"], case.plan.config)
    atomic_write_json(paths["roster"], case.roster)
    atomic_write_json(paths["materialization_receipt"], case.materialization_receipt)
    atomic_write_json(paths["development"], case.development_graph)
    atomic_write_json(paths["confirmation"], case.confirmation_graph)
    atomic_write_json(paths["full"], case.graph)
    return paths


def _prepare_args(case: ConfirmationCase, paths: dict[str, Path]) -> list[str]:
    return [
        "prepare",
        "--target",
        str(paths["target"]),
        "--expected-target-sha256",
        case.plan.target_sha256,
        "--config",
        str(paths["config"]),
        "--expected-config-sha256",
        case.plan.config_sha256,
        "--roster",
        str(paths["roster"]),
        "--expected-roster-sha256",
        case.plan.roster_sha256,
        "--materialization-receipt",
        str(paths["materialization_receipt"]),
        "--expected-materialization-receipt-sha256",
        case.materialization_receipt.receipt_sha256,
        "--pipeline-sha256",
        PIPELINE_SHA256,
        "--external-freeze-anchor",
        case.plan.external_freeze_anchor,
        "--output",
        str(paths["plan"]),
    ]


def _fit_args(case: ConfirmationCase, paths: dict[str, Path]) -> list[str]:
    return [
        "fit",
        "--plan",
        str(paths["plan"]),
        "--expected-plan-sha256",
        case.plan.plan_sha256,
        "--current-pipeline-sha256",
        PIPELINE_SHA256,
        "--development-graph",
        str(paths["development"]),
        "--output",
        str(paths["model"]),
    ]


def _materialize_args(
    case: ConfirmationCase,
    paths: dict[str, Path],
    outputs: dict[str, Path],
) -> list[str]:
    return [
        "materialize",
        "--target",
        str(paths["target"]),
        "--expected-target-sha256",
        case.plan.target_sha256,
        "--full-graph",
        str(paths["full"]),
        "--expected-full-graph-sha256",
        case.plan.full_graph_sha256,
        "--roster-output",
        str(outputs["roster"]),
        "--development-graph-output",
        str(outputs["development"]),
        "--confirmation-graph-output",
        str(outputs["confirmation"]),
        "--receipt-output",
        str(outputs["receipt"]),
    ]


def _confirmation_args(
    command: str,
    case: ConfirmationCase,
    paths: dict[str, Path],
) -> list[str]:
    return [
        command,
        "--plan",
        str(paths["plan"]),
        "--expected-plan-sha256",
        case.plan.plan_sha256,
        "--model",
        str(paths["model"]),
        "--expected-model-sha256",
        case.model.model_sha256,
        "--current-pipeline-sha256",
        PIPELINE_SHA256,
        "--full-graph",
        str(paths["full"]),
    ]


def _assert_self_hashed_receipt(receipt: dict[str, object]) -> None:
    assert receipt["receipt_sha256"] == hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_custodian_cli_materializes_and_exactly_replays_content_silent_outputs(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)
    outputs = {
        "roster": tmp_path / "custodian" / "roster.json",
        "development": tmp_path / "custodian" / "development.private.json",
        "confirmation": tmp_path / "custodian" / "confirmation.private.json",
        "receipt": tmp_path / "custodian" / "receipt.json",
    }
    opened: list[Path] = []
    original_reader = cli._read_regular_bytes

    def tracked_reader(path: Path, *, purpose: str) -> tuple[bytes, str]:
        opened.append(path)
        return original_reader(path, purpose=purpose)

    monkeypatch.setattr(cli, "_read_regular_bytes", tracked_reader)
    assert cli.main(_materialize_args(case, paths, outputs)) == 0
    materialize_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(materialize_receipt)
    assert opened == [paths["target"], paths["full"]]
    assert materialize_receipt["full_graph_outcomes_opened_by_custodian"] is True
    roster = LabelFreeGraphRosterV1.model_validate_json(
        outputs["roster"].read_text(encoding="utf-8")
    )
    typed_receipt = ConditionConfirmationMaterializationReceiptV1.model_validate_json(
        outputs["receipt"].read_text(encoding="utf-8")
    )
    assert roster == case.roster
    assert typed_receipt == case.materialization_receipt
    assert typed_receipt.effect_outcome_uncertainty_values_embedded is False

    opened.clear()
    validate_args = [
        "validate-materialization",
        "--target",
        str(paths["target"]),
        "--expected-target-sha256",
        case.plan.target_sha256,
        "--full-graph",
        str(paths["full"]),
        "--expected-full-graph-sha256",
        case.plan.full_graph_sha256,
        "--roster",
        str(outputs["roster"]),
        "--development-graph",
        str(outputs["development"]),
        "--confirmation-graph",
        str(outputs["confirmation"]),
        "--receipt",
        str(outputs["receipt"]),
        "--expected-receipt-sha256",
        typed_receipt.receipt_sha256,
    ]
    assert cli.main(validate_args) == 0
    validation_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(validation_receipt)
    assert validation_receipt["stage"] == "custodian_materialization_validated"
    assert opened == [
        paths["target"],
        paths["full"],
        outputs["roster"],
        outputs["development"],
        outputs["confirmation"],
        outputs["receipt"],
    ]


def test_custodian_cli_output_alias_symlink_and_existing_output_fail_closed(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)
    outputs = {
        "roster": tmp_path / "custodian-roster.json",
        "development": tmp_path / "custodian-development.json",
        "confirmation": tmp_path / "custodian-confirmation.json",
        "receipt": tmp_path / "custodian-receipt.json",
    }
    duplicate = dict(outputs)
    duplicate["receipt"] = duplicate["roster"]
    with pytest.raises(ConditionConfirmationError, match="outputs_must_be_distinct"):
        cli.main(_materialize_args(case, paths, duplicate))

    aliased = dict(outputs)
    aliased["roster"] = paths["full"]
    with pytest.raises(ConditionConfirmationError, match="must_not_alias_input"):
        cli.main(_materialize_args(case, paths, aliased))

    symlink_target = tmp_path / "existing-target.json"
    atomic_write_json(symlink_target, {"unrelated": True})
    outputs["receipt"].symlink_to(symlink_target)
    with pytest.raises(ConditionConfirmationError, match="output_symlink_forbidden"):
        cli.main(_materialize_args(case, paths, outputs))
    outputs["receipt"].unlink()

    atomic_write_json(outputs["development"], {"preexisting": True})
    with pytest.raises(OutputExistsError):
        cli.main(_materialize_args(case, paths, outputs))


def test_staged_cli_preserves_outcome_firewall_and_exactly_replays(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)
    opened: list[Path] = []
    original_reader = cli._read_regular_bytes

    def tracked_reader(path: Path, *, purpose: str) -> tuple[bytes, str]:
        opened.append(path)
        return original_reader(path, purpose=purpose)

    monkeypatch.setattr(cli, "_read_regular_bytes", tracked_reader)

    assert cli.main(_prepare_args(case, paths)) == 0
    prepare_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(prepare_receipt)
    assert opened == [
        paths["target"],
        paths["config"],
        paths["roster"],
        paths["materialization_receipt"],
    ]
    assert paths["development"] not in opened
    assert paths["full"] not in opened
    assert prepare_receipt["effect_or_outcome_values_opened"] is False
    plan = ConditionConfirmationPlanV1.model_validate_json(
        paths["plan"].read_text(encoding="utf-8")
    )
    assert plan == case.plan

    opened.clear()
    assert cli.main(_fit_args(case, paths)) == 0
    fit_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(fit_receipt)
    assert opened == [paths["plan"], paths["development"]]
    assert paths["full"] not in opened
    assert fit_receipt["confirmation_outcomes_opened"] is False
    model = ConditionConfirmationFrozenModelV1.model_validate_json(
        paths["model"].read_text(encoding="utf-8")
    )
    assert model == case.model

    opened.clear()
    confirm_args = [
        *_confirmation_args("confirm", case, paths),
        "--output",
        str(paths["assessment"]),
    ]
    assert cli.main(confirm_args) == 0
    confirm_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(confirm_receipt)
    assert confirm_receipt["write_disposition"] == "created"
    assert opened == [paths["plan"], paths["model"], paths["full"]]
    assessment_bytes = paths["assessment"].read_bytes()
    assessment = ConditionConfirmationAssessmentV1.model_validate_json(assessment_bytes)
    assert assessment.status == "confirmed"

    opened.clear()
    assert cli.main(confirm_args) == 0
    idempotent_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(idempotent_receipt)
    assert idempotent_receipt["write_disposition"] == "idempotent_existing_match"
    assert paths["assessment"].read_bytes() == assessment_bytes

    opened.clear()
    validate_args = [
        *_confirmation_args("validate", case, paths),
        "--assessment",
        str(paths["assessment"]),
    ]
    assert cli.main(validate_args) == 0
    validate_receipt = json.loads(capsys.readouterr().out)
    _assert_self_hashed_receipt(validate_receipt)
    assert validate_receipt["assessment_sha256"] == assessment.assessment_sha256
    assert validate_receipt["status"] == "confirmed"


def test_prepare_rejects_outcome_graph_json_and_fit_parser_has_no_test_path(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)
    arguments = _prepare_args(case, paths)
    roster_index = arguments.index("--roster") + 1
    arguments[roster_index] = str(paths["full"])
    with pytest.raises(
        ConditionConfirmationError,
        match="label_free_roster_contract_invalid",
    ):
        cli.main(arguments)
    assert not paths["plan"].exists()

    with pytest.raises(SystemExit):
        cli.main(
            [
                *_fit_args(case, paths),
                "--confirmation-graph",
                str(paths["full"]),
            ]
        )
    assert not paths["model"].exists()


@pytest.mark.parametrize(
    ("argument", "replacement", "expected_opened"),
    [
        ("--expected-plan-sha256", "0" * 64, ["plan"]),
        ("--current-pipeline-sha256", "0" * 64, ["plan"]),
        ("--expected-model-sha256", "0" * 64, ["plan", "model"]),
    ],
)
def test_confirm_refuses_before_full_graph_access_when_frozen_identity_mismatches(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    replacement: str,
    expected_opened: list[str],
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)
    atomic_write_json(paths["plan"], case.plan)
    atomic_write_json(paths["model"], case.model)
    opened: list[Path] = []
    original_reader = cli._read_regular_bytes

    def tracked_reader(path: Path, *, purpose: str) -> tuple[bytes, str]:
        opened.append(path)
        return original_reader(path, purpose=purpose)

    monkeypatch.setattr(cli, "_read_regular_bytes", tracked_reader)
    arguments = [
        *_confirmation_args("confirm", case, paths),
        "--output",
        str(paths["assessment"]),
    ]
    arguments[arguments.index(argument) + 1] = replacement
    with pytest.raises(ConditionConfirmationError, match="sha256_mismatch"):
        cli.main(arguments)
    assert opened == [paths[name] for name in expected_opened]
    assert paths["full"] not in opened
    assert not paths["assessment"].exists()


def test_output_alias_symlink_and_mismatched_existing_assessment_fail_closed(
    tmp_path: Path,
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    paths = _write_inputs(tmp_path, case)

    alias_arguments = _prepare_args(case, paths)
    alias_arguments[alias_arguments.index("--output") + 1] = str(paths["target"])
    with pytest.raises(ConditionConfirmationError, match="must_not_alias_input"):
        cli.main(alias_arguments)

    symlink_target = tmp_path / "unrelated-existing-output.json"
    atomic_write_json(symlink_target, {"unrelated": True})
    symlink_output = tmp_path / "plan-symlink.json"
    symlink_output.symlink_to(symlink_target)
    symlink_arguments = _prepare_args(case, paths)
    symlink_arguments[symlink_arguments.index("--output") + 1] = str(symlink_output)
    with pytest.raises(ConditionConfirmationError, match="output_symlink_forbidden"):
        cli.main(symlink_arguments)

    atomic_write_json(paths["plan"], {"preexisting": True})
    with pytest.raises(OutputExistsError):
        cli.main(_prepare_args(case, paths))

    paths["plan"].unlink()
    atomic_write_json(paths["plan"], case.plan)
    atomic_write_json(paths["model"], case.model)
    atomic_write_json(paths["assessment"], {"preexisting": True})
    arguments = [
        *_confirmation_args("confirm", case, paths),
        "--output",
        str(paths["assessment"]),
    ]
    with pytest.raises(
        ConditionConfirmationError,
        match="existing_assessment_contract_invalid",
    ):
        cli.main(arguments)
