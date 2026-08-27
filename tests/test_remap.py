from __future__ import annotations

from copy import deepcopy

import pytest

from literature_multiverse.remap import (
    RemapContractError,
    RemapInferenceOverrides,
    approval_template,
    evaluate_remap_candidate,
    execute_approved_remap,
    incremental_cv,
    not_run_trace,
    propose_remap,
    reconcile_echo_responses,
    validate_approval,
)


def _headline() -> dict[str, object]:
    return {
        "narrative_variant": "A",
        "moderator": {"name": "dose_regime"},
        "rendered_sentence": "Frozen pre-specified headline.",
    }


def _pairs() -> list[dict[str, object]]:
    return [
        {
            "pair_id": f"pair-{index}",
            "distance": index / 10,
            "distance_components": [{"field": "comparator", "distance": 0}],
            "left_citation": {"finding_id": f"left-{index}"},
            "right_citation": {"finding_id": f"right-{index}"},
        }
        for index in range(7)
    ]


def _moderator() -> dict[str, object]:
    return {
        "name": "assay_context",
        "type": "categorical",
        "kind": "paper_constant",
        "categories": ["alpha", "beta", "gamma"],
        "bins": None,
        "paper_summary": None,
        "permutation": "paper",
        "extraction_prompt": "Classify the assay context from the quoted method.",
        "rationale": "Residual pairs differ in their assay context.",
    }


def _proposal() -> dict[str, object]:
    return propose_remap(
        _pairs(),
        _headline(),
        base_moderator_names=["dose_regime", "training_status"],
        proposer=lambda _: _moderator(),
    )


def _approval(proposal: dict[str, object], *, approved: bool = True) -> dict[str, object]:
    approval = approval_template(proposal)
    approval.update(
        {
            "approved": approved,
            "approved_moderator": proposal["moderator"] if approved else None,
            "reviewer": "Reviewer A",
            "reason": "Approved for the frozen exploratory protocol.",
        }
    )
    return approval


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    directions = ("increase", "no_effect", "decrease")
    candidate = ("alpha", "beta", "gamma")
    for index in range(30):
        rows.append(
            {
                "finding_id": f"f{index:02d}",
                "paper_id": f"p{index:02d}",
                "effect_direction": directions[index % 3],
                "mod__dose_regime": "high" if index % 2 else "low",
                "evidence_quote": f"quote {index}",
                "outcome_name": "peak_power",
                "dose_raw": "fixture dose",
                "candidate": candidate[index % 3],
            }
        )
    return rows


def _execution(proposal: dict[str, object]) -> dict[str, object]:
    rows = _rows()
    reconciliation = reconcile_echo_responses(
        [str(row["finding_id"]) for row in rows],
        [{"finding_id": row["finding_id"], "value": row["candidate"]} for row in rows],
    )
    return {
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "moderator": proposal["moderator"],
        "reconciliation": {key: value for key, value in reconciliation.items() if key != "table"},
        "side_table": reconciliation["table"],
    }


def _passing_overrides() -> RemapInferenceOverrides:
    return RemapInferenceOverrides(
        incremental_cv={
            "status": "eligible",
            "k": 5,
            "delta_ll": 0.05,
            "positive_folds": 4,
            "folds": [
                {
                    "fold": index,
                    "train_indices": [0, 1],
                    "test_indices": [2, 3],
                    "train_papers": ["p0", "p1"],
                    "test_papers": ["p2", "p3"],
                }
                for index in range(5)
            ],
            "n_rows": 30,
            "n_papers": 30,
            "finding_ids": [f"f{index:02d}" for index in range(30)],
        },
        permutation={
            "status": "complete",
            "success_count": 100,
            "attempt_count": 100,
            "p_value": 0.05,
        },
        bootstrap={
            "status": "complete",
            "n_bootstraps": 200,
            "positive_fraction": 0.75,
        },
        all_valid_sensitivity={
            "incremental_delta_ll": 0.03,
            "positive_gain": True,
        },
    )


def test_proposal_is_hash_bound_to_frozen_headline_and_human_approval() -> None:
    proposal = _proposal()
    assert proposal["status"] == "proposed"
    assert proposal["residual_pair_ids"] == [f"pair-{index}" for index in range(7)]
    assert validate_approval(proposal, _approval(proposal)) == "approved"

    tampered = _approval(proposal)
    tampered["proposal_sha256"] = "x" * 64
    with pytest.raises(RemapContractError, match="approval_proposal_hash_mismatch"):
        validate_approval(proposal, tampered)

    unavailable = approval_template(proposal)
    with pytest.raises(RemapContractError, match="human_approval_unavailable"):
        validate_approval(proposal, unavailable)


def test_variant_b_cannot_enter_proposal_and_fewer_than_five_pairs_fail() -> None:
    with pytest.raises(RemapContractError, match="requires_variant_a"):
        propose_remap(
            _pairs(),
            {"narrative_variant": "B"},
            base_moderator_names=[],
            proposer=lambda _: _moderator(),
        )
    with pytest.raises(RemapContractError, match="insufficient_residual_pairs"):
        propose_remap(
            _pairs()[:4],
            _headline(),
            base_moderator_names=[],
            proposer=lambda _: _moderator(),
        )


def test_echo_join_allows_one_missing_of_twenty_but_never_duplicates_or_unknowns() -> None:
    expected = [f"f{index}" for index in range(20)]
    responses = [
        {"finding_id": finding_id, "value": None if index == 0 else "alpha"}
        for index, finding_id in enumerate(expected[:19])
    ]
    valid = reconcile_echo_responses(expected, responses)
    assert valid["technical_valid"] is True
    assert valid["join_fraction"] == pytest.approx(0.95)
    assert valid["null_count"] == 1
    assert valid["missing"] == ["f19"]

    duplicate = [*responses, {"finding_id": "f0", "value": "beta"}]
    assert reconcile_echo_responses(expected, duplicate)["technical_valid"] is False
    unknown = [*responses, {"finding_id": "outside", "value": "beta"}]
    assert reconcile_echo_responses(expected, unknown)["technical_valid"] is False


def test_execute_approved_maps_whole_primary_cohort_and_preserves_null_join() -> None:
    proposal = _proposal()
    rows = _rows()

    def mapper(request: dict[str, object]) -> list[dict[str, object]]:
        findings = request["findings"]
        assert isinstance(findings, list)
        return [
            {
                "finding_id": row["finding_id"],
                "value": None if index == 0 else ("alpha", "beta", "gamma")[index % 3],
            }
            for index, row in enumerate(findings)
        ]

    execution = execute_approved_remap(
        proposal=proposal,
        approval=_approval(proposal),
        primary_rows=rows,
        mapper=mapper,
    )
    assert execution["status"] == "complete"
    assert execution["reconciliation"]["expected_count"] == 30
    assert execution["reconciliation"]["join_fraction"] == 1.0
    assert execution["reconciliation"]["null_count"] == 1
    assert len(execution["side_table"]) == 30


def test_incremental_cv_uses_identical_rows_folds_and_paper_weights() -> None:
    rows = []
    for source in _rows():
        row = dict(source)
        row["__candidate__"] = source["candidate"]
        rows.append(row)
    result = incremental_cv(rows, base_moderator_key="mod__dose_regime", seed=17)
    assert result["status"] == "eligible"
    assert result["k"] >= 3
    assert result["delta_ll"] > 0
    for fold in result["folds"]:
        assert set(fold["train_papers"]).isdisjoint(fold["test_papers"])
        assert fold["test_weight_sum"] == pytest.approx(len(fold["test_papers"]))
        assert fold["before_log_loss"] >= 0
        assert fold["after_log_loss"] >= 0


def test_keep_discard_and_indeterminate_states_are_rule_driven() -> None:
    proposal = _proposal()
    execution = _execution(proposal)
    kept = evaluate_remap_candidate(
        proposal=proposal,
        execution=execution,
        primary_rows=_rows(),
        frozen_headline=_headline(),
        overrides=_passing_overrides(),
    )
    assert kept["decision"] == "kept_exploratory"
    assert kept["language_guard"] == (
        "cross-validated incremental gain, moderator proposed post hoc"
    )
    assert kept["frozen_headline_sha256"]
    assert kept["before_model"]["frozen"] is True

    numeric = deepcopy(_passing_overrides().incremental_cv)
    assert numeric is not None
    numeric["delta_ll"] = 0.01
    discarded = evaluate_remap_candidate(
        proposal=proposal,
        execution=execution,
        primary_rows=_rows(),
        frozen_headline=_headline(),
        overrides=RemapInferenceOverrides(
            incremental_cv=numeric,
            permutation=_passing_overrides().permutation,
            bootstrap=_passing_overrides().bootstrap,
            all_valid_sensitivity=_passing_overrides().all_valid_sensitivity,
        ),
    )
    assert discarded["decision"] == "discarded"
    assert discarded["reason"] == "numeric_rule_failed"

    invalid_execution = deepcopy(execution)
    invalid_execution["reconciliation"]["technical_valid"] = False
    invalid_execution["reconciliation"]["duplicates"] = ["f00"]
    invalid = evaluate_remap_candidate(
        proposal=proposal,
        execution=invalid_execution,
        primary_rows=_rows(),
        frozen_headline=_headline(),
        overrides=_passing_overrides(),
    )
    assert invalid["decision"] == "indeterminate"
    assert invalid["reason"] == "invalid_echo_join"


def test_undefined_exchangeability_and_99_of_125_are_indeterminate() -> None:
    proposal = _proposal()
    execution = _execution(proposal)
    undefined = deepcopy(proposal)
    undefined["moderator"]["kind"] = "within_paper"
    undefined["moderator"]["permutation"] = "paper_summary"
    undefined["moderator"]["paper_summary"] = None
    execution["proposal_sha256"] = undefined["proposal_sha256"]
    result = evaluate_remap_candidate(
        proposal=undefined,
        execution=execution,
        primary_rows=_rows(),
        frozen_headline=_headline(),
        overrides=_passing_overrides(),
    )
    assert result["decision"] == "indeterminate"
    assert result["reason"] == "undefined_exchangeability"

    incomplete_permutation = dict(_passing_overrides().permutation or {})
    incomplete_permutation.update(
        {"status": "indeterminate", "success_count": 99, "attempt_count": 125, "p_value": None}
    )
    incomplete = evaluate_remap_candidate(
        proposal=proposal,
        execution=_execution(proposal),
        primary_rows=_rows(),
        frozen_headline=_headline(),
        overrides=RemapInferenceOverrides(
            incremental_cv=_passing_overrides().incremental_cv,
            permutation=incomplete_permutation,
            bootstrap=_passing_overrides().bootstrap,
            all_valid_sensitivity=_passing_overrides().all_valid_sensitivity,
        ),
    )
    assert incomplete["decision"] == "indeterminate"
    assert incomplete["reason"] == "inference_incomplete_or_undefined"


def test_not_run_trace_reasons_are_closed() -> None:
    assert not_run_trace("human_approval_unavailable") == {
        "status": "not_run",
        "reason": "human_approval_unavailable",
    }
    with pytest.raises(RemapContractError):
        not_run_trace("made_up")
