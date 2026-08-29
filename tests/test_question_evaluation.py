from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import scripts.build_question_replay_state as bridge_cli
import scripts.evaluate_question_benchmark as cli

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.production_policy import (
    PRODUCTION_STOPPING_RULE,
)
from literature_multiverse.question_evaluation import (
    AuditCostBasis,
    BenchmarkEvidenceKind,
    PolicyInputProvenance,
    QuestionEvaluationContractError,
    ReferenceVerdictSource,
    ReplayPolicy,
    ReplayPolicyInput,
    ReplaySource,
    ReplayStoppingRule,
    compute_question_evaluation_pipeline_fingerprint,
    evaluate_question_benchmark,
    freeze_claim_question_benchmark_record,
    freeze_question_audit_event,
    freeze_question_replay_state,
    freeze_question_replay_state_from_certificate,
    freeze_reference_claim_verdict,
    load_question_benchmark,
    write_question_benchmark_jsonl,
)
from literature_multiverse.verifier import build_offline_fixture, run_verification


def _hash(label: str) -> str:
    return hash_canonical({"fixture": label})


def _resolve_local_dependency(
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
        candidates = [Path(*module.split(".")).with_suffix(".py")]
    else:
        return None
    return next(
        (
            candidate.as_posix()
            for candidate in candidates
            if (repository_root / candidate).is_file()
        ),
        None,
    )


def _independent_question_evaluation_dependency_closure(
    repository_root: Path,
) -> set[str]:
    pending = [
        "scripts/build_question_replay_state.py",
        "scripts/evaluate_question_benchmark.py",
        "src/literature_multiverse/question_evaluation.py",
    ]
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
                dependency = _resolve_local_dependency(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return observed


def _kind_contracts(kind: BenchmarkEvidenceKind):
    if kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        return (
            ReferenceVerdictSource.EXPERT_ADJUDICATION,
            AuditCostBasis.REALIZED_HUMAN_MINUTES,
            ReplaySource.LEGACY_DECLARED_PIPELINE_RERUN,
            2,
        )
    if kind is BenchmarkEvidenceKind.SIMULATION:
        return (
            ReferenceVerdictSource.PLANTED_SIMULATION,
            AuditCostBasis.SIMULATED_MINUTES,
            ReplaySource.PLANTED_SIMULATION,
            1,
        )
    return (
        ReferenceVerdictSource.DIAGNOSTIC_PROXY,
        AuditCostBasis.DIAGNOSTIC_MINUTES,
        ReplaySource.DIAGNOSTIC_APPROXIMATION,
        1,
    )


def _record(
    suffix: str,
    *,
    reference_supported: bool,
    exact_reference_decision: str | None = None,
    kind: BenchmarkEvidenceKind = BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED,
    fit_question_ids: list[str] | None = None,
    paper_id: str | None = None,
    omit_sequence: tuple[str, ...] | None = None,
    estimated_minutes_a: float = 1.0,
    estimated_minutes_b: float = 1.0,
    realized_minutes_a: float = 1.0,
    realized_minutes_b: float = 1.0,
    eligible_a: bool = True,
    eligible_b: bool = True,
    release_requires_audit: bool = False,
):
    question_id = f"question-{suffix}"
    claim_id = f"claim-{suffix}"
    source, cost_basis, replay_source, adjudicators = _kind_contracts(kind)
    reference = freeze_reference_claim_verdict(
        question_id=question_id,
        claim_id=claim_id,
        verdict=(
            exact_reference_decision
            if exact_reference_decision is not None
            else ("supported" if reference_supported else "contradicted")
        ),
        source=source,
        adjudicator_count=adjudicators,
        protocol_sha256=_hash(f"reference-protocol-{suffix}"),
        artifact_sha256=_hash(f"reference-artifact-{suffix}"),
    )
    item_a = f"item-{suffix}-a"
    item_b = f"item-{suffix}-b"
    events = [
        freeze_question_audit_event(
            item_id=item_a,
            disposition="corrected",
            completed_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
            realized_minutes=realized_minutes_a,
            cost_basis=cost_basis,
            adjudicator_count=adjudicators,
            protocol_sha256=_hash(f"audit-protocol-{suffix}"),
            artifact_sha256=_hash(f"audit-artifact-{suffix}-a"),
            correction_sha256=_hash(f"correction-{suffix}-a"),
        ),
        freeze_question_audit_event(
            item_id=item_b,
            disposition="confirmed",
            completed_at=datetime(2026, 8, 27, 12, 5, tzinfo=UTC),
            realized_minutes=realized_minutes_b,
            cost_basis=cost_basis,
            adjudicator_count=adjudicators,
            protocol_sha256=_hash(f"audit-protocol-{suffix}"),
            artifact_sha256=_hash(f"audit-artifact-{suffix}-b"),
        ),
    ]
    sequences = [(), (item_a,), (item_b,), (item_a, item_b), (item_b, item_a)]
    states = []
    for sequence in sequences:
        if sequence == omit_sequence:
            continue
        synthesis_sha = _hash(f"synthesis-{suffix}-{sequence}")
        remaining = sorted({item_a, item_b} - set(sequence))
        inputs = [
            ReplayPolicyInput(
                item_id=item_id,
                canonical_order=1 if item_id == item_a else 2,
                risk_score=0.9 if item_id == item_a else 0.1,
                risk_basis="calibrated_cell_rate_ucl",
                disagreement_score=0.8 if item_id == item_a else 0.2,
                influence_score=0.95 if item_id == item_a else 0.05,
                estimated_minutes=(
                    estimated_minutes_a if item_id == item_a else estimated_minutes_b
                ),
                eligible=eligible_a if item_id == item_a else eligible_b,
                ineligibility_reasons=(
                    []
                    if (eligible_a if item_id == item_a else eligible_b)
                    else ["prespecified_ineligible_fixture"]
                ),
                score_state_sha256=synthesis_sha,
            )
            for item_id in remaining
        ]
        error_repaired = item_a in sequence
        released = (
            True
            if exact_reference_decision is not None
            else (
                reference_supported and error_repaired
                if release_requires_audit
                else reference_supported or not error_repaired
            )
        )
        classification_supported = reference_supported or not error_repaired
        states.append(
            freeze_question_replay_state(
                question_id=question_id,
                pipeline_sha256=_hash("frozen-pipeline"),
                audit_sequence=sequence,
                policy_inputs=inputs,
                release_status="released" if released else "abstained",
                claim_classification=(
                    exact_reference_decision
                    if exact_reference_decision is not None
                    else ("supported" if classification_supported else "contradicted")
                ),
                release_reasons=(
                    []
                    if released
                    else (
                        ["prespecified_audit_gate_not_yet_satisfied"]
                        if classification_supported
                        else ["expert_correction_contradicts_target"]
                    )
                ),
                graph_sha256=_hash(f"graph-{suffix}-{sequence}"),
                synthesis_sha256=synthesis_sha,
                release_assessment_sha256=_hash(f"release-{suffix}-{sequence}"),
                replay_source=replay_source,
            )
        )
    return freeze_claim_question_benchmark_record(
        question_id=question_id,
        claim_id=claim_id,
        domain="domain-a" if suffix in {"a", "b"} else "domain-b",
        population_id="held-out-population",
        split="test",
        evidence_kind=kind,
        pipeline_sha256=_hash("frozen-pipeline"),
        corpus_sha256=_hash(f"corpus-{suffix}"),
        paper_ids=[paper_id or f"paper-{suffix}"],
        cohort_ids=[f"cohort-{suffix}"],
        policy_input_provenance=PolicyInputProvenance(
            artifact_sha256=_hash(f"policy-inputs-{suffix}"),
            fit_question_ids=fit_question_ids or [],
            fit_claim_ids=[],
            fit_paper_ids=[],
        ),
        reference_verdict=reference,
        audit_events=events,
        replay_states=states,
    )


def _records():
    return [
        _record("a", reference_supported=True),
        _record("b", reference_supported=False),
        _record("c", reference_supported=True),
        _record("d", reference_supported=False),
    ]


def test_real_question_replay_reports_decisive_cost_and_release_metrics(tmp_path) -> None:
    benchmark_path = tmp_path / "questions.jsonl"
    benchmark = write_question_benchmark_jsonl(benchmark_path, _records())

    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[0.0, 1.0],
        fixed_count=1,
        bootstrap_draws=100,
        bootstrap_seed=17,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )

    assert result.scientific_claim_eligible is False
    assert "certificate-bound v5 or final condition-v7 verifier states" in (
        result.causal_interpretation
    )
    assert any("order invariance" in row for row in result.replay_assumptions)
    assert {
        event.cost_accounting for record in benchmark.records for event in record.audit_events
    } == {"total_person_minutes_across_all_reviewers_and_final_adjudication"}
    no_audit = next(
        row
        for row in result.policy_results
        if row["policy"] == ReplayPolicy.NO_AUDIT.value
        and row["budget_minutes_per_question"] == 1.0
    )
    assert no_audit["metrics"]["release_coverage"] == 1.0
    assert no_audit["metrics"]["released_claim_error"] == 0.5
    assert no_audit["metrics"]["correct_releases_per_human_hour"] is None

    adaptive = next(
        row
        for row in result.policy_results
        if row["policy"] == ReplayPolicy.RISK_X_INFLUENCE_PER_COST.value
        and row["budget_minutes_per_question"] == 1.0
    )
    assert adaptive["metrics"]["release_coverage"] == 0.5
    assert adaptive["metrics"]["released_claim_error"] == 0.0
    assert adaptive["metrics"]["correct_releases_per_human_hour"] == 30.0
    assert adaptive["bootstrap"]["cluster_unit"] == "complete_claim_question"
    assert set(adaptive["domain_metrics"]) == {"domain-a", "domain-b"}
    assert adaptive["worst_domain_metrics"]["released_claim_error"] is not None

    paired = next(
        row
        for row in result.paired_policy_comparisons
        if row["baseline_policy"] == ReplayPolicy.NO_AUDIT.value
        and row["budget_minutes_per_question"] == 1.0
    )
    assert paired["primary_policy"] == ReplayPolicy.RISK_X_INFLUENCE_PER_COST.value
    assert paired["point_deltas"]["release_coverage"] == -0.5
    assert paired["point_deltas"]["released_claim_error"] == -0.5
    assert paired["point_deltas"]["released_claim_errors_per_question"] == -0.5
    assert paired["bootstrap"]["cluster_unit"] == "complete_claim_question"

    upper = result.audit_all_upper_bound
    assert upper["policy"] == "audit_all_upper_bound"
    assert upper["upper_bound"] is True
    assert upper["metrics"]["total_realized_minutes"] == 8.0
    assert result.evaluation_sha256 == hash_canonical(
        result.model_dump(mode="json", exclude={"evaluation_sha256"})
    )
    assert result.pipeline_sha256 == _hash("frozen-pipeline")
    assert result.evaluation_pipeline_fingerprint.components[0].component_id == (
        "question-benchmark-evaluation"
    )
    assert result.evaluation_pipeline_fingerprint.components[0].component_version == "8"


def test_question_evaluation_fingerprint_binds_exact_local_dependency_closure() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    component = compute_question_evaluation_pipeline_fingerprint(root=repository_root).components[0]
    observed = {row.path for row in component.files}
    expected_python = _independent_question_evaluation_dependency_closure(repository_root)

    assert observed == expected_python | {"pyproject.toml", "uv.lock"}
    assert component.component_version == "8"
    assert component.settings["in_repository_dependency_closure_bound"] is True
    assert {
        "src/literature_multiverse/claim_semantics.py",
        "src/literature_multiverse/condition_confirmation.py",
        "src/literature_multiverse/independence_identity.py",
        "src/literature_multiverse/meta_analysis.py",
        "src/literature_multiverse/native_extraction.py",
        "src/literature_multiverse/verifier.py",
    } <= observed


def test_five_way_released_decision_uses_exact_reference_equality(tmp_path) -> None:
    records = [
        _record(
            "five-way-condition",
            reference_supported=False,
            exact_reference_decision="condition_dependent",
        ),
        _record(
            "five-way-contradicted",
            reference_supported=False,
            exact_reference_decision="contradicted",
        ),
    ]
    benchmark = write_question_benchmark_jsonl(
        tmp_path / "five-way-questions.jsonl",
        records,
    )
    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[0.0],
        policies=[ReplayPolicy.NO_AUDIT],
        bootstrap_draws=100,
        bootstrap_seed=23,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )
    outcome = result.policy_results[0]
    assert outcome["metrics"]["released_claims"] == 2
    assert outcome["metrics"]["released_claim_errors"] == 0
    assert outcome["metrics"]["correct_releases"] == 2


def test_production_stop_rejects_legacy_declared_release_states(
    tmp_path,
) -> None:
    records = [
        _record(
            suffix,
            reference_supported=True,
            release_requires_audit=True,
        )
        for suffix in ("stop-a", "stop-b")
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "stopping.jsonl", records)

    with pytest.raises(
        QuestionEvaluationContractError,
        match="production_stopping_requires_all_states_certificate_bound",
    ):
        evaluate_question_benchmark(
            benchmark,
            budgets_minutes=[2.0],
            policies=[ReplayPolicy.RISK_ONLY],
            bootstrap_draws=100,
        )
    experimental = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[2.0],
        policies=[ReplayPolicy.RISK_ONLY],
        bootstrap_draws=100,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )

    assert experimental.evaluation_version == "question-policy-replay-evaluation-v7"
    assert experimental.production_policy_match is False
    assert "does not represent" in experimental.stopping_rule_semantics
    for record, experimental_row in zip(
        records,
        experimental.policy_results[0]["question_outcomes"],
        strict=True,
    ):
        first = next(
            row.item_id for row in record.replay_states[0].policy_inputs if row.risk_score == 0.9
        )
        second = next(
            row.item_id for row in record.replay_states[0].policy_inputs if row.risk_score == 0.1
        )
        assert experimental_row["selected_item_ids"] == [first, second]
        assert experimental_row["realized_minutes"] == 2.0
    assert experimental.policy_results[0]["production_policy_match"] is False

    contradictory = experimental.model_dump(mode="json", exclude={"evaluation_sha256"})
    contradictory["production_policy_match"] = True
    with pytest.raises(
        ValueError,
        match="question_evaluation_stopping_rule_contract_mismatch",
    ):
        type(experimental).model_validate(
            {
                **contradictory,
                "evaluation_sha256": hash_canonical(contradictory),
            }
        )


def test_certificate_bridge_projects_complete_production_stop_decision() -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    state = freeze_question_replay_state_from_certificate(certificate)
    binding = state.production_binding
    assert binding is not None

    assert PRODUCTION_STOPPING_RULE == ("stop_at_first_full_frozen_release_eligible_state")
    assert state.replay_source is ReplaySource.FROZEN_PIPELINE_RERUN
    assert state.replay_version == "question-replay-state-v5"
    assert binding.certificate_sha256 == certificate.certificate_sha256
    assert binding.production_stop_decision_sha256 == (
        certificate.production_stop_decision.decision_sha256
    )
    assert binding.evaluated_sequential_state_sha256 == (
        certificate.production_stop_decision.evaluated_state.state_sha256
    )
    assert binding.evaluated_audit_prefix == state.audit_sequence == []
    assert binding.evaluated_active_action_item_id is None
    assert binding.full_release_eligible is False
    assert binding.blocking_adapter_reasons == (
        certificate.production_stop_decision.blocking_adapter_reasons
    )
    assert state.release_status.value == "abstained"
    assert state.graph_sha256 == certificate.evidence_graph_sha256
    assert state.synthesis_sha256 == certificate.synthesis_sha256
    assert binding.source_evidence_graph_sha256 == (certificate.source_evidence_graph_sha256)
    assert binding.source_current_graph_lineage == certificate.current_state_hashes
    assert type(state).model_validate_json(state.model_dump_json()) == state


def test_certificate_bridge_rejects_stateless_v5_certificate() -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert certificate.production_stop_decision.evaluated_state is None
    with pytest.raises(
        QuestionEvaluationContractError,
        match="production_replay_requires_stateful_v5_certificate",
    ):
        freeze_question_replay_state_from_certificate(certificate)


def test_certificate_bridge_binds_active_action_as_release_blocker() -> None:
    manifest, corpus = build_offline_fixture()
    initial = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert initial.sequential_audit_state is not None
    assert initial.sequential_audit_state.session.active_action is not None
    active = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        sequential_audit_state=initial.sequential_audit_state,
        generated_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
    )
    state = freeze_question_replay_state_from_certificate(active)
    binding = state.production_binding
    assert binding is not None
    assert binding.production_outcome == "active_action_in_progress"
    assert binding.evaluated_active_action_item_id == (
        active.sequential_audit_state.session.active_action.item_id
    )
    assert binding.full_release_eligible is False
    assert state.release_status.value == "abstained"


@pytest.mark.parametrize(
    "tamper",
    [
        "status",
        "blocker",
        "evaluated_state",
        "audit_prefix",
        "graph",
        "synthesis",
        "certificate_swap",
    ],
)
def test_certificate_bridge_rejects_fully_rehashed_projection_tampering(tamper) -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    state = freeze_question_replay_state_from_certificate(certificate)
    payload = state.model_dump(mode="json")
    binding = payload["production_binding"]
    assert isinstance(binding, dict)

    if tamper == "status":
        payload["release_status"] = "released"
        payload["release_reasons"] = []
        payload["claim_classification"] = "supported"
        binding["release_status"] = "released"
        binding["release_reasons"] = []
        binding["claim_classification"] = "supported"
        binding["full_release_eligible"] = True
    elif tamper == "blocker":
        binding["blocking_adapter_reasons"] = []
    elif tamper == "evaluated_state":
        binding["evaluated_sequential_state_sha256"] = "f" * 64
    elif tamper == "audit_prefix":
        payload["audit_sequence"] = ["forged-prefix-item"]
        binding["evaluated_audit_prefix"] = ["forged-prefix-item"]
    elif tamper == "graph":
        payload["graph_sha256"] = "f" * 64
        binding["current_evidence_graph_sha256"] = "f" * 64
    elif tamper == "synthesis":
        payload["synthesis_sha256"] = "f" * 64
        binding["current_synthesis_sha256"] = "f" * 64
        for row in payload["policy_inputs"]:
            row["score_state_sha256"] = "f" * 64
    else:
        swapped = run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            allow_uncalibrated_sequential_analysis=True,
            generated_at=datetime(2026, 8, 27, 12, 0, 1, tzinfo=UTC),
        )
        assert swapped.certificate_sha256 != certificate.certificate_sha256
        binding["certificate"] = swapped.model_dump(mode="json")

    unsigned_binding = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = hash_canonical(unsigned_binding)
    unsigned_state = {key: value for key, value in payload.items() if key != "replay_sha256"}
    payload["replay_sha256"] = hash_canonical(unsigned_state)
    with pytest.raises(ValueError, match=r"production_replay|replay_release"):
        type(state).model_validate(payload)


def test_simulation_or_manual_row_cannot_impersonate_production_certificate() -> None:
    with pytest.raises(
        QuestionEvaluationContractError,
        match="requires_certificate_factory",
    ):
        freeze_question_replay_state(
            question_id="question-manual",
            pipeline_sha256=_hash("manual-pipeline"),
            audit_sequence=[],
            policy_inputs=[],
            release_status="abstained",
            claim_classification="inconclusive",
            release_reasons=["manual"],
            graph_sha256=_hash("manual-graph"),
            synthesis_sha256=_hash("manual-synthesis"),
            release_assessment_sha256=_hash("manual-release"),
            replay_source=ReplaySource.FROZEN_PIPELINE_RERUN,
        )


def test_certificate_bridge_cli_writes_round_trip_state(tmp_path, capsys) -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    certificate_path = tmp_path / "certificate.json"
    state_path = tmp_path / "replay-state.json"
    certificate_path.write_text(certificate.model_dump_json(), encoding="utf-8")

    assert (
        bridge_cli.main(
            [
                "--certificate",
                str(certificate_path),
                "--output",
                str(state_path),
            ]
        )
        == 0
    )
    state = freeze_question_replay_state_from_certificate(certificate)
    assert type(state).model_validate_json(state_path.read_text(encoding="utf-8")) == state
    summary = json.loads(capsys.readouterr().out)
    assert summary["certificate_sha256"] == certificate.certificate_sha256
    assert summary["replay_sha256"] == state.replay_sha256


def test_replay_uses_estimated_cost_for_feasibility_and_skips_unfit_or_ineligible_top(
    tmp_path,
) -> None:
    records = [
        _record(
            "unfit",
            reference_supported=True,
            estimated_minutes_a=2.0,
            estimated_minutes_b=1.0,
        ),
        _record(
            "ineligible",
            reference_supported=True,
            estimated_minutes_a=0.5,
            estimated_minutes_b=1.0,
            eligible_a=False,
        ),
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "feasibility.jsonl", records)

    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RISK_ONLY],
        bootstrap_draws=100,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )
    outcomes = result.policy_results[0]["question_outcomes"]

    assert outcomes[0]["selected_item_ids"] == ["item-ineligible-b"]
    assert outcomes[0]["resolved_item_ids"] == ["item-ineligible-b"]
    assert outcomes[1]["selected_item_ids"] == ["item-unfit-b"]
    assert outcomes[1]["resolved_item_ids"] == ["item-unfit-b"]
    assert all(row["historical_realized_minutes"] == 1.0 for row in outcomes)
    assert all(row["active_truncated_realized_minutes"] == 0.0 for row in outcomes)


def test_realized_overrun_leaves_active_action_unapplied_and_forces_abstention(
    tmp_path,
) -> None:
    records = [
        _record(
            "overrun-a",
            reference_supported=True,
            estimated_minutes_a=1.0,
            realized_minutes_a=2.0,
        ),
        _record(
            "overrun-b",
            reference_supported=False,
            estimated_minutes_a=1.0,
            realized_minutes_a=2.0,
        ),
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "overrun.jsonl", records)

    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RISK_ONLY],
        bootstrap_draws=100,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )
    policy = result.policy_results[0]
    outcomes = policy["question_outcomes"]

    for row in outcomes:
        expected_item = f"item-{row['question_id'].removeprefix('question-')}-a"
        assert row["selected_item_ids"] == [expected_item]
        assert row["attempted_item_ids"] == [expected_item]
        assert row["resolved_item_ids"] == []
        assert row["incomplete_item_ids"] == [expected_item]
        assert row["active_action_item_id"] == expected_item
        assert row["budget_exhausted_with_active_action"] is True
        assert row["historical_realized_minutes"] == 0.0
        assert row["active_truncated_realized_minutes"] == 1.0
        assert row["realized_minutes"] == 1.0
        assert row["release_status"] == "abstained"
        assert row["claim_classification"] == "supported"
        assert row["stop_reason"] == "budget_exhausted_with_active_action"
        assert "budget_exhausted_active_audit_action_unresolved" in row["release_reasons"]

    assert policy["metrics"]["total_realized_minutes"] == 2.0
    assert policy["metrics"]["historical_completed_realized_minutes"] == 0.0
    assert policy["metrics"]["active_truncated_realized_minutes"] == 2.0
    assert policy["metrics"]["attempted_audit_actions"] == 2
    assert policy["metrics"]["completed_audit_actions"] == 0
    assert policy["metrics"]["incomplete_audit_actions"] == 2
    assert policy["metrics"]["release_coverage"] == 0.0


def test_overrun_after_completed_correction_cannot_release_from_pre_action_state(
    tmp_path,
) -> None:
    records = [
        _record(
            suffix,
            reference_supported=True,
            estimated_minutes_a=0.5,
            realized_minutes_a=0.6,
            estimated_minutes_b=0.4,
            realized_minutes_b=0.5,
        )
        for suffix in ("second-overrun-a", "second-overrun-b")
    ]
    benchmark = write_question_benchmark_jsonl(
        tmp_path / "second-overrun.jsonl",
        records,
    )

    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RISK_ONLY],
        bootstrap_draws=100,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )

    for row in result.policy_results[0]["question_outcomes"]:
        suffix = row["question_id"].removeprefix("question-")
        assert row["selected_item_ids"] == [f"item-{suffix}-a", f"item-{suffix}-b"]
        assert row["resolved_item_ids"] == [f"item-{suffix}-a"]
        assert row["active_action_item_id"] == f"item-{suffix}-b"
        assert row["historical_realized_minutes"] == 0.6
        assert row["active_truncated_realized_minutes"] == pytest.approx(0.4)
        assert row["realized_minutes"] == 1.0
        assert row["claim_classification"] == "supported"
        assert row["release_status"] == "abstained"
        assert "budget_exhausted_active_audit_action_unresolved" in row["release_reasons"]


def test_no_estimated_candidate_fit_stops_without_opening_realized_cost(tmp_path) -> None:
    records = [
        _record(
            "nofit-a",
            reference_supported=True,
            estimated_minutes_a=2.0,
            estimated_minutes_b=2.0,
            realized_minutes_a=0.5,
            realized_minutes_b=0.5,
        ),
        _record(
            "nofit-b",
            reference_supported=True,
            estimated_minutes_a=2.0,
            estimated_minutes_b=2.0,
            realized_minutes_a=0.5,
            realized_minutes_b=0.5,
        ),
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "no-fit.jsonl", records)

    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RISK_ONLY],
        bootstrap_draws=100,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )

    for row in result.policy_results[0]["question_outcomes"]:
        assert row["selected_item_ids"] == []
        assert row["resolved_item_ids"] == []
        assert row["realized_minutes"] == 0.0
        assert row["release_status"] == "released"
        assert row["stop_reason"] == "no_eligible_candidate_fits_estimated_budget"


def test_seeded_random_replay_is_bitwise_deterministic(tmp_path) -> None:
    benchmark = write_question_benchmark_jsonl(tmp_path / "random.jsonl", _records())

    first = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RANDOM],
        random_seed=19,
        bootstrap_draws=100,
        bootstrap_seed=23,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )
    repeated = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        policies=[ReplayPolicy.RANDOM],
        random_seed=19,
        bootstrap_draws=100,
        bootstrap_seed=23,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )

    assert repeated == first


def test_simulation_is_labeled_and_rejected_without_explicit_opt_in(tmp_path) -> None:
    records = [
        _record("sim-a", reference_supported=True, kind=BenchmarkEvidenceKind.SIMULATION),
        _record("sim-b", reference_supported=False, kind=BenchmarkEvidenceKind.SIMULATION),
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "sim.jsonl", records)

    with pytest.raises(
        QuestionEvaluationContractError,
        match="non_real_benchmark_requires_explicit_allow_non_real",
    ):
        evaluate_question_benchmark(
            benchmark,
            budgets_minutes=[1.0],
            bootstrap_draws=100,
            stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
        )

    diagnostic = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=[1.0],
        bootstrap_draws=100,
        allow_non_real=True,
        stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
    )
    assert diagnostic.scientific_claim_eligible is False
    assert diagnostic.evidence_kind == "simulation"
    assert "cannot support" in diagnostic.claim_scope
    assert all(
        row["metrics"]["correct_releases_per_human_hour"] is None
        for row in diagnostic.policy_results
    )


def test_cross_question_paper_overlap_is_rejected() -> None:
    records = [
        _record("a", reference_supported=True, paper_id="shared-paper"),
        _record("b", reference_supported=False, paper_id="shared-paper"),
    ]

    with pytest.raises(QuestionEvaluationContractError, match="benchmark_paper_overlap"):
        write_question_benchmark_jsonl(Path("unused.jsonl"), records)


def test_questions_from_different_production_pipelines_cannot_be_pooled(tmp_path) -> None:
    first, second = _records()[:2]
    payload = second.model_dump(mode="json", exclude={"record_sha256"})
    different_pipeline = _hash("different-frozen-pipeline")
    payload["pipeline_sha256"] = different_pipeline
    for state in payload["replay_states"]:
        state["pipeline_sha256"] = different_pipeline
        unsigned_state = {key: value for key, value in state.items() if key != "replay_sha256"}
        state["replay_sha256"] = hash_canonical(unsigned_state)
    second = type(second).model_validate({**payload, "record_sha256": hash_canonical(payload)})

    with pytest.raises(
        QuestionEvaluationContractError,
        match="benchmark_pipeline_identity_mixed",
    ):
        write_question_benchmark_jsonl(tmp_path / "mixed-pipelines.jsonl", [first, second])


def test_policy_fit_overlap_with_any_evaluation_question_is_rejected(tmp_path) -> None:
    records = [
        _record(
            "a",
            reference_supported=True,
            fit_question_ids=["question-b"],
        ),
        _record("b", reference_supported=False),
    ]

    with pytest.raises(
        QuestionEvaluationContractError,
        match="benchmark_fit_evaluation_question_overlap",
    ):
        write_question_benchmark_jsonl(tmp_path / "leak.jsonl", records)


def test_missing_policy_selected_replay_state_fails_instead_of_approximating(
    tmp_path,
) -> None:
    records = [
        _record("a", reference_supported=True, omit_sequence=("item-a-a",)),
        _record("b", reference_supported=False),
    ]
    benchmark = write_question_benchmark_jsonl(tmp_path / "incomplete.jsonl", records)

    with pytest.raises(QuestionEvaluationContractError, match="replay_state_missing"):
        evaluate_question_benchmark(
            benchmark,
            budgets_minutes=[1.0],
            policies=[ReplayPolicy.RISK_ONLY],
            bootstrap_draws=100,
            stopping_rule=ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL,
        )


def test_cli_writes_hash_bound_evaluation(tmp_path, capsys) -> None:
    benchmark_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "evaluation.json"
    write_question_benchmark_jsonl(benchmark_path, _records())

    assert (
        cli.main(
            [
                "--benchmark",
                str(benchmark_path),
                "--output",
                str(output_path),
                "--budget-minutes",
                "0",
                "--budget-minutes",
                "1",
                "--fixed-count",
                "1",
                "--bootstrap-draws",
                "100",
                "--stopping-rule",
                ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL.value,
            ]
        )
        == 0
    )
    output = json.loads(output_path.read_text())
    summary = json.loads(capsys.readouterr().out)
    assert output["evaluation_sha256"] == summary["evaluation_sha256"]
    assert summary["question_count"] == 4
    assert summary["scientific_claim_eligible"] is False


def test_jsonl_row_hash_tampering_is_rejected(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    write_question_benchmark_jsonl(path, _records())
    lines = path.read_text().splitlines()
    payload = json.loads(lines[0])
    payload["domain"] = "tampered-domain"
    lines[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(QuestionEvaluationContractError, match="record_hash_mismatch"):
        load_question_benchmark(path)
