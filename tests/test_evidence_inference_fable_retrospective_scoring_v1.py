from __future__ import annotations

import json
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.private_cache_support import require_private_cache

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    MODEL,
    EvidenceInferenceFableCallSurfaceV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    authorize_evidence_inference_fable_workspace_v1,
    execute_evidence_inference_fable_paired_v1,
    freeze_evidence_inference_fable_budget_authorization_v1,
    prepare_evidence_inference_fable_workspace_v1,
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    EvidenceInferenceFableScoringError,
    PublicPairedSummaryV1,
    freeze_private_reference_label_bundle_v1,
    materialize_private_and_public_reports_v1,
    project_public_paired_summary_v1,
    replay_terminal_scoring_receipts_v1,
    repository_results_source_loader_v1,
    score_private_paired_report_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import hash_canonical

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("increase", "no_effect", "decrease")

pytestmark = pytest.mark.private_cache


class _FixtureClient:
    def __init__(
        self,
        *,
        plan: EvidenceInferenceFableRetrospectivePlanV1,
        source_loader: Any,
        expected_directions: dict[str, str],
        invalid_batch_key: str | None = None,
        eligible_false_key: str | None = None,
        empty_finding_key: str | None = None,
    ) -> None:
        self.requests = {item.request_key: item for item in plan.roster}
        self.source_loader = source_loader
        self.expected_directions = expected_directions
        self.invalid_batch_key = invalid_batch_key
        self.eligible_false_key = eligible_false_key
        self.empty_finding_key = empty_finding_key
        self.calls = 0

    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1:
        self.calls += 1
        request = self.requests[surface.request_key]
        lines = self.source_loader(request.article_id)
        line_key, source_line = next(
            (key, item)
            for key, item in lines.items()
            if item["text"] != "BODY.RESULTS:"
        )
        exact_quote = source_line["text"][:80]
        results: dict[str, Any] = {}
        for example_id in request.example_ids:
            if surface.request_key == self.eligible_false_key:
                item = {"eligible": False, "findings": []}
            elif surface.request_key == self.empty_finding_key:
                item = {"eligible": True, "findings": []}
            else:
                item = {
                    "eligible": True,
                    "findings": [
                        {
                            "direction": self.expected_directions[example_id],
                            "evidence_quote": exact_quote,
                            "evidence_lines": [line_key],
                        }
                    ],
                }
            results[example_id] = item
        if surface.request_key == self.invalid_batch_key:
            results.pop(request.example_ids[-1])
        payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": surface.request_key,
            "surface_sha256": surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "completed",
            "response_id": f"fixture-{self.calls}",
            "response_model": MODEL,
            "parsed_json": {"results": results},
            "input_tokens": 100,
            "output_tokens": 20,
            "reported_cost_usd_micros": 2_000,
            "charged_cost_usd_micros": 2_000,
            "cost_basis": "reported_usage",
            "response_text_sha256": None,
            "failure_code": None,
        }
        return EvidenceInferenceFableProviderResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )


@pytest.fixture(scope="module")
def pilot_runtime_contract() -> tuple[
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
]:
    require_private_cache(
        "data/cache/evidence-inference-gepa/manifest.json",
        "data/cache/evidence-inference-gepa/conversion_report.json",
        "data/cache/evidence-inference-gepa-pilot30/manifest.json",
        "data/cache/evidence-inference-gepa-pilot30/conversion_report.json",
        "data/cache/evidence-inference-gepa-low-budget/manifest.json",
        "data/cache/evidence-inference-gepa-low-budget/conversion_report.json",
        "data/cache/evidence-inference-2.0/prompts_merged.csv",
        "data/cache/evidence-inference-2.0/txt_files",
        "data/cache/evidence-inference-ollama-gepa-v1-final-v3/frozen-winner.json",
        "data/cache/evidence-inference-ollama-gepa-v1-final-v3/frozen-winner.md",
        "data/cache/evidence-inference-ollama-gepa-v1-final-v3/gepa-result.json",
        "data/cache/evidence-inference-ollama-gepa-v1-final-v3/optimization-plan.json",
    )
    return reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )


def _fixture_directions(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> dict[str, str]:
    example_ids = sorted(
        {example_id for item in plan.roster for example_id in item.example_ids}
    )
    return {
        example_id: DIRECTIONS[index % len(DIRECTIONS)]
        for index, example_id in enumerate(example_ids)
    }


def _execute_fixture_workspace(
    *,
    tmp_path: Path,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    client: _FixtureClient,
    budget: int,
) -> Path:
    workspace = tmp_path / "runtime"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        prepared=prepared,
    )
    authorization = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=budget,
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        authorization=authorization,
    )
    execute_evidence_inference_fable_paired_v1(
        workspace=workspace,
        plan=plan,
        client=client,
    )
    return workspace


def test_replayed_private_score_enforces_batch_ite_and_unconditional_grounding(
    tmp_path: Path,
    pilot_runtime_contract: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFablePreparedRuntimeV1,
    ],
) -> None:
    plan, prepared = pilot_runtime_contract
    source_loader = repository_results_source_loader_v1(repository_root=ROOT)
    directions = _fixture_directions(plan)
    invalid_request = plan.roster[0]
    ineligible_request = plan.roster[1]
    empty_request = plan.roster[2]
    client = _FixtureClient(
        plan=plan,
        source_loader=source_loader,
        expected_directions=directions,
        invalid_batch_key=invalid_request.request_key,
        eligible_false_key=ineligible_request.request_key,
        empty_finding_key=empty_request.request_key,
    )
    workspace = _execute_fixture_workspace(
        tmp_path=tmp_path,
        plan=plan,
        prepared=prepared,
        client=client,
        budget=plan.total_full_context_hard_liability_usd_micros,
    )
    label_load_count = 0

    def load_labels() -> Any:
        nonlocal label_load_count
        label_load_count += 1
        return freeze_private_reference_label_bundle_v1(
            plan=plan,
            expected_directions=directions,
        )

    private_report = score_private_paired_report_v1(
        plan=plan,
        runtime_workspace=workspace,
        source_loader=source_loader,
        private_label_loader=load_labels,
    )
    assert client.calls == plan.request_count == 14
    assert label_load_count == 1
    assert private_report.runtime_terminal_sha256 == (
        private_report.completion_certificate.runtime_terminal_sha256
    )
    assert private_report.completion_certificate.private_scored_rows_sha256 == (
        private_report.scored_rows.private_scored_rows_sha256
    )

    rows = {item.example_id: item for item in private_report.scored_rows.rows}
    for example_id in invalid_request.example_ids:
        score = getattr(rows[example_id], invalid_request.arm)
        assert score.primary_failure == "invalid_article_batch"
        assert score.structured_output_reliability == 0
        assert score.direction_accuracy == 0
        assert score.exact_grounding_reliability == 0
    for example_id in ineligible_request.example_ids:
        score = getattr(rows[example_id], ineligible_request.arm)
        assert score.primary_failure == "eligible_false"
        assert score.structured_output_reliability == 1
        assert score.exact_grounding_reliability == 0
        assert score.conditional_grounding_evaluated == 0
    for example_id in empty_request.example_ids:
        score = getattr(rows[example_id], empty_request.arm)
        assert score.primary_failure == "missing_finding"
        assert score.structured_output_reliability == 1
        assert score.exact_grounding_reliability == 0
        assert score.conditional_grounding_evaluated == 0

    untouched = next(
        item
        for item in private_report.scored_rows.rows
        if item.example_id not in set(invalid_request.example_ids)
        | set(ineligible_request.example_ids)
        | set(empty_request.example_ids)
    )
    assert untouched.seed.exact_grounding_reliability == 1
    assert untouched.winner.exact_grounding_reliability == 1
    assert private_report.paired_article_cluster_bootstrap.question_count == 30
    assert private_report.paired_article_cluster_bootstrap.article_cluster_count == 7

    public = project_public_paired_summary_v1(private_report)
    public_text = json.dumps(public.model_dump(mode="json"), sort_keys=True)
    for request in plan.roster:
        assert request.article_id not in public_text
        assert all(example_id not in public_text for example_id in request.example_ids)
    first_lines = source_loader(plan.roster[0].article_id)
    first_source_text = next(
        item["text"] for item in first_lines.values() if item["text"] != "BODY.RESULTS:"
    )
    assert first_source_text[:80] not in public_text
    assert '"rows"' not in public_text
    assert '"parsed_item"' not in public_text
    assert '"grounding_detail"' not in public_text
    assert public.runtime_terminal_sha256 == private_report.runtime_terminal_sha256
    assert public.completion_certificate_sha256 == (
        private_report.completion_certificate.certificate_sha256
    )
    assert public.gepa_optimization_improvement_authority is False
    assert public.scientific_effectiveness_authority is False
    assert public.generalization_authority is False

    private_path = tmp_path / "private" / "score.json"
    public_path = tmp_path / "public" / "summary.json"
    private_path.parent.mkdir()
    public_path.parent.mkdir()
    materialize_private_and_public_reports_v1(
        private_report=private_report,
        public_summary=public,
        private_path=private_path,
        public_path=public_path,
    )
    assert private_path.is_file() and public_path.is_file()
    with pytest.raises(
        EvidenceInferenceFableScoringError,
        match="scoring_report_target_not_fresh",
    ):
        materialize_private_and_public_reports_v1(
            private_report=private_report,
            public_summary=public,
            private_path=private_path,
            public_path=public_path,
        )
    tampered = public.model_dump(mode="json")
    tampered["examples"] += 1
    with pytest.raises(ValidationError):
        PublicPairedSummaryV1.model_validate(tampered)

    leaked_key = public.model_dump(mode="json")
    provider_counts = leaked_key["arms"]["seed"]["provider_outcome_counts"]
    original_key = next(iter(provider_counts))
    provider_counts[plan.roster[0].article_id] = provider_counts.pop(original_key)
    with pytest.raises(ValidationError):
        PublicPairedSummaryV1.model_validate(leaked_key)

    inconsistent_point = public.model_dump(mode="json")
    bootstrap = inconsistent_point["paired_article_cluster_bootstrap"]
    estimate = bootstrap["estimates"][0]
    denominator = estimate["denominator"]
    old_seed = estimate["seed_success_count"]
    new_seed = old_seed - 1 if old_seed == denominator else old_seed + 1
    estimate["seed_success_count"] = new_seed
    quantum = Decimal("0.000000001")
    estimate["seed_rate"] = str(
        (Decimal(new_seed) / Decimal(denominator)).quantize(
            quantum, rounding=ROUND_HALF_EVEN
        )
    )
    estimate["winner_minus_seed_difference"] = str(
        (
            Decimal(estimate["winner_success_count"] - new_seed)
            / Decimal(denominator)
        ).quantize(quantum, rounding=ROUND_HALF_EVEN)
    )
    estimate["percentile_95_lower"] = "-1.000000000"
    estimate["percentile_95_upper"] = "1.000000000"
    bootstrap["bootstrap_sha256"] = hash_canonical(
        {key: value for key, value in bootstrap.items() if key != "bootstrap_sha256"}
    )
    inconsistent_point["public_summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in inconsistent_point.items()
            if key != "public_summary_sha256"
        }
    )
    with pytest.raises(ValidationError, match="public_summary_alias_mismatch"):
        PublicPairedSummaryV1.model_validate(inconsistent_point)


def test_incomplete_runtime_blocks_before_private_label_loader(
    tmp_path: Path,
    pilot_runtime_contract: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFablePreparedRuntimeV1,
    ],
) -> None:
    plan, prepared = pilot_runtime_contract
    source_loader = repository_results_source_loader_v1(repository_root=ROOT)
    directions = _fixture_directions(plan)
    client = _FixtureClient(
        plan=plan,
        source_loader=source_loader,
        expected_directions=directions,
    )
    workspace = _execute_fixture_workspace(
        tmp_path=tmp_path,
        plan=plan,
        prepared=prepared,
        client=client,
        budget=1,
    )
    label_loaded = False

    def forbidden_label_load() -> Any:
        nonlocal label_loaded
        label_loaded = True
        raise AssertionError("private labels must remain unopened")

    with pytest.raises(
        EvidenceInferenceFableScoringError,
        match="completed_runtime_required_for_full_population_score",
    ):
        score_private_paired_report_v1(
            plan=plan,
            runtime_workspace=workspace,
            source_loader=source_loader,
            private_label_loader=forbidden_label_load,
        )
    assert client.calls == 0
    assert label_loaded is False


def test_runtime_hard_failures_replay_as_transport_or_ambiguity(
    tmp_path: Path,
    pilot_runtime_contract: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFablePreparedRuntimeV1,
    ],
) -> None:
    plan, prepared = pilot_runtime_contract

    class HardFailureClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _surface: EvidenceInferenceFableCallSurfaceV1) -> Any:
            self.calls += 1
            if self.calls % 2:
                raise RuntimeError("fixture provider exception")
            return {"invalid": "provider result"}

    client = HardFailureClient()
    workspace = tmp_path / "hard-failure-runtime"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        prepared=prepared,
    )
    authorization = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=(
            plan.total_full_context_hard_liability_usd_micros
        ),
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        authorization=authorization,
    )
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace,
        plan=plan,
        client=client,
    )
    assert terminal.status == "completed"
    assert client.calls == plan.request_count

    replayed_terminal, receipts = replay_terminal_scoring_receipts_v1(
        plan=plan,
        runtime_workspace=workspace,
    )
    assert replayed_terminal == terminal
    assert {receipt.provider_outcome for receipt in receipts} == {
        "transport_failed_or_ambiguous"
    }
