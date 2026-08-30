from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest
from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import ValidationError
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

import literature_multiverse.metasyn_contextual_frontier_recovery_v2 as recovery
import literature_multiverse.metasyn_contextual_frontier_runtime_v1 as transport
from literature_multiverse.metasyn_bounded_hosted_runtime import MetaSynHostedRuntimeError
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError

ROOT = Path(__file__).resolve().parents[1]
V1_WORKSPACE = ROOT / "data/cache/metasyn/contextual-frontier-runtime-v1"
RECOVERY_V2_PREPARED = "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json"

pytestmark = pytest.mark.private_cache


@pytest.fixture(scope="module")
def plan() -> recovery.MetaSynContextualFrontierRecoveryPlanV2:
    cached = os.environ.get("LM_FRONTIER_RECOVERY_V2_TEST_PLAN")
    if cached:
        raw = json.loads(Path(cached).read_text(encoding="utf-8"))
        return recovery.MetaSynContextualFrontierRecoveryPlanV2.model_validate(raw.get("plan", raw))
    require_private_cache(
        "data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json",
        "data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json",
        RECOVERY_V2_PREPARED,
    )
    return skip_when_historical_replay_is_stale(
        lambda: recovery.freeze_metasyn_contextual_frontier_recovery_plan_v2(repository_root=ROOT),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )


def _completed_branch(schema: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item for item in schema["oneOf"] if item["properties"]["status"].get("const") == "completed"
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_plan_grant_fields_false(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    grant_fields = {
        "graph_construction_mechanics_authority",
        "extraction_accuracy_authority",
        "reliability_authority",
        "generalization_authority",
        "synthesis_input_authority",
        "scientific_synthesis_authority",
        "scientific_effectiveness_authority",
        "calibration_authority",
        "claim_release_authority",
    }
    # The code-owned offline evaluator fixture predates this recovery and records
    # a mechanics observation.  It is not a recovery-v2 grant surface.  Check the
    # plan and every provider-facing recovery contract directly instead of
    # recursively treating that embedded fixture as new empirical authority.
    surfaces = {
        "plan": plan,
        "config": plan.config,
        "target_spec": plan.target_spec,
        "predecessor": plan.predecessor,
        "request": plan.request,
    }
    found: list[tuple[str, Any]] = []
    for surface_name, surface in surfaces.items():
        values = surface.model_dump(mode="json")
        found.extend(
            (f"{surface_name}.{key}", values[key]) for key in grant_fields if key in values
        )
    assert found
    assert all(item is False for _, item in found), found


def test_response_schema_is_an_exact_keyed_fifteen_field_contract(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    expected = [item.field_path for item in plan.target_spec.fields]
    assert len(expected) == 15
    assert expected == sorted(set(expected))
    assert expected == plan.request.required_field_paths

    completed = _completed_branch(plan.request.response_schema)
    claims = completed["properties"]["claims_by_field"]
    assert claims["type"] == "object"
    assert claims["additionalProperties"] is False
    assert claims["required"] == expected
    assert sorted(claims["properties"]) == expected
    for field_path in expected:
        field = claims["properties"][field_path]
        assert field["additionalProperties"] is False
        assert field["required"] == [
            "passage_id",
            "support_quote",
            "context",
            "token",
            "normalization",
        ]


def test_target_is_explicitly_500_mg_vs_placebo_primary_week_24(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    assert plan.target_spec.estimand == (
        "fedratinib 500-mg group versus placebo group for the primary end point "
        "spleen response at week 24"
    )
    assert plan.request.explicit_estimand == plan.target_spec.estimand
    assert "Do not select the 400-mg arm" in plan.request.prompt

    completed = _completed_branch(plan.request.response_schema)
    properties = completed["properties"]["claims_by_field"]["properties"]
    assert properties["treatment_arm.label"]["properties"]["token"]["const"] == "500-mg"
    assert properties["comparator_arm.label"]["properties"]["token"]["const"] == "placebo group"
    assert (
        properties["finding.endpoint_marker"]["properties"]["token"]["const"] == "primary end point"
    )


def test_event_and_total_answers_are_not_disclosed_as_target_or_schema_constants(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    extracted = set(plan.target_spec.extraction_scored_field_paths)
    assert extracted == {
        "effect.control_events",
        "effect.control_total",
        "effect.treatment_events",
        "effect.treatment_total",
    }
    by_path = {item.field_path: item for item in plan.target_spec.fields}
    completed = _completed_branch(plan.request.response_schema)
    properties = completed["properties"]["claims_by_field"]["properties"]
    for field_path in extracted:
        assert by_path[field_path].canonical_token is None
        assert by_path[field_path].token_policy == ("extract_exact_unsigned_integer_from_source")
        token_schema = properties[field_path]["properties"]["token"]
        assert "const" not in token_schema
        assert token_schema["pattern"] == r"^(?:0|[1-9][0-9]{0,9})$"
    assert not plan.target_spec.event_count_answers_disclosed_outside_source
    assert not plan.evaluator_numeric_targets_model_facing
    assert plan.evaluator_fixture_never_passed_to_request_builder


def test_recovery_request_is_materially_distinct_from_both_immutable_v1_requests(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    require_private_cache("data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json")
    frozen_v1 = transport.MetaSynContextualFrontierPlanV1.model_validate(
        json.loads((V1_WORKSPACE / "00-prepared.json").read_text(encoding="utf-8"))
    )
    old = [item.request for item in frozen_v1.roster]
    assert len(old) == 2
    for prior in old:
        assert plan.request.request_sha256 != prior.request_sha256
        assert plan.request.prompt_sha256 != prior.prompt_sha256
        assert plan.request.response_schema_sha256 != prior.original_schema_sha256
        assert (
            plan.request.transport_request.wire_call_surface_sha256
            != prior.wire_call_surface_sha256
        )
    assert not plan.request.exact_request_retry_permitted
    assert not plan.config.predecessor_v1_retry_permitted
    assert plan.predecessor_requests_retried == 0


def test_public_request_builder_has_no_evaluator_or_numeric_target_input(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    parameters = inspect.signature(
        recovery.freeze_metasyn_contextual_frontier_recovery_request_v2
    ).parameters
    assert set(parameters) == {
        "provider_context",
        "target_spec",
        "predecessor",
        "transport_config",
    }
    assert all("evaluator" not in name and "semantic_target" not in name for name in parameters)
    rebuilt = recovery.freeze_metasyn_contextual_frontier_recovery_request_v2(
        provider_context=plan.provider_context,
        target_spec=plan.target_spec,
        predecessor=plan.predecessor,
        transport_config=plan.transport_profile_config,
    )
    assert rebuilt == plan.request
    assert plan.evaluator_fixture_never_passed_to_request_builder
    assert not plan.evaluator_numeric_targets_model_facing


def test_v1_failure_is_bound_only_by_terminal_file_sha_without_semantic_salvage(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    predecessor = plan.predecessor
    assert predecessor.binding_method == "file_sha256_only_no_semantic_parse"
    assert predecessor.terminal_path == (
        "data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json"
    )
    assert predecessor.terminal_file_sha256 == (
        "ea3bca6df39aa914d1fede51edbd5abbb39e1624bb6adcb5792946b48411f76d"
    )
    assert not predecessor.exact_request_retry_permitted
    assert not predecessor.predecessor_workspace_mutation_permitted
    assert set(predecessor.model_dump(mode="json")) == {
        "provenance_version",
        "terminal_path",
        "terminal_file_sha256",
        "binding_method",
        "predecessor_diagnosis",
        "exact_request_retry_permitted",
        "predecessor_workspace_mutation_permitted",
        "claim_release_authority",
        "provenance_sha256",
    }


def test_all_sixteen_provider_visible_passages_are_frozen_and_citable(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    passages = plan.provider_visible_source_passages
    assert len(passages) == 16
    assert [item.passage_id for item in passages] == sorted(item.passage_id for item in passages)
    assert [item.passage_id for item in passages] == [
        item.passage_id for item in plan.provider_context.passages
    ]
    assert plan.validates_any_provider_visible_citation_not_exact_passage_roster


def test_both_v1_live_outputs_remain_invalid_and_scientifically_unsalvageable(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    require_private_cache(
        "data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json",
        "data/cache/metasyn/contextual-frontier-runtime-v1/provider-receipts",
    )
    terminal = json.loads((V1_WORKSPACE / "02-terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "roster_exhausted_without_typed_graph"
    assert len(terminal["validation_results"]) == 2
    assert all(
        item["status"] == "contextual_validation_failed_closed"
        for item in terminal["validation_results"]
    )

    parsed_by_key: dict[str, dict[str, Any]] = {}
    for path in sorted((V1_WORKSPACE / "provider-receipts").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        parsed = receipt["provider_result"]["parsed_json"]
        parsed_by_key[receipt["request_key"]] = parsed
        with pytest.raises(ValidationError):
            validate_json_schema(parsed, plan.request.response_schema)

    primary = {
        item["field_path"]: item["token"]
        for item in parsed_by_key["row17-candidate3-fable5-high"]["claims"]
    }
    fallback = {
        item["field_path"]: item["token"]
        for item in parsed_by_key["row17-candidate2-fable5-high"]["claims"]
    }
    assert primary["treatment_arm.label"] != "500-mg"
    assert fallback["finding.endpoint_marker"] != "primary end point"


def test_plan_is_zero_call_single_request_and_every_authority_is_false(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    assert plan.status == "offline_prepared_zero_provider_calls"
    assert plan.provider_calls_made == 0
    assert plan.maximum_provider_calls == 1
    assert plan.config.maximum_provider_calls == 1
    assert plan.predecessor_requests_retried == 0
    assert plan.hard_cost_liability_usd_micros == 11_600_000
    assert plan.diagnostic_known_surface_cost_usd_micros < (plan.hard_cost_liability_usd_micros)
    assert plan.request.transport_request.cost_ceiling.model_max_input_tokens == 1_000_000
    assert plan.request.transport_request.cost_ceiling.max_output_tokens == 32_000
    _assert_plan_grant_fields_false(plan)


def test_offline_fixture_completes_exact_four_number_grounding_without_empirical_authority(
    plan: recovery.MetaSynContextualFrontierRecoveryPlanV2,
) -> None:
    claims_by_field = {
        claim.field_path: {
            "passage_id": claim.passage_id,
            "support_quote": claim.support_quote,
            "context": claim.context,
            "token": claim.token,
            "normalization": claim.normalization,
        }
        for claim in plan.evaluator_fixture.model_outcome.claims
    }
    raw = {
        "response_version": "metasyn-contextual-frontier-recovery-response-v2",
        "status": "completed",
        "target_contract_sha256": plan.target_spec_sha256,
        "claims_by_field": claims_by_field,
    }
    evaluation = recovery.evaluate_metasyn_contextual_frontier_recovery_response_v2(
        plan=plan,
        raw_response=raw,
        provider_execution_binding_sha256="a" * 64,
    )
    assert evaluation.status == "typed_graph_mechanics_completed"
    assert evaluation.numeric_extraction_fields_evaluated == 4
    assert evaluation.numeric_evaluator_exact_match
    assert evaluation.extracted_numeric_values == {
        field_path: plan.evaluator_fixture.semantic_target.expected_normalized_values[field_path]
        for field_path in plan.target_spec.extraction_scored_field_paths
    }
    assert evaluation.native_projection is not None
    assert evaluation.grounded_effect is not None
    assert evaluation.typed_graph_mechanics_observed
    assert not evaluation.graph_construction_mechanics_authority
    assert not evaluation.extraction_accuracy_authority
    assert not evaluation.reliability_authority
    assert not evaluation.generalization_authority
    assert not evaluation.synthesis_input_authority
    assert not evaluation.scientific_synthesis_authority
    assert not evaluation.calibration_authority
    assert not evaluation.claim_release_authority
