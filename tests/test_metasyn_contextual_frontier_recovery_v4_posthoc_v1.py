from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tests.private_cache_support import require_private_cache

from literature_multiverse.lineage import sha256_file
from literature_multiverse.metasyn_contextual_frontier_recovery_v3 import _load_v2_plan
from literature_multiverse.metasyn_contextual_frontier_recovery_v4_posthoc_v1 import (
    DEFAULT_IMMUTABLE_WORKSPACE,
    EXPECTED_PLAN_FILE_SHA256,
    EXPECTED_RECEIPT_FILE_SHA256,
    EXPECTED_TERMINAL_FILE_SHA256,
    MetaSynContextualFrontierRecoveryV4PosthocError,
    _canonicalize_response,
    freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1,
    write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1,
)

ROOT = Path(__file__).resolve().parents[1]
IMMUTABLE = ROOT / DEFAULT_IMMUTABLE_WORKSPACE

pytestmark = pytest.mark.private_cache

_PRIVATE_CACHE_PATHS = (
    "data/cache/metasyn/contextual-frontier-recovery-v4",
    "data/cache/metasyn/contextual-frontier-recovery-v4-posthoc-v1",
)


def _immutable_hashes() -> tuple[str, str, str]:
    return (
        sha256_file(IMMUTABLE / "00-prepared.json"),
        sha256_file(IMMUTABLE / "provider-receipt.json"),
        sha256_file(IMMUTABLE / "02-terminal.json"),
    )


def _raw_response() -> dict[str, object]:
    receipt = json.loads((IMMUTABLE / "provider-receipt.json").read_text(encoding="utf-8"))
    return receipt["provider_result"]["parsed_json"]


def test_actual_posthoc_artifact_is_deterministic_and_mechanics_only(tmp_path: Path) -> None:
    require_private_cache(*_PRIVATE_CACHE_PATHS)
    before = _immutable_hashes()
    first = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=ROOT
    )
    second = write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=ROOT, workspace=tmp_path / "posthoc"
    )
    replay = write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=ROOT, workspace=tmp_path / "posthoc"
    )
    assert first == second == replay
    assert first.status == "typed_graph_mechanics_completed"
    assert first.recovery_label == "post_hoc_syntactic_canonicalization"
    assert first.canonicalizer_provider_calls_made == 0
    assert first.upstream_v4_provider_attempt_count == 1
    assert first.upstream_v4_provider_response_completed
    assert first.nested_projection_graph_mechanics_flag_dependency_only
    assert first.evaluation.native_projection.graph_construction_mechanics_authority
    assert not first.graph_construction_mechanics_authority
    assert first.blockers == ["post_hoc_source_span_repair"]
    assert first.original_context_containment_failure_count == 14
    assert first.original_invented_context_prefix_count == 1
    assert first.original_semantic_tuple_sha256 == first.canonical_semantic_tuple_sha256
    assert first.evaluation.runtime_pipeline_sha256 == (
        first.canonicalization_pipeline_sha256
    )
    assert first.evaluation.native_projection.runtime_pipeline_sha256 == (
        first.canonicalization_pipeline_sha256
    )
    assert first.evaluation.runtime_pipeline_sha256 != (
        first.evaluation.v3_evaluator_runtime_pipeline_sha256
    )
    assert {item.change_kind for item in first.canonicalization_changes} == {
        "minimal_local_context",
        "unicode_whitespace_exact_source_quote",
        "endpoint_marker_passage_binding",
        "endpoint_marker_quote_binding",
    }
    assert first.evaluation.numeric_evaluator_exact_match
    assert first.evaluation.extracted_numeric_values == {
        "effect.control_events": "1",
        "effect.control_total": "96",
        "effect.treatment_events": "39",
        "effect.treatment_total": "97",
    }
    assert all(
        not getattr(first, field)
        for field in (
            "graph_construction_mechanics_authority",
            "extraction_accuracy_authority",
            "reliability_authority",
            "generalization_authority",
            "synthesis_input_authority",
            "scientific_synthesis_authority",
            "scientific_effectiveness_authority",
            "calibration_authority",
            "claim_release_authority",
        )
    )
    assert _immutable_hashes() == before == (
        EXPECTED_PLAN_FILE_SHA256,
        EXPECTED_RECEIPT_FILE_SHA256,
        EXPECTED_TERMINAL_FILE_SHA256,
    )


def test_contexts_are_exact_deterministic_local_windows() -> None:
    require_private_cache(*_PRIVATE_CACHE_PATHS)
    artifact = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=ROOT
    )
    by_field = {item["field_path"]: item for item in artifact.canonicalized_response["claims"]}
    assert by_field["effect.control_events"]["context"] == "1 of 96"
    assert by_field["effect.control_total"]["context"] == "1 of 96"
    assert by_field["effect.treatment_events"]["context"] == "39 of 97"
    assert by_field["effect.treatment_total"]["context"] == "39 of 97"
    assert by_field["finding.timepoint.anchor"]["context"] == "week 24"
    assert by_field["finding.timepoint.value"]["context"] == "week 24"
    for field, claim in by_field.items():
        if field not in {
            "effect.control_events",
            "effect.control_total",
            "effect.treatment_events",
            "effect.treatment_total",
            "finding.timepoint.anchor",
            "finding.timepoint.value",
        }:
            assert claim["context"] == claim["token"]


def test_endpoint_repair_requires_strict_whitespace_only_equivalence() -> None:
    require_private_cache(*_PRIVATE_CACHE_PATHS)
    raw = deepcopy(_raw_response())
    raw["endpoint_quote"] += " changed"
    passages = {
        item.passage_id: item.text for item in _load_v2_plan(ROOT).provider_context.passages
    }
    with pytest.raises(
        MetaSynContextualFrontierRecoveryV4PosthocError,
        match="endpoint_quote_not_whitespace_equivalent",
    ):
        _canonicalize_response(raw_response=raw, passage_text_by_id=passages)


def test_artifact_validator_rejects_reordered_or_forged_transform_ledger() -> None:
    require_private_cache(*_PRIVATE_CACHE_PATHS)
    artifact = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=ROOT
    )
    reordered = artifact.model_dump(mode="json")
    reordered["canonicalization_changes"].reverse()
    with pytest.raises(ValueError, match="artifact_replay_mismatch"):
        type(artifact).model_validate(reordered)
    forged = artifact.model_dump(mode="json")
    forged["canonicalization_changes"][0]["json_pointer"] = "/claims/99/context"
    with pytest.raises(ValueError, match="artifact_replay_mismatch"):
        type(artifact).model_validate(forged)


def test_canonicalizer_does_not_consult_hidden_numeric_target() -> None:
    require_private_cache(*_PRIVATE_CACHE_PATHS)
    raw = deepcopy(_raw_response())
    for claim in raw["claims"]:
        if claim["field_path"] == "effect.treatment_events":
            claim["token"] = "35"
            claim["support_quote"] = "35 of 96 (36% [95% CI, 27%-46%])"
        elif claim["field_path"] == "effect.treatment_total":
            claim["token"] = "96"
            claim["support_quote"] = "35 of 96 (36% [95% CI, 27%-46%])"
    passages = {
        item.passage_id: item.text for item in _load_v2_plan(ROOT).provider_context.passages
    }
    canonical, _ = _canonicalize_response(
        raw_response=raw, passage_text_by_id=passages
    )
    by_field = {item["field_path"]: item for item in canonical["claims"]}
    assert by_field["effect.treatment_events"]["token"] == "35"
    assert by_field["effect.treatment_total"]["token"] == "96"
    assert by_field["effect.treatment_events"]["context"] == "35 of 96"
