from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.native_bounded_generation import (
    BoundedArm,
    BoundedCohortHeader,
    BoundedContrast,
    BoundedEvidence,
    BoundedFindingHeader,
    BoundedNumericSupport,
    BoundedStudyHeader,
    BoundedTimepoint,
    DirectStandardErrorEffect,
    NativeCandidateDescriptor,
    NativeCandidatePacket,
)
from literature_multiverse.native_bounded_ollama_diagnostic import (
    DEFAULT_CONFIG_PATH,
    NativeBoundedOllamaDiagnosticError,
    _classify_packet_response,
    finalize_bounded_diagnostic,
    freeze_bounded_prediction_ledger,
    prepare_bounded_input_bundle,
    run_bounded_prediction_stage,
    run_bounded_schema_compatibility_preflight,
    validate_bounded_finalized_artifacts_with_private_replay,
    validate_bounded_input_bundle,
    validate_bounded_public_summary,
    validate_current_bounded_context,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIGEST = "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b"


def _identity() -> OllamaIdentity:
    return OllamaIdentity(
        ollama_version="0.15.1",
        model="qwen2.5:3b-instruct",
        model_digest=MODEL_DIGEST,
        parameter_size="3.1B",
        quantization_level="Q4_K_M",
        model_format="gguf",
        model_family="qwen2",
    )


class FakeClient:
    def __init__(
        self,
        *,
        fail_on_generate_call: int | None = None,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.generate_calls = 0
        self.inspect_calls = 0
        self.fail_on_generate_call = fail_on_generate_call
        self.entered = entered
        self.release = release

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        self.inspect_calls += 1
        return _identity()

    def generate(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        del output_schema
        self.generate_calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=10)
        if self.fail_on_generate_call == self.generate_calls:
            raise LocalOllamaError("synthetic ambiguous transport failure")
        if "packet_version" in prompt and "unable_to_complete" in prompt:
            response = (
                '{"packet_version":"native-candidate-packet-v1",'
                '"packet_status":"unable_to_complete","candidate_index":1,'
                '"reason":"capacity_or_other_uncertainty"}'
            )
        else:
            response = (
                '{"inventory_version":"native-candidate-inventory-v1",'
                '"inventory_status":"no_candidate_found","candidates":[],'
                '"has_more_or_uncertain":false}'
            )
        return OllamaGenerationResult(
            model=config.model,
            response_text=response,
            done=True,
            done_reason="stop",
        )


class CandidateClient(FakeClient):
    def __init__(
        self,
        *,
        target_inventory_call: int,
        candidate: NativeCandidateDescriptor,
        packet: NativeCandidatePacket[DirectStandardErrorEffect],
        unable_packet: bool,
    ) -> None:
        super().__init__()
        self.target_inventory_call = target_inventory_call
        self.candidate = candidate
        self.packet = packet
        self.unable_packet = unable_packet

    def generate(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        del prompt, output_schema
        self.generate_calls += 1
        if self.generate_calls <= 19:
            if self.generate_calls == self.target_inventory_call:
                payload = {
                    "inventory_version": "native-candidate-inventory-v1",
                    "inventory_status": "candidates_found",
                    "candidates": [self.candidate.model_dump(mode="json")],
                    "has_more_or_uncertain": False,
                }
            else:
                payload = {
                    "inventory_version": "native-candidate-inventory-v1",
                    "inventory_status": "no_candidate_found",
                    "candidates": [],
                    "has_more_or_uncertain": False,
                }
        elif self.unable_packet:
            payload = {
                "packet_version": "native-candidate-packet-v1",
                "packet_status": "unable_to_complete",
                "candidate_index": 1,
                "reason": "insufficient_numeric_support",
            }
        else:
            payload = self.packet.model_dump(mode="json")
        return OllamaGenerationResult(
            model=config.model,
            response_text=json.dumps(payload, separators=(",", ":")),
            done=True,
            done_reason="stop",
        )


def _workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "inventory": workspace / "inventory-receipts",
        "packets": workspace / "packet-receipts",
        "intents": workspace / "pre-call-intents",
        "preflight": workspace / "schema-preflight",
        "ledger": workspace / "prediction-ledger.json",
    }


_SIMPLE_DECIMAL = re.compile(
    r"(?<![0-9A-Za-z.,])(-?(?:0|[1-9][0-9]{0,8})(?:\.[0-9]{1,8})?|"
    r"-?\.[0-9]{1,8})(?![0-9A-Za-z.,%])"
)


def _find_candidate_packet(
    bundle: dict[str, Any],
) -> tuple[int, NativeCandidateDescriptor, NativeCandidatePacket[DirectStandardErrorEffect]]:
    outcome = bundle["allowed_outcomes"][0]
    direction = bundle["outcome_positive_directions"][outcome]
    for row_index, row in enumerate(bundle["source_adapter"]["rows"], start=1):
        locator = row["source_record"]["source_document"]["source_locator"]
        for passage in row["source_projection"]:
            quote = passage["text"]
            matches = list(_SIMPLE_DECIMAL.finditer(quote))
            for estimate_match in matches:
                for standard_error_match in matches:
                    if estimate_match.span() == standard_error_match.span():
                        continue
                    try:
                        if Decimal(standard_error_match.group()) <= 0:
                            continue
                    except InvalidOperation:
                        continue
                    candidate = NativeCandidateDescriptor(
                        candidate_index=1,
                        outcome_name=outcome,
                        effect_kind="direct_standard_error",
                        line_ids=[passage["line_id"]],
                    )
                    try:
                        packet = NativeCandidatePacket[DirectStandardErrorEffect](
                            candidate_index=1,
                            study=BoundedStudyHeader(
                                key="study-1",
                                source_label="Study 1",
                                design="parallel controlled trial",
                                registration_ids=[],
                            ),
                            cohort=BoundedCohortHeader(
                                key="cohort-1",
                                source_labels=["Cohort 1"],
                                registry_ids=[],
                                dataset_ids=[],
                                population_description=None,
                                recruitment_period=None,
                                total_sample_size=None,
                            ),
                            treatment_arm=BoundedArm(
                                key="treatment",
                                label="Treatment",
                                role="intervention",
                                sample_size=None,
                            ),
                            comparator_arm=BoundedArm(
                                key="control",
                                label="Control",
                                role="control",
                                sample_size=None,
                            ),
                            contrast=BoundedContrast(
                                key="target",
                                label="treatment_vs_control",
                                estimand="between-group difference",
                                positive_direction_means=direction,
                            ),
                            finding=BoundedFindingHeader(
                                key="finding-1",
                                outcome_name=outcome,
                                timepoint=BoundedTimepoint(kind="not_reported"),
                                analysis_population=None,
                            ),
                            effect=DirectStandardErrorEffect(
                                effect_format="mean_difference",
                                estimate=estimate_match.group(),
                                standard_error=standard_error_match.group(),
                                unit=None,
                            ),
                            evidence=BoundedEvidence(
                                source_locator=locator,
                                quote=quote,
                                section=passage["section"],
                                line_ids=[passage["line_id"]],
                            ),
                            numeric_support=[
                                BoundedNumericSupport(
                                    field_path="effect.estimate",
                                    verbatim_token=estimate_match.group(),
                                    quote_start=str(estimate_match.start()),
                                    quote_end=str(estimate_match.end()),
                                ),
                                BoundedNumericSupport(
                                    field_path="effect.standard_error",
                                    verbatim_token=standard_error_match.group(),
                                    quote_start=str(standard_error_match.start()),
                                    quote_end=str(standard_error_match.end()),
                                ),
                            ],
                        )
                    except (ValidationError, ValueError):
                        continue
                    return row_index, candidate, packet
    raise AssertionError("frozen diagnostic projection has no two-token bounded packet")


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    return prepare_bounded_input_bundle(
        config_path=ROOT / DEFAULT_CONFIG_PATH,
        repository_root=ROOT,
    )


@pytest.fixture(scope="module")
def completed_run(
    tmp_path_factory: pytest.TempPathFactory,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    workspace = tmp_path_factory.mktemp("bounded-native-complete").resolve()
    paths = _workspace_paths(workspace)
    client = FakeClient()
    preflight = run_bounded_schema_compatibility_preflight(
        client=client,
        identity=_identity(),
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    ledger = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        client=client,
    )
    private, public = finalize_bounded_diagnostic(
        input_bundle=bundle,
        prediction_ledger=ledger,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
    )
    return {
        "workspace": workspace,
        "paths": paths,
        "preflight": preflight,
        "ledger": ledger,
        "private": private,
        "public": public,
        "client": client,
    }


def _rehash(value: dict[str, Any], field: str) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop(field, None)
    payload[field] = hash_canonical(payload)
    return payload


def test_prepare_freezes_exact_source_only_bundle_and_current_context(
    bundle: dict[str, Any],
) -> None:
    validated = validate_current_bounded_context(
        bundle,
        repository_root=ROOT,
        reverify_source_adapter=False,
    )

    assert validated["source_rows"] == 19
    assert validated["legacy_single_stage_generation_contract_authority"] is False
    assert validated["legacy_single_stage_receipts_accepted"] is False
    assert validated["prediction_stage_can_open_source_or_label_files"] is False
    assert len(validated["diagnostic_execution_identity"]["files"]) >= 47


def test_input_bundle_exact_fields_forbid_coherently_rehashed_label_smuggling(
    bundle: dict[str, Any],
) -> None:
    tampered = deepcopy(bundle)
    tampered["labels"] = ["hidden-answer"]
    tampered = _rehash(tampered, "input_bundle_sha256")

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="input_bundle_fields_invalid",
    ):
        validate_bounded_input_bundle(tampered)


def test_current_context_rederives_positive_direction_and_prompt(
    bundle: dict[str, Any],
) -> None:
    tampered = deepcopy(bundle)
    outcome = tampered["allowed_outcomes"][0]
    tampered["outcome_positive_directions"][outcome] = "reversed hidden semantics"
    tampered = _rehash(tampered, "input_bundle_sha256")

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="current_context_mismatch",
    ):
        validate_current_bounded_context(
            tampered,
            repository_root=ROOT,
            reverify_source_adapter=False,
        )


def test_coherent_source_row_rehash_cannot_cross_original_freeze_anchor(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    tampered = deepcopy(bundle)
    row = tampered["source_adapter"]["rows"][0]
    row["source_projection"][0]["text"] += " invented source suffix"
    row["source_projection"][0]["source_line_end_exclusive"] += len(
        " invented source suffix"
    )
    row["projected_characters"] += len(" invented source suffix")
    row["source_projection_sha256"] = hash_canonical(row["source_projection"])
    row_payload = deepcopy(row)
    row_payload.pop("input_row_sha256")
    row["input_row_sha256"] = hash_canonical(row_payload)
    adapter_payload = deepcopy(tampered["source_adapter"])
    adapter_payload.pop("source_adapter_sha256")
    tampered["source_adapter"]["source_adapter_sha256"] = hash_canonical(
        adapter_payload
    )
    tampered["source_adapter_sha256"] = tampered["source_adapter"][
        "source_adapter_sha256"
    ]
    tampered = _rehash(tampered, "input_bundle_sha256")
    paths = _workspace_paths(tmp_path.resolve())
    client = FakeClient()

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="input_bundle_freeze_anchor_mismatch",
    ):
        run_bounded_prediction_stage(
            input_bundle=tampered,
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=client,
        )

    assert client.inspect_calls == 0
    assert client.generate_calls == 0


def test_preflight_is_six_call_bundle_bound_and_idempotent(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    workspace = tmp_path.resolve()
    client = FakeClient()
    first = run_bounded_schema_compatibility_preflight(
        client=client,
        identity=_identity(),
        preflight_dir=workspace / "schema-preflight",
        bundle=bundle,
    )
    second = run_bounded_schema_compatibility_preflight(
        client=client,
        identity=_identity(),
        preflight_dir=workspace / "schema-preflight",
        bundle=bundle,
    )

    assert first == second
    assert first["synthetic_calls"] == 6
    assert first["input_bundle_sha256"] == bundle["input_bundle_sha256"]
    assert client.generate_calls == 6


def test_ambiguous_preflight_execution_poison_prevents_retry(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    workspace = tmp_path.resolve()
    first = FakeClient(fail_on_generate_call=1)
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="generation_transport_failure_no_receipt",
    ):
        run_bounded_schema_compatibility_preflight(
            client=first,
            identity=_identity(),
            preflight_dir=workspace / "schema-preflight",
            bundle=bundle,
        )
    second = FakeClient()
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="ambiguous_preflight_execution_workspace_poisoned",
    ):
        run_bounded_schema_compatibility_preflight(
            client=second,
            identity=_identity(),
            preflight_dir=workspace / "schema-preflight",
            bundle=bundle,
        )

    assert first.generate_calls == 1
    assert second.generate_calls == 0


def test_preflight_directory_rejects_unregistered_artifact_membership(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    preflight_dir = tmp_path.resolve() / "schema-preflight"
    client = FakeClient()
    run_bounded_schema_compatibility_preflight(
        client=client,
        identity=_identity(),
        preflight_dir=preflight_dir,
        bundle=bundle,
    )
    (preflight_dir / "legacy-one-stage-receipt.json").write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="preflight_directory_mixing_or_extra",
    ):
        run_bounded_schema_compatibility_preflight(
            client=client,
            identity=_identity(),
            preflight_dir=preflight_dir,
            bundle=bundle,
        )

    assert client.generate_calls == 6


def test_prediction_requires_completed_current_preflight_before_paper_call(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    paths = _workspace_paths(tmp_path.resolve())
    client = FakeClient()

    with pytest.raises(NativeBoundedOllamaDiagnosticError):
        run_bounded_prediction_stage(
            input_bundle=bundle,
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=client,
        )

    assert client.generate_calls == 0


def test_prediction_partial_complete_and_noop_resume_use_one_call_per_row(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    paths = _workspace_paths(tmp_path.resolve())
    client = FakeClient()
    run_bounded_schema_compatibility_preflight(
        client=client,
        identity=_identity(),
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    partial = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        client=client,
        inventory_limit=1,
    )
    complete = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        client=client,
    )
    resumed = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        client=client,
    )

    assert partial["inventory_receipts"] == 1
    assert complete["inventory_receipts"] == 19
    assert complete["all_expected_terminal_receipts_frozen"] is True
    assert resumed == complete
    assert client.generate_calls == 6 + 19


def test_ambiguous_paper_execution_poison_prevents_retry(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    paths = _workspace_paths(tmp_path.resolve())
    preflight_client = FakeClient()
    run_bounded_schema_compatibility_preflight(
        client=preflight_client,
        identity=_identity(),
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    failing = FakeClient(fail_on_generate_call=1)
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="generation_transport_failure_no_receipt",
    ):
        run_bounded_prediction_stage(
            input_bundle=bundle,
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=failing,
        )
    retry = FakeClient()
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="ambiguous_inventory_execution_workspace_poisoned",
    ):
        run_bounded_prediction_stage(
            input_bundle=bundle,
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=retry,
        )

    assert failing.generate_calls == 1
    assert retry.generate_calls == 0


def test_canonical_workspace_layout_rejects_alternate_lock_domains_before_calls(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    workspace = tmp_path.resolve()
    paths = _workspace_paths(workspace)
    client = FakeClient()

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="prediction_workspace_layout_invalid",
    ):
        run_bounded_prediction_stage(
            input_bundle=bundle,
            inventory_receipts_dir=workspace / "alternate-inventory",
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=client,
        )

    assert client.inspect_calls == 0
    assert client.generate_calls == 0


def test_concurrent_predictor_is_rejected_and_cannot_double_post(
    tmp_path: Path,
    bundle: dict[str, Any],
) -> None:
    paths = _workspace_paths(tmp_path.resolve())
    setup = FakeClient()
    run_bounded_schema_compatibility_preflight(
        client=setup,
        identity=_identity(),
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    entered = threading.Event()
    release = threading.Event()
    first = FakeClient(entered=entered, release=release)
    first_error: list[BaseException] = []

    def run_first() -> None:
        try:
            run_bounded_prediction_stage(
                input_bundle=bundle,
                inventory_receipts_dir=paths["inventory"],
                packet_receipts_dir=paths["packets"],
                attempt_intents_dir=paths["intents"],
                preflight_dir=paths["preflight"],
                prediction_ledger_path=paths["ledger"],
                repository_root=ROOT,
                expected_input_bundle_sha256=bundle["input_bundle_sha256"],
                client=first,
                inventory_limit=1,
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            first_error.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=10)
    second = FakeClient()
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="prediction_workspace_locked",
    ):
        run_bounded_prediction_stage(
            input_bundle=bundle,
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
            client=second,
        )
    release.set()
    worker.join(timeout=10)

    assert not first_error
    assert first.generate_calls == 1
    assert second.generate_calls == 0


def _packet_response(*, quote: str, locator: str, line_id: str) -> str:
    candidate = NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="outcome",
        effect_kind="direct_standard_error",
        line_ids=[line_id],
    )
    packet = NativeCandidatePacket[DirectStandardErrorEffect](
        candidate_index=1,
        study=BoundedStudyHeader(
            key="study-1",
            source_label="Study 1",
            design="parallel controlled trial",
            registration_ids=[],
        ),
        cohort=BoundedCohortHeader(
            key="cohort-1",
            source_labels=["Cohort 1"],
            registry_ids=[],
            dataset_ids=[],
            population_description=None,
            recruitment_period=None,
            total_sample_size=None,
        ),
        treatment_arm=BoundedArm(
            key="treatment", label="Treatment", role="intervention", sample_size=None
        ),
        comparator_arm=BoundedArm(
            key="control", label="Control", role="control", sample_size=None
        ),
        contrast=BoundedContrast(
            key="target",
            label="treatment_vs_control",
            estimand="between-group difference",
            positive_direction_means="larger outcome in treatment",
        ),
        finding=BoundedFindingHeader(
            key="finding-1",
            outcome_name="outcome",
            timepoint=BoundedTimepoint(kind="not_reported"),
            analysis_population=None,
        ),
        effect=DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate="0.5",
            standard_error="0.2",
            unit="units",
        ),
        evidence=BoundedEvidence(
            source_locator=locator,
            quote=quote,
            section="Results",
            line_ids=[line_id],
        ),
        numeric_support=[
            BoundedNumericSupport(
                field_path="effect.estimate",
                verbatim_token="0.5",
                quote_start=str(quote.index("0.5")),
                quote_end=str(quote.index("0.5") + 3),
            ),
            BoundedNumericSupport(
                field_path="effect.standard_error",
                verbatim_token="0.2",
                quote_start=str(quote.index("0.2")),
                quote_end=str(quote.index("0.2") + 3),
            ),
        ],
    )
    assert candidate.candidate_index == packet.candidate_index
    return packet.model_dump_json()


@pytest.mark.parametrize(
    ("source_text", "expected_status"),
    [
        ("The difference was 0.5 (SE 0.2).", "packet_completed"),
        ("No matching numerical sentence.", "packet_source_grounding_invalid"),
        (
            "The difference was 0.5 (SE 0.2). The difference was 0.5 (SE 0.2).",
            "packet_source_grounding_invalid",
        ),
    ],
)
def test_packet_quote_must_match_exactly_one_cited_frozen_passage(
    source_text: str,
    expected_status: str,
) -> None:
    line_id = "L1"
    locator = "synthetic:source"
    quote = "The difference was 0.5 (SE 0.2)."
    candidate = NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="outcome",
        effect_kind="direct_standard_error",
        line_ids=[line_id],
    )
    row = {
        "source_projection": [
            {
                "line_id": line_id,
                "passage_rank": 1,
                "section": "Results",
                "source_line_start": 100,
                "text": source_text,
            }
        ],
        "source_record": {"source_document": {"source_locator": locator}},
    }
    bundle = {
        "allowed_outcomes": ["outcome"],
        "outcome_positive_directions": {"outcome": "larger outcome in treatment"},
        "allowed_moderators": [],
        "allowed_sections": ["Results"],
    }
    result = OllamaGenerationResult(
        model="qwen2.5:3b-instruct",
        response_text=_packet_response(quote=quote, locator=locator, line_id=line_id),
        done=True,
        done_reason="stop",
    )

    status, _, validated, _, grounding = _classify_packet_response(
        result=result,
        bundle=bundle,
        row=row,
        candidate=candidate,
    )

    assert status == expected_status
    if expected_status == "packet_completed":
        assert validated is not None
        assert grounding is not None
        assert grounding["source_line_char_start"] == 100
        assert grounding["passage_utf8_start"] == 0
    else:
        assert validated is None
        assert grounding is None


def test_ledger_constructor_rejects_extra_receipt_keys(
    bundle: dict[str, Any],
    completed_run: dict[str, Any],
) -> None:
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="extra_inventory_receipt",
    ):
        freeze_bounded_prediction_ledger(
            bundle=bundle,
            identity=_identity(),
            inventory_receipts={"f" * 64: {}},
            packet_receipts={},
            preflight_receipt=completed_run["preflight"],
        )


def test_public_summary_is_strict_content_silent_and_truthful(
    completed_run: dict[str, Any],
) -> None:
    public = completed_run["public"]
    validated = validate_bounded_public_summary(public, repository_root=ROOT)

    assert validated["generation"]["synthetic_preflight_calls"] == 6
    assert (
        validated["generation"]["synthetic_preflight_status"]
        == "passed_and_bound_before_paper_calls"
    )
    assert validated["generation"]["paper_generation_calls"] == 19
    assert validated["official_native_v1_estimable_publications"] == 0
    assert validated["empirical_counts_require_private_receipt_replay"] is True


@pytest.mark.parametrize("attack", ["extra_field", "float_count", "identity_smuggle"])
def test_public_summary_rejects_coherently_rehashed_attacks(
    completed_run: dict[str, Any],
    attack: str,
) -> None:
    tampered = deepcopy(completed_run["public"])
    if attack == "extra_field":
        tampered["model"]["abstract_text"] = "source-bearing payload"
    elif attack == "float_count":
        key = next(iter(tampered["inventory_status_counts"]))
        tampered["inventory_status_counts"][key] = 19.0
    else:
        tampered["model"]["identity"]["client_version"] = "source-bearing payload"
        identity_payload = deepcopy(tampered["model"]["identity"])
        identity_payload.pop("identity_sha256")
        tampered["model"]["identity"]["identity_sha256"] = hash_canonical(
            identity_payload
        )
    tampered = _rehash(tampered, "summary_sha256")

    with pytest.raises(NativeBoundedOllamaDiagnosticError):
        validate_bounded_public_summary(tampered, repository_root=ROOT)


def test_private_replay_recomputes_every_empirical_aggregate(
    bundle: dict[str, Any],
    completed_run: dict[str, Any],
) -> None:
    paths = completed_run["paths"]
    private, public = validate_bounded_finalized_artifacts_with_private_replay(
        input_bundle=bundle,
        prediction_ledger=completed_run["ledger"],
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        private_report=completed_run["private"],
        public_summary=completed_run["public"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
    )

    assert private == completed_run["private"]
    assert public == completed_run["public"]


def test_private_replay_rejects_coherently_rehashed_private_report(
    bundle: dict[str, Any],
    completed_run: dict[str, Any],
) -> None:
    paths = completed_run["paths"]
    tampered = deepcopy(completed_run["private"])
    tampered["official_native_v1_findings"] = 1
    tampered = _rehash(tampered, "private_report_sha256")

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="private_report_full_replay_mismatch",
    ):
        validate_bounded_finalized_artifacts_with_private_replay(
            input_bundle=bundle,
            prediction_ledger=completed_run["ledger"],
            inventory_receipts_dir=paths["inventory"],
            packet_receipts_dir=paths["packets"],
            attempt_intents_dir=paths["intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["ledger"],
            private_report=tampered,
            public_summary=completed_run["public"],
            repository_root=ROOT,
            expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        )


@pytest.mark.parametrize(
    ("unable_packet", "expected_promoted"),
    [(False, 1), (True, 0)],
)
def test_candidate_packet_finalizer_promotes_only_complete_grounded_publication(
    tmp_path: Path,
    bundle: dict[str, Any],
    unable_packet: bool,
    expected_promoted: int,
) -> None:
    row_index, candidate, packet = _find_candidate_packet(bundle)
    paths = _workspace_paths(tmp_path.resolve())
    preflight_client = FakeClient()
    run_bounded_schema_compatibility_preflight(
        client=preflight_client,
        identity=_identity(),
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    client = CandidateClient(
        target_inventory_call=row_index,
        candidate=candidate,
        packet=packet,
        unable_packet=unable_packet,
    )
    ledger = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
        client=client,
    )
    private, public = finalize_bounded_diagnostic(
        input_bundle=bundle,
        prediction_ledger=ledger,
        inventory_receipts_dir=paths["inventory"],
        packet_receipts_dir=paths["packets"],
        attempt_intents_dir=paths["intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["ledger"],
        repository_root=ROOT,
        expected_input_bundle_sha256=bundle["input_bundle_sha256"],
    )

    target_row_key = bundle["source_adapter"]["rows"][row_index - 1]["row_key"]
    target_result = next(
        item for item in private["rows"] if item["row_key"] == target_row_key
    )
    assert preflight_client.generate_calls == 6
    assert client.generate_calls == 20
    assert ledger["inventory_status_counts"] == {
        "inventory_below_cap": 1,
        "inventory_no_candidate_non_authorizing": 18,
    }
    assert public["official_native_v1_estimable_publications"] == expected_promoted
    assert public["official_native_v1_findings"] == expected_promoted
    assert private["partial_packet_salvage_count"] == 0
    assert sum(item["official_output"] is not None for item in private["rows"]) == (
        expected_promoted
    )
    if unable_packet:
        assert ledger["packet_status_counts"] == {"packet_unable_to_complete": 1}
        assert target_result["official_output"] is None
        assert target_result["status"] == (
            "packet_set_non_authorizing:packet_unable_to_complete"
        )
    else:
        assert ledger["packet_status_counts"] == {"packet_completed": 1}
        assert target_result["status"] == "official_native_v1_estimable"
        assert target_result["official_output"]["extraction_schema_version"] == (
            "native-publication-extraction-v1"
        )
        assert target_result["official_output"]["status"] == "estimable"


def test_cli_requires_staged_anchor_and_has_no_all_in_one_command() -> None:
    from scripts.run_native_bounded_ollama_diagnostic import build_parser

    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices

    assert set(commands) == {
        "prepare",
        "preflight",
        "predict",
        "finalize",
        "validate-public",
        "validate-private",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["preflight"])
    with pytest.raises(SystemExit):
        parser.parse_args(["predict"])


def test_prepare_freshness_rejects_unknown_workspace_contamination(
    tmp_path: Path,
) -> None:
    from scripts.run_native_bounded_ollama_diagnostic import (
        _assert_fresh_prepare_workspace,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "unknown-label-cache.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="prepare_requires_fresh_workspace",
    ):
        _assert_fresh_prepare_workspace(workspace)


def test_finalize_pair_write_is_idempotent_and_recovers_missing_counterpart(
    tmp_path: Path,
) -> None:
    from scripts.run_native_bounded_ollama_diagnostic import (
        _write_or_validate_exact,
    )

    private_path = tmp_path / "private.json"
    public_path = tmp_path / "public.json"
    private = {"artifact": "private", "sha256": "a" * 64}
    public = {"artifact": "public", "sha256": "b" * 64}

    _write_or_validate_exact(
        private_path, private, code="existing_private_report_mismatch"
    )
    _write_or_validate_exact(public_path, public, code="existing_public_summary_mismatch")
    _write_or_validate_exact(
        private_path, private, code="existing_private_report_mismatch"
    )
    _write_or_validate_exact(public_path, public, code="existing_public_summary_mismatch")

    assert private_path.is_file()
    assert public_path.is_file()
    with pytest.raises(
        NativeBoundedOllamaDiagnosticError,
        match="existing_public_summary_mismatch",
    ):
        _write_or_validate_exact(
            public_path,
            {"artifact": "forged"},
            code="existing_public_summary_mismatch",
        )
