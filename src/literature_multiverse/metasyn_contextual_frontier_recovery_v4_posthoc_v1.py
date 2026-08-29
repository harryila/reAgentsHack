"""Offline, source-span-only canonicalization of the immutable recovery-v4 result.

The provider returned the exact field roster and target values, but some local
contexts extended beyond their support quotes and the redundant endpoint marker
was bound to the endpoint-definition passage rather than the endpoint-results
passage named by ``endpoint_passage_id``.  This module permits only a deterministic
repair of those source-span mechanics.  It does not call a provider and grants no
accuracy, generalization, synthesis, calibration, or release authority.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import ValidationError
from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.contextual_numeric_grounding_v3 import (
    _runtime_native_projection_from_fixture,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_recovery_v3 import _load_v2_plan
from literature_multiverse.metasyn_contextual_frontier_recovery_v4 import (
    MetaSynContextualFrontierRecoveryCoreEvaluationV4,
    MetaSynContextualFrontierRecoveryPlanV4,
    MetaSynContextualFrontierRecoveryReceiptV4,
    MetaSynContextualFrontierRecoveryTerminalV4,
    evaluate_metasyn_contextual_frontier_recovery_response_v4,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    _assert_secret_free,
    _persist_json,
)
from literature_multiverse.models import SHA256_RE, ContractModel

ARTIFACT_VERSION = "metasyn-contextual-frontier-recovery-v4-posthoc-artifact-v1"
RECOVERY_LABEL = "post_hoc_syntactic_canonicalization"
DEFAULT_IMMUTABLE_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-recovery-v4")
DEFAULT_WORKSPACE = Path(
    "data/cache/metasyn/contextual-frontier-recovery-v4-posthoc-v1"
)
RUNTIME_SOURCE_PATH = Path(
    "src/literature_multiverse/metasyn_contextual_frontier_recovery_v4_posthoc_v1.py"
)

EXPECTED_PLAN_SHA256 = "5b504b4f7bad1742ec6a773289141835715f05bb63454ab3e61026f25bd012c8"
EXPECTED_PLAN_FILE_SHA256 = (
    "936ac25e5ffdead2b5246126c2ff38632e8999adb5b45850e17753e94e707b86"
)
EXPECTED_TERMINAL_SHA256 = (
    "9c1bd812915afd5527585c389559f6716f868c02bbd8be1faf360cf8de158986"
)
EXPECTED_TERMINAL_FILE_SHA256 = (
    "e9a87352f85b9aafa826abdc59ccc575dfe818f1adf5040d94b1111a01d47d3f"
)
EXPECTED_RECEIPT_SHA256 = (
    "8ebe84f61a2136e79e06fb3fc2dd6b02462b5213df892fd0830840c571405bef"
)
EXPECTED_RECEIPT_FILE_SHA256 = (
    "c6d6b80dbf3dc83843fd86f1cd6199b5461c4a2aa4e001a191e32da77b6dd0e8"
)
EXPECTED_PROVIDER_RESULT_SHA256 = (
    "4cf8d252b609c528e6638c5cbf57431639f2331cfee366cafa4ed6a17e148165"
)
EXPECTED_ORIGINAL_RESPONSE_SHA256 = (
    "ee3229492abcc677e540d91522d6b0536e37ab2e06962a11c898ef50e6ff48a9"
)

_NUMERIC_FIELDS = {
    "effect.control_events",
    "effect.control_total",
    "effect.treatment_events",
    "effect.treatment_total",
}


class MetaSynContextualFrontierRecoveryV4PosthocError(ValueError):
    """The immutable inputs drifted or a syntactic-only repair was impossible."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(model.model_dump(mode="json", exclude={field})):
        raise ValueError(code)


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_source_artifact_unsafe"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_source_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_source_artifact_invalid"
        )
    _assert_secret_free(value)
    return value


def _unicode_whitespace_normalized(value: str) -> str:
    """Collapse all Unicode whitespace while preserving every non-whitespace codepoint."""

    return " ".join(value.split())


def _semantic_projection(response: dict[str, Any]) -> dict[str, Any]:
    claims = response.get("claims")
    if not isinstance(claims, list):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_claims_not_array"
        )
    return {
        "candidate_binding_sha256": response.get("candidate_binding_sha256"),
        "canonical_outcome_id": response.get("canonical_outcome_id"),
        "effect_kind": response.get("effect_kind"),
        "effect_format_token": response.get("effect_format_token"),
        "effect_computation": response.get("effect_computation"),
        "source_scope_acknowledgement": response.get("source_scope_acknowledgement"),
        "claims": [
            {
                "field_path": item.get("field_path"),
                "token": item.get("token"),
                "normalization": item.get("normalization"),
            }
            for item in claims
            if isinstance(item, dict)
        ],
        "numeric_values": {
            item.get("field_path"): item.get("token")
            for item in claims
            if isinstance(item, dict) and item.get("field_path") in _NUMERIC_FIELDS
        },
    }


def _recursively_true_authority_paths(
    value: Any, prefix: tuple[str, ...] = ()
) -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            if key.endswith("_authority") and child is True:
                matches.append(".".join(path))
            matches.extend(_recursively_true_authority_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_recursively_true_authority_paths(child, (*prefix, str(index))))
    return sorted(matches)


class MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1(_Frozen):
    json_pointer: str
    change_kind: Literal[
        "minimal_local_context",
        "unicode_whitespace_exact_source_quote",
        "endpoint_marker_passage_binding",
        "endpoint_marker_quote_binding",
    ]
    before_sha256: Sha256
    after_sha256: Sha256

    @model_validator(mode="after")
    def validate_change(
        self,
    ) -> MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1:
        if self.before_sha256 == self.after_sha256:
            raise ValueError("recovery_v4_posthoc_noop_ledger_entry")
        return self


class MetaSynContextualFrontierRecoveryV4PosthocArtifactV1(_Frozen):
    artifact_version: Literal[
        "metasyn-contextual-frontier-recovery-v4-posthoc-artifact-v1"
    ] = ARTIFACT_VERSION
    recovery_label: Literal["post_hoc_syntactic_canonicalization"] = RECOVERY_LABEL
    status: Literal["typed_graph_mechanics_completed"] = "typed_graph_mechanics_completed"
    offline_only: Literal[True] = True
    canonicalizer_provider_calls_made: Literal[0] = 0
    upstream_v4_provider_attempt_count: Literal[1] = 1
    upstream_v4_provider_response_completed: Literal[True] = True
    immutable_v4_plan_sha256: Sha256
    immutable_v4_plan_file_sha256: Sha256
    immutable_v4_terminal_sha256: Sha256
    immutable_v4_terminal_file_sha256: Sha256
    immutable_v4_receipt_sha256: Sha256
    immutable_v4_receipt_file_sha256: Sha256
    provider_result_sha256: Sha256
    original_response_sha256: Sha256
    canonicalized_response_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    canonicalizer_source_sha256: Sha256
    canonicalization_pipeline_sha256: Sha256
    v4_evaluator_dependency_sha256: Sha256
    canonicalization_changes: list[
        MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1
    ]
    canonicalization_change_membership_sha256: Sha256
    canonicalized_response: dict[str, Any]
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV4
    evaluation_sha256: Sha256
    original_semantic_tuple_sha256: Sha256
    canonical_semantic_tuple_sha256: Sha256
    original_context_containment_failure_count: Literal[14] = 14
    original_invented_context_prefix_count: Literal[1] = 1
    blockers: list[Literal["post_hoc_source_span_repair"]] = Field(
        default_factory=lambda: ["post_hoc_source_span_repair"]
    )
    raw_schema_validated_before_repair: Literal[True] = True
    exact_roster_and_order_validated_before_repair: Literal[True] = True
    no_target_or_gold_token_injection: Literal[True] = True
    immutable_transform_ledger: Literal[True] = True
    endpoint_quote_ascii_provider_to_exact_u2009_source: Literal[True] = True
    endpoint_source_match_unique: Literal[True] = True
    endpoint_marker_only_passage_id_change: Literal[True] = True
    grounding_uniqueness_and_offsets_recomputed: Literal[True] = True
    binary_pair_structure_and_arm_source_spans_verified: Literal[True] = True
    hidden_numeric_equality_not_consulted_by_repair: Literal[True] = True
    nested_projection_graph_mechanics_flag_dependency_only: Literal[True] = True
    field_set_unchanged: Literal[True] = True
    tokens_unchanged: Literal[True] = True
    normalizations_unchanged: Literal[True] = True
    numeric_values_unchanged: Literal[True] = True
    arm_outcome_semantics_unchanged: Literal[True] = True
    source_span_only_changes_validated: Literal[True] = True
    endpoint_quote_unicode_whitespace_equivalent: Literal[True] = True
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> MetaSynContextualFrontierRecoveryV4PosthocArtifactV1:
        claims = self.canonicalized_response.get("claims")
        if not isinstance(claims, list) or len(claims) != 15:
            raise ValueError("recovery_v4_posthoc_artifact_claim_roster_invalid")
        markers = [
            index
            for index, claim in enumerate(claims)
            if isinstance(claim, dict) and claim.get("field_path") == "finding.endpoint_marker"
        ]
        if len(markers) != 1:
            raise ValueError("recovery_v4_posthoc_artifact_endpoint_marker_invalid")
        marker_index = markers[0]
        permitted_changes = {
            "/endpoint_quote": "unicode_whitespace_exact_source_quote",
            f"/claims/{marker_index}/passage_id": "endpoint_marker_passage_binding",
            f"/claims/{marker_index}/support_quote": "endpoint_marker_quote_binding",
            **{
                f"/claims/{index}/context": "minimal_local_context"
                for index in range(len(claims))
            },
        }
        observed_changes = {
            item.json_pointer: item.change_kind for item in self.canonicalization_changes
        }
        pointers = [item.json_pointer for item in self.canonicalization_changes]
        true_authority_paths = _recursively_true_authority_paths(
            self.model_dump(mode="json")
        )
        if (
            len(self.canonicalization_changes) != 18
            or len(observed_changes) != len(self.canonicalization_changes)
            or pointers != sorted(pointers)
            or observed_changes != permitted_changes
            or self.canonicalized_response_sha256
            != hash_canonical(self.canonicalized_response)
            or self.canonicalization_change_membership_sha256
            != hash_canonical(
                [item.model_dump(mode="json") for item in self.canonicalization_changes]
            )
            or self.evaluation_sha256 != self.evaluation.evaluation_sha256
            or self.evaluation.response.model_dump(mode="json") != self.canonicalized_response
            or self.evaluation.status != "typed_graph_mechanics_completed"
            or self.evaluation.provider_execution_binding_sha256
            != self.provider_execution_binding_sha256
            or self.evaluation.plan_sha256 != self.immutable_v4_plan_sha256
            or self.evaluation.runtime_pipeline_sha256
            != self.canonicalization_pipeline_sha256
            or self.evaluation.native_projection.runtime_pipeline_sha256
            != self.canonicalization_pipeline_sha256
            or self.evaluation.native_projection.fragment is None
            or self.evaluation.native_projection.fragment.pipeline_fingerprint_sha256
            != self.canonicalization_pipeline_sha256
            or self.original_semantic_tuple_sha256 != self.canonical_semantic_tuple_sha256
            or self.blockers != ["post_hoc_source_span_repair"]
            or true_authority_paths
            != ["evaluation.native_projection.graph_construction_mechanics_authority"]
        ):
            raise ValueError("recovery_v4_posthoc_artifact_replay_mismatch")
        _self_hash(self, "artifact_sha256", "recovery_v4_posthoc_artifact_hash_mismatch")
        return self


def _record_change(
    *, pointer: str, kind: str, before: Any, after: Any
) -> MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1 | None:
    if before == after:
        return None
    return MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1(
        json_pointer=pointer,
        change_kind=kind,
        before_sha256=hash_canonical(before),
        after_sha256=hash_canonical(after),
    )


def _canonicalize_response(
    *, raw_response: dict[str, Any], passage_text_by_id: dict[str, str]
) -> tuple[
    dict[str, Any], list[MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1]
]:
    original = deepcopy(raw_response)
    canonical = deepcopy(raw_response)
    original_projection = _semantic_projection(original)
    claims = canonical.get("claims")
    if not isinstance(claims, list) or len(claims) != 15 or not all(
        isinstance(item, dict) for item in claims
    ):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_exact_claim_roster_missing"
        )
    by_field = {item["field_path"]: item for item in claims}
    if len(by_field) != len(claims):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_duplicate_field_path"
        )

    endpoint_id = canonical.get("endpoint_passage_id")
    provider_endpoint_quote = canonical.get("endpoint_quote")
    exact_endpoint_text = passage_text_by_id.get(endpoint_id)
    if not isinstance(provider_endpoint_quote, str) or not isinstance(exact_endpoint_text, str):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_endpoint_source_missing"
        )
    endpoint_matches = [
        passage_id
        for passage_id, text in passage_text_by_id.items()
        if text.replace("\u2009", " ") == provider_endpoint_quote
    ]
    if (
        _unicode_whitespace_normalized(provider_endpoint_quote)
        != _unicode_whitespace_normalized(exact_endpoint_text)
        or exact_endpoint_text.replace("\u2009", " ") != provider_endpoint_quote
        or endpoint_matches != [endpoint_id]
    ):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_endpoint_quote_not_whitespace_equivalent"
        )

    changes: list[MetaSynContextualFrontierRecoveryV4CanonicalizationChangeV1] = []
    change = _record_change(
        pointer="/endpoint_quote",
        kind="unicode_whitespace_exact_source_quote",
        before=canonical["endpoint_quote"],
        after=exact_endpoint_text,
    )
    if change is not None:
        changes.append(change)
    canonical["endpoint_quote"] = exact_endpoint_text

    marker = by_field.get("finding.endpoint_marker")
    if marker is None:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_endpoint_marker_missing"
        )
    original_marker_passage = passage_text_by_id.get(marker["passage_id"])
    if (
        not isinstance(original_marker_passage, str)
        or original_marker_passage.count(marker["support_quote"]) != 1
        or marker["support_quote"].count(marker["token"]) != 1
        or exact_endpoint_text.count(marker["token"]) != 1
    ):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_endpoint_marker_token_not_exact_in_both_passages"
        )
    for key, value, kind in (
        ("passage_id", endpoint_id, "endpoint_marker_passage_binding"),
        ("support_quote", exact_endpoint_text, "endpoint_marker_quote_binding"),
    ):
        change = _record_change(
            pointer=f"/claims/{claims.index(marker)}/{key}",
            kind=kind,
            before=marker[key],
            after=value,
        )
        if change is not None:
            changes.append(change)
        marker[key] = value

    numeric_context = {
        "effect.control_events": (
            f"{by_field['effect.control_events']['token']} of "
            f"{by_field['effect.control_total']['token']}"
        ),
        "effect.control_total": (
            f"{by_field['effect.control_events']['token']} of "
            f"{by_field['effect.control_total']['token']}"
        ),
        "effect.treatment_events": (
            f"{by_field['effect.treatment_events']['token']} of "
            f"{by_field['effect.treatment_total']['token']}"
        ),
        "effect.treatment_total": (
            f"{by_field['effect.treatment_events']['token']} of "
            f"{by_field['effect.treatment_total']['token']}"
        ),
    }
    for index, claim in enumerate(claims):
        field_path = claim["field_path"]
        if field_path in numeric_context:
            context = numeric_context[field_path]
        elif field_path in {"finding.timepoint.anchor", "finding.timepoint.value"}:
            if (
                by_field["finding.timepoint.anchor"]["token"] != "week"
                or by_field["finding.timepoint.value"]["token"] != "24"
            ):
                raise MetaSynContextualFrontierRecoveryV4PosthocError(
                    "recovery_v4_posthoc_timepoint_semantic_drift"
                )
            context = "week 24"
        else:
            context = claim["token"]
        support_quote = claim["support_quote"]
        token = claim["token"]
        passage_text = passage_text_by_id.get(claim["passage_id"])
        if (
            not isinstance(support_quote, str)
            or not isinstance(context, str)
            or not isinstance(token, str)
            or not isinstance(passage_text, str)
            or passage_text.count(support_quote) != 1
            or support_quote.count(context) != 1
            or context.count(token) != 1
        ):
            raise MetaSynContextualFrontierRecoveryV4PosthocError(
                f"recovery_v4_posthoc_local_context_not_exact:{field_path}"
            )
        change = _record_change(
            pointer=f"/claims/{index}/context",
            kind="minimal_local_context",
            before=claim["context"],
            after=context,
        )
        if change is not None:
            changes.append(change)
        claim["context"] = context

    if _semantic_projection(canonical) != original_projection:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_semantic_projection_changed"
        )
    for key in original:
        if key not in {"claims", "endpoint_quote"} and canonical[key] != original[key]:
            raise MetaSynContextualFrontierRecoveryV4PosthocError(
                "recovery_v4_posthoc_top_level_semantic_change"
            )
    original_claims = original["claims"]
    for before, after in zip(original_claims, claims, strict=True):
        allowed = {"context"}
        if before["field_path"] == "finding.endpoint_marker":
            allowed.update({"passage_id", "support_quote"})
        for key in before:
            if key not in allowed and before[key] != after[key]:
                raise MetaSynContextualFrontierRecoveryV4PosthocError(
                    f"recovery_v4_posthoc_claim_semantic_change:{before['field_path']}:{key}"
                )
    for prefix in ("treatment", "control"):
        events_token = by_field[f"effect.{prefix}_events"]["token"]
        total_token = by_field[f"effect.{prefix}_total"]["token"]
        if not (
            isinstance(events_token, str)
            and isinstance(total_token, str)
            and events_token.isascii()
            and total_token.isascii()
            and events_token.isdigit()
            and total_token.isdigit()
            and 0 <= int(events_token) <= int(total_token)
            and int(total_token) > 0
        ):
            raise MetaSynContextualFrontierRecoveryV4PosthocError(
                f"recovery_v4_posthoc_binary_pair_structure_invalid:{prefix}"
            )
    return canonical, changes


def freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
    *,
    repository_root: Path,
    immutable_workspace: Path | None = None,
) -> MetaSynContextualFrontierRecoveryV4PosthocArtifactV1:
    root = repository_root.resolve(strict=True)
    source_workspace = (
        (root / DEFAULT_IMMUTABLE_WORKSPACE)
        if immutable_workspace is None
        else immutable_workspace
    ).resolve(strict=True)
    plan_path = source_workspace / "00-prepared.json"
    receipt_path = source_workspace / "provider-receipt.json"
    terminal_path = source_workspace / "02-terminal.json"
    observed_files = (
        sha256_file(plan_path),
        sha256_file(receipt_path),
        sha256_file(terminal_path),
    )
    if observed_files != (
        EXPECTED_PLAN_FILE_SHA256,
        EXPECTED_RECEIPT_FILE_SHA256,
        EXPECTED_TERMINAL_FILE_SHA256,
    ):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_immutable_file_drift"
        )
    plan = MetaSynContextualFrontierRecoveryPlanV4.model_validate(_read_object(plan_path))
    receipt = MetaSynContextualFrontierRecoveryReceiptV4.model_validate(
        _read_object(receipt_path)
    )
    terminal = MetaSynContextualFrontierRecoveryTerminalV4.model_validate(
        _read_object(terminal_path)
    )
    result = receipt.provider_result
    if (
        plan.plan_sha256 != EXPECTED_PLAN_SHA256
        or terminal.terminal_sha256 != EXPECTED_TERMINAL_SHA256
        or receipt.receipt_sha256 != EXPECTED_RECEIPT_SHA256
        or result.result_sha256 != EXPECTED_PROVIDER_RESULT_SHA256
        or result.parsed_json_sha256 != EXPECTED_ORIGINAL_RESPONSE_SHA256
        or terminal.provider_receipt != receipt
        or terminal.status != "contextual_validation_failed_closed"
        or terminal.plan_sha256 != plan.plan_sha256
        or receipt.plan_sha256 != plan.plan_sha256
        or result.outcome != "completed"
        or result.parsed_json is None
    ):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_immutable_semantic_drift"
        )
    v2 = _load_v2_plan(root)
    if v2.provider_context_sha256 != plan.provider_context_sha256:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_provider_context_drift"
        )
    passage_text_by_id = {item.passage_id: item.text for item in v2.provider_context.passages}
    if len(passage_text_by_id) != len(v2.provider_context.passages):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_duplicate_passage_id"
        )
    original = deepcopy(result.parsed_json)
    try:
        validate_json_schema(original, plan.request.compiled_schema.original_schema)
    except ValidationError as exc:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_raw_schema_invalid"
        ) from exc
    observed_roster = [item.get("field_path") for item in original.get("claims", [])]
    required_roster = [item.field_path for item in plan.target_spec.fields]
    if observed_roster != required_roster or observed_roster != sorted(set(observed_roster)):
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_raw_roster_or_order_invalid"
        )
    original_containment_failures = sum(
        item["support_quote"].count(item["context"]) != 1 for item in original["claims"]
    )
    original_invented_prefixes = sum(
        _unicode_whitespace_normalized(passage_text_by_id[item["passage_id"]]).count(
            _unicode_whitespace_normalized(item["context"])
        )
        != 1
        for item in original["claims"]
    )
    if original_containment_failures != 14 or original_invented_prefixes != 1:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_failure_diagnosis_drift"
        )
    canonical, changes = _canonicalize_response(
        raw_response=original, passage_text_by_id=passage_text_by_id
    )
    provider_execution_binding = hash_canonical(
        {
            "plan_sha256": plan.plan_sha256,
            "authorization_sha256": receipt.authorization_sha256,
            "intent_sha256": receipt.intent_sha256,
            "receipt_sha256": receipt.receipt_sha256,
            "provider_result_sha256": receipt.provider_result_sha256,
        }
    )
    v4_evaluation = evaluate_metasyn_contextual_frontier_recovery_response_v4(
        repository_root=root,
        plan=plan,
        raw_response=canonical,
        provider_execution_binding_sha256=provider_execution_binding,
    )
    source_sha = sha256_file(root / RUNTIME_SOURCE_PATH)
    changes.sort(key=lambda item: item.json_pointer)
    change_membership_sha = hash_canonical(
        [item.model_dump(mode="json") for item in changes]
    )
    canonical_response_sha = hash_canonical(canonical)
    pipeline = hash_canonical(
        {
            "canonicalizer_source_sha256": source_sha,
            "immutable_v4_plan_sha256": plan.plan_sha256,
            "immutable_v4_plan_file_sha256": observed_files[0],
            "immutable_v4_terminal_sha256": terminal.terminal_sha256,
            "immutable_v4_terminal_file_sha256": observed_files[2],
            "immutable_v4_receipt_sha256": receipt.receipt_sha256,
            "immutable_v4_receipt_file_sha256": observed_files[1],
            "provider_result_sha256": result.result_sha256,
            "original_response_sha256": result.parsed_json_sha256,
            "canonicalized_response_sha256": canonical_response_sha,
            "canonicalization_change_membership_sha256": change_membership_sha,
            "v4_evaluator_dependency_sha256": v4_evaluation.evaluation_sha256,
            "recovery_label": RECOVERY_LABEL,
        }
    )
    projection = _runtime_native_projection_from_fixture(
        fixture_receipt=v2.evaluator_fixture,
        effect=v4_evaluation.grounded_effect,
        groundings=v4_evaluation.groundings,
        grounding_core_sha256=v4_evaluation.contextual_grounding_core_sha256,
        runtime_pipeline_sha256=pipeline,
        provider_execution_binding_sha256=provider_execution_binding,
    )
    evaluation_payload = v4_evaluation.model_dump(mode="json", exclude={"evaluation_sha256"})
    evaluation_payload.update(
        {
            "runtime_pipeline_sha256": pipeline,
            "native_projection": projection,
            "native_projection_sha256": projection.projection_sha256,
        }
    )
    evaluation = MetaSynContextualFrontierRecoveryCoreEvaluationV4.model_validate(
        {
            **evaluation_payload,
            "evaluation_sha256": hash_canonical(evaluation_payload),
        }
    )
    semantic_sha = hash_canonical(_semantic_projection(original))
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "recovery_label": RECOVERY_LABEL,
        "status": "typed_graph_mechanics_completed",
        "offline_only": True,
        "canonicalizer_provider_calls_made": 0,
        "upstream_v4_provider_attempt_count": terminal.provider_attempt_count_upper_bound,
        "upstream_v4_provider_response_completed": result.outcome == "completed",
        "immutable_v4_plan_sha256": plan.plan_sha256,
        "immutable_v4_plan_file_sha256": observed_files[0],
        "immutable_v4_terminal_sha256": terminal.terminal_sha256,
        "immutable_v4_terminal_file_sha256": observed_files[2],
        "immutable_v4_receipt_sha256": receipt.receipt_sha256,
        "immutable_v4_receipt_file_sha256": observed_files[1],
        "provider_result_sha256": result.result_sha256,
        "original_response_sha256": hash_canonical(original),
        "canonicalized_response_sha256": hash_canonical(canonical),
        "provider_execution_binding_sha256": provider_execution_binding,
        "canonicalizer_source_sha256": source_sha,
        "canonicalization_pipeline_sha256": pipeline,
        "v4_evaluator_dependency_sha256": v4_evaluation.evaluation_sha256,
        "canonicalization_changes": changes,
        "canonicalization_change_membership_sha256": change_membership_sha,
        "canonicalized_response": canonical,
        "evaluation": evaluation,
        "evaluation_sha256": evaluation.evaluation_sha256,
        "original_semantic_tuple_sha256": semantic_sha,
        "canonical_semantic_tuple_sha256": hash_canonical(_semantic_projection(canonical)),
        "original_context_containment_failure_count": original_containment_failures,
        "original_invented_context_prefix_count": original_invented_prefixes,
        "blockers": ["post_hoc_source_span_repair"],
        "raw_schema_validated_before_repair": True,
        "exact_roster_and_order_validated_before_repair": True,
        "no_target_or_gold_token_injection": True,
        "immutable_transform_ledger": True,
        "endpoint_quote_ascii_provider_to_exact_u2009_source": True,
        "endpoint_source_match_unique": True,
        "endpoint_marker_only_passage_id_change": True,
        "grounding_uniqueness_and_offsets_recomputed": True,
        "binary_pair_structure_and_arm_source_spans_verified": True,
        "hidden_numeric_equality_not_consulted_by_repair": True,
        "nested_projection_graph_mechanics_flag_dependency_only": True,
        "field_set_unchanged": True,
        "tokens_unchanged": True,
        "normalizations_unchanged": True,
        "numeric_values_unchanged": True,
        "arm_outcome_semantics_unchanged": True,
        "source_span_only_changes_validated": True,
        "endpoint_quote_unicode_whitespace_equivalent": True,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryV4PosthocArtifactV1.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
    *,
    repository_root: Path,
    workspace: Path | None = None,
    immutable_workspace: Path | None = None,
) -> MetaSynContextualFrontierRecoveryV4PosthocArtifactV1:
    root = repository_root.resolve(strict=True)
    target_workspace = root / DEFAULT_WORKSPACE if workspace is None else workspace.absolute()
    immutable = (
        root / DEFAULT_IMMUTABLE_WORKSPACE
        if immutable_workspace is None
        else immutable_workspace.absolute()
    )
    if target_workspace == immutable or immutable in target_workspace.parents:
        raise MetaSynContextualFrontierRecoveryV4PosthocError(
            "recovery_v4_posthoc_workspace_overlaps_immutable_source"
        )
    artifact = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
        repository_root=root, immutable_workspace=immutable
    )
    path = target_workspace / "artifact.json"
    if path.exists():
        observed = MetaSynContextualFrontierRecoveryV4PosthocArtifactV1.model_validate(
            _read_object(path)
        )
        if observed != artifact:
            raise MetaSynContextualFrontierRecoveryV4PosthocError(
                "recovery_v4_posthoc_artifact_replay_mismatch"
            )
        return observed
    _persist_json(path, artifact)
    return artifact
