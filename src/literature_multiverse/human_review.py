"""Fail-closed adjudication of confidence-blinded paper review packets.

Private reviewer files contain article-linked audit identifiers and never leave the
packet directory.  This module verifies the packet manifest and every private file,
validates two independently completed decision ledgers, isolates disagreements, and
emits only aggregate counts plus cryptographic bindings.  A consensus label is used
only when both reviewers agree on the complete scientific decision; otherwise a
separate third-adjudicator row is required.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import ContractModel


class HumanReviewContractError(ValueError):
    """A private review ledger cannot support an auditable scientific result."""


DirectionSummary = Literal[
    "positive",
    "negative",
    "no_effect",
    "mixed",
    "unclear",
    "not_applicable",
]
ReviewerSlot = Literal["reviewer_a", "reviewer_b", "adjudicator"]


class HumanFindingDecision(ContractModel):
    audit_finding_id: Annotated[str, Field(min_length=1)]
    atomic: bool
    supported_by_quote: bool
    direction_correct: bool
    pico_correct: bool
    notes: str | None = None

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class CompletedHumanReviewDecision(ContractModel):
    """One completed paper-level decision, including measured reviewer time."""

    human_review_decision_version: Literal["2"] = "2"
    audit_unit_id: Annotated[str, Field(min_length=1)]
    reviewer_slot: ReviewerSlot
    paper_eligible: bool
    any_target_finding_missed: bool
    all_emitted_findings_supported: bool
    paper_direction_summary: DirectionSummary
    finding_decisions: list[HumanFindingDecision]
    error_codes: list[str] = Field(default_factory=list)
    notes: str | None = None
    review_minutes: Annotated[float, Field(gt=0)]

    @field_validator("error_codes")
    @classmethod
    def validate_error_codes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized) or normalized != sorted(set(normalized)):
            raise ValueError("human_review_error_codes_must_be_sorted_unique")
        return normalized

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("review_minutes")
    @classmethod
    def validate_review_minutes(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("human_review_minutes_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> CompletedHumanReviewDecision:
        finding_ids = [item.audit_finding_id for item in self.finding_decisions]
        if finding_ids != sorted(set(finding_ids)):
            raise ValueError("human_review_finding_ids_must_be_sorted_unique")
        if not self.paper_eligible and self.paper_direction_summary not in {
            "not_applicable",
            "unclear",
        }:
            raise ValueError("ineligible_paper_direction_must_be_not_applicable_or_unclear")
        if self.all_emitted_findings_supported != all(
            item.atomic and item.supported_by_quote and item.direction_correct and item.pico_correct
            for item in self.finding_decisions
        ):
            raise ValueError("all_emitted_findings_supported_not_derived_from_findings")
        return self


class ReviewIdentity(ContractModel):
    audit_unit_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    doc_id: Annotated[str, Field(min_length=1)]
    finding_ids: list[str]
    selection_stratum: Literal[
        "pipeline_eligible_with_findings",
        "pipeline_eligible_zero_findings",
        "pipeline_ineligible_zero_findings",
    ]


class ReviewPacketFinding(ContractModel):
    audit_finding_id: Annotated[str, Field(min_length=1)]


class ReviewPacketSystemOutput(ContractModel):
    eligible: bool
    findings: list[ReviewPacketFinding]


class ReviewPacketIndex(ContractModel):
    audit_unit_id: Annotated[str, Field(min_length=1)]
    display_order: Annotated[int, Field(ge=1)]
    system_output: ReviewPacketSystemOutput


def _packet_index_from_row(row: Mapping[str, Any]) -> ReviewPacketIndex:
    system_output = row.get("system_output")
    if not isinstance(system_output, Mapping):
        raise HumanReviewContractError("human_review_packet_system_output_invalid")
    findings = system_output.get("findings")
    if not isinstance(findings, list) or any(not isinstance(item, Mapping) for item in findings):
        raise HumanReviewContractError("human_review_packet_findings_invalid")
    return ReviewPacketIndex.model_validate(
        {
            "audit_unit_id": row.get("audit_unit_id"),
            "display_order": row.get("display_order"),
            "system_output": {
                "eligible": system_output.get("eligible"),
                "findings": [
                    {"audit_finding_id": item.get("audit_finding_id")} for item in findings
                ],
            },
        }
    )


SCIENTIFIC_FIELDS = (
    "paper_eligible",
    "any_target_finding_missed",
    "all_emitted_findings_supported",
    "paper_direction_summary",
    "finding_decisions",
    "error_codes",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HumanReviewContractError(f"human_review_json_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise HumanReviewContractError(f"human_review_json_requires_object:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HumanReviewContractError(f"human_review_jsonl_unreadable:{path}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HumanReviewContractError(f"human_review_jsonl_invalid:{path}:{number}") from exc
        if not isinstance(value, dict):
            raise HumanReviewContractError(f"human_review_jsonl_requires_objects:{path}:{number}")
        rows.append(value)
    return rows


def verify_review_packet_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Resolve and rehash the exact four private packet files without following links."""

    if manifest_path.is_symlink():
        raise HumanReviewContractError("human_review_manifest_symlink_forbidden")
    manifest = _read_json(manifest_path)
    if manifest.get("human_review_packet_manifest_version") not in {"1", "2"}:
        raise HumanReviewContractError("human_review_manifest_version_unsupported")
    expected = {
        "identity_key",
        "review_packet",
        "reviewer_a_decisions",
        "reviewer_b_decisions",
    }
    entries = manifest.get("local_private_files")
    if not isinstance(entries, dict) or set(entries) != expected:
        raise HumanReviewContractError("human_review_private_file_set_invalid")
    packet_dir = manifest_path.parent
    if packet_dir.is_symlink():
        raise HumanReviewContractError("human_review_packet_directory_symlink_forbidden")
    try:
        root = packet_dir.resolve(strict=True)
    except OSError as exc:
        raise HumanReviewContractError("human_review_packet_directory_missing") from exc
    resolved: dict[str, Path] = {}
    for role, metadata in sorted(entries.items()):
        if not isinstance(metadata, Mapping):
            raise HumanReviewContractError("human_review_private_file_metadata_invalid")
        relative = metadata.get("path")
        expected_hash = metadata.get("sha256")
        expected_rows = metadata.get("rows")
        if not isinstance(relative, str) or not relative:
            raise HumanReviewContractError("human_review_private_file_path_invalid")
        relative_path = Path(relative)
        candidate = packet_dir / relative_path
        if relative_path.is_absolute() or ".." in relative_path.parts or candidate.is_symlink():
            raise HumanReviewContractError("human_review_private_file_path_unsafe")
        try:
            path = candidate.resolve(strict=True)
        except OSError as exc:
            raise HumanReviewContractError(f"human_review_private_file_missing:{role}") from exc
        if root not in path.parents or not path.is_file():
            raise HumanReviewContractError("human_review_private_file_path_unsafe")
        if sha256_file(path) != expected_hash:
            raise HumanReviewContractError(f"human_review_private_file_hash_mismatch:{role}")
        if not isinstance(expected_rows, int) or expected_rows < 1:
            raise HumanReviewContractError("human_review_private_file_rows_invalid")
        if len(_read_jsonl(path)) != expected_rows:
            raise HumanReviewContractError(f"human_review_private_file_row_mismatch:{role}")
        resolved[role] = path
    return manifest, resolved


def _unique_by_id(rows: Sequence[Any], *, label: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for row in rows:
        audit_id = row.audit_unit_id
        if audit_id in indexed:
            raise HumanReviewContractError(f"duplicate_{label}_audit_unit:{audit_id}")
        indexed[audit_id] = row
    return indexed


def _decision_signature(decision: CompletedHumanReviewDecision) -> str:
    payload = {
        field: decision.model_dump(mode="json")[field]
        for field in SCIENTIFIC_FIELDS
        if field != "finding_decisions"
    }
    payload["finding_decisions"] = [
        item.model_dump(mode="json", exclude={"notes"})
        for item in decision.finding_decisions
    ]
    return hash_canonical(payload)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total == 0:
        return {"successes": 0, "total": 0, "rate": None, "ci95": None}
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci95": [max(0.0, center - half), min(1.0, center + half)],
    }


def _field_agreement(
    a: Sequence[CompletedHumanReviewDecision],
    b_by_id: Mapping[str, CompletedHumanReviewDecision],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_findings = agreed_findings = 0
    for left in a:
        right = b_by_id[left.audit_unit_id]
        for field in SCIENTIFIC_FIELDS[:-2]:
            counts[field] += int(getattr(left, field) == getattr(right, field))
        left_findings = {row.audit_finding_id: row for row in left.finding_decisions}
        right_findings = {row.audit_finding_id: row for row in right.finding_decisions}
        if set(left_findings) != set(right_findings):
            raise HumanReviewContractError(
                f"reviewer_finding_identity_mismatch:{left.audit_unit_id}"
            )
        for finding_id in sorted(left_findings):
            total_findings += 1
            agreed_findings += int(
                left_findings[finding_id].model_dump(mode="json", exclude={"notes"})
                == right_findings[finding_id].model_dump(mode="json", exclude={"notes"})
            )
    n = len(a)
    return {
        "complete_decision": _wilson(
            sum(
                _decision_signature(left) == _decision_signature(b_by_id[left.audit_unit_id])
                for left in a
            ),
            n,
        ),
        "paper_fields": {field: _wilson(counts[field], n) for field in SCIENTIFIC_FIELDS[:-2]},
        "finding_decisions": _wilson(agreed_findings, total_findings),
    }


def _performance(
    resolutions: Mapping[str, CompletedHumanReviewDecision],
    identities: Mapping[str, ReviewIdentity],
    packet: Mapping[str, ReviewPacketIndex],
) -> dict[str, Any]:
    by_stratum: dict[str, list[str]] = {}
    for audit_id, identity in identities.items():
        by_stratum.setdefault(identity.selection_stratum, []).append(audit_id)

    def summarize(ids: Sequence[str]) -> dict[str, Any]:
        tp = fp = tn = fn = 0
        eligible = missed = 0
        findings_total = findings_correct = 0
        for audit_id in ids:
            resolved = resolutions[audit_id]
            predicted = packet[audit_id].system_output.eligible
            truth = resolved.paper_eligible
            tp += int(predicted and truth)
            fp += int(predicted and not truth)
            tn += int(not predicted and not truth)
            fn += int(not predicted and truth)
            eligible += int(truth)
            missed += int(truth and resolved.any_target_finding_missed)
            findings_total += len(resolved.finding_decisions)
            findings_correct += sum(
                item.atomic
                and item.supported_by_quote
                and item.direction_correct
                and item.pico_correct
                for item in resolved.finding_decisions
            )
        return {
            "items": len(ids),
            "eligibility_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "eligibility_precision": _wilson(tp, tp + fp),
            "eligibility_recall": _wilson(tp, tp + fn),
            "eligible_papers_without_missed_target_finding": _wilson(eligible - missed, eligible),
            "emitted_findings_fully_correct": _wilson(findings_correct, findings_total),
        }

    all_ids = sorted(resolutions)
    return {
        "sampling_warning": (
            "The selected set takes a census of pipeline-positive strata and samples "
            "pipeline-negative "
            "papers; pooled counts are feasibility diagnostics, not prevalence-weighted accuracy."
        ),
        "pooled_diagnostic": summarize(all_ids),
        "by_selection_stratum": {
            stratum: summarize(sorted(ids)) for stratum, ids in sorted(by_stratum.items())
        },
    }


def evaluate_human_review_packet(
    *,
    manifest_path: Path,
    reviewer_a_path: Path | None = None,
    reviewer_b_path: Path | None = None,
    adjudicator_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate completed reviews and return a public summary plus private conflict rows."""

    manifest, paths = verify_review_packet_manifest(manifest_path)
    identities = [ReviewIdentity.model_validate(row) for row in _read_jsonl(paths["identity_key"])]
    packet_rows = [_packet_index_from_row(row) for row in _read_jsonl(paths["review_packet"])]
    identity_by_id = _unique_by_id(identities, label="identity")
    packet_by_id = _unique_by_id(packet_rows, label="packet")
    expected_ids = set(identity_by_id)
    if set(packet_by_id) != expected_ids:
        raise HumanReviewContractError("human_review_packet_identity_set_mismatch")

    # The manifest binds immutable blank templates. Reviewers save completed copies to
    # new paths; editing the templates would destroy the packet freeze. When explicit
    # copies are absent, parsing the templates yields the honest prepared/incomplete
    # status.
    completed_paths = {
        "reviewer_a_decisions": reviewer_a_path or paths["reviewer_a_decisions"],
        "reviewer_b_decisions": reviewer_b_path or paths["reviewer_b_decisions"],
    }
    for role, path in completed_paths.items():
        if path.is_symlink() or not path.is_file():
            raise HumanReviewContractError(f"human_review_completed_file_invalid:{role}")

    completed: dict[str, list[CompletedHumanReviewDecision]] = {}
    parse_errors: dict[str, list[dict[str, Any]]] = {}
    for role in ("reviewer_a_decisions", "reviewer_b_decisions"):
        parsed: list[CompletedHumanReviewDecision] = []
        errors: list[dict[str, Any]] = []
        expected_slot = role.removesuffix("_decisions")
        raw_rows = _read_jsonl(completed_paths[role])
        for row_number, row in enumerate(raw_rows, start=1):
            try:
                decision = CompletedHumanReviewDecision.model_validate(row)
                if decision.reviewer_slot != expected_slot:
                    raise ValueError("reviewer_slot_mismatch")
                parsed.append(decision)
            except ValueError as exc:
                errors.append({"row": row_number, "reason": str(exc).splitlines()[0]})
        completed[role] = parsed
        parse_errors[role] = errors

    public_base: dict[str, Any] = {
        "human_review_evaluation_version": "1",
        "question_id": manifest.get("question_id"),
        "packet_manifest_sha256": sha256_file(manifest_path),
        "private_input_sha256s": {role: sha256_file(path) for role, path in sorted(paths.items())},
        "completed_decision_sha256s": {
            role: sha256_file(path) for role, path in sorted(completed_paths.items())
        },
        "sample_size": len(expected_ids),
        "contains_article_text": False,
        "contains_paper_identifiers": False,
        "contains_audit_unit_identifiers": False,
        "label_access_state": {
            "system_outputs_previously_opened": True,
            "review_labels_accessed_by_this_evaluator": bool(
                reviewer_a_path is not None and reviewer_b_path is not None
            ),
            "review_labels_used_for_pipeline_optimization": False,
            "pristine_question_level_holdout_eligible": False,
            "scientific_role": "single_question_stratified_diagnostic",
        },
    }
    if any(parse_errors.values()):
        payload = {
            **public_base,
            "status": "prepared_not_adjudicated",
            "completed_rows": {role: len(rows) for role, rows in sorted(completed.items())},
            "invalid_or_incomplete_rows": {
                role: len(errors) for role, errors in sorted(parse_errors.items())
            },
            "claim_boundary": (
                "No human-accuracy, timing, calibration, or release claim is "
                "licensed."
            ),
        }
        return {**payload, "evaluation_sha256": hash_canonical(payload)}, []

    left = completed["reviewer_a_decisions"]
    right = completed["reviewer_b_decisions"]
    left_by_id = _unique_by_id(left, label="reviewer_a")
    right_by_id = _unique_by_id(right, label="reviewer_b")
    if set(left_by_id) != expected_ids or set(right_by_id) != expected_ids:
        raise HumanReviewContractError("completed_reviewer_audit_unit_set_mismatch")
    for audit_id in sorted(expected_ids):
        expected_finding_ids = sorted(
            finding.audit_finding_id for finding in packet_by_id[audit_id].system_output.findings
        )
        for decision in (left_by_id[audit_id], right_by_id[audit_id]):
            observed = [item.audit_finding_id for item in decision.finding_decisions]
            if observed != expected_finding_ids:
                raise HumanReviewContractError(f"completed_review_finding_set_mismatch:{audit_id}")

    conflicts = [
        audit_id
        for audit_id in sorted(expected_ids)
        if _decision_signature(left_by_id[audit_id]) != _decision_signature(right_by_id[audit_id])
    ]
    conflict_rows = [
        {
            "human_review_decision_version": "2",
            "audit_unit_id": audit_id,
            "reviewer_slot": "adjudicator",
            "paper_eligible": None,
            "any_target_finding_missed": None,
            "all_emitted_findings_supported": None,
            "paper_direction_summary": None,
            "finding_decisions": [
                {
                    "audit_finding_id": finding.audit_finding_id,
                    "atomic": None,
                    "supported_by_quote": None,
                    "direction_correct": None,
                    "pico_correct": None,
                    "notes": None,
                }
                for finding in packet_by_id[audit_id].system_output.findings
            ],
            "error_codes": [],
            "notes": None,
            "review_minutes": None,
        }
        for audit_id in conflicts
    ]
    resolutions: dict[str, CompletedHumanReviewDecision] = {
        audit_id: left_by_id[audit_id]
        for audit_id in expected_ids
        if audit_id not in set(conflicts)
    }
    adjudicator_hash = None
    if conflicts and adjudicator_path is not None:
        if adjudicator_path.is_symlink():
            raise HumanReviewContractError("human_review_adjudicator_symlink_forbidden")
        adjudications = [
            CompletedHumanReviewDecision.model_validate(row)
            for row in _read_jsonl(adjudicator_path)
        ]
        adjudication_by_id = _unique_by_id(adjudications, label="adjudicator")
        if set(adjudication_by_id) != set(conflicts):
            raise HumanReviewContractError("human_review_adjudicator_conflict_set_mismatch")
        if any(row.reviewer_slot != "adjudicator" for row in adjudications):
            raise HumanReviewContractError("human_review_adjudicator_slot_invalid")
        for audit_id, decision in adjudication_by_id.items():
            expected_finding_ids = sorted(
                finding.audit_finding_id
                for finding in packet_by_id[audit_id].system_output.findings
            )
            if [
                item.audit_finding_id for item in decision.finding_decisions
            ] != expected_finding_ids:
                raise HumanReviewContractError(
                    f"human_review_adjudicator_finding_set_mismatch:{audit_id}"
                )
        resolutions.update(adjudication_by_id)
        adjudicator_hash = sha256_file(adjudicator_path)

    reviewer_minutes = [row.review_minutes for row in [*left, *right]]
    adjudication_minutes = (
        sum(
            row.review_minutes
            for row in resolutions.values()
            if row.reviewer_slot == "adjudicator"
        )
        if adjudicator_hash
        else 0.0
    )
    payload = {
        **public_base,
        "status": "complete" if len(resolutions) == len(expected_ids) else "awaiting_adjudication",
        "independent_review_complete": True,
        "agreement": _field_agreement(left, right_by_id),
        "conflicting_items": len(conflicts),
        "adjudicated_items": len(conflicts) if adjudicator_hash else 0,
        "review_time": {
            "reviewer_rows": len(reviewer_minutes),
            "independent_reviewer_person_minutes": sum(reviewer_minutes),
            "median_minutes_per_reviewer_item": median(reviewer_minutes),
            "adjudication_person_minutes": adjudication_minutes,
            "total_person_minutes": (
                sum(reviewer_minutes) + adjudication_minutes
            ),
            "basis": (
                "reviewer_reported_total_person_minutes_including_final_adjudication"
            ),
        },
        "adjudicator_file_sha256": adjudicator_hash,
        "performance": (
            _performance(resolutions, identity_by_id, packet_by_id)
            if len(resolutions) == len(expected_ids)
            else None
        ),
        "claim_boundary": (
            "Metadata-only diagnostic on a stratified single-question review packet; it is not "
            "a pristine question-level calibration or cross-domain generalization result."
            if len(resolutions) == len(expected_ids)
            else (
                "Independent review is complete, but no human-accuracy or calibration "
                "claim is licensed until every disagreement is adjudicated."
            )
        ),
    }
    return {**payload, "evaluation_sha256": hash_canonical(payload)}, conflict_rows


__all__ = [
    "CompletedHumanReviewDecision",
    "HumanFindingDecision",
    "HumanReviewContractError",
    "evaluate_human_review_packet",
    "verify_review_packet_manifest",
]
