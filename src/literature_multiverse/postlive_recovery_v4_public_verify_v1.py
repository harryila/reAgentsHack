"""Public ``lm verify`` diagnostic for the repaired recovery-v4 graph.

This module constructs a truthful, incomplete verification corpus bundle and an
actual claim manifest from the post-live joined graph.  All known source and
corpus limitations are blocking adapter issues.  The resulting public verifier
run is permanently analysis-only and must abstain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.certificate import VerificationCertificate
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_recovery_v4_posthoc_v1 import (
    MetaSynContextualFrontierRecoveryV4PosthocArtifactV1,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import _persist_json
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.postlive_recovery_v4_join_v1 import (
    PostLiveRecoveryV4JoinArtifactV1,
    validate_postlive_recovery_v4_join_artifact_v1,
)
from literature_multiverse.sequential_verification import SequentialVerificationState
from literature_multiverse.verifier import ClaimManifest

PREPARATION_VERSION = "postlive-recovery-v4-public-verify-preparation-v1"
VALIDATION_VERSION = "postlive-recovery-v4-public-verify-validation-v1"
DEFAULT_JOIN_PATH = Path("artifacts/diagnostics/postlive-recovery-v4-join-v1.json")
DEFAULT_POSTHOC_PATH = Path(
    "data/cache/metasyn/contextual-frontier-recovery-v4-posthoc-v1/artifact.json"
)
DEFAULT_WORKSPACE = Path(
    "data/cache/metasyn/postlive-recovery-v4-public-verify-v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1"
)

REQUIRED_ADAPTER_ISSUE_CODES = (
    "complete_corpus_not_available",
    "post_hoc_source_span_repair",
    "source_v4_terminal_failed_closed",
    "title_or_abstract_only_not_release_grade",
)
REQUIRED_CERTIFICATE_REASONS = tuple(
    sorted(
        {
            *(f"adapter:{code}" for code in REQUIRED_ADAPTER_ISSUE_CODES),
            "adapter:uncalibrated_sequential_audit_analysis_only",
            "adapter:unverified_source_provenance",
        }
    )
)


class PostLiveRecoveryV4PublicVerifyError(ValueError):
    """The public-verifier diagnostic contract failed closed."""


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
        raise PostLiveRecoveryV4PublicVerifyError("public_verify_source_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostLiveRecoveryV4PublicVerifyError(
            "public_verify_source_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise PostLiveRecoveryV4PublicVerifyError("public_verify_source_not_object")
    return value


class PostLiveRecoveryV4PublicVerifyPreparationV1(_Frozen):
    preparation_version: Literal[
        "postlive-recovery-v4-public-verify-preparation-v1"
    ] = PREPARATION_VERSION
    status: Literal["public_cli_inputs_frozen_non_authorizing"] = (
        "public_cli_inputs_frozen_non_authorizing"
    )
    source_join_artifact_sha256: Sha256
    source_join_artifact_file_sha256: Sha256
    source_posthoc_artifact_sha256: Sha256
    source_posthoc_artifact_file_sha256: Sha256
    source_canonicalization_pipeline_sha256: Sha256
    source_repair_change_membership_sha256: Sha256
    source_evidence_graph_sha256: Sha256
    source_join_external_replay_performed: Literal[True] = True
    source_posthoc_external_replay_performed: Literal[True] = True
    source_lineage: dict[str, Any]
    source_lineage_sha256: Sha256
    claim_manifest_sha256: Sha256
    corpus_bundle_sha256: Sha256
    required_adapter_issue_codes: tuple[
        Literal[
            "complete_corpus_not_available",
            "post_hoc_source_span_repair",
            "source_v4_terminal_failed_closed",
            "title_or_abstract_only_not_release_grade",
        ],
        ...,
    ]
    public_lm_verify_required: Literal[True] = True
    analysis_only_uncalibrated_audit_required: Literal[True] = True
    incomplete_corpus: Literal[True] = True
    title_or_abstract_only: Literal[True] = True
    source_v4_terminal_failed_closed: Literal[True] = True
    post_hoc_source_span_repair: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    preparation_sha256: Sha256

    @model_validator(mode="after")
    def validate_preparation(self) -> PostLiveRecoveryV4PublicVerifyPreparationV1:
        if (
            self.required_adapter_issue_codes != REQUIRED_ADAPTER_ISSUE_CODES
            or self.source_lineage_sha256 != hash_canonical(self.source_lineage)
        ):
            raise ValueError("public_verify_preparation_replay_mismatch")
        _self_hash(self, "preparation_sha256", "public_verify_preparation_hash_mismatch")
        return self


class PostLiveRecoveryV4PublicVerifyValidationV1(_Frozen):
    validation_version: Literal[
        "postlive-recovery-v4-public-verify-validation-v1"
    ] = VALIDATION_VERSION
    status: Literal["public_cli_abstention_validated"] = "public_cli_abstention_validated"
    preparation_sha256: Sha256
    claim_manifest_file_sha256: Sha256
    corpus_bundle_file_sha256: Sha256
    certificate_sha256: Sha256
    certificate_file_sha256: Sha256
    certificate_html_file_sha256: Sha256
    sequential_audit_state_sha256: Sha256
    sequential_audit_state_file_sha256: Sha256
    required_certificate_reasons: tuple[str, ...]
    selected_audit_item_id: str | None
    public_cli_exercised: Literal[True] = True
    analysis_only_uncalibrated_audit: Literal[True] = True
    release_status: Literal["abstained"] = "abstained"
    release_authorizing: Literal[False] = False
    external_certificate_model_replay: Literal[True] = True
    external_sequential_state_model_replay: Literal[True] = True
    metadata_lineage_preserved: Literal[True] = True
    all_required_blockers_preserved: Literal[True] = True
    validation_sha256: Sha256

    @field_validator("required_certificate_reasons")
    @classmethod
    def validate_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != REQUIRED_CERTIFICATE_REASONS:
            raise ValueError("public_verify_required_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> PostLiveRecoveryV4PublicVerifyValidationV1:
        _self_hash(self, "validation_sha256", "public_verify_validation_hash_mismatch")
        return self


def _freeze_source_contract(
    *, repository_root: Path, join_path: Path, posthoc_path: Path
) -> tuple[PostLiveRecoveryV4JoinArtifactV1, MetaSynContextualFrontierRecoveryV4PosthocArtifactV1]:
    join = validate_postlive_recovery_v4_join_artifact_v1(
        artifact=_read_object(join_path),
        repository_root=repository_root,
        posthoc_artifact_path=posthoc_path,
    )
    posthoc = MetaSynContextualFrontierRecoveryV4PosthocArtifactV1.model_validate(
        _read_object(posthoc_path)
    )
    if (
        join.source_posthoc_artifact_sha256 != posthoc.artifact_sha256
        or join.source_posthoc_artifact_file_sha256 != sha256_file(posthoc_path)
        or join.canonicalization_pipeline_sha256
        != posthoc.canonicalization_pipeline_sha256
        or join.source_repair_change_membership_sha256
        != posthoc.canonicalization_change_membership_sha256
        or join.evidence_graph_sha256 != hash_canonical(join.evidence_graph)
        or not set(REQUIRED_ADAPTER_ISSUE_CODES).issubset(join.blockers)
        or join.release_authorizing
        or join.claim_release_authority
        or not join.source_posthoc_external_replay_performed
        or join.source_posthoc_external_replay_sha256 is None
    ):
        raise PostLiveRecoveryV4PublicVerifyError("public_verify_source_join_mismatch")
    try:
        join_path.resolve(strict=True).relative_to(repository_root.resolve(strict=True))
        posthoc_path.resolve(strict=True).relative_to(repository_root.resolve(strict=True))
    except ValueError as exc:
        raise PostLiveRecoveryV4PublicVerifyError(
            "public_verify_source_outside_repository"
        ) from exc
    return join, posthoc


def freeze_postlive_recovery_v4_public_verify_inputs_v1(
    *,
    repository_root: Path,
    join_path: Path | None = None,
    posthoc_path: Path | None = None,
) -> tuple[PostLiveRecoveryV4PublicVerifyPreparationV1, dict[str, Any], dict[str, Any]]:
    root = repository_root.resolve(strict=True)
    source_join_path = root / DEFAULT_JOIN_PATH if join_path is None else join_path
    source_posthoc_path = root / DEFAULT_POSTHOC_PATH if posthoc_path is None else posthoc_path
    join, posthoc = _freeze_source_contract(
        repository_root=root,
        join_path=source_join_path,
        posthoc_path=source_posthoc_path,
    )
    graph = join.evidence_graph
    estimate = graph.outcome_estimates[0]
    publication = graph.publications[0]
    treatment = next(arm for arm in graph.arms if arm.role.value == "intervention")
    comparator = next(arm for arm in graph.arms if arm.role.value == "comparator")
    if (
        treatment.label != "500-mg"
        or comparator.label != "placebo group"
        or estimate.outcome_name != "spleen response"
        or estimate.timepoint is None
        or estimate.timepoint.raw_label != "week 24"
    ):
        raise PostLiveRecoveryV4PublicVerifyError(
            "public_verify_claim_semantics_not_exact"
        )
    source_lineage = {
        "join_artifact_sha256": join.artifact_sha256,
        "join_artifact_file_sha256": sha256_file(source_join_path),
        "posthoc_artifact_sha256": posthoc.artifact_sha256,
        "posthoc_artifact_file_sha256": sha256_file(source_posthoc_path),
        "canonicalization_pipeline_sha256": posthoc.canonicalization_pipeline_sha256,
        "repair_change_membership_sha256": (
            posthoc.canonicalization_change_membership_sha256
        ),
        "v4_terminal_sha256": posthoc.immutable_v4_terminal_sha256,
        "v4_terminal_file_sha256": posthoc.immutable_v4_terminal_file_sha256,
        "evidence_graph_sha256": join.evidence_graph_sha256,
        "recovery_witness_identity": "metasyn-row17-candidate3",
        "source_record_identity": "metasyn-source-id20",
        "source_locator": estimate.effect.provenance.source_locator,
        "join_external_replay_sha256": hash_canonical(
            {
                "join_artifact_sha256": join.artifact_sha256,
                "posthoc_external_replay_sha256": (
                    join.source_posthoc_external_replay_sha256
                ),
                "exact_rebuild_equality": True,
            }
        ),
        "posthoc_external_replay_sha256": join.source_posthoc_external_replay_sha256,
        "source_scope": "single_title_abstract_publication_incomplete_corpus",
        "release_authorizing": False,
    }
    source_lineage_sha = hash_canonical(source_lineage)
    claim = {
        "claim_manifest_version": "1",
        "question_id": "metasyn-row17-fedratinib-500mg-spleen-week24",
        "population_id": "primary-or-secondary-myelofibrosis",
        "domain": "oncology",
        "claim": {
            "statement": (
                "In patients with primary or secondary myelofibrosis, fedratinib "
                "500-mg versus placebo increases primary-endpoint spleen response at week 24."
            ),
            "direction": "increase",
            "outcome_name": estimate.outcome_name,
            "contrast_id": estimate.contrast_id,
            "estimand": graph.contrasts[0].estimand,
            "conditions": {},
        },
        "protocol": {
            "corpus_cutoff": (
                "diagnostic-metasyn-row17-candidate3-source-id20-"
                "title-abstract-snapshot"
            ),
            "inclusion_criteria": [
                "Patients with primary or secondary myelofibrosis",
                "Fedratinib 500-mg compared with placebo",
                "Primary-endpoint spleen response measured at week 24",
            ],
            "exclusion_criteria": [],
            "notes": (
                "Incomplete one-publication title/abstract diagnostic; not a frozen "
                "protocol-complete literature corpus."
            ),
        },
        "audit": {
            "error_probability": 0.5,
            "probability_basis": "heuristic",
            "probability_source": "uncalibrated-posthoc-diagnostic-only",
            "verification_minutes_per_item": 3.0,
            "item_error_probabilities": {},
            "item_verification_minutes": {},
            "decision_threshold": 0.5,
        },
    }
    manifest = ClaimManifest.model_validate(claim)
    claim = manifest.model_dump(mode="json")
    finding_id = estimate.effect.finding_id
    issues = [
        {
            "severity": "blocking",
            "code": "complete_corpus_not_available",
            "detail": (
                "The supplied graph contains one title/abstract publication and does not "
                "represent a protocol-complete or retrieval-saturated corpus."
            ),
        },
        {
            "severity": "blocking",
            "code": "post_hoc_source_span_repair",
            "detail": (
                "The typed graph depends on a deterministic post-hoc source-span repair; "
                "this licenses mechanics inspection only."
            ),
            "paper_id": publication.paper_id,
            "finding_id": finding_id,
        },
        {
            "severity": "blocking",
            "code": "source_v4_terminal_failed_closed",
            "detail": (
                "The original recovery-v4 runtime terminal failed contextual validation; "
                "the offline repair does not rewrite that terminal outcome."
            ),
            "paper_id": publication.paper_id,
            "finding_id": finding_id,
        },
        {
            "severity": "blocking",
            "code": "title_or_abstract_only_not_release_grade",
            "detail": (
                "Evidence is grounded only in title/abstract source spans and is not "
                "release-grade full-text evidence."
            ),
            "paper_id": publication.paper_id,
            "finding_id": finding_id,
        },
    ]
    issues.sort(key=lambda item: item["code"])
    metadata = {
        "diagnostic_version": PREPARATION_VERSION,
        "purpose": "public_lm_verify_fail_closed_integration",
        "source_lineage": source_lineage,
        "source_lineage_sha256": source_lineage_sha,
        "source_join_blockers": join.blockers,
        "declared_complete_corpus": False,
        "empirical_evidence_authority": False,
        "claim_release_authority": False,
    }
    bundle = {
        "corpus_bundle_version": "postlive-recovery-v4-public-verify-bundle-v1",
        "corpus_id": "metasyn-row17-fedratinib-500mg-spleen-week24-diagnostic",
        "graph": graph.model_dump(mode="json"),
        "eligibility": [
            {
                "paper_id": publication.paper_id,
                "title": publication.title,
                "status": "included",
                "reason": "Only publication present in the bounded diagnostic graph.",
                "source": "postlive-recovery-v4-join-v1",
            }
        ],
        "adapter_issues": issues,
        "metadata": metadata,
    }
    payload = {
        "preparation_version": PREPARATION_VERSION,
        "status": "public_cli_inputs_frozen_non_authorizing",
        "source_join_artifact_sha256": join.artifact_sha256,
        "source_join_artifact_file_sha256": sha256_file(source_join_path),
        "source_posthoc_artifact_sha256": posthoc.artifact_sha256,
        "source_posthoc_artifact_file_sha256": sha256_file(source_posthoc_path),
        "source_canonicalization_pipeline_sha256": (
            posthoc.canonicalization_pipeline_sha256
        ),
        "source_repair_change_membership_sha256": (
            posthoc.canonicalization_change_membership_sha256
        ),
        "source_evidence_graph_sha256": join.evidence_graph_sha256,
        "source_join_external_replay_performed": True,
        "source_posthoc_external_replay_performed": True,
        "source_lineage": source_lineage,
        "source_lineage_sha256": source_lineage_sha,
        "claim_manifest_sha256": hash_canonical(claim),
        "corpus_bundle_sha256": hash_canonical(bundle),
        "required_adapter_issue_codes": REQUIRED_ADAPTER_ISSUE_CODES,
        "public_lm_verify_required": True,
        "analysis_only_uncalibrated_audit_required": True,
        "incomplete_corpus": True,
        "title_or_abstract_only": True,
        "source_v4_terminal_failed_closed": True,
        "post_hoc_source_span_repair": True,
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    preparation = PostLiveRecoveryV4PublicVerifyPreparationV1.model_validate(
        {**payload, "preparation_sha256": hash_canonical(payload)}
    )
    return preparation, claim, bundle


def _write_exact(path: Path, value: Any) -> None:
    if path.exists():
        if _read_object(path) != value:
            raise PostLiveRecoveryV4PublicVerifyError(
                f"public_verify_existing_artifact_mismatch:{path.name}"
            )
        return
    _persist_json(path, value)


def write_postlive_recovery_v4_public_verify_inputs_v1(
    *, repository_root: Path, workspace: Path | None = None
) -> PostLiveRecoveryV4PublicVerifyPreparationV1:
    root = repository_root.resolve(strict=True)
    target = root / DEFAULT_WORKSPACE if workspace is None else workspace.absolute()
    preparation, claim, bundle = freeze_postlive_recovery_v4_public_verify_inputs_v1(
        repository_root=root
    )
    _write_exact(target / "claim.json", claim)
    _write_exact(target / "corpus-bundle.json", bundle)
    _write_exact(target / "preparation.json", preparation.model_dump(mode="json"))
    return preparation


def validate_postlive_recovery_v4_public_verify_output_v1(
    *,
    repository_root: Path,
    workspace: Path | None = None,
    output_dir: Path | None = None,
    write_report: bool = True,
) -> PostLiveRecoveryV4PublicVerifyValidationV1:
    root = repository_root.resolve(strict=True)
    source = root / DEFAULT_WORKSPACE if workspace is None else workspace.absolute()
    output = root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir.absolute()
    preparation = PostLiveRecoveryV4PublicVerifyPreparationV1.model_validate(
        _read_object(source / "preparation.json")
    )
    claim = _read_object(source / "claim.json")
    bundle = _read_object(source / "corpus-bundle.json")
    certificate_path = output / "verification-certificate.json"
    html_path = output / "verification-certificate.html"
    state_path = output / "sequential-audit-state.json"
    certificate = VerificationCertificate.model_validate(_read_object(certificate_path))
    state = SequentialVerificationState.model_validate(_read_object(state_path))
    required = set(REQUIRED_CERTIFICATE_REASONS)
    observed_issues = {
        item.get("code"): item.get("severity") for item in certificate.adapter_issues
    }
    html = html_path.read_text(encoding="utf-8")
    if (
        preparation.claim_manifest_sha256 != hash_canonical(claim)
        or preparation.corpus_bundle_sha256 != hash_canonical(bundle)
        or certificate.claim_manifest != claim
        or certificate.corpus_sha256
        != hash_canonical({"evidence": sha256_file(source / "corpus-bundle.json")})
        or certificate.source_evidence_graph_sha256
        != preparation.source_evidence_graph_sha256
        or certificate.corpus.get("metadata") != bundle.get("metadata")
        or certificate.corpus["metadata"].get("source_lineage_sha256")
        != preparation.source_lineage_sha256
        or certificate.status != "abstained"
        or certificate.release_assessment.status.value != "abstained"
        or not required.issubset(certificate.reasons)
        or any(observed_issues.get(code) != "blocking" for code in REQUIRED_ADAPTER_ISSUE_CODES)
        or observed_issues.get("uncalibrated_sequential_audit_analysis_only") != "blocking"
        or observed_issues.get("unverified_source_provenance") != "blocking"
        or certificate.sequential_audit_state is None
        or certificate.sequential_audit_state != state
        or certificate.adaptive_calibration_bundle is not None
        or certificate.adaptive_prospective_assessment is not None
        or certificate.certificate_sha256 not in html
        or "<script" in html.lower()
    ):
        raise PostLiveRecoveryV4PublicVerifyError(
            "public_verify_certificate_fail_closed_contract_mismatch"
        )
    active = state.session.active_action
    payload = {
        "validation_version": VALIDATION_VERSION,
        "status": "public_cli_abstention_validated",
        "preparation_sha256": preparation.preparation_sha256,
        "claim_manifest_file_sha256": sha256_file(source / "claim.json"),
        "corpus_bundle_file_sha256": sha256_file(source / "corpus-bundle.json"),
        "certificate_sha256": certificate.certificate_sha256,
        "certificate_file_sha256": sha256_file(certificate_path),
        "certificate_html_file_sha256": sha256_file(html_path),
        "sequential_audit_state_sha256": state.state_sha256,
        "sequential_audit_state_file_sha256": sha256_file(state_path),
        "required_certificate_reasons": REQUIRED_CERTIFICATE_REASONS,
        "selected_audit_item_id": active.item_id if active is not None else None,
        "public_cli_exercised": True,
        "analysis_only_uncalibrated_audit": True,
        "release_status": "abstained",
        "release_authorizing": False,
        "external_certificate_model_replay": True,
        "external_sequential_state_model_replay": True,
        "metadata_lineage_preserved": True,
        "all_required_blockers_preserved": True,
    }
    report = PostLiveRecoveryV4PublicVerifyValidationV1.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )
    if write_report:
        _write_exact(output / "external-validation.json", report.model_dump(mode="json"))
    return report
