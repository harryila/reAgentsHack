"""Cycle-free composition of extraction and verifier pipeline identities.

An extraction pipeline and a verifier pipeline are different identities.  This
module does not collapse them or treat a caller-supplied equality decision as
evidence.  It independently validates three computed pipeline proofs, derives a
calibration/release-pipeline identity from the extraction, verifier-core, and
join-policy proofs, and binds that identity to one deterministic corpus-ingress
projection.

The resulting join proves only pipeline composition and corpus lineage.  It has
no extraction-accuracy, calibration, scientific, or claim-release authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintError,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
    validate_pipeline_verification_integrity,
)

CORPUS_INGRESS_PROJECTION_VERSION = "corpus-ingress-projection-v1"
PIPELINE_COMPOSITION_CONTEXT_VERSION = "pipeline-composition-context-v1"
CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION = "corpus-pipeline-composition-join-v1"
COMPOSED_CALIBRATION_COMPONENT_ID = "frozen-corpus-extraction"
COMPOSED_CALIBRATION_COMPONENT_VERSION = "1"

CorpusIngressInterfaceV1 = Literal[
    "native-grounding-v4",
    "hosted-exact-once-native-grounding-v4",
]

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class CorpusPipelineCompositionError(ValueError):
    """A pipeline proof or corpus composition was invalid or non-matched."""


class _FrozenContract(ContractModel):
    """Closed immutable contract used for portable composition artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def _validated_matched_proof(
    proof: PipelineFingerprintVerification, *, role: str
) -> PipelineFingerprintVerification:
    """Reparse and require one independently computed, internally matched proof."""

    try:
        validated = validate_pipeline_verification_integrity(proof)
    except (PipelineFingerprintError, ValueError) as exc:
        raise ValueError(f"corpus_pipeline_{role}_proof_invalid") from exc
    if (
        validated.status != "matched"
        or validated.issues
        or validated.computed is None
        or validated.computed_pipeline_sha256 is None
        or validated.expected_pipeline_sha256 != validated.computed_pipeline_sha256
        or validated.computed.pipeline_sha256 != validated.computed_pipeline_sha256
    ):
        raise ValueError(f"corpus_pipeline_{role}_proof_not_matched")
    return validated


def _ingress_raw_payload(ingress: CorpusIngressProjectionV1) -> dict[str, Any]:
    """Return every caller-supplied ingress value, excluding derived hashes."""

    return {
        "projection_version": ingress.projection_version,
        "ingress_interface": ingress.ingress_interface,
        "corpus_id": ingress.corpus_id,
        "question_id": ingress.question_id,
        "corpus_cutoff": ingress.corpus_cutoff,
        "corpus_source_sha256": ingress.corpus_source_sha256,
        "grounding_package_sha256": ingress.grounding_package_sha256,
        "typed_corpus_sha256": ingress.typed_corpus_sha256,
        "extraction_pipeline_sha256": ingress.extraction_pipeline_sha256,
        "extraction_pipeline_verification_sha256": (
            ingress.extraction_pipeline_verification_sha256
        ),
        "source_manifest_sha256": ingress.source_manifest_sha256,
        "source_membership_sha256": ingress.source_membership_sha256,
        "question_config_sha256": ingress.question_config_sha256,
        "extraction_context_sha256": ingress.extraction_context_sha256,
        "extraction_context_receipt_sha256": (ingress.extraction_context_receipt_sha256),
        "hosted_run_sha256": ingress.hosted_run_sha256,
        "terminal_call_membership_sha256": ingress.terminal_call_membership_sha256,
        "terminal_fragment_membership_sha256": (ingress.terminal_fragment_membership_sha256),
        "grounding_validation_sha256": ingress.grounding_validation_sha256,
        "grounding_replay_sha256": ingress.grounding_replay_sha256,
        "cohort_reconciliation_receipt_sha256": (ingress.cohort_reconciliation_receipt_sha256),
        "reconciled_graph_sha256": ingress.reconciled_graph_sha256,
        "effective_graph_sha256": ingress.effective_graph_sha256,
        "hosted_bridge_receipt_sha256": ingress.hosted_bridge_receipt_sha256,
    }


def _projection_group_hash(
    *, group: str, ingress: CorpusIngressProjectionV1, fields: tuple[str, ...]
) -> str:
    raw = _ingress_raw_payload(ingress)
    return hash_canonical(
        {
            "projection_version": CORPUS_INGRESS_PROJECTION_VERSION,
            "projection_group": group,
            "ingress_interface": ingress.ingress_interface,
            "values": {field: raw[field] for field in fields},
        }
    )


def _expected_ingress_hashes(ingress: CorpusIngressProjectionV1) -> dict[str, str]:
    input_membership_sha256 = hash_canonical(_ingress_raw_payload(ingress))
    source_membership_projection_sha256 = _projection_group_hash(
        group="source-membership",
        ingress=ingress,
        fields=(
            "corpus_id",
            "question_id",
            "corpus_cutoff",
            "corpus_source_sha256",
            "grounding_package_sha256",
            "typed_corpus_sha256",
            "source_manifest_sha256",
            "source_membership_sha256",
        ),
    )
    extraction_execution_projection_sha256 = _projection_group_hash(
        group="extraction-execution",
        ingress=ingress,
        fields=(
            "corpus_id",
            "question_id",
            "corpus_cutoff",
            "grounding_package_sha256",
            "typed_corpus_sha256",
            "extraction_pipeline_sha256",
            "extraction_pipeline_verification_sha256",
            "question_config_sha256",
            "extraction_context_sha256",
            "extraction_context_receipt_sha256",
            "hosted_run_sha256",
            "terminal_call_membership_sha256",
            "hosted_bridge_receipt_sha256",
        ),
    )
    grounding_projection_sha256 = _projection_group_hash(
        group="source-grounding",
        ingress=ingress,
        fields=(
            "corpus_id",
            "question_id",
            "grounding_package_sha256",
            "typed_corpus_sha256",
            "terminal_fragment_membership_sha256",
            "grounding_validation_sha256",
            "grounding_replay_sha256",
        ),
    )
    reconciliation_projection_sha256 = _projection_group_hash(
        group="cohort-reconciliation",
        ingress=ingress,
        fields=(
            "corpus_id",
            "question_id",
            "grounding_package_sha256",
            "typed_corpus_sha256",
            "cohort_reconciliation_receipt_sha256",
            "reconciled_graph_sha256",
            "effective_graph_sha256",
        ),
    )
    ingress_projection_sha256 = hash_canonical(
        {
            "projection_version": CORPUS_INGRESS_PROJECTION_VERSION,
            "ingress_interface": ingress.ingress_interface,
            "ingress_input_membership_sha256": input_membership_sha256,
            "source_membership_projection_sha256": (source_membership_projection_sha256),
            "extraction_execution_projection_sha256": (extraction_execution_projection_sha256),
            "grounding_projection_sha256": grounding_projection_sha256,
            "reconciliation_projection_sha256": reconciliation_projection_sha256,
        }
    )
    return {
        "ingress_input_membership_sha256": input_membership_sha256,
        "source_membership_projection_sha256": (source_membership_projection_sha256),
        "extraction_execution_projection_sha256": (extraction_execution_projection_sha256),
        "grounding_projection_sha256": grounding_projection_sha256,
        "reconciliation_projection_sha256": reconciliation_projection_sha256,
        "ingress_projection_sha256": ingress_projection_sha256,
    }


class CorpusIngressProjectionV1(_FrozenContract):
    """Deterministic, per-corpus projection of the complete v4 ingress lineage."""

    projection_version: Literal["corpus-ingress-projection-v1"] = CORPUS_INGRESS_PROJECTION_VERSION
    ingress_interface: CorpusIngressInterfaceV1
    corpus_id: Annotated[str, Field(min_length=1)]
    question_id: Annotated[str, Field(min_length=1)]
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    corpus_source_sha256: Sha256
    grounding_package_sha256: Sha256
    typed_corpus_sha256: Sha256
    extraction_pipeline_sha256: Sha256
    extraction_pipeline_verification_sha256: Sha256
    source_manifest_sha256: Sha256
    source_membership_sha256: Sha256 | None = None
    question_config_sha256: Sha256
    extraction_context_sha256: Sha256
    extraction_context_receipt_sha256: Sha256
    hosted_run_sha256: Sha256 | None = None
    terminal_call_membership_sha256: Sha256 | None = None
    terminal_fragment_membership_sha256: Sha256
    grounding_validation_sha256: Sha256
    grounding_replay_sha256: Sha256
    cohort_reconciliation_receipt_sha256: Sha256
    reconciled_graph_sha256: Sha256
    effective_graph_sha256: Sha256
    hosted_bridge_receipt_sha256: Sha256 | None = None
    ingress_input_membership_sha256: Sha256
    source_membership_projection_sha256: Sha256
    extraction_execution_projection_sha256: Sha256
    grounding_projection_sha256: Sha256
    reconciliation_projection_sha256: Sha256
    ingress_projection_sha256: Sha256

    @field_validator("corpus_id", "question_id", "corpus_cutoff")
    @classmethod
    def validate_nonempty_identity(cls, value: str, info: Any) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(f"corpus_ingress_{info.field_name}_invalid")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> CorpusIngressProjectionV1:
        hosted_fields = (
            self.source_membership_sha256,
            self.hosted_run_sha256,
            self.terminal_call_membership_sha256,
            self.hosted_bridge_receipt_sha256,
        )
        if self.ingress_interface == "hosted-exact-once-native-grounding-v4":
            if not all(value is not None for value in hosted_fields):
                raise ValueError("hosted_exact_once_ingress_lineage_incomplete")
        elif any(value is not None for value in hosted_fields):
            raise ValueError("native_grounding_v4_ingress_forbids_hosted_lineage")
        if self.reconciled_graph_sha256 != self.effective_graph_sha256:
            raise ValueError("corpus_ingress_effective_graph_not_reconciled_graph")
        expected = _expected_ingress_hashes(self)
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"corpus_ingress_{field}_mismatch")
        return self


def build_corpus_ingress_projection_v1(
    *,
    ingress_interface: CorpusIngressInterfaceV1,
    corpus_id: str,
    question_id: str,
    corpus_cutoff: str,
    corpus_source_sha256: str,
    grounding_package_sha256: str,
    typed_corpus_sha256: str,
    extraction_pipeline_sha256: str,
    extraction_pipeline_verification_sha256: str,
    source_manifest_sha256: str,
    question_config_sha256: str,
    extraction_context_sha256: str,
    extraction_context_receipt_sha256: str,
    terminal_fragment_membership_sha256: str,
    grounding_validation_sha256: str,
    grounding_replay_sha256: str,
    cohort_reconciliation_receipt_sha256: str,
    reconciled_graph_sha256: str,
    effective_graph_sha256: str,
    source_membership_sha256: str | None = None,
    hosted_run_sha256: str | None = None,
    terminal_call_membership_sha256: str | None = None,
    hosted_bridge_receipt_sha256: str | None = None,
) -> CorpusIngressProjectionV1:
    """Build and self-hash one closed corpus-ingress projection."""

    unhashed: dict[str, Any] = {
        "projection_version": CORPUS_INGRESS_PROJECTION_VERSION,
        "ingress_interface": ingress_interface,
        "corpus_id": corpus_id,
        "question_id": question_id,
        "corpus_cutoff": corpus_cutoff,
        "corpus_source_sha256": corpus_source_sha256,
        "grounding_package_sha256": grounding_package_sha256,
        "typed_corpus_sha256": typed_corpus_sha256,
        "extraction_pipeline_sha256": extraction_pipeline_sha256,
        "extraction_pipeline_verification_sha256": (extraction_pipeline_verification_sha256),
        "source_manifest_sha256": source_manifest_sha256,
        "source_membership_sha256": source_membership_sha256,
        "question_config_sha256": question_config_sha256,
        "extraction_context_sha256": extraction_context_sha256,
        "extraction_context_receipt_sha256": extraction_context_receipt_sha256,
        "hosted_run_sha256": hosted_run_sha256,
        "terminal_call_membership_sha256": terminal_call_membership_sha256,
        "terminal_fragment_membership_sha256": terminal_fragment_membership_sha256,
        "grounding_validation_sha256": grounding_validation_sha256,
        "grounding_replay_sha256": grounding_replay_sha256,
        "cohort_reconciliation_receipt_sha256": (cohort_reconciliation_receipt_sha256),
        "reconciled_graph_sha256": reconciled_graph_sha256,
        "effective_graph_sha256": effective_graph_sha256,
        "hosted_bridge_receipt_sha256": hosted_bridge_receipt_sha256,
    }
    # Build an intentionally invalid hash shell only long enough to use the exact
    # model-owned projection algorithm.  ``model_construct`` performs no validation
    # and the fully populated result is then validated normally below.
    shell = CorpusIngressProjectionV1.model_construct(
        **unhashed,
        ingress_input_membership_sha256="0" * 64,
        source_membership_projection_sha256="0" * 64,
        extraction_execution_projection_sha256="0" * 64,
        grounding_projection_sha256="0" * 64,
        reconciliation_projection_sha256="0" * 64,
        ingress_projection_sha256="0" * 64,
    )
    return CorpusIngressProjectionV1.model_validate({**unhashed, **_expected_ingress_hashes(shell)})


class PipelineCompositionContextV1(_FrozenContract):
    """Portable context digest over three internally matched proof objects.

    ``composition_context_sha256`` is not a standard computed-pipeline identity
    and must never be supplied where a ``PipelineFingerprintVerification`` is
    required.  The standard composed calibration identity is produced by
    :func:`build_composed_calibration_pipeline_verification_v1` below.
    """

    composition_version: Literal["pipeline-composition-context-v1"] = (
        PIPELINE_COMPOSITION_CONTEXT_VERSION
    )
    extraction_pipeline_verification: PipelineFingerprintVerification
    verifier_core_pipeline_verification: PipelineFingerprintVerification
    join_policy_pipeline_verification: PipelineFingerprintVerification
    extraction_pipeline_sha256: Sha256
    verifier_core_pipeline_sha256: Sha256
    join_policy_pipeline_sha256: Sha256
    composition_context_sha256: Sha256

    @model_validator(mode="after")
    def validate_composition(self) -> PipelineCompositionContextV1:
        extraction = _validated_matched_proof(
            self.extraction_pipeline_verification, role="extraction"
        )
        core = _validated_matched_proof(
            self.verifier_core_pipeline_verification, role="verifier_core"
        )
        policy = _validated_matched_proof(
            self.join_policy_pipeline_verification, role="join_policy"
        )
        if self.extraction_pipeline_sha256 != extraction.expected_pipeline_sha256:
            raise ValueError("corpus_pipeline_extraction_alias_mismatch")
        if self.verifier_core_pipeline_sha256 != core.expected_pipeline_sha256:
            raise ValueError("corpus_pipeline_verifier_core_alias_mismatch")
        if self.join_policy_pipeline_sha256 != policy.expected_pipeline_sha256:
            raise ValueError("corpus_pipeline_join_policy_alias_mismatch")
        expected = compute_pipeline_composition_context_sha256_v1(
            extraction_pipeline_verification=extraction,
            verifier_core_pipeline_verification=core,
            join_policy_pipeline_verification=policy,
        )
        if self.composition_context_sha256 != expected:
            raise ValueError("corpus_pipeline_composition_context_hash_mismatch")
        return self


def compute_pipeline_composition_context_sha256_v1(
    *,
    extraction_pipeline_verification: PipelineFingerprintVerification,
    verifier_core_pipeline_verification: PipelineFingerprintVerification,
    join_policy_pipeline_verification: PipelineFingerprintVerification,
) -> str:
    """Hash declared proof objects; this is not a standard pipeline fingerprint."""

    extraction = _validated_matched_proof(extraction_pipeline_verification, role="extraction")
    core = _validated_matched_proof(verifier_core_pipeline_verification, role="verifier_core")
    policy = _validated_matched_proof(join_policy_pipeline_verification, role="join_policy")
    return hash_canonical(
        {
            "composition_version": PIPELINE_COMPOSITION_CONTEXT_VERSION,
            "extraction_pipeline_verification": extraction.model_dump(mode="json"),
            "verifier_core_pipeline_verification": core.model_dump(mode="json"),
            "join_policy_pipeline_verification": policy.model_dump(mode="json"),
        }
    )


def build_pipeline_composition_context_v1(
    *,
    extraction_pipeline_verification: PipelineFingerprintVerification,
    verifier_core_pipeline_verification: PipelineFingerprintVerification,
    join_policy_pipeline_verification: PipelineFingerprintVerification,
) -> PipelineCompositionContextV1:
    """Build a portable, non-authorizing proof-composition context."""

    extraction = _validated_matched_proof(extraction_pipeline_verification, role="extraction")
    core = _validated_matched_proof(verifier_core_pipeline_verification, role="verifier_core")
    policy = _validated_matched_proof(join_policy_pipeline_verification, role="join_policy")
    return PipelineCompositionContextV1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=core,
        join_policy_pipeline_verification=policy,
        extraction_pipeline_sha256=extraction.expected_pipeline_sha256,
        verifier_core_pipeline_sha256=core.expected_pipeline_sha256,
        join_policy_pipeline_sha256=policy.expected_pipeline_sha256,
        composition_context_sha256=compute_pipeline_composition_context_sha256_v1(
            extraction_pipeline_verification=extraction,
            verifier_core_pipeline_verification=core,
            join_policy_pipeline_verification=policy,
        ),
    )


def _externally_replay_matched_proof(
    *,
    proof: PipelineFingerprintVerification,
    role: str,
    repository_root: Path,
    current_components: Sequence[PipelineComponentSpec] | None = None,
) -> PipelineFingerprintVerification:
    """Rehash one proof's files from the supplied root and require exact replay."""

    validated = _validated_matched_proof(proof, role=role)
    assert validated.computed is not None
    try:
        replayed = require_pipeline_fingerprint_match(
            expected=validated.computed,
            root=repository_root,
            current_components=current_components,
        )
    except (OSError, PipelineFingerprintError, ValueError) as exc:
        raise CorpusPipelineCompositionError(
            f"corpus_pipeline_{role}_external_replay_failed"
        ) from exc
    if replayed != validated:
        raise CorpusPipelineCompositionError(f"corpus_pipeline_{role}_external_replay_mismatch")
    return replayed


def composed_calibration_pipeline_components_v1(
    *,
    verifier_core_components: Sequence[PipelineComponentSpec],
    ingress_interface: CorpusIngressInterfaceV1,
    extraction_pipeline_verification: PipelineFingerprintVerification,
    verifier_core_pipeline_verification: PipelineFingerprintVerification,
    join_policy_pipeline_verification: PipelineFingerprintVerification,
) -> list[PipelineComponentSpec]:
    """Add a settings-only extraction identity to the exact verifier-core manifest."""

    extraction = _validated_matched_proof(extraction_pipeline_verification, role="extraction")
    core = _validated_matched_proof(verifier_core_pipeline_verification, role="verifier_core")
    policy = _validated_matched_proof(join_policy_pipeline_verification, role="join_policy")
    components = [
        PipelineComponentSpec.model_validate(component.model_dump(mode="json"))
        for component in verifier_core_components
    ]
    component_ids = {component.component_id for component in components}
    if COMPOSED_CALIBRATION_COMPONENT_ID in component_ids:
        raise CorpusPipelineCompositionError("composed_calibration_component_id_collision")
    composition_context_sha256 = compute_pipeline_composition_context_sha256_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=core,
        join_policy_pipeline_verification=policy,
    )
    components.append(
        PipelineComponentSpec(
            component_id=COMPOSED_CALIBRATION_COMPONENT_ID,
            component_version=COMPOSED_CALIBRATION_COMPONENT_VERSION,
            file_paths=[],
            settings={
                "composition_contract": PIPELINE_COMPOSITION_CONTEXT_VERSION,
                "composition_context_sha256": composition_context_sha256,
                "extraction_pipeline_sha256": extraction.expected_pipeline_sha256,
                "extraction_pipeline_verification_sha256": (extraction.verification_sha256),
                "join_policy_pipeline_sha256": policy.expected_pipeline_sha256,
                "join_policy_pipeline_verification_sha256": (policy.verification_sha256),
                "native_ingress_interface": ingress_interface,
                "per_corpus_join_excluded_from_calibration_identity": True,
                "verifier_core_pipeline_sha256": core.expected_pipeline_sha256,
                "verifier_core_pipeline_verification_sha256": (core.verification_sha256),
            },
        )
    )
    return sorted(components, key=lambda component: component.component_id)


def build_composed_calibration_pipeline_verification_v1(
    *,
    repository_root: Path,
    verifier_core_components: Sequence[PipelineComponentSpec],
    join_policy_components: Sequence[PipelineComponentSpec],
    ingress_interface: CorpusIngressInterfaceV1,
    extraction_pipeline_verification: PipelineFingerprintVerification,
    verifier_core_pipeline_verification: PipelineFingerprintVerification,
    join_policy_pipeline_verification: PipelineFingerprintVerification,
    expected: PipelineFingerprint | None = None,
) -> PipelineFingerprintVerification:
    """Build a standard proof from externally prevalidated extraction inputs.

    The resulting ``computed_pipeline_sha256`` is the identity that existing
    item-risk, adaptive-calibration, sequential-state, and release contracts may
    use.  It is extraction-pipeline-specific but excludes per-corpus hashes so a
    calibration population may contain multiple independent questions produced by
    the same frozen extraction pipeline. This low-level object does not prove
    that the extraction proof came from a package/run replay. Production gates
    must require a corpus-specific external-replay receipt that performed that
    replay before calling this function.
    """

    root = repository_root.resolve(strict=True)
    extraction = _externally_replay_matched_proof(
        proof=extraction_pipeline_verification,
        role="extraction",
        repository_root=root,
    )
    core = _externally_replay_matched_proof(
        proof=verifier_core_pipeline_verification,
        role="verifier_core",
        repository_root=root,
        current_components=verifier_core_components,
    )
    policy = _externally_replay_matched_proof(
        proof=join_policy_pipeline_verification,
        role="join_policy",
        repository_root=root,
        current_components=join_policy_components,
    )
    components = composed_calibration_pipeline_components_v1(
        verifier_core_components=verifier_core_components,
        ingress_interface=ingress_interface,
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=core,
        join_policy_pipeline_verification=policy,
    )
    frozen = expected or compute_pipeline_fingerprint(root=root, components=components)
    try:
        return require_pipeline_fingerprint_match(
            expected=frozen,
            root=root,
            current_components=components,
        )
    except (OSError, PipelineFingerprintError, ValueError) as exc:
        raise CorpusPipelineCompositionError(
            "composed_calibration_pipeline_external_replay_failed"
        ) from exc


class CorpusPipelineCompositionJoinV1(_FrozenContract):
    """Portable integrity object for one declared corpus composition.

    This model can prove only that its nested values and aliases are internally
    consistent.  A production gate must additionally require a separate external
    replay receipt that rederives the values from current repository and corpus
    bytes.
    """

    join_version: Literal["corpus-pipeline-composition-join-v1"] = (
        CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION
    )
    compatibility_decision: Literal["portable-integrity-only"] = "portable-integrity-only"
    external_replay_required: Literal[True] = True
    external_replay_completed: Literal[False] = False
    endpoint_identity_equality_required: Literal[False] = False
    extraction_pipeline_verification: PipelineFingerprintVerification
    pipeline_composition_context: PipelineCompositionContextV1
    extraction_pipeline_sha256: Sha256
    verifier_core_pipeline_sha256: Sha256
    join_policy_pipeline_sha256: Sha256
    composition_context_sha256: Sha256
    corpus_ingress: CorpusIngressProjectionV1
    corpus_ingress_projection_sha256: Sha256
    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False
    join_sha256: Sha256

    @model_validator(mode="after")
    def validate_join(self) -> CorpusPipelineCompositionJoinV1:
        extraction = _validated_matched_proof(
            self.extraction_pipeline_verification, role="extraction"
        )
        composition = PipelineCompositionContextV1.model_validate(
            self.pipeline_composition_context.model_dump(mode="json")
        )
        ingress = CorpusIngressProjectionV1.model_validate(
            self.corpus_ingress.model_dump(mode="json")
        )
        aliases = {
            "extraction_pipeline_sha256": extraction.expected_pipeline_sha256,
            "verifier_core_pipeline_sha256": composition.verifier_core_pipeline_sha256,
            "join_policy_pipeline_sha256": composition.join_policy_pipeline_sha256,
            "composition_context_sha256": composition.composition_context_sha256,
            "corpus_ingress_projection_sha256": ingress.ingress_projection_sha256,
        }
        for field, expected in aliases.items():
            if getattr(self, field) != expected:
                raise ValueError(f"corpus_pipeline_join_{field}_mismatch")
        if (
            composition.extraction_pipeline_sha256 != extraction.expected_pipeline_sha256
            or composition.extraction_pipeline_verification.verification_sha256
            != extraction.verification_sha256
        ):
            raise ValueError("corpus_pipeline_join_release_extraction_mismatch")
        if (
            ingress.extraction_pipeline_sha256 != extraction.expected_pipeline_sha256
            or ingress.extraction_pipeline_verification_sha256 != extraction.verification_sha256
        ):
            raise ValueError("corpus_pipeline_join_extraction_ingress_mismatch")
        payload = self.model_dump(mode="json", exclude={"join_sha256"})
        if self.join_sha256 != hash_canonical(payload):
            raise ValueError("corpus_pipeline_join_self_hash_mismatch")
        return self


def build_corpus_pipeline_composition_join_v1(
    *,
    extraction_pipeline_verification: PipelineFingerprintVerification,
    verifier_core_pipeline_verification: PipelineFingerprintVerification,
    join_policy_pipeline_verification: PipelineFingerprintVerification,
    corpus_ingress: CorpusIngressProjectionV1,
) -> CorpusPipelineCompositionJoinV1:
    """Compose three matched proofs with one externally derived corpus ingress."""

    extraction = _validated_matched_proof(extraction_pipeline_verification, role="extraction")
    composition = build_pipeline_composition_context_v1(
        extraction_pipeline_verification=extraction,
        verifier_core_pipeline_verification=verifier_core_pipeline_verification,
        join_policy_pipeline_verification=join_policy_pipeline_verification,
    )
    ingress = CorpusIngressProjectionV1.model_validate(corpus_ingress.model_dump(mode="json"))
    payload = {
        "join_version": CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION,
        "compatibility_decision": "portable-integrity-only",
        "external_replay_required": True,
        "external_replay_completed": False,
        "endpoint_identity_equality_required": False,
        "extraction_pipeline_verification": extraction.model_dump(mode="json"),
        "pipeline_composition_context": composition.model_dump(mode="json"),
        "extraction_pipeline_sha256": extraction.expected_pipeline_sha256,
        "verifier_core_pipeline_sha256": composition.verifier_core_pipeline_sha256,
        "join_policy_pipeline_sha256": composition.join_policy_pipeline_sha256,
        "composition_context_sha256": composition.composition_context_sha256,
        "corpus_ingress": ingress.model_dump(mode="json"),
        "corpus_ingress_projection_sha256": ingress.ingress_projection_sha256,
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    return CorpusPipelineCompositionJoinV1.model_validate(
        {**payload, "join_sha256": hash_canonical(payload)}
    )


def validate_corpus_pipeline_composition_join_v1(
    join: CorpusPipelineCompositionJoinV1,
) -> CorpusPipelineCompositionJoinV1:
    """Reparse a join so post-construction nested mutation fails closed."""

    if not isinstance(join, CorpusPipelineCompositionJoinV1):
        raise CorpusPipelineCompositionError("corpus_pipeline_join_contract_invalid")
    try:
        return CorpusPipelineCompositionJoinV1.model_validate(join.model_dump(mode="json"))
    except (PipelineFingerprintError, ValueError) as exc:
        raise CorpusPipelineCompositionError("corpus_pipeline_join_integrity_changed") from exc


__all__ = [
    "COMPOSED_CALIBRATION_COMPONENT_ID",
    "COMPOSED_CALIBRATION_COMPONENT_VERSION",
    "CORPUS_INGRESS_PROJECTION_VERSION",
    "CORPUS_PIPELINE_COMPOSITION_JOIN_VERSION",
    "PIPELINE_COMPOSITION_CONTEXT_VERSION",
    "CorpusIngressInterfaceV1",
    "CorpusIngressProjectionV1",
    "CorpusPipelineCompositionError",
    "CorpusPipelineCompositionJoinV1",
    "PipelineCompositionContextV1",
    "build_composed_calibration_pipeline_verification_v1",
    "build_corpus_ingress_projection_v1",
    "build_corpus_pipeline_composition_join_v1",
    "build_pipeline_composition_context_v1",
    "composed_calibration_pipeline_components_v1",
    "compute_pipeline_composition_context_sha256_v1",
    "validate_corpus_pipeline_composition_join_v1",
]
