"""Prospective corpus and policy identities for composed extraction pipelines.

The v1 complete-corpus identity proves publication membership, but it does not
identify the extraction pipeline that produced the corpus.  Conversely, a
``CorpusPipelineCompositionJoinV1`` proves an exact extraction/verifier/ingress
composition, but is deliberately scoped to one question and carries no
calibration or release authority.

This module joins those two already-validated contracts without changing either
one.  ``CompleteCorpusIdentityV2`` retains every v1 membership field and its
original self-hash, then adds the extraction pipeline, corpus-independent
calibration pipeline, exact per-question join, and ingress projection.  Its own
composition hash is distinct from the embedded v1 membership hash.

``ManifestCorpusPolicyBindingV2`` gives a future sequential ledger one hash that
binds the claim manifest, complete V2 corpus, exact per-question join,
calibration pipeline, and deployed policy.  These are lineage artifacts only:
they confer no extraction-accuracy, calibration, scientific, or release
authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.adaptive_calibration import CompleteCorpusIdentity
from literature_multiverse.corpus_pipeline_composition import (
    COMPOSED_CALIBRATION_COMPONENT_ID,
    COMPOSED_CALIBRATION_COMPONENT_VERSION,
    PIPELINE_COMPOSITION_CONTEXT_VERSION,
    CorpusPipelineCompositionError,
    CorpusPipelineCompositionJoinV1,
    validate_corpus_pipeline_composition_join_v1,
)
from literature_multiverse.corpus_pipeline_composition_runtime import (
    CorpusPipelineCompositionExternalReplayReceiptV1,
    CorpusPipelineCompositionRuntimeError,
    require_external_replay_receipt_matches_corpus_load_result_v1,
    validate_corpus_pipeline_composition_external_replay_receipt_v1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprintError,
    PipelineFingerprintVerification,
    validate_pipeline_verification_integrity,
)

COMPLETE_CORPUS_IDENTITY_V2_VERSION = "complete-corpus-membership-v2"
MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION = "manifest-corpus-policy-binding-v2"
COMPLETE_CORPUS_IDENTITY_V3_VERSION = "complete-corpus-membership-v3"
MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION = "manifest-corpus-policy-binding-v3"

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class ComposedCorpusIdentityError(ValueError):
    """A composed corpus or manifest-policy identity failed closed."""


class _FrozenContract(ContractModel):
    """Closed immutable contract with explicit whitespace semantics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def _validated_v1_membership(
    membership: CompleteCorpusIdentity,
) -> CompleteCorpusIdentity:
    """Reparse v1 so mutable nested publication lists cannot bypass validation."""

    if not isinstance(membership, CompleteCorpusIdentity):
        raise ComposedCorpusIdentityError("complete_corpus_membership_v1_contract_invalid")
    try:
        return CompleteCorpusIdentity.model_validate(membership.model_dump(mode="json"))
    except ValueError as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_membership_v1_integrity_changed"
        ) from exc


def _validated_join(
    join: CorpusPipelineCompositionJoinV1,
) -> CorpusPipelineCompositionJoinV1:
    """Reparse the full join, including all three independently matched proofs."""

    try:
        return validate_corpus_pipeline_composition_join_v1(join)
    except (CorpusPipelineCompositionError, ValueError) as exc:
        raise ComposedCorpusIdentityError("complete_corpus_pipeline_join_invalid") from exc


def _validated_composed_pipeline_proof(
    proof: PipelineFingerprintVerification,
    *,
    join: CorpusPipelineCompositionJoinV1,
) -> PipelineFingerprintVerification:
    """Require the standard composed proof and its exact join-independent aliases."""

    if not isinstance(proof, PipelineFingerprintVerification):
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_proof_contract_invalid"
        )
    try:
        validated = validate_pipeline_verification_integrity(proof)
    except (PipelineFingerprintError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_proof_invalid"
        ) from exc
    if (
        validated.status != "matched"
        or validated.issues
        or validated.computed is None
        or validated.computed_pipeline_sha256 is None
        or validated.expected_pipeline_sha256 != validated.computed_pipeline_sha256
        or validated.computed.pipeline_sha256 != validated.computed_pipeline_sha256
    ):
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_proof_not_matched"
        )

    special = [
        component
        for component in validated.computed.components
        if component.component_id == COMPOSED_CALIBRATION_COMPONENT_ID
    ]
    if len(special) != 1:
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_component_missing_or_duplicated"
        )
    component = special[0]
    if (
        component.component_version != COMPOSED_CALIBRATION_COMPONENT_VERSION
        or component.files
    ):
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_component_shape_invalid"
        )

    context = join.pipeline_composition_context
    expected_settings = {
        "composition_contract": PIPELINE_COMPOSITION_CONTEXT_VERSION,
        "composition_context_sha256": join.composition_context_sha256,
        "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
        "extraction_pipeline_verification_sha256": (
            join.extraction_pipeline_verification.verification_sha256
        ),
        "join_policy_pipeline_sha256": join.join_policy_pipeline_sha256,
        "join_policy_pipeline_verification_sha256": (
            context.join_policy_pipeline_verification.verification_sha256
        ),
        "native_ingress_interface": join.corpus_ingress.ingress_interface,
        "per_corpus_join_excluded_from_calibration_identity": True,
        "verifier_core_pipeline_sha256": join.verifier_core_pipeline_sha256,
        "verifier_core_pipeline_verification_sha256": (
            context.verifier_core_pipeline_verification.verification_sha256
        ),
    }
    if component.settings != expected_settings:
        raise ComposedCorpusIdentityError(
            "composed_calibration_pipeline_component_alias_mismatch"
        )

    core = context.verifier_core_pipeline_verification
    if core.computed is None:
        raise ComposedCorpusIdentityError(
            "composed_calibration_verifier_core_computed_missing"
        )
    composed_core_components = [
        candidate
        for candidate in validated.computed.components
        if candidate.component_id != COMPOSED_CALIBRATION_COMPONENT_ID
    ]
    if [item.model_dump(mode="json") for item in composed_core_components] != [
        item.model_dump(mode="json") for item in core.computed.components
    ]:
        raise ComposedCorpusIdentityError(
            "composed_calibration_verifier_core_components_mismatch"
        )
    return validated


def _require_membership_join_aliases(
    *,
    membership: CompleteCorpusIdentity,
    join: CorpusPipelineCompositionJoinV1,
) -> None:
    """Require the v1 membership and join to describe the same exact corpus."""

    ingress = join.corpus_ingress
    aliases: tuple[tuple[str, Any, Any], ...] = (
        ("corpus_id", membership.corpus_id, ingress.corpus_id),
        (
            "corpus_source_sha256",
            membership.corpus_source_sha256,
            ingress.corpus_source_sha256,
        ),
        ("corpus_cutoff", membership.corpus_cutoff, ingress.corpus_cutoff),
        (
            "source_manifest_sha256",
            membership.source_manifest_sha256,
            ingress.source_manifest_sha256,
        ),
    )
    for field, membership_value, ingress_value in aliases:
        if membership_value != ingress_value:
            raise ValueError(f"complete_corpus_v2_{field}_join_alias_mismatch")
    # Every currently supported composed ingress is backed by an exact source
    # manifest.  A publication-list-only v1 identity cannot prove it is the same
    # membership universe and therefore must not be silently promoted to V2.
    if (
        membership.membership_basis != "source_manifest"
        or membership.source_manifest_sha256 is None
    ):
        raise ValueError("complete_corpus_v2_source_manifest_membership_required")


class CompleteCorpusIdentityV2(_FrozenContract):
    """Complete v1 membership plus an exact extraction/calibration corpus join."""

    identity_version: Literal["complete-corpus-membership-v2"] = (
        COMPLETE_CORPUS_IDENTITY_V2_VERSION
    )
    complete_corpus_membership_v1: CompleteCorpusIdentity

    # Exact aliases retain every v1 membership field at the V2 boundary.  The
    # embedded object also retains the v1 ``identity_version`` and validates its
    # original self-hash independently.
    corpus_id: Annotated[str, Field(min_length=1)]
    corpus_source_sha256: Sha256
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    membership_basis: Literal["source_manifest", "frozen_corpus_publications"]
    publication_ids: list[str]
    source_manifest_sha256: Sha256 | None = None
    membership_sha256: Sha256

    corpus_pipeline_join: CorpusPipelineCompositionJoinV1
    composed_pipeline_verification: PipelineFingerprintVerification
    composed_pipeline_verification_sha256: Sha256
    extraction_pipeline_sha256: Sha256
    calibration_pipeline_sha256: Sha256
    corpus_pipeline_join_sha256: Sha256
    corpus_ingress_projection_sha256: Sha256

    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False

    # This is the V2 self-hash.  It is intentionally not called
    # ``membership_sha256`` because that name remains the exact embedded v1 hash.
    membership_composition_sha256: Sha256

    @field_validator("corpus_id", "corpus_cutoff")
    @classmethod
    def validate_nonempty_identity(cls, value: str, info: Any) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(f"complete_corpus_v2_{info.field_name}_invalid")
        return value

    @field_validator("publication_ids")
    @classmethod
    def validate_publications(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("complete_corpus_v2_publications_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> CompleteCorpusIdentityV2:
        membership = _validated_v1_membership(self.complete_corpus_membership_v1)
        join = _validated_join(self.corpus_pipeline_join)
        composed = _validated_composed_pipeline_proof(
            self.composed_pipeline_verification,
            join=join,
        )

        membership_aliases: tuple[tuple[str, Any], ...] = (
            ("corpus_id", membership.corpus_id),
            ("corpus_source_sha256", membership.corpus_source_sha256),
            ("corpus_cutoff", membership.corpus_cutoff),
            ("membership_basis", membership.membership_basis),
            ("publication_ids", membership.publication_ids),
            ("source_manifest_sha256", membership.source_manifest_sha256),
            ("membership_sha256", membership.membership_sha256),
        )
        for field, expected in membership_aliases:
            if getattr(self, field) != expected:
                raise ValueError(f"complete_corpus_v2_{field}_v1_alias_mismatch")

        _require_membership_join_aliases(membership=membership, join=join)
        join_aliases = {
            "composed_pipeline_verification_sha256": composed.verification_sha256,
            "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
            "calibration_pipeline_sha256": composed.computed_pipeline_sha256,
            "corpus_pipeline_join_sha256": join.join_sha256,
            "corpus_ingress_projection_sha256": (
                join.corpus_ingress_projection_sha256
            ),
        }
        for field, expected in join_aliases.items():
            if getattr(self, field) != expected:
                raise ValueError(f"complete_corpus_v2_{field}_join_alias_mismatch")

        payload = self.model_dump(
            mode="json", exclude={"membership_composition_sha256"}
        )
        expected_hash = hash_canonical(payload)
        if self.membership_composition_sha256 != expected_hash:
            raise ValueError("complete_corpus_v2_membership_composition_hash_mismatch")
        if self.membership_composition_sha256 == membership.membership_sha256:
            raise ValueError("complete_corpus_v2_hash_domain_not_distinct_from_v1")
        return self


def freeze_complete_corpus_identity_v2(
    *,
    complete_corpus_membership_v1: CompleteCorpusIdentity,
    corpus_pipeline_join: CorpusPipelineCompositionJoinV1,
    composed_pipeline_verification: PipelineFingerprintVerification,
) -> CompleteCorpusIdentityV2:
    """Validate and freeze a complete membership under one exact composed join."""

    membership = _validated_v1_membership(complete_corpus_membership_v1)
    join = _validated_join(corpus_pipeline_join)
    composed = _validated_composed_pipeline_proof(
        composed_pipeline_verification,
        join=join,
    )
    assert composed.computed_pipeline_sha256 is not None
    _require_membership_join_aliases(membership=membership, join=join)
    payload: dict[str, Any] = {
        "identity_version": COMPLETE_CORPUS_IDENTITY_V2_VERSION,
        "complete_corpus_membership_v1": membership.model_dump(mode="json"),
        "corpus_id": membership.corpus_id,
        "corpus_source_sha256": membership.corpus_source_sha256,
        "corpus_cutoff": membership.corpus_cutoff,
        "membership_basis": membership.membership_basis,
        "publication_ids": membership.publication_ids,
        "source_manifest_sha256": membership.source_manifest_sha256,
        "membership_sha256": membership.membership_sha256,
        "corpus_pipeline_join": join.model_dump(mode="json"),
        "composed_pipeline_verification": composed.model_dump(mode="json"),
        "composed_pipeline_verification_sha256": composed.verification_sha256,
        "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
        "calibration_pipeline_sha256": composed.computed_pipeline_sha256,
        "corpus_pipeline_join_sha256": join.join_sha256,
        "corpus_ingress_projection_sha256": join.corpus_ingress_projection_sha256,
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    return CompleteCorpusIdentityV2.model_validate(
        {
            **payload,
            "membership_composition_sha256": hash_canonical(payload),
        }
    )


def validate_complete_corpus_identity_v2(
    identity: CompleteCorpusIdentityV2,
) -> CompleteCorpusIdentityV2:
    """Reparse V2 so mutations below frozen outer models fail closed."""

    if not isinstance(identity, CompleteCorpusIdentityV2):
        raise ComposedCorpusIdentityError("complete_corpus_identity_v2_contract_invalid")
    try:
        return CompleteCorpusIdentityV2.model_validate(identity.model_dump(mode="json"))
    except (ComposedCorpusIdentityError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_identity_v2_integrity_changed"
        ) from exc


class ManifestCorpusPolicyBindingV2(_FrozenContract):
    """One exact manifest/corpus/join/calibration/policy ledger identity."""

    binding_version: Literal["manifest-corpus-policy-binding-v2"] = (
        MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION
    )
    claim_manifest_sha256: Sha256
    complete_corpus_identity_v2: CompleteCorpusIdentityV2
    complete_corpus_membership_v2_sha256: Sha256
    corpus_pipeline_join: CorpusPipelineCompositionJoinV1
    composed_pipeline_verification: PipelineFingerprintVerification
    composed_pipeline_verification_sha256: Sha256
    corpus_pipeline_join_sha256: Sha256
    corpus_ingress_projection_sha256: Sha256
    extraction_pipeline_sha256: Sha256
    calibration_pipeline_sha256: Sha256
    policy_sha256: Sha256

    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False

    manifest_corpus_policy_binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> ManifestCorpusPolicyBindingV2:
        complete = validate_complete_corpus_identity_v2(
            self.complete_corpus_identity_v2
        )
        join = _validated_join(self.corpus_pipeline_join)
        composed = _validated_composed_pipeline_proof(
            self.composed_pipeline_verification,
            join=join,
        )

        if complete.corpus_pipeline_join != join:
            raise ValueError("manifest_corpus_policy_complete_join_object_mismatch")
        if complete.composed_pipeline_verification != composed:
            raise ValueError(
                "manifest_corpus_policy_complete_composed_proof_object_mismatch"
            )
        aliases = {
            "complete_corpus_membership_v2_sha256": (
                complete.membership_composition_sha256
            ),
            "composed_pipeline_verification_sha256": composed.verification_sha256,
            "corpus_pipeline_join_sha256": join.join_sha256,
            "corpus_ingress_projection_sha256": (
                join.corpus_ingress_projection_sha256
            ),
            "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
            "calibration_pipeline_sha256": composed.computed_pipeline_sha256,
        }
        for field, expected in aliases.items():
            if getattr(self, field) != expected:
                raise ValueError(f"manifest_corpus_policy_{field}_alias_mismatch")

        if (
            complete.corpus_pipeline_join_sha256 != join.join_sha256
            or complete.corpus_ingress_projection_sha256
            != join.corpus_ingress_projection_sha256
            or complete.extraction_pipeline_sha256 != join.extraction_pipeline_sha256
            or complete.composed_pipeline_verification_sha256
            != composed.verification_sha256
            or complete.calibration_pipeline_sha256
            != composed.computed_pipeline_sha256
        ):
            raise ValueError("manifest_corpus_policy_complete_corpus_alias_mismatch")

        payload = self.model_dump(
            mode="json", exclude={"manifest_corpus_policy_binding_sha256"}
        )
        if self.manifest_corpus_policy_binding_sha256 != hash_canonical(payload):
            raise ValueError("manifest_corpus_policy_binding_hash_mismatch")
        return self


def freeze_manifest_corpus_policy_binding_v2(
    *,
    claim_manifest_sha256: str,
    complete_corpus_identity_v2: CompleteCorpusIdentityV2,
    corpus_pipeline_join: CorpusPipelineCompositionJoinV1,
    composed_pipeline_verification: PipelineFingerprintVerification,
    policy_sha256: str,
) -> ManifestCorpusPolicyBindingV2:
    """Freeze the single policy identity a future sequential ledger can carry."""

    complete = validate_complete_corpus_identity_v2(complete_corpus_identity_v2)
    join = _validated_join(corpus_pipeline_join)
    composed = _validated_composed_pipeline_proof(
        composed_pipeline_verification,
        join=join,
    )
    assert composed.computed_pipeline_sha256 is not None
    if complete.corpus_pipeline_join != join:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_complete_join_object_mismatch"
        )
    if complete.composed_pipeline_verification != composed:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_complete_composed_proof_object_mismatch"
        )
    payload: dict[str, Any] = {
        "binding_version": MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION,
        "claim_manifest_sha256": claim_manifest_sha256,
        "complete_corpus_identity_v2": complete.model_dump(mode="json"),
        "complete_corpus_membership_v2_sha256": (
            complete.membership_composition_sha256
        ),
        "corpus_pipeline_join": join.model_dump(mode="json"),
        "composed_pipeline_verification": composed.model_dump(mode="json"),
        "composed_pipeline_verification_sha256": composed.verification_sha256,
        "corpus_pipeline_join_sha256": join.join_sha256,
        "corpus_ingress_projection_sha256": join.corpus_ingress_projection_sha256,
        "extraction_pipeline_sha256": join.extraction_pipeline_sha256,
        "calibration_pipeline_sha256": composed.computed_pipeline_sha256,
        "policy_sha256": policy_sha256,
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    try:
        return ManifestCorpusPolicyBindingV2.model_validate(
            {
                **payload,
                "manifest_corpus_policy_binding_sha256": hash_canonical(payload),
            }
        )
    except ValueError as exc:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v2_invalid"
        ) from exc


def validate_manifest_corpus_policy_binding_v2(
    binding: ManifestCorpusPolicyBindingV2,
) -> ManifestCorpusPolicyBindingV2:
    """Reparse a binding and all nested identities before ledger use."""

    if not isinstance(binding, ManifestCorpusPolicyBindingV2):
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v2_contract_invalid"
        )
    try:
        return ManifestCorpusPolicyBindingV2.model_validate(
            binding.model_dump(mode="json")
        )
    except (ComposedCorpusIdentityError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v2_integrity_changed"
        ) from exc


def _validated_external_replay_receipt(
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> CorpusPipelineCompositionExternalReplayReceiptV1:
    """Reparse one receipt without treating its self-hash as current-byte replay."""

    if not isinstance(receipt, CorpusPipelineCompositionExternalReplayReceiptV1):
        raise ComposedCorpusIdentityError(
            "complete_corpus_external_replay_receipt_contract_invalid"
        )
    try:
        return CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
            receipt.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_external_replay_receipt_integrity_changed"
        ) from exc


class CompleteCorpusIdentityV3(_FrozenContract):
    """V2 corpus composition plus the exact runtime external-replay receipt.

    Deserializing this object proves closed-contract integrity, not that repository
    bytes are still current.  Authoritative runtime consumers must additionally
    call :func:`validate_complete_corpus_identity_v3_external_replay` with the
    package and bridge paths before using its pipeline aliases.
    """

    identity_version: Literal["complete-corpus-membership-v3"] = (
        COMPLETE_CORPUS_IDENTITY_V3_VERSION
    )
    complete_corpus_identity_v2: CompleteCorpusIdentityV2
    external_replay_receipt: CorpusPipelineCompositionExternalReplayReceiptV1

    corpus_id: Annotated[str, Field(min_length=1)]
    corpus_source_sha256: Sha256
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    membership_basis: Literal["source_manifest", "frozen_corpus_publications"]
    publication_ids: list[str]
    source_manifest_sha256: Sha256 | None = None
    membership_sha256: Sha256
    membership_composition_v2_sha256: Sha256

    external_replay_receipt_sha256: Sha256
    corpus_pipeline_join_sha256: Sha256
    corpus_ingress_projection_sha256: Sha256
    extraction_pipeline_sha256: Sha256
    calibration_pipeline_sha256: Sha256
    release_pipeline_sha256: Sha256
    composed_pipeline_verification_sha256: Sha256

    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False

    membership_composition_sha256: Sha256

    @field_validator("corpus_id", "corpus_cutoff")
    @classmethod
    def validate_nonempty_identity(cls, value: str, info: Any) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(f"complete_corpus_v3_{info.field_name}_invalid")
        return value

    @field_validator("publication_ids")
    @classmethod
    def validate_publications(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("complete_corpus_v3_publications_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> CompleteCorpusIdentityV3:
        complete_v2 = validate_complete_corpus_identity_v2(
            self.complete_corpus_identity_v2
        )
        receipt = _validated_external_replay_receipt(self.external_replay_receipt)
        if (
            receipt.composition_join != complete_v2.corpus_pipeline_join
            or receipt.composed_pipeline_verification
            != complete_v2.composed_pipeline_verification
        ):
            raise ValueError("complete_corpus_v3_receipt_v2_object_mismatch")

        aliases: tuple[tuple[str, Any], ...] = (
            ("corpus_id", complete_v2.corpus_id),
            ("corpus_source_sha256", complete_v2.corpus_source_sha256),
            ("corpus_cutoff", complete_v2.corpus_cutoff),
            ("membership_basis", complete_v2.membership_basis),
            ("publication_ids", complete_v2.publication_ids),
            ("source_manifest_sha256", complete_v2.source_manifest_sha256),
            ("membership_sha256", complete_v2.membership_sha256),
            (
                "membership_composition_v2_sha256",
                complete_v2.membership_composition_sha256,
            ),
            ("external_replay_receipt_sha256", receipt.receipt_sha256),
            ("corpus_pipeline_join_sha256", receipt.composition_join_sha256),
            (
                "corpus_ingress_projection_sha256",
                receipt.corpus_ingress_projection_sha256,
            ),
            (
                "extraction_pipeline_sha256",
                receipt.composition_join.extraction_pipeline_sha256,
            ),
            ("calibration_pipeline_sha256", receipt.calibration_pipeline_sha256),
            ("release_pipeline_sha256", receipt.release_pipeline_sha256),
            (
                "composed_pipeline_verification_sha256",
                receipt.composed_pipeline_verification_sha256,
            ),
        )
        for field, expected in aliases:
            if getattr(self, field) != expected:
                raise ValueError(f"complete_corpus_v3_{field}_alias_mismatch")
        if self.calibration_pipeline_sha256 != self.release_pipeline_sha256:
            raise ValueError("complete_corpus_v3_pipeline_domain_mismatch")
        payload = self.model_dump(
            mode="json", exclude={"membership_composition_sha256"}
        )
        if self.membership_composition_sha256 != hash_canonical(payload):
            raise ValueError("complete_corpus_v3_membership_composition_hash_mismatch")
        if self.membership_composition_sha256 in {
            self.membership_sha256,
            self.membership_composition_v2_sha256,
            self.external_replay_receipt_sha256,
        }:
            raise ValueError("complete_corpus_v3_hash_domain_not_distinct")
        return self


def freeze_complete_corpus_identity_v3(
    *,
    complete_corpus_membership_v1: CompleteCorpusIdentity,
    external_replay_receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
    loaded_corpus: Any,
) -> CompleteCorpusIdentityV3:
    """Freeze V3 after matching the receipt to the exact loaded corpus object."""

    membership = _validated_v1_membership(complete_corpus_membership_v1)
    receipt = _validated_external_replay_receipt(external_replay_receipt)
    try:
        receipt = require_external_replay_receipt_matches_corpus_load_result_v1(
            receipt=receipt,
            corpus=loaded_corpus,
        )
    except CorpusPipelineCompositionRuntimeError as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_v3_loaded_corpus_replay_mismatch"
        ) from exc
    complete_v2 = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=membership,
        corpus_pipeline_join=receipt.composition_join,
        composed_pipeline_verification=receipt.composed_pipeline_verification,
    )
    payload: dict[str, Any] = {
        "identity_version": COMPLETE_CORPUS_IDENTITY_V3_VERSION,
        "complete_corpus_identity_v2": complete_v2,
        "external_replay_receipt": receipt,
        "corpus_id": complete_v2.corpus_id,
        "corpus_source_sha256": complete_v2.corpus_source_sha256,
        "corpus_cutoff": complete_v2.corpus_cutoff,
        "membership_basis": complete_v2.membership_basis,
        "publication_ids": complete_v2.publication_ids,
        "source_manifest_sha256": complete_v2.source_manifest_sha256,
        "membership_sha256": complete_v2.membership_sha256,
        "membership_composition_v2_sha256": (
            complete_v2.membership_composition_sha256
        ),
        "external_replay_receipt_sha256": receipt.receipt_sha256,
        "corpus_pipeline_join_sha256": receipt.composition_join_sha256,
        "corpus_ingress_projection_sha256": (
            receipt.corpus_ingress_projection_sha256
        ),
        "extraction_pipeline_sha256": (
            receipt.composition_join.extraction_pipeline_sha256
        ),
        "calibration_pipeline_sha256": receipt.calibration_pipeline_sha256,
        "release_pipeline_sha256": receipt.release_pipeline_sha256,
        "composed_pipeline_verification_sha256": (
            receipt.composed_pipeline_verification_sha256
        ),
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    try:
        return CompleteCorpusIdentityV3.model_validate(
            {
                **payload,
                "membership_composition_sha256": hash_canonical(payload),
            }
        )
    except ValueError as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_identity_v3_invalid"
        ) from exc


def validate_complete_corpus_identity_v3(
    identity: CompleteCorpusIdentityV3,
) -> CompleteCorpusIdentityV3:
    """Validate serialized integrity without claiming current repository replay."""

    if not isinstance(identity, CompleteCorpusIdentityV3):
        raise ComposedCorpusIdentityError("complete_corpus_identity_v3_contract_invalid")
    try:
        return CompleteCorpusIdentityV3.model_validate(identity.model_dump(mode="json"))
    except (ComposedCorpusIdentityError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_identity_v3_integrity_changed"
        ) from exc


def validate_complete_corpus_identity_v3_external_replay(
    *,
    identity: CompleteCorpusIdentityV3,
    loaded_corpus: Any,
    repository_root: Path,
    grounding_package_path: Path,
    hosted_bridge_receipt_path: Path,
) -> CompleteCorpusIdentityV3:
    """Rebuild current bytes and require the exact persisted V3 identity."""

    validated = validate_complete_corpus_identity_v3(identity)
    try:
        receipt = validate_corpus_pipeline_composition_external_replay_receipt_v1(
            receipt=validated.external_replay_receipt,
            repository_root=repository_root,
            grounding_package_path=grounding_package_path,
            hosted_bridge_receipt_path=hosted_bridge_receipt_path,
        )
        require_external_replay_receipt_matches_corpus_load_result_v1(
            receipt=receipt,
            corpus=loaded_corpus,
        )
    except CorpusPipelineCompositionRuntimeError as exc:
        raise ComposedCorpusIdentityError(
            "complete_corpus_identity_v3_external_replay_failed"
        ) from exc
    rebuilt = freeze_complete_corpus_identity_v3(
        complete_corpus_membership_v1=(
            validated.complete_corpus_identity_v2.complete_corpus_membership_v1
        ),
        external_replay_receipt=receipt,
        loaded_corpus=loaded_corpus,
    )
    if rebuilt != validated:
        raise ComposedCorpusIdentityError(
            "complete_corpus_identity_v3_external_replay_mismatch"
        )
    return validated


class ManifestCorpusPolicyBindingV3(_FrozenContract):
    """Manifest/policy ledger bound to the receipt-bearing V3 corpus identity."""

    binding_version: Literal["manifest-corpus-policy-binding-v3"] = (
        MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION
    )
    claim_manifest_sha256: Sha256
    complete_corpus_identity_v3: CompleteCorpusIdentityV3
    complete_corpus_membership_v3_sha256: Sha256
    external_replay_receipt_sha256: Sha256
    corpus_pipeline_join_sha256: Sha256
    corpus_ingress_projection_sha256: Sha256
    extraction_pipeline_sha256: Sha256
    calibration_pipeline_sha256: Sha256
    release_pipeline_sha256: Sha256
    policy_sha256: Sha256

    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_claim_truth_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False

    manifest_corpus_policy_binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> ManifestCorpusPolicyBindingV3:
        complete = validate_complete_corpus_identity_v3(
            self.complete_corpus_identity_v3
        )
        aliases = {
            "complete_corpus_membership_v3_sha256": (
                complete.membership_composition_sha256
            ),
            "external_replay_receipt_sha256": (
                complete.external_replay_receipt_sha256
            ),
            "corpus_pipeline_join_sha256": complete.corpus_pipeline_join_sha256,
            "corpus_ingress_projection_sha256": (
                complete.corpus_ingress_projection_sha256
            ),
            "extraction_pipeline_sha256": complete.extraction_pipeline_sha256,
            "calibration_pipeline_sha256": complete.calibration_pipeline_sha256,
            "release_pipeline_sha256": complete.release_pipeline_sha256,
        }
        for field, expected in aliases.items():
            if getattr(self, field) != expected:
                raise ValueError(f"manifest_corpus_policy_v3_{field}_alias_mismatch")
        payload = self.model_dump(
            mode="json", exclude={"manifest_corpus_policy_binding_sha256"}
        )
        if self.manifest_corpus_policy_binding_sha256 != hash_canonical(payload):
            raise ValueError("manifest_corpus_policy_v3_binding_hash_mismatch")
        return self


def freeze_manifest_corpus_policy_binding_v3(
    *,
    claim_manifest_sha256: str,
    complete_corpus_identity_v3: CompleteCorpusIdentityV3,
    policy_sha256: str,
) -> ManifestCorpusPolicyBindingV3:
    """Freeze one manifest/policy binding around the exact V3 corpus receipt."""

    complete = validate_complete_corpus_identity_v3(complete_corpus_identity_v3)
    payload: dict[str, Any] = {
        "binding_version": MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION,
        "claim_manifest_sha256": claim_manifest_sha256,
        "complete_corpus_identity_v3": complete,
        "complete_corpus_membership_v3_sha256": (
            complete.membership_composition_sha256
        ),
        "external_replay_receipt_sha256": complete.external_replay_receipt_sha256,
        "corpus_pipeline_join_sha256": complete.corpus_pipeline_join_sha256,
        "corpus_ingress_projection_sha256": (
            complete.corpus_ingress_projection_sha256
        ),
        "extraction_pipeline_sha256": complete.extraction_pipeline_sha256,
        "calibration_pipeline_sha256": complete.calibration_pipeline_sha256,
        "release_pipeline_sha256": complete.release_pipeline_sha256,
        "policy_sha256": policy_sha256,
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    try:
        return ManifestCorpusPolicyBindingV3.model_validate(
            {
                **payload,
                "manifest_corpus_policy_binding_sha256": hash_canonical(payload),
            }
        )
    except ValueError as exc:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v3_invalid"
        ) from exc


def validate_manifest_corpus_policy_binding_v3(
    binding: ManifestCorpusPolicyBindingV3,
) -> ManifestCorpusPolicyBindingV3:
    """Reparse a V3 policy binding and its complete receipt-bearing corpus."""

    if not isinstance(binding, ManifestCorpusPolicyBindingV3):
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v3_contract_invalid"
        )
    try:
        return ManifestCorpusPolicyBindingV3.model_validate(
            binding.model_dump(mode="json")
        )
    except (ComposedCorpusIdentityError, ValueError) as exc:
        raise ComposedCorpusIdentityError(
            "manifest_corpus_policy_binding_v3_integrity_changed"
        ) from exc


__all__ = [
    "COMPLETE_CORPUS_IDENTITY_V2_VERSION",
    "COMPLETE_CORPUS_IDENTITY_V3_VERSION",
    "MANIFEST_CORPUS_POLICY_BINDING_V2_VERSION",
    "MANIFEST_CORPUS_POLICY_BINDING_V3_VERSION",
    "CompleteCorpusIdentityV2",
    "CompleteCorpusIdentityV3",
    "ComposedCorpusIdentityError",
    "ManifestCorpusPolicyBindingV2",
    "ManifestCorpusPolicyBindingV3",
    "freeze_complete_corpus_identity_v2",
    "freeze_complete_corpus_identity_v3",
    "freeze_manifest_corpus_policy_binding_v2",
    "freeze_manifest_corpus_policy_binding_v3",
    "validate_complete_corpus_identity_v2",
    "validate_complete_corpus_identity_v3",
    "validate_complete_corpus_identity_v3_external_replay",
    "validate_manifest_corpus_policy_binding_v2",
    "validate_manifest_corpus_policy_binding_v3",
]
