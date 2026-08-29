"""Additive, fail-closed bridge from MetaSyn v2 grounded packets to typed corpora.

The bridge preserves the complete 32-publication terminal ledger.  It also projects
completed ``GroundedTypedEffectV2`` values into the pre-existing evidence graph so
that the quantitative kernels can be exercised.  That projection is deliberately
named a *quantitative compatibility projection*: legacy sentinel values for
significance, equivalence, moderators, and risk of bias are storage adapters, not
claims that those scientific properties were absent or assessed.

Nothing in this module authorizes synthesis conclusions or claim release.  External
replay revalidates the immutable v5 source surface, the additive execution bundle,
every inventory, packet, grounding, and assembly receipt, and every source byte.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.effects import (
    EquivalenceConclusion,
    ReportedSignificance,
)
from literature_multiverse.evidence_graph import (
    ArmNode,
    CohortIdentity,
    CohortIdentityBasis,
    CohortNode,
    ContrastNode,
    EvidenceGraph,
    EvidenceSpan,
    EvidenceSpanRole,
    OutcomeEstimateNode,
    PublicationIdentity,
    StudyNode,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    validate_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynExtractionRowInputV2,
    MetaSynPacketCandidateInputV2,
    validate_metasyn_packet_candidate_input_v2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.metasyn_v5_source_surface import (
    MetaSynV5SourceSurfaceRowV1,
    MetaSynV5SourceSurfaceV1,
    freeze_metasyn_v5_source_surface,
    validate_metasyn_v5_source_surface,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import NativeEffectPayload, NativeEvidenceSpan
from literature_multiverse.native_packet_assembly_v2 import (
    GroundedTypedEffectV2,
    NativePacketAssemblyCompletedV2,
    NativePacketAssemblyOutcomeV2,
    freeze_packet_assembly_protocol_orientation_v2,
    replay_metasyn_question_projection_spec_v2,
    validate_native_packet_assembly_v2,
)
from literature_multiverse.native_packet_grounding_v2 import (
    PacketGroundingAbstentionReceiptV2,
    PacketGroundingCompletedReceiptV2,
    PacketGroundingReceiptV2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.native_question_projection import QuestionProjectionSpecV1
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    SourceDocumentArtifact,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

BRIDGE_VERSION = "metasyn-grounded-publication-corpus-bridge-v2"
BRIDGE_COMPONENT_VERSION = "1"
TERMINAL_VERSION = "metasyn-grounded-candidate-terminal-v2"
PUBLICATION_JOIN_VERSION = "metasyn-grounded-publication-join-v2"
QUESTION_CORPUS_VERSION = "metasyn-grounded-question-corpus-v2"
EXPECTED_PUBLICATION_COUNT = 32
EXPECTED_QUESTION_COUNT = 10
BRIDGE_MODULE_PATH = "src/literature_multiverse/metasyn_grounded_publication_bridge_v2.py"

_GROUNDING_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)
_ASSEMBLY_ADAPTER = TypeAdapter(NativePacketAssemblyOutcomeV2)


class MetaSynGroundedPublicationBridgeV2Error(ValueError):
    """A row, receipt, or compatibility projection cannot be replayed exactly."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    if getattr(model, field_name) != hash_canonical(
        model.model_dump(mode="json", exclude={field_name})
    ):
        raise ValueError(f"metasyn_grounded_bridge_v2_self_hash_mismatch:{field_name}")


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{hash_canonical({'kind': prefix, 'identity': dict(payload)})[:40]}"


def _sorted_unique(values: Sequence[str]) -> list[str]:
    return sorted(set(values))


class MetaSynOptionalScientificCoverageV2(_FrozenExactModel):
    """Exact coverage ledger for fields lost in the legacy compatibility shape."""

    coverage_version: Literal["metasyn-optional-scientific-coverage-v2"] = (
        "metasyn-optional-scientific-coverage-v2"
    )
    typed_effect_sha256: Sha256
    positive_direction_coverage: Literal[
        "prespecified_in_frozen_protocol",
        "not_prespecified_in_frozen_protocol",
    ]
    reported_significance_coverage: Literal[
        "not_extracted_from_selected_support",
        "p_value_only_extracted_conclusion_not_extracted",
    ]
    equivalence_coverage: Literal[
        "not_extracted_from_selected_support",
        "margin_only_extracted_conclusion_not_extracted",
    ]
    moderator_coverage: Literal["not_extracted_from_selected_support"]
    analysis_population_coverage: Literal[
        "grounded_exact_text",
        "not_extracted_from_selected_support",
    ]
    compatibility_reported_significance_sentinel: Literal["not_reported"] = "not_reported"
    compatibility_equivalence_sentinel: Literal["not_tested"] = "not_tested"
    compatibility_moderators_storage_sentinel: Literal["empty_mapping"] = "empty_mapping"
    legacy_sentinels_are_scientific_observations: Literal[False] = False
    coverage_blockers: list[str]
    coverage_sha256: Sha256

    @field_validator("coverage_blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or not value:
            raise ValueError("metasyn_grounded_bridge_v2_coverage_blockers_invalid")
        return value

    @model_validator(mode="after")
    def validate_coverage(self) -> MetaSynOptionalScientificCoverageV2:
        expected = {
            "equivalence_conclusion_not_extracted",
            "moderators_not_extracted",
            "reported_significance_conclusion_not_extracted",
        }
        if self.positive_direction_coverage == "not_prespecified_in_frozen_protocol":
            expected.add("claim_direction_not_prespecified")
        if self.analysis_population_coverage == "not_extracted_from_selected_support":
            expected.add("analysis_population_not_extracted")
        if self.coverage_blockers != sorted(expected):
            raise ValueError("metasyn_grounded_bridge_v2_coverage_blocker_mismatch")
        _self_hash(self, "coverage_sha256")
        return self


class MetaSynQuantitativeEffectCompatibilityV2(_FrozenExactModel):
    """One exact typed-effect-to-graph projection plus its loss ledger."""

    compatibility_version: Literal["metasyn-quantitative-effect-compatibility-v2"] = (
        "metasyn-quantitative-effect-compatibility-v2"
    )
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    grounding_receipt_sha256: Sha256
    assembly_receipt_sha256: Sha256
    typed_effect_sha256: Sha256
    quantitative_signature_sha256: Sha256
    quantitative_content_sha256: Sha256
    canonical_outcome_id: str
    protocol_outcome_text: str
    outcome_concept_quote: str
    estimate_id: str
    evidence_span_id: str
    exact_source_locator: str
    exact_evidence_quote: str
    exact_line_ids: list[str]
    exact_source_char_start: Annotated[int, Field(ge=0)]
    exact_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    exact_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    exact_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    coverage: MetaSynOptionalScientificCoverageV2
    coverage_sha256: Sha256
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: Literal[True] = True
    scientific_optional_field_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    compatibility_sha256: Sha256

    @field_validator("exact_line_ids")
    @classmethod
    def validate_line_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or not value:
            raise ValueError("metasyn_grounded_bridge_v2_line_ids_invalid")
        return value

    @model_validator(mode="after")
    def validate_compatibility(self) -> MetaSynQuantitativeEffectCompatibilityV2:
        if self.coverage_sha256 != self.coverage.coverage_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_coverage_hash_alias_mismatch")
        if self.exact_source_char_end_exclusive - self.exact_source_char_start != len(
            self.exact_evidence_quote
        ):
            raise ValueError("metasyn_grounded_bridge_v2_source_quote_offsets_invalid")
        if self.exact_source_utf8_byte_end_exclusive - self.exact_source_utf8_byte_start != len(
            self.exact_evidence_quote.encode("utf-8")
        ):
            raise ValueError("metasyn_grounded_bridge_v2_source_byte_offsets_invalid")
        if self.outcome_concept_quote not in self.protocol_outcome_text:
            raise ValueError("metasyn_grounded_bridge_v2_outcome_quote_not_exact")
        _self_hash(self, "compatibility_sha256")
        return self


class MetaSynGroundedCandidateTerminalV2(_FrozenExactModel):
    """One candidate's complete packet/grounding/assembly terminal envelope."""

    terminal_version: Literal["metasyn-grounded-candidate-terminal-v2"] = TERMINAL_VERSION
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: str
    candidate_index: Annotated[int, Field(ge=1)]
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    packet_input: MetaSynPacketCandidateInputV2
    packet_input_sha256: Sha256
    grounding_receipt: PacketGroundingReceiptV2
    grounding_receipt_sha256: Sha256
    assembly_receipt: NativePacketAssemblyOutcomeV2
    assembly_receipt_sha256: Sha256
    terminal_status: Literal["typed_effect_completed", "unable_to_assemble"]
    terminal_blockers: list[str]
    authorizes_typed_effect: bool
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @field_validator("terminal_blockers")
    @classmethod
    def validate_terminal_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_grounded_bridge_v2_terminal_blockers_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_terminal(self) -> MetaSynGroundedCandidateTerminalV2:
        aliases = {
            "row_ordinal": self.packet_input.row_ordinal,
            "row_key": self.packet_input.row_key,
            "candidate_index": self.packet_input.candidate.candidate_index,
            "candidate_descriptor_sha256": self.packet_input.candidate_descriptor_sha256,
            "candidate_binding_sha256": self.packet_input.candidate_binding_sha256,
            "packet_input_sha256": self.packet_input.packet_input_sha256,
            "grounding_receipt_sha256": self.grounding_receipt.receipt_sha256,
            "assembly_receipt_sha256": self.assembly_receipt.assembly_receipt_sha256,
            "terminal_status": self.assembly_receipt.status,
            "authorizes_typed_effect": isinstance(
                self.assembly_receipt, NativePacketAssemblyCompletedV2
            ),
        }
        if any(getattr(self, key) != value for key, value in aliases.items()):
            raise ValueError("metasyn_grounded_bridge_v2_terminal_alias_mismatch")
        if self.grounding_receipt.candidate_binding.binding_sha256 != (
            self.candidate_binding_sha256
        ):
            raise ValueError("metasyn_grounded_bridge_v2_grounding_binding_mismatch")
        if self.assembly_receipt.grounding_receipt_sha256 != (self.grounding_receipt_sha256):
            raise ValueError("metasyn_grounded_bridge_v2_assembly_grounding_mismatch")
        expected_blockers = [
            *(
                [f"grounding:{self.grounding_receipt.model_outcome.reason}"]
                if isinstance(self.grounding_receipt, PacketGroundingAbstentionReceiptV2)
                else []
            ),
            *(
                []
                if isinstance(self.assembly_receipt, NativePacketAssemblyCompletedV2)
                else [
                    *(f"assembly:{item}" for item in self.assembly_receipt.blocker_codes),
                    *(f"missing:{item}" for item in self.assembly_receipt.missing_field_paths),
                ]
            ),
        ]
        if self.terminal_blockers != sorted(expected_blockers):
            raise ValueError("metasyn_grounded_bridge_v2_terminal_blocker_mismatch")
        _self_hash(self, "terminal_sha256")
        return self


class MetaSynGroundedPublicationJoinV2(_FrozenExactModel):
    """One publication retained with its full terminal and coverage accounting."""

    publication_join_version: Literal["metasyn-grounded-publication-join-v2"] = (
        PUBLICATION_JOIN_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: str
    question_id: str
    source_surface_row_sha256: Sha256
    extraction_row_input_sha256: Sha256
    publication_source_identity_sha256: Sha256
    source_record_sha256: Sha256
    publication: PublicationIdentity
    source_document: SourceDocumentArtifact
    source_artifact_binding_sha256: Sha256
    source_strength_surface_sha256: Sha256
    source_strength_blockers: list[str]
    protocol_outcome_text_by_id: dict[str, str]
    protocol_outcome_membership_sha256: Sha256
    inventory_receipt: MetaSynCandidateInventoryReceiptV2
    inventory_receipt_sha256: Sha256
    inventory_status: Literal[
        "candidates_authorized",
        "no_candidate_non_authorizing",
        "capacity_or_uncertainty_non_authorizing",
    ]
    candidate_descriptor_sha256s: list[Sha256]
    candidate_terminal_sha256s: list[Sha256]
    candidate_terminals: list[MetaSynGroundedCandidateTerminalV2]
    inventoried_candidate_count: Annotated[int, Field(ge=0)]
    authorized_candidate_count: Annotated[int, Field(ge=0)]
    terminal_candidate_count: Annotated[int, Field(ge=0)]
    completed_candidate_count: Annotated[int, Field(ge=0)]
    abstained_candidate_count: Annotated[int, Field(ge=0)]
    compatibility_effects: list[MetaSynQuantitativeEffectCompatibilityV2]
    compatibility_effect_membership_sha256: Sha256
    coverage_blockers: list[str]
    compatibility_fragment: PublicationEvidenceFragment
    compatibility_fragment_sha256: Sha256
    exact_terminal_roster_complete: Literal[True] = True
    exact_protocol_outcome_mapping_authority: Literal[True] = True
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: bool
    scientific_optional_field_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    publication_join_sha256: Sha256

    @field_validator(
        "source_strength_blockers",
        "candidate_descriptor_sha256s",
        "candidate_terminal_sha256s",
        "coverage_blockers",
    )
    @classmethod
    def validate_canonical_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"metasyn_grounded_bridge_v2_list_not_canonical:{info.field_name}")
        return value

    @field_validator("protocol_outcome_text_by_id")
    @classmethod
    def validate_outcome_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or value != dict(sorted(value.items())):
            raise ValueError("metasyn_grounded_bridge_v2_outcome_map_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_join(self) -> MetaSynGroundedPublicationJoinV2:
        if self.source_record_sha256 != hash_canonical(
            {
                "publication": self.publication,
                "source_document": self.source_document,
            }
        ):
            raise ValueError("metasyn_grounded_bridge_v2_publication_identity_mismatch")
        if self.protocol_outcome_membership_sha256 != hash_canonical(
            self.protocol_outcome_text_by_id
        ):
            raise ValueError("metasyn_grounded_bridge_v2_outcome_membership_mismatch")
        if self.inventory_receipt_sha256 != self.inventory_receipt.receipt_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_inventory_hash_alias_mismatch")
        if self.inventory_status != self.inventory_receipt.status:
            raise ValueError("metasyn_grounded_bridge_v2_inventory_status_alias_mismatch")
        if self.candidate_terminals != sorted(
            self.candidate_terminals, key=lambda item: item.candidate_index
        ):
            raise ValueError("metasyn_grounded_bridge_v2_terminals_not_candidate_order")
        descriptors = sorted(
            item.descriptor_sha256 for item in self.inventory_receipt.inventory.candidates
        )
        terminal_descriptors = sorted(
            item.candidate_descriptor_sha256 for item in self.candidate_terminals
        )
        if self.candidate_descriptor_sha256s != descriptors:
            raise ValueError("metasyn_grounded_bridge_v2_candidate_roster_mismatch")
        expected_terminal_descriptors = (
            descriptors if self.inventory_status == "candidates_authorized" else []
        )
        if terminal_descriptors != expected_terminal_descriptors:
            raise ValueError("metasyn_grounded_bridge_v2_terminal_roster_incomplete")
        if self.candidate_terminal_sha256s != sorted(
            item.terminal_sha256 for item in self.candidate_terminals
        ):
            raise ValueError("metasyn_grounded_bridge_v2_terminal_membership_mismatch")
        completed = sum(item.authorizes_typed_effect for item in self.candidate_terminals)
        expected_authorized_count = (
            len(descriptors) if self.inventory_status == "candidates_authorized" else 0
        )
        if (
            self.inventoried_candidate_count,
            self.authorized_candidate_count,
            self.terminal_candidate_count,
            self.completed_candidate_count,
            self.abstained_candidate_count,
        ) != (
            len(descriptors),
            expected_authorized_count,
            len(self.candidate_terminals),
            completed,
            len(self.candidate_terminals) - completed,
        ):
            raise ValueError("metasyn_grounded_bridge_v2_terminal_counts_mismatch")
        compatibility_hashes = sorted(
            item.compatibility_sha256 for item in self.compatibility_effects
        )
        if hash_canonical(compatibility_hashes) != (self.compatibility_effect_membership_sha256):
            raise ValueError("metasyn_grounded_bridge_v2_effect_membership_mismatch")
        if len(self.compatibility_effects) != self.completed_candidate_count:
            raise ValueError("metasyn_grounded_bridge_v2_effect_count_mismatch")
        if self.compatibility_fragment_sha256 != self.compatibility_fragment.fragment_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_fragment_hash_alias_mismatch")
        if self.compatibility_fragment.publication != self.publication:
            raise ValueError("metasyn_grounded_bridge_v2_fragment_publication_mismatch")
        expected_compatibility = self.completed_candidate_count > 0
        if self.quantitative_kernel_compatibility != expected_compatibility:
            raise ValueError("metasyn_grounded_bridge_v2_quantitative_status_mismatch")
        expected_status = (
            FragmentStatus.ESTIMABLE if expected_compatibility else FragmentStatus.NON_ESTIMABLE
        )
        if self.compatibility_fragment.status is not expected_status:
            raise ValueError("metasyn_grounded_bridge_v2_fragment_status_mismatch")
        _self_hash(self, "publication_join_sha256")
        return self


class MetaSynGroundedQuestionCorpusV2(_FrozenExactModel):
    """Question-scoped compatibility corpus; questions are never silently mixed."""

    question_corpus_version: Literal["metasyn-grounded-question-corpus-v2"] = (
        QUESTION_CORPUS_VERSION
    )
    question_id: str
    publication_join_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    publication_ids: Annotated[list[str], Field(min_length=1)]
    compatibility_corpus: TypedEvidenceCorpus
    compatibility_corpus_sha256: Sha256
    estimable_publication_count: Annotated[int, Field(ge=0)]
    non_estimable_publication_count: Annotated[int, Field(ge=0)]
    quantitative_effect_count: Annotated[int, Field(ge=0)]
    coverage_blockers: list[str]
    exact_projection_authority: Literal[True] = True
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: bool
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    question_corpus_sha256: Sha256

    @field_validator("publication_join_sha256s", "publication_ids", "coverage_blockers")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"metasyn_grounded_bridge_v2_question_list_not_canonical:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_question_corpus(self) -> MetaSynGroundedQuestionCorpusV2:
        corpus = self.compatibility_corpus
        if self.compatibility_corpus_sha256 != corpus.corpus_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_corpus_hash_alias_mismatch")
        if self.question_id != corpus.question_id:
            raise ValueError("metasyn_grounded_bridge_v2_corpus_question_mismatch")
        if self.publication_ids != sorted(item.publication_id for item in corpus.fragments):
            raise ValueError("metasyn_grounded_bridge_v2_corpus_publications_mismatch")
        if (
            self.estimable_publication_count,
            self.non_estimable_publication_count,
        ) != (
            len(corpus.estimable_publication_ids),
            len(corpus.non_estimable_publication_ids),
        ):
            raise ValueError("metasyn_grounded_bridge_v2_corpus_counts_mismatch")
        if self.quantitative_effect_count != len(corpus.graph.outcome_estimates):
            raise ValueError("metasyn_grounded_bridge_v2_corpus_effect_count_mismatch")
        if self.quantitative_kernel_compatibility != bool(self.quantitative_effect_count):
            raise ValueError("metasyn_grounded_bridge_v2_question_compatibility_mismatch")
        _self_hash(self, "question_corpus_sha256")
        return self


class MetaSynGroundedPublicationCorpusBridgeV2(_FrozenExactModel):
    """All-32 run package containing ten question-scoped compatibility corpora."""

    bridge_version: Literal["metasyn-grounded-publication-corpus-bridge-v2"] = BRIDGE_VERSION
    status: Literal["externally_replayable_complete_terminal_roster"] = (
        "externally_replayable_complete_terminal_roster"
    )
    execution_bundle: MetaSynPassageHostedExecutionBundleV2
    execution_bundle_sha256: Sha256
    source_surface: MetaSynV5SourceSurfaceV1
    source_surface_sha256: Sha256
    inventory_receipt_membership_sha256: Sha256
    terminal_membership_sha256: Sha256
    bridge_pipeline_fingerprint: PipelineFingerprint
    bridge_pipeline_sha256: Sha256
    publication_joins: Annotated[
        list[MetaSynGroundedPublicationJoinV2],
        Field(min_length=EXPECTED_PUBLICATION_COUNT, max_length=EXPECTED_PUBLICATION_COUNT),
    ]
    publication_join_membership_sha256: Sha256
    question_corpora: Annotated[
        list[MetaSynGroundedQuestionCorpusV2],
        Field(min_length=EXPECTED_QUESTION_COUNT, max_length=EXPECTED_QUESTION_COUNT),
    ]
    question_corpus_membership_sha256: Sha256
    question_count: Literal[10] = EXPECTED_QUESTION_COUNT
    publication_count: Literal[32] = EXPECTED_PUBLICATION_COUNT
    inventoried_candidate_count: Annotated[int, Field(ge=0)]
    authorized_candidate_count: Annotated[int, Field(ge=0)]
    terminal_candidate_count: Annotated[int, Field(ge=0)]
    completed_candidate_count: Annotated[int, Field(ge=0)]
    abstained_candidate_count: Annotated[int, Field(ge=0)]
    estimable_publication_count: Annotated[int, Field(ge=0)]
    quantitative_effect_count: Annotated[int, Field(ge=0)]
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    v5_hosted_outputs_consumed: Literal[False] = False
    v2_terminal_outputs_consumed: bool
    legacy_v4_grounding_package_emitted: Literal[False] = False
    exact_projection_authority: Literal[True] = True
    graph_construction_authority: Literal[True] = True
    quantitative_kernel_compatibility: bool
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    bridge_sha256: Sha256

    @model_validator(mode="after")
    def validate_bridge(self) -> MetaSynGroundedPublicationCorpusBridgeV2:
        if self.execution_bundle_sha256 != self.execution_bundle.execution_bundle_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_execution_hash_alias_mismatch")
        if self.source_surface_sha256 != self.source_surface.source_surface_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_source_hash_alias_mismatch")
        if self.bridge_pipeline_sha256 != self.bridge_pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_pipeline_hash_alias_mismatch")
        if len(self.bridge_pipeline_fingerprint.components) != 1:
            raise ValueError("metasyn_grounded_bridge_v2_pipeline_component_count_mismatch")
        component = self.bridge_pipeline_fingerprint.components[0]
        expected_settings = {
            "all_32_publications_required": True,
            "execution_bundle_sha256": self.execution_bundle_sha256,
            "source_surface_sha256": self.source_surface_sha256,
            "inventory_receipt_membership_sha256": (self.inventory_receipt_membership_sha256),
            "terminal_membership_sha256": self.terminal_membership_sha256,
            "question_scoped_corpora_required": True,
            "legacy_v4_grounding_package_emitted": False,
            "legacy_optional_sentinels_are_scientific_observations": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        if (
            component.component_id != "metasyn-grounded-publication-corpus-bridge-v2"
            or component.component_version != BRIDGE_COMPONENT_VERSION
            or [item.path for item in component.files] != [BRIDGE_MODULE_PATH]
            or component.settings != expected_settings
        ):
            raise ValueError("metasyn_grounded_bridge_v2_pipeline_component_mismatch")
        if (
            self.execution_bundle.extraction_inputs.upstream_source_surface_sha256
            != self.source_surface_sha256
        ):
            raise ValueError("metasyn_grounded_bridge_v2_upstream_source_alias_mismatch")
        if [item.row_ordinal for item in self.publication_joins] != list(
            range(EXPECTED_PUBLICATION_COUNT)
        ):
            raise ValueError("metasyn_grounded_bridge_v2_row_roster_invalid")
        if [item.row_key for item in self.publication_joins] != sorted(
            {item.row_key for item in self.publication_joins}
        ):
            raise ValueError("metasyn_grounded_bridge_v2_row_keys_invalid")
        if (
            hash_canonical([item.publication_join_sha256 for item in self.publication_joins])
            != self.publication_join_membership_sha256
        ):
            raise ValueError("metasyn_grounded_bridge_v2_join_membership_mismatch")
        if (
            hash_canonical([item.inventory_receipt_sha256 for item in self.publication_joins])
            != self.inventory_receipt_membership_sha256
        ):
            raise ValueError("metasyn_grounded_bridge_v2_inventory_membership_mismatch")
        expected_terminal_membership = [
            {
                "row_key": item.row_key,
                "terminal_sha256s": [
                    terminal.terminal_sha256 for terminal in item.candidate_terminals
                ],
            }
            for item in self.publication_joins
        ]
        if hash_canonical(expected_terminal_membership) != self.terminal_membership_sha256:
            raise ValueError("metasyn_grounded_bridge_v2_terminal_membership_mismatch")
        for source_row, extraction_row, joined in zip(
            self.source_surface.rows,
            self.execution_bundle.extraction_inputs.rows,
            self.publication_joins,
            strict=True,
        ):
            if (
                source_row.row_ordinal != joined.row_ordinal
                or extraction_row.row_ordinal != joined.row_ordinal
                or source_row.row_key != joined.row_key
                or extraction_row.row_key != joined.row_key
                or source_row.source_surface_row_sha256 != joined.source_surface_row_sha256
                or extraction_row.row_input_sha256 != joined.extraction_row_input_sha256
                or source_row.row_source_identity_sha256
                != joined.publication_source_identity_sha256
                or source_row.source_row.source_record.publication != joined.publication
                or source_row.source_row.source_record.source_document != joined.source_document
            ):
                raise ValueError("metasyn_grounded_bridge_v2_embedded_row_join_mismatch")
        if [item.question_id for item in self.question_corpora] != sorted(
            {item.question_id for item in self.question_corpora}
        ):
            raise ValueError("metasyn_grounded_bridge_v2_question_roster_invalid")
        if (
            hash_canonical([item.question_corpus_sha256 for item in self.question_corpora])
            != self.question_corpus_membership_sha256
        ):
            raise ValueError("metasyn_grounded_bridge_v2_question_membership_mismatch")
        joins_by_question: dict[str, list[MetaSynGroundedPublicationJoinV2]] = defaultdict(list)
        for item in self.publication_joins:
            joins_by_question[item.question_id].append(item)
        for corpus in self.question_corpora:
            question_joins = joins_by_question.get(corpus.question_id, [])
            if corpus.publication_join_sha256s != sorted(
                item.publication_join_sha256 for item in question_joins
            ) or corpus.publication_ids != sorted(
                item.publication.publication_id for item in question_joins
            ):
                raise ValueError("metasyn_grounded_bridge_v2_question_join_projection_mismatch")
        expected_counts = (
            sum(item.inventoried_candidate_count for item in self.publication_joins),
            sum(item.authorized_candidate_count for item in self.publication_joins),
            sum(len(item.candidate_terminals) for item in self.publication_joins),
            sum(item.completed_candidate_count for item in self.publication_joins),
            sum(item.abstained_candidate_count for item in self.publication_joins),
            sum(item.quantitative_kernel_compatibility for item in self.publication_joins),
            sum(len(item.compatibility_effects) for item in self.publication_joins),
        )
        if (
            self.inventoried_candidate_count,
            self.authorized_candidate_count,
            self.terminal_candidate_count,
            self.completed_candidate_count,
            self.abstained_candidate_count,
            self.estimable_publication_count,
            self.quantitative_effect_count,
        ) != expected_counts:
            raise ValueError("metasyn_grounded_bridge_v2_global_counts_mismatch")
        if self.v2_terminal_outputs_consumed != bool(self.terminal_candidate_count):
            raise ValueError("metasyn_grounded_bridge_v2_output_consumption_mismatch")
        if self.quantitative_kernel_compatibility != bool(self.quantitative_effect_count):
            raise ValueError("metasyn_grounded_bridge_v2_global_compatibility_mismatch")
        _self_hash(self, "bridge_sha256")
        return self


def _protocol_for_row(row: MetaSynExtractionRowInputV2) -> QuestionProjectionSpecV1:
    protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
    if protocol.question_spec_sha256 != (row.projection_v2.lineage_binding.question_spec_sha256):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_protocol_question_spec_mismatch"
        )
    return protocol


def _native_effect_payload(effect: GroundedTypedEffectV2) -> NativeEffectPayload:
    source = effect.effect.model_dump(mode="json")
    source.pop("effect_kind", None)
    source.pop("source_measurement_unit", None)
    integer_fields = {
        "treatment_n",
        "control_n",
        "treatment_events",
        "treatment_total",
        "control_events",
        "control_total",
    }
    for key, value in list(source.items()):
        if key == "effect_format" or value is None or key == "unit":
            continue
        source[key] = int(value) if key in integer_fields else float(Decimal(value))
    source.update(
        {
            "reported_p_value": (
                float(Decimal(effect.reported_p_value))
                if effect.reported_p_value is not None
                else None
            ),
            "reported_significance": ReportedSignificance.NOT_REPORTED,
            "equivalence_conclusion": EquivalenceConclusion.NOT_TESTED,
            "equivalence_margin": (
                float(Decimal(effect.equivalence_margin))
                if effect.equivalence_margin is not None
                else None
            ),
            "moderators": [],
            "extraction_method": effect.extraction_method,
        }
    )
    return NativeEffectPayload.model_validate(source)


def _coverage_for_effect(effect: GroundedTypedEffectV2) -> MetaSynOptionalScientificCoverageV2:
    blockers = {
        "equivalence_conclusion_not_extracted",
        "moderators_not_extracted",
        "reported_significance_conclusion_not_extracted",
    }
    if effect.positive_direction_coverage == "not_prespecified_in_frozen_protocol":
        blockers.add("claim_direction_not_prespecified")
    if effect.analysis_population_coverage == "not_extracted_from_selected_support":
        blockers.add("analysis_population_not_extracted")
    payload = {
        "coverage_version": "metasyn-optional-scientific-coverage-v2",
        "typed_effect_sha256": effect.typed_effect_sha256,
        "positive_direction_coverage": effect.positive_direction_coverage,
        "reported_significance_coverage": effect.reported_significance_coverage,
        "equivalence_coverage": effect.equivalence_coverage,
        "moderator_coverage": effect.moderator_coverage,
        "analysis_population_coverage": effect.analysis_population_coverage,
        "compatibility_reported_significance_sentinel": "not_reported",
        "compatibility_equivalence_sentinel": "not_tested",
        "compatibility_moderators_storage_sentinel": "empty_mapping",
        "legacy_sentinels_are_scientific_observations": False,
        "coverage_blockers": sorted(blockers),
    }
    return MetaSynOptionalScientificCoverageV2.model_validate(
        {**payload, "coverage_sha256": hash_canonical(payload)}
    )


def _identity_values(
    receipt: PacketGroundingCompletedReceiptV2,
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for item in receipt.identity_receipts:
        values[item.field_path].append(item.verbatim_identity_text)
    return {key: sorted(set(items)) for key, items in sorted(values.items())}


def _project_completed_effect(
    *,
    source_row: MetaSynV5SourceSurfaceRowV1,
    terminal: MetaSynGroundedCandidateTerminalV2,
) -> tuple[
    MetaSynQuantitativeEffectCompatibilityV2,
    StudyNode,
    CohortNode,
    list[ArmNode],
    ContrastNode,
    OutcomeEstimateNode,
    EvidenceSpan,
]:
    if not isinstance(terminal.assembly_receipt, NativePacketAssemblyCompletedV2):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_noncompleted_effect_projection"
        )
    if not isinstance(terminal.grounding_receipt, PacketGroundingCompletedReceiptV2):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_completed_assembly_without_grounding"
        )
    assembly = terminal.assembly_receipt
    grounding = terminal.grounding_receipt
    effect = assembly.typed_effect
    evidence = grounding.evidence_receipt
    publication = source_row.source_row.source_record.publication
    if effect.publication_source_identity_sha256 != source_row.row_source_identity_sha256:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_typed_effect_source_identity_mismatch"
        )

    effect_measure_identity = {
        "effect_kind": effect.effect.effect_kind,
        "effect_format": effect.effect.effect_format,
        "unit": getattr(effect.effect, "unit", None),
        "source_measurement_unit": getattr(effect.effect, "source_measurement_unit", None),
    }
    signature_payload = {
        "publication_source_identity_sha256": effect.publication_source_identity_sha256,
        "study_key": effect.study_key,
        "cohort_key": effect.cohort_key,
        "treatment_arm_key": effect.treatment_arm_key,
        "comparator_arm_key": effect.comparator_arm_key,
        "contrast_key": effect.contrast_key,
        "outcome_id": effect.canonical_outcome_id,
        "timepoint": effect.timepoint,
        "analysis_population": effect.analysis_population,
        "effect_measure_identity": effect_measure_identity,
        "analysis_policy_sha256": effect.analysis_policy_sha256,
    }
    quantitative_signature_sha256 = hash_canonical(signature_payload)
    quantitative_content_sha256 = hash_canonical(
        {
            "effect": effect.effect,
            "reported_p_value": effect.reported_p_value,
            "equivalence_margin": effect.equivalence_margin,
        }
    )
    outcome_text = source_row.question_spec.outcome_id_to_text.get(effect.canonical_outcome_id)
    if outcome_text is None or effect.outcome_concept_quote not in outcome_text:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_effect_outcome_not_exact_protocol"
        )
    estimate_id = _stable_id("estimate", signature_payload)
    span_id = _stable_id(
        "span",
        {
            "publication_id": publication.publication_id,
            "evidence_receipt_sha256": evidence.evidence_receipt_sha256,
        },
    )
    coverage = _coverage_for_effect(effect)
    compatibility_payload = {
        "compatibility_version": "metasyn-quantitative-effect-compatibility-v2",
        "candidate_descriptor_sha256": terminal.candidate_descriptor_sha256,
        "candidate_binding_sha256": terminal.candidate_binding_sha256,
        "grounding_receipt_sha256": terminal.grounding_receipt_sha256,
        "assembly_receipt_sha256": terminal.assembly_receipt_sha256,
        "typed_effect_sha256": effect.typed_effect_sha256,
        "quantitative_signature_sha256": quantitative_signature_sha256,
        "quantitative_content_sha256": quantitative_content_sha256,
        "canonical_outcome_id": effect.canonical_outcome_id,
        "protocol_outcome_text": outcome_text,
        "outcome_concept_quote": effect.outcome_concept_quote,
        "estimate_id": estimate_id,
        "evidence_span_id": span_id,
        "exact_source_locator": evidence.source_locator,
        "exact_evidence_quote": evidence.evidence_quote,
        "exact_line_ids": [evidence.line_id],
        "exact_source_char_start": evidence.quote_source_char_start,
        "exact_source_char_end_exclusive": evidence.quote_source_char_end_exclusive,
        "exact_source_utf8_byte_start": evidence.quote_source_utf8_byte_start,
        "exact_source_utf8_byte_end_exclusive": (evidence.quote_source_utf8_byte_end_exclusive),
        "coverage": coverage,
        "coverage_sha256": coverage.coverage_sha256,
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": True,
        "scientific_optional_field_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    compatibility = MetaSynQuantitativeEffectCompatibilityV2.model_validate(
        {
            **compatibility_payload,
            "compatibility_sha256": hash_canonical(compatibility_payload),
        }
    )

    identity = _identity_values(grounding)
    registry_ids = identity.get("study.registration_id", []) + identity.get(
        "cohort.registry_id", []
    )
    dataset_ids = identity.get("cohort.dataset_id", [])
    cohort_labels = list(effect.cohort_source_labels)
    if registry_ids:
        basis = CohortIdentityBasis.REPORTED_REGISTRY_ID
    elif dataset_ids:
        basis = CohortIdentityBasis.REPORTED_DATASET_ID
    else:
        basis = CohortIdentityBasis.SOURCE_REPORTED_LABEL
    cohort_identity = CohortIdentity(
        cohort_id=effect.cohort_key,
        basis=basis,
        source_labels=sorted(cohort_labels),
        registry_ids=sorted(set(registry_ids)),
        dataset_ids=sorted(set(dataset_ids)),
    )
    study = StudyNode(
        study_id=effect.study_key,
        publication_ids=[publication.publication_id],
        primary_publication_id=publication.publication_id,
        registration_ids=sorted(set(identity.get("study.registration_id", []))),
    )
    cohort = CohortNode(identity=cohort_identity, study_id=effect.study_key)
    arms = [
        ArmNode(
            arm_id=effect.treatment_arm_key,
            cohort_id=effect.cohort_key,
            label=effect.treatment_arm_label,
            role=effect.treatment_arm_role,
        ),
        ArmNode(
            arm_id=effect.comparator_arm_key,
            cohort_id=effect.cohort_key,
            label=effect.comparator_arm_label,
            role=effect.comparator_arm_role,
        ),
    ]
    contrast = ContrastNode(
        contrast_id=effect.contrast_key,
        cohort_id=effect.cohort_key,
        treatment_arm_id=effect.treatment_arm_key,
        comparator_arm_id=effect.comparator_arm_key,
        label=effect.contrast_label,
        estimand=effect.contrast_estimand,
        positive_direction_means=effect.positive_direction_means,
    )
    native_span = NativeEvidenceSpan(
        source_locator=evidence.source_locator,
        quote=evidence.evidence_quote,
        char_start=evidence.quote_source_char_start,
        char_end=evidence.quote_source_char_end_exclusive,
        line_ids=[evidence.line_id],
    )
    native_effect = _native_effect_payload(effect)
    effect_evidence = native_effect.to_effect(
        paper_id=publication.paper_id,
        finding_id=effect.finding_key,
        outcome=outcome_text,
        contrast=effect.contrast_label,
        evidence=native_span,
    )
    estimate = OutcomeEstimateNode(
        estimate_id=estimate_id,
        contrast_id=effect.contrast_key,
        outcome_name=outcome_text,
        timepoint=effect.timepoint.to_native(),
        analysis_population=effect.analysis_population,
        effect=effect_evidence,
        evidence_span_ids=[span_id],
    )
    span = EvidenceSpan(
        span_id=span_id,
        publication_id=publication.publication_id,
        source_locator=evidence.source_locator,
        quote=evidence.evidence_quote,
        char_start=evidence.quote_source_char_start,
        char_end=evidence.quote_source_char_end_exclusive,
        line_ids=[evidence.line_id],
        roles=[EvidenceSpanRole.NUMERICAL_RESULT],
    )
    return compatibility, study, cohort, arms, contrast, estimate, span


def _merge_node(target: dict[str, Any], key: str, value: Any, kind: str) -> None:
    existing = target.get(key)
    if existing is not None and hash_canonical(existing) != hash_canonical(value):
        raise MetaSynGroundedPublicationBridgeV2Error(
            f"metasyn_grounded_bridge_v2_identity_collision:{kind}:{key}"
        )
    target[key] = value


def _publication_graph(
    *,
    source_row: MetaSynV5SourceSurfaceRowV1,
    terminals: Sequence[MetaSynGroundedCandidateTerminalV2],
) -> tuple[EvidenceGraph | None, list[MetaSynQuantitativeEffectCompatibilityV2]]:
    publication = source_row.source_row.source_record.publication
    studies: dict[str, StudyNode] = {}
    cohorts: dict[str, CohortNode] = {}
    arms: dict[str, ArmNode] = {}
    contrasts: dict[str, ContrastNode] = {}
    estimates: dict[str, OutcomeEstimateNode] = {}
    spans: dict[str, EvidenceSpan] = {}
    compatibility: list[MetaSynQuantitativeEffectCompatibilityV2] = []
    signatures: dict[str, str] = {}
    for terminal in terminals:
        if not terminal.authorizes_typed_effect:
            continue
        projected = _project_completed_effect(source_row=source_row, terminal=terminal)
        item, study, cohort, projected_arms, contrast, estimate, span = projected
        existing = signatures.get(item.quantitative_signature_sha256)
        if existing is not None:
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_duplicate_quantitative_signature:"
                f"{item.quantitative_signature_sha256}:{existing}:"
                f"{terminal.candidate_descriptor_sha256}"
            )
        signatures[item.quantitative_signature_sha256] = terminal.candidate_descriptor_sha256
        compatibility.append(item)
        _merge_node(studies, study.study_id, study, "study")
        _merge_node(cohorts, cohort.cohort_id, cohort, "cohort")
        for arm in projected_arms:
            _merge_node(arms, arm.arm_id, arm, "arm")
        _merge_node(contrasts, contrast.contrast_id, contrast, "contrast")
        _merge_node(estimates, estimate.estimate_id, estimate, "estimate")
        _merge_node(spans, span.span_id, span, "span")
    if not compatibility:
        return None, []
    graph = EvidenceGraph(
        graph_schema_version="1",
        publications=[publication],
        studies=[studies[key] for key in sorted(studies)],
        cohorts=[cohorts[key] for key in sorted(cohorts)],
        arms=[arms[key] for key in sorted(arms)],
        contrasts=[contrasts[key] for key in sorted(contrasts)],
        outcome_estimates=[estimates[key] for key in sorted(estimates)],
        evidence_spans=[spans[key] for key in sorted(spans)],
    )
    return graph, sorted(compatibility, key=lambda item: item.quantitative_signature_sha256)


def _canonical_terminal_input(value: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    expected = {"packet_input", "grounding_receipt", "assembly_receipt"}
    if set(value) != expected:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_terminal_input_fields_invalid"
        )
    try:
        packet = MetaSynPacketCandidateInputV2.model_validate(value["packet_input"])
        grounding = _GROUNDING_ADAPTER.validate_python(value["grounding_receipt"])
        assembly = _ASSEMBLY_ADAPTER.validate_python(value["assembly_receipt"])
    except ValueError as exc:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_terminal_input_invalid"
        ) from exc
    return packet, grounding, assembly


def _freeze_terminal(
    *,
    row: MetaSynExtractionRowInputV2,
    inventory: MetaSynCandidateInventoryReceiptV2,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    value: Mapping[str, Any],
) -> MetaSynGroundedCandidateTerminalV2:
    packet, grounding_raw, assembly_raw = _canonical_terminal_input(value)
    packet = validate_metasyn_packet_candidate_input_v2(
        packet_input=packet,
        extraction_inputs=execution_bundle.extraction_inputs,
        inventory_receipt=inventory,
    )
    if packet.row_ordinal != row.row_ordinal or packet.row_key != row.row_key:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_terminal_row_mismatch"
        )
    grounding = validate_passage_packet_grounding_receipt_v2(
        receipt=grounding_raw,
        model_outcome=grounding_raw.model_outcome.model_dump(mode="json"),
        candidate=packet.candidate,
        projection=row.projection_v2,
    )
    protocol = _protocol_for_row(row)
    protocol_orientation = freeze_packet_assembly_protocol_orientation_v2(
        question_surface=row.question_surface
    )
    assembly = validate_native_packet_assembly_v2(
        assembly=assembly_raw,
        candidate=packet.candidate,
        projection=row.projection_v2,
        protocol=protocol,
        protocol_orientation=protocol_orientation,
        analysis_policy=execution_bundle.assembly_analysis_policy,
        grounding_receipt=grounding,
    )
    blockers = sorted(
        [
            *(
                [f"grounding:{grounding.model_outcome.reason}"]
                if isinstance(grounding, PacketGroundingAbstentionReceiptV2)
                else []
            ),
            *(
                []
                if isinstance(assembly, NativePacketAssemblyCompletedV2)
                else [
                    *(f"assembly:{item}" for item in assembly.blocker_codes),
                    *(f"missing:{item}" for item in assembly.missing_field_paths),
                ]
            ),
        ]
    )
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "candidate_index": packet.candidate.candidate_index,
        "candidate_descriptor_sha256": packet.candidate_descriptor_sha256,
        "candidate_binding_sha256": packet.candidate_binding_sha256,
        "packet_input": packet,
        "packet_input_sha256": packet.packet_input_sha256,
        "grounding_receipt": grounding,
        "grounding_receipt_sha256": grounding.receipt_sha256,
        "assembly_receipt": assembly,
        "assembly_receipt_sha256": assembly.assembly_receipt_sha256,
        "terminal_status": assembly.status,
        "terminal_blockers": blockers,
        "authorizes_typed_effect": isinstance(assembly, NativePacketAssemblyCompletedV2),
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedCandidateTerminalV2.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def compute_metasyn_grounded_publication_bridge_v2_pipeline_fingerprint(
    *,
    repository_root: Path,
    execution_bundle_sha256: str,
    source_surface_sha256: str,
    inventory_receipt_membership_sha256: str,
    terminal_membership_sha256: str,
) -> PipelineFingerprint:
    if repository_root.is_symlink():
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_repository_root_symlink"
        )
    root = repository_root.resolve(strict=True)
    if not (root / BRIDGE_MODULE_PATH).is_file():
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_repository_root_invalid"
        )
    component = PipelineComponentSpec(
        component_id="metasyn-grounded-publication-corpus-bridge-v2",
        component_version=BRIDGE_COMPONENT_VERSION,
        file_paths=[BRIDGE_MODULE_PATH],
        settings={
            "all_32_publications_required": True,
            "execution_bundle_sha256": execution_bundle_sha256,
            "source_surface_sha256": source_surface_sha256,
            "inventory_receipt_membership_sha256": (inventory_receipt_membership_sha256),
            "terminal_membership_sha256": terminal_membership_sha256,
            "question_scoped_corpora_required": True,
            "legacy_v4_grounding_package_emitted": False,
            "legacy_optional_sentinels_are_scientific_observations": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


def _non_estimability(
    *,
    row: MetaSynExtractionRowInputV2,
    inventory: MetaSynCandidateInventoryReceiptV2,
    terminals: Sequence[MetaSynGroundedCandidateTerminalV2],
) -> tuple[NonEstimabilityReason, str, list[str]]:
    blockers = set(row.source_strength.source_strength_blockers)
    source_incomplete = (
        not row.source_strength.release_grade_source_grounding_eligible
        or not row.source_strength.projection_v2_selection_complete
    )
    if source_incomplete:
        blockers.add("source:projection_or_grounding_surface_incomplete")
        reason = NonEstimabilityReason.SOURCE_DOCUMENT_INCOMPLETE
    elif inventory.status == "no_candidate_non_authorizing":
        blockers.add("inventory:no_candidate_non_authorizing_no_absence_claim")
        reason = NonEstimabilityReason.OTHER
    elif inventory.status == "capacity_or_uncertainty_non_authorizing":
        blockers.add("inventory:capacity_or_uncertainty_non_authorizing")
        reason = NonEstimabilityReason.OTHER
    else:
        blockers.update(blocker for terminal in terminals for blocker in terminal.terminal_blockers)
        blockers.add("candidate:all_grounding_or_assembly_paths_non_estimable")
        reason = NonEstimabilityReason.UNGROUNDED_NUMERICAL_RESULT
    ordered = sorted(blockers)
    detail = "metasyn_grounded_bridge_v2_non_estimable:" + (
        ",".join(ordered) if ordered else "all_authorized_candidates_abstained"
    )
    return reason, detail, ordered


def _publication_join(
    *,
    bridge_pipeline_sha256: str,
    source_row: MetaSynV5SourceSurfaceRowV1,
    row: MetaSynExtractionRowInputV2,
    inventory: MetaSynCandidateInventoryReceiptV2,
    terminals: Sequence[MetaSynGroundedCandidateTerminalV2],
) -> MetaSynGroundedPublicationJoinV2:
    source_record = source_row.source_row.source_record
    graph, effects = _publication_graph(source_row=source_row, terminals=terminals)
    effect_hashes = sorted(item.compatibility_sha256 for item in effects)
    coverage_blockers = {
        "cross_publication_cohort_reconciliation_not_performed",
        "quantitative_compatibility_projection_only",
        "retrieval_completeness_not_established",
        "risk_of_bias_not_assessed",
        *row.source_strength.source_strength_blockers,
    }
    coverage_blockers.update(
        blocker for item in effects for blocker in item.coverage.coverage_blockers
    )
    coverage_blockers.update(blocker for item in terminals for blocker in item.terminal_blockers)
    receipt_payload = {
        "row_key": row.row_key,
        "inventory_receipt_sha256": inventory.receipt_sha256,
        "terminal_sha256s": sorted(item.terminal_sha256 for item in terminals),
        "compatibility_effect_sha256s": effect_hashes,
    }
    publication_grounding_sha256 = hash_canonical(receipt_payload)
    if graph is not None:
        fragment = freeze_publication_evidence_fragment(
            question_id=row.question_surface.question_id,
            publication_id=source_record.publication.publication_id,
            paper_id=source_record.publication.paper_id,
            publication=source_record.publication,
            pipeline_fingerprint_sha256=bridge_pipeline_sha256,
            source_document=source_record.source_document,
            grounding_receipt_sha256=publication_grounding_sha256,
            status=FragmentStatus.ESTIMABLE,
            graph=graph,
            extractor_warnings=sorted(coverage_blockers),
        )
    else:
        reason, detail, blockers = _non_estimability(
            row=row, inventory=inventory, terminals=terminals
        )
        coverage_blockers.update(blockers)
        fragment = freeze_publication_evidence_fragment(
            question_id=row.question_surface.question_id,
            publication_id=source_record.publication.publication_id,
            paper_id=source_record.publication.paper_id,
            publication=source_record.publication,
            pipeline_fingerprint_sha256=bridge_pipeline_sha256,
            source_document=source_record.source_document,
            grounding_receipt_sha256=publication_grounding_sha256,
            status=FragmentStatus.NON_ESTIMABLE,
            non_estimability_reason=reason,
            non_estimability_detail=detail,
            extractor_warnings=sorted(coverage_blockers),
        )
    completed = sum(item.authorizes_typed_effect for item in terminals)
    source_record_identity = {
        "publication": source_record.publication,
        "source_document": source_record.source_document,
    }
    payload = {
        "publication_join_version": PUBLICATION_JOIN_VERSION,
        "row_ordinal": row.row_ordinal,
        "row_key": row.row_key,
        "question_id": row.question_surface.question_id,
        "source_surface_row_sha256": source_row.source_surface_row_sha256,
        "extraction_row_input_sha256": row.row_input_sha256,
        "publication_source_identity_sha256": source_row.row_source_identity_sha256,
        "source_record_sha256": hash_canonical(source_record_identity),
        "publication": source_record.publication,
        "source_document": source_record.source_document,
        "source_artifact_binding_sha256": source_row.artifact_binding_sha256,
        "source_strength_surface_sha256": row.source_strength_surface_sha256,
        "source_strength_blockers": sorted(row.source_strength.source_strength_blockers),
        "protocol_outcome_text_by_id": row.question_surface.allowed_outcome_text_by_id,
        "protocol_outcome_membership_sha256": hash_canonical(
            row.question_surface.allowed_outcome_text_by_id
        ),
        "inventory_receipt": inventory,
        "inventory_receipt_sha256": inventory.receipt_sha256,
        "inventory_status": inventory.status,
        "candidate_descriptor_sha256s": sorted(
            item.descriptor_sha256 for item in inventory.inventory.candidates
        ),
        "candidate_terminal_sha256s": sorted(item.terminal_sha256 for item in terminals),
        "candidate_terminals": list(terminals),
        "inventoried_candidate_count": len(inventory.inventory.candidates),
        "authorized_candidate_count": (
            len(inventory.inventory.candidates)
            if inventory.status == "candidates_authorized"
            else 0
        ),
        "terminal_candidate_count": len(terminals),
        "completed_candidate_count": completed,
        "abstained_candidate_count": len(terminals) - completed,
        "compatibility_effects": effects,
        "compatibility_effect_membership_sha256": hash_canonical(effect_hashes),
        "coverage_blockers": sorted(coverage_blockers),
        "compatibility_fragment": fragment,
        "compatibility_fragment_sha256": fragment.fragment_sha256,
        "exact_terminal_roster_complete": True,
        "exact_protocol_outcome_mapping_authority": True,
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": graph is not None,
        "scientific_optional_field_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedPublicationJoinV2.model_validate(
        {**payload, "publication_join_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_grounded_publication_bridge_v2(
    *,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2 | Mapping[str, Any],
    inventory_receipts_by_row: Mapping[str, MetaSynCandidateInventoryReceiptV2 | Mapping[str, Any]],
    candidate_terminals_by_row: Mapping[str, Sequence[Mapping[str, Any]]],
    repository_root: Path,
) -> MetaSynGroundedPublicationCorpusBridgeV2:
    """Freeze the complete all-32 bridge from explicit runtime result mappings.

    Each terminal mapping has exactly three keys: ``packet_input``,
    ``grounding_receipt``, and ``assembly_receipt``.  Every authorized inventory
    candidate must appear exactly once and non-authorizing rows must have no terminal.
    """

    root = repository_root.resolve(strict=True)
    try:
        preliminary_bundle = MetaSynPassageHostedExecutionBundleV2.model_validate(
            execution_bundle.model_dump(mode="json")
            if isinstance(execution_bundle, MetaSynPassageHostedExecutionBundleV2)
            else execution_bundle
        )
    except ValueError as exc:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_execution_bundle_invalid"
        ) from exc
    preliminary_rows = preliminary_bundle.extraction_inputs.rows
    preliminary_keys = [row.row_key for row in preliminary_rows]
    if set(inventory_receipts_by_row) != set(preliminary_keys):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_inventory_row_roster_incomplete"
        )
    if set(candidate_terminals_by_row) != set(preliminary_keys):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_terminal_row_roster_incomplete"
        )
    for row in preliminary_rows:
        preliminary_inventory = validate_metasyn_candidate_inventory_receipt_v2(
            inventory_receipts_by_row[row.row_key],
            row_context_sha256=row.upstream_row_context_sha256,
            projection_v2_sha256=row.projection_v2_sha256,
            allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
            passage_text_by_id={
                item.passage_id: item.text for item in row.projection_surface.passages
            },
        )
        raw_terminals = candidate_terminals_by_row[row.row_key]
        if isinstance(raw_terminals, (str, bytes)):
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_terminal_sequence_invalid"
            )
        for terminal in raw_terminals:
            if not isinstance(terminal, Mapping) or set(terminal) != {
                "packet_input",
                "grounding_receipt",
                "assembly_receipt",
            }:
                raise MetaSynGroundedPublicationBridgeV2Error(
                    "metasyn_grounded_bridge_v2_terminal_input_fields_invalid"
                )
        expected_terminal_count = (
            len(preliminary_inventory.inventory.candidates)
            if preliminary_inventory.status == "candidates_authorized"
            else 0
        )
        if len(raw_terminals) != expected_terminal_count:
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_terminal_candidate_roster_mismatch"
            )
    bundle = validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=execution_bundle,
        repository_root=root,
        external_replay=True,
    )
    source_surface = freeze_metasyn_v5_source_surface(repository_root=root)
    source_surface = validate_metasyn_v5_source_surface(
        source_surface=source_surface,
        repository_root=root,
        external_replay=False,
    )
    rows = bundle.extraction_inputs.rows
    source_rows = source_surface.rows
    inventories: list[MetaSynCandidateInventoryReceiptV2] = []
    frozen_terminals: list[list[MetaSynGroundedCandidateTerminalV2]] = []
    for row, source_row in zip(rows, source_rows, strict=True):
        if (
            row.row_ordinal != source_row.row_ordinal
            or row.row_key != source_row.row_key
            or row.upstream_source_surface_row_sha256 != source_row.source_surface_row_sha256
        ):
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_source_extraction_row_join_mismatch"
            )
        inventory = validate_metasyn_candidate_inventory_receipt_v2(
            inventory_receipts_by_row[row.row_key],
            row_context_sha256=row.upstream_row_context_sha256,
            projection_v2_sha256=row.projection_v2_sha256,
            allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
            passage_text_by_id={
                item.passage_id: item.text for item in row.projection_surface.passages
            },
        )
        raw_terminals = candidate_terminals_by_row[row.row_key]
        if isinstance(raw_terminals, (str, bytes)):
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_terminal_sequence_invalid"
            )
        terminals = [
            _freeze_terminal(
                row=row,
                inventory=inventory,
                execution_bundle=bundle,
                value=value,
            )
            for value in raw_terminals
        ]
        terminals.sort(key=lambda item: item.candidate_index)
        descriptors = [item.candidate_descriptor_sha256 for item in terminals]
        expected_descriptors = (
            [item.descriptor_sha256 for item in inventory.inventory.candidates]
            if inventory.status == "candidates_authorized"
            else []
        )
        if descriptors != expected_descriptors:
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_terminal_candidate_roster_mismatch"
            )
        if inventory.status != "candidates_authorized" and terminals:
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_non_authorizing_inventory_has_terminal"
            )
        inventories.append(inventory)
        frozen_terminals.append(terminals)

    inventory_membership = hash_canonical([item.receipt_sha256 for item in inventories])
    terminal_membership = hash_canonical(
        [
            {
                "row_key": row.row_key,
                "terminal_sha256s": [item.terminal_sha256 for item in terminals],
            }
            for row, terminals in zip(rows, frozen_terminals, strict=True)
        ]
    )
    fingerprint = compute_metasyn_grounded_publication_bridge_v2_pipeline_fingerprint(
        repository_root=root,
        execution_bundle_sha256=bundle.execution_bundle_sha256,
        source_surface_sha256=source_surface.source_surface_sha256,
        inventory_receipt_membership_sha256=inventory_membership,
        terminal_membership_sha256=terminal_membership,
    )
    joins = [
        _publication_join(
            bridge_pipeline_sha256=fingerprint.pipeline_sha256,
            source_row=source_row,
            row=row,
            inventory=inventory,
            terminals=terminals,
        )
        for row, source_row, inventory, terminals in zip(
            rows, source_rows, inventories, frozen_terminals, strict=True
        )
    ]
    publication_ids = [item.publication.publication_id for item in joins]
    paper_ids = [item.publication.paper_id for item in joins]
    if len(publication_ids) != len(set(publication_ids)):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_publication_id_collision"
        )
    if len(paper_ids) != len(set(paper_ids)):
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_paper_id_collision"
        )

    joins_by_question: dict[str, list[MetaSynGroundedPublicationJoinV2]] = defaultdict(list)
    for item in joins:
        joins_by_question[item.question_id].append(item)
    if len(joins_by_question) != EXPECTED_QUESTION_COUNT:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_question_count_mismatch"
        )
    question_corpora: list[MetaSynGroundedQuestionCorpusV2] = []
    for question_id in sorted(joins_by_question):
        question_joins = joins_by_question[question_id]
        corpus = assemble_typed_evidence_corpus(
            [item.compatibility_fragment for item in question_joins]
        )
        blockers = sorted(
            {blocker for item in question_joins for blocker in item.coverage_blockers}
        )
        question_payload = {
            "question_corpus_version": QUESTION_CORPUS_VERSION,
            "question_id": question_id,
            "publication_join_sha256s": sorted(
                item.publication_join_sha256 for item in question_joins
            ),
            "publication_ids": sorted(item.publication.publication_id for item in question_joins),
            "compatibility_corpus": corpus,
            "compatibility_corpus_sha256": corpus.corpus_sha256,
            "estimable_publication_count": len(corpus.estimable_publication_ids),
            "non_estimable_publication_count": len(corpus.non_estimable_publication_ids),
            "quantitative_effect_count": len(corpus.graph.outcome_estimates),
            "coverage_blockers": blockers,
            "exact_projection_authority": True,
            "graph_construction_authority": True,
            "quantitative_kernel_compatibility": bool(corpus.graph.outcome_estimates),
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        question_corpora.append(
            MetaSynGroundedQuestionCorpusV2.model_validate(
                {
                    **question_payload,
                    "question_corpus_sha256": hash_canonical(question_payload),
                }
            )
        )
    inventoried_count = sum(item.inventoried_candidate_count for item in joins)
    authorized_count = sum(item.authorized_candidate_count for item in joins)
    terminal_count = sum(item.terminal_candidate_count for item in joins)
    completed_count = sum(item.completed_candidate_count for item in joins)
    abstained_count = sum(item.abstained_candidate_count for item in joins)
    effect_count = sum(len(item.compatibility_effects) for item in joins)
    bridge_payload = {
        "bridge_version": BRIDGE_VERSION,
        "status": "externally_replayable_complete_terminal_roster",
        "execution_bundle": bundle,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "source_surface": source_surface,
        "source_surface_sha256": source_surface.source_surface_sha256,
        "inventory_receipt_membership_sha256": inventory_membership,
        "terminal_membership_sha256": terminal_membership,
        "bridge_pipeline_fingerprint": fingerprint,
        "bridge_pipeline_sha256": fingerprint.pipeline_sha256,
        "publication_joins": joins,
        "publication_join_membership_sha256": hash_canonical(
            [item.publication_join_sha256 for item in joins]
        ),
        "question_corpora": question_corpora,
        "question_corpus_membership_sha256": hash_canonical(
            [item.question_corpus_sha256 for item in question_corpora]
        ),
        "question_count": EXPECTED_QUESTION_COUNT,
        "publication_count": EXPECTED_PUBLICATION_COUNT,
        "inventoried_candidate_count": inventoried_count,
        "authorized_candidate_count": authorized_count,
        "terminal_candidate_count": terminal_count,
        "completed_candidate_count": completed_count,
        "abstained_candidate_count": abstained_count,
        "estimable_publication_count": sum(
            item.quantitative_kernel_compatibility for item in joins
        ),
        "quantitative_effect_count": effect_count,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "v5_hosted_outputs_consumed": False,
        "v2_terminal_outputs_consumed": bool(terminal_count),
        "legacy_v4_grounding_package_emitted": False,
        "exact_projection_authority": True,
        "graph_construction_authority": True,
        "quantitative_kernel_compatibility": bool(effect_count),
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynGroundedPublicationCorpusBridgeV2.model_validate(
        {**bridge_payload, "bridge_sha256": hash_canonical(bridge_payload)}
    )


def validate_metasyn_grounded_publication_bridge_v2(
    *,
    bridge: MetaSynGroundedPublicationCorpusBridgeV2 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> MetaSynGroundedPublicationCorpusBridgeV2:
    """Validate a saved bridge and optionally rebuild every upstream join."""

    try:
        canonical = MetaSynGroundedPublicationCorpusBridgeV2.model_validate(
            bridge.model_dump(mode="json")
            if isinstance(bridge, MetaSynGroundedPublicationCorpusBridgeV2)
            else bridge
        )
    except ValueError as exc:
        raise MetaSynGroundedPublicationBridgeV2Error(
            "metasyn_grounded_bridge_v2_contract_invalid"
        ) from exc
    if external_replay:
        inventories = {item.row_key: item.inventory_receipt for item in canonical.publication_joins}
        terminals = {
            item.row_key: [
                {
                    "packet_input": terminal.packet_input,
                    "grounding_receipt": terminal.grounding_receipt,
                    "assembly_receipt": terminal.assembly_receipt,
                }
                for terminal in item.candidate_terminals
            ]
            for item in canonical.publication_joins
        }
        replayed = freeze_metasyn_grounded_publication_bridge_v2(
            execution_bundle=canonical.execution_bundle,
            inventory_receipts_by_row=inventories,
            candidate_terminals_by_row=terminals,
            repository_root=repository_root,
        )
        if replayed != canonical:
            raise MetaSynGroundedPublicationBridgeV2Error(
                "metasyn_grounded_bridge_v2_external_replay_mismatch"
            )
    return canonical


__all__ = [
    "BRIDGE_VERSION",
    "MetaSynGroundedCandidateTerminalV2",
    "MetaSynGroundedPublicationBridgeV2Error",
    "MetaSynGroundedPublicationCorpusBridgeV2",
    "MetaSynGroundedPublicationJoinV2",
    "MetaSynGroundedQuestionCorpusV2",
    "MetaSynOptionalScientificCoverageV2",
    "MetaSynQuantitativeEffectCompatibilityV2",
    "compute_metasyn_grounded_publication_bridge_v2_pipeline_fingerprint",
    "freeze_metasyn_grounded_publication_bridge_v2",
    "validate_metasyn_grounded_publication_bridge_v2",
]
