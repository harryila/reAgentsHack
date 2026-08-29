"""Compact, offset-free model contract with trusted packet grounding.

Version 1 asked the model to calculate character offsets.  This independent v2
component deliberately does not expose offset fields to the model.  A model may
only bind itself to one frozen candidate, copy one exact evidence quote, name a
closed scientific field path, and copy the verbatim numeric token.  Trusted local
code then requires unique exact occurrences and derives every character and UTF-8
byte offset without repair, fuzzy matching, or whitespace normalization.

It accepts either the frozen v1 line descriptor or the additive MetaSyn p2 passage
descriptor; both paths end in the same compact model schema and trusted receipts.
This module is a grounding component, not a claim-release or extraction-accuracy
authority.  It does not modify the frozen v1 packet/runtime contracts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import (
    ConfigDict,
    Field,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from literature_multiverse.effects import EffectFormat
from literature_multiverse.evidence_graph import TimeUnit
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynPassageCandidateV2,
)
from literature_multiverse.metasyn_projection_v2 import (
    AnchoredPassageV2,
    FrozenMetaSynProjectionV2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    INVENTORY_SENTINEL_CAP,
    MAX_CANDIDATE_LINE_IDS,
    MAX_DECIMAL_LEXEME_CHARACTERS,
    MAX_EVIDENCE_QUOTE_CHARACTERS,
    MAX_NUMERIC_SUPPORT_ITEMS,
    EffectKind,
    NativeCandidateDescriptor,
    NumericFieldPath,
)
from literature_multiverse.native_question_projection import (
    FrozenProjectedPassageV1,
    FrozenSourceProjectionV1,
)

PACKET_GROUNDING_V2_VERSION = "native-packet-grounding-v2"
CANDIDATE_BINDING_V2_VERSION = "native-packet-candidate-binding-v2"
PASSAGE_CANDIDATE_BINDING_V2_VERSION = (
    "metasyn-passage-packet-candidate-binding-v2"
)
MODEL_OUTCOME_V2_VERSION = "native-packet-grounding-model-outcome-v2"
SCHEMA_BUNDLE_V2_VERSION = "native-packet-grounding-schema-bundle-v2"
EVIDENCE_RECEIPT_V2_VERSION = "native-packet-evidence-receipt-v2"
EFFECT_FORMAT_RECEIPT_V2_VERSION = "native-packet-effect-format-receipt-v2"
NORMALIZATION_RECEIPT_V2_VERSION = "native-packet-normalization-receipt-v2"
NUMERIC_RECEIPT_V2_VERSION = "native-packet-numeric-grounding-receipt-v2"
IDENTITY_RECEIPT_V2_VERSION = "native-packet-identity-grounding-receipt-v2"
COMPLETED_RECEIPT_V2_VERSION = "native-packet-grounding-completed-receipt-v2"
ABSTENTION_RECEIPT_V2_VERSION = "native-packet-grounding-abstention-receipt-v2"

MAX_IDENTITY_CLAIMS = 32
MAX_IDENTITY_TEXT_CHARACTERS = 512
MAX_OCCURRENCES = 100_000
MAX_DECIMAL_MAGNITUDE = Decimal("1e12")

_DECIMAL_RE = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,12})?|\.[0-9]{1,12})"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]{0,2}))?$"
)
_UNSIGNED_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]{0,9})$")
_NUMERIC_SIGN_CHARACTERS = frozenset(
    "+-\u2212\u2010\u2011\u2012\u2013\u2014\ufe63\uff0d\u207a\u207b\u208a\u208b"
)
_INEQUALITY_CHARACTERS = frozenset("<>\u2264\u2265\u2266\u2267")
_NUMERIC_GROUPING_WHITESPACE = " \t\u00a0\u2007\u2009\u202f"

_PERCENT_NORMALIZABLE_PATHS = frozenset(
    {"effect.ci_level", "effect.reported_p_value"}
)
_COMMON_NUMERIC_PATHS = frozenset(
    {
        "cohort.total_sample_size",
        "treatment_arm.sample_size",
        "comparator_arm.sample_size",
        "finding.timepoint.value",
        "finding.timepoint.lower",
        "finding.timepoint.upper",
        "effect.reported_p_value",
        "effect.equivalence_margin",
    }
)
_EFFECT_NUMERIC_PATHS: dict[EffectKind, frozenset[str]] = {
    "direct_standard_error": frozenset(
        {"effect.estimate", "effect.standard_error"}
    ),
    "direct_variance": frozenset({"effect.estimate", "effect.variance"}),
    "direct_confidence_interval": frozenset(
        {
            "effect.estimate",
            "effect.ci_lower",
            "effect.ci_upper",
            "effect.ci_level",
        }
    ),
    "continuous_group_statistics": frozenset(
        {
            "effect.treatment_mean",
            "effect.treatment_sd",
            "effect.treatment_n",
            "effect.control_mean",
            "effect.control_sd",
            "effect.control_n",
        }
    ),
    "binary_group_statistics": frozenset(
        {
            "effect.treatment_events",
            "effect.treatment_total",
            "effect.control_events",
            "effect.control_total",
        }
    ),
}
_INTEGER_PATHS = frozenset(
    {
        "cohort.total_sample_size",
        "treatment_arm.sample_size",
        "comparator_arm.sample_size",
        "effect.treatment_n",
        "effect.control_n",
        "effect.treatment_events",
        "effect.treatment_total",
        "effect.control_events",
        "effect.control_total",
    }
)
_STRICTLY_POSITIVE_PATHS = frozenset(
    {
        "cohort.total_sample_size",
        "treatment_arm.sample_size",
        "comparator_arm.sample_size",
        "effect.standard_error",
        "effect.variance",
        "effect.treatment_sd",
        "effect.control_sd",
        "effect.treatment_n",
        "effect.control_n",
        "effect.treatment_total",
        "effect.control_total",
        "effect.equivalence_margin",
    }
)
_UNIT_INTERVAL_PATHS = frozenset(
    {"effect.ci_level", "effect.reported_p_value"}
)

IDENTITY_FIELD_PATHS = (
    "study.source_label",
    "study.design",
    "study.registration_id",
    "cohort.source_label",
    "cohort.registry_id",
    "cohort.dataset_id",
    "cohort.population_description",
    "cohort.recruitment_period",
    "treatment_arm.label",
    "treatment_arm.description",
    "comparator_arm.label",
    "comparator_arm.description",
    "contrast.label",
    "contrast.estimand",
    "finding.analysis_population",
)
type IdentityFieldPath = Literal[
    "study.source_label",
    "study.design",
    "study.registration_id",
    "cohort.source_label",
    "cohort.registry_id",
    "cohort.dataset_id",
    "cohort.population_description",
    "cohort.recruitment_period",
    "treatment_arm.label",
    "treatment_arm.description",
    "comparator_arm.label",
    "comparator_arm.description",
    "contrast.label",
    "contrast.estimand",
    "finding.analysis_population",
]
type GroundedIdentityFieldPath = IdentityFieldPath | Literal[
    "effect.unit",
    "finding.timepoint.anchor",
    "finding.timepoint.raw_label",
]

NORMALIZATION_POLICY_V2 = {
    "policy_version": "native-packet-normalization-policy-v2",
    "identity": "verbatim decimal lexeme is preserved byte-for-byte",
    "percent_to_proportion": "verbatim percent-bearing decimal divided exactly by 100",
    "percent_normalizable_field_paths": sorted(_PERCENT_NORMALIZABLE_PATHS),
    "model_authored_offsets_permitted": False,
    "fuzzy_matching_permitted": False,
}
NORMALIZATION_POLICY_V2_SHA256 = hash_canonical(NORMALIZATION_POLICY_V2)

EFFECT_FORMAT_ALIAS_POLICY_V2 = {
    "policy_version": "native-packet-effect-format-alias-policy-v2",
    "normalization": "unicode_casefold_then_whitespace_collapse",
    "aliases": {
        "cohen d": "cohens_d",
        "cohen's d": "cohens_d",
        "cohens d": "cohens_d",
        "cohens_d": "cohens_d",
        "hedges g": "hedges_g",
        "hedges' g": "hedges_g",
        "hedges_g": "hedges_g",
        "log odds ratio": "log_odds_ratio",
        "log risk ratio": "log_risk_ratio",
        "log-odds ratio": "log_odds_ratio",
        "log-risk ratio": "log_risk_ratio",
        "log_odds_ratio": "log_odds_ratio",
        "log_risk_ratio": "log_risk_ratio",
        "mean difference": "mean_difference",
        "mean_difference": "mean_difference",
        "odds ratio": "odds_ratio",
        "odds_ratio": "odds_ratio",
        "relative risk": "risk_ratio",
        "risk ratio": "risk_ratio",
        "risk_ratio": "risk_ratio",
    },
    "unlisted_aliases_permitted": False,
    "model_authored_effect_format_permitted": False,
}
EFFECT_FORMAT_ALIAS_POLICY_V2_SHA256 = hash_canonical(
    EFFECT_FORMAT_ALIAS_POLICY_V2
)


class NativePacketGroundingV2Error(ValueError):
    """A compact model response or trusted grounding replay is unsafe."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[StrictStr, Field(pattern=SHA256_RE.pattern)]
NonEmptyText = Annotated[
    StrictStr, Field(min_length=1, max_length=MAX_IDENTITY_TEXT_CHARACTERS)
]
SourceLocatorText = Annotated[StrictStr, Field(min_length=1, max_length=2_048)]
EffectUnitText = Annotated[StrictStr, Field(min_length=1, max_length=64)]


def _validate_self_hash(model: _FrozenExactModel, field_name: str) -> None:
    payload = model.model_dump(mode="json", exclude={field_name})
    if getattr(model, field_name) != hash_canonical(payload):
        raise ValueError(f"packet_grounding_v2_self_hash_mismatch:{field_name}")


def _require_unpadded(value: str, *, code: str) -> str:
    if not value or value != value.strip():
        raise ValueError(code)
    return value


def _allowed_numeric_paths(effect_kind: EffectKind) -> tuple[str, ...]:
    return tuple(sorted(_COMMON_NUMERIC_PATHS | _EFFECT_NUMERIC_PATHS[effect_kind]))


def _effect_format_from_exact_token(value: str) -> tuple[str, EffectFormat]:
    _require_unpadded(
        value,
        code="packet_grounding_v2_effect_format_token_not_exact",
    )
    normalized = " ".join(value.casefold().split())
    raw = EFFECT_FORMAT_ALIAS_POLICY_V2["aliases"].get(normalized)
    if not isinstance(raw, str):
        raise ValueError("packet_grounding_v2_effect_format_alias_unsupported")
    return normalized, EffectFormat(raw)


def _parse_decimal(value: str, *, code: str) -> Decimal:
    if not _DECIMAL_RE.fullmatch(value):
        raise ValueError(f"{code}_lexeme_invalid")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - excluded by the regex
        raise ValueError(f"{code}_lexeme_invalid") from exc
    if not parsed.is_finite() or abs(parsed) > MAX_DECIMAL_MAGNITUDE:
        raise ValueError(f"{code}_magnitude_invalid")
    return parsed


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():  # pragma: no cover - guarded by parsing
        raise ValueError("packet_grounding_v2_nonfinite_decimal")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _normalized_numeric_lexeme(
    *,
    field_path: str,
    verbatim_numeric_token: str,
    normalization: Literal["identity", "percent_to_proportion"],
) -> str:
    if normalization == "percent_to_proportion":
        if field_path not in _PERCENT_NORMALIZABLE_PATHS:
            raise ValueError(
                f"packet_grounding_v2_percent_normalization_forbidden:{field_path}"
            )
        if not verbatim_numeric_token.endswith("%"):
            raise ValueError(
                f"packet_grounding_v2_percent_marker_missing:{field_path}"
            )
        parsed = _parse_decimal(
            verbatim_numeric_token[:-1], code="packet_grounding_v2_numeric_token"
        ) / Decimal("100")
        rendered = _canonical_decimal(parsed)
    else:
        if verbatim_numeric_token.endswith("%"):
            raise ValueError(
                f"packet_grounding_v2_unlicensed_percent_token:{field_path}"
            )
        parsed = _parse_decimal(
            verbatim_numeric_token, code="packet_grounding_v2_numeric_token"
        )
        rendered = verbatim_numeric_token

    if field_path in _INTEGER_PATHS:
        raw = rendered
        if not _UNSIGNED_INTEGER_RE.fullmatch(raw):
            raise ValueError(f"packet_grounding_v2_integer_token_invalid:{field_path}")
    if field_path in _STRICTLY_POSITIVE_PATHS and parsed <= 0:
        raise ValueError(f"packet_grounding_v2_value_not_positive:{field_path}")
    if field_path in {"effect.treatment_events", "effect.control_events"} and parsed < 0:
        raise ValueError(f"packet_grounding_v2_event_count_negative:{field_path}")
    if field_path.startswith("finding.timepoint.") and parsed < 0:
        raise ValueError(f"packet_grounding_v2_timepoint_negative:{field_path}")
    if field_path in _UNIT_INTERVAL_PATHS and not Decimal("0") <= parsed <= Decimal(
        "1"
    ):
        raise ValueError(f"packet_grounding_v2_unit_interval_invalid:{field_path}")
    if field_path == "effect.ci_level" and parsed in {Decimal("0"), Decimal("1")}:
        raise ValueError("packet_grounding_v2_ci_level_open_interval_required")
    return rendered


def _assert_numeric_token_boundary(
    *, quote: str, start: int, end: int, normalization: str, field_path: str
) -> None:
    prefix = quote[:start]
    suffix = quote[end:]
    stripped_prefix = prefix.rstrip(_NUMERIC_GROUPING_WHITESPACE)
    stripped_suffix = suffix.lstrip(_NUMERIC_GROUPING_WHITESPACE)
    immediate_prefix_invalid = bool(prefix) and (
        prefix[-1].isdigit()
        or (
            prefix[-1] in ".eE," and len(prefix) > 1 and prefix[-2].isdigit()
        )
    )
    expression_prefix_invalid = bool(stripped_prefix) and (
        stripped_prefix[-1] in _NUMERIC_SIGN_CHARACTERS
        or stripped_prefix[-1] in _INEQUALITY_CHARACTERS
        or (len(stripped_prefix) < len(prefix) and stripped_prefix[-1].isdigit())
    )
    immediate_suffix_invalid = bool(suffix) and (
        suffix[0].isdigit()
        or (suffix[0] in ".," and len(suffix) > 1 and suffix[1].isdigit())
        or (
            suffix[0] in "eE"
            and len(suffix) > 1
            and (
                suffix[1].isdigit()
                or (
                    suffix[1] in _NUMERIC_SIGN_CHARACTERS
                    and len(suffix) > 2
                    and suffix[2].isdigit()
                )
            )
        )
    )
    if immediate_prefix_invalid or expression_prefix_invalid or immediate_suffix_invalid:
        raise NativePacketGroundingV2Error(
            f"packet_grounding_v2_numeric_token_boundary:{field_path}"
        )
    if normalization == "identity" and stripped_suffix.startswith("%"):
        raise NativePacketGroundingV2Error(
            f"packet_grounding_v2_unlicensed_percent_token:{field_path}"
        )


def _occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    cursor = 0
    while True:
        found = haystack.find(needle, cursor)
        if found < 0:
            break
        starts.append(found)
        if len(starts) > MAX_OCCURRENCES:
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_occurrence_cap_exceeded"
            )
        cursor = found + 1
    return starts


def _numeric_token_occurrences(
    *,
    quote: str,
    token: str,
    normalization: Literal["identity", "percent_to_proportion"],
    field_path: str,
) -> list[int]:
    """Return exact occurrences that are also complete numeric lexemes.

    A short token such as ``8`` may occur inside another exact token such as
    ``-0.80``.  Those substring hits are not candidate occurrences and must not
    make an otherwise unique standalone token ambiguous.
    """

    valid: list[int] = []
    for start in _occurrences(quote, token):
        end = start + len(token)
        try:
            _assert_numeric_token_boundary(
                quote=quote,
                start=start,
                end=end,
                normalization=normalization,
                field_path=field_path,
            )
        except NativePacketGroundingV2Error:
            continue
        valid.append(start)
    return valid


class PacketCandidateBindingV2(_FrozenExactModel):
    binding_version: Literal["native-packet-candidate-binding-v2"] = (
        CANDIDATE_BINDING_V2_VERSION
    )
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    candidate_descriptor_sha256: Sha256
    outcome_name: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    effect_kind: EffectKind
    line_ids: Annotated[
        list[Annotated[StrictStr, Field(min_length=1, max_length=32)]],
        Field(min_length=1, max_length=MAX_CANDIDATE_LINE_IDS),
    ]
    projection_sha256: Sha256
    binding_sha256: Sha256

    @field_validator("line_ids")
    @classmethod
    def validate_line_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("packet_grounding_v2_binding_line_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> PacketCandidateBindingV2:
        candidate = NativeCandidateDescriptor(
            candidate_index=self.candidate_index,
            outcome_name=self.outcome_name,
            effect_kind=self.effect_kind,
            line_ids=self.line_ids,
        )
        if candidate.descriptor_sha256 != self.candidate_descriptor_sha256:
            raise ValueError("packet_grounding_v2_candidate_descriptor_hash_mismatch")
        _validate_self_hash(self, "binding_sha256")
        return self


class PacketPassageCandidateBindingV2(_FrozenExactModel):
    """Bind a packet to one additive passage-anchored candidate descriptor."""

    binding_version: Literal[
        "metasyn-passage-packet-candidate-binding-v2"
    ] = PASSAGE_CANDIDATE_BINDING_V2_VERSION
    candidate_index: Annotated[int, Field(ge=1, le=INVENTORY_SENTINEL_CAP)]
    candidate_descriptor_sha256: Sha256
    canonical_outcome_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    outcome_concept_quote: Annotated[
        StrictStr, Field(min_length=1, max_length=256)
    ]
    effect_kind: EffectKind
    passage_ids: Annotated[
        list[Annotated[StrictStr, Field(pattern=r"^p2-[0-9a-f]{64}$")]],
        Field(min_length=1, max_length=4),
    ]
    projection_sha256: Sha256
    binding_sha256: Sha256

    @field_validator("passage_ids")
    @classmethod
    def validate_passage_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                "packet_grounding_v2_binding_passage_ids_not_sorted_unique"
            )
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> PacketPassageCandidateBindingV2:
        candidate = MetaSynPassageCandidateV2(
            candidate_index=self.candidate_index,
            canonical_outcome_id=self.canonical_outcome_id,
            outcome_concept_quote=self.outcome_concept_quote,
            effect_kind=self.effect_kind,
            passage_ids=self.passage_ids,
        )
        if candidate.descriptor_sha256 != self.candidate_descriptor_sha256:
            raise ValueError("packet_grounding_v2_candidate_descriptor_hash_mismatch")
        _validate_self_hash(self, "binding_sha256")
        return self


type PacketCandidateBindingLikeV2 = (
    PacketCandidateBindingV2 | PacketPassageCandidateBindingV2
)
_CANDIDATE_BINDING_ADAPTER = TypeAdapter(PacketCandidateBindingLikeV2)


class PacketNumericClaimV2(_FrozenExactModel):
    field_path: NumericFieldPath
    verbatim_numeric_token: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_DECIMAL_LEXEME_CHARACTERS + 1)
    ]
    normalization: Literal["identity", "percent_to_proportion"]

    @model_validator(mode="after")
    def validate_claim(self) -> PacketNumericClaimV2:
        _normalized_numeric_lexeme(
            field_path=self.field_path,
            verbatim_numeric_token=self.verbatim_numeric_token,
            normalization=self.normalization,
        )
        return self


class PacketIdentityClaimV2(_FrozenExactModel):
    field_path: IdentityFieldPath
    verbatim_identity_text: NonEmptyText

    @field_validator("verbatim_identity_text")
    @classmethod
    def validate_identity_text(cls, value: str) -> str:
        return _require_unpadded(
            value, code="packet_grounding_v2_identity_text_not_exact"
        )


class PacketTimepointNotReportedV2(_FrozenExactModel):
    kind: Literal["not_reported"] = "not_reported"


class PacketTimepointExactV2(_FrozenExactModel):
    kind: Literal["exact"] = "exact"
    unit: TimeUnit
    anchor: NonEmptyText | None
    raw_label: NonEmptyText | None

    @field_validator("anchor", "raw_label")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _require_unpadded(
                value, code="packet_grounding_v2_timepoint_identity_not_exact"
            )
        )


class PacketTimepointRangeV2(_FrozenExactModel):
    kind: Literal["range"] = "range"
    unit: TimeUnit
    anchor: NonEmptyText | None
    raw_label: NonEmptyText | None

    @field_validator("anchor", "raw_label")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _require_unpadded(
                value, code="packet_grounding_v2_timepoint_identity_not_exact"
            )
        )


class PacketTimepointReportedTextV2(_FrozenExactModel):
    kind: Literal["reported_text"] = "reported_text"
    raw_label: NonEmptyText
    anchor: NonEmptyText | None

    @field_validator("anchor", "raw_label")
    @classmethod
    def validate_identity(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _require_unpadded(
                value, code="packet_grounding_v2_timepoint_identity_not_exact"
            )
        )


type PacketTimepointV2 = Annotated[
    PacketTimepointNotReportedV2
    | PacketTimepointExactV2
    | PacketTimepointRangeV2
    | PacketTimepointReportedTextV2,
    Field(discriminator="kind"),
]


class PacketGroundingModelCompletedV2(_FrozenExactModel):
    outcome_version: Literal["native-packet-grounding-model-outcome-v2"] = (
        MODEL_OUTCOME_V2_VERSION
    )
    packet_status: Literal["completed"] = "completed"
    candidate_binding_sha256: Sha256
    evidence_quote: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_EVIDENCE_QUOTE_CHARACTERS)
    ]
    effect_format_token: Annotated[
        StrictStr, Field(min_length=1, max_length=64)
    ] | None
    effect_unit: EffectUnitText | None
    numeric_claims: Annotated[
        list[PacketNumericClaimV2],
        Field(min_length=1, max_length=MAX_NUMERIC_SUPPORT_ITEMS),
    ]
    identity_claims: Annotated[
        list[PacketIdentityClaimV2], Field(max_length=MAX_IDENTITY_CLAIMS)
    ]
    timepoint: PacketTimepointV2

    @field_validator("evidence_quote")
    @classmethod
    def validate_evidence_quote(cls, value: str) -> str:
        return _require_unpadded(
            value,
            code="packet_grounding_v2_evidence_quote_not_exact",
        )

    @field_validator("effect_format_token")
    @classmethod
    def validate_effect_format_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _effect_format_from_exact_token(value)
        return value

    @field_validator("effect_unit")
    @classmethod
    def validate_effect_unit(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _require_unpadded(
                value,
                code="packet_grounding_v2_effect_unit_not_exact",
            )
        )

    @field_validator("numeric_claims")
    @classmethod
    def validate_numeric_claims(
        cls, value: list[PacketNumericClaimV2]
    ) -> list[PacketNumericClaimV2]:
        paths = [item.field_path for item in value]
        if paths != sorted(set(paths)):
            raise ValueError("packet_grounding_v2_numeric_paths_not_sorted_unique")
        return value

    @field_validator("identity_claims")
    @classmethod
    def validate_identity_claims(
        cls, value: list[PacketIdentityClaimV2]
    ) -> list[PacketIdentityClaimV2]:
        keys = [(item.field_path, item.verbatim_identity_text) for item in value]
        if keys != sorted(set(keys)):
            raise ValueError("packet_grounding_v2_identity_claims_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_timepoint_shape(self) -> PacketGroundingModelCompletedV2:
        paths = {item.field_path for item in self.numeric_claims}
        observed = paths.intersection(
            {
                "finding.timepoint.value",
                "finding.timepoint.lower",
                "finding.timepoint.upper",
            }
        )
        expected = {
            "not_reported": set(),
            "reported_text": set(),
            "exact": {"finding.timepoint.value"},
            "range": {"finding.timepoint.lower", "finding.timepoint.upper"},
        }[self.timepoint.kind]
        if observed != expected:
            raise ValueError("packet_grounding_v2_timepoint_numeric_shape_mismatch")
        return self


class PacketGroundingModelAbstentionV2(_FrozenExactModel):
    outcome_version: Literal["native-packet-grounding-model-outcome-v2"] = (
        MODEL_OUTCOME_V2_VERSION
    )
    packet_status: Literal["unable_to_complete"] = "unable_to_complete"
    candidate_binding_sha256: Sha256
    reason: Literal[
        "source_support_incomplete",
        "candidate_ambiguous",
        "numeric_token_ambiguous",
        "identity_not_groundable",
        "timepoint_not_groundable",
        "contract_cannot_be_satisfied",
    ]


type PacketGroundingModelOutcomeV2 = Annotated[
    PacketGroundingModelCompletedV2 | PacketGroundingModelAbstentionV2,
    Field(discriminator="packet_status"),
]
_MODEL_OUTCOME_ADAPTER = TypeAdapter(PacketGroundingModelOutcomeV2)


def freeze_packet_candidate_binding_v2(
    *, candidate: NativeCandidateDescriptor, projection: FrozenSourceProjectionV1
) -> PacketCandidateBindingV2:
    candidate = NativeCandidateDescriptor.model_validate(
        candidate.model_dump(mode="json")
    )
    projection = FrozenSourceProjectionV1.model_validate(
        projection.model_dump(mode="json")
    )
    if projection.projection_status != "ready":
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_projection_not_ready"
        )
    if not set(candidate.line_ids).issubset(set(projection.exposed_line_ids)):
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_candidate_line_not_exposed"
        )
    if candidate.outcome_name not in set(projection.allowed_outcomes):
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_candidate_outcome_not_allowed"
        )
    payload: dict[str, Any] = {
        "binding_version": CANDIDATE_BINDING_V2_VERSION,
        "candidate_index": candidate.candidate_index,
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "outcome_name": candidate.outcome_name,
        "effect_kind": candidate.effect_kind,
        "line_ids": candidate.line_ids,
        "projection_sha256": projection.projection_sha256,
    }
    return PacketCandidateBindingV2.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def freeze_passage_packet_candidate_binding_v2(
    *,
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
) -> PacketPassageCandidateBindingV2:
    """Bind an additive p2 candidate to the exact selected prompt surface."""

    candidate = MetaSynPassageCandidateV2.model_validate(
        candidate.model_dump(mode="json")
    )
    projection = FrozenMetaSynProjectionV2.model_validate(
        projection.model_dump(mode="json")
    )
    selected = set(projection.selected_passage_anchors)
    if not set(candidate.passage_ids).issubset(selected):
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_candidate_passage_not_exposed"
        )
    payload: dict[str, Any] = {
        "binding_version": PASSAGE_CANDIDATE_BINDING_V2_VERSION,
        "candidate_index": candidate.candidate_index,
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "canonical_outcome_id": candidate.canonical_outcome_id,
        "outcome_concept_quote": candidate.outcome_concept_quote,
        "effect_kind": candidate.effect_kind,
        "passage_ids": candidate.passage_ids,
        "projection_sha256": projection.projection_sha256,
    }
    return PacketPassageCandidateBindingV2.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def _specialized_model_schema(
    binding: PacketCandidateBindingLikeV2,
) -> dict[str, Any]:
    schema = deepcopy(_MODEL_OUTCOME_ADAPTER.json_schema())
    allowed_paths = list(_allowed_numeric_paths(binding.effect_kind))

    def non_null_schema(value: dict[str, Any]) -> dict[str, Any]:
        branches = value.get("anyOf")
        if not isinstance(branches, list):
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_nullable_schema_shape_invalid"
            )
        retained = [
            deepcopy(item)
            for item in branches
            if isinstance(item, dict) and item.get("type") != "null"
        ]
        if len(retained) != 1:
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_nullable_schema_shape_invalid"
            )
        return retained[0]

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                binding_property = properties.get("candidate_binding_sha256")
                if isinstance(binding_property, dict):
                    binding_property["const"] = binding.binding_sha256
                field_property = properties.get("field_path")
                if isinstance(field_property, dict) and (
                    "verbatim_numeric_token" in properties
                ):
                    field_property["enum"] = allowed_paths
                format_property = properties.get("effect_format_token")
                unit_property = properties.get("effect_unit")
                if isinstance(format_property, dict) and isinstance(
                    unit_property, dict
                ):
                    if binding.effect_kind in {
                        "direct_standard_error",
                        "direct_variance",
                        "direct_confidence_interval",
                    }:
                        properties["effect_format_token"] = non_null_schema(
                            format_property
                        )
                    elif binding.effect_kind == "continuous_group_statistics":
                        properties["effect_format_token"] = {
                            "const": None,
                            "type": "null",
                        }
                    else:
                        properties["effect_format_token"] = {
                            "const": None,
                            "type": "null",
                        }
                        properties["effect_unit"] = {
                            "const": None,
                            "type": "null",
                        }
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    schema["$id"] = (
        "urn:literature-multiverse:native-packet-grounding-v2:"
        f"{binding.binding_sha256}"
    )
    schema["x-literature-multiverse-model-authored-offsets"] = False
    schema["x-literature-multiverse-candidate-binding-sha256"] = (
        binding.binding_sha256
    )
    validator = validator_for(schema)
    try:
        validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - checked by focused tests
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_schema_invalid"
        ) from exc
    return json.loads(canonical_json_bytes(schema))


def _fixture_numeric_path(effect_kind: EffectKind) -> str:
    return {
        "direct_standard_error": "effect.estimate",
        "direct_variance": "effect.estimate",
        "direct_confidence_interval": "effect.estimate",
        "continuous_group_statistics": "effect.treatment_mean",
        "binary_group_statistics": "effect.treatment_events",
    }[effect_kind]


def _fixture_effect_format_token(effect_kind: EffectKind) -> str | None:
    return (
        "Hedges g"
        if effect_kind
        in {
            "direct_standard_error",
            "direct_variance",
            "direct_confidence_interval",
        }
        else None
    )


class PacketGroundingSchemaBundleV2(_FrozenExactModel):
    schema_bundle_version: Literal["native-packet-grounding-schema-bundle-v2"] = (
        SCHEMA_BUNDLE_V2_VERSION
    )
    candidate_binding_sha256: Sha256
    model_response_schema: dict[str, Any]
    model_response_schema_sha256: Sha256
    completed_fixture: dict[str, Any]
    completed_fixture_sha256: Sha256
    abstaining_fixture: dict[str, Any]
    abstaining_fixture_sha256: Sha256
    fixtures_are_synthetic: Literal[True] = True
    scientific_authority: Literal[False] = False
    schema_bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_bundle(self) -> PacketGroundingSchemaBundleV2:
        if hash_canonical(self.model_response_schema) != self.model_response_schema_sha256:
            raise ValueError("packet_grounding_v2_schema_hash_mismatch")
        if hash_canonical(self.completed_fixture) != self.completed_fixture_sha256:
            raise ValueError("packet_grounding_v2_completed_fixture_hash_mismatch")
        if hash_canonical(self.abstaining_fixture) != self.abstaining_fixture_sha256:
            raise ValueError("packet_grounding_v2_abstaining_fixture_hash_mismatch")
        validator = validator_for(self.model_response_schema)
        try:
            validator.check_schema(self.model_response_schema)
            validator(self.model_response_schema).validate(self.completed_fixture)
            validator(self.model_response_schema).validate(self.abstaining_fixture)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            raise ValueError("packet_grounding_v2_schema_fixture_invalid") from exc
        _validate_self_hash(self, "schema_bundle_sha256")
        return self


def freeze_packet_grounding_schema_bundle_v2(
    *, binding: PacketCandidateBindingLikeV2
) -> PacketGroundingSchemaBundleV2:
    binding = _CANDIDATE_BINDING_ADAPTER.validate_python(
        binding.model_dump(mode="json")
    )
    schema = _specialized_model_schema(binding)
    token = "10" if _fixture_numeric_path(binding.effect_kind) in _INTEGER_PATHS else "0.5"
    effect_format_token = _fixture_effect_format_token(binding.effect_kind)
    completed: dict[str, Any] = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "completed",
        "candidate_binding_sha256": binding.binding_sha256,
        "evidence_quote": (
            f"Synthetic Arm A fixture reports {effect_format_token} value {token}."
            if effect_format_token is not None
            else f"Synthetic Arm A fixture reports value {token}."
        ),
        "effect_format_token": effect_format_token,
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": _fixture_numeric_path(binding.effect_kind),
                "verbatim_numeric_token": token,
                "normalization": "identity",
            }
        ],
        "identity_claims": [
            {
                "field_path": "treatment_arm.label",
                "verbatim_identity_text": "Arm A",
            }
        ],
        "timepoint": {"kind": "not_reported"},
    }
    abstaining: dict[str, Any] = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "unable_to_complete",
        "candidate_binding_sha256": binding.binding_sha256,
        "reason": "source_support_incomplete",
    }
    payload: dict[str, Any] = {
        "schema_bundle_version": SCHEMA_BUNDLE_V2_VERSION,
        "candidate_binding_sha256": binding.binding_sha256,
        "model_response_schema": schema,
        "model_response_schema_sha256": hash_canonical(schema),
        "completed_fixture": completed,
        "completed_fixture_sha256": hash_canonical(completed),
        "abstaining_fixture": abstaining,
        "abstaining_fixture_sha256": hash_canonical(abstaining),
        "fixtures_are_synthetic": True,
        "scientific_authority": False,
    }
    return PacketGroundingSchemaBundleV2.model_validate(
        {**payload, "schema_bundle_sha256": hash_canonical(payload)}
    )


def _validate_model_outcome(
    *,
    value: Mapping[str, Any],
    binding: PacketCandidateBindingLikeV2,
    schema_bundle: PacketGroundingSchemaBundleV2,
) -> PacketGroundingModelCompletedV2 | PacketGroundingModelAbstentionV2:
    try:
        raw = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_outcome_not_canonical_json"
        ) from exc
    validator = validator_for(schema_bundle.model_response_schema)
    try:
        validator(schema_bundle.model_response_schema).validate(raw)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_outcome_schema_invalid"
        ) from exc
    try:
        outcome = _MODEL_OUTCOME_ADAPTER.validate_python(raw)
    except ValueError as exc:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_outcome_contract_invalid"
        ) from exc
    if outcome.model_dump(mode="json") != raw:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_outcome_normalized"
        )
    if outcome.candidate_binding_sha256 != binding.binding_sha256:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_model_candidate_binding_mismatch"
        )
    if isinstance(outcome, PacketGroundingModelCompletedV2):
        allowed = set(_allowed_numeric_paths(binding.effect_kind))
        if any(item.field_path not in allowed for item in outcome.numeric_claims):
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_numeric_path_incompatible"
            )
        if binding.effect_kind in {
            "direct_standard_error",
            "direct_variance",
            "direct_confidence_interval",
        }:
            if outcome.effect_format_token is None:
                raise NativePacketGroundingV2Error(
                    "packet_grounding_v2_direct_effect_format_missing"
                )
            _, effect_format = _effect_format_from_exact_token(
                outcome.effect_format_token
            )
            if (effect_format is EffectFormat.MEAN_DIFFERENCE) != (
                outcome.effect_unit is not None
            ):
                raise NativePacketGroundingV2Error(
                    "packet_grounding_v2_effect_format_unit_shape_mismatch"
                )
        elif outcome.effect_format_token is not None:
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_group_statistics_forbid_reported_effect_format"
            )
        elif (
            binding.effect_kind == "binary_group_statistics"
            and outcome.effect_unit is not None
        ):
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_binary_group_statistics_forbid_unit"
            )
    return outcome


class PacketEvidenceReceiptV2(_FrozenExactModel):
    evidence_receipt_version: Literal["native-packet-evidence-receipt-v2"] = (
        EVIDENCE_RECEIPT_V2_VERSION
    )
    projection_sha256: Sha256
    passage_text_sha256: Sha256
    source_locator: SourceLocatorText
    line_id: Annotated[StrictStr, Field(pattern=r"^L[1-9][0-9]*$")]
    passage_rank: Annotated[int, Field(ge=1)]
    passage_anchor: Annotated[
        StrictStr, Field(pattern=r"^p2-[0-9a-f]{64}$")
    ] | None = None
    passage_lineage_sha256: Sha256 | None = None
    source_origin_sha256: Sha256 | None = None
    source_occurrence_count: Annotated[int, Field(ge=1)] | None = None
    evidence_quote: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_EVIDENCE_QUOTE_CHARACTERS)
    ]
    evidence_quote_sha256: Sha256
    quote_start_in_passage: Annotated[int, Field(ge=0)]
    quote_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    quote_source_char_start: Annotated[int, Field(ge=0)]
    quote_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    quote_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    quote_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    evidence_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_evidence(self) -> PacketEvidenceReceiptV2:
        passage_metadata = (
            self.passage_anchor,
            self.passage_lineage_sha256,
            self.source_origin_sha256,
            self.source_occurrence_count,
        )
        if any(item is None for item in passage_metadata) and any(
            item is not None for item in passage_metadata
        ):
            raise ValueError("packet_grounding_v2_passage_metadata_partial")
        if hash_canonical(self.evidence_quote) != self.evidence_quote_sha256:
            raise ValueError("packet_grounding_v2_quote_hash_mismatch")
        if (
            self.quote_end_exclusive_in_passage - self.quote_start_in_passage
            != len(self.evidence_quote)
            or self.quote_source_char_end_exclusive - self.quote_source_char_start
            != len(self.evidence_quote)
            or self.quote_source_utf8_byte_end_exclusive
            - self.quote_source_utf8_byte_start
            != len(self.evidence_quote.encode("utf-8"))
        ):
            raise ValueError("packet_grounding_v2_quote_offset_shape_invalid")
        _validate_self_hash(self, "evidence_receipt_sha256")
        return self


class PacketEffectFormatGroundingReceiptV2(_FrozenExactModel):
    effect_format_receipt_version: Literal[
        "native-packet-effect-format-receipt-v2"
    ] = EFFECT_FORMAT_RECEIPT_V2_VERSION
    candidate_binding_sha256: Sha256
    evidence_quote_sha256: Sha256
    verbatim_effect_format_token: Annotated[
        StrictStr, Field(min_length=1, max_length=64)
    ]
    verbatim_effect_format_token_sha256: Sha256
    normalized_alias: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    effect_format: EffectFormat
    token_start_in_quote: Annotated[int, Field(ge=0)]
    token_end_exclusive_in_quote: Annotated[int, Field(gt=0)]
    token_source_char_start: Annotated[int, Field(ge=0)]
    token_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    token_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    token_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    alias_policy_sha256: Sha256
    effect_format_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_effect_format(self) -> PacketEffectFormatGroundingReceiptV2:
        if (
            hash_canonical(self.verbatim_effect_format_token)
            != self.verbatim_effect_format_token_sha256
        ):
            raise ValueError("packet_grounding_v2_effect_format_token_hash_mismatch")
        normalized, effect_format = _effect_format_from_exact_token(
            self.verbatim_effect_format_token
        )
        if (
            self.normalized_alias != normalized
            or self.effect_format is not effect_format
        ):
            raise ValueError("packet_grounding_v2_effect_format_mapping_mismatch")
        if self.alias_policy_sha256 != EFFECT_FORMAT_ALIAS_POLICY_V2_SHA256:
            raise ValueError("packet_grounding_v2_effect_format_policy_mismatch")
        token = self.verbatim_effect_format_token
        if (
            self.token_end_exclusive_in_quote - self.token_start_in_quote
            != len(token)
            or self.token_source_char_end_exclusive - self.token_source_char_start
            != len(token)
            or self.token_source_utf8_byte_end_exclusive
            - self.token_source_utf8_byte_start
            != len(token.encode("utf-8"))
        ):
            raise ValueError("packet_grounding_v2_effect_format_offset_shape_invalid")
        _validate_self_hash(self, "effect_format_receipt_sha256")
        return self


class PacketNormalizationReceiptV2(_FrozenExactModel):
    normalization_receipt_version: Literal[
        "native-packet-normalization-receipt-v2"
    ] = NORMALIZATION_RECEIPT_V2_VERSION
    field_path: NumericFieldPath
    verbatim_numeric_token: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_DECIMAL_LEXEME_CHARACTERS + 1)
    ]
    verbatim_numeric_token_sha256: Sha256
    normalization: Literal["identity", "percent_to_proportion"]
    normalized_numeric_lexeme: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_DECIMAL_LEXEME_CHARACTERS)
    ]
    normalization_policy_sha256: Sha256
    normalization_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_normalization(self) -> PacketNormalizationReceiptV2:
        if hash_canonical(self.verbatim_numeric_token) != self.verbatim_numeric_token_sha256:
            raise ValueError("packet_grounding_v2_numeric_token_hash_mismatch")
        if self.normalization_policy_sha256 != NORMALIZATION_POLICY_V2_SHA256:
            raise ValueError("packet_grounding_v2_normalization_policy_mismatch")
        expected = _normalized_numeric_lexeme(
            field_path=self.field_path,
            verbatim_numeric_token=self.verbatim_numeric_token,
            normalization=self.normalization,
        )
        if self.normalized_numeric_lexeme != expected:
            raise ValueError("packet_grounding_v2_normalized_value_mismatch")
        _validate_self_hash(self, "normalization_receipt_sha256")
        return self


class PacketNumericGroundingReceiptV2(_FrozenExactModel):
    numeric_receipt_version: Literal[
        "native-packet-numeric-grounding-receipt-v2"
    ] = NUMERIC_RECEIPT_V2_VERSION
    candidate_binding_sha256: Sha256
    evidence_quote_sha256: Sha256
    normalization_receipt: PacketNormalizationReceiptV2
    token_start_in_quote: Annotated[int, Field(ge=0)]
    token_end_exclusive_in_quote: Annotated[int, Field(gt=0)]
    token_source_char_start: Annotated[int, Field(ge=0)]
    token_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    token_source_utf8_byte_start: Annotated[int, Field(ge=0)]
    token_source_utf8_byte_end_exclusive: Annotated[int, Field(gt=0)]
    numeric_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_numeric_receipt(self) -> PacketNumericGroundingReceiptV2:
        token = self.normalization_receipt.verbatim_numeric_token
        if (
            self.token_end_exclusive_in_quote - self.token_start_in_quote != len(token)
            or self.token_source_char_end_exclusive - self.token_source_char_start
            != len(token)
            or self.token_source_utf8_byte_end_exclusive
            - self.token_source_utf8_byte_start
            != len(token.encode("utf-8"))
        ):
            raise ValueError("packet_grounding_v2_numeric_offset_shape_invalid")
        _validate_self_hash(self, "numeric_receipt_sha256")
        return self


class PacketIdentityGroundingReceiptV2(_FrozenExactModel):
    identity_receipt_version: Literal[
        "native-packet-identity-grounding-receipt-v2"
    ] = IDENTITY_RECEIPT_V2_VERSION
    projection_sha256: Sha256
    field_path: GroundedIdentityFieldPath
    verbatim_identity_text: NonEmptyText
    verbatim_identity_text_sha256: Sha256
    occurrence_count: Annotated[int, Field(ge=1, le=MAX_OCCURRENCES)]
    first_passage_rank: Annotated[int, Field(ge=1)]
    first_line_id: Annotated[StrictStr, Field(pattern=r"^L[1-9][0-9]*$")]
    first_passage_anchor: Annotated[
        StrictStr, Field(pattern=r"^p2-[0-9a-f]{64}$")
    ] | None = None
    first_source_origin_sha256: Sha256 | None = None
    first_start_in_passage: Annotated[int, Field(ge=0)]
    first_end_exclusive_in_passage: Annotated[int, Field(gt=0)]
    first_source_char_start: Annotated[int, Field(ge=0)]
    first_source_char_end_exclusive: Annotated[int, Field(gt=0)]
    identity_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity_receipt(self) -> PacketIdentityGroundingReceiptV2:
        if (self.first_passage_anchor is None) != (
            self.first_source_origin_sha256 is None
        ):
            raise ValueError("packet_grounding_v2_identity_passage_metadata_partial")
        if (
            hash_canonical(self.verbatim_identity_text)
            != self.verbatim_identity_text_sha256
        ):
            raise ValueError("packet_grounding_v2_identity_text_hash_mismatch")
        if (
            self.first_end_exclusive_in_passage - self.first_start_in_passage
            != len(self.verbatim_identity_text)
            or self.first_source_char_end_exclusive - self.first_source_char_start
            != len(self.verbatim_identity_text)
        ):
            raise ValueError("packet_grounding_v2_identity_offset_shape_invalid")
        _validate_self_hash(self, "identity_receipt_sha256")
        return self


def _timepoint_identity_claims(
    timepoint: PacketTimepointV2,
) -> list[tuple[GroundedIdentityFieldPath, str]]:
    claims: list[tuple[GroundedIdentityFieldPath, str]] = []
    anchor = getattr(timepoint, "anchor", None)
    raw_label = getattr(timepoint, "raw_label", None)
    if anchor is not None:
        claims.append(("finding.timepoint.anchor", anchor))
    if raw_label is not None:
        claims.append(("finding.timepoint.raw_label", raw_label))
    return claims


class PacketGroundingCompletedReceiptV2(_FrozenExactModel):
    grounding_version: Literal["native-packet-grounding-v2"] = (
        PACKET_GROUNDING_V2_VERSION
    )
    receipt_version: Literal[
        "native-packet-grounding-completed-receipt-v2"
    ] = COMPLETED_RECEIPT_V2_VERSION
    status: Literal["completed"] = "completed"
    candidate_binding: PacketCandidateBindingLikeV2
    schema_bundle_sha256: Sha256
    projection_sha256: Sha256
    model_outcome: PacketGroundingModelCompletedV2
    model_outcome_sha256: Sha256
    evidence_receipt: PacketEvidenceReceiptV2
    effect_format_receipt: PacketEffectFormatGroundingReceiptV2 | None
    numeric_receipts: Annotated[
        list[PacketNumericGroundingReceiptV2],
        Field(min_length=1, max_length=MAX_NUMERIC_SUPPORT_ITEMS),
    ]
    identity_receipts: Annotated[
        list[PacketIdentityGroundingReceiptV2], Field(max_length=MAX_IDENTITY_CLAIMS + 2)
    ]
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> PacketGroundingCompletedReceiptV2:
        if self.projection_sha256 != self.candidate_binding.projection_sha256:
            raise ValueError("packet_grounding_v2_receipt_projection_mismatch")
        if (
            self.model_outcome.candidate_binding_sha256
            != self.candidate_binding.binding_sha256
        ):
            raise ValueError("packet_grounding_v2_receipt_candidate_mismatch")
        if hash_canonical(self.model_outcome) != self.model_outcome_sha256:
            raise ValueError("packet_grounding_v2_model_outcome_hash_mismatch")
        if self.evidence_receipt.projection_sha256 != self.projection_sha256:
            raise ValueError("packet_grounding_v2_evidence_projection_mismatch")
        if self.model_outcome.evidence_quote != self.evidence_receipt.evidence_quote:
            raise ValueError("packet_grounding_v2_receipt_quote_mismatch")
        if isinstance(self.candidate_binding, PacketPassageCandidateBindingV2):
            if self.evidence_receipt.passage_anchor not in set(
                self.candidate_binding.passage_ids
            ):
                raise ValueError("packet_grounding_v2_evidence_passage_mismatch")
        elif self.evidence_receipt.passage_anchor is not None:
            raise ValueError("packet_grounding_v2_unexpected_evidence_passage_anchor")
        expects_reported_format = self.candidate_binding.effect_kind in {
            "direct_standard_error",
            "direct_variance",
            "direct_confidence_interval",
        }
        if expects_reported_format != (self.effect_format_receipt is not None):
            raise ValueError("packet_grounding_v2_effect_format_receipt_shape_mismatch")
        format_receipt = self.effect_format_receipt
        if format_receipt is None:
            if self.model_outcome.effect_format_token is not None:
                raise ValueError("packet_grounding_v2_unexpected_effect_format_token")
        elif (
            format_receipt.candidate_binding_sha256
            != self.candidate_binding.binding_sha256
            or format_receipt.evidence_quote_sha256
            != self.evidence_receipt.evidence_quote_sha256
            or format_receipt.verbatim_effect_format_token
            != self.model_outcome.effect_format_token
        ):
            raise ValueError("packet_grounding_v2_effect_format_context_mismatch")
        if format_receipt is not None:
            self._validate_effect_format_offsets(format_receipt)

        expected_numeric = [
            (item.field_path, item.verbatim_numeric_token, item.normalization)
            for item in self.model_outcome.numeric_claims
        ]
        observed_numeric = [
            (
                item.normalization_receipt.field_path,
                item.normalization_receipt.verbatim_numeric_token,
                item.normalization_receipt.normalization,
            )
            for item in self.numeric_receipts
        ]
        if observed_numeric != expected_numeric:
            raise ValueError("packet_grounding_v2_receipt_numeric_claim_mismatch")
        spans: list[tuple[int, int]] = []
        quote = self.evidence_receipt.evidence_quote
        for item in self.numeric_receipts:
            if (
                item.candidate_binding_sha256
                != self.candidate_binding.binding_sha256
                or item.evidence_quote_sha256
                != self.evidence_receipt.evidence_quote_sha256
            ):
                raise ValueError("packet_grounding_v2_numeric_context_mismatch")
            start = item.token_start_in_quote
            end = item.token_end_exclusive_in_quote
            token = item.normalization_receipt.verbatim_numeric_token
            if quote[start:end] != token:
                raise ValueError("packet_grounding_v2_numeric_quote_slice_mismatch")
            if (
                item.token_source_char_start
                != self.evidence_receipt.quote_source_char_start + start
                or item.token_source_char_end_exclusive
                != self.evidence_receipt.quote_source_char_start + end
                or item.token_source_utf8_byte_start
                != self.evidence_receipt.quote_source_utf8_byte_start
                + len(quote[:start].encode("utf-8"))
                or item.token_source_utf8_byte_end_exclusive
                != self.evidence_receipt.quote_source_utf8_byte_start
                + len(quote[:end].encode("utf-8"))
            ):
                raise ValueError("packet_grounding_v2_numeric_absolute_offset_mismatch")
            spans.append((start, end))
        if len(spans) != len(set(spans)):
            raise ValueError("packet_grounding_v2_numeric_source_span_reused")

        expected_identity = [
            (item.field_path, item.verbatim_identity_text)
            for item in self.model_outcome.identity_claims
        ] + _timepoint_identity_claims(self.model_outcome.timepoint)
        if self.model_outcome.effect_unit is not None:
            expected_identity.append(("effect.unit", self.model_outcome.effect_unit))
        expected_identity.sort()
        observed_identity = [
            (item.field_path, item.verbatim_identity_text)
            for item in self.identity_receipts
        ]
        if observed_identity != expected_identity:
            raise ValueError("packet_grounding_v2_receipt_identity_claim_mismatch")
        if any(
            item.projection_sha256 != self.projection_sha256
            for item in self.identity_receipts
        ):
            raise ValueError("packet_grounding_v2_identity_projection_mismatch")
        _validate_self_hash(self, "receipt_sha256")
        return self

    def _validate_effect_format_offsets(
        self, format_receipt: PacketEffectFormatGroundingReceiptV2
    ) -> None:
        if (
            self.model_outcome.effect_format_token
            != format_receipt.verbatim_effect_format_token
        ):
            raise ValueError("packet_grounding_v2_effect_format_context_mismatch")
        format_start = format_receipt.token_start_in_quote
        format_end = format_receipt.token_end_exclusive_in_quote
        if self.evidence_receipt.evidence_quote[format_start:format_end] != (
            format_receipt.verbatim_effect_format_token
        ):
            raise ValueError("packet_grounding_v2_effect_format_quote_slice_mismatch")
        if (
            format_receipt.token_source_char_start
            != self.evidence_receipt.quote_source_char_start + format_start
            or format_receipt.token_source_char_end_exclusive
            != self.evidence_receipt.quote_source_char_start + format_end
            or format_receipt.token_source_utf8_byte_start
            != self.evidence_receipt.quote_source_utf8_byte_start
            + len(
                self.evidence_receipt.evidence_quote[:format_start].encode("utf-8")
            )
            or format_receipt.token_source_utf8_byte_end_exclusive
            != self.evidence_receipt.quote_source_utf8_byte_start
            + len(self.evidence_receipt.evidence_quote[:format_end].encode("utf-8"))
        ):
            raise ValueError("packet_grounding_v2_effect_format_absolute_offset_mismatch")


class PacketGroundingAbstentionReceiptV2(_FrozenExactModel):
    grounding_version: Literal["native-packet-grounding-v2"] = (
        PACKET_GROUNDING_V2_VERSION
    )
    receipt_version: Literal[
        "native-packet-grounding-abstention-receipt-v2"
    ] = ABSTENTION_RECEIPT_V2_VERSION
    status: Literal["unable_to_complete"] = "unable_to_complete"
    candidate_binding: PacketCandidateBindingLikeV2
    schema_bundle_sha256: Sha256
    projection_sha256: Sha256
    model_outcome: PacketGroundingModelAbstentionV2
    model_outcome_sha256: Sha256
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> PacketGroundingAbstentionReceiptV2:
        if self.projection_sha256 != self.candidate_binding.projection_sha256:
            raise ValueError("packet_grounding_v2_abstention_projection_mismatch")
        if (
            self.model_outcome.candidate_binding_sha256
            != self.candidate_binding.binding_sha256
        ):
            raise ValueError("packet_grounding_v2_abstention_candidate_mismatch")
        if hash_canonical(self.model_outcome) != self.model_outcome_sha256:
            raise ValueError("packet_grounding_v2_abstention_outcome_hash_mismatch")
        _validate_self_hash(self, "receipt_sha256")
        return self


type PacketGroundingReceiptV2 = (
    PacketGroundingCompletedReceiptV2 | PacketGroundingAbstentionReceiptV2
)
_RECEIPT_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)


def _unique_quote_match(
    *, quote: str, candidate_line_ids: Sequence[str], projection: FrozenSourceProjectionV1
) -> tuple[FrozenProjectedPassageV1, int]:
    matches: list[tuple[FrozenProjectedPassageV1, int]] = []
    allowed = set(candidate_line_ids)
    for passage in projection.passages:
        if passage.line_id not in allowed:
            continue
        matches.extend((passage, start) for start in _occurrences(passage.text, quote))
    if not matches:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_evidence_quote_absent"
        )
    if len(matches) != 1:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_evidence_quote_not_unique"
        )
    return matches[0]


def _freeze_evidence_receipt(
    *,
    quote: str,
    passage: FrozenProjectedPassageV1,
    quote_start: int,
    projection: FrozenSourceProjectionV1,
) -> PacketEvidenceReceiptV2:
    quote_end = quote_start + len(quote)
    payload: dict[str, Any] = {
        "evidence_receipt_version": EVIDENCE_RECEIPT_V2_VERSION,
        "projection_sha256": projection.projection_sha256,
        "passage_text_sha256": passage.text_sha256,
        "source_locator": projection.source_locator,
        "line_id": passage.line_id,
        "passage_rank": passage.passage_rank,
        "passage_anchor": None,
        "passage_lineage_sha256": None,
        "source_origin_sha256": None,
        "source_occurrence_count": None,
        "evidence_quote": quote,
        "evidence_quote_sha256": hash_canonical(quote),
        "quote_start_in_passage": quote_start,
        "quote_end_exclusive_in_passage": quote_end,
        "quote_source_char_start": passage.source_char_start + quote_start,
        "quote_source_char_end_exclusive": passage.source_char_start + quote_end,
        "quote_source_utf8_byte_start": passage.source_utf8_byte_start
        + len(passage.text[:quote_start].encode("utf-8")),
        "quote_source_utf8_byte_end_exclusive": passage.source_utf8_byte_start
        + len(passage.text[:quote_end].encode("utf-8")),
    }
    return PacketEvidenceReceiptV2.model_validate(
        {**payload, "evidence_receipt_sha256": hash_canonical(payload)}
    )


def _freeze_effect_format_receipt(
    *,
    token: str,
    evidence: PacketEvidenceReceiptV2,
    binding: PacketCandidateBindingLikeV2,
) -> PacketEffectFormatGroundingReceiptV2:
    normalized, effect_format = _effect_format_from_exact_token(token)
    starts = _occurrences(evidence.evidence_quote, token)
    if not starts:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_effect_format_token_absent"
        )
    if len(starts) != 1:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_effect_format_token_not_unique"
        )
    start = starts[0]
    end = start + len(token)
    quote = evidence.evidence_quote
    payload: dict[str, Any] = {
        "effect_format_receipt_version": EFFECT_FORMAT_RECEIPT_V2_VERSION,
        "candidate_binding_sha256": binding.binding_sha256,
        "evidence_quote_sha256": evidence.evidence_quote_sha256,
        "verbatim_effect_format_token": token,
        "verbatim_effect_format_token_sha256": hash_canonical(token),
        "normalized_alias": normalized,
        "effect_format": effect_format,
        "token_start_in_quote": start,
        "token_end_exclusive_in_quote": end,
        "token_source_char_start": evidence.quote_source_char_start + start,
        "token_source_char_end_exclusive": evidence.quote_source_char_start + end,
        "token_source_utf8_byte_start": evidence.quote_source_utf8_byte_start
        + len(quote[:start].encode("utf-8")),
        "token_source_utf8_byte_end_exclusive": evidence.quote_source_utf8_byte_start
        + len(quote[:end].encode("utf-8")),
        "alias_policy_sha256": EFFECT_FORMAT_ALIAS_POLICY_V2_SHA256,
    }
    return PacketEffectFormatGroundingReceiptV2.model_validate(
        {**payload, "effect_format_receipt_sha256": hash_canonical(payload)}
    )


def _freeze_normalization_receipt(
    claim: PacketNumericClaimV2,
) -> PacketNormalizationReceiptV2:
    normalized = _normalized_numeric_lexeme(
        field_path=claim.field_path,
        verbatim_numeric_token=claim.verbatim_numeric_token,
        normalization=claim.normalization,
    )
    payload: dict[str, Any] = {
        "normalization_receipt_version": NORMALIZATION_RECEIPT_V2_VERSION,
        "field_path": claim.field_path,
        "verbatim_numeric_token": claim.verbatim_numeric_token,
        "verbatim_numeric_token_sha256": hash_canonical(
            claim.verbatim_numeric_token
        ),
        "normalization": claim.normalization,
        "normalized_numeric_lexeme": normalized,
        "normalization_policy_sha256": NORMALIZATION_POLICY_V2_SHA256,
    }
    return PacketNormalizationReceiptV2.model_validate(
        {**payload, "normalization_receipt_sha256": hash_canonical(payload)}
    )


def _freeze_numeric_receipts(
    *,
    claims: Sequence[PacketNumericClaimV2],
    evidence: PacketEvidenceReceiptV2,
    binding: PacketCandidateBindingLikeV2,
) -> list[PacketNumericGroundingReceiptV2]:
    receipts: list[PacketNumericGroundingReceiptV2] = []
    spans: set[tuple[int, int]] = set()
    quote = evidence.evidence_quote
    for claim in claims:
        starts = _numeric_token_occurrences(
            quote=quote,
            token=claim.verbatim_numeric_token,
            normalization=claim.normalization,
            field_path=claim.field_path,
        )
        if not starts:
            raise NativePacketGroundingV2Error(
                f"packet_grounding_v2_numeric_token_absent:{claim.field_path}"
            )
        if len(starts) != 1:
            raise NativePacketGroundingV2Error(
                f"packet_grounding_v2_numeric_token_not_unique:{claim.field_path}"
            )
        start = starts[0]
        end = start + len(claim.verbatim_numeric_token)
        if (start, end) in spans:
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_numeric_source_span_reused"
            )
        spans.add((start, end))
        normalization_receipt = _freeze_normalization_receipt(claim)
        payload: dict[str, Any] = {
            "numeric_receipt_version": NUMERIC_RECEIPT_V2_VERSION,
            "candidate_binding_sha256": binding.binding_sha256,
            "evidence_quote_sha256": evidence.evidence_quote_sha256,
            "normalization_receipt": normalization_receipt,
            "token_start_in_quote": start,
            "token_end_exclusive_in_quote": end,
            "token_source_char_start": evidence.quote_source_char_start + start,
            "token_source_char_end_exclusive": evidence.quote_source_char_start + end,
            "token_source_utf8_byte_start": evidence.quote_source_utf8_byte_start
            + len(quote[:start].encode("utf-8")),
            "token_source_utf8_byte_end_exclusive": evidence.quote_source_utf8_byte_start
            + len(quote[:end].encode("utf-8")),
        }
        receipts.append(
            PacketNumericGroundingReceiptV2.model_validate(
                {**payload, "numeric_receipt_sha256": hash_canonical(payload)}
            )
        )
    return receipts


def _all_identity_claims(
    outcome: PacketGroundingModelCompletedV2,
) -> list[tuple[GroundedIdentityFieldPath, str]]:
    claims: list[tuple[GroundedIdentityFieldPath, str]] = [
        (item.field_path, item.verbatim_identity_text)
        for item in outcome.identity_claims
    ]
    claims.extend(_timepoint_identity_claims(outcome.timepoint))
    if outcome.effect_unit is not None:
        claims.append(("effect.unit", outcome.effect_unit))
    claims.sort()
    if len(claims) != len(set(claims)):
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_identity_claim_duplicate"
        )
    return claims


def _freeze_identity_receipts(
    *,
    claims: Sequence[tuple[GroundedIdentityFieldPath, str]],
    projection: FrozenSourceProjectionV1,
) -> list[PacketIdentityGroundingReceiptV2]:
    receipts: list[PacketIdentityGroundingReceiptV2] = []
    for field_path, text in claims:
        matches: list[tuple[FrozenProjectedPassageV1, int]] = []
        for passage in projection.passages:
            matches.extend(
                (passage, start) for start in _occurrences(passage.text, text)
            )
        if not matches:
            raise NativePacketGroundingV2Error(
                f"packet_grounding_v2_identity_text_absent:{field_path}"
            )
        matches.sort(key=lambda item: (item[0].passage_rank, item[1]))
        passage, start = matches[0]
        end = start + len(text)
        payload: dict[str, Any] = {
            "identity_receipt_version": IDENTITY_RECEIPT_V2_VERSION,
            "projection_sha256": projection.projection_sha256,
            "field_path": field_path,
            "verbatim_identity_text": text,
            "verbatim_identity_text_sha256": hash_canonical(text),
            "occurrence_count": len(matches),
            "first_passage_rank": passage.passage_rank,
            "first_line_id": passage.line_id,
            "first_passage_anchor": None,
            "first_source_origin_sha256": None,
            "first_start_in_passage": start,
            "first_end_exclusive_in_passage": end,
            "first_source_char_start": passage.source_char_start + start,
            "first_source_char_end_exclusive": passage.source_char_start + end,
        }
        receipts.append(
            PacketIdentityGroundingReceiptV2.model_validate(
                {**payload, "identity_receipt_sha256": hash_canonical(payload)}
            )
        )
    return receipts


def _unique_passage_quote_match(
    *,
    quote: str,
    candidate_passage_ids: Sequence[str],
    projection: FrozenMetaSynProjectionV2,
) -> tuple[AnchoredPassageV2, int]:
    matches: list[tuple[AnchoredPassageV2, int]] = []
    allowed = set(candidate_passage_ids)
    for passage in projection.passages:
        if (
            passage.selection_status != "selected"
            or passage.passage_anchor not in allowed
        ):
            continue
        matches.extend((passage, start) for start in _occurrences(passage.text, quote))
    if not matches:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_evidence_quote_absent"
        )
    if len(matches) != 1:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_evidence_quote_not_unique"
        )
    return matches[0]


def _freeze_passage_evidence_receipt(
    *,
    quote: str,
    passage: AnchoredPassageV2,
    quote_start: int,
    projection: FrozenMetaSynProjectionV2,
) -> PacketEvidenceReceiptV2:
    if passage.prompt_rank is None or passage.selection_status != "selected":
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_evidence_passage_not_selected"
        )
    origin = passage.origins[0]
    quote_end = quote_start + len(quote)
    payload: dict[str, Any] = {
        "evidence_receipt_version": EVIDENCE_RECEIPT_V2_VERSION,
        "projection_sha256": projection.projection_sha256,
        "passage_text_sha256": passage.text_sha256,
        "source_locator": projection.source_locator,
        "line_id": origin.line_id,
        "passage_rank": passage.prompt_rank,
        "passage_anchor": passage.passage_anchor,
        "passage_lineage_sha256": passage.passage_lineage_sha256,
        "source_origin_sha256": origin.origin_sha256,
        "source_occurrence_count": passage.origin_count,
        "evidence_quote": quote,
        "evidence_quote_sha256": hash_canonical(quote),
        "quote_start_in_passage": quote_start,
        "quote_end_exclusive_in_passage": quote_end,
        "quote_source_char_start": origin.source_char_start + quote_start,
        "quote_source_char_end_exclusive": origin.source_char_start + quote_end,
        "quote_source_utf8_byte_start": origin.source_utf8_byte_start
        + len(passage.text[:quote_start].encode("utf-8")),
        "quote_source_utf8_byte_end_exclusive": origin.source_utf8_byte_start
        + len(passage.text[:quote_end].encode("utf-8")),
    }
    return PacketEvidenceReceiptV2.model_validate(
        {**payload, "evidence_receipt_sha256": hash_canonical(payload)}
    )


def _freeze_passage_identity_receipts(
    *,
    claims: Sequence[tuple[GroundedIdentityFieldPath, str]],
    projection: FrozenMetaSynProjectionV2,
) -> list[PacketIdentityGroundingReceiptV2]:
    receipts: list[PacketIdentityGroundingReceiptV2] = []
    selected = sorted(
        (item for item in projection.passages if item.selection_status == "selected"),
        key=lambda item: (item.prompt_rank or 0, item.passage_anchor),
    )
    for field_path, text in claims:
        matches: list[tuple[AnchoredPassageV2, int]] = []
        for passage in selected:
            matches.extend(
                (passage, start) for start in _occurrences(passage.text, text)
            )
        if not matches:
            raise NativePacketGroundingV2Error(
                f"packet_grounding_v2_identity_text_absent:{field_path}"
            )
        source_occurrence_count = sum(passage.origin_count for passage, _ in matches)
        if source_occurrence_count > MAX_OCCURRENCES:
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_occurrence_cap_exceeded"
            )
        passage, start = matches[0]
        if passage.prompt_rank is None:  # pragma: no cover - filtered above
            raise NativePacketGroundingV2Error(
                "packet_grounding_v2_identity_passage_not_selected"
            )
        origin = passage.origins[0]
        end = start + len(text)
        payload: dict[str, Any] = {
            "identity_receipt_version": IDENTITY_RECEIPT_V2_VERSION,
            "projection_sha256": projection.projection_sha256,
            "field_path": field_path,
            "verbatim_identity_text": text,
            "verbatim_identity_text_sha256": hash_canonical(text),
            "occurrence_count": source_occurrence_count,
            "first_passage_rank": passage.prompt_rank,
            "first_line_id": origin.line_id,
            "first_passage_anchor": passage.passage_anchor,
            "first_source_origin_sha256": origin.origin_sha256,
            "first_start_in_passage": start,
            "first_end_exclusive_in_passage": end,
            "first_source_char_start": origin.source_char_start + start,
            "first_source_char_end_exclusive": origin.source_char_start + end,
        }
        receipts.append(
            PacketIdentityGroundingReceiptV2.model_validate(
                {**payload, "identity_receipt_sha256": hash_canonical(payload)}
            )
        )
    return receipts


def freeze_packet_grounding_receipt_v2(
    *,
    model_outcome: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
    projection: FrozenSourceProjectionV1,
) -> PacketGroundingReceiptV2:
    """Validate one raw compact response and derive all offsets locally."""

    binding = freeze_packet_candidate_binding_v2(
        candidate=candidate, projection=projection
    )
    schema_bundle = freeze_packet_grounding_schema_bundle_v2(binding=binding)
    outcome = _validate_model_outcome(
        value=model_outcome, binding=binding, schema_bundle=schema_bundle
    )
    outcome_sha256 = hash_canonical(outcome)
    if isinstance(outcome, PacketGroundingModelAbstentionV2):
        payload: dict[str, Any] = {
            "grounding_version": PACKET_GROUNDING_V2_VERSION,
            "receipt_version": ABSTENTION_RECEIPT_V2_VERSION,
            "status": "unable_to_complete",
            "candidate_binding": binding,
            "schema_bundle_sha256": schema_bundle.schema_bundle_sha256,
            "projection_sha256": projection.projection_sha256,
            "model_outcome": outcome,
            "model_outcome_sha256": outcome_sha256,
            "claim_release_authority": False,
        }
        return PacketGroundingAbstentionReceiptV2.model_validate(
            {**payload, "receipt_sha256": hash_canonical(payload)}
        )

    passage, quote_start = _unique_quote_match(
        quote=outcome.evidence_quote,
        candidate_line_ids=candidate.line_ids,
        projection=projection,
    )
    evidence = _freeze_evidence_receipt(
        quote=outcome.evidence_quote,
        passage=passage,
        quote_start=quote_start,
        projection=projection,
    )
    effect_format = (
        _freeze_effect_format_receipt(
            token=outcome.effect_format_token,
            evidence=evidence,
            binding=binding,
        )
        if outcome.effect_format_token is not None
        else None
    )
    numeric = _freeze_numeric_receipts(
        claims=outcome.numeric_claims, evidence=evidence, binding=binding
    )
    identities = _freeze_identity_receipts(
        claims=_all_identity_claims(outcome), projection=projection
    )
    payload = {
        "grounding_version": PACKET_GROUNDING_V2_VERSION,
        "receipt_version": COMPLETED_RECEIPT_V2_VERSION,
        "status": "completed",
        "candidate_binding": binding,
        "schema_bundle_sha256": schema_bundle.schema_bundle_sha256,
        "projection_sha256": projection.projection_sha256,
        "model_outcome": outcome,
        "model_outcome_sha256": outcome_sha256,
        "evidence_receipt": evidence,
        "effect_format_receipt": effect_format,
        "numeric_receipts": numeric,
        "identity_receipts": identities,
        "claim_release_authority": False,
    }
    return PacketGroundingCompletedReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def freeze_passage_packet_grounding_receipt_v2(
    *,
    model_outcome: Mapping[str, Any],
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
) -> PacketGroundingReceiptV2:
    """Ground a compact response against one exact p2 candidate surface."""

    candidate = MetaSynPassageCandidateV2.model_validate(
        candidate.model_dump(mode="json")
    )
    projection = FrozenMetaSynProjectionV2.model_validate(
        projection.model_dump(mode="json")
    )
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate, projection=projection
    )
    schema_bundle = freeze_packet_grounding_schema_bundle_v2(binding=binding)
    outcome = _validate_model_outcome(
        value=model_outcome, binding=binding, schema_bundle=schema_bundle
    )
    outcome_sha256 = hash_canonical(outcome)
    if isinstance(outcome, PacketGroundingModelAbstentionV2):
        payload: dict[str, Any] = {
            "grounding_version": PACKET_GROUNDING_V2_VERSION,
            "receipt_version": ABSTENTION_RECEIPT_V2_VERSION,
            "status": "unable_to_complete",
            "candidate_binding": binding,
            "schema_bundle_sha256": schema_bundle.schema_bundle_sha256,
            "projection_sha256": projection.projection_sha256,
            "model_outcome": outcome,
            "model_outcome_sha256": outcome_sha256,
            "claim_release_authority": False,
        }
        return PacketGroundingAbstentionReceiptV2.model_validate(
            {**payload, "receipt_sha256": hash_canonical(payload)}
        )

    passage, quote_start = _unique_passage_quote_match(
        quote=outcome.evidence_quote,
        candidate_passage_ids=candidate.passage_ids,
        projection=projection,
    )
    evidence = _freeze_passage_evidence_receipt(
        quote=outcome.evidence_quote,
        passage=passage,
        quote_start=quote_start,
        projection=projection,
    )
    effect_format = (
        _freeze_effect_format_receipt(
            token=outcome.effect_format_token,
            evidence=evidence,
            binding=binding,
        )
        if outcome.effect_format_token is not None
        else None
    )
    numeric = _freeze_numeric_receipts(
        claims=outcome.numeric_claims, evidence=evidence, binding=binding
    )
    identities = _freeze_passage_identity_receipts(
        claims=_all_identity_claims(outcome), projection=projection
    )
    payload = {
        "grounding_version": PACKET_GROUNDING_V2_VERSION,
        "receipt_version": COMPLETED_RECEIPT_V2_VERSION,
        "status": "completed",
        "candidate_binding": binding,
        "schema_bundle_sha256": schema_bundle.schema_bundle_sha256,
        "projection_sha256": projection.projection_sha256,
        "model_outcome": outcome,
        "model_outcome_sha256": outcome_sha256,
        "evidence_receipt": evidence,
        "effect_format_receipt": effect_format,
        "numeric_receipts": numeric,
        "identity_receipts": identities,
        "claim_release_authority": False,
    }
    return PacketGroundingCompletedReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_passage_packet_grounding_receipt_v2(
    *,
    receipt: PacketGroundingReceiptV2 | Mapping[str, Any],
    model_outcome: Mapping[str, Any],
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
) -> PacketGroundingReceiptV2:
    """Externally replay a passage-anchored receipt from frozen inputs."""

    try:
        canonical = _RECEIPT_ADAPTER.validate_python(receipt)
    except ValueError as exc:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_saved_receipt_invalid"
        ) from exc
    replayed = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=model_outcome,
        candidate=candidate,
        projection=projection,
    )
    if canonical != replayed:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_external_replay_mismatch"
        )
    return canonical


def validate_packet_grounding_receipt_v2(
    *,
    receipt: PacketGroundingReceiptV2 | Mapping[str, Any],
    model_outcome: Mapping[str, Any],
    candidate: NativeCandidateDescriptor,
    projection: FrozenSourceProjectionV1,
) -> PacketGroundingReceiptV2:
    """Recompute a receipt from original inputs; coherent rehashes still fail."""

    try:
        canonical = _RECEIPT_ADAPTER.validate_python(receipt)
    except ValueError as exc:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_saved_receipt_invalid"
        ) from exc
    replayed = freeze_packet_grounding_receipt_v2(
        model_outcome=model_outcome,
        candidate=candidate,
        projection=projection,
    )
    if canonical != replayed:
        raise NativePacketGroundingV2Error(
            "packet_grounding_v2_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "ABSTENTION_RECEIPT_V2_VERSION",
    "CANDIDATE_BINDING_V2_VERSION",
    "COMPLETED_RECEIPT_V2_VERSION",
    "EFFECT_FORMAT_ALIAS_POLICY_V2",
    "EFFECT_FORMAT_ALIAS_POLICY_V2_SHA256",
    "EFFECT_FORMAT_RECEIPT_V2_VERSION",
    "MODEL_OUTCOME_V2_VERSION",
    "NORMALIZATION_POLICY_V2",
    "NORMALIZATION_POLICY_V2_SHA256",
    "PACKET_GROUNDING_V2_VERSION",
    "PASSAGE_CANDIDATE_BINDING_V2_VERSION",
    "SCHEMA_BUNDLE_V2_VERSION",
    "NativePacketGroundingV2Error",
    "PacketCandidateBindingV2",
    "PacketEffectFormatGroundingReceiptV2",
    "PacketGroundingAbstentionReceiptV2",
    "PacketGroundingCompletedReceiptV2",
    "PacketGroundingModelAbstentionV2",
    "PacketGroundingModelCompletedV2",
    "PacketGroundingReceiptV2",
    "PacketGroundingSchemaBundleV2",
    "PacketPassageCandidateBindingV2",
    "freeze_packet_candidate_binding_v2",
    "freeze_packet_grounding_receipt_v2",
    "freeze_packet_grounding_schema_bundle_v2",
    "freeze_passage_packet_candidate_binding_v2",
    "freeze_passage_packet_grounding_receipt_v2",
    "validate_packet_grounding_receipt_v2",
    "validate_passage_packet_grounding_receipt_v2",
]
