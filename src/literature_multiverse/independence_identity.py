"""Canonical authority identities for split integrity and question independence.

Graph-local publication, study, and cohort labels are useful for joining one frozen
graph, but they are not globally comparable identities.  This module keeps that
distinction explicit.  ``join-only`` tokens may connect nodes inside a graph; only
canonical ``authority-v1`` tokens may determine a held-out split or certify that
complete questions do not overlap.

Raw identifiers belong in private custody artifacts.  Release and calibration
artifacts use :class:`StrongIndependenceIdentityV1`, which retains only domain-
separated token digests and rename-invariant connected-component digests.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel, normalize_doi

type AuthorityIdentityKind = Literal[
    "doi",
    "pmid",
    "trial_registry",
    "dataset",
    "globally_scoped_study",
    "globally_scoped_cohort",
]
type StrongIdentityKind = Literal[
    "doi",
    "pmid",
    "registration_id",
    "registry_id",
    "dataset_id",
    "globally_scoped_study_id",
    "globally_scoped_cohort_id",
]

AUTHORITY_TOKEN_VERSION = "authority-identity-token-v1"
AUTHORITY_TOKEN_DOMAIN = "literature-multiverse-authority-identity-v1"
COMPONENT_SIGNATURE_VERSION = "authority-identity-component-v1"
JOIN_ONLY_TOKEN_PREFIX = "join-only-v1"

_AUTHORITY_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?)"
    r"(?:/[a-z0-9][a-z0-9._-]*)*$"
)
_MACHINE_REASON_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9_.:-]*[a-z0-9])?$")

_AUTHORITY_ALIASES = {
    "actrn": "anzctr.org.au",
    "anzctr": "anzctr.org.au",
    "anzctr.org.au": "anzctr.org.au",
    "chictr": "chictr.org.cn",
    "chictr.org.cn": "chictr.org.cn",
    "clinicaltrials.gov": "clinicaltrials.gov",
    "datadryad.org": "datadryad.org",
    "dataverse": "dataverse.org",
    "dataverse.org": "dataverse.org",
    "doi": "doi.org",
    "doi.org": "doi.org",
    "drks": "drks.de",
    "drks.de": "drks.de",
    "dryad": "datadryad.org",
    "ebi.ac.uk": "ebi.ac.uk",
    "figshare": "figshare.com",
    "figshare.com": "figshare.com",
    "geo": "ncbi.nlm.nih.gov/geo",
    "isrctn": "isrctn.com",
    "isrctn.com": "isrctn.com",
    "ncbi.nlm.nih.gov/geo": "ncbi.nlm.nih.gov/geo",
    "ncbi.nlm.nih.gov/sra": "ncbi.nlm.nih.gov/sra",
    "nct": "clinicaltrials.gov",
    "openneuro": "openneuro.org",
    "openneuro.org": "openneuro.org",
    "osf": "osf.io",
    "osf.io": "osf.io",
    "proteomexchange": "proteomexchange.org",
    "proteomexchange.org": "proteomexchange.org",
    "pubmed": "pubmed.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov": "pubmed.ncbi.nlm.nih.gov",
    "sra": "ncbi.nlm.nih.gov/sra",
    "zenodo": "zenodo.org",
    "zenodo.org": "zenodo.org",
}


class IndependenceIdentityError(ValueError):
    """An identity cannot safely certify a split or independence claim."""


def _sha256(value: str, name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"independence_identity_invalid_sha256:{name}")
    return value


def _machine_reasons(values: Sequence[str]) -> list[str]:
    reasons = sorted(set(values))
    if any(_MACHINE_REASON_RE.fullmatch(value) is None for value in reasons):
        raise IndependenceIdentityError("independence_identity_reason_not_machine_readable")
    return reasons


def _semantic_kind(kind: StrongIdentityKind | AuthorityIdentityKind) -> AuthorityIdentityKind:
    aliases: dict[str, AuthorityIdentityKind] = {
        "dataset": "dataset",
        "dataset_id": "dataset",
        "doi": "doi",
        "globally_scoped_cohort": "globally_scoped_cohort",
        "globally_scoped_cohort_id": "globally_scoped_cohort",
        "globally_scoped_study": "globally_scoped_study",
        "globally_scoped_study_id": "globally_scoped_study",
        "pmid": "pmid",
        "registration_id": "trial_registry",
        "registry_id": "trial_registry",
        "trial_registry": "trial_registry",
    }
    try:
        return aliases[str(kind)]
    except KeyError as exc:
        raise IndependenceIdentityError(
            f"independence_identity_kind_unsupported:{kind}"
        ) from exc


def canonicalize_join_only_identity(namespace: str, value: str) -> str:
    """Canonicalize an in-graph join token that can never certify independence."""

    normalized_namespace = namespace.strip().casefold().replace(" ", "_")
    normalized_value = " ".join(value.strip().casefold().split())
    if (
        not normalized_namespace
        or not normalized_value
        or ":" in normalized_namespace
        or any(character in normalized_value for character in ("\x00", "\n", "\r"))
    ):
        raise IndependenceIdentityError("join_only_identity_invalid")
    return f"{JOIN_ONLY_TOKEN_PREFIX}:{normalized_namespace}:{normalized_value}"


def is_join_only_identity_token(value: str) -> bool:
    return value.startswith(f"{JOIN_ONLY_TOKEN_PREFIX}:")


def _canonical_authority(value: str) -> str:
    normalized = value.strip().casefold().removeprefix("https://").removeprefix(
        "http://"
    )
    normalized = normalized.rstrip("/")
    normalized = _AUTHORITY_ALIASES.get(normalized, normalized)
    if not normalized or _AUTHORITY_RE.fullmatch(normalized) is None:
        raise IndependenceIdentityError("authority_identity_authority_unrecognized")
    # A bare, unknown word is not a globally resolvable authority.  Known compact
    # aliases were expanded above; other authorities must be domain-like.
    if "." not in normalized and normalized not in set(_AUTHORITY_ALIASES.values()):
        raise IndependenceIdentityError("authority_identity_authority_unrecognized")
    return normalized


def _explicit_authority_value(value: str) -> tuple[str, str] | None:
    raw = value.strip()
    if "://" in raw:
        scheme, _, remainder = raw.partition("://")
        if scheme.casefold() not in {"http", "https"} or "/" not in remainder:
            return None
        authority, _, identifier = remainder.partition("/")
        return authority, identifier
    authority, separator, identifier = raw.partition(":")
    if not separator:
        return None
    return authority, identifier


def _inferred_authority(
    kind: AuthorityIdentityKind, value: str
) -> tuple[str, str] | None:
    compact = re.sub(r"[\s_-]+", "", value).casefold()
    if kind == "trial_registry":
        patterns: tuple[tuple[str, str, str], ...] = (
            (r"nct\d{8}", "clinicaltrials.gov", compact),
            (r"isrctn\d{8}", "isrctn.com", compact),
            (r"actrn\d+", "anzctr.org.au", compact),
            (r"chictr[a-z0-9]+", "chictr.org.cn", compact),
            (r"drks\d+", "drks.de", compact),
        )
    elif kind == "dataset":
        patterns = (
            (r"gse\d+", "ncbi.nlm.nih.gov/geo", compact),
            (r"(?:sra|srp|srr|srs|srx)\d+", "ncbi.nlm.nih.gov/sra", compact),
            (r"pxd\d+", "proteomexchange.org", compact),
            (r"emt[a-z0-9]+", "ebi.ac.uk", compact),
        )
    else:
        return None
    for pattern, authority, normalized in patterns:
        if re.fullmatch(pattern, compact):
            return authority, normalized
    return None


def _normalize_explicit_value(value: str) -> str:
    normalized = value.strip().casefold()
    if (
        not normalized
        or any(character.isspace() for character in normalized)
        or any(character in normalized for character in ("\x00", "\n", "\r"))
    ):
        raise IndependenceIdentityError("authority_identity_value_invalid")
    return normalized


def _normalize_authority_value(
    *,
    kind: AuthorityIdentityKind,
    authority: str,
    value: str,
) -> str:
    normalized = _normalize_explicit_value(value)
    if (kind, authority) in {
        ("trial_registry", "anzctr.org.au"),
        ("trial_registry", "chictr.org.cn"),
        ("trial_registry", "clinicaltrials.gov"),
        ("trial_registry", "drks.de"),
        ("trial_registry", "isrctn.com"),
        ("dataset", "ebi.ac.uk"),
        ("dataset", "ncbi.nlm.nih.gov/geo"),
        ("dataset", "ncbi.nlm.nih.gov/sra"),
        ("dataset", "proteomexchange.org"),
    }:
        normalized = re.sub(r"[\s_-]+", "", normalized)
    return normalized


def _hash_authority_identity_token_unchecked(canonical_token: str) -> str:
    return hashlib.sha256(
        f"{AUTHORITY_TOKEN_DOMAIN}\0{canonical_token}".encode()
    ).hexdigest()


class AuthorityIdentityClaimV1(ContractModel):
    """Private node-scoped raw claim presented to the canonicalizer."""

    claim_version: Literal["authority-identity-claim-v1"] = (
        "authority-identity-claim-v1"
    )
    node_id: Annotated[str, Field(min_length=1)]
    kind: StrongIdentityKind | AuthorityIdentityKind
    raw_value: Annotated[str, Field(min_length=1)]

    @field_validator("node_id", "raw_value")
    @classmethod
    def normalize_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("authority_identity_claim_value_empty")
        return normalized


class CanonicalAuthorityIdentityV1(ContractModel):
    """One strict, globally scoped authority identity."""

    identity_version: Literal["canonical-authority-identity-v1"] = (
        "canonical-authority-identity-v1"
    )
    kind: AuthorityIdentityKind
    authority: Annotated[str, Field(min_length=1)]
    normalized_value: Annotated[str, Field(min_length=1)]
    token: Annotated[str, Field(min_length=1)]
    token_sha256: str

    @field_validator("token_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "authority_token")

    @model_validator(mode="after")
    def validate_identity(self) -> CanonicalAuthorityIdentityV1:
        if self.authority != _canonical_authority(self.authority):
            raise ValueError("authority_identity_authority_not_canonical")
        if self.kind == "doi":
            if self.authority != "doi.org":
                raise ValueError("authority_identity_doi_authority_mismatch")
            try:
                expected_value = normalize_doi(self.normalized_value)
            except ValueError as exc:
                raise ValueError("authority_identity_doi_invalid") from exc
            if not expected_value.startswith("10.") or "/" not in expected_value:
                raise ValueError("authority_identity_doi_invalid")
        elif self.kind == "pmid":
            if (
                self.authority != "pubmed.ncbi.nlm.nih.gov"
                or not self.normalized_value.isdigit()
            ):
                raise ValueError("authority_identity_pmid_semantics_mismatch")
            expected_value = str(int(self.normalized_value))
        else:
            expected_value = _normalize_authority_value(
                kind=self.kind,
                authority=self.authority,
                value=self.normalized_value,
            )
        if self.normalized_value != expected_value:
            raise ValueError("authority_identity_value_not_canonical")
        expected_token = (
            f"{AUTHORITY_TOKEN_VERSION}:{self.kind}:{self.authority}:"
            f"{self.normalized_value}"
        )
        if self.token != expected_token:
            raise ValueError("authority_identity_token_mismatch")
        if self.token_sha256 != _hash_authority_identity_token_unchecked(self.token):
            raise ValueError("authority_identity_token_hash_mismatch")
        return self


def canonicalize_authority_identity(
    *,
    kind: StrongIdentityKind | AuthorityIdentityKind,
    value: str,
) -> CanonicalAuthorityIdentityV1:
    """Normalize one raw identity or fail when its authority is ambiguous."""

    semantic_kind = _semantic_kind(kind)
    raw = value.strip()
    if not raw:
        raise IndependenceIdentityError("authority_identity_value_empty")
    if semantic_kind == "doi":
        try:
            normalized_value = normalize_doi(raw)
        except ValueError as exc:
            raise IndependenceIdentityError("authority_identity_doi_invalid") from exc
        authority = "doi.org"
    elif semantic_kind == "pmid":
        normalized_value = raw.casefold().removeprefix("pmid:").strip()
        if not normalized_value.isdigit():
            raise IndependenceIdentityError("authority_identity_pmid_invalid")
        normalized_value = str(int(normalized_value))
        authority = "pubmed.ncbi.nlm.nih.gov"
    else:
        explicit = _explicit_authority_value(raw)
        inferred = None if explicit is not None else _inferred_authority(semantic_kind, raw)
        if explicit is None and inferred is None:
            raise IndependenceIdentityError("authority_identity_namespace_unrecognized")
        raw_authority, raw_identifier = explicit or inferred  # type: ignore[misc]
        authority = _canonical_authority(raw_authority)
        normalized_value = _normalize_authority_value(
            kind=semantic_kind,
            authority=authority,
            value=raw_identifier,
        )
    token = (
        f"{AUTHORITY_TOKEN_VERSION}:{semantic_kind}:{authority}:{normalized_value}"
    )
    return CanonicalAuthorityIdentityV1(
        kind=semantic_kind,
        authority=authority,
        normalized_value=normalized_value,
        token=token,
        token_sha256=_hash_authority_identity_token_unchecked(token),
    )


def parse_canonical_authority_identity(token: str) -> CanonicalAuthorityIdentityV1:
    """Parse and fully revalidate a canonical token; opaque digests are not accepted."""

    prefix, separator, remainder = token.partition(":")
    if not separator or prefix != AUTHORITY_TOKEN_VERSION:
        raise IndependenceIdentityError("authority_identity_token_version_invalid")
    raw_kind, separator, remainder = remainder.partition(":")
    if not separator:
        raise IndependenceIdentityError("authority_identity_token_invalid")
    authority, separator, normalized_value = remainder.partition(":")
    if not separator:
        raise IndependenceIdentityError("authority_identity_token_invalid")
    return CanonicalAuthorityIdentityV1(
        kind=_semantic_kind(raw_kind),  # type: ignore[arg-type]
        authority=authority,
        normalized_value=normalized_value,
        token=token,
        token_sha256=_hash_authority_identity_token_unchecked(token),
    )


def hash_authority_identity_token(canonical_token: str) -> str:
    """Return a content-silent, domain-separated digest of one canonical token."""

    return parse_canonical_authority_identity(canonical_token).token_sha256


class AuthorityIdentityBindingV1(ContractModel):
    token: Annotated[str, Field(min_length=1)]
    token_sha256: str
    node_ids: Annotated[list[str], Field(min_length=1)]

    @field_validator("token_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "authority_binding_token")

    @field_validator("node_ids")
    @classmethod
    def validate_nodes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("authority_identity_binding_nodes_invalid")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> AuthorityIdentityBindingV1:
        parsed = parse_canonical_authority_identity(self.token)
        if self.token_sha256 != parsed.token_sha256:
            raise ValueError("authority_identity_binding_hash_mismatch")
        return self


class AuthorityIdentityConflictV1(ContractModel):
    kind: AuthorityIdentityKind
    normalized_value: Annotated[str, Field(min_length=1)]
    authorities: Annotated[list[str], Field(min_length=2)]
    node_ids: Annotated[list[str], Field(min_length=1)]
    conflict_sha256: str

    @field_validator("authorities", "node_ids")
    @classmethod
    def validate_sorted_values(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError(f"authority_identity_conflict_values_invalid:{info.field_name}")
        return value

    @field_validator("conflict_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "authority_conflict")

    @model_validator(mode="after")
    def validate_conflict(self) -> AuthorityIdentityConflictV1:
        if any(authority != _canonical_authority(authority) for authority in self.authorities):
            raise ValueError("authority_identity_conflict_authority_not_canonical")
        payload = self.model_dump(mode="json", exclude={"conflict_sha256"})
        if hash_canonical(payload) != self.conflict_sha256:
            raise ValueError("authority_identity_conflict_hash_mismatch")
        return self


class AuthorityIdentityLedgerV1(ContractModel):
    """Private canonicalization ledger; unrecognized raw values survive only as hashes."""

    ledger_version: Literal["authority-identity-ledger-v1"] = (
        "authority-identity-ledger-v1"
    )
    identities: list[CanonicalAuthorityIdentityV1]
    bindings: list[AuthorityIdentityBindingV1]
    conflicts: list[AuthorityIdentityConflictV1]
    unrecognized_claim_sha256s: list[str]
    ledger_sha256: str

    @field_validator("unrecognized_claim_sha256s")
    @classmethod
    def validate_unrecognized_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("authority_identity_unrecognized_hashes_not_sorted")
        return [_sha256(item, "unrecognized_claim") for item in value]

    @model_validator(mode="after")
    def validate_ledger(self) -> AuthorityIdentityLedgerV1:
        tokens = [item.token for item in self.identities]
        binding_tokens = [item.token for item in self.bindings]
        if tokens != sorted(set(tokens)) or binding_tokens != tokens:
            raise ValueError("authority_identity_ledger_identity_binding_mismatch")
        conflict_keys = [
            (item.kind, item.normalized_value, item.authorities, item.node_ids)
            for item in self.conflicts
        ]
        if conflict_keys != sorted(conflict_keys):
            raise ValueError("authority_identity_ledger_conflicts_not_sorted")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if hash_canonical(payload) != self.ledger_sha256:
            raise ValueError("authority_identity_ledger_hash_mismatch")
        return self

    @property
    def authority_identity_set_sha256(self) -> str:
        return authority_identity_set_sha256(
            [identity.token for identity in self.identities]
        )


def _unrecognized_claim_sha256(claim: AuthorityIdentityClaimV1) -> str:
    return hashlib.sha256(
        (
            f"{AUTHORITY_TOKEN_DOMAIN}\0unrecognized\0{claim.kind}\0"
            f"{claim.raw_value.strip().casefold()}"
        ).encode()
    ).hexdigest()


def canonicalize_authority_identity_claims(
    claims: Sequence[AuthorityIdentityClaimV1],
) -> AuthorityIdentityLedgerV1:
    """Canonicalize node-scoped claims, collapsing aliases and exposing conflicts."""

    parsed_claims = [
        AuthorityIdentityClaimV1.model_validate(claim.model_dump(mode="json"))
        for claim in claims
    ]
    bindings: dict[str, set[str]] = defaultdict(set)
    identities: dict[str, CanonicalAuthorityIdentityV1] = {}
    unrecognized: list[str] = []
    by_semantic_value: dict[
        tuple[AuthorityIdentityKind, str], dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    for claim in parsed_claims:
        try:
            identity = canonicalize_authority_identity(
                kind=claim.kind,
                value=claim.raw_value,
            )
        except IndependenceIdentityError:
            unrecognized.append(_unrecognized_claim_sha256(claim))
            continue
        identities[identity.token] = identity
        bindings[identity.token].add(claim.node_id)
        by_semantic_value[(identity.kind, identity.normalized_value)][
            identity.authority
        ].add(claim.node_id)
    conflicts: list[AuthorityIdentityConflictV1] = []
    for (kind, normalized_value), authority_nodes in by_semantic_value.items():
        if len(authority_nodes) < 2:
            continue
        conflict_payload: dict[str, Any] = {
            "kind": kind,
            "normalized_value": normalized_value,
            "authorities": sorted(authority_nodes),
            "node_ids": sorted(
                {node for nodes in authority_nodes.values() for node in nodes}
            ),
        }
        conflicts.append(
            AuthorityIdentityConflictV1.model_validate(
                {
                    **conflict_payload,
                    "conflict_sha256": hash_canonical(conflict_payload),
                }
            )
        )
    conflict_rows = sorted(
        conflicts,
        key=lambda item: (item.kind, item.normalized_value, item.authorities, item.node_ids),
    )
    identity_rows = [identities[token] for token in sorted(identities)]
    binding_rows = [
        AuthorityIdentityBindingV1(
            token=token,
            token_sha256=identities[token].token_sha256,
            node_ids=sorted(bindings[token]),
        )
        for token in sorted(identities)
    ]
    payload: dict[str, Any] = {
        "ledger_version": "authority-identity-ledger-v1",
        "identities": identity_rows,
        "bindings": binding_rows,
        "conflicts": conflict_rows,
        "unrecognized_claim_sha256s": sorted(set(unrecognized)),
    }
    return AuthorityIdentityLedgerV1.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def authority_identity_set_sha256(canonical_tokens: Sequence[str]) -> str:
    """Rename-invariant signature over recomputed authority-token digests."""

    identities = [parse_canonical_authority_identity(token) for token in canonical_tokens]
    token_hashes = sorted({identity.token_sha256 for identity in identities})
    if not token_hashes:
        raise IndependenceIdentityError("authority_identity_set_empty")
    return hash_canonical(
        {
            "component_version": COMPONENT_SIGNATURE_VERSION,
            "authority_identity_token_sha256s": token_hashes,
        }
    )


class StrongIndependenceIdentityV1(ContractModel):
    """Content-silent strong identity closure for one complete review question."""

    identity_version: Literal["strong-independence-identity-v1"] = (
        "strong-independence-identity-v1"
    )
    verification_status: Literal["verified", "unverified"]
    derivation_basis: Literal[
        "canonical authority identities and rename-invariant connected components; "
        "no raw identifiers, free text, or graph-local IDs"
    ] = (
        "canonical authority identities and rename-invariant connected components; "
        "no raw identifiers, free text, or graph-local IDs"
    )
    strong_identity_token_sha256s: list[str]
    strong_component_sha256s: list[str]
    unverification_reasons: list[str]
    identity_sha256: str

    @field_validator("strong_identity_token_sha256s", "strong_component_sha256s")
    @classmethod
    def validate_hashes(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"strong_identity_hashes_not_sorted:{info.field_name}")
        return [_sha256(item, info.field_name) for item in value]

    @field_validator("unverification_reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        return _machine_reasons(value)

    @field_validator("identity_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "strong_independence_identity")

    @model_validator(mode="after")
    def validate_identity(self) -> StrongIndependenceIdentityV1:
        verified = bool(
            self.strong_identity_token_sha256s
            and self.strong_component_sha256s
            and not self.unverification_reasons
        )
        if (self.verification_status == "verified") != verified:
            raise ValueError("strong_independence_verification_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"identity_sha256"})
        if hash_canonical(payload) != self.identity_sha256:
            raise ValueError("strong_independence_identity_hash_mismatch")
        return self

    @property
    def independence_identity_sha256(self) -> str:
        """Compatibility name used by the adaptive-v2 contract."""

        return self.identity_sha256


def freeze_strong_independence_identity(
    *,
    strong_components: Sequence[Sequence[str | CanonicalAuthorityIdentityV1]],
    reasons: Sequence[str] = (),
) -> StrongIndependenceIdentityV1:
    """Recompute all token/component hashes and discard every raw identifier."""

    component_hashes: list[str] = []
    token_hashes: set[str] = set()
    for component in strong_components:
        identities = [
            (
                parse_canonical_authority_identity(item)
                if isinstance(item, str)
                else CanonicalAuthorityIdentityV1.model_validate(
                    item.model_dump(mode="json")
                )
            )
            for item in component
        ]
        tokens = sorted({identity.token for identity in identities})
        if not tokens:
            raise IndependenceIdentityError("strong_identity_component_empty")
        component_token_hashes = {identity.token_sha256 for identity in identities}
        if token_hashes & component_token_hashes:
            raise IndependenceIdentityError(
                "strong_identity_token_spans_multiple_components"
            )
        component_hashes.append(authority_identity_set_sha256(tokens))
        token_hashes.update(component_token_hashes)
    normalized_reasons = _machine_reasons(reasons)
    if not component_hashes and not normalized_reasons:
        normalized_reasons = ["authority_identity_components_absent"]
    payload: dict[str, Any] = {
        "identity_version": "strong-independence-identity-v1",
        "verification_status": (
            "verified" if component_hashes and not normalized_reasons else "unverified"
        ),
        "derivation_basis": (
            "canonical authority identities and rename-invariant connected components; "
            "no raw identifiers, free text, or graph-local IDs"
        ),
        "strong_identity_token_sha256s": sorted(token_hashes),
        "strong_component_sha256s": sorted(set(component_hashes)),
        "unverification_reasons": normalized_reasons,
    }
    return StrongIndependenceIdentityV1.model_validate(
        {**payload, "identity_sha256": hash_canonical(payload)}
    )


def freeze_strong_independence_identity_from_component_tokens(
    *,
    component_token_sets: Sequence[Sequence[str]],
    reasons: Sequence[str] = (),
) -> StrongIndependenceIdentityV1:
    """Project native components by parsing tokens, never trusting supplied digests."""

    return freeze_strong_independence_identity(
        strong_components=component_token_sets,
        reasons=reasons,
    )


def freeze_strong_independence_identity_from_ledgers(
    *,
    component_ledgers: Sequence[AuthorityIdentityLedgerV1],
    reasons: Sequence[str] = (),
) -> StrongIndependenceIdentityV1:
    """Create a release identity while converting ambiguity into fail-closed reasons."""

    normalized_reasons = list(reasons)
    components: list[list[CanonicalAuthorityIdentityV1]] = []
    seen_token_hashes: set[str] = set()
    for index, ledger in enumerate(component_ledgers):
        try:
            parsed = AuthorityIdentityLedgerV1.model_validate(
                ledger.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise IndependenceIdentityError(
                "strong_identity_component_ledger_invalid"
            ) from exc
        if parsed.conflicts:
            normalized_reasons.extend(
                f"authority_identity_conflict:{row.conflict_sha256}"
                for row in parsed.conflicts
            )
        if parsed.unrecognized_claim_sha256s:
            normalized_reasons.extend(
                f"authority_identity_unrecognized:{digest}"
                for digest in parsed.unrecognized_claim_sha256s
            )
        if not parsed.identities:
            normalized_reasons.append(f"authority_identity_component_empty:{index}")
            continue
        current_hashes = {row.token_sha256 for row in parsed.identities}
        if current_hashes & seen_token_hashes:
            normalized_reasons.append("authority_identity_token_spans_components")
            # Preserve a single deterministic component occurrence so the output
            # remains structurally valid while its status is unverified.
            unique_rows = [
                row for row in parsed.identities if row.token_sha256 not in seen_token_hashes
            ]
            if unique_rows:
                components.append(unique_rows)
                seen_token_hashes.update(row.token_sha256 for row in unique_rows)
            continue
        components.append(list(parsed.identities))
        seen_token_hashes.update(current_hashes)
    return freeze_strong_independence_identity(
        strong_components=components,
        reasons=normalized_reasons,
    )


__all__ = [
    "AUTHORITY_TOKEN_DOMAIN",
    "AUTHORITY_TOKEN_VERSION",
    "AuthorityIdentityBindingV1",
    "AuthorityIdentityClaimV1",
    "AuthorityIdentityConflictV1",
    "AuthorityIdentityKind",
    "AuthorityIdentityLedgerV1",
    "CanonicalAuthorityIdentityV1",
    "IndependenceIdentityError",
    "StrongIdentityKind",
    "StrongIndependenceIdentityV1",
    "authority_identity_set_sha256",
    "canonicalize_authority_identity",
    "canonicalize_authority_identity_claims",
    "canonicalize_join_only_identity",
    "freeze_strong_independence_identity",
    "freeze_strong_independence_identity_from_component_tokens",
    "freeze_strong_independence_identity_from_ledgers",
    "hash_authority_identity_token",
    "is_join_only_identity_token",
    "parse_canonical_authority_identity",
]
