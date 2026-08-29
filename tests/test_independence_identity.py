from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.independence_identity import (
    AuthorityIdentityClaimV1,
    CanonicalAuthorityIdentityV1,
    IndependenceIdentityError,
    StrongIndependenceIdentityV1,
    authority_identity_set_sha256,
    canonicalize_authority_identity,
    canonicalize_authority_identity_claims,
    canonicalize_join_only_identity,
    freeze_strong_independence_identity,
    freeze_strong_independence_identity_from_ledgers,
    hash_authority_identity_token,
    parse_canonical_authority_identity,
)


def _claim(node_id: str, kind: str, raw_value: str) -> AuthorityIdentityClaimV1:
    return AuthorityIdentityClaimV1(
        node_id=node_id,
        kind=kind,
        raw_value=raw_value,
    )


def test_doi_pmid_and_recognized_registry_aliases_canonicalize_exactly() -> None:
    doi = canonicalize_authority_identity(
        kind="doi", value="HTTPS://DOI.ORG/10.1000/ABC"
    )
    pmid = canonicalize_authority_identity(kind="pmid", value="PMID:000123")
    inferred = canonicalize_authority_identity(
        kind="registration_id", value="NCT-01234567"
    )
    explicit = canonicalize_authority_identity(
        kind="registry_id", value="clinicaltrials.gov:NCT_01234567"
    )

    assert doi.token.endswith(":doi:doi.org:10.1000/abc")
    assert pmid.token.endswith(":pmid:pubmed.ncbi.nlm.nih.gov:123")
    assert inferred == explicit
    assert parse_canonical_authority_identity(doi.token) == doi
    assert hash_authority_identity_token(doi.token) == doi.token_sha256


def test_graph_local_and_unknown_identifiers_never_become_authority_tokens() -> None:
    join_token = canonicalize_join_only_identity("study", "Local Study 7")
    assert join_token == "join-only-v1:study:local study 7"
    with pytest.raises(
        IndependenceIdentityError,
        match="authority_identity_namespace_unrecognized",
    ):
        canonicalize_authority_identity(
            kind="globally_scoped_study_id",
            value="local-study-7",
        )
    ledger = canonicalize_authority_identity_claims(
        [_claim("study:7", "globally_scoped_study_id", "local-study-7")]
    )
    assert ledger.identities == []
    assert len(ledger.unrecognized_claim_sha256s) == 1
    assert "local-study-7" not in ledger.model_dump_json()


def test_aliases_collapse_but_conflicting_authorities_are_explicit() -> None:
    ledger = canonicalize_authority_identity_claims(
        [
            _claim("publication:a", "registry_id", "NCT01234567"),
            _claim(
                "publication:b",
                "registration_id",
                "clinicaltrials.gov:NCT-01234567",
            ),
            _claim("publication:c", "registration_id", "isrctn.com:nct01234567"),
        ]
    )

    assert len(ledger.identities) == 2
    clinical = next(
        row for row in ledger.bindings if row.token.endswith(":nct01234567")
        and ":clinicaltrials.gov:" in row.token
    )
    assert clinical.node_ids == ["publication:a", "publication:b"]
    assert len(ledger.conflicts) == 1
    assert ledger.conflicts[0].authorities == [
        "clinicaltrials.gov",
        "isrctn.com",
    ]


def test_component_signature_is_rename_invariant_and_content_silent() -> None:
    first = canonicalize_authority_identity(kind="doi", value="10.1000/a")
    second = canonicalize_authority_identity(kind="pmid", value="42")
    forward = authority_identity_set_sha256([first.token, second.token])
    reverse = authority_identity_set_sha256([second.token, first.token])
    assert forward == reverse

    frozen = freeze_strong_independence_identity(
        strong_components=[[first, second]]
    )
    assert frozen.verification_status == "verified"
    assert frozen.strong_component_sha256s == [forward]
    serialized = frozen.model_dump_json()
    assert "10.1000/a" not in serialized
    assert ":42" not in serialized


def test_conflict_and_unrecognized_claims_make_release_identity_unverified() -> None:
    ledger = canonicalize_authority_identity_claims(
        [
            _claim("p:a", "registry_id", "clinicaltrials.gov:shared1"),
            _claim("p:b", "registry_id", "isrctn.com:shared1"),
            _claim("s:a", "globally_scoped_study_id", "local-only"),
        ]
    )
    frozen = freeze_strong_independence_identity_from_ledgers(
        component_ledgers=[ledger]
    )

    assert frozen.verification_status == "unverified"
    assert any(
        reason.startswith("authority_identity_conflict:")
        for reason in frozen.unverification_reasons
    )
    assert any(
        reason.startswith("authority_identity_unrecognized:")
        for reason in frozen.unverification_reasons
    )


def test_same_authority_token_cannot_certify_two_components() -> None:
    identity = canonicalize_authority_identity(kind="doi", value="10.1000/shared")
    with pytest.raises(
        IndependenceIdentityError,
        match="strong_identity_token_spans_multiple_components",
    ):
        freeze_strong_independence_identity(
            strong_components=[[identity], [identity]]
        )


def test_canonical_and_release_identity_hashes_fail_closed_on_tamper() -> None:
    identity = canonicalize_authority_identity(kind="doi", value="10.1000/a")
    identity_payload = deepcopy(identity.model_dump(mode="json"))
    identity_payload["normalized_value"] = "10.1000/b"
    with pytest.raises(ValidationError):
        CanonicalAuthorityIdentityV1.model_validate(identity_payload)

    frozen = freeze_strong_independence_identity(strong_components=[[identity]])
    frozen_payload = deepcopy(frozen.model_dump(mode="json"))
    frozen_payload["verification_status"] = "unverified"
    with pytest.raises(ValidationError):
        StrongIndependenceIdentityV1.model_validate(frozen_payload)
