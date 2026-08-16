from __future__ import annotations

from copy import deepcopy

import pytest

from literature_multiverse.cohort import (
    CohortContractError,
    canonical_primary_rows,
    cohort_sha256,
)


def test_nested_and_flattened_moderators_have_identical_complete_cohort_hash(
    finding_payload,
) -> None:
    nested = deepcopy(finding_payload)
    flat = deepcopy(finding_payload)
    moderators = flat.pop("moderators")
    flat.update({f"mod__{name}": value for name, value in moderators.items()})
    assert cohort_sha256([nested]) == cohort_sha256([flat])
    canonical = canonical_primary_rows([flat])[0]
    assert canonical["moderators"] == nested["moderators"]

    changed_quote = deepcopy(flat)
    changed_quote["evidence_quote"] = "A different source passage."
    assert cohort_sha256([changed_quote]) != cohort_sha256([flat])
    changed_moderator = deepcopy(flat)
    changed_moderator["mod__dose_regime"] = "high"
    assert cohort_sha256([changed_moderator]) != cohort_sha256([flat])


def test_cohort_hash_is_order_independent_but_rejects_duplicate_ids(finding_payload) -> None:
    first = deepcopy(finding_payload)
    second = deepcopy(finding_payload)
    second["finding_id"] = f"{second['finding_id']}-other"
    second["array_position"] = 1
    assert cohort_sha256([first, second]) == cohort_sha256([second, first])
    with pytest.raises(CohortContractError, match="not_unique"):
        cohort_sha256([first, first])


def test_flat_and_nested_moderator_conflict_is_rejected(finding_payload) -> None:
    row = deepcopy(finding_payload)
    row["mod__dose_regime"] = "high"
    with pytest.raises(CohortContractError, match="representation_conflict"):
        cohort_sha256([row])
