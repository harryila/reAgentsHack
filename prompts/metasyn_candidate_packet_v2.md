# Literature Multiverse passage-anchored candidate packet

Prompt version: `metasyn-passage-candidate-packet-v2`

Extract exactly the one frozen candidate bound below. Inspect only the frozen
question/protocol and the candidate's exact passage surface. Return exactly one JSON
object matching the supplied candidate-specific schema. Do not return prose,
markdown, publication identifiers, filesystem paths, review conclusions, or fields
outside the schema.

## Frozen extraction question and protocol

`[[QUESTION_SURFACE_JSON]]`

## Frozen candidate binding

`[[CANDIDATE_BINDING_JSON]]`

## Full exposed projection for grounded identities

`[[PROJECTION_V2_JSON]]`

## Exact candidate passage surface

`[[CANDIDATE_PASSAGE_SURFACE_JSON]]`

The `candidate_binding_sha256` is immutable. Copy that exact hash into the response.
Never add, remove, rename, merge, or replace the candidate or its passage IDs. Return
either one `completed` response or the closed `unable_to_complete` branch.

- Preserve the frozen intervention/exposure-minus-comparator orientation. Raw
  positive-direction semantics are measurement semantics only; they are not clinical
  benefit, support, contradiction, or review-conclusion labels.
- Use only the frozen canonical outcome and outcome-concept quote. Do not substitute
  a different endpoint, subgroup, contrast, timepoint, or effect family.
- `evidence_quote` must be an exact, case-sensitive substring of exactly one passage
  listed in the candidate passage surface and must contain every emitted numerical
  token. Other passages in the full projection may not support `evidence_quote`. Do
  not calculate or report character or byte offsets; trusted local code derives them.
- Every scientific number must be copied as one verbatim JSON string in
  `verbatim_numeric_token`. Never compute, reverse, combine, round, repair, or infer a
  value. Use `percent_to_proportion` only for a verbatim percent confidence level or
  p-value; otherwise use `identity`.
- Emit only numeric field paths permitted by the supplied schema for the frozen effect
  family. Never invent uncertainty, sample sizes, units, equivalence margins,
  moderators, identities, or timepoints.
- Identity claims may copy exact, case-sensitive text visible anywhere in the full
  exposed projection because trusted grounding validates them against that full
  surface. Omit an optional identity rather than normalizing or guessing it.
- For an exact or ranged timepoint, copy its numeric token or tokens into the required
  numeric claims and copy any anchor/raw label exactly. Use `reported_text` only when a
  safe numeric timepoint is unavailable, and `not_reported` only when no timepoint is
  reported for this candidate.

If the frozen candidate cannot be completed under these rules, return
`unable_to_complete` with the closest closed reason. Never substitute another
candidate or emit a partial numerical claim.
