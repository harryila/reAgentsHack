# Literature Multiverse quote-to-direction verification

Prompt version: `quote-verification-v1`

Independently assess whether each supplied source passage supports its proposed measured direction
under the locked outcome and comparator definitions. This is not a relevance screen and not a
benefit/harm judgment. Return exactly the supplied strict JSON schema and echo every `finding_id`
once, in input order. Never add or omit an ID.

Allowed `model_status` values:

- `agree`: the passage directly supports the proposed direction relative to the comparator;
- `disagree`: the passage supports a different direction; or
- `unverifiable`: the passage is insufficient, ambiguous, internally incompatible, or not the
  paper's own result.

Give a short rationale grounded only in the provided passage and definitions. Do not change the
human `adjudication` field; code initializes it to `none`, and only a human may later set
`accept` or `reject`.

Locked definitions:

`[[DIRECTION_DEFINITIONS_JSON]]`

Findings:

`[[FINDINGS_JSON]]`
