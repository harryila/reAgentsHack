# Literature Multiverse MetaSyn bounded candidate packet

Prompt version: `metasyn-candidate-packet-v1`

Extract exactly one frozen candidate from the supplied source projection. Return exactly one
JSON object matching the supplied candidate-specific schema. Do not return prose, markdown,
publication identifiers, filesystem paths, extra candidates, or fields outside the schema.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

## Frozen candidate descriptor

`__FROZEN_CANDIDATE_JSON__`

The candidate descriptor is immutable. Copy its candidate index, canonical outcome ID,
effect kind, and exact source line IDs. Never add, remove, rename, or replace a candidate.
Return either one `completed` packet or the value-free `unable_to_complete` branch.

- Preserve the frozen intervention/exposure-versus-comparator orientation. The frozen
  `positive_direction_means` describes raw outcome/effect orientation only. It is not a
  clinical-benefit label. Never infer beneficial, harmful, supported, contradicted, or review
  conclusion direction.
- Use the exact canonical outcome ID. The untruncated outcome text remains frozen in the
  question mapping; do not emit it in place of the ID.
- Use stable local study/cohort/arm/contrast keys. Metadata repeated across candidate packets
  must be byte-identical; any conflict abstains the whole publication.
- Emit only values reported verbatim in the cited exact projection. Never invent uncertainty,
  sample size, unit, equivalence, moderator, timepoint, or a derived effect.
- Every scientific number must be a bounded JSON string and have exactly one numeric-support
  receipt with exact token and quote offsets. `percent_to_proportion` is permitted only for a
  reported confidence level or p-value.
- Evidence must use the exact frozen source locator, canonical exposed section enum, and exact
  candidate line IDs. The quote must occur uniquely in those cited frozen passages.
- Title/Abstract-only source packets remain diagnostic even if structurally complete.

If the candidate cannot be completed safely, return `unable_to_complete`. One unable, invalid,
missing, conflicting, or ungrounded packet abstains the whole publication; no subset survives.
