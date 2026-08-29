# Literature Multiverse bounded native candidate packet

Prompt version: `native-candidate-packet-v1`

Extract exactly one frozen candidate from the supplied source projection. Return exactly one
JSON object matching the supplied candidate-specific schema. Do not return prose, markdown,
publication identifiers, filesystem paths, extra candidates, or fields outside the schema.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

## Frozen candidate descriptor

`__FROZEN_CANDIDATE_JSON__`

The candidate descriptor is immutable routing context. Copy its candidate index, canonical
outcome, effect kind, and exact source line IDs. Never add, remove, rename, or replace a
candidate. Return either one `completed` packet or the value-free `unable_to_complete`
branch. A completed response must contain one study header, one cohort header, exactly the
two arms in the target contrast, one contrast, one finding, one effect payload of the
selected kind, one exact evidence span, and complete numeric-support receipts. Other
candidates are handled by separate calls.

- Use stable local keys. Repeated study, cohort, arm, or contrast keys across candidate calls
  must have byte-identical metadata; conflicts make the whole publication abstain.
- The treatment arm is the intervention/exposure named by the frozen target relation and the
  comparator arm is its concurrent control. Preserve that orientation.
- Use exactly the frozen canonical endpoint. Never substitute a raw measurement name.
- Encode a positive effect as the frozen beneficial direction. V1 permits only values
  reported verbatim by the source. Do not reverse, derive, calculate, or transform a direct
  estimate. Complete reported group summaries may be harmonized deterministically after
  extraction.
- Report only the selected effect representation. Never invent uncertainty, sample sizes,
  units, equivalence margins, moderator values, or timepoints.
- Every scientific number must be a bounded JSON string preserving the source numeric
  lexeme (for example `"0.5"`, `"20"`, or `".05"`), never a JSON number. For every emitted
  number, add exactly one `numeric_support` item with its exact field path, verbatim token,
  and zero-based start/end offsets within the evidence quote. The token at those offsets
  must parse to exactly the emitted value. `percent_to_proportion` is permitted only for a
  reported confidence level or p-value (for example source `95` to emitted `0.95`); it is
  forbidden for estimates and all other fields. Do not use thousands separators or
  inequality signs as numeric tokens in v1.
- Every moderator name must be prespecified. Missing moderators are omitted, not guessed.
- The evidence quote must be the shortest exact source substring supporting groups, outcome,
  timepoint, numerical value, and uncertainty. Use the exact source locator and exact frozen
  line IDs.

If the frozen candidate cannot be completed safely or every numeric leaf cannot be supported,
return `unable_to_complete` with the closest closed reason. Do not substitute a different
candidate or partial packet. An unable, invalid, or length-truncated packet blocks the whole
publication.
