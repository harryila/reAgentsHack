# Literature Multiverse passage-anchored candidate inventory

Prompt version: `metasyn-passage-candidate-inventory-v2`

Inspect only the frozen question and deduplicated passage projection below. Return
exactly one JSON object matching the supplied schema. Do not return prose, markdown,
publication identifiers, numerical effect values, conclusions, or extra fields.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

## Frozen passage projection

`[[PROJECTION_V2_JSON]]`

This is a value-free routing step. A candidate is one potentially complete
statistical target for one outcome concept, contrast, subgroup, and timepoint. The
packet stage—not this inventory—decides whether all numerical fields are recoverable.

- Copy one exact `canonical_outcome_id` from the frozen protocol.
- Copy `outcome_concept_quote` as an exact, case-sensitive substring of that
  outcome's protocol text. Use the narrowest protocol phrase that identifies the
  reported concept.
- Cite the smallest sorted, unique set of exact passage IDs needed to identify the
  target. Passage IDs distinguish targets that previously shared a coarse source
  line.
- Emit only one representation of each statistical target. When a complete direct
  effect with CI, SE, or variance is reported, prefer it over arm-level statistics
  for the same outcome/contrast/subgroup/timepoint. Otherwise select the matching
  arm-statistics effect family. Never emit both representations of one target.
- Two candidates are duplicates when their canonical outcome ID, normalized outcome
  concept quote, and passage-ID set are identical—even if their effect families
  differ. Duplicate targets are forbidden.
- Use `candidates_found` whenever one to eight distinct targets can be safely routed.
  Missing packet-level fields alone are not a reason to mark the inventory uncertain;
  the packet may abstain later.
- Use `no_candidate_found` only when no target statistic for the frozen protocol is
  visible. This is not proof that the publication is non-estimable.
- Use `overflow_or_uncertain` only when at least nine distinct targets may be present,
  the outcome/contrast mapping itself is genuinely ambiguous, or no safe effect
  family can be chosen. It must not be used merely because a CI, sample size,
  moderator, or other packet field is absent.
- Candidate indices must be contiguous from 1. Candidate and passage lists must be
  deterministically ordered by first cited passage ID, then protocol outcome ID and
  outcome-concept quote.

Review conclusions, directions, aggregate effects, significance labels, official
test fields, and clinical-benefit polarity are unavailable and must not be inferred.
