# Literature Multiverse extraction prompt

Prompt version: `extraction-v3`

You are extracting the paper's own empirical results for one pre-registered scientific question.
Return exactly one JSON value matching the supplied schema. Do not add prose, markdown, citations,
paper identifiers, or fields outside that schema.

## Locked question

- Research question: `[[RESEARCH_QUESTION]]`
- Exposure/intervention: `[[EXPOSURE]]`
- Comparator: `[[COMPARATOR]]`
- Primary outcome family: `[[PRIMARY_OUTCOME_FAMILY]]`
- Included primary endpoints: `[[PRIMARY_ENDPOINTS_JSON]]`
- `increase`: `[[INCREASE_DEFINITION]]`
- `decrease`: `[[DECREASE_DEFINITION]]`
- `no_effect`: `[[NO_EFFECT_DEFINITION]]`

The primary family and endpoint list steer the downstream headline analysis — they are **not**
an eligibility filter and **not** a restriction on which findings to extract. A paper is
eligible when it tests the target relation on **any** outcome attributable to that relation
(functional, physiological, molecular, mitochondrial, redox, damage, inflammatory, or other).
Extract **every** qualifying finding regardless of family, and describe each outcome under the
paper's own outcome name; family assignment happens downstream.

## Eligibility

Include only the paper's own eligible empirical comparison under the locked rules below. Reviews,
editorials, protocols without results, cited studies, reference titles, background claims, and
results outside the target relation are not the paper's own findings.

`[[ELIGIBILITY_RULES_JSON]]`

If the paper is ineligible, return `eligible=false`, one concise exclusion reason, and an empty
`findings` array. If it is eligible but reports no usable target finding, return `eligible=true`,
`exclusion_reason=null`, and an empty array. Never invent a finding to avoid an empty result.

## Atomic finding rules

Create a separate finding for every distinct outcome, endpoint, timepoint, intervention dose,
comparator, or population stratum. Do not combine incompatible results into one row. Preserve the
paper's original dose, duration, timepoint, effect-size, and p-value wording in the corresponding
raw fields.

Direction always means the measured outcome relative to the locked comparator—never beneficial or
harmful. Use only `increase`, `no_effect`, `decrease`, `mixed`, or `unclear`. Use `mixed` only when an
otherwise atomic result contains genuinely incompatible directions; use `unclear` when the source
does not safely determine direction. The direction field must never be JSON null.

For every finding, copy the shortest verbatim passage that directly supports the reported result
and its exact `L<number>` source line or line range. The passage MUST come from the paper's
Results body or a results table — **never from the abstract, discussion, or conclusion**. When
the abstract summarizes a result, locate and quote the corresponding Results-section sentence or
table row instead; if the body genuinely never reports it, do not create the finding. The passage
must itself contain the between-group evidence for the direction you assign: the two groups'
values, a between-group p-value or effect size, or an explicit group×time interaction statement.
A quote showing only one group's change from its own baseline cannot support a between-group
direction. Do not use a reference title, bibliography entry, author speculation, or a cited study
as result evidence. Every nullable schema key must still be present with JSON null when
unavailable.

## Topic moderators

Populate only moderator keys present in the supplied schema, following these locked definitions:

`[[MODERATOR_RULES_JSON]]`

Do not convert animal mg/kg dosing into human per-day dosing. Preserve ambiguous and combination
doses in `dose_raw`; use null for a normalized value that the paper does not support.
