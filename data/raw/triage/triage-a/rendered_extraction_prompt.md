# Literature Multiverse extraction prompt

Prompt version: `extraction-v1`

You are extracting the paper's own empirical results for one pre-registered scientific question.
Return exactly one JSON value matching the supplied schema. Do not add prose, markdown, citations,
paper identifiers, or fields outside that schema.

## Locked question

- Research question: `Under which conditions does time-restricted eating change cardiometabolic outcomes relative to an unrestricted eating window?`
- Exposure/intervention: `time-restricted eating (a deliberately restricted daily eating window)`
- Comparator: `unrestricted eating window or a non-time-restricted control arm`
- Primary outcome family: `None`
- Included primary endpoints: `[]`
- `increase`: `the measured outcome value is higher with time-restricted eating than in the comparison arm`
- `decrease`: `the measured outcome value is lower with time-restricted eating than in the comparison arm`
- `no_effect`: `no detectable between-group difference in the outcome is reported`

## Eligibility

Include only the paper's own eligible empirical comparison under the locked rules below. Reviews,
editorials, protocols without results, cited studies, reference titles, background claims, and
results outside the target relation are not the paper's own findings.

`{"article_types": ["research-article"], "exclude": ["reviews, meta-analyses, editorials, commentaries, and protocols", "observational religious-fasting studies without a controlled comparison", "studies without an eating-window contrast"], "exclude_article_types": ["review-article"], "include": ["randomized or controlled primary trials of time-restricted eating with a concurrent comparison arm", "direct measurement of at least one cardiometabolic outcome"]}`

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
and its exact `L<number>` source line or line range. Prefer Results/table-result text. Do not use a
reference title, bibliography entry, author speculation, or a cited study as result evidence.
Every nullable schema key must still be present with JSON null when unavailable.

## Topic moderators

Populate only moderator keys present in the supplied schema, following these locked definitions:

`[{"allowed_values": ["early", "midday_or_late", "self_selected", "not_reported"], "bins": null, "display_name": "Eating-window timing", "kind": "paper_constant", "name": "window_timing", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["intended_isocaloric", "combined_with_energy_restriction", "ad_libitum"], "bins": null, "display_name": "Caloric design", "kind": "paper_constant", "name": "caloric_design", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["healthy", "overweight_or_obese", "prediabetes_or_type2_diabetes", "metabolic_syndrome"], "bins": null, "display_name": "Baseline metabolic state", "kind": "paper_constant", "name": "baseline_state", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["four_to_six_hours", "eight_hours", "ten_hours_or_more", "not_reported"], "bins": null, "display_name": "Eating-window length", "kind": "paper_constant", "name": "window_length", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["under_eight_weeks", "eight_to_sixteen_weeks", "over_sixteen_weeks"], "bins": null, "display_name": "Intervention duration", "kind": "paper_constant", "name": "intervention_duration", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "fixed", "type": "categorical"}, {"allowed_values": ["younger_adult", "older_adult", "mixed_or_unclear"], "bins": null, "display_name": "Age group", "kind": "paper_constant", "name": "age_group", "paper_summary": null, "permutation": "paper", "role": "descriptive", "source": "fixed", "type": "categorical"}]`

Do not convert animal mg/kg dosing into human per-day dosing. Preserve ambiguous and combination
doses in `dose_raw`; use null for a normalized value that the paper does not support.
