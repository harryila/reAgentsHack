# Literature Multiverse extraction prompt

Prompt version: `extraction-v3`

You are extracting the paper's own empirical results for one pre-registered scientific question.
Return exactly one JSON value matching the supplied schema. Do not add prose, markdown, citations,
paper identifiers, or fields outside that schema.

## Locked question

- Research question: `Under which experimental conditions does antioxidant supplementation (vitamin C and/or E) blunt, enhance, or leave unchanged adaptations to exercise training?`
- Exposure/intervention: `vitamin C and/or vitamin E supplementation taken during a structured exercise-training program`
- Comparator: `placebo or no-supplement control completing the same training program`
- Primary outcome family: `functional_adaptation`
- Included primary endpoints: `["muscle_strength", "muscle_mass", "aerobic_capacity", "endurance_performance", "sprint_or_power_performance"]`
- `increase`: `the training adaptation is larger in the supplemented group than in the control group`
- `decrease`: `the training adaptation is smaller in the supplemented group than in the control group`
- `no_effect`: `no detectable between-group difference in the training adaptation is reported`

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

`{"article_types": ["research-article"], "exclude": ["reviews, meta-analyses, editorials, commentaries, and protocols", "acute single-session studies without a training program", "studies without an antioxidant-versus-control contrast"], "exclude_article_types": ["review-article"], "include": ["human participants (animal and in-vitro studies are ineligible)", "controlled primary studies combining a structured exercise-training program with vitamin C and/or E supplementation", "a concurrent placebo or no-supplement comparison completing the same training", "direct measurement of at least one training-adaptation endpoint"]}`

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

`[{"allowed_values": ["below_500_mg_daily", "500_to_999_mg_daily", "1000_mg_daily_or_more", "not_reported_as_daily_dose"], "bins": null, "display_name": "Vitamin C dose", "kind": "paper_constant", "name": "vitc_dose_regime", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["endurance", "resistance", "sprint_interval", "concurrent_or_mixed"], "bins": null, "display_name": "Training modality", "kind": "paper_constant", "name": "training_modality", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["trained", "untrained", "mixed_or_unclear"], "bins": null, "display_name": "Training status", "kind": "paper_constant", "name": "training_status", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "fixed", "type": "categorical"}, {"allowed_values": ["younger_adult", "older_adult", "mixed_or_unclear"], "bins": null, "display_name": "Age group", "kind": "paper_constant", "name": "age_group", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "fixed", "type": "categorical"}, {"allowed_values": ["vitamin_c_only", "vitamin_e_only", "vitamin_c_and_e"], "bins": null, "display_name": "Supplement composition", "kind": "paper_constant", "name": "supplement_combo", "paper_summary": null, "permutation": "paper", "role": "tested", "source": "topic", "type": "categorical"}, {"allowed_values": ["healthy", "clinical_or_sarcopenic", "mixed_or_unclear"], "bins": null, "display_name": "Population state", "kind": "paper_constant", "name": "population_state", "paper_summary": null, "permutation": "paper", "role": "descriptive", "source": "fixed", "type": "categorical"}]`

Do not convert animal mg/kg dosing into human per-day dosing. Preserve ambiguous and combination
doses in `dose_raw`; use null for a normalized value that the paper does not support.
