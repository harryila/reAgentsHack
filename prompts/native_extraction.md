# Literature Multiverse native numerical evidence extraction

Prompt version: `native-extraction-v3`

Extract the publication's own empirical evidence for the frozen scientific question below.
Return exactly one JSON object matching the supplied schema. Do not return prose, markdown,
publication identifiers, filesystem paths, or fields outside the schema.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

## Schema boundary

The frozen question above is input context, not output. Never copy its keys or maps into
`studies`: in particular, do not output `eligibility`, `moderators`, `outcomes`, `endpoint_map`,
`family_map`, `included_primary_endpoints`, `primary_family`, `research_question`, or
`target_relation` anywhere in the response.

The top-level object has exactly these six keys:
`extraction_schema_version`, `status`, `studies`, `non_estimability_reason`,
`non_estimability_detail`, and `warnings`. If no complete, safely grounded numerical target
contrast is present, return exactly this shape (with the most specific allowed reason substituted
when appropriate):

```json
{
  "extraction_schema_version": "native-publication-extraction-v1",
  "status": "non_estimable",
  "studies": [],
  "non_estimability_reason": "numerical_result_absent",
  "non_estimability_detail": null,
  "warnings": []
}
```

Do not put partial study descriptions into a non-estimable response. Do not invent fields to
explain a decision. Use the full estimable schema only when every required study, cohort, arm,
contrast, finding, numerical-effect, and evidence field can be completed from the exposed source.

## Scientific unit rules

- Separate publication, study, cohort, arms, contrast, outcome, and timepoint.
- Use stable local keys only within this response. The caller injects global identities.
- Split distinct cohorts, population strata, doses, contrasts, outcomes, and timepoints.
- Copy reported registry and dataset identifiers exactly. Never invent one.
- Do not claim that cohorts in different publications are identical; reconciliation occurs later.

## Numerical effect rules

- Set `outcome_name` to exactly one canonical endpoint named in
  `outcomes.included_primary_endpoints`, applying the frozen `outcomes.endpoint_map`.
  Do not emit a raw measurement label as `outcome_name`; if the mapping is ambiguous,
  do not guess.
- For the target relation, the treatment arm is the vitamin C/E supplement arm and the
  comparator arm is placebo or no supplement under the same training program. Preserve
  that orientation in every contrast and numerical effect.
- Define `positive_direction_means` and encode the numerical effect so a positive value
  means a larger beneficial training adaptation in the supplemented arm. Respect any
  frozen endpoint direction override. If the paper reports the inverse contrast or a
  lower-is-better measurement, reverse it only by transparent algebra, set
  `extraction_method=computed_from_reported_statistics`, and retain a quote supporting
  the reported inputs; otherwise return the finding as non-estimable. Never silently
  reverse a reported contrast or outcome.
- Represent the between-group difference in training adaptation or change requested by
  the protocol, not an unadjusted post-training difference when that would answer a
  different estimand.
- Use an explicit reported follow-up or training-completion timepoint whenever available.
  Never invent one; `not_reported` remains valid but may prevent compatible synthesis.
- Prefer a directly reported effect estimate with exactly one uncertainty source: standard error,
  variance, or both confidence limits.
- Otherwise report complete continuous group summaries or complete binary event counts.
- Set the exact reported effect format. Never mix raw, standardized, ratio, and log-ratio scales.
- Do not infer an exact zero from `not significant`, `no difference`, or a missing estimate.
- Keep reported significance and equivalence conclusions separate from the numerical estimate.
- Never invent uncertainty, sample sizes, units, equivalence margins, or moderator values.
- If the paper has no safely estimable target result, return `status=non_estimable`, no studies,
  and the most specific non-estimability reason.
- Decide the top-level status before constructing the object. `status=non_estimable` requires
  `studies=[]`; never describe candidate studies or cohorts in that branch. `status=estimable`
  requires at least one fully grounded numerical finding.

## Grounding rules

- Every numerical finding requires the shortest exact source quote, table row, or figure-region
  transcription that contains the estimate or complete group statistics.
- Supply an exact page/table/row/figure locator and source line identifiers when available.
- Use Results text, tables, figures, or supplements—not an abstract, introduction, discussion,
  reference title, or a cited study's result.
- The quote must support the represented groups, outcome, timepoint, value, and uncertainty.

## Conditions and moderators

- Output only prespecified moderator names from the frozen question.
- Preserve reported units and categories. Do not normalize ambiguous values.
- A missing or ambiguous moderator value is JSON null, not a guessed category.

Every nullable schema field must still be present with JSON null or an empty array as appropriate.
All arrays of identifiers, source-name strings, line IDs, and warnings must be unique and sorted
lexicographically. Arm, contrast, and finding references must copy their local keys exactly.
