# Literature Multiverse bounded native candidate inventory

Prompt version: `native-candidate-inventory-v1`

Inspect only the frozen source projection for the frozen scientific question below. Return
exactly one JSON object matching the supplied inventory schema. Do not return prose,
markdown, numerical effect values, publication identifiers, or fields outside the schema.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

The inventory is a value-free routing step, not a scientific conclusion. A candidate is one
potentially complete numerical target contrast for one prespecified endpoint and one source
anchor. Distinct studies, cohorts, contrasts, outcomes, or timepoints are distinct candidates.

- Use `candidates_found` only when at least one safely identifiable candidate exists and the
  complete candidate inventory is below the schema capacity.
- Use `no_candidate_found` with an empty list only when you found no candidate in the exposed
  projection. This is a model routing output, not proof that the publication is non-estimable.
- Use `overflow_or_uncertain` and `has_more_or_uncertain=true` whenever the candidate set may
  reach the schema capacity, the candidate boundaries are ambiguous, or a safe effect kind
  cannot be selected.
- Never fill the candidate list to capacity merely to satisfy the schema. Reaching capacity
  causes the complete publication to abstain.
- Candidate indices must be contiguous from 1. Use only the supplied canonical endpoint enum
  and exposed line-ID enum. Include no numerical result or free-text summary.
- Select the effect kind that exactly matches the reported statistics: direct estimate plus
  SE, variance, or confidence interval; complete continuous group statistics; or complete
  binary group statistics. If none is safe, use `overflow_or_uncertain`.

All candidate line-ID lists must be sorted, unique, nonempty, and no longer than the schema
limit.

