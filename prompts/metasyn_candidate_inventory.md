# Literature Multiverse MetaSyn bounded candidate inventory

Prompt version: `metasyn-candidate-inventory-v1`

Inspect only the frozen source projection for the frozen P/I(E)/C/O question below.
Return exactly one JSON object matching the supplied inventory schema. Do not return prose,
markdown, numerical effect values, publication identifiers, or fields outside the schema.

## Frozen question and protocol

`[[QUESTION_SPEC_JSON]]`

This is a label-blind oracle-corpus extraction-yield study. Matched-paper membership is
evaluator supplied; the review conclusion, direction, aggregate effect, significance, and
official test fields are unavailable and must never be inferred or requested.

The inventory is a value-free routing step, not a scientific conclusion. A candidate is one
potentially complete numerical target contrast for one canonical outcome ID and one source
anchor. Distinct studies, cohorts, contrasts, outcomes, or timepoints are distinct candidates.

- Use `candidates_found` only when at least one safely identifiable candidate exists and the
  complete candidate inventory is below the schema capacity.
- Use `no_candidate_found` with an empty list only when no candidate is visible in the exposed
  projection. This is not proof that the publication is non-estimable.
- Use `overflow_or_uncertain` with `has_more_or_uncertain=true` whenever the candidate set may
  reach capacity, boundaries are ambiguous, or a safe effect kind cannot be selected.
- Use only supplied canonical outcome IDs and exposed line IDs. Do not copy or invent a raw
  outcome name, clinical direction, or numerical result.
- Title/Abstract-only projections are diagnostic source surfaces and must not be described as
  full-text evidence.

All candidate line-ID lists must be sorted, unique, nonempty, and within the schema cap.
