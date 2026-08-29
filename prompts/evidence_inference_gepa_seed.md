# Evidence Inference extraction prompt

Prompt version: `evidence-inference-2.0-v1`

Given one clinical-trial article's BODY.RESULTS source lines, answer exactly one locked
intervention-comparator-outcome question. Return exactly one JSON object matching the supplied
schema. Do not add prose, markdown, citations, or fields outside that schema.

## Locked question

- Outcome: `[[OUTCOME]]`
- Intervention: `[[INTERVENTION]]`
- Comparator: `[[COMPARATOR]]`

Direction describes the reported intervention-versus-comparator difference for this outcome:

- `increase`: the outcome was reported as significantly increased for the intervention.
- `decrease`: the outcome was reported as significantly decreased for the intervention.
- `no_effect`: the article reported no significant difference between the groups.

Do not reinterpret direction as beneficial or harmful. If the article does not answer the locked
question, return `eligible=false` and an empty `findings` array. Otherwise return `eligible=true`
and exactly one finding.

Copy the shortest verbatim Results passage that directly supports the direction. Cite the exact
source line or inclusive line range, using `L12` or `L12-L14`. The quote must be contained in the
cited line text. Do not infer a significant difference from numerical ordering alone, and do not
turn absence of a reported p-value into `no_effect`.
