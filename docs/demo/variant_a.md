# Demo Script Template — Variant A

**Selection condition:** at least one pre-specified moderator has passed every M4 rule; the script
uses only the deterministic §6.3 winner.
**Target:** 80–90 seconds; the spoken copy below is intentionally under 200 words before rendering.

## Renderer contract

s7 may replace only these validated tokens:

- `{{manifest.spoken_question}}`
- `{{manifest.paper_funnel.searched_documents}}`
- `{{manifest.paper_funnel.identity_deduped_papers}}`
- `{{manifest.paper_funnel.primary_grounded_papers}}`
- `{{manifest.paper_funnel.primary_grounded_findings}}`
- `{{manifest.release_selection.rendered_disclosure}}`
- `{{headline.rendered_sentence}}`

Export fails on a missing value, an unexpected type, any remaining `{{...}}`, or more than 225
words under “Spoken copy.” The rendered script is copied to `demo/demo_script.md`; this source
template is never presented as evidence.

## Spoken copy

**0–10 seconds — Question**

“The papers do not give one stable answer to this question: {{manifest.spoken_question}}”

**10–23 seconds — Reconciled corpus funnel**

“We retrieved {{manifest.paper_funnel.searched_documents}} documents, reconciled them to
{{manifest.paper_funnel.identity_deduped_papers}} papers, and kept every exclusion, failure, and
zero-finding paper visible. The headline uses {{manifest.paper_funnel.primary_grounded_findings}}
grounded findings from {{manifest.paper_funnel.primary_grounded_papers}} papers.”

**23–48 seconds — Controlled conditional pattern**

“{{headline.rendered_sentence}}”

**48–64 seconds — Full moderator table**

“That was selected from the complete pre-specified moderator table using paper-grouped
cross-validation, a family-wise permutation control, a practical-effect threshold, and paper
bootstraps. The losing and insufficient moderators remain visible.”

**64–79 seconds — Tree and evidence cards**

“The tree is only a readable map of the selected pattern. These cards show the underlying source
passages and line references; missing or weakly supported leaves are grey and are not narrated.”

**79–90 seconds — Close**

“So this is a predictive pattern in our retrieved corpus, not a causal law or a claim about all
published literature. Every number and quote here can be traced back to the frozen evidence.”

## Release disclosure (display; not spoken)

{{manifest.release_selection.rendered_disclosure}}
