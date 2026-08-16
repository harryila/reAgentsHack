# Demo Script Template — Variant B

**Selection condition:** G3 story viability failed, no pre-specified moderator passed every M4
rule, or the fixed M4 battery did not complete.
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
- `{{headline.residuals.rendered_sentence}}`

Export fails on a missing value, an unexpected type, any remaining unreplaced template token, or more than 225
words under “Spoken copy.” The rendered script is copied to `demo/demo_script.md`; this source
template is never presented as evidence.

## Spoken copy

**0–10 seconds — Question**

“We asked whether our retrieved corpus gives one stable answer to this question:
{{manifest.spoken_question}}”

**10–23 seconds — Reconciled corpus funnel**

“We retrieved {{manifest.paper_funnel.searched_documents}} documents, reconciled them to
{{manifest.paper_funnel.identity_deduped_papers}} papers, keeping every exclusion and failure visible. The analysis uses {{manifest.paper_funnel.primary_grounded_findings}}
grounded findings from {{manifest.paper_funnel.primary_grounded_papers}} papers.”

**23–43 seconds — Honest result**

“{{headline.rendered_sentence}}”

**43–58 seconds — Complete moderator table**

“Every pre-specified moderator is shown with its support status. None passed the pre-registered
conditional-pattern gates, so no exploratory tree or post-hoc remap was promoted.”

**58–75 seconds — Residual result**

“{{headline.residuals.rendered_sentence}}”

**75–90 seconds — Evidence gaps and close**

“The empty and sparse cells show exactly where our retrieved corpus cannot distinguish the
explanations — a limit of the corpus, not proof that no explanation exists. The system refused
to invent a hidden variable; it shows what remains unresolved and what evidence is missing.”

## Release disclosure (display; not spoken)

{{manifest.release_selection.rendered_disclosure}}
