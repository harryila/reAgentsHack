# Demo Script Template — Variant B

**Selection condition:** G3 story viability failed, no pre-specified moderator passed every M4
rule, or the fixed M4 battery did not complete.
**Target:** 80–90 seconds; the spoken copy below is intentionally under 200 words before rendering.

## Renderer contract

s7 may replace only these validated tokens:

- `Does vitamin C or E supplementation change how people adapt to exercise training?`
- `649`
- `648`
- `6`
- `12`
- `Release: frozen v1 with 6 grounded primary papers; no scaled candidate was promoted.`
- `In our retrieved, grounded corpus, paper-level directional disagreement is 58% (90% bootstrap interval 0–63%), but the corpus did not meet the pre-registered disagreement-support gate for a conditional-pattern claim. 0 grounded opposite-direction paper pairs remain, and 32 of 32 pre-registered evidence cells contain fewer than five papers.`
- `No grounded opposite-direction pair met the matching rule; the empty residual view is part of the result.`

Export fails on a missing value, an unexpected type, any remaining unreplaced template token, or more than 225
words under “Spoken copy.” The rendered script is copied to `demo/demo_script.md`; this source
template is never presented as evidence.

## Spoken copy

**0–10 seconds — Question**

“We asked whether our retrieved corpus gives one stable answer to this question:
Does vitamin C or E supplementation change how people adapt to exercise training?”

**10–23 seconds — Reconciled corpus funnel**

“We retrieved 649 documents, reconciled them to
648 papers, keeping every exclusion and failure visible. The analysis uses 12
grounded findings from 6 papers.”

**23–43 seconds — Honest result**

“In our retrieved, grounded corpus, paper-level directional disagreement is 58% (90% bootstrap interval 0–63%), but the corpus did not meet the pre-registered disagreement-support gate for a conditional-pattern claim. 0 grounded opposite-direction paper pairs remain, and 32 of 32 pre-registered evidence cells contain fewer than five papers.”

**43–58 seconds — Complete moderator table**

“Every pre-specified moderator is shown with its support status. None passed the pre-registered
conditional-pattern gates, so no exploratory tree or post-hoc remap was promoted.”

**58–75 seconds — Residual result**

“No grounded opposite-direction pair met the matching rule; the empty residual view is part of the result.”

**75–90 seconds — Evidence gaps and close**

“The empty and sparse cells show exactly where our retrieved corpus cannot distinguish the
explanations — a limit of the corpus, not proof that no explanation exists. The system refused
to invent a hidden variable; it shows what remains unresolved and what evidence is missing.”

## Release disclosure (display; not spoken)

Release: frozen v1 with 6 grounded primary papers; no scaled candidate was promoted.
