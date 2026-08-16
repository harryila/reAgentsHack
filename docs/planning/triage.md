# Topic Triage Decision — 2026-08-15 evening (M2 / G2)

**Decision: LOCK topic C — antioxidant (vitamin C/E) supplementation × exercise-training
adaptations — as `configs/questions/antiox-training.yaml`
(config sha256 `9f1a61f67ae809aa12d64c98d8f5f0186ee40aacf34f7c2cdb37a9a0a13acf34`).**

Both candidates ran the identical protocol: live s1 searches (`--all`, per-query limit 100,
confirmed reviews excluded at retrieval) → deterministic s2 screen/dedupe → stratified,
seeded, keyword-prefiltered 10-paper probe (exact provider set built via a dedicated paper
repo and verified by CSV export) → live schema'd extraction with prompt `extraction-v1` →
scored with the tested `evaluate_g2` gate. Full numbers:
[triage-c-score.json](triage/triage-c-score.json), [triage-a-score.json](triage/triage-a-score.json).

| Criterion | C: antioxidants × training | A: time-restricted eating |
|---|---|---|
| s2 included papers | 305 | 175 |
| Probe eligible (of 10) | 5 | 2 |
| Classifiable findings | 28 | 15 |
| Direction split (findings) | 9 increase / 6 no_effect / 13 decrease | 0 / 12 / 3 |
| Distinct papers by direction | 2 / 1 / 4 | 0 / 2 / 1 |
| Paper-modal entropy (90% CI) | **0.96** (0.46–0.96) | 0.00 (0–0) |
| Relation purity (operator-labeled) | 28/28 = 1.00 | 1.00 (2 papers only) |
| Est. usable primaries | **152.5** | 35.0 |
| G2 | **PASS (all rules)** | FAIL (entropy, direction support, volume) |

## Why the pre-scan ranking survived and the volume story inverted

The planning pre-scan favored C for its science (aggregate-null vs dose-threshold narrative)
but feared its volume, and trusted A's volume. Empirically the reverse held: A's probe found
mostly ineligible papers (pilot/single-arm designs, no eating-window contrast) and its two
eligible trials were **null-dominant with zero paper-level disagreement** — exactly the
null-dominance risk the pre-scan flagged. C's corpus is large (305 screened, ~152 est.
usable) and its probe shows **near-maximal paper-level directional disagreement** with all
three directions populated.

## What the probe also taught (encoded into the locked config)

1. Outcome names arrive as free text → `outcomes.endpoint_map` regex canonicalization onto
   five locked functional endpoints; molecular/redox/damage/inflammation families are mapped
   but non-primary.
2. Primary family = `functional_adaptation`: the headline question is whether supplementation
   changes what training *does* (strength, mass, aerobic capacity, performance) — redox
   markers move mechanically and would drown the story (probe family concentration shows it).
3. Human-only eligibility made explicit (probe caught rat studies excluded only by model
   improvisation).
4. Anchors: `PMC12384908` (functional adaptations LARGER with C+E in older sarcopenic women)
   and `PMC2129167` (functional adaptations unchanged in trained kayakers) — an
   in-corpus, in-probe verified opposite-direction pair.

## Triage-process fixes discovered live (all pinned in code/tests)

- `-c` is not a count; `--json` doesn't exist (CSV export path built); `merge` cannot see
  server-side ids (repo-based exact sets built); `-j`/named workers gated to GXL testers;
  CLI reports many errors in-band with exit 0 (boundary hardened); triage prompts previously
  rendered `None`/`[]` for unlocked outcomes (exploratory clause added).

## Escalation ladder status

Not needed — C passed every G2 floor. A remains recorded as the fallback; the C+B metformin
merge rung was never reached.
