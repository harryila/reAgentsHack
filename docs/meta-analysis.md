# Quantitative synthesis contract

The quantitative path accepts source-grounded `EffectEvidence` records, not legacy
`effect_direction` labels. Its central semantic rule is:

> A non-significant difference is not a zero effect.

`EffectEvidence` therefore records the point estimate, reported significance conclusion,
equivalence-test conclusion, and information availability independently. A literal point estimate
of zero becomes `exact_zero`; no other field can create that label. Missing uncertainty produces an
explicit insufficiency result while preserving the point-estimate sign for the fallback analysis.

## Supported harmonization

`harmonize_effect` supports:

- mean differences in a common named unit, directly reported or calculated from group means, SDs,
  and sample sizes;
- standardized mean differences, represented as bias-corrected Hedges' g;
- odds ratios and risk ratios, represented on log scales, directly reported or calculated from 2x2
  event counts;
- standard errors, variances, or confidence intervals as uncertainty sources.

Zero cells use a disclosed 0.5 continuity correction. Different outcomes, treatment contrasts,
effect scales, or mean-difference units are never silently pooled. Every result retains finding IDs
and source locators. `effect_evidence_json_schema()` exposes the closed extraction contract for a
harvester or structured-output model.

## Integration status

The current quantitative implementation is a standalone, tested library path. It is not yet wired
into the production s3/s5 pipeline: the legacy `FindingRow.effect_size_raw` string does not identify
an estimand, scale, unit, or uncertainty source safely enough to convert automatically. A real
pipeline run therefore needs an explicit, source-grounded adapter that emits `EffectEvidence`
records and a frozen synthesis artifact. Until that adapter exists, only the constructed simulation
exercises this quantitative path; legacy directional s5 output is not effect-size meta-analysis.

## Synthesis APIs

```python
from literature_multiverse.effects import EffectEvidence, harmonize_effects
from literature_multiverse.meta_analysis import (
    categorical_meta_regression,
    synthesize_with_directional_fallback,
)

records = [EffectEvidence.model_validate(payload) for payload in payloads]
harmonized = harmonize_effects(records)
synthesis = synthesize_with_directional_fallback(harmonized)

estimable = [item.effect for item in harmonized if item.effect is not None]
dose_model = categorical_meta_regression(estimable, "dose", reference_level="low")
```

The random-effects implementation uses generalized Paule–Mandel tau-squared estimation, modified
Knapp–Hartung uncertainty, Cochran's Q/I-squared heterogeneity summaries, and a prediction interval
when at least three papers are available. Multiple compatible rows from one paper are first reduced
to one equally weighted paper effect. The default assumes within-paper correlation `rho=1`, so
duplicate outcomes do not create artificial precision; a smaller value must be prespecified and
justified.

Categorical meta-regression uses the same one-effect-per-paper reduction and requires a moderator to
be constant within each paper. It reports predictive corpus associations only. Sparse levels,
within-paper conflicts, incompatible scales, and inadequate residual degrees of freedom return
stable `status="insufficient"` reason codes.

If magnitude synthesis is unavailable, `directional_synthesis` uses paper-level point-estimate signs
only. Conflicting signs within one paper become `mixed`; missing estimates remain `unavailable`; and
reported non-significance or equivalence never becomes `exact_zero`. The result explicitly states
that magnitudes and precision were not combined.

## Planted method check

`scripts/simulate_meta_analysis.py` creates paper-independent train/test corpora in which the two
moderator levels have different study precision. This is a deliberately adversarial setting for
vote counting: converting `p < .05` into a positive direction makes precision look like an effect
modifier. The effect-size model uses the reported variances instead.

```bash
uv run python scripts/simulate_meta_analysis.py \
  --output artifacts/paper/meta-simulation-200.json \
  --replicates 200 --seed 20260826 \
  --alpha 0.05 --moderator-effect 0.35 \
  --papers-per-level 30 --heldout-papers-per-level 30
```

With no planted moderator, meta-regression selected one in 0.050 of replicates, versus 0.955 for
significance voting. With a planted moderator, the rates were 1.000 and 0.070, respectively.
Meta-regression also had lower held-out Brier loss for the observed point-estimate sign (0.117 vs.
0.371 under the null; 0.049 vs. 0.226 with a moderator). These results demonstrate the constructed
precision-confounding failure and validate the implementation. They do not estimate performance on
real systematic reviews.
