# Evidence Inference item-risk diagnostic

## Result and permitted interpretation

This is a real-data but explicitly retrospective and non-pristine diagnostic of the
item-risk calibration machinery. It uses frozen paired seed/winner predictions from the
Evidence Inference GEPA test run and the benchmark's official direction annotations. It is
not a verifier calibration result, a human-audit result, or evidence that the risk score
ranks errors well.

The source had 524 prompt rows from 191 papers. A namespaced hash selected one prompt per
paper before any reference-direction field was read in this rerun. A separate namespaced
paper hash assigned 82 papers to development and 109 to calibration, with zero paper or
question overlap. The bins were prespecified and did not use development or calibration
labels.

The calibration result is:

| Label-free score cell | Interpretation | Errors / units | Empirical error | Simultaneous upper bound |
|---|---|---:|---:|---:|
| `[0.0, 0.1)` | no failure flags | 42 / 64 | 0.6563 | 0.7789 |
| `[0.1, 0.3)` | one failure flag | 21 / 27 | 0.7778 | 0.9213 |
| `[0.3, 1.0]` | at least two failure flags | 12 / 18 | 0.6667 | 0.8779 |

These are Bonferroni-adjusted, one-sided Clopper–Pearson upper confidence limits with a
0.05 familywise error budget over the three biomedical domain-by-bin cells. They estimate
group-average error rates within cells. They are not calibrated probabilities for
individual items. Their nominal coverage assumes the paper-level calibration units are
exchangeable with the target population within each cell. Paper/question disjointness does
not establish cohort independence, and this diagnostic did not have authority-scoped trial
or cohort identifiers with which to test that stronger condition.

The observed ordering is nonmonotonic, and all three error rates are high. Therefore this
study does not support a claim that this simple five-flag score usefully ranks extraction
errors. Its defensible use is narrower: it proves that real benchmark rows can enter the
item-scheduling UCL contract without fabricating model confidence, human adjudication, or
a distribution-shift result. The bundle remains
`release_probability_authority=false`.

## Frozen score and independence rule

The risk score is the mean of five binary, label-free flags:

- seed and winner predicted directions disagree;
- seed structured output is invalid;
- winner structured output is invalid;
- seed exact grounding is invalid;
- winner exact grounding is invalid.

The feature projector does not read the official direction, direction accuracy,
distribution fidelity, the GEPA scalar objective, a model confidence field, or a human
label. The official direction is used only afterward to define whether the frozen winner
prediction is wrong.

Evidence Inference contains multiple prompts for many papers. Treating all 524 rows as
independent calibration units would be pseudoreplication. This diagnostic instead keeps
one deterministic prompt per paper and assigns the complete paper to one split. The
selection and split rules are hash-bound in the design receipt.

## Prediction-time lineage boundary

The historical prediction source—not the current verifier—defines the prediction pipeline
identity. The public artifact includes a content-silent, self-hashed projection of the
frozen GEPA plan, winner, prompt, schema, local-model, test-split, source-code, and paired
report hashes. The prediction-source lineage hash is
`d4fa141b5b767a78943460a9e1eb2e41854d2667f518e9dd542e761d1bbc6808`.

The predictions came from local `llama3.2:1b`, reported by the frozen runtime as a 1.2B
parameter model. The standalone diagnostic pipeline hash is
`bf1c5c585cead5c2a3b9521d1d4b89f405e15aa3e22f201fa11a598ede0bc95a`.
The public artifact explicitly declares `current_verifier_pipeline_compatible=false`.

## Access order and its limit

The clean rerun used three stages:

1. Freeze the score, bins, sampling/adjudication protocols, historical prediction lineage,
   and standalone pipeline before opening the private paired report.
2. Recompute the pipeline, validate the GEPA plan/winner/manifest, open the paired report,
   select and split rows using identifiers, project label-free features, and only then read
   the official-direction field.
3. Fit simultaneous bounds from calibration units and publish an identifier-free aggregate.

This is an exact logical access-order receipt, but not a claim of pristine blinding. The
paired report physically co-locates predictions and official labels, and all benchmark
labels in this checkout were historically opened before this protocol existed. The study
is retrospective by construction.

## Reproduction

The config self-hash is
`a472f54cad7f55c394446e7037639220e7f317e4ee4e9e2289a695cf198773e2`.
On a checkout where the configured ignored run directory does not yet exist:

```bash
.venv/bin/python scripts/run_evidence_inference_item_risk_diagnostic.py freeze-design \
  --config configs/benchmarks/evidence-inference-item-risk-v1.json \
  --repository-root . \
  --work-dir data/cache/evidence-inference-item-risk-v1-final-v5
```

The frozen design receipt is
`ab832131044a3a3d54b045cac591eb47346a3155cc050fa04345a466670aeb9c`.

```bash
.venv/bin/python scripts/run_evidence_inference_item_risk_diagnostic.py materialize-units \
  --config configs/benchmarks/evidence-inference-item-risk-v1.json \
  --repository-root . \
  --work-dir data/cache/evidence-inference-item-risk-v1-final-v5 \
  --expected-design-receipt-sha256 \
    ab832131044a3a3d54b045cac591eb47346a3155cc050fa04345a466670aeb9c
```

The materialization receipt is
`f8e53a902a53faf68e25f5e46e6734c8b305ffe00ee0f11e7b178937d2b07ba6`.

```bash
.venv/bin/python scripts/calibrate_item_risk.py calibrate \
  --expected-pipeline \
    data/cache/evidence-inference-item-risk-v1-final-v5/diagnostic-pipeline-fingerprint.json \
  --pipeline-root . \
  --fixed-bins data/cache/evidence-inference-item-risk-v1-final-v5/fixed-risk-bins.json \
  --development-units \
    data/cache/evidence-inference-item-risk-v1-final-v5/development-units.jsonl \
  --calibration-units \
    data/cache/evidence-inference-item-risk-v1-final-v5/calibration-units.jsonl \
  --familywise-delta 0.05 \
  --sampling-protocol-sha256 \
    e502ec296f1946f23f147951f446392999c143c1b1adf01cb011f9744065ce51 \
  --error-event-definition \
    'winner predicted direction differs from the official Evidence Inference direction for the deterministically selected paper-level example' \
  --shift-detector-id not-assessed-retrospective-diagnostic-v1 \
  --shift-detector-sha256 \
    abaef4df4faa3cbd5ea2fb6d3c3f23071728513551849d6a933238bc5ee2e80d \
  --supported-domain biomedical-evidence-inference \
  --output data/cache/evidence-inference-item-risk-v1-final-v5/calibration-run.json
```

The calibration receipt is
`eb02572de442e1d2b36b2d05028c727d44c4f18b921fc2c2b92c9e0bd7512d4a`,
and its non-authoritative item-risk bundle is
`b461d872de3f66c8f0447f158fecd74c794d5af12504d90fecc123e1123e421e`.

After all three private receipt hashes have been checked, regenerate the public projection in
an experimental copy of the checkout:

```bash
.venv/bin/python scripts/run_evidence_inference_item_risk_diagnostic.py summarize \
  --config configs/benchmarks/evidence-inference-item-risk-v1.json \
  --repository-root . \
  --work-dir data/cache/evidence-inference-item-risk-v1-final-v5 \
  --calibration-run data/cache/evidence-inference-item-risk-v1-final-v5/calibration-run.json \
  --expected-design-receipt-sha256 \
    ab832131044a3a3d54b045cac591eb47346a3155cc050fa04345a466670aeb9c \
  --expected-materialization-receipt-sha256 \
    f8e53a902a53faf68e25f5e46e6734c8b305ffe00ee0f11e7b178937d2b07ba6 \
  --expected-calibration-run-receipt-sha256 \
    eb02572de442e1d2b36b2d05028c727d44c4f18b921fc2c2b92c9e0bd7512d4a \
  --output artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json \
  --force
```

The checked-in aggregate can be validated without the ignored row cache:

```bash
.venv/bin/python scripts/run_evidence_inference_item_risk_diagnostic.py validate-public \
  --summary artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json
```

Its payload hash is
`61c8620e02fd2b17dbf495b7f6b97324791ec6e65f5340aa206736cfed8c7c3b`.
The public registry additionally recomputes the standalone diagnostic pipeline from public
source/configuration files. It does not pretend to replay row-level metrics when the
ignored paired report is absent.

## Non-claims

- No labels are pristine, secret, or prospectively held out.
- No independent human adjudication was performed; labels are official benchmark
  annotations.
- No confidence value was invented or calibrated.
- No distribution-shift test or prospective deployment scoring was run.
- Paper-level disjointness is not evidence of cohort independence or exchangeability.
- No claim-release probability, end-to-end verifier accuracy, or scientific truth claim is
  authorized.
- The result does not generalize beyond this frozen local 1.2B model run and this
  biomedical benchmark sample.
