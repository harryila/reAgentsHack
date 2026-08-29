# Misspecified adaptive-verification stress study

This study asks a narrow engineering question: when the assumptions behind an
audit scheduler are wrong, how do its release-error/coverage curves change at a
fixed simulated human budget? It is an adversarial simulation and is not evidence
of real scientific accuracy, calibrated release risk, or human-review efficiency.

## What is compared

The fixed-budget arms receive the same nominal simulated person-minute budgets:

- `adaptive_static`: use the production ranker to freeze a ranking by item-risk
  score × exact leave-one-out probability influence / cost;
- `adaptive_sequential`: after every reviewer result, rerun the nonlinear synthesis,
  recompute leave-one-out influence, update risk for unresolved items in the same
  cohort using only the observed correction, and invoke the production ranker again;
- deterministic random review;
- risk-only and risk-per-cost;
- disagreement-only and disagreement-per-cost;
- influence-only and influence-per-cost;
- a fixed five-item source-order gate, subject to the same minute budget.

`no_audit` is a zero-minute reference. `audit_all` reviews every accessible item and
is reported at its actual, unmatched cost. Because reviewers can be wrong and some
full texts are unavailable, `audit_all` is an exhaustive-review reference—not an
oracle upper bound.

All ranking functions receive only `StressVisibleItem`: observed contribution,
noisy risk and disagreement scores, expected person-minutes, cohort identity, and
full-text availability. Hidden extraction errors, true contributions, and reviewer
correctness live in a separate `StressOracleItem` and enter only at the simulated
adjudication/evaluation boundary.

## Why the generator is misspecified

The proposed allocation score is not the data-generating objective. Seven frozen
populations independently vary:

- shared-cohort error correlation;
- reversal/noise in the item-risk sensor;
- nonlinear within-cohort interactions in the actual claim synthesis;
- reviewer mistakes that may retain or amplify extraction distortion;
- missing full text, which leaves influential items unauditable;
- a combined target-domain shift containing all of the above;
- an approximately IID source-like control.

Extraction errors have mixed signs. The current conclusion is produced by rerunning
the same nonlinear synthetic synthesis after each correction, so two corrections
can interact. Sequential risk updates observe only the difference between the old
value and the reviewer-returned value; they do not know whether the reviewer was
right.

The release score is also distinct from the allocation score. For each final audit
state, a frozen, threshold-blind joint perturbation ensemble samples correlated,
mixed-sign changes to every unresolved item and records the fraction that flips the
current conclusion. The predeclared threshold family is then replayed over that one
state. These are sensitivity operating points, not calibrated probabilities.

## Outcomes and uncertainty

The primary curves report, at each release-risk threshold:

- coverage: released questions / all complete questions;
- released-claim error: incorrect released conclusions / released questions;
- correctly released claims per simulated person-hour;
- all numerators, denominators, and mean minutes per question.

The independent sampling unit is one complete simulated question. Curve rates have
question-level Wilson intervals. At the frozen primary operating point, all three
primary metrics have deterministic nonparametric percentile intervals obtained by
resampling complete questions. The proposed-versus-baseline contrasts use the same
question bootstrap indices. Intervals are pointwise and have no simultaneous or
multiple-comparison guarantee.

## Lineage and rerunning

The CLI seals the generator profiles, scenario list, seed, policy family, budgets,
release thresholds, Monte Carlo size, uncertainty procedure, and exact source file
and environment-lock hashes before generating the first result. Python, NumPy, and
PCG64 runtime identity are also recorded. Every complete-question receipt has a
self-hash. The public aggregate contains the ordered receipt-hash manifest, a
manifest hash, the frozen configuration hash, and an artifact self-hash. Full
synthetic receipts can be regenerated exactly from the recorded configuration.

Run the prespecified public study with:

```bash
.venv/bin/python scripts/run_adaptive_stress_study.py --force
```

The default aggregate is
`artifacts/diagnostics/adaptive-stress-study-v1.json`.

## Frozen result

The prespecified study contains 1,120 complete questions (160 in each of seven
scenarios) and 26,880 evidence items. At 30 simulated person-minutes and a maximum
policy-visible perturbation flip rate of 0.10, sequential adaptive auditing released
63.66% of questions with a released-claim error rate of 16.13% and 1.1117 correctly
released claims per simulated person-hour.

Against random review at the same nominal budget, the paired error-rate difference was
-0.0876 (pointwise question-bootstrap 95% interval [-0.1170, -0.0595]) and the
efficiency difference was +0.1448 correct releases per person-hour [0.0823, 0.2047].
Against the fixed-five gate, the error-rate difference was -0.1012 [-0.1308, -0.0723].
The sequential-versus-static adaptive error difference was only -0.0163
[-0.0334, 0.0012], so this simulation does not establish a sequential-over-static
advantage. In the combined shifted scenario, sequential adaptive coverage fell to
51.25% while released-claim error rose to 37.80%, demonstrating rather than concealing
the lack of shift robustness.

These error rates are intentionally not presented as acceptable production risk. The
0.10 threshold is a synthetic perturbation-sensitivity setting, not the calibrated
complete-question risk target used by the verifier under its declared sampling and
exchangeability assumptions.

## Interpretation boundary

A favorable result would show that the allocation mechanism remains useful under
these particular departures from its assumptions. An unfavorable or mixed result
would identify where abstention or a different scheduler is necessary. Neither
result estimates performance on real literature. The study contains no real paper,
expert adjudication, measured review time, or scientific truth label, and therefore
cannot support a production release guarantee.
