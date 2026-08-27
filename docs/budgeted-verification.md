# Budgeted human verification

This module answers a narrow decision question:

> Given a fixed human-audit budget, which item should be checked next to reduce the
> risk of the wrong corpus-level claim?

It does **not** infer that an item-level confidence score is a real probability. It
does **not** replace human adjudication. The current paper artifact is a planted
method check, not a human study or real-review result.

## Contracts and separation

`budgeted_verification.py` defines three separate contracts:

- `AuditCandidate` contains policy-visible information: an item ID, baseline and
  counterfactual claim contributions, an error probability or score, its declared
  basis and source, verification cost and unit, disagreement, and scenario
  provenance.
- `ClaimModel` converts the additive snapshot into a claim probability and a frozen
  binary decision. A non-additive synthesizer must first rerun the conclusion under
  each candidate correction or leave-one-out scenario and encode an equivalent local
  replacement. The approximation must be validated against those reruns.
- `AuditOracle` contains adjudicated error and correction labels. Ranking functions
  do not accept this object; it is available only to retrospective evaluation.

The API rejects duplicate identities, mixed cost units, nonfinite values, nonpositive
costs, and incomplete oracle identity sets. A leave-one-out scenario must replace its
item contribution with zero.

## Priority and its assumptions

For item \(i\), let \(r_i\) be its declared error probability, \(p\) the current
claim probability, \(p_{-i}\) the probability under its documented counterfactual,
and \(c_i\) its human-verification cost. The proposed priority is

\[
  \frac{r_i |p-p_{-i}|}{c_i}.
\]

Here \(|p-p_{-i}|\) is conclusion influence under an absolute
claim-probability-loss scenario. The numerator is an expected claim-loss reduction
only if all of the following are credible:

1. \(r_i\) is a calibrated probability for the deployment population;
2. the candidate counterfactual represents what correction would do;
3. a one-item counterfactual is an adequate local approximation; and
4. audit correction is accurate.

The contract therefore labels probability basis as `calibrated`, `heuristic`, or
`planted_simulation`. Heuristic values must be described as scores, not probabilities,
in empirical claims. Interactions among multiple corrections can make realized loss
reduction differ from the sum of individual priorities.

## Fixed-budget comparison

`evaluate_fixed_budgets` compares nine policies on exactly the same cost caps. The
full component family prevents the proposed product from receiving a unique benefit
from cost normalization:

1. seeded random ranking;
2. inverse verification cost only;
3. risk/error score only;
4. verifier disagreement only;
5. conclusion influence only;
6. risk × influence without cost normalization;
7. risk / verification cost;
8. influence / verification cost; and
9. risk × influence / verification cost.

Every policy uses deterministic ordering, including an item-ID tie break. Allocation
is an explicit rank-then-pack procedure: scan the ranking, select an item if it fits,
and continue past items that do not fit. It is not claimed to solve the global
knapsack optimum.

The retrospective evaluator applies a correction only when the hidden audit oracle
marks the selected item erroneous. It reports budget spent, items selected, errors
found, error recall, absolute claim-probability loss relative to the fully corrected
oracle claim, loss recovered, and whether the binary conclusion was repaired.

## Prospective release guard

`assess_prospective_release_guard` is the audit-specific guard used after allocation.
By default it blocks when any unresolved item:

- has a counterfactual that flips the current binary conclusion;
- changes claim probability by more than 0.05;
- contributes to a summed scenario-based expected loss above 0.05; or
- uses an error score that is not explicitly calibrated.

The thresholds are frozen in `ReleaseGuardConfig`. The summed scenario loss is a
triage burden, not a simultaneous probability bound; correction scenarios can
interact. Turning off the calibrated-probability requirement permits experimental
heuristic triage, but does not create a risk guarantee.

An item counts as resolved only after human adjudication is complete. Being selected,
assigned, or in progress does not resolve it. If the reviewer confirms an error, the
correction must be incorporated into a new corpus-level baseline and all influences
must be rerun before the item is marked resolved. If the budget ends while a material
item remains unresolved, the audit guard stays blocked: the system must abstain or
obtain more review budget.

An `eligible_for_downstream_gates` result is deliberately not `release`. Retrieval,
grounding, statistical, and calibrated question-level release checks must still pass.

## Frozen planted experiment

Regenerate the artifact with:

```bash
uv run python scripts/simulate_budgeted_verification.py \
  --output artifacts/paper/budgeted-verification-simulation-200.json \
  --replicates 200 --seed 20260827 --force
```

The generator creates 60 auditable items per replicate. Error labels are Bernoulli
draws from known planted probabilities. Candidate correction influence and simulated
human-minute costs vary across items. The observed corpus claim starts just above the
decision threshold, while planted positive extraction distortions can make it differ
from the fully corrected claim. The evaluator tests budgets of 5, 10, 20, and 40
simulated human minutes.

All probabilities, corrections, costs, and labels in this artifact are planted. The
experiment validates implementation behavior and illustrates when influence-aware
allocation can help. It cannot support a claim about real reviewers, real minutes,
real extraction errors, or generalization to scientific corpora. Those require a
preregistered audit with blinded adjudicators and a frozen real pipeline.

### Frozen uncertainty analysis

The independent sampling unit is one generated planted corpus. Aggregate arithmetic
means use two-sided Student-t intervals across the 200 replicates, with sample
variance and 199 degrees of freedom. Aggregate binary repair rates use two-sided
Wilson score intervals. At the predeclared budget of 5 simulated cost units, policy
contrasts are paired within the same generated corpus:

- recovery-fraction differences use a two-sided paired Student-t interval; and
- repair-rate differences use a 20,000-draw paired nonparametric percentile
  bootstrap that resamples whole corpus replicates with replacement.

The bootstrap uses PCG64 and frozen base seed `20260827`. Comparator-specific seeds
are deterministically derived from the base seed and recorded in the artifact
(`8361293546211811922` for risk-only and `14628598057257489332` for random), along
with draw count, quantile method, and all four paired binary-outcome counts. These
are nominal descriptive 95% intervals. They are not adjusted for multiple
comparisons, are not a preregistered hypothesis test, and quantify Monte Carlo
replicate variation inside the planted generator only—not uncertainty about transfer
to real scientific corpora.

Across 200 frozen replicates, the proposed policy recovered 34.0% (95% t interval
31.7%--36.4%) of oracle-attainable absolute claim-probability loss at budget 5,
versus 20.9% (18.9%--22.8%) for risk-only and 7.7% (6.4%--9.0%) for random
allocation. Its paired advantage was 13.2 percentage points over risk-only (95%
paired-t interval 10.8--15.5) and 26.4 points over random (24.0--28.8).

The proposed, risk-only, and random policies repaired the planted binary conclusion
in 169/200 (84.5%; 95% Wilson interval 78.8%--88.9%), 126/200 (63.0%;
56.1%--69.4%), and 43/200 (21.5%; 16.4%--27.7%) replicates, respectively. The paired
proposed-minus-risk-only difference was 21.5 percentage points (95% paired-bootstrap
interval 14.5--28.5; proposed-only/risk-only-only/both/neither counts
53/10/116/21). The proposed-minus-random difference was 63.0 points (56.0--69.5;
paired counts 127/1/42/30).

These are planted mechanics results. The units are simulated and must never be
presented as reviewer minutes or empirical scientific-verification performance.
