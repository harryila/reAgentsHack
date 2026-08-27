# Task and evaluation contract

Status: **methodological contract; no benchmark result is implied by this file.**

This document fixes the task, units, labels, split boundaries, and metrics for the
Literature Multiverse evaluation. It must be versioned before a frozen test split is
opened. Changing it after test access creates a new pipeline version and requires a new
test evaluation.

## Task

For one scientific question, one corpus cutoff, and one explicitly defined target
estimand, the system must:

1. identify the evidence that bears on the estimand;
2. represent publications, studies/cohorts, arms, contrasts, outcome estimates, and
   exact evidence spans without treating them as independent when they are not;
3. synthesize compatible estimates, using directional evidence only as a declared
   fallback;
4. propose one claim status and any supported conditions;
5. rank evidence items for human verification under a fixed cost budget; and
6. release the claim only through a policy frozen on other review questions.

The harvester is a replaceable corpus adapter. “Across scientific domains” means across
the domains represented in the benchmark; it does not mean access to every paper or
validity in every discipline.

## Question-level output labels

Each system output has exactly one of these statuses:

- `supported`: the synthesis supports the prespecified claim for the target population,
  intervention/exposure, comparator, outcome, and timepoint.
- `contradicted`: the synthesis supports a prespecified claim incompatible with the
  target claim. Mere failure to reject the null is not contradiction.
- `condition_dependent`: prespecified, auditable conditions produce materially different
  supported conclusions, and the conditions improve out-of-sample prediction. An
  exploratory subgroup pattern is not sufficient.
- `inconclusive`: evidence bears on the estimand, but uncertainty, heterogeneity,
  dependence, bias, or unresolved high-influence evidence prevents the other labels.
- `not_evaluable`: the corpus does not contain an extractable compatible estimand, graph
  identity is unresolved, or a required pipeline contract fails.

`abstained` is a release decision, not a sixth scientific label. The system can infer a
provisional status and still abstain from releasing it.

## Effect semantics

The following fields are distinct and must never be collapsed:

- point estimate and direction;
- uncertainty or confidence interval;
- reported statistical significance;
- result of a prespecified equivalence test; and
- availability/estimability.

“Not statistically significant,” missing reporting, an imprecise interval, and an
unextractable estimate are not zero effects. `exact_zero` is allowed only when the
reported or derived point estimate is numerically zero. `equivalent` is allowed only
when a reported equivalence procedure and margin support that conclusion.

## Independent units and identities

- The calibration and final-claim unit is one complete review question and corpus.
- The synthesis unit is a study/cohort, not a finding row or publication. Multiple
  publications from one study must share an identity or force abstention when the
  identity cannot be resolved safely.
- Multiple estimates within a study require an explicit dependence model or a
  conservative prespecified reduction.
- Findings, bootstrap draws, outcome rows, and papers from the same question are never
  counted as independent calibration examples.
- Eligible papers that yield zero extracted findings remain in the recall denominator
  and human-audit sampling frame.

## Gold labels and adjudication

The reference packet for each question contains a corpus cutoff, candidate and included
papers, an expert extraction table, target estimands, source spans where legally
available, and an expert claim verdict. Published review conclusions may seed the packet
but are not automatically treated as truth.

Human adjudicators are blinded to system confidence and audit-policy rank. Each item is
independently reviewed by two adjudicators; disagreements are resolved by a third. The
adjudication ledger records item IDs, timestamps, decisions, and reasons, but no names in
public artifacts. Until this review occurs, outputs are labelled `unadjudicated` and
cannot be reported as human-audit accuracy.

## Split and freeze protocol

- Split by whole review question, never by rows or papers.
- No paper identity may cross development, calibration, or test.
- Where sample size permits, reserve an entire domain for the final transfer analysis.
- Fit prompts and pipeline components on development reviews only.
- Fit the scalar error-risk model on development reviews.
- Choose the selective-release threshold on separate calibration reviews.
- Freeze code, prompts, feature schema, population definition, identities, and hashes
  before opening test labels.
- The test split estimates risk and coverage; it does not tune the policy.

With too few independent review questions, report empirical risk–coverage curves and
intervals only. Do not describe a simulation-based or undersized calibration result as
a real-world error guarantee.

## Metrics

All primary intervals are question-clustered or study-clustered as appropriate.

### Corpus and extraction

- screening recall over gold-included papers;
- screening precision as a workload metric;
- atomic estimate precision, recall, and F1 under one-to-one matching;
- exact-span grounding accuracy;
- absolute and standardized effect-estimate error;
- uncertainty-field completeness and error;
- study/publication identity resolution accuracy; and
- risk-of-bias field agreement.

An oracle-corpus ablation evaluates extraction and synthesis with the gold included set.
The gap between retrieved-corpus and oracle-corpus performance estimates retrieval's
downstream contribution; it is not itself a causal decomposition.

### Synthesis and release

- question-level status accuracy;
- condition/moderator precision, recall, and out-of-sample predictive improvement;
- empirical selective risk among released claims;
- answer coverage;
- simultaneous confidence bounds used by the frozen release policy; and
- risk–coverage area or a predeclared set of operating points.

Accuracy on answered questions is always shown together with coverage. Abstentions are
not counted as correct answers.

### Human-verification budget

The budget is a prespecified sum of item verification costs (preferably adjudicated
minutes; otherwise a clearly labelled proxy). At each budget, compare:

- random order, averaged over fixed random seeds;
- model confidence/error probability;
- verifier disagreement;
- conclusion influence;
- risk times influence;
- inverse cost, risk per cost, and influence per cost;
- expected reduction in claim loss per unit cost.

The primary budget outcome is the number or proportion of correct released claims after
verified corrections, reported with coverage. A planted-error simulation validates the
mechanism but is not evidence of human-review performance. Real audit results require
the blinded adjudication protocol above.

## Required ablations

On the identical frozen test questions, compare:

1. handwritten versus GEPA-optimized evidence processing;
2. retrieved versus oracle included corpus;
3. effect-size synthesis versus directional fallback where both are possible;
4. fixed-count gates versus the frozen risk-calibrated policy;
5. random/confidence/disagreement versus influence-aware audit allocation; and
6. full system versus removal of study grouping, grounding verification, and abstention.

GEPA may optimize text components on development data. It may not change scientific
hypotheses, moderator families, gold labels, calibration policy, or test decisions after
their respective freezes.

## Claim-reporting rule

Every reported number must identify whether it comes from a fixture, simulation, cached
benchmark, live model run, or blinded human adjudication. Missing experimental cells
remain explicitly missing. Software contracts and passing tests establish engineering
behavior; they do not establish scientific accuracy.
