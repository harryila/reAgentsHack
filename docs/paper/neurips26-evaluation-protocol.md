# Literature Multiverse — NeurIPS 2026 Evaluation Protocol

**Frozen intent date:** 2026-08-26  
**Target:** AI for Science workshop, Verifier Systems track  
**Status:** prospective protocol for all experiments started after this file was written

## Central claim

Literature Multiverse is a provenance-grounded selective verifier that releases a
condition-dependent scientific claim only when the estimated risk of an unsupported claim is below
a predeclared tolerance; otherwise it abstains with an auditable reason.

The paper does **not** claim to verify scientific truth. Its target loss is disagreement with a
frozen benchmark label or blinded expert adjudication over a specified retrieved corpus.

## Primary hypothesis

At matched unsupported-claim risk, condition-aware synthesis with a calibrated release policy has
higher question-level coverage than unconditional synthesis and fixed paper-count gates.

## Secondary hypotheses

1. GEPA-optimized extraction improves exact evidence-span support and structured-field accuracy over
   the handwritten prompt at equal model and evaluation budgets.
2. Effect-size meta-regression has lower false-moderator discovery and better held-out predictive
   score than significance-derived direction counting when compatible effect estimates exist.
3. A source-agnostic open harvester preserves the provenance and terminal-ledger invariants of the
   existing Paperclip-backed pipeline.

## Units and leakage controls

- Prompt optimization unit: paper, grouped so that no paper or duplicate cohort crosses a split.
- Statistical unit: independent paper/cohort, never an extracted row.
- Calibration unit: complete question–corpus, never a paper, finding, fold, or bootstrap draw.
- Splits: development → calibration → test. Test examples are excluded from fitting, prompt
  optimization, threshold selection, and model selection. Record-level test labels are opened by
  the evaluator only after the relevant pipeline artifact freezes; public schemas and aggregate
  benchmark statistics may be inspected and must be disclosed.
- Duplicate DOI/PMID/preprint/publication/cohort families remain in one split.
- Simulation and real-review corpora are distinct calibration populations and cannot be used to
  transfer a formal risk guarantee between them.

## Output semantics

Study-level records separate:

- signed effect estimate;
- standard error or confidence interval and sample size;
- observed point direction (`higher`, `lower`, or `unknown`);
- reported significance as metadata only;
- equivalence, which requires a predeclared equivalence margin;
- exact source span and study conditions.

Claim-level verdicts are `supports_claim`, `contradicts_claim`, `condition_dependent`,
`insufficient_evidence`, or `not_evaluable`. A moderator proposed and tested on overlapping papers
is reported only as an `exploratory_condition_signal`.

## Primary metrics

- question-level accepted-claim error and coverage;
- risk–coverage curve and area under that curve;
- worst-domain accepted-claim error;
- retrieval recall against a frozen included-study set;
- screening precision/recall;
- evidence-span support and structured-field accuracy;
- effect-estimate error and interval coverage;
- false-moderator rate and power in planted simulations;
- cost and wall-clock time.

## Baselines and ablations

- unconditional majority/consensus synthesis;
- current fixed G3/M4 gates;
- raw model confidence or self-consistency;
- bootstrap-stability-only gate;
- calibrated gate without GEPA;
- handwritten prompt vs GEPA at equal budgets;
- oracle corpus and oracle extraction;
- effect-size synthesis vs sign-only fallback;
- no paper grouping and no provenance verification (diagnostic ablations only).

## Formal calibration rule

Candidate release thresholds use simultaneous one-sided Clopper–Pearson upper bounds with a
Bonferroni correction across the predeclared threshold family. A threshold is eligible only when
its upper risk bound is at most `alpha`; otherwise the policy abstains on every question. Test-set
intervals are descriptive and are computed once after policy freeze.

## Claims forbidden without new evidence

- “domain-general” from a one-domain evaluation;
- “human audit” for model-agent review;
- “no effect” from a nonsignificant comparison;
- causal language for study-level moderator associations;
- a real-world false-claim guarantee calibrated only on simulations;
- “percentage disagreement” for normalized entropy;
- “preregistered” unless this protocol is externally time-stamped before the affected experiment.
