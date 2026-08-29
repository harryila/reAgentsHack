# MetaSyn fixed-`Positive` official-test control

This directory is one frozen, trivial question-level constant baseline. Every one of
the 86 official-test review IDs receives `predicted_direction="Positive"`; no question
content, retrieval IDs, or risk features are used or emitted.

`Positive` was chosen before test evaluation because it is the most frequent labeled
development class (68 of 157 non-`NR` reviews). This follows the conventional
"majority-class baseline" naming, but it is technically a plurality rather than an
absolute majority.

The access boundary is represented by two separate stages:

1. `predictions.jsonl` and `freeze_receipt.json` were generated through the
   split-scoped model-input loader. The receipt records `labels_opened=false`, the
   explicit class/config, and hashes of the manifest, named test input, and frozen
   predictions.
2. `evaluation.json` was produced afterward by a single, separate private-evaluator
   invocation. It records the complete per-review results and evaluator-side labels.

This is a retrospective, non-pristine control: the MetaSyn official-test labels were
historically opened elsewhere in this repository, and the staged boundary above does
not restore pristine-holdout status.

The direction result is 42/86 correct (`0.4883720930` accuracy; `0.21875` macro F1).
All 86 retrieval outputs are missing by design, so retrieval coverage and strict
recall are zero.

This artifact is not end-to-end retrieval evidence, calibrated system evidence,
study-level meta-analysis evidence, or a scientific-truth evaluation. Direction
errors mean disagreement with the frozen review-level MetaSyn annotation only.
