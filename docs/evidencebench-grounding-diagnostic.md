# EvidenceBench grounding diagnostic

This diagnostic evaluates one narrow production prerequisite: given a biomedical
hypothesis and all sentences from its source paper, can a frozen retriever place
sentences covering the annotated evidence aspects near the top? It does **not** evaluate
screening, numerical effect extraction, meta-analysis, claim correctness, adaptive
auditing, or calibrated release.

## Evidence status

The original EvidenceBench release contains 426 questions split into 96 training, 37
development, and 293 test questions. The pinned upstream repository describes
all-aspect Evidence Retrieval at 10 and Results-aspect Evidence Retrieval at 5. The
test split is distributed under CC-BY; training and development are CC-BY-NC-SA. The
exact upstream commit, file hashes, row counts, licenses, method roster, metrics, and
bootstrap protocol are frozen in
`configs/benchmarks/evidencebench-grounding-v1.json`.

The test file is publicly accessible, not a secret evaluator. The project therefore
labels the output `retrospective_public_test_diagnostic`. The staged runner records the
narrower chronology that this method roster and development-set selection were frozen
before this project's semantic test-label access, but that chronology was not externally
preregistered and is not presented as contamination-proof.

## Frozen result

The staged `final-v1` run passed exact replay over all 293 questions from 284 distinct upstream
paper IDs. The development-selected method was `rrf-word-char-v1`. On test it achieved:

- all-aspect Recall@10: **0.375113 [0.341668, 0.409161]**;
- Results-aspect Recall@5: **0.253415 [0.214730, 0.294222]** over the 288 questions with at least
  one Results aspect.

The selected method exceeded the deterministic-random control by
**0.143761 [0.104823, 0.181408]** on all-aspect Recall@10 and
**0.143397 [0.099923, 0.187581]** on Results Recall@5. It also exceeded BM25 and word TF-IDF on
both metrics. The broad all-aspect paired difference versus character TF-IDF was inconclusive
and compatible with small differences in either direction; character TF-IDF was better on Results
coverage by **0.024045 [0.000581, 0.049274]** when expressed as character-TFIDF minus fusion.

The position-only first-sentences control produced a revealing reversal: it reached
**0.495027 [0.457227, 0.532733]** on broad all-aspect Recall@10, exceeding the selected method by
**0.119914 [0.075558, 0.163036]**, but reached only
**0.093968 [0.065200, 0.125913]** on Results Recall@5. The selected method's Results advantage was
**0.159447 [0.110563, 0.210588]**. This suggests that broad aspect recall rewards background and
design information concentrated near the beginning of papers, whereas the Results-restricted
metric better tests outcome-evidence localization. The intervals are descriptive and unadjusted
across five prespecified paired comparisons; no family-wise significance claim is made.

The public summary self-hash is
`37efd76dcac1a0b0347dab505e40be8b70a362862d8c49d488bedd07cb8f096b`; the exact-replay receipt
self-hash is `a086a1b0e56190adb15c308f640eeb7d232587141a7bc05345801616cb029cca`.
Both are aggregate-only and cross-bound to the frozen plan, materialization, predictions,
implementation, protocol, runtime, and pinned test bytes.

## Frozen methods and metrics

Four deterministic lexical methods are selectable on the development split: BM25,
word TF-IDF, character TF-IDF, and reciprocal-rank fusion of the two TF-IDF rankings.
The development rule maximizes mean all-aspect recall@10, then Results-aspect recall@5,
then ascending method identifier. First-sentence and deterministic-random rankings are
frozen controls and cannot be selected.

The primary test metric is mean all-aspect recall@10 over all 293 questions. The
secondary metric is mean Results-aspect recall@5 over only questions with at least one
Results aspect; the eligible denominator is reported explicitly. Uncertainty uses a
5,000-replicate percentile bootstrap over upstream `paper_id` clusters. The public
summary includes paired cluster-bootstrap differences between the development-selected
method and every frozen comparator.

## Label firewall

The runner enforces five immutable stages:

1. `prepare` accepts a development path and an expected test digest, but its API has no
   test path. It freezes aggregate development scores, the selected method, exact source
   closure, runtime versions, protocol hash, and test-file digest.
2. `materialize` first revalidates that frozen environment, then verifies the pinned test
   bytes and separates them into a visible projection and private gold. Visible rows
   contain only question ID, paper ID, hypothesis, candidate sentences, and sentence
   types. Private rows contain IDs, sentence counts, and aspect mappings, but no
   hypotheses or candidate text.
3. `predict` reads only the visible projection and writes all frozen rankings plus a
   self-hashed prediction receipt.
4. `score` validates the prediction receipt, prediction bytes, plan, materialization
   receipt, implementation, protocol, and runtime **before** opening private gold. It
   emits an aggregate-only public summary with no question IDs, paper IDs, aspect IDs,
   sentences, rankings, or row metrics.
5. `audit` exactly replays every stage and emits a content-silent, self-hashed audit
   receipt. Any source, dependency lock, runtime, config, input, prediction, or result
   drift fails closed.

Scientific artifacts are immutable: the runner has no force/overwrite flag. Use a fresh
run directory for a new execution.

## Reproduction

Download the three pinned upstream JSON files from the commit in the protocol config.
The runner itself never checks them into the repository; visible inputs, private gold,
and row-level predictions belong under ignored `data/cache/` storage.

```bash
.venv/bin/python scripts/run_evidencebench_diagnostic.py prepare \
  --development /path/to/evidencebench_dev_set.json \
  --plan data/cache/evidencebench-grounding-v1/plan.json

.venv/bin/python scripts/run_evidencebench_diagnostic.py materialize \
  --plan data/cache/evidencebench-grounding-v1/plan.json \
  --test /path/to/evidencebench_test_set.json \
  --visible data/cache/evidencebench-grounding-v1/visible-test.json \
  --gold data/cache/evidencebench-grounding-v1/private-test-gold.json \
  --receipt data/cache/evidencebench-grounding-v1/materialization-receipt.json

.venv/bin/python scripts/run_evidencebench_diagnostic.py predict \
  --plan data/cache/evidencebench-grounding-v1/plan.json \
  --materialization-receipt data/cache/evidencebench-grounding-v1/materialization-receipt.json \
  --visible data/cache/evidencebench-grounding-v1/visible-test.json \
  --predictions data/cache/evidencebench-grounding-v1/predictions.json \
  --receipt data/cache/evidencebench-grounding-v1/prediction-receipt.json

.venv/bin/python scripts/run_evidencebench_diagnostic.py score \
  --plan data/cache/evidencebench-grounding-v1/plan.json \
  --materialization-receipt data/cache/evidencebench-grounding-v1/materialization-receipt.json \
  --gold data/cache/evidencebench-grounding-v1/private-test-gold.json \
  --predictions data/cache/evidencebench-grounding-v1/predictions.json \
  --prediction-receipt data/cache/evidencebench-grounding-v1/prediction-receipt.json \
  --output artifacts/diagnostics/evidencebench-grounding-v1/summary.json

.venv/bin/python scripts/run_evidencebench_diagnostic.py audit \
  --development /path/to/evidencebench_dev_set.json \
  --test /path/to/evidencebench_test_set.json \
  --plan data/cache/evidencebench-grounding-v1/plan.json \
  --visible data/cache/evidencebench-grounding-v1/visible-test.json \
  --gold data/cache/evidencebench-grounding-v1/private-test-gold.json \
  --materialization-receipt data/cache/evidencebench-grounding-v1/materialization-receipt.json \
  --predictions data/cache/evidencebench-grounding-v1/predictions.json \
  --prediction-receipt data/cache/evidencebench-grounding-v1/prediction-receipt.json \
  --summary artifacts/diagnostics/evidencebench-grounding-v1/summary.json \
  --receipt artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json
```

Upstream sources: [EvidenceBench repository](https://github.com/EvidenceBench/EvidenceBench/tree/bf1d9633c694381c7b016fd56ee9f95f48593cc3)
and [COLM 2025 paper](https://openreview.net/forum?id=lEQnUI5lEA).
