# MetaSyn protocol-aware screening reranking study

This is a staged, provider-free study of whether protocol-aware features can move
already-retrieved included studies earlier in the screening queue. It does not test a
new retriever and does not claim to identify every scientifically eligible article.

The study consumes only the frozen top-200 reciprocal-rank-fusion (RRF) candidates
from `data/cache/metasyn/retrieval-study-v1/`. It uses the pinned 140,585-article
MetaSyn corpus at revision `c8fa07d89c44093d623f9a213c6bf070f40ab960`.

## Staged protocol

Run the stages separately:

```bash
uv run python scripts/run_metasyn_screening_study.py prepare
uv run python scripts/run_metasyn_screening_study.py fit
uv run python scripts/run_metasyn_screening_study.py evaluate
uv run python scripts/run_metasyn_screening_study.py validate
```

`prepare` opens the development and calibration model inputs, the frozen RRF rankings,
and candidate titles/abstracts. It freezes 17 deterministic numeric features for every
question-document pair. It does not open a label artifact, the source review table, or
any official-test input or label. The receipt also freezes the complete candidate
family, grouped-CV rule, selection metric, evaluation depths, bootstrap method, and
seed before development labels are materialized.

The features are frozen RRF/BM25/TF-IDF rank transformations plus token-overlap
statistics between candidate title/abstract and the research-question/PIECO fields.
No review conclusion, effect direction, effect size, matched identifier, or evaluator
label is a feature.

`fit` materializes only development rows from the official-train Parquet using the
manifest review-ID predicate. It evaluates three prespecified candidates with
five-fold, connected-component-disjoint cross-validation:

- the original RRF order;
- an L2-regularized, class-balanced logistic reranker; and
- a class-balanced histogram-gradient reranker with nonnegative monotonic constraints.

The development selection score is the unweighted mean of question-macro absolute
recall at depths 10, 20, 50, and 100. Recall@200 is excluded from selection because all
rerankers have exactly the same top-200 set. Ties are resolved by ascending candidate
name. After selection, the winner is fit on all development questions and its
calibration ordering is hash-frozen before calibration labels are materialized.

`evaluate` materializes calibration labels, scores only that winner and the
prespecified RRF comparator, and writes an aggregate-only public result. Uncertainty
uses 20,000 paired percentile bootstrap replicates over MetaSyn connected review
components. The same sampled components are used for both methods and every depth.

The official-test model-input JSONL, official-test source Parquet, and shared
all-split evaluator-label JSONL are never opened by this study. This is only a
within-study access statement: official-test labels were historically opened elsewhere
in this repository, so they are not a pristine holdout and are not evaluated here. Development and
calibration rows physically share the official-train Parquet, so their access boundary
is logical (predicate-materialized rows), not a claim that unrelated Parquet pages
were physically unread. Automated tests make the test files and shared evaluator file
inaccessible and execute all three stages successfully.

## Estimands and denominators

Two denominators are reported because they answer different questions:

- **Absolute matched-subset recall** divides retained matched references by every
  released MetaSyn matched identifier for the review. Questions with no matched item
  in RRF top-200 remain in the macro denominator with recall zero.
- **Conditional survival** divides by matched identifiers already present somewhere
  in RRF top-200. Questions with zero retrievable matched identifiers have an undefined
  conditional fraction, are counted explicitly, and are excluded only from that
  conditional macro denominator.

Every non-matched top-200 candidate is an implicit training negative. MetaSyn's
released matching is not an exhaustive eligibility annotation, so some implicit
negatives may be eligible. This prevents a screening-precision or eligibility-accuracy
claim; the defensible estimand is survival of the released matched subset under a
smaller screening workload.

## Frozen result

The logistic reranker won development component-disjoint cross-validation with a
selection score of 0.4154; the monotonic histogram model scored 0.4136 and RRF scored
0.3794. The winner was then evaluated on all 161 calibration questions (152 connected
components, 2,329 released matched references).

| Documents screened per review | Macro absolute recall, logistic | Macro absolute recall, RRF | Paired macro delta (95% CI) | Micro absolute recall, logistic / RRF | Conditional macro survival, logistic / RRF |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.2498 | 0.2327 | +0.0170 [-0.0060, 0.0413] | 0.1511 / 0.1314 | 0.3341 / 0.2940 |
| 20 | 0.3637 | 0.3516 | +0.0120 [-0.0138, 0.0374] | 0.2422 / 0.2224 | 0.4866 / 0.4530 |
| 50 | 0.5232 | 0.4761 | +0.0471 [0.0242, 0.0714] | 0.3851 / 0.3469 | 0.7168 / 0.6417 |
| 100 | 0.6156 | 0.5918 | +0.0238 [0.0036, 0.0415] | 0.5006 / 0.4646 | 0.8747 / 0.8253 |
| 200 | 0.6827 | 0.6827 | 0.0000 [0.0000, 0.0000] | 0.5766 / 0.5766 | 1.0000 / 1.0000 |

At depth 50, the winner retained 897 matched references versus 808 for RRF. Its paired
micro-recall delta was +0.0382 [0.0251, 0.0526], and its paired conditional macro
survival delta was +0.0751 [0.0414, 0.1093]. Eight calibration questions had no released
matched reference anywhere in RRF top-200, so no within-set reranker could help them.

The result is not uniformly better on every question-level endpoint. At depth 50, the
full-inclusion rate was 0.1615 for the logistic reranker and 0.1677 for RRF. At depth
100, zero-retained rates were 0.0745 and 0.0621, respectively. This supports a mean
matched-reference-survival improvement, not dominance, eligibility accuracy, or a
guarantee that every review is helped.

## Claim boundary

Development, calibration, and official-test labels were historically opened somewhere
in this repository before this protocol, so these results are retrospective and there
is no pristine MetaSyn final holdout. This study evaluates calibration only; it does not
re-open or score official test. The enforced stage order cannot eliminate analyst memory or indirect
design feedback from those earlier openings. Candidate abstracts can encode language
correlated with inclusion, but no gold/conclusion/effect field is supplied directly.
The result does not validate retrieval beyond top-200, protocol-faithful exclusion
reasons, extraction, synthesis, or scientific truth.

Identifier-bearing features, fold assignments, rankings, and stage receipts remain in
ignored `data/cache/metasyn/screening-study-v1/`. The tracked
`artifacts/diagnostics/metasyn-screening-study-v1.json` is self-hashed and aggregate
only: it contains no question/component/article identifiers, titles, abstracts,
protocol text, per-question labels, or absolute paths.
