# Local real-data benchmark suite

Run the complete offline suite from the repository root:

```bash
uv run python scripts/run_local_benchmarks.py --force
```

For CI or a checkout without the locally licensed corpora, validate only the static
hash/access contract:

```bash
uv run python scripts/run_local_benchmarks.py --contract-only
```

The suite manifest is `configs/benchmarks/local-suite-v1.json`. The MetaSyn article
corpus is pinned to revision `c8fa07d89c44093d623f9a213c6bf070f40ab960`; its six
Parquet shard hashes and 140,585-row total are separately frozen in
`configs/benchmarks/metasyn-corpus-c8fa07d.json`.

## Retrieval baseline

The runner fits scikit-learn TF-IDF on corpus title + abstract only, then transforms
the model-facing research-question and PI/ECO fields. Configuration is fixed at
English stopwords, 1--2 grams, `min_df=2`, 200,000 maximum features, sublinear term
frequency, L2 normalization, and float32. It excludes each review's
`source_review_corpus_ids` before selecting 200 articles. Ties are resolved by
descending score and then ascending corpus ID.

The corpus-only fit avoids the exploratory prototype's transductive fit on the query
set. Gold matched-paper IDs are opened only after predictions and their receipt have
been frozen. The current deterministic report records:

| Split | Reviews | Macro Recall@200 | Micro Recall@200 |
|---|---:|---:|---:|
| Development | 158 | 0.6630240171 | 0.6044663134 |
| Calibration | 161 | 0.6771717303 | 0.5727780163 |
| Combined | 319 | 0.6701643990 | 0.5896197948 |

These are retrospective local results: all MetaSyn labels have previously been
opened, so no split is a pristine final holdout. The primary runner deliberately does
not evaluate the 86-review test split. Recall is against MetaSyn's released matched
subset, not an exhaustive set of every scientifically eligible paper.

The aggregate metadata-only report is
`artifacts/benchmarks/local-suite-v1/benchmark-report.json`. Predictions contain only
review IDs and retrieved corpus IDs; the report contains no question text, article
text, or per-review gold labels.

## Licensing and missing data

The MetaSyn notice says its annotation license does not replace upstream terms for
article metadata, abstracts, or PMC-derived text. Evidence Inference 2.0 similarly
bundles article text without one dataset-wide redistribution license. Corpus payloads
therefore remain in the ignored local cache; only metadata, hashes, predictions, and
aggregate metrics are emitted. The runner also reports that this repository currently
has no top-level code license, which blocks a public software release but not local
evaluation.

If a required cache file is absent or its hash differs, the runner writes a structured
blocked report and exits with status 2. It never downloads a replacement or fabricates
a result.
