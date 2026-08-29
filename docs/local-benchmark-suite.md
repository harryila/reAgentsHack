# Local real-data benchmark suite

Run the configured MetaSyn retrieval and screening studies from the repository root:

```bash
uv run python scripts/run_local_benchmarks.py --force
```

`--force` is intentional: a scientific rerun must replace every staged retrieval and
screening receipt. Without it, the runner fails before execution if either private work
directory, either public summary, or the suite report already exists. Retrieval is never
silently reused while screening is recomputed. Colliding or nested retrieval/screening work
directories are rejected even with `--force`.

For CI or a checkout without the locally licensed corpora, validate only the static
hash/access contract:

```bash
uv run python scripts/run_local_benchmarks.py --contract-only
```

Contract-only mode reads the suite manifest itself but does not open any pinned local payload or
execute a study. It validates the exact corpus/split inventory, historical non-pristine access
states, candidate families, evaluation splits, license policy, safe relative paths, and syntactic
SHA-256 records. It is not evidence that locally cached payload bytes are present or correct.

The suite manifest is `configs/benchmarks/local-suite-v1.json` (current file SHA-256
`f481ed2539f6ada9d8c9fb540cc15856ed0cc3c64def335b0e56b3babf771592`). The MetaSyn article
corpus is pinned to revision `c8fa07d89c44093d623f9a213c6bf070f40ab960`; its six
Parquet shard hashes and 140,585-row total are separately frozen in
`configs/benchmarks/metasyn-corpus-c8fa07d.json`. That corpus-manifest file is itself a pinned
suite input, preventing a replacement manifest from silently redefining the shard set.

## Staged retrieval study

The primary retrieval diagnostic now has an enforced three-stage sequence:

1. freeze every candidate prediction for development and calibration without opening
   the evaluator-label artifact or official-test model inputs;
2. compare the fixed candidates on development and select the largest
   question-weighted macro matched-subset Recall@200 (ties sort by candidate ID); and
3. score only that frozen winner once on calibration.

The three prespecified candidates are corpus-only-fit TF-IDF, fixed streaming BM25,
and reciprocal-rank fusion (RRF). TF-IDF uses title + abstract, English stopwords,
1--2 grams, `min_df=2`, 200,000 maximum features, sublinear term frequency, L2
normalization, and float32. BM25 uses `k1=1.2`, `b=0.75`, `k3=8`, title/PICO boosts,
and the repository-frozen tokenizer. RRF uses the top 1,000 rankings from both inputs,
rank constant 60, and returns 200 records. Every method excludes the review's own
`source_review_corpus_ids`; score ties end with ascending corpus ID. No hyperparameter
grid is selected from either benchmark split.

The fixed real-data run produced:

| Stage and candidate | Reviews | Macro Recall@200 (component-bootstrap 95% interval) | Micro Recall@200 (95% interval) |
|---|---:|---:|---:|
| Development: BM25 | 158 | 0.648862 [0.594666, 0.705371] | 0.576079 [0.516258, 0.644660] |
| Development: TF-IDF | 158 | 0.663024 [0.609366, 0.718885] | 0.604466 [0.545364, 0.670721] |
| Development: RRF (selected) | 158 | 0.664857 [0.611034, 0.721013] | 0.600303 [0.539695, 0.668679] |
| Calibration: selected RRF only | 161 | 0.682748 [0.630194, 0.733980] | 0.576642 [0.500209, 0.656347] |

RRF beat TF-IDF on the prespecified development selection metric by only `0.001833`;
the paired component-bootstrap interval for that difference was
`[-0.007729, 0.012414]`. Selection follows the frozen point-estimate rule, but this is
not evidence of a reliably superior fusion method. On calibration, RRF recovered
1,343 of 2,329 released matched identifiers; 8 of 161 questions had zero recall.
Intervals use 20,000 deterministic paired resamples of the pre-split review components
(141 development clusters and 152 calibration clusters). They describe sampling
variation in this retrospective snapshot; they are not finite-sample guarantees.

The access boundary remains deliberately blunt: all development, calibration, and
official-test labels were opened before this study was designed. The staged procedure
prevents calibration feedback in this run, but cannot make calibration pristine. The
shared evaluator JSONL physically contains all splits, so an evaluation stage must scan
that file even though it retains and scores only its named split. The 86-review
official test is not evaluated. The pinned snapshot contains 422 source reviews
(336 official train and 86 official test); 17 train reviews linked to test components
are quarantined. Do not silently generalize the result to a later MetaSyn release with
a different review inventory.

Recall is against MetaSyn's released matched-paper subset, not an exhaustive set of
scientifically eligible papers. All 54 source-review exclusions were checked after
labels were opened; none overlapped a gold matched identifier. This result validates a
real, label-blind retrieval implementation against the pinned matched subset. It does
not validate protocol-faithful eligibility screening, extraction, meta-analysis, or
scientific truth.

Identifier-bearing rankings, predictions, and staged receipts remain under ignored
`data/cache/metasyn/retrieval-study-v1/`. CI enforces that the public local-suite directory tracks
only its aggregate `benchmark-report.json`; predictions and freeze receipts are forbidden there.
The tracked
`artifacts/diagnostics/metasyn-retrieval-study-v1.json` is aggregate-only, self-hashed,
and contains no question/article text, per-question/article identifiers, labels, or
absolute paths. Its file SHA-256 is
`b1570cb45e690e8f66a9250aedf2be768411df1e600a52a1ad05ca7024ac2fd5`.

## Protocol-aware screening reranking

After retrieval freezes all three candidates, screening consumes the prespecified frozen RRF
top-200 candidate—not an arbitrary or caller-selected ranking. Seventeen label-blind protocol and
lexical features are frozen before development labels are materialized. Three rerankers are then
compared with connected-component-disjoint development cross-validation, and only the winner is
evaluated once on calibration against the prespecified RRF comparator.

The selected logistic reranker retained 897 released matched references in the first 50 documents,
versus 808 for RRF. Question-macro absolute recall was 0.5232 versus 0.4761, a paired
component-bootstrap difference of +0.0471 [0.0242, 0.0714]. This is a retrospective
matched-subset-survival result. Non-matched candidates are implicit negatives and can include
unannotated eligible articles, so the result is not screening precision, eligibility accuracy, or
exhaustive included-study recall. The complete protocol and denominator definitions are in
[the screening study](metasyn-screening-study.md).

The aggregate-only screening summary has file SHA-256
`8853ed4578f10eac54755ec38f3b483a4659e5aa4a32e24486bf24295a365e99` and canonical payload
SHA-256 `b5fdc31a1b1c3b3430b77b904256712095a626f61c94ca2d611a37bc22400322`.
The integrated runner independently verifies both public self-hashes, the physical retrieval
freeze receipt, the freeze's canonical payload hash, the benchmark/corpus manifest hashes, and the
screening-to-retrieval freeze link before writing a result. Its final report is also canonically
self-hashed; the scientific-payload hash includes both physical summary hashes and both canonical
payload hashes. The current report file SHA-256 is
`6af096fcbb367cfc7619489162795fd8d3e3c909ca25ed563130ea2567f114c9`, its canonical report-payload
SHA-256 is `880222ea443b64660cb5af137acad160595e91147b2159816d237b31577e7bbf`, and its scientific-payload
SHA-256 is `3f69ded0b94bbef250ba3cb2c634163257b9656e3f2beb65aa98a930ae9606ad`.

The successful-run `network_calls: 0` claim is executable rather than only declarative for these
Python stages: `socket.socket.connect`, `connect_ex`, and `socket.create_connection` are disabled
during corpus verification, both studies, and artifact validation. The implemented study path has
no subprocess or alternate networking client. This is a process-level guard, not an
operating-system network sandbox.

Evidence Inference 2.0 and the antiox training corpus are hash-checked inventories in this runner;
they do not produce metrics here. The report lists them under
`inventory_only_no_metric_in_this_runner` so a `complete` MetaSyn run cannot be mistaken for a
complete evaluation of every cached corpus.

## Licensing and missing data

The MetaSyn notice says its annotation license does not replace upstream terms for
article metadata, abstracts, or PMC-derived text. Evidence Inference 2.0 similarly
bundles article text without one dataset-wide redistribution license. Those two article-payload
caches remain local and ignored; identifier-bearing predictions and receipts from the current
runner are emitted only to private ignored work directories.

That statement does **not** apply to the repository's legacy Antiox trees. They currently include
tracked PMC-derived source lines, raw command outputs, third-party abstracts and metadata, provider
responses, evidence quotes, and derived exports whose redistribution basis has not been
established. The code's MIT license does not grant rights to that material. The content-silent
[public-data rights audit](public-data-rights.md) classifies the exact Git-index inventory and its
strict pre-publication mode fails while any such collection remains unresolved.

If a required cache file is absent or its hash differs, the runner writes a structured
blocked report and exits with status 2. It never downloads a replacement or fabricates
a result.
