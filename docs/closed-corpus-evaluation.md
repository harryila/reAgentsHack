# Closed-corpus evaluation and human audit

This workflow separates three questions that must not be collapsed into one score:

1. Did the system retrieve the frozen included-paper set?
2. Did it emit at least one accepted target finding for each included paper?
3. Did its final conclusion agree with the frozen benchmark annotation?

The implementation is in
[`closed_corpus.py`](../src/literature_multiverse/closed_corpus.py). It is offline and
does not call a provider.

## Evaluator contract

The private gold JSONL contains one `ClosedCorpusGoldQuestion` per question:

```json
{"question_id":"q-1","split":"test","gold_paper_ids":["paper-a","paper-b"],"gold_conclusion":"positive"}
```

The prediction JSONL may contain up to three named arms for each question:

- `system`: ordinary system retrieval, extraction, and synthesis.
- `oracle_corpus`: the evaluator replaces retrieval with the exact gold paper set;
  extraction and synthesis remain system outputs.
- `oracle_extraction`: the evaluator supplies the exact gold corpus and paper-level
  extraction coverage; final synthesis remains a system output.

Oracle arms are evaluator interventions. The evaluator rejects an oracle-corpus row
whose retrieved IDs are not exactly the gold set and rejects an oracle-extraction row
whose extracted IDs are not exactly the gold set. The oracle-extraction helper never
copies the gold conclusion into the prediction.

Run a frozen evaluation with:

```bash
uv run python scripts/evaluate_closed_corpus.py \
  --gold data/cache/private-gold.jsonl \
  --predictions data/cache/frozen-predictions.jsonl \
  --output artifacts/paper/closed-corpus-evaluation.json
```

Missing retrieval is scored as zero recall and is distinct from an explicit empty
retrieval. `extracted_paper_ids` records papers that produced at least one accepted
target finding. Every gold included paper remains in end-to-end extraction recall.
Consequently, a retrieved eligible paper with zero findings is a miss, not a removed
denominator. Conditional extraction yield among retrieved gold papers is reported
separately from end-to-end paper recall.

The public output is aggregate and metadata-only: it contains no question text,
article text, per-question labels, or paper identifiers.

## Confidence-blinded 60-paper packet

The paper-level audit pool includes every screened-in, successfully mapped paper. The
freezer first includes every paper that the pipeline marked eligible—including papers
with zero findings—and then fills the remaining slots by fixed-seed random sampling
from pipeline-ineligible papers. This gives a census of pipeline-positive papers plus
a false-negative sample; the strata must be analyzed separately rather than treated as
a prevalence-weighted random sample.

The reviewer sees the research question, inclusion criteria, source lines, the emitted
finding fields, and blank paper-level and per-finding review forms. The closed reviewer schema has no model
confidence, risk, disagreement, influence, cost, priority, score, rank, or selection
stratum. System outputs are visible because reviewers must adjudicate them. Article
identity may still be apparent from the article text, so this is confidence/selection
blinding—not author or source blinding.

Two identical decision templates are emitted for independent reviewers. The identity
key and selection strata are stored separately. Generate the current packet with:

```bash
uv run python scripts/prepare_human_review_packet.py \
  --question-config configs/questions/antiox-training.yaml \
  --papers data/processed/antiox-training/papers.parquet \
  --findings data/processed/antiox-training/findings.parquet \
  --source-lines data/raw/map/antiox-training/source_lines.json \
  --output-dir data/cache/human-audit/antiox-training-60-v2 \
  --sample-size 60 \
  --seed 20260827
```

The packet manifest hashes the blank templates. Reviewers therefore save completed
copies instead of editing those files in place. Version-2 decision rows require a
positive `review_minutes` value measured after each review; estimated model-side cost
is not accepted as human time. Validate readiness or completed copies with:

```bash
uv run python scripts/evaluate_human_review.py \
  --manifest data/cache/human-audit/antiox-training-60-v2/manifest.json \
  --reviewer-a data/cache/human-audit/antiox-training-60-v2/completed-a.private.jsonl \
  --reviewer-b data/cache/human-audit/antiox-training-60-v2/completed-b.private.jsonl \
  --conflicts-output data/cache/human-audit/antiox-training-60-v2/adjudicator.private.jsonl \
  --output artifacts/diagnostics/antiox-human-review.json
```

When every scientific field agrees, the shared decision is used. Any paper-level or
finding-level disagreement is withheld until a third adjudicator completes the exact
conflict ledger. The public output contains only aggregate agreement, stratum-specific
counts, Wilson intervals, measured-time summaries, and hashes of the private inputs;
it contains no article text, paper identifiers, or audit-unit identifiers. Pooled
accuracy is explicitly diagnostic because pipeline-positive strata are censused while
pipeline-negative papers are sampled.

The packet contains article text and stays under ignored `data/cache/`. Every private
file is resolved inside the packet directory, rejected if missing or symlinked, and
rehashed before only its metadata-only hash and counts are copied to the public audit
artifact.

## Frozen local feasibility result

[`closed-corpus-local-audit.json`](../artifacts/paper/closed-corpus-local-audit.json)
was produced without network or API calls:

```bash
uv run python scripts/audit_local_corpora.py --force
```

Version 3 self-hashes the complete metadata payload and binds the audit CLI, its
direct/transitive project modules, `pyproject.toml`, and `uv.lock`. These unkeyed hashes
provide reproducibility and tamper evidence, not signatures, freshness, authorship, or
rollback protection.

The result establishes these boundaries:

- **MetaSyn:** 158 development, 161 calibration, and 86 official test reviews are
  leakage-separated by connected paper/review components. The revision-pinned official
  corpus contains 140,585 article rows and all 6,576 distinct released matched-paper IDs
  required across those splits. A real retrospective lexical retrieval study is therefore
  runnable and complete against the released matched-paper subset. Its development-selected
  RRF ranker reaches calibration macro Recall@200 of 0.682748 (component-clustered 95%
  interval [0.630194, 0.733980]); this is not exhaustive eligible-study recall and the
  historically opened official test is not scored. Within the frozen top 200, a
  development-selected protocol-aware logistic reranker increases calibration
  question-macro absolute recall at depth 50 from 0.476129 to 0.523243, a paired increase
  of 0.047114 (95% interval [0.024178, 0.071360]). It still cannot recover articles absent
  from the top 200, treats nonmatched candidates as implicit negatives, and is not a
  protocol-screening accuracy result. Separately, the cached fixed-Positive question-only
  control answered 86/86 and matched 42/86 frozen review directions (0.488372), emitted no
  retrieval IDs, and performed no extraction; its own retrieval and extraction recall are
  therefore zero. Both oracle arms remain `not_run`.
- **Evidence Inference 2.0:** 4,454 full texts are local. The converted
  train/development/test sets contain 4,371/522/524 examples over 1,477/192/191 papers,
  with zero paper overlap. This is a single-paper extraction benchmark and has no
  review-level included-corpus labels, so retrieval recall is not identifiable.
  The cached diagnostic flag is emitted only after the self-hashed public receipt
  audit passes its contract. That audit finds 10 clean common development receipts,
  two missing mutation responses, and excludes the archived trace scalar fail closed;
  it is not a GEPA-improvement result.
- **Antiox training corpus:** 646 papers are screened in and successfully mapped. The
  pipeline marks 19 eligible and emits 28 accepted findings from 10 of them; the other
  9 eligible papers emit zero findings. The 60-paper packet contains all 19
  pipeline-eligible papers (10 with findings and 9 without) plus 41 fixed-seed random
  pipeline-ineligible papers. Human decisions are blank.

No closed-corpus end-to-end accuracy, exhaustive-eligibility retrieval recall,
human-accuracy, or real calibration claim follows from this artifact. MetaSyn now supports
retrospective retrieval agreement with its released matched-paper subset, but its typed
extractor and end-to-end synthesis arms are not connected. Antiox retrieval recall requires
an external gold included-study set; accuracy and calibration require completed independent
human adjudication.
