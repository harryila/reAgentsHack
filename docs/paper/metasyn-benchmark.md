# MetaSyn offline benchmark protocol

This adapter turns the downloaded MetaSyn review metadata into a hash-locked,
review-level benchmark for two tasks:

1. predicting the published review annotation `Positive`, `Negative`, or `Mixed`; and
2. recovering the review's matched MetaSyn corpus identifiers.

It does **not** reconstruct study-level effect estimates, rerun a random-effects model,
or establish scientific truth. A direction error is disagreement with the frozen
MetaSyn review-level annotation. That distinction must remain explicit in the paper.

## Frozen split

The generated paper artifact uses seed `20260826` and the source hashes recorded in
[`manifest.json`](../../artifacts/paper/metasyn-benchmark/manifest.json):

| Split/status | Reviews | Positive | Negative | Mixed | NR |
|---|---:|---:|---:|---:|---:|
| Development | 158 | 68 | 34 | 55 | 1 |
| Calibration | 161 | 77 | 27 | 56 | 1 |
| Official test | 86 | 42 | 15 | 29 | 0 |
| Quarantined official train | 17 | — | — | — | — |

Every one of the 86 official test reviews remains in test. The two official parquet
files share 112 matched corpus identifiers across their nominal boundary. Fifteen
official-train reviews overlap test directly; component closure adds two more linked
train reviews. All 17 are quarantined. The remaining train reviews are connected by
exact shared matched-paper ID, shared source-review ID, normalized exact title, or
normalized exact research question. Whole components are assigned by a seeded hash,
without consulting direction labels. Consequently no review component or matched
corpus identifier crosses development, calibration, and test.

This is stricter than reporting the official split without checking paper overlap. The
quarantined rows are not silently discarded: their IDs, component IDs, reasons, and
linked test review IDs are recorded in the manifest.

## Leakage boundary

Model-facing JSONL contains only the review/question IDs, research question, and
PICO/PECO fields. Split membership comes from the separate filename; component IDs are
kept out of model inputs because they were derived partly from gold matched-paper
links. The evaluator file and manifest are separate and hash-locked. The evaluator is
named `evaluator_labels.private.jsonl` to mark an access boundary, not because it
contains credentials. A runner should render only the question/PICO fields to a model,
using IDs solely to join the returned prediction.

Optimizer/model code must use the split-scoped loader, which verifies and opens only
the requested input file. It does not hash or open evaluator labels or any other split:

```python
from pathlib import Path

from literature_multiverse.metasyn_benchmark import load_metasyn_inputs

development = load_metasyn_inputs(
    Path("artifacts/paper/metasyn-benchmark/manifest.json"),
    split="development",
)
```

`load_metasyn_manifest` is intentionally evaluator-side: it verifies the complete
bundle and therefore must not be called from an optimizer or provider-facing path.

The following never appear in model-facing rows: abstract, conclusion summary,
conclusion paragraph, effect direction/category/type/value, significance, confidence
or heterogeneity statistics, key insights, extracted/raw included-study titles, and
gold matched corpus identifiers. Tests enforce the exact allowlist.

The included [MetaSyn license notice](../../artifacts/paper/metasyn-benchmark/METASYN_LICENSE.txt)
states that its project-authored annotations are MIT licensed, while third-party
metadata/text retains upstream terms. The generated model-facing inputs deliberately
omit titles, abstracts, persistent article identifiers, and source-review excerpts.
The private evaluator retains only MetaSyn's integer corpus-row keys needed to score
retrieval, not DOI/PMID/title metadata or article text.

## Prepare

Preparation is offline and makes no provider or network calls:

```bash
uv run python scripts/metasyn_benchmark.py prepare \
  --train-parquet data/cache/metasyn/reviews-train.parquet \
  --test-parquet data/cache/metasyn/reviews-test.parquet \
  --output-dir artifacts/paper/metasyn-benchmark \
  --seed 20260826 \
  --calibration-fraction 0.5
```

Changing either parquet, any generated JSONL, or the manifest is detected by SHA-256
verification before evaluation.

## Frozen fixed-direction control

The paper artifact includes a deliberately trivial constant-direction control. Its
class was selected from the development labels before this control's staged evaluator
invocation opened the official-test labels: `Positive` is the most frequent labeled
development class (68/157). This is the usual "majority-class baseline" convention,
although 68/157 is technically a plurality rather than an absolute majority. The
control is retrospective and non-pristine because those test labels had been opened
historically elsewhere in the repository; staging does not restore holdout status.

Prediction generation is label-blind and split-scoped. It calls only
`load_metasyn_inputs` for the explicitly named split, uses `review_id` solely as an
evaluator join key, and emits no retrieval IDs or risk features:

```bash
uv run python scripts/metasyn_benchmark.py freeze-fixed-direction \
  --manifest artifacts/paper/metasyn-benchmark/manifest.json \
  --split test \
  --direction Positive \
  --selection-note "Development-set most-frequent class control ('majority-class' baseline convention): Positive was 68/157 labeled development reviews; this is technically a plurality, not an absolute majority." \
  --output-dir artifacts/paper/metasyn-fixed-positive-test
```

The immutable freeze receipt records the manifest and named-input hashes, selected
class, config and prediction hashes, and `labels_opened=false` for that prediction
stage. This control's evaluator invocation opened the private labels only afterward,
in a separate process:

```bash
uv run python scripts/metasyn_benchmark.py evaluate \
  --manifest artifacts/paper/metasyn-benchmark/manifest.json \
  --predictions artifacts/paper/metasyn-fixed-positive-test/predictions.jsonl \
  --output artifacts/paper/metasyn-fixed-positive-test/evaluation.json
```

On all 86 official-test reviews, the constant control obtains 42/86 correct,
accuracy `0.4883720930`, and macro F1 `0.21875`. Retrieval coverage is zero by
construction. The frozen predictions, receipt, complete per-review evaluation, and
interpretation notes are in
[`metasyn-fixed-positive-test`](../../artifacts/paper/metasyn-fixed-positive-test/).
This is a question-level constant control only. It is **not** an end-to-end retrieval
result, calibrated system evidence, study-level meta-analysis evidence, or a claim of
scientific correctness.

## Prediction contract and evaluation

Predictions are JSONL with one row per supplied review. Missing rows are allowed and
are scored as missing output. An explicit empty retrieval list is distinct from an
omitted retrieval result.

```json
{
  "prediction_version": "1",
  "review_id": 23,
  "predicted_direction": "Positive",
  "retrieved_corpus_ids": [4556, 4557],
  "risk_features": {
    "bootstrap_instability": 0.12,
    "grounding_failure_fraction": 0.03
  }
}
```

`predicted_direction` accepts `Positive`, `Negative`, `Mixed`, `NR`, `Abstain`, or a
missing value. Direction reporting includes coverage, missing/NR/abstention counts,
accuracy on answered reviews, strict accuracy with unanswered reviews counted as
errors, macro F1, per-class results, and a confusion matrix. Retrieval reporting
includes missing and explicit-empty counts, macro recall on supplied outputs, strict
macro/micro recall with missing outputs counted as zero, and precision diagnostics.

```bash
uv run python scripts/metasyn_benchmark.py evaluate \
  --manifest artifacts/paper/metasyn-benchmark/manifest.json \
  --predictions path/to/frozen-predictions.jsonl \
  --output artifacts/paper/metasyn-evaluation.json
```

Evaluation defaults to the official `test` split held out from model optimization. Its schema and
aggregate label statistics were inspected while building the benchmark, so it is not described as
pristine or untouched. Development/calibration
diagnostics require an explicit `--split development` or `--split calibration`; use
`--split all` only for clearly labeled descriptive diagnostics, never as the held-out
paper result.

When—and only when—real frozen predictions include finite `risk_features`, the same
command can emit question-level calibration rows:

```bash
uv run python scripts/metasyn_benchmark.py evaluate \
  --manifest artifacts/paper/metasyn-benchmark/manifest.json \
  --predictions path/to/frozen-predictions.jsonl \
  --output artifacts/paper/metasyn-evaluation.json \
  --risk-examples-output artifacts/paper/metasyn-risk-examples.jsonl \
  --pipeline-sha256 <64-lowercase-hex-code-hash>
```

No predictions, risk features, corpora, or losses are fabricated. Gold `NR`, missing
system directions, predicted `NR`, explicit abstentions, predictions without risk
features, and empty/missing retrieved corpora are omitted from `RiskExample` output.
Each row's `paper_ids` records the system's actual retrieved corpus rather than the gold
matched set; cross-split retrieval overlap fails the calibration integrity check. The
population domain is labeled `metasyn_systematic_reviews`, without implying a validated
domain taxonomy. Its binary loss means disagreement with the frozen benchmark direction
and uses `label_source="benchmark_annotation"`; it must not be described as expert
adjudication or scientific-truth error.

## Known limitations

- The model-facing research-question/PICO fields are MetaSyn annotations derived from
  source reviews. They exclude explicit results, but could retain framing from the
  original authors; this is a benchmark limitation worth stating.
- Exact identifiers and exact normalized text catch auditable links, not every semantic
  near-duplicate or shared cohort.
- Retrieval recall is against MetaSyn's matched subset, not all scientifically eligible
  literature.
- Review-level direction compresses multi-outcome syntheses and cannot validate the
  numerical meta-analysis implementation. Study-level effect records are required for
  that separate evaluation.
