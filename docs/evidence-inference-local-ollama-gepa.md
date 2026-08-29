# Local-Ollama official-GEPA Evidence Inference study

This study is a staged, provider-free diagnostic of prompt optimization for Evidence Inference
2.0. It uses the official `gepa.optimize` API, the exact pinned local Ollama model for both task
and reflection calls, paper-level train/development selection, a frozen winner, and a single paired
test comparison.

It is not a pristine held-out experiment. Every Evidence Inference label in this checkout was
opened historically, and `llama3.2:1b` is a 1.2B-parameter diagnostic model. The public artifact
therefore cannot support a confirmatory or frontier-model claim even when its paired interval
excludes zero.

## Frozen contract

- model: `llama3.2:1b`
- model digest: `baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878`
- Ollama: `0.15.1`
- deterministic generation seed: `20260827`
- official GEPA: `0.1.4`
- optimization: an 800 logical-metric stopping threshold, 12-example reflection minibatches,
  Pareto selection with a hybrid instance/objective frontier; official GEPA checks the threshold
  between atomic evaluations, so the finalized artifact must report any batch-boundary overshoot
  rather than treating 800 as an exact realized count
- optimization population: whole papers selected by a label-blind SHA-256 ordering until at
  least 256 train and 96 development examples are included
- task context: the same frozen, label-blind Results passage projection used by the local EI
  diagnostic (at most six passages and 9,000 characters); official grounding still replays
  against the complete source-line object
- primary test estimand: winner-minus-seed direction-accuracy difference
- uncertainty: prompt-weighted, article-clustered percentile bootstrap with 20,000 replicates
- improvement rule: the winner must differ from the seed, the paired point difference must be
  positive, and the paired 95% lower bound must be above zero

The scalar GEPA objective is prespecified as 55% direction accuracy, 20% formal exact grounding,
10% official-schema validity, 10% generation success, 2.5% output-token efficiency, and 2.5%
latency-SLA success. Cost/runtime terms are deliberately small; direction and exact provenance
remain the scientific objectives. During train/development scoring only, a zero-scalar-weight
direction-distribution-fidelity objective is exposed to the hybrid Pareto frontier to make
single-label mode collapse visible without rewarding a merely balanced but incorrect classifier.
It is a diagnostic guardrail, not a correctness substitute, and it is never computed during input
selection or from test targets. The paired test instead reports article-clustered macro direction
recall as a correctness metric.

Ollama 0.15.1 was observed to terminate natively when its generation schema contained the official
Evidence Inference citation regex. Task generation therefore uses a row-specific, regex-free
schema whose citation enum contains only exposed line IDs. The official benchmark schema remains
the post-generation evaluator. Both schema algorithms and hashes are frozen and reported.

The passage projection reads only PICO replacements, source lines, and section metadata. It never
reads labels or annotations and never falls back to them when no Results passage survives. Because
projection can omit answer-bearing evidence, a failed row is a projection-plus-extraction failure,
not a pure extractor-error estimate.

## Stage boundary

Run each command separately from the repository root:

```bash
uv run --extra gepa python scripts/run_ollama_gepa_study.py prepare
uv run --extra gepa python scripts/run_ollama_gepa_study.py optimize
uv run --extra gepa python scripts/run_ollama_gepa_study.py test
uv run --extra gepa python scripts/run_ollama_gepa_study.py audit
```

`prepare` and `optimize` open only the manifest plus train/development payloads. The manifest
contains test membership metadata, which is stated explicitly, but these stages do not open or
hash the test JSONL. `optimize` freezes the exact seed, winner, GEPA result, checkpoint, model
identity, generation configuration, source implementation, and input hashes. Only `test` may open
the test JSONL, and it validates that complete freeze first.

The paired test alternates seed-first and winner-first order by a deterministic hash parity rule.
Arm-specific receipt namespaces prevent collisions even when GEPA retains the seed. Interrupted
runs resume from self-hashed private receipts; a replay consumes the same logical GEPA metric unit
without making a second physical model call. Completed test and summary stages are idempotent so
rerunning the command cannot become repeated test tuning.

Do not run the heavy stages while another local Ollama evaluation owns the server. `status` and
unit tests do not call Ollama:

```bash
uv run python scripts/run_ollama_gepa_study.py status
uv run pytest -q tests/test_ollama_gepa_study.py
```

## Artifacts and privacy

All prompts, article/question IDs, labels, row-level predictions, GEPA candidates, trajectories,
and model responses remain under the ignored directory
`data/cache/evidence-inference-ollama-gepa-v1-final-v3/`. The only trackable study output is
`artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json`, which contains aggregate
metrics, model/configuration metadata, caveats, and lineage hashes. Its validator rejects article
or question IDs, text, per-example labels or predictions, candidate text, and absolute paths.

The public result must say `seed_retained_no_improvement_claim` when the handwritten seed wins.
When a changed winner does not clear the paired article-clustered rule, it must say
`no_improvement_claim`. Even a positive diagnostic result remains explicitly non-pristine and
non-confirmatory.

## Frozen diagnostic result

The clean `final-v3` run passed frozen-winner, paired-test, private-receipt replay, public
self-hash, and current-source validation. It evaluated 7 accepted candidates using 540
optimization task calls and 8 reflection calls, then made 524 seed and 524 winner calls on the
complete non-pristine test split (191 articles). All 1,588 task receipts and 8 reflection receipts
validate; there were no failed local-model calls.

GEPA selected a changed winner (`seed_retained=false`), but the prespecified improvement rule did
not pass:

- Direction accuracy: winner **0.322519**, seed **0.326336**, paired difference
  **-0.003817 [-0.024905, 0.015152]**.
- Direction macro recall: winner **0.324593**, seed **0.331742**, paired difference
  **-0.007150 [-0.026702, 0.010726]**.
- Formal grounding validity: winner **0.652672**, seed **0.688931**, paired difference
  **-0.036260 [-0.086869, 0.012590]**.
- Structured-output validity: winner **0.994275**, seed **0.990458**, paired difference
  **0.003817 [0.000000, 0.009785]**.

Accordingly, the public status is `no_improvement_claim`, not a GEPA improvement result. The
aggregate is `artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json` with self-hash
`1039156083798863e85761ecf94b76578c74066af2ef7b7691fd4d724f4967ce`. It is explicitly
non-pristine, non-confirmatory, and specific to the pinned local `llama3.2:1b` runtime.

The study evaluates structured direction extraction and formal quote/line containment. It does
not evaluate retrieval, semantic entailment, clinical truth, numerical effect extraction,
meta-analysis, or the full claim-release verifier.

A later full-population `claude-fable-5` transfer also favored the seed on all three prespecified
metrics. That separate result changes the engineering choice for its runtime surface, not this
study's non-pristine or non-confirmatory authority. See
[the full Fable retrospective](evidence-inference-fable-retrospective-full.md).
