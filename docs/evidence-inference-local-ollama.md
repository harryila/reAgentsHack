# Offline local-model Evidence Inference diagnostic

This diagnostic measures one deliberately modest baseline: deterministic structured
extraction with the locally installed `llama3.2:1b` model. It is not a held-out result.
Every Evidence Inference test label in this checkout was previously opened, so every
metric is marked non-pristine and diagnostic-only.

The design enforces a physical stage boundary:

1. `prepare` validates the provider-free report, its redacted prediction ledger, and the
   full split manifest. It selects the 482 prompts on 179 papers with no successful test
   call in the complete bound local provider-receipt registry. It writes only IDs, PICO
   input fields, and a fixed Results-only source projection under ignored `data/cache/`.
2. `predict` accepts only that input-only bundle. It has no manifest or label-file
   argument. Each localhost response receives a self-hashed receipt binding the input
   row, rendered prompt, model name and digest, Ollama version, all generation settings,
   row-specific JSON Schema, raw response, parsed output, and telemetry. The prediction
   ledger is resumable while partial and becomes a complete frozen ledger only when all
   rows have receipts.
3. `score` first revalidates the input bundle, complete ledger, and every receipt. Only
   after those checks does its scoring-only loader materialize labels for evaluation. It
   replays the existing fixed lexical baseline and requires every replayed output to match
   the prior lexical ledger before constructing paired metrics.

The source JSONL physically co-locates inputs and labels, so the trusted preparation
process necessarily reads row bytes that also contain protected fields. Its closed
projection selects only IDs, PICO inputs, and source lines, recursively rejects label
keys, and writes an input bundle that the separate prediction API can validate. The
scientific separation claimed here is that prediction cannot receive or open labels—not
that the upstream storage format already provides a physically separate input file.

The fixed retrieval projection ranks sentence-like passages only from Results. Ranking
uses PICO term coverage, exact outcome-phrase presence, and a statistical-signal flag.
At most six passages and 9,000 source characters are exposed. All weights and limits are
hash-bound. This is a retrieval heuristic, not learned retrieval and not evidence of
corpus recall.

The local generation contract is pinned to:

- model: `llama3.2:1b` (`1.2B`, `Q8_0`);
- digest: `baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878`;
- Ollama: `0.15.1`;
- seed: `20260827`;
- temperature: `0.0`, `top_k=1`, `top_p=1.0`;
- context: 8,192 tokens; output cap: 384 tokens; `keep_alive=30m`.

Ollama 0.15.1 crashed in its native `schema_to_grammar` routine when given the official
Evidence Inference line-ID regular expression. The diagnostic therefore uses an
Ollama-safe generation schema whose line-ID enum is created from each row's exposed
passages, then validates the parsed response against the unchanged official benchmark
schema. The generation template, realized row schema, and official evaluation schema
have separate hashes. This preserves constrained generation without silently weakening
evaluation.

Before the complete freeze, a three-row input-only smoke was used solely to test runtime,
schema, and citation syntax. The first attempt exposed the Ollama regex crash; the next
showed that inline `[L23]` markers were being copied into quotes. The final prompt places
line metadata outside explicit source-text delimiters, and its row schema enumerates only
exposed line IDs. No labels or scores were opened during these smoke iterations, and the
retrieval ranking or direction decision rule was not tuned from test outcomes. The failed
smoke receipts remain private under ignored cache for auditability.

Run the stages explicitly:

```bash
.venv/bin/python scripts/evaluate_evidence_inference_ollama.py prepare --force
.venv/bin/python scripts/evaluate_evidence_inference_ollama.py predict --limit 3
.venv/bin/python scripts/evaluate_evidence_inference_ollama.py predict
.venv/bin/python scripts/evaluate_evidence_inference_ollama.py score --force
```

`--retry-failures` replaces only validated execution-failure receipts. It never replaces
a successful model response. The private input bundle, receipts, ledger, and full report
remain under ignored `data/cache/evidence-inference-ollama/`. The tracked public summary
contains aggregate numbers and hashes only; validation rejects article text, quotes,
raw predictions, row IDs, paper IDs, or absolute paths.

Reported metrics include exact structured validity, task-shape consistency, direction
accuracy, formal quote/line copy-grounding, the joint schema-direction-grounding rate,
and agreement with the annotated evidence span. Point estimates are prompt-weighted;
95% percentile intervals resample papers as clusters. Every metric is paired against the
frozen fixed lexical baseline on the same rows, and paired differences use the same
paper-cluster resampling unit.

## Completed diagnostic result

The complete run covers 482 prompts clustered in 179 articles. The local model produced
473 officially schema-valid outputs (98.13%) and nine frozen JSON-decode failures. All
nine failure receipts report `done_reason=length` and `eval_count=384`, exactly the
configured output cap; the tracked aggregate summary exposes these counts without
exposing response text. The frozen ledger contains one terminal receipt for every selected
row, so no selected row was omitted; it does not preserve enough attempt history to support a
no-retry claim. Its output
distribution was 472 `increase`, one `no_effect`, zero `decrease`, and nine execution
failures. This near-total direction mode collapse is the dominant scientific finding.

Direction accuracy was 33.61% (article-clustered 95% CI 28.32–39.00%) versus 44.61%
(39.33–50.31%) for the fixed lexical baseline. The paired local-minus-lexical difference
was −11.00 percentage points (−18.78 to −3.09), so this run is evidence against a local
model direction-improvement claim. Conversely, formal quote/line copy-grounding improved
by 18.88 points (11.74–26.04) and gold-line agreement improved by 19.29 points
(13.42–25.44). Quote-token F1 improved by 3.64 points, but its interval crossed zero
(−2.20 to 9.56). The joint schema-direction-provenance rate fell by 15.98 points
(−23.31 to −8.25).

These mixed results localize the opportunity: a future optimizer must correct direction
collapse without sacrificing the model's stronger source localization. They do not show
GEPA improvement. The metadata-only result and exact run lineage are in
`artifacts/diagnostics/evidence-inference-ollama/summary.json`.

That lineage binds the input and prediction ledgers, model/runtime identity, generation
and evaluation schemas, retrieval and prompt configuration, bootstrap settings, and a
declared hash inventory covering the evaluator CLI, its direct/transitive project modules,
`pyproject.toml`, and `uv.lock`. These are reproducibility hashes, not signatures or proof
of authorship; durable rollback resistance still requires an external signed or
append-only anchor.

Formal grounding here means only that the exact quoted bytes occur in the cited source
line. It is not semantic entailment. Gold-span disagreement can also occur when a model
cites a different valid span. The diagnostic does not evaluate native numerical effect
extraction, GEPA improvement, end-to-end retrieval, or literature-level scientific
synthesis. It makes zero hosted-provider calls and incurs zero provider charges; local
energy and hardware costs are not measured.
