# Evidence Inference 2.0 benchmark adapter

This adapter turns the official Evidence Inference 2.0 release into a real
extraction-and-grounding benchmark for the optional GEPA prompt optimizer. It preserves the
upstream article-level train/validation/test split and uses exact PMC article identifiers as both
`paper_id` and `group_id`, preventing prompts from the same paper from crossing splits.

The staged provider-free official-GEPA protocol is documented separately in
[Local-Ollama official-GEPA Evidence Inference study](evidence-inference-local-ollama-gepa.md).

## Official sources and scope

- Data and format documentation: <https://evidence-inference.ebm-nlp.com/download/>
- Version 2.0 archive: <https://evidence-inference.ebm-nlp.com/v2.0.tar.gz>
- Official code repository: <https://github.com/jayded/evidence-inference>
- Dataset paper: <https://aclanthology.org/2020.bionlp-1.13/>

The archive contains `prompts_merged.csv`, `annotations_merged.csv`, 4,454 plain-text article
files, and official article-ID splits. The converter retains only prompts with verifier-accepted
labels and rationales, agreement across accepted label codes, and a physician-selected quote that
mechanically grounds in `BODY.RESULTS`. By default it also excludes prompt IDs that the upstream
README labels incorrect, questionable, or somewhat malformed.

This is a benchmark for extracting a reported significance direction and supporting span. It is
not a benchmark for estimating an effect size or deciding scientific truth. The exact direction is
scored as the scientific label. The human span is retained for audit, while generated evidence is
scored for schema validity and exact source-line grounding; requiring equality to one annotated
span would incorrectly penalize alternate valid evidence.

## Download and convert

Keep the upstream and converted data under `data/cache/`, which is gitignored:

```bash
mkdir -p data/cache/evidence-inference-2.0
curl -L https://evidence-inference.ebm-nlp.com/v2.0.tar.gz \
  -o data/cache/evidence-inference-2.0/v2.0.tar.gz
tar -xzf data/cache/evidence-inference-2.0/v2.0.tar.gz \
  -C data/cache/evidence-inference-2.0

uv run python scripts/convert_evidence_inference.py \
  --output-dir data/cache/evidence-inference-gepa \
  --metadata-summary data/cache/evidence-inference-gepa/metadata-summary.private.json
```

For a cheap structural smoke test, append `--max-examples-per-split 3`. The cap uses a
deterministic, label-blind hash ranking and selects distinct papers first. Conversion is local and
does not import GEPA, contact a model provider, or read credentials.

The checked first-pass configuration uses 12 examples from 12 distinct papers per split:

```bash
uv run python scripts/convert_evidence_inference.py \
  --output-dir data/cache/evidence-inference-gepa-low-budget \
  --max-examples-per-split 12 \
  --metadata-summary \
    data/cache/evidence-inference-gepa-low-budget/metadata-summary.private.json
```

Validate the generated split contract without opening the test JSONL:

```bash
uv run python scripts/optimize_prompts.py validate \
  --manifest data/cache/evidence-inference-gepa/manifest.json
```

For a future fresh optimizer run, pass the benchmark-specific template explicitly and use a new
gitignored run directory:

```bash
uv run --extra gepa python scripts/optimize_prompts.py optimize \
  --manifest data/cache/evidence-inference-gepa-low-budget/manifest.json \
  --run-dir data/cache/gepa/evidence-inference-next-run \
  --extraction-template prompts/evidence_inference_extraction.md \
  --seed 20260826 \
  --reflection-lm anthropic/claude-sonnet-4-6 \
  --reflection-max-tokens 1200 \
  --max-metric-calls-per-prompt 40 \
  --max-reflection-cost-usd-per-prompt 2.5 \
  --reflection-batch-headroom-usd-per-prompt 0.5 \
  --reflection-minibatch-size 2 \
  --cost-cap-usd 0.03 \
  --model claude-sonnet-5 \
  --effort low \
  --max-tokens 1200 \
  --max-budget-usd 8 \
  --live
```

The 40-call task cap accommodates the seed evaluation and approximately two
reflection/minibatch/candidate rounds on the 12-example dev set. GEPA evaluates in batches, so its
call stopper may finish a batch at the boundary; the provider's separate `$8` ledger cap remains
hard. The `$2.50` reflection stopper is separate and uses zero retries.
Before credentials or clients are loaded, the CLI counts active prompt kinds from train/dev and
reserves, against a `$50` planning ceiling, existing archived provider estimates, the entire `$8`
task budget, `$2.50` reflection per kind, and `$0.50` batch-boundary reflection headroom per kind. It
also lowers the task provider's global ledger limit by the reflection reservation. Reflection runs
through LiteLLM rather than the archived task-provider ledger, so the combined `$50` value is a
conservative preflight policy under the stated headroom, not a mechanically hard cross-provider
ledger.

## Superseded 12-example archive

The first provider-backed archive under
`data/cache/gepa/evidence-inference-first-pass-v2` is retained locally for forensic
replay, but it is no longer a citable optimization result. A receipt-level audit found
only 10 clean common development responses and two missing mutation responses. The
archived trace scalar near `0.51258` is therefore excluded fail closed. The public,
self-hashed audit at `artifacts/diagnostics/evidence-inference/summary.json` exposes the
receipt counts and missing-response bounds without publishing text, identifiers,
predictions, or labels.

The provider-free official-GEPA study replaces that pilot with a frozen development
selection and one paired evaluation on all 524 non-pristine test examples from 191
articles. The clean `final-v3` run completed 540 optimization task calls, 8 reflection
calls, and 524 seed plus 524 winner test calls; all 1,596 receipts validate. GEPA chose a
changed prompt, but direction accuracy was **0.322519** versus **0.326336** for the seed,
a paired article-clustered difference of **-0.003817 [-0.024905, 0.015152]**. The frozen
improvement rule is false and the status is `no_improvement_claim`. The aggregate at
`artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json` has self-hash
`1039156083798863e85761ecf94b76578c74066af2ef7b7691fd4d724f4967ce` and passes current-source
and private-receipt replay validation. This is a useful negative local-model result, not
confirmatory evidence. Reproduce and audit it with `scripts/run_ollama_gepa_study.py` and
the contract in `docs/evidence-inference-local-ollama-gepa.md`.

## Leakage boundary

Each `OptimizationExample` has three model-facing replacements—`OUTCOME`, `INTERVENTION`, and
`COMPARATOR`—sourced only from `prompts_merged.csv`. The model also receives every Results line in
the paper, never a gold-selected window. `UserID`, validity fields, label text/code, annotation
text, and evidence offsets are confined to evaluator-only output construction. The GEPA adapter
renders replacements and source lines, not `expected_output`.

The bundle contains separately hash-locked `train.jsonl`, `dev.jsonl`, and `test.jsonl` files.
Optimization for this run opens train and dev only; staged test evaluation opens test only after
the winner is frozen. That sequencing prevents within-run test optimization but does not restore
pristine-holdout status to the historically opened 524-example test split.

## Provider schema boundary and failed pilot

The benchmark keeps constraints such as evidence-line `pattern`, non-empty quote `minLength`, and
finding-count limits in its original evaluator schema. Anthropic's structured-output endpoint does
not accept every JSON Schema keyword directly. The provider therefore applies the official SDK's
[`anthropic.transform_schema`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/_parse/_transform.py)
only to the copy sent through `output_config.format`. Each provider receipt archives the strict
original and provider-transformed schema hashes and the SDK version. Returned JSON must still pass
the untouched original schema locally before it can receive a score.

The first live pilot on 2026-08-26 was stopped after 68 archived task attempts were rejected with
HTTP 400 before generation because the raw schema had been sent directly. It produced no completed
task calls or experimental measurements and must not be included in results. Its immutable failure
receipts remain diagnostic provenance. A corrected pilot must use a fresh run directory and fresh
request namespace; the one-shot provider never retries archived request keys.

## Licensing caveat

The associated GitHub repository carries an MIT license, but the downloaded v2.0 archive does not
contain a license file or state a dataset-specific redistribution license. It also contains PMC
article text whose reuse terms may vary by article. Treat the archive and generated JSONL as local
research data and do not redistribute text-bearing artifacts until rights have been confirmed.
