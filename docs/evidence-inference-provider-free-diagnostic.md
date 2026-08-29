# Evidence Inference provider-free diagnostic

This command evaluates the largest locally available Evidence Inference 2.0
extraction set without making provider calls. Every result is a **non-pristine
diagnostic**: test labels are locally co-located with inputs and aggregate pilot
results were previously inspected. Nothing here is confirmatory evidence of prompt
improvement, semantic support, or scientific validity.

## Populations

The full conversion contains 524 official-test prompts from 191 articles. Archived,
hash-validated provider receipts identify 12 actual prior test attempts on 12 articles.
Excluding every prompt on those articles removes 42 rows, including 30 sibling prompts.
The resulting 482-prompt, 179-article set is named the
`provider_call_unseen_paper` diagnostic subset. “Unseen” is limited to the bound local
receipt registry; it does not prove global nonexposure and is neither primary nor
pristine. All 524 opened test rows are also reported separately.

The exposure manifest must exactly match every test attempt recoverable from the bound
successful-run receipts and the archived test-evaluation artifact. A missing receipt,
manifest mismatch, or test identity mismatch fails closed.

## Endpoints and interpretation

The fixed lexical baseline receives only the intervention, comparator, outcome, and
source lines. Predictions are frozen before metric computation. Reported endpoints are:

- exact JSON-Schema validity;
- explicit task-shape consistency (`eligible=true` requires one finding and
  `eligible=false` requires none);
- intervention-versus-comparator direction accuracy;
- formal quote/line provenance validity;
- joint schema, task-shape, direction, and formal-provenance validity; and
- separate gold evidence-line agreement and quote token-agreement F1.

Formal provenance means only that quoted bytes occur in the cited source lines under
the repository contract. It does **not** measure entailment, semantic support, causal
validity, or whether the quotation justifies the direction. Gold-span agreement is
separate because a different formally valid span may also be acceptable. Intervals are
deterministic 95% article-clustered percentile bootstraps.

## Archived GEPA replay

The replay recovers the handwritten seed and every exact logged proposal. For the valid
archive, each of six proposals is paired with the seed on GEPA’s exact recorded
two-example adaptive training minibatch. These are adaptive training diagnostics, not
held-out comparisons. Nine raw-schema proposals are counted separately as an excluded
failed archive.

Each receipt is bound to its candidate, example, exact prompt, schema, and request hash.
Paired arms must match provider, model, effort, maximum tokens, and system prompt; the
valid run also binds the exact trace configuration. Nonidentical duplicate
candidate/example receipts fail closed, while byte-identical duplicates may be
deduplicated explicitly. The report separately records calls, reported input/output
tokens, estimated cost and cost basis. The old trace lacks actual reflection-LM usage,
so candidate-generation/reflection cost remains explicitly unreconstructed.

The archived GEPA 0.1.4 score near 0.51258 is excluded only when the exact known run ID,
trace hash, library version, optimizer, candidate hash, and score all match. For that
artifact, direct replay finds 10 common development receipts and bounds two missing
mutation responses. Those counts are computed from receipts rather than assumed. The
trace score is non-citable.

The 12 archived seed test attempts are called `opened_test_seed_archive_replays`.
No mutation has a full test response archive.

## Artifacts and reproduction

Run locally:

```bash
uv run python scripts/evaluate_evidence_inference_diagnostic.py --force
```

This writes two ignored artifacts:

- `data/cache/evidence-inference-diagnostic/provider-free-report.json`
- `data/cache/evidence-inference-diagnostic/prediction-ledger.json`

The ledger stores IDs, directions, cited line IDs, and SHA-256 quote/output commitments,
but no article text, evidence quotes, source lines, or gold labels. Report and ledger
bind one another, their inputs, and the relevant implementation files.

To explicitly create the metadata-only public summary:

```bash
uv run python scripts/evaluate_evidence_inference_diagnostic.py \
  --public-summary-output \
  artifacts/diagnostics/evidence-inference/summary.json \
  --force
```

The public summary contains no article text, quotes, label values, paper/example IDs, or
raw predictions. It is self-hashed and binds the ignored full report and ledger.

Verification:

```bash
uv run ruff check .
uv run pytest -q tests/test_evidence_inference_diagnostic.py \
  tests/test_evidence_inference.py tests/test_prompt_optimization.py
```

## Explicit opt-in live comparison

The diagnostic never inspects credentials or starts provider execution. A separately
approved operator may use the existing explicit `--live` comparison command described by
`scripts/optimize_prompts.py`. Live runs incur external cost, require a unique output,
and remain non-pristine when they use the already opened test split. The native
production extraction prompt and pipeline remain outside this diagnostic’s scope.
