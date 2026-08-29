# Bounded native Ollama diagnostic

This diagnostic is a retrospective engineering study over the 19 historically opened
Antiox publications. It does not estimate extraction accuracy, calibrate claim risk, or
authorize claim release. Its purpose is narrower: test whether a small local model can
produce finite, mechanically replayable numerical-extraction packets without the
runaway schema filling observed in the historical one-stage pilot.

The executed model is pinned to `qwen2.5:3b-instruct`, digest
`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`,
under Ollama 0.15.1 (`3.1B`, `Q4_K_M`, GGUF, Qwen2 family). Model and schema selection
used the same already-opened development population, so any resulting aggregate is
descriptive and non-confirmatory.

## Frozen result

The completed run froze 19 inventory calls and 33 candidate-packet calls with zero
retries, after the six source-free schema preflight calls passed. Seven publication
inventories were valid and below the candidate cap, ten returned the typed no-candidate
outcome, and two were contract-invalid. Every one of the 33 requested candidate packets
was contract-invalid. Consequently, **zero publications and zero findings** were
promoted to `NativePublicationExtraction` v1. No partial packet was salvaged.

This is a negative structured-generation result for the pinned 3.1B local model, not an
extraction-accuracy estimate. In particular, the aggregate's statements that lexical
numeric support and source grounding are verified *for promoted packets* are vacuously
true because there were no promoted packets; they are not evidence that any numerical
extraction was correct.

A post-freeze, aggregate-only failure diagnosis found that all 33 packet responses passed
their stored generation JSON Schema but failed the stricter Pydantic packet contract.
Supplying only the omitted `packet_status` discriminator in a counterfactual diagnostic
did not rescue any row: all 33 still violated the bounded timepoint-decimal contract, with
additional numeric-support token/offset, effect-estimate, and sorted-identifier failures.
This diagnosis does not alter or relabel v1. It identifies an underconstrained v1
generation schema and motivates a separately versioned schema-tightening study.

The frozen lineage is:

- input bundle: `f48876719dbfd95308bcf0d405add1a78ee7882a68636243654f8e5b9afa24dd`;
- prediction ledger: `38872854af5eb720442ce95e5486b3286237552d2b140522e721c9f3a9a1cda2`;
- private report: `632284469d7828b8fe6d9823e74665cf6d1108ead39882144b021f1e0e7d2dbc`;
- public summary: `af6680f7f2d94ad5bea6bb59e6fcddc46e140ee8377447c96b55d7413ae788dc`.

At freeze time, current-source validation recomputed the code, configuration, prompt,
model, downstream verifier, strict-schema, and aggregate-semantic bindings. The public
registry now preserves v1 as an immutable historical result by requiring its exact
registered self-hash and frozen lineage identities; it deliberately does not reinterpret
v1 under later bounded-schema implementations. The tracked summary alone cannot prove
the empirical counts. The ignored private v1 replay independently rechecks all intents,
receipts, the prediction ledger, the all-or-nothing finalizer, and those counts; that
replay passes for the frozen workspace.

## Frozen stage order

1. `prepare` opens the current source-preparation inputs, projects only question,
   protocol, and source fields, and freezes a content-bearing bundle under ignored
   `data/cache/`. It prints the bundle SHA-256 that every later stage requires as an
   explicit freeze anchor.
2. `preflight` makes six source-free calls: one inventory grammar and one grammar for
   each of the five packet effect families. Every call has a durable pre-call intent and
   immutable response receipt. An ambiguous call poisons the workspace; it is never
   retried. The complete preflight is bound to the frozen bundle and current verifier
   pipeline.
3. `predict` first validates the exact bundle anchor, current code/config/prompts,
   complete preflight, canonical workspace layout, source-row contracts, model identity,
   and an OS-exclusive workspace lock. It then requests a value-free candidate inventory
   for each publication and exactly one candidate-bound packet per authorized candidate.
4. `finalize` reopens current source inputs, replays every intent and receipt, verifies
   every completed quote against exactly one cited frozen projected passage, and assembles
   an unchanged `NativePublicationExtraction` v1 only when the whole publication is
   valid. There is no partial-packet salvage.
5. `validate-public` checks current code/config lineage and a strict aggregate-only
   schema. `validate-private` additionally replays the source bundle, preflight, model
   receipts, prediction ledger, projection grounding, and all empirical counts.

Those two commands describe the frozen v1 runtime while its exact sources are present.
The cross-version public registry uses the immutable historical binding above, so future
v2 code cannot be mistaken for a replay of v1.

Use a fresh ignored workspace and preserve the SHA printed by `prepare`:

```bash
python scripts/run_native_bounded_ollama_diagnostic.py prepare \
  --workspace data/cache/native-antiox-bounded-v1-final-v1

python scripts/run_native_bounded_ollama_diagnostic.py preflight \
  --workspace data/cache/native-antiox-bounded-v1-final-v1 \
  --expected-input-bundle-sha256 <PREPARE_SHA256>

python scripts/run_native_bounded_ollama_diagnostic.py predict \
  --workspace data/cache/native-antiox-bounded-v1-final-v1 \
  --expected-input-bundle-sha256 <PREPARE_SHA256>

python scripts/run_native_bounded_ollama_diagnostic.py finalize \
  --workspace data/cache/native-antiox-bounded-v1-final-v1 \
  --expected-input-bundle-sha256 <PREPARE_SHA256>

python scripts/run_native_bounded_ollama_diagnostic.py validate-public

python scripts/run_native_bounded_ollama_diagnostic.py validate-private \
  --workspace data/cache/native-antiox-bounded-v1-final-v1 \
  --expected-input-bundle-sha256 <PREPARE_SHA256>
```

Source-bearing bundles, prompts, intents, responses, ledgers, and assembled extractions
must remain under ignored `data/cache/**`. The only permitted public output is the
aggregate summary at
`artifacts/diagnostics/native-antiox-bounded-ollama/summary.json`. The public artifact
explicitly states that its empirical counts require private replay; its public-only
validator cannot independently prove those counts without the private receipts.

## Fail-closed boundaries

- Inventory saturation, invalid JSON, schema failure, output-cap truncation, an
  `unable_to_complete` packet, missing packet, unsupported numeric syntax, quote/source
  mismatch, ambiguous quote occurrence, conflict, or official-schema failure blocks the
  entire publication.
- All model-facing arrays, strings, and numeric lexemes are bounded. Scientific numeric
  leaves require field-specific verbatim tokens, exact quote offsets, conservative
  `Decimal` replay, and distinct source spans. Derived statistics are unsupported in v1.
- A direct risk ratio or odds ratio and its interval bounds must be strictly positive.
- Exact lexical support is not semantic entailment. It does not prove that a reported
  value belongs to the claimed arm, contrast, outcome, or analysis population. Human
  verification and independent claim-level calibration remain required.
- A below-cap inventory is not evidence that extraction is complete. The inventory can
  still miss eligible effects, and conservative duplicate handling can reduce recall.
- Title/abstract packets may be structurally diagnosable for the generic adapter but are
  not release-grade full-text grounding.

The historical one-stage native summary remains historical-only. Its receipts, model
settings, and prompt lineage are never accepted by this versioned two-stage workspace.
