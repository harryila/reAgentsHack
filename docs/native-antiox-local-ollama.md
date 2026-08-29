# Native Antiox local-Ollama diagnostic

> **Historical, superseded diagnostic.** This one-stage `llama3.2:1b` run is retained
> only for provenance and compatibility auditing. It is not the current native
> extraction result and has no extraction, calibration, or claim-release authority.
> The current fail-closed study is the
> [bounded two-stage diagnostic](native-bounded-ollama-diagnostic.md).

This diagnostic tests whether the frozen 19-publication Antiox source subset can pass
through native numerical extraction, exact source grounding, typed evidence assembly,
v4 source-manifest packaging, replay, statistical synthesis, and the unified verifier.
It uses the existing local `llama3.2:1b` model and makes no hosted-provider calls.

This is not an extraction-accuracy evaluation. The subset was selected by historically
opened legacy eligibility decisions, no independent native numerical gold annotations
exist, and the 19 publications are not evidence of retrieval completeness. The resulting
certificate must abstain. The run measures structured-output validity, extraction yield,
mechanical grounding, downstream graph/synthesis behavior, and verifier blockers.

## Enforced stage boundary

The workflow has three physical stages:

1. `prepare` verifies the v2 source bridge, requires
   `selection_scope=legacy_eligible`, binds its exact run hash, validates the 19-record
   native source manifest, and resolves only the authoritative `source_lines` bytes
   named by each manifest locator. It hashes but does not parse the full locked question
   YAML; the model-facing question projection is a closed, anchor-free configuration.
   No legacy finding, direction, anchor expectation, or label is copied into the bundle.
   The fixed downstream claim is represented only by a hash; its payload is opened from
   the already-frozen diagnostic config only after all 19 model receipts are complete.
2. `predict` accepts only the ignored input bundle. It has no question-config, source,
   manifest, legacy-output, or label argument. Each row receives a deterministic,
   regex-free Ollama generation schema. Every response is strict-parsed once and then
   post-validated against the unchanged official `NativePublicationExtraction`
   contract. JSON errors, truncations, and schema failures in an actual model response
   are terminal receipts and are never retried or silently repaired. The exact
   runtime/model identity is rechecked before every request. A missing runtime or a
   transport failure with no model response stops the stage without converting that
   infrastructure outage into a permanent paper-level scientific result; already-frozen
   responses remain immutable when the command resumes. Before contacting Ollama, the
   command also recomputes both the native execution identity and the full downstream
   verifier fingerprint. Code, prompt, dependency, or configuration drift after
   `prepare` therefore fails before the first generation request. It then compiles the
   full expanded generation-schema shape in a short synthetic preflight containing no
   publication, claim, source text, eligibility label, or expected answer. A grammar
   incompatibility stops before any scientific row request and never becomes a paper
   receipt. This local compatibility call is deliberately outside the paper-level
   response-bearing receipt counters.

3. `finalize` first validates the complete 19-receipt freeze. It converts every official
   output or terminal failure through the shared grounding projection, requires one
   terminal fragment per manifest record, constructs a v4 grounding package bound to
   cutoff `antiox-legacy-eligible-diagnostic-2026-08-27`, replays current source bytes,
   loads the package through the unified verifier, and writes an abstaining certificate.

Generation-call and retry counters have the explicit scope
`response_bearing_terminal_receipts_only`. A client-side timeout or disconnect after a
POST can be ambiguous: the local server may have executed a call whose response never
reached the coordinator. Such an attempt is not represented as a scientific row result
and cannot be durably counted by this client. Consequently, a clean run has exactly 19
response-bearing paper calls plus one synthetic schema-compatibility call per `predict`
invocation; the paper-level ledger reports only the 19 receipts. After a transport
abort, rerunning may add another synthetic preflight and an unobserved infrastructure
attempt even though no returned paper response is ever retried.

The source bridge identity is fixed to run SHA-256
`62d87d08b116d2af95da3e8646a293f9cdaf3507fd8b3e84319b2007367f26b3`.
The model boundary is pinned to Ollama `0.15.1`, `llama3.2:1b`, model digest
`baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878`,
seed `20260827`, temperature `0`, `top_k=1`, `top_p=1`, a 16,384-token context,
and a 3,072-token output cap.

Ollama 0.15.1 previously crashed while compiling a JSON Schema regular expression.
This diagnostic therefore removes every `pattern` keyword only from the generation
schema and enumerates each row's exposed line IDs and exact source locator. The frozen
response is still post-validated against the official native schema, whose independent
hash is reported. The generation schema also exposes the official mutually exclusive
top-level status invariant: non-estimable outputs must contain zero studies, while
estimable outputs must contain at least one. Relaxing regex grammar does not relax
downstream acceptance. For either paper whose label-blind projection contains no source
passage, the line-ID grammar exposes only a non-source sentinel; it never falls back to a
real-looking hidden line ID, and the top-level grammar permits only the non-estimable
branch. Each composition branch declares `type: object` explicitly, every constrained
branch property repeats its primitive/container type, and each status discriminator
uses a typed singleton `enum`. Ollama 0.15.1 rejects otherwise-valid property-only
`oneOf` subschemas, constraint-only nested property schemas, and `const` discriminators
inside `oneOf` as an invalid generation format. These compatibility constraints are
regression-tested without changing strict official post-validation.

Prompt version `native-extraction-v3` also closes the statistical handoff: endpoint names
must use the frozen canonical map, supplement is always the treatment arm, positive
effects mean a larger beneficial adaptation under supplementation, and the requested
estimand is the between-group difference in training adaptation. Explicit reported
timepoints are required when available and are never guessed.

## Execution

Run the stages separately so the input and prediction freezes remain auditable:

```bash
.venv/bin/python scripts/run_native_ollama_diagnostic.py prepare \
  --workspace data/cache/native-antiox-ollama-v2-final-v1
.venv/bin/python scripts/run_native_ollama_diagnostic.py predict \
  --workspace data/cache/native-antiox-ollama-v2-final-v1
.venv/bin/python scripts/run_native_ollama_diagnostic.py finalize \
  --workspace data/cache/native-antiox-ollama-v2-final-v1 --force
```

The workspace must be new or contain only an unconsumed input bundle. `prepare` refuses
to overwrite a workspace containing generation receipts, a prediction ledger, or final
artifacts, even with `--force`; choose another ignored workspace instead. This prevents
new input bytes from being mixed with receipts from an earlier schema or pipeline.

The private bundle, prompts, raw responses, official outputs, receipts, grounding
package, replay, and certificate stay under ignored
`data/cache/native-antiox-ollama-v2-final-v1/`. The only tracked output is the aggregate
summary at `artifacts/diagnostics/native-antiox-ollama/summary.json`.

Public-summary validation recursively rejects article identifiers, source locators,
source text, quotes, prompts, predictions, raw responses, and absolute paths. It reports
only population counts; schema/status/yield counts; exact-grounding and downgrade counts;
graph counts; synthesis mode, status, and aggregate failure reason; certificate version,
reasons, and blocker codes; hashes; local-model token and duration telemetry;
coordinator-process peak RSS; and explicit caveats. RSS does not include the separate
Ollama server process, and energy use is not measured.

## Interpretation boundary

An exact grounding receipt proves that a quoted string and cited coordinates are present
in the frozen source bytes. It does not prove that the extracted effect is semantically
correct, that the chosen comparison is scientifically appropriate, or that the source
quote contains every represented field. Parsed source lines also do not provide genuine
multimodal reading of tables, figures, or supplements.

Accordingly, this diagnostic cannot support an accuracy, retrieval-recall, pristine
holdout, human-audit, multimodal-extraction, GEPA-improvement, calibrated-release, or
scientific-truth claim. It is an end-to-end engineering and failure-localization result.
