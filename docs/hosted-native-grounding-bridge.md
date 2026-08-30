# Hosted at-most-once native grounding bridge

The hosted bridge is an offline conversion boundary for a future production numerical
extraction run. It makes no provider calls and opens no benchmark labels. Its input is
one closed `hosted-native-extraction-run-v1` artifact built with the factories in
`literature_multiverse.hosted_native_extraction_contract`.

Existing MetaSyn passage-yield, contextual-frontier, recovery, post-hoc, feasibility,
and code-owned fixture artifacts are not accepted. Those contracts explicitly deny
synthesis-input and claim-release authority. Rehashing or relabeling them would not
create production evidence.

The first two-record hosted execution is documented in
[hosted-native-numeric-pilot.md](hosted-native-numeric-pilot.md). Its immutable v3
run terminated with two non-retried HTTP 400 failures and correctly bridged as a
zero-completion package. The fresh v4 recovery contract is a new prompt-JSON execution
identity with a source-free canary; it is not a retry or upgrade of v3.

## Required run lineage

The implementation and serialized setting retain the historical name
`hosted_exact_once`, but the precise transport guarantee is **at most one provider
attempt per frozen request identity**, not exactly one successful invocation. A crash
after the durable intent but before an observable response poisons that identity; it
is never retried. This sacrifices liveness to prevent duplicate scientific calls.

The run contains one terminal call for every record in a frozen
`native-source-manifest-v1`. Every call binds all of the following before transport:

- the locked question configuration and corpus cutoff;
- a computed pipeline fingerprint that includes the hosted contract, bridge, native
  extraction, and native grounding implementations;
- the exact full-text source record, source-document bytes, and locator;
- provider, model, model revision, API base, runtime, SDK, and non-secret runtime
  metadata, plus the exact runtime source paths inside the computed fingerprint;
- exact rendered prompt bytes and repository template bytes;
- the exact provider generation schema and official native postvalidation schema;
- the exact credential-free wire-request JSON bytes;
- a durable one-attempt intent and matching authorization with application and SDK
  retries fixed to zero; and
- one terminal provider receipt. A completed receipt contains exact response JSON bytes
  and an RFC 6901 pointer that must resolve to the postvalidated
  `NativePublicationExtraction`. Provider failure and ambiguous-attempt poison are
  terminal non-estimable records and are never silently dropped or retried.

The run must declare production extraction, full-text source scope, no diagnostic or
fixture origin, no code-owned predictions, and no opened reference/test labels. These
declarations are necessary but not sufficient: trusted code still externally rehashes
the pipeline, prompt templates, every source document, every terminal call, and every
grounding coordinate.

## Build and replay

Validate without writing:

```text
python scripts/build_hosted_native_grounding_package.py validate \
  --repository-root . \
  --run private/hosted-native-extraction-run.json
```

Build and immediately replay the standard v4 package:

```text
python scripts/build_hosted_native_grounding_package.py build \
  --repository-root . \
  --run private/hosted-native-extraction-run.json \
  --output-dir private/hosted-native-grounding
```

For a multi-publication corpus, a completed external cohort-reconciliation artifact may
be supplied with `--reviewer-reconciliation`. Without it, the normal verifier gate may
block release because cross-publication cohort identity is incomplete.

The emitted `typed_evidence_grounding_package.json` is already accepted by both public
paths:

```text
lm verify --claim claim.yaml \
  --corpus private/hosted-native-grounding/typed_evidence_grounding_package.json
```

An acquisition manifest may reference the same file through its existing
`typed_grounding_package` mode. Acquisition replays membership and then routes the
package through the same verifier loader; it does not confer new authority.

## Authority boundary

Successful conversion establishes only that a complete exact-once hosted execution and
its exact source-grounded typed projection were replayed into a version-four provenance
input. The bridge receipt deliberately keeps `claim_release_authority=false`. Claim
release still depends on the separate protocol, synthesis, cohort identity, calibration,
audit, residual-risk, and release-policy gates. It also does not turn this source-only
run into an extraction-accuracy benchmark or evidence of scientific truth.

## Fingerprint integration requirement

Before a hosted run is frozen, the verifier's `native-extraction` fingerprint component
must be bumped from version 12 to version 13 and must add:

- `scripts/build_hosted_native_grounding_package.py`;
- `src/literature_multiverse/hosted_native_extraction_contract.py`; and
- `src/literature_multiverse/hosted_native_grounding_bridge.py`.

The component settings and native entry-point list should also name the
`hosted-native-extraction-run-v1` / `hosted_exact_once` boundary. A run prepared against
the old component or before those bytes are included must fail fingerprint replay and
must be regenerated, not patched post hoc.

This bump is now in place: the current native-extraction component version is 13.
