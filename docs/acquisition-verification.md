# Frozen acquisition to claim verification

`lm verify` can replay a frozen local acquisition before it enters the scientific
verifier:

```bash
uv run lm verify \
  --claim path/to/claim.yaml \
  --acquisition-manifest path/to/frozen-acquisition.json \
  --budget-minutes 60 \
  --analysis-only-uncalibrated-audit \
  --output-dir artifacts/verification/example
```

`--acquisition-manifest` and `--corpus` are mutually exclusive. The existing
`lm verify --corpus ...` path is unchanged. Acquisition replay requires an explicit
output directory because the immutable harvester archive and, for offline extraction
ledgers, the derived native package are part of the run lineage.

This command does not query the open web or call an extraction model. It replays the
manifest's exact local `search_results` and consumes a pre-existing complete extraction
ledger or typed package. The integration therefore establishes deterministic orchestration
and fail-closed membership checks, not empirical retrieval recall, protocol-screening
accuracy, numerical-extraction accuracy, or claim-verification effectiveness.

## Contract

A `frozen-acquisition-manifest-v1` binds:

- the repository-relative frozen corpus file and its raw SHA-256;
- every query family/query pair, page size, per-query limit, and an aware retrieval
  timestamp;
- exact expected retrieved-document and terminal deterministic-screen memberships;
- the allowed article types;
- optionally, a complete self-hashed `protocol-screening-receipt-v1`; and
- either a complete `frozen-native-extraction-ledger-v1` or an existing
  `typed-evidence-grounding-package` artifact, with its raw SHA-256.

The manifest is self-hashed over its normalized content. Code constructing a manifest
should use `freeze_acquisition_manifest`; an offline extraction ledger should use
`freeze_native_extraction_ledger`. Both helpers validate closed schemas before sealing
the content hash.

The computed verifier fingerprint covers this path. Its current `native-extraction`
component is version 13 and includes the acquisition and harvester modules as well as the
native extraction/grounding surface. Any covered byte drift changes the computed pipeline
identity. The component version is an integrity boundary, not evidence that a retrieval or
extraction model is scientifically valid.

The frozen corpus must declare an exact `search_results` list for every query. Replay
rejects an undeclared query or a result list larger than `per_query_limit`; it never
silently accepts a truncated page as complete. It also rejects changed input bytes,
unexpected retrieval or screen membership, unresolved fuzzy-identity pairs, an
included paper without archived full text, and any missing or extra native extraction.

## Native inputs

`frozen_extraction_ledger` is the provider-free test/offline adapter. It requires one
terminal `NativePublicationExtraction` for every included paper. Text, XML, and HTML
full text are stored in the immutable harvester archive and addressed by a
`harvest-sha256:<content-sha256>` source locator. Native grounding verifies the archive
blob hash, deterministically projects its text, and then applies the same exact quote
and coordinate checks as other native sources. Unsupported media such as PDF fails
closed; it is not converted or guessed by this adapter.

This mode creates a membership-bound version-three typed grounding package. It has no
provider/model execution context and therefore remains analysis-only. It is useful for
complete offline tests, not for asserting that an extraction model was actually run.

`typed_grounding_package` accepts an externally produced package and replays it through
the normal native grounding loader. Its source-manifest `(doc_id, paper_id)` membership
must exactly equal the acquisition screen's included membership. A version-four package
can carry its real provider/model execution authority; the acquisition layer does not
manufacture or upgrade that authority.

## Outputs and fail-closed boundary

Alongside the normal JSON/HTML verification certificate, the command writes
`acquisition-replay-receipt.json`. The self-hashed receipt binds the manifest, claim
protocol, raw frozen corpus, occurrence and screen memberships, archive entries, native
source manifest, typed package, and stage counts. The same receipt is embedded in the
certificate's corpus metadata, and the corpus identity combines native and acquisition
lineage.

The deterministic screen enforces identity deduplication and article-type rules. It does
not itself adjudicate free-text population/intervention/exposure/comparator/outcome
eligibility. Without an external receipt, the orchestrated corpus therefore carries the
blocking issue `protocol_eligibility_screening_unverified`.

An optional `protocol-screening-receipt-v1` must cover every canonical retrieved paper,
bind the exact claim protocol and corpus cutoff, and preserve deterministic exclusions.
It records declared provenance and complete screening decisions, but it contains no
trusted reviewer identities, independent raw-decision artifacts, signatures, or
attestation registry. Consequently, every version-one receipt remains non-production
and adds the blocking issue `missing_verified_screening_adjudication_package`, including
one that self-declares `blinded_human` provenance and two adjudicators. A future
production screening boundary must replay an externally verifiable, hash-bound
adjudication package; this adapter never upgrades a self-asserted receipt into authority.

Exact frozen query replay also does not establish empirical retrieval recall outside the
declared query result lists; the receipt states this limitation explicitly.

Terminal non-estimability is not treated as a missing record: it remains a publication
fragment and a blocking corpus issue, so the verifier can explain why it abstained.
Structural incompleteness—missing query membership, full text, screen identity, source
record, extraction, or fragment—raises an error before synthesis. No empty graph row,
effect estimate, screening label, or provider receipt is fabricated.
