# Native extraction grounding and verifier package

The native extraction path is fail closed. An estimable publication fragment is valid
only when it links to an authorizing `native-grounding-receipt-v1`. The receipt contains
the full `NativePublicationExtraction`, its canonical hash, the immutable source
artifact identity, and one self-hashed grounding result per finding.

Supported source locators are deliberately closed:

- `json:<repository-relative-path>#/<escaped-doc-id>` for Antiox numbered source lines;
- `parquet:<repository-relative-path>#row_group=<n>&row_in_group=<n>&index_base=0&ID=<id>`
  for a MetaSyn physical Parquet row.

Resolution rejects absolute or escaping paths, symlinks, file-hash drift, malformed
JSON/Parquet payloads, duplicate locator keys, physical-row drift, and ID mismatches.
The artifact SHA-256 binds the exact physical JSON/Parquet bytes. Resolved logical text
preserves decoded leading/trailing whitespace, tabs, embedded CRLF, and non-ASCII text.
Each resolved line records both Unicode-code-point offsets into the canonical joined
`source_text` and UTF-8 byte offsets into that logical text; these are deliberately not
presented as physical offsets into the surrounding JSON/Parquet container.
Numerical findings authorize only when the authoritative source locator, exact quote,
and at least one supplied line or character-offset coordinate all verify. Citation
relocation, citation refinement, forbidden/unknown sections, missing coordinates, and
quote mismatches do not authorize an estimable graph.
For this native path, "exact quote" means a raw contiguous substring of both the resolved
source text and the cited raw lines (or supplied raw offset slice). Unicode normalization,
whitespace collapsing, and ellipsis splicing may aid diagnostics in the legacy matcher but
cannot authorize a native estimable fragment.

`scripts/s3_extract_typed.py` writes a release-capable version-four grounding package
only when the caller supplies a self-hashed `native-provider-execution-receipt-v1` for
every provider batch. The package is bound to the complete supplied
`native-source-manifest-v1`, exact corpus-cutoff identifier, and exact extraction
execution context:

- `grounding_receipts.jsonl`;
- `publication_fragments.jsonl`;
- `typed_evidence_corpus.json` as the lower-level assembled corpus;
- `typed_evidence_grounding_package.json` as the public verifier input;
- `native_extraction_context.json` as the private exact execution-context record;
- `reconciled_evidence_graph.json` as the exact graph selected by the package's
  cohort-reconciliation receipt;
- `native_extraction_run.json`, which binds the receipt-file, joined-validation,
  package, context, provider-receipt, and corpus hashes and reports expected non-estimable
  extractions separately from failed estimable grounding attempts.

The private execution context embeds the canonical locked question configuration, exact rendered
prompt text, official postvalidation and generation schemas, provider/model/runtime identity,
exact raw per-call ledger, source and map artifact digests, package/corpus-cutoff backlinks, and
the computed code fingerprint. Every accessible repository file or artifact link is reopened and
rehashed during package replay. The public certificate deliberately includes only the context,
config, prompt, schema, provider-receipt, and execution-identity hashes plus typed nonidentifying
runtime metadata; it does not copy raw prompts, configurations, or call ledgers.

Archived or live Paperclip ingestion therefore requires repeatable `--execution-receipt` inputs
whose execution IDs exactly match the archived map batches. Omitting them still produces a
membership-bound version-three package for historical analysis, but that package is permanently
release-ineligible. The local Ollama diagnostic constructs the same receipt automatically from
the inspected Ollama identity and its exact response-bearing per-call ledger; a transport failure
does not fabricate a scientific receipt.

An internally self-consistent package is integrity evidence, not sufficient scientific
verification. `lm verify` requires a repository root and replays every receipt against
the current source bytes. It then deterministically rebuilds every receipt-linked
terminal fragment—including non-estimable and failed-grounding projections—from the
receipt-bound extraction and authoritative publication identity. It also requires an
exact one-to-one match between manifest records, terminal fragments, graph publications,
and certificate eligibility rows. Missing or changed source bytes, a tampered extraction,
a dropped terminal failure, or a graph that differs from this projection causes the
public loader to reject the package.

Every terminal fragment contributes its `PublicationIdentity`, including a non-estimable
fragment. Only study, cohort, arm, contrast, estimate, and evidence-span nodes require an
estimable extraction. The certificate therefore binds a canonical terminal-fragment membership
ledger and requires its publication IDs, the manifest, eligibility rows, and source-graph
publications to agree exactly. An all-non-estimable package can validly contain every publication
with zero studies and zero estimates; it must abstain for the recorded non-estimability issues.

## Cross-publication study/cohort identity

Native fragment IDs are intentionally publication-scoped. Membership-bound version-three and
version-four grounding packages therefore contain a self-hashed
`native-cohort-reconciliation-receipt-v1`
and replay recomputes that receipt from the original typed corpus. The derived graph
retains every publication-specific estimate and evidence span while deterministically
rewriting study, cohort, arm, and contrast references.

Automatic reconciliation never uses titles, cohort labels, author names, sample size,
or other free text. Only exact matches after conservative Unicode/case/whitespace
normalization of reported registry and dataset IDs can create candidates. A component
with disjoint non-empty IDs, a transitive conflict, or more than one candidate identity
from the same publication fails closed as `requires_reviewer`; it is not partially
merged. Distinct cohorts reported in one publication always remain distinct.

Even a conflict-free strong-ID pass has status
`strong_identifier_reconciled_limited`: identifiers can be missing, so it does not
claim exhaustive cross-publication deduplication. For a multi-publication estimable
corpus, `lm verify` adds the blocking
`cross_publication_cohort_reconciliation_incomplete` issue until an external reviewer
artifact explicitly partitions every original study and cohort. The artifact is bound
to the exact corpus and graph hashes, a pseudonymous reviewer hash, a review-protocol
hash, completion time, and a rationale for every group. Its groups may not merge two
identities from one publication, and cohort groups must be nested in the declared study
groups. Accepted reviewer reconciliation has status `reviewer_complete`.

The certificate exposes a versioned `corpus.provenance_assurance` record containing the assurance
status, reason, replay hash, and release eligibility. Raw evidence-graph JSON, graph bundles, and
legacy findings are intentionally still loadable for analysis, but they receive the blocking
`unverified_source_provenance` issue. Only a successfully replayed, membership-bound version-four
native package with a valid extraction context can satisfy the provenance release gate. Versions
one through three remain parseable for explicitly analysis-only compatibility and fail closed at
release. The clearly marked embedded synthetic fixture exercises the remaining mechanics but
always abstains.

To rebuild a package from archived fragments and receipts:

```text
python scripts/build_typed_evidence_corpus.py \
  --fragments artifacts/run/publication_fragments.jsonl \
  --grounding-receipts artifacts/run/grounding_receipts.jsonl \
  --source-manifest artifacts/run/native_source_manifest.json \
  --corpus-cutoff exact-frozen-corpus-v1 \
  --extraction-context artifacts/run/native_extraction_context.json \
  --output-dir artifacts/rebuilt
```

To apply a completed external reconciliation receipt, add:

```text
  --reviewer-reconciliation artifacts/review/reviewer_cohort_reconciliation.json
```

The build emits both the untouched fragment-projected `evidence_graph.json` and the
package-selected `reconciled_evidence_graph.json`. Only the package is a public verifier
input; loading either graph alone intentionally receives the raw-graph provenance gate.

The build command rejects version-three fragments when their exact extraction context is absent,
or when any fragment, corpus, source manifest, cutoff, pipeline, prompt, schema, provider receipt,
or artifact backlink disagrees. It also rejects receipt-linked fragments when the actual receipt set is absent,
duplicated, unreferenced, or cannot be projected exactly. Unlinked non-estimable fragments
remain supported only for provider failures that never produced a native extraction. Omitting
the extraction context deliberately builds at most a version-three analysis-only package even
when source membership is present. Omitting both `--source-manifest` and `--corpus-cutoff` builds
the older version-two analysis package. Neither can release a claim.

Manifest completeness is relative to the supplied frozen manifest. These checks do not establish
that the search protocol retrieved every eligible publication, that retrieval was saturated, or
that publication bias is absent. Those remain separate empirical and calibration obligations.
