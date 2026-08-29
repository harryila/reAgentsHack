# Tracked public-data rights audit

The repository now has a content-silent inventory gate for research data that Git would publish.
It is an engineering control, not legal advice and not a grant of rights.

Run the CI policy-coverage check from the repository root:

```bash
uv run python scripts/audit_public_data_rights.py
```

That command exits successfully only when every monitored or high-risk tracked path matches
exactly one declaration in `configs/public-data-rights-v1.json`, every declared license file has
its frozen hash, and licensed structured payloads contain none of the policy's forbidden
article-text field names. A newly tracked Parquet, JSONL, CSV, text, XML, stdout/stderr, source-line,
or other monitored research-data path fails closed when it is undeclared or matches ambiguously.

Before publishing the repository, use the stricter gate:

```bash
uv run python scripts/audit_public_data_rights.py --require-release-ready
```

This also exits with status 2 when any nonempty collection is honestly declared
`redistribution_not_established`. The current Git-index inventory is **not release-ready**.
In particular, legacy Antiox paths contain PMC-derived source lines, raw command outputs,
third-party abstracts and metadata, provider responses, evidence quotes, and copied derived
exports. No item-by-item redistribution audit or dataset-wide grant has been established for
that mixed collection.

The policy-coverage command remains useful in CI while this blocker is unresolved: it prevents a
new corpus path from silently escaping classification. Passing that command must never be reported
as passing the stricter public-release gate.

## What the report reveals

The scanner treats the Git index as the prospective public inventory. It reads bytes only to hash
them and emits:

- declared path patterns and content classes;
- aggregate file counts and byte counts;
- extension counts;
- counts, but never values, of PMC-like and SHA-256-like path identifiers;
- one canonical inventory hash per collection and one across all classified index blobs;
- aggregate worktree-difference counts;
- policy and release blockers; and
- a canonical full-payload self-hash.

It never emits article text, question text, source quotes, abstracts, labels, predictions,
individual corpus identifiers, or a per-file path list. Undeclared and ambiguous path samples are
represented only by SHA-256 hashes of their relative paths.

An optional one-shot report can be written outside the repository:

```bash
uv run python scripts/audit_public_data_rights.py \
  --output /tmp/lm-public-data-rights.json
```

Existing output is not replaced unless `--force` is explicit.

## Interpreting the statuses

- `project_authored` means the policy identifies a repository-authored aggregate, manifest, or
  synthetic fixture. It does not make a claim about similarly named future files.
- `redistribution_established` requires hash-bound license evidence. For JSON/JSONL payloads, the
  scanner also audits field names for obvious article-text fields without returning any values.
- `redistribution_not_established` is a mandatory public-release blocker. It does not assert that
  redistribution is unlawful; it records that this repository has not established a sufficient
  basis for it.

The bundled MetaSyn notice covers project-authored annotations, while explicitly leaving article
metadata, abstracts, identifiers, review excerpts, and PMC-derived text subject to upstream terms.
The policy therefore covers only the checked annotation inputs and derived constant-prediction
bundle under that notice. Cached MetaSyn article payloads and Evidence Inference article text remain
local and ignored. The legacy tracked Antiox trees are a separate, unresolved boundary.

## Existing-metadata feasibility check

An aggregate-only review of the metadata already present in the repository covered **671 distinct
PMC-like path tokens**. It found **zero authoritative, machine-readable per-document rights
records** that could support automatic reclassification. **Five** documents contain an
unstructured license-shaped URI and are therefore manual-review candidates only; those strings do
not establish a license, bind one to a particular source payload, or make that subset release-ready.
The collection remains `redistribution_not_established`.

Resolving an item requires all seven of the following provenance fields:

1. canonical document identity and version;
2. authoritative license or open-access source, with retrieval timestamp and source-record hash;
3. normalized license URI, type, version, and open-access status;
4. copyright holder and required attribution;
5. permitted reuse scope;
6. an exact hash binding the rights record to the source payload; and
7. a reviewed allow-or-deny decision with rationale.

These are evidence requirements for this repository's publication gate, not a legal conclusion.
The review used existing metadata only and did not inspect or quote article text, browse external
sources, or alter any policy classification.

## Limitations

This scanner cannot decide copyright, database rights, privacy, contractual restrictions, or
whether a generic field embeds third-party prose. Before public release, a qualified human must
either establish and document the applicable rights for every blocked collection or change the
prospective public inventory. Any such decision requires a new aggregate audit and review of the
strict gate; the scanner does not perform deletion, history rewriting, commits, or pushes.
