# Public repository cleanup plan

This plan is a proposal only. It does not delete, move, rewrite history, stage, commit, or push
anything. It records the current gate status, every non-clean collection with its remediation
options, the decisions already made, the contradiction resolved in Phase 0, the surfaces the gate
deliberately does not monitor, and a recommended sequence with its explicit approval points.

Per the audit's own disclosure contract, this document contains counts, path patterns, and hashes
only — never article text, question text, labels, or per-row values.

## Files awaiting staging (operator action)

The following four paths exist in this working tree but are not yet indexed (`git add`ed). Phase 0
was forbidden to run `git add`, so staging them is an operator action outside this plan's scope.
Each breaks a specific check on a fresh clone until it is staged and committed:

- `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json` —
  a fresh clone will not contain this file. `_validate_evidence_inference_item_risk`
  (`src/literature_multiverse/public_artifacts.py:1450-1457`) reads it as the historical pipeline
  fingerprint; the missing file raises `OSError`, which is caught and re-raised as
  `PublicArtifactValidationError("evidence_inference_item_risk_historical_fingerprint_invalid")`, so
  `scripts/validate_public_artifacts.py` fails at the item-risk entry instead of reporting all
  artifacts valid.
- `tests/private_cache_support.py` — a fresh clone will not contain this module. All 28 test
  modules that `from tests.private_cache_support import ...` (the converted private-cache-aware
  suites) raise `ModuleNotFoundError` at collection, so `pytest -m "not live"` collapses at
  collection instead of running.
- `docs/public-repository-cleanup-plan.md` — this document itself. A fresh clone will not contain
  it, so no reviewer of that clone has any record of the non-clean collections, the remediation
  options, or (self-referentially) this staging requirement, until it is staged.
- `docs/superpowers/plans/2026-08-29-phase0-repository-stabilization.md` — a fresh clone will not
  contain this companion planning document. The rights audit and any reviewer of that clone cannot
  see the Phase 0 plan this cleanup plan was written against until it is staged and indexed
  alongside it.

## 1. Gate status

Run: `uv run python scripts/audit_public_data_rights.py`

- `policy_complete`: **true** — every monitored tracked path matches exactly one declared
  collection (`undeclared_file_count`: 0, `ambiguous_file_count`: 0).
- `release_ready`: **false** — twelve collections are honestly declared
  `redistribution_not_established` and are nonempty.
- `tracked_files_total`: 3,757. `audited_candidate_files` / `classified_files`: 3,220.
  `classified_bytes`: 133,293,588.
- `rights_status_file_counts`: `project_authored` 38, `redistribution_established` 8,
  `redistribution_not_established` 3,174.
- Policy file: `configs/public-data-rights-v1.json`, `file_sha256`
  `d37d3ea0376b208928d1c57b7734530d9de1caaa9443aa791c0d2ab7b095e1dc` (this hash changes with any
  future policy edit; nothing in the codebase pins it — see the Phase 0 report for the grep that
  confirmed this).
- `uv run python scripts/audit_public_data_rights.py --require-release-ready` exits 2, as expected
  while any collection above is nonempty and `redistribution_not_established`.

## 2. Non-clean collections and remediation options

Every option below is a choice for a human operator to approve; none has been executed by this
plan.

### The ten legacy Antiox collections

Combined: **3,163 files, ≈130 MB** (129,707,214 bytes), dominated by `antiox_raw_map_payloads`
(`data/raw/map/**`) at **2,658 files / ≈120 MB** (120,338,148 bytes). All ten share
`rights_status: redistribution_not_established` and `public_release_allowed: false`.

| Collection | Path glob | Files | Bytes |
| --- | --- | ---: | ---: |
| `antiox_derived_public_artifacts` | `artifacts/antiox-training/**` | 88 | 1,056,274 |
| `antiox_extracted_records` | `data/extracted/**` | 4 | 897,991 |
| `antiox_manual_patches` | `data/patches/**` | 1 | 508 |
| `antiox_processed_records` | `data/processed/**` | 13 | 248,740 |
| `antiox_provider_payloads` | `data/raw/providers/**` | 14 | 242,457 |
| `antiox_raw_map_payloads` | `data/raw/map/**` | 2,658 | 120,338,148 |
| `antiox_raw_screening_records` | `data/raw/screen/**` | 15 | 1,199,376 |
| `antiox_raw_search_records` | `data/raw/search/**` | 181 | 5,200,487 |
| `antiox_smoke_payloads` | `data/raw/smoke/**` | 28 | 32,133 |
| `antiox_triage_payloads` | `data/raw/triage/**` | 161 | 491,100 |

Options, per collection or for the group as a whole:

1. **Keep private.** Remove the paths from the Git index (`git rm --cached`) and add them to
   `.gitignore` so a future commit cannot silently reintroduce them. Any bytes already reachable
   from prior commits stay in history; scrubbing history (e.g. `git filter-repo`) is a separate,
   higher-risk action that requires explicit operator approval and is out of scope for this plan.
2. **Establish rights.** For each collection, obtain and bind hash-locked `license_evidence`
   (per-source license file plus its SHA-256) the way `metasyn_benchmark_license` and
   `metasyn_benchmark_model_inputs` already do, then reclassify to `redistribution_established`
   after confirming no `established_rights_forbidden_field_names` are present.
3. **Regenerate without third-party text.** Re-derive a narrower artifact that excludes the fields
   the audit's forbidden-field scan flags (e.g. `article_text`, `abstract`, `evidence_quote`,
   `full_text`), then reclassify only the regenerated artifact as `project_authored`.

Given the combined size (≈130 MB, almost entirely in one sub-tree) and the number of distinct
upstream sources implied by 671 distinct PMC-like path tokens (see
`docs/public-data-rights.md` § Existing-metadata feasibility check), option 1 (keep private) is
the lowest-risk default recommendation for this group as a whole; options 2 and 3 remain available
per-collection for any operator who wants a narrower public subset instead.

### `metasyn_derived_diagnostics_with_source_text`

**6 files**, `redistribution_not_established`, blocks release.

Path globs (all under `artifacts/diagnostics/`):
`contextual-grounding-offline-feasibility-suite-v3.json`,
`metasyn-passage-offline-feasibility-audit-v1.json`, `postlive-recovery-v4-join-v1.json`,
`postlive-recovery-v4-public-verify-v1/sequential-audit-state.json`,
`postlive-recovery-v4-public-verify-v1/verification-certificate.html`,
`postlive-recovery-v4-public-verify-v1/verification-certificate.json`.

These diagnostics carry `quote` / `evidence_quote` / `title` fields with verbatim MetaSyn-linked
article text (one file is the HTML rendering of the same certificate as its JSON sibling).
`METASYN_LICENSE.txt` covers the benchmark's project-authored annotations
(`metasyn_benchmark_license`) but explicitly reserves article metadata and excerpts, so it does not
cover this collection.

Options:

1. **Keep private.** Remove these six paths from the Git index; the diagnostics remain available
   locally for the operator's own investigation.
2. **Establish rights.** Obtain a redistribution grant for the specific quoted spans (or their
   upstream source) and bind it as `license_evidence`, as above.
3. **Regenerate without source text.** Produce a variant of each diagnostic that keeps its
   aggregate result (pass/fail, counts, hashes) but drops the `quote` / `evidence_quote` / `title`
   fields, then reclassify the regenerated files as `project_authored` under a new collection.

## 3. Operator roster decision

Collection: `project_authored_evidence_inference_rosters_with_pmc_identifiers` — **5 files**,
815,797 bytes.

Path globs (all under `artifacts/diagnostics/evidence-inference/`):
`fable-retrospective-full-plan-v1.json`, `fable-retrospective-pilot-recovery-v2-exclusions.json`,
`fable-retrospective-pilot30-plan-v1.json`, `fable-retrospective-pilot30-recovery-v2-plan-v1.json`,
`gepa-candidate-search-plan-v1.json`.

These are frozen request rosters and exclusion ledgers: project-authored lists that name public
PMC article identifiers (191/90/7/7/6 distinct across the five files) and Evidence Inference
example identifiers. They contain no article text, labels, or predictions — only identifiers and
project-authored plan/exclusion metadata.

**Operator decision recorded 2026-08-29:** `rights_status: redistribution_not_established`,
`public_release_allowed: false`. Rationale: Evidence Inference 2.0 ships no dataset-specific
redistribution license for its downloaded archive (see `docs/evidence-inference-benchmark.md`);
whether republishing a project-authored list of upstream identifiers requires the same
redistribution basis as republishing the underlying dataset has not been reviewed. The operator
chose the conservative classification — block public release — until that identifier-redistribution
question is reviewed, rather than assume identifiers-only lists are automatically exempt.

This is a decision already made and recorded in the policy's `rationale` field, not an open
question needing further options; it is listed here for visibility rather than for remediation
choice. It can be revisited (and reclassified to `redistribution_established` or narrowed) once
that review happens.

## 4. Contradictions resolved in Phase 0

**`local_suite_identifier_receipts`.** Before Phase 0, one repository test asserted this collection
was indexed with `file_count: 2` (`freeze_receipt.json`, `predictions.jsonl`) while `.gitignore`
(`/artifacts/benchmarks/local-suite-v1/freeze_receipt.json`,
`/artifacts/benchmarks/local-suite-v1/predictions.jsonl`) and the CI aggregate-only step both
forbid indexing them — the test encoded a state the rest of the system had already made
unreachable. Phase 0 replaced that test
(`test_repository_rights_report_surfaces_indexed_local_suite_receipts_as_content_silent_blocker`)
with `test_local_suite_identifier_receipts_stay_declared_but_unindexed`, which asserts the correct,
currently-true state: `file_count: 0`, `extension_counts: {}`, no `policy_blockers` entry for this
collection (its `allow_empty: true` declaration is exactly what makes an empty collection
non-blocking), and no `release_blockers` entry (a `redistribution_not_established` collection is
only a release blocker when it is nonempty). The declaration itself is intentionally kept — with
`allow_empty: true` — so that if either the `.gitignore` rule or the CI step is ever relaxed and
these files become indexed again, the audit fails closed instead of silently republishing
identifier-bearing receipts.

## 5. Unmonitored-by-design surfaces

These are not gate gaps; they are outside the monitored surface by the policy's own suffix/prefix
rules (`_is_monitored` in `src/literature_multiverse/public_data_rights.py`), recorded here for
visibility.

- **14 `prompts/*.md` files.** Markdown is on the policy's safe-extension list
  (`.css`, `.html`, `.js`, `.lock`, `.md`, `.py`, `.sh`, `.toml`, `.yaml`, `.yml`) and `prompts/` is
  not itself a monitored prefix, so these stay unmonitored regardless of the new
  `project_authored_prompt_templates` collection (which only matches the three `.txt` prompts,
  monitored by suffix). This collection's own declaration (`prompts/**`, `rationale`) is itself the
  operator-visible record that the Markdown prompts were reviewed and judged to hold no article
  text, abstracts, labels, or benchmark rows.
- **`artifacts/paper/*.json` files outside the two monitored `artifacts/paper/**` sub-prefixes**
  (`artifacts/paper/metasyn-benchmark`, `artifacts/paper/metasyn-fixed-positive-test` remain
  monitored and declared as before):
  - `artifacts/paper/budgeted-verification-simulation-200.json`
  - `artifacts/paper/calibration-simulation-100.json`
  - `artifacts/paper/closed-corpus-local-audit.json`
  - `artifacts/paper/harvester/validation_summary.json`
  - `artifacts/paper/meta-simulation-200.json`

  `.json` is not on the deny-by-default suffix list, so these five files are unmonitored unless a
  future path token or prefix change brings them in scope.
- **`.html` files outside monitored prefixes.** None. The only tracked `.html` file in the
  repository is `artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/verification-certificate.html`,
  which is now inside the new `artifacts/diagnostics` monitored prefix and declared under
  `metasyn_derived_diagnostics_with_source_text` (§2). No tracked `.html` file falls outside a
  monitored prefix.

## 6. Recommended sequence and approvals required

1. **No action required now.** `policy_complete` is true and CI's default gate
   (`scripts/audit_public_data_rights.py` with no flag) passes; nothing is silently undeclared.
2. **Operator review of the roster decision (§3).** The `redistribution_not_established`
   classification for the five PMC/example-identifier rosters is conservative by design; an
   operator may revisit it once the identifier-redistribution question has been reviewed. Changing
   it requires editing `configs/public-data-rights-v1.json` and is itself the kind of policy edit
   this audit gate exists to keep honest — no code change is needed beyond the JSON.
3. **Operator decision per Antiox collection or as a group (§2).** Choose keep-private,
   establish-rights, or regenerate, per collection. Keep-private is the lowest-risk default given
   the size and provenance uncertainty of the combined ≈130 MB tree.
4. **Operator decision for `metasyn_derived_diagnostics_with_source_text` (§2).** Choose
   keep-private, establish-rights, or regenerate-without-source-text for the six diagnostics.
5. **If keep-private is chosen for any collection:** removing tracked paths from the Git index
   (`git rm --cached`) is a distinct, explicit operator action, separate from this plan. If those
   bytes were ever committed, they remain reachable from history; a history rewrite
   (e.g. `git filter-repo`) to fully purge them requires its own explicit operator approval and is
   out of scope here.
6. **Re-run the gate after any reclassification or index change:**
   `uv run python scripts/audit_public_data_rights.py --require-release-ready` — the repository is
   release-ready only once every nonempty collection is `project_authored` or
   `redistribution_established`.

No step in this plan has been executed beyond producing this document and today's test/policy
changes recorded in the Phase 0 report; every remaining step above is explicitly deferred to
operator approval.
