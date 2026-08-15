# Literature Multiverse — Implementation Plan

**Date:** 2026-08-15

**Status:** v1.0 — coding-ready companion to design v2.3; product code starts only after Harry's
go-ahead

**Normative design:** `docs/superpowers/specs/2026-08-15-literature-multiverse-design.md`

**Hard deadline:** demo freeze Sun Aug 16 at 9:45 AM PT; submit at 10:15 AM PT

## 1. Objective and non-negotiable outcome

Build one offline, reproducible hackathon demonstration for one empirically selected scientific
question. It must turn a retrieved paper corpus into:

1. a lossless paper ledger;
2. atomic, source-grounded findings;
3. honest disagreement estimates;
4. a paper-balanced moderator analysis;
5. either a controlled conditional-pattern narrative (Variant A) or a complete no-explanation
   narrative with residual contradictions and evidence gaps (Variant B);
6. a frozen Streamlit demo that makes no network calls.

The product is not a general arbitrary-question service this weekend. Corpus scale, an agent remap,
and overnight confirmation are additive. A correct 30-paper Variant B beats a fragile 300-paper
Variant A.

## 2. Source of truth and change protocol

Authority is deliberately split:

1. design v2.3 defines global product, statistical, stage, and artifact invariants;
2. the locked `QuestionConfig` defines topic-specific values allowed by the design;
3. this plan defines dependency order, file ownership, commands, and acceptance tests;
4. executable models and generated JSON Schemas must match those authorities exactly;
5. fixtures exercise the contracts; root context, research notes, and reviews are history only.

For global invariants, the design wins. For permitted topic-specific values, the locked config wins
over examples in the design. Neither this plan, a test fixture, nor a generated schema may silently
change scientific semantics.

If implementation exposes a design defect, stop that workstream, write the proposed deviation into
the design's review trail, update the affected contract/test first, and only then code it. No silent
method changes after real outcomes are inspected. Any post-G2 config change creates a new hash and
must be recorded in `docs/planning/triage.md`.

## 3. Planned repository shape

```text
pyproject.toml
.python-version
configs/questions/
  triage-c.yaml                     # status: triage candidate; never a production input
  triage-a.yaml
  fixture-a.yaml
  fixture-b-story.yaml
  fixture-b-m4.yaml
  fixture-b-incomplete.yaml
  <locked-qid>.yaml
  <locked-qid>.patches.yaml
prompts/
  extraction.md
  moderator_proposal.md
  targeted_remap.md
  quote_verification.md
schemas/
  extraction.<qid>.schema.json       # generated snapshot, never hand-edited
src/literature_multiverse/
  __init__.py
  config.py                          # QuestionConfig + ModeratorSpec + locked-state validation
  models.py                          # PaperRecord, FindingRow, run/audit/remap models
  schemas.py                         # topic-aware strict JSON Schema generation
  paths.py                           # all qid/stage paths; no ad-hoc strings elsewhere
  lineage.py                         # canonical hashing, run.json, consistency checks
  paperclip_cli.py                   # argv-only subprocess wrapper + raw archival
  search.py
  screen.py
  extract.py
  normalize.py
  grounding.py
  disagreement.py
  moderators.py
  resampling.py
  tree.py
  contradictions.py
  evidence_gaps.py
  remap.py
  audit.py
  export.py
scripts/
  generate_fixture.py
  s0_smoke_test.py
  triage_probe.py
  s1_search.py
  s2_screen.py
  s3_extract.py
  s4_normalize.py
  s5_analyze.py
  s6_remap.py
  s7_export_demo.py
  generate_baseline.py
  audit_findings.py
  verify_quotes.py
  verify_demo.py
app/streamlit_app.py
docs/demo/variant_a.md
docs/demo/variant_b.md
tests/
  fixtures/raw/probe_map_m_2bc51e4b.txt
  test_config.py
  test_models.py
  test_schemas.py
  test_lineage.py
  test_screen.py
  test_extract_parser.py
  test_normalize.py
  test_grounding.py
  test_disagreement.py
  test_moderators.py
  test_resampling.py
  test_tree.py
  test_contradictions.py
  test_export.py
  test_app.py
  test_pipeline_fixture.py
```

Avoid a generic agent framework, database, task queue, web backend, or frontend stack change.

## 4. CLI and stage contracts

All Python CLIs accept `--question <qid>`, use fixed paths from `paths.py`, refuse overwrite unless
`--force`, log structured summaries, and return nonzero on contract failure. Default tests never
touch the network; live tests require both `@pytest.mark.live` and an explicit `--live` CLI flag.
In command blocks, replace the safe `QUESTION_ID=selected-question` assignment once G2 has produced
the real locked qid; do not paste angle-bracket placeholders into a shell.

| Stage | Reads | Writes | Hard postcondition |
|---|---|---|---|
| s0 | env, smoke config, archived probe | `data/raw/smoke/*`, `run.json` | G1b assertion report; secrets never serialized |
| triage_probe | triage config, logged s2 sample of exactly 10 papers | isolated `data/raw/triage/<candidate>/` raw/ledgers/report | strict candidate schema; cannot write or become an upstream of s3–s7 |
| s1 | triage/locked config | per-query raw JSON, `candidate_papers.jsonl`, `run.json` | every doc-id/query-family occurrence represented; `--all` recorded |
| s2 | s1 candidates | `screened_papers.jsonl`, include/exclude ID files, dedupe log, `run.json` | one canonical identity cluster disposition; alternates preserved |
| s3 | locked config, complete s2 ledger + include/exclude lists | raw map output, `papers.jsonl`, `findings.jsonl`, quarantine, `run.json` | map only includes; exactly one terminal PaperRecord per s2 paper; no orphan findings |
| s4 | s3 ledgers, optional human patches | `papers.parquet`, `findings.parquet`, normalization report, `run.json` | strict types, one extraction tuple, referential integrity |
| s5 | processed ledgers, `g3_gate.json`, optional remap side table or explicit frozen interruption checkpoint | exact §4.5 analysis inventory incl. `m4_checkpoint.json`, `m4_gate.json`, `headline.json`, `run.json` | trust failure refused; deterministic complete table incl. losers; A/B selected by gate status; a frozen interruption is terminalized, never exported as a partial run |
| s6 | v1 primary cohort, residuals, approved proposal | raw remap, remap parquet, quarantine, `trace.json`, `run.json` | whole-v1 coverage attempted; exact echo-back join; rule-based decision |
| s7 | candidate processed/analysis/audit/scripts, optional frozen-v1 fallback | staged release incl. `release_selection.json`, then `artifacts/<qid>/demo/` | exact inventory/hashes/offline startup; selected and rejected-attempt lineage separate; previous good release retained |

s7 builds into a temporary sibling directory, validates it, then atomically promotes it. A failed
candidate export never damages the last known-good `demo/` release.

## 5. Workstreams and merge discipline

After Phase 1 contracts are frozen, three workstreams may proceed in parallel:

- **A — live ingestion:** Paperclip wrapper and s0–s3.
- **B — fixture science:** fixture, s4, s5, statistics and analysis artifacts.
- **C — frozen demo:** s7, scripts and Streamlit against fixture artifacts.

They share only committed contracts from `config.py`, `models.py`, `schemas.py`, `paths.py`, and
`lineage.py`. Do not edit those five files concurrently. Integrate at M1 (raw parser), M2.5
(fixture release), M3 (real data), and M4 (narrative decision). Each integration runs the full
offline suite.

## 6. Phase 0 — planning baseline and safe workspace

### Task 0.1 — planning handoff

Files:

- design v2.3;
- this implementation plan;
- root-context normative notice;
- prewritten `docs/demo/variant_a.md` and `variant_b.md` renderer templates.

Actions after coding approval:

1. Confirm `.env` is ignored and mode `0600`; never print or parse secret values in tests.
2. Create the repository's initial commit containing planning/smoke artifacts but not `.env`.
3. Record the starting commit as the first `code_version`.

Acceptance:

- `git check-ignore .env` succeeds;
- `git status --short` has no unexpected paths after the baseline commit;
- no tracked file matches common secret-key patterns.

## 7. Phase 1 — scaffold and freeze contracts (M0)

### Task 1.1 — Python project

Create `pyproject.toml` and `.python-version`; pin Python 3.12 and direct runtime dependencies from
the design. Add pytest and ruff as development dependencies. Do not add optional science/UI
packages until a task consumes them.

Commands:

```bash
uv sync --python 3.12
uv run python --version
uv run pytest -q
uv run ruff check .
```

Acceptance: Python reports 3.12.x; empty/scaffold suite and lint pass.

### Task 1.2 — paths and lineage

Implement `paths.py` and `lineage.py` first. Canonical hashes serialize sorted validated config
data, rendered prompt, schema, input/output file hashes, and code version. `run.json` writes
atomically. A unit test proves that changing any eligibility rule, outcome family, moderator bin,
prompt byte, schema byte, or the code-owned closed direction-alias table changes the appropriate
config, schema, prompt, or code hash.

Acceptance: stale input, mixed tuple, missing hash, and dirty code each produce distinct stable
errors; diagnostic overrides cannot pass s7 without explicit `--allow-dirty-demo`.
For s5 frozen-incomplete finalization, tests additionally prove that run ID/timestamps/argv derive
from the checkpoint rather than invocation time and that a second finalization produces identical
run and analysis hashes. The release replay test supplies the complete immutable export input set,
including baseline/template/selection inputs, and then requires identical release hashes.

### Task 1.3 — exact models and schema generator

Implement §4.1–§4.4 in `models.py`, `config.py`, and `schemas.py`.

Required tests:

- `paper_id` priority DOI → PMID → doc ID and preprint/published preferred identity;
- every PaperRecord terminal status invariant, including ineligible/failed/zero-finding;
- accepted + quarantined = raw finding counts for every successful PaperRecord;
- model-produced identity is absent from extraction schema;
- canonical direction is exactly `increase|no_effect|decrease|mixed|unclear`, accepts only the
  closed legacy aliases, quarantines JSON-null direction, and maps the string `"null"` to
  `no_effect`;
- topic moderator keys/types come only from locked config; extra keys fail;
- `status: triage` can run only s1/s2/`triage_probe`, the probe hard-caps at 10, and its run record
  is rejected as an upstream by s3–s7; `status: locked` cannot omit primary outcome, direction,
  moderator/permutation, anchor, Variant-B, or demo fields;
- locked configs reject more than six tested/two descriptive moderators, outcome family as a tested
  moderator, invalid kind/permutation pairs, and any base moderator marked as remap-sourced;
- locked configs require a `demo.spoken_question` of at most 20 words for deterministic script
  timing;
- `fixture_mode=true` works only for the four named fixture qids with explicit `--fixture`; any
  production/live-provider combination fails;
- finding IDs are deterministic within a map and differ across map result IDs;
- remap duplicate/unmatched IDs fail.
- the `release_selection.json` model permits only the six design §4.5 state rows, four literal
  non-null scaled failure codes, exact nullable hash keys, and selected-release identity/hash
  equality; and
- `completion_mode=frozen_incomplete` is legal only for a complete s5 run with a matching
  checkpoint hash and `m4_gate.status=incomplete`.
- the `m4_checkpoint.json` union accepts only the exact not-applicable placeholder or a
  frozen-incomplete wrapper around a complete `checkpoint_version=1`,
  `checkpoint_status=running_snapshot` source object whose canonical hash matches.

### Task 1.4 — fixture config and generator skeleton

Create four locked fixture configs and `generate_fixture.py --all`. Every scenario writes both
ledgers with one consistent dummy extraction tuple, frozen authoritative content/section lines,
20 deterministic human-audit decisions, one verifier decision per exact-grounded finding, anchor
expectations, and a production-schema `baseline.json` marked `source=fixture_stub`. Fixture baseline
text is deterministic and never calls an LLM; production s7 rejects `fixture_stub` unless both the
config and CLI explicitly set fixture mode.

The shared data includes every pathology in design §2. Scenario outcomes are fixed:

- `fixture-a`: G3 passes and one moderator passes every M4 rule;
- `fixture-b-story`: trust passes, story fails, and residual count is zero;
- `fixture-b-m4`: G3 passes, full M4 completes with no passing moderator, and a residual pair exists;
- `fixture-b-incomplete`: G3 passes, a fixture-only interruption checkpoint leaves M4 partial;
  the test then invokes the real frozen-incomplete finalizer, whose complete s5 stage selects
  Variant B with `m4_incomplete`.

The pipeline test derives `g3_gate.json`, s5 artifacts, `release_selection.json`, manifest, and the
rendered script; the generator does not forge their expected outputs. Fixture-only fault injection
is impossible without `--fixture` and is rejected for any non-fixture qid.

M0 exit criteria:

- all contract tests pass;
- generated fixture schema has `additionalProperties:false` recursively;
- exact planned stage paths can be produced without creating outputs;
- contracts are committed before workstreams split.

## 8. Phase 2A — live ingestion and G1b (M1)

### Task 2A.1 — Paperclip subprocess boundary

`paperclip_cli.py` accepts an argv list only, never a shell string. It captures stdout/stderr/exit
code/timestamps, archives raw bytes before parsing, redacts key-like text, and maps only observed
transient/rate-limit failures to retry. No command is retried merely because parsing failed.

### Task 2A.2 — archived parser contract

Copy the existing raw probe to the test fixture as text despite its historical `.json` name. Parse
exactly four authoritative envelope doc IDs, four terminal papers, six raw findings, six accepted
findings, zero quarantines, one reference-section-flagged review-pollution finding, five
non-section-flagged findings, and two clean ineligible zero-finding papers. Inject `paper_id`,
`doc_id`, result ID, and array position locally. Add a minimal frozen authoritative line/section
fixture for the cited ranges so the offline test derives—not hard-codes—the one References flag.

### Task 2A.3 — remaining live assertions

Implement `s0_smoke_test.py --live` to execute design §5.1 assertions. The resume probe uses at
exactly ten papers, is killed only after a recorded completion, resumes the same map ID, and
reconciles the final set to the original include set.

Commands:

```bash
uv run pytest -q tests/test_extract_parser.py tests/test_lineage.py
uv run python scripts/s0_smoke_test.py --question fixture --live
```

G1b exit criteria: every assertion is machine-readable in the smoke report; no >10-paper live map
is allowed until it passes.

### Task 2A.4 — s1/s2/s3

Implement search provenance, deterministic filters, exact and fuzzy dedupe, terminal paper ledger,
raw archival, normalization and quarantine. Fuzzy matches never auto-merge below the pinned score;
ambiguous pairs are logged for human disposition. A reconciliation test checks:

```text
s2 included + s2 excluded = identity-deduped candidates
map success + map failure + not-mapped deterministic exclusions = s2 total
sum(PaperRecord.raw_finding_count) = raw model findings across successful maps
sum(PaperRecord.accepted_finding_count) = accepted FindingRows
sum(PaperRecord.quarantined_finding_count) = quarantine rows
for each successful paper: accepted_finding_count + quarantined_finding_count = raw_finding_count
set(FindingRow.paper_id) ⊆ set(PaperRecord.paper_id)
```

## 9. Phase 2B — fixture science vertical slice (M2.5)

### Task 2B.1 — s4 normalization and grounding

Implement strict parquet dtypes, config mappings, patch application with selector + expected old
value + reason, mechanical quote/line grounding, section resolution, and reports. A patch that
matches zero/multiple rows or the wrong old value fails.

Tests cover Unicode/whitespace quote matching, multi-line ranges, missing/mismatch/unverifiable
statuses, banned sections, unresolved quote⇒direction verification, nullable `Int64`, version
mixing, and paper joins. Unresolved verification disagreements remain visible but cannot enter the
primary headline cohort until adjudicated. Duplicate, unknown, or missing requested finding IDs in
`verification.json` fail reconciliation; absence is never interpreted as agreement.
Tests recompute every §4.5 quality denominator: verifier `disagree`/`unverifiable` count as non-
agreement, adjudication changes row inclusion but never model agreement, and a zero quarantine
denominator cannot pass G3.

### Task 2B.2 — disagreement

Implement fixed-log(3) normalized entropy, paper-balanced finding entropy, modal-paper entropy,
tie/exclusion accounting, majority agreement, paper bootstrap intervals, and pure G2/G3 gate
functions that emit every observed value and reason code.

Tests use hand-computable distributions: unanimous = 0; balanced three-class = 1; duplicating one
paper's rows does not change paper-balanced class proportions; paper-modal ties become `mixed` for
primary exclusion and `unresolved` in the 4-class sensitivity.
The 4-class unresolved sensitivity uses `log(4)`, is never compared with the primary estimate, and
all direction/tie exclusions reconcile. A fixture with G2's upper interval bound ≥0.40 but G3's
lower bound <0.40 passes only the permissive G2 disagreement rule.

### Task 2B.3 — moderator CV

Implement the exact §6.2 pipeline. Splits are created once per moderator subset and shared with the
baseline. If no all-class k≥3 split exists, output exploratory/insufficient—never silently skip.
Before one-hot encoding, pool levels represented in fewer than three **distinct papers** into the
model-only `__OTHER__`; never use row-count `min_frequency`, and never narrate `__OTHER__`.

Tests prove:

- no paper crosses train/test;
- baseline and model score identical rows with identical paper weights;
- probabilities are label-aligned, floored and renormalized;
- a binary observed subset still emits the full canonical probability array, while a one-paper rare
  class cannot manufacture feasible CV;
- outcome-correlated missingness alone gives no artificial gain;
- duplicating the 15-finding paper changes headline ΔLL/rank only within numerical tolerance;
- a rare level repeated across many findings in one paper is still pooled;
- all config moderators appear in output.

### Task 2B.4 — resampling and M4 gate

Implement paper-aware permutations, Westfall–Young max statistic, paper bootstrap checkpoints,
stable seed and the exact Boolean gate. Checkpoint writes are atomic, content-addressed, and
resumable. They contain the complete design §6.5 identity, budget, index, result, error, and artifact
hash fields—not only progress counters. Degenerate resamples remain in denominators with reason
codes.

Implement `s5_analyze.py --finalize-incomplete-from PATH` as a mutually exclusive path. It validates
and content-addresses the named canonical checkpoint, emits the exact §4.5 wrapper containing the
full source object plus its canonical-object hash, performs no further resampling, completes only
the deterministic descriptive/residual/gap work, and emits a complete s5 `run.json` with
`completion_mode=frozen_incomplete` alongside an explicitly incomplete M4 gate and Variant-B
headline. Ordinary s5, `--resume`, and frozen finalization cannot be combined. A checkpoint whose
registered attempt/bootstrap budget is already terminal is refused by the incomplete finalizer.

Fixture acceptance:

- planted moderator ranks first, ΔLL ≥0.02, adjusted p<0.10, and top-three bootstrap frequency
  ≥0.80;
- the same-subset conditional agreement improves on its global majority by ≥10 points, two
  supported levels have different unique modes, and the same named contrast/directions/material
  gain recur in ≥60% of all 200 bootstraps;
- planted null moderators do not pass adjusted p<0.10;
- within-paper moderator without summary permutation cannot headline;
- a tied same-subset global mode or tied regime mode cannot headline, and paper coverage uses all
  distinct primary-cohort papers as its denominator;
- fewer than 100 successful permutations in 125 attempts fails the permutation rule;
- Westfall–Young uses the maximum over the complete permutation-eligible family and both raw and
  adjusted values use the add-one formula;
- same-direction regimes, a tied global/regime mode, or material gain below 0.10 fails M4 even if
  ΔLL and adjusted p pass;
- hand-calculated rows reproduce Q, P, D, the support-maximized contrast, and rendered headline;
- flipping any M4 requirement deterministically selects Variant B;
- `headline.json` numbers recompute exactly from rows;
- killing a fixture run after a checkpoint leaves an ineligible partial run; finalizing that exact
  checkpoint emits typed incomplete slots plus `m4_incomplete`, never performs the next draw, and
  reproduces identical scientific/run hashes when repeated;
- the frozen wrapper's source hash equals the canonical serialization of its complete embedded
  checkpoint, its wrapper-file hash remains separate, and a forged wrapper status/hash fails; and
- changing the checkpoint bytes, qid, config/code/cohort/G3/input hash, seed, budget, completed
  index set, or canonical archive path makes finalization fail before any output is promoted.

### Task 2B.5 — tree, contradictions, evidence gaps

Implement a depth≤3 paper-weighted tree and separate JSON renderer spec; never serialize a sklearn
plot. Grey audited leaves <5 papers. Contradictions are same-primary-family, opposite-direction,
share ≥2 fields, and expose distance components. Evidence gaps always produce rows for declared
moderator levels even when count is zero. Generate the full config-frozen
axis-level × primary-endpoint Cartesian product with the exact §6.5 columns and `empty`/`sparse`/
`supported` boundaries; never infer the grid from observed rows.

Tests prove zero-count cells survive, status boundaries are exact at 0/1/5 grounded papers, entropy
stays null without three classifiable papers and two directions, and both citations in every
rendered contradiction resolve through the paper ledger.

## 10. Phase 2C — frozen export and UI (M3.5)

### Task 2C.1 — exact export

Implement the §4.5 allowlist. Reject missing and extra files, hash mismatches, stale inputs,
unreconciled counts, invalid `m4_gate.json`/headline values, manifest scalars that do not recompute,
and a missing selected script. Build/validate a candidate release before atomic promotion; preserve
the last good release. s7 writes `release_selection.json` before the manifest: selected-release
hashes describe the bundled corpus, while a rejected scaled candidate is recorded separately with
nullable hashes through its last completed stage. Neither record contains an s7/manifest hash.
Validate typed `not_run`/`incomplete` tree, permutation, and bootstrap components; skipped work never
authorizes a missing file or fabricated zero-valued result.

The exporter computes funnel, grounding, quarantine, and exclusion fields from the complete bundled
paper/finding ledgers; audit accuracy from `audit.json`; and cross-model agreement from
`verification.json` joined to the ledger-defined request set. Tests deliberately make the 20-row
audit unrepresentative and prove it cannot change any whole-ledger rate. Encode the six-row §4.5
release-selection state table literally: the four retained-v1 dispositions accept only
`scaled_incomplete`, `scaled_artifact_integrity_failed`,
`scaled_ledger_reconciliation_failed`, or `scaled_trust_or_offline_validation_failed` with their
specified status; no generic/free-text failure is accepted. Validate the exact nullable
stage/evidence keys, contiguous `last_completed_stage`, required selected hashes, and equality of
selected release identity/size/hashes to either frozen v1 or the promoted scaled attempt. A
scientifically finalized `m4_incomplete` scaled corpus is not a technically incomplete attempt and
must still promote after all trust/lineage/offline checks pass. Tests pin the design's deterministic
integrity → reconciliation → explicit trust/offline failure → otherwise unfinished precedence when
more than one symptom is present.

Derive `release_id` only from the selected scientific/lineage hashes. Set manifest `created_at` to
the selected s5 run's fixed `completed_at`, never s7 wall time; s7 records its operational package
time only in the external run log. Release replay receives the same processed ledgers,
G3/audit/verification files, s5 outputs, immutable one-shot baseline, template bytes, config/code,
and frozen-v1/scaled-attempt selection inputs, and must reproduce every bundled hash without
calling a provider or regenerating an upstream artifact.

### Task 2C.2 — both demo scripts

Write `docs/demo/variant_a.md` and `variant_b.md` before real s5 output. Each is ≤90 seconds at
150 spoken words/minute, names only fields available in the exact artifact inventory, and includes
the retrieved-corpus and predictive/non-causal caveats. Post-hoc language belongs in the separately
validated trace panel when a remap was actually attempted; the core scripts do not imply one.
Source templates may use only allowlisted
`{{artifact.json.path}}` tokens. s7 renders the selected template from the validated bundle and
fails if any `{{...}}` token, missing path, type mismatch, unapproved prose branch, or more than
225 rendered spoken words remains.

### Task 2C.3 — Streamlit

The app reads only `artifacts/<qid>/demo/`. Startup validates manifest/schema/hashes. It renders:

1. question + retrieved-corpus funnel;
2. grounding/exclusion strip;
3. baseline vs primary global direction;
4. complete moderator table;
5. human-readable tree/Variant B residual panel;
6. paper-joined evidence cards;
7. optional trace and contradictions.

Leaf selection uses a stable leaf ID/selectbox fallback; it must not depend on a fragile custom
Plotly click event. No import or callback can reach Paperclip/Claude/network.

Acceptance:

```bash
uv run python scripts/generate_fixture.py --all --force
uv run pytest -q tests/test_export.py tests/test_app.py tests/test_pipeline_fixture.py
```

`test_pipeline_fixture.py` runs s4→G3→s5→s7 for all four qids in explicit fixture mode, with network
disabled, and proves no baseline/provider call occurred. `streamlit.testing.v1.AppTest` loads every
release without exceptions and finds the four core views, corpus qualifier, created-at footer, and
evidence quote/line link. Variant B fixtures exercise non-null and zero-pair residual states plus all
three selection reasons; the incomplete fixture runs partial s5 → explicit checkpoint finalization
→ s7 and proves the partial record alone is refused; every spoken sentence and release-selection
hash resolves. Export tests
also (a) reject a staged scaled candidate and prove the fallback bundle's selected hashes are v1
while its `scaled_attempt` hashes are separate, and (b) promote a valid scaled A→B change. No
manifest or release-selection record contains a recursive s7/self hash. The state-matrix test
exercises all six dispositions, all four literal failure codes, illegal status/code/hash
combinations, and a valid scaled frozen-M4-incomplete promotion.

## 11. Phase 3 — empirical topic lock (M2)

Counts may start after M0, probes require M1/G1b, and scoring requires Task 2B.2. Search/count work
may overlap Phase 2A after M0. For each candidate qid, run the probe only after
the archived parser and live G1b assertions pass; calculate G2 only with the tested disagreement/
gate functions from Task 2B.2:

```bash
CANDIDATE_QID=triage-c
uv run python scripts/s1_search.py --question "$CANDIDATE_QID" --all --force
uv run python scripts/s2_screen.py --question "$CANDIDATE_QID" --force
uv run python scripts/triage_probe.py --question "$CANDIDATE_QID" --paper-count 10 --force
```

1. Run C and A searches with all query families and `--all`; run B/C+B counts only.
2. Dedupe and deterministic-screen before estimating usable primary volume.
3. Select ten probe papers per topic using a fixed, logged stratification across query family and
   publication era—not top relevance alone.
4. Extract/inspect only the isolated probes; calculate the exact G2 criteria. Never copy their raw
   or normalized rows into the winner's production corpus. Human-label every accepted finding for
   target intervention/comparator/outcome, average within paper, then take the unweighted mean over
   papers with findings; record zero-finding papers separately and fail an undefined denominator.
5. For C, count distinct papers on both sides of the dose threshold; do not count findings.
6. Write `docs/planning/triage.md` with raw counts, formulas, intervals, relation-purity audit,
   content tiers, outcome concentration, moderator support, and the winning/fallback decision.
7. Produce one locked config with primary family/endpoints, direction meanings, moderator
   family/kinds/permutation rules, bins, anchors and expected values, recovery statement, and demo
   copy.

G2 exit command:

```bash
QUESTION_ID=selected-question
uv run python -m literature_multiverse.config validate \
  "configs/questions/$QUESTION_ID.yaml" --require-locked
```

If no topic passes, follow the precommitted escalation ladder. Do not weaken a threshold to keep a
favorite topic.

## 12. Phase 4 — real v1 and G3 (M3)

Commands:

```bash
QUESTION_ID=selected-question
uv run python scripts/s1_search.py --question "$QUESTION_ID" --all --force
uv run python scripts/s2_screen.py --question "$QUESTION_ID" --force
uv run python scripts/s3_extract.py --question "$QUESTION_ID" --target-papers 30 --force
uv run python scripts/s4_normalize.py --question "$QUESTION_ID" --force
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --sample-size 20 --seed 20260815
uv run python scripts/verify_quotes.py --question "$QUESTION_ID" --scope grounded-v1
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --finalize-g3
```

1. Run s1–s3 on ~30 papers using the locked config and reconcile the paper ledger.
2. Run s4; inspect quarantine, grounding, section, content-tier and outcome-family reports.
3. Audit the fixed-seed 20-row sample and every anchor; record raw decisions in `audit.json`.
4. Run independent grounded quote⇒direction verification on all grounded v1 rows, reconcile one
   decision per requested finding ID, and record any human adjudications; disagreements enter the
   human queue and cannot enter the primary cohort unresolved.
5. Evaluate G3 trust and story Booleans separately.

If an extraction prompt changes, create a new prompt version/cfghash and rerun the **entire v1
include set**. Never append a revised subset or use `--allow-mixed` for analysis. Delta-audit the
affected fields plus five controls.

G3 exit criteria are exactly design §5.3. On second trust failure, remove the failing fields or
pivot; no scientific release may be built from a trust-failed cohort. If trust passes but story
viability fails, atomically select Variant B with `selection_reason=g3_story_not_viable`, run only
the descriptive s5 outputs needed for disagreement, residuals, moderator eligibility/status, and
evidence gaps, and skip M4 inference/remap. A topic pivot is optional only if there is time to
restart at G2 with a new locked config; do not weaken the gate.

For `g3_gate.action=select_variant_b_story`, the ordinary s5 command reads that action and emits the
descriptive Variant-B branch without starting permutation/bootstrap inference:

```bash
QUESTION_ID=selected-question
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" --force
```

## 13. Phase 5 — real analysis and atomic narrative decision (M4)

Run:

```bash
QUESTION_ID=selected-question
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" --force
uv run pytest -q
uv run ruff check .
```

s5 must emit `m4_gate.json` with status, each §6.3 rule, threshold, observed value/pass/fail when
completed (null plus reason when not), selected variant/reason, seed, cohort hash, and artifact
hashes. The selection is atomic and deterministic.

- **Variant A:** generate audited tree/headline; proceed to optional M4.5.
- **Variant B:** mark tree exploratory, write not-run trace, cut remap, polish contradictions and
  evidence gaps. Never manually relabel it A.

If the registered 200 bootstraps or 100 successful permutations do not finish, select Variant B
with `selection_reason=m4_incomplete`; do not treat a partial battery as a failed moderator result
or hand a partial run to s7. This means an interrupted run before its 125-attempt/200-bootstrap
budget. Choose the last clean content-addressed checkpoint and terminalize it explicitly:

```bash
QUESTION_ID=selected-question
CHECKPOINT_PATH=data/checkpoints/selected-question/s5/approved-checkpoint.json
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" \
  --finalize-incomplete-from "$CHECKPOINT_PATH" --force
```

Replace the safe checkpoint assignment with the reviewed repository-relative checkpoint path; do
not use the most recent file without checking its qid/config/code/cohort/G3/input hashes and budget
counters. The command must perform no additional resampling and must emit a complete stage record
whose `m4_gate.status` remains `incomplete`; rerunning it from that frozen checkpoint must reproduce
all s5 hashes. Replaying s7 reproduces release hashes only when the separately frozen
baseline/template/selection and remaining export inputs are also identical. Reaching 125 attempts
with fewer than 100 permutation successes is instead a completed failed rule and
`m4_no_moderator`. A finalized incomplete-M4 release is allowed only when
G3 trust and all descriptive, residual, evidence-gap, grounding, script, and bundle checks pass.

This phase is entered only when G3 story viability passed. A G3-story failure already selected
Variant B in Phase 4 and records M4 as `not_run`, rather than pretending the moderators failed tests
that were never executed.

Before **any** s7 export—including a G3-story Variant B—generate `baseline.json` once for the exact
cohort hash and archive its raw response, prompt/model/version, and inputs. Subsequent s7 runs copy
it and fail if an input hash changed. A new scaled cohort receives one separately archived call;
retries for the same cohort are forbidden. A failed one-shot writes the exact `unavailable` schema
and UI state; it does not block the majority baseline or release.

## 14. Phase 6 — optional agent remap (M4.5)

Only Variant A enters this phase.

```bash
QUESTION_ID=selected-question
uv run python scripts/s6_remap.py --question "$QUESTION_ID" --propose-only
```

The first command writes the proposal and stops. After Harry records approval/rejection in the
generated remap config, an approved candidate runs with:

```bash
QUESTION_ID=selected-question
REMAP_FIELD=approved-field
uv run python scripts/s6_remap.py --question "$QUESTION_ID" --execute-approved --field "$REMAP_FIELD"
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" --with-remap "$REMAP_FIELD"
uv run python scripts/s6_remap.py --question "$QUESTION_ID" --finalize "$REMAP_FIELD"
```

1. Select 5–10 residual pairs by the fixed contradiction ranking.
2. Call the proposal model with only logged inputs and strict structured output.
3. Harry approves/rejects the exact moderator name, type, categories/bins, extraction prompt, and
   permutation rule before any remap.
4. If approved, remap the full eligible v1 primary cohort using echo-back IDs.
5. Require no unmatched/duplicate responses and valid unique responses for ≥95% of expected IDs;
   nulls count toward join success, not scientific coverage.
6. Validate/quarantine, left-join a side table in s5, and compare the frozen headline model against
   that same model plus the candidate on identical non-missing rows, folds, paper weights, and
   preprocessing.
7. Record `kept_exploratory`, `discarded`, or `indeterminate` verbatim. Keep requires ≥60%
   non-null primary-paper coverage, two levels with ≥5 papers, `k>=3`, incremental ΔLL ≥0.02,
   positive gain in `ceil(0.6*k)` folds, candidate-only add-one permutation `p<0.10` from 100
   successes within 125 attempts, positive gain in ≥60% of all paper bootstraps, and an all-valid
   sensitivity with the same gain sign. Paper-constant candidates shuffle across non-missing
   papers; within-paper candidates use only the approved paper-summary exchangeability rule.
   Invalid joins, incomplete runs, or undefined permutation schemes are indeterminate; any failed
   numeric rule is discarded. A kept post-hoc moderator remains exploratory and cannot replace the
   pre-specified M4 headline.

If approval is unavailable, write `not_run: human_approval_unavailable`; do not block the demo.

Acceptance tests fail each remap rule one at a time, prove before/after rows and folds are identical,
exercise paper-constant and approved within-paper shuffles, and cover pass, numeric discard,
invalid-join indeterminate, undefined-exchangeability indeterminate, and 99-of-125 indeterminate.

## 15. Phase 7 — real release, rehearsal, and additive scale (M4.75–M7)

### First shippable freeze

```bash
QUESTION_ID=selected-question
uv run python scripts/generate_baseline.py --question "$QUESTION_ID"
uv run python scripts/s7_export_demo.py --question "$QUESTION_ID" --candidate v1 --force
uv run python scripts/verify_demo.py --question "$QUESTION_ID" --offline
uv run streamlit run app/streamlit_app.py -- --question "$QUESTION_ID"
```

1. Run the commands above on real v1.
2. Start the app with network disabled/unavailable and rehearse selected script end-to-end.
3. Verify every displayed number against manifest/headline source fields.
4. Capture screenshots for top three evidence claims and record a backup demo video.
5. Preserve this release unchanged before overnight work.

### Overnight

Scale only with healthy G1b resume/heartbeat and observed quota behavior. Target ≥50 eligible papers,
then 50–150; 150–300 is stretch. Checkpoint terminal PaperRecords and raw results. Freeze the last
clean checkpoint by ~7 AM. Full grounded verification is concurrent and never blocks the v1 demo.

```bash
QUESTION_ID=selected-question
SCALE_TARGET=150
uv run python scripts/s3_extract.py --question "$QUESTION_ID" --target-papers "$SCALE_TARGET" --resume
uv run python scripts/verify_quotes.py --question "$QUESTION_ID" --scope full --resume
```

### Sunday promotion rule

Stage the overnight result separately. If it is operationally complete, reconciled, lineage-valid,
passes the trust/grounding gates, and its offline release validates, it **supersedes v1 even when
the headline weakens or Variant A becomes Variant B**. Recompute M4 and ship the mechanically
selected scaled narrative; never retain a favorable pilot merely because scale changed the story.
Retain v1 only when the scaled run is technically incomplete, corrupt, unreconciled, or unvalidated
by freeze, and record that reason plus the frozen v1 corpus size in the manifest and script. At
9:45 AM, stop all changes and produce the submission artifact; submit at 10:15.

Promotion reruns s4, every anchor, whole-candidate quarantine, and a new fixed-seed stratified
20-row audit with at least `min(10, newly added audit-eligible papers)` rows drawn from newly added
distinct audit-eligible papers. Audit-eligible means at least one accepted primary-family canonical
3-class exact-grounded, unflagged row before verification. Record both the newly added eligible-
paper and audit-eligible-paper denominators; select one row per distinct new audit-eligible paper
before any repeat. It completes verification on every
grounded scaled row; each disagreement is adjudicated or
explicitly excluded. Apply the original G3 trust thresholds, then recompute story viability and s5
M4 or the descriptive Variant-B path. Generate a one-shot baseline for the scaled cohort hash, run
s7, and run `verify_demo.py --offline`. The release-selection record stores separate selected-v1 or
scaled upstream hashes and rejected-scaled-attempt hashes through the last completed pre-s7 stage;
it never hashes itself or s7. No v1 analysis or baseline artifact is mixed into the
scaled candidate; the frozen v1 release remains only as a recoverable historical bundle. If scaled
s5 is interrupted before freeze, use Phase 5's explicit checkpoint finalizer: a successfully
finalized `m4_incomplete` candidate is scientifically complete and follows normal promotion, while
an unfinalized partial remains the technical `scaled_incomplete` fallback case.

```bash
QUESTION_ID=selected-question
uv run python scripts/s4_normalize.py --question "$QUESTION_ID" --force
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --scope scaled --sample-size 20 --new-paper-target 10 --seed 20260815
uv run python scripts/verify_quotes.py --question "$QUESTION_ID" --scope full --resume
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --scope scaled --finalize-g3
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" --force
uv run python scripts/generate_baseline.py --question "$QUESTION_ID"
uv run python scripts/s7_export_demo.py --question "$QUESTION_ID" --candidate scaled --fallback frozen-v1 --force
uv run python scripts/verify_demo.py --question "$QUESTION_ID" --offline
```

## 16. Test matrix

| Layer | Default? | Covers |
|---|---:|---|
| unit | yes | config, IDs, aliases, types, entropy, weights, gate functions |
| contract | yes | generated schemas, PaperRecord terminals, manifests, artifact allowlists |
| fixture integration | yes | s4→s7 and app for A and B, no network |
| archived raw parser | yes | exact known probe envelope/rows |
| live smoke | no | auth, citation resolution, failure codes, resume, rate-limit observation |
| real-data reconciliation | manual gate + tests | counts, anchors, audit, grounding, verification |

Full offline quality command:

```bash
uv run ruff check .
uv run pytest -q -m "not live"
```

No milestone is complete with skipped/failing offline tests. Live failures are recorded as gate
results, never hidden with broad retries.

## 17. Definition of Done

### Contracts and reproducibility

- One authoritative locked config; exact generated schema committed/archived.
- Every retrieved document reconciles to a canonical PaperRecord or logged alternate; every
  canonical paper reaches a terminal extraction status.
- No orphan findings, mixed extraction tuples, stale analysis, or unrecorded dirty demo lineage.
- No partial s5 `run.json` is exportable; an interrupted M4 can ship only through explicit
  checkpoint finalization with a complete stage record and incomplete scientific gate.
- Re-running s4–s7 from the same inputs/seed reproduces hashes except declared non-scientific
  timestamps; frozen-incomplete finalization reproduces run and scientific hashes from its
  content-addressed checkpoint, and release replay reproduces selection/manifest/bundle hashes from
  the complete immutable export input set. Manifest snapshot time is itself deterministic.
- Release selection satisfies one of the six literal state rows, and every quality scalar
  recomputes from its contractually named full-ledger, audit, or verification source.

### Scientific integrity

- Atomic findings preserve outcome/timepoint/comparison distinctions.
- Primary headline cohort is grounded, unflagged, primary-family, canonical 3-class evidence.
- Paper balancing is used in fit, score, majority, tree and bootstrap.
- All pre-specified moderators, losers and insufficient rows appear.
- G3 trust/story and M4 `complete|not_run|incomplete` status are machine-readable; A/B is never
  chosen by taste, and incomplete rules are never mislabeled failed.
- Residual contradictions and evidence gaps are visible in either variant.
- Claims say “our retrieved corpus,” predictive rather than causal, and post-hoc when applicable.

### Reliability and demo

- Exact demo inventory and all hashes validate; app reads nothing outside `demo/`.
- App loads offline and renders four core views plus evidence.
- Selected ≤90-second script contains no unresolved placeholders and matches `headline.json`.
- Last known-good real-data release, screenshots, and backup video exist before overnight scaling.
- No secret is tracked, logged, copied into artifacts, or rendered.

### Minimum shippable outcome

- ≥20 grounded primary findings from multiple papers;
- visible directional disagreement or an honest agreement/no-pattern conclusion;
- a complete moderator eligibility table and, whenever grouped CV is feasible, at least one valid
  paper-balanced test with a paper-aware control, even if it fails;
- source-grounded evidence cards plus a residual artifact with an honest empty state when no
  opposite-direction pair exists;
- a complete Variant A or Variant B offline demo.

Anything beyond this Definition of Done is stretch and may not delay the freeze.
