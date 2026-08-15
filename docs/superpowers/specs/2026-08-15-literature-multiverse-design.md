# Literature Multiverse — Design Document

**Date:** 2026-08-15 · **Status:** v2.3 — contract-complete, paired with an executable
implementation plan; ready for the coding go/no-go · **Deadline:** submission Sun Aug 16 (freeze
9:45 AM, submit 10:15 — a deliberate 30-min buffer vs the official 10:45 AM PT deadline)

**v2.3 updates:** one normative-source hierarchy; exact `PaperRecord`, `FindingRow`,
`QuestionConfig`, direction, grounding, and lineage contracts; paper-ledger flow through s1–s7;
paper-balanced and fully specified CV; distinct G2/G3 purposes; a statistical + material M4 gate;
exact `headline.json` and agent keep/discard rules; a buildable Variant B; and a file-by-file
implementation plan with acceptance tests. v2.2's verified state remains unchanged: both API keys
work, **G1 passed**, and the empirical smoke artifacts are archived under `data/raw/smoke/`.

**Normative authority:** this document defines global product, statistical, stage, and artifact
invariants. The locked `QuestionConfig` defines only the topic-specific values permitted here
(eligibility, queries, outcomes, direction wording, moderators, categories, and bins). The paired
implementation plan is authoritative for dependency order, file ownership, commands, and
acceptance tests. Executable models and generated schemas must implement those authorities exactly;
they are not an independent place to change the contract. For global invariants this design wins;
for permitted topic-specific values the locked config wins over examples in this design.
`LITERATURE_MULTIVERSE_PROJECT_CONTEXT.md` is conceptual background; its example schemas, file
tree, and pseudo-code are non-normative when they differ from v2.3.
**Research:** `docs/planning/research-notes.md` · **Review trail:** `docs/planning/reviews/`

One sentence: *Turn a retrieved body of literature into a structured experimental multiverse,
measure where it disagrees, and test whether pre-specified context variables predict that
disagreement — with every row carrying provenance and grounding status, and every demo claim
traceable to source lines.*

v2 incorporates 8 blocker and 12 major findings from a 7-lens adversarial review
(`docs/planning/reviews/2026-08-15-round1-synthesis.md`). Biggest structural changes vs v1:
fixture-first build order (dashboard skeleton Saturday), statistics mechanics corrected
(subset-paired baselines, StratifiedGroupKFold, two-tier permutation with family-wise correction),
data-identity contracts pinned (finding_id, echo-back remap joins, version-mixing guard), every
gate given a failure branch, Sunday decompressed, heatmap + LLM screening pass cut.

---

## 1. Scope

### In scope
1. Question-config-driven pipeline: search → screen (deterministic) → extract → normalize/validate
   → analyze → targeted remap → export.
2. `PaperRecord` ledger: every identity-deduped paper survives even when excluded, failed,
   quarantined, or mapped to zero findings. Every raw retrieved document remains traceable through
   s1 candidates, `alternate_doc_ids`, and the dedupe log; query, coverage, and extraction
   provenance stay auditable without treating duplicate deposits as independent papers.
3. `FindingRow` atomic unit: one paper × comparison × outcome × timepoint. Never one row per paper.
4. Disagreement quantification: normalized entropy over 3-class direction, **finding-level and
   paper-level co-primary**, with grounding/exclusion rates always displayed.
5. Moderator ranking: coverage, paper-balanced grouped-CV log-loss gain (fold-paired,
   subset-internal baseline),
   two-tier grouped permutation + Westfall–Young family-wise p, paper-level bootstrap stability.
   MI as descriptive companion only.
6. Shallow decision tree (depth ≤ 3, paper-weighted, post-fit leaf audit) as the multiverse map.
7. Residual contradiction pairs (similar context, opposite direction), plus an always-produced
   evidence-gap table for the no-pattern narrative.
8. One agentic iteration, run while awake: residual conflicts → moderator proposal (Claude API) →
   human approval → targeted re-map → incremental gain test →
   `kept_exploratory|discarded|indeterminate`, logged to `trace.json`. Discarded and indeterminate
   outcomes are successful, auditable agent-loop results, not pipeline failures.
9. Streamlit dashboard reading cached artifacts only; **skeleton built Saturday against fixtures**.
10. Credibility layer: human audit (raw counts + Wilson interval), overnight cross-model
   verification of quote⇒direction, evidence-section flags, provenance links, publication-bias
   caveat card, baselines panel (majority vote + archived one-shot LLM consensus).

**MVP boundary:** the hackathon build is a frozen, offline viewer for one empirically selected and
preconfigured question. “Give us a question” describes the future product, not a live arbitrary-
question feature in this weekend's demo. All quantitative prose says “our retrieved corpus,” never
“the literature” without that qualifier.

### Out of scope / cut
Next-experiment planner, temporal backtesting, mechanism layer, cross-domain, React UI, universal
quality score, network MA. **Cut in v2:** context heatmap (tree carries the story), s2 LLM
`filter` pass (deterministic screening + map's own eligibility call suffice; `filter` stays in the
back pocket for extreme review pollution), topic E from triage (pre-scan rank 3; outcome sprawl).
Numeric random-effects MA remains a stretch only if effect sizes are commensurate. Bias funnel
plot: 15-min stretch only (not to be confused with the corpus 4-bar, which is core).

---

## 2. Architecture

Staged pipeline, cached artifacts, each stage an idempotent CLI (`scripts/s*_*.py --question <qid>`).

```
configs/questions/<qid>.yaml
  s0 smoke_test   → data/raw/smoke/               contract-pinning probe suite (§5.1)
  triage_probe    → data/raw/triage/<candidate>/  isolated ≤10-paper extraction using a
                                                  status:triage candidate config; generates a
                                                  candidate-specific strict schema but cannot
                                                  write or feed s3–s7 production artifacts
  s1 search       → data/raw/search/<qid>/        per-query JSON + candidate_papers.jsonl; one row
                                                  per doc_id × query family, before identity dedupe
                                                  (s1 only removes repeated doc_id/query hits)
  s2 screen       → data/raw/screen/<qid>/        screened_papers.jsonl: canonical PaperRecords
                                                  after DETERMINISTIC article-type/refine,
                                                  DOI/PMID + fuzzy identity dedupe; preserves
                                                  alternate doc_ids and every exclusion reason;
                                                  also emits dedupe_log.jsonl plus explicit
                                                  include_paper_ids.json/exclude_paper_ids.json
  s3 extract      → data/raw/map/<qid>/           raw map --json stdout archived verbatim; work
                                                  queue = s2 include list (excluded ids are never
                                                  mapped — funnel counts stay honest)
                  → data/extracted/<qid>/papers.jsonl     one terminal ledger row per s2 paper:
                                                  excluded / map_success / map_failed / ineligible;
                                                  zero-finding papers therefore never disappear
                  → data/extracted/<qid>/findings.jsonl   inject authoritative paper/doc/map IDs →
                                                  pre-pydantic normalizer → strict pydantic →
                                                  quarantine.jsonl with reason codes; normalizer
                                                  stamps the extraction contract tuple on each row
  s4 normalize    → data/processed/<qid>/papers.parquet   regenerated from the s3 paper ledger
                  → data/processed/<qid>/findings.parquet  regenerated from jsonl, never edited;
                                                  explicit Int64 nullable casts; manual fixes from
                                                  configs/questions/<qid>.patches.yaml
                                                  (human-authored, each with a reason);
                                                  validates finding→paper referential integrity;
                                                  FAILS LOUDLY on mixed contract tuples
  s5 analyze      → artifacts/<qid>/analysis/     entropy, moderator table, tree, pairs,
                                                  evidence_gaps, bootstrap/permutation,
                                                  headline.json and stage run.json;
                                                  --with-remap left-joins remap side tables
  s6 remap        → data/raw/map/<qid>/remap_*    raw echo-back-keyed output, then through the SAME
                  → data/processed/<qid>/remap_<field>.parquet   normalizer/quarantine path into a
                                                  side table (finding_id, value) with its own
                                                  version sidecar — NEVER merged into
                                                  findings.jsonl (base rows keep their original
                                                  tuple, so the s4 guard never trips);
                                                  trace.json → artifacts/<qid>/analysis/
  s7 export_demo  → artifacts/<qid>/demo/         PACKAGE-ONLY freeze; verifies complete lineage
                                                  and exact inventory in §4.5; generates only
                                                  packaging metadata and rendered script
app/streamlit_app.py — reads artifacts/<qid>/demo/ only; asserts manifest schema_version at
startup; refuses mixed/stale lineage; shows created_at and corpus qualifier in footer.
```

**Build order is fixture-first (B1):** the scientific data boundary is always the pair
`papers.jsonl` + `findings.jsonl`, never findings alone. The four-scenario suite in the implementation
plan shares a synthetic corpus shape (~25 PaperRecords/~60 findings: planted true/null moderators,
mixed/unclear and grounded/ungrounded rows, combo-dose strings, one 15-finding paper, a preprint/
published twin, zero-finding/failed papers, quarantine rows, and one dummy contract tuple). It also
supplies deterministic audit, verifier, anchor, section, and fixture-baseline inputs so every
G3/M4/A/B/export branch runs offline; derived gates/headlines/manifests are never forged. The suite
must prove that duplicating a paper's findings does not materially change the paper-balanced rank.
s3's raw-envelope parser is an explicit post-G1 stub. §41-10 forbids *polishing* UI before a result,
not plumbing it: skeleton Saturday, data-swap + package Sunday.

**Lineage guard (B4, still no cache machinery):** every stage has a fixed output path per
`(qid, stage)`, requires `--force` to replace it, and writes a strict `run.json` with:

```text
run_record_version: "1"
run_id, question_id, stage, stage_version, status: complete|partial|failed
completion_mode: normal|frozen_incomplete, checkpoint_sha256: str|null
started_at, completed_at|null, code_version, command_argv[]  # secrets redacted
config_path, config_sha256
prompt_path|null, prompt_sha256|null, schema_path|null, schema_sha256|null, cfghash|null
upstream[]: {stage, run_id, run_record_path, run_record_sha256}
inputs[]/outputs[]: {path, sha256, bytes, rows|null}
external_result_ids: {provider: [ids...]}
counts: {stage-specific reconciled integer counts}
warnings: [stable warning codes]
```

`completion_mode=frozen_incomplete` is legal only for a complete s5 stage finalized from an
immutable interruption checkpoint as specified in §6.5; `checkpoint_sha256` is required there and
is null everywhere else. In that case `run.json.status=complete` means that s5 produced and
validated its entire terminal artifact contract—it does not claim that M4 inference completed.

Timestamps are timezone-aware ISO-8601 and paths are repository-relative. `config_sha256` hashes
the *entire canonical validated locked config*, not only `target_relation`; `code_version` is the git
commit or `dirty:<source-tree-hash>`. Extraction defines
`cfghash = sha256(canonical config + rendered prompt + output schema)` and stamps
`(prompt_version, schema_version, cfghash)` on every finding. A stage becomes `complete` only after
output hashes and its count identities validate. s4 hard-fails on mixed extraction tuples
(`--allow-mixed` exists for diagnosis only). s5 records the exact papers/findings/remap inputs. s7
recomputes every upstream hash and **refuses to export `--allow-mixed`, stale, partial/failed, or
dirty-lineage combinations** unless a diagnostic dirty override is explicitly recorded. Humans
decide re-runs; there is no cache-hit logic.

Stack: `uv`, Python 3.12 pin, pydantic v2, pandas, scikit-learn, plotly, streamlit, tenacity,
rapidfuzz, pyyaml. Tests (pytest, no network) for models, normalizer, disagreement, moderators,
tree audit, dedupe.

---

## 3. Key decisions

**D1 — Extraction engine: Paperclip CLI `map --output-schema` via subprocess — ✅ VERIFIED
(G1 PASSED, probe `m_2bc51e4b`).** Argv-list invocation, **no shell**, raw stdout archived before
parsing. The named workers (`structured-extraction`, `eligibility-screen`,
`exhaustive-extraction`) are **gated to GXL testers** — ask organizers for enablement; the
default quick-reader worker with `--output-schema` is proven sufficient. *Fallback (cold
standby):* pull `content.lines` + direct Claude API structured extraction into the same pydantic
models — key verified, only built if map degrades at scale. Both keys live in `.env`
(`PAPERCLIP_API_KEY` verified against the server; `ANTHROPIC_API_KEY` verified with a live call).

**D2 — Extraction schema returns an ARRAY of findings per paper — ✅ VERIFIED.** The probe RCT
yielded 5 findings (one per outcome) with dose_raw, population, sample_size, timepoint_raw,
verbatim `evidence_quote`, and `evidence_lines` (format: `["L18"]`); ineligible papers return
`{eligible:false, exclusion_reason, findings:[]}` (no row drop). Fallbacks retired.

**D3 — One map per corpus, not offset windows** (per CLI guidance + measured behavior): a single
`map -j <concurrency>` call over the search result set, with **built-in `--resume MAP_ID` /
`--retry-failed` / `--cancel`** replacing most of the custom checkpointed runner — ours shrinks
to a thin wrapper that archives raw output, invokes `--resume` on restart, and heartbeats.
Measured latency: **4 papers in 8.0 s wall** (1.3–4.6 s/paper, parallel) — a 300-paper corpus is
minutes, not hours, subject to unknown quota ceilings. All raw JSON archived immediately; local
files are the source of truth (result-ID TTL still unknown).

**D4 — Screening is deterministic** (A20): `--article-type research-article` where available,
`refine` metadata filters, DOI/PMID dedupe, **fuzzy dedupe pass** (normalized title + first-author
surname + year±1 via rapidfuzz; prefer published over preprint; log to `dedupe_log.jsonl` — itself
demo material). Eligibility judgment happens once, inside extraction (`eligible=false` +
`exclusion_reason`). Trial-ID regex (NCT\d+ etc.) → `dataset_or_cohort_id`. The Yfanti 2010/2011
twin pair is seeded into dedupe tests now. G3 same-cohort hand-pass criteria (A17): author
overlap + year±2 + identical n/intervention → shared `dataset_or_cohort_id`.

**D5 — One canonical direction vocabulary.** Internal, exported, and dashboard values use exactly
`increase | no_effect | decrease | mixed | unclear`; `effect_direction` is never JSON null. These
mean measured outcome higher than comparator, no detectable difference, lower than comparator,
incompatible directions inside an otherwise atomic row, and unsafe to determine. They never mean
beneficial/harmful. The locked config pins the measurement meaning for every included endpoint.
The pre-normalizer accepts a closed legacy/fallback alias map (`positive→increase`,
`negative→decrease`, string `"null"`/`neutral→no_effect`, `indeterminate→unclear`) after case/spacing
normalization; value-laden terms such as `improved` or `harmful` are never auto-mapped. A JSON null
direction violates the required extraction contract and quarantines the finding; an unknown source
must explicitly emit `unclear`. Primary fitting is 3-class; `mixed`/`unclear` are retained and
excluded, and their **exclusion rate is displayed next to every entropy figure**. A separately
labeled 4-class sensitivity maps row-level `mixed`, `unclear`, and paper-modal ties to one
`unresolved` class, normalizes by `log(4)`, and is never numerically compared with the primary
3-class entropy. Also emit unresolved-rate × content-tier. Finding-level and paper-level entropy
are co-primary; paper modal ties become unresolved and are excluded from the 3-class estimate with
their rate reported.

**D6 — Moderator statistics: pre-registered, see §6.** No deviations without writing them down
before looking at results.

**D7 — Moderator proposal agent (Pass 3) = Claude API call** (`claude-opus-5`, adaptive thinking,
`output_config.effort: "high"` — per D11), input: config, schema, missingness,
outcome distribution, 5–10 residual pairs with quotes; output: JSON {moderator, reason,
extraction_prompt, search_terms} via structured outputs (`output_config.format`). **Human approves before the remap fires; the whole chain runs
at M4.5 while awake, only after Variant A passes** (B5). Everything is logged to `trace.json` as it runs (input residual
 pairs shown to the agent, proposal verbatim, approval, result IDs, before/after gain,
decision status, per-rule results, timestamps). Residual pairs are used only to *propose* the field;
the approved field is re-mapped across every eligible v1 paper in the locked primary outcome
cohort, never only the contradictory examples. The exact keep/discard rule is in §6.4. Language
rule (A11):
say **"cross-validated incremental gain (moderator proposed post hoc)"**, never "held-out" — and say the
caveat before a judge does.

**D8 — Provenance and grounding:** the model never supplies identity. s2 creates a stable local
`paper_id`; s3 injects it plus the authoritative `doc_id` from the map envelope. `PaperRecord`
preserves DOI/PMID, title, alternate doc IDs, query/search/map provenance, content tier, publication
status, eligibility, extraction terminal status, and cohort ID. Every finding carries the local and
authoritative join keys plus evidence quote/lines. Cached quote + resolved section + line numbers is
the primary evidence rendering; the live URL is a caption. `grounding_status` is computed
mechanically (§4.3), never self-reported by the model. Screenshots of top-3 claims are pre-captured
as offline backup. Paperclip-repo claim verification: stretch, ≤30 min, only if smooth.

**D9 — Dashboard: Streamlit skeleton built Saturday against fixtures** (B1); Sunday is data-swap +
copy. Degradation = cut sections in reverse narrative order, **never switch stacks** (B8).
Must-have four: global stacked bar, moderator table, tree, evidence cards. Cheap adds: agent trace
as preformatted text, contradictions as plain table. Tree renderer is a named Saturday
deliverable: human-readable split labels (moderator→display-name map), leaves as stacked direction
bars captioned "n=X findings / Y papers", depth ≤ 3, one screen. Never ship `plot_tree`.

**D10 — Topic locked by triage (§5.2)** from: **C (antioxidants × exercise) and A (TRE/IF) with
10-paper probes; B and C+B merge count-only**. E dropped. Tie-break toward C fires **only if** a
corpus grep estimates **≥5 distinct papers on each side of the ~500 mg/day threshold**; C's
fallback headline moderators pre-committed: training status, vitamin identity, population state.
C+B merge is the first rung of the G2 escalation ladder, not a default, and fires **only if C's
confirmed counts come in < ~60 usable primaries AND A disappoints**.
At G2, topic lock also freezes the primary outcome family, included endpoint mapping, per-endpoint
direction semantics, eligibility, candidate-moderator family (including constant vs within-paper
kind), bins, and the recovery-check statement. Other outcome families remain labeled
descriptive sensitivities and cannot silently enter the headline cohort.

**D11 — Claude model tiering (Harry's directive, ~$50 API budget):**
- **Wiring / connectivity pings / test fixtures:** `claude-haiku-4-5` ($1/$5 per MTok) — the
  "test dummy."
- **Volume LLM work** — extraction fallback if ever needed, and the **cross-model
  quote⇒direction verification** (A19): `claude-sonnet-5` ($2/$10 intro pricing through
  2026-08-31), run through the **Batches API at 50% off** for the overnight full-corpus pass
  (~1,000 rows × ~2K tokens ≈ low single-digit dollars). Sonnet also satisfies the
  "different model than the extractor" independence requirement (Paperclip's map worker is the
  extractor).
- **Genuine reasoning** — moderator proposal (D7), and any judgment-heavy one-off:
  `claude-opus-5` ($5/$25), adaptive thinking (default-on), `effort: "high"` (`xhigh` if a
  proposal round underwhelms). Low call volume keeps this under ~$5.
- Rough budget envelope: verification batch ~$3–6, proposals ~$2–5, wiring ~$1, reserve the
  rest for the extraction fallback (sonnet-5, ~$10–30 if the whole corpus ever needs it).
  Swap to event credits when issued; personal key until then. If the
merge ever wins, the intervention-class split is presented first as the trivial baseline, then
moderators within class.

---

## 4. Data contracts

### 4.1 Identity and `PaperRecord` (pydantic v2)

`paper_id` is created once, after s2 identity dedupe, and is the only local grouping/join key:
`doi:<normalized-doi>` if available, else `pmid:<digits>`, else `doc:<preferred-doc-id>`. For a
fuzzy-matched publication/preprint cluster, s2 chooses the peer-reviewed/published record as the
preferred record and preserves all alternates. IDs are never model-generated.

Every s3 input paper produces exactly one terminal `PaperRecord`, including zero-finding and failed
maps. Required fields and types:

- Identity: `paper_id: str`, `doc_id: str`, `alternate_doc_ids: list[str]`, `doi: str|null`,
  `pmid: str|null`, `title: str`, `first_author: str|null`, `pub_year: int|null`.
- Retrieval: `source: str`, `article_type: str|null`, `query_families: list[str]`,
  `search_result_ids: list[str]`, `content_tier: full_text|abstract_only|unknown`,
  `publication_status: peer_reviewed|preprint|unknown`.
- Screening/dedupe: `screen_status: included|excluded`, `screen_reason: str|null`,
  `dedupe_cluster_id: str|null`, `dedupe_preferred: bool`.
- Extraction: `map_status: not_mapped|success|failed`, `eligible: bool|null`,
  `exclusion_reason: str|null`, `map_result_id: str|null`, `raw_artifact_path: str|null`,
  `raw_finding_count: int`, `accepted_finding_count: int`,
  `quarantined_finding_count: int`, `failure_code: str|null`.
- Dependence/lineage: `dataset_or_cohort_id: str|null`, `prompt_version: str|null`,
  `schema_version: str`, `config_sha256: str`, `cfghash: str|null`, `created_at: datetime`.

The model is strict (`extra="forbid"`). `paper_id` is unique, alternate IDs are sorted/unique, and
all counts are non-negative. Excluded papers are `not_mapped`; included papers end in `success` or
`failed`. Success requires `map_result_id`, `raw_artifact_path`, `prompt_version`, and `cfghash`;
failure requires a stable `failure_code` and zero finding counts. After a successful map,
`accepted_finding_count + quarantined_finding_count = raw_finding_count`; model-ineligible papers
have zero findings, while model-eligible zero-finding papers are valid and remain visible.

The paper ledger, not `FindingRow`, produces funnel counts, exclusion reasons, coverage status,
publication-status sensitivities, query-provenance views, and cohort-collapsed analyses.

### 4.2 `FindingRow` (pydantic v2)

Pipeline-filled fields: `finding_id: str`, `paper_id: str`, `doc_id: str`, `map_result_id: str`,
`array_position: int`, `prompt_version: str`, `schema_version: str`, `cfghash: str`,
`grounding_status: exact|missing|mismatch|unverifiable`, `evidence_section: str|null`,
`section_flagged: bool`,
`normalization_warnings: list[str]`.

Fixed extraction fields (all nullable unless explicitly stated): `study_type: str`,
`species: str`, `model: str`, `population_state: str`, `intervention: str`,
`intervention_class: str`, `comparator: str`, `dose_raw: str`, `duration_raw: str`,
`timing_context: str`, **`outcome_name: non-empty str`**, `outcome_family: str`,
`timepoint_raw: str`, **`effect_direction: CanonicalDirection`**, `effect_size_raw: str`,
`p_value: float[0,1]`, `significant: bool`, `sample_size: int>=1`, `evidence_quote: str`,
`evidence_lines: list[str]`, `confidence: float[0,1]`.
`effect_direction` may not be JSON null; `mixed` or `unclear` represents non-classifiability.

Topic-specific normalized values live in one `moderators` object whose exact properties are
generated from the locked config's `ModeratorSpec`s with `additionalProperties:false`; s4 flattens
them to `mod__<name>` parquet columns. For topic C, these include
`vitc_dose_mg_per_day: float|null` and `vite_dose_iu_per_day: float|null`; `dose_raw` is always
preserved and animal mg/kg values are a different moderator/stratum. This resolves build ordering:
M0 implements the fixed model + schema generator, while G2 supplies the topic-specific properties.

`finding_id = {paper_id}:{map_result_id}:{array_position}:{hash8(outcome_name|timepoint_raw|
dose_raw|effect_direction)}`. It is stable only within one raw map artifact. Audit records use
`(map_result_id, array_position)`; joins across extraction runs are forbidden.

### 4.3 Map, validation, and grounding contracts

The model output schema is top-level `{eligible, exclusion_reason, findings:[…]}` — **no
model-filled paper ID** — using JSON Schema 2020-12 and `additionalProperties:false` at every
object. `eligible` is required boolean; `exclusion_reason` is required nullable string; `findings`
is a required array. An ineligible paper must have zero findings. For eligible findings,
`outcome_name`, canonical non-null `effect_direction`, and every nullable field key are required;
omission is represented by JSON null.

Boundary: raw payload → inject envelope/local identity → config-driven pre-normalizer (casefold,
aliases, string-`"null"` vs JSON-null rule) → strict Pydantic → finding quarantine with stable reason
codes. Quarantine never deletes the parent PaperRecord. A red dashboard banner appears above 10%.

Grounding is mechanical. Normalize Unicode and whitespace, concatenate the cited `content.lines`
in line order, then test whether the normalized quote is a substring. Status is `exact` on match,
`missing` if quote or lines are absent, `mismatch` on a failed substring check, and `unverifiable`
when source lines cannot be accessed. Resolve `evidence_section` from the line-to-section map;
Abstract/Discussion/Conclusion/References/unknown set `section_flagged=true`. The primary headline
cohort requires `grounding_status=exact` and `section_flagged=false`; all other valid rows remain in
the dataset and appear only in explicitly labeled all-valid-row sensitivities. Cross-model
verification is computed over grounded rows, with ungrounded counts reported separately.

Remap uses echo-back keying. The prompt embeds each base `finding_id` plus quote/outcome/dose; the
schema requires `{finding_id: <enum of supplied IDs>, value}`. It runs over the entire eligible v1
primary cohort, writes a versioned `(finding_id, value)` side table, and sends missing, duplicate,
or unmatched IDs to quarantine. It never mutates base JSONL.

### 4.4 Locked `QuestionConfig` (YAML)

The implementation validates two states. `status: triage` must contain the candidate question,
target relation/direction meanings, eligibility/search rules, candidate outcome map, candidate
moderators, and analysis seed/labels, but may omit anchors, recovery, Variant-B axes, and final demo
copy. It is accepted only by s1, s2, and the isolated `triage_probe` command, which maps at most ten
logged papers and writes outside production s3–s7 paths. `status: locked` is required by s3–s7 and
contains all of the following:

```yaml
schema_version: "1"
status: locked
question_id: <slug>
research_question: <text>
target_relation:
  exposure: <text>
  comparator: <text>
  outcome: <text>
  increase_definition: <text>
  decrease_definition: <text>
  no_effect_definition: <text>
  desirability_claims_allowed: false
eligibility: {include: [...], exclude: [...], article_types: [...]}
search:
  sources: [...]
  query_families: [{id: <slug>, queries: [...]}]
  use_all: true
outcomes:
  primary_family: <slug>
  family_map: {<raw-or-pattern>: <family>}
  included_primary_endpoints: [...]
  endpoint_direction_overrides: {}
moderators:
  - {name: <slug>, type: categorical|float|int|bool, source: fixed|topic,
     role: tested|descriptive,
     kind: paper_constant|within_paper, permutation: paper|paper_summary|none,
     paper_summary: <rule|null>, display_name: <text>, allowed_values: null, bins: null}
analysis:
  seed: 20260815
  labels: [increase, no_effect, decrease]
  max_missingness_headline: 0.40
  min_papers_per_level: 5
  cv_max_folds: 5
  permutation_count: 100
  bootstrap_count: 200
variant_b:
  axes: [<moderator-slug>, ...]
  # Levels come from each categorical allowed_values or numeric bins above.
  primary_endpoints: [<locked-primary-endpoint>, ...]
triage: {candidate_topic: <slug>, score_record: <path>}
anchor_papers: [...]
recovery_check: {kind: published|narrative|none, statement: <text>, source: <text|null>}
demo: {hook: <text>, spoken_question: <20-words-or-fewer>, moderator_display_names: {},
       corpus_qualifier: "our retrieved corpus", fixture_mode: false}
```

G2 writes `docs/planning/triage.md`, then changes the winning config to `locked`. Bins, outcome
mapping, candidate family, and direction semantics are frozen before s3 scale extraction or s5
real-data analysis. A locked config has at most six `tested` and two `descriptive` moderators;
outcome family cannot itself be a tested moderator. `paper_constant` requires paper permutation;
`within_paper` requires a frozen paper summary and `paper_summary` permutation to be inference-
eligible. Every `variant_b.axes` entry must name a configured moderator with a complete
`allowed_values` or `bins` list, and every listed endpoint must belong to the locked primary family;
missing is not a scientific level. A post-hoc remap moderator lives only in its approved remap
sidecar/config, never this pre-specified family. Any later base-config change creates a new config
hash and must be logged as a deviation.

`demo.fixture_mode=true` is reserved for the four named implementation-plan fixture qids. It
requires an explicit CLI `--fixture`, marks the manifest, forbids live providers, and is rejected by
production export/promotion. No other qid may enable it.

### 4.5 Demo artifacts (A18) — s7 is package-only; exact `demo/` inventory

s7 never recomputes a scientific result. It copies validated upstream artifacts and generates only
the manifest, release-selection record, and allowlisted rendered script while staging the release.

Every fixed analysis slot exists in every branch. `m4_checkpoint.json` has
`status:not_applicable|frozen_incomplete`. Every ordinary branch writes exactly
`{status:"not_applicable", reason:"m4_completed|m4_not_run"}`. A finalized interruption instead
writes exactly `{status:"frozen_incomplete", source_checkpoint_sha256:"...", checkpoint:{...}}`,
where `checkpoint` is the complete source-checkpoint object and
`source_checkpoint_sha256=sha256(canonical_json(checkpoint))`; source checkpoints themselves use
that same canonical JSON serialization. The wrapper's own file hash is a separate ordinary output
hash. `tree.json` has
`status:supporting|exploratory|not_run|incomplete`, a reason, and `nodes:[]` unless complete;
`permutation.json` has `status:complete|not_run|incomplete`, success/attempt counts, and reasons;
`bootstrap.json` records entropy and model-stability components separately so G3 entropy can be
complete while M4 stability is `not_run`/`incomplete`. `moderators.parquet` always contains every
configured moderator with support/eligibility/status even when inference did not run. Empty
contradiction/evidence tables retain their exact schemas. A skipped branch is a typed artifact, not
a missing file or fabricated zero.

- `papers.parquet` and `findings.parquet` — mutually referential processed copies; evidence cards
  join them inside `demo/` and never read working directories.
- `analysis/moderators.parquet`, `m4_checkpoint.json`, `tree.json`, `contradictions.parquet`,
  `evidence_gaps.parquet`, `bootstrap.json`, `permutation.json` — exact s5 copies.
- `m4_gate.json` — `status: complete|not_run|incomplete`, every §6.3 rule, threshold, observed value and
  pass/fail when run, selected moderator or null, selection reason, selected variant, seed, cohort
  hash, and input/output hashes. A G3-story failure uses `not_run`, null observations, and explicit
  reason codes; it never fabricates failed moderator tests.
- `headline.json` — the exact discriminated-union branch in §6.3/§7, including the fully rendered
  sentence and literal `narrative_variant: A` or `B`.
- `baseline.json` — majority-vote line + one-shot LLM consensus paragraph, generated once by the
  dedicated script for each exact cohort hash; the raw response, prompt/model/version, and input
  hash are archived and a second call for that cohort is forbidden. A scaled cohort gets its own
  one-shot baseline and never reuses v1's. The LLM paragraph is visibly labeled an ungrounded
  comparison output, is never used in a statistic or evidence card, and is never spoken as a
  finding.

  Minimum fields are `{cohort_hash, status:complete|unavailable,
  source:live_llm|fixture_stub, majority:{direction, agreement},
  llm:{provider, model, prompt_sha256, raw_response_sha256, paragraph}|null, attempted_at,
  failure_code|null}`. Production makes exactly one archived attempt per cohort. A provider failure
  yields `status=unavailable` and a visible “LLM baseline unavailable” state rather than a retry or
  blocked release. `fixture_stub` is legal only under fixture mode.
- `trace.json` — s6 agent-loop inputs, proposal, approval, map result, rule-by-rule decision and
  timestamps; if any gate selects Variant B, a valid `not_run` trace records the exact reason remap
  was cut.
- `audit.json` — fixed-seed sampled finding IDs, per-field human decisions, adjudications, raw
  counts, Wilson interval, denominators, and error taxonomy. A scaled audit additionally records
  newly added eligible-paper and newly added audit-eligible-paper denominators plus sampled new
  distinct-paper count.
- `verification.json` — verifier model/prompt/version plus one unique decision per requested
  `finding_id`, with `model_status: agree|disagree|unverifiable` and a separate
  `adjudication: none|accept|reject`. Raw model status is immutable. s5 accepts `model_status=agree`
  or `adjudication=accept` into the primary cohort; reject/unresolved rows stay visible but out.
  Duplicate, unknown, or missing requested IDs fail reconciliation rather than becoming agreement.
- `g3_gate.json` — trust and story as separate Booleans, every threshold/observed value/reason,
  cohort/config/audit/verification hashes, and the resulting action
  `block_release|run_m4|select_variant_b_story`.
- `release_selection.json` — non-recursive selected-release lineage plus a separate nullable scaled-
  attempt record, written during staged s7 validation and hashed by the manifest; exact below.
- `demo_script.md` — rendered Variant A or B template selected by `g3_gate` then, when run,
  `m4_gate`.
- `manifest.json` — schema/config/code versions, complete stage lineage and sha256s; paper-funnel
  counts (searched → identity-deduped → deterministic-included → extraction-eligible → findings);
  all scalar quality/exclusion/grounding rates; primary cohort definition; variant; created_at.

The manifest fields consumed by the UI and script renderer are fixed (additional artifact entries
are allowed only under `artifacts`):

```json
{
  "manifest_version": "1",
  "schema_version": "1",
  "fixture": false,
  "question_id": "...",
  "research_question": "...",
  "spoken_question": "...",
  "corpus_qualifier": "our retrieved corpus",
  "narrative_variant": "A|B",
  "created_at": "ISO-8601-with-timezone",
  "config_sha256": "...",
  "code_version": "...",
  "primary_cohort_definition": "primary_grounded_unflagged",
  "paper_funnel": {
    "searched_documents": 0,
    "identity_deduped_papers": 0,
    "deterministic_included_papers": 0,
    "extraction_eligible_papers": 0,
    "primary_grounded_papers": 0,
    "primary_grounded_findings": 0
  },
  "quality": {
    "audit_correct": 0, "audit_total": 0,
    "grounded_fraction": 0.0, "quarantine_fraction": 0.0,
    "cross_model_agreement": 0.0
  },
  "exclusions": {
    "mixed_or_unclear_fraction": 0.0,
    "section_flagged_fraction": 0.0,
    "verification_excluded_fraction": 0.0
  },
  "release_selection": {
    "record_sha256": "...",
    "disposition": "v1_frozen|scaled_promoted|v1_retained_scaled_incomplete|v1_retained_scaled_corrupt|v1_retained_scaled_unreconciled|v1_retained_scaled_unvalidated",
    "frozen_v1_primary_papers": 0,
    "selected_release": {
      "corpus_role": "v1|scaled",
      "release_id": "...",
      "primary_grounded_papers": 0,
      "stage_run_sha256s": {"s3": "...", "s4": "...", "s5": "..."},
      "evidence_sha256s": {"g3_gate": "...", "audit": "...",
                              "verification": "...", "headline": "...",
                              "baseline": "..."}
    },
    "scaled_attempt": null,
    "rendered_disclosure": "..."
  },
  "lineage": [{"stage": "...", "run_id": "...", "run_sha256": "..."}],
  "artifacts": [{"path": "...", "sha256": "...", "bytes": 0, "rows": null}]
}
```

`manifest.created_at` is the deterministic scientific-snapshot time, not the s7 invocation time:
it equals the selected s5 `run.json.completed_at`. For `completion_mode=frozen_incomplete`, that is
the checkpoint's fixed timestamp. Any operational package-at time belongs only in s7's external
run log and is not bundled. `release_id` is deterministically derived from question/corpus role,
cohort, config/code, selected stage, and evidence hashes—never a random ID or wall clock.

All counts are integers ≥0 and all fractions are in `[0,1]`. The exporter recomputes funnel counts
from the bundled `papers.parquet`; primary counts from the headline cohort joined back to the
bundled ledgers; grounded, quarantine, and exclusion numerators and denominators from the complete
bundled `papers.parquet` + `findings.parquet`; human-audit counts from `audit.json`; and cross-model
agreement from `verification.json` joined to the exact-grounded request set in the bundled finding
ledger. It recomputes artifact hashes from the staged bundle and never copies these scalars from UI
code. Release
disposition is a closed enum: any complete/trust-valid/validated scaled corpus must be
`scaled_promoted`; retaining v1 requires the matching technical-failure code. The rendered
disclosure is produced from that enum and the two corpus sizes, not free text, and is shown in the
selected script/footer.

`release_selection.json` contains the same `disposition`, `frozen_v1_primary_papers`,
`selected_release`, and `rendered_disclosure`. Its `scaled_attempt` is null for an initial v1 freeze;
otherwise it is `{status:selected|rejected|incomplete, failure_code, last_completed_stage,
candidate_release_id, primary_grounded_papers, stage_run_sha256s, evidence_sha256s}`.
`candidate_release_id` and `primary_grounded_papers` are non-null only after the candidate identity
and count recompute successfully; both are required for `status=selected`. `last_completed_stage`
is `null|s3|s4|s5` and names the
highest stage with a complete, hash-valid run and valid upstream prefix. `stage_run_sha256s` has
exactly the nullable keys `s3,s4,s5`: keys through `last_completed_stage` are non-null and later
keys are null. `evidence_sha256s` has exactly the nullable keys
`g3_gate,audit,verification,headline,baseline`; a value is non-null only when that artifact and its
declared stage/upstreams were hash-valid, so G3/audit/verification require a non-null s4 hash and
headline/baseline require a non-null s5 hash. A selected attempt requires every stage and evidence
hash to be non-null.

The only legal state combinations are:

| `disposition` | `scaled_attempt.status` | literal `failure_code` | selected corpus |
|---|---|---|---|
| `v1_frozen` | attempt must be null | n/a | frozen v1 |
| `scaled_promoted` | `selected` | null | scaled; its ID, size, and hashes equal the attempt |
| `v1_retained_scaled_incomplete` | `incomplete` | `scaled_incomplete` | frozen v1 |
| `v1_retained_scaled_corrupt` | `rejected` | `scaled_artifact_integrity_failed` | frozen v1 |
| `v1_retained_scaled_unreconciled` | `rejected` | `scaled_ledger_reconciliation_failed` | frozen v1 |
| `v1_retained_scaled_unvalidated` | `rejected` | `scaled_trust_or_offline_validation_failed` | frozen v1 |

These four strings are the entire non-null `failure_code` enum. `scaled_incomplete` means the
scaled pipeline or export did not reach a validated candidate by freeze; it does **not** describe a
scientifically complete s5 stage finalized with `m4_gate.status=incomplete`, which is promotable if
all trust, lineage, and offline checks pass. Classification is deterministic when conditions
overlap: a detected hash/schema/allowlist integrity failure wins first, then a reconciliation
failure on integrity-valid ledgers, then an explicit G3-trust or offline-validation failure; only
an otherwise failure-free attempt that has not reached all required checks by freeze is
`scaled_incomplete`. A candidate that passes all checks is `scaled_promoted`. For every retained-v1 row,
`selected_release.corpus_role=v1`, its `primary_grounded_papers=frozen_v1_primary_papers`, and its
release ID, stage hashes, and evidence hashes exactly equal the archived frozen-v1 record; the
attempt's hashes remain only under `scaled_attempt`. On `scaled_promoted`,
`selected_release.corpus_role=scaled` and its release ID, primary-paper count, stage hashes, and
evidence hashes exactly equal the scaled attempt. Hash fields for a rejected or
incomplete attempt are populated only through the last independently validated artifact and are
null thereafter; corruption never blesses the mismatching artifact's hash. Neither record includes
an s7/manifest hash, so there is no recursive checksum.
s7 first stages and validates a scaled candidate; if validation fails, it rebuilds from the frozen
v1 bundle and writes the rejected attempt into the new fallback selection record. The manifest
hashes `release_selection.json` like every other bundled artifact.

The disclosure templates are exact: `v1_frozen` → “Release: frozen v1 with V grounded primary
papers; no scaled candidate was promoted.” `scaled_promoted` → “Release: validated scaled corpus
with N grounded primary papers, superseding frozen v1 with V.” A retained-v1 disposition →
“Release: frozen v1 with V grounded primary papers; the scaled candidate was not promoted because
[fixed reason].” Fixed reasons are respectively “it did not complete,” “artifact integrity failed,”
“ledger reconciliation failed,” and “trust or offline release validation did not pass.”

Quality denominators are exact:

- `grounded_fraction = exact_grounded / accepted_primary_family`, where the denominator is every
  accepted, non-quarantined finding from a deterministically included, model-eligible paper in the
  locked primary family, before direction, section, or verifier filtering;
- `quarantine_fraction = quarantined_raw / (accepted + quarantined_raw)` across all findings
  returned by successfully mapped, deterministically included papers; a zero denominator is null
  and cannot pass G3; and
- `cross_model_agreement = model_status_agree / requested_exact_grounded`, where every exact-
  grounded accepted finding is requested. `disagree` and `unverifiable` remain in the denominator;
  adjudication never rewrites or improves this model-agreement rate. Missing decisions invalidate
  the artifact. The G3 ≥85% threshold uses this value.

The mixed/unclear and section-flagged exclusion fractions use `accepted_primary_family` above as
their denominator. `verification_excluded_fraction` uses exact-grounded, unflagged, canonical
3-class primary rows as denominator and counts rows lacking model agreement or an `accept`
adjudication. All numerators and denominators are also stored, even when the rendered manifest shows
only the fraction.

s7 recomputes hashes, validates paper/finding joins and all displayed scalar values, and refuses
missing, extra, stale, mixed, or dirty-lineage artifacts unless the dirtiness is explicitly
recorded and `--allow-dirty-demo` is supplied. The dashboard asserts the same manifest at startup.

---

## 5. Empirical protocols

### 5.1 s0 smoke test — STATUS: core probes DONE, G1 PASSED (`data/raw/smoke/S0_FINDINGS.md`)
**Done (archived smoke run):** auth verified (both keys); array-schema probe on 4 papers produced
6 raw/accepted findings and 0 quarantines: 5 RCT rows plus 1 schema-valid review-pollution row that
is reference-section flagged; the other 2 papers were clean ineligible `findings:[]` responses.
Thus 5 rows are non-section-flagged candidates, not a production headline cohort. Latency measured
(8 s / 4 papers); evidence_lines
format pinned (`["L18"]`); count semantics learned (`-c` = min(n, hits) — NOT a total; true
counts come from `sql`); default search is a **recency-weighted slice** (all 2024–2026) —
corpus construction MUST use `--all`; anchor coverage checked: **Paulsen 2014, TREAT 2020, and
Liu NEJM 2022 are absent from the corpus** (paywalled-journal PMC deposits missing; corpus
otherwise spans 1781–2027 with 73 pre-2023 TRE papers) → audit anchors are picked FROM the
corpus at triage, and coverage is disclosed honestly in the demo. Named map workers are
**gated to GXL testers** (ask organizers). Live confirmation of §41-2/review pollution: a
review self-marked eligible and extracted a *cited* study's finding with evidence lines pointing
at a reference-list title — deterministic screening + section flags are load-bearing, not
theoretical.

**Ingestion-readiness assertions (G1b; fold into M0/M1 and pass before any >10-paper map):**
1. Assert `evidence_quote` is a substring of the cited `content.lines` range on 3 probe rows;
   construct one citation URL, resolve it, and save the result in the smoke report.
2. Parse the archived probe output (`probe_map_m_2bc51e4b.json`) into exactly 4 PaperRecords,
   6 raw findings, 6 accepted FindingRows, 0 quarantines, and 1 reference-section-flagged row;
   exactly 5 rows remain non-section-flagged. Authoritative doc IDs must equal the four envelope
   IDs, not model output.
3. Force bogus-result-ID and invalid-schema failures; assert stable failure codes, nonzero exits,
   raw stderr archival, and that only observed transient/rate-limit codes enter the retry allowlist.
4. Kill an exactly 10-paper map after at least one completion, resume it, and assert final PaperRecord IDs
   equal the input include set with no duplicate terminal records or findings.
5. Watch for rate-limit signals during triage probes; ask organizers for quota numbers.

G1 means schema feasibility is proven. G1b means the local ingestion/resume path is safe. Neither a
4-paper latency result nor G1 alone authorizes the 150–300-paper scale target.

### 5.2 Topic triage and G2 (≤90 min)

Auth is resolved, so the full protocol runs as soon as the plan is approved; the old auth
degradation ladder is retired.
**Volume estimation = `paperclip sql` counts** (the `-c` flag does not count; s0 finding).
Priors already measured: title-level matches in PMC — **antioxidants/vitC/vitE × exercise/
training: 315; time-restricted eating: 229 (73 pre-2023)**; abstract-level counts will run at
triage. **All corpus-building searches use `--all`** (default search is a recency-weighted
slice) plus one recency-slice pass so both eras are represented; audit anchors are selected
from papers confirmed present in the corpus.
Per topic (C, A): search families incl. explicit null/negative phrasings → merge/dedupe →
deterministic screen → count gate → isolated `triage_probe` extraction of exactly 10 logged papers.
B and C+B: counts only (~2 min). Probe outputs are scoring inputs only and are never promoted or
mixed into the winner's s3 corpus; the locked winner is extracted afresh under its final hash.
Score: est. usable primaries, probe paper-level entropy point estimate + bootstrap interval,
moderator-field coverage, outcome-family concentration, review-pollution rate, full-text rate,
**relation-purity check** (human labels every accepted probe finding's intervention/comparator/
outcome as on/off target). For each probe paper with accepted findings, compute its on-target
fraction; relation purity is the unweighted mean of those paper fractions, so a five-finding paper
still weighs one. Zero-finding/ineligible papers are reported separately and do not enter this
denominator; no accepted findings means undefined and fails G2. Below 70% → tighten eligibility
before trusting entropy. Plus the
**dose-regime grep count** for C (D10 tie-break condition).
**Gate G2 is permissive topic viability, not a scientific claim.** A topic passes when its
paper-level normalized-entropy point estimate is ≥0.30, its 90% bootstrap upper bound is ≥0.40,
at least two canonical directions each occur in ≥2 distinct probe papers, relation purity is
≥70%, and est. usable primaries is ≥40 (est. = identity-deduped deterministic-included count ×
probe extraction-eligible fraction). For C, D10's ≥5-paper-per-dose-side support also applies.
**Any requirement fails → escalation ladder**, decided now: C+B merge (trigger per D10) → topic D
(keto × cancer) → ask a venue domain expert for a known-contested question. At lock, freeze the
complete §4.4 config—not only dose bins—and record every score and decision in
`docs/planning/triage.md` (demo material: “we chose empirically”).

### 5.3 Extraction validation (before scaling past ~30 papers)
- Automated: §4.3 schema, referential-integrity, grounding, section, direction, range, and
  version/lineage rules + quarantine with stable reason codes.
- Human audit: fixed-seed stratified sample of 20 headline-eligible rows, balanced as far as
  possible across canonical direction, primary endpoint, and paper. A row is correct only when
  eligibility, atomicity, intervention/comparator/outcome/timepoint, direction, and quote support
  all pass. Audit the first 10 papers while the next 20 extract; after a prompt fix, delta-audit the
  affected fields plus five unchanged controls. Report raw counts (for example `17/20`) with Wilson
  interval and the error taxonomy, never a bare percentage. Fewer than 20 available rows cannot
  pass G3.
- Anchor-paper check: every locked anchor's eligibility, finding count/atomicity, direction, and
  cited evidence must match the human-authored expected values in config.
- **Cross-model verification (A19):** a non-extractor model (Claude API, different model than
  extraction; independent of the Paperclip runner) checks quote⇒direction entailment. Runs in two
  passes: **v1 subset at G3 time** (gate input), full corpus overnight (concurrent with M5 and not
  allowed to endanger the frozen v1 demo); denominator is all grounded rows; ungrounded rows are
  reported outside that rate. The exact §4.5 formula counts model `disagree` and `unverifiable` as
  non-agreement and never lets adjudication improve the ≥85% model-agreement gate. Disagree/
  unverifiable rows enter the audit queue; adjudication controls row inclusion only. The full pass
  and resulting dispositions are required before a scaled candidate may supersede v1.
**Gate G3 separates extraction trust from story viability.** Trust passes iff human audit ≥17/20,
all anchors pass, v1 grounded-row cross-model agreement ≥85%, quarantine ≤10%, and G1b passed.
Story viability passes iff the primary grounded paper-level entropy 90% bootstrap lower bound is
≥0.40, at least 20 classifiable papers remain, and at least two canonical directions each have ≥5
distinct papers. An upper bound merely reaching 0.40 cannot pass G3. Both trust and story are
required for Narrative A to continue to M4. Trust failure → one prompt fix, then drop failing
fields/topic after a second failure; no scientific demo is exported from an untrusted cohort.
Trust passes but story viability fails → select Variant B with
`selection_reason="g3_story_not_viable"`, skip M4 inference and remap, and show the agreement/
insufficient-support result plus evidence gaps. A topic pivot is allowed only by restarting at G2
with a new locked config while time remains; never prompt-fish or weaken G3.

---

## 6. Statistical methodology (pre-registered; D6)

### 6.1 Cohorts, weighting, and disagreement

The **primary headline cohort** is fixed before analysis: deterministically included and
extraction-eligible papers; locked primary outcome family; canonical 3-class direction;
`grounding_status=exact`; `section_flagged=false`; and quote⇒direction verification either agrees or
has been human-adjudicated. Unresolved verification disagreements, `mixed`, `unclear`, quarantined,
secondary-family, ungrounded, and section-flagged rows remain counted and are shown in exclusion
tables. An **all-valid-row sensitivity** uses every non-quarantined 3-class primary-family row and
explicitly reports how many rows relax each primary filter. Other named sensitivities are
cohort-collapsed, peer-reviewed-only, and unweighted-findings.

Within any analyzed subset S, finding `i` from paper `p` has weight
`w_i = 1 / n_findings(p, S)`; normalize weights within each train/test fold to sum to the number of
distinct papers. This weighting is used in model fitting, log loss, majority agreement, and the
tree. Grouping alone is not treated as a pseudoreplication fix.

Finding-level normalized entropy uses the paper-balanced class proportions and denominator
`log(3)` even when a class is absent. Paper-level entropy first takes each paper's unweighted modal
direction; ties become `mixed` and are excluded with the tie rate reported. Both estimates, their
90% intervals from 1,000 fixed-seed paper bootstraps, class counts, paper counts, and every
exclusion rate are emitted. These cheap entropy bootstraps are separate from the 200 full-model
stability bootstraps.

### 6.2 Moderator model and feasible grouped CV

For each moderator, work only on its non-missing primary-cohort subset and recompute both model and
baseline on identical rows/folds. Let `C_X` be the canonical labels represented in that subset;
at least two are required. Candidate `k` starts at
`min(5, minimum distinct-paper count across C_X)` and decreases until every train and test fold
contains every label in `C_X`. Use `StratifiedGroupKFold(shuffle=True, random_state=seed)`.
`k>=3` is required for an M4/agent keep decision; `k=2` is labeled exploratory; no feasible split
is “insufficient for CV.” **Do not skip individual folds and then compare unequal fold sets.**

All numeric moderators are converted to config-pinned bins before outcomes are inspected. Before
encoding, pool levels represented in fewer than three distinct papers into model-only `__OTHER__`
using only moderator values and paper IDs; `__OTHER__` can never be narrated. The fixed model is
`OneHotEncoder(handle_unknown="ignore", drop=None)` followed by L2 multinomial logistic regression
(`C=1`, `solver="lbfgs"`, `max_iter=2000`, no class weights). Row-count `min_frequency` is forbidden
because a 15-finding paper must not make a rare level look supported. Pass paper weights to fitting
and scoring. The fold baseline is a custom canonical-label prior with Laplace `alpha=1` fitted on
that fold's weighted training labels; do not use a full-dataset `DummyClassifier`. Align
probabilities to `[increase,no_effect,decrease]`, fill absent model columns with zero, floor at
`1e-3`, then renormalize before log loss. A within-paper moderator's config-frozen paper summary is
its inferential representation; the row-varying form is descriptive. Without that summary, it has
no permutation p-value and cannot headline.

Rank by mean fold-paired `ΔLL = LL_baseline - LL_moderator` in natural-log units per
paper-balanced finding. Also report each fold, standard deviation, positive-fold count, coverage
by findings and distinct papers, support by level/class, and descriptive MI. Every config-listed
moderator appears, including failures and insufficient rows.

### 6.3 Exact headline contract and M4 gate

For moderator X, `S_X` contains primary rows in original, non-missing, non-`__OTHER__` levels
supported by ≥5 distinct papers; for within-paper moderators these are the frozen inferential
paper-summary values. Recompute paper weights inside `S_X`. `coverage_papers` is the number of
distinct primary-cohort papers represented in `S_X` divided by all distinct primary-cohort papers.
Define the same-subset global baseline `Q_X = max_y Σ(w_i·1[y_i=y])/Σw_i`. The weighted global mode
and every supported level's weighted modal direction `c_l` must be unique; ties are not broken by
label/config order and make X headline-ineligible. Define conditional agreement
`P_X = Σ(w_i·1[y_i=c_{X_i}])/Σw_i` and material gain `D_X=P_X-Q_X`. The displayed contrast is the
two differently directed levels with greatest combined distinct-paper support; ties follow locked
config order. The tree supports visualization but never selects X, Q, P, or the contrast.

`headline.json` contains, at minimum:

```json
{
  "narrative_variant": "A",
  "cohort_definition": "primary_grounded_unflagged",
  "analysis_labels": ["increase", "no_effect", "decrease"],
  "comparison_subset": {"n_findings": 0, "n_papers": 0, "coverage_papers": 0.0},
  "global_baseline": {"modal_direction": "increase", "agreement_q": 0.0},
  "within_regime": {"agreement_p": 0.0, "absolute_gain": 0.0},
  "contrast": {"level_a": "...", "direction_a": "...", "n_papers_a": 0,
               "level_b": "...", "direction_b": "...", "n_papers_b": 0},
  "moderator": {"name": "...", "k": 0, "delta_ll": 0.0, "positive_folds": 0,
                "westfall_young_p": 1.0},
  "stability": {"pattern_fraction": 0.0, "eligible_fraction": 0.0,
                "top3_fraction": 0.0, "n_bootstraps": 200},
  "rendered_sentence": "..."
}
```

Variant A passes M4 only when one **pre-specified** moderator satisfies all of:

1. feasible `k>=3`, ≥60% distinct-paper coverage, and every narrated level has ≥5 papers;
2. mean paper-balanced ΔLL ≥0.02 and positive in at least `ceil(0.6·k)` folds;
3. Westfall–Young family-wise adjusted `p<0.10` from 100 paper-aware permutations;
4. the same-subset global mode is unique and at least two supported levels have different unique
   modal directions;
5. `D_X≥0.10` on the exact moderator comparison subset;
6. the same named contrast, both directions, and `D≥0.10` recur in ≥60% of **all** 200 paper
   bootstraps; ineligible resamples stay in the denominator and are reported separately;
7. all-valid-row sensitivity preserves positive gain and both contrast directions.

If multiple moderators pass, choose greatest ΔLL, then lower adjusted p, then locked config order.

Variant A is selected iff at least one moderator satisfies **every** condition; §6.3's deterministic
tie-break chooses the sole headline winner even when several pass. If no moderator satisfies all
conditions, M4 selects Variant B; the tree is labeled exploratory, remap is cut, and no “hidden
variable explains disagreement” sentence is emitted. There is no manual override after looking at
results without a logged deviation that forces Variant B wording.

The exact Variant A sentence is: “Among N papers (M grounded directional findings) reporting
[moderator], one global direction matches Q% of paper-balanced findings; regime-specific directions
match P% (+D points), changing from [direction A] for [level A] to [direction B] for [level B]. The
same material contrast appeared in B% of paper bootstraps (Westfall–Young adjusted p=p).” Store
unrounded values; render percentages with decimal half-up rounding to whole points and p to three
decimals.

### 6.4 Agent-proposed moderator keep/discard rule

Variant B never runs the remap. For Variant A, freeze the selected M4 headline model before showing
residual pairs to the proposal agent. The post-hoc candidate is **exploratory**, is never added to
the original Westfall–Young family, and cannot replace the pre-specified headline.

The remap comparison is incremental on the exact intersection of non-missing rows:

- before: the frozen headline-moderator model;
- after: that identical model plus the proposed moderator;
- both: identical rows, folds, paper weights, label alignment, and preprocessing.

A technically valid remap has no unmatched or duplicate returned IDs and valid unique responses
for ≥95% of expected `finding_id`s. Null values count toward join success but not moderator
coverage. The decision is `kept_exploratory`, `discarded`, or `indeterminate`.
`kept_exploratory` requires every one of:

1. technical join success ≥95%;
2. non-null coverage ≥60% of primary-cohort papers;
3. at least two levels with ≥5 papers each;
4. feasible `k>=3`;
5. mean incremental `ΔLL = LL_before - LL_after >=0.02`;
6. positive incremental gain in at least `ceil(0.6·k)` folds;
7. candidate-only paper-aware add-one permutation `p<0.10`, holding the frozen baseline model,
   outcomes, missingness, and folds fixed; require 100 successes within 125 attempts and compute
   `p=(1 + #{incremental ΔLL_b >= incremental ΔLL_obs})/101`;
8. positive incremental gain in ≥60% of **all** paper bootstraps; and
9. all-valid-row sensitivity preserves the incremental-gain sign.

Any failed numeric rule means `discarded`. An incomplete run, invalid echo-back join, undefined
permutation, or invalid exchangeability scheme means `indeterminate`; it may be descriptive but
cannot be called retained. `trace.json` records every denominator, threshold, observed value,
rule result, and reason. Regardless of result, say “cross-validated incremental gain, moderator
proposed post hoc,” never “held-out,” “confirmed,” or “discovered explanation.”

For that candidate-only permutation, a paper-constant value is shuffled among its non-missing
papers and replicated to findings. A within-paper value uses only the human-approved paper summary
and exchangeability rule; without one it is `indeterminate`. Fewer than 100 successful permutations
within 125 attempts is also `indeterminate`, not `discarded`.

### 6.5 Permutation, bootstrap, tree, and residuals

- Paper-constant moderators: permute values across non-missing papers and replicate to their
  findings. Within-paper moderators: use the config's pre-specified paper-summary permutation;
  otherwise emit no p-value and make the moderator ineligible for headline/keep. Missingness and
  folds remain fixed. Each permutation records the maximum ΔLL across the eligible pre-specified
  family for single-step Westfall–Young adjustment. Require 100 successful permutations, retrying
  guarded failures up to 125 total attempts; fewer than 100 makes the M4 permutation rule fail.
  Use add-one values `p_raw=(1 + #{ΔLL_b >= ΔLL_obs})/(R+1)` and
  `p_WY=(1 + #{max_j ΔLL_b,j >= ΔLL_obs,j})/(R+1)`, where the maximum spans the complete
  pre-specified permutation-eligible family. Demo prose quotes only `p_WY`.
- Run 200 paper bootstraps and 100 successful permutations with the fixed seed; every resample is
  guarded, skip/error counts are explicit, and partials checkpoint every 25. Resampled copies
  retain the original paper group ID; multiplicity is represented by weight. The full battery runs
  on v1 at M4 before any overnight confirmation.

A run that reaches all 125 permitted permutation attempts but yields fewer than 100 successes is
terminal `status=complete` with a failed permutation rule; no moderator can pass, so the reason is
`m4_no_moderator`. `m4_incomplete` is reserved for interruption before the registered attempt/
bootstrap budget finishes. The raw partial run never ships; its explicitly finalized incomplete
scientific state may ship only Variant B, only after G3 trust passes and all
descriptive/residual/gap artifacts validate. Unevaluated rules are null/`not_evaluated`, not failed.
It never supports a claim that moderators were tested and rejected.

An interrupted run is never exported directly. The ordinary s5 process writes atomic,
content-addressed checkpoints under `data/checkpoints/<qid>/s5/` initially and after every 25
bootstrap/permutation attempts. The source object has literal `checkpoint_version:"1"` and
`checkpoint_status:"running_snapshot"`; it records its source run ID and fixed timestamps;
question/config/code/cohort/G3/input hashes; seed and registered budgets; completed draw and attempt
indices with their results; successful-permutation indices; guard failures; and hashes of the
complete descriptive, residual, and evidence-gap inputs/outputs. To freeze an interruption,
the operator must run the mutually exclusive operation
`s5_analyze.py --finalize-incomplete-from <checkpoint>`. The finalizer:

1. verifies the checkpoint hash, exact qid/config/code/upstream hashes, G3 trust/story pass, and that
   at least one registered bootstrap or permutation budget is genuinely unfinished;
2. archives the source under its canonical content-addressed checkpoint path, emits the
   `analysis/m4_checkpoint.json` wrapper defined in §4.5 with the full source object embedded, does
   not resume or draw another resample, and completes/validates the descriptive, residual, and
   evidence-gap artifacts from the frozen inputs;
3. emits typed `incomplete` bootstrap/permutation/tree components, null/`not_evaluated` unfinished
   M4 rules, a Variant-B `headline.json` with `selection_reason=m4_incomplete`, and a `not_run` remap
   trace with that same reason; and
4. replaces the ineligible partial s5 record with `run.json.status=complete`,
   `completion_mode=frozen_incomplete`, and the required `checkpoint_sha256` equal to the wrapper's
   `source_checkpoint_sha256`, while
   `m4_gate.status=incomplete` remains explicit.

Finalization is deterministic: `run_id` is derived from the source checkpoint hash, `started_at` and
`completed_at` come from the source-run/checkpoint timestamps, `command_argv` names the canonical
repository-relative archived checkpoint, and no finalizer wall-clock value enters a hashed
artifact. Re-running finalization from the same checkpoint with the same code and scientific inputs
must reproduce every s5 hash. Re-running s7 reproduces every release hash only from the complete
frozen export input set: those s5 outputs plus the processed ledgers, G3/audit/verification
artifacts, immutable one-shot `baseline.json`, exact script-template bytes, config/code, and the
same frozen-v1/scaled-attempt selection inputs. s7 never regenerates any of them. A checkpoint at a
terminal 125-attempt/200-bootstrap budget is ineligible for this path and must be finalized by the
ordinary completed-rule logic instead.

- Tree: depth ≤3, same primary cohort and paper weights, config-binned numerics, explicit visually
  distinct “not reported” branch. A post-fit audit greys (does not pretend to statistically merge)
  leaves with <5 distinct papers; grey/missingness leaves are never narrated. Report grouped-CV
  tree ΔLL vs its same-row baseline and root-split bootstrap frequency.
- Residual contradictions compare opposite-direction rows only within the same locked primary
  outcome family, require at least two shared non-missing context fields, use a declared Gower-like
  distance, and preserve both citations. G2 freezes `variant_b.axes` and their complete level/bin
  lists. `evidence_gaps.parquet` emits every Cartesian axis-level × locked-primary-endpoint cell,
  including zeros, with `cell_id`, `primary_endpoint`, `axis_values`, `n_papers_total`,
  `n_papers_grounded`, `n_findings`, `grounded_fraction`, `classifiable_fraction`, `paper_entropy`,
  and `status`. Status is `empty` for zero grounded papers, `sparse` for one–four, and `supported`
  for five or more. Entropy is null unless at least three classifiable papers and two directions
  exist. Variant B renders this as a simple table, not an undefined “map.”

---

## 7. Milestones & tripwires (PT)

The detailed dependency order, file ownership, tests, and safe parallel work are normative in the
paired implementation plan. Relative targets below are rebased when coding starts; gates, not wall-
clock optimism, authorize the next data scale. Only Sunday 9:45 AM demo freeze and 10:15 submission
are hard. Surplus goes to extraction quality, then corpus scale, then a second remap—never the
reverse.

| # | Target | Deliverable | Gate/tripwire |
|---|---|---|---|
| P0 | before product code | v2.3 + implementation plan complete; contracts, gates, Variant A/B scripts and Definition of Done frozen | user approves coding start |
| M0 | T+45 min | Scaffold; `PaperRecord`, fixed `FindingRow`, `QuestionConfig`, schema generator, lineage models; contract tests green; initial docs baseline committed | contracts compile |
| M1 | T+90 min | Archived parser test + live G1b resume/failure/evidence assertions; 10-paper paper-ledger reconciliation | **G1b** |
| M2 | T+150 min, parallel with fixture work | C/A 10-paper probes; B/C+B counts; `triage.md`; complete locked winning config | **G2** |
| M2.5 | T+240 min | Pathological papers+findings fixture drives s4→s7; planted moderator recovered, nulls rejected, duplication-invariance test passes; fixture demo exported | fixture gate |
| M3 | after M2/M2.5 | Real v1 (~30 papers) extracted; paper ledger reconciles every input; audit + anchors + grounded cross-model v1 check complete; trust failure blocks release, story failure selects B | **G3 trust + story recorded separately** |
| M3.5 | alongside M3 | Streamlit renders the exact fixture demo inventory offline, including human tree renderer, paper-joined evidence cards, and both scripted variants | offline UI gate |
| M4 | only after G3 trust+story pass, while awake | Complete real-v1 s5 inference: full moderator family, tree, residuals/gaps, 200 bootstraps + 100 successful permutations. Evaluate every §6.3 condition and atomically select A or B | **M4 Boolean** |
| M4.5 | only if Variant A, while Harry is awake | Residuals → proposal → approval → whole-v1 remap → §6.4 decision. Variant B spends this block on residual/evidence-gap cards instead | agent gate |
| M4.75 | before stopping for the night | s7 freezes a real-v1 shippable demo; both lineage checks and offline rehearsal pass; backup video recorded | shippable freeze |
| M5 | late night / overnight, additive only | First target ≥50 extraction-eligible papers; then 50–150 quality target; **150–300 is stretch only after G1b + observed quotas/heartbeat are healthy**. Full grounded-row verification batch runs concurrently | freeze last clean checkpoint by Sun ~7 AM |
| M6 | Sun ~9:00 AM | Recompute gates on the staged scale result. A complete, trust-valid, lineage-valid scaled corpus supersedes v1 even if A→B; retain v1 only for recorded technical incompleteness/corruption/unvalidated state | scaled-promotion rule |
| M7 | Sun 9:45 AM | **Demo freeze.** 9:45–10:15 submission artifact production; submit 10:15 | **hard stop** |

Rubric + submission format from organizers **by the 9:45 PM checkpoint tonight at the latest**.

**Scaled promotion re-runs G3; it is not a row-count swap.** Reconcile the whole scaled ledger;
rerun every anchor; recompute quarantine on the whole candidate; run a new fixed-seed, stratified
20-row audit containing at least `min(10, number of newly added audit-eligible papers)` rows from
newly added **distinct** audit-eligible papers (one row per such paper before repeats); and complete quote⇒direction
verification for every grounded scaled row. Every
verifier disagreement is adjudicated or explicitly excluded from the primary cohort. Apply the
same ≥17/20 audit, 100% anchor, ≥85% verifier-agreement, and ≤10% quarantine thresholds, then
recompute story viability and M4/B selection. The manifest's release-selection hashes bind this G3
record, audit, and verification artifact. A trust-valid scaled result supersedes v1 even when the
story weakens; a trust failure is `v1_retained_scaled_unvalidated`, not a narrative choice.

For this quota, an audit-eligible paper is newly added, deterministically included, model-eligible,
and has at least one accepted locked-primary-family, canonical 3-class, exact-grounded,
non-section-flagged row before cross-model verification. Eligible zero-finding or otherwise
row-ineligible new papers remain in the first denominator but cannot satisfy the audit-row quota.

**Cut order under pressure:** repo-claim verification → bias funnel plot stretch → second remap →
first remap if M4 does not require it → Streamlit sections in reverse narrative order
(trace → contradictions → baselines). The registered 200 stability bootstraps and 100 successful
permutations are not reducible for Variant A; if they cannot finish, select Variant B with the
incomplete inference recorded. Never cut the paper/finding contracts, lineage, primary disagreement
counts, complete moderator support/eligibility table, the four core views, evidence cards, or
selected demo script. Any Variant A claim still requires the complete paper-balanced grouped-CV,
permutation, and bootstrap battery. Never switch stacks.

**Variant B is a complete product, not a placeholder.** It is selected because G3 story viability
failed, no moderator passed M4, or the fixed M4 battery did not complete. Its exact 90-second order
is: question and retrieval funnel → global grounded disagreement → complete moderator table with
the statement that no moderator was promoted through the pre-registered conditional-pattern gates
→ residual panel with the top grounded pair or the validated empty state →
`evidence_gaps.parquet` table showing which contexts lack grounded evidence → closer: “The system
refused to invent a hidden variable; it shows exactly what remains unresolved and what evidence is
missing.” The tree is explicitly exploratory, M4.5 remap is cut, and its time goes to these evidence
cards. The implementation plan creates and tests both scripts before real outcomes are seen.

For Variant B, `headline.json` is the other branch of a discriminated union and contains at least:

```json
{
  "narrative_variant": "B",
  "selection_reason": "g3_story_not_viable|m4_no_moderator|m4_incomplete",
  "cohort_definition": "primary_grounded_unflagged",
  "analysis_labels": ["increase", "no_effect", "decrease"],
  "disagreement": {"n_papers": 0, "n_findings": 0, "paper_entropy": 0.0,
                    "interval_90": [0.0, 0.0]},
  "m4": {"status": "not_run|failed|incomplete",
         "failures": [{"moderator": "...", "failed_rules": ["..."]}]},
  "residuals": {"pair_count": 0, "top_pair_id": null, "rendered_sentence": "..."},
  "evidence_gaps": {"sparse_or_empty_cells": 0, "total_cells": 0},
  "rendered_sentence": "..."
}
```

The Variant B sentence is rendered from exactly one pre-approved clause:

- `g3_story_not_viable`: “the corpus did not meet the pre-registered disagreement-support gate for
  a conditional-pattern claim”; or
- `m4_no_moderator`: “no pre-specified moderator passed every pre-registered inference,
  stability, support, sensitivity, and materiality rule”;
  or
- `m4_incomplete`: “the pre-registered moderator-inference battery did not complete before freeze”.

Exact sentence: “In our retrieved, grounded corpus, paper-level directional disagreement is H
(90% bootstrap interval L–U), but [approved clause]. R grounded opposite-direction paper pairs
remain, and G of T pre-registered evidence cells contain fewer than five papers.” Store unrounded
values and use the same rounding rules as Variant A. `trace.json` has `status="not_run"` and reason
`g3_story_not_viable`, `m4_selected_variant_b`, or `m4_incomplete`.

The residual sentence is also pre-approved: when `pair_count>0`, “This top pair remains opposite
despite matching on at least two recorded conditions; both source passages and line references are
shown.” When it is zero, `top_pair_id` must be null and the sentence is “No grounded opposite-
direction pair met the matching rule; the empty residual view is part of the result.” No other
branch is valid.

---

## 8. Risk register (each row owned by a named gate)

| Risk | Trigger/owner | Mitigation / fallback |
|---|---|---|
| Map rejects array schema | G1 | two-pass map (array-of-strings probe validates premise) or Claude-API extraction; G1 not passed until one path proven |
| Rate limits choke extraction | M5 runner (latency measured fast; quotas still unknown) | if throttled, cap corpus stratified **across query families and years** (else fame bias returns); phrase all stats as "of our N-paper retrieved corpus"; `--resume` makes partial runs safe |
| Paper silently disappears at screen/map/flatten | s1–s4 reconciliation | every s1 document resolves to one canonical PaperRecord/alternate disposition; every canonical s2 paper gets a terminal row; stage count identities and finding→paper integrity are hard assertions |
| Corpus coverage gaps (CONFIRMED: Paulsen 2014, TREAT, Liu NEJM absent) | triage | audit anchors picked from in-corpus papers; coverage disclosed in demo + Q&A sentence; recovery-check framing per topic unaffected |
| Named map workers gated to GXL testers (CONFIRMED) | s0 | default worker proven sufficient; ask organizers for tester enablement as an upgrade, never a dependency |
| Search-result bias (§41-1) | s1/G2 | multi-family queries incl. null/negative phrasings; query provenance per paper; stratified cap rule |
| Conclusion-section bias (§41-2) | s3/G3 | prompt prioritizes Results; mechanical section flag + demotion from headline leaves; cross-model quote⇒direction check |
| Outcome collapse (§41-3) | G2/G3 | outcome_family tiers; entropy per family; primary analysis on dominant family |
| Moderator hallucination (§41-5) | M4.5 | proposals must be extractable fields; echo-back remap; incremental CV gain + permutation before display; discard/indeterminate are scripted outcomes |
| Small-stratum storytelling (§41-6) | M4 | min-support in §6; leaf audit; N on every element; never narrate missingness branches |
| **Publication bias (§41-7)** | demo/Q&A | risk row + dashboard caveat card ("proportions are of published findings") + one rehearsed Q&A sentence — in front of the Forking-Paths coauthor |
| Duplicate cohorts (§41-8) | s2/G3 | DOI/PMID + fuzzy dedupe; trial-ID regex; G3 same-cohort hand-pass over headline-leaf papers; cohort-collapsed sensitivity rerun |
| Pseudoreplication | §6 | grouping prevents leakage; paper-balanced weights govern fit/score/tree; paper bootstrap and findings-per-paper counts remain visible |
| Our own forking paths | §6 | pre-registered stats; full ranked table; Westfall–Young family-wise p |
| Version/stale-artifact mixing | s4/s5/s7 | full config/input/output/code lineage; mixed input fails; s7 independently recomputes and refuses stale demo artifacts |
| ~~Auth/keys delayed~~ RESOLVED | — | both keys verified in `.env`; model identifiers and structured-output/batch parameters still receive cheap M0 connectivity tests before dependence |
| Overnight runner dies silently | M5 | checkpoint by paper_id, heartbeat file, additive-only, 2 AM freeze tripwire |
| Python 3.14 wheel gaps | M0 | pin 3.12 |
| Demo-time network flakiness | M7 | frozen artifacts only; cached quotes primary, URLs as captions; screenshots of top-3 claims; backup video from Saturday night |
| Grounded-only cohort too small | G3 | report denominator; fail story gate or use Variant B; never backfill headline claims with ungrounded/section-flagged rows |

---

## 9. Validation & credibility (demo-facing)

1. Audit as raw counts + Wilson interval; error taxonomy compared to published LLM-extraction
   profiles (omissions dominate — otto-SR, 27-study review).
2. Mechanical quote/line grounding status over every row; cross-model quote⇒direction agreement
   over the grounded denominator; ungrounded and section-flagged counts beside it.
3. When M4 is eligible, paper-balanced fold-paired grouped CV + family-wise permutation p +
   bootstrap stability on every tested moderator; the full table shows losers and ineligible
   candidates. A G3-story Variant B instead labels M4 `not_run` and shows support/eligibility only.
4. Recovery check per §4.4 config slot (A: published early/late NMA; C: honest "first
   quantitative test" framing).
5. Residual contradictions displayed, not hidden.
6. Language discipline: "strongest observed moderator," "stratified by," "cross-validated gain
   (proposed post hoc)," never "causes," never "held-out."
7. Baselines panel: majority vote + archived one-shot LLM consensus vs conditional structure, with
   Variant B explicitly showing when conditional structure did not survive controls.

## 10. Demo narrative (90 s)

**Variant A:** the prewritten `docs/demo/variant_a.md` sequence is normative: question → reconciled
corpus funnel → exact `headline.json` sentence → complete moderator table → audited tree and source
cards → qualified close. An optional agent trace is visibly post hoc and uses exactly
`kept_exploratory|discarded|indeterminate`; it cannot displace the core 90-second sequence.

**Variant B:** the exact sequence and closer are pinned in §7 and prewritten in
`docs/demo/variant_b.md`. The app renders the selected script from `demo/demo_script.md`; a demo
operator never improvises a stronger claim than the selected variant.

**Differentiation (rewritten, A18c):** three parts — (1) built from the raw literature with
row-level provenance, no hand-built dataset; (2) disagreement *decomposed*, losers and residuals
shown, not just a pooled estimate; (3) the agent extends its own schema, gated by cross-validated
testing. Call an agent-proposed variable “post-hoc and kept exploratory” only when every incremental
rule passes; do not call it a discovered explanation. Config-listed moderators like dose are
*pre-specified candidates*, and we say so.

## 11. Open questions

1. **Harry:** review v2.3 plus the paired implementation plan and approve the coding start → M0.
2. **Organizers, when convenient tonight:** judging rubric; submission artifact format;
   Paperclip quota numbers; **GXL-tester enablement for the gated map workers**
   (`structured-extraction` / `eligibility-screen` — nice-to-have, not a dependency);
   Claude-credit mechanism.
3. ~~Core s0 schema feasibility~~ ✅ resolved; G1b ingestion assertions remain the first live-code
   gate — see §5.1 and `data/raw/smoke/S0_FINDINGS.md`.

---

## Appendix: review trail

- Round 1 (7 lenses + synthesis judge): 58 findings → 8 blockers + 12 majors accepted, 8 rejected
  with reasons, 7 conflicts adjudicated. Full record: `docs/planning/reviews/`.
- Round 2 (compliance diff vs decision doc + fresh-eyes implementer read): both pass-with-fixes;
  24 issues, all applied in v2.1. The three must-fixes were: remap side-table contract (was
  contradicting the version guard), s7 runs never scheduled in the milestone table, and the
  variant-B gate statistic/time ambiguity. One deliberate deviation from the round-1 decision:
  the G3 cross-model clause runs on the v1 subset at gate time (full corpus overnight), resolving
  the decision doc's own timing tension.
- Notable: the ΔLL defect in context-doc §31 pseudo-code was independently verified by the
  adjudicator before being accepted as B3. v1's "dashboard last" was overturned unanimously (C5)
  as a misreading of §41-10.
- Round 3 (v2.2, empirical + directives): Approach A approved by Harry; both API keys verified;
  s0 core probes run — **G1 passed** (array schema on default worker), map latency measured,
  `--resume` checkpointing discovered (simplifies D3), `-c`/recency-slice semantics corrected
  (§5.2 now counts via `sql`), anchor-coverage gaps confirmed, review-pollution failure mode
  reproduced live. Schedule re-paced per Harry: targets not caps, build continues past 11 PM.
  Model tiering added as D11 (haiku wiring / sonnet-5 volume+batches / opus-5 reasoning).
- Round 4 (v2.3, pre-coding contract audit): made the design authoritative over the stale context
  examples; added the paper ledger, dynamic topic-moderator schema, exact types/grounding/lineage;
  replaced ambiguous CV and gates with executable rules; made paper balancing primary; pinned the
  headline and post-hoc keep/discard thresholds; fully specified Variant B; separated G1/G1b and
  G2/G3 purposes; and added the dependency-ordered implementation plan and Definition of Done.
