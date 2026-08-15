# Research Notes — Planning Inputs (2026-08-15)

Condensed from a 6-agent research sweep (Paperclip docs, hackathon page, Zou paper, prior art,
topic pre-scan, completeness audit). Full raw output archived in the session workflow transcript.
Facts marked **UNVERIFIED** must be resolved by the smoke test or by asking organizers — do not
build on them.

---

## 1. Hackathon ground truth (from luma.com/g6org075, verbatim where quoted)

- **Submission deadline: Sunday Aug 16, 10:45 AM PT.** Demos + live judging 12:30–2:00 PM Sun.
- Build sessions today (Sat Aug 15): 1:00–3:30, 3:30–6:30, 7:15–9:45 PM; overnight checkpoint
  9:45–10:15 PM; day close 11:00 PM. Sunday final build 9:00–10:45 AM.
- Track B verbatim: "…find the pattern in what you assembled that no single paper could show
  you, and demo that."
- Resources: "Free Paperclip API access, access to Proto, Claude API credits, and compute and
  credits from our sponsors." Distribution mechanism **UNVERIFIED** (presumably at event).
- Judging criteria: **not published**. Submission artifact format (video/repo/writeup):
  **not published**. Live demo clearly expected. → Ask organizers today.
- Hosts include James Zou (Garden of Forking Paths coauthor) and Brian Hie. Co-hosts: GXL,
  Arc Institute, Anthropic, BenchFlow, future.bio, Biohub.

## 2. Paperclip verified capabilities (docs + GitHub README + installed CLI)

**Confirmed:**
- Auth: `paperclip login` (browser OAuth) or `PAPERCLIP_API_KEY` env var / `--api-key`.
  Keys minted at paperclip.gxl.ai/keys. Creds at `~/.paperclip/credentials.json`.
- Search: `-n` default 100 / max 1000; `--source pmc,biorxiv,medrxiv,arxiv,abstracts…`;
  `--ranking hybrid|bm25|vector`; `-e` exact; `--year`; `--since`; `--article-type
  review-article|research-article` (PMC only — useful hard review filter); `--tag`
  accumulation; `-c` count-only; `--json`. `paperclip searches Q1 Q2 …` = parallel queries.
- `map --from RESULT_ID "question" --output-schema JSON [--limit N --offset N]` — JSON Schema
  draft 2020-12, supports `required`, `additionalProperties:false`, nullables. Output validated;
  **exactly one correction retry** on invalid output. Docs guidance: keep maps to **3–10 papers**;
  window larger sets with `--limit/--offset` against one result ID. SDK timeout for map: 300 s.
- Every search/map/reduce returns a persistent result ID (`s_…`, `m_…`) — reproducibility
  mechanism; TTL **UNVERIFIED** → save everything locally (`results --save`, `--json`).
- `filter --from ID "claim"` LLM relevance filter (per-paper cached); `refine` deterministic
  metadata filter; `merge/intersect/subtract` result sets; `sql` (SELECT-only, 200 rows, columns
  incl. article_type, pub_year, journal_title); corpus-wide `grep` (trigram, fast); `lookup doi|
  pmid|title …`; `ask-image` on figures; `reduce --strategy table|consensus|…`.
- Paper VFS: `/papers/<id>/{meta.json, content.lines, sections/, figures/}`. Line-numbered full
  text → citation URLs `https://paperclip.gxl.ai/citations/papers/<doc_id>#L<n>`.
- Repos: `repo add <id> "claim" --lines A-B` + `repo commit` runs **claim verification against
  full text by parallel subagents**. Opt-in; strong provenance story for top demo claims.
- Built-in routine `paperclip-meta-analysis` exists (line-pinned JSON claims + deterministic
  compile/QA). Worth 10 minutes to inspect once authed; do not depend on it.
- Python SDK `gxl_paperclip`: `PaperclipClient.from_env()`, `search()`, `map_()` (streaming
  generator), `execute(command, args)` escape hatch. Sync only.

**UNVERIFIED / smoke-test targets:**
1. **Does `map --output-schema` accept an array-of-findings schema** (multiple findings per
   paper)? Docs example is a flat object. *Biggest single unknown in the whole plan.*
2. SDK `map_` schema parameter (CLI flag is confirmed; SDK signature doesn't show it → likely
   go through `execute()` or subprocess CLI).
3. Rate limits / quotas / allowed concurrency on the hackathon tier. RateLimitError class
   exists; numbers unknown. Determines overnight extraction ceiling.
4. Full-text coverage of anchor papers (exercise-physiology journals often paywalled → may be
   abstract-only, which can't populate dose/timing fields). Check via `lookup` for: Paulsen 2014
   (J Physiol), Yfanti 2010/2011, Lowe/TREAT 2020, Liu NEJM 2022, 2–3 ITP rapamycin papers.
5. Map behavior when a paper has zero findings (null-fill vs row-drop).
6. Result-ID TTL; map parallelism ("20 sub-agents / ~15 s per paper" is from a search snippet —
   don't budget on it).
7. API key prefix discrepancy (`gxl_` vs `pk_`) — never validate key format in code.

## 3. Garden of Forking Paths (arXiv 2607.01507) — what to borrow

- Vocabulary to echo: *analysis path*, *analysis space / multiverse*, *defensible analyses*,
  *selective exploration and reporting*, *m-value*, *Agentic Bootstrap*, *specification curve*.
  Our framing: "They estimate a distribution over analysis paths on one fixed dataset; we
  estimate a distribution over **experimental paths across many papers** — the forks move from
  researcher degrees of freedom to experimental-design degrees of freedom."
- Evaluation ideas to analogize (cheap, high-credibility):
  - **Permuted-null control**: shuffle moderator labels across papers (grouped) → MI / CV gain
    must collapse to baseline. Direct anti-overfitting evidence.
  - **Cross-model review**: a different model (not the extractor) audits extracted rows.
  - **Stratified human audit** with reported agreement.
  - **Recovery test**: validate against a published meta-analysis with known subgroup effects.
- Their warning that applies to us verbatim: cheap multiverse exploration **amplifies selective
  reporting** — moderator search is itself a garden of forking paths. Guards: grouped CV,
  permutation baseline, paper-level bootstrap stability, min-support thresholds, report ALL
  moderators tested (not just winners).

## 4. Prior art — differentiation and credibility ammo

- **Nearest neighbor: AutoSynthesis** (arXiv 2607.15247, Jul 2026): agentic end-to-end
  meta-analysis incl. heterogeneity analysis over *user-specified* moderators. Our
  differentiation: automatic moderator **discovery** + disagreement decomposition (entropy) +
  residual contradiction pairs. Pitch line: say "automatic moderator discovery," never "first
  automated meta-analysis."
- Elicit (extraction, no moderator analysis), Consensus (vote-count meter), Scite / FutureHouse
  ContraCrow (flag contradictions pairwise, don't explain), OpenScholar (QA synthesis).
- Extraction credibility citations: otto-SR (medRxiv 2025, preprint/self-reported): LLM
  extraction 93.1% vs human 79.7%; 27-study review (J Biomed Inform 2026): categorical fields
  74–96%, hallucinations 0.08–6%, **errors dominated by omissions** → design for omission
  (nulls allowed, never force values); Claude beat GPT on extraction head-to-head.
- Methods lineage to cite: metafor (I², tau², meta-regression), **MetaForest** (clustered
  bootstrap RF variable importance), **metaCART** (tree leaves = subgroups). Honest framing:
  "MetaForest for the raw literature — no hand-built dataset required."
- MI-based moderator discovery across studies: we found no prior work (claim as "we found
  none," not "none exists").

## 5. Topic pre-scan verdict (to be confirmed by empirical triage)

Ranking for triage: **C (antioxidants × exercise) > A (TRE/IF → metabolic) > E (rapamycin,
scoped to one outcome family)**. D (keto × cancer) backup; B (metformin × exercise) too small
alone (~25–50 papers, lab-concentrated).

- **C's money shot**: published 2019 SR/MA reports *null* pooled effect on aerobic adaptations,
  while the literature narratively attributes the Paulsen-2014-vs-Yfanti contradiction to a
  **vitamin C dose threshold (~500 vs 1000 mg/day)** — an aggregate null that a dose split may
  resolve into two opposite-signed regimes. Exactly our thesis. No published meta-regression on
  dose confirmed (**verify; absence strengthens the demo**). Corpus size 60–120 papers is
  **inferred, not counted** — triage must confirm ≥~60 usable primaries or C drops behind A.
- **Strategic merge option**: define corpus as "pharmacological/nutritional blunting of exercise
  adaptation" → C + B share one judge-friendly question ("does the pill cancel the workout?"),
  doubling corpus, adding intervention-class as a top-level moderator.
- **A**: largest corpus (150–300+ RCTs), pre-registered moderator structure (early vs late
  window × isocaloric vs deficit), many nulls → treat null as first-class third category.
- **E**: richest moderators (dose, sex, age-at-init, intermittent vs continuous, species,
  compound identity) but worst outcome sprawl; only viable scoped to one outcome family
  (mouse lifespan, or immune/vaccine response). Best 15-s hook: the immune paradox.
- Triage risk flags: review pollution (A worst), endpoint tiers needed (C: molecular vs
  functional), compound conflation (E: rapamycin ≠ everolimus ≠ RTB101).

## 6. Decisions this research forces into the plan

1. Smoke test (auth → counts → anchor lookups → **one nested-array-schema map on 3–5 papers**)
   is the first empirical act; extraction architecture is chosen by its outcome, by end of
   Build Session I.
2. Fallback extraction path must be designed now: pull `content.lines` via `papers.cat`/`pull`
   + direct Claude API extraction into the same pydantic models (Claude credits are provided).
3. Save every raw response locally; never rely on server-side result IDs surviving to Sunday.
4. Ask organizers today: rate limits, map schema expressiveness, judging rubric, submission
   format.
5. Extraction schema must allow nulls everywhere; never force a value (omission is the
   dominant LLM error mode).
