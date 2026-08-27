# Literature Multiverse

Literature Multiverse is a budget-aware verifier for AI-generated scientific claims. Given a
directional claim, proposed conditions, a frozen literature corpus, and a human-verification
budget, it determines whether that corpus supports, contradicts, qualifies, or cannot evaluate the
claim. It verifies **literature support under the declared corpus**, not scientific truth.

The primary executable is one fail-closed command:

```bash
lm verify \
  --claim claim.yaml \
  --corpus frozen-corpus-or-evidence-graph \
  --budget-minutes 60 \
  --output-dir artifacts/verification/run
```

It produces a self-hashed JSON certificate and a static, no-JavaScript HTML rendering containing
the corpus ledger, typed evidence graph, synthesis, graph-derived counterfactual reruns, ranked
verification actions, residual-risk bound, release/abstain decision, caveats, and lineage.

## Why this exists

A conventional literature summary tends to compress heterogeneous experiments into one answer.
This project keeps paper identity, comparison, outcome, timepoint, source passage, and context
separate, then asks which recorded conditions predict the reported direction out of sample. The
design deliberately makes a complete negative result preferable to a fragile positive story.

The normative contracts are the
[design](docs/superpowers/specs/2026-08-15-literature-multiverse-design.md) and
[implementation plan](docs/superpowers/plans/2026-08-15-literature-multiverse-implementation.md).
The much larger root context is historical background, not an executable specification.

## Supported verifier path

```text
AI-generated claim + protocol
        -> frozen corpus and eligibility ledger
        -> typed publication/study/cohort evidence graph
        -> cohort-aware effect synthesis and condition analysis
        -> graph-derived counterfactual audit priorities
        -> bounded residual-risk release or abstention
        -> hash-bound JSON/HTML verification certificate
```

Important guardrails:

- numerical synthesis accepts typed estimands, effect scales, uncertainty, cohort identities, and
  exact source provenance; legacy directional rows are retained but cannot silently become effect
  estimates;
- audit influence comes from rerunning the actual synthesis with an evidence item removed, not
  from a hand-authored additive score;
- partial-audit release requires calibrated marginal error-probability **upper bounds** whose
  dependency-robust union bound is below the frozen tolerance;
- `condition_dependent` requires a prespecified moderator with multiplicity-adjusted,
  opposite-direction subgroup intervals and remains a predictive rather than causal conclusion;
- missing calibration, unresolved cohort identity, untyped effects, distribution shift, or an
  unstable conclusion causes abstention;
- each certificate states that it assesses literature support, not scientific truth.

For a provider-free integration check:

```bash
uv run lm verify --fixture --budget-minutes 30 --output-dir /tmp/lm-fixture
```

## Current real-data boundary

The checked-in local benchmark runner uses no network or provider calls and never evaluates the
MetaSyn test split:

```bash
uv run python scripts/run_local_benchmarks.py --force
```

On the pinned 140,585-article MetaSyn release, the deterministic corpus-only TF-IDF baseline
achieves macro Recall@200 of **0.6630** on 158 development reviews and **0.6772** on 161 calibration
reviews (**0.6702** across 319 reviews). These are retrospective, previously opened splits and are
not a pristine holdout result. The exact receipt and claim boundary are in
`artifacts/benchmarks/local-suite-v1/benchmark-report.json`.

A real claim was also executed through `lm verify` against the frozen 648-paper
`antiox-training` corpus. It correctly abstains because the legacy extractor did not produce typed
effect estimates or resolved cohort identities and no real question-level calibration bundle
exists. That run validates integration and fail-closed behavior, not claim accuracy.

The full Evidence Inference 2.0 cache is available locally, but the existing 12-example GEPA pilot
is a negative pilot: its only mutation lost to the handwritten seed. MetaSyn test labels and the
Evidence Inference test labels have already been opened and are ineligible for a pristine final
claim.

## Legacy ingestion compatibility path

The original hackathon pipeline remains available as an ingestion and compatibility adapter:

```text
s1 search -> s2 screen/dedupe -> s3 extract/terminal ledger -> s4 normalize/ground
                                                        |
                    human audit + independent verification -> G3 trust/story gate
                                                        |
                     s5 paper-balanced analysis -> atomic A/B selection
                                                        |
                     optional approved s6 remap (A only)
                                                        |
                     s7 validate/stage/atomically promote
                                                        |
                     frozen Streamlit app (no network)
```

Its original release guardrails remain enforced:

- one terminal `PaperRecord` for every identity-deduplicated paper, including exclusions,
  extraction failures, and eligible zero-finding papers;
- atomic `FindingRow` records with exact authoritative line references;
- no paper crosses a train/test split, and every paper contributes unit mass within a subset;
- rare categorical levels are pooled by **distinct-paper** count, never row count;
- G3 trust and story viability are separate; trust failure blocks release;
- M4 selection is deterministic and atomic—there is no manual promotion to Variant A;
- a partial inference run is never exported unless its exact checkpoint is explicitly finalized as
  a typed incomplete Variant B;
- the legacy viewer reads only `artifacts/<question>/demo/` and validates every bundled hash at
  startup.

The s1–s7 path is no longer the terminal scientific interface. Its output must enter `lm verify`
through the processed-directory adapter or be replaced by a native typed evidence graph.

## Local setup

Requirements: Python 3.12 and `uv`. The Paperclip CLI is needed only for the legacy Paperclip live
ingestion adapter; the open harvester does not use it.

```bash
uv sync --python 3.12
uv run python --version
uv run ruff check .
uv run pytest -q -m "not live"
```

Credentials are optional for the entire fixture pipeline. For live work only:

```bash
cp .env.example .env
chmod 600 .env
```

Populate `PAPERCLIP_API_KEY` and/or `ANTHROPIC_API_KEY`. `.env` is ignored, its mode is checked
before Anthropic use, and secret values are never copied into run or provider archives.

## Deterministic fixture vertical slice

The four fixtures cover the complete narrative state space:

| Question ID | Expected release |
|---|---|
| `fixture-a` | G3 passes; planted moderator passes M4; Variant A |
| `fixture-b-story` | trust passes, G3 story viability fails; Variant B; zero residual pairs |
| `fixture-b-m4` | G3 passes, M4 completes without a winner; Variant B with residuals |
| `fixture-b-incomplete` | G3 passes, partial M4 is explicitly checkpoint-finalized; Variant B |

Generate and exercise them without network access:

```bash
uv run python scripts/generate_fixture.py --all --force
uv run pytest -q tests/test_pipeline_fixture.py tests/test_export.py tests/test_app.py
```

Individual stage CLIs all require `--question`; fixture configs additionally require the explicit
`--fixture` flag. Scientific outputs refuse overwrite unless the relevant CLI explicitly supports
`--force`.

## Paperclip-free open literature harvest

The source-agnostic s1 harvester can replace the Paperclip search step without changing s2 or the
`PaperRecord` ledger. OpenAlex supplies cross-domain search; provider-declared open URLs, Europe
PMC, and arXiv supply full text when available. Every response and local frozen-corpus input is
stored in a content-addressed archive with a SHA-256 receipt before candidates are materialized.

```bash
uv run python scripts/s1_harvest.py \
  --question selected-question --all --openalex --mailto "$OPENALEX_MAILTO"
uv run python scripts/s2_screen.py --question selected-question --force
```

For a reproducible offline replay, replace `--openalex --mailto ...` with
`--frozen-corpus path/to/corpus.json --frozen-sha256 <sha256>`. Full-text resolution is enabled by
default and can be disabled explicitly with `--no-fetch-full-text`. `--force` may regenerate the
derived candidate view and run record, but never replaces a content-addressed raw archive object.

## Verifier components

The unified verifier assigns separate jobs to separate methods:

- the optional official GEPA adapter optimizes extraction and quote-verification prompts on
  paper/group-disjoint train and development sets, then freezes a hash-locked winner before opening
  test;
- compatible estimates are represented in a typed publication -> study -> cohort -> arm ->
  contrast -> outcome-estimate graph, then harmonized and synthesized with a conservative
  random-effects boundary; literal estimate-direction synthesis is an explicit fallback;
- a question-level calibrated policy decides whether a claim can be released or whether the system
  must abstain;
- a sequential audit policy ranks evidence by declared error risk, graph-derived conclusion
  influence, and verification cost, then recomputes priorities after each adjudicated correction;
- a residual-risk guard permits partial audit only under calibrated probability upper bounds; and
- a hash-bound claim-release boundary requires synthesis, audit, and frozen calibration to pass
  together before producing a certificate.

Install the optional GEPA runtime with `uv sync --extra gepa`. Supporting entry points are
`scripts/optimize_prompts.py` and `scripts/calibrate_risk_gate.py`. Simulations remain mechanics
checks, not effectiveness evidence. See the [evidence graph](docs/evidence-graph.md),
[meta-analysis](docs/meta-analysis.md),
[budgeted verification](docs/budgeted-verification.md),
[calibration](docs/calibration.md), [prospective claim release](docs/claim-release.md), the
[closed-corpus evaluator](docs/closed-corpus-evaluation.md), the
[Evidence Inference benchmark adapter](docs/evidence-inference-benchmark.md), and the prospective
[evaluation protocol](docs/paper/neurips26-evaluation-protocol.md) for the scientific contracts.

## Legacy live-ingestion workflow

Live calls are never a prerequisite for tests. The compatibility ingestion sequence is:

```bash
QUESTION_ID=selected-question
uv run python scripts/s0_smoke_test.py --question "$QUESTION_ID" --live
uv run python scripts/s1_search.py --question "$QUESTION_ID" --all --force
uv run python scripts/s2_screen.py --question "$QUESTION_ID" --force
uv run python scripts/s3_extract.py --question "$QUESTION_ID" --target-papers 30 --force
uv run python scripts/s4_normalize.py --question "$QUESTION_ID" --force
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --sample-size 20 --seed 20260815
uv run python scripts/verify_quotes.py --question "$QUESTION_ID" --scope grounded-v1 --live
uv run python scripts/audit_findings.py --question "$QUESTION_ID" --finalize-g3
uv run python scripts/s5_analyze.py --question "$QUESTION_ID" --force
uv run python scripts/generate_baseline.py --question "$QUESTION_ID" --live
uv run python scripts/s7_export_demo.py --question "$QUESTION_ID" --candidate v1 --force
uv run python scripts/verify_demo.py --question "$QUESTION_ID" --offline
uv run streamlit run app/streamlit_app.py -- --question "$QUESTION_ID"
```

For scientific verification, pass the resulting processed directory to `lm verify`; the legacy
demo export is not the terminal release decision.

Replace the safe `selected-question` value with a validated locked config slug. G1b deliberately
blocks live maps over ten papers until the supervised failure/resume assertions pass.

Anthropic use follows the documented
[model, no-retry, and $50 global credit policy](docs/operations/anthropic-provider-policy.md).
Fixtures use an in-memory provider; transport wiring uses low-effort Sonnet; genuine independent
verification uses a higher-capability model. A failed one-shot baseline remains visibly
unavailable and never blocks the majority baseline or changes a statistic.

## Repository map

- `configs/questions/` — triage/locked question contracts
- `schemas/` — generated, topic-aware extraction schemas
- `prompts/` — versioned strict prompts
- `src/literature_multiverse/verifier.py` — the single claim-to-certificate orchestrator
- `src/literature_multiverse/certificate.py` — self-hashed JSON and static HTML certificates
- `src/literature_multiverse/meta_analysis.py` — synthesis, condition tests, and graph reruns
- `src/literature_multiverse/budgeted_verification.py` — sequential VOI scheduling and residual
  risk guards
- `src/literature_multiverse/` — remaining contracts, adapters, lineage, and legacy export
- `scripts/` — one CLI per stage and trust operation
- `data/` — raw, extracted, processed, and checkpoint state
- `artifacts/<qid>/analysis/` — s5 scientific artifacts
- `artifacts/<qid>/demo/` — exact 19-file offline release, including the non-self-hashed manifest
- `app/streamlit_app.py` — validated legacy-release viewer
- `tests/` — unit, contract, archived-parser, mutation, and fixture integration coverage

## Interpretation

The headline always describes **our retrieved corpus**. Moderator performance is predictive, not
causal. A post-hoc remap stays exploratory even if it passes its separate incremental tests. Empty
residual or evidence-gap views are valid results; the pipeline never invents a hidden variable to
make the demo more dramatic.
