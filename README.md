# Literature Multiverse

Literature Multiverse is a budget-aware verifier for AI-generated scientific claims. Given a
directional claim, proposed conditions, a frozen literature corpus, and a human-verification
budget, it determines whether that corpus supports, contradicts, qualifies, or cannot evaluate the
claim. It verifies **literature support under the declared corpus**, not scientific truth.

The terminal verifier is one fail-closed command. Its `--corpus` input is a previously built,
version-four native typed grounding package; literature acquisition, eligibility screening,
native extraction, and grounding are independently replayable upstream stages rather than hidden
work performed by this command:

```bash
lm verify \
  --claim claim.yaml \
  --corpus typed_evidence_grounding_package.json \
  --budget-minutes 60 \
  --adaptive-calibration path/to/adaptive-calibration-bundle.json \
  --output-dir artifacts/verification/run
```

For a manifest-v1/v2 claim, omitting `--adaptive-calibration` is a deliberate fail-closed
integration check: it creates no sequential audit state and selects no action. Uncalibrated action
selection requires the explicit `--analysis-only-uncalibrated-audit` opt-in, and that state can
never later become release-capable. Manifest-v3 global condition claims use the separate
confirmation-aware inputs and v6-to-v7 terminal path documented in
[held-out condition confirmation](docs/heldout-condition-confirmation.md).

It produces a self-hashed JSON certificate and a static, no-JavaScript HTML rendering containing
the corpus ledger, typed evidence graph, synthesis, graph-derived counterfactual reruns, ranked
verification actions, unresolved item-cell UCL burden, release/abstain decision, caveats, and
lineage. The item statistic is explicitly not a claim-decision-risk bound. The
certificate also freezes the complete preselection release assessment and selected-action
transition, so the rule “stop at the first full frozen release-eligible state” is independently
auditable and shared literally with the primary policy evaluation. Certificate v5 embeds the
original source-replayed graph separately from the current post-adjudication graph and binds the
ordered selection/checkpoint/correction ledger plus completed correction receipts between them. It
also carries the complete corpus identity, the self-contained item-risk scoring receipt, and the
full adaptive calibration/candidate/assessment objects needed to recompute the release gate. The
candidate is verifier-derived from append-only preselection checkpoints; callers cannot supply it,
and both the policy-context and bundle hashes are committed when the sequential state is created.
The verifier and certificate scientifically replay every predecessor checkpoint, so an adaptive
bundle cannot be activated, removed, switched, or downgraded even at prefix zero within that state.
For a manifest-v3 global condition claim, the verifier first freezes an immutable, outcome-free,
always-abstained certificate v6. Only the dedicated terminal finalizer may join its exact held-out
assessment and confirmation-aware v2 calibration lineage to create a distinct final certificate
v7; same-corpus moderator analysis cannot authorize that transition.
This is therefore one terminal verification command, not yet one raw-claim-to-literature command.

The exact fingerprint, sequential-audit, item-risk, native-extraction, human-review, and
question-benchmark commands are in the
[verifier v2 reproducibility guide](docs/reproducibility-v2.md).

## Why this exists

A conventional literature summary tends to compress heterogeneous experiments into one answer.
This project keeps paper identity, comparison, outcome, timepoint, source passage, and context
separate, then asks which recorded conditions predict the reported direction out of sample. The
design deliberately makes a complete negative result preferable to a fragile positive story.

The dated [design](docs/superpowers/specs/2026-08-15-literature-multiverse-design.md) and
[implementation plan](docs/superpowers/plans/2026-08-15-literature-multiverse-implementation.md)
document the original hackathon pipeline and are retained as historical background. They are not
normative for the current verifier. The executable schemas and validators, together with the
[verifier v2 reproducibility guide](docs/reproducibility-v2.md), define the supported contract.

## Supported verifier path

```text
AI-generated claim + protocol
        -> frozen corpus and eligibility ledger
        -> typed publication/study/cohort evidence graph
        -> cohort-aware effect synthesis and condition analysis
        -> graph-derived counterfactual audit priorities
        -> complete-question risk-controlled release or abstention
        -> hash-bound JSON/HTML verification certificate
```

Important guardrails:

- numerical synthesis accepts typed estimands, effect scales, uncertainty, cohort identities, and
  exact source provenance; legacy directional rows are retained but cannot silently become effect
  estimates;
- estimable release requires a version-four native typed grounding package replayed against
  current source bytes and bound one-to-one to its complete supplied source manifest and exact
  corpus cutoff; bare evidence graphs, legacy packages, graph bundles, and legacy findings remain
  analysis inputs but carry a blocking provenance issue;
- cross-publication study/cohort identity is a separate hash-bound replay stage: exact normalized
  registry or dataset IDs may reconcile unambiguous candidates, free-text labels never do, and a
  multi-publication claim remains blocked until a reviewer artifact partitions every original
  study and cohort;
- audit influence comes from rerunning the actual synthesis with an evidence item removed, not
  from a hand-authored additive score;
- a resumed corrected audit state is replayed from the original source graph; every predecessor,
  realized-cost receipt, correction target, synthesis, and counterfactual candidate set is checked
  before the corrected current graph may enter release assessment;
- fixed-stratum conditional claims use exact typed predicates and an explicit meaningful-effect
  threshold; the stratum and its conditions must be prespecified, while post hoc strata stay
  exploratory;
- item calibration produces simultaneous domain-by-score-bin group-average error-rate UCLs;
  their sum is an audit-triage/blocking feature, not an individual-risk union bound or release
  guarantee, and partial-audit release still requires the frozen complete-question policy;
- a global `condition_dependent` verdict requires a prospectively frozen manifest-v3 target, a
  development-only moderator fit, a confirmed held-out assessment, and a confirmation-aware v2
  complete-question calibration bundle. Same-corpus multiplicity-adjusted moderator signals remain
  exploratory, and even a confirmed result is a predictive rather than causal conclusion;
- missing calibration, unresolved cohort identity, untyped effects, distribution shift, or an
  unstable conclusion causes abstention;
- each certificate states that it assesses literature support, not scientific truth;
- source-manifest completeness is relative to the supplied frozen manifest and does not itself
  prove retrieval saturation, protocol-wide eligible-study recall, or absence of publication bias;
- audit budgets are charged from measured realized person-minutes summed across reviewers and
  adjudication; selecting an action or estimating its cost never counts as completed review; and
- if a selected review is incomplete at the hard deadline, only time actually spent is charged,
  its adjudication is not applied, and the active action forces abstention.

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

On the pinned 140,585-article MetaSyn release, reciprocal-rank fusion of fixed TF-IDF and BM25
retrievers was selected once on 158 development reviews (macro matched-subset Recall@200
**0.6649**) and evaluated once on 161 calibration reviews (**0.6827**, component-cluster bootstrap
95% interval **[0.6302, 0.7340]**; micro **0.5766**, **1,343/2,329** matched references). The paired
development difference from TF-IDF alone is small and uncertain, so this is evidence for the
retrieval baseline—not a fusion-improvement claim. These are retrospective, previously opened
splits, the matched references are not an exhaustive eligibility gold set, and official test was
not evaluated. The exact aggregate receipt and claim boundary are in
`artifacts/diagnostics/metasyn-retrieval-study-v1.json` and
`artifacts/benchmarks/local-suite-v1/benchmark-report.json`.

Within that frozen top-200 set, a protocol-aware logistic reranker selected by
component-disjoint development cross-validation retained 897 of the calibration matched
references in the first 50 documents, versus 808 for the original RRF order. Question-macro
absolute recall was **0.5232** versus **0.4761**, a paired component-bootstrap difference of
**+0.0471 [0.0242, 0.0714]**. This is a retrospective matched-reference-survival result, not
eligibility accuracy: non-matched candidates are only implicit negatives, included-study matching
is not exhaustive, all MetaSyn labels are non-pristine in this repository, and no official-test
score is reported. See [the screening study](docs/metasyn-screening-study.md).

A separate 1,120-question misspecified simulation stress-tests allocation under shared-cohort
errors, risk-score reversal, nonlinear interactions, reviewer mistakes, missing full text, and
combined domain shift. At the frozen 30-minute operating point, sequential adaptive review reduced
released-claim error versus random by **0.0876 [0.0595, 0.1170]**, but its advantage over static
adaptive review was not decisive; shifted-domain error rose to **0.3780**. This validates stress
mechanics and exposes failure modes—it is not real-world accuracy or release calibration. See
[the adaptive stress study](docs/adaptive-stress-study.md).

The metadata-only local-corpus audit inventories 648 `antiox-training` paper records, of which the
legacy pipeline labeled 19 eligible. It does **not** contain a tracked v5 certificate for a
648-paper `lm verify` run, so no public end-to-end claim is made from that inventory. The separate
native 19-publication diagnostic below is the explicitly bounded integration study.
The legacy Antiox data trees also contain tracked text-bearing and identifier-bearing material
whose redistribution basis has not been established; the strict
[public-data rights gate](docs/public-data-rights.md) therefore blocks a public repository release.

The new two-stage Antiox extraction diagnostic ran the pinned local
`qwen2.5:3b-instruct` 3.1B model over those 19 historically opened publications. After six
source-free preflight calls, it froze 19 inventory calls and 33 candidate-packet calls with zero
retries. Seven inventories were valid and below cap, ten returned no candidate, and two were
contract-invalid; all 33 candidate packets were contract-invalid. The fail-closed finalizer
therefore promoted **0 publications and 0 findings**, with no partial salvage. This is an honest
negative structured-generation result—not extraction accuracy, calibration, or claim-release
evidence. Post-freeze diagnosis found that the stored v1 generation schema was weaker than the
runtime packet contract; all 33 responses passed that JSON Schema but failed strict packet
validation. The v1 public registry fixes the exact historical lineage and aggregate self-hash instead
of reinterpreting the run under later schema code; its empirical counts were additionally confirmed
by the ignored private v1 receipt replay. The public summary self-hash is
`af6680f7f2d94ad5bea6bb59e6fcddc46e140ee8377447c96b55d7413ae788dc`; see the
[bounded native diagnostic](docs/native-bounded-ollama-diagnostic.md). The earlier one-stage
`llama3.2:1b` Antiox summary is retained as explicitly historical provenance only.

The full Evidence Inference 2.0 cache is available locally. The archived 12-example GEPA pilot is
superseded: a receipt audit recovered only 10 common development responses and excludes its trace
scalar from citation. The clean `final-v3` official-GEPA diagnostic completed 540 optimization
task calls, 8 reflection calls, and a 1,048-call paired test over all 524 examples from 191
articles. GEPA selected a changed prompt, but it did **not** improve held-out direction accuracy:
**0.3225** versus **0.3263** for the handwritten seed, a paired article-clustered difference of
**-0.0038 [-0.0249, 0.0152]**. The result is therefore `no_improvement_claim`; its aggregate
self-hash is `1039156083798863e85761ecf94b76578c74066af2ef7b7691fd4d724f4967ce`.
MetaSyn and Evidence Inference labels were historically opened, so this remains a non-pristine
local-model diagnostic rather than confirmatory evidence. See the
[official-GEPA study](docs/evidence-inference-local-ollama-gepa.md).

A separate staged EvidenceBench diagnostic adds a real 293-question, 284-paper grounding result
without publishing row data. The development-selected lexical fusion reached all-aspect
Recall@10 **0.3751 [0.3417, 0.4092]** and Results-aspect Recall@5
**0.2534 [0.2147, 0.2942]**. It beat the deterministic-random control on both metrics, but the
first-sentences control was substantially better on the broad all-aspect metric
(**0.4950 [0.4572, 0.5327]**) while much worse on Results evidence
(**0.0940 [0.0652, 0.1259]**). This mixed result exposes a position bias in broad aspect coverage
and supports only a within-paper evidence-retrieval diagnostic—not extraction, synthesis, or claim
verification. The public aggregate and exact-replay receipt are described in the
[EvidenceBench contract](docs/evidencebench-grounding-diagnostic.md).

An additional staged, offline `llama3.2:1b` diagnostic evaluates the 482 prompts on 179 test
papers absent from the complete bound local successful-provider-call registry. Prediction reads
only a frozen Results-passage projection; scoring cannot open labels until every local-model
receipt is frozen and validated. Its exact contract and non-pristine interpretation limits are in
[the local Ollama diagnostic](docs/evidence-inference-local-ollama.md).

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

The s1–s7 path is no longer the terminal scientific interface. Its output may enter `lm verify`
through the processed-directory adapter for analysis, but scientific release requires replacement
with a source-replayed native typed grounding package.

## Local setup

Requirements: Python 3.12 and `uv`. The Paperclip CLI is needed only for the legacy Paperclip live
ingestion adapter; the open harvester does not use it.

```bash
uv sync --python 3.12 --frozen --group dev --extra gepa
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
- a complete-question trajectory policy replays every frozen threshold/policy arm from audit
  prefix zero and calibrates exactly one first-release outcome per independent question; the legacy
  fixed-state bundle is analysis-only for sequential runs;
- a sequential audit policy ranks evidence by declared error risk, graph-derived conclusion
  influence, and verification cost, then recomputes priorities after each adjudicated correction;
- an item-cell UCL guard supplies conservative scheduling/blocking features without claiming an
  individual or residual decision-risk bound; and
- a hash-bound claim-release boundary requires every applicable corpus/provenance, synthesis,
  audit, complete-question calibration, and held-out-confirmation gate to pass together before a
  claim can be released in its terminal certificate.

The hashes provide replayable self-consistency and tamper evidence, not authentication or
nonrepudiation. Preventing someone from discarding a run and starting a different prefix-zero state
under another bundle requires prospective external anchoring (for example, preregistration or a
signed append-only store).

Install the optional GEPA runtime with `uv sync --extra gepa`. Supporting entry points are
`scripts/optimize_prompts.py` and `scripts/calibrate_risk_gate.py`. Simulations remain mechanics
checks, not effectiveness evidence. See the [evidence graph](docs/evidence-graph.md),
[meta-analysis](docs/meta-analysis.md),
[budgeted verification](docs/budgeted-verification.md),
[calibration](docs/calibration.md), [prospective claim release](docs/claim-release.md), the
[verifier v2 reproducibility guide](docs/reproducibility-v2.md), the
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
- `src/literature_multiverse/budgeted_verification.py` — sequential VOI scheduling and unresolved
  audit-burden guards
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
