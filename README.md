# Literature Multiverse

Literature Multiverse is a provenance-first pipeline for a deceptively hard question: when papers
appear to disagree, is there a reproducible conditional pattern—or is the honest result that the
available corpus does not explain the disagreement?

It retrieves and reconciles papers, extracts atomic source-grounded findings, measures directional
disagreement with paper-balanced statistics, tests a pre-registered moderator family, and freezes
either:

- **Variant A:** one moderator passed every inference, stability, support, sensitivity, and
  materiality rule; or
- **Variant B:** trust passed, but no controlled conditional-pattern claim is warranted, so the
  release shows residual contradictions and exact evidence gaps instead.

The app is an offline viewer, not an arbitrary-question web service. Every displayed number and
quote comes from a frozen, hash-validated release.

## Why this exists

A conventional literature summary tends to compress heterogeneous experiments into one answer.
This project keeps paper identity, comparison, outcome, timepoint, source passage, and context
separate, then asks which recorded conditions predict the reported direction out of sample. The
design deliberately makes a complete negative result preferable to a fragile positive story.

The normative contracts are the
[design](docs/superpowers/specs/2026-08-15-literature-multiverse-design.md) and
[implementation plan](docs/superpowers/plans/2026-08-15-literature-multiverse-implementation.md).
The much larger root context is historical background, not an executable specification.

## Pipeline

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

Important guardrails:

- one terminal `PaperRecord` for every identity-deduplicated paper, including exclusions,
  extraction failures, and eligible zero-finding papers;
- atomic `FindingRow` records with exact authoritative line references;
- no paper crosses a train/test split, and every paper contributes unit mass within a subset;
- rare categorical levels are pooled by **distinct-paper** count, never row count;
- G3 trust and story viability are separate; trust failure blocks release;
- M4 selection is deterministic and atomic—there is no manual promotion to Variant A;
- a partial inference run is never exported unless its exact checkpoint is explicitly finalized as
  a typed incomplete Variant B;
- the final app reads only `artifacts/<question>/demo/` and validates every bundled hash at startup.

## Local setup

Requirements: Python 3.12, `uv`, and the Paperclip CLI for explicit live ingestion.

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

## Real-data workflow

Live calls are never a prerequisite for tests. The intended production sequence is:

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
- `src/literature_multiverse/` — contracts, ingestion, science, lineage, and export
- `scripts/` — one CLI per stage and trust operation
- `data/` — raw, extracted, processed, and checkpoint state
- `artifacts/<qid>/analysis/` — s5 scientific artifacts
- `artifacts/<qid>/demo/` — exact 19-file offline release, including the non-self-hashed manifest
- `app/streamlit_app.py` — validated offline viewer
- `tests/` — unit, contract, archived-parser, mutation, and fixture integration coverage

## Interpretation

The headline always describes **our retrieved corpus**. Moderator performance is predictive, not
causal. A post-hoc remap stays exploratory even if it passes its separate incremental tests. Empty
residual or evidence-gap views are valid results; the pipeline never invents a hidden variable to
make the demo more dramatic.
