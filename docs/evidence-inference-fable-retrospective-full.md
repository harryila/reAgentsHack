# Full Fable Evidence Inference retrospective

## Result

The frozen full run compares the handwritten extraction prompt (`seed`) with the prompt selected
by the earlier GEPA study (`winner`) on all 524 Evidence Inference test questions from 191
articles. Both arms used the same `claude-fable-5` runtime and frozen article-batched interface.
The seed was better on every prespecified metric:

| Metric | Seed | GEPA-selected winner | Winner minus seed, paired article-cluster 95% interval |
|---|---:|---:|---:|
| Direction accuracy | 0.8359 | 0.7653 | -0.0706 [-0.1281, -0.0179] |
| Structured-output reliability | 0.8855 | 0.8092 | -0.0763 [-0.1335, -0.0239] |
| Formal exact-grounding reliability | 0.8721 | 0.7958 | -0.0763 [-0.1342, -0.0232] |

The operational decision is therefore to retain the handwritten seed. This result does not
authorize a confirmatory claim that GEPA is harmful or that the seed generalizes better. It is an
exploratory, retrospective cross-model and batched-interface transfer test on a benchmark whose
labels were historically opened.

## Frozen execution

- Population: 524 questions in 191 article clusters.
- Logical requests: 382, one article-batched request per arm and article.
- New provider attempts: 358, with zero retries.
- Reused immutable terminal receipts: 22.
- Inherited ambiguous failures: 2.
- Target-accounted spend: $36.218770, of which $31.702130 came from new attempts. This
  is the run's conservative accounting measure, not a provider invoice.
- Bootstrap: 20,000 paired article-cluster resamples with seed 20260829.
- Full plan SHA-256:
  `75d94201849e815561165d06467b63c03662828285d8aa6c3fd39933a4bf5864`.
- Runtime terminal SHA-256:
  `75b8d93118f0b8b26903cf11a9fea1c59b9b6c6dd82b898ff37b974c5a07a154`.
- Union plan SHA-256:
  `48d7a8e6b7002c6dd643881db168df0942f7958e3f39b86c144bd39159999246`.
- Union terminal SHA-256:
  `b9714b585c0e2d7269bdf7dae79c160277a318a6658a9486c6e55f98e80b0cbb`.

The priority-union run reused only receipts whose complete request identity matched. It preserved
ambiguous historical attempts as zero-credit terminal failures and never retried them. A new
runtime incident was likewise recorded, zero-credited, and not retried.

## Failure burden

The aggregate sidecar distinguishes transport/runtime incidents from otherwise terminal but
unusable provider responses:

| Burden | Seed | Winner | Total |
|---|---:|---:|---:|
| All forced-zero requests | 13 | 22 | 35 |
| All forced-zero locked questions | 51 | 80 | 131 |
| Runtime/transport incident requests | 1 | 2 | 3 |
| Runtime/transport incident locked questions | 15 | 16 | 31 |

The aggregate public artifacts intentionally omit article and question identities, so they do not
expose which incident-bearing requests share a cluster. The wider forced-zero difference also
includes provider responses that arrived but failed the frozen structured-response contract. No
arm-comparative failure-cause inference is made from these burden counts.

## Interpretation boundary

The paired bootstrap estimates a question-weighted rate difference while resampling article
clusters and preserving arm pairing. It does not model run-to-run generation variability, prompt
selection uncertainty, corpus or domain shift, or dependencies between related studies. The three
intervals are exploratory and have no multiplicity adjustment.

Formal exact grounding means that the returned quote and line references satisfy the frozen exact
containment contract. It is not semantic entailment, numerical effect extraction, or scientific
correctness. All retained benchmark examples are eligibility-positive, so the run does not
evaluate literature screening, retrieval, meta-analysis, calibration, human-audit efficiency, or
claim release.

Accordingly, this study supports only the engineering decision to retain the seed for this runtime
surface. It does not provide pristine-holdout, GEPA-improvement, calibration, generalization, or
claim-release authority.

## Public artifacts

- Aggregate score:
  `artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json`
  (self-hash `bf08f5292293a32ee802a07a5cb49544305dc01091683339f23cebf748413413`).
- Union and failure-burden projection:
  `artifacts/diagnostics/evidence-inference/fable-retrospective-full-union-evaluation-v2.json`
  (self-hash `4141ee56dbb51c289a3c41915d68cf099592567a90af95fef2c764de53c3b6f4`).

Both public artifacts are aggregate-only. They contain no article or question identifiers, source
text, evidence quotes, row-level predictions, or reference labels. Row-level scoring material is
kept in the private run workspace and is not a public result surface.
