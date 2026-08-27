# Prospective directional claim release

`literature_multiverse.claim_release` is the fail-closed boundary that wires the
evidence graph, synthesis, budgeted verification, and frozen risk calibration into one
prospective decision. It assesses only a prespecified `increase` or `decrease` claim.
There is deliberately no `no_effect` or equivalence target.

## Gate sequence

1. The target outcome is selected from a referentially valid `EvidenceGraph`.
2. `select_effect_evidence` checks timepoint and cohort/publication independence, and
   `synthesize_evidence_graph` performs paper-balanced quantitative synthesis or its
   sign-only fallback.
3. A quantitative direction is supported only when its confidence interval is wholly
   on the target side of zero. When configured (the default), its prediction interval
   must also exist and be wholly on that side. A directional fallback uses the exact
   binomial interval around the fraction of positive paper signs; it cannot satisfy the
   default prediction-interval requirement.
4. Audit candidates must cover the matching outcome-estimate IDs exactly. The fixed
   budget ranks and assigns candidates, but assignment never counts as resolution. The
   joined release path requires a completed `AuditResolutionReceipt` for **every**
   matching estimate. Each self-hashed receipt names `blinded_human` or
   `benchmark_adjudication` provenance, an aware completion timestamp, protocol and
   adjudication-artifact hashes, audited evidence/graph/synthesis hashes, and the current
   corrected evidence/graph/synthesis/candidate hashes. If the audited and current
   snapshots differ, a correction-lineage hash is mandatory. Partial audit always
   abstains, even when caller-supplied influence is zero.
5. A fixed, label-free feature vector combines graph completeness, synthesis
   uncertainty, and residual audit burden. A `ReleaseCandidate` is then scored by an
   already frozen `FrozenCalibrationBundle`. Missing calibration, schema/population/
   pipeline drift, split overlap, an abstain-all policy, or a score over the threshold
   prevents release. A bundle whose labels came from `simulation` is surfaced as such
   and cannot authorize this scientific release boundary.

The supplied binary `ClaimModel` must also classify the current baseline as the supported
target before the audit gate can run. This catches a reversed or disconnected baseline;
it does not prove that caller-supplied probability influence is numerically derived from
the meta-analysis. Until a graph-derived counterfactual adapter exists, requiring all
receipts is the fail-closed protection against zero or understated influence.

The final output includes hashes for the graph, synthesis, configuration, audit input,
external resolution receipts and their collection, allocation, audit guard, risk
features, calibration assessment, and the complete decision. These hashes make a
decision reproducible; they do not validate scientific truth or prove that a declared
human adjudicator was competent.

## Closed request CLI

The command accepts one JSON object and rejects unknown top-level fields, including any
attempt to pass an audit oracle or correctness label:

```bash
uv run python scripts/assess_claim_release.py \
  --input data/claim-release-request.json \
  --output artifacts/claim-release-assessment.json
```

Required top-level fields are:

- `graph`, `question_id`, `population_id`, `domain`, `pipeline_sha256`, and `target`;
- `audit_candidates`, `claim_model`, `audit_resolution_receipts`, and `audit_budget`; and
- `frozen_calibration_bundle`, which may be JSON `null` but then forces abstention.

Optional `config` and `audit_guard_config` objects freeze the declared operating point.
The target has the shape
`{"direction":"increase","outcome_name":"...","contrast_id":null}`. Audit candidate
IDs are the graph's outcome-estimate IDs, not paper titles or row positions.

This is a two-pass boundary. A first request with an empty receipt list abstains and
returns `audit.candidate_input_sha256` plus the current graph and synthesis hashes. After
external adjudication and any correction/rerun, freeze one receipt per estimate with
`freeze_audit_resolution_receipt`; the final request supplies those receipts. Bare
resolved IDs are no longer accepted. This is an intentional, backwards-incompatible
migration from the earlier ID-only declaration.

## Interpretation limits

`released` means every declared gate passed for the frozen label-risk policy and
population. It is not a guarantee of scientific truth, equivalence, causal effect,
external validity, robustness under distribution shift, or real-world calibration.
Simulation-trained bundles remain simulation evidence and are rejected by the joined
scientific release path (the lower-level calibration API may still assess simulation
candidates inside a declared simulation study). Corrections found during adjudication
must first be incorporated into the current graph, synthesis, and candidate baseline;
the receipt binds all three current hashes and requires correction lineage when they
differ from the audited snapshot.
