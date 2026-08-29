# Prospective claim release

`literature_multiverse.claim_release` is the fail-closed boundary that wires the
evidence graph, synthesis, budgeted verification, and frozen risk calibration into one
prospective decision. Version 1 assesses an unqualified, prespecified `increase` or
`decrease` claim. Version 2 adds exact typed condition predicates and an explicit
meaningful-effect threshold. There is deliberately no `no_effect` or equivalence target.
The public verifier's manifest-v3 path is different: it prospectively specifies a global
qualitative effect-modification claim and cannot classify it as `condition_dependent` from the
same-corpus moderator analysis. That verdict requires the separately frozen held-out confirmation
and confirmation-aware complete-question calibration contracts described below.

## Gate sequence

1. The target outcome is selected from a referentially valid `EvidenceGraph`. For a
   version-2 target, exact `equals`, `in`, or numeric `between` predicates are applied
   before harmonization. Missing values and type mismatches fail closed; values are never
   fuzzily coerced. A condition discovered on the evaluated evidence cannot release.
2. `select_effect_evidence` requires resolved cohort identities, compatible timepoints, and one
   contrast orientation. `synthesize_evidence_graph` then reduces compatible reports to one
   conservative contribution per explicit cohort and performs cohort-unit quantitative synthesis
   or its cohort-balanced sign-only fallback. A shared cohort across publications is one unit;
   distinct explicit cohorts reported by one publication remain distinct units.
3. A quantitative direction is supported only when its confidence interval is wholly
   on the target side of zero. When configured (the default), its prediction interval
   must also exist and be wholly on that side. A directional fallback uses the exact
   binomial interval around the fraction of positive cohort signs; it cannot satisfy the
   default prediction-interval requirement.
4. Audit candidates must cover the matching outcome-estimate IDs exactly. Selecting an
   action never counts as resolution or spent review time. The preferred v2 path uses a
   hash-chained `SequentialVerificationState`: one action is adjudicated, measured human
   minutes are charged, any correction is applied, and the actual synthesis and
   counterfactual priorities are rerun before another action can be selected. The serialized
   state carries an append-only transition ledger for selection, active-cost checkpoints, and
   corrections; production replays it from the original source graph and independently recomputes
   every corrected synthesis and candidate set before release assessment. A legacy
   v1 receipt path remains for compatibility but cannot be mixed with a sequential state.
   Item calibration supplies simultaneous domain-by-score-bin group-average error-rate UCLs.
   Their sum is a conservative audit-triage/blocking feature only: adaptive selection means it
   is neither an individual-item union bound nor residual claim-decision risk. A small sum never
   supersedes counterfactual flip or influence gates. Heuristic scores and item-cell UCLs can rank
   or block actions but cannot authorize release; complete-question calibration remains mandatory.
5. The deployed sequential path uses an `AdaptiveCalibrationBundle`, not the legacy
   fixed-state bundle. Its unit is one independent complete question trajectory. Each
   policy arm is generated once by the frozen, threshold-blind scheduler from prefix zero
   through budget exhaustion, no feasible action, or complete resolution. Reference
   verdicts are separate sidecars and never enter policy-visible features or scheduler
   state. Score models and the candidate threshold family are frozen on development only.
   Every threshold and policy arm then replays each calibration trajectory from prefix zero
   and stops at its first full release, producing exactly one accepted/error Bernoulli pair
   per question; explicit abstentions and empty corpora remain in the denominator.
   Bonferroni-corrected one-sided Clopper--Pearson bounds are simultaneous across the full
   arm-by-threshold family. Production may assess after every resolved audit only because
   each call supplies and replays the entire observed prefix and must match the exact frozen
   pipeline, allocation policy, budget, stopping rule, release/audit configuration, target
   semantics, corpus protocol, and complete publication membership. A legacy
   `FrozenCalibrationBundle` remains available for explicit one-shot analysis but fails
   closed whenever a `SequentialVerificationState` is present.
6. A manifest-v3 global condition claim adds two gates without weakening any of the preceding
   ones. Its online trajectory uses only the development partition and an outcome-free projection.
   A held-out `ConditionConfirmationAssessmentV1` may be opened only at the exact, replayed
   condition-gate invocation state. Scientific confirmation is still insufficient by itself: the
   confirmation-aware v2 bundle must calibrate the joint online-policy-plus-terminal-gate rule,
   including at least one accepted confirmed-condition calibration release and a passing
   simultaneous error bound in every frozen deployment domain.

The supplied binary `ClaimModel` must also classify the current baseline as the supported
target before the audit gate can run. This catches a reversed or disconnected baseline.
The public verifier builds influence by leave-one-item-out reruns of the actual synthesis;
item error probabilities require separate artifact-backed calibration proof.

The generic version-1/version-2 v5 output embeds the graph, synthesis, configuration, audit input,
external resolution receipts, allocation, audit guard, risk features, complete adaptive
calibration bundle, verifier-derived whole-prefix candidate, prospective assessment, policy
context, complete corpus identity, item-risk scoring receipt, and all corresponding hashes. These
artifacts make a decision reproducible; they do not validate scientific truth, authenticate an
unkeyed hash, or prove that a declared human adjudicator was competent.

The unified v5 verification certificate makes source/current semantics explicit. It embeds the
original source-replayed graph and the current post-adjudication graph separately. Corpus
membership and provenance validate against the source graph; scientific synthesis, audit
candidates, and release validate against the current graph. Its `audit_correction_replay` lineage
stage binds the ordered transition ledger and completed correction receipts. When no correction
exists, source and current graphs must be identical.

Manifest v3 uses type-distinct terminal artifacts. The first production artifact is an immutable,
outcome-free, always-abstained `ConditionVerificationCertificateV6`; it binds the exact scheduler
state, condition projection, plan, development-only model, and confirmation-aware v2 bundle before
the held-out assessment is opened. The dedicated finalizer joins that exact v6 to the held-out
assessment, terminal gate result, and prospective v2 qualification. Only the resulting certificate
v7 can release. V6 is never a final release certificate, and a confirmed assessment without a
passing v2 bundle remains abstained. Calibration-data collection uses separate always-abstained
source/receipt types and cannot be substituted for either production artifact. See
[held-out condition confirmation](heldout-condition-confirmation.md) and the
[adaptive calibration contract](adaptive-calibration-contract.md).

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

This is the legacy two-pass boundary. A first request with an empty receipt list abstains
and returns `audit.candidate_input_sha256` plus the current graph and synthesis hashes. After
external adjudication and any correction/rerun, freeze one receipt per estimate with
`freeze_audit_resolution_receipt`; the final request supplies those receipts. Bare
resolved IDs are no longer accepted. This is an intentional, backwards-incompatible
migration from the earlier ID-only declaration.

For new work, use the public `lm verify` and `lm audit-resolve` sequence documented in the
[verifier v2 reproducibility guide](reproducibility-v2.md). A resumed no-active state is assessed
against every release gate first and automatically selects the next feasible action only if it
cannot release. `lm audit-select` is a lower-level non-adaptive diagnostic interface; it cannot
authorize adaptive release because it does not own the full verifier replay context.

Stateful risk-controlled release requires the frozen adaptive bundle:

```bash
lm verify ... \
  --adaptive-calibration path/to/adaptive-question-trajectory-freeze.json
```

The verifier, not the caller, derives the prospective whole-prefix candidate from the exact
preselection checkpoints in its replayed sequential transition ledger. Adaptive calibration must
be supplied when the sequential state is created, before any selection. The same policy-context and
bundle hashes remain bound at prefix zero and every later selection; calibration cannot be
activated, removed, switched, or silently downgraded within that state. Both the verifier and the
certificate independently reconstruct every selection predecessor and rerun its synthesis,
counterfactual candidates, non-calibration release gates, and risk-feature projection. The public
low-level claim-release function therefore rejects adaptive release: only the unified verifier has
the complete artifacts needed to authorize it.

`--calibration` now denotes the legacy single-decision compatibility bundle and can never
authorize a sequential release.

The global manifest-v3 path uses explicit, version-separated inputs:

```bash
lm verify \
  --claim path/to/claim-v3.yaml \
  --corpus path/to/typed-evidence-grounding-package-v4.json \
  --budget-minutes 60 \
  --condition-adaptive-calibration path/to/adaptive-calibration-bundle-v2.json \
  --condition-plan path/to/condition-plan.json \
  --condition-development-graph path/to/development-graph.json \
  --condition-model path/to/frozen-condition-model.json \
  --condition-assessment path/to/heldout-assessment.json \
  --output-dir path/to/final-verification
```

The assessment path is not read unless the outcome-free scheduler proves the exact gate-invocation
state. Omitting it leaves the source v6 abstained. Passing it at a gate-ready state performs the
dedicated immutable-v6-to-v7 join; it does not rerun or mutate the online trajectory.

## Interpretation limits

`released` means every declared gate passed for the frozen trajectory policy and
population. Its exact finite-sample target is binary disagreement with the supplied
reference verdict under exchangeable independent complete-question trajectories. It is
not a guarantee of scientific truth, equivalence, causal effect, external validity,
robustness under distribution shift, or real-world calibration.
For a manifest-v3 `condition_dependent` release, the confirmation-aware v2 contract additionally
requires a simultaneous error bound for confirmed condition releases within every frozen
deployment domain. A generic v1 bundle cannot authorize that release.
Observed-domain matching is a conservative shift guard, not proof of exchangeability within a
named domain.
The self-hashes are not signatures. They detect inconsistency inside an observed lineage but cannot
prove when it was created, prevent rollback to an unobserved state, or prevent bundle shopping
before a run is externally anchored. A real prospective deployment must preregister the bundle and
genesis state or place them in authenticated append-only storage before labels or audit outcomes
are opened.
Simulation-trained bundles remain simulation evidence and are rejected by the joined
scientific release path (the lower-level calibration API may still assess simulation
candidates inside a declared simulation study). Corrections found during adjudication
must first be incorporated into the current graph, synthesis, and candidate baseline.
Sequential v2 receipts additionally bind realized cost and the complete prior state hash.
A selected or active action is never equivalent to a completed review. If measured work reaches
the hard budget deadline before adjudication is complete, the session checkpoints that partial
time as `active_realized_cost`, leaves the graph and synthesis unchanged, and must abstain with
`active_audit_action_unresolved`; neither the eventual adjudication nor its eventual full duration
may be applied retrospectively at that deadline.
