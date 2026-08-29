# Adaptive first-release calibration contract

This module controls one narrow quantity: the error rate among claims released by a
frozen adaptive verification policy, where an error is an exact mismatch between the
released claim decision and a verdict produced under one frozen adjudication protocol.
It does not calibrate scientific truth, item-level extraction probabilities, or a new
domain whose question trajectories are not exchangeable with calibration.

## Statistical unit and policy

The independent unit is one complete review question, not a paper, estimate, audit
step, or intermediate scheduler state. For each question and policy arm, the label-free
trajectory contains prefix zero and every later no-active-action state through a
recomputably terminal scheduler state. A threshold is replayed over that full path and
produces exactly one outcome: the first qualifying release, or one abstention.

The terminal proof carries the complete terminal action universe, eligibility and cost
of every action, resolved prefix, remaining budget, scheduler-state identity, and
source candidate-input identity. It recomputes whether all items are resolved, the
budget is exhausted, no action is feasible, a non-confirmation context gate is
irreparably blocked, or every ordinary release gate has passed and terminal condition
confirmation may be invoked. Generic v5 and final condition-v7 certificate projections
also verify every selected action against the next resolved prefix and the exact
selection/correction transition chain.

## What is frozen before labels open

Development freezes all of the following before calibration labels are joined:

- exact pipeline, scheduler, budget, release, audit, target, and corpus-protocol context;
- development-fitted score model and feature schema for every arm;
- the finite threshold family for every arm;
- the risk limit `alpha`, familywise failure probability `delta`, multiplicity method,
  exact decision-mismatch loss, and deterministic candidate-selection rule;
- the complete label-free calibration roster, including every empty or permanently
  abstaining question, every full trajectory, and complete corpus membership.

Calibration must join exactly one question-bound reference verdict to every frozen
roster entry. Omission, substitution, duplicate questions, mixed adjudication protocols,
question or publication overlap, and any rehashed nested mutation fail closed.

## Staged artifact builder

The standalone builder makes the label-access order executable rather than relying on
operator convention. Prepare four physically separate inputs:

- `policy-contexts.json`: a JSON array of self-hashed `AdaptivePolicyContext` objects;
- `development.jsonl`: development-only, self-hashed `LabeledQuestionTrajectory` rows;
- `calibration-visible.jsonl`: calibration-only, self-hashed
  `PolicyVisibleQuestionTrajectory` rows with no reference verdicts; and
- `calibration-labels.private.jsonl`: calibration-only, self-hashed
  `QuestionReferenceVerdict` rows in sorted question-ID order.

Freeze development and the exact label-free calibration roster without passing the
calibration-label path to the process:

```bash
uv run python scripts/calibrate_adaptive_release.py freeze-development \
  --development-trajectories artifacts/development.jsonl \
  --policy-contexts artifacts/policy-contexts.json \
  --calibration-visible-trajectories artifacts/calibration-visible.jsonl \
  --alpha 0.10 \
  --delta 0.05 \
  --seed 20260827 \
  --candidate-threshold adaptive=0.05 \
  --candidate-threshold adaptive=0.10 \
  --candidate-threshold random=0.05 \
  --candidate-threshold random=0.10 \
  --output artifacts/adaptive-development-freeze.json
```

If `--candidate-threshold` is omitted, the finite candidate family is derived only
from development scores. If any explicit threshold is supplied, every frozen policy
arm must be represented. Record the printed `development_freeze_sha256` outside the
label-bearing workspace before proceeding.

Only after that hash is recorded should the calibration-label file become readable.
The second stage reparses every nested self-hash and matches the externally recorded
freeze hash before its first read of that file:

```bash
uv run python scripts/calibrate_adaptive_release.py build-calibration-bundle \
  --development-freeze artifacts/adaptive-development-freeze.json \
  --expected-development-freeze-sha256 RECORD_THE_STAGE_ONE_HASH_HERE \
  --calibration-labels private/calibration-labels.private.jsonl \
  --output private/adaptive-calibration-bundle.json
```

Each command prints a canonical, self-hashed JSON receipt. The receipt reports the
typed artifact's internal self-hash, the physical input/output file hashes, counts,
and an explicit access-order ledger; it never prints verdicts. The raw outputs remain
`AdaptiveDevelopmentFreeze` and `AdaptiveCalibrationBundle` objects, so the latter can
be passed directly to `lm verify --adaptive-calibration`.

Both stages reject test trajectories, wrong split identities, question/publication/
corpus/manifest overlap, unsorted or incomplete rosters, extra label IDs, and nested
self-hash tampering. Outputs are atomic and immutable by default. Replacing an existing
regular output requires `--force`; an output may never alias an input, and symlinked
inputs or outputs are rejected even with `--force`.

The development freeze contains development verdicts, and the final bundle contains
both development and calibration verdicts. Treat both artifacts as private. For a
stronger physical separation than process access order alone, use filesystem ACLs or a
separate mount and grant access to `calibration-labels.private.jsonl` only between the
two commands. Held-out test labels are not an input to either command.

## Confirmation-aware v2 contract

The version-2 path is separate from v1; it never filters a threshold selected under
the v1 loss and never silently downgrades a v2 input. It calibrates exact mismatch over
the five-way verdict vocabulary (`supported`, `contradicted`, `condition_dependent`,
`inconclusive`, and `not_evaluable`). A provisional `condition_dependent` state can be
released only after all of these independently replayable events occur in order:

1. the calibration-collection scheduler follows its complete threshold-blind path
   using the development graph only;
2. an outcome-free `ConditionGateInvocationProofV2` freezes the first eligible state,
   exact available-action roster, remaining budget, and unopened-outcome firewall when
   every ordinary gate passes; a typed ordinary proof instead records an already
   blocked terminal path without opening confirmation;
3. a type-distinct, always-abstained `ConditionCalibrationCollectionSourceV1` freezes
   the exact policy-visible trajectory, plan, model, graphs, pipeline, scheduler state,
   and invocation before any held-out assessment, reference verdict, or v2 bundle is
   available;
4. a `ConditionCalibrationCollectionSourceRosterV1` externally replays every source
   and freezes exact sorted question-by-arm membership plus content-silent anchors;
5. a `ConditionCalibrationAssessmentReceiptV1` embeds and externally replays that
   immutable source, validates the held-out assessment against its exact plan, model,
   and full graph only after matching the frozen source roster, then derives a
   calibration-only `ConditionCalibrationGateResultV1`;
6. the complete calibration roster embeds and replays one full receipt for every
   gate-required arm before reference verdicts are opened. Bare terminal-gate results
   and caller-authored source hashes are rejected;
7. the exact five-way joint rule and all overall/condition-domain risk bounds are
   calibrated from that frozen receipt roster and the subsequently opened references;
8. prospective production freezes a separate, bundle-bound, outcome-free v6 source
   certificate plus a post-bundle threshold qualification proof; and
9. only the dedicated v6-to-v7 finalizer may validate a prospective held-out
   assessment, derive the production-only `ConditionTerminalGateResultV2`, and
   authorize final release.

The invocation proof is deliberately independent of gate outcomes, reference verdicts,
and the eventual calibration threshold. It can stop further literature-item auditing
at the first state where every ordinary non-confirmation release gate passes, even if
feasible audit actions remain. Terminal confirmation is never a selectable audit
feature or action. The post-calibration qualification proof is distinct so calibration
labels cannot retroactively change the invocation point or action roster.

The v2 projection binds the exact global-condition target, claim/config/corpus/cutoff,
plan and materialization receipts, full/development/confirmation graphs, partition
hashes, pipeline and runner hashes, and prespecified moderator family. Its firewall
requires the online graph to equal the development graph. Confirmation outcomes are
forbidden from policy states and score features.

Question independence is checked with authority-qualified, domain-separated identity
tokens and rename-invariant connected-component hashes from the shared identity
canonicalizer. DOI/PMID, recognized or explicitly namespaced trial-registry and dataset
identifiers, and explicitly globally scoped study/cohort identifiers are eligible;
free text, graph-local labels, and ambiguous raw IDs are not. Token or component
overlap within development/calibration, across those splits, or between a prospective
question and either frozen split fails closed. Missing or ambiguous strong identity
makes v2 release ineligible, even when a statistical candidate passes.

The executable workflow uses two outcome-free collection passes before the five
access-separated calibration stages. This is required because no complete multi-arm
trajectory exists until every arm has independently reached a terminal state:

1. Run `lm condition-collect` independently for every prespecified arm without
   `--policy-visible-trajectory`. These first-pass sources must remain outcome-free and
   each must contain exactly its own arm.
2. Build the canonical family from those complete sources. The builder externally
   replays every source against the current repository fingerprint, requires one exact
   question/split/population/domain/corpus/source graph/target/independence identity,
   rejects duplicate arms, and accepts no assessment, gate-result, reference-label, or
   calibration-bundle argument:

```bash
uv run python scripts/build_condition_calibration_trajectory.py \
  --source private/first-pass/question-arm-a/condition-calibration-collection-source.json \
  --source private/first-pass/question-arm-b/condition-calibration-collection-source.json \
  --pipeline-root . \
  --output private/complete-multi-arm-visible-trajectory-v2.json
```

3. Repeat `lm condition-collect` for every arm from the same starting inputs and pass
   that exact combined file through `--policy-visible-trajectory`. Only these
   second-pass sources share the same trajectory hash and may enter the frozen source
   roster. First-pass sources are construction inputs, never roster members.

For example, repeat the following command once per arm, changing only the policy
context and arm-specific state/output paths:

```bash
uv run lm condition-collect \
  --claim path/to/claim-v3.yaml \
  --corpus private/typed-evidence-grounding-package.json \
  --budget-minutes 60 \
  --split calibration \
  --policy-context private/one-policy-context.json \
  --condition-plan private/heldout-condition-plan-v1.json \
  --condition-development-graph private/condition-development-graph.json \
  --condition-model private/heldout-condition-model-v1.json \
  --policy-visible-trajectory private/complete-multi-arm-visible-trajectory-v2.json \
  --pipeline-fingerprint private/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir private/second-pass/question-arm

# Canonically sort only the second-pass sources into the JSONL supplied below.
uv run python scripts/calibrate_adaptive_release.py freeze-collection-sources-v2 \
  --collection-sources private/calibration-collection-sources-v1.jsonl \
  --output private/calibration-collection-source-roster-v1.json

uv run python scripts/calibrate_adaptive_release.py freeze-development-v2 \
  --development-trajectories private/development-v2.jsonl \
  --policy-contexts artifacts/policy-contexts.json \
  --calibration-visible-trajectories private/calibration-visible-v2.jsonl \
  --calibration-source-roster private/calibration-collection-source-roster-v1.json \
  --expected-calibration-source-roster-sha256 RECORDED_SOURCE_ROSTER_HASH \
  --expected-calibration-source-membership-sha256 RECORDED_SOURCE_MEMBERSHIP_HASH \
  --alpha 0.10 --delta 0.05 \
  --output private/adaptive-development-freeze-v2.json

# Repeat once per frozen source, only after recording both roster hashes.
uv run lm condition-finalize-calibration \
  --source-roster private/calibration-collection-source-roster-v1.json \
  --expected-source-roster-sha256 RECORDED_SOURCE_ROSTER_HASH \
  --expected-source-membership-sha256 RECORDED_SOURCE_MEMBERSHIP_HASH \
  --source private/one-collection-source-v1.json \
  --condition-assessment private/one-held-out-assessment-v1.json \
  --output private/one-calibration-assessment-receipt-v1.json

uv run python scripts/calibrate_adaptive_release.py freeze-terminal-gates-v2 \
  --development-freeze private/adaptive-development-freeze-v2.json \
  --expected-development-freeze-sha256 RECORDED_DEVELOPMENT_FREEZE_HASH \
  --calibration-assessment-receipts private/calibration-assessment-receipts-v1.jsonl \
  --expected-source-roster-sha256 RECORDED_SOURCE_ROSTER_HASH \
  --expected-source-membership-sha256 RECORDED_SOURCE_MEMBERSHIP_HASH \
  --output private/gate-complete-calibration-roster-v2.json

uv run python scripts/calibrate_adaptive_release.py build-calibration-bundle-v2 \
  --development-freeze private/adaptive-development-freeze-v2.json \
  --expected-development-freeze-sha256 RECORDED_DEVELOPMENT_FREEZE_HASH \
  --gate-complete-roster private/gate-complete-calibration-roster-v2.json \
  --expected-gate-complete-roster-sha256 RECORDED_GATE_ROSTER_HASH \
  --calibration-labels private/calibration-labels-v2.jsonl \
  --output private/adaptive-calibration-bundle-v2.json
```

The builder output is deterministic under input reordering, is written atomically,
refuses overwrite unless `--force` is explicit, and rejects input, output, parent-path,
and pipeline-root symlinks. `--force` may replace only a regular output and never an
input alias. Any scientific-context mismatch or source replay failure aborts without
writing a trajectory.

The source-roster and development-freeze stages have no assessment, terminal-outcome,
or calibration-reference argument. The condition finalizer validates the complete
source roster and both externally recorded hashes before it opens the assessment path;
each linear-size receipt embeds only its exact source, matching content-silent anchor,
and roster/membership hashes. Collect those receipts into a canonically sorted JSONL
before the fourth command. The gate-roster stage matches the externally recorded
development and source-roster identities, externally replays every complete receipt,
and freezes the exact receipt roster while references remain unopened. It never accepts
a bare `ConditionCalibrationGateResultV1` or production
`ConditionTerminalGateResultV2`. The final calibration stage matches both externally
recorded hashes and their lineage before opening references. Each stage emits a
self-hashed, content-silent access-order receipt. A simulation-labeled,
independence-unverified, incomplete, or statistically non-passing v2 bundle is
explicitly ineligible for real scientific release.

## Guarantee

For each predeclared arm/threshold candidate, replay produces one `(accepted, error)`
pair per complete question. The implementation computes one one-sided exact
Clopper--Pearson bound over all accepted decisions and an additional bound over
confirmed `condition_dependent` releases in every domain frozen in the calibration
roster. With `C` candidates and `D` frozen domains, every bound uses
`delta / (C * (1 + D))`. Bonferroni therefore covers candidate selection across the
finite arm-by-threshold family and all candidate-specific condition/domain strata.
A candidate with zero confirmed condition release in any frozen deployment domain has
an undefined stratum bound and fails closed, even if its pooled bound passes. The
deployed candidate is chosen deterministically by maximum calibration coverage, then
lower overall upper risk, then the frozen tie breakers.

This statement requires independent, exchangeable complete-question trajectories from
the declared population and stable verdict semantics. It is invalid under calibration
roster selection after labels open, correlated questions not captured by corpus-overlap
checks, distribution shift, a changed pipeline/policy/budget, or continued auditing
after the first qualifying production prefix. Those conditions cause abstention or a
contract error where they are mechanically detectable.

Empty-corpus and always-abstaining questions remain in the coverage denominator. They
do not enter the selective-risk denominator because no claim was released for them.

## Production checks

Every repeated production assessment derives the whole observed prefix from the verifier's
replayed transition ledger. State genesis commits the policy-context and calibration-bundle hashes;
each selection then commits the exact unscored preselection state, resolved-item prefix,
realized historical cost, graph, synthesis, scheduler state, policy context, calibration bundle,
and pipeline-bound scheduler artifact. Adaptive recording cannot be activated after genesis,
removed, switched, or downgraded at prefix zero. The verifier and the generic v5 or condition-v7
certificate independently reconstruct and scientifically replay every predecessor checkpoint,
re-score every state with the frozen development model, check exact context, budget, feature,
corpus, population, domain, question, and publication lineage, and reject a prefix that continued
after an earlier qualifying release. A condition-v6 source is always abstained and outcome-free;
only its exact v7 finalization may join the held-out assessment after bundle qualification.
Simulation-calibrated bundles can never authorize a scientific release. These unkeyed hashes
establish internal replay consistency, not authentication: resisting rollback or bundle shopping
before genesis requires preregistration or authenticated append-only storage outside this process.
