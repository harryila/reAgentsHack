# Held-out condition-confirmation contract

Status: implemented release/certificate gate with synthetic contract tests. A terminal
`condition_dependent` decision can enter the public version-seven certificate only through
the staged, held-out confirmation and calibration path described below. This is still not
an empirical effectiveness result: no checked-in real corpus has yet passed that path, and
the checked-in tests use synthetic effects solely to exercise the statistical, outcome-firewall,
and lineage contracts.

## What this confirms

The protocol tests whether a categorical moderator selected under a prespecified
development family improves held-out effect-sign prediction and independently
replicates opposite-polarity subgroup effects. A passing assessment is a held-out
predictive literature association. It is not a causal moderator, a proof of scientific
truth, or evidence that the corpus is complete or transportable.

Same-corpus moderator discovery remains exploratory. It must not be rendered as
`condition_dependent`. Only a `confirmed` `ConditionConfirmationAssessmentV1` from
this held-out protocol can pass the scientific terminal gate. Confirmation alone
cannot release: the exact joint confirmation-aware v2 complete-question policy must
also pass its frozen calibration rule.

## Immutable identity bindings

`ConditionConfirmationTargetV1` binds all of the following in its self-hash:

- `claim_spec_sha256`: exactly the self-hash of the manifest-v3
  `GlobalConditionDependenceTargetV1`, including its qualified global claim semantics;
- `question_config_sha256`: the exact frozen extraction/question configuration identity;
- `corpus_snapshot_sha256`: exactly
  `CompleteCorpusIdentity.membership_sha256`, not a graph file hash or package hash;
- `corpus_cutoff`: exactly `CompleteCorpusIdentity.corpus_cutoff`;
- the harmonized measure/unit, outcome, prespecified moderator family, optional
  contrast ID, one unambiguous graph contrast label, nonempty estimand, exact meaning
  of a positive effect, and treatment/comparator arm roles.

The custodian materialization receipt, plan, model, assessment, and CLI receipts
duplicate and cross-check these bindings. The materialization receipt contains no
effect, outcome, uncertainty, moderator value, source quote, or source location. It
binds only identities, counts, canonical hashes, and access declarations. The plan
requires that receipt and separately binds the condition config, current pipeline
fingerprint, and an externally recorded freeze anchor. A string in
`external_freeze_anchor` is only a binding: a pristine scientific claim requires
publishing that commit, timestamp-authority receipt, or immutable registry entry
outside this process before model fitting.

If a claim has a non-null `contrast_id`, target roster rows must map to exactly that
graph contrast ID and one label. If it is null, all target rows must still map to exactly
one prespecified label. The target estimand, positive-direction semantics, and referenced
arm roles must also match exactly. Ambiguous or changed mappings make the plan
insufficient.

## Outcome firewall and roles

The workflow assumes physically separate access roles or storage controls:

1. After the target is externally frozen, a data custodian runs `materialize`. This is
   an explicitly outcome-accessing role: it opens the complete graph and emits the
   strict label-free roster, exact private development graph, exact private
   confirmation graph, and a content-silent self-hashed materialization receipt.
2. The planning operator receives only the roster and materialization receipt.
   `prepare` has no graph argument and no free graph-hash arguments. It independently
   reconstructs the component assignment from the roster and requires all receipt
   identities, counts, partition hashes, and graph hashes to agree. The roster permits
   identity/linkage, target membership, oriented contrast semantics, and prespecified
   categorical predictor values. It forbids availability, estimability, effect format,
   estimates, uncertainty, observed direction, significance, and source outcome text.
3. The development operator receives only the plan and exact development graph. `fit`
   has no full-graph or confirmation-graph argument.
4. The confirmation operator opens the full graph only after externally supplied plan,
   model, and current-pipeline hashes match. The script reconstructs both partitions,
   reruns the development fit, and then evaluates the confirmation partition.

Moderator values are predictors, not outcomes, but they can still reveal cohort
composition. They must be prespecified and frozen; they must not be selected after
examining confirmation effects.

## Independence and split

The roster is converted to connected components with union-find over publication,
exact paper, study, and cohort identities. Repeated normalized DOI, PMID, document ID,
study registration ID, cohort registry ID, and cohort dataset ID join components.
Titles and source labels never join components. A target cohort with a
`legacy_placeholder` identity makes the plan insufficient.

The component membership hash still records all graph-local identities for audit and
post-freeze tamper detection, but it is deliberately not the split seed. Graph-local
publication, paper, study, cohort, arm, contrast, estimate, span, and document IDs can
be renamed by an operator and therefore cannot safely determine held-out assignment.

The neutral `independence_identity` contract canonicalizes node-scoped DOI, PMID,
trial-registry, dataset, and explicitly authority-qualified global study/cohort claims.
Known namespace aliases collapse to one token. Unknown raw IDs remain conservative
join-only tokens: they may merge nodes but can never certify a split. Conflicting
authority aliases merge the affected nodes conservatively and make every affected
target component insufficient. The split identity is the shared contract's
domain-separated hash over the sorted canonical authority-token digests.

For release eligibility, every target-scope contributing publication needs a canonical
DOI or PMID. Every target component also needs at least one canonical
trial-registry/dataset/global-study/global-cohort linkage token shared across all of its
target-scope reports. Thus deleting a cohort linkage cannot turn a multi-report study
into several eligible DOI-only components, and an unscoped shared string cannot certify
independence. Authority provenance must itself be bound to the frozen source package;
this contract does not authenticate an arbitrary string merely because it appears in
an identity field.

For each authority-scoped component identity:

```text
h = SHA256(
  "literature-multiverse-condition-confirmation-v2" || NUL ||
  question_id || NUL ||
  authority_identity_set_sha256([
    canonical DOI/PMID/trial/dataset/global-study/global-cohort tokens
  ])
)
confirmation iff int(h[0:8], big-endian) mod 3 == 0
```

There is no user seed, balancing, reroll, or component movement. Every publication,
study, cohort, arm, contrast, estimate, and source-span node—including out-of-scope and
non-estimable nodes—belongs to exactly one frozen partition. Scientific support counts
only target components; non-target nodes remain in the graph and hashes.
Renaming graph-local IDs changes the membership hash and invalidates downstream
artifacts, but it cannot change the development/confirmation assignment.

## Development-only fitting

All compatible effects within one connected publication/study/cohort identity
component are conservatively reduced to one contribution. Passing component IDs into
the existing lower-level cohort aggregation API prevents multiple cohorts or reports
from masquerading as independent development observations. A component containing
conflicting levels for a prespecified moderator fails closed, and minimum support is
counted in independent components, not cohorts. Directional-only, incompatible,
missing, or unresolved target evidence also fails closed. The unconditional model and
every prespecified categorical moderator use generalized Paule–Mandel heterogeneity
and modified Knapp–Hartung covariance from the existing cohort-aware synthesis
implementation.

The development family applies the existing Bonferroni qualitative rule across all
prespecified moderator omnibus tests and level intervals. Exactly one passing moderator
is frozen: the lowest family-adjusted omnibus p-value, with moderator name as the
lexical tie-break. Its largest fitted positive level and smallest fitted negative level
are frozen, with lexical tie-breaks; both must already have adjusted development
intervals strictly beyond zero. The frozen artifact stores the unconditional mean and
variance, conditional coefficients and covariance, heterogeneity estimates, design,
reference, levels, support, and complete development diagnostics.

## Confirmation gates

For each held-out cohort with estimate `y_i`, sampling variance `v_i`, and frozen design
row `x_i`, the artifact stores:

```text
p_cond_i = Phi((x_i beta) /
               sqrt(v_i + tau_cond^2 + x_i Cov(beta) x_i))
p_uncond_i = Phi(mu_0 /
                 sqrt(v_i + tau_0^2 + Var(mu_0)))
z_i = 1[y_i > 0]
```

An exact zero, missing/conflicted moderator, unseen level, incompatible effect, or an
unrepresented target component yields `insufficient`; no row is silently dropped.

A result is `confirmed` only if all three gates pass:

1. Paired conditional-minus-unconditional Brier loss is averaged within each independent
   component and then equally across components. There must be at least 20 confirmation
   components. A deterministic 10,000-resample paired component bootstrap uses NumPy
   quantile method `higher`; its one-sided 95% upper bound must be strictly below the
   negative prespecified minimum improvement.
2. The two development-frozen polarity levels each have at least five independent test
   components and pass separate random-effects syntheses at two-sided confidence 0.975
   (Bonferroni family size two). The positive lower bound must exceed zero and the
   negative upper bound must be below zero. A component containing both levels is
   insufficient.
3. The single development-frozen moderator has confirmation-only omnibus p-value below
   0.05. There is no test-time moderator selection.

Complete but failed gates produce `not_confirmed`. Identity, support, compatibility, or
missing-data failures produce `insufficient`.

## Commands

All expected hashes below are externally recorded values, not values copied from a
command after it has already opened the next stage. The custodian command is the only
pre-confirmation command authorized to open the full graph:

```bash
.venv/bin/python scripts/confirm_condition_dependence.py materialize \
  --target work/condition/target.json \
  --expected-target-sha256 "$TARGET_SHA256" \
  --full-graph private/condition/full-graph.json \
  --expected-full-graph-sha256 "$FULL_GRAPH_SHA256" \
  --roster-output work/condition/label-free-roster.json \
  --development-graph-output private/condition/development-graph.json \
  --confirmation-graph-output private/condition/confirmation-graph.json \
  --receipt-output work/condition/materialization-receipt.json
```

An independent custodian replay must reproduce all four artifacts exactly:

```bash
.venv/bin/python scripts/confirm_condition_dependence.py validate-materialization \
  --target work/condition/target.json \
  --expected-target-sha256 "$TARGET_SHA256" \
  --full-graph private/condition/full-graph.json \
  --expected-full-graph-sha256 "$FULL_GRAPH_SHA256" \
  --roster work/condition/label-free-roster.json \
  --development-graph private/condition/development-graph.json \
  --confirmation-graph private/condition/confirmation-graph.json \
  --receipt work/condition/materialization-receipt.json \
  --expected-receipt-sha256 "$MATERIALIZATION_RECEIPT_SHA256"
```

The planning operator then uses no graph:

```bash
.venv/bin/python scripts/confirm_condition_dependence.py prepare \
  --target work/condition/target.json \
  --expected-target-sha256 "$TARGET_SHA256" \
  --config work/condition/config.json \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --roster work/condition/label-free-roster.json \
  --expected-roster-sha256 "$ROSTER_SHA256" \
  --materialization-receipt work/condition/materialization-receipt.json \
  --expected-materialization-receipt-sha256 \
    "$MATERIALIZATION_RECEIPT_SHA256" \
  --pipeline-sha256 "$PIPELINE_SHA256" \
  --external-freeze-anchor "$EXTERNAL_FREEZE_ANCHOR" \
  --output work/condition/plan.json
```

Physically withhold the full and confirmation graphs from the development environment:

```bash
.venv/bin/python scripts/confirm_condition_dependence.py fit \
  --plan work/condition/plan.json \
  --expected-plan-sha256 "$PLAN_SHA256" \
  --current-pipeline-sha256 "$PIPELINE_SHA256" \
  --development-graph private/condition/development-graph.json \
  --output work/condition/frozen-model.json
```

After externally anchoring both plan and model, run the one-shot confirmation. An exact
rerun validates and returns the existing output without rewriting it; any different or
invalid existing output fails closed.

```bash
.venv/bin/python scripts/confirm_condition_dependence.py confirm \
  --plan work/condition/plan.json \
  --expected-plan-sha256 "$PLAN_SHA256" \
  --model work/condition/frozen-model.json \
  --expected-model-sha256 "$MODEL_SHA256" \
  --current-pipeline-sha256 "$PIPELINE_SHA256" \
  --full-graph private/condition/full-graph.json \
  --output work/condition/assessment.json
```

Independent replay uses the same frozen public inputs and recomputes the split,
development model, predictions, bootstrap, subgroup syntheses, and omnibus test:

```bash
.venv/bin/python scripts/confirm_condition_dependence.py validate \
  --plan work/condition/plan.json \
  --expected-plan-sha256 "$PLAN_SHA256" \
  --model work/condition/frozen-model.json \
  --expected-model-sha256 "$MODEL_SHA256" \
  --current-pipeline-sha256 "$PIPELINE_SHA256" \
  --full-graph private/condition/full-graph.json \
  --assessment work/condition/assessment.json
```

Every command prints a canonical self-hashed receipt. The materializer writes its
self-hashed semantic receipt last, after the three data artifacts. Outputs are atomic,
exclusive regular files; output aliases, symlinks, and pre-existing outputs are
rejected. An exact confirmation rerun is the sole idempotent existing-output case.

## Integration boundary and invalidation

The implemented integration is terminal-only. Online synthesis, risk scoring, and
audit selection see the development partition and outcome-free projection, never
confirmation effects or assessment status. Calibration collection first freezes one
always-abstained source per question and policy arm, then externally replays the exact
complete source roster before any assessment path is opened. Multi-arm questions use
the two-pass trajectory builder documented in
[adaptive-calibration-contract.md](adaptive-calibration-contract.md): first collect
each arm independently, build one complete outcome-free trajectory, then rerun every
arm against that identical trajectory before freezing the roster.

Only a gate-ready calibration source may be joined to an assessment. The public
finalizer requires the full frozen source roster and both externally recorded hashes,
proves exact source and anchor membership, and only then opens the assessment:

```bash
uv run lm condition-finalize-calibration \
  --source-roster private/calibration-collection-source-roster-v1.json \
  --expected-source-roster-sha256 "$SOURCE_ROSTER_SHA256" \
  --expected-source-membership-sha256 "$SOURCE_MEMBERSHIP_SHA256" \
  --source private/one-collection-source-v1.json \
  --condition-assessment private/one-assessment-v1.json \
  --output private/one-calibration-assessment-receipt-v1.json
```

Each linear-size receipt embeds its exact source, matching content-silent anchor, both
roster hashes, replayed assessment, and derived calibration-only gate result. V2 then
freezes and replays the complete receipt roster before opening reference verdicts;
bare gate results and caller-authored source hashes are invalid. Prospective production
freezes a separate bundle-bound, outcome-free, always-abstained v6 certificate. Only
the dedicated terminal join may create v7. A scientifically confirmed assessment with
no passing confirmation-aware v2 bundle remains abstained.

Any subsequent correction, extraction-config change, corpus-membership change, claim
amendment, graph change, policy change, or pipeline-fingerprint change invalidates the
assessment and every dependent source, receipt, bundle, v6, and v7 artifact. An
assessment must never be reused across corpus or domain shift. V7 certifies a
predictive literature association under this declared protocol, not causal proof or
scientific truth.
