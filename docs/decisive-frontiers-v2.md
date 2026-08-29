# Decisive evaluation frontiers v2

The v2 extension is additive: it does not alter the decisive v1 freeze, result schema, or any v1
hash. It consumes an already-scored v1 result and reports two views needed for the central
budget-aware verification hypothesis:

- policy coverage and correct releases at prespecified common **realized person-minute ceilings**;
- policy coverage and correct releases under a prespecified released-claim error ceiling.

## Current empirical status

The checked-in public readiness receipt at
`artifacts/diagnostics/decisive-claim-evaluation-v1-real-readiness-blocked.json` has
`status="blocked"` and zero development, calibration, and evaluation questions. It has no split
manifest, fit receipts, label manifest/root, or trajectory bundle, and
`real_scored_run_candidate=false`. Consequently, neither the v1 evaluator nor these v2 frontiers
currently provide a real adaptive-verification effectiveness or human-efficiency result. Passing
tests or producing a descriptive frontier from non-authoritative input validates mechanics only.

## Compile production trajectories without labels

The label-blind bridge from append-only verifier workspaces to the decisive v1 trajectory bundle is
`scripts/run_decisive_trajectory_compiler_v1.py`. First freeze one relative workspace roster entry
and one adjudication replay package for every evaluation question (plus any required standalone or
condition-v7 certificate binding), then compile and externally replay it:

```bash
uv run python scripts/run_decisive_trajectory_compiler_v1.py freeze-roster --help
uv run python scripts/run_decisive_trajectory_compiler_v1.py compile --help
uv run python scripts/run_decisive_trajectory_compiler_v1.py validate --help
```

For example, a roster entry uses `--workspace QUESTION_ID=workspaces/question`,
`--adjudication-package QUESTION_ID=adjudication/question/package.json`, and any standalone
`--certificate` or final-v7 `--condition-binding` inputs. The package enumerates safe relative paths
and exact file hashes for a frozen operator reviewer/role registry, protocol, independent reviewer
decisions, per-person timing records, final resolution, and correction payload. The compiler
replays question and item membership, reviewer roles, blind-review flags, protocol identities,
decision and resolution joins, timing intervals and their total, and every transaction-receipt
payload hash against those raw bytes. Missing packages, symlinks, unbound files, or self-declared
`blinded_human` receipts without this replay all fail closed.

The registry is deliberately an operator trust root, not an external credential system. Hash
binding proves which declared roster and workflow files were used; it does **not** cryptographically
prove reviewer identity, licensure, independence, or expertise. Compiler receipts expose that
boundary as `external_reviewer_identity_or_expertise_proven=false` and retain no scientific-claim
authority.

The resulting bundle is not accepted as real merely because its `evidence_kind` says
`real_expert_adjudicated`. Every real decisive-v1 `readiness`, `freeze`, `score`, and external
validation invocation must also provide all three of:

```text
--trajectory-compilation-result path/to/compilation-receipt.json
--trajectory-compilation-source-roster path/to/source-roster.json
--trajectory-compilation-source-root path/to/frozen-sources
```

Before readiness can become `ready`, and again before any sealed evaluation label is opened, v1
reads the exact compiler-result bytes without following symlinks, replays the compiler against the
exact roster/workspaces/adjudication packages and current compiler code identity, and requires its
embedded bundle to equal the supplied trajectory bundle. The portable freeze binds the resulting
external-replay proof. A bare or hand-authored real bundle is therefore blocked and is never a
`real_scored_run_candidate`.

The compiler holds each transactional workspace lock while replaying its artifacts, joins only
completed, raw-workflow-replayed blinded-human adjudications with positive realized
total-person-minutes, and retains the
exact union of prefixes visited by the prespecified policy roster. It rejects fixtures,
simulations, diagnostics, benchmark-replay states, uncalibrated or ungrounded certificates, missing
policy-visited prefixes, and incomplete evaluation rosters. It never accepts or opens evaluation
reference verdicts. Its bundle and receipt have no scientific-claim or claim-release authority;
they become inputs to the separately frozen v1 label-opening lifecycle.

The realized-cost view is deliberately a common-ceiling frontier. At each cutoff, it selects the
compiled point with the greatest observed spend that does not exceed the cutoff, with a frozen
nominal-budget tie break. It reports unused minutes and whether spend happens to equal the cutoff.
It never describes equal nominal deadlines as equal realized spending.

The fixed-error view reports every compiled point. An observed error rate below the ceiling is
descriptive test-set evidence only. It does not become an error-control guarantee. An authoritative
point additionally requires a real `AdaptiveCalibrationBundleV2` whose simultaneous upper risk
bound is no larger than the frozen ceiling and whose exact bundle is embedded in every terminal
production replay used by that point.

## Freeze before scoring

The checked-in default is
`configs/benchmarks/decisive-claim-evaluation-frontiers-v2.json`. A different grid must be frozen
before looking at evaluation outcomes:

```bash
uv run python scripts/run_decisive_claim_evaluation_v2.py freeze-config \
  --realized-minutes-per-question-cutoff 15 \
  --realized-minutes-per-question-cutoff 30 \
  --realized-minutes-per-question-cutoff 60 \
  --released-error-ceiling 0.05 \
  --bootstrap-draws 2000 \
  --output /tmp/decisive-frontier-config-v2.json
```

## Build and replay

```bash
uv run python scripts/run_decisive_claim_evaluation_v2.py build \
  --config /tmp/decisive-frontier-config-v2.json \
  --source-result path/to/decisive-v1-result.json \
  --repository-root . \
  --calibration path/to/exact-arm-calibration-v2.json \
  --output /tmp/decisive-frontiers-v2.json

uv run python scripts/run_decisive_claim_evaluation_v2.py validate \
  --artifact /tmp/decisive-frontiers-v2.json \
  --source-result path/to/decisive-v1-result.json \
  --repository-root . \
  --calibration path/to/exact-arm-calibration-v2.json
```

Repeat `--calibration` for distinct arm/budget points. Duplicate arm/budget calibrations are
rejected. When several budgets for one arm are calibrated, the fixed-error comparison freezes the
largest nominal budget before consulting its evaluation outcome; it never falls back to another
calibrated budget if that preselected point exceeds the observed ceiling. Omitting calibration
still yields a descriptive artifact, but all scientific frontier authority stays false.

## Fail-closed provenance

The builder rejects planted simulation, diagnostics, legacy declared replay states, real-v1
artifacts without compiler-lineage replay proof, non-expert
references, mixed adjudication protocols, unequal question rosters, mixed pipeline identities, and
tampered scored outcomes. It recomputes every outcome from the v1 frozen policy question plus its
embedded opened reference before aggregation.

A supplied calibration must be real, independently verified, expert-adjudicated, and selected
before test labels. Its development and calibration question rosters must equal the corresponding
v1 split rosters and remain disjoint from evaluation. Its policy context must use the same pipeline
and include this pre-calibration bridge in `corpus_protocol_context`:

```json
{
  "decisive_split_manifest_sha256": "<v1 split manifest SHA>",
  "decisive_identity_membership_sha256": "<v1 full identity-membership SHA>",
  "decisive_evaluation_population_membership_sha256": "<v2 source-anchor question/population SHA>"
}
```

Missing or unequal bridge fields reject the supplied calibration rather than silently joining two
populations. Even a matching bundle has no point authority unless the exact bundle SHA is embedded
in every terminal certificate for that arm/budget point.

## Uncertainty and limits

All intervals use PCG64 and paired or marginal, domain-stratified resampling of whole independent
questions. Point selection is frozen outside each bootstrap draw. The artifact reports aggregate
and worst-domain intervals for coverage, released error, correct releases per question, and correct
releases per human hour. These are resampling summaries, not causal, prospective, conformal, or
finite-sample guarantees. The artifact evaluates policies; it never authorizes release of an
individual scientific claim.
