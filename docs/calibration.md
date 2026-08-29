# Question-level release calibration

This page documents the legacy one-shot question-risk primitive and its physically separated
freeze/evaluate workflow. It remains useful for mechanism checks and diagnostics, but it cannot
authorize a stateful sequential release. The deployed generic verifier instead uses the exact
complete-trajectory contract in
[adaptive-calibration-contract.md](adaptive-calibration-contract.md); a global manifest-v3
`condition_dependent` verdict requires that contract's confirmation-aware v2 path.

In every question-level contract, one complete question--corpus run is one calibration unit.
Papers, findings, bootstrap draws, and model samples are features or dependent observations; they
are not counted as independent calibration examples.

## Contract

Each `RiskExample` records a development, calibration, or test question; the complete sorted paper
set; one frozen pipeline hash; one population; a fixed numeric feature schema; and a binary loss.
The loss can mean disagreement with a frozen benchmark annotation or blinded expert adjudication.
Simulation labels are accepted only in a separate simulation population. In all cases, this is a
label-risk target, not a claim to know scientific truth.

The split-integrity checks used by fitting, calibration, and evaluation refuse:

- repeated question IDs;
- any paper shared across question units, including within one split;
- pipeline, population, feature-schema, or label-source drift across splits; and
- a missing partition required by the active stage.

Development data fit a standardized logistic score. Calibration data select the highest-coverage
candidate threshold whose one-sided exact Clopper--Pearson risk bound is at most `alpha`, using a
Bonferroni correction over the complete candidate family. The family should be prespecified; when
none is supplied, the implementation derives it from development scores, never from calibration
scores. If no candidate is certified, the policy abstains on every question. A held-out test split
is opened only after the model and policy are frozen, and only to produce a descriptive final
risk/coverage estimate and per-domain summary.

The calibration code exposes both a two-stage empirical workflow and a label-free prospective
deployment boundary. `ReleaseCandidate` deliberately contains no correctness label;
`assess_release_candidate` checks pipeline, population, feature-schema, question, and paper
identity against a frozen bundle before returning release or abstention. The production s3--s7
workflow still does not construct complete `RiskExample`/`ReleaseCandidate` records, so this API
is not itself an end-to-end calibrated production result. The checked artifact below validates
simulated mechanics only.

Every test or prospective scoring boundary first serializes and revalidates a fresh bundle
snapshot. This is necessary because nested Python lists can otherwise be mutated in place after
Pydantic assignment validation. A changed coefficient, threshold family, model hash, policy hash,
or bundle hash now fails with `frozen_bundle_integrity_changed`; downstream scoring uses the newly
validated copy rather than the caller's mutable object.

The old one-file CLI was not an adequate operational boundary: although the fitting functions used
only the intended rows, it deserialized test labels before freezing the decision rule. The empirical
workflow now requires two physical inputs and two commands. The freeze bundle contains no test
rows, test counts, or test labels. It does contain sorted development/calibration question and paper
identity ledgers so the later command can reject cross-stage overlap.

## Empirical two-stage commands

First create a JSONL containing **development and calibration rows only**. Test rows are forbidden
even if code outside this script has already exposed their labels. Fit and calibrate, then freeze a
self-hashed bundle:

```bash
uv run python scripts/calibrate_risk_gate.py \
  freeze \
  --input path/to/development-calibration-risk-examples.jsonl \
  --output artifacts/paper/real-risk-policy-freeze.json \
  --alpha 0.10 --delta 0.05 --seed 20260826 \
  --candidate-threshold 0.01 \
  --candidate-threshold 0.02 \
  --candidate-threshold 0.03 \
  --candidate-threshold 0.05 \
  --candidate-threshold 0.08 \
  --candidate-threshold 0.10 \
  --candidate-threshold 0.15 \
  --candidate-threshold 0.20 \
  --candidate-threshold 0.30
```

Record the printed `bundle_sha256` in the run ledger before opening test labels. In a separate
process, use a JSONL containing **test rows only**. The command validates the bundle's internal
hash first, optionally compares it with the externally recorded hash, and only then opens the test
file:

```bash
uv run python scripts/calibrate_risk_gate.py \
  evaluate-test \
  --bundle artifacts/paper/real-risk-policy-freeze.json \
  --expected-freeze-sha256 <recorded-64-lowercase-hex-hash> \
  --input path/to/held-out-test-risk-examples.jsonl \
  --output artifacts/paper/real-risk-heldout-evaluation.json
```

The test stage refuses model/policy/bundle hash tampering; development/calibration question or
paper overlap; non-test rows; and pipeline, population, label-source, or feature-schema drift. The
test artifact binds its descriptive evaluation and risk--coverage curve to the prior
`bundle_sha256`. A content hash cannot prove that a human never viewed test labels, so the run
ledger and access controls remain part of the scientific protocol.

For backward-compatible mechanism checks, both the combined CLI and its artifact builder are
explicitly diagnostic and restricted to `label_source="simulation"`:

```bash
uv run python scripts/calibrate_risk_gate.py \
  diagnostic-one-shot \
  --input path/to/simulation-risk-examples.jsonl \
  --output artifacts/paper/simulation-one-shot-diagnostic.json
```

It must not be used for an empirical held-out result.

## Prospective release after freeze

After a real development/calibration bundle has been frozen, deployment code should construct an
unlabelled `ReleaseCandidate` and call `assess_release_candidate`. The function refuses any
question or paper used during fitting/calibration, any pipeline or population change, and any
feature-schema mismatch. It never accepts a correctness label, so release cannot depend on an
opened test or deployment outcome. A released result remains covered only by the frozen
label-risk policy under its declared assumptions; it is not certified as scientific truth.
The generic calibration boundary can be used inside a declared simulation study. The joined
scientific `claim_release` boundary is stricter and refuses `label_source="simulation"` as release
authority.

The same boundary is available without importing Python:

```bash
uv run python scripts/calibrate_risk_gate.py \
  assess-release \
  --bundle artifacts/paper/real-risk-policy-freeze.json \
  --expected-freeze-sha256 <recorded-64-lowercase-hex-hash> \
  --input path/to/unlabelled-release-candidate.json \
  --output artifacts/paper/prospective-release-assessment.json
```

## Planted mechanism validation

Validate the mechanism on planted independent question-corpora:

```bash
uv run python scripts/simulate_risk_calibration.py \
  --output artifacts/paper/calibration-simulation-100.json \
  --replicates 100 --seed 20260826 \
  --alpha 0.10 --delta 0.05 \
  --development-count 400 --calibration-count 2000 --test-count 2000 \
  --force
```

The committed 100-replicate planted study reports mean coverage 0.252 and mean planted selective
risk 0.062 for the calibrated policy; 0 of 87 nonempty calibrated replicates exceeded the 0.10
planted-risk target. The fixed five-paper gate had mean coverage 0.994, mean planted risk 0.177,
and exceeded the target in all 100 replicates. These results validate the calibration mechanics and
show why paper count alone is inadequate. They do not establish a real-world guarantee.
