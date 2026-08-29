# Reproducibility guide: verifier v2

This page documents the currently supported, provider-free verifier interfaces. The command
contracts and emitted hashes are authoritative. All commands run from the repository root.

Literature Multiverse assesses whether a **frozen corpus** supports a prespecified AI-generated
claim. It does not establish scientific truth. A successful transport, parsing, extraction, or
simulation run is not evidence that the claim verifier is accurate.

## Fast local verification

Install the locked development environment and run the same offline checks as CI:

```bash
uv sync --frozen --group dev --extra gepa
uv run ruff check .
uv run pytest -q -m "not live"
git show --check --format= HEAD
```

The shortest complete claim-to-certificate smoke test uses only embedded fixtures:

```bash
uv run lm verify \
  --fixture \
  --budget-minutes 30 \
  --output-dir /tmp/lm-v2-fixture
```

This writes `verification-certificate.json` and `verification-certificate.html`. Because the smoke
test supplies no adaptive calibration bundle, it deliberately creates no sequential audit state
and records `adaptive_calibration_required_before_audit_genesis`. It validates integration and
fail-closed behavior only. Fixture outputs cannot support an effectiveness, calibration, or
human-efficiency claim.

To inspect every supported public interface without executing a scientific run:

```bash
uv run lm --help
uv run lm fingerprint --help
uv run lm verify --help
uv run lm condition-collect --help
uv run lm condition-finalize-calibration --help
uv run lm audit-select --help
uv run lm audit-checkpoint --help
uv run lm audit-resolve --help
uv run python scripts/build_condition_calibration_trajectory.py --help
uv run python scripts/calibrate_adaptive_release.py --help
uv run python scripts/calibrate_risk_gate.py --help
uv run python scripts/calibrate_item_risk.py --help
uv run python scripts/build_native_source_manifest.py --help
uv run python scripts/s3_extract_typed.py --help
uv run python scripts/build_typed_evidence_corpus.py --help
uv run python scripts/evaluate_question_benchmark.py --help
uv run python scripts/evaluate_human_review.py --help
```

## Freeze the executable identity

Freeze the current verifier code and prompt identity before opening calibration or evaluation
labels:

```bash
uv run lm fingerprint \
  --pipeline-root . \
  --output /tmp/lm-v2/pipeline-fingerprint.json
```

The artifact contains a computed `pipeline_sha256` and the file-level component manifest. Later
commands reread and rehash those files. Supplying a hash in a claim manifest is not proof of
pipeline identity, and any covered code or prompt edit invalidates the frozen artifact.

## Verify a real claim and run the sequential audit

Start a run with an explicit computed fingerprint:

```bash
uv run lm verify \
  --claim path/to/claim.yaml \
  --corpus path/to/typed_evidence_grounding_package.json \
  --budget-minutes 60 \
  --adaptive-calibration /tmp/lm-v2/adaptive-trajectory-policy.json \
  --pipeline-fingerprint /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/verify-00
```

Every action-selecting, release-capable run must supply the adaptive bundle at state genesis.
Omitting it produces a no-selection, no-state fail-closed diagnostic. To study an uncalibrated
scheduler explicitly, pass `--analysis-only-uncalibrated-audit`; that creates a state marked
permanently analysis-only, and a later adaptive bundle cannot upgrade it. The verifier derives the
candidate internally from the entire label-free preselection ledger, exact complete corpus
membership, and recomputed policy context. Supply the exact frozen bundle when the state is
created and on every resume. Its policy
context and bundle hashes are committed at state genesis, including prefix zero. Each
selection transition appends its preselection checkpoint automatically; caller-authored prefix
files are deliberately unsupported.
`--calibration` is retained only for fixed single-decision compatibility and always abstains when
a sequential state is present. Static `--receipts` are legacy analysis-only inputs and never form
a release-capable sequential trajectory. Item-risk artifacts, when available, are passed
separately as shown below; item-level calibration cannot substitute for the complete-question
release gate.

The unified verifier and its generic v5 certificate independently reconstruct every selection
predecessor and rerun its scientific assessment. The global condition-dependent path instead
freezes an always-abstained, outcome-free v6 source and permits only an exact v6-to-v7 terminal
finalization after confirmation-aware calibration. The lower-level claim-release API intentionally
rejects adaptive authorization because it lacks those replay artifacts. The resulting unkeyed
hashes are tamper evidence, not signatures: preregister or externally timestamp the bundle and
genesis-state hashes to prevent rollback or bundle shopping outside the observed run.

Version-2 claim manifests bind exact typed condition predicates and an explicit meaningful-effect
threshold; they may release only inside prospectively prespecified typed strata. A stratum or
moderator discovered on the evaluated corpus remains exploratory and cannot authorize release.
The global effect-modification verdict is a separate manifest-v3 path using the held-out,
confirmation-aware v2 calibration contract described below. Version-1 manifests cannot carry
nonempty conditions.

When a calibrated run cannot yet release safely, it emits a state with one selected action. An
explicitly opted-in uncalibrated analysis run may do the same, but it remains release-ineligible.
A human review then resolves that action. The protocol and payload files below are external,
immutable records; do not substitute a self-authored boolean or a selected item ID:

```bash
uv run lm audit-resolve \
  --state /tmp/lm-v2/verify-00/sequential-audit-state.json \
  --claim path/to/claim.yaml \
  --disposition no_change \
  --adjudication-protocol path/to/adjudication-protocol.pdf \
  --adjudication-payload path/to/completed-adjudication.json \
  --correction-protocol path/to/correction-protocol.pdf \
  --correction-payload path/to/no-change-record.json \
  --provenance blinded_human \
  --adjudicator-count 2 \
  --realized-minutes 7.4 \
  --pipeline-fingerprint /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/audit-01
```

For `--disposition corrected`, also pass `--corrected-corpus` containing the corrected typed graph.
That file is an adjudication input, not a replacement source package: only the selected estimate
may change (or be removed), and the transition records the original and corrected graph hashes,
adjudication, correction provenance, deterministic synthesis/candidate reruns, and realized-cost
receipt. The verifier first replays the complete original item-risk scoring receipt against the
source graph, then deterministically projects its bounds only onto unchanged unresolved items.
A resolved selected item may be modified or removed; a new, changed, or removed unresolved item
fails closed rather than inheriting a stale score. For `no_change`, a corrected corpus is forbidden and the rerun must reproduce the exact
pre-action graph, synthesis, and candidates. Blinded human resolutions require at least two
adjudicators. The budget is charged from the supplied positive, measured `--realized-minutes`;
this value is total person-minutes summed across every reviewer and final adjudication, not parallel
wall-clock elapsed time. Estimated cost is used for ranking but never treated as spent human time.

If the hard budget deadline arrives before that selected review is complete, checkpoint cumulative
time on the active action instead of creating an adjudication:

```bash
uv run lm audit-checkpoint \
  --state /tmp/lm-v2/verify-00/sequential-audit-state.json \
  --active-realized-minutes 60 \
  --pipeline-fingerprint /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/audit-deadline
```

`--active-realized-minutes` is cumulative for the current action, not an increment. It must be
monotone, uses the same total-person-minute accounting, and cannot make total session spending
exceed the frozen budget. This command accepts no
adjudication, disposition, correction, or correctness label; it preserves the pre-action graph and
synthesis, leaves the action active, and therefore forces any resumed release assessment to
abstain. A completion observed after the deadline is not applied to the matched-budget result.
Every select, checkpoint, and resolve command rehashes the current supported pipeline and rejects a
state created by different code.

After a resolution, the normal production path is to resume `lm verify`. It first recomputes and
assesses the complete no-active-action state. If every scientific, audit, provenance, calibration,
and release gate passes, it stops and releases. Otherwise, if budget remains, it automatically
selects the next feasible action and persists that selection in the certificate and state.

`lm audit-select` remains available only as a lower-level analysis operator command. Adaptive
production selection must run through `lm verify` so the frozen bundle, policy context, and
prospective checkpoint are joined before selection. For an already-uncalibrated analysis state,
the standalone command requires the same explicit opt-in:

```bash
uv run lm audit-select \
  --state /tmp/lm-v2/audit-01/sequential-audit-state.json \
  --analysis-only-uncalibrated-audit \
  --pipeline-fingerprint /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/audit-02
```

or run terminal release assessment on the resolved state:

```bash
uv run lm verify \
  --claim path/to/claim.yaml \
  --corpus path/to/original-frozen-typed-evidence-grounding-package.json \
  --budget-minutes 60 \
  --adaptive-calibration /tmp/lm-v2/adaptive-trajectory-policy.json \
  --audit-state /tmp/lm-v2/audit-01/sequential-audit-state.json \
  --pipeline-fingerprint /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/verify-01
```

The resumed `--corpus` must be the original source-replayed package whose graph equals the audit
state's initial graph. The state—not an edited source package—carries the corrected current graph.
The v5 certificate embeds both graphs: corpus counts and native manifest membership validate
against the original source graph, while synthesis, counterfactuals, audit candidates, and release
validate against the corrected current graph. Its `audit_correction_replay` lineage stage binds the
ordered selection/checkpoint/correction ledger, completed correction receipts, and final state.
`post-evidence-graph.json` remains an inspection artifact and cannot independently cross the
source-provenance gate.

Do not edit a state file. Sequential state v2 replays every selection, active-cost checkpoint, and
correction from genesis. The original/current graphs, predecessor chain, action, adjudication,
correction provenance, receipt costs, synthesis, candidates, claim-manifest policy, pipeline,
budget, and cost unit are hash-bound and recomputed where possible. A stale, reordered, truncated,
or scientifically inconsistent resume fails closed.

## Complete-question release calibration

The deployed release-calibration unit is one complete independent question trajectory, never a
paper, state, row, model sample, or bootstrap draw. Each trajectory contains every preselection
state generated by a threshold-blind frozen scheduler from prefix zero through a terminal budget
or scheduler condition. Reference verdicts are stored in a separate sidecar and are not visible
to the allocation or stopping policy.

For the generic non-condition v1 path, use `fit_adaptive_development` on development trajectories
first. It fits question-weighted score models and freezes the full policy-arm/threshold family with
`freeze_state="calibration_labels_unopened"`. Only then call
`calibrate_adaptive_first_release` on a physically separate calibration set. Every frozen
arm/threshold replays each complete question from prefix zero and yields exactly one accepted/error
Bernoulli pair; abstentions and empty corpora remain denominator questions. The bundle applies
simultaneous Bonferroni one-sided Clopper--Pearson bounds across all arms and thresholds.

For a global `condition_dependent` release, use only the version-separated confirmation-aware v2
contract in [adaptive-calibration-contract.md](adaptive-calibration-contract.md). Its enforced
access order is:

1. First-pass `lm condition-collect` writes one independently collected, always-abstained,
   outcome-free single-arm source per policy arm. Then
   `scripts/build_condition_calibration_trajectory.py` externally replays every first-pass source,
   verifies one exact question/split/population/domain/corpus/source graph/target/independence
   context with unique arms, and emits the canonical multi-arm trajectory. Repeat
   `lm condition-collect` for every arm with that exact `--policy-visible-trajectory`; only these
   second-pass sources may enter calibration. The builder has no assessment, gate-result,
   reference-label, or calibration-bundle input. See the linked contract for complete commands.
2. `freeze-collection-sources-v2 --collection-sources ... --output ...` externally replays and
   freezes the complete source roster before any held-out assessment is readable.
3. `freeze-development-v2` requires `--calibration-source-roster`,
   `--expected-calibration-source-roster-sha256`, and
   `--expected-calibration-source-membership-sha256` while freezing the score models, threshold
   family, and visible calibration roster.
4. `lm condition-finalize-calibration` requires `--source-roster`, both corresponding
   `--expected-source-*` hashes, one exact `--source`, and one `--condition-assessment`; it checks
   source membership before opening the assessment and emits a linear-size replayed receipt.
5. `freeze-terminal-gates-v2` accepts only the canonically sorted full
   `--calibration-assessment-receipts` JSONL plus the exact development/source hashes. Bare gate
   results are rejected.
6. `build-calibration-bundle-v2` verifies the gate-complete receipt roster and its externally
   recorded hash before opening `--calibration-labels`.

The resulting bundle remains real-release-ineligible unless every complete question has verified
strong independence, every required receipt/source replay succeeds, and the simultaneous pooled
and per-domain condition bounds pass. Calibration collection artifacts are structurally distinct
from production v6/v7 artifacts and can never authorize production release.

The older command below produces `FrozenCalibrationBundle` v2 for one-shot analysis only. It is
not a valid adaptive production artifact:

```bash
uv run python scripts/calibrate_risk_gate.py freeze \
  --input path/to/development-calibration-risk-examples.jsonl \
  --output /tmp/lm-v2/question-risk-policy.json \
  --alpha 0.10 \
  --delta 0.05 \
  --seed 20260827 \
  --candidate-threshold 0.01 \
  --candidate-threshold 0.02 \
  --candidate-threshold 0.05 \
  --candidate-threshold 0.10
```

Record the emitted legacy bundle hash before opening a physically separate held-out test file. The
test evaluation is descriptive and does not feed back into the frozen release rule:

```bash
uv run python scripts/calibrate_risk_gate.py evaluate-test \
  --bundle /tmp/lm-v2/question-risk-policy.json \
  --expected-freeze-sha256 <recorded-64-hex-bundle-hash> \
  --input path/to/held-out-test-risk-examples.jsonl \
  --output /tmp/lm-v2/question-risk-heldout-evaluation.json
```

The adaptive bundle rejects question overlap and overlap in the complete corpus publication
membership—not merely target-matching estimates—as well as pipeline, allocation policy, budget,
stopping rule, release/audit configuration, target-semantics, corpus-protocol, population, feature,
or domain drift. Simulation labels cannot authorize scientific release. The guarantee concerns
exact decision-mismatch risk against the frozen adjudication protocol under exchangeable,
independent complete-question trajectories; it is not scientific
truth or robustness under domain shift. With locally opened labels this workflow remains a
diagnostic and requires newly adjudicated unopened questions for a real prospective claim.

## Artifact-backed item cell-rate UCLs

Raw model scores may rank audit actions as heuristics, but they cannot authorize partial-audit
release. A simultaneous group-average domain-by-score-bin error-rate UCL requires four physically
and logically separated stages:

```bash
uv run python scripts/calibrate_item_risk.py freeze-bins \
  --definition path/to/development-bin-definition.json \
  --output /tmp/lm-v2/fixed-bins.json

uv run python scripts/calibrate_item_risk.py calibrate \
  --expected-pipeline /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --fixed-bins /tmp/lm-v2/fixed-bins.json \
  --development-units path/to/development-units.jsonl \
  --calibration-units path/to/calibration-units.jsonl \
  --familywise-delta 0.05 \
  --sampling-protocol-sha256 <64-hex-protocol-hash> \
  --error-event-definition "Any adjudicated material extraction error in the item." \
  --shift-detector-id frozen-domain-monitor-v1 \
  --shift-detector-sha256 <64-hex-detector-hash> \
  --supported-domain cardiology \
  --output /tmp/lm-v2/item-risk-calibration.json

uv run python scripts/calibrate_item_risk.py assess-shift \
  --calibration-run /tmp/lm-v2/item-risk-calibration.json \
  --detector-receipt path/to/external-shift-receipt.json \
  --detector-artifact path/to/external-shift-output.json \
  --output /tmp/lm-v2/item-risk-shift.json

uv run python scripts/calibrate_item_risk.py score \
  --calibration-run /tmp/lm-v2/item-risk-calibration.json \
  --expected-pipeline /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --shift-assessment /tmp/lm-v2/item-risk-shift.json \
  --candidates path/to/prospective-item-risk-inputs.jsonl \
  --output /tmp/lm-v2/item-risk-scoring-receipt.json
```

Replace angle-bracketed hash placeholders with hashes of the frozen external artifacts. The
development and calibration JSONL files must be different regular files and contain unique
question/paper units. The prospective candidate file may not declare probability fields. The
scorer emits a scheduling-only cell-rate UCL only when pipeline, score model, population, domain,
development/calibration identity non-overlap, and external shift assessment all match. Because the
current score model is opaque rather than executable, `usable_for_release` is always false.

Pass the resulting self-contained v2 scoring receipt into `lm verify`, `lm audit-resolve`, and
every resumed run. It embeds and cross-binds the calibration run, shift assessment, exact source
candidate snapshots, computed bounds, and pipeline proof:

```bash
  --item-risk-scoring-receipt /tmp/lm-v2/item-risk-scoring-receipt.json
```

Item-cell UCLs cover the group-average frequency of the declared item-error event in a frozen
domain/score cell. They are not individual probabilities, and their adaptively selected sum is not
a residual claim-decision-risk or union bound. They do not cover retrieval omissions, publication
bias, synthesis misspecification, reviewer mistakes, or claim-level error. A separate calibration
over complete, independent review-question trajectories under the exact stopping policy is still
required for claim release.

## Native source manifests and typed extraction

Build a hash-bound source manifest from the locally archived Antiox corpus:

```bash
uv run python scripts/build_native_source_manifest.py \
  --repository-root . \
  --question-id antiox-training \
  --output-dir /tmp/lm-v2/antiox-source \
  antiox \
  --scope legacy_eligible
```

Or select explicit rows from the revision-pinned MetaSyn cache:

```bash
uv run python scripts/build_native_source_manifest.py \
  --repository-root . \
  --question-id metasyn-diagnostic \
  --output-dir /tmp/lm-v2/metasyn-source \
  metasyn \
  --corpus-ids-file path/to/selected-corpus-ids.txt
```

The Antiox `legacy_eligible` scope uses the already-opened pipeline eligibility column; it is a
tractable real-paper extraction diagnostic, not an independent screening result. Both bridges
label their outputs `diagnostic_only=true`, `labels_previously_opened=true`, and
`pristine_final_holdout_eligible=false`. A source-only record means extraction was **not assessed**;
it must not be silently counted as an extraction failure or a non-estimable study.

After a mapper has produced archived envelopes conforming to `NativePublicationExtraction`, bind
authoritative source identities and assemble the typed corpus:

```bash
uv run python scripts/s3_extract_typed.py \
  --question antiox-training \
  --map-output path/to/native-map-output.results.txt \
  --execution-receipt path/to/native-provider-execution-receipt.json \
  --source-manifest /tmp/lm-v2/antiox-source/native_source_manifest.json \
  --corpus-cutoff antiox-frozen-corpus-v1 \
  --pipeline-fingerprint-artifact /tmp/lm-v2/pipeline-fingerprint.json \
  --pipeline-root . \
  --output-dir /tmp/lm-v2/antiox-typed
```

Repeat `--execution-receipt` once per provider batch. Each self-hashed receipt binds the exact
provider/model/runtime identity and raw call ledger, and its execution ID must match the archived
map batch. Without these receipts the command writes a version-three analysis-only package.

An explicitly authorized provider map can instead use `--live --from-result <s_result_id>`
(or `--live --resume-map-id <m_map_id>`). Live mode refuses a bare hexadecimal pipeline
hash: the computed artifact must be supplied and reverified first. Merely reading this guide or
running CI never launches that provider path.

Legacy map payloads do not automatically become typed effect records. The native parser emits an
estimable graph only when effect scale, uncertainty, cohort/arm/contrast identity, and exact source
spans satisfy the contract. Terminal provider failures become explicit non-estimable fragments.
Multiple already frozen fragment files can be merged without provider calls:

```bash
uv run python scripts/build_typed_evidence_corpus.py \
  --fragments path/to/fragments-a.jsonl path/to/fragments-b.jsonl \
  --grounding-receipts path/to/receipts-a.jsonl path/to/receipts-b.jsonl \
  --source-manifest /tmp/lm-v2/antiox-source/native_source_manifest.json \
  --corpus-cutoff antiox-frozen-corpus-v1 \
  --extraction-context path/to/native_extraction_context.json \
  --output-dir /tmp/lm-v2/typed-corpus
```

The public release boundary accepts the resulting
`typed_evidence_grounding_package.json` and replays it using `--pipeline-root`. A bare
`typed_evidence_corpus.json`, `evidence_graph.json`, graph bundle, or adapted legacy findings file
can still be synthesized for diagnostics, but its certificate records
`unverified_source_provenance` as a blocking issue and cannot release a claim. The embedded fixture
also runs every scientific and release gate for integration testing, but it always records the
specific blocking issue `embedded_synthetic_fixture_not_empirical`; it can never release a
scientific claim.

The release-capable package is `typed-evidence-grounding-package-v4`. In addition to its manifest,
exact source artifact identities, one terminal fragment per manifest record, and corpus cutoff,
it binds a self-hashed extraction-context receipt. That context contains the canonical locked
question configuration, exact rendered prompt and evaluation schemas, provider/model execution
receipts and raw call ledger, map/source artifacts, and computed code fingerprint. Replay reopens
every accessible repository or artifact link. The public certificate exposes hashes and typed
nonidentifying runtime metadata, not exact private prompts, configurations, or call ledgers.

`lm verify` also checks semantics rather than trusting package labels: the claim question,
estimand, prespecified conditions, endpoint/moderator registry, inclusion/exclusion protocol, and
corpus cutoff must agree with the locked extraction configuration. A missing or incompatible field
is a blocking issue.

The manifest, context, terminal membership, and cutoff are carried into
the self-contained certificate and checked against the evidence graph and eligibility ledger.
Every terminal fragment retains its publication node, including non-estimable fragments; only
study/cohort/effect nodes depend on estimability. Thus manifest IDs, terminal-fragment IDs,
eligibility membership, and source-graph publication IDs must agree exactly, while an all-
non-estimable package may validly have zero studies and zero estimates and must abstain.
Versions one through three remain parseable for explicitly analysis-only compatibility and cannot
release. Building without an extraction context produces at most a version-three analysis package;
building without both source-membership arguments produces the older version-two analysis package.
"Complete" here means complete relative to the supplied frozen manifest; it
does not prove protocol-wide retrieval completeness, search saturation, or absence of publication
bias.

## Blinded human review

Prepare the existing single-question feasibility packet without opening system confidence to the
reviewers:

```bash
uv run python scripts/prepare_human_review_packet.py \
  --question-config configs/questions/antiox-training.yaml \
  --papers data/processed/antiox-training/papers.parquet \
  --findings data/processed/antiox-training/findings.parquet \
  --source-lines data/raw/map/antiox-training/source_lines.json \
  --output-dir /tmp/lm-v2/human-review \
  --sample-size 60 \
  --seed 20260827
```

Each reviewer completes a copy of their immutable blank template and records positive measured
minutes for every paper. Evaluate both copies independently:

```bash
uv run python scripts/evaluate_human_review.py \
  --manifest /tmp/lm-v2/human-review/manifest.json \
  --reviewer-a path/to/completed-reviewer-a.jsonl \
  --reviewer-b path/to/completed-reviewer-b.jsonl \
  --conflicts-output /tmp/lm-v2/third-adjudicator.private.jsonl \
  --output /tmp/lm-v2/human-review-summary.json
```

If reviewers disagree on any complete scientific decision, a third adjudicator must complete
exactly the emitted conflict rows before a complete result exists:

```bash
uv run python scripts/evaluate_human_review.py \
  --manifest /tmp/lm-v2/human-review/manifest.json \
  --reviewer-a path/to/completed-reviewer-a.jsonl \
  --reviewer-b path/to/completed-reviewer-b.jsonl \
  --adjudicator path/to/completed-third-adjudication.jsonl \
  --output /tmp/lm-v2/human-review-final.json \
  --require-complete
```

Until both independent ledgers are complete and every disagreement is adjudicated, no human
accuracy, timing, calibration, or release claim is licensed. Even when complete, this stratified
single-question packet is a metadata-only feasibility diagnostic, not a prevalence-weighted,
pristine question-level, or cross-domain result. Timing reports independent-reviewer and
third-adjudication person-minutes separately and their sum as `total_person_minutes`; parallel
wall-clock elapsed time is not the human-cost denominator.

## Question-level policy evaluation

The decisive evaluation unit is a complete independent claim question, not a paper or extraction
row. A benchmark JSONL must bind expert reference verdicts, realized human audit events, policy
inputs frozen outside the evaluation identities, and actual pipeline replay states for every audit
prefix that completes within budget. Missing completed-prefix states fail instead of being
approximated; a deadline-truncated active action uses the preceding state only to force abstention,
never to claim the action's correction was applied.

All questions in one evaluation must come from the same benchmark split and computed
production-pipeline identity; mixed verifier versions or prospective/retrospective designs are
rejected rather than pooled. Every replay prefix repeats that production hash, and every outcome,
policy row, paired comparison, and upper-bound row carries it forward. The
`question-policy-replay-evaluation-v7` artifact separately embeds evaluator fingerprint component
version 8. That fingerprint mechanically closes over the evaluator, both replay/evaluation CLI
entry points, every transitive in-repository Python import, the package initializer, project and
lock files, and the active Python/NumPy/Pydantic runtime identity. Thus the production system being
evaluated and the exact code computing the reported policy curves have distinct, explicit
identities; any newly imported local dependency changes the evaluator fingerprint.

Retrospective replay follows the production scheduler exactly at the budget boundary. At every
state, it ranks using only current policy inputs, skips ineligible items and items whose
**estimated** time does not fit the remaining budget, and selects the highest-priority remaining
feasible item. Only after selection does the evaluator open that item's realized duration. If the
review would finish after the hard deadline, the evaluator charges only the remaining minutes,
records the item as an active incomplete action, does not apply its adjudication or post-audit
state, and forces abstention. It never falls back to a pre-action release while that action is
active. Completed time and deadline-truncated active time are reported separately. Seeded random
priorities are stable per item and use the same deterministic construction as production; all
policies rerank after each completed correction. Per-question rows distinguish attempted/selected
items (including the one active action) from resolved items whose adjudications were actually
applied.

The default registered budgeted arms use the **production stopping rule**: before selecting another
action, replay checks the current **complete** frozen release assessment and stops at the first
release-eligible state. Complete means the synthesis, audit, calibration, and adapter/corpus gates
all pass; audit-guard eligibility alone is not a stopping condition. That check does not open a
future audit outcome, realized duration, or reference verdict. Consequently, the primary
evaluation never charges or applies post-release audit work. `no_audit` and the exhaustive upper
bound remain separate.

Every verification certificate embeds a self-hashed `ProductionStopDecision`. It contains the
exact state and full assessment evaluated before selection, the exact blocking adapter reasons,
and either the complete selected-action transition or an explicit stop/no-feasible/active-action
outcome. The certificate cross-checks that transition against its final sequential state. This
keeps the no-active prefix used by evaluation auditable even though a nonreleasable production run
normally returns with the next action already active and a postselection abstention assessment.

Evaluator component v8, which emits the `question-policy-replay-evaluation-v7` schema, does not
accept a manually entered `release_status` as a production stop decision. Generate each
completed-prefix replay row directly from a validated generic v5 or final condition-v7
certificate:

```bash
uv run python scripts/build_question_replay_state.py \
  --certificate path/to/verification-certificate.json \
  --output path/to/question-prefix-replay-state.json
```

The resulting `question-replay-state-v5` wrapper embeds the complete certificate and separately
projects and self-hashes its certificate digest, evaluated sequential state/session/transition
ledger, ordered resolved audit prefix, selected and active action, full release eligibility, exact
five-way decision/classification/reasons, graph and synthesis lineage, and policy inputs. Generic
v5 rows bind `ProductionStopDecision`; condition-v7 rows additionally bind their outcome-free v6
source and production-only terminal result. The model recomputes every projection from the
embedded certificate. Changing an outer status, blocker, prefix, graph, synthesis, policy input,
or certificate and then recomputing only the replay hash therefore still fails validation.

The production stopping rule is available only when every state in every real question is such a
certificate-bound `frozen_pipeline_rerun`. Older manually declared rows must use the explicit
`legacy_declared_pipeline_rerun` source, are ineligible for production stopping and scientific
released-claim claims, and may be used only with `allocate_to_cap_experimental`. Planted
simulations and diagnostic approximations remain distinct replay sources and cannot carry a
production binding.

An explicit `allocate_to_cap_experimental` rule is available as a controlled comparison. It keeps
auditing to the hard cap even after an intermediate state could release, is labeled non-production
in every result row, and must not be presented as the deployed verifier's behavior. Run it only by
passing `--stopping-rule allocate_to_cap_experimental`.

This is an off-policy retrospective comparison, not a randomized prospective policy trial. Real
durations are total person-minutes across all reviewers and final adjudication. The replay reuses
each item's observed adjudication and duration across policies, which declares an
item-outcome and item-cost order-invariance assumption. The evaluation artifact records that
assumption explicitly. A prospective policy comparison is still required to identify learning,
fatigue, context, or reviewer-order effects.

Run every registered policy at several realized-minute budgets:

```bash
uv run python scripts/evaluate_question_benchmark.py \
  --benchmark path/to/real-expert-question-benchmark.jsonl \
  --output /tmp/lm-v2/question-evaluation.json \
  --budget-minutes 0 \
  --budget-minutes 15 \
  --budget-minutes 30 \
  --budget-minutes 60 \
  --fixed-count 5 \
  --primary-policy risk_x_influence_per_cost \
  --bootstrap-draws 2000 \
  --bootstrap-seed 20260827
```

The evaluator reports release coverage, released-claim error, correct releases per realized human
hour, abstention utility, domain metrics, worst-domain performance, and complete-question clustered
bootstrap intervals for random, risk, disagreement, influence, risk-times-influence,
cost-normalized, fixed-count, no-audit, and audit-all policies.
The prespecified primary policy is also compared directly with every baseline at every budget
using paired, domain-stratified question resampling; the artifact reports primary-minus-baseline
point deltas and paired bootstrap intervals rather than inferring differences from two marginal
intervals.

Simulation or diagnostic input is rejected unless `--allow-non-real` is explicit. Opting in does
not upgrade the result: the artifact remains marked ineligible for a real scientific or
human-efficiency claim. The evaluator also rejects overlap in evaluation question, claim, corpus,
paper, cohort, or policy-fitting identities, as well as mixed production-pipeline hashes.

## Claim boundary checklist

Before describing any output as empirical evidence, verify all of the following:

- the pipeline and protocol were frozen before the relevant labels were opened;
- the final questions and domain were not used to fit prompts, risk scores, bins, thresholds, or
  audit policies;
- reference verdicts use at least two experts, and every reviewer disagreement received separate
  third adjudication;
- audit cost is measured realized human time, not an estimate or simulated duration;
- every selected correction triggers the actual graph, synthesis, counterfactual, and release
  rerun;
- item-level bounds and complete-question release calibration match the deployment population;
- no shift detector, required source, typed identity, or calibration gate failed; and
- the reported conclusion is literature support under the declared frozen corpus, not truth.

The locally cached MetaSyn and Evidence Inference test labels have already been opened. They remain
valuable diagnostics but are not pristine final holdouts. No simulation, fixture, opened-label
diagnostic, source-manifest bridge, or incomplete human-review packet may authorize release or a
main scientific effectiveness claim.
