# Evidence graph contract

The version-1 graph replaces an ambiguous flat scientific unit with an explicit hierarchy:

```text
Publication -> Study -> Cohort -> Arm -> Contrast -> OutcomeEstimate -> EvidenceSpan
```

The implementation lives in `literature_multiverse.evidence_graph`. It is additive: existing
`PaperRecord` and `FindingRow` artifacts remain readable, but a legacy row is not automatically
treated as quantitative evidence.

## Why publication and cohort identities are separate

A publication is a document. A cohort is an independent participant or sample set. One paper can
report several cohorts, and several papers can report the same trial cohort. Treating paper IDs as
independent scientific units can therefore cause either pseudoreplication or over-collapsing.

`PublicationIdentity` records the graph publication ID plus its existing paper/document, DOI, and
PMID identities. `StudyNode` links all reports of one study. `CohortIdentity` states how the sample
identity was established:

- reported registry ID;
- reported dataset ID;
- source-reported cohort label;
- reviewer-reconciled identity; or
- an explicit legacy placeholder.

Reviewer-reconciled and placeholder identities require a rationale. A placeholder is a blocking
state at the quantitative synthesis boundary, not an ordinary missing value.

## Scientific nodes

- `ArmNode` identifies each intervention, comparison, exposure, or control group inside a cohort.
- `ContrastNode` orients two distinct arms and requires a plain-language statement of what a
  positive estimate means.
- `OutcomeEstimateNode` binds a typed `EffectEvidence` record to one contrast, one outcome, one
  explicit timepoint, risk-of-bias state, and one or more source spans.
- `EvidenceSpan` stores the publication, immutable source locator, optional quote, section/page,
  exact offsets or line IDs, and the scientific role of the passage. A locator alone is not an
  evidence span: at least one nonblank quote, ordered offset pair, or line ID is required.

`OutcomeTimepoint` distinguishes exact values, numeric ranges, verbatim reported text, and an
explicit `not_reported` state. The legacy adapter preserves text such as “post” as text; it does not
guess that it means a particular number of weeks.

`RiskOfBiasAssessment` distinguishes a real low-risk judgement from `not_assessed`. Assessments
name their tool, overall judgement, domain judgements, rationale, assessor, and supporting span
IDs. Graph validation rejects dangling span references.

## Graph validation

`EvidenceGraph` is a closed Pydantic contract and checks:

- uniqueness of node IDs, paper IDs, DOIs, and PMIDs;
- global node-ID collisions;
- publication membership for studies and effects;
- study/cohort/arm/contrast hierarchy;
- that both contrast arms are distinct and belong to the same cohort;
- that an effect's outcome and contrast label match its graph nodes;
- that every span names an existing publication, and every estimate span comes from the
  exact publication named by that estimate's `EffectEvidence.paper_id` (not merely another
  report linked to the same study);
- that every span has a quote, exact character offsets, or exact line IDs rather than only a
  document/page locator;
- that risk-of-bias spans exist; and
- that the numerical effect provenance locator is represented by an attached evidence span.

`evidence_graph_json_schema()` exports the closed Draft 2020-12 schema with the stable ID
`urn:literature-multiverse:evidence-graph:v1`.

## Conservative adapters

Both adapters require a `GraphAdapterContext`. This is intentional: `FindingRow` and
`EffectEvidence` do not contain enough information to infer cross-publication cohort identity,
arm identity, estimate orientation, or sometimes timepoint.

```python
from literature_multiverse.evidence_graph import (
    GraphAdapterContext,
    adapt_effect_evidence,
    adapt_finding_row,
)

typed_result = adapt_effect_evidence(effect, context=context)
legacy_result = adapt_finding_row(finding, context=context)
```

`adapt_effect_evidence` preserves the typed estimate, uncertainty, reported significance, and
equivalence conclusion. An unresolved cohort placeholder yields `requires_review`.

`adapt_finding_row` is deliberately non-numerical. It retains `effect_size_raw` and the old
categorical direction as legacy source statements, while emitting an unavailable/inconclusive
`EffectEvidence` record. In particular:

- `significant=False` becomes `reported_significance="not_significant"`, never estimate zero;
- `effect_direction="no_effect"` remains an ambiguous legacy label, never `exact_zero`;
- no equivalence claim is created unless an actual equivalence analysis is separately extracted;
- free-text effect sizes are not parsed without their estimand, scale, unit, and uncertainty.

The adapter result includes machine-readable blocking issues so downstream code cannot mistake a
structurally valid compatibility graph for synthesis-ready evidence.

## Quantitative boundary

`synthesize_evidence_graph()` provides the safe bridge to the current meta-analysis implementation.
It filters a graph, harmonizes the selected `EffectEvidence` records, and uses the existing
random-effects or explicit sign-only fallback path only if all of these conditions hold:

- cohort identities are resolved;
- selected timepoints are explicit and compatible (unless the caller deliberately relaxes the
  missing-timepoint check); and
- selected contrast nodes agree on both their contrast label and the plain-language meaning of a
  positive estimate; and
- the selected cohort-to-publication mapping is one-to-one.

The last requirement is conservative. The current estimator clusters by publication, so the
bridge refuses a cohort reported in several publications or a publication reporting several
cohorts. It returns a stable `status="insufficient"` reason instead of silently applying the wrong
dependence model.

## Prospective risk features

`graph_risk_features()` emits deterministic, label-free diagnostics for a future claim-release
model:

- number of estimates, publications, and cohorts;
- fraction of non-estimable effects;
- fraction missing source quotes or timepoints;
- fraction with unassessed or high/critical risk of bias; and
- fraction with unresolved cohort identity.

The synthesis wrapper exposes these under `evidence_graph.risk_features`. They are **inputs**, not
a probability of error and not a release guarantee. Their mapping to claim risk must be fitted and
calibrated on independent question-corpus units; test correctness labels must never enter feature
construction. `EvidenceGraphRiskFeatures.as_calibration_features()` returns the stable, sorted
numeric mapping expected by the prospective `ReleaseCandidate.features` boundary; the version tag
remains separate from the learned feature vector.

## Exact unresolved live-pipeline gaps

1. Production s3 still emits `FindingRow`; it does not extract typed estimates, arm-level sample
   sizes, uncertainty, equivalence analyses, or structured risk of bias.
2. No production resolver creates study/cohort identities across publications. The adapter requires
   caller-supplied identity evidence and blocks placeholders.
3. The numerical engine is publication-clustered. Multi-report cohorts and multi-cohort papers need
   a cohort-aware hierarchical dependence model; the current graph bridge refuses them.
4. There is no frozen real-review graph artifact or end-to-end closed-corpus run yet.
5. Risk-of-bias judgements and graph fields have contract tests, not human inter-rater validation.
6. The graph risk features have not been fitted or calibrated against real claim-level errors.

Those are scientific integration tasks, not schema-validation failures. The current contract makes
them explicit and prevents the legacy path from manufacturing quantitative evidence while they are
unfinished.
