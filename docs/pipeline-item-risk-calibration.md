# Computed pipeline identity and item-risk calibration

These modules replace two declarations with executable proof contracts:

- `pipeline_fingerprint.py` recomputes pipeline identity from explicit file bytes and
  non-secret JSON settings; and
- `item_risk_calibration.py` maps a prospective raw scheduling score to a simultaneous
  group-average domain-by-score-bin error-rate UCL when calibration and scope checks pass.

A claim manifest's `pipeline_sha256` or `error_probability` is not proof for either API.

## Pipeline identity

Build a closed list of `PipelineComponentSpec` objects. Each component identifies its
version, repository-relative files, and scientific settings. Then:

1. `compute_pipeline_fingerprint(...)` hashes each file, each component, and the complete
   pipeline.
2. Freeze the resulting `PipelineFingerprint` as the expected identity.
3. `verify_pipeline_fingerprint(...)` rereads every file and returns a self-hashed
   `PipelineFingerprintVerification` with `matched`, `mismatch`, or `unverifiable` status.
4. Calibration and prospective item scoring accept a matched verification object, not a
   bare expected hash.

Paths must be normalized and repository-relative. Symlinks, missing files, path escape,
duplicate component IDs, and assigning one file to several components fail closed. The
component manifest must enumerate every file that can affect the scientific result;
unlisted files are outside the identity by definition.

The supported verifier manifest also binds `pyproject.toml`, `uv.lock`, shared hashing,
schema, model, configuration, and parquet-reader code, plus the actually imported Python,
platform, NumPy, SciPy, scikit-learn, pandas, PyArrow, Pydantic, and PyYAML versions. Thus a
matching lockfile with a different active numerical runtime is a different pipeline rather
than silently reusing a calibration or audit state.

## Item-risk calibration

The supported method is deliberately simple and conservative:

- Risk-bin edges are frozen before calibration labels through a self-hashed
  `FixedRiskBinFamily`.
- Each calibration row has a hash-bound raw score and expert/benchmark adjudication.
- There may be at most one row per question and per paper across development and
  calibration. This prevents correlated estimates from inflating the nominal sample
  size.
- Calibration is stratified by declared domain and fixed score bin.
- Every nonempty domain/bin cell receives a one-sided exact Clopper--Pearson upper confidence
  limit for that cell's group-average error rate.
- Bonferroni divides the requested family-wise error probability across the complete
  domain-by-bin family, so all reported cell bounds hold simultaneously under the stated
  sampling assumptions.
- An empty cell produces no bound.

At deployment, `score_item_risk_bound(...)` only uses the externally supplied raw score to locate
its fixed bin. Because the current bundle contains an opaque score-model hash rather than an
executable model and evidence-feature contract, every v2 `RiskBound` is scheduling-only and sets
`usable_for_release=false`. It carries a cell-rate UCL only when the computed pipeline,
score-model hash, population, domain, split non-overlap, and bundle-bound shift assessment all
match. Otherwise it returns a self-hashed failure status and no numeric rate.

`verified_audit_cell_rate_ucl_fields(...)` is the narrow scheduling adapter. It returns
`rate_basis="calibrated_cell_rate_ucl"` only for a validated v2 result bound to the supplied
calibration bundle and current pipeline verification. The legacy
`verified_audit_probability_fields(...)` entry point now fails closed so a cell rate cannot be
promoted to an item or release probability.

## Scope of the bound

The UCL estimates the group-average frequency of the prespecified adjudicated item-error event in
the declared population, domain, and score bin. It assumes the unique calibration units are
exchangeable with the corresponding deployment cell. It is not a uniform per-item bound; after
adaptive influence/cost selection, neither an individual UCL nor a sum of cell UCLs is a
conditional residual decision-risk bound. It does not cover retrieval omissions, statistical
model misspecification, publication bias, human-adjudication errors, or latent domain
shift unless those events are explicitly included in the error definition and sampling
protocol. A separate complete-question release calibration over the exact deployed adaptive
stopping policy remains necessary for a claim-level risk statement.
