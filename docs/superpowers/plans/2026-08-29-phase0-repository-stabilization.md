# Phase 0 — Public Repository Stabilization Implementation Plan (rev 2, post adversarial review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public repository lint-clean, make every public validator and the complete non-live test suite pass on a checkout without private `data/cache` files, and produce a clean-public-repository plan — without adding architecture, contract versions, provider calls, or fabricated evidence.

**Architecture:** Phase 0 changes only (a) the public-artifact registry's *interpretation* of already-frozen artifacts (a historical pin instead of current-tree recomputation for the one artifact whose closure drifted; registration of every tracked diagnostic through its existing producer model), (b) the rights policy's coverage, (c) test-suite partitioning (private-cache tests self-skip; historical-bundle staleness is pinned by tests that prove the divergence is identity-only; fixtures whose outputs are byte-frozen in tracked artifacts load those artifacts instead), and (d) small hygiene fixes. No artifact bytes are rewritten, no hash is edited to fake compatibility, no historical workspace is regenerated.

**Tech Stack:** Python 3.12.11, uv (offline from local cache), pytest (`-m "not live"`, `--strict-markers`), ruff 0.16.3 (`py312`, `UP` selected), pydantic v2 contracts, the repository's own validators (`scripts/validate_public_artifacts.py`, `scripts/audit_public_data_rights.py`).

**Spec:** the operator's Phase 0 instructions (chat, 2026-08-29) plus `docs/claudecode-handoff-2026-08-29.md` §"Non-negotiable operator constraints" and §P1. Revision brief: adversarial review 2026-08-29 (7 lenses, verdict-adjudicated; edits E1–E24 folded in below).

## Global Constraints

- The worker never runs any `git` or `gh` command. The project's own read-only git inventory subprocesses are unchanged and may execute (`scripts/audit_public_data_rights.py` → `ls-files --stage`, `diff-files`, `cat-file blob` via `public_data_rights.py:61-69`; `tests/test_manuscript_boundary.py:39-66` → `ls-files`, `check-ignore`). The report lists them under "git invocations (read-only, by project code)".
- Because git is forbidden, every **new** file this plan creates stays in the working tree, unindexed (the companion manifest in Task 2, `tests/private_cache_support.py`, `docs/public-repository-cleanup-plan.md`, this plan). Staging is the operator's action and is listed under remaining blockers. Consequences are made visible, never hidden (Task 5 Step 4b).
- Do not touch `paper/`, `docs/paper/**`, `Formatting_Instructions_For_NeurIPS_2026 (2)/`, `artifacts/submission/`, or the GitHub Support purge material. `docs/paper` and `artifacts/paper` are *evaluation assets*, are tracked, and must remain present in every copy.
- Do not modify, retry, regenerate, or relabel any historical hosted v3/v4/v5 workspace, any `data/cache/**` file, or any tracked artifact's bytes. The only new tracked-intended artifact is a byte copy of existing provenance metadata (Task 2) — never model outputs, labels, or source text.
- Task 1 and Task 5 edit files that are bound by the handoff's authoritative v2 composition receipt (`data/cache/hosted-native-numeric-yield-pilot-v5-composition-v2/external-replay-receipt.json`: `question_evaluation.py` is already stale against it; `pyproject.toml` becomes stale with the marker edit). Do **not** write a v3 receipt, and do not run `lm fingerprint`/`lm verify` against the hosted-v5 artifacts in Phase 0; that is P1 work after the last source change.
- Do not fabricate human labels, reviewer identity, timings, adjudication, or provenance. Do not print or persist secrets. No provider/API calls; no network beyond none (`uv sync` runs `--offline`). Project spend stays at ≈$61.14 accounted.
- Introduce no new contract version. Registry entries, validator kinds, a pytest marker, and rights collections are not contract versions.
- Never edit a hash to fake compatibility. The item-risk pin `2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15` stays literally in code and is *reinterpreted* as a historical pin (Task 2 states plainly that this withdraws the artifact's former recompute-and-match guarantee).
- Every task ends with its named offline commands actually run and their output recorded for the Phase 0 report.

---

## Baseline (measured 2026-08-29, HEAD `288f8b7`, dirty working tree)

| Gate | State | Root cause |
|---|---|---|
| `uv run ruff check .` | 3 × F401 | unused `ConditionVerificationCertificateV8`, `FinalConditionVerificationCertificateV9`, `VerificationCertificateV8` at `src/literature_multiverse/question_evaluation.py:31-36` |
| `scripts/validate_public_artifacts.py` | exit 1 | `_validate_evidence_inference_item_risk` (`public_artifacts.py:1206-1286`) requires the *current-tree* diagnostic fingerprint to equal the frozen pin `2949fde1…`; the 23-file closure includes `pipeline_fingerprint.py`, changed 2026-08-29 → current `f1510d0d…`. Fail-fast at registry index 4; entries 5–22 never run in CI. |
| `scripts/audit_public_data_rights.py` | exit 2 (`policy_complete=false`) | 3 undeclared monitored files: `prompts/hosted_native_numeric_yield_pilot_v{1,4,5}.txt` (deny-by-default `.txt`, no `prompts/` collection) |
| `pytest -m "not live"` (full, 43 min, older tree) | 7 failed / 137 errors / 1823 passed | errors: 118 × `metasyn_pilot_prepare_external_replay_mismatch` (`metasyn_typed_pilot.py:1676`) + 19 × `metasyn_hosted_adapter_upstream_stale` (`metasyn_bounded_hosted_runtime.py:796`); failures: item-risk ×2, `test_manuscript_boundary` (4 deleted `artifacts/paper/evidence-inference*` entries), `test_public_artifacts` ×2 (18 tracked diagnostics unregistered; CLI red), `test_public_data_rights` ×2 (3 undeclared; stale local-suite expectation) |
| Clean checkout (no `data/cache`) | additionally broken | 37 test files read `data/cache/**`; many fixtures raise `FileNotFoundError`/wrapped open errors in CI |

---

## Design decisions (operator approval 2026-08-29: D1 demote; D3 Option A; EI rosters `redistribution_not_established`; E16 hosted-runtime restore DECLINED — record the 16 tests as recoverable; execution subagent-driven)

### D1 — Item-risk public artifact: demote to a historical pin (recommended) rather than reproduce

*Chosen:* keep `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1.json` byte-identical; add a **byte copy of its frozen pipeline-fingerprint manifest** (`data/cache/evidence-inference-item-risk-v1-final-v5/diagnostic-pipeline-fingerprint.json` → `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json`; contents = 23 repository-relative paths, byte counts, SHA-256s, and version/feature settings — no data, labels, or text) and change the registry validator to bind `lineage.diagnostic_pipeline_sha256` to that self-consistent manifest **and** to the literal pin, requiring only that the *closure definition* (component id/version and the ordered file-path list) still equals the current code's definition. Hash drift of closure files is what makes the artifact historical (today: exactly one of 23 files, `src/literature_multiverse/pipeline_fingerprint.py`; Python/numpy/pydantic/scipy versions identical).

*Honesty note (from review E7):* `current_verifier_pipeline_compatible=false` is a frozen component *setting* meaning the standalone projector was never composed with the verifier (`evidence_inference_item_risk.py:1-9, :456`; the summary validator at `:1421` requires it False). It is **not** a declaration that the diagnostic pipeline is non-current. The artifact's `protocol.materialization_access_order[2]` = `standalone_diagnostic_pipeline_recomputed_and_matched` and the current registry limitation (`public_artifacts.py:176-178`) promised recompute-and-match. **Phase 0 withdraws that guarantee** and replaces it with self-consistency + closure-definition checks. The report states this as an evidence narrowing, not a no-op.

*Rejected — reproduce under the current pipeline:* offline and cheap, but it emits a second, numerically identical artifact and re-pins `f1510d0d…`, which the next dependency bump (the closure includes `pyproject.toml`/`uv.lock` and embeds interpreter/library versions) breaks identically. If the operator prefers reproduction, Task 2 is replaced by that run; everything else is unchanged.

### D2 — Register every tracked diagnostic through its existing producer model

All 18 unregistered `artifacts/diagnostics/**/*.json` files get `PublicArtifactSpec` entries with their existing self-hash field (all 18 conventions verified: `hash_canonical(payload minus field)`), `source_map_bindings=()` (none embeds a reserved map — verified), `result_recomputed_from_public_inputs=False`, and a *scoped* limitation string (E21: "historical, embedded pipeline identity drifted" only where verified; "current blocked receipt" for the readiness file; "frozen plan/roster, embeds PMC and example identifiers" for the five roster files; the ledger has current implementation hashes but a drifted embedded question-evaluation pipeline identity). Six thin validator kinds wrap **existing** pydantic models (`VerificationCertificate`, `EvidenceBoundaryLedgerV1`, `DecisiveEvaluationReadinessV1`, `MetaSynPassageOfflineFeasibilityAuditV1`/`MetaSynContextualFrontierV1FailureAudit`, `PublicPairedSummaryV1`, `EvidenceInferenceFableFullUnionPublicEvaluationV2`), each in the file's own `try/except ValueError → PublicArtifactValidationError(code)` convention (`public_artifacts.py:1212-1228`). The producer contracts already fail closed on authority escalation before any registry post-check runs (verified: `unverified_corpus_cannot_have_released_certificate`, `decisive_evaluation_v1_real_candidate_mismatch`, `literal_error` on `claim_release_authority`). No parallel module, no new contract.

### D3 — Rights gate: Option A (monitor `artifacts/diagnostics`, declare every file) — corrected split

Add `artifacts/diagnostics` to `monitored_path_prefixes` and declare four collections: `project_authored_prompt_templates` (`prompts/**`; only the 3 `.txt` files are monitored, the 14 `.md` are unmonitored by suffix), `metasyn_derived_diagnostics_with_source_text` (**6** files: 5 JSON + the HTML certificate; `redistribution_not_established`, release-blocking), `project_authored_evidence_inference_rosters_with_pmc_identifiers` (**5** files carrying 191/90/7/7/6 distinct PMC ids and hundreds of `ei2-prompt-*` example ids; no article text; Evidence Inference ships no dataset license, `docs/evidence-inference-benchmark.md:166-169` — its `rights_status`/`public_release_allowed` is set by the operator's reply), and `project_authored_diagnostic_aggregates` (**22** explicit paths = 36 files incl. the companion − 6 − 5 − 3 already declared). `external-validation.json` is hash/flag-only and belongs to the aggregates (E18). Offline simulation against the parsed index: 0 undeclared / 0 ambiguous / 0 empty; CI gate green; `--require-release-ready` stays exit 2 (honest). Cost: every future diagnostic needs one policy line; the honesty of the text/aggregate split is enforced by a content test (E20), not by hand.

*Fallback Option B:* prompts collection only; leaves the HTML and five text-bearing JSON files invisible to the gate. Not recommended.

Both options fix the stale local-suite test and produce `docs/public-repository-cleanup-plan.md`. **No file is deleted and no history is rewritten in Phase 0.**

### D4 — Private-cache tests: one skip mechanism, honest staleness pins, restore what tracked artifacts already freeze

A `private_cache` marker (selection only) plus `tests/private_cache_support.py` with exactly two functions and two constants. Every fixture/test that opens `data/cache/**` calls `require_private_cache(...)` as its first statement (no autouse guard — E12: it cannot pre-empt session/module fixtures). Fixtures that replay a frozen bundle wrap the replay in `skip_when_historical_replay_is_stale(...)`, which converts *only* the two documented stale codes into a skip. The skip is honest only because two pinned tests prove the divergence is confined to pipeline identity (E14: typed-pilot rebuild equals the frozen bundle outside identity fields — today 7 identity leaves; hosted adapter contract valid, upstream pilot sha differs). Fixture families whose outputs are byte-frozen in **tracked public artifacts** (`evidence-boundary-ledger-v1.json`, `contextual-grounding-offline-feasibility-suite-v3.json`, `metasyn-passage-offline-feasibility-audit-v1.json`) load those artifacts through their producer models — restoring 45 tests in public CI with zero cache reads (E15; handoff §P1 lines 274-275 allow exactly this). Optional (operator's choice, E16): restore the 16 hosted-runtime stage-machine tests privately by mirroring the file's own bypass (`tests/test_metasyn_bounded_hosted_runtime.py:166-182`), ~511 s per private run.

*Rejected:* regenerated or synthetic fixtures for the MetaSyn bounded/hosted families (forbidden workspace regeneration or new runs). Coverage still lost after this plan is recorded per file in the report.

### D5 — Hygiene

Remove the three unused imports (Phase 1 re-adds them with real dispatch); drop the four deleted paths from `tests/test_manuscript_boundary.py`; correct the two docs lines that still say native-extraction component "version 12" (code and tests say 13). Doc-only edits outside Steps 1–6 are labelled as such in the report.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/literature_multiverse/question_evaluation.py:31-36` | remove 3 unused imports | 1 |
| `docs/reproducibility-v2.md:72`, `docs/acquisition-verification.md:45` | "version 12" → 13 | 1 |
| `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json` (new, byte copy, unindexed) | historical fingerprint manifest | 2 |
| `src/literature_multiverse/public_artifacts.py` | item-risk validator → historical binding; 6 validator kinds; 18 registry entries | 2, 3 |
| `tests/test_evidence_inference_item_risk.py` | historical-binding tests | 2 |
| `tests/test_public_artifacts.py` | companion set; registry-shape loop; tamper tests | 2, 3 |
| `docs/evidence-inference-item-risk-diagnostic.md`, `docs/reproducibility-v2.md` | withdrawn-guarantee wording; fix the stale `bf1c5c58…` hash | 2 |
| `configs/public-data-rights-v1.json`, `docs/public-data-rights.md` | prefix + 4 collections | 4 |
| `tests/test_public_data_rights.py` | local-suite fix; coverage + content-split tests | 4 |
| `docs/public-repository-cleanup-plan.md` (new) | every unresolved collection + options + operator decisions | 4 |
| `tests/conftest.py` | `private_cache` marker registered via `pytest_configure` (NOT `pyproject.toml`, which `metasyn_screening_study._source_code_hashes()` hashes — see fix round 1) | 5 |
| `tests/private_cache_support.py` (new) | `require_private_cache`, `skip_when_historical_replay_is_stale` | 5 |
| 37 test files listed in Task 5 | guards, artifact-backed fixtures, 2 staleness pins | 5 |
| `tests/test_manuscript_boundary.py` | drop 4 deleted entries | 6 |

---

### Task 0: Re-measure the baseline on the current tree

- [ ] **Step 1:** `uv run ruff check . | tail -3` → record.
- [ ] **Step 2:** `uv run python scripts/validate_public_artifacts.py 2>&1 | tail -1` → record (expect the item-risk lineage error).
- [ ] **Step 3:** `uv run python scripts/audit_public_data_rights.py > /dev/null; echo exit=$?` → record (expect 2).
- [ ] **Step 4:** `uv run pytest -q -p no:cacheprovider tests/test_manuscript_boundary.py tests/test_public_artifacts.py tests/test_public_data_rights.py tests/test_evidence_inference_item_risk.py 2>&1 | tail -12` → record the exact failing test ids. The full-suite counts are re-measured in Task 7; the report carries measured numbers, never the table's.

---

### Task 1: Lint clean and version-text consistency

**Files:** `src/literature_multiverse/question_evaluation.py:31-36`; `docs/reproducibility-v2.md:72`; `docs/acquisition-verification.md:45`; `docs/hosted-native-grounding-bridge.md` (append after line 108).

- [ ] **Step 1: Remove the three unused imports** so the block reads exactly:

```python
from literature_multiverse.certificate import (
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
)
```

(Phase 1 reintroduces `VerificationCertificateV8`, `ConditionVerificationCertificateV8`, `FinalConditionVerificationCertificateV9` together with real v8/v9 dispatch.)

- [ ] **Step 2: Fix the stale docs lines.** `grep -rn "version 12" docs README.md --exclude-dir=superpowers` → exactly three hits: `docs/reproducibility-v2.md:72`, `docs/acquisition-verification.md:45`, `docs/hosted-native-grounding-bridge.md:99`. Replace `version 12` with `version 13` in the first two. In the bridge doc insert, as a new paragraph after line 108: `This bump is now in place: the current native-extraction component version is 13.` Re-run the grep → only the bridge doc's historical line 99 remains.

- [ ] **Step 3: Verify.** `uv run ruff check .` → `All checks passed!`; `uv run ruff format --check src/literature_multiverse/question_evaluation.py`; `uv run pytest -q -p no:cacheprovider tests/test_question_evaluation.py` → 28 passed.

---

### Task 2: Item-risk artifact — historical binding (D1)

**Files:**
- Create (unindexed): `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json`
- Modify: `src/literature_multiverse/public_artifacts.py` (`_validate_evidence_inference_item_risk` ~1206-1286; the item-risk `PublicArtifactSpec` ~173-190; module imports)
- Modify: `tests/test_evidence_inference_item_risk.py` (imports at 1-28; append tests)
- Modify: `tests/test_public_artifacts.py:79-93` (`bound_companions`)
- Modify: `docs/evidence-inference-item-risk-diagnostic.md` (lines 69-70 and a new paragraph), `docs/reproducibility-v2.md` ("Artifact-backed item cell-rate UCLs")

**Interfaces:**
- Produces: `_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_FINGERPRINT: str`, `_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_PIPELINE_SHA256: str` (= the existing literal pin); error codes `evidence_inference_item_risk_historical_fingerprint_invalid`, `evidence_inference_item_risk_historical_closure_definition_mismatch`; the unchanged code `evidence_inference_item_risk_current_lineage_mismatch` now means "artifact lineage ≠ historical manifest/pin".

- [ ] **Step 1: Create the manifest copy and record exactly what drifted**

```bash
cp data/cache/evidence-inference-item-risk-v1-final-v5/diagnostic-pipeline-fingerprint.json \
   artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json
uv run python - <<'PY'
import json, hashlib
from pathlib import Path
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
path = Path("artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json")
m = PipelineFingerprint.model_validate(json.loads(path.read_text(encoding="utf-8")))
assert m.pipeline_sha256 == "2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15"
assert len(m.components) == 1 and m.components[0].component_id == "evidence-inference-retrospective-item-risk"
assert m.components[0].component_version == "2"
text = path.read_text(encoding="utf-8")
assert "/Users/" not in text and "PMC" not in text and "data/cache" not in text
drift = [f.path for f in m.components[0].files
         if hashlib.sha256(Path(f.path).read_bytes()).hexdigest() != f.sha256]
s = m.components[0].settings
print("DIFF:", drift, f"({len(drift)} of {len(m.components[0].files)})")
print("versions:", s["python_version"], s["numpy_version"], s["pydantic_version"], s["scipy_version"])
PY
```
Expected: `DIFF: ['src/literature_multiverse/pipeline_fingerprint.py'] (1 of 23)` and `versions: 3.12.11 2.5.2 2.13.4 1.18.0`. Record the output; if the drift list is anything else, stop and report before continuing. **The file is written to the working tree only; staging is deferred to the operator.** Until staged, the rights audit (index-based) cannot see it and a fresh clone fails `validate_public_artifacts.py` at index 4 with `evidence_inference_item_risk_historical_fingerprint_invalid` (demonstrated in Task 5 Step 4b).

- [ ] **Step 2: Write the failing tests.** Edit the import groups at `tests/test_evidence_inference_item_risk.py:1-28`: add `from types import SimpleNamespace` to the stdlib group; add `PublicArtifactValidationError` to the existing `from literature_multiverse.public_artifacts import (...)` block. Then append:

```python
_HISTORICAL_FINGERPRINT = Path(
    "artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json"
)


def _stand_in_fingerprint(**overrides: Any):
    def factory(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        fingerprint, source = _REAL_COMPUTE(**kwargs)
        component = fingerprint.components[0]
        return (
            SimpleNamespace(
                pipeline_sha256=overrides.get("pipeline_sha256", fingerprint.pipeline_sha256),
                components=[
                    SimpleNamespace(
                        component_id=component.component_id,
                        component_version=component.component_version,
                        files=overrides.get("files", component.files),
                    )
                ],
            ),
            source,
        )

    return factory


_REAL_COMPUTE = public_artifacts_module.compute_ei_item_risk_pipeline_fingerprint


def test_item_risk_public_validation_binds_historical_fingerprint_not_current_tree(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        public_artifacts_module,
        "compute_ei_item_risk_pipeline_fingerprint",
        _stand_in_fingerprint(pipeline_sha256="0" * 64),
    )
    _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_closure_definition_drift_fails_closed(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, _ = _REAL_COMPUTE(
        repository_root=repo_root,
        config=load_config(repo_root / _CONFIG),
        gepa_public_summary=json.loads(
            (repo_root / load_config(repo_root / _CONFIG).gepa_public_summary_path).read_text(
                encoding="utf-8"
            )
        ),
    )
    monkeypatch.setattr(
        public_artifacts_module,
        "compute_ei_item_risk_pipeline_fingerprint",
        _stand_in_fingerprint(files=list(reversed(real.components[0].files))),
    )
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_historical_closure_definition_mismatch",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_historical_fingerprint_manifest_tamper_fails_closed(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = public_artifacts_module._load_json_object
    target = (repo_root / _HISTORICAL_FINGERPRINT).resolve()

    def tampered(path: Path) -> dict[str, Any]:
        value = original(path)
        if Path(path).resolve() == target:
            value = json.loads(json.dumps(value))
            value["components"][0]["files"][0]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(public_artifacts_module, "_load_json_object", tampered)
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_historical_fingerprint_invalid",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)


def test_item_risk_coherently_rehashed_manifest_cannot_replace_the_pin(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = public_artifacts_module._load_json_object
    target = (repo_root / _HISTORICAL_FINGERPRINT).resolve()

    def rehashed(path: Path) -> dict[str, Any]:
        value = original(path)
        if Path(path).resolve() == target:
            value = json.loads(json.dumps(value))
            component = value["components"][0]
            component["files"][0]["sha256"] = "1" * 64
            component["component_sha256"] = hash_canonical(
                {k: v for k, v in component.items() if k != "component_sha256"}
            )
            value["pipeline_sha256"] = hash_canonical(
                {k: v for k, v in value.items() if k != "pipeline_sha256"}
            )
        return value

    monkeypatch.setattr(public_artifacts_module, "_load_json_object", rehashed)
    with pytest.raises(
        PublicArtifactValidationError,
        match="evidence_inference_item_risk_current_lineage_mismatch",
    ):
        _validate_evidence_inference_item_risk(_load_summary(repo_root), root=repo_root)
```

- [ ] **Step 3: Run; expect three failures and one pass.** `uv run ruff check tests/test_evidence_inference_item_risk.py` → clean. `uv run pytest -q -p no:cacheprovider tests/test_evidence_inference_item_risk.py -k "historical or closure_definition or binds_historical"` → `binds_historical` FAIL, `closure_definition_drift` FAIL, `manifest_tamper` FAIL (codes not raised yet); `rehashed_manifest` already passes (kept as a regression pin).

- [ ] **Step 4: Implement the historical binding** in `public_artifacts.py`. Add after `_EVIDENCE_INFERENCE_ITEM_RISK_CONFIG`:

```python
_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_FINGERPRINT = (
    "artifacts/diagnostics/evidence-inference/"
    "item-risk-calibration-v1-pipeline-fingerprint.json"
)
_EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_PIPELINE_SHA256 = (
    "2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15"
)
```

Add `from literature_multiverse.pipeline_fingerprint import PipelineFingerprint` to the module imports (`pipeline_fingerprint` imports only `lineage`/`models`; no cycle). In `_validate_evidence_inference_item_risk`, directly after the existing `try/except` that computes `fingerprint, recomputed_prediction_source`, insert:

```python
    try:
        historical = PipelineFingerprint.model_validate(
            _load_json_object(root / _EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_FINGERPRINT)
        )
    except (OSError, ValueError) as exc:
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_historical_fingerprint_invalid"
        ) from exc
    current_component = fingerprint.components[0]
    if (
        len(historical.components) != 1
        or historical.components[0].component_id != current_component.component_id
        or historical.components[0].component_version != current_component.component_version
        or [record.path for record in historical.components[0].files]
        != [record.path for record in current_component.files]
    ):
        raise PublicArtifactValidationError(
            "evidence_inference_item_risk_historical_closure_definition_mismatch"
        )
```

In the `if (...)` that raises `evidence_inference_item_risk_current_lineage_mismatch`, replace

```python
        or lineage.get("diagnostic_pipeline_sha256") != fingerprint.pipeline_sha256
        or lineage.get("diagnostic_pipeline_sha256")
        != "2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15"
```

with

```python
        or lineage.get("diagnostic_pipeline_sha256") != historical.pipeline_sha256
        or historical.pipeline_sha256
        != _EVIDENCE_INFERENCE_ITEM_RISK_HISTORICAL_PIPELINE_SHA256
```

Docstring: `"""Validate the public aggregate against its frozen historical pipeline manifest and the public GEPA lineage. Before 2026-08-29 this recomputed the diagnostic fingerprint from current bytes and required equality; it now consults the current tree only for the closure definition."""`

Registry entry `limitation` for `item-risk-calibration-v1.json`:

```python
        "historical diagnostic: its frozen 23-file pipeline fingerprint (2949fde1...) is "
        "bound by the tracked companion manifest and no longer equals the current tree "
        "(pipeline_fingerprint.py changed 2026-08-29). Before 2026-08-29 public CI recomputed "
        "this fingerprint from current bytes and required equality; it no longer does, and "
        "checks only manifest self-consistency, the closure definition, aggregate semantics, "
        "and public GEPA lineage. Ignored row-level paired predictions/labels are unavailable "
        "for metric replay, and this historically opened diagnostic has no claim-release "
        "authority",
```

- [ ] **Step 5: Register the companion.** In `tests/test_public_artifacts.py::test_every_public_diagnostic_json_is_registered_or_a_bound_companion` add `"artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json"` to `bound_companions`.

- [ ] **Step 6: Verify.** `uv run pytest -q -p no:cacheprovider tests/test_evidence_inference_item_risk.py` → all pass. `uv run python scripts/validate_public_artifacts.py 2>&1 | tail -1` → no item-risk error; if a *later* entry fails, record which (Task 3 input).

- [ ] **Step 7: Docs.** In `docs/evidence-inference-item-risk-diagnostic.md` replace lines 69-70 with: `The standalone diagnostic pipeline hash frozen with the tracked artifact is 2949fde1ce2f3df25f57d968a075352aa7f36b7f77f0af62ef24cf06e5680f15 (an earlier superseded rerun hashed bf1c5c58…; that run is not the public artifact).` and add under "Result and permitted interpretation": "Lineage status (2026-08-29): historical. The fingerprint above is frozen in the companion `item-risk-calibration-v1-pipeline-fingerprint.json`; the same 23-file closure under the current tree hashes differently because `pipeline_fingerprint.py` changed on 2026-08-29 (interpreter and library versions unchanged). Before 2026-08-29 public CI recomputed the fingerprint from current bytes and required equality; it now validates the companion's self-consistency, the closure definition, the public GEPA lineage, and the aggregate cells only. Reproducing the diagnostic under a new pipeline would produce a new, separately versioned artifact." Append one pointer sentence to `docs/reproducibility-v2.md` "Artifact-backed item cell-rate UCLs". Verify `grep -rn bf1c5c58 docs README.md src tests configs artifacts --exclude-dir=superpowers` → only the new parenthetical.

---

### Task 3: Register the 18 unregistered diagnostics (D2)

**Files:** `src/literature_multiverse/public_artifacts.py` (`SemanticValidator` 73-95; `PUBLIC_RESULT_REGISTRY` 135-327; `_semantic_validate` ~2401-2466; new functions); `tests/test_public_artifacts.py`.

**Interfaces:** new `SemanticValidator` members `decisive_readiness_blocked`, `evidence_boundary_ledger`, `fable_public_paired_summary`, `fable_public_union_evaluation_v2`, `historical_verification_certificate_v5`, `metasyn_offline_audit_model_only`; 18 new registry paths; error codes `historical_verification_certificate_v5_invalid`, `historical_verification_certificate_version_mismatch`, `historical_verification_certificate_status_mismatch`, `evidence_boundary_ledger_invalid`, `evidence_boundary_ledger_implementation_stale`, `decisive_readiness_invalid`, `decisive_readiness_registered_state_mismatch`, `metasyn_offline_audit_invalid`, `metasyn_offline_audit_unknown_artifact`, `fable_public_paired_summary_invalid`, `fable_public_paired_summary_version_mismatch`, `fable_public_union_evaluation_v2_invalid`.

- [ ] **Step 1: Re-prove the self-hash conventions (all 18 pass today).** Run the probe below; if any line prints `FAIL`, stop and report — do not register that file with a different field.

```bash
uv run python - <<'PY'
import json
from literature_multiverse import public_artifacts as pa
plan = {
 "artifacts/diagnostics/contextual-grounding-offline-feasibility-suite-v3.json": "suite_sha256",
 "artifacts/diagnostics/decisive-claim-evaluation-v1-real-readiness-blocked.json": "readiness_sha256",
 "artifacts/diagnostics/evidence-boundary-ledger-v1.json": "ledger_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json": "plan_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json": "public_summary_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-full-union-evaluation-v2.json": "evaluation_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot-recovery-v2-exclusions.json": "exclusion_ledger_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot-recovery-v2-execution-policy.json": "policy_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json": "plan_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-recovery-v2-plan-v1.json": "plan_sha256",
 "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-recovery-v2-summary-v1.json": "public_summary_sha256",
 "artifacts/diagnostics/evidence-inference/gepa-candidate-search-plan-v1.json": "plan_sha256",
 "artifacts/diagnostics/metasyn-contextual-frontier-v1-failure-audit-v1.json": "audit_sha256",
 "artifacts/diagnostics/metasyn-passage-offline-feasibility-audit-v1.json": "audit_sha256",
 "artifacts/diagnostics/postlive-recovery-v4-join-v1.json": "artifact_sha256",
 "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/external-validation.json": "validation_sha256",
 "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/sequential-audit-state.json": "state_sha256",
 "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/verification-certificate.json": "certificate_sha256",
}
for path, field in plan.items():
    try:
        pa._validate_self_hash(json.load(open(path)), field=field, artifact_path=path); print("OK  ", path)
    except pa.PublicArtifactValidationError as exc:
        print("FAIL", path, exc)
PY
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_public_artifacts.py`)

```python
def test_registered_diagnostics_that_are_not_recomputed_are_pinned_non_current(
    repo_root: Path,
) -> None:
    checked = 0
    for spec in PUBLIC_RESULT_REGISTRY:
        if not spec.path.startswith("artifacts/diagnostics/"):
            continue
        if spec.result_recomputed_from_public_inputs:
            continue
        assert spec.source_map_bindings is not None, spec.path
        assert spec.limitation, spec.path
        value = _load_json_object(repo_root / spec.path)
        if spec.self_hash_field is not None:
            _validate_self_hash(value, field=spec.self_hash_field, artifact_path=spec.path)
        _semantic_validate(spec.semantic_validator, value, root=repo_root, artifact_path=spec.path)
        checked += 1
    assert checked >= 18


@pytest.mark.parametrize(
    ("path", "field", "replacement", "error"),
    [
        (
            "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/verification-certificate.json",
            "status",
            "released",
            "historical_verification_certificate_v5_invalid",
        ),
        (
            "artifacts/diagnostics/decisive-claim-evaluation-v1-real-readiness-blocked.json",
            "real_scored_run_candidate",
            True,
            "decisive_readiness_invalid",
        ),
        (
            "artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json",
            "claim_release_authority",
            True,
            "fable_public_paired_summary_invalid",
        ),
        (
            "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-recovery-v2-summary-v1.json",
            "scientific_effectiveness_authority",
            True,
            "fable_public_paired_summary_invalid",
        ),
    ],
)
def test_historical_diagnostic_validators_reject_authority_escalation(
    repo_root: Path, path: str, field: str, replacement: Any, error: str
) -> None:
    # The producer contracts fail closed before the registry's secondary status checks;
    # those secondary checks are defense-in-depth and are not independently reachable
    # by mutation (verified 2026-08-29).
    spec = next(item for item in PUBLIC_RESULT_REGISTRY if item.path == path)
    value = json.loads(json.dumps(_load_json_object(repo_root / path)))
    value[field] = replacement
    with pytest.raises(PublicArtifactValidationError, match=error):
        _semantic_validate(spec.semantic_validator, value, root=repo_root, artifact_path=path)
```

(Add `from typing import Any` to the file's stdlib imports if absent.) Run: `uv run pytest -q -p no:cacheprovider tests/test_public_artifacts.py -k "pinned_non_current or authority_escalation"` → FAIL (`StopIteration` / count < 18).

- [ ] **Step 3: Add the validator kinds and functions.** Keep imports lazy inside the functions: `evidence_boundary_ledger_v1.py:59` imports `public_artifacts` at module level (a real cycle), and `certificate.py`/`decisive_claim_evaluation_v1.py` pull in the verifier chain.

```python
def _validate_historical_verification_certificate_v5(value: Mapping[str, Any]) -> None:
    from literature_multiverse.certificate import VerificationCertificate

    try:
        certificate = VerificationCertificate.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "historical_verification_certificate_v5_invalid"
        ) from exc
    if certificate.certificate_version != "literature-multiverse-verification-v5":
        raise PublicArtifactValidationError(
            "historical_verification_certificate_version_mismatch"
        )
    if certificate.status != "abstained":
        raise PublicArtifactValidationError(
            "historical_verification_certificate_status_mismatch"
        )


def _validate_evidence_boundary_ledger_public(
    value: Mapping[str, Any], *, root: Path
) -> None:
    from literature_multiverse.evidence_boundary_ledger_v1 import (
        _implementation_hashes,
        validate_evidence_boundary_ledger,
    )

    try:
        ledger = validate_evidence_boundary_ledger(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("evidence_boundary_ledger_invalid") from exc
    if ledger.ledger_implementation_file_sha256s != _implementation_hashes(root):
        raise PublicArtifactValidationError("evidence_boundary_ledger_implementation_stale")


def _validate_decisive_readiness_blocked(value: Mapping[str, Any]) -> None:
    from literature_multiverse.decisive_claim_evaluation_v1 import (
        DecisiveEvaluationReadinessV1,
    )

    try:
        readiness = DecisiveEvaluationReadinessV1.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("decisive_readiness_invalid") from exc
    if readiness.status != "blocked" or readiness.real_scored_run_candidate is not False:
        raise PublicArtifactValidationError("decisive_readiness_registered_state_mismatch")


def _validate_metasyn_offline_audit_model_only(
    value: Mapping[str, Any], *, artifact_path: str
) -> None:
    if artifact_path.endswith("metasyn-passage-offline-feasibility-audit-v1.json"):
        from literature_multiverse.metasyn_passage_offline_feasibility_audit_v1 import (
            MetaSynPassageOfflineFeasibilityAuditV1 as model,
        )
    elif artifact_path.endswith("metasyn-contextual-frontier-v1-failure-audit-v1.json"):
        from literature_multiverse.metasyn_contextual_frontier_v1_failure_audit import (
            MetaSynContextualFrontierV1FailureAudit as model,
        )
    else:
        raise PublicArtifactValidationError(
            f"metasyn_offline_audit_unknown_artifact:{artifact_path}"
        )
    try:
        model.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("metasyn_offline_audit_invalid") from exc


def _validate_fable_public_paired_summary(value: Mapping[str, Any]) -> None:
    from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
        PublicPairedSummaryV1,
    )

    try:
        PublicPairedSummaryV1.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError("fable_public_paired_summary_invalid") from exc
    if value.get("public_summary_version") != "evidence-inference-fable-public-paired-summary-v1":
        raise PublicArtifactValidationError("fable_public_paired_summary_version_mismatch")


def _validate_fable_public_union_evaluation_v2(value: Mapping[str, Any]) -> None:
    from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
        EvidenceInferenceFableFullUnionPublicEvaluationV2,
    )

    try:
        EvidenceInferenceFableFullUnionPublicEvaluationV2.model_validate(value)
    except ValueError as exc:
        raise PublicArtifactValidationError(
            "fable_public_union_evaluation_v2_invalid"
        ) from exc
```

Before finalizing, confirm the attribute names against the models (`DecisiveEvaluationReadinessV1.status`/`.real_scored_run_candidate`; `VerificationCertificate.status`/`.certificate_version`; `EvidenceBoundaryLedgerV1.ledger_implementation_file_sha256s`; `_implementation_hashes(root)` at `evidence_boundary_ledger_v1.py:1858-1862`). If a name differs, use the model's actual field — never relax the assertion. Both `fable-retrospective-*-summary-v1.json` files use `fable_public_paired_summary`; the union evaluation uses `fable_public_union_evaluation_v2`; the sequential-audit-state file is registered `generic` (its bytes are already bound inside the certificate); plans, exclusions, execution policy, candidate-search plan, join artifact, suite v3, and external-validation are `generic`.

Insert the six literals into `SemanticValidator` in sorted position **without reordering the existing entries**; add the `elif` branches to `_semantic_validate` immediately before `elif kind == "generic":` (pass `root=root` to the ledger validator and `artifact_path=artifact_path` to the MetaSyn one).

- [ ] **Step 4: Add the 18 registry entries** — one `PublicArtifactSpec` each, `source_map_bindings=()`, `result_recomputed_from_public_inputs=False`, with *scoped* limitations:

- `postlive-recovery-v4-public-verify-v1/verification-certificate.json` → `certificate_sha256`, `historical_verification_certificate_v5`: "historical abstained v5 certificate from the single-row MetaSyn post-live recovery diagnostic; its embedded pipeline identity predates the 2026-08-29 fingerprint bump and is not recomputed; embeds MetaSyn source quotes and titles (rights policy: release-blocking); no release or effectiveness authority".
- `.../sequential-audit-state.json` → `state_sha256`, `generic`: "historical sequential state bound byte-for-byte inside the sibling v5 certificate; embeds MetaSyn quotes/titles; no authority".
- `.../external-validation.json` → `validation_sha256`, `generic`: "hash/flag-only validation receipt that byte-binds the three sibling files; embeds no source text; no authority".
- `evidence-boundary-ledger-v1.json` → `ledger_sha256`, `evidence_boundary_ledger`: "implementation identity current (checked against the tree); embedded question-evaluation pipeline identity historical (drifted after 2026-08-29); every authority boundary in the ledger is false; no authority".
- `decisive-claim-evaluation-v1-real-readiness-blocked.json` → `readiness_sha256`, `decisive_readiness_blocked`: "current blocked readiness receipt with zero development/calibration/evaluation questions; no run, no pipeline identity; `real_scored_run_candidate=false`".
- the five roster/plan files (`fable-retrospective-full-plan-v1`, `fable-retrospective-pilot30-plan-v1`, `fable-retrospective-pilot30-recovery-v2-plan-v1`, `fable-retrospective-pilot-recovery-v2-exclusions`, `gepa-candidate-search-plan-v1`) → `plan_sha256`/`exclusion_ledger_sha256`, `generic`: "frozen plan/roster; no execution identity; embeds public Evidence Inference PMC article and benchmark example identifiers, no article text; no authority".
- `fable-retrospective-pilot-recovery-v2-execution-policy.json` → `policy_sha256`, `generic`: "frozen execution policy; no authority".
- the two `fable-retrospective-*-summary-v1.json` → `public_summary_sha256`, `fable_public_paired_summary`: "public aggregate of a retrospective cross-model transfer on historically opened labels; every authority flag typed False; exploratory only".
- `fable-retrospective-full-union-evaluation-v2.json` → `evaluation_sha256`, `fable_public_union_evaluation_v2`: same boundary wording.
- `contextual-grounding-offline-feasibility-suite-v3.json` → `suite_sha256`, `generic`; `metasyn-passage-offline-feasibility-audit-v1.json` and `metasyn-contextual-frontier-v1-failure-audit-v1.json` → `audit_sha256`, `metasyn_offline_audit_model_only`; `postlive-recovery-v4-join-v1.json` → `artifact_sha256`, `generic`: "historical offline diagnostic; embedded pipeline identity drifted after 2026-08-29 and is not recomputed; [embeds MetaSyn quotes/titles — for suite v3, passage audit, join]; no authority". Verify the frontier failure audit's text content before writing its limitation (the forbidden-field scan found none).

- [ ] **Step 5: Verify.** `uv run pytest -q -p no:cacheprovider tests/test_public_artifacts.py tests/test_evidence_inference_item_risk.py` → all pass. In `test_public_validator_cli_runs_outside_repository_working_directory` add `assert result["registered_artifacts"] == 41` and `assert all(item["result_recomputed_from_public_inputs"] is False for item in result["artifacts"] if item["path"].startswith("artifacts/diagnostics/") and item["path"] not in {p for p in records if records[p]["result_recomputed_from_public_inputs"]})` — simpler: assert the 18 new paths each have `result_recomputed_from_public_inputs is False`. `uv run python scripts/validate_public_artifacts.py | uv run python -c "import json,sys; r=json.load(sys.stdin); print(r['status'], r['registered_artifacts'])"` → `public_artifact_integrity_valid_with_scoped_semantics 41`. `uv run ruff check .` → clean.

---

### Task 4: Rights gate — classify, extend, fix the stale test, write the cleanup plan (D3, Option A)

**Files:** `configs/public-data-rights-v1.json`; `tests/test_public_data_rights.py`; `docs/public-data-rights.md`; create `docs/public-repository-cleanup-plan.md`.

- [ ] **Step 1: Failing tests** (append to `tests/test_public_data_rights.py`; add `from literature_multiverse.public_data_rights import _is_monitored, _load_policy, _matches, _structured_field_names` to the existing import block — confirm the exact helper names in `public_data_rights.py` first; if `_structured_field_names` is named differently, use the module's own helper that the forbidden-field scan calls at `:462-482`).

```python
def _tracked_paths_from_index(repo_root: Path) -> set[str]:
    # Parse .git/index (v2) directly; the worker never runs git.
    import struct

    data = (repo_root / ".git/index").read_bytes()
    signature, version, count = struct.unpack(">4sII", data[:12])
    assert signature == b"DIRC" and version == 2, (signature, version)
    position, paths = 12, set()
    for _ in range(count):
        flags = struct.unpack(">H", data[position + 60 : position + 62])[0]
        name_length = flags & 0x0FFF
        assert name_length < 0x0FFF
        paths.add(data[position + 62 : position + 62 + name_length].decode("utf-8"))
        position += ((62 + name_length + 8) // 8) * 8
    return paths


def test_project_authored_prompt_templates_are_declared(repo_root: Path) -> None:
    policy = json.loads(
        (repo_root / "configs/public-data-rights-v1.json").read_text(encoding="utf-8")
    )
    collection = next(
        item for item in policy["collections"]
        if item["collection_id"] == "project_authored_prompt_templates"
    )
    assert collection["path_globs"] == ["prompts/**"]
    assert collection["rights_status"] == "project_authored"
    assert collection["public_release_allowed"] is True
    assert "license_evidence" not in collection
    report = audit_public_data_rights(repository_root=repo_root)
    summary = next(
        item for item in report["collection_summaries"]
        if item["collection_id"] == "project_authored_prompt_templates"
    )
    assert summary["file_count"] >= 3
    assert set(summary["extension_counts"]) <= {".txt"}
    assert report["undeclared_file_count"] == 0


def test_every_monitored_diagnostic_and_prompt_on_disk_matches_exactly_one_collection(
    repo_root: Path,
) -> None:
    policy = _load_policy(repo_root / "configs/public-data-rights-v1.json")
    candidates = [
        path.relative_to(repo_root).as_posix()
        for base in ("artifacts/diagnostics", "prompts")
        for path in (repo_root / base).rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    monitored = [rel for rel in candidates if _is_monitored(rel, policy)]
    assert monitored
    for rel in monitored:
        matches = [c["collection_id"] for c in policy["collections"] if _matches(rel, c)]
        assert len(matches) == 1, (rel, matches)


def test_diagnostic_rights_split_matches_structured_field_content(repo_root: Path) -> None:
    policy = _load_policy(repo_root / "configs/public-data-rights-v1.json")
    forbidden = set(policy["established_rights_forbidden_field_names"])
    by_id = {c["collection_id"]: c for c in policy["collections"]}
    tracked = _tracked_paths_from_index(repo_root)
    text_free = by_id["project_authored_diagnostic_aggregates"]["path_globs"] + by_id[
        "project_authored_evidence_inference_rosters_with_pmc_identifiers"
    ]["path_globs"]
    text_bearing = by_id["metasyn_derived_diagnostics_with_source_text"]["path_globs"]
    for rel in text_free:
        if rel.endswith(".json") and (repo_root / rel).is_file():
            assert not (_structured_field_names(repo_root / rel) & forbidden), rel
    for rel in text_bearing:
        if rel.endswith(".json"):
            assert _structured_field_names(repo_root / rel) & forbidden, rel
    declared_tracked = {rel for rel in text_free + text_bearing if rel in tracked}
    monitored_tracked = {
        rel for rel in tracked if rel.startswith("artifacts/diagnostics/") and _is_monitored(rel, policy)
    }
    already_declared = {
        "artifacts/diagnostics/evidencebench-grounding-v1/audit-receipt.json",
        "artifacts/diagnostics/evidencebench-grounding-v1/summary.json",
        "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json",
    }
    assert declared_tracked | already_declared == monitored_tracked
```

Run: `uv run pytest -q -p no:cacheprovider tests/test_public_data_rights.py -k "prompt_templates or exactly_one_collection or rights_split"` → FAIL.

- [ ] **Step 2: Edit the policy** (`configs/public-data-rights-v1.json`): add `"artifacts/diagnostics"` to `monitored_path_prefixes` (keep the two existing sub-prefixes; keep the list sorted); add four collections in sorted `collection_id` order; no `allow_empty` on any of them (all non-empty).

```json
{
  "collection_id": "project_authored_prompt_templates",
  "path_globs": ["prompts/**"],
  "content_class": "project_authored_prompt_templates_without_third_party_text",
  "rights_status": "project_authored",
  "public_release_allowed": true,
  "rationale": "Prompt templates written for this project: placeholders and instructions only; no article text, abstracts, labels, or benchmark rows. Only the three hosted numeric-yield pilot .txt prompts are monitored by suffix; the Markdown prompts are unmonitored."
}
```

```json
{
  "collection_id": "metasyn_derived_diagnostics_with_source_text",
  "path_globs": [
    "artifacts/diagnostics/contextual-grounding-offline-feasibility-suite-v3.json",
    "artifacts/diagnostics/metasyn-passage-offline-feasibility-audit-v1.json",
    "artifacts/diagnostics/postlive-recovery-v4-join-v1.json",
    "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/sequential-audit-state.json",
    "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/verification-certificate.html",
    "artifacts/diagnostics/postlive-recovery-v4-public-verify-v1/verification-certificate.json"
  ],
  "content_class": "project_authored_diagnostics_embedding_metasyn_article_quotes_and_titles",
  "rights_status": "redistribution_not_established",
  "public_release_allowed": false,
  "rationale": "These diagnostics embed verbatim quotes, titles, and passages from MetaSyn-linked articles under quote/evidence_quote/title fields. METASYN_LICENSE.txt covers the benchmark annotations (see metasyn_benchmark_license) and reserves article metadata and excerpts; the redistribution basis for derived article text has not been established, so they block public release until reviewed or regenerated without source text."
}
```

```json
{
  "collection_id": "project_authored_evidence_inference_rosters_with_pmc_identifiers",
  "path_globs": [
    "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json",
    "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot-recovery-v2-exclusions.json",
    "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json",
    "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-recovery-v2-plan-v1.json",
    "artifacts/diagnostics/evidence-inference/gepa-candidate-search-plan-v1.json"
  ],
  "content_class": "project_authored_rosters_of_public_evidence_inference_pmc_and_example_identifiers_no_text",
  "rights_status": "redistribution_not_established",
  "public_release_allowed": false,
  "rationale": "Frozen request rosters and exclusion ledgers naming public PMC article identifiers (191/90/7/7/6 distinct) and Evidence Inference example identifiers; no article text, labels, or predictions. Evidence Inference 2.0 ships no dataset license (docs/evidence-inference-benchmark.md); the operator's classification is recorded here."
}
```

(Operator decision recorded 2026-08-29: `redistribution_not_established` / `false` — conservative until the identifier-redistribution question is reviewed.)

`project_authored_diagnostic_aggregates`: `path_globs` = the explicit sorted list of the **22** remaining paths (36 files on disk incl. the companion, minus the 6 text-bearing, the 5 roster, and the 3 already-declared `evidencebench-grounding-v1/{audit-receipt,summary}.json` and `metasyn-synthesis-yield-v1/summary.json`), generated by rglob over `artifacts/diagnostics`, sorted with `LC_ALL=C`, and naming `artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json` verbatim; `content_class: project_authored_aggregate_metrics_hashes_receipts_and_fingerprints`; `rights_status: project_authored`; `public_release_allowed: true`; rationale: "aggregate counts, hashes, plans without identifiers, receipts, and pipeline fingerprints authored by this project; no article text, labels, per-row predictions, or per-row identifiers (each JSON file is also validated by the public-artifact registry)".

- [ ] **Step 3: Fix the stale local-suite test.** Replace `test_repository_rights_report_surfaces_indexed_local_suite_receipts_as_content_silent_blocker` with:

```python
def test_local_suite_identifier_receipts_stay_declared_but_unindexed(repo_root: Path) -> None:
    report = audit_public_data_rights(repository_root=repo_root)
    local_suite = next(
        item for item in report["collection_summaries"]
        if item["collection_id"] == "local_suite_identifier_receipts"
    )
    assert local_suite["path_globs"] == [
        "artifacts/benchmarks/local-suite-v1/freeze_receipt.json",
        "artifacts/benchmarks/local-suite-v1/predictions.jsonl",
    ]
    assert local_suite["rights_status"] == "redistribution_not_established"
    assert local_suite["public_release_allowed"] is False
    # .gitignore and the CI aggregate-only step forbid indexing these receipts; the
    # declaration stays (allow_empty) so that indexing them can never be silent.
    assert local_suite["file_count"] == 0
    assert local_suite["extension_counts"] == {}
    assert [
        item for item in report["policy_blockers"]
        if item.get("collection_id") == "local_suite_identifier_receipts"
    ] == []
    serialized = json.dumps(report)
    assert "git_object_id" not in serialized
    assert "ARTICLE SENTENCE" not in serialized
```

- [ ] **Step 4: Confirm nothing pins the policy hash.** `grep -rn "5ad40d9b398257c09bbe30018eb1bfd24709f7714db9500fef433914168a0b3a" src tests artifacts configs docs README.md --exclude-dir=superpowers` → no hits (`report["policy"]["file_sha256"]` changes; nothing pins it).

- [ ] **Step 5: Verify the gate.** `uv run python scripts/audit_public_data_rights.py > /dev/null; echo exit=$?` → `exit=0`. `uv run python scripts/audit_public_data_rights.py --require-release-ready > /dev/null; echo exit=$?` → `exit=2` (expected). Note in the report: the local audit reads the git index and cannot see the untracked companion; the filesystem coverage test is the local proxy. `uv run pytest -q -p no:cacheprovider tests/test_public_data_rights.py` → all pass (`test_metasyn_synthesis_yield_rights_scope…` still asserts sorted prefixes and collection ids).

- [ ] **Step 6: Documents.** Update `docs/public-data-rights.md`: list the four collections and the `artifacts/diagnostics` prefix; reword "never emits … a per-file path list" to "emits declared path patterns verbatim, never per-file inventories or values". Write `docs/public-repository-cleanup-plan.md` with: (1) gate status (`policy_complete`, `release_ready`, counts); (2) every non-clean collection — the 10 Antiox collections (3,163 files, ≈130 MB, dominated by `data/raw/map/**` 2,658 files / 120 MB) and `metasyn_derived_diagnostics_with_source_text` (6 files) — with options per collection: keep private (remove from index; history rewrite requires operator approval), establish rights (license evidence), or regenerate without third-party text; (3) the operator's roster decision and its rationale; (4) contradictions resolved in Phase 0 (local-suite receipts); (5) unmonitored-by-design surfaces: 14 `prompts/*.md`; the `artifacts/paper/*.json` files outside monitored prefixes (list them by path); `.html` files outside prefixes; (6) recommended sequence and what needs explicit approval. No values, no per-row content.

---

### Task 5: Private-cache test separation and honest staleness (D4)

**Files:** `tests/conftest.py` (marker registration; `pyproject.toml` stays untouched — amended after fix round 1); create `tests/private_cache_support.py`; modify the 37 test files below.

Files that error today (stale replay): `test_metasyn_bounded_hosted_runtime.py`, `test_contextual_numeric_grounding_v3.py`, `test_metasyn_grounded_publication_bridge_v2.py`, `test_metasyn_contextual_frontier_runtime_v1.py`, `test_metasyn_v5_source_surface.py`, `test_metasyn_extraction_inputs_v2.py`, `test_metasyn_passage_offline_feasibility_audit_v1.py`, `test_metasyn_contextual_frontier_recovery_v2.py`, `test_evidence_boundary_ledger_v1.py`, `test_metasyn_passage_hosted_bundle_v2.py`, `test_metasyn_passage_packet_rescue_v3.py`, `test_metasyn_passage_hosted_runtime_v2.py`, `test_metasyn_contextual_frontier_recovery_lifecycle_v2.py`, `test_metasyn_grounded_analysis_v2.py`. Files that pass today but read the cache unguarded or partially guarded: `test_metasyn_synthesis_yield.py`, `test_verification_certificate_v8.py`, `test_metasyn_typed_pilot.py`, `test_evidence_inference_fable_full_union_reuse_v2.py`, `test_evidence_inference_item_risk.py`, `test_evidence_inference_fable_full_reuse_v1.py`, `test_corpus_pipeline_composition_runtime.py`, `test_metasyn_projection_v2.py`, `test_metasyn_contextual_frontier_recovery_v3.py`, `test_metasyn_bounded_runtime.py`, `test_evidence_inference_gepa_scaled_readiness_v1.py`, `test_composed_corpus_identity_v3.py`, `test_postlive_recovery_v4_join_v1.py`, `test_native_question_projection.py`, `test_native_packet_grounding_v2.py`, `test_native_packet_assembly_v2.py`, `test_native_ollama_diagnostic.py`, `test_native_bounded_generation.py`, `test_metasyn_contextual_frontier_recovery_v4.py`, `test_metasyn_bounded_adapter.py`, `test_hosted_native_numeric_canary_v4.py`, `test_metasyn_contextual_frontier_recovery_v4_posthoc_v1.py`, `test_postlive_recovery_v4_public_verify_v1.py`. (`test_synthesis_authorization_review.py` only writes `data/cache`-named paths under `tmp_path` — untouched.)

**Interfaces (`tests/private_cache_support.py`):** `require_private_cache(*relative: str) -> Path`; `skip_when_historical_replay_is_stale[T](build: Callable[[], T], *, stale_errors: tuple[type[BaseException], ...], stale_codes: frozenset[str]) -> T`; `TYPED_PILOT_STALE_CODES`, `HOSTED_ADAPTER_STALE_CODES`. Import form in every converted module is exactly `from tests.private_cache_support import <names>`, placed after `import pytest` (and any `import scripts…`) and before the blank line preceding `from literature_multiverse…` (repo convention, e.g. `tests/test_calibrate_adaptive_release_cli.py:6-9`).

- [ ] **Step 0: Prepare the clean copy used by the per-file check**

```bash
S=/private/tmp/claude-501/-Users-harry-Desktop-temp-reAgentsHack/82ed9c8a-d4be-4b41-af0d-a4204a9bf952/scratchpad/clean-checkout
rsync -a --delete --exclude /data/cache/ --exclude /.venv/ --exclude /.pytest_cache/ --exclude /.ruff_cache/ --exclude /paper/ --exclude "/Formatting_Instructions_For_NeurIPS_2026 (2)/" --exclude /artifacts/submission/ /Users/harry/Desktop/temp/reAgentsHack/ "$S/"
test -d "$S/docs/paper" && test -d "$S/artifacts/paper" && ! test -e "$S/paper" && test -d "$S/.git" && ! test -e "$S/data/cache" && echo copy-ok
cd "$S" && uv sync --frozen --offline --group dev --extra gepa
```
`docs/paper` and `artifacts/paper` are evaluation assets and must be present; only the top-level manuscript directory is excluded. The copy must use its own venv; never point it at the main `.venv`. If `--offline` fails, stop and report. Re-run the rsync (with `--delete`) after every batch of test edits so `$S` mirrors the working tree.

- [ ] **Step 1: Marker and helper.** ~~`pyproject.toml`~~ **Amended after fix round 1:** `pyproject.toml` must stay byte-identical to HEAD (its hash is part of `metasyn_screening_study._source_code_hashes()`, so the marker line raised `prepare_source_code_drift` in nine MetaSyn replay chains). Register the marker in `tests/conftest.py` instead:

```python
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "private_cache: reads ignored local data/cache artifacts; skipped in checkouts that lack them",
    )
```

The original (superseded) `pyproject.toml` text was:

```toml
markers = [
  "live: requires explicit network/provider access and is excluded from default CI",
  "private_cache: reads ignored local data/cache artifacts; skipped in checkouts that lack them",
]
```

`tests/private_cache_support.py`:

```python
"""Helpers that keep private-cache integration tests out of the public offline contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_CACHE_ROOT = REPOSITORY_ROOT / "data/cache"
TYPED_PILOT_STALE_CODES = frozenset({"metasyn_pilot_prepare_external_replay_mismatch"})
HOSTED_ADAPTER_STALE_CODES = frozenset({"metasyn_hosted_adapter_upstream_stale"})


def require_private_cache(*relative: str) -> Path:
    """Skip unless every named ignored path exists; never fabricate a substitute."""

    missing = [item for item in relative if not (REPOSITORY_ROOT / item).exists()]
    if missing:
        pytest.skip(f"private local cache artifact unavailable in this checkout: {missing[0]}")
    return REPOSITORY_ROOT


def skip_when_historical_replay_is_stale[T](
    build: Callable[[], T],
    *,
    stale_errors: tuple[type[BaseException], ...],
    stale_codes: frozenset[str],
) -> T:
    """Run a frozen-bundle replay; convert only the documented stale codes into a skip."""

    try:
        return build()
    except stale_errors as exc:
        if str(exc) in stale_codes:
            pytest.skip(
                "historical private bundle is stale under the current pipeline "
                f"({exc}); the identity-only staleness test pins this"
            )
        raise
```

`uv run ruff check tests/private_cache_support.py pyproject.toml` → clean. No `conftest.py` change.

- [ ] **Step 2: Two identity-only staleness pins** (both `@pytest.mark.private_cache`, both call `require_private_cache` first).

`tests/test_metasyn_typed_pilot.py` (confirm `_IDENTITY_FIELDS` names against `MetaSynTypedPilotPrepareBundleV1`; today exactly 7 differing leaves, all identity fields — if the differing leaves are under other names, use those names and never widen beyond identity/fingerprint fields):

```python
_IDENTITY_FIELDS = {
    "pilot_pipeline_fingerprint",
    "pilot_pipeline_sha256",
    "downstream_verifier_pipeline_sha256",
    "prepare_bundle_sha256",
}


@pytest.mark.private_cache
def test_historical_typed_oracle_pilot_v2_diverges_from_current_pipeline_only_in_identity() -> None:
    root = require_private_cache("data/cache/metasyn/typed-oracle-pilot-v2")
    workspace = root / "data/cache/metasyn/typed-oracle-pilot-v2"
    with pytest.raises(MetaSynTypedPilotError, match="metasyn_pilot_prepare_external_replay_mismatch"):
        validate_metasyn_typed_pilot_prepare(repository_root=root, workspace=workspace)
    private = metasyn_typed_pilot._private_workspace(workspace, repository_root=root)
    bundle = MetaSynTypedPilotPrepareBundleV1.model_validate(
        json.loads((private / PREPARE_BUNDLE_FILENAME).read_text(encoding="utf-8"))
    )
    inputs = bundle.repository_inputs
    rebuilt = _build_prepare_bundle(
        repository_root=root,
        screening_work_dir=(root / inputs["screening_fit_receipt"]).parent,
        reviews_train_path=root / inputs["reviews_train"],
        corpus_manifest_path=root / inputs["corpus_manifest"],
    )
    assert rebuilt.model_dump(mode="json", exclude=_IDENTITY_FIELDS) == bundle.model_dump(
        mode="json", exclude=_IDENTITY_FIELDS
    )
    assert rebuilt.pilot_pipeline_sha256 != bundle.pilot_pipeline_sha256
```

`tests/test_metasyn_bounded_hosted_runtime.py`:

```python
@pytest.mark.private_cache
def test_historical_qwen_attempt06_adapter_is_stale_only_in_upstream_pilot_identity() -> None:
    root = require_private_cache(
        "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/execution-bundle.private.json",
        "data/cache/metasyn/typed-oracle-pilot-v2",
    )
    frozen = json.loads(
        (root / "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/execution-bundle.private.json")
        .read_text(encoding="utf-8")
    )
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(frozen["adapter_bundle"])
    current_pilot_sha, _downstream = runtime._pilot_downstream_sha(root)
    assert current_pilot_sha != adapter.upstream_pilot_pipeline_sha256
    config, config_sha = load_metasyn_hosted_runtime_config(repository_root=root)
    with pytest.raises(MetaSynHostedRuntimeError, match="metasyn_hosted_adapter_upstream_stale"):
        freeze_metasyn_hosted_execution_bundle(
            adapter_bundle=adapter,
            runtime_config=config,
            config_file_sha256=config_sha,
            pilot_workspace_relative="data/cache/metasyn/typed-oracle-pilot-v2",
            repository_root=root,
        )
```

(`runtime` = `import literature_multiverse.metasyn_bounded_hosted_runtime as runtime`, matching the file's existing alias if one exists.)

- [ ] **Step 3: Per-file conversion recipe** (one worker per file; files are disjoint; no worktree isolation).

Required private paths per file (verify while converting; never widen `stale_codes`): `test_evidence_inference_gepa_scaled_readiness_v1` → `data/cache/evidence-inference-gepa/{manifest,conversion_report}.json`; `test_metasyn_bounded_hosted_runtime` → `data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/execution-bundle.private.json`, `data/cache/metasyn/typed-oracle-pilot-v2`; `test_metasyn_contextual_frontier_recovery_v{2,3,4}` and `_lifecycle_v2` → `data/cache/metasyn/contextual-frontier-runtime-v1/{00-prepared,02-terminal}.json`, `data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json`; `test_metasyn_extraction_inputs_v2`, `test_metasyn_v5_source_surface` → `data/cache/metasyn/bounded-anthropic-yield-v5`, `data/cache/metasyn/typed-oracle-pilot-v2`; runtime-v1 / hosted-bundle-v2 / hosted-runtime-v2 / publication-bridge-v2 / grounded-analysis-v2 / packet-rescue-v3 → `data/cache/metasyn/passage-hosted-yield-v2`, `data/cache/metasyn/bounded-anthropic-yield-v5`; `…recovery_v4_posthoc_v1` → `data/cache/metasyn/contextual-frontier-recovery-v4`, `…-posthoc-v1`; `test_postlive_recovery_v4_{join,public_verify}_v1` → `data/cache/metasyn/contextual-frontier-recovery-v4-posthoc-v1/artifact.json`; fable full-reuse-v1 / union-reuse-v2 → the workspaces named at their lines 34-45 / 55-66; the v8/composition/certificate files → the hosted-v5 paths their `HAS_REAL_V5` guards already name.

(a) `uv run pytest -q -p no:cacheprovider tests/<file> -x 2>&1 | tail -30`; capture the first error.
(b) Every fixture/test that opens a `data/cache` path calls `require_private_cache("<relative>", ...)` as its first statement and is marked `@pytest.mark.private_cache` (module-level `pytestmark = pytest.mark.private_cache` only when every test in the module needs the cache). The marker never substitutes for the in-fixture guard.
(c) If the fixture's replay raises one of the two stale codes, wrap the raising call: `bundle = skip_when_historical_replay_is_stale(lambda: <original call>, stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError), stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES)` (import the two error types where used).
(c′) **Artifact-backed fixtures (E15):** in `test_evidence_boundary_ledger_v1.py`, `test_contextual_numeric_grounding_v3.py`, and `test_metasyn_passage_offline_feasibility_audit_v1.py`, the module fixture becomes `Model.model_validate(json.loads((REPOSITORY_ROOT / "<tracked artifact>").read_text(encoding="utf-8")))` using `EvidenceBoundaryLedgerV1` / `ContextualGroundingOfflineFeasibilitySuiteV3` / `MetaSynPassageOfflineFeasibilityAuditV1` on `artifacts/diagnostics/evidence-boundary-ledger-v1.json` / `contextual-grounding-offline-feasibility-suite-v3.json` / `metasyn-passage-offline-feasibility-audit-v1.json`; only tests that invoke external replay against the repository keep `private_cache` + the stale-skip. Expected: 45 tests restored in public CI.
(d) Do not touch tests that only *mention* `data/cache` inside `tmp_path` fixtures.
(e1) `uv run pytest -q -p no:cacheprovider tests/<file>` in the working tree → `0 errors, 0 failed`; record `passed/skipped` and the skipped test names.
(e2) `rsync` the edited file into `$S` and run `cd "$S" && uv run pytest -q -p no:cacheprovider tests/<file>` → pass or skip only; any ERROR means a wrong guard path — fix the guard, never widen `stale_codes`.
(f) `uv run ruff check tests/<file>` and `uv run ruff format --check tests/<file>`.

- [ ] **Step 3b (optional, operator's choice — E16): restore the hosted-runtime stage machine privately.** In `tests/test_metasyn_bounded_hosted_runtime.py`, the session fixture saves `ORIGINAL_PILOT_DOWNSTREAM_SHA = runtime._pilot_downstream_sha`, calls `require_private_cache(...)`, builds `adapter`/`config` as today, and freezes the bundle under a patch scoped to the freeze call:

```python
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            runtime,
            "_pilot_downstream_sha",
            lambda _root: (
                adapter.upstream_pilot_pipeline_sha256,
                frozen["downstream_verifier_pipeline_sha256"],
            ),
        )
        return freeze_metasyn_hosted_execution_bundle(...)
```

If later stage tests re-check the upstream identity and fail with the stale code, widen the patch to the session (mirroring lines 166-182) and make the Step 2 staleness test restore `ORIGINAL_PILOT_DOWNSTREAM_SHA` via `monkeypatch.setattr` before asserting. `test_hosted_bundle_is_closed_and_current` (line 203) becomes the inverted assertion under the original function; `test_cli_validation_is_offline_and_live_stage_requires_explicit_flag` (line 883, subprocess) stays skipped. Expected: 16 restored / 1 converted / 1 skipped; ≈511 s per private run. If declined, record the 16 as recoverable in the report.

- [ ] **Step 4a: Public no-cache proof on the rsync copy.** Re-sync `$S`, then `cd "$S" && uv run ruff check . && uv run python scripts/validate_public_artifacts.py > /dev/null && uv run python scripts/audit_public_data_rights.py > /dev/null && uv run pytest -q -m "not live" -p no:cacheprovider` → `0 failed, 0 errors`; record passed/skipped.

- [ ] **Step 4b: Index-faithful fresh-clone proof (E3).** Build `$S2` from the parsed index (no git command):

```bash
S2=/private/tmp/claude-501/-Users-harry-Desktop-temp-reAgentsHack/82ed9c8a-d4be-4b41-af0d-a4204a9bf952/scratchpad/index-checkout
uv run python - "$S2" <<'PY'
import shutil, struct, sys
from pathlib import Path
root = Path("/Users/harry/Desktop/temp/reAgentsHack"); dest = Path(sys.argv[1])
data = (root / ".git/index").read_bytes()
signature, version, count = struct.unpack(">4sII", data[:12])
assert signature == b"DIRC" and version == 2, (signature, version)
position, paths = 12, []
for _ in range(count):
    flags = struct.unpack(">H", data[position + 60 : position + 62])[0]
    name_length = flags & 0x0FFF
    assert name_length < 0x0FFF
    paths.append(data[position + 62 : position + 62 + name_length].decode("utf-8"))
    position += ((62 + name_length + 8) // 8) * 8
if dest.exists():
    shutil.rmtree(dest)
for rel in paths:
    target = dest / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, target)
shutil.copytree(root / ".git", dest / ".git", symlinks=True)
print("copied", len(paths), "indexed paths")
PY
cd "$S2" && uv sync --frozen --offline --group dev --extra gepa && uv run python scripts/validate_public_artifacts.py 2>&1 | tail -1
```
Expected **before staging**: `evidence_inference_item_risk_historical_fingerprint_invalid` (the companion is not indexed) — record it as the expected fresh-clone state. Then copy the three new files into `$S2` (`artifacts/diagnostics/evidence-inference/item-risk-calibration-v1-pipeline-fingerprint.json`, `tests/private_cache_support.py`, `docs/public-repository-cleanup-plan.md`) and re-run the three validators and `uv run pytest -q -m "not live" -p no:cacheprovider` → green. Record both results; the difference is exactly what the operator's `git add` will close.

---

### Task 6: Manuscript-boundary test reflects the index (D5)

- [ ] **Step 1:** In `tests/test_manuscript_boundary.py:12-33` remove `artifacts/paper/evidence-inference-2/failed-raw-schema-pilot30-summary.json`, `artifacts/paper/evidence-inference-benchmark-summary.json`, `artifacts/paper/evidence-inference-gepa-pilot-summary.json`, `artifacts/paper/evidence-inference-low-budget-summary.json` from `EVALUATION_ASSETS`; add the comment `# 2026-08-28: the four evidence-inference-* paper summaries were removed from the index.` (the remaining 18 entries match the index — verified).
- [ ] **Step 2:** `uv run pytest -q -p no:cacheprovider tests/test_manuscript_boundary.py` → 2 passed.

---

### Task 7: Full verification and Phase 0 report

- [ ] **Step 1:** `uv run ruff check .` → clean; `uv run ruff format --check` on every edited/created Python file → formatted; for every new/edited non-Python file: no trailing whitespace and a final newline (`grep -nE " +$" <file>`; `tail -c1 <file> | od -c` ends in `\n`) — CI runs `git show --check` (`ci.yml:76-77`).
- [ ] **Step 2:** `uv run python scripts/validate_public_artifacts.py` → exit 0, 41 artifacts.
- [ ] **Step 3:** `uv run python scripts/audit_public_data_rights.py` → exit 0, `policy_complete=true`, `release_ready=false`; `--require-release-ready` → exit 2 (documented as expected). Note the index/companion caveat.
- [ ] **Step 4:** the 28 CI smoke commands at `.github/workflows/ci.yml:38-65` (27 `--help` + `run_local_benchmarks.py --contract-only`) → exit 0 each.
- [ ] **Step 5:** `uv run pytest -q -m "not live" -p no:cacheprovider` in the working tree (with cache): `0 failed, 0 errors`; record passed/skipped; `uv run pytest -q -m private_cache -p no:cacheprovider` → record passed/skipped (the staleness pins must be among the passes). Then Task 5 Steps 4a and 4b.
- [ ] **Step 6: Report** (chat) with exactly: **files changed** (edited vs created; created files are *not yet indexed — operator must `git add`*); **tests and validators run** with counts, including coverage lost per file and the git invocations made by project code; **remaining blockers** (Antiox rights; the 6 text-bearing diagnostics; the roster decision; MetaSyn historical bundles and the families still skipped; the stale v2 composition receipt and v8 smoke — P1; the CI history step is expected green on main/pub from offline evidence, unfetched origin branches unverifiable; the GitHub Support purge); **provider calls and spend** (none; ≈$61.14 unchanged); **whether any scientific claim became newly authorized** (no — and the item-risk artifact's guarantee was *narrowed*, stated explicitly); **doc-only edits outside Steps 1–6** labelled.

---

## Refuted review claims (do not "fix" these)

- CI's history-reachability step is expected green on `main`/`pub` (offline walk found zero forbidden paths in reachable commits); unfetched origin branches cannot be checked without git. Outside Phase 0 scope.
- The baseline HEAD citation (`288f8b7`) is correct; only the dirty-tree pytest counts need re-measuring (Task 0).
- The bare `private_cache_support` import form is ruff-clean but is a different module object; the canonical `from tests.private_cache_support import …` form is required for consistency, not for lint.
- `PipelineFingerprint.model_validate` recomputes component and pipeline hashes, so the manifest tamper test fires; both monkeypatch targets intercept the real call sites (`public_artifacts.py:25`, `:1219`); all 18 self-hash conventions pass; all six producer models validate their tracked files in <0.1 s with zero `data/cache` reads; `.html` is monitored by prefix under Option A; `prompts/**` matches exactly the 3 `.txt` files; the 18 remaining `EVALUATION_ASSETS` match the index; `uv sync --frozen --offline` works from the local cache; the tests' git shell-outs need no identity under `actions/checkout`.

## Residual risks to disclose

1. Fresh-clone red until the operator stages the three new files (Task 5 Step 4b demonstrates exactly this).
2. The item-risk artifact's recompute-and-match guarantee is withdrawn; the closure-definition check is the only remaining tether to current code.
3. `question_evaluation.py` and `pyproject.toml` are stale against the v2 composition receipt; v3 receipt + v8 smoke are P1.
4. Staleness pins never run in public CI (no `data/cache`); the report gives local `-m private_cache` counts.
5. Rights decisions pending the operator: roster classification; the 6 text-bearing MetaSyn diagnostics stay honest release blockers next to the 10 Antiox collections, so `--require-release-ready` stays exit 2.
6. Unmonitored-by-design surfaces remain (14 `prompts/*.md`, `artifacts/paper/*.json` outside prefixes, `.html` outside prefixes).
7. Coverage genuinely lost after E15/E16: MetaSyn v5 source-surface / extraction-inputs / hosted-bundle families still skip; hosted stage machine too if E16 is declined. Recorded per file.
8. Ledger identity is mixed (implementation current; embedded question-evaluation pipeline drifted) — limitation states both.
9. Runtime: the full non-live suite (~43 min) runs twice (working tree, rsync copy) plus the index-faithful copy; the optional hosted restore adds ≈511 s per private run.
