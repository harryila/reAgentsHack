"""Offline-only Streamlit viewer for a frozen Literature Multiverse release.

The application deliberately has no search, extraction, or model-provider imports.  It
validates a self-contained ``artifacts/<question-id>/demo`` directory before rendering
anything and never reads scientific data from the working pipeline directories.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTION_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# ``manifest.json`` cannot hash itself.  Every other bundled path is fixed.
BUNDLED_PATHS = frozenset(
    {
        "analysis/bootstrap.json",
        "analysis/contradictions.parquet",
        "analysis/evidence_gaps.parquet",
        "analysis/headline.json",
        "analysis/m4_checkpoint.json",
        "analysis/m4_gate.json",
        "analysis/moderators.parquet",
        "analysis/permutation.json",
        "analysis/tree.json",
        "audit.json",
        "baseline.json",
        "demo_script.md",
        "findings.parquet",
        "g3_gate.json",
        "papers.parquet",
        "release_selection.json",
        "trace.json",
        "verification.json",
    }
)
ALL_DEMO_PATHS = BUNDLED_PATHS | {"manifest.json"}


class BundleValidationError(ValueError):
    """The selected frozen release is missing, altered, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class DemoBundle:
    root: Path
    manifest: Mapping[str, Any]
    headline: Mapping[str, Any]
    baseline: Mapping[str, Any]
    g3_gate: Mapping[str, Any]
    m4_gate: Mapping[str, Any]
    tree: Mapping[str, Any]
    trace: Mapping[str, Any]
    release_selection: Mapping[str, Any]
    verification: Mapping[str, Any]
    papers: pd.DataFrame
    findings: pd.DataFrame
    moderators: pd.DataFrame
    contradictions: pd.DataFrame
    evidence_gaps: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid_json:{path.name}") from exc
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"json_root_must_be_object:{path.name}")
    return value


def _safe_bundle_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or relative.startswith("./") or ".." in candidate.parts:
        raise BundleValidationError(f"unsafe_artifact_path:{relative}")
    target = root / candidate
    if target.is_symlink():
        raise BundleValidationError(f"symlink_artifact_forbidden:{relative}")
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BundleValidationError(f"artifact_escapes_bundle:{relative}") from exc
    return target


def _validate_artifact_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise BundleValidationError("manifest_artifacts_must_be_array")

    indexed: dict[str, Mapping[str, Any]] = {}
    for row in artifact_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise BundleValidationError("manifest_artifact_row_invalid")
        relative = str(row["path"])
        if relative in indexed:
            raise BundleValidationError(f"duplicate_artifact_entry:{relative}")
        indexed[relative] = row
    if set(indexed) != BUNDLED_PATHS:
        missing = sorted(BUNDLED_PATHS - set(indexed))
        extra = sorted(set(indexed) - BUNDLED_PATHS)
        raise BundleValidationError(f"artifact_allowlist_mismatch:missing={missing}:extra={extra}")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != ALL_DEMO_PATHS:
        missing = sorted(ALL_DEMO_PATHS - actual_files)
        extra = sorted(actual_files - ALL_DEMO_PATHS)
        raise BundleValidationError(f"bundle_inventory_mismatch:missing={missing}:extra={extra}")

    for relative, row in indexed.items():
        target = _safe_bundle_path(root, relative)
        if not target.is_file():
            raise BundleValidationError(f"missing_artifact:{relative}")
        if row.get("bytes") != target.stat().st_size:
            raise BundleValidationError(f"artifact_size_mismatch:{relative}")
        expected_hash = row.get("sha256")
        if not isinstance(expected_hash, str) or _sha256_file(target) != expected_hash:
            raise BundleValidationError(f"artifact_hash_mismatch:{relative}")


def _required_manifest_fields(manifest: Mapping[str, Any]) -> None:
    required = {
        "manifest_version",
        "schema_version",
        "fixture",
        "question_id",
        "research_question",
        "spoken_question",
        "corpus_qualifier",
        "narrative_variant",
        "created_at",
        "config_sha256",
        "code_version",
        "primary_cohort_definition",
        "paper_funnel",
        "quality",
        "exclusions",
        "release_selection",
        "lineage",
        "artifacts",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise BundleValidationError(f"manifest_fields_missing:{missing}")
    if manifest.get("manifest_version") != "1" or manifest.get("schema_version") != "1":
        raise BundleValidationError("manifest_version_unsupported")
    qid = manifest.get("question_id")
    if not isinstance(qid, str) or not QUESTION_ID_RE.fullmatch(qid):
        raise BundleValidationError("manifest_question_id_invalid")
    if manifest.get("corpus_qualifier") != "our retrieved corpus":
        raise BundleValidationError("manifest_corpus_qualifier_invalid")
    if manifest.get("narrative_variant") not in {"A", "B"}:
        raise BundleValidationError("manifest_variant_invalid")
    if manifest.get("primary_cohort_definition") != "primary_grounded_unflagged":
        raise BundleValidationError("manifest_primary_cohort_definition_invalid")


def _validate_cross_artifact_state(
    manifest: Mapping[str, Any],
    *,
    bundle_root: Path,
    headline: Mapping[str, Any],
    g3_gate: Mapping[str, Any],
    m4_gate: Mapping[str, Any],
    release_selection: Mapping[str, Any],
) -> None:
    variant = manifest["narrative_variant"]
    if headline.get("narrative_variant") != variant:
        raise BundleValidationError("headline_variant_mismatch")
    selected = release_selection.get("selected_release")
    manifest_selection = manifest.get("release_selection")
    if not isinstance(selected, Mapping) or not isinstance(manifest_selection, Mapping):
        raise BundleValidationError("release_selection_shape_invalid")
    if release_selection.get("disposition") != manifest_selection.get("disposition"):
        raise BundleValidationError("release_disposition_mismatch")
    if manifest_selection.get("record_sha256") != _sha256_file(
        bundle_root / "release_selection.json"
    ):
        raise BundleValidationError("release_selection_record_hash_mismatch")

    action = g3_gate.get("action")
    if action == "block_release":
        raise BundleValidationError("blocked_g3_release_must_not_be_bundled")
    if variant == "A" and action != "run_m4":
        raise BundleValidationError("variant_a_requires_run_m4")
    if variant == "A" and m4_gate.get("selected_variant") != "A":
        raise BundleValidationError("variant_a_m4_mismatch")
    if variant == "B" and m4_gate.get("selected_variant") not in {"B", None}:
        raise BundleValidationError("variant_b_m4_mismatch")


def _read_parquet(path: Path, name: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pandas backends expose several implementation errors
        raise BundleValidationError(f"invalid_parquet:{name}") from exc


def load_demo_bundle(demo_dir: str | Path) -> DemoBundle:
    """Validate and load one frozen bundle without consulting any upstream directory."""

    root = Path(demo_dir).resolve()
    if not root.is_dir():
        raise BundleValidationError(f"demo_directory_missing:{root}")
    manifest = dict(_load_json(root / "manifest.json"))
    _required_manifest_fields(manifest)
    _validate_artifact_inventory(root, manifest)

    headline = _load_json(root / "analysis" / "headline.json")
    baseline = _load_json(root / "baseline.json")
    g3_gate = _load_json(root / "g3_gate.json")
    m4_gate = _load_json(root / "analysis" / "m4_gate.json")
    tree = _load_json(root / "analysis" / "tree.json")
    trace = _load_json(root / "trace.json")
    release_selection = _load_json(root / "release_selection.json")
    _validate_cross_artifact_state(
        manifest,
        bundle_root=root,
        headline=headline,
        g3_gate=g3_gate,
        m4_gate=m4_gate,
        release_selection=release_selection,
    )
    return DemoBundle(
        root=root,
        manifest=manifest,
        headline=headline,
        baseline=baseline,
        g3_gate=g3_gate,
        m4_gate=m4_gate,
        tree=tree,
        trace=trace,
        release_selection=release_selection,
        verification=_load_json(root / "verification.json"),
        papers=_read_parquet(root / "papers.parquet", "papers"),
        findings=_read_parquet(root / "findings.parquet", "findings"),
        moderators=_read_parquet(root / "analysis" / "moderators.parquet", "moderators"),
        contradictions=_read_parquet(
            root / "analysis" / "contradictions.parquet", "contradictions"
        ),
        evidence_gaps=_read_parquet(
            root / "analysis" / "evidence_gaps.parquet", "evidence_gaps"
        ),
    )


def _question_from_argv(argv: Iterable[str]) -> str | None:
    values = list(argv)
    for index, value in enumerate(values):
        if value == "--question":
            if index + 1 >= len(values):
                raise BundleValidationError("--question requires a value")
            qid = values[index + 1]
            if not QUESTION_ID_RE.fullmatch(qid):
                raise BundleValidationError("invalid --question value")
            return qid
        if value.startswith("--question="):
            qid = value.partition("=")[2]
            if not QUESTION_ID_RE.fullmatch(qid):
                raise BundleValidationError("invalid --question value")
            return qid
    return None


def select_demo_dir(argv: Iterable[str] = ()) -> Path:
    """Resolve the explicit question or, for convenience, the only frozen local release."""

    qid = _question_from_argv(argv)
    if qid is not None:
        return REPO_ROOT / "artifacts" / qid / "demo"
    releases = sorted((REPO_ROOT / "artifacts").glob("*/demo/manifest.json"))
    if len(releases) == 1:
        return releases[0].parent
    if not releases:
        raise BundleValidationError("no frozen demo release found; pass --question after --")
    labels = [path.parents[1].name for path in releases]
    selected = st.sidebar.selectbox("Frozen release", labels)
    return REPO_ROOT / "artifacts" / selected / "demo"


def _format_fraction(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{100 * float(value):.1f}%"
    return "Unavailable"


def _render_funnel(manifest: Mapping[str, Any]) -> None:
    funnel = manifest["paper_funnel"]
    ordered = [
        ("Searched documents", "searched_documents"),
        ("Identity-deduped papers", "identity_deduped_papers"),
        ("Deterministically included", "deterministic_included_papers"),
        ("Extraction eligible", "extraction_eligible_papers"),
        ("Primary grounded papers", "primary_grounded_papers"),
        ("Primary grounded findings", "primary_grounded_findings"),
    ]
    st.dataframe(
        pd.DataFrame(
            [{"Stage": label, "Count": funnel.get(key)} for label, key in ordered]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _render_quality(manifest: Mapping[str, Any]) -> None:
    quality = manifest["quality"]
    exclusions = manifest["exclusions"]
    columns = st.columns(3)
    columns[0].metric("Mechanically grounded", _format_fraction(quality.get("grounded_fraction")))
    columns[1].metric(
        "Cross-model agreement",
        _format_fraction(quality.get("cross_model_agreement")),
    )
    columns[2].metric("Quarantined", _format_fraction(quality.get("quarantine_fraction")))
    st.caption(
        "Excluded from the primary headline: "
        f"mixed/unclear {_format_fraction(exclusions.get('mixed_or_unclear_fraction'))}; "
        f"flagged sections {_format_fraction(exclusions.get('section_flagged_fraction'))}; "
        f"verification {_format_fraction(exclusions.get('verification_excluded_fraction'))}."
    )
    st.caption(
        f"Human audit: {quality.get('audit_correct', 0)} of "
        f"{quality.get('audit_total', 0)} sampled rows correct."
    )


def _render_baseline(bundle: DemoBundle) -> None:
    majority = bundle.baseline.get("majority")
    if not isinstance(majority, Mapping):
        majority = {}
    columns = st.columns(2)
    columns[0].metric("Paper-balanced majority", str(majority.get("direction", "Unavailable")))
    columns[1].metric("Majority agreement", _format_fraction(majority.get("agreement")))
    rendered = bundle.headline.get("rendered_sentence")
    if isinstance(rendered, str) and rendered:
        st.success(rendered)
    llm = bundle.baseline.get("llm")
    if bundle.baseline.get("status") == "complete" and isinstance(llm, Mapping):
        st.info(f"Ungrounded LLM comparison (not evidence): {llm.get('paragraph', '')}")
    else:
        st.info("LLM baseline unavailable; it is not used in any scientific result.")


def _render_moderators(bundle: DemoBundle) -> None:
    if bundle.moderators.empty:
        st.info("No moderator rows were available.")
        return
    preferred = [
        "display_name",
        "moderator",
        "status",
        "paper_coverage",
        "delta_log_loss",
        "adjusted_p",
        "bootstrap_top3_frequency",
        "reason",
    ]
    columns = [column for column in preferred if column in bundle.moderators.columns]
    st.dataframe(
        bundle.moderators[columns] if columns else bundle.moderators,
        hide_index=True,
        use_container_width=True,
    )


def _render_pattern_or_residuals(bundle: DemoBundle) -> None:
    if bundle.manifest["narrative_variant"] == "A":
        st.subheader("Conditional pattern")
        nodes = bundle.tree.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            st.info("The validated tree contains no displayable nodes.")
            return
        leaves = [node for node in nodes if isinstance(node, Mapping) and node.get("is_leaf")]
        leaf_ids = [str(node.get("leaf_id", node.get("node_id"))) for node in leaves]
        if leaf_ids:
            chosen = st.selectbox("Tree leaf", leaf_ids)
            node = leaves[leaf_ids.index(chosen)]
            st.json(dict(node), expanded=True)
        else:
            st.json(nodes, expanded=False)
        return

    st.subheader("Residual contradictions and evidence gaps")
    if bundle.contradictions.empty:
        st.info("No grounded opposite-direction residual pairs were found.")
    else:
        st.dataframe(bundle.contradictions, hide_index=True, use_container_width=True)
    if bundle.evidence_gaps.empty:
        st.info("No pre-registered evidence-gap cells were emitted.")
    else:
        status_counts = (
            bundle.evidence_gaps["status"].value_counts().rename_axis("Status").reset_index(name="Cells")
            if "status" in bundle.evidence_gaps
            else bundle.evidence_gaps
        )
        st.dataframe(status_counts, hide_index=True, use_container_width=True)


def _render_evidence(bundle: DemoBundle) -> None:
    if bundle.findings.empty or "finding_id" not in bundle.findings:
        st.info("No evidence findings are available in this release.")
        return
    papers = bundle.papers.copy()
    findings = bundle.findings.copy()
    if "paper_id" not in papers or "paper_id" not in findings:
        st.error("The bundled ledgers cannot be joined by paper_id.")
        return
    paper_columns = [
        column
        for column in ("paper_id", "title", "first_author", "pub_year", "doc_id", "doi", "pmid")
        if column in papers
    ]
    joined = findings.merge(
        papers[paper_columns].drop_duplicates("paper_id"),
        on="paper_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_paper"),
    )
    joined = joined.sort_values("finding_id", kind="stable")
    identifiers = [str(value) for value in joined["finding_id"]]
    chosen = st.selectbox("Evidence finding", identifiers)
    row = joined.iloc[identifiers.index(chosen)]
    quote = row.get("evidence_quote")
    if isinstance(quote, str) and quote:
        st.markdown(f"> {quote}")
    else:
        st.info("This valid row has no displayable quote.")
    st.write(
        f"**Paper:** {row.get('title', row.get('paper_id'))}  \n"
        f"**Direction:** {row.get('effect_direction', 'Unavailable')}  \n"
        f"**Outcome/timepoint:** {row.get('outcome_name', 'Unavailable')} / "
        f"{row.get('timepoint_raw', 'not reported')}"
    )
    lines = row.get("evidence_lines")
    if isinstance(lines, str):
        line_label = lines
    elif lines is None:
        line_label = "not reported"
    else:
        try:
            line_label = ", ".join(str(value) for value in lines)
        except TypeError:
            line_label = str(lines)
    st.caption(
        f"Source document {row.get('doc_id', row.get('doc_id_paper', 'unknown'))}; "
        f"authoritative line reference(s): {line_label}."
    )


def _render_multiverse(bundle: DemoBundle) -> None:
    """Cross-family view: what the same corpus says beyond the primary endpoint.

    Only claim-grade rows (verbatim-grounded, unflagged section, independent-model agree)
    count; everything else is shown as excluded mass so scale is never overstated."""

    status_by_id = {
        str(decision.get("finding_id")): str(decision.get("model_status"))
        for decision in bundle.verification.get("decisions", [])
    }
    rows = bundle.findings.to_dict(orient="records")
    summary: dict[str, dict[str, int]] = {}
    excluded = 0
    for row in rows:
        direction = str(row.get("effect_direction"))
        if direction not in {"increase", "no_effect", "decrease"}:
            continue
        family = str(row.get("outcome_family") or "unmapped")
        claim_grade = (
            str(row.get("grounding_status")) == "exact"
            and not bool(row.get("section_flagged"))
            and status_by_id.get(str(row.get("finding_id"))) == "agree"
        )
        if not claim_grade:
            excluded += 1
            continue
        bucket = summary.setdefault(family, {"increase": 0, "no_effect": 0, "decrease": 0})
        bucket[direction] += 1
    if not summary:
        st.write("No claim-grade findings outside the exclusion sets.")
        return
    table = pd.DataFrame(
        [
            {"outcome family": family, **counts, "total": sum(counts.values())}
            for family, counts in sorted(summary.items())
        ]
    )
    st.dataframe(table, hide_index=True)
    st.caption(
        "Direction counts use only claim-grade rows: verbatim quote verified on the cited "
        "source lines, from an allowed section, with an independent model agreeing the quote "
        f"supports the direction. {excluded} grounded rows that failed any of those checks "
        "are excluded from this table and counted in the exclusions panel above. "
        "This cross-family view is descriptive context, not a gated claim."
    )


def render_app(bundle: DemoBundle) -> None:
    manifest = bundle.manifest
    st.set_page_config(page_title="Papertrail", layout="wide", page_icon="🧾")
    st.title("🧾 Papertrail")
    st.markdown(
        "**Turn an entire literature into one audited, provenance-locked dataset, "
        "and let pre-registered gates decide what it may claim. Every answer has a paper trail.**"
    )
    st.caption(
        f"{manifest['corpus_qualifier'].capitalize()} · frozen snapshot "
        f"{manifest['created_at']} · Variant {manifest['narrative_variant']}"
    )

    st.header("Question and corpus")
    st.subheader(str(manifest["spoken_question"]))
    st.write(str(manifest["research_question"]))
    _render_funnel(manifest)

    st.header("Grounding and exclusions")
    _render_quality(manifest)

    st.header("Baseline vs primary global direction")
    _render_baseline(bundle)

    st.header("Complete moderator analysis")
    _render_moderators(bundle)
    _render_pattern_or_residuals(bundle)

    st.header("Beyond the primary endpoint: the full multiverse")
    _render_multiverse(bundle)

    st.header("Source-grounded evidence")
    _render_evidence(bundle)

    with st.expander("Agent/remap trace"):
        st.json(dict(bundle.trace), expanded=False)
    with st.expander("Release lineage"):
        st.json(dict(bundle.release_selection), expanded=False)
    st.caption(f"Created at {manifest['created_at']} from a validated offline release.")


def main() -> None:
    try:
        demo_dir = select_demo_dir(sys.argv[1:])
        bundle = load_demo_bundle(demo_dir)
    except BundleValidationError as exc:
        st.set_page_config(page_title="Literature Multiverse", layout="wide")
        st.error(f"Frozen release validation failed: {exc}")
        st.stop()
    render_app(bundle)


if __name__ == "__main__":
    main()
