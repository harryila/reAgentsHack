"""Hash-bound, self-contained verification certificate artifacts.

The JSON certificate is the normative artifact.  The HTML file is a dependency-free
human-readable rendering of that exact JSON payload; it never fetches remote assets or
executes JavaScript.  A certificate can therefore be archived, inspected, and verified
without the application that created it.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.claim_release import ClaimReleaseAssessment
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel


class CertificateLineageStage(ContractModel):
    """One deterministic hand-off in the unified verifier."""

    stage: Annotated[str, Field(min_length=1)]
    input_sha256s: dict[str, str]
    output_sha256s: dict[str, str]
    method: Annotated[str, Field(min_length=1)]

    @field_validator("input_sha256s", "output_sha256s")
    @classmethod
    def validate_hash_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())):
            raise ValueError("certificate_lineage_hashes_must_be_sorted")
        if any(not SHA256_RE.fullmatch(digest) for digest in value.values()):
            raise ValueError("certificate_lineage_hash_invalid")
        return value


class VerificationCertificate(ContractModel):
    """Complete frozen record of one claim-verification run."""

    certificate_version: Literal["literature-multiverse-verification-v1"] = (
        "literature-multiverse-verification-v1"
    )
    run_id: Annotated[str, Field(pattern=r"^verify-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["released", "abstained"]
    reasons: list[str]
    claim_manifest: dict[str, Any]
    claim_manifest_sha256: str
    corpus: dict[str, Any]
    corpus_sha256: str
    evidence_graph: EvidenceGraph
    evidence_graph_sha256: str
    adapter_issues: list[dict[str, Any]]
    synthesis: dict[str, Any]
    synthesis_sha256: str
    counterfactual_reruns: list[dict[str, Any]]
    audit_candidates: list[dict[str, Any]]
    release_assessment: ClaimReleaseAssessment
    lineage: list[CertificateLineageStage]
    certificate_sha256: str
    interpretation: Literal[
        "literature-support verification under the declared corpus; not scientific truth"
    ] = "literature-support verification under the declared corpus; not scientific truth"

    @field_validator(
        "claim_manifest_sha256",
        "corpus_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("verification_certificate_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("verification_certificate_reasons_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> VerificationCertificate:
        if hash_canonical(self.claim_manifest) != self.claim_manifest_sha256:
            raise ValueError("verification_certificate_claim_manifest_hash_mismatch")
        if hash_canonical(self.evidence_graph) != self.evidence_graph_sha256:
            raise ValueError("verification_certificate_evidence_graph_hash_mismatch")
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("verification_certificate_synthesis_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if hash_canonical(payload) != self.certificate_sha256:
            raise ValueError("verification_certificate_hash_mismatch")
        if self.status == "released" and self.reasons:
            raise ValueError("released_verification_certificate_cannot_have_reasons")
        if self.status == "abstained" and not self.reasons:
            raise ValueError("abstained_verification_certificate_requires_reason")
        return self


class CertificateArtifacts(ContractModel):
    """Paths and byte hashes written by :func:`write_certificate_artifacts`."""

    json_path: str
    json_sha256: str
    html_path: str
    html_sha256: str

    @field_validator("json_sha256", "html_sha256")
    @classmethod
    def validate_artifact_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("verification_artifact_sha256_invalid")
        return value


def freeze_verification_certificate(
    *,
    generated_at: datetime,
    status: Literal["released", "abstained"],
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    counterfactual_reruns: list[dict[str, Any]],
    audit_candidates: list[dict[str, Any]],
    release_assessment: ClaimReleaseAssessment,
    lineage: list[CertificateLineageStage],
) -> VerificationCertificate:
    """Freeze and self-hash one complete verification result."""

    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("verification_certificate_generated_at_requires_timezone")
    generated_at_json = generated_at.isoformat()
    if generated_at_json.endswith("+00:00"):
        generated_at_json = f"{generated_at_json[:-6]}Z"
    claim_hash = hash_canonical(claim_manifest)
    graph_hash = hash_canonical(evidence_graph)
    synthesis_hash = hash_canonical(synthesis)
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": claim_hash,
            "corpus_sha256": corpus_sha256,
            "evidence_graph_sha256": graph_hash,
            "release_decision_sha256": release_assessment.decision_sha256,
        }
    )
    payload: dict[str, Any] = {
        "certificate_version": "literature-multiverse-verification-v1",
        "run_id": f"verify-{run_identity[:16]}",
        "generated_at": generated_at_json,
        "status": status,
        "reasons": sorted(set(reasons)),
        "claim_manifest": claim_manifest,
        "claim_manifest_sha256": claim_hash,
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "evidence_graph": evidence_graph,
        "evidence_graph_sha256": graph_hash,
        "adapter_issues": adapter_issues,
        "synthesis": synthesis,
        "synthesis_sha256": synthesis_hash,
        "counterfactual_reruns": counterfactual_reruns,
        "audit_candidates": audit_candidates,
        "release_assessment": release_assessment,
        "lineage": lineage,
        "interpretation": (
            "literature-support verification under the declared corpus; not scientific truth"
        ),
    }
    return VerificationCertificate.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return html.escape(str(value))


def _eligibility_rows(certificate: VerificationCertificate) -> str:
    rows = certificate.corpus.get("eligibility", [])
    if not isinstance(rows, list) or not rows:
        return '<tr><td colspan="4">No paper-level eligibility ledger was supplied.</td></tr>'
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rendered.append(
            "<tr>"
            f"<td>{_cell(row.get('paper_id'))}</td>"
            f"<td>{_cell(row.get('title'))}</td>"
            f"<td>{_cell(row.get('status'))}</td>"
            f"<td>{_cell(row.get('reason'))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _estimate_rows(certificate: VerificationCertificate) -> str:
    span_by_id = {span.span_id: span for span in certificate.evidence_graph.evidence_spans}
    rows: list[str] = []
    for estimate in certificate.evidence_graph.outcome_estimates:
        first_span = span_by_id[estimate.evidence_span_ids[0]]
        rows.append(
            "<tr>"
            f"<td>{_cell(estimate.estimate_id)}</td>"
            f"<td>{_cell(estimate.effect.paper_id)}</td>"
            f"<td>{_cell(estimate.outcome_name)}</td>"
            f"<td>{_cell(estimate.effect.estimate)}</td>"
            f"<td>{_cell(estimate.effect.effect_format)}</td>"
            f"<td>{_cell(first_span.source_locator)}</td>"
            f"<td>{_cell(first_span.quote)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _audit_rows(certificate: VerificationCertificate) -> str:
    rows: list[str] = []
    for item in certificate.release_assessment.audit.ranking:
        rows.append(
            "<tr>"
            f"<td>{item.rank}</td>"
            f"<td>{_cell(item.item_id)}</td>"
            f"<td>{_cell(item.selected_for_audit)}</td>"
            f"<td>{_cell(item.resolved_before_release)}</td>"
            f"<td>{item.probability_influence:.6f}</td>"
            f"<td>{item.expected_claim_loss_reduction_per_cost:.6f}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="6">No matching evidence items were available to rank.</td></tr>'
    return "".join(rows)


def render_certificate_html(certificate: VerificationCertificate) -> str:
    """Render a static HTML view containing the complete canonical certificate JSON."""

    manifest_claim = certificate.claim_manifest.get("claim", {})
    if not isinstance(manifest_claim, dict):
        manifest_claim = {}
    evidence = certificate.release_assessment.evidence
    reason_items = "".join(f"<li>{_cell(reason)}</li>" for reason in certificate.reasons)
    if not reason_items:
        reason_items = "<li>All declared gates passed.</li>"
    canonical = json.dumps(
        certificate.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    status_class = "released" if certificate.status == "released" else "abstained"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verification certificate — {html.escape(certificate.run_id)}</title>
  <style>
    :root {{ color-scheme: light dark; --ok:#0b6b42; --stop:#9b2c2c; --line:#8b8b8b55; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; line-height:1.45; margin:0 auto;
            max-width:1200px; padding:2rem; }}
    .banner {{ border-left:.55rem solid var(--stop); padding:1rem 1.2rem; background:#9b2c2c12; }}
    .banner.released {{ border-color:var(--ok); background:#0b6b4212; }}
    h1,h2 {{ line-height:1.15; }} h2 {{ margin-top:2rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th,td {{ border:1px solid var(--line); padding:.5rem; text-align:left; vertical-align:top; }}
    th {{ background:#80808014; }}
    code,pre {{ font-family:ui-monospace, SFMono-Regular, monospace; }}
    pre {{ overflow:auto; max-height:42rem; border:1px solid var(--line); padding:1rem; }}
    .hash {{ overflow-wrap:anywhere; }} .muted {{ opacity:.75; }}
  </style>
</head>
<body>
  <div class="banner {status_class}">
    <h1>{html.escape(certificate.status.upper())}</h1>
    <p><strong>{_cell(manifest_claim.get('statement'))}</strong></p>
    <p>This decision verifies support in the declared literature corpus; it is not a claim of
       scientific truth.</p>
  </div>

  <h2>Decision</h2>
  <ul>{reason_items}</ul>
  <table>
    <tr><th>Evidence classification</th><td>{_cell(evidence.classification)}</td></tr>
    <tr><th>Synthesis mode</th><td>{_cell(evidence.mode)}</td></tr>
    <tr><th>Papers synthesized</th><td>{evidence.n_papers}</td></tr>
    <tr><th>Audit budget / spent</th><td>{certificate.release_assessment.audit.budget:g} /
      {certificate.release_assessment.audit.spent:g}
      {_cell(certificate.release_assessment.audit.cost_unit)}</td></tr>
    <tr><th>Calibration</th><td>{_cell(certificate.release_assessment.calibration.status)} —
      {_cell(certificate.release_assessment.calibration.reason)}</td></tr>
  </table>

  <h2>Corpus eligibility</h2>
  <table><thead><tr><th>Paper</th><th>Title</th><th>Status</th><th>Reason</th></tr></thead>
    <tbody>{_eligibility_rows(certificate)}</tbody></table>

  <h2>Evidence graph</h2>
  <table><thead><tr><th>Estimate</th><th>Paper</th><th>Outcome</th><th>Value</th>
    <th>Format</th><th>Source</th><th>Grounding</th></tr></thead>
    <tbody>{_estimate_rows(certificate)}</tbody></table>

  <h2>Ranked verification actions</h2>
  <table><thead><tr><th>Rank</th><th>Evidence item</th><th>Selected</th><th>Resolved</th>
    <th>Influence</th><th>Expected loss reduction / minute</th></tr></thead>
    <tbody>{_audit_rows(certificate)}</tbody></table>

  <h2>Integrity</h2>
  <p class="hash"><strong>Certificate SHA-256:</strong> {certificate.certificate_sha256}</p>
  <p class="hash"><strong>Graph SHA-256:</strong> {certificate.evidence_graph_sha256}</p>
  <p class="hash"><strong>Synthesis SHA-256:</strong> {certificate.synthesis_sha256}</p>

  <details><summary><strong>Complete normative JSON payload</strong></summary>
    <pre>{html.escape(canonical)}</pre>
  </details>
  <p class="muted">Generated {html.escape(certificate.generated_at.isoformat())};
    no remote assets.</p>
</body>
</html>
"""


def write_certificate_artifacts(
    certificate: VerificationCertificate,
    output_dir: Path,
    *,
    force: bool = False,
) -> CertificateArtifacts:
    """Atomically write the normative JSON and its static HTML rendering."""

    json_path = output_dir / "verification-certificate.json"
    html_path = output_dir / "verification-certificate.html"
    if not force:
        existing = [path.as_posix() for path in (json_path, html_path) if path.exists()]
        if existing:
            raise FileExistsError(f"verification_certificate_outputs_exist:{existing}")
    rendered_html = render_certificate_html(certificate)
    atomic_write_json(json_path, certificate, force=force)
    atomic_write_text(html_path, rendered_html, force=force)
    return CertificateArtifacts(
        json_path=json_path.as_posix(),
        json_sha256=sha256_file(json_path),
        html_path=html_path.as_posix(),
        html_sha256=sha256_file(html_path),
    )


__all__ = [
    "CertificateArtifacts",
    "CertificateLineageStage",
    "VerificationCertificate",
    "freeze_verification_certificate",
    "render_certificate_html",
    "write_certificate_artifacts",
]
