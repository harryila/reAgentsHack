"""Read-only HTTP API over a frozen Literature Multiverse release.

The API serves the same hash-verified bundle the dashboard renders: audited claim rows
with verbatim provenance, gate verdicts, the corpus funnel, and the cross-family
multiverse view. It also exposes the entry point for new questions: config validation
against the locked contract.

Run:  uv run uvicorn app.api:app --port 8799
Bundle directory comes from LM_DEMO_DIR (default: artifacts/antiox-training/demo).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from literature_multiverse.config import QuestionConfig

PRIMARY_DIRECTIONS = {"increase", "no_effect", "decrease"}

app = FastAPI(
    title="Papertrail API",
    description=(
        "Audited, provenance-locked literature datasets. Every claim row carries a "
        "verbatim quote and source line numbers, re-verified by code, census-audited "
        "by adversarial agents, and cross-examined by a second model. Pre-registered "
        "gates decide what each release may claim."
    ),
    version="1.0",
)


def _bundle_root() -> Path:
    return Path(os.environ.get("LM_DEMO_DIR", "artifacts/antiox-training/demo")).resolve()


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    root = _bundle_root()
    if not root.is_dir():
        raise RuntimeError(f"demo bundle missing: {root}")
    manifest = json.loads((root / "manifest.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    findings = pd.read_parquet(root / "findings.parquet").to_dict(orient="records")
    papers = pd.read_parquet(root / "papers.parquet").to_dict(orient="records")
    decisions = {
        str(d["finding_id"]): d for d in verification.get("decisions", [])
    }
    audit = json.loads((root / "audit.json").read_text())
    audited = {str(d["finding_id"]): d for d in audit.get("decisions", [])}
    g3 = json.loads((root / "g3_gate.json").read_text())
    return {
        "root": root,
        "manifest": manifest,
        "findings": findings,
        "papers": papers,
        "decisions": decisions,
        "audit": audited,
        "g3": g3,
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def _claim_grade(row: dict[str, Any], decisions: dict[str, Any]) -> bool:
    decision = decisions.get(str(row.get("finding_id")), {})
    verified = decision.get("model_status") == "agree" or decision.get("adjudication") == "accept"
    return (
        str(row.get("grounding_status")) == "exact"
        and not bool(row.get("section_flagged"))
        and str(row.get("effect_direction")) in PRIMARY_DIRECTIONS
        and verified
    )


def _claim_view(row: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    fid = str(row.get("finding_id"))
    decision = bundle["decisions"].get(fid)
    audit = bundle["audit"].get(fid)
    return _plain(
        {
            "finding_id": fid,
            "paper_id": row.get("paper_id"),
            "doc_id": row.get("doc_id"),
            "outcome_name": row.get("outcome_name"),
            "outcome_family": row.get("outcome_family"),
            "effect_direction": row.get("effect_direction"),
            "timepoint": row.get("timepoint_raw"),
            "dose": row.get("dose_raw"),
            "comparator": row.get("comparator"),
            "claim_grade": _claim_grade(row, bundle["decisions"]),
            "provenance": {
                "evidence_quote": row.get("evidence_quote"),
                "evidence_lines": row.get("evidence_lines"),
                "evidence_section": row.get("evidence_section"),
                "grounding_status": row.get("grounding_status"),
                "section_flagged": row.get("section_flagged"),
            },
            "verification": decision,
            "audit": audit,
            "moderators": row.get("moderators"),
        }
    )


@app.get("/")
def service_info() -> dict[str, Any]:
    bundle = _bundle()
    manifest = bundle["manifest"]
    return {
        "service": "papertrail",
        "questions": [manifest["question_id"]],
        "release": {
            "variant": manifest["narrative_variant"],
            "created_at": manifest["created_at"],
            "disposition": manifest["release_selection"]["disposition"],
        },
        "endpoints": [
            "/questions/{qid}",
            "/questions/{qid}/claims",
            "/questions/{qid}/claims/{finding_id}",
            "/questions/{qid}/gates",
            "/questions/{qid}/funnel",
            "/questions/{qid}/multiverse",
            "POST /questions/validate",
        ],
    }


def _require_question(qid: str) -> dict[str, Any]:
    bundle = _bundle()
    if qid != bundle["manifest"]["question_id"]:
        raise HTTPException(status_code=404, detail=f"unknown question: {qid}")
    return bundle


@app.get("/questions/{qid}")
def question_summary(qid: str) -> dict[str, Any]:
    bundle = _require_question(qid)
    manifest = bundle["manifest"]
    return _plain(
        {
            "question_id": qid,
            "spoken_question": manifest["spoken_question"],
            "research_question": manifest["research_question"],
            "variant": manifest["narrative_variant"],
            "funnel": manifest["paper_funnel"],
            "quality": manifest["quality"],
            "created_at": manifest["created_at"],
        }
    )


@app.get("/questions/{qid}/claims")
def claims(
    qid: str,
    family: str | None = None,
    direction: str | None = None,
    claim_grade: bool = True,
) -> dict[str, Any]:
    bundle = _require_question(qid)
    rows = []
    for row in bundle["findings"]:
        if claim_grade and not _claim_grade(row, bundle["decisions"]):
            continue
        if family and row.get("outcome_family") != family:
            continue
        if direction and row.get("effect_direction") != direction:
            continue
        rows.append(_claim_view(row, bundle))
    return {"question_id": qid, "count": len(rows), "claims": rows}


@app.get("/questions/{qid}/claims/{finding_id:path}")
def claim(qid: str, finding_id: str) -> dict[str, Any]:
    bundle = _require_question(qid)
    for row in bundle["findings"]:
        if str(row.get("finding_id")) == finding_id:
            return _claim_view(row, bundle)
    raise HTTPException(status_code=404, detail=f"unknown finding: {finding_id}")


@app.get("/questions/{qid}/gates")
def gates(qid: str) -> dict[str, Any]:
    bundle = _require_question(qid)
    return _plain(bundle["g3"])


@app.get("/questions/{qid}/funnel")
def funnel(qid: str) -> dict[str, Any]:
    bundle = _require_question(qid)
    manifest = bundle["manifest"]
    return _plain(
        {
            "question_id": qid,
            "funnel": manifest["paper_funnel"],
            "exclusions": manifest["exclusions"],
        }
    )


@app.get("/questions/{qid}/multiverse")
def multiverse(qid: str) -> dict[str, Any]:
    bundle = _require_question(qid)
    table: dict[str, dict[str, int]] = defaultdict(
        lambda: {"increase": 0, "no_effect": 0, "decrease": 0}
    )
    excluded = 0
    for row in bundle["findings"]:
        direction = str(row.get("effect_direction"))
        if direction not in PRIMARY_DIRECTIONS:
            continue
        if not _claim_grade(row, bundle["decisions"]):
            excluded += 1
            continue
        table[str(row.get("outcome_family") or "unmapped")][direction] += 1
    return {
        "question_id": qid,
        "claim_grade_by_family": dict(sorted(table.items())),
        "excluded_grounded_rows": excluded,
        "note": "claim-grade rows only: exact grounding, allowed section, second-model verified",
    }


class ConfigSubmission(BaseModel):
    config_yaml: str


@app.post("/questions/validate")
def validate_config(submission: ConfigSubmission) -> dict[str, Any]:
    """Validate a candidate question config against the locked contract.

    This is the same validation the pipeline enforces, so a config that passes here
    is runnable as-is: search families, outcome maps, moderators, gates and all.
    """

    try:
        raw = yaml.safe_load(submission.config_yaml)
    except yaml.YAMLError as exc:
        return {"valid": False, "errors": [f"yaml: {exc}"]}
    if not isinstance(raw, dict):
        return {"valid": False, "errors": ["config root must be a mapping"]}
    try:
        config = QuestionConfig.model_validate(raw)
    except ValidationError as exc:
        return {
            "valid": False,
            "errors": [
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            ],
        }
    return {
        "valid": True,
        "question_id": config.question_id,
        "status": config.status,
        "queries": sum(len(family.queries) for family in config.search.query_families),
        "moderators": [spec.name for spec in config.moderators],
        "next": "run scripts/s1_search.py --question <id> --all --live",
    }
