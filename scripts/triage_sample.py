#!/usr/bin/env python3
"""Select the logged, stratified 10-paper triage probe sample and its exact provider set.

Selection is deterministic: included s2 papers are stratified across (query family,
publication era), shuffled inside each stratum with the config seed, and drawn round-robin.
The exact provider set is built with a dedicated paper repo — ``repo add`` accepts explicit
document IDs, and a ``--repo-only`` search over that repo yields one saved result whose CSV
export must equal the sampled set exactly.  (The build's ``merge`` command cannot see
server-side result IDs, so per-paper searches + merge are not a viable construction.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    atomic_write_jsonl,
    code_version,
    sha256_file,
    write_run_record,
)
from literature_multiverse.live import paperclip_executable, search_id_from_stdout
from literature_multiverse.models import RunRecord, UpstreamRef
from literature_multiverse.paperclip_cli import require_success, run_paperclip
from literature_multiverse.paths import PATHS
from literature_multiverse.search import parse_search_csv

PROBE_SIZE = 10


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument("--force", action="store_true")
    return result


def _era(pub_year: int | None) -> str:
    if pub_year is None:
        return "era-unknown"
    if pub_year < 2015:
        return "era-pre-2015"
    if pub_year < 2020:
        return "era-2015-2019"
    if pub_year < 2023:
        return "era-2020-2022"
    return "era-2023-plus"


def _strata(papers: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for paper in papers:
        families = paper.get("query_families") or ["family-unknown"]
        key = (str(sorted(families)[0]), _era(paper.get("pub_year")))
        grouped[key].append(paper)
    return grouped


def _deterministic_order(
    strata: dict[tuple[str, str], list[dict]], *, seed: int, question_id: str
) -> list[dict]:
    """Round-robin across sorted strata, seeded-shuffled within each stratum."""

    ordered_strata: list[list[dict]] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda paper: str(paper["paper_id"]))
        rng = random.Random(f"{seed}:{question_id}:{key[0]}:{key[1]}")
        rng.shuffle(members)
        ordered_strata.append(members)
    order: list[dict] = []
    depth = 0
    while any(depth < len(members) for members in ordered_strata):
        for members in ordered_strata:
            if depth < len(members):
                order.append(members[depth])
        depth += 1
    return order


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question)
    if config.status != "triage":
        raise ValueError("triage_sample_requires_status_triage_config")
    config.authorize_stage("s2", explicit_fixture=False, live_provider=True)

    screen_dir = PATHS.raw_screen_dir(config.question_id)
    screened_path = screen_dir / "screened_papers.jsonl"
    screened = [
        json.loads(line) for line in screened_path.read_text().splitlines() if line.strip()
    ]
    included = [paper for paper in screened if paper.get("screen_status") == "included"]
    if len(included) < PROBE_SIZE:
        raise ValueError(
            f"triage sample requires at least {PROBE_SIZE} included papers; "
            f"found {len(included)}"
        )

    # Config-declared probe relevance prefilter over title + retrieved abstract text.
    # The production corpus stays recall-heavy; this only decides which papers may spend
    # one of the ten probe slots.
    keyword_groups = [
        [term.casefold() for term in group] for group in config.triage.probe_keyword_groups
    ]
    prefilter_pool = len(included)
    if keyword_groups:
        candidates_path = PATHS.raw_search_dir(config.question_id) / "candidate_papers.jsonl"
        doc_text: dict[str, str] = {}
        for line in candidates_path.read_text().splitlines():
            if not line.strip():
                continue
            candidate = json.loads(line)
            metadata = candidate.get("raw_metadata") or {}
            text = " ".join(
                str(part)
                for part in (candidate.get("title"), metadata.get("abstract"))
                if part
            ).casefold()
            existing = doc_text.get(str(candidate["doc_id"]), "")
            if len(text) > len(existing):
                doc_text[str(candidate["doc_id"])] = text

        def _matches(paper: dict) -> bool:
            doc_ids = [str(paper["doc_id"]), *map(str, paper.get("alternate_doc_ids", []))]
            text = max(
                (doc_text.get(doc_id, "") for doc_id in doc_ids), key=len, default=""
            ) or str(paper.get("title", "")).casefold()
            return all(any(term in text for term in group) for group in keyword_groups)

        included = [paper for paper in included if _matches(paper)]
    exclude_terms = [term.casefold() for term in config.triage.probe_exclude_terms]
    if exclude_terms:
        included = [
            paper
            for paper in included
            if not any(term in str(paper.get("title", "")).casefold() for term in exclude_terms)
        ]
    if len(included) < PROBE_SIZE:
        raise ValueError(
            f"probe prefilter left {len(included)} of {prefilter_pool} papers; "
            f"at least {PROBE_SIZE} are required — loosen the triage probe filters"
        )

    destination = PATHS.triage_dir(config.question_id)
    destination.mkdir(parents=True, exist_ok=True)
    search_archive = destination / "sample_searches"
    order = _deterministic_order(
        _strata(included), seed=config.analysis.seed, question_id=config.question_id
    )
    chosen = order[:PROBE_SIZE]
    chosen_doc_ids = sorted(str(paper["doc_id"]) for paper in chosen)

    # The repo name is derived from the sample content, so a rerun with the same sample
    # reuses the same repo idempotently and a changed sample gets a fresh, distinct repo.
    sample_digest = hashlib.sha256("\n".join(chosen_doc_ids).encode()).hexdigest()[:10]
    repo_name = f"{config.question_id}-probe-{sample_digest}"
    executable = paperclip_executable()

    init_run = run_paperclip(
        [executable, "repo", "init", repo_name, "isolated triage probe set"],
        archive_dir=search_archive,
        archive_stem="repo-init",
        timeout_seconds=120,
        force=args.force,
    )
    init_text = (init_run.final.stdout + init_run.final.stderr).decode("utf-8", "replace")
    if not init_run.ok and "already exists" not in init_text.casefold():
        raise ValueError(f"probe repo init failed: see {init_run.final.stderr_path}")
    add_run = run_paperclip(
        [executable, "--repo", repo_name, "repo", "add", *chosen_doc_ids],
        archive_dir=search_archive,
        archive_stem="repo-add",
        timeout_seconds=300,
        force=args.force,
    )
    require_success(add_run)

    probe_query = config.research_question
    search_run = run_paperclip(
        [
            executable,
            "--repo",
            repo_name,
            "--repo-only",
            "search",
            "-s",
            ",".join(config.search.sources),
            "-n",
            "50",
            "--all",
            probe_query,
        ],
        archive_dir=search_archive,
        archive_stem="repo-probe-search",
        timeout_seconds=300,
        force=args.force,
    )
    search_attempt = require_success(search_run)
    probe_result_id = search_id_from_stdout(search_attempt.stdout)

    probe_csv = search_archive / "probe-set.results.csv"
    if probe_csv.exists() and not args.force:
        raise FileExistsError(f"refusing to replace {probe_csv}")
    export_run = run_paperclip(
        [executable, "results", probe_result_id, "--save", str(probe_csv)],
        archive_dir=search_archive,
        archive_stem="probe-set.export",
        timeout_seconds=300,
        force=args.force,
    )
    require_success(export_run)
    observed_ids = sorted(
        str(record["doc_id"]) for record in parse_search_csv(probe_csv.read_bytes())
    )
    if observed_ids != chosen_doc_ids:
        raise ValueError(
            "repo-scoped provider set does not equal the sampled probe set: "
            f"expected={chosen_doc_ids} observed={observed_ids}"
        )

    probe_papers_path = destination / "probe_papers.jsonl"
    atomic_write_jsonl(probe_papers_path, chosen, force=args.force)
    strata_summary = {
        f"{family}|{era}": len(members)
        for (family, era), members in sorted(_strata(included).items())
    }
    sample_log = {
        "log_version": "2",
        "question_id": config.question_id,
        "seed": config.analysis.seed,
        "included_paper_count": prefilter_pool,
        "probe_keyword_groups": config.triage.probe_keyword_groups,
        "prefiltered_paper_count": len(included),
        "strata": strata_summary,
        "sampled_paper_ids": [paper["paper_id"] for paper in chosen],
        "sampled_doc_ids": chosen_doc_ids,
        "probe_repo": repo_name,
        "probe_result_id": probe_result_id,
        "probe_verified_doc_ids": observed_ids,
    }
    sample_log_path = destination / "probe_sample.json"
    atomic_write_json(sample_log_path, sample_log, force=args.force)

    s2_run_path = PATHS.run_record_path(config.question_id, "s2")
    s2_run = RunRecord.model_validate_json(s2_run_path.read_text())
    record = RunRecord(
        run_id=f"triage-sample-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s2",
        stage_version="1-sample",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/triage_sample.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(PATHS.config_path(config.question_id)),
        config_sha256=config_sha256(config),
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[
            UpstreamRef(
                stage="s2",
                run_id=s2_run.run_id,
                run_record_path=PATHS.repository_relative(s2_run_path),
                run_record_sha256=sha256_file(s2_run_path),
            )
        ],
        inputs=[artifact_ref(screened_path, root=PATHS.root, rows=len(screened))],
        outputs=[
            artifact_ref(probe_papers_path, root=PATHS.root, rows=len(chosen)),
            artifact_ref(sample_log_path, root=PATHS.root),
        ],
        external_result_ids={"paperclip": [probe_result_id]},
        counts={
            "included_papers": len(included),
            "sampled_papers": len(chosen),
        },
        warnings=["isolated_triage_artifacts_forbidden_as_production_upstream"],
    )
    # The sampler is a logged s2-substage; its record lives beside the triage artifacts
    # rather than under a production stage directory.
    write_run_record(destination / "sample_run.json", record, force=args.force)
    print(
        f"triage sample complete: {len(chosen)} papers, repo {repo_name}, "
        f"provider set {probe_result_id} verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
