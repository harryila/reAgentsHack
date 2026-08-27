"""Evidence Inference 2.0 conversion for leakage-safe prompt optimization.

The upstream archive stores question metadata, physician annotations, official
article-level splits, and plain-text articles separately.  This module preserves those
boundaries: prompt replacements come only from the question table, source lines come
only from the article, and annotation fields are written only to ``expected_output``.

Only verified, label-consistent examples with an exactly grounded BODY.RESULTS evidence
span are retained.  This is intentionally a conservative extraction/grounding benchmark,
not a claim that every row in the upstream dataset is error-free.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from literature_multiverse.grounding import normalize_evidence_text, quote_content_contained
from literature_multiverse.lineage import (
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    OptimizationSplitManifest,
    SplitArtifact,
)

OfficialSplit = Literal["train", "dev", "test"]

CONVERTER_VERSION = "evidence-inference-2.0-v1"
OFFICIAL_DOWNLOAD_URL = "https://evidence-inference.ebm-nlp.com/v2.0.tar.gz"
OFFICIAL_DATA_PAGE = "https://evidence-inference.ebm-nlp.com/download/"
OFFICIAL_CODE_URL = "https://github.com/jayded/evidence-inference"
OFFICIAL_PAPER_URL = "https://aclanthology.org/2020.bionlp-1.13/"

PROMPT_HEADERS = ("PromptID", "PMCID", "Outcome", "Intervention", "Comparator")
ANNOTATION_HEADERS = (
    "UserID",
    "PromptID",
    "PMCID",
    "Valid Label",
    "Valid Reasoning",
    "Label",
    "Annotations",
    "Label Code",
    "In Abstract",
    "Evidence Start",
    "Evidence End",
)
SPLIT_FILES: dict[OfficialSplit, str] = {
    "train": "train_article_ids.txt",
    "dev": "validation_article_ids.txt",
    "test": "test_article_ids.txt",
}
LABEL_CODE_TO_DIRECTION = {"-1": "decrease", "0": "no_effect", "1": "increase"}
FLAGGED_README_HEADINGS = ("Incorrect", "Questionable", "Somewhat malformed")
_UNICODE_LINE_ENDINGS = "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"

EVIDENCE_INFERENCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "eligible": {"type": "boolean"},
        "findings": {
            "type": "array",
            "minItems": 0,
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["increase", "no_effect", "decrease"],
                    },
                    "evidence_quote": {"type": "string", "minLength": 1},
                    "evidence_lines": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "pattern": r"^L[1-9][0-9]*(?:-L[1-9][0-9]*)?$",
                        },
                    },
                },
                "required": ["direction", "evidence_quote", "evidence_lines"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["eligible", "findings"],
    "additionalProperties": False,
}


class EvidenceInferenceContractError(ValueError):
    """Raised when an upstream dataset or generated benchmark violates the contract."""


@dataclass(frozen=True, slots=True)
class EvidenceInferenceConversion:
    """Paths and counts for one immutable conversion bundle."""

    output_dir: Path
    manifest_path: Path
    report_path: Path
    rows: dict[OfficialSplit, int]


def write_evidence_inference_metadata_summary(
    manifest_path: str | Path,
    report_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write a trackable benchmark summary containing identifiers but no article text."""

    manifest_source = Path(manifest_path)
    report_source = Path(report_path)
    try:
        manifest = OptimizationSplitManifest.model_validate_json(
            manifest_source.read_text(encoding="utf-8")
        )
        report = json.loads(report_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceContractError("cannot load conversion metadata") from exc
    splits: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        artifact = getattr(manifest, split)
        report_split = report["splits"][split]
        splits[split] = {
            "rows": artifact.rows,
            "papers": len(artifact.paper_ids),
            "example_ids": artifact.example_ids,
            "paper_ids": artifact.paper_ids,
            "jsonl_sha256": artifact.sha256,
            "direction_counts": report_split["direction_counts"],
        }
    summary = {
        "benchmark_summary_version": "1",
        "converter_version": report["converter_version"],
        "dataset": report["dataset"],
        "official_sources": report["official_sources"],
        "official_split_files": report["official_split_files"],
        "official_input_hashes": report["input_hashes"],
        "split_policy": report["split_policy"],
        "manifest_sha256": report["manifest_sha256"],
        "selected_text_corpus_sha256": report["selected_text_corpus_sha256"],
        "filters": report["filters"],
        "excluded_prompt_counts": report["excluded_prompt_counts"],
        "splits": splits,
        "contains_article_text": False,
        "contains_per_example_labels": False,
        "license_caveat": report["license_caveat"],
    }
    target = Path(output_path)
    atomic_write_json(target, summary)
    return target


@dataclass(frozen=True, slots=True)
class _SourceLine:
    number: int
    start: int
    end: int
    text: str
    section: str


@dataclass(frozen=True, slots=True)
class _QualifiedPrompt:
    prompt: Mapping[str, str]
    annotation: Mapping[str, str]
    split: OfficialSplit
    quote: str
    line_numbers: tuple[int, ...]


def _read_csv(path: Path, required_headers: Sequence[str]) -> list[dict[str, str]]:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise EvidenceInferenceContractError(f"cannot read required file: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        observed = tuple(reader.fieldnames or ())
        missing = sorted(set(required_headers) - set(observed))
        if missing:
            raise EvidenceInferenceContractError(
                f"{path.name} is missing required columns: {missing}"
            )
        return [{key: value or "" for key, value in row.items()} for row in reader]


def _safe_integer_id(value: str, *, field: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise EvidenceInferenceContractError(f"invalid positive integer {field}: {value!r}")
    return int(value)


def _parse_truth(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes"}


def _officially_flagged_prompt_ids(readme_path: Path) -> set[str]:
    """Read the three noisy-prompt lists published in the upstream README."""

    try:
        text = readme_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EvidenceInferenceContractError(f"cannot read required file: {readme_path}") from exc
    flagged: set[str] = set()
    for heading in FLAGGED_README_HEADINGS:
        match = re.search(
            rf"^###\s+{re.escape(heading)}\s*:\s*$\n(?P<body>.*?)(?=^###\s+|\Z)",
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if match is not None:
            flagged.update(re.findall(r"\b[1-9][0-9]*\b", match.group("body")))
    return flagged


def _canonical_section(heading: str) -> str:
    upper = heading.upper()
    if upper.startswith("RESULT"):
        return "Results"
    if upper.startswith("METHOD"):
        return "Methods"
    if upper.startswith("DISCUSSION"):
        return "Discussion"
    if upper.startswith("CONCLUSION"):
        return "Conclusion"
    if upper.startswith("INTRODUCTION") or upper.startswith("BACKGROUND"):
        return "Introduction"
    if upper.startswith("REFERENCE"):
        return "References"
    return "Other"


def _source_lines(text: str) -> list[_SourceLine]:
    """Map raw character offsets to stable non-empty source-line numbers."""

    lines: list[_SourceLine] = []
    current_section = "Abstract"
    character_offset = 0
    for raw_line in text.splitlines(keepends=True):
        # ``str.splitlines`` recognizes Unicode separators such as U+2028.  Leaving one
        # attached to ``content`` would make a valid JSON string look like two records to
        # JSONL readers that also use ``splitlines`` (the upstream corpus contains U+2028).
        content = raw_line.rstrip(_UNICODE_LINE_ENDINGS)
        start = character_offset
        end = start + len(content) - 1
        character_offset += len(raw_line)
        stripped = content.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("ABSTRACT.") and upper.endswith(":"):
            current_section = "Abstract"
        elif upper.startswith("BODY.") and upper.endswith(":"):
            top_level = upper[5:-1].split(".", maxsplit=1)[0]
            current_section = _canonical_section(top_level)
        lines.append(
            _SourceLine(
                number=len(lines) + 1,
                start=start,
                end=end,
                text=content,
                section=current_section,
            )
        )
    return lines


def _normalized_contains(quote: str, cited_text: str) -> bool:
    normalized_quote = normalize_evidence_text(quote)
    normalized_cited = normalize_evidence_text(cited_text)
    return quote_content_contained(normalized_quote, normalized_cited)


def _candidate_for_annotation(
    annotation: Mapping[str, str],
    *,
    source_text: str,
    source_lines: Sequence[_SourceLine],
) -> tuple[str, tuple[int, ...]] | None:
    try:
        start = int(annotation["Evidence Start"])
        end = int(annotation["Evidence End"])
    except (KeyError, TypeError, ValueError):
        return None
    if start < 0 or end < start or end >= len(source_text):
        return None
    overlapping = tuple(
        line for line in source_lines if line.start <= end and line.end >= start
    )
    if not overlapping or any(line.section != "Results" for line in overlapping):
        return None
    quote = annotation.get("Annotations", "").strip()
    if not quote:
        return None
    cited_text = " ".join(line.text for line in overlapping)
    if not _normalized_contains(quote, cited_text):
        return None
    return quote, tuple(line.number for line in overlapping)


def _read_official_splits(dataset_root: Path) -> dict[str, OfficialSplit]:
    lookup: dict[str, OfficialSplit] = {}
    for split, filename in SPLIT_FILES.items():
        path = dataset_root / "splits" / filename
        try:
            article_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        except OSError as exc:
            raise EvidenceInferenceContractError(f"cannot read required split: {path}") from exc
        if not article_ids:
            raise EvidenceInferenceContractError(f"official split is empty: {path}")
        for article_id in article_ids:
            _safe_integer_id(article_id, field="official split PMCID")
            previous = lookup.get(article_id)
            if previous is not None:
                raise EvidenceInferenceContractError(
                    f"official paper split leakage: PMC{article_id} in {previous} and {split}"
                )
            lookup[article_id] = split
    return lookup


def _line_reference(numbers: Sequence[int]) -> list[str]:
    if not numbers:
        raise EvidenceInferenceContractError("gold evidence has no source lines")
    expected = tuple(range(numbers[0], numbers[-1] + 1))
    if tuple(numbers) != expected:
        raise EvidenceInferenceContractError("gold evidence source lines are not contiguous")
    if len(numbers) == 1:
        return [f"L{numbers[0]}"]
    return [f"L{numbers[0]}-L{numbers[-1]}"]


def _prompt_replacements(prompt: Mapping[str, str]) -> dict[str, str]:
    """Return model-facing values sourced exclusively from ``prompts_merged.csv``."""

    return {
        "COMPARATOR": prompt["Comparator"].strip(),
        "INTERVENTION": prompt["Intervention"].strip(),
        "OUTCOME": prompt["Outcome"].strip(),
    }


def _content_lines(lines: Sequence[_SourceLine]) -> dict[str, dict[str, str | int]]:
    """Expose every Results line, never a gold-selected context window."""

    return {
        f"L{line.number}": {
            "line_number": line.number,
            "section": line.section,
            "text": line.text,
        }
        for line in lines
        if line.section == "Results"
    }


def _optimization_example(
    qualified: _QualifiedPrompt, *, source_lines: Sequence[_SourceLine]
) -> OptimizationExample:
    prompt_id = qualified.prompt["PromptID"]
    pmcid = qualified.prompt["PMCID"]
    direction = LABEL_CODE_TO_DIRECTION[qualified.annotation["Label Code"]]
    paper_id = f"PMC{pmcid}"
    return OptimizationExample(
        example_id=f"ei2-prompt-{prompt_id}",
        paper_id=paper_id,
        group_id=paper_id,
        prompt_kind="extraction",
        replacements=_prompt_replacements(qualified.prompt),
        expected_output={
            "eligible": True,
            "findings": [
                {
                    "direction": direction,
                    "evidence_quote": qualified.quote,
                    "evidence_lines": _line_reference(qualified.line_numbers),
                }
            ],
        },
        # Direction is the unique scientific target.  Evidence text may have several
        # equally valid spans, so it is assessed mechanically by the grounding objective
        # rather than by brittle exact equality to one physician-selected span.
        label_paths=["/findings/0/direction"],
        output_schema=EVIDENCE_INFERENCE_OUTPUT_SCHEMA,
        content_lines=_content_lines(source_lines),
        source_accessible=True,
    )


def _qualified_prompts(
    dataset_root: Path,
    *,
    include_flagged: bool,
) -> tuple[list[_QualifiedPrompt], Counter[str], set[str]]:
    prompts = _read_csv(dataset_root / "prompts_merged.csv", PROMPT_HEADERS)
    annotations = _read_csv(dataset_root / "annotations_merged.csv", ANNOTATION_HEADERS)
    split_lookup = _read_official_splits(dataset_root)
    flagged = _officially_flagged_prompt_ids(dataset_root / "README.md")

    prompts_by_id: dict[str, dict[str, str]] = {}
    prompts_by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for prompt in prompts:
        prompt_id = prompt["PromptID"].strip()
        pmcid = prompt["PMCID"].strip()
        _safe_integer_id(prompt_id, field="PromptID")
        _safe_integer_id(pmcid, field="PMCID")
        if prompt_id in prompts_by_id:
            raise EvidenceInferenceContractError(f"duplicate PromptID: {prompt_id}")
        if any(not prompt[field].strip() for field in ("Outcome", "Intervention", "Comparator")):
            raise EvidenceInferenceContractError(
                f"blank model-facing field for PromptID {prompt_id}"
            )
        prompts_by_id[prompt_id] = prompt
        prompts_by_paper[pmcid].append(prompt)

    annotations_by_prompt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for annotation in annotations:
        prompt_id = annotation["PromptID"].strip()
        prompt = prompts_by_id.get(prompt_id)
        if prompt is None:
            continue
        if annotation["PMCID"].strip() != prompt["PMCID"].strip():
            raise EvidenceInferenceContractError(
                f"annotation/prompt PMCID mismatch for PromptID {prompt_id}"
            )
        annotations_by_prompt[prompt_id].append(annotation)

    exclusions: Counter[str] = Counter()
    qualified: list[_QualifiedPrompt] = []
    selected_papers: set[str] = set()
    for pmcid in sorted(prompts_by_paper, key=int):
        split = split_lookup.get(pmcid)
        if split is None:
            exclusions["paper_missing_from_official_split"] += len(prompts_by_paper[pmcid])
            continue
        text_path = dataset_root / "txt_files" / f"PMC{pmcid}.txt"
        try:
            source_text = text_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            exclusions["missing_or_invalid_source_text"] += len(prompts_by_paper[pmcid])
            continue
        if not source_text.strip():
            exclusions["missing_or_invalid_source_text"] += len(prompts_by_paper[pmcid])
            continue
        source_lines = _source_lines(source_text)
        for prompt in sorted(prompts_by_paper[pmcid], key=lambda item: int(item["PromptID"])):
            prompt_id = prompt["PromptID"]
            if prompt_id in flagged and not include_flagged:
                exclusions["officially_flagged_prompt"] += 1
                continue
            accepted = [
                annotation
                for annotation in annotations_by_prompt.get(prompt_id, [])
                if _parse_truth(annotation["Valid Label"])
                and _parse_truth(annotation["Valid Reasoning"])
                and annotation["Label Code"] in LABEL_CODE_TO_DIRECTION
            ]
            if not accepted:
                exclusions["no_verified_label_and_reasoning"] += 1
                continue
            label_codes = {annotation["Label Code"] for annotation in accepted}
            if len(label_codes) != 1:
                exclusions["verified_label_disagreement"] += 1
                continue
            candidates: list[tuple[str, tuple[int, ...], Mapping[str, str]]] = []
            for annotation in accepted:
                candidate = _candidate_for_annotation(
                    annotation,
                    source_text=source_text,
                    source_lines=source_lines,
                )
                if candidate is not None:
                    quote, line_numbers = candidate
                    candidates.append((quote, line_numbers, annotation))
            if not candidates:
                exclusions["no_exact_body_results_evidence"] += 1
                continue
            quote, line_numbers, annotation = min(
                candidates,
                key=lambda item: (
                    len(normalize_evidence_text(item[0])),
                    int(item[2]["Evidence Start"]),
                    item[2]["UserID"],
                ),
            )
            qualified.append(
                _QualifiedPrompt(
                    prompt=prompt,
                    annotation=annotation,
                    split=split,
                    quote=quote,
                    line_numbers=line_numbers,
                )
            )
            selected_papers.add(pmcid)
    return qualified, exclusions, selected_papers


def _stream_jsonl(path: Path, rows: Iterable[OptimizationExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    count = 0
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(row) + b"\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _selected_text_sha256(dataset_root: Path, paper_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for pmcid in sorted(set(paper_ids), key=int):
        payload = (dataset_root / "txt_files" / f"PMC{pmcid}.txt").read_bytes()
        identifier = f"PMC{pmcid}".encode("ascii")
        digest.update(len(identifier).to_bytes(8, "big"))
        digest.update(identifier)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _input_hashes(dataset_root: Path) -> dict[str, str]:
    paths = {
        "README.md": dataset_root / "README.md",
        "prompts_merged.csv": dataset_root / "prompts_merged.csv",
        "annotations_merged.csv": dataset_root / "annotations_merged.csv",
        **{
            f"splits/{filename}": dataset_root / "splits" / filename
            for filename in SPLIT_FILES.values()
        },
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    archive = dataset_root / "v2.0.tar.gz"
    if archive.is_file():
        hashes["v2.0.tar.gz"] = sha256_file(archive)
    return hashes


def _split_artifact(path: Path, examples: Sequence[_QualifiedPrompt]) -> SplitArtifact:
    return SplitArtifact(
        path=path.name,
        sha256=sha256_file(path),
        rows=len(examples),
        example_ids=sorted(f"ei2-prompt-{row.prompt['PromptID']}" for row in examples),
        paper_ids=sorted({f"PMC{row.prompt['PMCID']}" for row in examples}),
        group_ids=sorted({f"PMC{row.prompt['PMCID']}" for row in examples}),
    )


def _smoke_sample(
    rows: Sequence[_QualifiedPrompt], *, split: OfficialSplit, limit: int
) -> list[_QualifiedPrompt]:
    """Select a reproducible, label-blind, paper-diverse subset for cheap wiring runs."""

    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{CONVERTER_VERSION}:{split}:PMC{row.prompt['PMCID']}:"
                f"{row.prompt['PromptID']}".encode("ascii")
            ).hexdigest(),
            int(row.prompt["PromptID"]),
        ),
    )
    selected: list[_QualifiedPrompt] = []
    used_papers: set[str] = set()
    for row in ranked:
        pmcid = row.prompt["PMCID"]
        if pmcid in used_papers:
            continue
        selected.append(row)
        used_papers.add(pmcid)
        if len(selected) == limit:
            return selected
    selected_ids = {row.prompt["PromptID"] for row in selected}
    for row in ranked:
        if row.prompt["PromptID"] in selected_ids:
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def convert_evidence_inference(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    include_flagged: bool = False,
    max_examples_per_split: int | None = None,
) -> EvidenceInferenceConversion:
    """Convert official Evidence Inference 2.0 files into GEPA split artifacts.

    The official article-level split is authoritative.  ``max_examples_per_split`` takes
    a deterministic, label-blind, paper-diverse subset within each official split and is
    intended for smoke tests; it never reassigns an article to another split.
    """

    source = Path(dataset_root)
    destination = Path(output_dir)
    if not source.is_dir():
        raise EvidenceInferenceContractError(f"dataset root does not exist: {source}")
    if destination.exists():
        raise EvidenceInferenceContractError(f"output directory already exists: {destination}")
    if max_examples_per_split is not None and max_examples_per_split < 1:
        raise ValueError("max_examples_per_split must be positive")

    qualified, exclusions, _all_selected_papers = _qualified_prompts(
        source, include_flagged=include_flagged
    )
    selected: dict[OfficialSplit, list[_QualifiedPrompt]] = {}
    for split in ("train", "dev", "test"):
        rows = sorted(
            (row for row in qualified if row.split == split),
            key=lambda row: int(row.prompt["PromptID"]),
        )
        if max_examples_per_split is not None:
            rows = _smoke_sample(
                rows,
                split=split,
                limit=max_examples_per_split,
            )
        if not rows:
            raise EvidenceInferenceContractError(
                f"no benchmark examples remain in official {split} split"
            )
        selected[split] = rows

    destination.mkdir(parents=True, exist_ok=False)
    split_paths = {split: destination / f"{split}.jsonl" for split in selected}
    for split, rows in selected.items():
        cached_pmcid: str | None = None
        cached_lines: list[_SourceLine] = []

        def examples(
            split_rows: Sequence[_QualifiedPrompt] = rows,
        ) -> Iterable[OptimizationExample]:
            nonlocal cached_pmcid, cached_lines
            for row in sorted(
                split_rows,
                key=lambda item: (int(item.prompt["PMCID"]), int(item.prompt["PromptID"])),
            ):
                pmcid = row.prompt["PMCID"]
                if pmcid != cached_pmcid:
                    source_text = (
                        source / "txt_files" / f"PMC{pmcid}.txt"
                    ).read_bytes().decode("utf-8")
                    cached_lines = _source_lines(source_text)
                    cached_pmcid = pmcid
                yield _optimization_example(row, source_lines=cached_lines)

        observed_rows = _stream_jsonl(split_paths[split], examples())
        if observed_rows != len(rows):  # pragma: no cover - defensive stream invariant
            raise EvidenceInferenceContractError(f"{split} output row count drift")

    artifacts = {
        split: _split_artifact(split_paths[split], rows)
        for split, rows in selected.items()
    }
    total_rows = sum(len(rows) for rows in selected.values())
    train_fraction = len(selected["train"]) / total_rows
    dev_fraction = len(selected["dev"]) / total_rows
    if not math.isfinite(train_fraction + dev_fraction) or train_fraction + dev_fraction >= 1:
        raise EvidenceInferenceContractError("generated official split fractions are invalid")
    source_digest = hash_canonical(
        {split: artifacts[split].sha256 for split in ("train", "dev", "test")}
    )
    manifest = OptimizationSplitManifest(
        algorithm="official-paper-groups-v1",
        seed=0,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
        source_examples_sha256=source_digest,
        train=artifacts["train"],
        dev=artifacts["dev"],
        test=artifacts["test"],
    )
    manifest_path = destination / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    selected_papers = {
        row.prompt["PMCID"] for rows in selected.values() for row in rows
    }
    split_summary: dict[str, Any] = {}
    for split, rows in selected.items():
        split_summary[split] = {
            "rows": len(rows),
            "papers": len({row.prompt["PMCID"] for row in rows}),
            "direction_counts": dict(
                sorted(
                    Counter(
                        LABEL_CODE_TO_DIRECTION[row.annotation["Label Code"]] for row in rows
                    ).items()
                )
            ),
            "jsonl_sha256": artifacts[split].sha256,
        }
    report = {
        "conversion_report_version": "1",
        "converter_version": CONVERTER_VERSION,
        "dataset": "Evidence Inference 2.0",
        "official_sources": {
            "data_page": OFFICIAL_DATA_PAGE,
            "download": OFFICIAL_DOWNLOAD_URL,
            "code": OFFICIAL_CODE_URL,
            "paper": OFFICIAL_PAPER_URL,
        },
        "input_hashes": _input_hashes(source),
        "selected_text_corpus_sha256": _selected_text_sha256(source, selected_papers),
        "manifest_sha256": sha256_file(manifest_path),
        "official_split_files": dict(SPLIT_FILES),
        "split_policy": "preserve official article-level train/validation/test membership",
        "sampling": {
            "max_examples_per_split": max_examples_per_split,
            "deterministic": True,
            "policy": "label-blind hash ranking with distinct papers first",
        },
        "filters": {
            "require_verified_label": True,
            "require_verified_reasoning": True,
            "require_unanimous_verified_label_code": True,
            "require_exact_body_results_grounding": True,
            "exclude_upstream_readme_flagged_prompts": not include_flagged,
        },
        "excluded_prompt_counts": dict(sorted(exclusions.items())),
        "splits": split_summary,
        "model_input_boundary": {
            "replacements_source": "prompts_merged.csv only",
            "source_lines_source": "all BODY.RESULTS lines from the article; never a gold window",
            "annotation_fields_rendered_to_model": False,
            "labels_location": "expected_output and label_paths only",
        },
        "scoring": {
            "exact_label_paths": ["/findings/0/direction"],
            "evidence_policy": (
                "gold span retained for audit; generated evidence assessed by schema and "
                "mechanical source-line grounding, not equality to one valid span"
            ),
        },
        "license_caveat": (
            "The associated GitHub code repository is MIT-licensed, but the v2.0 data "
            "archive contains no LICENSE file and bundles PMC article text whose reuse "
            "terms may vary. Keep text-bearing conversions local until redistribution "
            "rights are confirmed."
        ),
    }
    report_path = destination / "conversion_report.json"
    atomic_write_json(report_path, report)
    return EvidenceInferenceConversion(
        output_dir=destination,
        manifest_path=manifest_path,
        report_path=report_path,
        rows={split: len(rows) for split, rows in selected.items()},
    )


__all__ = [
    "CONVERTER_VERSION",
    "EVIDENCE_INFERENCE_OUTPUT_SCHEMA",
    "EvidenceInferenceContractError",
    "EvidenceInferenceConversion",
    "convert_evidence_inference",
    "write_evidence_inference_metadata_summary",
]
