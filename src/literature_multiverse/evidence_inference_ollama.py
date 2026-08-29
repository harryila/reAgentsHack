"""Staged offline Ollama diagnostic for Evidence Inference 2.0.

This module deliberately isolates three stages:

1. prepare a label-stripped, hash-bound projection of the provider-call-unseen papers;
2. run a deterministic localhost model and freeze per-row prediction receipts;
3. validate the frozen ledger *before* opening the co-located benchmark labels.

The test labels in this repository were previously opened.  Every result is therefore a
non-pristine diagnostic.  This is a small local-model baseline, not a GEPA improvement,
not semantic entailment evaluation, and not native numerical effect extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from jsonschema.validators import validator_for

import literature_multiverse.evidence_inference_diagnostic as lexical_diagnostic
from literature_multiverse.evidence_inference import EVIDENCE_INFERENCE_OUTPUT_SCHEMA
from literature_multiverse.evidence_inference_diagnostic import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    article_clustered_interval,
    score_diagnostic_output,
    summarize_scored_rows,
    validate_diagnostic_report,
)
from literature_multiverse.evidence_inference_diagnostic import (
    validate_prediction_ledger as validate_lexical_prediction_ledger,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaClientProtocol,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    OptimizationSplitManifest,
    load_manifest_split,
)

OLLAMA_DIAGNOSTIC_VERSION = "evidence-inference-local-ollama-diagnostic-v1"
INPUT_BUNDLE_VERSION = "evidence-inference-local-ollama-input-bundle-v1"
PREDICTION_RECEIPT_VERSION = "evidence-inference-local-ollama-receipt-v1"
PREDICTION_LEDGER_VERSION = "evidence-inference-local-ollama-ledger-v1"
PRIVATE_REPORT_VERSION = "evidence-inference-local-ollama-report-v1"
PUBLIC_SUMMARY_VERSION = "evidence-inference-local-ollama-public-summary-v1"
RETRIEVAL_ALGORITHM = "fixed-results-passage-projection-v1"
GENERATION_SCHEMA_ALGORITHM = "row-line-id-enumerated-json-schema-v1"

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_MODEL_DIGEST = (
    "baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878"
)
DEFAULT_OLLAMA_VERSION = "0.15.1"
DEFAULT_MODEL_PARAMETER_SIZE = "1.2B"

DEFAULT_GENERATION_CONFIG = OllamaGenerationConfig(
    model=DEFAULT_MODEL,
    model_digest=DEFAULT_MODEL_DIGEST,
    expected_ollama_version=DEFAULT_OLLAMA_VERSION,
    seed=20260827,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    num_ctx=8192,
    num_predict=384,
    keep_alive="30m",
)

DEFAULT_RETRIEVAL_CONFIG: dict[str, Any] = {
    "algorithm": RETRIEVAL_ALGORITHM,
    "allowed_section": "Results",
    "max_passages": 6,
    "max_total_characters": 9000,
    "max_passage_characters": 1800,
    "minimum_passage_characters": 24,
    "outcome_term_weight": 6.0,
    "outcome_phrase_weight": 3.0,
    "intervention_term_weight": 1.0,
    "comparator_term_weight": 1.0,
    "statistical_signal_weight": 1.0,
}

_METRIC_FIELDS = (
    "exact_structured_output_validity",
    "task_shape_consistency",
    "direction_accuracy",
    "formal_quote_line_provenance_validity",
    "schema_direction_provenance_joint_validity",
    "gold_evidence_line_agreement",
    "gold_evidence_quote_token_agreement_f1",
)
_SAFE_LINE_ID = re.compile(r"^L[1-9][0-9]*$")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STATISTICAL_SIGNAL = re.compile(
    r"\b(?:significant|significantly|higher|lower|increase|decrease|reduced|"
    r"difference|similar|comparable|confidence interval|odds ratio|risk ratio)\b|"
    r"\bp\s*[<=>\u2264\u2265]",
    flags=re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "versus",
        "vs",
        "was",
        "were",
        "with",
    }
)
_FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "expected_output",
        "label_paths",
        "expected_direction",
        "gold_direction",
        "gold_label",
        "gold_quote",
        "gold_lines",
        "annotations",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "rows_by_id",
        "example_id",
        "paper_id",
        "raw_prediction",
        "parsed_output",
        "response_text",
        "prompt",
        "passages",
        "content_lines",
        "expected_output",
        "direction_confusion",
    }
)
_PUBLIC_FORBIDDEN_VALUE_PATTERNS = (
    re.compile(r"\bei2-prompt-[0-9]+\b", re.I),
    re.compile(r"\bPMC[0-9]+\b", re.I),
    re.compile(r"(?:^|\s)/(?:Users|home|private|tmp|var)/"),
)

PROMPT_TEMPLATE = """You are performing one Evidence Inference extraction from fixed
Results excerpts.

Intervention: {intervention}
Comparator: {comparator}
Outcome: {outcome}

Determine the direction of the OUTCOME for the INTERVENTION relative to the COMPARATOR:
- increase: the outcome is higher/increased for the intervention;
- decrease: the outcome is lower/decreased for the intervention;
- no_effect: the comparison is reported as null or not statistically significant.

Use only the excerpts below. If an excerpt directly supports a direction, set eligible=true and
return exactly one finding. Copy a short, exact, contiguous evidence_quote only from between an
excerpt's BEGIN_EXACT_SOURCE_TEXT and END_EXACT_SOURCE_TEXT delimiters. The LINE_ID and delimiters
are metadata, not source text: never include them in evidence_quote. In evidence_lines, return only
the bare line ID, for example ["L23"] -- never "[L23]", never the quote, and never any other text.
Never invent a quote or line number.
If the supplied excerpts do not support a direction for this exact PICO comparison, set
eligible=false and findings=[]. Output only the JSON object required by the schema.

RESULTS EXCERPTS:
{passages}
"""
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
OLLAMA_GENERATION_SCHEMA: dict[str, Any] = {
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
                        "maxItems": 1,
                        "items": {"type": "string"},
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
GENERATION_SCHEMA_SHA256 = hash_canonical(OLLAMA_GENERATION_SCHEMA)
EVALUATION_SCHEMA_SHA256 = hash_canonical(EVIDENCE_INFERENCE_OUTPUT_SCHEMA)


def canonical_json_file_sha256(value: Any) -> str:
    """Hash the exact newline-terminated bytes written by ``atomic_write_json``."""

    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


class EvidenceInferenceOllamaError(ValueError):
    """A staged diagnostic artifact violated its leakage or lineage contract."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceOllamaError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceInferenceOllamaError(f"JSON artifact must be an object: {path}")
    return dict(payload)


def _without_hash(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop(hash_field, None)
    return payload


def _validate_self_hash(
    value: Mapping[str, Any], hash_field: str, *, artifact_name: str
) -> None:
    observed = value.get(hash_field)
    if (
        not isinstance(observed, str)
        or hash_canonical(_without_hash(value, hash_field)) != observed
    ):
        raise EvidenceInferenceOllamaError(f"{artifact_name} hash mismatch")


def _forbidden_key_path(value: Any, forbidden: frozenset[str], prefix: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text.casefold() in forbidden:
                return path
            nested = _forbidden_key_path(item, forbidden, path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _forbidden_key_path(item, forbidden, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    return None


def _stem(token: str) -> str:
    value = token.casefold()
    for suffix in ("ization", "ation", "ments", "ment", "ing", "ies", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    return value


def _terms(text: str) -> list[str]:
    return [
        stemmed
        for token in _TOKEN.findall(text)
        if len(stemmed := _stem(token)) >= 2 and stemmed not in _STOPWORDS
    ]


def _normalized_phrase(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [0]
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        if (
            character == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            continue
        cursor = index + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor].isupper() or text[cursor].isdigit():
            boundaries.append(cursor)
    boundaries.append(len(text))
    spans: list[tuple[int, int]] = []
    for start, end in pairwise(boundaries):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def _bounded_passage_spans(
    text: str, *, outcome_terms: Sequence[str], max_characters: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start, end in _sentence_spans(text):
        if end - start <= max_characters:
            spans.append((start, end))
            continue
        sentence = text[start:end]
        anchors = [
            match.start()
            for term in sorted(set(outcome_terms))
            for match in re.finditer(rf"\b{re.escape(term)}", sentence, flags=re.I)
        ]
        if not anchors:
            anchors = list(range(0, len(sentence), max_characters))
        for anchor in anchors:
            window_start = max(0, anchor - max_characters // 2)
            window_end = min(len(sentence), window_start + max_characters)
            window_start = max(0, window_end - max_characters)
            while window_start < window_end and sentence[window_start].isspace():
                window_start += 1
            while window_end > window_start and sentence[window_end - 1].isspace():
                window_end -= 1
            spans.append((start + window_start, start + window_end))
    return sorted(set(spans))


def _coverage(query: Sequence[str], candidate: set[str]) -> float:
    unique = set(query)
    return 0.0 if not unique else sum(term in candidate for term in unique) / len(unique)


def project_results_passages(
    row: Mapping[str, Any], retrieval_config: Mapping[str, Any] = DEFAULT_RETRIEVAL_CONFIG
) -> list[dict[str, Any]]:
    """Create a deterministic, label-free Results-only retrieval projection."""

    if dict(retrieval_config) != DEFAULT_RETRIEVAL_CONFIG:
        raise EvidenceInferenceOllamaError("only the frozen retrieval configuration is supported")
    replacements = row.get("replacements")
    content_lines = row.get("content_lines")
    line_sections = row.get("line_sections")
    if not isinstance(replacements, Mapping) or not isinstance(content_lines, Mapping):
        raise EvidenceInferenceOllamaError("input row lacks PICO replacements or content lines")
    for key in ("OUTCOME", "INTERVENTION", "COMPARATOR"):
        if not isinstance(replacements.get(key), str):
            raise EvidenceInferenceOllamaError(f"input row lacks string {key}")
    if line_sections is not None and not isinstance(line_sections, Mapping):
        raise EvidenceInferenceOllamaError("line_sections must be an object or null")

    outcome = replacements["OUTCOME"]
    intervention = replacements["INTERVENTION"]
    comparator = replacements["COMPARATOR"]
    outcome_terms = _terms(outcome)
    intervention_terms = _terms(intervention)
    comparator_terms = _terms(comparator)
    normalized_outcome = _normalized_phrase(outcome)
    candidates: list[dict[str, Any]] = []
    for raw_line_id, raw_value in content_lines.items():
        if not isinstance(raw_line_id, str) or not _SAFE_LINE_ID.fullmatch(raw_line_id):
            raise EvidenceInferenceOllamaError("content line ID is invalid")
        if not isinstance(raw_value, Mapping) or not isinstance(raw_value.get("text"), str):
            raise EvidenceInferenceOllamaError("content line value is invalid")
        section = None
        if isinstance(line_sections, Mapping):
            section = line_sections.get(raw_line_id)
        if not isinstance(section, str):
            section = raw_value.get("section")
        if not isinstance(section, str) or section.casefold() != "results":
            continue
        text = raw_value["text"]
        line_number = raw_value.get("line_number")
        if not isinstance(line_number, int) or line_number < 1:
            line_number = int(raw_line_id[1:])
        for start, end in _bounded_passage_spans(
            text,
            outcome_terms=outcome_terms,
            max_characters=int(retrieval_config["max_passage_characters"]),
        ):
            passage = text[start:end]
            if len(passage) < int(retrieval_config["minimum_passage_characters"]):
                continue
            terms = set(_terms(passage))
            normalized = _normalized_phrase(passage)
            outcome_coverage = _coverage(outcome_terms, terms)
            intervention_coverage = _coverage(intervention_terms, terms)
            comparator_coverage = _coverage(comparator_terms, terms)
            phrase_match = bool(normalized_outcome and normalized_outcome in normalized)
            signal = bool(_STATISTICAL_SIGNAL.search(passage))
            score = (
                float(retrieval_config["outcome_term_weight"]) * outcome_coverage
                + float(retrieval_config["outcome_phrase_weight"]) * float(phrase_match)
                + float(retrieval_config["intervention_term_weight"])
                * intervention_coverage
                + float(retrieval_config["comparator_term_weight"]) * comparator_coverage
                + float(retrieval_config["statistical_signal_weight"]) * float(signal)
            )
            candidates.append(
                {
                    "line_id": raw_line_id,
                    "line_number": line_number,
                    "start_character": start,
                    "end_character_exclusive": end,
                    "text": passage,
                    "ranking": {
                        "score": score,
                        "outcome_coverage": outcome_coverage,
                        "outcome_phrase_match": phrase_match,
                        "intervention_coverage": intervention_coverage,
                        "comparator_coverage": comparator_coverage,
                        "statistical_signal": signal,
                    },
                }
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["ranking"]["score"]),
            -float(item["ranking"]["outcome_coverage"]),
            -int(bool(item["ranking"]["outcome_phrase_match"])),
            -int(bool(item["ranking"]["statistical_signal"])),
            int(item["line_number"]),
            int(item["start_character"]),
            str(item["line_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    used_characters = 0
    seen_text: set[tuple[str, int, int]] = set()
    for item in ranked:
        identity = (
            str(item["line_id"]),
            int(item["start_character"]),
            int(item["end_character_exclusive"]),
        )
        if identity in seen_text:
            continue
        next_characters = len(str(item["text"]))
        if used_characters + next_characters > int(retrieval_config["max_total_characters"]):
            continue
        selected.append(item)
        seen_text.add(identity)
        used_characters += next_characters
        if len(selected) >= int(retrieval_config["max_passages"]):
            break
    return [
        {
            "passage_rank": index,
            **item,
        }
        for index, item in enumerate(selected, start=1)
    ]


def _load_input_only_test_rows(
    manifest_path: Path,
    *,
    selected_ids: set[str],
) -> tuple[OptimizationSplitManifest, list[dict[str, Any]]]:
    try:
        manifest = OptimizationSplitManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EvidenceInferenceOllamaError("cannot load the full split manifest") from exc
    split_path = manifest_path.parent / manifest.test.path
    if sha256_file(split_path) != manifest.test.sha256:
        raise EvidenceInferenceOllamaError("test split hash does not match its manifest")
    rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    try:
        handle = split_path.open(encoding="utf-8")
    except OSError as exc:
        raise EvidenceInferenceOllamaError("cannot open test split") from exc
    with handle:
        for raw_line in handle:
            try:
                source = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise EvidenceInferenceOllamaError("test split contains invalid JSONL") from exc
            if not isinstance(source, Mapping):
                raise EvidenceInferenceOllamaError("test split row is not an object")
            example_id = source.get("example_id")
            if not isinstance(example_id, str) or example_id not in selected_ids:
                continue
            paper_id = source.get("paper_id")
            replacements = source.get("replacements")
            if not isinstance(paper_id, str) or not isinstance(replacements, Mapping):
                raise EvidenceInferenceOllamaError("selected input row identity/PICO is invalid")
            passages = project_results_passages(source)
            input_row_payload = {
                "example_id": example_id,
                "paper_id": paper_id,
                "query": {
                    key: replacements.get(key)
                    for key in ("INTERVENTION", "COMPARATOR", "OUTCOME")
                },
                "source_accessible": source.get("source_accessible") is True,
                "passages": passages,
                "projection_sha256": hash_canonical(passages),
            }
            forbidden = _forbidden_key_path(
                input_row_payload,
                frozenset(key.casefold() for key in _FORBIDDEN_INPUT_KEYS),
            )
            if forbidden is not None:
                raise EvidenceInferenceOllamaError(
                    f"prepared row retained a protected label field: {forbidden}"
                )
            input_row_payload["input_row_sha256"] = hash_canonical(input_row_payload)
            rows.append(input_row_payload)
            observed_ids.add(example_id)
    if observed_ids != selected_ids or len(rows) != len(selected_ids):
        raise EvidenceInferenceOllamaError(
            "provider-call-unseen subset is not present exactly once"
        )
    return manifest, sorted(rows, key=lambda row: str(row["example_id"]))


def prepare_input_bundle(
    *,
    manifest_path: Path,
    provider_free_report_path: Path,
    lexical_prediction_ledger_path: Path,
) -> dict[str, Any]:
    """Validate the prior diagnostic and emit only label-free model inputs."""

    report = validate_diagnostic_report(_read_json_object(provider_free_report_path))
    ledger = validate_lexical_prediction_ledger(
        _read_json_object(lexical_prediction_ledger_path)
    )
    if report["prediction_ledger"]["ledger_sha256"] != ledger["ledger_sha256"]:
        raise EvidenceInferenceOllamaError("provider-free report and lexical ledger are unbound")
    manifest_sha256 = sha256_file(manifest_path)
    if report["manifest_file_sha256"] != manifest_sha256:
        raise EvidenceInferenceOllamaError(
            "full manifest differs from validated provider-free report"
        )
    subset = ledger.get("provider_call_unseen_paper_subset")
    if not isinstance(subset, Mapping) or not isinstance(subset.get("rows"), list):
        raise EvidenceInferenceOllamaError("lexical ledger lacks provider-call-unseen subset")
    selected_rows = subset["rows"]
    identities: dict[str, str] = {}
    for row in selected_rows:
        if not isinstance(row, Mapping):
            raise EvidenceInferenceOllamaError("lexical subset row is invalid")
        example_id = row.get("example_id")
        paper_id = row.get("paper_id")
        if not isinstance(example_id, str) or not isinstance(paper_id, str):
            raise EvidenceInferenceOllamaError("lexical subset row identity is invalid")
        if example_id in identities:
            raise EvidenceInferenceOllamaError("lexical subset example IDs are duplicated")
        identities[example_id] = paper_id
    manifest, rows = _load_input_only_test_rows(
        manifest_path, selected_ids=set(identities)
    )
    if any(row["paper_id"] != identities[row["example_id"]] for row in rows):
        raise EvidenceInferenceOllamaError("manifest and validated lexical ledger paper IDs differ")
    if len(rows) != report["provider_call_unseen_paper_diagnostic_rows"]:
        raise EvidenceInferenceOllamaError("provider-call-unseen row count differs from report")
    article_count = len({str(row["paper_id"]) for row in rows})
    if article_count != report["provider_call_unseen_paper_diagnostic_articles"]:
        raise EvidenceInferenceOllamaError("provider-call-unseen article count differs from report")
    payload = {
        "input_bundle_version": INPUT_BUNDLE_VERSION,
        "status": "input_only_provider_call_unseen_non_pristine_diagnostic",
        "diagnostic_version": OLLAMA_DIAGNOSTIC_VERSION,
        "contains_gold_labels": False,
        "contains_expected_outputs": False,
        "contains_label_paths": False,
        "source_split_physically_colocates_inputs_and_labels": True,
        "prediction_stage_can_access_source_split": False,
        "test_labels_previously_opened": True,
        "test_split_pristine": False,
        "manifest_file_sha256": manifest_sha256,
        "test_split_jsonl_sha256": manifest.test.sha256,
        "provider_free_report_sha256": report["report_sha256"],
        "lexical_prediction_ledger_sha256": ledger["ledger_sha256"],
        "provider_call_unseen_subset_ledger_sha256": subset["ledger_sha256"],
        "retrieval_config": deepcopy(DEFAULT_RETRIEVAL_CONFIG),
        "retrieval_config_sha256": hash_canonical(DEFAULT_RETRIEVAL_CONFIG),
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_schema_sha256": GENERATION_SCHEMA_SHA256,
        "evaluation_schema_sha256": EVALUATION_SCHEMA_SHA256,
        "rows": rows,
        "row_count": len(rows),
        "article_count": article_count,
    }
    return {**payload, "input_bundle_sha256": hash_canonical(payload)}


def validate_input_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(bundle))
    _validate_self_hash(snapshot, "input_bundle_sha256", artifact_name="input bundle")
    if (
        snapshot.get("input_bundle_version") != INPUT_BUNDLE_VERSION
        or snapshot.get("status")
        != "input_only_provider_call_unseen_non_pristine_diagnostic"
        or snapshot.get("contains_gold_labels") is not False
        or snapshot.get("contains_expected_outputs") is not False
        or snapshot.get("contains_label_paths") is not False
        or snapshot.get("prediction_stage_can_access_source_split") is not False
        or snapshot.get("test_labels_previously_opened") is not True
        or snapshot.get("test_split_pristine") is not False
        or snapshot.get("retrieval_config") != DEFAULT_RETRIEVAL_CONFIG
        or snapshot.get("retrieval_config_sha256")
        != hash_canonical(DEFAULT_RETRIEVAL_CONFIG)
        or snapshot.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
        or snapshot.get("generation_schema_algorithm")
        != GENERATION_SCHEMA_ALGORITHM
        or snapshot.get("generation_schema_sha256") != GENERATION_SCHEMA_SHA256
        or snapshot.get("evaluation_schema_sha256") != EVALUATION_SCHEMA_SHA256
    ):
        raise EvidenceInferenceOllamaError("input bundle scope/configuration mismatch")
    forbidden = _forbidden_key_path(
        snapshot,
        frozenset(key.casefold() for key in _FORBIDDEN_INPUT_KEYS),
    )
    if forbidden is not None:
        raise EvidenceInferenceOllamaError(
            f"input bundle contains a protected label field: {forbidden}"
        )
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or len(rows) != snapshot.get("row_count"):
        raise EvidenceInferenceOllamaError("input bundle rows are invalid")
    identities: set[str] = set()
    papers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise EvidenceInferenceOllamaError("input bundle row is not an object")
        row_snapshot = deepcopy(dict(row))
        row_hash = row_snapshot.pop("input_row_sha256", None)
        if not isinstance(row_hash, str) or hash_canonical(row_snapshot) != row_hash:
            raise EvidenceInferenceOllamaError("input bundle row hash mismatch")
        example_id = row.get("example_id")
        paper_id = row.get("paper_id")
        if not isinstance(example_id, str) or example_id in identities:
            raise EvidenceInferenceOllamaError("input bundle example identity is invalid")
        if not isinstance(paper_id, str):
            raise EvidenceInferenceOllamaError("input bundle paper identity is invalid")
        if row.get("projection_sha256") != hash_canonical(row.get("passages")):
            raise EvidenceInferenceOllamaError("input bundle projection hash mismatch")
        identities.add(example_id)
        papers.add(paper_id)
    if len(papers) != snapshot.get("article_count"):
        raise EvidenceInferenceOllamaError("input bundle article count mismatch")
    if [row["example_id"] for row in rows] != sorted(identities):
        raise EvidenceInferenceOllamaError("input bundle rows must be sorted by example ID")
    return dict(bundle)


def render_prediction_prompt(row: Mapping[str, Any]) -> str:
    """Render the exact fixed model prompt from one validated input-only row."""

    query = row.get("query")
    passages = row.get("passages")
    if not isinstance(query, Mapping) or not isinstance(passages, list):
        raise EvidenceInferenceOllamaError("prediction row lacks query/passages")
    for key in ("INTERVENTION", "COMPARATOR", "OUTCOME"):
        if not isinstance(query.get(key), str):
            raise EvidenceInferenceOllamaError(f"prediction query lacks {key}")
    rendered_passages: list[str] = []
    for passage in passages:
        if not isinstance(passage, Mapping):
            raise EvidenceInferenceOllamaError("prediction passage is invalid")
        line_id = passage.get("line_id")
        text = passage.get("text")
        if not isinstance(line_id, str) or not isinstance(text, str):
            raise EvidenceInferenceOllamaError("prediction passage lacks line/text")
        rendered_passages.append(
            f"LINE_ID: {line_id}\nBEGIN_EXACT_SOURCE_TEXT\n{text}\n"
            "END_EXACT_SOURCE_TEXT"
        )
    if not rendered_passages:
        rendered_passages.append("[NO_RESULTS_PASSAGE] No Results excerpt survived projection.")
    return PROMPT_TEMPLATE.format(
        intervention=query["INTERVENTION"],
        comparator=query["COMPARATOR"],
        outcome=query["OUTCOME"],
        passages="\n\n".join(rendered_passages),
    )


def generation_schema_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Constrain citations to the exact line IDs exposed in this row's projection."""

    passages = row.get("passages")
    if not isinstance(passages, list):
        raise EvidenceInferenceOllamaError("prediction row passages are invalid")
    line_ids = sorted(
        {
            str(passage["line_id"])
            for passage in passages
            if isinstance(passage, Mapping)
            and isinstance(passage.get("line_id"), str)
            and _SAFE_LINE_ID.fullmatch(str(passage["line_id"]))
        }
    )
    if not line_ids:
        line_ids = ["L1"]
    schema = deepcopy(OLLAMA_GENERATION_SCHEMA)
    schema["properties"]["findings"]["items"]["properties"]["evidence_lines"][
        "items"
    ]["enum"] = line_ids
    return schema


def _schema_validation_error(output: Any) -> str | None:
    try:
        validator = validator_for(EVIDENCE_INFERENCE_OUTPUT_SCHEMA)
        validator.check_schema(EVIDENCE_INFERENCE_OUTPUT_SCHEMA)
        validator(EVIDENCE_INFERENCE_OUTPUT_SCHEMA).validate(output)
    except (SchemaError, JSONSchemaValidationError) as exc:
        message = exc.message if hasattr(exc, "message") else str(exc)
        return f"{type(exc).__name__}:{message}"
    return None


def _model_identity_payload(identity: OllamaIdentity) -> dict[str, Any]:
    return {
        **identity.model_dump(mode="json"),
        "identity_sha256": identity.identity_sha256,
    }


def _receipt_path(receipts_dir: Path, example_id: str) -> Path:
    opaque = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:32]
    return receipts_dir / f"{opaque}.json"


def _generation_result_payload(result: OllamaGenerationResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _build_request_identity(
    *,
    row: Mapping[str, Any],
    prompt: str,
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
) -> dict[str, Any]:
    generation_schema = generation_schema_for_row(row)
    request_payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "format": generation_schema,
        "keep_alive": config.keep_alive,
        "options": config.request_options(),
    }
    return {
        "input_row_sha256": row["input_row_sha256"],
        "model_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generation_schema_sha256": hash_canonical(generation_schema),
        "evaluation_schema_sha256": EVALUATION_SCHEMA_SHA256,
        "request_sha256": hash_canonical(request_payload),
    }


def _make_prediction_receipt(
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    client: OllamaClientProtocol,
) -> dict[str, Any]:
    prompt = render_prediction_prompt(row)
    request_identity = _build_request_identity(
        row=row,
        prompt=prompt,
        config=config,
        identity=identity,
    )
    result: OllamaGenerationResult | None = None
    parsed_output: Any = None
    execution_failure: str | None = None
    schema_error: str | None = None
    try:
        generation_schema = generation_schema_for_row(row)
        result = client.generate(
            prompt=prompt,
            output_schema=generation_schema,
            config=config,
        )
        try:
            parsed_output = json.loads(result.response_text)
        except json.JSONDecodeError:
            execution_failure = "response_json_decode_error"
        if execution_failure is None:
            schema_error = _schema_validation_error(parsed_output)
    except LocalOllamaError as exc:
        detail = str(exc).casefold()
        if "connectionrefusederror" in detail:
            failure_code = "local_server_connection_refused"
        elif "timeouterror" in detail or "timed out" in detail:
            failure_code = "local_server_timeout"
        elif "http " in detail:
            failure_code = "local_server_http_error"
        elif "invalid json" in detail:
            failure_code = "local_server_invalid_wrapper_json"
        else:
            failure_code = "local_server_contract_error"
        execution_failure = f"local_ollama_error:{failure_code}"
    except Exception as exc:
        execution_failure = f"client_error:{type(exc).__name__}"

    if execution_failure is not None:
        status = "execution_failure"
    elif schema_error is not None:
        status = "schema_invalid"
    else:
        status = "complete_schema_valid"
    response_text = result.response_text if result is not None else None
    payload = {
        "prediction_receipt_version": PREDICTION_RECEIPT_VERSION,
        "status": status,
        "diagnostic_status": "non_pristine_local_model_baseline",
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "example_id": row["example_id"],
        "paper_id": row["paper_id"],
        **request_identity,
        "model": config.model,
        "model_digest": config.model_digest,
        "ollama_version": identity.ollama_version,
        "model_parameter_size": identity.parameter_size,
        "generation_config": config.model_dump(mode="json"),
        "generation_config_sha256": config.config_sha256,
        "generation_schema": generation_schema_for_row(row),
        "generation_schema_sha256": hash_canonical(generation_schema_for_row(row)),
        "evaluation_schema_sha256": EVALUATION_SCHEMA_SHA256,
        "response_text": response_text,
        "response_text_sha256": (
            hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            if response_text is not None
            else None
        ),
        "parsed_output": parsed_output,
        "parsed_output_sha256": (
            hash_canonical(parsed_output) if parsed_output is not None else None
        ),
        "execution_failure": execution_failure,
        "schema_error": schema_error,
        "telemetry": (
            _generation_result_payload(result) if result is not None else None
        ),
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def validate_prediction_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    row: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    _validate_self_hash(snapshot, "receipt_sha256", artifact_name="prediction receipt")
    prompt = render_prediction_prompt(row)
    expected_request = _build_request_identity(
        row=row,
        prompt=prompt,
        config=config,
        identity=identity,
    )
    if (
        snapshot.get("prediction_receipt_version") != PREDICTION_RECEIPT_VERSION
        or snapshot.get("diagnostic_status") != "non_pristine_local_model_baseline"
        or snapshot.get("input_bundle_sha256") != bundle.get("input_bundle_sha256")
        or snapshot.get("example_id") != row.get("example_id")
        or snapshot.get("paper_id") != row.get("paper_id")
        or snapshot.get("model") != config.model
        or snapshot.get("model_digest") != config.model_digest
        or snapshot.get("ollama_version") != identity.ollama_version
        or snapshot.get("generation_config") != config.model_dump(mode="json")
        or snapshot.get("generation_config_sha256") != config.config_sha256
        or snapshot.get("generation_schema") != generation_schema_for_row(row)
        or snapshot.get("generation_schema_sha256")
        != hash_canonical(generation_schema_for_row(row))
        or snapshot.get("evaluation_schema_sha256") != EVALUATION_SCHEMA_SHA256
        or any(snapshot.get(key) != value for key, value in expected_request.items())
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
    ):
        raise EvidenceInferenceOllamaError("prediction receipt lineage/config mismatch")
    response_text = snapshot.get("response_text")
    expected_response_hash = (
        hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if isinstance(response_text, str)
        else None
    )
    if snapshot.get("response_text_sha256") != expected_response_hash:
        raise EvidenceInferenceOllamaError("prediction receipt response text hash mismatch")
    parsed = snapshot.get("parsed_output")
    expected_parsed_hash = hash_canonical(parsed) if parsed is not None else None
    if snapshot.get("parsed_output_sha256") != expected_parsed_hash:
        raise EvidenceInferenceOllamaError("prediction receipt parsed output hash mismatch")
    status = snapshot.get("status")
    if status not in {"complete_schema_valid", "schema_invalid", "execution_failure"}:
        raise EvidenceInferenceOllamaError("prediction receipt status is invalid")
    observed_schema_error = _schema_validation_error(parsed) if parsed is not None else None
    if status == "complete_schema_valid" and (
        snapshot.get("execution_failure") is not None
        or snapshot.get("schema_error") is not None
        or observed_schema_error is not None
    ):
        raise EvidenceInferenceOllamaError("valid receipt does not contain a valid output")
    if status == "schema_invalid" and (
        snapshot.get("execution_failure") is not None
        or snapshot.get("schema_error") != observed_schema_error
        or observed_schema_error is None
    ):
        raise EvidenceInferenceOllamaError("schema-invalid receipt is inconsistent")
    if status == "execution_failure" and not isinstance(
        snapshot.get("execution_failure"), str
    ):
        raise EvidenceInferenceOllamaError("failed receipt lacks an execution failure")
    return dict(receipt)


def _prediction_ledger_from_receipts(
    *,
    bundle: Mapping[str, Any],
    config: OllamaGenerationConfig,
    identity: OllamaIdentity,
    receipt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_rows = int(bundle["row_count"])
    status = (
        "complete_frozen_non_pristine_diagnostic"
        if len(receipt_rows) == expected_rows
        else "partial_resumable_non_pristine_diagnostic"
    )
    status_counts = Counter(str(row["status"]) for row in receipt_rows)
    payload = {
        "prediction_ledger_version": PREDICTION_LEDGER_VERSION,
        "status": status,
        "diagnostic_version": OLLAMA_DIAGNOSTIC_VERSION,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "input_row_count": expected_rows,
        "input_article_count": bundle["article_count"],
        "receipt_count": len(receipt_rows),
        "all_expected_predictions_frozen": len(receipt_rows) == expected_rows,
        "prediction_stage_received_label_fields": False,
        "prediction_stage_opened_manifest_or_label_file": False,
        "provider_call_unseen_subset": True,
        "test_labels_previously_opened": True,
        "test_split_pristine": False,
        "model_identity": _model_identity_payload(identity),
        "generation_config": config.model_dump(mode="json"),
        "generation_config_sha256": config.config_sha256,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_schema_sha256": GENERATION_SCHEMA_SHA256,
        "evaluation_schema_sha256": EVALUATION_SCHEMA_SHA256,
        "status_counts": dict(sorted(status_counts.items())),
        "receipts": [
            {
                "example_id": row["example_id"],
                "paper_id": row["paper_id"],
                "input_row_sha256": row["input_row_sha256"],
                "receipt_sha256": row["receipt_sha256"],
                "status": row["status"],
                "parsed_output_sha256": row["parsed_output_sha256"],
            }
            for row in receipt_rows
        ],
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "local_ollama_requests": len(receipt_rows),
    }
    return {**payload, "prediction_ledger_sha256": hash_canonical(payload)}


def validate_prediction_ledger(
    ledger: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    require_complete: bool,
) -> dict[str, Any]:
    validated_bundle = validate_input_bundle(bundle)
    snapshot = deepcopy(dict(ledger))
    _validate_self_hash(
        snapshot, "prediction_ledger_sha256", artifact_name="prediction ledger"
    )
    try:
        parsed_config = OllamaGenerationConfig.model_validate(
            snapshot.get("generation_config")
        )
    except ValueError as exc:
        raise EvidenceInferenceOllamaError(
            "prediction ledger generation config is invalid"
        ) from exc
    if (
        snapshot.get("prediction_ledger_version") != PREDICTION_LEDGER_VERSION
        or snapshot.get("diagnostic_version") != OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("input_bundle_sha256")
        != validated_bundle["input_bundle_sha256"]
        or snapshot.get("input_row_count") != validated_bundle["row_count"]
        or snapshot.get("input_article_count") != validated_bundle["article_count"]
        or snapshot.get("prediction_stage_received_label_fields") is not False
        or snapshot.get("prediction_stage_opened_manifest_or_label_file") is not False
        or snapshot.get("provider_call_unseen_subset") is not True
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
        or snapshot.get("generation_config_sha256") != parsed_config.config_sha256
        or snapshot.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
        or snapshot.get("generation_schema_algorithm")
        != GENERATION_SCHEMA_ALGORITHM
        or snapshot.get("generation_schema_sha256") != GENERATION_SCHEMA_SHA256
        or snapshot.get("evaluation_schema_sha256") != EVALUATION_SCHEMA_SHA256
    ):
        raise EvidenceInferenceOllamaError("prediction ledger scope/lineage mismatch")
    receipts = snapshot.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != snapshot.get("receipt_count"):
        raise EvidenceInferenceOllamaError("prediction ledger receipt manifest is invalid")
    expected_rows = {
        str(row["example_id"]): row for row in validated_bundle["rows"]
    }
    observed_ids: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise EvidenceInferenceOllamaError("prediction ledger receipt row is invalid")
        example_id = receipt.get("example_id")
        if (
            not isinstance(example_id, str)
            or example_id in observed_ids
            or example_id not in expected_rows
            or receipt.get("paper_id") != expected_rows[example_id]["paper_id"]
            or receipt.get("input_row_sha256")
            != expected_rows[example_id]["input_row_sha256"]
        ):
            raise EvidenceInferenceOllamaError("prediction ledger receipt identity is invalid")
        observed_ids.add(example_id)
    if [receipt["example_id"] for receipt in receipts] != sorted(observed_ids):
        raise EvidenceInferenceOllamaError("prediction ledger receipts must be sorted")
    complete = len(receipts) == validated_bundle["row_count"]
    if snapshot.get("all_expected_predictions_frozen") is not complete:
        raise EvidenceInferenceOllamaError("prediction ledger completion flag is inconsistent")
    expected_status = (
        "complete_frozen_non_pristine_diagnostic"
        if complete
        else "partial_resumable_non_pristine_diagnostic"
    )
    if snapshot.get("status") != expected_status:
        raise EvidenceInferenceOllamaError("prediction ledger status is inconsistent")
    if require_complete and not complete:
        raise EvidenceInferenceOllamaError("scoring requires a complete frozen prediction ledger")
    return dict(ledger)


def run_prediction_stage(
    *,
    input_bundle: Mapping[str, Any],
    receipts_dir: Path,
    prediction_ledger_path: Path,
    client: OllamaClientProtocol,
    config: OllamaGenerationConfig = DEFAULT_GENERATION_CONFIG,
    limit: int | None = None,
    retry_failures: bool = False,
) -> dict[str, Any]:
    """Run or resume label-blind prediction and atomically freeze its receipt ledger."""

    bundle = validate_input_bundle(input_bundle)
    if limit is not None and limit < 1:
        raise EvidenceInferenceOllamaError("prediction limit must be positive")
    identity = client.inspect_identity(config)
    if (
        identity.model != config.model
        or identity.model_digest != config.model_digest
        or identity.ollama_version != config.expected_ollama_version
    ):
        raise EvidenceInferenceOllamaError("observed Ollama identity differs from config")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {str(row["example_id"]): row for row in bundle["rows"]}
    validated_receipts: dict[str, dict[str, Any]] = {}
    for row in bundle["rows"]:
        path = _receipt_path(receipts_dir, str(row["example_id"]))
        if not path.exists():
            continue
        receipt = validate_prediction_receipt(
            _read_json_object(path),
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
        )
        if receipt["status"] == "execution_failure" and retry_failures:
            continue
        validated_receipts[str(row["example_id"])] = receipt
    missing = [
        row
        for row in bundle["rows"]
        if str(row["example_id"]) not in validated_receipts
    ]
    to_run = missing if limit is None else missing[:limit]
    for row in to_run:
        receipt = _make_prediction_receipt(
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
            client=client,
        )
        validate_prediction_receipt(
            receipt,
            bundle=bundle,
            row=row,
            config=config,
            identity=identity,
        )
        path = _receipt_path(receipts_dir, str(row["example_id"]))
        atomic_write_json(path, receipt, force=path.exists())
        validated_receipts[str(row["example_id"])] = receipt
    ordered_receipts = [
        validated_receipts[example_id]
        for example_id in sorted(validated_receipts)
    ]
    if set(validated_receipts) - set(rows_by_id):
        raise EvidenceInferenceOllamaError("receipt directory contains an unexpected row")
    ledger = _prediction_ledger_from_receipts(
        bundle=bundle,
        config=config,
        identity=identity,
        receipt_rows=ordered_receipts,
    )
    validate_prediction_ledger(ledger, bundle=bundle, require_complete=False)
    atomic_write_json(prediction_ledger_path, ledger, force=prediction_ledger_path.exists())
    return ledger


def validate_frozen_predictions_before_label_access(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    receipts_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate every frozen byte before a caller is allowed to open labels."""

    bundle = validate_input_bundle(input_bundle)
    ledger = validate_prediction_ledger(
        prediction_ledger,
        bundle=bundle,
        require_complete=True,
    )
    try:
        config = OllamaGenerationConfig.model_validate(ledger["generation_config"])
        identity_payload = deepcopy(dict(ledger["model_identity"]))
        identity_sha256 = identity_payload.pop("identity_sha256")
        identity = OllamaIdentity.model_validate(identity_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceInferenceOllamaError("prediction ledger model contract is invalid") from exc
    if identity.identity_sha256 != identity_sha256:
        raise EvidenceInferenceOllamaError("prediction ledger model identity hash mismatch")
    receipts: dict[str, dict[str, Any]] = {}
    rows_by_id = {str(row["example_id"]): row for row in bundle["rows"]}
    for ledger_row in ledger["receipts"]:
        example_id = str(ledger_row["example_id"])
        receipt = validate_prediction_receipt(
            _read_json_object(_receipt_path(receipts_dir, example_id)),
            bundle=bundle,
            row=rows_by_id[example_id],
            config=config,
            identity=identity,
        )
        for key in (
            "receipt_sha256",
            "status",
            "parsed_output_sha256",
            "input_row_sha256",
            "paper_id",
        ):
            if receipt.get(key) != ledger_row.get(key):
                raise EvidenceInferenceOllamaError(
                    "prediction receipt differs from frozen ledger"
                )
        receipts[example_id] = receipt
    if set(receipts) != set(rows_by_id):
        raise EvidenceInferenceOllamaError("frozen receipt population is incomplete")
    return bundle, ledger, receipts


def _verify_baseline_prediction(
    *,
    example: OptimizationExample,
    expected_ledger_row: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    stripped = lexical_diagnostic._label_stripped_input(example)
    output, disposition = lexical_diagnostic.lexical_extraction_output(stripped)
    observed = lexical_diagnostic._redacted_prediction_ledger_row(
        {
            "example_id": example.example_id,
            "paper_id": example.paper_id,
            "output": output,
            "disposition": disposition,
        }
    )
    if observed != dict(expected_ledger_row):
        raise EvidenceInferenceOllamaError(
            "replayed lexical baseline differs from its validated frozen ledger"
        )
    return output, disposition


def _paired_metric_summary(
    model_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if len(model_rows) != len(baseline_rows):
        raise EvidenceInferenceOllamaError("paired score populations have different lengths")
    baseline_by_id = {
        str(row["example_id"]): row for row in baseline_rows
    }
    paired: dict[str, Any] = {}
    for metric in _METRIC_FIELDS:
        delta_key = f"delta_{metric}"
        delta_rows: list[dict[str, Any]] = []
        for model_row in model_rows:
            example_id = str(model_row["example_id"])
            baseline_row = baseline_by_id.get(example_id)
            if baseline_row is None or baseline_row["paper_id"] != model_row["paper_id"]:
                raise EvidenceInferenceOllamaError("paired score identities differ")
            delta_rows.append(
                {
                    "paper_id": model_row["paper_id"],
                    delta_key: float(model_row[metric]) - float(baseline_row[metric]),
                }
            )
        paired[metric] = {
            "local_ollama": article_clustered_interval(
                model_rows,
                metric,
                seed=seed,
                replicates=replicates,
            ),
            "fixed_lexical": article_clustered_interval(
                baseline_rows,
                metric,
                seed=seed,
                replicates=replicates,
            ),
            "local_ollama_minus_fixed_lexical": article_clustered_interval(
                delta_rows,
                delta_key,
                seed=seed,
                replicates=replicates,
            ),
        }
    discordance = Counter()
    for model_row in model_rows:
        baseline_row = baseline_by_id[str(model_row["example_id"])]
        model_correct = float(model_row["direction_accuracy"]) == 1.0
        baseline_correct = float(baseline_row["direction_accuracy"]) == 1.0
        if model_correct and baseline_correct:
            discordance["both_correct"] += 1
        elif model_correct:
            discordance["local_ollama_only_correct"] += 1
        elif baseline_correct:
            discordance["fixed_lexical_only_correct"] += 1
        else:
            discordance["both_incorrect"] += 1
    return {
        "metrics": paired,
        "direction_accuracy_pairing": dict(sorted(discordance.items())),
        "interval_unit": "article_cluster",
        "point_estimate_unit": "prompt_row",
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def _aggregate_telemetry(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integer_fields = (
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "eval_count",
        "eval_duration_ns",
    )
    sums = {field: 0 for field in integer_fields}
    observed = {field: 0 for field in integer_fields}
    for receipt in receipts:
        telemetry = receipt.get("telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        for field in integer_fields:
            value = telemetry.get(field)
            if isinstance(value, int) and value >= 0:
                sums[field] += value
                observed[field] += 1
    return {
        "receipt_rows": len(receipts),
        "status_counts": dict(
            sorted(Counter(str(receipt["status"]) for receipt in receipts).items())
        ),
        "reported_field_sums": sums,
        "reported_field_observation_counts": observed,
        "reported_total_duration_seconds": sums["total_duration_ns"] / 1_000_000_000,
        "reported_load_duration_seconds": sums["load_duration_ns"] / 1_000_000_000,
        "reported_prompt_tokens": sums["prompt_eval_count"],
        "reported_generated_tokens": sums["eval_count"],
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "cost_basis": "local_existing_hardware_no_provider_charge_energy_not_measured",
    }


def _load_labels_after_freeze(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> list[OptimizationExample]:
    """The sole scoring-only label-opening boundary."""

    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise EvidenceInferenceOllamaError("scoring manifest hash differs from input bundle")
    return load_manifest_split(manifest_path, "test")


def score_frozen_predictions(
    *,
    input_bundle: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    receipts_dir: Path,
    manifest_path: Path,
    provider_free_report_path: Path,
    lexical_prediction_ledger_path: Path,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    label_loader: Callable[..., list[OptimizationExample]] = _load_labels_after_freeze,
) -> dict[str, Any]:
    """Validate the frozen prediction population, then and only then open labels."""

    # Nothing below this call can influence an already self-hashed prediction receipt.
    bundle, ledger, receipts = validate_frozen_predictions_before_label_access(
        input_bundle=input_bundle,
        prediction_ledger=prediction_ledger,
        receipts_dir=receipts_dir,
    )
    provider_free_report = validate_diagnostic_report(
        _read_json_object(provider_free_report_path)
    )
    lexical_ledger = validate_lexical_prediction_ledger(
        _read_json_object(lexical_prediction_ledger_path)
    )
    if (
        provider_free_report["report_sha256"]
        != bundle["provider_free_report_sha256"]
        or lexical_ledger["ledger_sha256"]
        != bundle["lexical_prediction_ledger_sha256"]
        or provider_free_report["prediction_ledger"]["ledger_sha256"]
        != lexical_ledger["ledger_sha256"]
    ):
        raise EvidenceInferenceOllamaError("scoring references differ from input freeze")

    # This function is intentionally invoked only after every receipt and both ledgers
    # have passed their cryptographic/structural checks.
    examples = label_loader(
        manifest_path,
        expected_manifest_sha256=bundle["manifest_file_sha256"],
    )
    examples_by_id = {example.example_id: example for example in examples}
    subset_ids = {str(row["example_id"]) for row in bundle["rows"]}
    if not subset_ids.issubset(examples_by_id):
        raise EvidenceInferenceOllamaError("frozen input subset is absent from scoring labels")
    baseline_subset = lexical_ledger["provider_call_unseen_paper_subset"]["rows"]
    baseline_ledger_by_id = {
        str(row["example_id"]): row for row in baseline_subset
    }
    if set(baseline_ledger_by_id) != subset_ids:
        raise EvidenceInferenceOllamaError("paired lexical ledger population differs")

    model_scored: list[dict[str, Any]] = []
    baseline_scored: list[dict[str, Any]] = []
    model_output_distribution: Counter[str] = Counter()
    baseline_output_distribution: Counter[str] = Counter()
    for input_row in bundle["rows"]:
        example_id = str(input_row["example_id"])
        example = examples_by_id[example_id]
        if example.paper_id != input_row["paper_id"]:
            raise EvidenceInferenceOllamaError("scoring paper identity differs from freeze")
        # Recompute the input-only projection from the authoritative row after labels are
        # opened; it must be byte-identical to what prediction saw.
        recomputed_passages = project_results_passages(example.model_dump(mode="json"))
        if (
            recomputed_passages != input_row["passages"]
            or hash_canonical(recomputed_passages) != input_row["projection_sha256"]
        ):
            raise EvidenceInferenceOllamaError("scoring source projection differs from freeze")
        receipt = receipts[example_id]
        execution_failure = (
            str(receipt["execution_failure"])
            if receipt["status"] == "execution_failure"
            else None
        )
        if execution_failure is not None:
            model_output_distribution["execution_failure"] += 1
        elif not receipt["parsed_output"]["findings"]:
            model_output_distribution["no_finding"] += 1
        else:
            model_output_distribution[
                str(receipt["parsed_output"]["findings"][0]["direction"])
            ] += 1
        model_scored.append(
            score_diagnostic_output(
                example,
                receipt["parsed_output"],
                execution_failure=execution_failure,
                baseline_disposition=str(receipt["status"]),
            )
        )
        baseline_output, baseline_disposition = _verify_baseline_prediction(
            example=example,
            expected_ledger_row=baseline_ledger_by_id[example_id],
        )
        if not baseline_output["findings"]:
            baseline_output_distribution["no_finding"] += 1
        else:
            baseline_output_distribution[
                str(baseline_output["findings"][0]["direction"])
            ] += 1
        baseline_scored.append(
            score_diagnostic_output(
                example,
                baseline_output,
                baseline_disposition=baseline_disposition,
            )
        )

    model_summary = summarize_scored_rows(
        model_scored,
        seed=seed,
        replicates=replicates,
    )
    baseline_summary = summarize_scored_rows(
        baseline_scored,
        seed=seed,
        replicates=replicates,
    )
    model_summary["prediction_output_distribution"] = dict(
        sorted(model_output_distribution.items())
    )
    baseline_summary["prediction_output_distribution"] = dict(
        sorted(baseline_output_distribution.items())
    )
    paired = _paired_metric_summary(
        model_scored,
        baseline_scored,
        seed=seed,
        replicates=replicates,
    )
    model_identity = deepcopy(dict(ledger["model_identity"]))
    execution_inputs = {
        "diagnostic_version": OLLAMA_DIAGNOSTIC_VERSION,
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "input_bundle_file_sha256": canonical_json_file_sha256(bundle),
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "prediction_ledger_file_sha256": canonical_json_file_sha256(ledger),
        "provider_free_report_sha256": provider_free_report["report_sha256"],
        "lexical_prediction_ledger_sha256": lexical_ledger["ledger_sha256"],
        "manifest_file_sha256": bundle["manifest_file_sha256"],
        "test_split_jsonl_sha256": bundle["test_split_jsonl_sha256"],
        "module_sha256": sha256_file(Path(__file__)),
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "retrieval_config_sha256": bundle["retrieval_config_sha256"],
        "generation_config_sha256": ledger["generation_config_sha256"],
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "generation_schema_sha256": GENERATION_SCHEMA_SHA256,
        "evaluation_schema_sha256": EVALUATION_SCHEMA_SHA256,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }
    payload = {
        "private_report_version": PRIVATE_REPORT_VERSION,
        "diagnostic_version": OLLAMA_DIAGNOSTIC_VERSION,
        "status": "complete_non_pristine_offline_local_model_diagnostic",
        "confirmatory_claims_allowed": False,
        "test_labels_previously_opened": True,
        "test_split_pristine": False,
        "provider_call_unseen_limited_to_bound_local_receipt_registry": True,
        "prediction_ledger_validated_before_label_file_open": True,
        "prediction_stage_received_label_fields": False,
        "prediction_stage_opened_manifest_or_label_file": False,
        "scoring_stage_opened_labels": True,
        "source_rows_physically_colocate_inputs_and_labels": True,
        "predictions_frozen_before_scoring": True,
        "training_labels_used": False,
        "semantic_support_or_entailment_measured": False,
        "formal_quote_line_provenance_definition": (
            "quoted bytes occur in cited authoritative source lines; not entailment"
        ),
        "gold_span_agreement_reported_separately": True,
        "native_numeric_effect_extraction_evaluated": False,
        "gepa_optimization_or_improvement_evaluated": False,
        "model_scope": "local_1.2B_parameter_baseline",
        "model_identity": model_identity,
        "generation_config": deepcopy(ledger["generation_config"]),
        "retrieval_config": deepcopy(bundle["retrieval_config"]),
        "population": {
            "rows": len(model_scored),
            "articles": len({str(row["paper_id"]) for row in model_scored}),
            "selection": "provider_call_unseen_papers_in_bound_local_registry",
        },
        "local_ollama": model_summary,
        "fixed_lexical": baseline_summary,
        "paired_comparison": paired,
        "inference_failure_taxonomy": dict(
            sorted(Counter(str(receipt["status"]) for receipt in receipts.values()).items())
        ),
        "telemetry": _aggregate_telemetry(list(receipts.values())),
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "execution_inputs": execution_inputs,
        "execution_fingerprint_sha256": hash_canonical(execution_inputs),
        "interpretation_limits": [
            "all Evidence Inference test labels and metrics are non-pristine diagnostics",
            "provider-call-unseen means unseen in the complete bound local "
            "successful-call registry",
            "the evaluated model is a 1.2B local baseline, not a frontier model",
            "formal copy-grounding is byte validity and does not establish semantic entailment",
            "gold-span disagreement may still reflect an alternative valid source span",
            "this run does not demonstrate GEPA improvement",
            "this run does not evaluate native numerical effect extraction",
            "no hosted provider calls or provider charges were incurred",
        ],
    }
    return {**payload, "report_sha256": hash_canonical(payload)}


def validate_private_report(report: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(report))
    _validate_self_hash(snapshot, "report_sha256", artifact_name="private report")
    population = snapshot.get("population")
    model_identity = snapshot.get("model_identity")
    if (
        snapshot.get("private_report_version") != PRIVATE_REPORT_VERSION
        or snapshot.get("diagnostic_version") != OLLAMA_DIAGNOSTIC_VERSION
        or snapshot.get("status")
        != "complete_non_pristine_offline_local_model_diagnostic"
        or snapshot.get("confirmatory_claims_allowed") is not False
        or snapshot.get("test_labels_previously_opened") is not True
        or snapshot.get("test_split_pristine") is not False
        or snapshot.get("prediction_ledger_validated_before_label_file_open") is not True
        or snapshot.get("prediction_stage_received_label_fields") is not False
        or snapshot.get("prediction_stage_opened_manifest_or_label_file") is not False
        or snapshot.get("scoring_stage_opened_labels") is not True
        or snapshot.get("predictions_frozen_before_scoring") is not True
        or snapshot.get("semantic_support_or_entailment_measured") is not False
        or snapshot.get("native_numeric_effect_extraction_evaluated") is not False
        or snapshot.get("gepa_optimization_or_improvement_evaluated") is not False
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
        or not isinstance(population, Mapping)
        or population.get("rows", 0) < 1
        or population.get("articles", 0) < 1
        or not isinstance(model_identity, Mapping)
        or model_identity.get("model") != DEFAULT_MODEL
        or model_identity.get("model_digest") != DEFAULT_MODEL_DIGEST
        or model_identity.get("ollama_version") != DEFAULT_OLLAMA_VERSION
        or model_identity.get("parameter_size") != DEFAULT_MODEL_PARAMETER_SIZE
    ):
        raise EvidenceInferenceOllamaError("private report scope contract mismatch")
    execution_inputs = snapshot.get("execution_inputs")
    if (
        not isinstance(execution_inputs, Mapping)
        or snapshot.get("execution_fingerprint_sha256")
        != hash_canonical(execution_inputs)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(execution_inputs.get("input_bundle_file_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(execution_inputs.get("prediction_ledger_file_sha256", "")),
        )
    ):
        raise EvidenceInferenceOllamaError("private report execution fingerprint mismatch")
    for system_name in ("local_ollama", "fixed_lexical"):
        system_summary = snapshot.get(system_name)
        distribution = (
            system_summary.get("prediction_output_distribution")
            if isinstance(system_summary, Mapping)
            else None
        )
        if (
            not isinstance(distribution, Mapping)
            or any(
                key
                not in {
                    "increase",
                    "no_effect",
                    "decrease",
                    "no_finding",
                    "execution_failure",
                }
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in distribution.items()
            )
            or sum(distribution.values()) != population["rows"]
        ):
            raise EvidenceInferenceOllamaError(
                f"private report prediction distribution mismatch: {system_name}"
            )
    return dict(report)


def _public_metric_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": summary["rows"],
        "articles": summary["articles"],
        "metrics": deepcopy(summary["metrics"]),
        "failure_taxonomy": deepcopy(summary["failure_taxonomy"]),
        "prediction_output_distribution": deepcopy(
            summary["prediction_output_distribution"]
        ),
    }


def build_public_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Produce a tracked, self-hashed summary without text, IDs, or predictions."""

    validated = validate_private_report(report)
    payload = {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "diagnostic_version": OLLAMA_DIAGNOSTIC_VERSION,
        "status": "metadata_only_non_pristine_offline_local_model_diagnostic",
        "contains_article_text": False,
        "contains_evidence_quotes": False,
        "contains_gold_labels": False,
        "contains_per_example_labels": False,
        "contains_paper_or_example_ids": False,
        "contains_raw_predictions": False,
        "contains_absolute_paths": False,
        "full_private_report_sha256": validated["report_sha256"],
        "execution_fingerprint_sha256": validated["execution_fingerprint_sha256"],
        "input_bundle_sha256": validated["execution_inputs"]["input_bundle_sha256"],
        "input_bundle_file_sha256": validated["execution_inputs"][
            "input_bundle_file_sha256"
        ],
        "prediction_ledger_sha256": validated["execution_inputs"][
            "prediction_ledger_sha256"
        ],
        "prediction_ledger_file_sha256": validated["execution_inputs"][
            "prediction_ledger_file_sha256"
        ],
        "population": deepcopy(validated["population"]),
        "model_identity": deepcopy(validated["model_identity"]),
        "generation_config": deepcopy(validated["generation_config"]),
        "generation_config_sha256": validated["execution_inputs"][
            "generation_config_sha256"
        ],
        "retrieval_config": deepcopy(validated["retrieval_config"]),
        "retrieval_config_sha256": validated["execution_inputs"][
            "retrieval_config_sha256"
        ],
        "prompt_template_sha256": validated["execution_inputs"][
            "prompt_template_sha256"
        ],
        "generation_schema_sha256": validated["execution_inputs"][
            "generation_schema_sha256"
        ],
        "generation_schema_algorithm": GENERATION_SCHEMA_ALGORITHM,
        "evaluation_schema_sha256": validated["execution_inputs"][
            "evaluation_schema_sha256"
        ],
        "local_ollama": _public_metric_summary(validated["local_ollama"]),
        "fixed_lexical": _public_metric_summary(validated["fixed_lexical"]),
        "paired_comparison": deepcopy(validated["paired_comparison"]),
        "inference_failure_taxonomy": deepcopy(
            validated["inference_failure_taxonomy"]
        ),
        "telemetry": deepcopy(validated["telemetry"]),
        "scientific_scope": {
            "confirmatory_claims_allowed": False,
            "test_labels_previously_opened": True,
            "test_split_pristine": False,
            "provider_call_unseen_limited_to_bound_local_receipt_registry": True,
            "semantic_support_or_entailment_measured": False,
            "formal_quote_line_provenance_is_copy_validity_only": True,
            "gold_span_agreement_reported_separately": True,
            "native_numeric_effect_extraction_evaluated": False,
            "gepa_optimization_or_improvement_evaluated": False,
            "local_model_parameter_scale": DEFAULT_MODEL_PARAMETER_SIZE,
        },
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
        "interpretation_limits": deepcopy(validated["interpretation_limits"]),
    }
    summary = {**payload, "public_summary_sha256": hash_canonical(payload)}
    validate_public_summary(summary)
    return summary


def validate_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(summary))
    _validate_self_hash(snapshot, "public_summary_sha256", artifact_name="public summary")
    if (
        snapshot.get("public_summary_version") != PUBLIC_SUMMARY_VERSION
        or snapshot.get("status")
        != "metadata_only_non_pristine_offline_local_model_diagnostic"
        or snapshot.get("contains_article_text") is not False
        or snapshot.get("contains_evidence_quotes") is not False
        or snapshot.get("contains_gold_labels") is not False
        or snapshot.get("contains_per_example_labels") is not False
        or snapshot.get("contains_paper_or_example_ids") is not False
        or snapshot.get("contains_raw_predictions") is not False
        or snapshot.get("contains_absolute_paths") is not False
        or snapshot.get("external_provider_calls") != 0
        or snapshot.get("external_provider_cost_usd") != 0.0
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(snapshot.get("input_bundle_file_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(snapshot.get("prediction_ledger_file_sha256", "")),
        )
    ):
        raise EvidenceInferenceOllamaError("public summary scope contract mismatch")
    forbidden = _forbidden_key_path(
        snapshot,
        frozenset(key.casefold() for key in _FORBIDDEN_PUBLIC_KEYS),
    )
    if forbidden is not None:
        raise EvidenceInferenceOllamaError(
            f"public summary contains a protected field: {forbidden}"
        )
    serialized = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    if any(pattern.search(serialized) for pattern in _PUBLIC_FORBIDDEN_VALUE_PATTERNS):
        raise EvidenceInferenceOllamaError("public summary contains a protected value")
    population = snapshot.get("population")
    for system_name in ("local_ollama", "fixed_lexical"):
        system_summary = snapshot.get(system_name)
        distribution = (
            system_summary.get("prediction_output_distribution")
            if isinstance(system_summary, Mapping)
            else None
        )
        if (
            not isinstance(population, Mapping)
            or not isinstance(distribution, Mapping)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in distribution.values()
            )
            or sum(distribution.values()) != population.get("rows")
        ):
            raise EvidenceInferenceOllamaError(
                f"public summary prediction distribution mismatch: {system_name}"
            )
    return dict(summary)
