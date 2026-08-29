"""Provider-free Evidence Inference extraction and archived-response diagnostics.

The full-split baseline is a deliberately simple, fixed lexical extractor.  It sees only
the model-facing PICO fields and article source lines.  Historical provider receipts are
rescored without network access, and every GEPA mutation is compared with the handwritten
seed only on examples for which both arms have an archived call.

This module is diagnostic by construction.  The repository's Evidence Inference test
labels have previously been opened, so no result emitted here is a pristine held-out
estimate or evidence of prospective prompt improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from jsonschema.validators import validator_for

from literature_multiverse.grounding import GroundingContractError, ground_evidence
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    OptimizationSplitManifest,
    load_manifest_split,
    load_split_manifest,
)
from literature_multiverse.prompting import PromptContractError, render_prompt_text
from literature_multiverse.providers import sha256_json

DIAGNOSTIC_VERSION = "evidence-inference-provider-free-diagnostic-v2"
LEXICAL_BASELINE_VERSION = "fixed-input-only-lexical-extractor-v1"
PREDICTION_LEDGER_VERSION = "evidence-inference-prediction-ledger-v1"
DEFAULT_BOOTSTRAP_SEED = 20260827
DEFAULT_BOOTSTRAP_REPLICATES = 2000
_KNOWN_CACHE_DEFECT_EVIDENCE = {
    "archive_run_id": "gepa/evidence-inference-first-pass-v2",
    "optimization_trace_sha256": (
        "01c9f68a993a80ac4b5698c504424622a4acc725b40ac82de3ed6b44282db999"
    ),
    "trace_run_id": "gepa-b29034d6a6f046563116",
    "gepa_version": "0.1.4",
    "optimizer": "gepa.optimize",
    "candidate_sha256": (
        "a8675bde5230408c42a7b944ffe328b42f13eacec26f16a6cc080acc8a398127"
    ),
    "trace_score": 0.5125799999999999,
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
_REQUEST_KEY = re.compile(
    r"^(?P<example_id>ei2-prompt-[1-9][0-9]*)-(?P<candidate_prefix>[0-9a-f]{16})$"
)
_PROPOSED_PROMPT = re.compile(
    r"^Iteration (?P<iteration>\d+): Proposed new text for extraction_prompt: "
    r"(?P<prompt>.*?)(?=\n^Iteration (?P=iteration): "
    r"(?:New subsample score|Accepted candidate))",
    flags=re.MULTILINE | re.DOTALL,
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_P_VALUE = re.compile(
    r"\bp\s*(?P<operator><=|>=|=|<|>|≤|≥)\s*(?P<value>(?:0?\.\d+|1(?:\.0+)?))",
    flags=re.IGNORECASE,
)
_NO_EFFECT_PATTERNS = (
    re.compile(
        r"\b(?:no|not)\s+(?:statistically\s+)?significant(?:ly)?\s+"
        r"(?:difference|differences|change|changes|effect|effects|association)",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bdid\s+not\s+(?:differ|change|improve|increase|decrease)\b", re.I),
    re.compile(r"\bneither\b.{0,100}\bstatistically\s+significant\b", re.I),
    re.compile(r"\b(?:similar|comparable)\s+(?:between|in)\s+(?:the\s+)?(?:groups|arms)\b", re.I),
)
_INCREASE = re.compile(
    r"\b(?:higher|increas(?:e|ed|ing)|greater|more|improv(?:e|ed|ement)|rose|gain(?:ed)?)\b",
    flags=re.IGNORECASE,
)
_DECREASE = re.compile(
    r"\b(?:lower|decreas(?:e|ed|ing)|reduc(?:e|ed|tion)|less|fewer|declin(?:e|ed)|fell)\b",
    flags=re.IGNORECASE,
)
_SIGNAL = re.compile(
    r"\b(?:significant|significantly|higher|lower|increase|decrease|reduced|difference|"
    r"similar|comparable)\b|\bp\s*[<=>≤≥]",
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
_GENERIC_GROUP_TERMS = frozenset(
    {"arm", "control", "group", "intervention", "placebo", "treatment"}
)


class EvidenceInferenceDiagnosticError(ValueError):
    """A local benchmark or archive violated the diagnostic contract."""


@dataclass(frozen=True, slots=True)
class _TextSpan:
    line_id: str
    line_number: int
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class _ArchivedCall:
    path: Path
    example_id: str
    candidate_sha256: str
    receipt: Mapping[str, Any]
    schema_matches_current: bool
    request_hash_contract: str


@dataclass(frozen=True, slots=True)
class _LabelStrippedExtractionInput:
    """The only fields made available to the provider-free prediction stage."""

    example_id: str
    paper_id: str
    prompt_kind: str
    replacements: Mapping[str, str]
    content_lines: Mapping[str, Any] | list[Any] | None
    line_sections: Mapping[str, str | None] | None
    source_accessible: bool


def _label_stripped_input(example: OptimizationExample) -> _LabelStrippedExtractionInput:
    return _LabelStrippedExtractionInput(
        example_id=example.example_id,
        paper_id=example.paper_id,
        prompt_kind=example.prompt_kind,
        replacements=deepcopy(example.replacements),
        content_lines=deepcopy(example.content_lines),
        line_sections=deepcopy(example.line_sections),
        source_accessible=example.source_accessible,
    )


def _stem(token: str) -> str:
    value = token.casefold()
    for suffix in ("ization", "ation", "ments", "ment", "ing", "ies", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    return value


def _terms(text: str, *, remove_generic_groups: bool = False) -> list[str]:
    values = [_stem(token) for token in _TOKEN.findall(text)]
    return [
        token
        for token in values
        if len(token) >= 2
        and token not in _STOPWORDS
        and (not remove_generic_groups or token not in _GENERIC_GROUP_TERMS)
    ]


def _normalized_phrase(text: str) -> str:
    return " ".join(token.casefold() for token in _TOKEN.findall(text))


def _line_number(line_id: str, value: Mapping[str, Any]) -> int:
    raw = value.get("line_number")
    if isinstance(raw, int) and raw > 0:
        return raw
    if line_id.startswith("L") and line_id[1:].isdigit():
        return int(line_id[1:])
    raise EvidenceInferenceDiagnosticError(f"invalid source line identity: {line_id}")


def _source_line_rows(
    example: OptimizationExample | _LabelStrippedExtractionInput,
) -> list[tuple[str, int, str]]:
    content = example.content_lines
    if not isinstance(content, Mapping):
        raise EvidenceInferenceDiagnosticError(
            f"Evidence Inference content lines must be an object: {example.example_id}"
        )
    rows: list[tuple[str, int, str]] = []
    for raw_line_id, raw_value in content.items():
        if not isinstance(raw_line_id, str) or not isinstance(raw_value, Mapping):
            raise EvidenceInferenceDiagnosticError(
                f"invalid source line object: {example.example_id}"
            )
        text = raw_value.get("text")
        if not isinstance(text, str):
            raise EvidenceInferenceDiagnosticError(
                f"invalid source line text: {example.example_id}:{raw_line_id}"
            )
        rows.append((raw_line_id, _line_number(raw_line_id, raw_value), text))
    return sorted(rows, key=lambda row: (row[1], row[0]))


def _sentence_offsets(text: str) -> list[tuple[int, int]]:
    """Return stable sentence-like spans without splitting decimal p-values."""

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
        if cursor < len(text) and text[cursor].isupper():
            boundaries.append(cursor)
    boundaries.append(len(text))
    spans: list[tuple[int, int]] = []
    for start, end in pairwise(boundaries):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 12:
            spans.append((start, end))
    return spans


def _focused_window(text: str, start: int, end: int, anchor: int) -> tuple[int, int]:
    if end - start <= 1400:
        return start, end
    left_limit = max(start, anchor - 700)
    right_limit = min(end, anchor + 700)
    left_candidates = [
        text.rfind(separator, left_limit, anchor)
        for separator in (". ", "; ", "  ", ": ")
    ]
    left = max((position for position in left_candidates if position >= 0), default=left_limit)
    if left > left_limit:
        left += 2
    right_candidates = [
        text.find(separator, anchor, right_limit)
        for separator in (". ", "; ", "  ")
    ]
    present_right = [position for position in right_candidates if position >= 0]
    right = min(present_right) + 1 if present_right else right_limit
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def _candidate_spans(
    example: OptimizationExample | _LabelStrippedExtractionInput,
) -> list[_TextSpan]:
    outcome_terms = _terms(example.replacements.get("OUTCOME", ""))
    spans: list[_TextSpan] = []
    seen: set[tuple[str, int, int]] = set()
    for line_id, line_number, text in _source_line_rows(example):
        if text.strip().upper().startswith("BODY.RESULTS") and text.strip().endswith(":"):
            continue
        for sentence_start, sentence_end in _sentence_offsets(text):
            sentence = text[sentence_start:sentence_end]
            lowered_terms = _terms(sentence)
            anchors = [
                match.start()
                for term in outcome_terms
                for match in re.finditer(rf"\b{re.escape(term)}", sentence, flags=re.I)
            ]
            windows = [(sentence_start, sentence_end)]
            if sentence_end - sentence_start > 1400 and anchors:
                windows = [
                    _focused_window(
                        text,
                        sentence_start,
                        sentence_end,
                        sentence_start + anchor,
                    )
                    for anchor in anchors
                ]
            elif not lowered_terms:
                continue
            for start, end in windows:
                key = (line_id, start, end)
                if key in seen or end - start < 12:
                    continue
                seen.add(key)
                spans.append(
                    _TextSpan(
                        line_id=line_id,
                        line_number=line_number,
                        start=start,
                        text=text[start:end],
                    )
                )
    return spans


def _term_coverage(query: Sequence[str], candidate: set[str]) -> float:
    if not query:
        return 0.0
    return sum(term in candidate for term in set(query)) / len(set(query))


def _rank_span(
    example: OptimizationExample | _LabelStrippedExtractionInput, span: _TextSpan
) -> tuple[float, ...]:
    outcome = example.replacements.get("OUTCOME", "")
    outcome_terms = _terms(outcome)
    intervention_terms = _terms(
        example.replacements.get("INTERVENTION", ""), remove_generic_groups=True
    )
    comparator_terms = _terms(
        example.replacements.get("COMPARATOR", ""), remove_generic_groups=True
    )
    candidate_terms = set(_terms(span.text))
    outcome_coverage = _term_coverage(outcome_terms, candidate_terms)
    normalized_outcome = _normalized_phrase(outcome)
    phrase_match = bool(
        normalized_outcome and normalized_outcome in _normalized_phrase(span.text)
    )
    group_coverage = 0.5 * _term_coverage(intervention_terms, candidate_terms)
    group_coverage += 0.5 * _term_coverage(comparator_terms, candidate_terms)
    signal = 1.0 if _SIGNAL.search(span.text) else 0.0
    score = 6.0 * outcome_coverage + 3.0 * phrase_match + group_coverage + signal
    return (
        score,
        outcome_coverage,
        float(phrase_match),
        signal,
        -len(span.text),
        -span.line_number,
        -span.start,
    )


def _first_term_position(text: str, terms: Sequence[str]) -> int | None:
    positions = [
        match.start()
        for term in set(terms)
        for match in [re.search(rf"\b{re.escape(term)}", text, flags=re.I)]
        if match is not None
    ]
    return min(positions) if positions else None


def _explicit_nonsignificant_p_value(text: str) -> bool:
    for match in _P_VALUE.finditer(text):
        value = float(match.group("value"))
        operator = match.group("operator")
        if operator in {">", ">=", "≥"} and value >= 0.05:
            return True
        if operator == "=" and value > 0.05:
            return True
    return False


def _direction_from_span(
    example: OptimizationExample | _LabelStrippedExtractionInput, span: _TextSpan
) -> str | None:
    text = span.text
    if any(pattern.search(text) for pattern in _NO_EFFECT_PATTERNS):
        return "no_effect"
    if _explicit_nonsignificant_p_value(text):
        return "no_effect"
    directional = [
        *((match.start(), "increase") for match in _INCREASE.finditer(text)),
        *((match.start(), "decrease") for match in _DECREASE.finditer(text)),
    ]
    if not directional:
        return None
    outcome_positions = [
        match.start()
        for term in _terms(example.replacements.get("OUTCOME", ""))
        for match in re.finditer(rf"\b{re.escape(term)}", text, flags=re.I)
    ]
    anchor = min(outcome_positions) if outcome_positions else len(text) // 2
    direction_position, direction = min(
        directional, key=lambda item: (abs(item[0] - anchor), item[0], item[1])
    )
    intervention_position = _first_term_position(
        text,
        _terms(example.replacements.get("INTERVENTION", ""), remove_generic_groups=True),
    )
    comparator_position = _first_term_position(
        text,
        _terms(example.replacements.get("COMPARATOR", ""), remove_generic_groups=True),
    )
    if comparator_position is not None and (
        intervention_position is None
        or abs(comparator_position - direction_position)
        < abs(intervention_position - direction_position)
    ):
        return "decrease" if direction == "increase" else "increase"
    return direction


def lexical_extraction_output(
    example: OptimizationExample | _LabelStrippedExtractionInput,
) -> tuple[dict[str, Any], str]:
    """Produce one fixed input-only extraction output and a diagnostic disposition."""

    if example.prompt_kind != "extraction":
        raise EvidenceInferenceDiagnosticError("lexical baseline supports extraction only")
    spans = _candidate_spans(example)
    if not spans:
        return {"eligible": False, "findings": []}, "no_candidate_span"
    ranked = sorted(spans, key=lambda span: _rank_span(example, span), reverse=True)
    selected = ranked[0]
    rank = _rank_span(example, selected)
    if rank[1] <= 0:
        return {"eligible": False, "findings": []}, "no_outcome_term_match"
    direction = _direction_from_span(example, selected)
    if direction is None:
        return {"eligible": False, "findings": []}, "direction_ambiguous"
    return (
        {
            "eligible": True,
            "findings": [
                {
                    "direction": direction,
                    "evidence_quote": selected.text,
                    "evidence_lines": [selected.line_id],
                }
            ],
        },
        "finding_emitted",
    )


def _schema_error(example: OptimizationExample, output: Any) -> str | None:
    try:
        schema = deepcopy(example.output_schema)
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema).validate(output)
    except (SchemaError, JSONSchemaValidationError) as exc:
        return f"{type(exc).__name__}:{exc.message if hasattr(exc, 'message') else exc}"
    return None


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = Counter(_TOKEN.findall(predicted.casefold()))
    expected_tokens = Counter(_TOKEN.findall(expected.casefold()))
    if not predicted_tokens or not expected_tokens:
        return 0.0
    overlap = sum((predicted_tokens & expected_tokens).values())
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _expected_finding(example: OptimizationExample) -> Mapping[str, Any]:
    findings = example.expected_output.get("findings")
    if not isinstance(findings, list) or len(findings) != 1 or not isinstance(findings[0], Mapping):
        raise EvidenceInferenceDiagnosticError(
            f"diagnostic requires one gold finding: {example.example_id}"
        )
    return findings[0]


def score_diagnostic_output(
    example: OptimizationExample,
    output: Any,
    *,
    execution_failure: str | None = None,
    baseline_disposition: str | None = None,
) -> dict[str, Any]:
    """Score syntax, task shape, direction, and formal quote/line provenance.

    Formal provenance means that the quoted bytes occur in the cited source lines under
    the repository grounding contract.  It is not semantic support or entailment.
    """

    expected = _expected_finding(example)
    schema_error = None if output is None else _schema_error(example, output)
    structured = execution_failure is None and output is not None and schema_error is None
    predicted_direction: str | None = None
    formal_provenance_valid = False
    gold_line_agreement = False
    quote_f1 = 0.0
    eligible = False
    task_shape_consistent = False
    if structured and isinstance(output, Mapping):
        eligible = output.get("eligible") is True
        findings = output.get("findings")
        task_shape_consistent = isinstance(output.get("eligible"), bool) and isinstance(
            findings, list
        )
        if task_shape_consistent:
            task_shape_consistent = (eligible and len(findings) == 1) or (
                not eligible and len(findings) == 0
            )
        if eligible and isinstance(findings, list) and len(findings) == 1:
            finding = findings[0]
            if isinstance(finding, Mapping):
                raw_direction = finding.get("direction")
                if isinstance(raw_direction, str):
                    predicted_direction = raw_direction
                try:
                    provenance = ground_evidence(
                        finding.get("evidence_quote"),
                        finding.get("evidence_lines"),
                        example.content_lines,
                        line_sections=example.line_sections,
                        source_accessible=example.source_accessible,
                    )
                    formal_provenance_valid = (
                        provenance.get("grounding_status") == "exact"
                        and provenance.get("section_flagged") is False
                    )
                except (GroundingContractError, TypeError, ValueError):
                    formal_provenance_valid = False
                raw_lines = finding.get("evidence_lines")
                gold_line_agreement = raw_lines == expected.get("evidence_lines")
                raw_quote = finding.get("evidence_quote")
                expected_quote = expected.get("evidence_quote")
                if isinstance(raw_quote, str) and isinstance(expected_quote, str):
                    quote_f1 = _token_f1(raw_quote, expected_quote)
    direction_correct = predicted_direction == expected.get("direction")
    joint_validity = (
        structured
        and task_shape_consistent
        and direction_correct
        and formal_provenance_valid
    )
    if execution_failure is not None:
        primary_failure = execution_failure
    elif not structured:
        primary_failure = "invalid_exact_structured_output"
    elif not task_shape_consistent:
        primary_failure = "task_shape_inconsistent"
    elif not eligible:
        primary_failure = "model_abstention"
    elif not direction_correct:
        primary_failure = "direction_incorrect"
    elif not formal_provenance_valid:
        primary_failure = "formal_quote_line_provenance_invalid"
    else:
        primary_failure = "success"
    return {
        "example_id": example.example_id,
        "paper_id": example.paper_id,
        "expected_direction": expected.get("direction"),
        "predicted_direction": predicted_direction,
        "baseline_disposition": baseline_disposition,
        "schema_error": schema_error,
        "primary_failure": primary_failure,
        "exact_structured_output_validity": float(structured),
        "task_shape_consistency": float(task_shape_consistent),
        "direction_accuracy": float(direction_correct),
        "formal_quote_line_provenance_validity": float(formal_provenance_valid),
        "schema_direction_provenance_joint_validity": float(joint_validity),
        "gold_evidence_line_agreement": float(gold_line_agreement),
        "gold_evidence_quote_token_agreement_f1": quote_f1,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def article_clustered_interval(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Prompt-weighted point estimate with an article-cluster percentile bootstrap."""

    if replicates < 100:
        raise EvidenceInferenceDiagnosticError("cluster bootstrap requires at least 100 replicates")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        paper_id = row.get("paper_id")
        value = row.get(metric)
        if not isinstance(paper_id, str) or not isinstance(value, (float, int)):
            raise EvidenceInferenceDiagnosticError(f"invalid clustered metric row: {metric}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise EvidenceInferenceDiagnosticError(f"nonfinite clustered metric: {metric}")
        clusters[paper_id].append(numeric)
    if not clusters:
        return {
            "status": "no_rows",
            "estimate": None,
            "lower": None,
            "upper": None,
            "rows": 0,
            "articles": 0,
            "replicates": replicates,
            "method": "article_cluster_percentile_bootstrap_95",
        }
    point = math.fsum(value for values in clusters.values() for value in values) / sum(
        len(values) for values in clusters.values()
    )
    paper_ids = sorted(clusters)
    metric_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{metric}".encode()).digest()[:8], "big"
    )
    generator = random.Random(metric_seed)
    bootstrapped: list[float] = []
    for _ in range(replicates):
        sampled = [paper_ids[generator.randrange(len(paper_ids))] for _ in paper_ids]
        numerator = math.fsum(value for paper_id in sampled for value in clusters[paper_id])
        denominator = sum(len(clusters[paper_id]) for paper_id in sampled)
        bootstrapped.append(numerator / denominator)
    bootstrapped.sort()
    return {
        "status": "estimated",
        "estimate": point,
        "lower": _quantile(bootstrapped, 0.025),
        "upper": _quantile(bootstrapped, 0.975),
        "rows": sum(len(values) for values in clusters.values()),
        "articles": len(clusters),
        "replicates": replicates,
        "method": "article_cluster_percentile_bootstrap_95",
    }


def summarize_scored_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    failures = Counter(str(row.get("primary_failure")) for row in rows)
    confusion = Counter(
        f"{row.get('expected_direction')}->{row.get('predicted_direction') or 'missing'}"
        for row in rows
    )
    return {
        "rows": len(rows),
        "articles": len({str(row.get("paper_id")) for row in rows}),
        "metrics": {
            metric: article_clustered_interval(
                rows, metric, seed=seed, replicates=replicates
            )
            for metric in _METRIC_FIELDS
        },
        "failure_taxonomy": dict(sorted(failures.items())),
        "direction_confusion": dict(sorted(confusion.items())),
    }


def _redacted_prediction_ledger_row(prediction: Mapping[str, Any]) -> dict[str, Any]:
    output = prediction.get("output")
    eligible = output.get("eligible") if isinstance(output, Mapping) else None
    findings = output.get("findings") if isinstance(output, Mapping) else None
    finding = findings[0] if isinstance(findings, list) and len(findings) == 1 else None
    quote = finding.get("evidence_quote") if isinstance(finding, Mapping) else None
    lines = finding.get("evidence_lines") if isinstance(finding, Mapping) else None
    return {
        "example_id": prediction["example_id"],
        "paper_id": prediction["paper_id"],
        "eligible": eligible,
        "predicted_direction": (
            finding.get("direction") if isinstance(finding, Mapping) else None
        ),
        "evidence_lines": deepcopy(lines) if isinstance(lines, list) else None,
        "evidence_quote_sha256": (
            hashlib.sha256(quote.encode("utf-8")).hexdigest()
            if isinstance(quote, str)
            else None
        ),
        "exact_output_sha256": hash_canonical(output),
        "disposition": prediction["disposition"],
    }


def _run_full_lexical_diagnostic_with_ledger(
    examples: Sequence[OptimizationExample],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    for example in examples:
        # The extractor receives a closed projection with no expected output or labels.
        output, disposition = lexical_extraction_output(_label_stripped_input(example))
        dispositions[disposition] += 1
        predictions.append(
            {
                "example_id": example.example_id,
                "paper_id": example.paper_id,
                "output": output,
                "disposition": disposition,
            }
        )
    prediction_freeze_sha256 = hash_canonical(predictions)
    by_id = {example.example_id: example for example in examples}
    scored = [
        score_diagnostic_output(
            by_id[prediction["example_id"]],
            prediction["output"],
            baseline_disposition=prediction["disposition"],
        )
        for prediction in predictions
    ]
    ledger_payload = {
        "prediction_ledger_version": PREDICTION_LEDGER_VERSION,
        "status": "diagnostic_only_non_pristine",
        "contains_article_text": False,
        "contains_gold_labels": False,
        "quote_representation": "sha256_only",
        "rows": [_redacted_prediction_ledger_row(row) for row in predictions],
    }
    ledger = {**ledger_payload, "ledger_sha256": hash_canonical(ledger_payload)}
    result = {
        "baseline_id": LEXICAL_BASELINE_VERSION,
        "provider_calls": 0,
        "training_labels_used": False,
        "prediction_stage_received_label_fields": False,
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "redacted_prediction_ledger_sha256": ledger["ledger_sha256"],
        "prediction_rows": len(predictions),
        "model_facing_inputs_only": [
            "OUTCOME",
            "INTERVENTION",
            "COMPARATOR",
            "content_lines",
        ],
        "dispositions": dict(sorted(dispositions.items())),
        "summary": summarize_scored_rows(scored, seed=seed, replicates=replicates),
    }
    return result, ledger


def run_full_lexical_diagnostic(
    examples: Sequence[OptimizationExample],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the lexical diagnostic while withholding the separate redacted ledger."""

    result, _ = _run_full_lexical_diagnostic_with_ledger(
        examples, seed=seed, replicates=replicates
    )
    return result


def _candidate_sha256(candidate: Mapping[str, str]) -> str:
    if set(candidate) != {"extraction_prompt"} or not isinstance(
        candidate.get("extraction_prompt"), str
    ):
        raise EvidenceInferenceDiagnosticError("archived extraction candidate is invalid")
    return hash_canonical(dict(candidate))


def _load_candidate_file(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceDiagnosticError(f"cannot read candidate archive: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvidenceInferenceDiagnosticError(f"candidate archive is empty: {path}")
    candidates: list[dict[str, str]] = []
    for value in payload:
        if not isinstance(value, Mapping):
            raise EvidenceInferenceDiagnosticError(f"candidate archive row is invalid: {path}")
        candidate = {str(key): item for key, item in value.items() if isinstance(item, str)}
        _candidate_sha256(candidate)
        candidates.append(candidate)
    hashes = [_candidate_sha256(candidate) for candidate in candidates]
    if len(hashes) != len(set(hashes)):
        raise EvidenceInferenceDiagnosticError(f"candidate archive has duplicates: {path}")
    return candidates


def _load_run_candidates(
    run_dir: Path, candidate_path: Path
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Recover accepted and rejected proposals, preserving exact logger bytes."""

    frozen_candidates = _load_candidate_file(candidate_path)
    candidates: list[dict[str, str]] = [frozen_candidates[0]]
    provenance: list[dict[str, Any]] = [
        {"source": "candidates_json_seed", "iteration": None}
    ]
    run_log_path = run_dir / "extraction" / "run_log.txt"
    if run_log_path.is_file():
        try:
            run_log_text = run_log_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise EvidenceInferenceDiagnosticError(
                f"cannot read GEPA proposal log: {run_log_path}"
            ) from exc
        for match in _PROPOSED_PROMPT.finditer(run_log_text):
            candidates.append({"extraction_prompt": match.group("prompt")})
            provenance.append(
                {
                    "source": "run_log_exact_proposal",
                    "iteration": int(match.group("iteration")),
                }
            )
    known_hashes = {_candidate_sha256(candidate) for candidate in candidates}
    for candidate in frozen_candidates[1:]:
        candidate_sha256 = _candidate_sha256(candidate)
        if candidate_sha256 in known_hashes:
            continue
        candidates.append(candidate)
        provenance.append({"source": "candidates_json_only", "iteration": None})
        known_hashes.add(candidate_sha256)
    hashes = [_candidate_sha256(candidate) for candidate in candidates]
    if len(hashes) != len(set(hashes)):
        raise EvidenceInferenceDiagnosticError(
            f"GEPA run log repeats a mutation candidate: {run_dir}"
        )
    return candidates, provenance


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceDiagnosticError(f"cannot read archived JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceInferenceDiagnosticError(f"archived JSON must be an object: {path}")
    return payload


def _proposal_pairing_contract(
    run_dir: Path,
) -> tuple[dict[int, list[str]], dict[str, Any] | None]:
    """Recover GEPA's exact adaptive minibatches from immutable run inputs.

    GEPA's recorded ``subsample_ids`` index the physical training JSONL, not the
    sorted example-id list written to the optimization trace.  Resolving them
    against the physical file prevents a silent, scientifically material pairing
    error.
    """

    trace_path = run_dir / "optimization_trace.json"
    log_path = run_dir / "extraction" / "run_log.json"
    if not trace_path.is_file() or not log_path.is_file():
        return {}, None
    trace = _read_json_object(trace_path)
    manifest_value = trace.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise EvidenceInferenceDiagnosticError(
            f"optimization trace lacks its source manifest: {trace_path}"
        )
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = load_split_manifest(manifest_path)
    manifest_file_sha256 = sha256_file(manifest_path)
    if trace.get("manifest_sha256") != manifest_file_sha256:
        raise EvidenceInferenceDiagnosticError("GEPA trace manifest hash mismatch")
    if trace.get("manifest_seed") != manifest.seed:
        raise EvidenceInferenceDiagnosticError("GEPA trace manifest seed mismatch")
    if trace.get("train_example_ids") != manifest.train.example_ids:
        raise EvidenceInferenceDiagnosticError("GEPA trace train identities mismatch")
    if trace.get("dev_example_ids") != manifest.dev.example_ids:
        raise EvidenceInferenceDiagnosticError("GEPA trace development identities mismatch")
    if (
        trace.get("optimization_splits") != ["train", "dev"]
        or trace.get("test_split_opened") is not False
        or trace.get("test_evaluated") is not False
    ):
        raise EvidenceInferenceDiagnosticError("GEPA trace split contract mismatch")
    physical_train = load_manifest_split(manifest_path, "train")
    physical_dev = load_manifest_split(manifest_path, "dev")
    physical_ids = [example.example_id for example in physical_train]
    if (
        len(physical_ids) != manifest.train.rows
        or sorted(example.example_id for example in physical_dev)
        != manifest.dev.example_ids
    ):
        raise EvidenceInferenceDiagnosticError("GEPA pairing manifest row mismatch")
    try:
        raw_log = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceDiagnosticError(f"cannot read GEPA run log: {log_path}") from exc
    if not isinstance(raw_log, list):
        raise EvidenceInferenceDiagnosticError(f"GEPA run log must be a list: {log_path}")
    by_iteration: dict[int, list[str]] = {}
    for row in raw_log:
        if not isinstance(row, Mapping):
            raise EvidenceInferenceDiagnosticError("GEPA run-log row must be an object")
        zero_based_iteration = row.get("i")
        raw_indices = row.get("subsample_ids")
        tasks = row.get("tasks")
        if (
            not isinstance(zero_based_iteration, int)
            or zero_based_iteration < 0
            or not isinstance(raw_indices, list)
            or not all(isinstance(value, int) for value in raw_indices)
            or row.get("n_tasks") != 1
            or not isinstance(tasks, list)
            or len(tasks) != 1
            or not isinstance(tasks[0], Mapping)
            or tasks[0].get("parent_idx") != 0
            or tasks[0].get("subsample_ids") != raw_indices
        ):
            raise EvidenceInferenceDiagnosticError("GEPA run-log pairing contract mismatch")
        if any(index < 0 or index >= len(physical_ids) for index in raw_indices):
            raise EvidenceInferenceDiagnosticError("GEPA subsample index is out of range")
        iteration = zero_based_iteration + 1
        if iteration in by_iteration:
            raise EvidenceInferenceDiagnosticError("GEPA run log repeats an iteration")
        by_iteration[iteration] = [physical_ids[index] for index in raw_indices]
    trace_config_fields = (
        "run_id",
        "optimizer",
        "gepa_version",
        "manifest_sha256",
        "manifest_seed",
        "optimizer_seed",
        "optimization_splits",
        "objective_weights",
        "cost_cap_usd",
        "max_metric_calls_per_prompt",
        "max_reflection_cost_usd_per_prompt",
        "reflection_minibatch_size",
        "reflection_lm_kwargs",
        "reflection_lm_identity",
        "task_provider_identity_sha256",
    )
    trace_configuration = {field: trace.get(field) for field in trace_config_fields}
    metadata = {
        "status": "verified",
        "optimization_trace_sha256": sha256_file(trace_path),
        "trace_configuration_sha256": hash_canonical(trace_configuration),
        "trace_configuration": trace_configuration,
        "run_log_json_sha256": sha256_file(log_path),
        "source_manifest_sha256": manifest_file_sha256,
        "source_train_jsonl_sha256": manifest.train.sha256,
        "source_dev_jsonl_sha256": manifest.dev.sha256,
        "physical_train_rows": len(physical_ids),
        "physical_dev_rows": len(physical_dev),
        "index_semantics": "zero_based_physical_train_jsonl_order",
        "reflection_budget": {
            "max_reflection_cost_usd_per_prompt": trace.get(
                "max_reflection_cost_usd_per_prompt"
            ),
            "reflection_minibatch_size": trace.get("reflection_minibatch_size"),
            "reflection_lm_kwargs": trace.get("reflection_lm_kwargs"),
            "actual_reflection_usage_status": (
                "verified_from_trace"
                if isinstance(trace.get("reflection_lm_usage"), Mapping)
                else "unavailable_in_archived_trace"
            ),
            "actual_reflection_usage": trace.get("reflection_lm_usage"),
            "candidate_generation_cost_reconstructed": False,
        },
    }
    return by_iteration, metadata


def _provider_prompt(example: OptimizationExample, candidate: Mapping[str, str]) -> str:
    try:
        rendered, _ = render_prompt_text(
            candidate["extraction_prompt"], example.replacements
        )
    except (KeyError, PromptContractError) as exc:
        raise EvidenceInferenceDiagnosticError(
            f"cannot render archived candidate: {example.example_id}"
        ) from exc
    source_json = json.dumps(
        example.content_lines,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        f"{rendered.rstrip()}\n\n"
        "## Paper source lines supplied by the evaluation harness\n\n"
        f"```json\n{source_json}\n```\n"
    )


def _validate_archived_request(
    *,
    path: Path,
    receipt: Mapping[str, Any],
    example: OptimizationExample,
    candidate: Mapping[str, str],
    allow_archived_schema_mismatch: bool,
) -> tuple[bool, str]:
    candidate_sha256 = _candidate_sha256(candidate)
    expected_key = f"{example.example_id}-{candidate_sha256[:16]}"
    schema_matches_current = receipt.get("output_schema") == example.output_schema
    if (
        receipt.get("operation") != "gepa-extraction"
        or receipt.get("request_key") != expected_key
        or receipt.get("prompt") != _provider_prompt(example, candidate)
        or (not schema_matches_current and not allow_archived_schema_mismatch)
    ):
        raise EvidenceInferenceDiagnosticError(
            f"archived provider request does not match candidate/example: {path}"
        )
    request_fields = (
        "operation",
        "request_key",
        "provider",
        "model",
        "effort",
        "max_tokens",
        "system",
        "prompt",
        "output_schema",
        "output_schema_original_sha256",
        "output_schema_provider",
        "output_schema_provider_sha256",
        "output_schema_transform",
    )
    request_payload = {field: receipt.get(field) for field in request_fields}
    request_hash_contract = "provider-schema-transform-v2"
    observed_request_sha256 = receipt.get("request_sha256")
    request_hash_valid = sha256_json(request_payload) == observed_request_sha256
    if not request_hash_valid and allow_archived_schema_mismatch:
        legacy_fields = request_fields[:9]
        legacy_payload = {field: receipt.get(field) for field in legacy_fields}
        request_hash_valid = sha256_json(legacy_payload) == observed_request_sha256
        request_hash_contract = "legacy-raw-schema-v1"
    if not request_hash_valid:
        raise EvidenceInferenceDiagnosticError(
            f"archived provider request hash mismatch: {path}"
        )
    return schema_matches_current, request_hash_contract


def _receipt_call(
    path: Path,
    candidate_hashes: Mapping[str, str],
    candidates_by_sha256: Mapping[str, Mapping[str, str]],
    examples_by_id: Mapping[str, tuple[str, OptimizationExample]],
    allow_archived_schema_mismatch: bool = False,
) -> _ArchivedCall | None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, Mapping):
        return None
    request_key = receipt.get("request_key")
    if not isinstance(request_key, str) or (match := _REQUEST_KEY.fullmatch(request_key)) is None:
        return None
    candidate_sha256 = candidate_hashes.get(match.group("candidate_prefix"))
    example_entry = examples_by_id.get(match.group("example_id"))
    if candidate_sha256 is None or example_entry is None:
        return None
    schema_matches_current, request_hash_contract = _validate_archived_request(
        path=path,
        receipt=receipt,
        example=example_entry[1],
        candidate=candidates_by_sha256[candidate_sha256],
        allow_archived_schema_mismatch=allow_archived_schema_mismatch,
    )
    return _ArchivedCall(
        path=path,
        example_id=match.group("example_id"),
        candidate_sha256=candidate_sha256,
        receipt=receipt,
        schema_matches_current=schema_matches_current,
        request_hash_contract=request_hash_contract,
    )


def _archived_output(call: _ArchivedCall) -> tuple[Any, str | None]:
    status = call.receipt.get("status")
    if status != "complete":
        failure = call.receipt.get("failure")
        suffix = str(failure) if failure else str(status or "unknown")
        return None, f"archived_provider_failure:{suffix}"
    response_text = call.receipt.get("response_text")
    if not isinstance(response_text, str):
        return None, "archived_response_text_missing"
    try:
        output = json.loads(response_text)
    except json.JSONDecodeError:
        return None, "archived_exact_json_invalid"
    if not isinstance(output, dict):
        return None, "archived_output_root_not_object"
    return output, None


def _score_archived_call(
    call: _ArchivedCall | None, example: OptimizationExample
) -> dict[str, Any]:
    if call is None:
        return score_diagnostic_output(
            example, None, execution_failure="missing_archived_response"
        )
    output, failure = _archived_output(call)
    return score_diagnostic_output(example, output, execution_failure=failure)


def _paired_delta_summary(
    seed_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if len(seed_rows) != len(mutation_rows):
        raise EvidenceInferenceDiagnosticError("paired archive arms have unequal calls")
    delta_rows = [
        {
            "paper_id": seed_row["paper_id"],
            **{
                metric: float(mutation_row[metric]) - float(seed_row[metric])
                for metric in _METRIC_FIELDS
            },
        }
        for seed_row, mutation_row in zip(seed_rows, mutation_rows, strict=True)
    ]
    return {
        metric: article_clustered_interval(
            delta_rows, metric, seed=seed, replicates=replicates
        )
        for metric in _METRIC_FIELDS
    }


def _relative_run_id(run_dir: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            relative = run_dir.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return f"{root.name}/{relative.as_posix()}"
    return run_dir.name


def _binary_missing_response_bounds(
    rows: Sequence[Mapping[str, Any]], total_rows: int
) -> dict[str, dict[str, float | int]]:
    if total_rows < len(rows):
        raise EvidenceInferenceDiagnosticError("missing-response denominator is invalid")
    missing = total_rows - len(rows)
    return {
        metric: {
            "observed_successes": sum(int(float(row[metric])) for row in rows),
            "observed_rows": len(rows),
            "missing_rows": missing,
            "all_missing_fail_lower": (
                sum(float(row[metric]) for row in rows) / total_rows
                if total_rows
                else 0.0
            ),
            "all_missing_succeed_upper": (
                (sum(float(row[metric]) for row in rows) + missing) / total_rows
                if total_rows
                else 0.0
            ),
        }
        for metric in _METRIC_FIELDS
        if metric != "gold_evidence_quote_token_agreement_f1"
    }


def _receipt_execution_contract(call: _ArchivedCall) -> dict[str, Any]:
    fields = {
        "provider": call.receipt.get("provider"),
        "model": call.receipt.get("model"),
        "effort": call.receipt.get("effort"),
        "max_tokens": call.receipt.get("max_tokens"),
        "system": call.receipt.get("system"),
    }
    missing = [
        field
        for field in ("provider", "model", "effort")
        if not isinstance(fields[field], str) or not fields[field]
    ]
    if not isinstance(fields["max_tokens"], int) or fields["max_tokens"] <= 0:
        missing.append("max_tokens")
    if fields["system"] is not None and not isinstance(fields["system"], str):
        missing.append("system")
    if missing:
        return {
            "status": "unverified_missing_or_invalid_fields",
            "missing_or_invalid_fields": sorted(set(missing)),
            "fields": fields,
            "contract_sha256": None,
        }
    return {
        "status": "verified",
        "missing_or_invalid_fields": [],
        "fields": fields,
        "contract_sha256": hash_canonical(fields),
    }


def _paired_execution_contract(
    seed_calls: Sequence[_ArchivedCall],
    mutation_calls: Sequence[_ArchivedCall],
    pairing_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len(seed_calls) != len(mutation_calls):
        raise EvidenceInferenceDiagnosticError("paired execution arms have unequal calls")
    if not seed_calls:
        return {
            "status": "unverified_no_paired_calls",
            "receipt_contract_verified": False,
            "trace_configuration_verified": pairing_metadata is not None,
        }
    contracts = [
        _receipt_execution_contract(call) for call in (*seed_calls, *mutation_calls)
    ]
    invalid = [contract for contract in contracts if contract["status"] != "verified"]
    if invalid:
        return {
            "status": "unverified_missing_receipt_execution_fields",
            "receipt_contract_verified": False,
            "trace_configuration_verified": pairing_metadata is not None,
            "issues": [contract["missing_or_invalid_fields"] for contract in invalid],
        }
    contract_hashes = {str(contract["contract_sha256"]) for contract in contracts}
    if len(contract_hashes) != 1:
        raise EvidenceInferenceDiagnosticError(
            "paired seed/mutation receipts use unequal provider execution contracts"
        )
    contract = contracts[0]
    if pairing_metadata is None:
        return {
            "status": "receipt_contract_verified_trace_configuration_unavailable",
            "receipt_contract_verified": True,
            "trace_configuration_verified": False,
            "receipt_execution_contract": contract["fields"],
            "receipt_execution_contract_sha256": contract["contract_sha256"],
            "trace_configuration_sha256": None,
        }
    return {
        "status": "verified",
        "receipt_contract_verified": True,
        "trace_configuration_verified": True,
        "receipt_execution_contract": contract["fields"],
        "receipt_execution_contract_sha256": contract["contract_sha256"],
        "trace_configuration_sha256": pairing_metadata["trace_configuration_sha256"],
    }


def _archived_call_telemetry(calls: Sequence[_ArchivedCall]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    token_rows = 0
    estimated_cost = 0.0
    cost_rows = 0
    cost_bases: Counter[str] = Counter()
    for call in calls:
        usage = call.receipt.get("usage")
        if isinstance(usage, Mapping):
            raw_input = usage.get("input_tokens")
            raw_output = usage.get("output_tokens")
            if (
                isinstance(raw_input, int)
                and raw_input >= 0
                and isinstance(raw_output, int)
                and raw_output >= 0
            ):
                input_tokens += raw_input
                output_tokens += raw_output
                token_rows += 1
        raw_cost = call.receipt.get("estimated_cost_usd")
        if isinstance(raw_cost, (int, float)) and math.isfinite(float(raw_cost)) and raw_cost >= 0:
            estimated_cost += float(raw_cost)
            cost_rows += 1
        raw_basis = call.receipt.get("cost_basis")
        if isinstance(raw_basis, str) and raw_basis:
            cost_bases[raw_basis] += 1
    return {
        "archived_calls": len(calls),
        "complete_calls": sum(call.receipt.get("status") == "complete" for call in calls),
        "failed_calls": sum(call.receipt.get("status") != "complete" for call in calls),
        "token_accounting_status": (
            "verified_from_all_receipts" if token_rows == len(calls) else "incomplete"
        ),
        "input_tokens": input_tokens if token_rows == len(calls) else None,
        "output_tokens": output_tokens if token_rows == len(calls) else None,
        "estimated_cost_accounting_status": (
            "verified_from_all_receipts" if cost_rows == len(calls) else "incomplete"
        ),
        "estimated_cost_usd": estimated_cost if cost_rows == len(calls) else None,
        "cost_basis_counts": dict(sorted(cost_bases.items())),
        "cost_interpretation": (
            "receipt estimate under recorded basis; not necessarily billed cost"
        ),
    }


def evaluate_archived_gepa_runs(
    *,
    archive_roots: Sequence[Path],
    examples_by_id: Mapping[str, tuple[str, OptimizationExample]],
    test_examples: Sequence[OptimizationExample],
    seed_prompt_text: str,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Rescore all discovered mutations on exact seed-paired historical calls."""

    roots = [path for path in archive_roots if path.is_dir()]
    candidate_paths = sorted(
        {
            path
            for root in roots
            for path in root.rglob("extraction/candidates.json")
            if path.is_file()
        }
    )
    seed_sha256 = _candidate_sha256({"extraction_prompt": seed_prompt_text})
    runs: list[dict[str, Any]] = []
    opened_test_seed_replays: list[dict[str, Any]] = []
    archive_files: list[dict[str, str]] = []
    mutation_counts: dict[str, Counter[str]] = {
        "eligible_diagnostic": Counter(),
        "excluded_failed": Counter(),
    }
    bound_test_attempt_example_ids: set[str] = set()
    bound_test_attempt_paper_ids: set[str] = set()
    bound_test_manifest_sha256s: set[str] = set()
    for candidate_path in candidate_paths:
        run_dir = candidate_path.parent.parent
        run_id = _relative_run_id(run_dir, roots)
        archive_status: Literal["diagnostic_archive", "excluded_failed_archive"] = (
            "excluded_failed_archive"
            if "gepa-failed-runs" in run_dir.parts
            else "diagnostic_archive"
        )
        candidates, candidate_provenance = _load_run_candidates(run_dir, candidate_path)
        proposal_pairings, pairing_metadata = _proposal_pairing_contract(run_dir)
        candidate_shas = [_candidate_sha256(candidate) for candidate in candidates]
        if seed_sha256 not in candidate_shas:
            raise EvidenceInferenceDiagnosticError(
                f"archived run lacks exact handwritten seed: {run_dir}"
            )
        trace_path = run_dir / "optimization_trace.json"
        trace = _read_json_object(trace_path) if trace_path.is_file() else None
        if trace is not None:
            component_traces = trace.get("component_traces")
            component = (
                component_traces.get("extraction")
                if isinstance(component_traces, Mapping)
                else None
            )
            if not isinstance(component, Mapping):
                raise EvidenceInferenceDiagnosticError("GEPA extraction trace is missing")
            trace_candidate_payloads = component.get("candidates")
            trace_candidate_hashes = component.get("candidate_sha256s")
            seed_prompt_hashes = trace.get("seed_prompt_sha256s")
            if (
                not isinstance(trace_candidate_payloads, list)
                or not isinstance(trace_candidate_hashes, list)
                or [_candidate_sha256(candidate) for candidate in trace_candidate_payloads]
                != trace_candidate_hashes
                or not set(trace_candidate_hashes) <= set(candidate_shas)
                or trace_candidate_hashes[0] != seed_sha256
                or not isinstance(seed_prompt_hashes, Mapping)
                or seed_prompt_hashes.get("extraction")
                != hashlib.sha256(seed_prompt_text.encode("utf-8")).hexdigest()
            ):
                raise EvidenceInferenceDiagnosticError("GEPA trace candidate binding mismatch")
        prefixes: dict[str, str] = {}
        candidates_by_sha256 = {
            _candidate_sha256(candidate): candidate for candidate in candidates
        }
        for candidate_sha in candidate_shas:
            prefix = candidate_sha[:16]
            if prefix in prefixes:
                raise EvidenceInferenceDiagnosticError("archived candidate hash prefix collision")
            prefixes[prefix] = candidate_sha
        receipt_paths = sorted(run_dir.rglob("*.provider.json"))
        archive_files.append(
            {
                "path": f"{run_id}/{candidate_path.relative_to(run_dir).as_posix()}",
                "sha256": sha256_file(candidate_path),
            }
        )
        archive_files.extend(
            {
                "path": f"{run_id}/{path.relative_to(run_dir).as_posix()}",
                "sha256": sha256_file(path),
            }
            for path in receipt_paths
        )
        for auxiliary_path in (
            run_dir / "optimization_trace.json",
            run_dir / "extraction" / "run_log.json",
            run_dir / "extraction" / "run_log.txt",
            run_dir / "heldout-test.json",
        ):
            if auxiliary_path.is_file():
                archive_files.append(
                    {
                        "path": f"{run_id}/{auxiliary_path.relative_to(run_dir).as_posix()}",
                        "sha256": sha256_file(auxiliary_path),
                    }
                )
        calls: dict[str, dict[str, _ArchivedCall]] = defaultdict(dict)
        identical_duplicate_calls = 0
        ignored_receipts = 0
        for receipt_path in receipt_paths:
            call = _receipt_call(
                receipt_path,
                prefixes,
                candidates_by_sha256,
                examples_by_id,
                allow_archived_schema_mismatch=(
                    archive_status == "excluded_failed_archive"
                ),
            )
            if call is None or call.example_id not in examples_by_id:
                ignored_receipts += 1
                continue
            if call.example_id in calls[call.candidate_sha256]:
                prior = calls[call.candidate_sha256][call.example_id]
                if sha256_file(prior.path) != sha256_file(call.path):
                    raise EvidenceInferenceDiagnosticError(
                        "nonidentical duplicate candidate/example provider receipts"
                    )
                identical_duplicate_calls += 1
                continue
            calls[call.candidate_sha256][call.example_id] = call
        if archive_status == "diagnostic_archive":
            for candidate_calls in calls.values():
                for call in candidate_calls.values():
                    split_name, example = examples_by_id[call.example_id]
                    if split_name == "test":
                        bound_test_attempt_example_ids.add(call.example_id)
                        bound_test_attempt_paper_ids.add(example.paper_id)
        seed_calls = calls.get(seed_sha256, {})
        run_mutations: list[dict[str, Any]] = []
        count_key = (
            "eligible_diagnostic"
            if archive_status == "diagnostic_archive"
            else "excluded_failed"
        )
        for candidate_index, candidate_sha in enumerate(candidate_shas):
            if candidate_sha == seed_sha256:
                continue
            mutation_counts[count_key]["discovered"] += 1
            mutation_calls = calls.get(candidate_sha, {})
            all_paired_ids = sorted(set(seed_calls) & set(mutation_calls))
            iteration = candidate_provenance[candidate_index].get("iteration")
            recorded_ids = proposal_pairings.get(iteration) if isinstance(iteration, int) else None
            if recorded_ids is not None:
                missing_recorded = [
                    example_id
                    for example_id in recorded_ids
                    if example_id not in seed_calls or example_id not in mutation_calls
                ]
                if missing_recorded:
                    raise EvidenceInferenceDiagnosticError(
                        "GEPA proposal minibatch lacks an exact seed/mutation receipt pair: "
                        f"{run_id}:{iteration}:{missing_recorded}"
                    )
                paired_ids = list(recorded_ids)
                pairing_basis = "gepa_recorded_adaptive_training_minibatch"
            else:
                paired_ids = all_paired_ids
                pairing_basis = "all_exact_seed_mutation_receipt_intersections"
            if paired_ids:
                mutation_counts[count_key]["with_paired_calls"] += 1
            seed_rows: list[dict[str, Any]] = []
            mutation_rows: list[dict[str, Any]] = []
            split_counts: Counter[str] = Counter()
            for example_id in paired_ids:
                split_name, example = examples_by_id[example_id]
                split_counts[split_name] += 1
                seed_rows.append(_score_archived_call(seed_calls[example_id], example))
                mutation_rows.append(
                    _score_archived_call(mutation_calls[example_id], example)
                )
            pair_set_sha256 = hash_canonical(
                [
                    {
                        "example_id": example_id,
                        "paper_id": examples_by_id[example_id][1].paper_id,
                        "split": examples_by_id[example_id][0],
                    }
                    for example_id in paired_ids
                ]
            )
            paired_seed_calls = [seed_calls[example_id] for example_id in paired_ids]
            paired_mutation_calls = [
                mutation_calls[example_id] for example_id in paired_ids
            ]
            run_mutations.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_sha256": candidate_sha,
                    "candidate_provenance": candidate_provenance[candidate_index],
                    "prompt_text_sha256": hashlib.sha256(
                        candidates[candidate_index]["extraction_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "comparison_status": (
                        "paired_archived_calls" if paired_ids else "no_paired_archived_calls"
                    ),
                    "pairing_basis": pairing_basis,
                    "calls_per_arm": len(paired_ids),
                    "equal_call_contract_satisfied": len(seed_rows) == len(mutation_rows),
                    "paired_execution_contract": _paired_execution_contract(
                        paired_seed_calls, paired_mutation_calls, pairing_metadata
                    ),
                    "seed_arm_telemetry": _archived_call_telemetry(paired_seed_calls),
                    "mutation_arm_telemetry": _archived_call_telemetry(
                        paired_mutation_calls
                    ),
                    "paired_set_sha256": pair_set_sha256,
                    "split_counts": dict(sorted(split_counts.items())),
                    "seed": summarize_scored_rows(
                        seed_rows, seed=seed, replicates=replicates
                    ),
                    "mutation": summarize_scored_rows(
                        mutation_rows, seed=seed, replicates=replicates
                    ),
                    "paired_delta_mutation_minus_seed": _paired_delta_summary(
                        seed_rows,
                        mutation_rows,
                        seed=seed,
                        replicates=replicates,
                    ),
                    "all_exact_receipt_intersection_rows": len(all_paired_ids),
                    "all_exact_receipt_intersection_split_counts": dict(
                        sorted(
                            Counter(
                                examples_by_id[example_id][0]
                                for example_id in all_paired_ids
                            ).items()
                        )
                    ),
                }
            )
        unique_calls = [
            call
            for candidate_calls in calls.values()
            for call in candidate_calls.values()
        ]
        run_payload: dict[str, Any] = {
            "archive_run_id": run_id,
            "archive_status": archive_status,
            "candidate_file_sha256": sha256_file(candidate_path),
            "candidate_count_including_seed": len(candidates),
            "mutation_count": len(candidates) - 1,
            "recognized_receipts": len(unique_calls),
            "ignored_receipts": ignored_receipts,
            "byte_identical_duplicate_receipts_deduplicated": identical_duplicate_calls,
            "nonidentical_duplicate_receipts_accepted": 0,
            "archived_schema_mismatch_receipts": sum(
                not call.schema_matches_current for call in unique_calls
            ),
            "legacy_raw_schema_request_hash_receipts": sum(
                call.request_hash_contract == "legacy-raw-schema-v1"
                for call in unique_calls
            ),
            "archived_unique_call_telemetry": _archived_call_telemetry(unique_calls),
            "optimization_budget": (
                {
                    "status": "verified_from_trace",
                    "task_cost_cap_usd": pairing_metadata["trace_configuration"][
                        "cost_cap_usd"
                    ],
                    "max_metric_calls_per_prompt": pairing_metadata[
                        "trace_configuration"
                    ]["max_metric_calls_per_prompt"],
                    "reflection": pairing_metadata["reflection_budget"],
                }
                if pairing_metadata is not None
                else {
                    "status": "unavailable_no_optimization_trace",
                    "task_cost_cap_usd": None,
                    "max_metric_calls_per_prompt": None,
                    "reflection": None,
                }
            ),
            "trace_aggregate_scores_used": False,
            "direct_receipt_replay_bypasses_known_gepa_0_1_4_cache_id_collision": False,
            "proposal_pairing_contract": pairing_metadata,
            "mutations": run_mutations,
        }
        if trace is not None:
            component = trace.get("component_traces", {}).get("extraction", {})
            trace_candidates = component.get("candidate_sha256s", [])
            trace_scores = component.get("val_aggregate_scores", [])
            known_candidate_sha256 = str(
                _KNOWN_CACHE_DEFECT_EVIDENCE["candidate_sha256"]
            )
            known_mutation_index = (
                trace_candidates.index(known_candidate_sha256)
                if isinstance(trace_candidates, list)
                and known_candidate_sha256 in trace_candidates
                else None
            )
            if (
                pairing_metadata is not None
                and run_id == _KNOWN_CACHE_DEFECT_EVIDENCE["archive_run_id"]
                and pairing_metadata["optimization_trace_sha256"]
                == _KNOWN_CACHE_DEFECT_EVIDENCE["optimization_trace_sha256"]
                and trace.get("run_id") == _KNOWN_CACHE_DEFECT_EVIDENCE["trace_run_id"]
                and trace.get("gepa_version")
                == _KNOWN_CACHE_DEFECT_EVIDENCE["gepa_version"]
                and trace.get("optimizer") == _KNOWN_CACHE_DEFECT_EVIDENCE["optimizer"]
                and known_mutation_index is not None
                and isinstance(trace_scores, list)
                and len(trace_scores) == len(trace_candidates)
                and isinstance(trace_scores[known_mutation_index], (float, int))
                and math.isclose(
                    float(trace_scores[known_mutation_index]),
                    float(_KNOWN_CACHE_DEFECT_EVIDENCE["trace_score"]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                mutation_sha = known_candidate_sha256
                dev_ids = trace.get("dev_example_ids")
                if not isinstance(dev_ids, list) or not all(
                    isinstance(value, str) for value in dev_ids
                ):
                    raise EvidenceInferenceDiagnosticError(
                        "GEPA trace dev identities are invalid"
                    )
                common_dev_ids = sorted(
                    set(dev_ids)
                    & set(seed_calls)
                    & set(calls.get(mutation_sha, {}))
                )
                clean_seed_rows = [
                    _score_archived_call(seed_calls[example_id], examples_by_id[example_id][1])
                    for example_id in common_dev_ids
                ]
                clean_mutation_rows = [
                    _score_archived_call(
                        calls[mutation_sha][example_id], examples_by_id[example_id][1]
                    )
                    for example_id in common_dev_ids
                ]
                missing_dev_ids = sorted(set(dev_ids) - set(common_dev_ids))
                run_payload["cache_integrity_finding"] = {
                    "status": "fail_closed_trace_score_excluded",
                    "evidence_binding_status": "verified_exact_archived_run",
                    "evidence_binding": deepcopy(_KNOWN_CACHE_DEFECT_EVIDENCE),
                    "gepa_version": trace["gepa_version"],
                    "defect": (
                        "train_and_dev_loaders_reused_split_unnamespaced_numeric_cache_ids"
                    ),
                    "finding_scope": "this_exact_hash_bound_archived_run_only",
                    "archived_trace_mutation_dev_score": trace_scores[
                        known_mutation_index
                    ],
                    "archived_trace_mutation_dev_score_usable": False,
                    "archived_trace_score_citation_allowed": False,
                    "direct_receipt_replay_only": True,
                    "expected_dev_rows": len(dev_ids),
                    "clean_common_dev_receipts": len(common_dev_ids),
                    "missing_mutation_dev_receipts": len(missing_dev_ids),
                    "clean_common_dev_set_sha256": hash_canonical(common_dev_ids),
                    "missing_dev_set_sha256": hash_canonical(missing_dev_ids),
                    "seed_on_clean_common_dev": summarize_scored_rows(
                        clean_seed_rows, seed=seed, replicates=replicates
                    ),
                    "mutation_on_clean_common_dev": summarize_scored_rows(
                        clean_mutation_rows, seed=seed, replicates=replicates
                    ),
                    "paired_delta_on_clean_common_dev": _paired_delta_summary(
                        clean_seed_rows,
                        clean_mutation_rows,
                        seed=seed,
                        replicates=replicates,
                    ),
                    "mutation_missing_response_bounds_over_all_dev": (
                        _binary_missing_response_bounds(clean_mutation_rows, len(dev_ids))
                    ),
                    "interpretation": (
                        f"{len(common_dev_ids)} clean common receipts are diagnostic; "
                        f"{len(missing_dev_ids)} missing mutation responses are bounded, "
                        "not imputed"
                    ),
                }
                run_payload[
                    "direct_receipt_replay_bypasses_known_gepa_0_1_4_cache_id_collision"
                ] = True
        runs.append(run_payload)
        test_ids = {example.example_id for example in test_examples}
        available_test_ids = sorted(test_ids & set(seed_calls))
        if available_test_ids:
            heldout_path = run_dir / "heldout-test.json"
            if archive_status == "diagnostic_archive":
                if trace is None or not heldout_path.is_file():
                    raise EvidenceInferenceDiagnosticError(
                        "opened test receipts lack their hash-bound evaluation artifact"
                    )
                heldout = _read_json_object(heldout_path)
                heldout_ids = heldout.get("test_example_ids")
                if (
                    heldout.get("manifest_sha256") != trace.get("manifest_sha256")
                    or not isinstance(heldout_ids, list)
                    or sorted(heldout_ids) != available_test_ids
                    or heldout.get("split") != "test"
                ):
                    raise EvidenceInferenceDiagnosticError(
                        "opened test evaluation artifact does not bind its receipts"
                    )
                bound_test_manifest_sha256s.add(str(heldout["manifest_sha256"]))
            attempted_rows = [
                _score_archived_call(seed_calls[example_id], examples_by_id[example_id][1])
                for example_id in available_test_ids
            ]
            all_rows = [
                _score_archived_call(
                    seed_calls.get(example.example_id), example
                )
                for example in test_examples
            ]
            opened_test_seed_replays.append(
                {
                    "archive_run_id": run_id,
                    "candidate_sha256": seed_sha256,
                    "full_test_rows": len(test_examples),
                    "full_test_articles": len({example.paper_id for example in test_examples}),
                    "archived_calls": len(available_test_ids),
                    "archive_coverage": len(available_test_ids) / len(test_examples),
                    "archived_call_set_sha256": hash_canonical(available_test_ids),
                    "evaluation_artifact_sha256": (
                        sha256_file(heldout_path) if heldout_path.is_file() else None
                    ),
                    "attempted_call_telemetry": _archived_call_telemetry(
                        [seed_calls[example_id] for example_id in available_test_ids]
                    ),
                    "attempted_call_metrics": summarize_scored_rows(
                        attempted_rows, seed=seed, replicates=replicates
                    ),
                    "full_test_missing_as_failure": summarize_scored_rows(
                        all_rows, seed=seed, replicates=replicates
                    ),
                }
            )
    archive_files = sorted(archive_files, key=lambda row: (row["path"], row["sha256"]))
    eligible_counts = mutation_counts["eligible_diagnostic"]
    excluded_counts = mutation_counts["excluded_failed"]
    return {
        "handwritten_seed_candidate_sha256": seed_sha256,
        "archive_roots_present": [path.as_posix() for path in roots],
        "archive_input_files": len(archive_files),
        "archive_input_corpus_sha256": hash_canonical(archive_files),
        "runs": runs,
        "eligible_diagnostic_mutation_accounting": {
            "discovered": eligible_counts["discovered"],
            "reported": sum(
                len(run["mutations"])
                for run in runs
                if run["archive_status"] == "diagnostic_archive"
            ),
            "with_paired_calls": eligible_counts["with_paired_calls"],
            "without_paired_calls": (
                eligible_counts["discovered"] - eligible_counts["with_paired_calls"]
            ),
        },
        "excluded_failed_mutation_accounting": {
            "discovered": excluded_counts["discovered"],
            "reported": sum(
                len(run["mutations"])
                for run in runs
                if run["archive_status"] == "excluded_failed_archive"
            ),
            "with_paired_calls": excluded_counts["with_paired_calls"],
            "without_paired_calls": (
                excluded_counts["discovered"] - excluded_counts["with_paired_calls"]
            ),
        },
        "bound_local_test_exposure_registry": {
            "status": "verified_against_all_bound_successful_run_test_attempts",
            "provider_test_attempt_rows": len(bound_test_attempt_example_ids),
            "provider_test_attempt_articles": len(bound_test_attempt_paper_ids),
            "example_ids": sorted(bound_test_attempt_example_ids),
            "paper_ids": sorted(bound_test_attempt_paper_ids),
            "bound_manifest_sha256s": sorted(bound_test_manifest_sha256s),
            "scope_limit": "local hash-bound archives only; not proof of global nonexposure",
        },
        "opened_test_seed_archive_replays": opened_test_seed_replays,
    }


def _all_split_examples(
    manifest_path: Path,
) -> tuple[OptimizationSplitManifest, dict[str, tuple[str, OptimizationExample]]]:
    manifest = load_split_manifest(manifest_path)
    lookup: dict[str, tuple[str, OptimizationExample]] = {}
    for split_name in ("train", "dev", "test"):
        for example in load_manifest_split(manifest_path, split_name):
            if example.example_id in lookup:
                raise EvidenceInferenceDiagnosticError("example appears in multiple splits")
            lookup[example.example_id] = (split_name, example)
    return manifest, lookup


def _diagnostic_code_fingerprint() -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    component_names = (
        "evidence_inference_diagnostic.py",
        "grounding.py",
        "prompt_optimization.py",
        "prompting.py",
        "providers.py",
    )
    components = [
        {"path": name, "sha256": sha256_file(module_root / name)}
        for name in component_names
    ]
    return {
        "components": components,
        "code_fingerprint_sha256": hash_canonical(components),
    }


def build_provider_free_diagnostic_bundle(
    *,
    manifest_path: Path,
    previously_opened_manifest_path: Path,
    seed_prompt_path: Path,
    archive_roots: Sequence[Path],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a self-hashed report and separate redacted prediction ledger."""

    manifest, examples_by_id = _all_split_examples(manifest_path)
    test_examples = [
        example for split_name, example in examples_by_id.values() if split_name == "test"
    ]
    test_examples.sort(key=lambda example: example.example_id)
    exposure_manifest = load_split_manifest(previously_opened_manifest_path)
    declared_exposure_paper_ids = set(exposure_manifest.test.paper_ids)
    full_test_paper_ids = {example.paper_id for example in test_examples}
    if (
        not declared_exposure_paper_ids
        or not declared_exposure_paper_ids <= full_test_paper_ids
    ):
        raise EvidenceInferenceDiagnosticError(
            "provider-exposure manifest is not a subset of the full test split"
        )
    try:
        seed_prompt_text = seed_prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceInferenceDiagnosticError("cannot read handwritten seed prompt") from exc
    archived = evaluate_archived_gepa_runs(
        archive_roots=archive_roots,
        examples_by_id=examples_by_id,
        test_examples=test_examples,
        seed_prompt_text=seed_prompt_text,
        seed=seed,
        replicates=replicates,
    )
    exposure_registry = archived["bound_local_test_exposure_registry"]
    receipt_exposure_example_ids = set(exposure_registry["example_ids"])
    receipt_exposure_paper_ids = set(exposure_registry["paper_ids"])
    exposure_manifest_sha256 = sha256_file(previously_opened_manifest_path)
    if (
        receipt_exposure_example_ids != set(exposure_manifest.test.example_ids)
        or receipt_exposure_paper_ids != declared_exposure_paper_ids
        or exposure_registry["bound_manifest_sha256s"] != [exposure_manifest_sha256]
    ):
        raise EvidenceInferenceDiagnosticError(
            "declared provider-exposure registry does not match bound local receipts"
        )
    provider_call_unseen_examples = [
        example
        for example in test_examples
        if example.paper_id not in receipt_exposure_paper_ids
    ]
    all_test_lexical, all_test_ledger = _run_full_lexical_diagnostic_with_ledger(
        test_examples, seed=seed, replicates=replicates
    )
    provider_unseen_lexical, provider_unseen_ledger = (
        _run_full_lexical_diagnostic_with_ledger(
            provider_call_unseen_examples, seed=seed, replicates=replicates
        )
    )
    code_fingerprint = _diagnostic_code_fingerprint()
    input_fingerprint_payload = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "lexical_baseline_version": LEXICAL_BASELINE_VERSION,
        "full_manifest_sha256": sha256_file(manifest_path),
        "provider_exposure_manifest_sha256": exposure_manifest_sha256,
        "test_split_jsonl_sha256": manifest.test.sha256,
        "seed_prompt_sha256": sha256_file(seed_prompt_path),
        "archive_input_corpus_sha256": archived["archive_input_corpus_sha256"],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }
    input_fingerprint_sha256 = hash_canonical(input_fingerprint_payload)
    execution_fingerprint_sha256 = hash_canonical(
        {
            "code_fingerprint_sha256": code_fingerprint["code_fingerprint_sha256"],
            "input_fingerprint_sha256": input_fingerprint_sha256,
        }
    )
    ledger_payload = {
        "prediction_ledger_version": PREDICTION_LEDGER_VERSION,
        "status": "diagnostic_only_non_pristine",
        "contains_article_text": False,
        "contains_evidence_quotes": False,
        "contains_gold_labels": False,
        "contains_raw_source_lines": False,
        "execution_fingerprint_sha256": execution_fingerprint_sha256,
        "provider_call_unseen_paper_subset": provider_unseen_ledger,
        "all_opened_test_rows": all_test_ledger,
    }
    prediction_ledger = {
        **ledger_payload,
        "ledger_sha256": hash_canonical(ledger_payload),
    }
    excluded_rows_on_touched_articles = sum(
        example.paper_id in receipt_exposure_paper_ids for example in test_examples
    )
    payload: dict[str, Any] = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "diagnostic_only_non_pristine",
        "scientific_scope": (
            "direction classification, exact output syntax/task shape, formal quote-line "
            "copy provenance, and separate gold-span agreement; not entailment, semantic "
            "support, effect-size estimation, or scientific truth"
        ),
        "test_labels_previously_opened": True,
        "test_split_pristine": False,
        "source_rows_physically_colocate_inputs_and_labels": True,
        "physical_label_open_order_blind": False,
        "prediction_function_received_label_fields": False,
        "predictions_hashed_before_metric_computation": True,
        "confirmatory_claims_allowed": False,
        "semantic_support_or_entailment_measured": False,
        "gold_evidence_spans_available": True,
        "gold_span_agreement_reported_separately": True,
        "provider_calls_made": 0,
        "provider_credentials_read": False,
        "test_rows": len(test_examples),
        "test_articles": len({example.paper_id for example in test_examples}),
        "registered_previous_provider_test_attempt_rows": len(
            receipt_exposure_example_ids
        ),
        "provider_touched_test_articles": len(receipt_exposure_paper_ids),
        "excluded_rows_on_provider_touched_articles": excluded_rows_on_touched_articles,
        "excluded_sibling_rows_on_provider_touched_articles": (
            excluded_rows_on_touched_articles - len(receipt_exposure_example_ids)
        ),
        "provider_call_unseen_paper_diagnostic_rows": len(
            provider_call_unseen_examples
        ),
        "provider_call_unseen_paper_diagnostic_articles": len(
            {example.paper_id for example in provider_call_unseen_examples}
        ),
        "manifest_file_sha256": sha256_file(manifest_path),
        "provider_exposure_manifest_file_sha256": exposure_manifest_sha256,
        "test_split_jsonl_sha256": manifest.test.sha256,
        "seed_prompt_file_sha256": sha256_file(seed_prompt_path),
        "bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "unit": "article",
            "interval": "percentile_95",
        },
        "provider_call_unseen_paper_input_only_lexical_diagnostic": (
            provider_unseen_lexical
        ),
        "all_opened_test_input_only_lexical_diagnostic": all_test_lexical,
        "archived_gepa_response_replay": archived,
        "provider_exposure_registry_validation": {
            "status": "verified_against_all_bound_local_successful_run_test_attempts",
            "declared_attempt_rows": exposure_manifest.test.rows,
            "receipt_attempt_rows": len(receipt_exposure_example_ids),
            "declared_touched_articles": len(declared_exposure_paper_ids),
            "receipt_touched_articles": len(receipt_exposure_paper_ids),
            "coverage_limit": (
                "provider-call-unseen means unseen only in the hash-bound local archive "
                "registry; global exposure is not proven"
            ),
        },
        "code_fingerprint": code_fingerprint,
        "input_fingerprint": {
            "components": input_fingerprint_payload,
            "input_fingerprint_sha256": input_fingerprint_sha256,
        },
        "execution_fingerprint_sha256": execution_fingerprint_sha256,
        "prediction_ledger": {
            "artifact_required_for_reproduction": True,
            "ledger_sha256": prediction_ledger["ledger_sha256"],
            "contains_article_text": False,
            "contains_gold_labels": False,
            "provider_call_unseen_rows": len(provider_call_unseen_examples),
            "all_opened_test_rows": len(test_examples),
        },
        "interpretation_limits": [
            "all_test_results_are_diagnostic_because_labels_were_previously_opened",
            "no_subset_is_called_primary_or_pristine",
            "provider_call_unseen_is_limited_to_the_bound_local_exposure_registry",
            "the_lexical_baseline_is_not_a_language_model_or_prompt_quality_comparison",
            "archived_prompt_comparisons_use_only_exact_seed_mutation_call_intersections",
            "optimization_split_archive_comparisons_are_not_heldout_prompt_selection_evidence",
            "the_archived_gepa_0_1_4_trace_score_is_excluded_after_a_cache_id_collision",
            "formal_quote_line_provenance_is_copy_validity_not_entailment_or_support",
            "gold_span_agreement_is_separate_and_alternate_provenance_valid_spans_can_exist",
            "the_native_production_extraction_prompt_and_pipeline_are_out_of_scope",
            "no_new_provider_calls_or_costs_were_incurred",
        ],
        "live_execution": {
            "status": "not_run",
            "reason": "provider execution requires explicit user opt-in and budget approval",
            "credentials_inspected": False,
        },
    }
    report = {**payload, "report_sha256": hash_canonical(payload)}
    return report, prediction_ledger


def build_provider_free_diagnostic_report(
    *,
    manifest_path: Path,
    previously_opened_manifest_path: Path,
    seed_prompt_path: Path,
    archive_roots: Sequence[Path],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Build the report; use the bundle API when persisting its required ledger."""

    report, _ = build_provider_free_diagnostic_bundle(
        manifest_path=manifest_path,
        previously_opened_manifest_path=previously_opened_manifest_path,
        seed_prompt_path=seed_prompt_path,
        archive_roots=archive_roots,
        seed=seed,
        replicates=replicates,
    )
    return report


def validate_prediction_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(ledger))
    observed = snapshot.pop("ledger_sha256", None)
    if not isinstance(observed, str) or hash_canonical(snapshot) != observed:
        raise EvidenceInferenceDiagnosticError("prediction ledger hash mismatch")
    if (
        snapshot.get("prediction_ledger_version") != PREDICTION_LEDGER_VERSION
        or snapshot.get("status") != "diagnostic_only_non_pristine"
        or snapshot.get("contains_article_text") is not False
        or snapshot.get("contains_evidence_quotes") is not False
        or snapshot.get("contains_gold_labels") is not False
        or snapshot.get("contains_raw_source_lines") is not False
    ):
        raise EvidenceInferenceDiagnosticError("prediction ledger scope mismatch")
    for subset_name in ("provider_call_unseen_paper_subset", "all_opened_test_rows"):
        subset = snapshot.get(subset_name)
        if not isinstance(subset, Mapping) or not isinstance(subset.get("rows"), list):
            raise EvidenceInferenceDiagnosticError("prediction ledger subset is invalid")
        subset_snapshot = deepcopy(dict(subset))
        subset_hash = subset_snapshot.pop("ledger_sha256", None)
        if not isinstance(subset_hash, str) or hash_canonical(subset_snapshot) != subset_hash:
            raise EvidenceInferenceDiagnosticError("prediction ledger subset hash mismatch")
        for row in subset["rows"]:
            if not isinstance(row, Mapping) or any(
                forbidden in row
                for forbidden in (
                    "evidence_quote",
                    "expected_output",
                    "expected_direction",
                    "content_lines",
                )
            ):
                raise EvidenceInferenceDiagnosticError(
                    "prediction ledger contains protected source or label fields"
                )
    return dict(ledger)


def build_public_diagnostic_summary(
    report: Mapping[str, Any], prediction_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a check-in-safe numeric summary without row identities or predictions."""

    validated_report = validate_diagnostic_report(report)
    validated_ledger = validate_prediction_ledger(prediction_ledger)
    if validated_report["prediction_ledger"]["ledger_sha256"] != validated_ledger[
        "ledger_sha256"
    ]:
        raise EvidenceInferenceDiagnosticError("report and prediction ledger are unbound")
    archived = validated_report["archived_gepa_response_replay"]
    cache_findings = []
    for run in archived["runs"]:
        finding = run.get("cache_integrity_finding")
        if isinstance(finding, Mapping):
            cache_findings.append(
                {
                    "status": finding["status"],
                    "evidence_binding_status": finding["evidence_binding_status"],
                    "expected_dev_rows": finding["expected_dev_rows"],
                    "clean_common_dev_receipts": finding[
                        "clean_common_dev_receipts"
                    ],
                    "missing_mutation_dev_receipts": finding[
                        "missing_mutation_dev_receipts"
                    ],
                    "mutation_missing_response_bounds_over_all_dev": deepcopy(
                        finding["mutation_missing_response_bounds_over_all_dev"]
                    ),
                    "archived_trace_score_citation_allowed": False,
                }
            )
    provider_unseen_summary = validated_report[
        "provider_call_unseen_paper_input_only_lexical_diagnostic"
    ]["summary"]
    all_test_summary = validated_report[
        "all_opened_test_input_only_lexical_diagnostic"
    ]["summary"]

    def public_score_summary(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "rows": value["rows"],
            "articles": value["articles"],
            "metrics": deepcopy(value["metrics"]),
            "failure_taxonomy": deepcopy(value["failure_taxonomy"]),
        }

    payload = {
        "public_summary_version": "evidence-inference-diagnostic-public-summary-v1",
        "status": "metadata_only_diagnostic_non_pristine",
        "contains_article_text": False,
        "contains_evidence_quotes": False,
        "contains_labels": False,
        "contains_per_example_labels": False,
        "contains_paper_or_example_ids": False,
        "contains_raw_predictions": False,
        "full_report_sha256": validated_report["report_sha256"],
        "prediction_ledger_sha256": validated_ledger["ledger_sha256"],
        "execution_fingerprint_sha256": validated_report[
            "execution_fingerprint_sha256"
        ],
        "population": {
            "all_opened_test_rows": validated_report["test_rows"],
            "all_opened_test_articles": validated_report["test_articles"],
            "registered_previous_provider_test_attempt_rows": validated_report[
                "registered_previous_provider_test_attempt_rows"
            ],
            "provider_touched_test_articles": validated_report[
                "provider_touched_test_articles"
            ],
            "excluded_rows_on_provider_touched_articles": validated_report[
                "excluded_rows_on_provider_touched_articles"
            ],
            "provider_call_unseen_paper_diagnostic_rows": validated_report[
                "provider_call_unseen_paper_diagnostic_rows"
            ],
            "provider_call_unseen_paper_diagnostic_articles": validated_report[
                "provider_call_unseen_paper_diagnostic_articles"
            ],
        },
        "provider_call_unseen_paper_lexical_diagnostic": public_score_summary(
            provider_unseen_summary
        ),
        "all_opened_test_lexical_diagnostic": public_score_summary(all_test_summary),
        "eligible_diagnostic_mutation_accounting": deepcopy(
            archived["eligible_diagnostic_mutation_accounting"]
        ),
        "excluded_failed_mutation_accounting": deepcopy(
            archived["excluded_failed_mutation_accounting"]
        ),
        "archived_run_telemetry": [
            {
                "archive_status": run["archive_status"],
                "mutation_count": run["mutation_count"],
                "recognized_receipts": run["recognized_receipts"],
                "archived_unique_call_telemetry": deepcopy(
                    run["archived_unique_call_telemetry"]
                ),
                "optimization_budget": deepcopy(run["optimization_budget"]),
            }
            for run in archived["runs"]
        ],
        "cache_integrity_findings": cache_findings,
        "interpretation_boundaries": [
            "all labels and reported metrics are non-pristine diagnostics",
            "provider-call-unseen is limited to the bound local receipt registry",
            "formal quote-line provenance is byte-copy validity, not entailment",
            "gold-span agreement is separate and alternative valid spans may exist",
            "GEPA mutation comparisons are adaptive training diagnostics",
            "archived receipt costs are estimates; candidate/reflection cost is unverified "
            "when trace telemetry is absent",
            "no provider calls were made",
        ],
    }
    return {**payload, "public_summary_sha256": hash_canonical(payload)}


def validate_diagnostic_report(report: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(report))
    observed = snapshot.pop("report_sha256", None)
    if not isinstance(observed, str) or hash_canonical(snapshot) != observed:
        raise EvidenceInferenceDiagnosticError("diagnostic report hash mismatch")
    if (
        snapshot.get("diagnostic_version") != DIAGNOSTIC_VERSION
        or snapshot.get("status") != "diagnostic_only_non_pristine"
        or snapshot.get("provider_calls_made") != 0
        or snapshot.get("provider_credentials_read") is not False
        or snapshot.get("test_labels_previously_opened") is not True
        or snapshot.get("test_split_pristine") is not False
        or snapshot.get("semantic_support_or_entailment_measured") is not False
        or "primary_paper_clean_input_only_lexical_baseline" in snapshot
        or "paper_clean_diagnostic_rows" in snapshot
        or not isinstance(snapshot.get("execution_fingerprint_sha256"), str)
    ):
        raise EvidenceInferenceDiagnosticError("diagnostic report scope contract mismatch")
    ledger_ref = snapshot.get("prediction_ledger")
    if (
        not isinstance(ledger_ref, Mapping)
        or ledger_ref.get("contains_article_text") is not False
        or ledger_ref.get("contains_gold_labels") is not False
        or not isinstance(ledger_ref.get("ledger_sha256"), str)
    ):
        raise EvidenceInferenceDiagnosticError("diagnostic ledger binding is invalid")
    exposure_validation = snapshot.get("provider_exposure_registry_validation")
    if (
        not isinstance(exposure_validation, Mapping)
        or exposure_validation.get("status")
        != "verified_against_all_bound_local_successful_run_test_attempts"
        or exposure_validation.get("declared_attempt_rows")
        != exposure_validation.get("receipt_attempt_rows")
        or exposure_validation.get("declared_touched_articles")
        != exposure_validation.get("receipt_touched_articles")
    ):
        raise EvidenceInferenceDiagnosticError("provider exposure registry is unverified")
    archived = snapshot.get("archived_gepa_response_replay")
    if (
        not isinstance(archived, Mapping)
        or not isinstance(archived.get("runs"), list)
        or not isinstance(
            archived.get("eligible_diagnostic_mutation_accounting"), Mapping
        )
        or not isinstance(archived.get("excluded_failed_mutation_accounting"), Mapping)
        or "mutation_accounting" in archived
        or "heldout_seed_archive_replays" in archived
    ):
        raise EvidenceInferenceDiagnosticError("diagnostic GEPA archive section is invalid")
    for run in archived["runs"]:
        if (
            not isinstance(run, Mapping)
            or run.get("trace_aggregate_scores_used") is not False
            or run.get("nonidentical_duplicate_receipts_accepted") != 0
            or (
                run.get("archive_status") == "diagnostic_archive"
                and run.get("ignored_receipts") != 0
            )
        ):
            raise EvidenceInferenceDiagnosticError("GEPA trace aggregates must fail closed")
        mutations = run.get("mutations")
        if not isinstance(mutations, list):
            raise EvidenceInferenceDiagnosticError("GEPA mutation rows are invalid")
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                raise EvidenceInferenceDiagnosticError("GEPA mutation row is invalid")
            execution_contract = mutation.get("paired_execution_contract")
            if not isinstance(execution_contract, Mapping):
                raise EvidenceInferenceDiagnosticError(
                    "GEPA paired execution contract is absent"
                )
            if (
                run.get("archive_status") == "diagnostic_archive"
                and mutation.get("calls_per_arm", 0) > 0
                and execution_contract.get("status") != "verified"
            ):
                raise EvidenceInferenceDiagnosticError(
                    "eligible GEPA comparison lacks a verified execution contract"
                )
        finding = run.get("cache_integrity_finding")
        if finding is None:
            continue
        if not isinstance(finding, Mapping):
            raise EvidenceInferenceDiagnosticError("GEPA cache finding is invalid")
        expected = finding.get("expected_dev_rows")
        clean = finding.get("clean_common_dev_receipts")
        missing = finding.get("missing_mutation_dev_receipts")
        evidence_binding = finding.get("evidence_binding")
        pairing_contract = run.get("proposal_pairing_contract")
        if (
            finding.get("status") != "fail_closed_trace_score_excluded"
            or finding.get("evidence_binding_status")
            != "verified_exact_archived_run"
            or evidence_binding != _KNOWN_CACHE_DEFECT_EVIDENCE
            or run.get("archive_run_id")
            != _KNOWN_CACHE_DEFECT_EVIDENCE["archive_run_id"]
            or not isinstance(pairing_contract, Mapping)
            or pairing_contract.get("optimization_trace_sha256")
            != _KNOWN_CACHE_DEFECT_EVIDENCE["optimization_trace_sha256"]
            or finding.get("archived_trace_mutation_dev_score_usable") is not False
            or finding.get("archived_trace_score_citation_allowed") is not False
            or not isinstance(expected, int)
            or not isinstance(clean, int)
            or not isinstance(missing, int)
            or expected != clean + missing
        ):
            raise EvidenceInferenceDiagnosticError(
                "GEPA cache-collision finding did not fail closed"
            )
    return dict(report)


def validate_public_diagnostic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(summary))
    observed = snapshot.pop("public_summary_sha256", None)
    if not isinstance(observed, str) or hash_canonical(snapshot) != observed:
        raise EvidenceInferenceDiagnosticError("public diagnostic summary hash mismatch")
    if (
        snapshot.get("status") != "metadata_only_diagnostic_non_pristine"
        or snapshot.get("contains_article_text") is not False
        or snapshot.get("contains_evidence_quotes") is not False
        or snapshot.get("contains_labels") is not False
        or snapshot.get("contains_per_example_labels") is not False
        or snapshot.get("contains_paper_or_example_ids") is not False
        or snapshot.get("contains_raw_predictions") is not False
    ):
        raise EvidenceInferenceDiagnosticError("public diagnostic summary scope mismatch")

    forbidden_keys = {
        "content_lines",
        "direction_confusion",
        "evidence_quote",
        "example_id",
        "example_ids",
        "expected_direction",
        "expected_output",
        "labels",
        "paper_id",
        "paper_ids",
        "predicted_direction",
        "raw_predictions",
    }

    def inspect(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in forbidden_keys or inspect(item) for key, item in value.items()
            )
        if isinstance(value, list):
            return any(inspect(item) for item in value)
        return isinstance(value, str) and bool(
            re.search(r"\bei2-prompt-[0-9]+\b|\bPMC[0-9]+\b", value)
        )

    if inspect(snapshot):
        raise EvidenceInferenceDiagnosticError(
            "public diagnostic summary contains row-level identities or predictions"
        )
    return dict(summary)


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DIAGNOSTIC_VERSION",
    "LEXICAL_BASELINE_VERSION",
    "PREDICTION_LEDGER_VERSION",
    "EvidenceInferenceDiagnosticError",
    "article_clustered_interval",
    "build_provider_free_diagnostic_bundle",
    "build_provider_free_diagnostic_report",
    "build_public_diagnostic_summary",
    "evaluate_archived_gepa_runs",
    "lexical_extraction_output",
    "run_full_lexical_diagnostic",
    "score_diagnostic_output",
    "summarize_scored_rows",
    "validate_diagnostic_report",
    "validate_prediction_ledger",
    "validate_public_diagnostic_summary",
]
