from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from scripts import convert_evidence_inference as conversion_cli

from literature_multiverse.evidence_inference import (
    EvidenceInferenceContractError,
    convert_evidence_inference,
    write_evidence_inference_metadata_summary,
)
from literature_multiverse.grounding import ground_evidence
from literature_multiverse.prompt_optimization import (
    GEPAPromptAdapter,
    load_manifest_split,
    load_optimization_examples,
    load_split_manifest,
    provider_request_key,
)
from literature_multiverse.providers import FixtureProvider


@pytest.fixture
def evidence_fixture(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "evidence_inference_v2"


def test_converter_preserves_official_split_ids_and_grounding(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    result = convert_evidence_inference(evidence_fixture, tmp_path / "benchmark")

    assert result.rows == {"train": 1, "dev": 1, "test": 1}
    manifest = load_split_manifest(result.manifest_path)
    assert manifest.algorithm == "official-paper-groups-v1"
    assert manifest.train.paper_ids == ["PMC1001"]
    assert manifest.dev.paper_ids == ["PMC1002"]
    assert manifest.test.paper_ids == ["PMC1003"]
    assert manifest.train.group_ids == ["PMC1001"]
    assert not (
        set(manifest.train.paper_ids)
        & set(manifest.dev.paper_ids)
        & set(manifest.test.paper_ids)
    )

    train = load_manifest_split(result.manifest_path, "train")
    example = train[0]
    assert example.example_id == "ei2-prompt-1"
    assert example.paper_id == example.group_id == "PMC1001"
    assert example.replacements == {
        "COMPARATOR": "placebo",
        "INTERVENTION": "treatment A",
        "OUTCOME": "recovery score",
    }
    assert example.label_paths == ["/findings/0/direction"]
    assert example.expected_output["findings"][0]["direction"] == "increase"
    assert all(line["section"] == "Results" for line in example.content_lines.values())
    assert not any(
        "Abstract-only decoy" in line["text"] for line in example.content_lines.values()
    )
    finding = example.expected_output["findings"][0]
    grounded = ground_evidence(
        finding["evidence_quote"],
        finding["evidence_lines"],
        example.content_lines,
        source_accessible=example.source_accessible,
    )
    assert grounded["grounding_status"] == "exact"
    assert grounded["section_flagged"] is False


def test_annotation_metadata_is_not_rendered_to_model(
    evidence_fixture: Path, repo_root: Path, tmp_path: Path
) -> None:
    result = convert_evidence_inference(evidence_fixture, tmp_path / "benchmark")
    example = load_manifest_split(result.manifest_path, "train")[0]
    candidate = {
        "extraction_prompt": (
            repo_root / "prompts" / "evidence_inference_extraction.md"
        ).read_text(encoding="utf-8")
    }
    request_key = provider_request_key(example, candidate)

    class RecordingFixtureProvider(FixtureProvider):
        def __init__(self, responses: dict[tuple[str, str], dict[str, Any]]) -> None:
            super().__init__(responses)
            self.prompts: list[str] = []

        def generate(self, **kwargs: Any):  # type: ignore[no-untyped-def]
            self.prompts.append(kwargs["prompt"])
            return super().generate(**kwargs)

    provider = RecordingFixtureProvider(
        {("gepa-extraction", request_key): example.expected_output}
    )
    evaluation = GEPAPromptAdapter(provider).evaluate([example], candidate)

    assert evaluation.scores == [1.0]
    assert len(provider.prompts) == 1
    rendered = provider.prompts[0]
    assert "secret-reviewer-token" not in rendered
    assert "DO-NOT-RENDER-ANNOTATION-METADATA" not in rendered
    assert "Valid Label" not in rendered
    assert "Evidence Start" not in rendered


def test_converter_excludes_readme_flags_and_label_disagreement(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    default = convert_evidence_inference(evidence_fixture, tmp_path / "default")
    report = json.loads(default.report_path.read_text(encoding="utf-8"))

    assert report["excluded_prompt_counts"]["officially_flagged_prompt"] == 1
    assert report["excluded_prompt_counts"]["verified_label_disagreement"] == 1
    assert "license" in " ".join(report).casefold() or "license_caveat" in report

    included = convert_evidence_inference(
        evidence_fixture,
        tmp_path / "included",
        include_flagged=True,
    )
    assert included.rows["train"] == 2
    train = load_manifest_split(included.manifest_path, "train")
    assert {example.example_id for example in train} == {
        "ei2-prompt-1",
        "ei2-prompt-4",
    }


def test_official_article_split_overlap_is_rejected(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "overlap-source"
    shutil.copytree(evidence_fixture, copied)
    test_ids = copied / "splits" / "test_article_ids.txt"
    test_ids.write_text("1003\n1001\n", encoding="utf-8")

    with pytest.raises(EvidenceInferenceContractError, match="split leakage"):
        convert_evidence_inference(copied, tmp_path / "benchmark")


def test_manifest_train_dev_loader_and_cli_smoke_cap(
    evidence_fixture: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "cli-benchmark"
    assert (
        conversion_cli.main(
            [
                "--dataset-root",
                str(evidence_fixture),
                "--output-dir",
                str(output),
                "--max-examples-per-split",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == {"dev": 1, "test": 1, "train": 1}
    train, dev = load_optimization_examples(output / "manifest.json")
    assert len(train) == len(dev) == 1


def test_existing_output_directory_is_never_overwritten(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-user.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(EvidenceInferenceContractError, match="already exists"):
        convert_evidence_inference(evidence_fixture, output)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_unicode_line_separator_cannot_corrupt_jsonl(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "unicode-source"
    shutil.copytree(evidence_fixture, copied)
    source = copied / "txt_files" / "PMC1001.txt"
    original = source.read_text(encoding="utf-8")
    # This separator occurs in nine real EI2 articles. It is a source line break but must
    # not survive literally inside one JSONL record.
    source.write_text(
        original.replace("Participants were", "Participants\u2028were"),
        encoding="utf-8",
    )

    result = convert_evidence_inference(copied, tmp_path / "benchmark")
    assert len(load_manifest_split(result.manifest_path, "train")) == 1


def test_trackable_summary_contains_ids_but_no_article_text(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    result = convert_evidence_inference(evidence_fixture, tmp_path / "benchmark")
    summary_path = write_evidence_inference_metadata_summary(
        result.manifest_path,
        result.report_path,
        tmp_path / "summary.json",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["contains_article_text"] is False
    assert summary["contains_per_example_labels"] is False
    assert summary["splits"]["train"]["example_ids"] == ["ei2-prompt-1"]
    assert summary["splits"]["train"]["paper_ids"] == ["PMC1001"]
    assert "Treatment A significantly" not in summary_path.read_text(encoding="utf-8")
