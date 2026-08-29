from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from scripts.evaluate_synthesis_authorization_review import main as evaluate_review_cli
from scripts.prepare_synthesis_authorization_review import main as prepare_review_cli
from tests.test_cohort_reconciliation import _two_publication_corpus
from tests.test_synthesis_unit_authorization import _materialize_sources

from literature_multiverse.cohort_reconciliation import reconcile_native_cohorts
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.synthesis_authorization_review import (
    RequestedSynthesisUnit,
    SynthesisAuthorizationReviewError,
    SynthesisAuthorizationReviewPacket,
    compute_synthesis_authorization_review_pipeline_fingerprint,
    evaluate_synthesis_authorization_review,
    freeze_synthesis_authorization_review_request,
    prepare_synthesis_authorization_review,
    reverify_synthesis_authorization_review_evaluation,
    verify_synthesis_authorization_review_manifest,
)

CREATED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _materialize_review_pipeline(repository_root: Path) -> None:
    fingerprint = compute_synthesis_authorization_review_pipeline_fingerprint(REPOSITORY_ROOT)
    for item in fingerprint.components[0].files:
        source = REPOSITORY_ROOT / item.path
        destination = repository_root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _case(
    tmp_path: Path,
    *,
    first_registry: list[str] | None = None,
    second_registry: list[str] | None = None,
    identifier_text_suffix: str = "",
    name: str = "case",
):
    _materialize_review_pipeline(tmp_path)
    corpus = _materialize_sources(
        _two_publication_corpus(
            first_registry=first_registry or ["NCT-ONE"],
            second_registry=second_registry or ["NCT-TWO"],
        ),
        tmp_path,
        identifier_text_suffix=identifier_text_suffix,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    assert reconciliation.reconciled_graph is not None
    request = freeze_synthesis_authorization_review_request(
        [
            RequestedSynthesisUnit(
                synthesis_unit_id="unit-primary",
                estimate_ids=sorted(
                    item.estimate_id for item in reconciliation.reconciled_graph.outcome_estimates
                ),
            )
        ]
    )
    output_dir = tmp_path / "data/cache/synthesis-authorization-review" / name
    manifest = prepare_synthesis_authorization_review(
        corpus=corpus,
        reconciliation=reconciliation,
        request=request,
        repository_root=tmp_path,
        output_dir=output_dir,
        created_at=CREATED_AT,
    )
    _, packet, reviewer_a, reviewer_b = verify_synthesis_authorization_review_manifest(
        manifest_path=output_dir / "manifest.private.json",
        repository_root=tmp_path,
    )
    return {
        "repository_root": tmp_path,
        "corpus": corpus,
        "reconciliation": reconciliation,
        "request": request,
        "output_dir": output_dir,
        "manifest": manifest,
        "manifest_path": output_dir / "manifest.private.json",
        "packet": packet,
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
    }


def _fill_template(
    template: Any,
    packet: SynthesisAuthorizationReviewPacket,
    *,
    reviewer_identity: str,
    relationship: str | None = None,
    rationale: str = "The exact source review establishes the requested relationship.",
    start_at: datetime | None = None,
    blank_citations: bool = False,
    quote_identifier_only: bool = False,
) -> dict[str, Any]:
    payload = template.model_dump(mode="json")
    source_by_publication = {item.publication_id: item for item in packet.source_materials}
    target_by_id = {item.target_id: item for item in packet.targets}
    cursor = start_at or (packet.created_at + timedelta(minutes=5))
    for row in payload["decisions"]:
        target = target_by_id[row["target_id"]]
        row["relationship"] = relationship or target.required_relationship
        row["rationale"] = rationale
        if not blank_citations:
            for citation in row["citations"]:
                identifier = citation["eligible_source_identifiers"][0]
                source = source_by_publication[citation["publication_id"]]
                line = next(
                    item for item in source.lines if identifier.casefold() in item.text.casefold()
                )
                citation["quote"] = identifier if quote_identifier_only else line.text
                citation["line_ids"] = [line.line_id]
                citation["cited_identifier"] = identifier
        row["review_started_at"] = cursor.isoformat().replace("+00:00", "Z")
        cursor += timedelta(minutes=2)
        row["review_completed_at"] = cursor.isoformat().replace("+00:00", "Z")
        row["review_minutes"] = 2.0
        cursor += timedelta(minutes=1)
    payload["reviewer_identity_sha256"] = reviewer_identity
    payload["submitted_at"] = cursor.isoformat().replace("+00:00", "Z")
    return payload


def _write_form(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    path = tmp_path / name
    atomic_write_json(path, value)
    return path


def _evaluate(
    case: dict[str, Any],
    tmp_path: Path,
    a: dict[str, Any],
    b: dict[str, Any],
    adjudicator: dict[str, Any] | None = None,
):
    form_dir = case["output_dir"] / "completed" / tmp_path.name
    a_path = _write_form(form_dir, "completed-a.json", a)
    b_path = _write_form(form_dir, "completed-b.json", b)
    adjudicator_path = (
        _write_form(form_dir, "completed-adjudicator.json", adjudicator)
        if adjudicator is not None
        else None
    )
    return evaluate_synthesis_authorization_review(
        manifest_path=case["manifest_path"],
        corpus=case["corpus"],
        reconciliation=case["reconciliation"],
        repository_root=case["repository_root"],
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
        adjudicator_path=adjudicator_path,
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_prepare_freezes_exact_private_sources_and_hides_system_scores(tmp_path: Path) -> None:
    case = _case(tmp_path)
    packet = case["packet"]
    assert case["manifest"].review_target_count == 1
    assert packet.system_scores_included is False
    assert packet.source_identity_visible is True
    assert packet.publication_source_content_visible is True
    assert packet.benchmark_reference_labels_accessed is False
    assert packet.benchmark_review_verdicts_accessed is False
    assert packet.pipeline_fingerprint_sha256 == packet.pipeline_fingerprint.pipeline_sha256
    assert case["manifest"].pipeline_fingerprint == packet.pipeline_fingerprint
    assert len(packet.source_materials) == 2
    assert len(packet.cohort_sources) == 2
    assert packet.targets[0].required_relationship == "independent_cohorts"
    keys = _all_keys(packet.model_dump(mode="json"))
    assert not {
        "model_confidence",
        "error_probability",
        "risk_score",
        "audit_priority",
        "influence",
        "value_of_information",
        "synthesis_result",
    }.intersection(keys)
    assert case["reviewer_a"].reviewer_identity_sha256 is None
    assert case["reviewer_b"].reviewer_identity_sha256 is None
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="output_directory_exists",
    ):
        prepare_synthesis_authorization_review(
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            request=case["request"],
            repository_root=tmp_path,
            output_dir=case["output_dir"],
            created_at=CREATED_AT,
        )


def test_pipeline_fingerprint_uses_ast_closed_runtime_and_byte_exact_transitive_files(
    tmp_path: Path,
) -> None:
    _materialize_review_pipeline(tmp_path)
    before = compute_synthesis_authorization_review_pipeline_fingerprint(tmp_path)
    paths = {item.path for item in before.components[0].files}
    assert {
        "scripts/evaluate_synthesis_authorization_review.py",
        "scripts/prepare_synthesis_authorization_review.py",
        "src/literature_multiverse/synthesis_authorization_review.py",
        "src/literature_multiverse/synthesis_unit_authorization.py",
        "src/literature_multiverse/cohort_reconciliation.py",
        "src/literature_multiverse/native_grounding.py",
        "src/literature_multiverse/typed_extraction.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/pipeline_fingerprint.py",
        "src/literature_multiverse/models.py",
        "pyproject.toml",
        "uv.lock",
    } <= paths
    models_path = tmp_path / "src/literature_multiverse/models.py"
    models_path.write_text(
        models_path.read_text(encoding="utf-8") + "\n# transitive drift\n",
        encoding="utf-8",
    )
    after = compute_synthesis_authorization_review_pipeline_fingerprint(tmp_path)
    assert after.pipeline_sha256 != before.pipeline_sha256
    before_models = next(
        item.sha256 for item in before.components[0].files if item.path.endswith("models.py")
    )
    after_models = next(
        item.sha256 for item in after.components[0].files if item.path.endswith("models.py")
    )
    assert after_models != before_models


def test_manifest_and_full_replay_fail_closed_after_transitive_code_drift(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64)
    form_dir = case["output_dir"] / "completed/code-drift"
    a_path = _write_form(form_dir, "completed-a.json", a)
    b_path = _write_form(form_dir, "completed-b.json", b)
    private, public, template = evaluate_synthesis_authorization_review(
        manifest_path=case["manifest_path"],
        corpus=case["corpus"],
        reconciliation=case["reconciliation"],
        repository_root=tmp_path,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
    )
    assert template is None
    models_path = tmp_path / "src/literature_multiverse/models.py"
    models_path.write_text(
        models_path.read_text(encoding="utf-8") + "\n# post-freeze drift\n",
        encoding="utf-8",
    )
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="pipeline_fingerprint_mismatch",
    ):
        verify_synthesis_authorization_review_manifest(
            manifest_path=case["manifest_path"],
            repository_root=tmp_path,
        )
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="pipeline_fingerprint_mismatch",
    ):
        reverify_synthesis_authorization_review_evaluation(
            manifest_path=case["manifest_path"],
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            repository_root=tmp_path,
            reviewer_a_path=a_path,
            reviewer_b_path=b_path,
            private_evaluation=private,
            public_summary=public,
        )


def test_coherently_rehashed_manifest_pipeline_cannot_replace_computed_identity(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    forged = case["manifest"].model_dump(mode="json")
    pipeline = forged["pipeline_fingerprint"]
    component = pipeline["components"][0]
    component["settings"]["unknown_or_missing_policy"] = "forged_release"
    component["component_sha256"] = hash_canonical(
        {key: value for key, value in component.items() if key != "component_sha256"}
    )
    pipeline["pipeline_sha256"] = hash_canonical(
        {key: value for key, value in pipeline.items() if key != "pipeline_sha256"}
    )
    forged["pipeline_fingerprint_sha256"] = pipeline["pipeline_sha256"]
    forged["manifest_sha256"] = hash_canonical(
        {key: value for key, value in forged.items() if key != "manifest_sha256"}
    )
    forged_manifest = type(case["manifest"]).model_validate(forged)
    atomic_write_json(case["manifest_path"], forged_manifest, force=True)
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="pipeline_fingerprint_mismatch",
    ):
        verify_synthesis_authorization_review_manifest(
            manifest_path=case["manifest_path"],
            repository_root=tmp_path,
        )


def test_unknown_agreement_is_complete_but_abstains_and_public_is_identifier_free(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    a = _fill_template(
        case["reviewer_a"],
        case["packet"],
        reviewer_identity="a" * 64,
        relationship="unknown",
        rationale="The available source does not establish cohort independence.",
        blank_citations=True,
    )
    b = _fill_template(
        case["reviewer_b"],
        case["packet"],
        reviewer_identity="b" * 64,
        relationship="unknown",
        rationale="The available source does not establish cohort independence.",
        blank_citations=True,
    )
    private, public, conflicts = _evaluate(case, tmp_path, a, b)
    assert conflicts is None
    assert private.status == "complete"
    assert private.unit_outcomes[0].authorizes_synthesis_input is False
    assert private.unit_outcomes[0].assertions == []
    assert "source_relationship_unknown" in private.unit_outcomes[0].blocker_codes
    assert public.resolved_unknown_count == 1
    assert public.authorized_unit_count == 0
    serialized = public.model_dump_json()
    assert "private_evaluation_sha256" not in serialized
    private_literals = {
        "publication-1",
        "publication-2",
        "NCT-ONE",
        "NCT-TWO",
        "unit-primary",
        "a" * 64,
        "b" * 64,
        *[item.original_cohort_id for item in case["packet"].cohort_sources],
        *[item.source_locator for item in case["packet"].source_materials],
    }
    assert not any(value in serialized for value in private_literals)


def test_exact_independent_agreement_builds_v1_assertion_and_authorizes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64)
    private, public, conflicts = _evaluate(case, tmp_path, a, b)
    assert conflicts is None
    assert private.status == "complete"
    assert private.pipeline_fingerprint == case["packet"].pipeline_fingerprint
    assert public.pipeline_fingerprint == private.pipeline_fingerprint
    assert public.pipeline_fingerprint_sha256 == private.pipeline_fingerprint_sha256
    outcome = private.unit_outcomes[0]
    assert outcome.authorizes_synthesis_input is True
    assert outcome.blocker_codes == []
    assert len(outcome.assertions) == 1
    assert outcome.assertions[0].assertion_version == "source-identity-assertion-v1"
    assert outcome.assertions[0].relationship == "independent_cohorts"
    assert outcome.authorization_receipt is not None
    assert outcome.authorization_receipt.authorizes_synthesis_input is True
    assert public.source_assertion_count == 1
    assert public.authorized_unit_count == 1


def test_merged_cross_publication_cohort_requires_and_builds_same_cohort_assertion(
    tmp_path: Path,
) -> None:
    case = _case(
        tmp_path,
        first_registry=["NCT 00000001"],
        second_registry=["nct 00000001"],
    )
    assert case["packet"].targets[0].required_relationship == "same_cohort"
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64)
    private, public, conflicts = _evaluate(case, tmp_path, a, b)
    assert conflicts is None
    outcome = private.unit_outcomes[0]
    assert outcome.authorizes_synthesis_input is True
    assert len(outcome.assertions) == 1
    assert outcome.assertions[0].relationship == "same_cohort"
    assert public.same_cohort_target_count == 1
    assert public.independence_target_count == 0


def test_full_replay_rejects_coherently_rehashed_private_or_public_projection(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64)
    form_dir = case["output_dir"] / "completed/replay"
    a_path = _write_form(form_dir, "completed-a.json", a)
    b_path = _write_form(form_dir, "completed-b.json", b)
    private, public, template = evaluate_synthesis_authorization_review(
        manifest_path=case["manifest_path"],
        corpus=case["corpus"],
        reconciliation=case["reconciliation"],
        repository_root=tmp_path,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
    )
    assert template is None
    replayed_private, replayed_public, _ = reverify_synthesis_authorization_review_evaluation(
        manifest_path=case["manifest_path"],
        corpus=case["corpus"],
        reconciliation=case["reconciliation"],
        repository_root=tmp_path,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
        private_evaluation=private,
        public_summary=public,
    )
    assert replayed_private == private
    assert replayed_public == public

    forged_private_payload = private.model_dump(mode="json")
    forged_unit = forged_private_payload["unit_outcomes"][0]
    forged_unit["synthesis_unit_id"] = "coherently-forged-unit"
    forged_unit_payload = {
        key: value for key, value in forged_unit.items() if key != "unit_review_sha256"
    }
    forged_unit["unit_review_sha256"] = hash_canonical(forged_unit_payload)
    forged_private_payload["final_transition_sha256"] = hash_canonical(
        {
            "comparison_transition_sha256": private.comparison.transition_sha256,
            "adjudicator_submission_sha256": None,
            "resolution_sha256s": [item.resolution_sha256 for item in private.resolutions],
            "unit_review_sha256s": [forged_unit["unit_review_sha256"]],
        }
    )
    private_without_hash = {
        key: value for key, value in forged_private_payload.items() if key != "evaluation_sha256"
    }
    forged_private_payload["evaluation_sha256"] = hash_canonical(private_without_hash)
    forged_private = type(private).model_validate(forged_private_payload)
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="private_evaluation_replay_mismatch",
    ):
        reverify_synthesis_authorization_review_evaluation(
            manifest_path=case["manifest_path"],
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            repository_root=tmp_path,
            reviewer_a_path=a_path,
            reviewer_b_path=b_path,
            private_evaluation=forged_private,
            public_summary=public,
        )

    forged_public_payload = public.model_dump(mode="json")
    forged_public_payload["authorized_unit_count"] = 0
    forged_public_payload["abstained_unit_count"] = 1
    public_without_hash = {
        key: value for key, value in forged_public_payload.items() if key != "summary_sha256"
    }
    forged_public_payload["summary_sha256"] = hash_canonical(public_without_hash)
    forged_public = type(public).model_validate(forged_public_payload)
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="public_summary_replay_mismatch",
    ):
        reverify_synthesis_authorization_review_evaluation(
            manifest_path=case["manifest_path"],
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            repository_root=tmp_path,
            reviewer_a_path=a_path,
            reviewer_b_path=b_path,
            private_evaluation=private,
            public_summary=forged_public,
        )


def test_any_scientific_disagreement_requires_distinct_third_adjudicator(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(
        case["reviewer_b"],
        case["packet"],
        reviewer_identity="b" * 64,
        relationship="unknown",
        rationale="The available source does not establish cohort independence.",
        blank_citations=True,
    )
    pending, public, template = _evaluate(case, tmp_path, a, b)
    assert pending.status == "awaiting_adjudication"
    assert template is not None
    assert template.input_transition_sha256 == pending.comparison.transition_sha256
    assert template.reviewer_slot == "adjudicator"
    assert pending.unit_outcomes[0].authorizes_synthesis_input is False
    assert public.disagreement_count == 1
    assert public.adjudicated_count == 0

    adjudicator = _fill_template(
        template,
        case["packet"],
        reviewer_identity="c" * 64,
        start_at=datetime(2026, 8, 28, 13, 0, tzinfo=UTC),
    )
    same_person = deepcopy(adjudicator)
    same_person["reviewer_identity_sha256"] = "a" * 64
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="adjudicator_not_identity_independent",
    ):
        _evaluate(case, tmp_path / "invalid-adjudicator", a, b, same_person)
    completed, completed_public, no_template = _evaluate(
        case, tmp_path / "adjudicated", a, b, adjudicator
    )
    assert no_template is None
    assert completed.status == "complete"
    assert completed.adjudicator_submission is not None
    assert completed.unit_outcomes[0].authorizes_synthesis_input is True
    assert completed_public.adjudicated_count == 1


def test_same_label_different_valid_rationales_and_citations_preserve_both_without_third_review(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(
        case["reviewer_b"],
        case["packet"],
        reviewer_identity="b" * 64,
        rationale="A different, independently written source rationale reaches the same label.",
        quote_identifier_only=True,
    )
    private, public, template = _evaluate(case, tmp_path, a, b)
    assert private.status == "complete"
    assert public.relationship_agreement_count == 1
    assert public.support_divergence_count == 1
    assert public.disagreement_count == 0
    assert template is None
    resolution = private.resolutions[0]
    assert resolution.support_evidence_diverged is True
    assert [item.reviewer_slot for item in resolution.source_decisions] == [
        "reviewer_a",
        "reviewer_b",
    ]
    assert len(resolution.citations) == 4
    assert "Reviewer A:" in resolution.rationale
    assert "Reviewer B:" in resolution.rationale
    assert private.unit_outcomes[0].authorizes_synthesis_input is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("same_identity", "independent_reviewers_not_distinct"),
        ("wrong_minutes", "minutes_not_measured_elapsed_time"),
        ("tamper_source", "citation_prefill_tampered"),
        ("wrong_predecessor", "submission_lineage_mismatch"),
        ("missing_citation", "partial_citation_roster"),
    ],
)
def test_adversarial_reviewer_transition_tampering_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    case = _case(tmp_path)
    a = _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64)
    b = _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64)
    if mutation == "same_identity":
        b["reviewer_identity_sha256"] = "a" * 64
    elif mutation == "wrong_minutes":
        a["decisions"][0]["review_minutes"] = 0.5
    elif mutation == "tamper_source":
        a["decisions"][0]["citations"][0]["publication_id"] = "invented-publication"
    elif mutation == "wrong_predecessor":
        b["input_transition_sha256"] = "f" * 64
    else:
        a["decisions"][0]["citations"][0]["quote"] = None
        a["decisions"][0]["citations"][0]["line_ids"] = []
        a["decisions"][0]["citations"][0]["cited_identifier"] = None
    with pytest.raises((SynthesisAuthorizationReviewError, ValueError), match=message):
        _evaluate(case, tmp_path, a, b)


def test_identifier_prefix_collision_and_post_freeze_source_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    collision = _case(
        tmp_path,
        first_registry=["NCT123"],
        second_registry=["NCT999"],
        identifier_text_suffix="4",
        name="prefix-collision",
    )
    a = _fill_template(collision["reviewer_a"], collision["packet"], reviewer_identity="a" * 64)
    b = _fill_template(collision["reviewer_b"], collision["packet"], reviewer_identity="b" * 64)
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="identifier_outside_exact_lines",
    ):
        _evaluate(collision, tmp_path, a, b)

    clean_root = tmp_path / "source-mutation"
    clean_root.mkdir()
    clean = _case(clean_root)
    source_path = clean_root / clean["corpus"].fragments[0].source_document.artifact_path
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    a = _fill_template(clean["reviewer_a"], clean["packet"], reviewer_identity="a" * 64)
    b = _fill_template(clean["reviewer_b"], clean["packet"], reviewer_identity="b" * 64)
    with pytest.raises(ValueError, match="source_artifact_hash_mismatch"):
        _evaluate(clean, clean_root, a, b)


def test_prepare_and_evaluate_clis_write_only_mandated_private_and_public_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _materialize_review_pipeline(tmp_path)
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    assert reconciliation.reconciled_graph is not None
    corpus_path = tmp_path / "corpus.json"
    reconciliation_path = tmp_path / "reconciliation.json"
    request_path = tmp_path / "request.json"
    atomic_write_json(corpus_path, corpus)
    atomic_write_json(reconciliation_path, reconciliation)
    atomic_write_json(
        request_path,
        {
            "request_version": "synthesis-authorization-review-request-v1",
            "synthesis_units": [
                {
                    "synthesis_unit_id": "unit-primary",
                    "estimate_ids": sorted(
                        item.estimate_id
                        for item in reconciliation.reconciled_graph.outcome_estimates
                    ),
                }
            ],
        },
    )
    output_dir = tmp_path / "data/cache/synthesis-authorization-review/cli-packet"
    assert (
        prepare_review_cli(
            [
                "--corpus",
                str(corpus_path),
                "--reconciliation",
                str(reconciliation_path),
                "--request",
                str(request_path),
                "--repository-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--created-at",
                "2026-08-28T12:00:00Z",
            ]
        )
        == 0
    )
    capsys.readouterr()
    _, packet, reviewer_a, reviewer_b = verify_synthesis_authorization_review_manifest(
        manifest_path=output_dir / "manifest.private.json",
        repository_root=tmp_path,
    )
    completed_dir = output_dir / "completed"
    a_path = _write_form(
        completed_dir,
        "cli-a.json",
        _fill_template(reviewer_a, packet, reviewer_identity="a" * 64),
    )
    b_path = _write_form(
        completed_dir,
        "cli-b.json",
        _fill_template(reviewer_b, packet, reviewer_identity="b" * 64),
    )
    private_output = (
        tmp_path
        / "data/cache/synthesis-authorization-review/cli-evaluation/private-evaluation.json"
    )
    public_output = (
        tmp_path / "artifacts/diagnostics/synthesis-authorization-review/cli-summary.json"
    )
    assert (
        evaluate_review_cli(
            [
                "--corpus",
                str(corpus_path),
                "--reconciliation",
                str(reconciliation_path),
                "--repository-root",
                str(tmp_path),
                "--manifest",
                str(output_dir / "manifest.private.json"),
                "--reviewer-a",
                str(a_path),
                "--reviewer-b",
                str(b_path),
                "--private-output",
                str(private_output),
                "--public-output",
                str(public_output),
                "--require-complete",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert printed["authorized_unit_count"] == 1
    assert private_output.is_file()
    public = json.loads(public_output.read_text(encoding="utf-8"))
    assert public["aggregate_only"] is True
    assert public["contains_source_identifiers"] is False


def test_cli_require_complete_disagreement_creates_no_outputs(tmp_path: Path) -> None:
    case = _case(tmp_path)
    completed_dir = case["output_dir"] / "completed/require-complete"
    a_path = _write_form(
        completed_dir,
        "reviewer-a.json",
        _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64),
    )
    b_path = _write_form(
        completed_dir,
        "reviewer-b.json",
        _fill_template(
            case["reviewer_b"],
            case["packet"],
            reviewer_identity="b" * 64,
            relationship="unknown",
            rationale="The source relationship remains unknown.",
            blank_citations=True,
        ),
    )
    corpus_path = case["output_dir"] / "corpus.private.json"
    reconciliation_path = case["output_dir"] / "reconciliation.private.json"
    atomic_write_json(corpus_path, case["corpus"])
    atomic_write_json(reconciliation_path, case["reconciliation"])
    private_output = (
        tmp_path / "data/cache/synthesis-authorization-review/require-complete/private.json"
    )
    public_output = (
        tmp_path / "artifacts/diagnostics/synthesis-authorization-review/require-complete.json"
    )
    with pytest.raises(ValueError, match="review_not_complete"):
        evaluate_review_cli(
            [
                "--corpus",
                str(corpus_path),
                "--reconciliation",
                str(reconciliation_path),
                "--repository-root",
                str(tmp_path),
                "--manifest",
                str(case["manifest_path"]),
                "--reviewer-a",
                str(a_path),
                "--reviewer-b",
                str(b_path),
                "--private-output",
                str(private_output),
                "--public-output",
                str(public_output),
                "--require-complete",
            ]
        )
    assert not private_output.exists()
    assert not public_output.exists()


def test_manifest_or_output_path_escape_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    tampered = deepcopy(case["manifest"].model_dump(mode="json"))
    tampered["private_files"][0]["path"] = "../packet.private.json"
    with pytest.raises(ValueError, match="private_path_unsafe"):
        type(case["manifest"]).model_validate(tampered)

    outside = tmp_path / "not-ignored/review"
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="outside_ignored_root",
    ):
        prepare_synthesis_authorization_review(
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            request=case["request"],
            repository_root=tmp_path,
            output_dir=outside,
            created_at=CREATED_AT,
        )

    outside_reviewer = _write_form(
        tmp_path,
        "outside-reviewer.json",
        _fill_template(case["reviewer_a"], case["packet"], reviewer_identity="a" * 64),
    )
    inside_reviewer = _write_form(
        case["output_dir"] / "completed/path-check",
        "reviewer-b.json",
        _fill_template(case["reviewer_b"], case["packet"], reviewer_identity="b" * 64),
    )
    with pytest.raises(
        SynthesisAuthorizationReviewError,
        match="reviewer_a_outside_ignored_root",
    ):
        evaluate_synthesis_authorization_review(
            manifest_path=case["manifest_path"],
            corpus=case["corpus"],
            reconciliation=case["reconciliation"],
            repository_root=tmp_path,
            reviewer_a_path=outside_reviewer,
            reviewer_b_path=inside_reviewer,
        )
