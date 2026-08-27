"""Offline contracts for the pinned MetaSyn corpus and retrieval runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.run_local_benchmarks import BenchmarkSuiteError, load_suite

from literature_multiverse.lineage import sha256_file
from literature_multiverse.metasyn_retrieval import (
    MetaSynCorpusError,
    _stable_top_k,
    inspect_corpus_coverage,
    verify_corpus_manifest,
)


def _corpus_manifest(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shard = corpus / "part.parquet"
    pd.DataFrame(
        [
            {"ID": 1, "title": "alpha trial", "abstract": "outcome one"},
            {"ID": 2, "title": "beta trial", "abstract": "outcome two"},
            {"ID": 3, "title": None, "abstract": "outcome three"},
        ]
    ).to_parquet(shard, index=False)
    license_path = tmp_path / "NOTICE.txt"
    license_path.write_text("fixture terms\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_manifest_version": "1",
                "dataset": "MetaSyn corpus",
                "source_repository": "fixture/metasyn",
                "source_revision": "a" * 40,
                "local_root": "corpus",
                "license_notice": {
                    "path": "NOTICE.txt",
                    "sha256": sha256_file(license_path),
                    "status": "local_evaluation_only_third_party_terms_apply",
                },
                "shards": [{"path": shard.name, "rows": 3, "sha256": sha256_file(shard)}],
                "total_rows": 3,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_corpus_manifest_verifies_hashes_rows_schema_and_gold_coverage(
    tmp_path: Path,
) -> None:
    manifest_path = _corpus_manifest(tmp_path)
    _, shards = verify_corpus_manifest(manifest_path, repository_root=tmp_path)
    # Benchmark manifests normalize paper identifiers to strings for overlap checks;
    # corpus Parquet stores the same identifiers as integers.
    coverage = inspect_corpus_coverage(shards, required_corpus_ids={"1", "3"})

    assert coverage == {
        "rows": 3,
        "rows_with_title": 2,
        "rows_with_abstract": 3,
        "required_gold_ids": 2,
        "required_gold_ids_present": 2,
        "required_gold_ids_missing": 0,
        "required_gold_ids_complete": True,
    }

    shard = shards[0]
    shard.write_bytes(shard.read_bytes() + b"mutation")
    with pytest.raises(MetaSynCorpusError, match="corpus_shard_hash_mismatch"):
        verify_corpus_manifest(manifest_path, repository_root=tmp_path)


def test_stable_top_k_excludes_source_review_and_breaks_ties_by_corpus_id() -> None:
    ranked = _stable_top_k(
        np.asarray([0.7, 0.7, 0.9, 0.7], dtype=np.float32),
        corpus_ids=np.asarray([20, 10, 30, 5], dtype=np.int64),
        excluded_ids={30},
        top_k=2,
    )

    assert ranked == [5, 10]


def test_suite_contract_forbids_opened_labels_from_pristine_holdout(tmp_path: Path) -> None:
    suite = {
        "benchmark_suite_version": "1",
        "network_calls": 0,
        "corpora": {
            "metasyn": {
                "access_state": {
                    "test": {
                        "labels_previously_opened": True,
                        "pristine_final_holdout_eligible": True,
                    }
                }
            }
        },
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(BenchmarkSuiteError, match="opened_labels_cannot_be_pristine"):
        load_suite(path)
