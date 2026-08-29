#!/usr/bin/env python3
"""Prepare or externally validate the recovery-v4 public-verifier diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.postlive_recovery_v4_public_verify_v1 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKSPACE,
    validate_postlive_recovery_v4_public_verify_output_v1,
    write_postlive_recovery_v4_public_verify_inputs_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate-output"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace or args.repository_root / DEFAULT_WORKSPACE
    output = args.output_dir or args.repository_root / DEFAULT_OUTPUT_DIR
    if args.command == "prepare":
        result = write_postlive_recovery_v4_public_verify_inputs_v1(
            repository_root=args.repository_root,
            workspace=workspace,
        )
        summary = {
            "claim_manifest_sha256": result.claim_manifest_sha256,
            "corpus_bundle_sha256": result.corpus_bundle_sha256,
            "preparation_sha256": result.preparation_sha256,
            "status": result.status,
        }
    else:
        result = validate_postlive_recovery_v4_public_verify_output_v1(
            repository_root=args.repository_root,
            workspace=workspace,
            output_dir=output,
        )
        summary = {
            "certificate_sha256": result.certificate_sha256,
            "release_status": result.release_status,
            "selected_audit_item_id": result.selected_audit_item_id,
            "status": result.status,
            "validation_sha256": result.validation_sha256,
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
