#!/usr/bin/env python3
"""Project a validated production certificate into one question replay state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.certificate import (
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
)
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.question_evaluation import (
    freeze_question_replay_state_from_certificate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = args.certificate.read_text(encoding="utf-8")
    value = json.loads(raw)
    version = value.get("certificate_version") if isinstance(value, dict) else None
    if version == "literature-multiverse-verification-v5":
        certificate = VerificationCertificate.model_validate(value)
    elif version == "literature-multiverse-condition-verification-v7":
        certificate = FinalConditionVerificationCertificateV7.model_validate(value)
    else:
        raise ValueError(f"question_replay_certificate_version_unsupported:{version}")
    state = freeze_question_replay_state_from_certificate(certificate)
    atomic_write_json(args.output, state, force=args.force)
    binding = state.production_binding
    assert binding is not None
    print(
        json.dumps(
            {
                "audit_prefix": state.audit_sequence,
                "certificate_sha256": binding.certificate_sha256,
                "full_release_eligible": binding.full_release_eligible,
                "output": args.output.as_posix(),
                "production_stop_decision_sha256": (binding.production_stop_decision_sha256),
                "question_id": state.question_id,
                "replay_sha256": state.replay_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
