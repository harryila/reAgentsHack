"""Additive, source-only projection of the immutable MetaSyn hosted v5 run.

The v5 execution bundle is externally replayed before this module exposes anything.
Only the frozen question specification, independence-component lineage, and source row
are projected.  Hosted call intents, receipts, row results, ledgers, private reports,
and provider outputs are neither parameters nor inputs to the projection.

This is deliberately not a hosted-result migration.  It is a narrow upstream source
surface from which a separately versioned successor may construct new prompts and make
new calls without silently consuming or reinterpreting v5 model outputs.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    MetaSynHostedExecutionBundleV1,
    load_current_metasyn_hosted_execution_bundle,
)
from literature_multiverse.metasyn_typed_pilot import (
    EXPECTED_SELECTED_COMPONENTS,
    EXPECTED_SELECTED_PAPERS,
    EXPECTED_SELECTED_QUESTIONS,
    MetaSynPilotQuestionSpecV1,
    MetaSynPilotSourceProjectionRowV1,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

SOURCE_SURFACE_VERSION = "metasyn-v5-source-surface-v1"
SOURCE_SURFACE_ROW_VERSION = "metasyn-v5-source-surface-row-v1"
SOURCE_ARTIFACT_BINDING_VERSION = "metasyn-v5-source-artifact-binding-v1"
COMPONENT_BINDING_VERSION = "metasyn-v5-independence-component-binding-v1"
SOURCE_SURFACE_COMPONENT_VERSION = "1"

V5_EXECUTION_WORKSPACE = Path("data/cache/metasyn/bounded-anthropic-yield-v5")
V5_EXECUTION_BUNDLE_RELATIVE = "execution-bundle.json"

# These are the externally replayed, immutable v5 identities.  A merely self-consistent
# replacement bundle is not accepted as v5.
EXPECTED_V5_EXECUTION_BUNDLE_SHA256 = (
    "d53fedfb58ab4937fe314d10d1612d300d573f60eb0a968de0e861b67b5c3aa7"
)
EXPECTED_V5_ADAPTER_BUNDLE_SHA256 = (
    "15226fe7d6838b27886447ae152d7ef69e733c8b031baae1007ba945041220e8"
)
EXPECTED_V5_ADAPTER_PIPELINE_SHA256 = (
    "55783c45dbd1df33b5a8ec2a8f41ee61b43ad0fd59f278aca17a6d55b2007281"
)
EXPECTED_V5_RUNTIME_PIPELINE_SHA256 = (
    "d813267fc90717a3842fc70baf3b6f62ace5b12c582595b03365b33141973dea"
)
EXPECTED_V5_UPSTREAM_PILOT_PIPELINE_SHA256 = (
    "1696e7626ecc7236af5437913d4645bbbe306b90904bd2836551a007ec8f2e1f"
)
EXPECTED_V5_DOWNSTREAM_VERIFIER_PIPELINE_SHA256 = (
    "1f332b803be247af6ee7c8e25e9eb0320dbe32203e3d36a0dc9e366a68dac475"
)
EXPECTED_V5_CONFIG_SHA256 = "a8d0d0da289b9ac74c4ba43d2f6b2303ffc0f0d0e0d1405b987ace834df0c22f"
EXPECTED_V5_NATIVE_SCHEMA_V2_CONTRACT_SHA256 = (
    "1197e39cbff831501bb9f635df49a06280c66dd742d1cc2abc44454a4789369f"
)
EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256 = (
    "104c7bdaaf79bc15af00077d34d7dc4aa879a58f3e5e708f0c5a0a4b7173bfbe"
)
EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256 = (
    "524f03800d3a19ddc4153d992ad46c329fb22d8b42bc57f01234b2d59415d0a7"
)
EXPECTED_V5_ROW_MEMBERSHIP_SHA256 = (
    "ce15e4a9e98b8b2f5d7ca6920227215acad3561144a1b97c740e3d8a78b56cac"
)

PROJECTED_ROW_CONTEXT_FIELDS: tuple[str, ...] = (
    "independence_component_id",
    "independence_component_membership_sha256",
    "independence_component_review_ids",
    "projection_sha256",
    "question_bundle_sha256",
    "question_spec",
    "question_spec_sha256",
    "row_context_sha256",
    "source_row",
    "source_row_sha256",
)

CONSUMED_V5_RUNTIME_ARTIFACT_KINDS: tuple[str, ...] = ("execution_bundle",)
FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS: tuple[str, ...] = (
    "attempt_intents",
    "call_incidents",
    "call_receipts",
    "cost_authorization_receipt",
    "hosted_ledger",
    "preflight_receipt",
    "private_yield_reports",
    "provider_outputs",
    "row_results",
    "smoke_receipt",
)

_SOURCE_SURFACE_ENTRYPOINTS = ("src/literature_multiverse/metasyn_v5_source_surface.py",)

_EXPECTED_V5_ANCHORS = {
    "adapter_bundle_sha256": EXPECTED_V5_ADAPTER_BUNDLE_SHA256,
    "adapter_pipeline_sha256": EXPECTED_V5_ADAPTER_PIPELINE_SHA256,
    "component_membership_sha256": EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256,
    "config_sha256": EXPECTED_V5_CONFIG_SHA256,
    "downstream_verifier_pipeline_sha256": (EXPECTED_V5_DOWNSTREAM_VERIFIER_PIPELINE_SHA256),
    "execution_bundle_sha256": EXPECTED_V5_EXECUTION_BUNDLE_SHA256,
    "native_schema_v2_contract_sha256": (EXPECTED_V5_NATIVE_SCHEMA_V2_CONTRACT_SHA256),
    "question_membership_sha256": EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256,
    "row_membership_sha256": EXPECTED_V5_ROW_MEMBERSHIP_SHA256,
    "runtime_pipeline_sha256": EXPECTED_V5_RUNTIME_PIPELINE_SHA256,
    "upstream_pilot_pipeline_sha256": EXPECTED_V5_UPSTREAM_PILOT_PIPELINE_SHA256,
}


class MetaSynV5SourceSurfaceError(ValueError):
    """The pinned v5 bundle or its source-only projection failed closed."""


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_v5_source_surface_sha256_invalid:{field_name}")
    return value


def _canonical_repository_root(repository_root: Path) -> Path:
    root = Path(os.path.abspath(repository_root))
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode):
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_repository_root_symlink_forbidden"
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_repository_root_unreadable"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_repository_root_not_directory")
    return resolved


def _checked_repository_path(
    *, repository_root: Path, relative_path: str, require_directory: bool
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative_path.startswith("./")
        or relative.as_posix() != relative_path
    ):
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_path_not_canonical_relative")

    candidate = repository_root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise MetaSynV5SourceSurfaceError(
                "metasyn_v5_source_surface_path_unreadable_or_missing"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_path_symlink_forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_path_unreadable_or_missing"
        ) from exc
    if not resolved.is_relative_to(repository_root):
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_path_outside_repository")
    if require_directory and not resolved.is_dir():
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_path_not_directory")
    if not require_directory and not resolved.is_file():
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_path_not_regular_file")
    return resolved


def _pinned_v5_workspace(*, repository_root: Path, execution_workspace: Path | None) -> Path:
    expected_relative = V5_EXECUTION_WORKSPACE.as_posix()
    expected = _checked_repository_path(
        repository_root=repository_root,
        relative_path=expected_relative,
        require_directory=True,
    )
    _checked_repository_path(
        repository_root=repository_root,
        relative_path=(V5_EXECUTION_WORKSPACE / V5_EXECUTION_BUNDLE_RELATIVE).as_posix(),
        require_directory=False,
    )
    if execution_workspace is None:
        return expected

    requested = Path(execution_workspace)
    if not requested.is_absolute():
        requested = repository_root / requested
    requested_absolute = Path(os.path.abspath(requested))
    try:
        requested_mode = requested_absolute.lstat().st_mode
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_execution_workspace_unreadable"
        ) from exc
    if stat.S_ISLNK(requested_mode):
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_execution_workspace_symlink_forbidden"
        )
    try:
        requested_resolved = requested_absolute.resolve(strict=True)
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_execution_workspace_unreadable"
        ) from exc
    if requested_resolved != expected:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_execution_workspace_not_pinned_v5"
        )
    return expected


def _sha256_open_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_artifact_open_failed") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_artifact_not_regular_file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise MetaSynV5SourceSurfaceError(
                "metasyn_v5_source_surface_artifact_changed_during_hash"
            )
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def _rehash_source_artifact(
    *, repository_root: Path, artifact_path: str, expected_sha256: str
) -> tuple[str, int]:
    """Hash actual source bytes after canonical-path and no-symlink checks."""

    _validate_sha256(expected_sha256, "expected_source_artifact")
    root = _canonical_repository_root(repository_root)
    path = _checked_repository_path(
        repository_root=root,
        relative_path=artifact_path,
        require_directory=False,
    )
    observed_sha256, observed_bytes = _sha256_open_regular_file(path)
    if observed_sha256 != expected_sha256:
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_artifact_sha256_mismatch")
    return observed_sha256, observed_bytes


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _source_surface_python_dependency_closure(repository_root: Path) -> list[str]:
    """Walk every direct/transitive in-repository import of the projection."""

    root = _canonical_repository_root(repository_root)
    pending = list(_SOURCE_SURFACE_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        try:
            source_path = _checked_repository_path(
                repository_root=root,
                relative_path=relative,
                require_directory=False,
            )
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except MetaSynV5SourceSurfaceError:
            raise
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynV5SourceSurfaceError(
                f"metasyn_v5_source_surface_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def _pipeline_settings() -> dict[str, Any]:
    return {
        "consumed_v5_runtime_artifact_kinds": list(CONSUMED_V5_RUNTIME_ARTIFACT_KINDS),
        "expected_component_count": EXPECTED_SELECTED_COMPONENTS,
        "expected_publication_count": EXPECTED_SELECTED_PAPERS,
        "expected_question_count": EXPECTED_SELECTED_QUESTIONS,
        "external_v5_replay_required": True,
        "forbidden_v5_runtime_artifact_kinds": list(FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS),
        "official_test_labels_opened": False,
        "projected_row_context_fields": list(PROJECTED_ROW_CONTEXT_FIELDS),
        "projection_input_contract": (
            "externally_replayed_v5_execution_bundle_embedded_adapter_rows_only"
        ),
        "reference_fields_unopened": True,
        "source_artifact_bytes_rehashed": True,
        "source_surface_version": SOURCE_SURFACE_VERSION,
        "v5_adapter_bundle_sha256": EXPECTED_V5_ADAPTER_BUNDLE_SHA256,
        "v5_adapter_pipeline_sha256": EXPECTED_V5_ADAPTER_PIPELINE_SHA256,
        "v5_component_membership_sha256": (EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256),
        "v5_execution_bundle_sha256": EXPECTED_V5_EXECUTION_BUNDLE_SHA256,
        "v5_question_membership_sha256": EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256,
        "v5_row_membership_sha256": EXPECTED_V5_ROW_MEMBERSHIP_SHA256,
        "v5_runtime_pipeline_sha256": EXPECTED_V5_RUNTIME_PIPELINE_SHA256,
    }


def compute_metasyn_v5_source_surface_pipeline_fingerprint(
    *, root: Path | None = None
) -> PipelineFingerprint:
    """Compute the AST dependency-closed identity of this projection component."""

    repository_root = _canonical_repository_root(root or Path(__file__).resolve().parents[2])
    component = PipelineComponentSpec(
        component_id="metasyn-v5-source-surface-projection",
        component_version=SOURCE_SURFACE_COMPONENT_VERSION,
        file_paths=_source_surface_python_dependency_closure(repository_root),
        settings=_pipeline_settings(),
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


class MetaSynV5SourceArtifactBindingV1(ContractModel):
    binding_version: Literal["metasyn-v5-source-artifact-binding-v1"] = (
        SOURCE_ARTIFACT_BINDING_VERSION
    )
    artifact_path: Annotated[str, Field(min_length=1, max_length=2048)]
    source_document_sha256: str
    projection_artifact_sha256: str
    observed_artifact_sha256: str
    observed_artifact_bytes: Annotated[int, Field(ge=1)]
    artifact_binding_sha256: str

    @field_validator(
        "source_document_sha256",
        "projection_artifact_sha256",
        "observed_artifact_sha256",
        "artifact_binding_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        relative = PurePosixPath(value)
        if (
            "\\" in value
            or relative.is_absolute()
            or "." in relative.parts
            or ".." in relative.parts
            or value.startswith("./")
            or relative.as_posix() != value
        ):
            raise ValueError("metasyn_v5_source_surface_artifact_path_unsafe")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> MetaSynV5SourceArtifactBindingV1:
        if (
            len(
                {
                    self.source_document_sha256,
                    self.projection_artifact_sha256,
                    self.observed_artifact_sha256,
                }
            )
            != 1
        ):
            raise ValueError("metasyn_v5_source_surface_artifact_hash_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"artifact_binding_sha256"})
        if hash_canonical(payload) != self.artifact_binding_sha256:
            raise ValueError("metasyn_v5_source_surface_artifact_binding_hash_mismatch")
        return self


class MetaSynV5IndependenceComponentBindingV1(ContractModel):
    binding_version: Literal["metasyn-v5-independence-component-binding-v1"] = (
        COMPONENT_BINDING_VERSION
    )
    independence_component_id: Annotated[str, Field(min_length=1, max_length=256)]
    independence_component_review_ids: Annotated[list[int], Field(min_length=1, max_length=512)]
    independence_component_membership_sha256: str
    component_binding_sha256: str

    @field_validator("independence_component_membership_sha256", "component_binding_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("independence_component_review_ids")
    @classmethod
    def validate_review_ids(cls, value: list[int]) -> list[int]:
        if value != sorted(set(value)) or any(item < 0 for item in value):
            raise ValueError("metasyn_v5_source_surface_component_review_ids_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> MetaSynV5IndependenceComponentBindingV1:
        if self.independence_component_membership_sha256 != hash_canonical(
            self.independence_component_review_ids
        ):
            raise ValueError("metasyn_v5_source_surface_component_membership_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"component_binding_sha256"})
        if hash_canonical(payload) != self.component_binding_sha256:
            raise ValueError("metasyn_v5_source_surface_component_binding_hash_mismatch")
        return self


class MetaSynV5SourceSurfaceRowV1(ContractModel):
    source_surface_row_version: Literal["metasyn-v5-source-surface-row-v1"] = (
        SOURCE_SURFACE_ROW_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_SELECTED_PAPERS)]
    row_key: Annotated[str, Field(min_length=1, max_length=512)]
    upstream_row_context_sha256: str
    question_bundle_sha256: str
    question_spec: MetaSynPilotQuestionSpecV1
    question_spec_sha256: str
    protocol_row_sha256: str
    component_binding: MetaSynV5IndependenceComponentBindingV1
    component_binding_sha256: str
    source_row: MetaSynPilotSourceProjectionRowV1
    source_row_sha256: str
    diagnostic_source_record_sha256: str
    row_source_identity_sha256: str
    projection_spec_sha256: str
    projection_sha256: str
    source_payload_sha256: str
    source_text_sha256: str
    artifact_binding: MetaSynV5SourceArtifactBindingV1
    artifact_binding_sha256: str
    source_surface_row_sha256: str

    @field_validator(
        "upstream_row_context_sha256",
        "question_bundle_sha256",
        "question_spec_sha256",
        "protocol_row_sha256",
        "component_binding_sha256",
        "source_row_sha256",
        "diagnostic_source_record_sha256",
        "row_source_identity_sha256",
        "projection_spec_sha256",
        "projection_sha256",
        "source_payload_sha256",
        "source_text_sha256",
        "artifact_binding_sha256",
        "source_surface_row_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_row(self) -> MetaSynV5SourceSurfaceRowV1:
        projection = self.source_row.projection
        expected_row_key = f"{self.question_spec.question_id}::{self.source_row.doc_id}"
        if self.row_key != expected_row_key:
            raise ValueError("metasyn_v5_source_surface_row_key_mismatch")
        if self.question_spec_sha256 != self.question_spec.question_spec_sha256:
            raise ValueError("metasyn_v5_source_surface_question_hash_alias_mismatch")
        if self.protocol_row_sha256 != self.question_spec.protocol_row_sha256:
            raise ValueError("metasyn_v5_source_surface_protocol_hash_alias_mismatch")
        if self.question_spec.review_id not in (
            self.component_binding.independence_component_review_ids
        ):
            raise ValueError("metasyn_v5_source_surface_question_outside_component")
        if self.component_binding_sha256 != (self.component_binding.component_binding_sha256):
            raise ValueError("metasyn_v5_source_surface_component_hash_alias_mismatch")
        if self.source_row.question_id != self.question_spec.question_id:
            raise ValueError("metasyn_v5_source_surface_question_source_mismatch")
        if self.source_row_sha256 != self.source_row.source_row_sha256:
            raise ValueError("metasyn_v5_source_surface_source_row_hash_alias_mismatch")
        if self.diagnostic_source_record_sha256 != (
            self.source_row.diagnostic_source_record_sha256
        ):
            raise ValueError("metasyn_v5_source_surface_diagnostic_source_hash_alias_mismatch")
        expected_projection_aliases = {
            "projection_sha256": projection.projection_sha256,
            "projection_spec_sha256": projection.projection_spec_sha256,
            "row_source_identity_sha256": projection.row_source_identity_sha256,
            "source_payload_sha256": projection.source_payload_sha256,
            "source_text_sha256": projection.source_text_sha256,
        }
        if any(getattr(self, key) != value for key, value in expected_projection_aliases.items()):
            raise ValueError("metasyn_v5_source_surface_projection_hash_alias_mismatch")
        if self.artifact_binding_sha256 != self.artifact_binding.artifact_binding_sha256:
            raise ValueError("metasyn_v5_source_surface_artifact_binding_alias_mismatch")
        if self.artifact_binding.artifact_path != projection.artifact_path:
            raise ValueError("metasyn_v5_source_surface_artifact_path_alias_mismatch")
        if self.artifact_binding.source_document_sha256 != (
            self.source_row.source_record.source_document.sha256
        ):
            raise ValueError("metasyn_v5_source_surface_source_document_hash_alias_mismatch")
        if self.artifact_binding.projection_artifact_sha256 != projection.artifact_sha256:
            raise ValueError("metasyn_v5_source_surface_projection_artifact_hash_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"source_surface_row_sha256"})
        if hash_canonical(payload) != self.source_surface_row_sha256:
            raise ValueError("metasyn_v5_source_surface_row_hash_mismatch")
        return self


class MetaSynV5SourceSurfaceV1(ContractModel):
    source_surface_version: Literal["metasyn-v5-source-surface-v1"] = SOURCE_SURFACE_VERSION
    status: Literal["externally_replayed_v5_source_lineage_only"] = (
        "externally_replayed_v5_source_lineage_only"
    )
    v5_execution_workspace_relative: Literal["data/cache/metasyn/bounded-anthropic-yield-v5"] = (
        V5_EXECUTION_WORKSPACE.as_posix()
    )
    v5_execution_bundle_sha256: str
    v5_adapter_bundle_sha256: str
    v5_adapter_pipeline_sha256: str
    v5_runtime_pipeline_sha256: str
    v5_upstream_pilot_pipeline_sha256: str
    v5_downstream_verifier_pipeline_sha256: str
    v5_config_sha256: str
    v5_native_schema_v2_contract_sha256: str
    v5_question_membership_sha256: str
    v5_component_membership_sha256: str
    v5_row_membership_sha256: str
    source_surface_pipeline_fingerprint: PipelineFingerprint
    source_surface_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    projected_row_context_fields: list[str]
    rows: Annotated[list[MetaSynV5SourceSurfaceRowV1], Field(min_length=32, max_length=32)]
    projected_row_hash_membership_sha256: str
    source_artifact_membership_sha256: str
    projection_input_contract: Literal[
        "externally_replayed_v5_execution_bundle_embedded_adapter_rows_only"
    ] = "externally_replayed_v5_execution_bundle_embedded_adapter_rows_only"
    consumed_v5_runtime_artifact_kinds: list[str]
    forbidden_v5_runtime_artifact_kinds: list[str]
    upstream_v5_execution_bundle_consumed: Literal[True] = True
    upstream_v5_call_receipts_consumed: Literal[False] = False
    upstream_v5_row_results_consumed: Literal[False] = False
    upstream_v5_provider_outputs_consumed: Literal[False] = False
    source_artifact_bytes_rehashed: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    directional_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    source_surface_sha256: str

    @field_validator(
        "v5_execution_bundle_sha256",
        "v5_adapter_bundle_sha256",
        "v5_adapter_pipeline_sha256",
        "v5_runtime_pipeline_sha256",
        "v5_upstream_pilot_pipeline_sha256",
        "v5_downstream_verifier_pipeline_sha256",
        "v5_config_sha256",
        "v5_native_schema_v2_contract_sha256",
        "v5_question_membership_sha256",
        "v5_component_membership_sha256",
        "v5_row_membership_sha256",
        "source_surface_pipeline_sha256",
        "projected_row_hash_membership_sha256",
        "source_artifact_membership_sha256",
        "source_surface_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator(
        "projected_row_context_fields",
        "consumed_v5_runtime_artifact_kinds",
        "forbidden_v5_runtime_artifact_kinds",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"metasyn_v5_source_surface_values_not_sorted_unique:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_surface(self) -> MetaSynV5SourceSurfaceV1:
        observed_anchors = {
            "adapter_bundle_sha256": self.v5_adapter_bundle_sha256,
            "adapter_pipeline_sha256": self.v5_adapter_pipeline_sha256,
            "component_membership_sha256": self.v5_component_membership_sha256,
            "config_sha256": self.v5_config_sha256,
            "downstream_verifier_pipeline_sha256": (self.v5_downstream_verifier_pipeline_sha256),
            "execution_bundle_sha256": self.v5_execution_bundle_sha256,
            "native_schema_v2_contract_sha256": (self.v5_native_schema_v2_contract_sha256),
            "question_membership_sha256": self.v5_question_membership_sha256,
            "row_membership_sha256": self.v5_row_membership_sha256,
            "runtime_pipeline_sha256": self.v5_runtime_pipeline_sha256,
            "upstream_pilot_pipeline_sha256": (self.v5_upstream_pilot_pipeline_sha256),
        }
        if observed_anchors != _EXPECTED_V5_ANCHORS:
            raise ValueError("metasyn_v5_source_surface_anchor_mismatch")
        if self.projected_row_context_fields != list(PROJECTED_ROW_CONTEXT_FIELDS):
            raise ValueError("metasyn_v5_source_surface_projection_whitelist_mismatch")
        if self.consumed_v5_runtime_artifact_kinds != list(CONSUMED_V5_RUNTIME_ARTIFACT_KINDS):
            raise ValueError("metasyn_v5_source_surface_consumed_artifacts_mismatch")
        if self.forbidden_v5_runtime_artifact_kinds != list(FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS):
            raise ValueError("metasyn_v5_source_surface_forbidden_artifacts_mismatch")

        fingerprint = self.source_surface_pipeline_fingerprint
        if self.source_surface_pipeline_sha256 != fingerprint.pipeline_sha256:
            raise ValueError("metasyn_v5_source_surface_pipeline_hash_alias_mismatch")
        if len(fingerprint.components) != 1:
            raise ValueError("metasyn_v5_source_surface_pipeline_component_count_mismatch")
        component = fingerprint.components[0]
        if (
            component.component_id != "metasyn-v5-source-surface-projection"
            or component.component_version != SOURCE_SURFACE_COMPONENT_VERSION
            or component.settings != _pipeline_settings()
        ):
            raise ValueError("metasyn_v5_source_surface_pipeline_component_mismatch")

        if len(self.rows) != self.publication_count:
            raise ValueError("metasyn_v5_source_surface_publication_count_mismatch")
        if [row.row_ordinal for row in self.rows] != list(range(EXPECTED_SELECTED_PAPERS)):
            raise ValueError("metasyn_v5_source_surface_row_ordinals_mismatch")
        row_keys = [row.row_key for row in self.rows]
        if row_keys != sorted(set(row_keys)):
            raise ValueError("metasyn_v5_source_surface_rows_not_sorted_unique")
        if hash_canonical(row_keys) != self.v5_row_membership_sha256:
            raise ValueError("metasyn_v5_source_surface_row_membership_mismatch")
        question_ids = sorted({row.question_spec.question_id for row in self.rows})
        if (
            len(question_ids) != self.question_count
            or hash_canonical(question_ids) != self.v5_question_membership_sha256
        ):
            raise ValueError("metasyn_v5_source_surface_question_membership_mismatch")

        components_by_id: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            binding = row.component_binding
            descriptor = {
                "independence_component_id": binding.independence_component_id,
                "independence_component_review_ids": (binding.independence_component_review_ids),
                "independence_component_membership_sha256": (
                    binding.independence_component_membership_sha256
                ),
            }
            prior = components_by_id.setdefault(binding.independence_component_id, descriptor)
            if prior != descriptor:
                raise ValueError("metasyn_v5_source_surface_component_conflict")
        components = [components_by_id[key] for key in sorted(components_by_id)]
        if (
            len(components) != self.component_count
            or hash_canonical(components) != self.v5_component_membership_sha256
        ):
            raise ValueError("metasyn_v5_source_surface_component_membership_mismatch")

        row_hashes = [row.source_surface_row_sha256 for row in self.rows]
        if hash_canonical(row_hashes) != self.projected_row_hash_membership_sha256:
            raise ValueError("metasyn_v5_source_surface_projected_row_hash_membership_mismatch")
        artifacts_by_path: dict[str, MetaSynV5SourceArtifactBindingV1] = {}
        for row in self.rows:
            binding = row.artifact_binding
            prior = artifacts_by_path.setdefault(binding.artifact_path, binding)
            if prior != binding:
                raise ValueError("metasyn_v5_source_surface_artifact_binding_conflict")
        artifacts = [
            artifacts_by_path[path].model_dump(mode="json") for path in sorted(artifacts_by_path)
        ]
        if hash_canonical(artifacts) != self.source_artifact_membership_sha256:
            raise ValueError("metasyn_v5_source_surface_artifact_membership_mismatch")

        payload = self.model_dump(mode="json", exclude={"source_surface_sha256"})
        if hash_canonical(payload) != self.source_surface_sha256:
            raise ValueError("metasyn_v5_source_surface_hash_mismatch")
        return self


def _assert_v5_anchors(bundle: MetaSynHostedExecutionBundleV1) -> None:
    observed = {name: getattr(bundle, name) for name in _EXPECTED_V5_ANCHORS}
    if observed != _EXPECTED_V5_ANCHORS:
        mismatched = sorted(
            name
            for name, expected in _EXPECTED_V5_ANCHORS.items()
            if observed.get(name) != expected
        )
        suffix = ",".join(mismatched)
        raise MetaSynV5SourceSurfaceError(
            f"metasyn_v5_source_surface_immutable_anchor_mismatch:{suffix}"
        )
    if (
        bundle.question_count != EXPECTED_SELECTED_QUESTIONS
        or bundle.component_count != EXPECTED_SELECTED_COMPONENTS
        or bundle.publication_count != EXPECTED_SELECTED_PAPERS
    ):
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_v5_roster_count_mismatch")
    if (
        not bundle.reference_fields_unopened
        or bundle.official_test_labels_opened
        or bundle.provider_calls_made
    ):
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_v5_label_or_execution_boundary_mismatch"
        )


def _freeze_component_binding(
    *,
    component_id: str,
    review_ids: list[int],
    membership_sha256: str,
) -> MetaSynV5IndependenceComponentBindingV1:
    payload = {
        "binding_version": COMPONENT_BINDING_VERSION,
        "independence_component_id": component_id,
        "independence_component_review_ids": review_ids,
        "independence_component_membership_sha256": membership_sha256,
    }
    return MetaSynV5IndependenceComponentBindingV1.model_validate(
        {**payload, "component_binding_sha256": hash_canonical(payload)}
    )


def _freeze_artifact_binding(
    *,
    artifact_path: str,
    source_document_sha256: str,
    projection_artifact_sha256: str,
    observed_artifact_sha256: str,
    observed_artifact_bytes: int,
) -> MetaSynV5SourceArtifactBindingV1:
    payload = {
        "binding_version": SOURCE_ARTIFACT_BINDING_VERSION,
        "artifact_path": artifact_path,
        "source_document_sha256": source_document_sha256,
        "projection_artifact_sha256": projection_artifact_sha256,
        "observed_artifact_sha256": observed_artifact_sha256,
        "observed_artifact_bytes": observed_artifact_bytes,
    }
    return MetaSynV5SourceArtifactBindingV1.model_validate(
        {**payload, "artifact_binding_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_v5_source_surface(
    *,
    repository_root: Path | None = None,
    execution_workspace: Path | None = None,
) -> MetaSynV5SourceSurfaceV1:
    """Externally replay pinned v5 and return its source-only successor surface."""

    root = _canonical_repository_root(repository_root or Path(__file__).resolve().parents[2])
    workspace = _pinned_v5_workspace(
        repository_root=root,
        execution_workspace=execution_workspace,
    )
    canonical_workspace, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=root,
        external_replay=True,
    )
    if canonical_workspace != workspace:
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_external_replay_workspace_mismatch"
        )
    _assert_v5_anchors(bundle)

    contexts = bundle.adapter_bundle.row_contexts
    if [context.row_key for context in contexts] != sorted(
        {context.row_key for context in contexts}
    ):
        raise MetaSynV5SourceSurfaceError(
            "metasyn_v5_source_surface_upstream_rows_not_sorted_unique"
        )

    artifact_cache: dict[str, tuple[str, int, str]] = {}
    rows: list[MetaSynV5SourceSurfaceRowV1] = []
    for ordinal, context in enumerate(contexts):
        question_spec = MetaSynPilotQuestionSpecV1.model_validate(
            context.question_spec.model_dump(mode="json")
        )
        source_row = MetaSynPilotSourceProjectionRowV1.model_validate(
            context.source_row.model_dump(mode="json")
        )
        projection = source_row.projection
        artifact_path = projection.artifact_path
        expected_artifact_sha256 = source_row.source_record.source_document.sha256
        cached = artifact_cache.get(artifact_path)
        if cached is None:
            observed_sha256, observed_bytes = _rehash_source_artifact(
                repository_root=root,
                artifact_path=artifact_path,
                expected_sha256=expected_artifact_sha256,
            )
            cached = (observed_sha256, observed_bytes, expected_artifact_sha256)
            artifact_cache[artifact_path] = cached
        observed_sha256, observed_bytes, cached_expected_sha256 = cached
        if cached_expected_sha256 != expected_artifact_sha256:
            raise MetaSynV5SourceSurfaceError(
                "metasyn_v5_source_surface_shared_artifact_hash_conflict"
            )

        component_binding = _freeze_component_binding(
            component_id=context.independence_component_id,
            review_ids=list(context.independence_component_review_ids),
            membership_sha256=context.independence_component_membership_sha256,
        )
        artifact_binding = _freeze_artifact_binding(
            artifact_path=artifact_path,
            source_document_sha256=expected_artifact_sha256,
            projection_artifact_sha256=projection.artifact_sha256,
            observed_artifact_sha256=observed_sha256,
            observed_artifact_bytes=observed_bytes,
        )
        row_payload = {
            "source_surface_row_version": SOURCE_SURFACE_ROW_VERSION,
            "row_ordinal": ordinal,
            "row_key": context.row_key,
            "upstream_row_context_sha256": context.row_context_sha256,
            "question_bundle_sha256": context.question_bundle_sha256,
            "question_spec": question_spec,
            "question_spec_sha256": context.question_spec_sha256,
            "protocol_row_sha256": question_spec.protocol_row_sha256,
            "component_binding": component_binding,
            "component_binding_sha256": component_binding.component_binding_sha256,
            "source_row": source_row,
            "source_row_sha256": context.source_row_sha256,
            "diagnostic_source_record_sha256": (source_row.diagnostic_source_record_sha256),
            "row_source_identity_sha256": projection.row_source_identity_sha256,
            "projection_spec_sha256": projection.projection_spec_sha256,
            "projection_sha256": context.projection_sha256,
            "source_payload_sha256": projection.source_payload_sha256,
            "source_text_sha256": projection.source_text_sha256,
            "artifact_binding": artifact_binding,
            "artifact_binding_sha256": artifact_binding.artifact_binding_sha256,
        }
        rows.append(
            MetaSynV5SourceSurfaceRowV1.model_validate(
                {
                    **row_payload,
                    "source_surface_row_sha256": hash_canonical(row_payload),
                }
            )
        )

    fingerprint = compute_metasyn_v5_source_surface_pipeline_fingerprint(root=root)
    artifacts_by_path = {row.artifact_binding.artifact_path: row.artifact_binding for row in rows}
    artifact_membership = [
        artifacts_by_path[path].model_dump(mode="json") for path in sorted(artifacts_by_path)
    ]
    surface_payload = {
        "source_surface_version": SOURCE_SURFACE_VERSION,
        "status": "externally_replayed_v5_source_lineage_only",
        "v5_execution_workspace_relative": V5_EXECUTION_WORKSPACE.as_posix(),
        "v5_execution_bundle_sha256": bundle.execution_bundle_sha256,
        "v5_adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "v5_adapter_pipeline_sha256": bundle.adapter_pipeline_sha256,
        "v5_runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "v5_upstream_pilot_pipeline_sha256": (bundle.upstream_pilot_pipeline_sha256),
        "v5_downstream_verifier_pipeline_sha256": (bundle.downstream_verifier_pipeline_sha256),
        "v5_config_sha256": bundle.config_sha256,
        "v5_native_schema_v2_contract_sha256": (bundle.native_schema_v2_contract_sha256),
        "v5_question_membership_sha256": bundle.question_membership_sha256,
        "v5_component_membership_sha256": bundle.component_membership_sha256,
        "v5_row_membership_sha256": bundle.row_membership_sha256,
        "source_surface_pipeline_fingerprint": fingerprint,
        "source_surface_pipeline_sha256": fingerprint.pipeline_sha256,
        "question_count": bundle.question_count,
        "component_count": bundle.component_count,
        "publication_count": bundle.publication_count,
        "projected_row_context_fields": list(PROJECTED_ROW_CONTEXT_FIELDS),
        "rows": rows,
        "projected_row_hash_membership_sha256": hash_canonical(
            [row.source_surface_row_sha256 for row in rows]
        ),
        "source_artifact_membership_sha256": hash_canonical(artifact_membership),
        "projection_input_contract": (
            "externally_replayed_v5_execution_bundle_embedded_adapter_rows_only"
        ),
        "consumed_v5_runtime_artifact_kinds": list(CONSUMED_V5_RUNTIME_ARTIFACT_KINDS),
        "forbidden_v5_runtime_artifact_kinds": list(FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS),
        "upstream_v5_execution_bundle_consumed": True,
        "upstream_v5_call_receipts_consumed": False,
        "upstream_v5_row_results_consumed": False,
        "upstream_v5_provider_outputs_consumed": False,
        "source_artifact_bytes_rehashed": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "directional_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynV5SourceSurfaceV1.model_validate(
        {
            **surface_payload,
            "source_surface_sha256": hash_canonical(surface_payload),
        }
    )


def validate_metasyn_v5_source_surface(
    *,
    source_surface: MetaSynV5SourceSurfaceV1 | dict[str, Any],
    repository_root: Path | None = None,
    external_replay: bool = True,
) -> MetaSynV5SourceSurfaceV1:
    """Revalidate a source surface and optionally replay v5 and every source byte."""

    try:
        if isinstance(source_surface, MetaSynV5SourceSurfaceV1):
            canonical = MetaSynV5SourceSurfaceV1.model_validate(
                source_surface.model_dump(mode="json")
            )
        else:
            canonical = MetaSynV5SourceSurfaceV1.model_validate(source_surface)
    except ValueError as exc:
        raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_contract_invalid") from exc
    if external_replay:
        replayed = freeze_metasyn_v5_source_surface(repository_root=repository_root)
        if replayed != canonical:
            raise MetaSynV5SourceSurfaceError("metasyn_v5_source_surface_external_replay_mismatch")
    return canonical


__all__ = [
    "CONSUMED_V5_RUNTIME_ARTIFACT_KINDS",
    "EXPECTED_V5_ADAPTER_BUNDLE_SHA256",
    "EXPECTED_V5_ADAPTER_PIPELINE_SHA256",
    "EXPECTED_V5_COMPONENT_MEMBERSHIP_SHA256",
    "EXPECTED_V5_EXECUTION_BUNDLE_SHA256",
    "EXPECTED_V5_QUESTION_MEMBERSHIP_SHA256",
    "EXPECTED_V5_ROW_MEMBERSHIP_SHA256",
    "EXPECTED_V5_RUNTIME_PIPELINE_SHA256",
    "FORBIDDEN_V5_RUNTIME_ARTIFACT_KINDS",
    "PROJECTED_ROW_CONTEXT_FIELDS",
    "SOURCE_SURFACE_VERSION",
    "V5_EXECUTION_WORKSPACE",
    "MetaSynV5IndependenceComponentBindingV1",
    "MetaSynV5SourceArtifactBindingV1",
    "MetaSynV5SourceSurfaceError",
    "MetaSynV5SourceSurfaceRowV1",
    "MetaSynV5SourceSurfaceV1",
    "compute_metasyn_v5_source_surface_pipeline_fingerprint",
    "freeze_metasyn_v5_source_surface",
    "validate_metasyn_v5_source_surface",
]
