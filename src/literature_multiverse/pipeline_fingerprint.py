"""Computed, self-verifying identity for a frozen scientific pipeline.

The fingerprint is intentionally built from explicit repository-relative files and
JSON settings.  A caller-supplied hexadecimal string is not a pipeline identity:
the files are read, hashed, and compared with the expected fingerprint before a
downstream calibration artifact may rely on it.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel


class PipelineFingerprintError(ValueError):
    """The pipeline identity could not be computed or verified."""


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _validate_relative_file_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("pipeline_file_path_must_use_posix_separators")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
    ):
        raise ValueError("pipeline_file_path_must_be_normalized_repository_relative")
    return value


class PipelineComponentSpec(ContractModel):
    """The exact files and non-secret settings that define one pipeline component."""

    component_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")]
    component_version: Annotated[str, Field(min_length=1)]
    file_paths: list[str]
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("file_paths")
    @classmethod
    def validate_file_paths(cls, value: list[str]) -> list[str]:
        normalized = [_validate_relative_file_path(path) for path in value]
        if normalized != sorted(set(normalized)):
            raise ValueError("pipeline_component_files_must_be_sorted_unique")
        return normalized

    @model_validator(mode="after")
    def validate_identity_is_nonempty(self) -> PipelineComponentSpec:
        if not self.file_paths and not self.settings:
            raise ValueError("pipeline_component_identity_empty")
        return self


class PipelineFileHash(ContractModel):
    """Digest and byte count observed for one explicit pipeline file."""

    path: str
    sha256: str
    bytes: Annotated[int, Field(ge=0)]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_file_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "pipeline_file")


class PipelineComponentFingerprint(ContractModel):
    """Self-hashed identity of one component and all of its declared inputs."""

    fingerprint_version: Literal["pipeline-component-v1"] = "pipeline-component-v1"
    component_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")]
    component_version: Annotated[str, Field(min_length=1)]
    files: list[PipelineFileHash]
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    component_sha256: str

    @field_validator("component_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "pipeline_component")

    @model_validator(mode="after")
    def validate_component_integrity(self) -> PipelineComponentFingerprint:
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("pipeline_component_files_must_be_sorted_unique")
        if not self.files and not self.settings:
            raise ValueError("pipeline_component_identity_empty")
        payload = self.model_dump(mode="json", exclude={"component_sha256"})
        if hash_canonical(payload) != self.component_sha256:
            raise ValueError("pipeline_component_hash_mismatch")
        return self


class PipelineFingerprint(ContractModel):
    """Portable pipeline identity whose digest covers every component and file hash."""

    fingerprint_version: Literal["computed-pipeline-v1"] = "computed-pipeline-v1"
    components: list[PipelineComponentFingerprint]
    pipeline_sha256: str

    @field_validator("pipeline_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "pipeline")

    @model_validator(mode="after")
    def validate_pipeline_integrity(self) -> PipelineFingerprint:
        component_ids = [component.component_id for component in self.components]
        if not component_ids or component_ids != sorted(set(component_ids)):
            raise ValueError("pipeline_components_must_be_nonempty_sorted_unique")
        paths = [file.path for component in self.components for file in component.files]
        if not paths:
            raise ValueError("pipeline_fingerprint_requires_at_least_one_file")
        if len(paths) != len(set(paths)):
            raise ValueError("pipeline_file_may_belong_to_only_one_component")
        payload = self.model_dump(mode="json", exclude={"pipeline_sha256"})
        if hash_canonical(payload) != self.pipeline_sha256:
            raise ValueError("pipeline_fingerprint_hash_mismatch")
        return self


class PipelineFingerprintVerification(ContractModel):
    """Self-hashed expected-versus-computed verification result."""

    verification_version: Literal["pipeline-verification-v1"] = "pipeline-verification-v1"
    expected_pipeline_sha256: str
    computed_pipeline_sha256: str | None
    status: Literal["matched", "mismatch", "unverifiable"]
    issues: list[str]
    computed: PipelineFingerprint | None
    verification_sha256: str

    @field_validator("expected_pipeline_sha256")
    @classmethod
    def validate_expected_hash(cls, value: str) -> str:
        return _validate_sha256(value, "expected_pipeline")

    @field_validator("computed_pipeline_sha256")
    @classmethod
    def validate_computed_hash(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_sha256(value, "computed_pipeline")
        return value

    @field_validator("verification_sha256")
    @classmethod
    def validate_verification_hash(cls, value: str) -> str:
        return _validate_sha256(value, "pipeline_verification")

    @model_validator(mode="after")
    def validate_verification_integrity(self) -> PipelineFingerprintVerification:
        if self.issues != sorted(set(self.issues)):
            raise ValueError("pipeline_verification_issues_must_be_sorted_unique")
        if self.computed is None:
            if self.computed_pipeline_sha256 is not None or self.status != "unverifiable":
                raise ValueError("pipeline_verification_missing_computed_state_mismatch")
            if not self.issues:
                raise ValueError("unverifiable_pipeline_requires_issue")
        else:
            if self.computed.pipeline_sha256 != self.computed_pipeline_sha256:
                raise ValueError("pipeline_verification_computed_hash_mismatch")
            expected_status = (
                "matched"
                if self.expected_pipeline_sha256 == self.computed_pipeline_sha256
                else "mismatch"
            )
            if self.status != expected_status:
                raise ValueError("pipeline_verification_status_mismatch")
            if (self.status == "matched") != (not self.issues):
                raise ValueError("pipeline_verification_issue_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"verification_sha256"})
        if hash_canonical(payload) != self.verification_sha256:
            raise ValueError("pipeline_verification_hash_mismatch")
        return self


def _checked_file(root: Path, relative: str) -> Path:
    """Resolve one file without permitting symlinks or repository escape."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise PipelineFingerprintError("pipeline_root_not_directory")
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise PipelineFingerprintError(f"pipeline_file_symlink_forbidden:{relative}")
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PipelineFingerprintError(f"pipeline_file_missing:{relative}") from exc
    if not resolved.is_relative_to(root):
        raise PipelineFingerprintError(f"pipeline_file_outside_root:{relative}")
    if not resolved.is_file():
        raise PipelineFingerprintError(f"pipeline_path_not_file:{relative}")
    return resolved


def _component_fingerprint(
    *, root: Path, spec: PipelineComponentSpec
) -> PipelineComponentFingerprint:
    files = [
        PipelineFileHash(
            path=relative,
            sha256=sha256_file(path := _checked_file(root, relative)),
            bytes=path.stat().st_size,
        )
        for relative in spec.file_paths
    ]
    payload = {
        "fingerprint_version": "pipeline-component-v1",
        "component_id": spec.component_id,
        "component_version": spec.component_version,
        "files": files,
        "settings": spec.settings,
    }
    return PipelineComponentFingerprint.model_validate(
        {**payload, "component_sha256": hash_canonical(payload)}
    )


def compute_pipeline_fingerprint(
    *, root: Path, components: list[PipelineComponentSpec]
) -> PipelineFingerprint:
    """Hash the current bytes of an explicit, closed component manifest."""

    if not components:
        raise PipelineFingerprintError("pipeline_component_manifest_empty")
    try:
        normalized = sorted(
            [
                PipelineComponentSpec.model_validate(component.model_dump(mode="json"))
                for component in components
            ],
            key=lambda component: component.component_id,
        )
    except (AttributeError, ValueError) as exc:
        raise PipelineFingerprintError("pipeline_component_manifest_invalid") from exc
    if len({component.component_id for component in normalized}) != len(normalized):
        raise PipelineFingerprintError("pipeline_component_id_duplicate")
    paths = [path for component in normalized for path in component.file_paths]
    if len(paths) != len(set(paths)):
        raise PipelineFingerprintError("pipeline_file_assigned_to_multiple_components")
    computed_components = [
        _component_fingerprint(root=root, spec=component) for component in normalized
    ]
    payload = {
        "fingerprint_version": "computed-pipeline-v1",
        "components": computed_components,
    }
    return PipelineFingerprint.model_validate(
        {**payload, "pipeline_sha256": hash_canonical(payload)}
    )


def validate_pipeline_fingerprint_integrity(
    fingerprint: PipelineFingerprint,
) -> PipelineFingerprint:
    """Reparse a snapshot so in-place mutation of nested data cannot bypass hashes."""

    if not isinstance(fingerprint, PipelineFingerprint):
        raise PipelineFingerprintError("pipeline_fingerprint_contract_invalid")
    try:
        return PipelineFingerprint.model_validate(fingerprint.model_dump(mode="json"))
    except ValueError as exc:
        raise PipelineFingerprintError("pipeline_fingerprint_integrity_changed") from exc


def validate_pipeline_verification_integrity(
    verification: PipelineFingerprintVerification,
) -> PipelineFingerprintVerification:
    """Revalidate a verification report and all of its nested computed identity."""

    if not isinstance(verification, PipelineFingerprintVerification):
        raise PipelineFingerprintError("pipeline_verification_contract_invalid")
    try:
        return PipelineFingerprintVerification.model_validate(
            verification.model_dump(mode="json")
        )
    except ValueError as exc:
        raise PipelineFingerprintError("pipeline_verification_integrity_changed") from exc


def _verification_result(
    *,
    expected_sha256: str,
    computed: PipelineFingerprint | None,
    status: Literal["matched", "mismatch", "unverifiable"],
    issues: list[str],
) -> PipelineFingerprintVerification:
    payload = {
        "verification_version": "pipeline-verification-v1",
        "expected_pipeline_sha256": expected_sha256,
        "computed_pipeline_sha256": None if computed is None else computed.pipeline_sha256,
        "status": status,
        "issues": sorted(set(issues)),
        "computed": computed,
    }
    return PipelineFingerprintVerification.model_validate(
        {**payload, "verification_sha256": hash_canonical(payload)}
    )


def verify_pipeline_fingerprint(
    *, expected: PipelineFingerprint, root: Path
) -> PipelineFingerprintVerification:
    """Recompute every expected file and report exact expected-versus-current identity."""

    expected = validate_pipeline_fingerprint_integrity(expected)
    specs = [
        PipelineComponentSpec(
            component_id=component.component_id,
            component_version=component.component_version,
            file_paths=[file.path for file in component.files],
            settings=component.settings,
        )
        for component in expected.components
    ]
    try:
        computed = compute_pipeline_fingerprint(root=root, components=specs)
    except (OSError, ValueError) as exc:
        return _verification_result(
            expected_sha256=expected.pipeline_sha256,
            computed=None,
            status="unverifiable",
            issues=[str(exc) or exc.__class__.__name__],
        )

    issues: list[str] = []
    expected_components = {item.component_id: item for item in expected.components}
    for observed_component in computed.components:
        expected_component = expected_components[observed_component.component_id]
        expected_files = {item.path: item for item in expected_component.files}
        for observed_file in observed_component.files:
            expected_file = expected_files[observed_file.path]
            if observed_file.sha256 != expected_file.sha256:
                issues.append(f"file_sha256_mismatch:{observed_file.path}")
            if observed_file.bytes != expected_file.bytes:
                issues.append(f"file_bytes_mismatch:{observed_file.path}")
        if observed_component.component_sha256 != expected_component.component_sha256:
            issues.append(f"component_sha256_mismatch:{observed_component.component_id}")
    if computed.pipeline_sha256 != expected.pipeline_sha256:
        issues.append("pipeline_sha256_mismatch")
    return _verification_result(
        expected_sha256=expected.pipeline_sha256,
        computed=computed,
        status="matched" if not issues else "mismatch",
        issues=issues,
    )


def require_pipeline_fingerprint_match(
    *, expected: PipelineFingerprint, root: Path
) -> PipelineFingerprintVerification:
    """Return a matched proof or fail closed on missing, stale, or changed inputs."""

    verification = verify_pipeline_fingerprint(expected=expected, root=root)
    if verification.status != "matched":
        raise PipelineFingerprintError(
            f"pipeline_fingerprint_not_matched:{verification.status}:{','.join(verification.issues)}"
        )
    return verification


__all__ = [
    "PipelineComponentFingerprint",
    "PipelineComponentSpec",
    "PipelineFileHash",
    "PipelineFingerprint",
    "PipelineFingerprintError",
    "PipelineFingerprintVerification",
    "compute_pipeline_fingerprint",
    "require_pipeline_fingerprint_match",
    "validate_pipeline_fingerprint_integrity",
    "validate_pipeline_verification_integrity",
    "verify_pipeline_fingerprint",
]
