"""Content-addressed, append-only storage for raw harvester responses."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import canonical_json_bytes

from .contracts import RetrievedPayload


class ArchiveIntegrityError(RuntimeError):
    """An existing content-addressed object did not contain the expected bytes."""


@dataclass(frozen=True, slots=True)
class ArchivedPayload:
    sha256: str
    bytes: int
    media_type: str | None
    blob_path: str
    receipt_path: str
    url: str
    retrieved_at: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


_MEDIA_SUFFIXES = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/xml": ".xml",
    "application/xhtml+xml": ".xhtml",
    "text/html": ".html",
    "text/plain": ".txt",
    "text/xml": ".xml",
}


def _suffix(media_type: str | None) -> str:
    normalized = (media_type or "").partition(";")[0].strip().casefold()
    return _MEDIA_SUFFIXES.get(normalized, ".bin")


def _write_content_addressed(path: Path, content: bytes) -> None:
    """Create once; an identical existing object is an idempotent success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise ArchiveIntegrityError(
                f"archive_object_hash_collision_or_mutation:{path}"
            ) from None
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


class ImmutableArchive:
    """Store raw bytes and receipts under their SHA-256 identities.

    Existing objects are never replaced, including when a derived candidate view is
    regenerated with ``--force``.  ``path_base`` only controls paths written into
    provenance; it must contain the archive root.
    """

    def __init__(self, root: Path, *, path_base: Path | None = None) -> None:
        self.root = root.resolve()
        self.path_base = (path_base or self.root).resolve()
        try:
            self.root.relative_to(self.path_base)
        except ValueError as exc:
            raise ValueError("archive_root_outside_path_base") from exc

    def _display_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.path_base).as_posix()

    def archive(
        self,
        payload: RetrievedPayload,
        *,
        kind: str,
        source_name: str,
        context: dict[str, Any] | None = None,
    ) -> ArchivedPayload:
        if not kind or not source_name:
            raise ValueError("archive_kind_and_source_required")
        digest = hashlib.sha256(payload.body).hexdigest()
        blob = self.root / "blobs" / digest[:2] / f"{digest}{_suffix(payload.media_type)}"
        _write_content_addressed(blob, payload.body)

        receipt = {
            "archive_version": "1",
            "kind": kind,
            "source": source_name,
            "url": payload.url,
            "retrieved_at": payload.retrieved_at.isoformat(),
            "status_code": payload.status_code,
            "media_type": payload.media_type,
            "response_headers": dict(sorted(payload.response_headers.items())),
            "content_sha256": digest,
            "content_bytes": len(payload.body),
            "blob_path": self._display_path(blob),
            "context": context or {},
        }
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
        receipt_path = self.root / "receipts" / receipt_hash[:2] / f"{receipt_hash}.json"
        _write_content_addressed(receipt_path, receipt_bytes)
        return ArchivedPayload(
            sha256=digest,
            bytes=len(payload.body),
            media_type=payload.media_type,
            blob_path=self._display_path(blob),
            receipt_path=self._display_path(receipt_path),
            url=payload.url,
            retrieved_at=payload.retrieved_at.isoformat(),
        )

    def verify(self, entry: ArchivedPayload) -> None:
        blob = self.path_base / entry.blob_path
        receipt = self.path_base / entry.receipt_path
        if not blob.is_file() or hashlib.sha256(blob.read_bytes()).hexdigest() != entry.sha256:
            raise ArchiveIntegrityError(f"archive_blob_hash_mismatch:{entry.blob_path}")
        try:
            receipt_bytes = receipt.read_bytes()
            receipt_value = json.loads(receipt_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchiveIntegrityError(f"archive_receipt_invalid:{entry.receipt_path}") from exc
        receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
        if receipt.stem != receipt_digest:
            raise ArchiveIntegrityError(f"archive_receipt_hash_mismatch:{entry.receipt_path}")
        if (
            receipt_value.get("content_sha256") != entry.sha256
            or receipt_value.get("content_bytes") != entry.bytes
            or receipt_value.get("blob_path") != entry.blob_path
        ):
            raise ArchiveIntegrityError(f"archive_receipt_mismatch:{entry.receipt_path}")


__all__ = ["ArchiveIntegrityError", "ArchivedPayload", "ImmutableArchive"]
