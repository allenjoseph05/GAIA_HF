"""Content-addressed storage for attachments and derived evidence artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path, PurePath

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gaia_max.attachment_validation import AttachmentValidationResult
from gaia_max.domain import ArtifactRef, ArtifactSource

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ArtifactStoreError(RuntimeError):
    """Base error for content-addressed artifact storage."""


class ArtifactNotFoundError(ArtifactStoreError):
    """The requested artifact does not exist."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Stored bytes or metadata no longer match their content identity."""


class UnsafeArtifactNameError(ArtifactStoreError):
    """An untrusted original filename is unsafe to record."""


class ArtifactOrigin(BaseModel):
    """One observed provenance record for immutable content bytes."""

    model_config = ConfigDict(extra="forbid")

    original_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=255)
    source: ArtifactSource
    retrieved_at: datetime
    task_id: str | None = None
    source_locator: str | None = None
    validation: AttachmentValidationResult | None = None

    def identity(self) -> tuple[str, str, str, str | None, str | None]:
        """Identity excluding retrieval time, so repeated fetches do not add duplicates."""

        return (
            self.original_name,
            self.media_type,
            self.source.value,
            self.task_id,
            self.source_locator,
        )


class ArtifactMetadata(BaseModel):
    """Portable metadata stored next to immutable content."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(gt=0)
    relative_content_path: str = Field(min_length=1)
    created_at: datetime
    origins: list[ArtifactOrigin] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> ArtifactMetadata:
        if self.artifact_id != self.sha256:
            raise ValueError("artifact ID must equal its SHA-256")
        relative = PurePath(self.relative_content_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact content path must be safe and relative")
        return self


class ArtifactStore:
    """Store immutable bytes by SHA-256 and maintain answer-free provenance metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.content_root = self.root / "sha256"
        self.content_root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def put_bytes(
        self,
        content: bytes,
        *,
        original_name: str,
        media_type: str,
        source: ArtifactSource,
        retrieved_at: datetime | None = None,
        task_id: str | None = None,
        source_locator: str | None = None,
        validation: AttachmentValidationResult | None = None,
    ) -> ArtifactRef:
        """Atomically store bytes and return a runtime reference."""

        self._validate_original_name(original_name)
        if not content:
            raise ArtifactStoreError("cannot store an empty artifact")
        if not media_type.strip():
            raise ArtifactStoreError("artifact media type cannot be empty")

        observed_at = retrieved_at or datetime.now(UTC)
        digest = hashlib.sha256(content).hexdigest()
        self._validate_report(
            validation,
            digest=digest,
            size_bytes=len(content),
            original_name=original_name,
            media_type=media_type,
        )
        artifact_dir = self._artifact_dir(digest)
        content_path = artifact_dir / "content"
        metadata_path = artifact_dir / "metadata.json"
        origin = ArtifactOrigin(
            original_name=original_name,
            media_type=media_type,
            source=source,
            retrieved_at=observed_at,
            task_id=task_id,
            source_locator=source_locator,
            validation=validation,
        )

        with self._write_lock:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            self._assert_within_root(artifact_dir)

            if content_path.exists():
                self._verify_content(
                    content_path,
                    expected_sha256=digest,
                    expected_size=len(content),
                )
            else:
                self._atomic_write(content_path, content)

            if metadata_path.exists():
                metadata = self._load_metadata_file(metadata_path)
                self._verify_metadata_identity(metadata, digest, len(content), content_path)
                existing_origin_ids = {item.identity() for item in metadata.origins}
                if origin.identity() not in existing_origin_ids:
                    metadata.origins.append(origin)
                    self._atomic_write(
                        metadata_path,
                        metadata.model_dump_json(indent=2).encode("utf-8"),
                    )
            else:
                relative_path = content_path.relative_to(self.root).as_posix()
                metadata = ArtifactMetadata(
                    artifact_id=digest,
                    sha256=digest,
                    size_bytes=len(content),
                    relative_content_path=relative_path,
                    created_at=observed_at,
                    origins=[origin],
                )
                self._atomic_write(
                    metadata_path,
                    metadata.model_dump_json(indent=2).encode("utf-8"),
                )

        return ArtifactRef(
            artifact_id=digest,
            sha256=digest,
            original_name=original_name,
            media_type=media_type,
            size_bytes=len(content),
            local_path=content_path,
            source=source,
            retrieved_at=observed_at,
        )

    def get(self, artifact_id: str) -> ArtifactRef:
        """Load and fully verify a stored artifact by its SHA-256 identity."""

        self._validate_digest(artifact_id)
        artifact_dir = self._artifact_dir(artifact_id)
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ArtifactNotFoundError(f"artifact metadata not found: {artifact_id}")

        metadata = self._load_metadata_file(metadata_path)
        content_path = self._resolve_relative_content_path(metadata.relative_content_path)
        self._verify_metadata_identity(
            metadata,
            artifact_id,
            metadata.size_bytes,
            content_path,
        )
        self._verify_content(
            content_path,
            expected_sha256=metadata.sha256,
            expected_size=metadata.size_bytes,
        )
        origin = metadata.origins[0]
        return ArtifactRef(
            artifact_id=metadata.artifact_id,
            sha256=metadata.sha256,
            original_name=origin.original_name,
            media_type=origin.media_type,
            size_bytes=metadata.size_bytes,
            local_path=content_path,
            source=origin.source,
            retrieved_at=origin.retrieved_at,
        )

    def read_bytes(self, artifact: ArtifactRef) -> bytes:
        """Read bytes only after verifying the reference and current file contents."""

        if artifact.artifact_id != artifact.sha256:
            raise ArtifactIntegrityError("artifact reference ID and SHA-256 differ")
        expected_path = (self._artifact_dir(artifact.sha256) / "content").resolve()
        provided_path = artifact.local_path.resolve()
        self._assert_within_root(provided_path)
        if provided_path != expected_path:
            raise ArtifactIntegrityError("artifact reference points to an unexpected path")
        self._verify_content(
            provided_path,
            expected_sha256=artifact.sha256,
            expected_size=artifact.size_bytes,
        )
        return provided_path.read_bytes()

    def metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Return validated metadata for inspection and provenance reporting."""

        self._validate_digest(artifact_id)
        path = self._artifact_dir(artifact_id) / "metadata.json"
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact metadata not found: {artifact_id}")
        metadata = self._load_metadata_file(path)
        content_path = self._resolve_relative_content_path(metadata.relative_content_path)
        self._verify_metadata_identity(
            metadata,
            artifact_id,
            metadata.size_bytes,
            content_path,
        )
        return metadata

    def _artifact_dir(self, digest: str) -> Path:
        self._validate_digest(digest)
        path = self.content_root / digest[:2] / digest
        self._assert_within_root(path)
        return path

    def _resolve_relative_content_path(self, relative_path: str) -> Path:
        pure_path = PurePath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ArtifactIntegrityError("metadata contains an unsafe content path")
        resolved = (self.root / Path(relative_path)).resolve()
        self._assert_within_root(resolved)
        return resolved

    def _verify_metadata_identity(
        self,
        metadata: ArtifactMetadata,
        expected_sha256: str,
        expected_size: int,
        expected_content_path: Path,
    ) -> None:
        if metadata.artifact_id != expected_sha256 or metadata.sha256 != expected_sha256:
            raise ArtifactIntegrityError("artifact metadata hash identity mismatch")
        if metadata.size_bytes != expected_size:
            raise ArtifactIntegrityError("artifact metadata size mismatch")
        stored_path = self._resolve_relative_content_path(metadata.relative_content_path)
        if stored_path != expected_content_path.resolve():
            raise ArtifactIntegrityError("artifact metadata points to unexpected content")

    @staticmethod
    def _verify_content(path: Path, *, expected_sha256: str, expected_size: int) -> None:
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(f"artifact content is missing: {path}") from exc
        if len(content) != expected_size:
            raise ArtifactIntegrityError("artifact content size no longer matches metadata")
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ArtifactIntegrityError("artifact content SHA-256 verification failed")

    @staticmethod
    def _load_metadata_file(path: Path) -> ArtifactMetadata:
        try:
            return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise ArtifactIntegrityError(f"artifact metadata is invalid: {path}") from exc

    @staticmethod
    def _atomic_write(target: Path, payload: bytes) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _assert_within_root(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.root)
        except ValueError as exc:
            raise ArtifactIntegrityError(f"artifact path escapes store root: {path}") from exc

    @staticmethod
    def _validate_original_name(original_name: str) -> None:
        if not original_name or len(original_name) > 255 or "\x00" in original_name:
            raise UnsafeArtifactNameError("artifact original name is empty or invalid")
        if Path(original_name).name != original_name or any(
            separator in original_name for separator in ("/", "\\")
        ):
            raise UnsafeArtifactNameError("artifact original name must not contain a path")
        if original_name in {".", ".."}:
            raise UnsafeArtifactNameError("artifact original name is unsafe")

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ArtifactStoreError("artifact ID must be a lowercase SHA-256 value")

    @staticmethod
    def _validate_report(
        validation: AttachmentValidationResult | None,
        *,
        digest: str,
        size_bytes: int,
        original_name: str,
        media_type: str,
    ) -> None:
        if validation is None:
            return
        if (
            validation.sha256 != digest
            or validation.size_bytes != size_bytes
            or validation.original_name != original_name
            or validation.canonical_media_type != media_type
        ):
            raise ArtifactStoreError("attachment validation report does not match artifact")
