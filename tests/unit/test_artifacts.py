from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaia_max.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    UnsafeArtifactNameError,
)
from gaia_max.attachment_validation import AttachmentValidator
from gaia_max.domain import ArtifactSource

NOW = datetime(2026, 8, 27, tzinfo=UTC)


def put_fixture(store: ArtifactStore, content: bytes = b"fixture-content"):
    return store.put_bytes(
        content,
        original_name="fixture.bin",
        media_type="application/octet-stream",
        source=ArtifactSource.FIXTURE,
        retrieved_at=NOW,
        task_id="fixture-task",
        source_locator="fixture://local",
    )


def test_put_stores_content_in_sha256_sharded_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    artifact = put_fixture(store)

    assert artifact.local_path.is_file()
    assert artifact.local_path.name == "content"
    assert artifact.local_path.parent.name == artifact.sha256
    assert artifact.local_path.parent.parent.name == artifact.sha256[:2]
    assert store.read_bytes(artifact) == b"fixture-content"


def test_identical_bytes_deduplicate_and_record_distinct_origins(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = put_fixture(store)

    second = store.put_bytes(
        b"fixture-content",
        original_name="same-bytes.dat",
        media_type="application/octet-stream",
        source=ArtifactSource.DERIVED,
        retrieved_at=NOW,
        task_id="other-task",
        source_locator="derived://other",
    )
    metadata = store.metadata(first.artifact_id)

    assert first.artifact_id == second.artifact_id
    assert first.local_path == second.local_path
    assert len(metadata.origins) == 2


def test_repeated_same_origin_is_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = put_fixture(store)
    second = put_fixture(store)

    assert first.artifact_id == second.artifact_id
    assert len(store.metadata(first.artifact_id).origins) == 1


def test_different_bytes_have_different_artifact_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    first = put_fixture(store, b"first")
    second = put_fixture(store, b"second")

    assert first.artifact_id != second.artifact_id
    assert first.local_path != second.local_path


@pytest.mark.parametrize(
    "unsafe_name",
    ["../secret.txt", "folder/file.txt", "folder\\file.txt", "..", "", "bad\x00name"],
)
def test_unsafe_original_names_are_rejected(tmp_path: Path, unsafe_name: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(UnsafeArtifactNameError):
        store.put_bytes(
            b"content",
            original_name=unsafe_name,
            media_type="text/plain",
            source=ArtifactSource.FIXTURE,
        )


def test_empty_content_and_media_type_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactStoreError, match="empty artifact"):
        store.put_bytes(
            b"",
            original_name="empty.bin",
            media_type="application/octet-stream",
            source=ArtifactSource.FIXTURE,
        )
    with pytest.raises(ArtifactStoreError, match="media type"):
        store.put_bytes(
            b"content",
            original_name="content.bin",
            media_type=" ",
            source=ArtifactSource.FIXTURE,
        )


def test_tampered_content_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = put_fixture(store)
    artifact.local_path.write_bytes(b"tampered-content")

    with pytest.raises(ArtifactIntegrityError):
        store.read_bytes(artifact)


def test_tampered_metadata_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = put_fixture(store)
    metadata_path = artifact.local_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        store.get(artifact.artifact_id)


def test_read_rejects_reference_to_unexpected_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = put_fixture(store)
    other_path = store.root / "other-content"
    other_path.write_bytes(b"fixture-content")
    changed_reference = artifact.model_copy(update={"local_path": other_path})

    with pytest.raises(ArtifactIntegrityError, match="unexpected path"):
        store.read_bytes(changed_reference)


def test_missing_artifact_has_specific_error(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactNotFoundError):
        store.get("a" * 64)


def test_atomic_writes_leave_no_temporary_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = put_fixture(store)

    temporary_files = list(artifact.local_path.parent.glob("*.tmp"))

    assert temporary_files == []


def test_store_rejects_validation_report_for_different_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    validation = AttachmentValidator().validate(
        b"print('first')\n",
        original_name="fixture.py",
        media_type="text/x-python",
    )

    with pytest.raises(ArtifactStoreError, match="validation report does not match"):
        store.put_bytes(
            b"print('different')\n",
            original_name="fixture.py",
            media_type="text/x-python",
            source=ArtifactSource.FIXTURE,
            validation=validation,
        )
