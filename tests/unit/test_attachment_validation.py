from __future__ import annotations

import hashlib
import io
import struct
import zlib
from collections.abc import Iterator
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from gaia_max.attachment_validation import (
    MAX_IMAGE_PIXELS,
    AttachmentKind,
    AttachmentRejectionReason,
    AttachmentValidationError,
    AttachmentValidator,
)


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def valid_png(*, width: int = 1, height: int = 1) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\x00\x00\x00" * width)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanline * height))
        + png_chunk(b"IEND", b"")
    )


def valid_jpeg() -> bytes:
    frame = b"\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    scan = b"\x01\x01\x00\x00\x3f\x00"
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(frame) + 2)
        + frame
        + b"\xff\xda"
        + struct.pack(">H", len(scan) + 2)
        + scan
        + b"\x00\xff\xd9"
    )


def valid_mp3() -> bytes:
    return b"\xff\xfb\x90\x64" + (b"\x00" * 64)


def xlsx_bytes(extra_members: dict[str, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        archive.writestr(
            "xl/workbook.xml",
            b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>',
        )
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    return output.getvalue()


def valid_cases() -> Iterator[object]:
    yield pytest.param(valid_png(), "board.png", "image/png", AttachmentKind.PNG, id="png")
    yield pytest.param(valid_jpeg(), "photo.jpeg", "image/jpeg", AttachmentKind.JPEG, id="jpeg")
    yield pytest.param(valid_mp3(), "audio.mp3", "audio/mpeg", AttachmentKind.MP3, id="mp3")
    yield pytest.param(
        b"print(42)\n",
        "answer.py",
        "text/x-python",
        AttachmentKind.PYTHON,
        id="python",
    )
    yield pytest.param(
        xlsx_bytes(),
        "sales.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        AttachmentKind.XLSX,
        id="xlsx",
    )


@pytest.mark.parametrize(("content", "name", "media_type", "expected_kind"), tuple(valid_cases()))
def test_supported_synthetic_attachments_pass(
    content: bytes,
    name: str,
    media_type: str,
    expected_kind: AttachmentKind,
) -> None:
    result = AttachmentValidator().validate(
        content,
        original_name=name,
        media_type=media_type,
    )

    assert result.kind is expected_kind
    assert result.size_bytes == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert "size_within_limit" in result.checks


def test_extension_and_media_type_normalization() -> None:
    result = AttachmentValidator().validate(
        b"print('ok')\n",
        original_name="PROGRAM.PY",
        media_type="Text/Plain; charset=utf-8",
    )

    assert result.kind is AttachmentKind.PYTHON
    assert result.extension == ".py"
    assert result.supplied_media_type == "text/plain"


def test_generic_binary_media_type_requires_and_records_strong_content_check() -> None:
    result = AttachmentValidator().validate(
        valid_mp3(),
        original_name="audio.mp3",
        media_type="application/octet-stream",
    )

    assert "generic_media_type_with_strong_content_validation" in result.checks


def assert_rejected(
    content: bytes,
    *,
    name: str,
    media_type: str | None,
    reason: AttachmentRejectionReason,
    validator: AttachmentValidator | None = None,
) -> None:
    with pytest.raises(AttachmentValidationError) as caught:
        (validator or AttachmentValidator()).validate(
            content,
            original_name=name,
            media_type=media_type,
        )
    assert caught.value.reason is reason
    assert str(caught.value).startswith(f"{reason.value}:")


def test_empty_and_oversized_attachments_have_distinct_reason_codes() -> None:
    assert_rejected(
        b"",
        name="empty.png",
        media_type="image/png",
        reason=AttachmentRejectionReason.EMPTY,
    )
    assert_rejected(
        b"four",
        name="large.py",
        media_type="text/x-python",
        reason=AttachmentRejectionReason.TOO_LARGE,
        validator=AttachmentValidator(max_bytes=3),
    )


@pytest.mark.parametrize("name", ["../board.png", "folder/board.png", "folder\\board.png", ""])
def test_unsafe_filenames_are_rejected(name: str) -> None:
    assert_rejected(
        valid_png(),
        name=name,
        media_type="image/png",
        reason=AttachmentRejectionReason.UNSAFE_FILENAME,
    )


def test_unsupported_extension_is_rejected() -> None:
    assert_rejected(
        b"plain text",
        name="attachment.txt",
        media_type="text/plain",
        reason=AttachmentRejectionReason.UNSUPPORTED_EXTENSION,
    )


def test_missing_and_mismatched_media_types_are_rejected() -> None:
    assert_rejected(
        valid_png(),
        name="board.png",
        media_type=None,
        reason=AttachmentRejectionReason.MISSING_MEDIA_TYPE,
    )
    assert_rejected(
        valid_png(),
        name="board.png",
        media_type="audio/mpeg",
        reason=AttachmentRejectionReason.MEDIA_TYPE_MISMATCH,
    )


@pytest.mark.parametrize(
    ("content", "media_type"),
    [
        (b"<!doctype html><title>404</title>", "image/png"),
        (valid_png(), "text/html; charset=utf-8"),
    ],
)
def test_html_error_responses_are_detected_from_bytes_or_media_type(
    content: bytes,
    media_type: str,
) -> None:
    assert_rejected(
        content,
        name="board.png",
        media_type=media_type,
        reason=AttachmentRejectionReason.HTML_ERROR_RESPONSE,
    )


def test_extension_and_actual_content_must_agree() -> None:
    assert_rejected(
        valid_png(),
        name="disguised.jpg",
        media_type="image/jpeg",
        reason=AttachmentRejectionReason.CONTENT_MISMATCH,
    )


@pytest.mark.parametrize(
    ("content", "name", "media_type", "reason"),
    [
        (
            b"\x89PNG\r\n\x1a\ntruncated",
            "bad.png",
            "image/png",
            AttachmentRejectionReason.MALFORMED_CONTENT,
        ),
        (
            b"\xff\xd8truncated",
            "bad.jpg",
            "image/jpeg",
            AttachmentRejectionReason.MALFORMED_CONTENT,
        ),
        (
            b"ID3\x04\x00\x00\x80\x00\x00\x00",
            "bad.mp3",
            "audio/mpeg",
            AttachmentRejectionReason.MALFORMED_CONTENT,
        ),
        (
            b"def broken(:\n",
            "bad.py",
            "text/x-python",
            AttachmentRejectionReason.MALFORMED_CONTENT,
        ),
        (
            b"PKnot-a-zip",
            "bad.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            AttachmentRejectionReason.MALFORMED_CONTENT,
        ),
    ],
)
def test_malformed_supported_formats_are_rejected(
    content: bytes,
    name: str,
    media_type: str,
    reason: AttachmentRejectionReason,
) -> None:
    assert_rejected(content, name=name, media_type=media_type, reason=reason)


def test_image_decompression_dimensions_are_bounded() -> None:
    oversized_header = struct.pack(">IIBBBBB", MAX_IMAGE_PIXELS + 1, 1, 8, 2, 0, 0, 0)
    oversized_png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", oversized_header)
        + png_chunk(b"IEND", b"")
    )
    assert_rejected(
        oversized_png,
        name="huge.png",
        media_type="image/png",
        reason=AttachmentRejectionReason.DECOMPRESSION_RISK,
    )


@pytest.mark.parametrize("unsafe_member", ["../outside.xml", "C:/outside.xml"])
def test_xlsx_archive_member_cannot_escape_extraction_root(unsafe_member: str) -> None:
    assert_rejected(
        xlsx_bytes({unsafe_member: b"unsafe"}),
        name="unsafe.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        reason=AttachmentRejectionReason.UNSAFE_ARCHIVE,
    )


def test_xlsx_archive_rejects_raw_windows_member_separator() -> None:
    with pytest.raises(AttachmentValidationError) as caught:
        AttachmentValidator._validate_zip_member_name("bad\\name.xml")

    assert caught.value.reason is AttachmentRejectionReason.UNSAFE_ARCHIVE
