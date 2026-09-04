"""Defensive validation for untrusted downloaded attachments."""

from __future__ import annotations

import ast
import hashlib
import io
import struct
import tokenize
import zlib
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
MAX_XLSX_ENTRIES = 10_000
MAX_XLSX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_XLSX_COMPRESSION_RATIO = 1_000
SUPPORTED_ATTACHMENT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".mp3", ".py", ".xlsx"})


class AttachmentKind(StrEnum):
    """Attachment formats accepted by the current GAIA solver system."""

    PNG = "png"
    JPEG = "jpeg"
    MP3 = "mp3"
    PYTHON = "python"
    XLSX = "xlsx"


class AttachmentRejectionReason(StrEnum):
    """Stable machine-readable reasons for rejecting an attachment."""

    EMPTY = "empty"
    TOO_LARGE = "too_large"
    UNSAFE_FILENAME = "unsafe_filename"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    MISSING_MEDIA_TYPE = "missing_media_type"
    MEDIA_TYPE_MISMATCH = "media_type_mismatch"
    HTML_ERROR_RESPONSE = "html_error_response"
    CONTENT_MISMATCH = "content_mismatch"
    MALFORMED_CONTENT = "malformed_content"
    DECOMPRESSION_RISK = "decompression_risk"
    UNSAFE_ARCHIVE = "unsafe_archive"


class AttachmentValidationError(ValueError):
    """An attachment failed validation with a stable reason code."""

    def __init__(self, reason: AttachmentRejectionReason, message: str) -> None:
        self.reason = reason
        super().__init__(f"{reason.value}: {message}")


class AttachmentValidationResult(BaseModel):
    """Validated attachment facts safe to record as provenance metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_name: str
    kind: AttachmentKind
    extension: str
    supplied_media_type: str
    canonical_media_type: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: tuple[str, ...]


_EXTENSION_KIND = {
    ".png": AttachmentKind.PNG,
    ".jpg": AttachmentKind.JPEG,
    ".jpeg": AttachmentKind.JPEG,
    ".mp3": AttachmentKind.MP3,
    ".py": AttachmentKind.PYTHON,
    ".xlsx": AttachmentKind.XLSX,
}

_CANONICAL_MEDIA_TYPE = {
    AttachmentKind.PNG: "image/png",
    AttachmentKind.JPEG: "image/jpeg",
    AttachmentKind.MP3: "audio/mpeg",
    AttachmentKind.PYTHON: "text/x-python",
    AttachmentKind.XLSX: (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
}

_ALLOWED_MEDIA_TYPES = {
    AttachmentKind.PNG: {"image/png"},
    AttachmentKind.JPEG: {"image/jpeg", "image/jpg"},
    AttachmentKind.MP3: {"audio/mpeg", "audio/mp3"},
    AttachmentKind.PYTHON: {
        "text/x-python",
        "text/plain",
        "application/x-python-code",
    },
    AttachmentKind.XLSX: {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
    },
}

_GENERIC_MEDIA_TYPES = {"application/octet-stream", "binary/octet-stream"}
_HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}


class AttachmentValidator:
    """Validate filename, media type, size, signature, and internal structure."""

    def __init__(self, *, max_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes

    def validate(
        self,
        content: bytes,
        *,
        original_name: str,
        media_type: str | None,
    ) -> AttachmentValidationResult:
        """Return immutable validated facts or raise a reason-coded error."""

        self._validate_filename(original_name)
        if not content:
            self._reject(AttachmentRejectionReason.EMPTY, "attachment contains no bytes")
        if len(content) > self.max_bytes:
            self._reject(
                AttachmentRejectionReason.TOO_LARGE,
                f"attachment is {len(content)} bytes; maximum is {self.max_bytes}",
            )

        extension = Path(original_name).suffix.lower()
        kind = _EXTENSION_KIND.get(extension)
        if kind is None:
            self._reject(
                AttachmentRejectionReason.UNSUPPORTED_EXTENSION,
                f"unsupported attachment extension: {extension or '<none>'}",
            )

        normalized_media_type = self._normalize_media_type(media_type)
        if self._looks_like_html(content) or normalized_media_type in _HTML_MEDIA_TYPES:
            self._reject(
                AttachmentRejectionReason.HTML_ERROR_RESPONSE,
                "received an HTML page instead of an attachment",
            )
        self._validate_media_type(kind, normalized_media_type)
        checks = self._validate_content(kind, content)

        media_check = (
            "generic_media_type_with_strong_content_validation"
            if normalized_media_type in _GENERIC_MEDIA_TYPES
            else "media_type_matches_extension"
        )
        return AttachmentValidationResult(
            original_name=original_name,
            kind=kind,
            extension=extension,
            supplied_media_type=normalized_media_type,
            canonical_media_type=_CANONICAL_MEDIA_TYPE[kind],
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            checks=("safe_filename", "size_within_limit", media_check, *checks),
        )

    @classmethod
    def _validate_filename(cls, original_name: str) -> None:
        if (
            not original_name
            or len(original_name) > 255
            or "\x00" in original_name
            or Path(original_name).name != original_name
            or any(separator in original_name for separator in ("/", "\\"))
            or original_name in {".", ".."}
        ):
            cls._reject(
                AttachmentRejectionReason.UNSAFE_FILENAME,
                "attachment name must be a safe basename",
            )

    @classmethod
    def _normalize_media_type(cls, media_type: str | None) -> str:
        if media_type is None or not media_type.strip():
            cls._reject(
                AttachmentRejectionReason.MISSING_MEDIA_TYPE,
                "attachment response did not include a media type",
            )
        return media_type.partition(";")[0].strip().lower()

    @classmethod
    def _validate_media_type(cls, kind: AttachmentKind, media_type: str) -> None:
        if media_type not in _ALLOWED_MEDIA_TYPES[kind] | _GENERIC_MEDIA_TYPES:
            cls._reject(
                AttachmentRejectionReason.MEDIA_TYPE_MISMATCH,
                f"{media_type!r} is inconsistent with a {kind.value} attachment",
            )

    @classmethod
    def _validate_content(cls, kind: AttachmentKind, content: bytes) -> tuple[str, ...]:
        validators = {
            AttachmentKind.PNG: cls._validate_png,
            AttachmentKind.JPEG: cls._validate_jpeg,
            AttachmentKind.MP3: cls._validate_mp3,
            AttachmentKind.PYTHON: cls._validate_python,
            AttachmentKind.XLSX: cls._validate_xlsx,
        }
        return validators[kind](content)

    @classmethod
    def _validate_png(cls, content: bytes) -> tuple[str, ...]:
        signature = b"\x89PNG\r\n\x1a\n"
        if not content.startswith(signature):
            cls._content_mismatch("PNG signature is missing")

        offset = len(signature)
        chunk_number = 0
        saw_iend = False
        while offset < len(content):
            if offset + 12 > len(content):
                cls._malformed("PNG has a truncated chunk")
            chunk_length = struct.unpack(">I", content[offset : offset + 4])[0]
            chunk_type = content[offset + 4 : offset + 8]
            chunk_end = offset + 12 + chunk_length
            if chunk_end > len(content):
                cls._malformed("PNG chunk exceeds the file boundary")
            chunk_data = content[offset + 8 : offset + 8 + chunk_length]
            expected_crc = struct.unpack(">I", content[offset + 8 + chunk_length : chunk_end])[0]
            actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                cls._malformed("PNG chunk CRC validation failed")

            if chunk_number == 0:
                if chunk_type != b"IHDR" or chunk_length != 13:
                    cls._malformed("PNG must begin with a 13-byte IHDR chunk")
                width, height = struct.unpack(">II", chunk_data[:8])
                if width == 0 or height == 0:
                    cls._malformed("PNG dimensions must be positive")
                cls._validate_pixel_count(width, height, "PNG")
            if chunk_type == b"IEND":
                if chunk_length != 0 or chunk_end != len(content):
                    cls._malformed("PNG IEND chunk is invalid or not final")
                saw_iend = True
                break
            offset = chunk_end
            chunk_number += 1

        if not saw_iend:
            cls._malformed("PNG IEND chunk is missing")
        return ("png_signature", "png_chunk_crc", "png_structure")

    @classmethod
    def _validate_jpeg(cls, content: bytes) -> tuple[str, ...]:
        if not content.startswith(b"\xff\xd8"):
            cls._content_mismatch("JPEG start-of-image signature is missing")
        if not content.endswith(b"\xff\xd9"):
            cls._malformed("JPEG end-of-image marker is missing")

        offset = 2
        saw_frame = False
        start_of_frame_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while offset < len(content) - 2:
            if content[offset] != 0xFF:
                cls._malformed("JPEG marker structure is invalid")
            while offset < len(content) and content[offset] == 0xFF:
                offset += 1
            if offset >= len(content):
                cls._malformed("JPEG marker is truncated")
            marker = content[offset]
            offset += 1
            if marker == 0xD9:
                break
            if marker in {0x01, *range(0xD0, 0xD8)}:
                continue
            if offset + 2 > len(content):
                cls._malformed("JPEG segment length is truncated")
            segment_length = struct.unpack(">H", content[offset : offset + 2])[0]
            if segment_length < 2 or offset + segment_length > len(content):
                cls._malformed("JPEG segment exceeds the file boundary")
            segment = content[offset + 2 : offset + segment_length]
            if marker in start_of_frame_markers:
                if len(segment) < 6:
                    cls._malformed("JPEG frame header is truncated")
                height, width = struct.unpack(">HH", segment[1:5])
                if width == 0 or height == 0:
                    cls._malformed("JPEG dimensions must be positive")
                cls._validate_pixel_count(width, height, "JPEG")
                saw_frame = True
            if marker == 0xDA:
                if not saw_frame:
                    cls._malformed("JPEG scan appears before a frame header")
                break
            offset += segment_length

        if not saw_frame:
            cls._malformed("JPEG frame header is missing")
        return ("jpeg_signature", "jpeg_frame_structure")

    @classmethod
    def _validate_mp3(cls, content: bytes) -> tuple[str, ...]:
        offset = 0
        if content.startswith(b"ID3"):
            if len(content) < 10 or any(value & 0x80 for value in content[6:10]):
                cls._malformed("MP3 ID3 header is invalid")
            tag_size = (
                (content[6] << 21)
                | (content[7] << 14)
                | (content[8] << 7)
                | content[9]
            )
            offset = 10 + tag_size + (10 if content[5] & 0x10 else 0)
            if offset > len(content):
                cls._malformed("MP3 ID3 tag exceeds the file boundary")

        search_end = min(len(content) - 3, offset + 65_536)
        for index in range(offset, max(offset, search_end)):
            header = int.from_bytes(content[index : index + 4], "big")
            if cls._is_valid_mp3_frame_header(header):
                return ("mp3_frame_sync", "mp3_header_fields")
        cls._content_mismatch("no valid MPEG audio frame was found")

    @staticmethod
    def _is_valid_mp3_frame_header(header: int) -> bool:
        return (
            header >> 21 == 0x7FF
            and (header >> 19) & 0b11 != 0b01
            and (header >> 17) & 0b11 != 0b00
            and (header >> 12) & 0b1111 not in {0, 0b1111}
            and (header >> 10) & 0b11 != 0b11
        )

    @classmethod
    def _validate_python(cls, content: bytes) -> tuple[str, ...]:
        if b"\x00" in content:
            cls._content_mismatch("Python source contains binary NUL bytes")
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(content).readline)
            source = content.decode(encoding)
            ast.parse(source, filename="<attachment>")
        except (LookupError, SyntaxError, UnicodeDecodeError) as exc:
            cls._malformed(f"Python source cannot be decoded and parsed: {exc}")
        return ("python_text_encoding", "python_ast_parse")

    @classmethod
    def _validate_xlsx(cls, content: bytes) -> tuple[str, ...]:
        if not content.startswith(b"PK"):
            cls._content_mismatch("XLSX ZIP signature is missing")
        try:
            with ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > MAX_XLSX_ENTRIES:
                    cls._decompression_risk("XLSX contains too many ZIP entries")
                names = {member.filename for member in members}
                if len(names) != len(members):
                    cls._reject(
                        AttachmentRejectionReason.UNSAFE_ARCHIVE,
                        "XLSX contains duplicate ZIP member names",
                    )
                required = {"[Content_Types].xml", "xl/workbook.xml"}
                if not required.issubset(names):
                    cls._content_mismatch("ZIP file is not an XLSX workbook")

                expanded_size = 0
                compressed_size = 0
                for member in members:
                    cls._validate_zip_member_name(member.filename)
                    if member.flag_bits & 0x1:
                        cls._reject(
                            AttachmentRejectionReason.UNSAFE_ARCHIVE,
                            "encrypted XLSX entries are not supported",
                        )
                    expanded_size += member.file_size
                    compressed_size += member.compress_size
                if expanded_size > MAX_XLSX_EXPANDED_BYTES:
                    cls._decompression_risk("XLSX expands beyond the safe byte limit")
                if (
                    compressed_size > 0
                    and expanded_size / compressed_size > MAX_XLSX_COMPRESSION_RATIO
                ):
                    cls._decompression_risk("XLSX compression ratio is suspicious")
                bad_member = archive.testzip()
                if bad_member is not None:
                    cls._malformed(f"XLSX ZIP member failed CRC validation: {bad_member}")
        except BadZipFile as exc:
            cls._malformed(f"XLSX ZIP structure is invalid: {exc}")
        return ("xlsx_zip_signature", "xlsx_required_parts", "xlsx_crc")

    @classmethod
    def _validate_zip_member_name(cls, member_name: str) -> None:
        path = PurePosixPath(member_name)
        windows_drive = bool(path.parts and ":" in path.parts[0])
        if (
            not member_name
            or path.is_absolute()
            or windows_drive
            or ".." in path.parts
            or "\\" in member_name
        ):
            cls._reject(
                AttachmentRejectionReason.UNSAFE_ARCHIVE,
                f"XLSX contains an unsafe ZIP member: {member_name}",
            )

    @classmethod
    def _validate_pixel_count(cls, width: int, height: int, label: str) -> None:
        if width * height > MAX_IMAGE_PIXELS:
            cls._decompression_risk(f"{label} dimensions exceed the safe pixel limit")

    @staticmethod
    def _looks_like_html(content: bytes) -> bool:
        prefix = content[:1024].lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
        return prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))

    @classmethod
    def _content_mismatch(cls, message: str) -> NoReturn:
        cls._reject(AttachmentRejectionReason.CONTENT_MISMATCH, message)

    @classmethod
    def _malformed(cls, message: str) -> NoReturn:
        cls._reject(AttachmentRejectionReason.MALFORMED_CONTENT, message)

    @classmethod
    def _decompression_risk(cls, message: str) -> NoReturn:
        cls._reject(AttachmentRejectionReason.DECOMPRESSION_RISK, message)

    @staticmethod
    def _reject(reason: AttachmentRejectionReason, message: str) -> NoReturn:
        raise AttachmentValidationError(reason, message)
