"""Page-addressable PDF text, layout fallback, rendering, OCR, and exact context."""

from __future__ import annotations

import re
from collections.abc import Sequence
from io import BytesIO
from typing import Protocol, cast

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator
from pypdf import PdfReader

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactRef, ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN


class PdfExtractionError(RuntimeError):
    """PDF is invalid, unsafe, encrypted, inconsistent, or not extractable."""


class PdfTextBackend(Protocol):
    """Extract one text string per physical PDF page."""

    @property
    def name(self) -> str: ...

    def extract_pages(self, pdf_bytes: bytes) -> Sequence[str]: ...


class PdfPageRenderer(Protocol):
    """Render one zero-based physical page to PNG bytes."""

    def render_png(self, pdf_bytes: bytes, page_index: int) -> bytes: ...


class PdfOcrEngine(Protocol):
    """OCR a rendered PNG without receiving the rest of the document."""

    @property
    def name(self) -> str: ...

    def recognize(self, png_bytes: bytes, page_number: int) -> str: ...


class PyMuPdfTextBackend:
    name = "pymupdf"

    def extract_pages(self, pdf_bytes: bytes) -> tuple[str, ...]:
        try:
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
                if cast(bool, document.needs_pass):  # pyright: ignore[reportUnknownMemberType]
                    raise PdfExtractionError("encrypted PDF requires a password")
                texts: list[str] = []
                for page in document:
                    raw_text = cast(
                        object,
                        page.get_text(  # pyright: ignore[reportUnknownMemberType]
                            "text",
                            sort=True,
                        ),
                    )
                    if not isinstance(raw_text, str):
                        raise PdfExtractionError("PyMuPDF returned non-text page data")
                    texts.append(raw_text)
                return tuple(texts)
        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError("PyMuPDF could not parse the document") from exc


class PyPdfTextBackend:
    name = "pypdf"

    def extract_pages(self, pdf_bytes: bytes) -> tuple[str, ...]:
        try:
            reader = PdfReader(BytesIO(pdf_bytes), strict=True)
            if reader.is_encrypted:
                raise PdfExtractionError("encrypted PDF requires a password")
            return tuple(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError("pypdf could not parse the document") from exc


class PyMuPdfRenderer:
    def __init__(self, *, dpi: int = 180) -> None:
        if not 72 <= dpi <= 400:
            raise ValueError("PDF render DPI must be between 72 and 400")
        self._dpi = dpi

    def render_png(self, pdf_bytes: bytes, page_index: int) -> bytes:
        try:
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
                page_count = cast(
                    int,
                    document.page_count,  # pyright: ignore[reportUnknownMemberType]
                )
                if not 0 <= page_index < page_count:
                    raise PdfExtractionError("PDF render page is outside the document")
                pixmap = document[page_index].get_pixmap(  # pyright: ignore[reportUnknownMemberType]
                    dpi=self._dpi,
                    alpha=False,
                )
                return cast(bytes, pixmap.tobytes("png"))
        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError("PDF page rendering failed") from exc


class PdfPageExtraction(BaseModel):
    """One physical page and the immutable artifacts used to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    method: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    text_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rendered_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    character_count: int = Field(ge=0)
    requires_ocr: bool = False

    @model_validator(mode="after")
    def validate_artifacts(self) -> PdfPageExtraction:
        if self.character_count > 0 and self.text_artifact_id is None:
            raise ValueError("non-empty PDF page requires a text artifact")
        if self.requires_ocr and self.rendered_artifact_id is None:
            raise ValueError("OCR-required PDF page needs a rendered artifact")
        if self.requires_ocr and self.text_artifact_id is not None:
            raise ValueError("unresolved OCR page cannot contain extracted text")
        return self


class PdfDocumentExtraction(BaseModel):
    """Complete page inventory for one immutable source PDF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    page_count: int = Field(ge=1)
    pages: tuple[PdfPageExtraction, ...] = Field(min_length=1)
    fallback_used: bool
    unresolved_ocr_pages: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_pages(self) -> PdfDocumentExtraction:
        expected = tuple(range(1, self.page_count + 1))
        if tuple(page.page_number for page in self.pages) != expected:
            raise ValueError("PDF pages must be complete, ordered, and one-based")
        unresolved = tuple(page.page_number for page in self.pages if page.requires_ocr)
        if self.unresolved_ocr_pages != unresolved:
            raise ValueError("unresolved OCR inventory must match page records")
        return self


class PdfContextWindow(BaseModel):
    """Exact literal match and bounded same-page context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    query: StrictStr = Field(min_length=1)
    matched_text: StrictStr = Field(min_length=1)
    context: StrictStr = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text_artifact_id: str = Field(pattern=SHA256_PATTERN)


class PdfExtractionLadder:
    """Prefer native text, then layout text, then rendered OCR per weak page."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        primary: PdfTextBackend | None = None,
        secondary: PdfTextBackend | None = None,
        renderer: PdfPageRenderer | None = None,
        ocr: PdfOcrEngine | None = None,
        min_page_characters: int = 20,
        max_pages: int = 1000,
    ) -> None:
        if not 0 <= min_page_characters <= 10_000:
            raise ValueError("minimum PDF page characters must be between 0 and 10000")
        if not 1 <= max_pages <= 10_000:
            raise ValueError("maximum PDF pages must be between 1 and 10000")
        self._store = store
        self._primary = primary or PyMuPdfTextBackend()
        self._secondary = secondary or PyPdfTextBackend()
        self._renderer = renderer or PyMuPdfRenderer()
        self._ocr = ocr
        self._minimum = min_page_characters
        self._max_pages = max_pages

    def extract(self, artifact: ArtifactRef) -> PdfDocumentExtraction:
        pdf_bytes = self._store.read_bytes(artifact)
        if not pdf_bytes.startswith(b"%PDF-"):
            raise PdfExtractionError("artifact does not have a PDF signature")
        primary_pages = self._validated_pages(self._primary, pdf_bytes)
        if not primary_pages or len(primary_pages) > self._max_pages:
            raise PdfExtractionError("PDF page count is empty or exceeds configured maximum")
        weak = [index for index, text in enumerate(primary_pages) if not self._enough(text)]
        secondary_pages: tuple[str, ...] | None = None
        if weak:
            secondary_pages = self._validated_pages(self._secondary, pdf_bytes)
            if len(secondary_pages) != len(primary_pages):
                raise PdfExtractionError("PDF text backends disagree on page count")

        pages: list[PdfPageExtraction] = []
        fallback_used = False
        for page_index, primary_text in enumerate(primary_pages):
            page_number = page_index + 1
            normalized = _normalize_pdf_text(primary_text)
            method = self._primary.name
            rendered_id: str | None = None
            requires_ocr = False
            if not self._enough(normalized):
                fallback_used = True
                if secondary_pages is None:
                    raise RuntimeError("missing PDF secondary extraction state")
                secondary_text = _normalize_pdf_text(secondary_pages[page_index])
                if len(secondary_text) > len(normalized):
                    normalized = secondary_text
                    method = self._secondary.name
            if not self._enough(normalized):
                png = self._renderer.render_png(pdf_bytes, page_index)
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise PdfExtractionError("PDF renderer did not return PNG bytes")
                rendered = self._store.put_bytes(
                    png,
                    original_name=f"pdf-page-{page_number}.png",
                    media_type="image/png",
                    source=ArtifactSource.DERIVED,
                    source_locator=f"artifact:{artifact.artifact_id}#page={page_number}",
                )
                rendered_id = rendered.artifact_id
                if self._ocr is not None:
                    normalized = _normalize_pdf_text(self._ocr.recognize(png, page_number))
                    method = f"ocr_{self._ocr.name}"
                if not self._enough(normalized):
                    requires_ocr = True
                    method = "rendered_ocr_required"

            text_id: str | None = None
            if normalized:
                text_artifact = self._store.put_bytes(
                    normalized.encode("utf-8"),
                    original_name=f"pdf-page-{page_number}.txt",
                    media_type="text/plain; charset=utf-8",
                    source=ArtifactSource.DERIVED,
                    source_locator=f"artifact:{artifact.artifact_id}#page={page_number}",
                )
                text_id = text_artifact.artifact_id
            pages.append(
                PdfPageExtraction(
                    page_number=page_number,
                    method=method,
                    text_artifact_id=text_id,
                    rendered_artifact_id=rendered_id,
                    character_count=len(normalized),
                    requires_ocr=requires_ocr,
                )
            )
        return PdfDocumentExtraction(
            source_artifact_id=artifact.artifact_id,
            page_count=len(pages),
            pages=tuple(pages),
            fallback_used=fallback_used,
            unresolved_ocr_pages=tuple(page.page_number for page in pages if page.requires_ocr),
        )

    def exact_contexts(
        self,
        extraction: PdfDocumentExtraction,
        query: str,
        *,
        case_sensitive: bool = False,
        context_characters: int = 250,
    ) -> tuple[PdfContextWindow, ...]:
        """Find every literal occurrence with bounded context on its physical page."""

        if not query or query != query.strip():
            raise ValueError("PDF context query must be non-empty and trimmed")
        if not 0 <= context_characters <= 5000:
            raise ValueError("PDF context size must be between 0 and 5000")
        needle = query if case_sensitive else query.casefold()
        windows: list[PdfContextWindow] = []
        for page in extraction.pages:
            if page.text_artifact_id is None:
                continue
            text = self._store.read_bytes(self._store.get(page.text_artifact_id)).decode("utf-8")
            haystack = text if case_sensitive else text.casefold()
            offset = 0
            while True:
                start = haystack.find(needle, offset)
                if start < 0:
                    break
                end = start + len(query)
                context_start = max(0, start - context_characters)
                context_end = min(len(text), end + context_characters)
                windows.append(
                    PdfContextWindow(
                        page_number=page.page_number,
                        query=query,
                        matched_text=text[start:end],
                        context=text[context_start:context_end],
                        start_offset=start,
                        end_offset=end,
                        text_artifact_id=page.text_artifact_id,
                    )
                )
                offset = end
        return tuple(windows)

    @staticmethod
    def _validated_pages(backend: PdfTextBackend, pdf_bytes: bytes) -> tuple[str, ...]:
        raw_pages = cast(Sequence[object], backend.extract_pages(pdf_bytes))
        if any(not isinstance(text, str) for text in raw_pages):
            raise PdfExtractionError(f"PDF backend {backend.name} returned non-text page data")
        pages = cast(tuple[str, ...], tuple(raw_pages))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", backend.name):
            raise PdfExtractionError("PDF backend name is not a safe identifier")
        return pages

    def _enough(self, text: str) -> bool:
        return len(_normalize_pdf_text(text)) >= self._minimum


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


__all__ = [
    "PdfContextWindow",
    "PdfDocumentExtraction",
    "PdfExtractionError",
    "PdfExtractionLadder",
    "PdfOcrEngine",
    "PdfPageExtraction",
    "PdfPageRenderer",
    "PdfTextBackend",
    "PyMuPdfRenderer",
    "PyMuPdfTextBackend",
    "PyPdfTextBackend",
]
