from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactRef, ArtifactSource
from gaia_max.retrieval import PdfExtractionError, PdfExtractionLadder

PNG = b"\x89PNG\r\n\x1a\nsynthetic-render"


class TextBackend:
    def __init__(self, name: str, pages: tuple[str, ...]) -> None:
        self.name = name
        self.pages = pages
        self.calls = 0

    def extract_pages(self, pdf_bytes: bytes) -> tuple[str, ...]:
        self.calls += 1
        assert pdf_bytes.startswith(b"%PDF-")
        return self.pages


class Renderer:
    def __init__(self) -> None:
        self.pages: list[int] = []

    def render_png(self, pdf_bytes: bytes, page_index: int) -> bytes:
        assert pdf_bytes.startswith(b"%PDF-")
        self.pages.append(page_index)
        return PNG


class Ocr:
    name = "synthetic"

    def recognize(self, png_bytes: bytes, page_number: int) -> str:
        assert png_bytes == PNG
        return f"Scanned deposition statement on physical page {page_number}."


def source_artifact(store: ArtifactStore, content: bytes) -> ArtifactRef:
    return store.put_bytes(
        content,
        original_name="paper.pdf",
        media_type="application/pdf",
        source=ArtifactSource.WEB_RETRIEVAL,
        source_locator="https://example.org/paper.pdf",
    )


def make_real_pdf() -> bytes:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text(  # pyright: ignore[reportUnknownMemberType]
        (72, 72),
        "Acknowledgments: this work was supported by award NNX-SYNTHETIC-01.",
    )
    second = document.new_page()
    second.insert_text(  # pyright: ignore[reportUnknownMemberType]
        (72, 72),
        "Specimens will be deposited at the Synthetic National Collection.",
    )
    content = document.tobytes()  # pyright: ignore[reportUnknownMemberType]
    document.close()
    return content


def test_real_pymupdf_primary_extraction_is_page_addressable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = source_artifact(store, make_real_pdf())
    ladder = PdfExtractionLadder(store=store, min_page_characters=20)

    extraction = ladder.extract(artifact)
    contexts = ladder.exact_contexts(extraction, "NNX-SYNTHETIC-01")

    assert extraction.page_count == 2
    assert extraction.fallback_used is False
    assert [page.method for page in extraction.pages] == ["pymupdf", "pymupdf"]
    assert len(contexts) == 1
    assert contexts[0].page_number == 1
    assert "Acknowledgments" in contexts[0].context


def test_layout_fallback_replaces_only_weak_pages(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = source_artifact(store, b"%PDF-1.7 synthetic")
    primary = TextBackend("primary", ("Readable first page with enough content.", "x"))
    secondary = TextBackend(
        "layout",
        (
            "Different first page that must not replace primary.",
            "Layout-sensitive table row with complete scholarly evidence.",
        ),
    )
    ladder = PdfExtractionLadder(
        store=store,
        primary=primary,
        secondary=secondary,
        renderer=Renderer(),
        min_page_characters=20,
    )

    extraction = ladder.extract(artifact)

    assert [page.method for page in extraction.pages] == ["primary", "layout"]
    assert extraction.fallback_used is True
    assert extraction.unresolved_ocr_pages == ()
    second_text_id = extraction.pages[1].text_artifact_id
    assert second_text_id is not None
    assert b"Layout-sensitive" in store.read_bytes(store.get(second_text_id))


def test_scanned_page_is_rendered_and_ocr_text_keeps_page_provenance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = source_artifact(store, b"%PDF-1.7 scanned")
    renderer = Renderer()
    ladder = PdfExtractionLadder(
        store=store,
        primary=TextBackend("primary", ("",)),
        secondary=TextBackend("secondary", ("",)),
        renderer=renderer,
        ocr=Ocr(),
        min_page_characters=20,
    )

    extraction = ladder.extract(artifact)
    contexts = ladder.exact_contexts(extraction, "deposition statement")

    page = extraction.pages[0]
    assert renderer.pages == [0]
    assert page.method == "ocr_synthetic"
    assert page.rendered_artifact_id is not None
    assert page.text_artifact_id is not None
    assert page.requires_ocr is False
    assert contexts[0].page_number == 1


def test_scanned_page_without_ocr_is_explicitly_unresolved(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = source_artifact(store, b"%PDF-1.7 scanned")
    ladder = PdfExtractionLadder(
        store=store,
        primary=TextBackend("primary", ("",)),
        secondary=TextBackend("secondary", ("",)),
        renderer=Renderer(),
        min_page_characters=20,
    )

    extraction = ladder.extract(artifact)

    assert extraction.unresolved_ocr_pages == (1,)
    assert extraction.pages[0].method == "rendered_ocr_required"
    assert extraction.pages[0].text_artifact_id is None
    assert ladder.exact_contexts(extraction, "anything") == ()


def test_backend_page_count_disagreement_and_non_pdf_fail_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    pdf = source_artifact(store, b"%PDF-1.7 synthetic")
    ladder = PdfExtractionLadder(
        store=store,
        primary=TextBackend("primary", ("",)),
        secondary=TextBackend("secondary", ("one", "two")),
        renderer=Renderer(),
    )
    with pytest.raises(PdfExtractionError, match="page count"):
        ladder.extract(pdf)

    not_pdf = source_artifact(store, b"not actually a PDF")
    with pytest.raises(PdfExtractionError, match="signature"):
        ladder.extract(not_pdf)


def test_exact_context_is_literal_case_configurable_and_bounded(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = source_artifact(store, b"%PDF-1.7 synthetic")
    ladder = PdfExtractionLadder(
        store=store,
        primary=TextBackend("primary", ("Before AWARD-42 after. Then award-42 again.",)),
        secondary=TextBackend("secondary", ("",)),
        renderer=Renderer(),
        min_page_characters=1,
    )
    extraction = ladder.extract(artifact)

    insensitive = ladder.exact_contexts(extraction, "award-42", context_characters=7)
    sensitive = ladder.exact_contexts(
        extraction,
        "award-42",
        case_sensitive=True,
        context_characters=7,
    )

    assert len(insensitive) == 2
    assert len(sensitive) == 1
    assert sensitive[0].matched_text == "award-42"
    assert len(sensitive[0].context) <= len("award-42") + 14
