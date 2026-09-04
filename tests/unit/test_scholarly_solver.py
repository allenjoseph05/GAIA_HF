from __future__ import annotations

from pathlib import Path

import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.retrieval import PdfDocumentExtraction
from gaia_max.solvers import ScholarlyEvidenceError, ScholarlyEvidenceExtractor


def document(store: ArtifactStore, pages: tuple[str, ...]) -> PdfDocumentExtraction:
    page_records: list[dict[str, object]] = []
    for page_number, text in enumerate(pages, start=1):
        artifact = store.put_bytes(
            text.encode("utf-8"),
            original_name=f"pdf-page-{page_number}.txt",
            media_type="text/plain; charset=utf-8",
            source=ArtifactSource.DERIVED,
            source_locator=f"artifact:{'a' * 64}#page={page_number}",
        )
        page_records.append(
            {
                "page_number": page_number,
                "method": "pymupdf",
                "text_artifact_id": artifact.artifact_id,
                "character_count": len(text),
                "requires_ocr": False,
            }
        )
    return PdfDocumentExtraction.model_validate(
        {
            "source_artifact_id": "a" * 64,
            "page_count": len(pages),
            "pages": page_records,
            "fallback_used": False,
            "unresolved_ocr_pages": (),
        }
    )


def test_funding_requires_author_agency_award_and_relationship_in_same_passage(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    paper = document(
        store,
        (
            "A. Researcher is affiliated with Example University.",
            "Acknowledgments: A. Researcher was supported by NASA award "
            "NNX-SYNTHETIC-01. The authors thank the reviewers.",
        ),
    )

    result = ScholarlyEvidenceExtractor(store).funding_for_author(
        paper,
        author_aliases=("A. Researcher", "A.R."),
    )

    assert result.award_identifier == "NNX-SYNTHETIC-01"
    assert result.author_alias == "A. Researcher"
    assert result.agency_keyword == "NASA"
    assert result.passage.page_number == 2
    assert result.passage.text_artifact_id == paper.pages[1].text_artifact_id


def test_funding_does_not_transfer_another_authors_award_or_affiliation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    paper = document(
        store,
        (
            "A. Researcher works at the NASA Example Center. "
            "Acknowledgments: B. Other was supported by NASA award NNX-OTHER-99.",
        ),
    )

    with pytest.raises(ScholarlyEvidenceError, match="exactly one"):
        ScholarlyEvidenceExtractor(store).funding_for_author(
            paper,
            author_aliases=("A. Researcher",),
        )


def test_multiple_author_awards_are_ambiguous_without_adjudication(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    paper = document(
        store,
        (
            "Acknowledgments: A. Researcher was supported by NASA awards "
            "NNX-FIRST-01 and NNX-SECOND-02.",
        ),
    )

    with pytest.raises(ScholarlyEvidenceError, match="exactly one"):
        ScholarlyEvidenceExtractor(store).funding_for_author(
            paper,
            author_aliases=("A. Researcher",),
        )


def test_specimen_depository_is_distinguished_from_author_affiliation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    paper = document(
        store,
        (
            "Authors are affiliated with Hanoi Example University.",
            "The holotype and paratype specimens will eventually be deposited at "
            "Synthetic National Zoological Collection (SNZC).",
        ),
    )

    result = ScholarlyEvidenceExtractor(store).specimen_depository(paper)

    assert result.institution == "Synthetic National Zoological Collection (SNZC)"
    assert result.passage.page_number == 2
    assert "Hanoi" not in result.institution


def test_depository_requires_specimen_language_and_unique_institution(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    no_specimen = document(
        store,
        ("The dataset will be deposited at General Repository.",),
    )
    with pytest.raises(ScholarlyEvidenceError, match="exactly one"):
        ScholarlyEvidenceExtractor(store).specimen_depository(no_specimen)

    ambiguous = document(
        store,
        (
            "Specimens were deposited at First Museum. "
            "Other specimens were deposited at Second Museum.",
        ),
    )
    with pytest.raises(ScholarlyEvidenceError, match="exactly one"):
        ScholarlyEvidenceExtractor(store).specimen_depository(ambiguous)
