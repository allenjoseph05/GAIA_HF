"""Exact scholarly relationship extraction from page-addressable PDF evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from gaia_max.artifacts import ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN
from gaia_max.retrieval.pdf import PdfDocumentExtraction

_RELATION = re.compile(r"\b(?:support(?:ed)?|fund(?:ed|ing)?|grant|award|contract)\b", re.I)
_ACKNOWLEDGMENT = re.compile(r"\backnowledg(?:e?ment|es|ed|ing)s?\b", re.I)
_DEFAULT_AWARD = re.compile(
    r"\b(?:NNX[0-9A-Z-]{5,}|NAG[0-9A-Z-]{3,}|80NSSC[0-9A-Z-]{4,}|"
    r"[A-Z]{2,10}[- ]?\d[A-Z0-9-]{3,})\b"
)
_SPECIMEN = re.compile(r"\b(?:specimens?|types?|holotypes?|paratypes?|vouchers?)\b", re.I)
_DEPOSITION = re.compile(
    r"\b(?:will\s+(?:eventually\s+)?be\s+deposited|"
    r"(?:are|were|is|was|have been)\s+(?:eventually\s+)?deposited|"
    r"deposited)\s+(?:in|at|with|to)\s+(?:the\s+)?"
    r"(?P<institution>[A-Z][^.;\n]{3,240}?)"
    r"(?=\s+(?:after|upon|once|when)\b|[.;\n]|$)",
    re.I,
)


class ScholarlyEvidenceError(ValueError):
    """A primary-paper relationship is absent or ambiguous."""


class ScholarlyPassage(BaseModel):
    """Exact page-level passage supporting one extracted relationship."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    text_artifact_id: str = Field(pattern=SHA256_PATTERN)
    text: StrictStr = Field(min_length=1, max_length=4000)


class FundingRelationship(BaseModel):
    """One named author explicitly connected to one agency award."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    author_alias: StrictStr = Field(min_length=1)
    agency_keyword: StrictStr = Field(min_length=1)
    award_identifier: StrictStr = Field(min_length=3, max_length=100)
    passage: ScholarlyPassage


class SpecimenDepositionRelationship(BaseModel):
    """Specimen deposition institution, distinct from author affiliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    institution: StrictStr = Field(min_length=3, max_length=300)
    passage: ScholarlyPassage


class ScholarlyEvidenceExtractor:
    """Require exact same-passage relationships from primary PDF page text."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def funding_for_author(
        self,
        document: PdfDocumentExtraction,
        *,
        author_aliases: tuple[str, ...],
        agency_keywords: tuple[str, ...] = ("NASA",),
        award_pattern: re.Pattern[str] = _DEFAULT_AWARD,
    ) -> FundingRelationship:
        """Extract an award only when author, agency, and funding relation co-occur."""

        aliases = _normalized_terms(author_aliases, "author aliases")
        agencies = _normalized_terms(agency_keywords, "agency keywords")
        matches: list[FundingRelationship] = []
        for passage in self._passages(document):
            folded = passage.text.casefold()
            alias = next((value for value in aliases if value.casefold() in folded), None)
            agency = next((value for value in agencies if value.casefold() in folded), None)
            if alias is None or agency is None or _RELATION.search(passage.text) is None:
                continue
            if _ACKNOWLEDGMENT.search(passage.text) is None and "supported" not in folded:
                continue
            identifiers = tuple(dict.fromkeys(award_pattern.findall(passage.text)))
            for identifier in identifiers:
                if isinstance(identifier, tuple):
                    raise ScholarlyEvidenceError("award pattern must not use capture groups")
                matches.append(
                    FundingRelationship(
                        author_alias=alias,
                        agency_keyword=agency,
                        award_identifier=" ".join(identifier.split()),
                        passage=passage,
                    )
                )
        unique = {
            (match.award_identifier.casefold(), match.author_alias.casefold()): match
            for match in matches
        }
        if len(unique) != 1:
            raise ScholarlyEvidenceError(
                "primary paper must contain exactly one author-to-award relationship"
            )
        return next(iter(unique.values()))

    def specimen_depository(
        self,
        document: PdfDocumentExtraction,
    ) -> SpecimenDepositionRelationship:
        """Extract an institution only from an explicit specimen-deposition passage."""

        matches: list[SpecimenDepositionRelationship] = []
        for passage in self._passages(document):
            if _SPECIMEN.search(passage.text) is None:
                continue
            for match in _DEPOSITION.finditer(passage.text):
                institution = " ".join(match.group("institution").split()).strip(" ,")
                matches.append(
                    SpecimenDepositionRelationship(
                        institution=institution,
                        passage=passage,
                    )
                )
        unique = {match.institution.casefold(): match for match in matches}
        if len(unique) != 1:
            raise ScholarlyEvidenceError(
                "primary paper must contain exactly one specimen-deposition institution"
            )
        return next(iter(unique.values()))

    def _passages(self, document: PdfDocumentExtraction) -> tuple[ScholarlyPassage, ...]:
        passages: list[ScholarlyPassage] = []
        for page in document.pages:
            if page.text_artifact_id is None:
                continue
            artifact = self._store.get(page.text_artifact_id)
            if artifact.source is not ArtifactSource.DERIVED:
                raise ScholarlyEvidenceError("PDF page text must be a derived artifact")
            page_text = self._store.read_bytes(artifact).decode("utf-8")
            for paragraph in re.split(
                r"\n\s*\n|(?<=[a-z0-9)])\.\s+(?=[A-Z])|(?<=[!?])\s+(?=[A-Z])",
                page_text,
            ):
                normalized = " ".join(paragraph.split())
                if normalized:
                    passages.append(
                        ScholarlyPassage(
                            page_number=page.page_number,
                            text_artifact_id=page.text_artifact_id,
                            text=normalized[:4000],
                        )
                    )
        return tuple(passages)


def _normalized_terms(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label} cannot be empty")
    normalized = tuple(" ".join(value.split()) for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{label} cannot contain empty values")
    if len({value.casefold() for value in normalized}) != len(normalized):
        raise ValueError(f"{label} must be unique case-insensitively")
    return normalized


__all__ = [
    "FundingRelationship",
    "ScholarlyEvidenceError",
    "ScholarlyEvidenceExtractor",
    "ScholarlyPassage",
    "SpecimenDepositionRelationship",
]
