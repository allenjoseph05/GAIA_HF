"""Deterministic extraction from identity-bound historical Wikipedia wikitext."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from gaia_max.artifacts import ArtifactIntegrityError, ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN
from gaia_max.retrieval.mediawiki import MediaWikiRevisionContent

_HEADING = re.compile(r"^(?P<marks>={2,6})\s*(?P<title>.*?)\s*(?P=marks)\s*$", re.MULTILINE)
_TABLE = re.compile(r"^\{\|.*?^\|\}\s*$", re.MULTILINE | re.DOTALL)
_TABLE_ROW = re.compile(r"^\|-.*$", re.MULTILINE)
_YEAR = re.compile(r"(?<!\d)(?P<year>1[89]\d{2}|20\d{2}|21\d{2})(?!\d)")
_RELEASED_YEAR = re.compile(
    r"\breleas(?:ed|e\s+date)\s*:?[^\n]{0,100}?"
    r"(?P<year>1[89]\d{2}|20\d{2}|21\d{2})",
    re.IGNORECASE,
)
_ITALIC_TITLE = re.compile(r"''(?!')(?P<title>.+?)(?<!')''")
_LINK = re.compile(r"\[\[(?P<target>[^\]|#]+)(?:#[^\]|]+)?(?:\|(?P<label>[^\]]+))?\]\]")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.IGNORECASE | re.DOTALL)
_USER_LINK = re.compile(
    r"\[\[\s*User(?:[ _]talk)?\s*:\s*(?P<account>[^\]|/#]+)"
    r"(?:[/#][^\]|]*)?(?:\|(?P<label>[^\]]+))?\]\]",
    re.IGNORECASE,
)
_USER_TEMPLATE = re.compile(
    r"\{\{\s*(?:user|nominator|fac nom)\s*\|\s*(?P<account>[^|}]+)",
    re.IGNORECASE,
)
_NOMINATION_LANGUAGE = re.compile(
    r"\b(?:(?:I\s+am|we\s+are)\s+(?:co-)?nominat(?:ing|ors?)|"
    r"co-nominat(?:ing|ors?)|nominators?\s*:)",
    re.IGNORECASE,
)


class HistoricalWikipediaExtractionError(ValueError):
    """Historical content is missing, ambiguous, or not provenance-bound."""


@dataclass(frozen=True, slots=True)
class WikitextSection:
    """One heading and its body through the next peer/ancestor heading."""

    title: str
    level: int
    body: str


class DiscographyAlbum(BaseModel):
    """One deterministically recognized studio-album record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: StrictStr = Field(min_length=1, max_length=500)
    release_year: int = Field(ge=1800, le=2199)
    extraction_method: str = Field(pattern=r"^(?:wikitable|list)$")


class HistoricalDiscographyResult(BaseModel):
    """Auditable inclusive count from one exact historical revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_id: int = Field(gt=0)
    revision_timestamp: datetime
    cutoff: datetime
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    section_title: StrictStr = Field(min_length=1)
    year_start: int = Field(ge=1800, le=2199)
    year_end: int = Field(ge=1800, le=2199)
    albums: tuple[DiscographyAlbum, ...] = Field(min_length=1)
    matching_titles: tuple[str, ...]
    count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> HistoricalDiscographyResult:
        if self.year_start > self.year_end:
            raise ValueError("discography year start cannot follow year end")
        expected = tuple(
            album.title
            for album in self.albums
            if self.year_start <= album.release_year <= self.year_end
        )
        if self.matching_titles != expected or self.count != len(expected):
            raise ValueError("discography count must match the inclusive filtered records")
        return self


class NominatorIdentity(BaseModel):
    """One signature identity attached to nomination language."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_name: StrictStr = Field(min_length=1, max_length=255)
    display_name: StrictStr = Field(min_length=1, max_length=255)


class FeaturedArticleNominationResult(BaseModel):
    """Nominators extracted only from nomination-bearing lead paragraphs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision_id: int = Field(gt=0)
    revision_timestamp: datetime
    cutoff: datetime
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    nominators: tuple[NominatorIdentity, ...] = Field(min_length=1)
    nomination_excerpt: StrictStr = Field(min_length=1, max_length=4000)


def parse_wikitext_sections(wikitext: str) -> tuple[WikitextSection, ...]:
    """Parse MediaWiki headings while preserving exact section bodies."""

    matches = list(_HEADING.finditer(wikitext))
    sections: list[WikitextSection] = []
    for index, match in enumerate(matches):
        level = len(match.group("marks"))
        body_end = len(wikitext)
        for later in matches[index + 1 :]:
            if len(later.group("marks")) <= level:
                body_end = later.start()
                break
        title = _plain_text(match.group("title"))
        if not title:
            raise HistoricalWikipediaExtractionError("wikitext heading has no title")
        sections.append(
            WikitextSection(
                title=title,
                level=level,
                body=wikitext[match.end() : body_end].strip(),
            )
        )
    return tuple(sections)


def extract_studio_albums(wikitext: str) -> tuple[str, tuple[DiscographyAlbum, ...]]:
    """Extract studio records while excluding sibling live/compilation sections."""

    sections = parse_wikitext_sections(wikitext)
    studio = [section for section in sections if _heading_key(section.title) == "studio albums"]
    if len(studio) != 1:
        raise HistoricalWikipediaExtractionError(
            "historical page must contain exactly one Studio albums section"
        )
    section = studio[0]
    records: list[DiscographyAlbum] = []
    table_spans = list(_TABLE.finditer(section.body))
    for match in table_spans:
        records.extend(_albums_from_table(match.group(0)))

    without_tables = _TABLE.sub("", section.body)
    for line in without_tables.splitlines():
        stripped = line.strip()
        if not re.match(r"^[*#](?![*#])", stripped):
            continue
        album = _album_from_record(stripped[1:].strip(), method="list")
        if album is not None:
            records.append(album)

    if not records:
        raise HistoricalWikipediaExtractionError(
            "Studio albums section contains no recognized album records"
        )
    identities = [(album.title.casefold(), album.release_year) for album in records]
    if len(set(identities)) != len(identities):
        raise HistoricalWikipediaExtractionError("studio album records are duplicated")
    return section.title, tuple(records)


def extract_fac_nominators(wikitext: str) -> tuple[tuple[NominatorIdentity, ...], str]:
    """Extract signatures only from paragraphs that explicitly nominate the article."""

    cleaned = _REF.sub("", _COMMENT.sub("", wikitext))
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned)]
    nomination_paragraphs = [
        paragraph for paragraph in paragraphs if _NOMINATION_LANGUAGE.search(paragraph)
    ]
    if not nomination_paragraphs:
        raise HistoricalWikipediaExtractionError("FAC page has no nomination-bearing paragraph")

    identities: list[NominatorIdentity] = []
    seen_accounts: set[str] = set()
    for paragraph in nomination_paragraphs:
        for match in _USER_LINK.finditer(paragraph):
            account = _normalized_identity(match.group("account"))
            if account.casefold() in seen_accounts:
                continue
            label = match.group("label")
            display = _plain_text(label) if label else account
            if not display or display.casefold() in {"talk", "contribs", "contributions"}:
                display = account
            identities.append(NominatorIdentity(account_name=account, display_name=display))
            seen_accounts.add(account.casefold())
        for match in _USER_TEMPLATE.finditer(paragraph):
            account = _normalized_identity(match.group("account"))
            if account.casefold() not in seen_accounts:
                identities.append(
                    NominatorIdentity(account_name=account, display_name=account)
                )
                seen_accounts.add(account.casefold())

    if not identities:
        raise HistoricalWikipediaExtractionError(
            "nomination paragraph contains no attributable user signatures"
        )
    excerpt = "\n\n".join(nomination_paragraphs)
    if len(excerpt) > 4000:
        excerpt = excerpt[:4000]
    return tuple(identities), excerpt


class HistoricalWikipediaExtractor:
    """Read exact revision artifacts and produce typed deterministic results."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def count_studio_albums(
        self,
        historical: MediaWikiRevisionContent,
        *,
        year_start: int,
        year_end: int,
    ) -> HistoricalDiscographyResult:
        """Count studio albums in an inclusive range from the selected revision only."""

        if not 1800 <= year_start <= year_end <= 2199:
            raise ValueError("discography year range must be ordered within 1800-2199")
        wikitext = self._historical_wikitext(historical)
        section_title, albums = extract_studio_albums(wikitext)
        matching = tuple(
            album.title
            for album in albums
            if year_start <= album.release_year <= year_end
        )
        return HistoricalDiscographyResult(
            revision_id=historical.selection.revision.revision_id,
            revision_timestamp=historical.selection.revision.timestamp,
            cutoff=historical.selection.cutoff,
            artifact_id=historical.artifact_id,
            section_title=section_title,
            year_start=year_start,
            year_end=year_end,
            albums=albums,
            matching_titles=matching,
            count=len(matching),
        )

    def fac_nominators(
        self,
        historical: MediaWikiRevisionContent,
    ) -> FeaturedArticleNominationResult:
        """Extract nominator identities without treating reviewers as nominators."""

        wikitext = self._historical_wikitext(historical)
        nominators, excerpt = extract_fac_nominators(wikitext)
        return FeaturedArticleNominationResult(
            revision_id=historical.selection.revision.revision_id,
            revision_timestamp=historical.selection.revision.timestamp,
            cutoff=historical.selection.cutoff,
            artifact_id=historical.artifact_id,
            nominators=nominators,
            nomination_excerpt=excerpt,
        )

    def _historical_wikitext(self, historical: MediaWikiRevisionContent) -> str:
        revision = historical.selection.revision
        if revision.timestamp > historical.selection.cutoff:
            raise HistoricalWikipediaExtractionError(
                "selected revision is newer than the historical cutoff"
            )
        artifact = self._store.get(historical.artifact_id)
        metadata = self._store.metadata(historical.artifact_id)
        expected_locator = str(revision.revision_url)
        if not any(
            origin.source_locator == expected_locator
            and origin.source is ArtifactSource.WEB_RETRIEVAL
            for origin in metadata.origins
        ):
            raise ArtifactIntegrityError(
                "historical artifact lacks provenance for the selected revision"
            )
        try:
            return self._store.read_bytes(artifact).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HistoricalWikipediaExtractionError(
                "historical revision is not valid UTF-8 wikitext"
            ) from exc


def _albums_from_table(table: str) -> list[DiscographyAlbum]:
    records: list[DiscographyAlbum] = []
    rows = _TABLE_ROW.split(table)
    for row in rows[1:]:
        album = _album_from_record(row, method="wikitable")
        if album is not None:
            records.append(album)
    return records


def _album_from_record(record: str, *, method: str) -> DiscographyAlbum | None:
    released = _RELEASED_YEAR.search(record)
    years = [int(match.group("year")) for match in _YEAR.finditer(record)]
    if released is not None:
        year = int(released.group("year"))
    elif years:
        year = years[0]
    else:
        return None

    italic = _ITALIC_TITLE.search(record)
    if italic is not None:
        title = _plain_text(italic.group("title"))
    else:
        links = [
            _plain_text(match.group("label") or match.group("target"))
            for match in _LINK.finditer(record)
            if not match.group("target").casefold().startswith(
                ("file:", "category:", "help:", "wikipedia:")
            )
        ]
        title = links[0] if links else _fallback_table_title(record, year)
    if not title:
        return None
    return DiscographyAlbum(title=title, release_year=year, extraction_method=method)


def _fallback_table_title(record: str, year: int) -> str:
    cells = [cell.strip() for cell in re.split(r"\|\||\n[|!]", record)]
    for cell in cells:
        plain = _plain_text(cell.lstrip("!|").strip())
        if plain and str(year) not in plain and not plain.casefold().startswith("released:"):
            return plain
    return ""


def _plain_text(value: str) -> str:
    without_templates = _TEMPLATE.sub("", value)

    def replace_link(match: re.Match[str]) -> str:
        return match.group("label") or match.group("target")

    unlinked = _LINK.sub(replace_link, without_templates)
    plain = re.sub(r"'{2,5}", "", unlinked)
    plain = re.sub(r"<[^>]+>", "", plain)
    return (
        " ".join(plain.replace("_", " ").split())
        .strip(" -:;|")
        .strip("\u2013\u2014")
    )


def _heading_key(title: str) -> str:
    return _plain_text(title).casefold()


def _normalized_identity(value: str) -> str:
    normalized = " ".join(value.replace("_", " ").split()).strip()
    if not normalized or len(normalized) > 255:
        raise HistoricalWikipediaExtractionError("FAC user identity is empty or too long")
    return normalized


__all__ = [
    "DiscographyAlbum",
    "FeaturedArticleNominationResult",
    "HistoricalDiscographyResult",
    "HistoricalWikipediaExtractionError",
    "HistoricalWikipediaExtractor",
    "NominatorIdentity",
    "WikitextSection",
    "extract_fac_nominators",
    "extract_studio_albums",
    "parse_wikitext_sections",
]
