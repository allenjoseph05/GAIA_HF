from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaia_max.artifacts import ArtifactIntegrityError, ArtifactStore
from gaia_max.domain import ArtifactSource
from gaia_max.retrieval import MediaWikiRevisionContent
from gaia_max.solvers import (
    HistoricalWikipediaExtractionError,
    HistoricalWikipediaExtractor,
    extract_fac_nominators,
    extract_studio_albums,
    parse_wikitext_sections,
)

CUTOFF = datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC)
REVISION_TIME = datetime(2022, 10, 4, 12, 30, tzinfo=UTC)
RETRIEVED = datetime(2026, 8, 30, 13, 0, tzinfo=UTC)
REVISION_URL = "https://en.wikipedia.org/w/index.php?oldid=300"


def historical_content(
    store: ArtifactStore,
    wikitext: str,
    *,
    source_locator: str = REVISION_URL,
    revision_time: datetime = REVISION_TIME,
) -> MediaWikiRevisionContent:
    artifact = store.put_bytes(
        wikitext.encode("utf-8"),
        original_name="revision-300.wikitext",
        media_type="text/x-wiki",
        source=ArtifactSource.WEB_RETRIEVAL,
        retrieved_at=RETRIEVED,
        source_locator=source_locator,
    )
    return MediaWikiRevisionContent.model_validate(
        {
            "selection": {
                "requested_title": "Synthetic artist",
                "resolved_title": "Synthetic artist",
                "cutoff": CUTOFF,
                "revision": {
                    "page_id": 42,
                    "title": "Synthetic artist",
                    "revision_id": 300,
                    "parent_revision_id": 299,
                    "timestamp": revision_time,
                    "sha1": "a" * 40,
                    "revision_url": REVISION_URL,
                },
            },
            "artifact_id": artifact.artifact_id,
            "retrieved_at": RETRIEVED,
            "content_model": "wikitext",
            "content_format": "text/x-wiki",
        }
    )


def test_section_parser_stops_at_peer_heading_and_preserves_nested_body() -> None:
    sections = parse_wikitext_sections(
        "Lead\n== Discography ==\nIntro\n=== Studio albums ===\n* A\n"
        "=== Live albums ===\n* B\n== References ==\nRefs"
    )

    discography = next(section for section in sections if section.title == "Discography")
    studio = next(section for section in sections if section.title == "Studio albums")
    assert "Studio albums" in discography.body
    assert "References" not in discography.body
    assert studio.body == "* A"


def test_discography_list_extraction_uses_only_studio_section_and_inclusive_years(
    tmp_path: Path,
) -> None:
    historical = historical_content(
        ArtifactStore(tmp_path / "artifacts"),
        """
== Discography ==
=== Studio albums ===
* ''[[Before]]'' (1999)
* ''[[Lower boundary]]'' (2000)
* ''[[Middle album|Middle]]'' — released 14 March 2005
* ''[[Upper boundary]]'' (2009)
* ''[[After]]'' (2010)
=== Live albums ===
* ''[[Live distraction]]'' (2004)
=== Compilation albums ===
* ''[[Compilation distraction]]'' (2006)
""".strip(),
    )
    store = ArtifactStore(tmp_path / "artifacts")

    result = HistoricalWikipediaExtractor(store).count_studio_albums(
        historical,
        year_start=2000,
        year_end=2009,
    )

    assert result.count == 3
    assert result.matching_titles == ("Lower boundary", "Middle", "Upper boundary")
    assert result.revision_id == 300
    assert result.revision_timestamp == REVISION_TIME
    assert result.cutoff == CUTOFF


def test_discography_wikitable_extracts_one_album_per_row() -> None:
    section, albums = extract_studio_albums(
        """
== Studio albums ==
{| class="wikitable"
! Title !! Album details
|-
! scope="row" | ''[[First record]]''
|
* Released: 2 February 2001
* Label: Example
|-
! scope="row" | ''[[Second record|Displayed title]]''
|
* Released: 3 March 2008
* Reissued: 2018
|}
== Live albums ==
{| class="wikitable"
|-
| ''[[Wrong category]]'' || Released: 2004
|}
""".strip()
    )

    assert section == "Studio albums"
    assert [(album.title, album.release_year) for album in albums] == [
        ("First record", 2001),
        ("Displayed title", 2008),
    ]
    assert all(album.extraction_method == "wikitable" for album in albums)


def test_discography_requires_one_unambiguous_studio_section() -> None:
    with pytest.raises(HistoricalWikipediaExtractionError, match="exactly one"):
        extract_studio_albums("== Discography ==\n* ''Album'' (2001)")
    with pytest.raises(HistoricalWikipediaExtractionError, match="exactly one"):
        extract_studio_albums(
            "== Studio albums ==\n* ''One'' (2001)\n"
            "== Studio albums ==\n* ''Two'' (2002)"
        )


def test_fac_nominators_come_only_from_nomination_paragraph_not_reviewers() -> None:
    wikitext = """
I am nominating this article for featured article because the prose and sources
are now comprehensive. [[User:Alpha_editor|Alice Example]]
([[User talk:Alpha_editor|talk]]) and co-nominator {{user|Beta editor}}.

====Support====
'''Support''' after review. [[User:Reviewer|Review Person]]

====Source review====
I support this nomination. [[User:Another reviewer|Another Person]]
""".strip()

    nominators, excerpt = extract_fac_nominators(wikitext)

    assert [(item.account_name, item.display_name) for item in nominators] == [
        ("Alpha editor", "Alice Example"),
        ("Beta editor", "Beta editor"),
    ]
    assert "Reviewer" not in excerpt


def test_fac_high_level_result_stays_bound_to_revision_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    historical = historical_content(
        store,
        "We are nominating this article after a rewrite. "
        "[[User:One|Person One]] and [[User:Two|Person Two]]\n\n"
        "====Comments====\n[[User:Reviewer|Reviewer]]",
    )

    result = HistoricalWikipediaExtractor(store).fac_nominators(historical)

    assert [item.display_name for item in result.nominators] == [
        "Person One",
        "Person Two",
    ]
    assert result.revision_id == 300
    assert result.artifact_id == historical.artifact_id


def test_extractor_rejects_wrong_revision_provenance_and_post_cutoff_content(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    wrong_locator = historical_content(
        store,
        "== Studio albums ==\n* ''Album'' (2001)",
        source_locator="https://en.wikipedia.org/w/index.php?oldid=999",
    )
    with pytest.raises(ArtifactIntegrityError, match="lacks provenance"):
        HistoricalWikipediaExtractor(store).count_studio_albums(
            wrong_locator,
            year_start=2000,
            year_end=2009,
        )

    newer = historical_content(
        store,
        "== Studio albums ==\n* ''Album'' (2001)",
        revision_time=datetime(2023, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(HistoricalWikipediaExtractionError, match="newer"):
        HistoricalWikipediaExtractor(store).count_studio_albums(
            newer,
            year_start=2000,
            year_end=2009,
        )
