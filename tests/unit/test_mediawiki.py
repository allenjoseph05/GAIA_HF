from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource
from gaia_max.retrieval import (
    MediaWikiClient,
    MediaWikiError,
    MediaWikiNotFoundError,
    MediaWikiRevisionRef,
    MediaWikiTemporalError,
    select_latest_revision,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC)
SHA1 = "a" * 40


def settings(**updates: object) -> Settings:
    return Settings.model_validate(
        {
            "page_max_attempts": 2,
            "page_retry_base_seconds": 0,
            "page_retry_max_seconds": 0,
            "max_page_bytes": 100_000,
            "mediawiki_max_history_pages": 3,
            "mediawiki_revision_batch_size": 50,
            **updates,
        }
    )


def revision(revid: int, timestamp: str, *, content: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "revid": revid,
        "parentid": revid - 1,
        "timestamp": timestamp,
        "sha1": SHA1,
    }
    if content is not None:
        item["slots"] = {
            "main": {
                "contentmodel": "wikitext",
                "contentformat": "text/x-wiki",
                "content": content,
            }
        }
    return item


def history_payload(
    revisions: list[dict[str, Any]],
    *,
    title: str = "Canonical title",
    continuation: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": {
            "redirects": [{"from": "Alias", "to": title}],
            "pages": [
                {
                    "pageid": 42,
                    "ns": 0,
                    "title": title,
                    "revisions": revisions,
                }
            ],
        }
    }
    if continuation is not None:
        payload["continue"] = {"continue": "||", "rvcontinue": continuation}
    return payload


def response(
    request: httpx.Request,
    payload: dict[str, Any],
    *,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(status_code, request=request, json=payload)


def make_ref(revid: int, timestamp: datetime) -> MediaWikiRevisionRef:
    return MediaWikiRevisionRef.model_validate(
        {
            "page_id": 42,
            "title": "Synthetic page",
            "revision_id": revid,
            "parent_revision_id": revid - 1,
            "timestamp": timestamp,
            "sha1": SHA1,
            "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
        }
    )


def test_pure_revision_selection_is_inclusive_order_independent_and_deterministic() -> None:
    exact = make_ref(30, CUTOFF)
    earlier = make_ref(20, datetime(2022, 6, 1, tzinfo=UTC))
    later = make_ref(40, datetime(2023, 1, 1, tzinfo=UTC))

    selected = select_latest_revision((earlier, later, exact), CUTOFF)

    assert selected.revision_id == 30


def test_revision_selection_rejects_naive_cutoff_duplicates_and_empty_eligibility() -> None:
    item = make_ref(20, datetime(2023, 1, 1, tzinfo=UTC))

    with pytest.raises(ValueError, match="timezone-aware"):
        select_latest_revision((item,), datetime(2023, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        select_latest_revision((item, item), CUTOFF)
    with pytest.raises(MediaWikiTemporalError):
        select_latest_revision((item,), CUTOFF)


@pytest.mark.asyncio
async def test_client_selects_latest_inclusive_revision_and_preserves_redirect(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["rvstart"] == "2022-12-31T23:59:59Z"
        assert request.url.params["rvdir"] == "older"
        return response(
            request,
            history_payload(
                [
                    revision(10, "2021-01-01T00:00:00Z"),
                    revision(30, "2022-12-31T23:59:59Z"),
                    revision(20, "2022-06-01T00:00:00Z"),
                ]
            ),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
    )

    selected = await client.latest_revision_at_or_before("Alias", CUTOFF)

    assert selected.requested_title == "Alias"
    assert selected.resolved_title == "Canonical title"
    assert selected.revision.revision_id == 30
    assert selected.revision.timestamp == CUTOFF
    assert str(selected.revision.revision_url).endswith("oldid=30")
    await http_client.aclose()


@pytest.mark.asyncio
async def test_client_follows_bounded_history_continuation_until_candidate(
    tmp_path: Path,
) -> None:
    calls: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        continuation = request.url.params.get("rvcontinue")
        calls.append(continuation)
        if continuation is None:
            return response(request, history_payload([], continuation="next-token"))
        return response(
            request,
            history_payload([revision(20, "2022-01-01T00:00:00Z")]),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
    )

    selected = await client.latest_revision_at_or_before("Alias", CUTOFF)

    assert selected.revision.revision_id == 20
    assert calls == [None, "next-token"]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_exact_revision_content_is_identity_bound_and_stored_with_oldid_provenance(
    tmp_path: Path,
) -> None:
    wikitext = "== Discography ==\n* Synthetic album (2007)"

    async def handler(request: httpx.Request) -> httpx.Response:
        if "titles" in request.url.params:
            return response(
                request,
                history_payload([revision(30, "2022-12-31T23:59:59Z")]),
            )
        assert request.url.params["revids"] == "30"
        return response(
            request,
            history_payload(
                [revision(30, "2022-12-31T23:59:59Z", content=wikitext)]
            ),
        )

    store = ArtifactStore(tmp_path / "artifacts")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=store,
        settings=settings(),
        http_client=http_client,
        clock=lambda: NOW,
    )

    historical = await client.fetch_page_at_or_before("Alias", CUTOFF)

    artifact = store.get(historical.artifact_id)
    metadata = store.metadata(historical.artifact_id)
    assert store.read_bytes(artifact).decode("utf-8") == wikitext
    assert artifact.source is ArtifactSource.WEB_RETRIEVAL
    assert metadata.origins[0].source_locator is not None
    assert metadata.origins[0].source_locator.endswith("oldid=30")
    assert historical.selection.cutoff == CUTOFF
    await http_client.aclose()


@pytest.mark.asyncio
async def test_revision_content_identity_drift_is_rejected(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if "titles" in request.url.params:
            item = revision(30, "2022-12-31T23:59:59Z")
        else:
            item = revision(31, "2022-12-31T23:59:59Z", content="changed")
        return response(request, history_payload([item]))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
    )
    selection = await client.latest_revision_at_or_before("Alias", CUTOFF)

    with pytest.raises(MediaWikiError, match="revision_identity_mismatch"):
        await client.fetch_revision_content(selection)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_missing_page_and_no_historical_revision_are_distinct(tmp_path: Path) -> None:
    responses = [
        {"query": {"pages": [{"ns": 0, "title": "Missing", "missing": True}]}},
        history_payload([]),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return response(request, responses.pop(0))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
    )

    with pytest.raises(MediaWikiNotFoundError):
        await client.latest_revision_at_or_before("Missing", CUTOFF)
    with pytest.raises(MediaWikiTemporalError):
        await client.latest_revision_at_or_before("Too old", CUTOFF)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_transient_http_failure_retries_without_changing_query(tmp_path: Path) -> None:
    attempts = 0
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return response(
            request,
            history_payload([revision(20, "2022-01-01T00:00:00Z")]),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
        sleep=sleep,
    )

    selected = await client.latest_revision_at_or_before("Alias", CUTOFF)

    assert selected.revision.revision_id == 20
    assert attempts == 2
    assert waits == [0]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_page_search_returns_safe_title_identity_without_html_snippets(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["list"] == "search"
        return response(
            request,
            {
                "query": {
                    "search": [
                        {
                            "pageid": 7,
                            "title": "Synthetic result",
                            "snippet": "<span>untrusted markup</span>",
                        }
                    ]
                }
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = MediaWikiClient(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=http_client,
    )

    results = await client.search_pages("Synthetic query", limit=1)

    assert len(results) == 1
    assert results[0].title == "Synthetic result"
    assert str(results[0].page_url).endswith("/wiki/Synthetic_result")
    assert not hasattr(results[0], "snippet")
    await http_client.aclose()
