from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource
from gaia_max.retrieval import (
    PageAttemptStatus,
    PageFailureKind,
    PageLadderExhaustedError,
    PageRetrievalError,
    PageRetrievalLadder,
    PageRetrievalRequest,
    PageRetryPolicy,
    PageStrategy,
    ProviderContractError,
    StaticHttpPageProvider,
    retrieve_document,
)

NOW = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)


def settings(**updates: object) -> Settings:
    return Settings.model_validate(
        {
            "page_min_extracted_chars": 40,
            "max_page_bytes": 4096,
            **updates,
        }
    )


def document_payload(
    provider: str,
    request: PageRetrievalRequest,
    *,
    strategy: PageStrategy,
) -> dict[str, object]:
    final_url = (
        "https://archive.example/snapshot/https://example.org/article"
        if strategy is PageStrategy.ARCHIVE
        else "https://example.org/article"
    )
    return {
        "provider": provider,
        "requested_url": str(request.url),
        "final_url": final_url,
        "status_code": 200,
        "title": f"Synthetic {strategy.value} document",
        "retrieved_at": NOW,
        "artifact_id": "a" * 64,
        "text_artifact_id": "b" * 64,
        "links": (),
        "extraction_method": f"{strategy.value}_fixture",
        "injection_flags": (),
    }


ScriptedResponse = object | Callable[[PageRetrievalRequest], object]


class ScriptedPageProvider:
    def __init__(
        self,
        name: str,
        strategy: PageStrategy,
        responses: list[ScriptedResponse],
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.responses = responses
        self.requests: list[PageRetrievalRequest] = []

    async def retrieve(self, request: PageRetrievalRequest) -> object:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected call to {self.name}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(request)
        return response


def success(
    provider: str,
    strategy: PageStrategy,
) -> Callable[[PageRetrievalRequest], object]:
    return lambda request: document_payload(provider, request, strategy=strategy)


def request() -> PageRetrievalRequest:
    return PageRetrievalRequest.model_validate({"url": "https://example.org/article"})


@pytest.mark.asyncio
async def test_static_html_is_extracted_without_navigation_and_stored_by_hash(
    tmp_path: Path,
) -> None:
    html = b"""<!doctype html>
    <html><head>
      <title>Synthetic article</title>
      <meta property="article:published_time" content="2024-03-02T08:00:00Z">
    </head><body>
      <header>Site chrome that should disappear</header>
      <main><h1>Evidence heading</h1>
      <p>This synthetic primary passage is deliberately long enough for extraction.</p>
      <p>It contains another complete sentence for deterministic fixture coverage.</p>
      <a href="/source">Primary source</a>
      <a href="mailto:test@example.org">Email</a></main>
      <footer>Footer navigation should disappear</footer>
    </body></html>"""

    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/html; charset=utf-8"},
            content=html,
        )

    store = ArtifactStore(tmp_path / "artifacts")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=store,
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )

    document = await retrieve_document(provider, request())

    assert document.provider == "static-http"
    assert document.extraction_method == "static_html"
    assert document.title == "Synthetic article"
    assert str(document.published_at) == "2024-03-02"
    assert [str(link.url) for link in document.links] == ["https://example.org/source"]
    assert document.text_artifact_id is not None
    raw = store.get(document.artifact_id)
    text = store.get(document.text_artifact_id)
    assert raw.source is ArtifactSource.WEB_RETRIEVAL
    assert text.source is ArtifactSource.DERIVED
    assert store.read_bytes(raw) == html
    extracted = store.read_bytes(text).decode("utf-8")
    assert "Evidence heading" in extracted
    assert "Site chrome" not in extracted
    assert "Footer navigation" not in extracted
    assert "Synthetic article" not in extracted
    await client.aclose()


@pytest.mark.asyncio
async def test_static_plain_text_gets_a_separate_normalized_text_artifact(
    tmp_path: Path,
) -> None:
    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=b"Line one.\n\n  Line two with spaces.  ",
        )

    store = ArtifactStore(tmp_path / "artifacts")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=store,
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )

    document = await retrieve_document(provider, request())

    assert document.extraction_method == "static_plain_text"
    assert document.text_artifact_id is not None
    text = store.read_bytes(store.get(document.text_artifact_id)).decode("utf-8")
    assert text == "Line one.\nLine two with spaces."
    await client.aclose()


@pytest.mark.asyncio
async def test_javascript_shell_is_classified_with_raw_artifact(
    tmp_path: Path,
) -> None:
    html = (
        b"<html><head><title>App</title></head><body><div id='root'></div>"
        b"<script src='app.js'></script></body></html>"
    )

    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/html"},
            content=html,
        )

    store = ArtifactStore(tmp_path / "artifacts")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=store,
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )

    with pytest.raises(PageRetrievalError) as caught:
        await provider.retrieve(request())

    assert caught.value.kind is PageFailureKind.JS_REQUIRED
    assert caught.value.code == "javascript_shell"
    assert caught.value.artifact_id is not None
    assert store.read_bytes(store.get(caught.value.artifact_id)) == html
    await client.aclose()


@pytest.mark.asyncio
async def test_block_page_disguised_as_http_200_is_classified_and_preserved(
    tmp_path: Path,
) -> None:
    html = (
        b"<html><head><title>Access Denied</title></head>"
        b"<body><p>Verify you are human to continue.</p></body></html>"
    )

    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/html"},
            content=html,
        )

    store = ArtifactStore(tmp_path / "artifacts")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=store,
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )

    with pytest.raises(PageRetrievalError) as caught:
        await provider.retrieve(request())

    assert caught.value.kind is PageFailureKind.BLOCKED
    assert caught.value.code == "blocked_response_content"
    assert caught.value.artifact_id is not None
    assert store.read_bytes(store.get(caught.value.artifact_id)) == html
    await client.aclose()


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (403, PageFailureKind.BLOCKED),
        (404, PageFailureKind.NOT_FOUND),
        (429, PageFailureKind.TRANSIENT),
        (503, PageFailureKind.TRANSIENT),
    ],
)
@pytest.mark.asyncio
async def test_http_statuses_are_classified_before_extraction(
    tmp_path: Path,
    status: int,
    kind: PageFailureKind,
) -> None:
    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=incoming, content=b"synthetic error")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=client,
    )

    with pytest.raises(PageRetrievalError) as caught:
        await provider.retrieve(request())

    assert caught.value.kind is kind
    await client.aclose()


@pytest.mark.asyncio
async def test_streamed_page_size_is_bounded_before_artifact_storage(
    tmp_path: Path,
) -> None:
    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/plain"},
            content=b"0123456789",
        )

    store = ArtifactStore(tmp_path / "artifacts")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=store,
        settings=settings(max_page_bytes=8),
        http_client=client,
    )

    with pytest.raises(PageRetrievalError) as caught:
        await provider.retrieve(request())

    assert caught.value.kind is PageFailureKind.PERMANENT
    assert caught.value.code == "page_too_large"
    assert list((tmp_path / "artifacts" / "sha256").iterdir()) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_public_redirect_is_followed_and_final_url_is_preserved(
    tmp_path: Path,
) -> None:
    requested_urls: list[str] = []

    async def handler(incoming: httpx.Request) -> httpx.Response:
        requested_urls.append(str(incoming.url))
        if incoming.url.host == "example.org":
            return httpx.Response(
                302,
                request=incoming,
                headers={"location": "https://www.example.org/final"},
            )
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/plain"},
            content=b"Redirected public content.",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )

    document = await retrieve_document(provider, request())

    assert requested_urls == [
        "https://example.org/article",
        "https://www.example.org/final",
    ]
    assert str(document.requested_url) == "https://example.org/article"
    assert str(document.final_url) == "https://www.example.org/final"
    await client.aclose()


@pytest.mark.asyncio
async def test_private_redirect_target_is_rejected_before_second_request(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(incoming: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            302,
            request=incoming,
            headers={"location": "http://127.0.0.1/private"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = StaticHttpPageProvider(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=client,
    )

    with pytest.raises(PageRetrievalError) as caught:
        await provider.retrieve(request())

    assert caught.value.kind is PageFailureKind.INVALID_REQUEST
    assert caught.value.code == "unsafe_redirect_target"
    assert calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_static_fixture_stops_at_direct_path(tmp_path: Path) -> None:
    html = (
        b"<html><body><main><p>A sufficiently long static fixture proves that "
        b"no fallback provider is called for ordinary HTML content."
        b"</p></main></body></html>"
    )

    async def handler(incoming: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=incoming,
            headers={"content-type": "text/html"},
            content=html,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    direct = StaticHttpPageProvider(
        store=ArtifactStore(tmp_path / "artifacts"),
        settings=settings(),
        http_client=client,
        clock=lambda: NOW,
    )
    reader = ScriptedPageProvider("reader", PageStrategy.READER, [])
    browser = ScriptedPageProvider("browser", PageStrategy.BROWSER, [])
    archive = ScriptedPageProvider("archive", PageStrategy.ARCHIVE, [])
    ladder = PageRetrievalLadder(
        direct=direct,
        reader=reader,
        browser=browser,
        archive=archive,
    )

    outcome = await ladder.retrieve(request())

    assert outcome.winning_strategy is PageStrategy.DIRECT
    assert [attempt.strategy for attempt in outcome.attempts] == [PageStrategy.DIRECT]
    assert reader.requests == browser.requests == archive.requests == []
    await client.aclose()


@pytest.mark.asyncio
async def test_javascript_fixture_uses_reader_then_browser() -> None:
    direct = ScriptedPageProvider(
        "direct",
        PageStrategy.DIRECT,
        [PageRetrievalError(PageFailureKind.JS_REQUIRED, code="javascript_shell")],
    )
    reader = ScriptedPageProvider(
        "reader",
        PageStrategy.READER,
        [
            PageRetrievalError(
                PageFailureKind.EXTRACTION_INCOMPLETE,
                code="reader_incomplete",
            )
        ],
    )
    browser = ScriptedPageProvider(
        "browser",
        PageStrategy.BROWSER,
        [success("browser", PageStrategy.BROWSER)],
    )
    archive = ScriptedPageProvider("archive", PageStrategy.ARCHIVE, [])

    outcome = await PageRetrievalLadder(
        direct=direct,
        reader=reader,
        browser=browser,
        archive=archive,
        policy=PageRetryPolicy(max_attempts=1),
    ).retrieve(request())

    assert outcome.winning_strategy is PageStrategy.BROWSER
    assert [attempt.strategy for attempt in outcome.attempts] == [
        PageStrategy.DIRECT,
        PageStrategy.READER,
        PageStrategy.BROWSER,
    ]
    assert archive.requests == []


@pytest.mark.asyncio
async def test_blocked_fixture_uses_reader_without_expensive_browser() -> None:
    direct = ScriptedPageProvider(
        "direct",
        PageStrategy.DIRECT,
        [PageRetrievalError(PageFailureKind.BLOCKED, code="http_403")],
    )
    reader = ScriptedPageProvider(
        "reader",
        PageStrategy.READER,
        [success("reader", PageStrategy.READER)],
    )
    browser = ScriptedPageProvider("browser", PageStrategy.BROWSER, [])
    archive = ScriptedPageProvider("archive", PageStrategy.ARCHIVE, [])

    outcome = await PageRetrievalLadder(
        direct=direct,
        reader=reader,
        browser=browser,
        archive=archive,
        policy=PageRetryPolicy(max_attempts=1),
    ).retrieve(request())

    assert outcome.winning_strategy is PageStrategy.READER
    assert [attempt.strategy for attempt in outcome.attempts] == [
        PageStrategy.DIRECT,
        PageStrategy.READER,
    ]
    assert browser.requests == archive.requests == []


@pytest.mark.asyncio
async def test_missing_current_page_routes_directly_to_archive() -> None:
    direct = ScriptedPageProvider(
        "direct",
        PageStrategy.DIRECT,
        [PageRetrievalError(PageFailureKind.NOT_FOUND, code="http_404")],
    )
    reader = ScriptedPageProvider("reader", PageStrategy.READER, [])
    browser = ScriptedPageProvider("browser", PageStrategy.BROWSER, [])
    archive = ScriptedPageProvider(
        "archive",
        PageStrategy.ARCHIVE,
        [success("archive", PageStrategy.ARCHIVE)],
    )

    outcome = await PageRetrievalLadder(
        direct=direct,
        reader=reader,
        browser=browser,
        archive=archive,
        policy=PageRetryPolicy(max_attempts=1),
    ).retrieve(request())

    assert outcome.winning_strategy is PageStrategy.ARCHIVE
    assert [attempt.strategy for attempt in outcome.attempts] == [
        PageStrategy.DIRECT,
        PageStrategy.ARCHIVE,
    ]
    assert outcome.document.final_url.host == "archive.example"
    assert reader.requests == browser.requests == []


@pytest.mark.asyncio
async def test_transient_direct_failure_retries_before_reader_fallback() -> None:
    direct = ScriptedPageProvider(
        "direct",
        PageStrategy.DIRECT,
        [
            PageRetrievalError(PageFailureKind.TRANSIENT, code="http_503"),
            PageRetrievalError(PageFailureKind.TRANSIENT, code="http_503"),
        ],
    )
    reader = ScriptedPageProvider(
        "reader",
        PageStrategy.READER,
        [success("reader", PageStrategy.READER)],
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    outcome = await PageRetrievalLadder(
        direct=direct,
        reader=reader,
        policy=PageRetryPolicy(
            max_attempts=2,
            retry_base_seconds=0.5,
            retry_max_seconds=2,
            jitter_ratio=0,
        ),
        sleep=record_sleep,
    ).retrieve(request())

    assert sleeps == [0.5]
    assert len(direct.requests) == 2
    assert outcome.winning_strategy is PageStrategy.READER
    assert [attempt.status for attempt in outcome.attempts] == [
        PageAttemptStatus.TRANSIENT_FAILURE,
        PageAttemptStatus.TRANSIENT_FAILURE,
        PageAttemptStatus.SUCCESS,
    ]


@pytest.mark.asyncio
async def test_exhausted_ladder_exposes_only_safe_attempt_metadata() -> None:
    raw_secret = "provider-body-secret"
    providers = (
        ScriptedPageProvider(
            "direct",
            PageStrategy.DIRECT,
            [
                PageRetrievalError(
                    PageFailureKind.JS_REQUIRED,
                    code="javascript_shell",
                )
            ],
        ),
        ScriptedPageProvider(
            "reader",
            PageStrategy.READER,
            [
                PageRetrievalError(
                    PageFailureKind.EXTRACTION_INCOMPLETE,
                    code="reader_incomplete",
                )
            ],
        ),
        ScriptedPageProvider(
            "browser",
            PageStrategy.BROWSER,
            [PageRetrievalError(PageFailureKind.PERMANENT, code="browser_failed")],
        ),
        ScriptedPageProvider(
            "archive",
            PageStrategy.ARCHIVE,
            [PageRetrievalError(PageFailureKind.NOT_FOUND, code="archive_missing")],
        ),
    )

    with pytest.raises(PageLadderExhaustedError) as caught:
        await PageRetrievalLadder(
            direct=providers[0],
            reader=providers[1],
            browser=providers[2],
            archive=providers[3],
            policy=PageRetryPolicy(max_attempts=1),
        ).retrieve(request())

    assert [attempt.error_code for attempt in caught.value.attempts] == [
        "javascript_shell",
        "reader_incomplete",
        "browser_failed",
        "archive_missing",
    ]
    assert raw_secret not in repr(caught.value)


@pytest.mark.asyncio
async def test_invalid_and_malformed_provider_output_stops_fallback() -> None:
    secondary = ScriptedPageProvider("reader", PageStrategy.READER, [])
    invalid = ScriptedPageProvider(
        "direct",
        PageStrategy.DIRECT,
        [PageRetrievalError(PageFailureKind.INVALID_REQUEST, code="bad_url")],
    )
    with pytest.raises(PageRetrievalError):
        await PageRetrievalLadder(
            direct=invalid,
            reader=secondary,
            policy=PageRetryPolicy(max_attempts=1),
        ).retrieve(request())
    assert secondary.requests == []

    malformed = ScriptedPageProvider(
        "malformed",
        PageStrategy.DIRECT,
        [
            {
                **document_payload("malformed", request(), strategy=PageStrategy.DIRECT),
                "requested_url": "https://example.org/different",
            }
        ],
    )
    with pytest.raises(ProviderContractError, match="different requested URL"):
        await PageRetrievalLadder(
            direct=malformed,
            reader=secondary,
            policy=PageRetryPolicy(max_attempts=1),
        ).retrieve(request())
    assert secondary.requests == []


def test_ladder_rejects_provider_in_wrong_strategy_slot() -> None:
    browser = ScriptedPageProvider("browser", PageStrategy.BROWSER, [])

    with pytest.raises(ValueError, match="strategy direct"):
        PageRetrievalLadder(direct=browser)
