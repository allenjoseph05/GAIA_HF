"""Direct HTTP extraction and typed reader/browser/archive fallback routing."""

from __future__ import annotations

import asyncio
import ipaddress
import random
import re
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, date, datetime
from enum import StrEnum
from html.parser import HTMLParser
from typing import ClassVar, Protocol
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from gaia_max.artifacts import ArtifactStore
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource
from gaia_max.retrieval.providers import (
    DocumentLink,
    PageRetrievalProvider,
    PageRetrievalRequest,
    ProviderContractError,
    RetrievedDocument,
    retrieve_document,
)

AsyncSleep = Callable[[float], Awaitable[None]]
RandomFraction = Callable[[], float]

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,98}[A-Za-z0-9])?")
_SAFE_ERROR_CODE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_MEDIA_TYPES = {"text/plain"}
_BLOCKED_PHRASES = (
    "access denied",
    "attention required",
    "checking your browser",
    "enable cookies",
    "unusual traffic",
    "verify you are human",
    "captcha",
)
_BLOCKED_HEADERS = ("cf-mitigated", "x-sucuri-block", "x-captcha")
_DATE_META_NAMES = {
    "article:published_time",
    "date",
    "datecreated",
    "datepublished",
    "dc.date",
    "dc.date.issued",
    "publication_date",
}


class PageStrategy(StrEnum):
    """Ordered page-acquisition mechanisms in the C03 ladder."""

    DIRECT = "direct"
    READER = "reader"
    BROWSER = "browser"
    ARCHIVE = "archive"


class PageFailureKind(StrEnum):
    """Stable failure classes that decide the next retrieval edge."""

    TRANSIENT = "transient"
    NOT_FOUND = "not_found"
    BLOCKED = "blocked"
    JS_REQUIRED = "js_required"
    EXTRACTION_INCOMPLETE = "extraction_incomplete"
    UNSUPPORTED_CONTENT = "unsupported_content"
    INVALID_REQUEST = "invalid_request"
    PERMANENT = "permanent"


class PageRetrievalError(RuntimeError):
    """Sanitized strategy failure; raw page bodies never enter control state."""

    def __init__(
        self,
        kind: PageFailureKind,
        *,
        code: str,
        artifact_id: str | None = None,
    ) -> None:
        if _SAFE_ERROR_CODE.fullmatch(code) is None:
            raise ValueError("page retrieval error code must be a safe identifier")
        if artifact_id is not None and re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None:
            raise ValueError("page retrieval error artifact must be a SHA-256 identifier")
        self.kind = kind
        self.code = code
        self.artifact_id = artifact_id
        super().__init__(f"page retrieval failure: {kind.value}/{code}")


class StrategyPageProvider(PageRetrievalProvider, Protocol):
    """Page adapter that declares which ladder mechanism it implements."""

    @property
    def strategy(self) -> PageStrategy: ...


class PageRetryPolicy(BaseModel):
    """Bounded retry policy shared by every retrieval strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=2, ge=1, le=10)
    retry_base_seconds: float = Field(default=0.5, ge=0, le=60)
    retry_max_seconds: float = Field(default=4.0, ge=0, le=300)
    jitter_ratio: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_retry_window(self) -> PageRetryPolicy:
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("maximum page retry delay cannot be smaller than base delay")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> PageRetryPolicy:
        return cls(
            max_attempts=settings.page_max_attempts,
            retry_base_seconds=settings.page_retry_base_seconds,
            retry_max_seconds=settings.page_retry_max_seconds,
            jitter_ratio=settings.page_retry_jitter_ratio,
        )


class PageAttemptStatus(StrEnum):
    """Answer-free status for one actual strategy call."""

    SUCCESS = "success"
    FALLBACK_REQUIRED = "fallback_required"
    TRANSIENT_FAILURE = "transient_failure"


class PageAttempt(BaseModel):
    """Sanitized audit record for one retrieval attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: PageStrategy
    provider: str = Field(min_length=1, max_length=100)
    attempt_number: int = Field(ge=1, le=10)
    status: PageAttemptStatus
    failure_kind: PageFailureKind | None = None
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    raw_artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    retry_delay_seconds: float | None = Field(default=None, ge=0, le=300)

    @model_validator(mode="after")
    def validate_status_fields(self) -> PageAttempt:
        if self.status is PageAttemptStatus.SUCCESS:
            if any(
                value is not None
                for value in (
                    self.failure_kind,
                    self.error_code,
                    self.retry_delay_seconds,
                )
            ):
                raise ValueError("successful page attempt cannot include failure details")
        elif self.failure_kind is None or self.error_code is None:
            raise ValueError("failed page attempt requires kind and safe error code")
        return self


class PageRetrievalOutcome(BaseModel):
    """One validated document and the exact successful ladder path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: RetrievedDocument
    attempts: tuple[PageAttempt, ...] = Field(min_length=1)
    winning_strategy: PageStrategy

    @model_validator(mode="after")
    def require_matching_success(self) -> PageRetrievalOutcome:
        if self.attempts[-1].status is not PageAttemptStatus.SUCCESS:
            raise ValueError("retrieval outcome must end in a successful attempt")
        if self.attempts[-1].strategy is not self.winning_strategy:
            raise ValueError("winning strategy must match final successful attempt")
        return self


class PageLadderExhaustedError(RuntimeError):
    """All applicable strategies failed without producing an untrusted placeholder."""

    def __init__(
        self,
        requested_url: str,
        attempts: tuple[PageAttempt, ...],
    ) -> None:
        self.requested_url = requested_url
        self.attempts = attempts
        codes = ",".join(attempt.error_code or "success" for attempt in attempts)
        super().__init__(f"page retrieval ladder exhausted ({codes})")


class PageRetrievalLadder:
    """Route direct failures to the least-expensive applicable fallback strategy."""

    def __init__(
        self,
        *,
        direct: StrategyPageProvider,
        reader: StrategyPageProvider | None = None,
        browser: StrategyPageProvider | None = None,
        archive: StrategyPageProvider | None = None,
        policy: PageRetryPolicy | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        random_fraction: RandomFraction = random.random,
    ) -> None:
        self._direct = self._validate_provider(direct, PageStrategy.DIRECT)
        self._reader = self._validate_optional_provider(reader, PageStrategy.READER)
        self._browser = self._validate_optional_provider(browser, PageStrategy.BROWSER)
        self._archive = self._validate_optional_provider(archive, PageStrategy.ARCHIVE)
        names = tuple(
            provider.name
            for provider in (self._direct, self._reader, self._browser, self._archive)
            if provider is not None
        )
        if len(set(names)) != len(names):
            raise ValueError("page ladder providers must have distinct names")
        self._policy = policy or PageRetryPolicy()
        self._sleep = sleep
        self._random_fraction = random_fraction

    async def retrieve(self, request: PageRetrievalRequest) -> PageRetrievalOutcome:
        attempts: list[PageAttempt] = []
        document, direct_failure = await self._run_strategy(
            self._direct,
            request,
            attempts,
        )
        if document is not None:
            return self._outcome(document, attempts, PageStrategy.DIRECT)
        if direct_failure is None:
            raise RuntimeError("direct retrieval returned neither document nor failure")

        for provider in self._fallbacks_for(direct_failure):
            if provider is None:
                continue
            document, _failure = await self._run_strategy(provider, request, attempts)
            if document is not None:
                return self._outcome(document, attempts, provider.strategy)

        raise PageLadderExhaustedError(str(request.url), tuple(attempts))

    async def _run_strategy(
        self,
        provider: StrategyPageProvider,
        request: PageRetrievalRequest,
        attempts: list[PageAttempt],
    ) -> tuple[RetrievedDocument | None, PageFailureKind | None]:
        for attempt_number in range(1, self._policy.max_attempts + 1):
            try:
                document = await retrieve_document(provider, request)
            except ProviderContractError:
                raise
            except PageRetrievalError as exc:
                if exc.kind is PageFailureKind.INVALID_REQUEST:
                    raise
                retryable = exc.kind is PageFailureKind.TRANSIENT
                delay = None
                if retryable and attempt_number < self._policy.max_attempts:
                    delay = self._retry_delay(attempt_number)
                attempts.append(
                    PageAttempt(
                        strategy=provider.strategy,
                        provider=provider.name,
                        attempt_number=attempt_number,
                        status=(
                            PageAttemptStatus.TRANSIENT_FAILURE
                            if retryable
                            else PageAttemptStatus.FALLBACK_REQUIRED
                        ),
                        failure_kind=exc.kind,
                        error_code=exc.code,
                        raw_artifact_id=exc.artifact_id,
                        retry_delay_seconds=delay,
                    )
                )
                if delay is not None:
                    await self._sleep(delay)
                    continue
                return None, exc.kind
            attempts.append(
                PageAttempt(
                    strategy=provider.strategy,
                    provider=provider.name,
                    attempt_number=attempt_number,
                    status=PageAttemptStatus.SUCCESS,
                    raw_artifact_id=document.artifact_id,
                )
            )
            return document, None
        raise RuntimeError("unreachable page retry state")

    def _fallbacks_for(
        self,
        failure: PageFailureKind,
    ) -> tuple[StrategyPageProvider | None, ...]:
        if failure is PageFailureKind.NOT_FOUND:
            return (self._archive,)
        if failure is PageFailureKind.BLOCKED:
            return (self._reader, self._archive)
        return (self._reader, self._browser, self._archive)

    def _retry_delay(self, completed_attempt: int) -> float:
        ceiling = min(
            self._policy.retry_max_seconds,
            self._policy.retry_base_seconds * 2 ** (completed_attempt - 1),
        )
        jitter = 1 + self._policy.jitter_ratio * (2 * self._random_fraction() - 1)
        return min(self._policy.retry_max_seconds, max(0.0, ceiling * jitter))

    @staticmethod
    def _outcome(
        document: RetrievedDocument,
        attempts: list[PageAttempt],
        strategy: PageStrategy,
    ) -> PageRetrievalOutcome:
        return PageRetrievalOutcome(
            document=document,
            attempts=tuple(attempts),
            winning_strategy=strategy,
        )

    @staticmethod
    def _validate_provider(
        provider: StrategyPageProvider,
        expected_strategy: PageStrategy,
    ) -> StrategyPageProvider:
        name = getattr(provider, "name", None)
        if not isinstance(name, str) or _SAFE_IDENTIFIER.fullmatch(name) is None:
            raise ValueError("page provider name must be a safe stable identifier")
        if getattr(provider, "strategy", None) is not expected_strategy:
            raise ValueError(f"page provider must declare strategy {expected_strategy.value}")
        return provider

    @classmethod
    def _validate_optional_provider(
        cls,
        provider: StrategyPageProvider | None,
        expected_strategy: PageStrategy,
    ) -> StrategyPageProvider | None:
        if provider is None:
            return None
        return cls._validate_provider(provider, expected_strategy)


class _StructuredHTMLParser(HTMLParser):
    """Small deterministic extractor that omits scripts and navigation chrome."""

    _SUPPRESSED_TAGS: ClassVar[set[str]] = {
        "aside",
        "footer",
        "form",
        "header",
        "head",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
        "template",
    }
    _BLOCK_TAGS: ClassVar[set[str]] = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str | None]] = []
        self.published_at: date | None = None
        self.script_count = 0
        self.js_marker = False
        self._suppressed_stack: list[str] = []
        self._in_title = False
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if normalized_tag == "title":
            self._in_title = True
        if normalized_tag == "script":
            self.script_count += 1
        if normalized_tag == "noscript":
            self.js_marker = True
        if (
            attributes.get("id", "").casefold() in {"app", "root", "__next"}
            or "data-reactroot" in attributes
            or "ng-app" in attributes
        ):
            self.js_marker = True
        if normalized_tag == "meta":
            self._read_meta_date(attributes)
        if normalized_tag in self._SUPPRESSED_TAGS:
            self._suppressed_stack.append(normalized_tag)
            return
        if self._suppressed_stack:
            return
        if normalized_tag in self._BLOCK_TAGS:
            self.text_parts.append("\n")
        if normalized_tag == "a":
            self._anchor_href = attributes.get("href") or None
            self._anchor_text = []

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag == "title":
            self._in_title = False
        if self._suppressed_stack:
            if normalized_tag == self._suppressed_stack[-1]:
                self._suppressed_stack.pop()
            return
        if normalized_tag == "a" and self._anchor_href is not None:
            self._store_link(self._anchor_href, " ".join(self._anchor_text))
            self._anchor_href = None
            self._anchor_text = []
        if normalized_tag in self._BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._suppressed_stack:
            return
        self.text_parts.append(data)
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    @property
    def title(self) -> str | None:
        normalized = _normalize_inline_text(" ".join(self.title_parts))
        return normalized or None

    @property
    def text(self) -> str:
        return _normalize_extracted_text("".join(self.text_parts))

    def document_links(self, *, limit: int = 500) -> tuple[DocumentLink, ...]:
        seen: set[str] = set()
        documents: list[DocumentLink] = []
        for url, text in self.links:
            if url in seen:
                continue
            seen.add(url)
            documents.append(DocumentLink(url=HttpUrl(url), text=text))
            if len(documents) >= limit:
                break
        return tuple(documents)

    def _store_link(self, href: str, text: str) -> None:
        absolute = urljoin(self.base_url, href.strip())
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return
        normalized_text = _normalize_inline_text(text)
        self.links.append((absolute, normalized_text or None))

    def _read_meta_date(self, attributes: dict[str, str]) -> None:
        raw_name = (
            attributes.get("property")
            or attributes.get("name")
            or attributes.get("itemprop")
            or ""
        )
        name = raw_name.casefold()
        if name not in _DATE_META_NAMES:
            return
        content = attributes.get("content", "").strip()
        parsed = _parse_publication_date(content)
        if parsed is not None:
            self.published_at = parsed


class StaticHttpPageProvider:
    """Direct HTTP adapter with bounded bytes and deterministic HTML/text extraction."""

    name = "static-http"
    strategy = PageStrategy.DIRECT

    def __init__(
        self,
        *,
        store: ArtifactStore,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(UTC))
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "gaia-max/0.1 evidence-retriever"},
        )

    async def __aenter__(self) -> StaticHttpPageProvider:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def retrieve(self, request: PageRetrievalRequest) -> object:
        requested_url = str(request.url)
        _validate_fetch_url(requested_url, code="unsafe_request_url")
        response = await self._fetch_with_redirects(requested_url)
        content = response.content
        if not content:
            raise PageRetrievalError(
                PageFailureKind.EXTRACTION_INCOMPLETE,
                code="empty_response",
            )
        if len(content) > self._settings.max_page_bytes:
            raise PageRetrievalError(
                PageFailureKind.PERMANENT,
                code="page_too_large",
            )

        retrieved_at = self._clock()
        final_url = str(response.url)
        media_type = _response_media_type(response)
        raw_artifact = self._store.put_bytes(
            content,
            original_name=_raw_artifact_name(media_type),
            media_type=media_type,
            source=ArtifactSource.WEB_RETRIEVAL,
            retrieved_at=retrieved_at,
            source_locator=final_url,
        )

        if _response_indicates_block(response):
            raise PageRetrievalError(
                PageFailureKind.BLOCKED,
                code="blocked_response_header",
                artifact_id=raw_artifact.artifact_id,
            )
        if media_type in _HTML_MEDIA_TYPES or _looks_like_html(content):
            return self._extract_html(
                request,
                response,
                content,
                media_type,
                raw_artifact.artifact_id,
                retrieved_at,
            )
        if media_type in _TEXT_MEDIA_TYPES:
            return self._extract_plain_text(
                request,
                response,
                raw_artifact.artifact_id,
                retrieved_at,
            )
        raise PageRetrievalError(
            PageFailureKind.UNSUPPORTED_CONTENT,
            code="unsupported_media_type",
            artifact_id=raw_artifact.artifact_id,
        )

    async def _fetch_with_redirects(self, requested_url: str) -> httpx.Response:
        current_url = requested_url
        visited: set[str] = set()
        for redirect_count in range(self._settings.page_max_redirects + 1):
            if current_url in visited:
                raise PageRetrievalError(
                    PageFailureKind.PERMANENT,
                    code="redirect_loop",
                )
            visited.add(current_url)
            response = await self._fetch_once(current_url)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                raise PageRetrievalError(
                    PageFailureKind.PERMANENT,
                    code="redirect_without_location",
                )
            if redirect_count == self._settings.page_max_redirects:
                raise PageRetrievalError(
                    PageFailureKind.PERMANENT,
                    code="too_many_redirects",
                )
            current_url = urljoin(current_url, location)
            _validate_fetch_url(current_url, code="unsafe_redirect_target")
        raise RuntimeError("unreachable redirect state")

    async def _fetch_once(self, url: str) -> httpx.Response:
        try:
            async with self._client.stream(
                "GET",
                url,
                follow_redirects=False,
            ) as streamed:
                if streamed.status_code in {301, 302, 303, 307, 308}:
                    return httpx.Response(
                        streamed.status_code,
                        headers=streamed.headers,
                        request=streamed.request,
                    )
                self._raise_for_status(streamed)
                declared_length = _content_length(streamed)
                if (
                    declared_length is not None
                    and declared_length > self._settings.max_page_bytes
                ):
                    raise PageRetrievalError(
                        PageFailureKind.PERMANENT,
                        code="page_too_large",
                    )
                chunks: list[bytes] = []
                received = 0
                async for chunk in streamed.aiter_bytes():
                    received += len(chunk)
                    if received > self._settings.max_page_bytes:
                        raise PageRetrievalError(
                            PageFailureKind.PERMANENT,
                            code="page_too_large",
                        )
                    chunks.append(chunk)
                return httpx.Response(
                    streamed.status_code,
                    headers=streamed.headers,
                    content=b"".join(chunks),
                    request=streamed.request,
                )
        except httpx.TransportError as exc:
            raise PageRetrievalError(
                PageFailureKind.TRANSIENT,
                code="http_transport_failure",
            ) from exc

    def _extract_html(
        self,
        request: PageRetrievalRequest,
        response: httpx.Response,
        content: bytes,
        media_type: str,
        raw_artifact_id: str,
        retrieved_at: datetime,
    ) -> RetrievedDocument:
        parser = _StructuredHTMLParser(str(response.url))
        try:
            parser.feed(response.text)
            parser.close()
        except (UnicodeError, ValueError) as exc:
            raise PageRetrievalError(
                PageFailureKind.EXTRACTION_INCOMPLETE,
                code="html_parse_failure",
                artifact_id=raw_artifact_id,
            ) from exc

        text = parser.text
        blocked_title = (parser.title or "").casefold()
        blocked_lead = text[:1000].casefold()
        if any(phrase in blocked_title for phrase in _BLOCKED_PHRASES) or (
            len(text) < 2000
            and any(phrase in blocked_lead for phrase in _BLOCKED_PHRASES)
        ):
            raise PageRetrievalError(
                PageFailureKind.BLOCKED,
                code="blocked_response_content",
                artifact_id=raw_artifact_id,
            )
        if len(text) < self._settings.page_min_extracted_chars:
            js_likely = parser.js_marker or parser.script_count > 0
            raise PageRetrievalError(
                (
                    PageFailureKind.JS_REQUIRED
                    if js_likely
                    else PageFailureKind.EXTRACTION_INCOMPLETE
                ),
                code="javascript_shell" if js_likely else "insufficient_static_text",
                artifact_id=raw_artifact_id,
            )

        text_artifact = self._store.put_bytes(
            text.encode("utf-8"),
            original_name="extracted-page.txt",
            media_type="text/plain; charset=utf-8",
            source=ArtifactSource.DERIVED,
            retrieved_at=retrieved_at,
            source_locator=str(response.url),
        )
        return RetrievedDocument(
            provider=self.name,
            requested_url=request.url,
            final_url=HttpUrl(str(response.url)),
            status_code=response.status_code,
            title=parser.title,
            published_at=parser.published_at,
            retrieved_at=retrieved_at,
            artifact_id=raw_artifact_id,
            text_artifact_id=text_artifact.artifact_id,
            links=parser.document_links(),
            extraction_method=(
                "static_xhtml" if media_type == "application/xhtml+xml" else "static_html"
            ),
            injection_flags=(),
        )

    def _extract_plain_text(
        self,
        request: PageRetrievalRequest,
        response: httpx.Response,
        raw_artifact_id: str,
        retrieved_at: datetime,
    ) -> RetrievedDocument:
        text = _normalize_extracted_text(response.text)
        if not text:
            raise PageRetrievalError(
                PageFailureKind.EXTRACTION_INCOMPLETE,
                code="empty_plain_text",
                artifact_id=raw_artifact_id,
            )
        text_artifact = self._store.put_bytes(
            text.encode("utf-8"),
            original_name="extracted-page.txt",
            media_type="text/plain; charset=utf-8",
            source=ArtifactSource.DERIVED,
            retrieved_at=retrieved_at,
            source_locator=str(response.url),
        )
        return RetrievedDocument(
            provider=self.name,
            requested_url=request.url,
            final_url=HttpUrl(str(response.url)),
            status_code=response.status_code,
            retrieved_at=retrieved_at,
            artifact_id=raw_artifact_id,
            text_artifact_id=text_artifact.artifact_id,
            links=(),
            extraction_method="static_plain_text",
            injection_flags=(),
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status == 404 or status == 410:
            raise PageRetrievalError(PageFailureKind.NOT_FOUND, code=f"http_{status}")
        if status in {401, 403, 451}:
            raise PageRetrievalError(PageFailureKind.BLOCKED, code=f"http_{status}")
        if status == 429 or status >= 500:
            raise PageRetrievalError(PageFailureKind.TRANSIENT, code=f"http_{status}")
        if response.is_error:
            raise PageRetrievalError(PageFailureKind.PERMANENT, code=f"http_{status}")


def _normalize_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_extracted_text(value: str) -> str:
    lines = (_normalize_inline_text(line) for line in value.splitlines())
    return "\n".join(line for line in lines if line)


def _parse_publication_date(value: str) -> date | None:
    if not value:
        return None
    candidate = value[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _response_media_type(response: httpx.Response) -> str:
    value = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if value:
        return value
    return "text/html" if _looks_like_html(response.content) else "application/octet-stream"


def _looks_like_html(content: bytes) -> bool:
    prefix = content[:512].lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def _response_indicates_block(response: httpx.Response) -> bool:
    return any(header in response.headers for header in _BLOCKED_HEADERS)


def _raw_artifact_name(media_type: str) -> str:
    if media_type in _HTML_MEDIA_TYPES:
        return "retrieved-page.html"
    if media_type in _TEXT_MEDIA_TYPES:
        return "retrieved-page.txt"
    return "retrieved-page.bin"


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        return None
    return length if length >= 0 else None


def _validate_fetch_url(value: str, *, code: str) -> None:
    parsed = urlparse(value)
    hostname = parsed.hostname
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PageRetrievalError(PageFailureKind.INVALID_REQUEST, code=code)
    normalized_host = hostname.casefold().rstrip(".")
    if normalized_host == "localhost" or normalized_host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise PageRetrievalError(PageFailureKind.INVALID_REQUEST, code=code)
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return
    if not address.is_global:
        raise PageRetrievalError(PageFailureKind.INVALID_REQUEST, code=code)


def page_strategy_names(
    providers: Iterable[StrategyPageProvider],
) -> tuple[str, ...]:
    """Return stable strategy names for safe diagnostics and demonstrations."""

    return tuple(f"{provider.strategy.value}:{provider.name}" for provider in providers)


__all__ = [
    "PageAttempt",
    "PageAttemptStatus",
    "PageFailureKind",
    "PageLadderExhaustedError",
    "PageRetrievalError",
    "PageRetrievalLadder",
    "PageRetrievalOutcome",
    "PageRetryPolicy",
    "PageStrategy",
    "StaticHttpPageProvider",
    "StrategyPageProvider",
    "page_strategy_names",
]
