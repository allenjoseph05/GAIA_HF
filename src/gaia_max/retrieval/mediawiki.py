"""Revision-aware MediaWiki retrieval with exact temporal provenance."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import quote, urlparse

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictStr,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from gaia_max.artifacts import ArtifactStore
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource
from gaia_max.domain.models import SHA256_PATTERN

AsyncSleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]

_HTTP_URL = TypeAdapter(HttpUrl)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class MediaWikiError(RuntimeError):
    """Sanitized MediaWiki failure that never exposes response content."""

    def __init__(self, code: str) -> None:
        if _SAFE_CODE.fullmatch(code) is None:
            raise ValueError("MediaWiki error code must be a safe identifier")
        self.code = code
        super().__init__(f"MediaWiki request failed: {code}")


class MediaWikiNotFoundError(MediaWikiError):
    """The requested page or revision does not exist."""


class MediaWikiTemporalError(MediaWikiError):
    """No eligible revision exists at or before the requested cutoff."""


class MediaWikiSearchResult(BaseModel):
    """One page-discovery result from a specific wiki."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, le=500)
    page_id: int = Field(gt=0)
    title: StrictStr = Field(min_length=1, max_length=1000)
    page_url: HttpUrl

    @field_serializer("page_url")
    def serialize_url(self, value: HttpUrl) -> str:
        return str(value)


class MediaWikiRevisionRef(BaseModel):
    """Immutable identity and timestamp for one MediaWiki revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: int = Field(gt=0)
    title: StrictStr = Field(min_length=1, max_length=1000)
    revision_id: int = Field(gt=0)
    parent_revision_id: int | None = Field(default=None, ge=0)
    timestamp: AwareDatetime
    sha1: str | None = Field(default=None, pattern=_SHA1.pattern)
    revision_url: HttpUrl

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_serializer("revision_url")
    def serialize_url(self, value: HttpUrl) -> str:
        return str(value)


class MediaWikiRevisionSelection(BaseModel):
    """A dated page request bound to the latest eligible exact revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_title: StrictStr = Field(min_length=1, max_length=1000)
    resolved_title: StrictStr = Field(min_length=1, max_length=1000)
    cutoff: AwareDatetime
    revision: MediaWikiRevisionRef

    @field_validator("cutoff")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


class MediaWikiRevisionContent(BaseModel):
    """Exact old-revision wikitext stored as an immutable artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection: MediaWikiRevisionSelection
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    retrieved_at: AwareDatetime
    content_model: StrictStr = Field(min_length=1, max_length=100)
    content_format: StrictStr = Field(min_length=1, max_length=100)

    @field_validator("retrieved_at")
    @classmethod
    def normalize_retrieval_time(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


def select_latest_revision(
    revisions: Sequence[MediaWikiRevisionRef],
    cutoff: datetime,
) -> MediaWikiRevisionRef:
    """Select the latest revision whose timestamp is inclusively before cutoff."""

    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("MediaWiki cutoff must be timezone-aware")
    cutoff_utc = cutoff.astimezone(UTC)
    revision_ids = [revision.revision_id for revision in revisions]
    if len(set(revision_ids)) != len(revision_ids):
        raise ValueError("MediaWiki revision history contains duplicate IDs")
    eligible = [revision for revision in revisions if revision.timestamp <= cutoff_utc]
    if not eligible:
        raise MediaWikiTemporalError("no_revision_before_cutoff")
    return max(eligible, key=lambda revision: (revision.timestamp, revision.revision_id))


class MediaWikiClient:
    """Read-only Action API adapter for dated revision discovery and content."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        settings: Settings,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        http_client: httpx.AsyncClient | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        parsed_url = _HTTP_URL.validate_python(api_url)
        if parsed_url.scheme != "https":
            raise ValueError("MediaWiki API URL must use HTTPS")
        self._api_url = str(parsed_url)
        parsed = urlparse(self._api_url)
        self._origin = f"{parsed.scheme}://{parsed.netloc}"
        self._store = store
        self._settings = settings
        self._sleep = sleep
        self._clock = clock
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "gaia-max-score-agent/0.1 (historical research)"},
        )

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def search_pages(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[MediaWikiSearchResult, ...]:
        """Search page titles/content; results are discovery evidence only."""

        if not query or query != query.strip() or len(query) > 500:
            raise ValueError("MediaWiki search query must be non-empty and trimmed")
        if not 1 <= limit <= 50:
            raise ValueError("MediaWiki search limit must be between 1 and 50")
        payload = await self._request(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
                "srprop": "",
            }
        )
        query_data = self._mapping(payload.get("query"), "missing_query")
        raw_results = self._sequence(query_data.get("search"), "missing_search_results")
        results: list[MediaWikiSearchResult] = []
        for rank, raw in enumerate(raw_results, start=1):
            item = self._mapping(raw, "invalid_search_result")
            page_id = self._positive_int(item.get("pageid"), "invalid_search_page_id")
            title = self._text(item.get("title"), "invalid_search_title")
            results.append(
                MediaWikiSearchResult(
                    rank=rank,
                    page_id=page_id,
                    title=title,
                    page_url=self._page_url(title),
                )
            )
        if len(results) > limit:
            raise MediaWikiError("search_limit_exceeded")
        return tuple(results)

    async def latest_revision_at_or_before(
        self,
        title: str,
        cutoff: datetime,
    ) -> MediaWikiRevisionSelection:
        """Resolve redirects and bind a title to its latest inclusive revision."""

        requested_title = self._validated_title(title)
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("MediaWiki cutoff must be timezone-aware")
        cutoff_utc = cutoff.astimezone(UTC)
        continuation: str | None = None
        resolved_title: str | None = None
        candidates: list[MediaWikiRevisionRef] = []

        for _page_number in range(self._settings.mediawiki_max_history_pages):
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": requested_title,
                "redirects": "1",
                "rvprop": "ids|timestamp|sha1",
                "rvlimit": str(self._settings.mediawiki_revision_batch_size),
                "rvdir": "older",
                "rvstart": self._timestamp(cutoff_utc),
            }
            if continuation is not None:
                params["rvcontinue"] = continuation
            payload = await self._request(params)
            page, current_title = self._single_page(payload)
            if resolved_title is not None and current_title != resolved_title:
                raise MediaWikiError("history_page_identity_changed")
            resolved_title = current_title
            page_id = self._positive_int(page.get("pageid"), "invalid_page_id")
            raw_revisions = self._sequence(page.get("revisions", ()), "invalid_revisions")
            candidates.extend(
                self._revision_ref(page_id, resolved_title, raw)
                for raw in raw_revisions
            )
            if candidates:
                revision = select_latest_revision(candidates, cutoff_utc)
                return MediaWikiRevisionSelection(
                    requested_title=requested_title,
                    resolved_title=resolved_title,
                    cutoff=cutoff_utc,
                    revision=revision,
                )
            continuation = self._continuation(payload)
            if continuation is None:
                break

        raise MediaWikiTemporalError("no_revision_before_cutoff")

    async def fetch_revision_content(
        self,
        selection: MediaWikiRevisionSelection,
    ) -> MediaWikiRevisionContent:
        """Fetch and store the exact selected revision, rejecting identity drift."""

        expected = selection.revision
        payload = await self._request(
            {
                "action": "query",
                "prop": "revisions",
                "revids": str(expected.revision_id),
                "rvprop": "ids|timestamp|sha1|content|contentmodel",
                "rvslots": "main",
            }
        )
        page, resolved_title = self._single_page(payload)
        page_id = self._positive_int(page.get("pageid"), "invalid_page_id")
        raw_revisions = self._sequence(page.get("revisions"), "missing_revision_content")
        if len(raw_revisions) != 1:
            raise MediaWikiError("revision_content_cardinality")
        actual = self._revision_ref(page_id, resolved_title, raw_revisions[0])
        if (
            actual.revision_id != expected.revision_id
            or actual.page_id != expected.page_id
            or actual.timestamp != expected.timestamp
            or (expected.sha1 is not None and actual.sha1 != expected.sha1)
        ):
            raise MediaWikiError("revision_identity_mismatch")

        revision_data = self._mapping(raw_revisions[0], "invalid_revision_content")
        slots = self._mapping(revision_data.get("slots"), "missing_revision_slots")
        main = self._mapping(slots.get("main"), "missing_main_slot")
        content = self._text(main.get("content"), "missing_revision_text", allow_empty=True)
        if not content:
            raise MediaWikiError("empty_revision_text")
        content_model = self._text(
            main.get("contentmodel", revision_data.get("contentmodel", "wikitext")),
            "invalid_content_model",
        )
        content_format = self._text(
            main.get("contentformat", "text/x-wiki"),
            "invalid_content_format",
        )
        retrieved_at = self._clock()
        if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
            raise ValueError("MediaWiki clock must return a timezone-aware datetime")
        artifact = self._store.put_bytes(
            content.encode("utf-8"),
            original_name=f"revision-{expected.revision_id}.wikitext",
            media_type=content_format,
            source=ArtifactSource.WEB_RETRIEVAL,
            retrieved_at=retrieved_at,
            source_locator=str(expected.revision_url),
        )
        return MediaWikiRevisionContent(
            selection=selection,
            artifact_id=artifact.artifact_id,
            retrieved_at=retrieved_at,
            content_model=content_model,
            content_format=content_format,
        )

    async def fetch_page_at_or_before(
        self,
        title: str,
        cutoff: datetime,
    ) -> MediaWikiRevisionContent:
        """Resolve and fetch a historical page in one provenance-safe operation."""

        selection = await self.latest_revision_at_or_before(title, cutoff)
        return await self.fetch_revision_content(selection)

    async def _request(self, params: Mapping[str, str]) -> Mapping[str, Any]:
        complete_params = {
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
            **params,
        }
        for attempt in range(1, self._settings.page_max_attempts + 1):
            try:
                response = await self._client.get(self._api_url, params=complete_params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self._settings.page_max_attempts:
                    raise MediaWikiError("network_exhausted") from exc
                await self._sleep(self._retry_delay(attempt))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self._settings.page_max_attempts:
                    raise MediaWikiError("transient_http_exhausted")
                await self._sleep(self._retry_delay(attempt))
                continue
            if response.status_code == 404:
                raise MediaWikiNotFoundError("api_not_found")
            if not 200 <= response.status_code < 300:
                raise MediaWikiError("permanent_http_error")
            if len(response.content) > self._settings.max_page_bytes:
                raise MediaWikiError("response_too_large")
            try:
                decoded = response.json()
            except ValueError as exc:
                raise MediaWikiError("invalid_json") from exc
            payload = self._mapping(decoded, "invalid_payload")
            if "error" in payload:
                raise MediaWikiError("api_error")
            return payload
        raise RuntimeError("unreachable MediaWiki retry state")

    def _retry_delay(self, failed_attempt: int) -> float:
        return min(
            self._settings.page_retry_base_seconds * (2 ** (failed_attempt - 1)),
            self._settings.page_retry_max_seconds,
        )

    def _single_page(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
        query = self._mapping(payload.get("query"), "missing_query")
        pages = self._sequence(query.get("pages"), "missing_pages")
        if len(pages) != 1:
            raise MediaWikiError("page_cardinality")
        page = self._mapping(pages[0], "invalid_page")
        if "missing" in page or "invalid" in page:
            raise MediaWikiNotFoundError("page_not_found")
        title = self._text(page.get("title"), "invalid_page_title")
        return page, title

    def _revision_ref(
        self,
        page_id: int,
        title: str,
        raw: object,
    ) -> MediaWikiRevisionRef:
        item = self._mapping(raw, "invalid_revision")
        revision_id = self._positive_int(item.get("revid"), "invalid_revision_id")
        parent_raw = item.get("parentid")
        parent_id = None
        if parent_raw is not None:
            if not isinstance(parent_raw, int) or isinstance(parent_raw, bool) or parent_raw < 0:
                raise MediaWikiError("invalid_parent_revision_id")
            parent_id = parent_raw
        timestamp = self._parse_timestamp(item.get("timestamp"))
        sha1_raw = item.get("sha1")
        sha1 = None
        if sha1_raw is not None:
            if not isinstance(sha1_raw, str) or _SHA1.fullmatch(sha1_raw) is None:
                raise MediaWikiError("invalid_revision_sha1")
            sha1 = sha1_raw
        return MediaWikiRevisionRef(
            page_id=page_id,
            title=title,
            revision_id=revision_id,
            parent_revision_id=parent_id,
            timestamp=timestamp,
            sha1=sha1,
            revision_url=self._revision_url(revision_id),
        )

    def _continuation(self, payload: Mapping[str, Any]) -> str | None:
        raw = payload.get("continue")
        if raw is None:
            return None
        continuation = self._mapping(raw, "invalid_continuation")
        value = continuation.get("rvcontinue")
        if not isinstance(value, str) or not value or len(value) > 500:
            raise MediaWikiError("invalid_revision_continuation")
        return value

    def _parse_timestamp(self, value: object) -> datetime:
        if not isinstance(value, str):
            raise MediaWikiError("invalid_revision_timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MediaWikiError("invalid_revision_timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise MediaWikiError("naive_revision_timestamp")
        return parsed.astimezone(UTC)

    def _page_url(self, title: str) -> HttpUrl:
        return _HTTP_URL.validate_python(
            f"{self._origin}/wiki/{quote(title.replace(' ', '_'), safe='/:')}"
        )

    def _revision_url(self, revision_id: int) -> HttpUrl:
        return _HTTP_URL.validate_python(
            f"{self._origin}/w/index.php?oldid={revision_id}"
        )

    @staticmethod
    def _mapping(value: object, code: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise MediaWikiError(code)
        return cast(Mapping[str, Any], value)

    @staticmethod
    def _sequence(value: object, code: str) -> Sequence[object]:
        if not isinstance(value, list | tuple):
            raise MediaWikiError(code)
        return cast(Sequence[object], value)

    @staticmethod
    def _positive_int(value: object, code: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MediaWikiError(code)
        return value

    @staticmethod
    def _text(value: object, code: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str) or (not allow_empty and not value):
            raise MediaWikiError(code)
        return value

    @staticmethod
    def _validated_title(title: str) -> str:
        if not title or title != title.strip() or len(title) > 1000 or "\x00" in title:
            raise ValueError("MediaWiki title must be non-empty, trimmed, and bounded")
        return title

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "MediaWikiClient",
    "MediaWikiError",
    "MediaWikiNotFoundError",
    "MediaWikiRevisionContent",
    "MediaWikiRevisionRef",
    "MediaWikiRevisionSelection",
    "MediaWikiSearchResult",
    "MediaWikiTemporalError",
    "select_latest_revision",
]
