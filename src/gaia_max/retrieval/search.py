"""Independent search-path orchestration with retries, throttling, and caching."""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from gaia_max.config import Settings
from gaia_max.retrieval.providers import (
    ProviderContractError,
    SearchProvider,
    SearchRequest,
    SearchResult,
    execute_search,
)

AsyncSleep = Callable[[float], Awaitable[None]]
MonotonicClock = Callable[[], float]
WallClock = Callable[[], datetime]
RandomFraction = Callable[[], float]

_BACKEND_FAMILY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,98}[A-Za-z0-9])?")


class SearchConfigurationError(ValueError):
    """The primary and fallback search paths are not safely independent."""


class SearchFailureKind(StrEnum):
    """Stable provider failure classes used by the retry/fallback policy."""

    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    PERMANENT = "permanent"


class SearchProviderError(RuntimeError):
    """Sanitized provider error raised by concrete vendor adapters.

    Adapters keep response bodies and credentials out of this exception. The
    coordinator retries only transient/rate-limited failures, never malformed
    requests, authentication failures, or deterministic permanent failures.
    """

    def __init__(
        self,
        kind: SearchFailureKind,
        *,
        code: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", code) is None:
            raise ValueError("search provider error code must be a safe identifier")
        if retry_after_seconds is not None and retry_after_seconds < 0:
            raise ValueError("retry-after delay cannot be negative")
        self.kind = kind
        self.code = code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"search provider failure: {kind.value}/{code}")


class IndependentSearchProvider(SearchProvider, Protocol):
    """Search provider that declares the actual backend family it uses."""

    @property
    def backend_family(self) -> str: ...


class SearchRetryPolicy(BaseModel):
    """Bounded retries, jitter, cache lifetime, and provider request spacing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_seconds: float = Field(default=0.5, ge=0, le=60)
    retry_max_seconds: float = Field(default=8.0, ge=0, le=300)
    jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    cache_ttl_seconds: float = Field(default=86_400, ge=0, le=31_536_000)
    primary_min_interval_seconds: float = Field(default=0, ge=0, le=60)
    secondary_min_interval_seconds: float = Field(default=0, ge=0, le=60)

    @model_validator(mode="after")
    def validate_retry_window(self) -> SearchRetryPolicy:
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("maximum retry delay cannot be smaller than base delay")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> SearchRetryPolicy:
        """Build the search policy from validated application configuration."""

        return cls(
            max_attempts=settings.search_max_attempts,
            retry_base_seconds=settings.search_retry_base_seconds,
            retry_max_seconds=settings.search_retry_max_seconds,
            jitter_ratio=settings.search_retry_jitter_ratio,
            cache_ttl_seconds=settings.search_cache_ttl_seconds,
            primary_min_interval_seconds=(
                settings.search_primary_min_interval_seconds
            ),
            secondary_min_interval_seconds=(
                settings.search_secondary_min_interval_seconds
            ),
        )


class SearchPlan(BaseModel):
    """Ordered query variants: initial wording, then deliberate reformulations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: tuple[SearchRequest, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def require_distinct_variants(self) -> SearchPlan:
        fingerprints = tuple(request.model_dump_json() for request in self.requests)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("search query variants must be distinct")
        return self


def formulate_search_request(
    terms: str,
    *,
    exact_phrases: Sequence[str] = (),
    limit: int = 10,
    site_domains: Sequence[str] = (),
    preferred_domains: Sequence[str] = (),
    language: str | None = None,
    published_after: date | None = None,
    published_before: date | None = None,
) -> SearchRequest:
    """Build one quoted/site/native-language variant without vendor syntax leaks."""

    if terms != terms.strip():
        raise ValueError("search terms must be a trimmed string")
    rendered_parts = [terms] if terms else []
    for phrase in exact_phrases:
        if (
            phrase != phrase.strip()
            or not phrase
            or '"' in phrase
        ):
            raise ValueError("exact search phrases must be non-empty, trimmed, and unquoted")
        rendered_parts.append(f'"{phrase}"')
    if not rendered_parts:
        raise ValueError("search formulation requires terms or an exact phrase")
    return SearchRequest(
        query=" ".join(rendered_parts),
        limit=limit,
        site_domains=tuple(site_domains),
        preferred_domains=tuple(preferred_domains),
        language=language,
        published_after=published_after,
        published_before=published_before,
    )


class SearchAttemptStatus(StrEnum):
    """Answer-free audit status for one cache lookup or provider call."""

    CACHE_HIT = "cache_hit"
    RESULTS = "results"
    EMPTY = "empty"
    TRANSIENT_FAILURE = "transient_failure"
    RATE_LIMITED = "rate_limited"
    TERMINAL_FAILURE = "terminal_failure"


class SearchAttempt(BaseModel):
    """Sanitized trace of one step through the primary/secondary search ladder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(pattern=r"^(primary|secondary)$")
    provider: str = Field(min_length=1, max_length=100)
    backend_family: str = Field(min_length=1, max_length=100)
    query_index: int = Field(ge=0)
    query: StrictStr = Field(min_length=1, max_length=500)
    attempt_number: int = Field(ge=0)
    status: SearchAttemptStatus
    result_count: int = Field(default=0, ge=0, le=50)
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    retry_delay_seconds: float | None = Field(default=None, ge=0, le=3600)


class SearchOutcome(BaseModel):
    """Validated discovery results plus the exact fallback path that produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[SearchResult, ...]
    attempts: tuple[SearchAttempt, ...]
    winning_provider: str | None = None
    winning_backend_family: str | None = None
    winning_query_index: int | None = Field(default=None, ge=0)
    used_secondary: bool = False
    from_cache: bool = False

    @model_validator(mode="after")
    def validate_winner(self) -> SearchOutcome:
        winner_fields = (
            self.winning_provider,
            self.winning_backend_family,
            self.winning_query_index,
        )
        if self.results and any(value is None for value in winner_fields):
            raise ValueError("non-empty search outcome requires complete winner provenance")
        if not self.results and any(value is not None for value in winner_fields):
            raise ValueError("empty search outcome cannot name a winner")
        if self.from_cache and (
            not self.results
            or not self.attempts
            or self.attempts[-1].status is not SearchAttemptStatus.CACHE_HIT
        ):
            raise ValueError("cached outcome requires a winning cache-hit attempt")
        return self


class SearchCache(Protocol):
    """Replaceable cache port; persistent implementations can be added at wiring time."""

    async def get(
        self,
        provider: str,
        request: SearchRequest,
        *,
        max_age: timedelta,
    ) -> tuple[SearchResult, ...] | None: ...

    async def put(
        self,
        provider: str,
        request: SearchRequest,
        results: tuple[SearchResult, ...],
    ) -> None: ...


class InMemorySearchCache:
    """Concurrency-safe bounded-lifetime cache used locally and in unit tests."""

    def __init__(self, *, clock: WallClock | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._entries: dict[
            tuple[str, str], tuple[datetime, tuple[SearchResult, ...]]
        ] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        provider: str,
        request: SearchRequest,
        *,
        max_age: timedelta,
    ) -> tuple[SearchResult, ...] | None:
        key = (provider, request.model_dump_json())
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, results = entry
            if self._clock() - stored_at > max_age:
                del self._entries[key]
                return None
            return results

    async def put(
        self,
        provider: str,
        request: SearchRequest,
        results: tuple[SearchResult, ...],
    ) -> None:
        key = (provider, request.model_dump_json())
        async with self._lock:
            self._entries[key] = (self._clock(), results)


class _ProviderRateLimiter:
    def __init__(
        self,
        min_interval_seconds: float,
        *,
        sleep: AsyncSleep,
        monotonic: MonotonicClock,
    ) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_allowed_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._monotonic()
            delay = max(0.0, self._next_allowed_at - now)
            if delay:
                await self._sleep(delay)
                now = self._monotonic()
            self._next_allowed_at = now + self._min_interval_seconds


class SearchCoordinator:
    """Run reformulated primary searches before an independent secondary backend."""

    def __init__(
        self,
        *,
        primary: IndependentSearchProvider,
        secondary: IndependentSearchProvider,
        policy: SearchRetryPolicy | None = None,
        cache: SearchCache | None = None,
        sleep: AsyncSleep = asyncio.sleep,
        monotonic: MonotonicClock = time.monotonic,
        random_fraction: RandomFraction = random.random,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._policy = policy or SearchRetryPolicy()
        self._cache = cache if cache is not None else InMemorySearchCache()
        self._sleep = sleep
        self._random_fraction = random_fraction
        self._primary_identity = self._identity(primary)
        self._secondary_identity = self._identity(secondary)
        if self._primary_identity[0] == self._secondary_identity[0]:
            raise SearchConfigurationError("primary and secondary provider names must differ")
        if self._primary_identity[1] == self._secondary_identity[1]:
            raise SearchConfigurationError(
                "secondary search must use a genuinely separate backend family"
            )
        self._limiters = {
            "primary": _ProviderRateLimiter(
                self._policy.primary_min_interval_seconds,
                sleep=sleep,
                monotonic=monotonic,
            ),
            "secondary": _ProviderRateLimiter(
                self._policy.secondary_min_interval_seconds,
                sleep=sleep,
                monotonic=monotonic,
            ),
        }

    async def search(self, plan: SearchPlan) -> SearchOutcome:
        """Return the first non-empty result set after the complete bounded ladder."""

        attempts: list[SearchAttempt] = []
        for path, provider, identity in (
            ("primary", self._primary, self._primary_identity),
            ("secondary", self._secondary, self._secondary_identity),
        ):
            for query_index, request in enumerate(plan.requests):
                found = await self._search_variant(
                    path=path,
                    provider=provider,
                    identity=identity,
                    query_index=query_index,
                    request=request,
                    attempts=attempts,
                )
                if found is None:
                    # Provider unavailability is not fixed by changing the query.
                    # Continue with the independent backend immediately.
                    break
                results, from_cache = found
                if results:
                    return SearchOutcome(
                        results=results,
                        attempts=tuple(attempts),
                        winning_provider=identity[0],
                        winning_backend_family=identity[1],
                        winning_query_index=query_index,
                        used_secondary=path == "secondary",
                        from_cache=from_cache,
                    )
        return SearchOutcome(results=(), attempts=tuple(attempts), used_secondary=True)

    async def _search_variant(
        self,
        *,
        path: str,
        provider: IndependentSearchProvider,
        identity: tuple[str, str],
        query_index: int,
        request: SearchRequest,
        attempts: list[SearchAttempt],
    ) -> tuple[tuple[SearchResult, ...], bool] | None:
        provider_name, backend_family = identity
        cached = await self._cache.get(
            provider_name,
            request,
            max_age=timedelta(seconds=self._policy.cache_ttl_seconds),
        )
        if cached is not None:
            attempts.append(
                self._attempt(
                    path,
                    provider_name,
                    backend_family,
                    query_index,
                    request,
                    attempt_number=0,
                    status=SearchAttemptStatus.CACHE_HIT,
                    result_count=len(cached),
                )
            )
            return cached, True

        for attempt_number in range(1, self._policy.max_attempts + 1):
            await self._limiters[path].acquire()
            try:
                results = await execute_search(provider, request)
            except ProviderContractError:
                # A malformed response is a violated trust boundary, not ordinary
                # provider unavailability. Do not hide it behind another backend.
                raise
            except SearchProviderError as exc:
                if exc.kind is SearchFailureKind.INVALID_REQUEST:
                    raise
                retryable = exc.kind in {
                    SearchFailureKind.TRANSIENT,
                    SearchFailureKind.RATE_LIMITED,
                }
                delay = None
                if retryable and attempt_number < self._policy.max_attempts:
                    delay = self._retry_delay(attempt_number, exc.retry_after_seconds)
                attempts.append(
                    self._attempt(
                        path,
                        provider_name,
                        backend_family,
                        query_index,
                        request,
                        attempt_number=attempt_number,
                        status=(
                            SearchAttemptStatus.RATE_LIMITED
                            if exc.kind is SearchFailureKind.RATE_LIMITED
                            else SearchAttemptStatus.TRANSIENT_FAILURE
                            if retryable
                            else SearchAttemptStatus.TERMINAL_FAILURE
                        ),
                        error_code=exc.code,
                        retry_delay_seconds=delay,
                    )
                )
                if delay is not None:
                    await self._sleep(delay)
                    continue
                return None
            except (TimeoutError, ConnectionError):
                delay = None
                if attempt_number < self._policy.max_attempts:
                    delay = self._retry_delay(attempt_number, None)
                attempts.append(
                    self._attempt(
                        path,
                        provider_name,
                        backend_family,
                        query_index,
                        request,
                        attempt_number=attempt_number,
                        status=SearchAttemptStatus.TRANSIENT_FAILURE,
                        error_code="transport_failure",
                        retry_delay_seconds=delay,
                    )
                )
                if delay is not None:
                    await self._sleep(delay)
                    continue
                return None

            await self._cache.put(provider_name, request, results)
            attempts.append(
                self._attempt(
                    path,
                    provider_name,
                    backend_family,
                    query_index,
                    request,
                    attempt_number=attempt_number,
                    status=(
                        SearchAttemptStatus.RESULTS if results else SearchAttemptStatus.EMPTY
                    ),
                    result_count=len(results),
                )
            )
            return results, False
        raise RuntimeError("unreachable search retry state")

    def _retry_delay(
        self,
        completed_attempt: int,
        retry_after_seconds: float | None,
    ) -> float:
        ceiling = min(
            self._policy.retry_max_seconds,
            self._policy.retry_base_seconds * 2 ** (completed_attempt - 1),
        )
        jitter_multiplier = 1 + self._policy.jitter_ratio * (
            2 * self._random_fraction() - 1
        )
        jittered = max(0.0, ceiling * jitter_multiplier)
        return min(
            self._policy.retry_max_seconds,
            max(jittered, retry_after_seconds or 0.0),
        )

    @staticmethod
    def _identity(provider: IndependentSearchProvider) -> tuple[str, str]:
        provider_name = getattr(provider, "name", None)
        backend_family = getattr(provider, "backend_family", None)
        if (
            not isinstance(provider_name, str)
            or _BACKEND_FAMILY.fullmatch(provider_name) is None
        ):
            raise SearchConfigurationError("search provider name must be a safe identifier")
        if (
            not isinstance(backend_family, str)
            or _BACKEND_FAMILY.fullmatch(backend_family) is None
        ):
            raise SearchConfigurationError(
                "coordinated search providers must declare a safe backend_family"
            )
        return provider_name, backend_family

    @staticmethod
    def _attempt(
        path: str,
        provider: str,
        backend_family: str,
        query_index: int,
        request: SearchRequest,
        *,
        attempt_number: int,
        status: SearchAttemptStatus,
        result_count: int = 0,
        error_code: str | None = None,
        retry_delay_seconds: float | None = None,
    ) -> SearchAttempt:
        return SearchAttempt(
            path=path,
            provider=provider,
            backend_family=backend_family,
            query_index=query_index,
            query=request.query,
            attempt_number=attempt_number,
            status=status,
            result_count=result_count,
            error_code=error_code,
            retry_delay_seconds=retry_delay_seconds,
        )


__all__ = [
    "InMemorySearchCache",
    "IndependentSearchProvider",
    "SearchAttempt",
    "SearchAttemptStatus",
    "SearchCache",
    "SearchConfigurationError",
    "SearchCoordinator",
    "SearchFailureKind",
    "SearchOutcome",
    "SearchPlan",
    "SearchProviderError",
    "SearchRetryPolicy",
    "formulate_search_request",
]
