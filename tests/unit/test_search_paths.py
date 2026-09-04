from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from gaia_max.config import Settings
from gaia_max.retrieval import (
    InMemorySearchCache,
    ProviderContractError,
    SearchAttemptStatus,
    SearchConfigurationError,
    SearchCoordinator,
    SearchFailureKind,
    SearchPlan,
    SearchProviderError,
    SearchRequest,
    SearchRetryPolicy,
    formulate_search_request,
)

NOW = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)


def result_payload(provider: str, query: str, *, url_suffix: str = "source") -> dict[str, object]:
    return {
        "provider": provider,
        "query": query,
        "rank": 1,
        "url": f"https://example.org/{url_suffix}",
        "title": "Synthetic discovery result",
        "snippet": "A search snippet is not final evidence.",
        "retrieved_at": NOW,
    }


ScriptedResponse = object | Callable[[SearchRequest], object]


class ScriptedProvider:
    def __init__(
        self,
        name: str,
        backend_family: str,
        responses: list[ScriptedResponse],
    ) -> None:
        self.name = name
        self.backend_family = backend_family
        self.responses = responses
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> object:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected call to {self.name}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(request)
        return response


def response_for(provider: str, suffix: str = "source") -> Callable[[SearchRequest], object]:
    return lambda request: [
        result_payload(provider, request.query, url_suffix=suffix)
    ]


def no_wait_policy(**updates: object) -> SearchRetryPolicy:
    return SearchRetryPolicy.model_validate(
        {
            "max_attempts": 2,
            "retry_base_seconds": 0,
            "retry_max_seconds": 0,
            "jitter_ratio": 0,
            **updates,
        }
    )


def test_retry_policy_uses_validated_application_settings() -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        search_max_attempts=4,
        search_retry_base_seconds=1,
        search_retry_max_seconds=9,
        search_retry_jitter_ratio=0.1,
        search_cache_ttl_seconds=120,
        search_primary_min_interval_seconds=0.25,
        search_secondary_min_interval_seconds=0.5,
    )

    policy = SearchRetryPolicy.from_settings(settings)

    assert policy.max_attempts == 4
    assert policy.retry_max_seconds == 9
    assert policy.cache_ttl_seconds == 120
    assert policy.primary_min_interval_seconds == 0.25
    assert policy.secondary_min_interval_seconds == 0.5


def test_query_formulation_keeps_quoted_site_date_and_native_constraints_typed() -> None:
    request = formulate_search_request(
        "lokalny tytul",
        exact_phrases=("dokladna fraza",),
        limit=7,
        site_domains=("TVP.PL",),
        preferred_domains=("filmpolski.pl",),
        language="pl-PL",
        published_after=date(2020, 1, 1),
        published_before=date(2024, 12, 31),
    )

    assert request.query == 'lokalny tytul "dokladna fraza"'
    assert request.site_domains == ("tvp.pl",)
    assert request.preferred_domains == ("filmpolski.pl",)
    assert request.language == "pl-PL"
    assert request.limit == 7


@pytest.mark.parametrize(
    ("terms", "phrases"),
    [
        (" padded ", ()),
        ("", ()),
        ("topic", ("",)),
        ("topic", (" already padded ",)),
        ("topic", ('broken " phrase',)),
    ],
)
def test_query_formulation_rejects_ambiguous_text(
    terms: str,
    phrases: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        formulate_search_request(terms, exact_phrases=phrases)


def test_search_plan_requires_distinct_ordered_reformulations() -> None:
    request = SearchRequest(query="one query")

    with pytest.raises(ValidationError, match="must be distinct"):
        SearchPlan(requests=(request, request))


@pytest.mark.asyncio
async def test_empty_primary_query_reformulates_before_secondary() -> None:
    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [[], response_for("primary-search", "reformulated")],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])
    plan = SearchPlan(
        requests=(
            SearchRequest(query="initial wording"),
            SearchRequest(query='reformulated "exact phrase"'),
        )
    )
    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=no_wait_policy(max_attempts=1),
    )

    outcome = await coordinator.search(plan)

    assert [request.query for request in primary.requests] == [
        "initial wording",
        'reformulated "exact phrase"',
    ]
    assert secondary.requests == []
    assert outcome.winning_provider == "primary-search"
    assert outcome.winning_query_index == 1
    assert outcome.used_secondary is False
    assert [attempt.status for attempt in outcome.attempts] == [
        SearchAttemptStatus.EMPTY,
        SearchAttemptStatus.RESULTS,
    ]


@pytest.mark.asyncio
async def test_transient_primary_failure_retries_then_uses_separate_backend() -> None:
    primary = ScriptedProvider(
        "primary-search",
        "google-family",
        [TimeoutError("secret transport detail"), TimeoutError("still unavailable")],
    )
    secondary = ScriptedProvider(
        "secondary-search",
        "duckduckgo-family",
        [response_for("secondary-search", "fallback")],
    )
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=SearchRetryPolicy(
            max_attempts=2,
            retry_base_seconds=0.5,
            retry_max_seconds=2,
            jitter_ratio=0,
        ),
        sleep=record_sleep,
    )

    outcome = await coordinator.search(SearchPlan(requests=(SearchRequest(query="query"),)))

    assert len(primary.requests) == 2
    assert len(secondary.requests) == 1
    assert sleeps == [0.5]
    assert outcome.used_secondary is True
    assert outcome.winning_backend_family == "duckduckgo-family"
    assert [attempt.status for attempt in outcome.attempts] == [
        SearchAttemptStatus.TRANSIENT_FAILURE,
        SearchAttemptStatus.TRANSIENT_FAILURE,
        SearchAttemptStatus.RESULTS,
    ]
    assert "secret transport detail" not in repr(outcome)


def test_fallback_must_declare_a_genuinely_different_backend_family() -> None:
    primary = ScriptedProvider("search-one", "shared-backend", [])
    alias = ScriptedProvider("search-two", "shared-backend", [])

    with pytest.raises(SearchConfigurationError, match="genuinely separate"):
        SearchCoordinator(primary=primary, secondary=alias)


@pytest.mark.asyncio
async def test_rate_limit_error_honors_retry_after_before_success() -> None:
    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [
            SearchProviderError(
                SearchFailureKind.RATE_LIMITED,
                code="http_429",
                retry_after_seconds=3,
            ),
            response_for("primary-search"),
        ],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=SearchRetryPolicy(
            max_attempts=2,
            retry_base_seconds=0.25,
            retry_max_seconds=4,
            jitter_ratio=0,
        ),
        sleep=record_sleep,
    )

    outcome = await coordinator.search(SearchPlan(requests=(SearchRequest(query="query"),)))

    assert sleeps == [3]
    assert outcome.attempts[0].status is SearchAttemptStatus.RATE_LIMITED
    assert outcome.attempts[0].error_code == "http_429"
    assert outcome.attempts[0].retry_delay_seconds == 3


@pytest.mark.asyncio
async def test_authentication_failure_is_not_retried_and_falls_back() -> None:
    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [
            SearchProviderError(
                SearchFailureKind.AUTHENTICATION,
                code="invalid_credential",
            )
        ],
    )
    secondary = ScriptedProvider(
        "secondary-search",
        "backend-b",
        [response_for("secondary-search")],
    )
    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=no_wait_policy(max_attempts=3),
    )

    outcome = await coordinator.search(
        SearchPlan(
            requests=(
                SearchRequest(query="query"),
                SearchRequest(query="reformulated query"),
            )
        )
    )

    assert len(primary.requests) == 1
    assert len(secondary.requests) == 1
    assert outcome.attempts[0].status is SearchAttemptStatus.TERMINAL_FAILURE
    assert outcome.used_secondary is True


@pytest.mark.asyncio
async def test_unknown_provider_exception_remains_visible() -> None:
    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [RuntimeError("unexpected adapter defect")],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])

    with pytest.raises(RuntimeError, match="unexpected adapter defect"):
        await SearchCoordinator(
            primary=primary,
            secondary=secondary,
            policy=no_wait_policy(),
        ).search(SearchPlan(requests=(SearchRequest(query="query"),)))
    assert secondary.requests == []


@pytest.mark.asyncio
async def test_invalid_request_and_contract_failures_do_not_hide_behind_fallback() -> None:
    invalid_request = ScriptedProvider(
        "primary-search",
        "backend-a",
        [SearchProviderError(SearchFailureKind.INVALID_REQUEST, code="bad_query")],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])
    plan = SearchPlan(requests=(SearchRequest(query="query"),))

    with pytest.raises(SearchProviderError) as caught:
        await SearchCoordinator(
            primary=invalid_request,
            secondary=secondary,
            policy=no_wait_policy(),
        ).search(plan)
    assert caught.value.kind is SearchFailureKind.INVALID_REQUEST
    assert secondary.requests == []

    malformed = ScriptedProvider(
        "malformed-search",
        "backend-c",
        [[result_payload("forged-provider", "query")]],
    )
    with pytest.raises(ProviderContractError, match="mismatched provenance"):
        await SearchCoordinator(
            primary=malformed,
            secondary=secondary,
            policy=no_wait_policy(),
        ).search(plan)
    assert secondary.requests == []


@pytest.mark.asyncio
async def test_successful_results_are_cached_by_provider_and_exact_request() -> None:
    cache = InMemorySearchCache(clock=lambda: NOW)
    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [response_for("primary-search")],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])
    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        cache=cache,
        policy=no_wait_policy(),
    )
    plan = SearchPlan(requests=(SearchRequest(query="query"),))

    first = await coordinator.search(plan)
    second = await coordinator.search(plan)

    assert len(primary.requests) == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.results == first.results
    assert second.attempts[-1].status is SearchAttemptStatus.CACHE_HIT


@pytest.mark.asyncio
async def test_expired_cache_entry_calls_provider_again() -> None:
    current = [NOW]
    cache = InMemorySearchCache(clock=lambda: current[0])
    request = SearchRequest(query="query")
    provider = "primary-search"
    await cache.put(
        provider,
        request,
        (),
    )

    assert await cache.get(provider, request, max_age=timedelta(seconds=1)) == ()
    current[0] = NOW + timedelta(seconds=2)
    assert await cache.get(provider, request, max_age=timedelta(seconds=1)) is None


@pytest.mark.asyncio
async def test_per_provider_spacing_applies_only_to_real_provider_calls() -> None:
    now = [0.0]
    sleeps: list[float] = []

    async def advance(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    primary = ScriptedProvider(
        "primary-search",
        "backend-a",
        [[], response_for("primary-search")],
    )
    secondary = ScriptedProvider("secondary-search", "backend-b", [])
    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=no_wait_policy(
            max_attempts=1,
            primary_min_interval_seconds=2,
        ),
        sleep=advance,
        monotonic=lambda: now[0],
    )
    plan = SearchPlan(
        requests=(SearchRequest(query="first"), SearchRequest(query="second"))
    )

    await coordinator.search(plan)

    assert sleeps == [2]


@pytest.mark.asyncio
async def test_both_backends_can_return_empty_without_manufacturing_results() -> None:
    primary = ScriptedProvider("primary-search", "backend-a", [[], []])
    secondary = ScriptedProvider("secondary-search", "backend-b", [[], []])
    coordinator = SearchCoordinator(
        primary=primary,
        secondary=secondary,
        policy=no_wait_policy(max_attempts=1),
    )
    plan = SearchPlan(
        requests=(SearchRequest(query="first"), SearchRequest(query="second"))
    )

    outcome = await coordinator.search(plan)

    assert outcome.results == ()
    assert outcome.winning_provider is None
    assert outcome.used_secondary is True
    assert len(outcome.attempts) == 4
