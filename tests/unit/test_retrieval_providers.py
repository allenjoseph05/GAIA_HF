from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from gaia_max.retrieval import (
    PageRetrievalRequest,
    ProviderContractError,
    RetrievedDocument,
    SearchRequest,
    SearchResult,
    execute_search,
    retrieve_document,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def search_payload(provider: str, query: str, *, rank: int = 1) -> dict[str, object]:
    return {
        "provider": provider,
        "query": query,
        "rank": rank,
        "url": "https://example.org/source",
        "title": "Synthetic primary source",
        "snippet": "Discovery text is not final evidence.",
        "published_at": "2024-02-03",
        "retrieved_at": NOW,
    }


class ListSearchProvider:
    name = "mock-list-search"

    async def search(self, request: SearchRequest) -> object:
        return [search_payload(self.name, request.query)]


class TupleSearchProvider:
    name = "mock-tuple-search"

    async def search(self, request: SearchRequest) -> object:
        return (
            SearchResult.model_validate(search_payload(self.name, request.query)),
        )


@pytest.mark.parametrize("provider", [ListSearchProvider(), TupleSearchProvider()])
@pytest.mark.asyncio
async def test_two_search_providers_are_interchangeable_and_keep_provenance(
    provider: ListSearchProvider | TupleSearchProvider,
) -> None:
    request = SearchRequest(
        query='site:example.org "exact phrase"',
        limit=3,
        site_domains=("Example.ORG",),
        preferred_domains=("archive.example.org",),
        language="en-GB",
        published_after=date(2020, 1, 1),
        published_before=date(2025, 1, 1),
    )

    results = await execute_search(provider, request)

    assert request.site_domains == ("example.org",)
    assert len(results) == 1
    assert results[0].provider == provider.name
    assert results[0].query == request.query
    assert results[0].rank == 1
    assert str(results[0].url) == "https://example.org/source"
    assert results[0].retrieved_at == NOW


@pytest.mark.parametrize(
    "updates",
    [
        {"query": " padded "},
        {"site_domains": ("https://example.org/path",)},
        {"preferred_domains": ("example.org", "EXAMPLE.ORG")},
        {"language": "not a language tag"},
        {
            "published_after": date(2025, 1, 1),
            "published_before": date(2024, 1, 1),
        },
        {"unexpected_answer": "forbidden"},
    ],
)
def test_search_request_rejects_ambiguous_or_unexpected_constraints(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "valid query", **updates})


class PayloadSearchProvider:
    name = "mock-search"

    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def search(self, request: SearchRequest) -> object:
        del request
        return self.payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([search_payload("different-provider", "query")], "mismatched provenance"),
        ([search_payload("mock-search", "different query")], "query provenance"),
        ([search_payload("mock-search", "query", rank=2)], "non-contiguous ranks"),
        ([{"not": "a result"}], "invalid result schema"),
    ],
)
@pytest.mark.asyncio
async def test_search_gateway_rejects_poisoned_provider_output(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ProviderContractError, match=message):
        await execute_search(
            PayloadSearchProvider(payload),
            SearchRequest(query="query"),
        )


@pytest.mark.asyncio
async def test_search_gateway_rejects_duplicate_urls_and_excess_results() -> None:
    first = search_payload("mock-search", "query")
    second = {
        **search_payload("mock-search", "query", rank=2),
        "title": "Duplicate",
    }

    with pytest.raises(ProviderContractError, match="duplicate"):
        await execute_search(
            PayloadSearchProvider([first, second]),
            SearchRequest(query="query", limit=2),
        )
    distinct_second = {
        **second,
        "url": "https://example.org/second",
    }
    with pytest.raises(ProviderContractError, match="exceeded"):
        await execute_search(
            PayloadSearchProvider([first, distinct_second]),
            SearchRequest(query="query", limit=1),
        )


def document_payload(provider: str, requested_url: str) -> dict[str, object]:
    return {
        "provider": provider,
        "requested_url": requested_url,
        "final_url": "https://example.org/final",
        "status_code": 200,
        "title": "Synthetic document",
        "published_at": "2024-02-03",
        "retrieved_at": NOW,
        "artifact_id": "a" * 64,
        "text_artifact_id": "b" * 64,
        "links": [
            {"url": "https://example.org/reference", "text": "Reference"},
        ],
        "extraction_method": "fixture_text",
        "injection_flags": [],
    }


class MappingPageProvider:
    name = "mock-mapping-page"

    async def retrieve(self, request: PageRetrievalRequest) -> object:
        return document_payload(self.name, str(request.url))


class ModelPageProvider:
    name = "mock-model-page"

    async def retrieve(self, request: PageRetrievalRequest) -> object:
        return RetrievedDocument.model_validate(
            document_payload(self.name, str(request.url))
        )


@pytest.mark.parametrize("provider", [MappingPageProvider(), ModelPageProvider()])
@pytest.mark.asyncio
async def test_two_page_providers_are_interchangeable_and_keep_artifact_provenance(
    provider: MappingPageProvider | ModelPageProvider,
) -> None:
    request = PageRetrievalRequest.model_validate(
        {
            "url": "https://example.org/original",
            "preferred_language": "en",
        }
    )

    document = await retrieve_document(provider, request)

    assert document.provider == provider.name
    assert document.requested_url == request.url
    assert str(document.final_url) == "https://example.org/final"
    assert document.artifact_id == "a" * 64
    assert document.text_artifact_id == "b" * 64
    assert document.extraction_method == "fixture_text"
    assert str(document.links[0].url) == "https://example.org/reference"


class PayloadPageProvider:
    name = "mock-page"

    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def retrieve(self, request: PageRetrievalRequest) -> object:
        del request
        return self.payload


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            document_payload("different-provider", "https://example.org/original"),
            "mismatched provenance",
        ),
        (
            document_payload("mock-page", "https://example.org/different"),
            "different requested URL",
        ),
        (
            {
                **document_payload("mock-page", "https://example.org/original"),
                "artifact_id": "not-a-hash",
            },
            "invalid document schema",
        ),
    ],
)
@pytest.mark.asyncio
async def test_page_gateway_rejects_poisoned_document_provenance(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ProviderContractError, match=message):
        await retrieve_document(
            PayloadPageProvider(payload),
            PageRetrievalRequest.model_validate(
                {"url": "https://example.org/original"}
            ),
        )


@pytest.mark.asyncio
async def test_provider_runtime_failure_remains_visible_to_future_fallback_policy() -> None:
    class FailingProvider:
        name = "failing-search"

        async def search(self, request: SearchRequest) -> object:
            del request
            raise TimeoutError("synthetic provider timeout")

    with pytest.raises(TimeoutError, match="synthetic provider timeout"):
        await execute_search(FailingProvider(), SearchRequest(query="query"))
