"""Provider-neutral search and retrieved-document contracts."""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictStr,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from gaia_max.domain.models import SHA256_PATTERN

_PROVIDER_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,98}[A-Za-z0-9])?")
_LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")
_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


class ProviderContractError(ValueError):
    """A provider definition or response violated the trusted boundary."""


def _normalize_domain(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("search domains must be strings")
    domain = value.strip().casefold().removesuffix(".")
    labels = domain.split(".")
    if (
        len(domain) > 253
        or len(labels) < 2
        or any(_DOMAIN_LABEL.fullmatch(label) is None for label in labels)
    ):
        raise ValueError("search domains must be bare DNS names without paths or schemes")
    return domain


def _provider_name(provider: object) -> str:
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or _PROVIDER_NAME.fullmatch(name) is None:
        raise ProviderContractError("provider name must be a safe stable identifier")
    return name


class SearchRequest(BaseModel):
    """One exact query plus provider-neutral search constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: StrictStr = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)
    site_domains: tuple[str, ...] = Field(default=(), max_length=20)
    preferred_domains: tuple[str, ...] = Field(default=(), max_length=20)
    language: str | None = None
    published_after: date | None = None
    published_before: date | None = None

    @field_validator("query")
    @classmethod
    def require_exact_trimmed_query(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("search query must not have surrounding whitespace")
        return value

    @field_validator("site_domains", "preferred_domains", mode="before")
    @classmethod
    def normalize_domains(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("search domain constraints must be a list or tuple")
        domains = cast(list[object] | tuple[object, ...], value)
        normalized = tuple(_normalize_domain(domain) for domain in domains)
        if len(set(normalized)) != len(normalized):
            raise ValueError("search domain constraints must be unique")
        return normalized

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is not None and _LANGUAGE_TAG.fullmatch(value) is None:
            raise ValueError("search language must be a BCP-47-like language tag")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> SearchRequest:
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValueError("search publication start cannot follow end")
        return self


class SearchResult(BaseModel):
    """One discovery result with immutable request and provider provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=_PROVIDER_NAME.pattern)
    query: StrictStr = Field(min_length=1, max_length=500)
    rank: int = Field(ge=1)
    url: HttpUrl
    title: StrictStr = Field(min_length=1)
    snippet: StrictStr | None = None
    published_at: date | None = None
    retrieved_at: AwareDatetime

    @field_serializer("url")
    def serialize_url(self, url: HttpUrl) -> str:
        return str(url)


class PageRetrievalRequest(BaseModel):
    """Provider-neutral request for one HTTP(S) document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    preferred_language: str | None = None

    @field_validator("preferred_language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is not None and _LANGUAGE_TAG.fullmatch(value) is None:
            raise ValueError("page language must be a BCP-47-like language tag")
        return value

    @field_serializer("url")
    def serialize_url(self, url: HttpUrl) -> str:
        return str(url)


class DocumentLink(BaseModel):
    """One extracted link without navigation or scripting content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl
    text: StrictStr | None = None

    @field_serializer("url")
    def serialize_url(self, url: HttpUrl) -> str:
        return str(url)


class RetrievedDocument(BaseModel):
    """Successful page retrieval bound to immutable content artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(pattern=_PROVIDER_NAME.pattern)
    requested_url: HttpUrl
    final_url: HttpUrl
    status_code: int = Field(ge=200, lt=300)
    title: StrictStr | None = None
    published_at: date | None = None
    retrieved_at: AwareDatetime
    artifact_id: str = Field(pattern=SHA256_PATTERN)
    text_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    links: tuple[DocumentLink, ...] = ()
    extraction_method: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    injection_flags: tuple[str, ...] = ()

    @field_serializer("requested_url", "final_url")
    def serialize_urls(self, url: HttpUrl) -> str:
        return str(url)


class SearchProvider(Protocol):
    """Async vendor adapter; output is untrusted until ``execute_search``."""

    @property
    def name(self) -> str: ...

    async def search(self, request: SearchRequest) -> object: ...


class PageRetrievalProvider(Protocol):
    """Async retrieval adapter; output is untrusted until gateway validation."""

    @property
    def name(self) -> str: ...

    async def retrieve(self, request: PageRetrievalRequest) -> object: ...


_SEARCH_RESULTS = TypeAdapter(tuple[SearchResult, ...])


async def execute_search(
    provider: SearchProvider,
    request: SearchRequest,
) -> tuple[SearchResult, ...]:
    """Call one provider and enforce schema, provenance, rank, and limit invariants."""

    provider_name = _provider_name(provider)
    raw_results = await provider.search(request)
    try:
        results = _SEARCH_RESULTS.validate_python(raw_results)
    except ValidationError as exc:
        raise ProviderContractError(
            f"search provider {provider_name!r} returned an invalid result schema"
        ) from exc
    if len(results) > request.limit:
        raise ProviderContractError(
            f"search provider {provider_name!r} exceeded the requested result limit"
        )
    expected_ranks = tuple(range(1, len(results) + 1))
    if tuple(result.rank for result in results) != expected_ranks:
        raise ProviderContractError(
            f"search provider {provider_name!r} returned non-contiguous ranks"
        )
    if any(result.provider != provider_name for result in results):
        raise ProviderContractError(
            f"search provider {provider_name!r} returned mismatched provenance"
        )
    if any(result.query != request.query for result in results):
        raise ProviderContractError(
            f"search provider {provider_name!r} returned mismatched query provenance"
        )
    urls = tuple(str(result.url) for result in results)
    if len(set(urls)) != len(urls):
        raise ProviderContractError(
            f"search provider {provider_name!r} returned duplicate result URLs"
        )
    return results


async def retrieve_document(
    provider: PageRetrievalProvider,
    request: PageRetrievalRequest,
) -> RetrievedDocument:
    """Call one retriever and bind its typed document to the exact request."""

    provider_name = _provider_name(provider)
    raw_document = await provider.retrieve(request)
    try:
        document = RetrievedDocument.model_validate(raw_document)
    except ValidationError as exc:
        raise ProviderContractError(
            f"page provider {provider_name!r} returned an invalid document schema"
        ) from exc
    if document.provider != provider_name:
        raise ProviderContractError(
            f"page provider {provider_name!r} returned mismatched provenance"
        )
    if str(document.requested_url) != str(request.url):
        raise ProviderContractError(
            f"page provider {provider_name!r} returned a different requested URL"
        )
    return document


__all__ = [
    "DocumentLink",
    "PageRetrievalProvider",
    "PageRetrievalRequest",
    "ProviderContractError",
    "RetrievedDocument",
    "SearchProvider",
    "SearchRequest",
    "SearchResult",
    "execute_search",
    "retrieve_document",
]
