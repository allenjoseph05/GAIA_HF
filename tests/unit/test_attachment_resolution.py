from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_resolution import (
    OfficialAttachmentRejectedError,
    OfficialAttachmentResolver,
    OfficialAttachmentStatus,
    OfficialFallbackReason,
)
from gaia_max.attachment_validation import AttachmentRejectionReason, AttachmentValidator
from gaia_max.clients import GaiaApiClient
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource, Question


def settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        gaia_api_url="https://course.test",
        attachment_max_attempts=3,
        attachment_retry_base_seconds=0.25,
        **overrides,
    )


def attachment_question(file_name: str | None = "task.py") -> Question:
    return Question(
        task_id="task-1",
        question="Inspect the attached file.",
        file_name=file_name,
    )


async def no_sleep(_delay: float) -> None:
    return None


def resolver(
    tmp_path: Path,
    http_client: httpx.AsyncClient,
    *,
    app_settings: Settings | None = None,
    validator: AttachmentValidator | None = None,
    sleep=no_sleep,
) -> OfficialAttachmentResolver:
    resolved_settings = app_settings or settings()
    return OfficialAttachmentResolver(
        client=GaiaApiClient(resolved_settings, http_client=http_client),
        store=ArtifactStore(tmp_path / "artifacts"),
        validator=validator
        or AttachmentValidator(max_bytes=resolved_settings.max_attachment_bytes),
        settings=resolved_settings,
        sleep=sleep,
    )


@pytest.mark.asyncio
async def test_success_validates_stores_and_records_official_provenance(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"print(42)\n",
            headers={"content-type": "text/x-python; charset=utf-8"},
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://course.test",
    ) as http_client:
        attachment_resolver = resolver(tmp_path, http_client)
        outcome = await attachment_resolver.resolve(attachment_question())

    assert outcome.status is OfficialAttachmentStatus.RESOLVED
    assert outcome.attempts == 1
    assert outcome.artifact is not None
    assert outcome.artifact.source is ArtifactSource.OFFICIAL_API
    assert outcome.artifact.media_type == "text/x-python"
    assert outcome.artifact.local_path.read_bytes() == b"print(42)\n"

    metadata = ArtifactStore(tmp_path / "artifacts").metadata(outcome.artifact.artifact_id)
    origin = metadata.origins[0]
    assert origin.task_id == "task-1"
    assert origin.source_locator == "https://course.test/files/task-1"
    assert origin.validation is not None
    assert origin.validation.sha256 == outcome.artifact.sha256


@pytest.mark.asyncio
async def test_transient_failures_retry_with_exponential_delays_then_succeed(
    tmp_path: Path,
) -> None:
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            content=b"print('recovered')\n",
            headers={"content-type": "text/x-python"},
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client, sleep=record_sleep).resolve(
            attachment_question()
        )

    assert outcome.status is OfficialAttachmentStatus.RESOLVED
    assert outcome.attempts == 3
    assert calls == 3
    assert delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_404_routes_to_fallback_once_without_retry(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client).resolve(attachment_question())

    assert outcome.status is OfficialAttachmentStatus.FALLBACK_REQUIRED
    assert outcome.fallback_reason is OfficialFallbackReason.NOT_FOUND
    assert outcome.attempts == 1
    assert calls == 1


@pytest.mark.asyncio
async def test_exhausted_transient_failures_route_to_fallback(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client).resolve(attachment_question())

    assert outcome.status is OfficialAttachmentStatus.FALLBACK_REQUIRED
    assert outcome.fallback_reason is OfficialFallbackReason.TRANSIENT_RETRIES_EXHAUSTED
    assert outcome.attempts == 3
    assert calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403])
async def test_permanent_response_failure_routes_to_fallback_without_retry(
    tmp_path: Path,
    status_code: int,
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client).resolve(attachment_question())

    assert outcome.fallback_reason is OfficialFallbackReason.RESPONSE_REJECTED
    assert calls == 1


@pytest.mark.asyncio
async def test_html_attachment_routes_to_fallback_without_retry(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=b"<!doctype html><title>file unavailable</title>",
            headers={"content-type": "text/html"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client).resolve(attachment_question())

    assert outcome.fallback_reason is OfficialFallbackReason.VALIDATION_REJECTED
    assert outcome.fallback_validation_reason is AttachmentRejectionReason.HTML_ERROR_RESPONSE
    assert calls == 1


@pytest.mark.asyncio
async def test_unsupported_expected_file_type_stops_instead_of_bypassing_policy(
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"%PDF-content",
            headers={"content-type": "application/pdf"},
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://course.test",
    ) as http_client:
        with pytest.raises(OfficialAttachmentRejectedError) as caught:
            await resolver(tmp_path, http_client).resolve(attachment_question("task.pdf"))

    assert caught.value.reason is AttachmentRejectionReason.UNSUPPORTED_EXTENSION
    assert caught.value.attempts == 1


@pytest.mark.asyncio
async def test_streaming_size_limit_stops_before_storage(tmp_path: Path) -> None:
    app_settings = settings(max_attachment_bytes=8)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"print('larger than eight bytes')\n",
            headers={"content-type": "text/x-python"},
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://course.test",
    ) as http_client:
        attachment_resolver = resolver(
            tmp_path,
            http_client,
            app_settings=app_settings,
        )
        with pytest.raises(OfficialAttachmentRejectedError) as caught:
            await attachment_resolver.resolve(attachment_question())

    assert caught.value.reason is AttachmentRejectionReason.TOO_LARGE
    assert list((tmp_path / "artifacts" / "sha256").rglob("content")) == []


@pytest.mark.asyncio
async def test_task_without_attachment_never_calls_endpoint(tmp_path: Path) -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    ) as http_client:
        with pytest.raises(ValueError, match="no attachment filename"):
            await resolver(tmp_path, http_client).resolve(attachment_question(None))

    assert called is False


def test_resolver_public_surface_has_no_submission_capability() -> None:
    forbidden_names: tuple[str, ...] = ("submit", "submit_answers")
    assert all(not hasattr(OfficialAttachmentResolver, name) for name in forbidden_names)
