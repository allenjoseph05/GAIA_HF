from __future__ import annotations

import httpx
import pytest

from gaia_max.clients import (
    GaiaApiClient,
    GaiaApiNotFoundError,
    GaiaApiResponseError,
    GaiaApiTransientError,
)
from gaia_max.config import Settings


def settings() -> Settings:
    return Settings(_env_file=None, gaia_api_url="https://course.test")


@pytest.mark.asyncio
async def test_get_questions_parses_official_level_alias_and_empty_filename() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/questions"
        return httpx.Response(
            200,
            json=[
                {
                    "task_id": "task-1",
                    "question": "What is the result?",
                    "Level": "1",
                    "file_name": "",
                }
            ],
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        questions = await client.get_questions()

    assert questions[0].level == "1"
    assert questions[0].file_name is None


@pytest.mark.asyncio
async def test_get_questions_rejects_non_list_json() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"bad": "shape"}))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiResponseError, match="JSON list"):
            await client.get_questions()


@pytest.mark.asyncio
async def test_get_questions_rejects_invalid_records() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=[{"task_id": "x"}]))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiResponseError, match="invalid question"):
            await client.get_questions()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_statuses_are_classified_for_future_retry(status: int) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiTransientError):
            await client.get_questions()


@pytest.mark.asyncio
async def test_attachment_404_is_classified_for_dataset_fallback() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(404))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiNotFoundError):
            await client.get_attachment("task-1")


@pytest.mark.asyncio
async def test_attachment_returns_raw_bytes_and_metadata() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"example-bytes",
            headers={
                "content-type": "application/octet-stream",
                "content-disposition": 'attachment; filename="example.bin"',
            },
        )
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        attachment = await client.get_attachment("task-1")

    assert attachment.content == b"example-bytes"
    assert attachment.content_type == "application/octet-stream"
    assert "example.bin" in (attachment.content_disposition or "")


@pytest.mark.asyncio
async def test_empty_attachment_is_rejected() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b""))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiResponseError, match="was empty"):
            await client.get_attachment("task-1")


@pytest.mark.asyncio
async def test_timeout_is_classified_as_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://course.test"
    ) as http_client:
        client = GaiaApiClient(settings(), http_client=http_client)
        with pytest.raises(GaiaApiTransientError):
            await client.get_questions()


def test_client_exposes_no_submission_method() -> None:
    assert not hasattr(GaiaApiClient, "submit")
    assert not hasattr(GaiaApiClient, "submit_answers")
