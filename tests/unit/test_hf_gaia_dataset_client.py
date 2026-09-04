from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path

import httpx
import pytest

from gaia_max.clients import (
    GaiaGatedDatasetClient,
    GatedDatasetAccessError,
    GatedDatasetAttachmentNotFoundError,
    GatedDatasetDownloadTooLargeError,
    GatedDatasetPolicyError,
    GatedDatasetResponseError,
    GatedDatasetTransientError,
    GatedRawAttachment,
)
from gaia_max.config import Settings

TOKEN = "hf_private_test_token"
TASK_ID = "f918266a-b3e0-4914-865d-4faa564f1aef"
FILE_NAME = f"{TASK_ID}.py"


def settings(*, token: str | None = TOKEN) -> Settings:
    return Settings(
        _env_file=None,
        hf_token=token,
        download_timeout_seconds=5,
    )


@pytest.mark.asyncio
async def test_download_uses_only_exact_answer_blind_attachment_path() -> None:
    observed_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(request)
        return httpx.Response(
            200,
            content=b"print(42)\n",
            headers={
                "content-type": "text/plain; charset=utf-8",
                "x-repo-commit": "abc123commit",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(), http_client=http_client)
        attachment = await client.download_attachment(
            task_id=TASK_ID,
            expected_file_name=FILE_NAME,
            max_bytes=1024,
        )

    assert len(observed_requests) == 1
    request = observed_requests[0]
    assert request.url.path == (
        "/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/" + FILE_NAME
    )
    assert request.url.params["download"] == "true"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    forbidden_fragments = ("/rows", "parquet", "metadata")
    assert all(fragment not in str(request.url).lower() for fragment in forbidden_fragments)
    assert attachment.content == b"print(42)\n"
    assert attachment.repo_path == f"2023/validation/{FILE_NAME}"
    assert attachment.resolved_revision == "abc123commit"


@pytest.mark.asyncio
async def test_missing_token_fails_before_any_http_request() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(token=None), http_client=http_client)
        with pytest.raises(GatedDatasetAccessError, match="HF_TOKEN is required"):
            await client.download_attachment(
                task_id=TASK_ID,
                expected_file_name=FILE_NAME,
                max_bytes=1024,
            )

    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_denied_access_fails_loudly_without_exposing_token(status_code: int) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status_code))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(), http_client=http_client)
        with pytest.raises(GatedDatasetAccessError) as caught:
            await client.download_attachment(
                task_id=TASK_ID,
                expected_file_name=FILE_NAME,
                max_bytes=1024,
            )

    assert TOKEN not in str(caught.value)
    assert TOKEN not in repr(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (404, GatedDatasetAttachmentNotFoundError),
        (429, GatedDatasetTransientError),
        (500, GatedDatasetTransientError),
        (400, GatedDatasetResponseError),
    ],
)
async def test_hub_statuses_are_classified(
    status_code: int,
    error_type: type[Exception],
) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status_code))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(), http_client=http_client)
        with pytest.raises(error_type):
            await client.download_attachment(
                task_id=TASK_ID,
                expected_file_name=FILE_NAME,
                max_bytes=1024,
            )


@pytest.mark.asyncio
async def test_empty_and_oversized_downloads_are_rejected() -> None:
    responses = iter(
        [
            httpx.Response(200, content=b""),
            httpx.Response(200, content=b"ninebytes"),
        ]
    )
    transport = httpx.MockTransport(lambda _request: next(responses))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(), http_client=http_client)
        with pytest.raises(GatedDatasetResponseError, match="empty"):
            await client.download_attachment(
                task_id=TASK_ID,
                expected_file_name=FILE_NAME,
                max_bytes=8,
            )
        with pytest.raises(GatedDatasetDownloadTooLargeError):
            await client.download_attachment(
                task_id=TASK_ID,
                expected_file_name=FILE_NAME,
                max_bytes=8,
            )


@pytest.mark.parametrize(
    ("task_id", "file_name"),
    [
        ("../task", FILE_NAME),
        (TASK_ID, "../" + FILE_NAME),
        (TASK_ID, f"different-{TASK_ID}.py"),
        (TASK_ID, f"{TASK_ID}.parquet"),
        (TASK_ID, f"{TASK_ID}.json"),
    ],
)
def test_policy_permits_only_exact_supported_task_attachment(
    task_id: str,
    file_name: str,
) -> None:
    with pytest.raises(GatedDatasetPolicyError):
        GaiaGatedDatasetClient.answer_blind_repo_path(task_id, file_name)


def test_policy_returns_only_current_validation_attachment_path() -> None:
    path = GaiaGatedDatasetClient.answer_blind_repo_path(TASK_ID, FILE_NAME)

    assert path == f"2023/validation/{FILE_NAME}"
    assert Path(path).suffix == ".py"


def test_client_exposes_no_dataset_row_or_submission_methods() -> None:
    forbidden = (
        "get_rows",
        "get_parquet",
        "load_dataset",
        "metadata",
        "submit",
        "submit_answers",
    )
    assert all(not hasattr(GaiaGatedDatasetClient, name) for name in forbidden)


def test_raw_attachment_schema_contains_no_answer_fields() -> None:
    field_names = {field.name for field in fields(GatedRawAttachment)}

    assert all("answer" not in name.lower() for name in field_names)


@pytest.mark.asyncio
async def test_download_logs_neither_token_nor_attachment_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = b"print('sensitive attachment content')\n"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=content,
            headers={"content-type": "text/x-python"},
        )
    )
    caplog.set_level(logging.DEBUG)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        client = GaiaGatedDatasetClient(settings(), http_client=http_client)
        await client.download_attachment(
            task_id=TASK_ID,
            expected_file_name=FILE_NAME,
            max_bytes=1024,
        )

    captured = caplog.text
    assert TOKEN not in captured
    assert content.decode() not in captured
    assert "final answer" not in captured.lower()
