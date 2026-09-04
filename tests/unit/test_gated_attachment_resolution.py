from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_resolution import (
    OfficialAttachmentOutcome,
    OfficialAttachmentStatus,
    OfficialFallbackReason,
)
from gaia_max.attachment_validation import AttachmentRejectionReason, AttachmentValidator
from gaia_max.clients import (
    GaiaGatedDatasetClient,
    GatedDatasetAccessError,
    GatedDatasetAttachmentNotFoundError,
)
from gaia_max.config import Settings
from gaia_max.domain import ArtifactSource, Question
from gaia_max.gated_attachment_resolution import (
    GatedAttachmentRejectedError,
    GatedAttachmentResolution,
    GatedAttachmentResolver,
    GatedAttachmentRetriesExhaustedError,
)

TOKEN = "hf_private_test_token"
TASK_ID = "f918266a-b3e0-4914-865d-4faa564f1aef"
FILE_NAME = f"{TASK_ID}.py"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "hf_token": TOKEN,
        "attachment_max_attempts": 3,
        "attachment_retry_base_seconds": 0.25,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def question(file_name: str | None = FILE_NAME) -> Question:
    return Question(
        task_id=TASK_ID,
        question="Inspect the attached Python file.",
        file_name=file_name,
    )


def fallback(
    *,
    task_id: str = TASK_ID,
    file_name: str = FILE_NAME,
) -> OfficialAttachmentOutcome:
    return OfficialAttachmentOutcome(
        status=OfficialAttachmentStatus.FALLBACK_REQUIRED,
        task_id=task_id,
        expected_file_name=file_name,
        attempts=1,
        fallback_reason=OfficialFallbackReason.NOT_FOUND,
    )


async def no_sleep(_delay: float) -> None:
    return None


def resolver(
    tmp_path: Path,
    http_client: httpx.AsyncClient,
    *,
    app_settings: Settings | None = None,
    sleep=no_sleep,
) -> GatedAttachmentResolver:
    resolved_settings = app_settings or settings()
    return GatedAttachmentResolver(
        client=GaiaGatedDatasetClient(resolved_settings, http_client=http_client),
        store=ArtifactStore(tmp_path / "artifacts"),
        validator=AttachmentValidator(max_bytes=resolved_settings.max_attachment_bytes),
        settings=resolved_settings,
        sleep=sleep,
    )


@pytest.mark.asyncio
async def test_fallback_validates_stores_and_records_gated_provenance(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"print(42)\n",
            headers={
                "content-type": "application/octet-stream",
                "x-repo-commit": "commit-sha-123",
            },
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client).resolve(question(), fallback())

    assert outcome.attempts == 1
    assert outcome.repo_path == f"2023/validation/{FILE_NAME}"
    assert outcome.resolved_revision == "commit-sha-123"
    assert outcome.official_fallback_reason is OfficialFallbackReason.NOT_FOUND
    assert outcome.artifact.source is ArtifactSource.OFFICIAL_GATED_DATASET
    assert outcome.artifact.local_path.read_bytes() == b"print(42)\n"

    metadata = ArtifactStore(tmp_path / "artifacts").metadata(outcome.artifact.artifact_id)
    origin = metadata.origins[0]
    assert origin.task_id == TASK_ID
    assert origin.source_locator == (
        f"hf://datasets/gaia-benchmark/GAIA@commit-sha-123/2023/validation/{FILE_NAME}"
    )
    assert origin.validation is not None
    assert origin.validation.sha256 == outcome.artifact.sha256


@pytest.mark.asyncio
async def test_transient_hub_errors_retry_with_backoff_then_succeed(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            content=b"print('ok')\n",
            headers={"content-type": "text/x-python"},
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        outcome = await resolver(tmp_path, http_client, sleep=record_sleep).resolve(
            question(),
            fallback(),
        )

    assert outcome.attempts == 3
    assert calls == 3
    assert delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_transient_retries_exhaust_loudly(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        with pytest.raises(GatedAttachmentRetriesExhaustedError) as caught:
            await resolver(tmp_path, http_client).resolve(question(), fallback())

    assert caught.value.attempts == 3
    assert calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_missing_gated_access_fails_loudly_without_retry(
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
        base_url="https://huggingface.test",
    ) as http_client:
        with pytest.raises(GatedDatasetAccessError):
            await resolver(tmp_path, http_client).resolve(question(), fallback())

    assert calls == 1


@pytest.mark.asyncio
async def test_missing_exact_gated_file_fails_loudly_without_searching_rows(
    tmp_path: Path,
) -> None:
    observed_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_paths.append(request.url.path)
        return httpx.Response(404)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        with pytest.raises(GatedDatasetAttachmentNotFoundError):
            await resolver(tmp_path, http_client).resolve(question(), fallback())

    assert observed_paths == [
        f"/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{FILE_NAME}"
    ]


@pytest.mark.asyncio
async def test_invalid_gated_bytes_stop_and_are_never_stored(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            content=b"<!doctype html><title>denied</title>",
            headers={"content-type": "text/html"},
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://huggingface.test",
    ) as http_client:
        with pytest.raises(GatedAttachmentRejectedError) as caught:
            await resolver(tmp_path, http_client).resolve(question(), fallback())

    assert caught.value.reason is AttachmentRejectionReason.HTML_ERROR_RESPONSE
    assert list((tmp_path / "artifacts" / "sha256").rglob("content")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "handoff", "message"),
    [
        (question(None), fallback(), "no attachment filename"),
        (question(), fallback(task_id="different-task"), "task ID"),
        (question(), fallback(file_name=f"{TASK_ID}.mp3"), "filename"),
    ],
)
async def test_mismatched_a09_handoff_never_calls_gated_dataset(
    tmp_path: Path,
    task: Question,
    handoff: OfficialAttachmentOutcome,
    message: str,
) -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://huggingface.test",
    ) as http_client:
        with pytest.raises(ValueError, match=message):
            await resolver(tmp_path, http_client).resolve(task, handoff)

    assert called is False


def test_gated_models_contain_no_answer_fields() -> None:
    result_fields = set(GatedAttachmentResolution.model_fields)

    assert all("answer" not in name.lower() for name in result_fields)
