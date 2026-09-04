"""Read-only client for the official Hugging Face Agents Course API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from gaia_max.config import Settings
from gaia_max.domain import Question


class GaiaApiError(RuntimeError):
    """Base error for an official GAIA API read operation."""


class GaiaApiNotFoundError(GaiaApiError):
    """The requested official resource does not exist."""


class GaiaApiTransientError(GaiaApiError):
    """A read may succeed later, for example after a timeout or 5xx response."""


class GaiaApiResponseError(GaiaApiError):
    """The service returned a permanent or malformed response."""


class GaiaApiDownloadTooLargeError(GaiaApiResponseError):
    """An attachment exceeded its configured streaming byte ceiling."""

    def __init__(self, task_id: str, max_bytes: int) -> None:
        self.task_id = task_id
        self.max_bytes = max_bytes
        super().__init__(
            f"official attachment for task {task_id} exceeded {max_bytes} bytes"
        )


@dataclass(frozen=True, slots=True)
class RawAttachment:
    """Unvalidated attachment bytes from the preferred official endpoint.

    MIME, magic-byte, filename, and size validation belongs to story GAIA-A08.
    """

    task_id: str
    content: bytes
    content_type: str | None
    content_disposition: str | None


class GaiaApiClient:
    """Typed, read-only adapter around `/questions`, `/random-question`, and `/files`.

    This class deliberately has no submission method. The write capability is
    deferred to the final deployment milestone and will live in a separate client.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=str(settings.gaia_api_url),
            timeout=httpx.Timeout(settings.http_timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": "gaia-max/0.1 read-only-course-client"},
        )

    async def __aenter__(self) -> GaiaApiClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_questions(self) -> list[Question]:
        response = await self._request("GET", "/questions")
        payload = self._parse_json(response)
        if not isinstance(payload, list):
            raise GaiaApiResponseError("/questions must return a JSON list")

        try:
            questions = [Question.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise GaiaApiResponseError("/questions returned an invalid question record") from exc

        if not questions:
            raise GaiaApiResponseError("/questions returned an empty list")
        return questions

    async def get_random_question(self) -> Question:
        response = await self._request("GET", "/random-question")
        payload = self._parse_json(response)
        try:
            return Question.model_validate(payload)
        except ValidationError as exc:
            raise GaiaApiResponseError("/random-question returned an invalid record") from exc

    async def get_attachment(
        self,
        task_id: str,
        *,
        max_bytes: int | None = None,
    ) -> RawAttachment:
        if not task_id.strip():
            raise ValueError("task_id cannot be empty")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("max_bytes must be positive")

        path = f"/files/{quote(task_id, safe='')}"
        try:
            async with self._client.stream(
                "GET",
                path,
                timeout=httpx.Timeout(self._settings.download_timeout_seconds),
            ) as response:
                self._raise_for_status(response, path)
                declared_length = self._content_length(response)
                if (
                    max_bytes is not None
                    and declared_length is not None
                    and declared_length > max_bytes
                ):
                    raise GaiaApiDownloadTooLargeError(task_id, max_bytes)

                chunks: list[bytes] = []
                received_bytes = 0
                async for chunk in response.aiter_bytes():
                    received_bytes += len(chunk)
                    if max_bytes is not None and received_bytes > max_bytes:
                        raise GaiaApiDownloadTooLargeError(task_id, max_bytes)
                    chunks.append(chunk)

                content = b"".join(chunks)
                if not content:
                    raise GaiaApiResponseError(
                        f"official attachment for task {task_id} was empty"
                    )
                return RawAttachment(
                    task_id=task_id,
                    content=content,
                    content_type=response.headers.get("content-type"),
                    content_disposition=response.headers.get("content-disposition"),
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GaiaApiTransientError(f"temporary GAIA API failure for {path}") from exc

    async def _request(self, method: str, path: str) -> httpx.Response:
        try:
            response = await self._client.request(method, path)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GaiaApiTransientError(f"temporary GAIA API failure for {path}") from exc

        self._raise_for_status(response, path)
        return response

    @staticmethod
    def _raise_for_status(response: httpx.Response, path: str) -> None:
        if response.status_code == 404:
            raise GaiaApiNotFoundError(f"official GAIA resource was not found: {path}")
        if response.status_code == 429 or response.status_code >= 500:
            raise GaiaApiTransientError(
                f"temporary GAIA API HTTP {response.status_code} for {path}"
            )
        if response.is_error:
            raise GaiaApiResponseError(
                f"permanent GAIA API HTTP {response.status_code} for {path}"
            )

    @staticmethod
    def _content_length(response: httpx.Response) -> int | None:
        value = response.headers.get("content-length")
        if value is None:
            return None
        try:
            length = int(value)
        except ValueError:
            return None
        return length if length >= 0 else None

    @staticmethod
    def _parse_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise GaiaApiResponseError("official GAIA API returned malformed JSON") from exc
