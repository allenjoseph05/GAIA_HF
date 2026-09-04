"""Answer-blind file access for the official gated GAIA dataset repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from gaia_max.attachment_validation import SUPPORTED_ATTACHMENT_EXTENSIONS
from gaia_max.config import Settings

GAIA_DATASET_REPO_ID = "gaia-benchmark/GAIA"
GAIA_DATASET_REVISION = "main"
GAIA_ATTACHMENT_PREFIX = "2023/validation"


class GatedDatasetError(RuntimeError):
    """Base error for official gated-dataset attachment access."""


class GatedDatasetAccessError(GatedDatasetError):
    """A token is absent, invalid, or lacks accepted gated-dataset access."""


class GatedDatasetAttachmentNotFoundError(GatedDatasetError):
    """The exact allowlisted attachment path does not exist."""


class GatedDatasetTransientError(GatedDatasetError):
    """The official Hub request may succeed if retried later."""


class GatedDatasetResponseError(GatedDatasetError):
    """The official Hub returned a permanent or malformed response."""


class GatedDatasetDownloadTooLargeError(GatedDatasetResponseError):
    """The gated attachment exceeded the configured streaming limit."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"gated attachment exceeded {max_bytes} bytes")


class GatedDatasetPolicyError(GatedDatasetError):
    """A requested path would violate the answer-blind attachment allowlist."""


@dataclass(frozen=True, slots=True)
class GatedRawAttachment:
    """Unvalidated bytes from one exact file in the official gated repository."""

    content: bytes
    content_type: str | None
    repo_id: str
    repo_path: str
    requested_revision: str
    resolved_revision: str


class GaiaGatedDatasetClient:
    """Download exact attachment files without requesting dataset records or metadata."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url="https://huggingface.co",
            timeout=httpx.Timeout(settings.download_timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": "gaia-max/0.1 answer-blind-attachment-client"},
        )

    async def __aenter__(self) -> GaiaGatedDatasetClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def download_attachment(
        self,
        *,
        task_id: str,
        expected_file_name: str,
        max_bytes: int,
    ) -> GatedRawAttachment:
        """Stream one exact current-validation attachment through a strict path policy."""

        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        token = self._token()
        repo_path = self.answer_blind_repo_path(task_id, expected_file_name)
        request_path = self._resolve_request_path(repo_path)

        try:
            async with self._client.stream(
                "GET",
                request_path,
                params={"download": "true"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(self._settings.download_timeout_seconds),
            ) as response:
                self._raise_for_status(response, repo_path)
                declared_length = self._content_length(response)
                if declared_length is not None and declared_length > max_bytes:
                    raise GatedDatasetDownloadTooLargeError(max_bytes)

                chunks: list[bytes] = []
                received_bytes = 0
                async for chunk in response.aiter_bytes():
                    received_bytes += len(chunk)
                    if received_bytes > max_bytes:
                        raise GatedDatasetDownloadTooLargeError(max_bytes)
                    chunks.append(chunk)
                content = b"".join(chunks)
                if not content:
                    raise GatedDatasetResponseError("gated attachment response was empty")

                resolved_revision = response.headers.get(
                    "x-repo-commit",
                    GAIA_DATASET_REVISION,
                )
                return GatedRawAttachment(
                    content=content,
                    content_type=response.headers.get("content-type"),
                    repo_id=GAIA_DATASET_REPO_ID,
                    repo_path=repo_path,
                    requested_revision=GAIA_DATASET_REVISION,
                    resolved_revision=resolved_revision,
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise GatedDatasetTransientError("temporary gated-dataset download failure") from exc

    @staticmethod
    def answer_blind_repo_path(task_id: str, expected_file_name: str) -> str:
        """Build the only dataset path shape this client is permitted to request."""

        if not task_id or Path(task_id).name != task_id or any(
            separator in task_id for separator in ("/", "\\")
        ):
            raise GatedDatasetPolicyError("task ID is unsafe for attachment resolution")
        if (
            not expected_file_name
            or Path(expected_file_name).name != expected_file_name
            or any(separator in expected_file_name for separator in ("/", "\\"))
        ):
            raise GatedDatasetPolicyError("expected attachment must be a safe basename")
        expected_path = Path(expected_file_name)
        if expected_path.suffix.lower() not in SUPPORTED_ATTACHMENT_EXTENSIONS:
            raise GatedDatasetPolicyError("expected attachment extension is not allowlisted")
        if expected_path.stem != task_id:
            raise GatedDatasetPolicyError("attachment filename must match its public task ID")
        return f"{GAIA_ATTACHMENT_PREFIX}/{expected_file_name}"

    def _token(self) -> str:
        if self._settings.hf_token is None:
            raise GatedDatasetAccessError(
                "HF_TOKEN is required after accepting access to gaia-benchmark/GAIA"
            )
        token = self._settings.hf_token.get_secret_value().strip()
        if not token:
            raise GatedDatasetAccessError(
                "HF_TOKEN is required after accepting access to gaia-benchmark/GAIA"
            )
        return token

    @staticmethod
    def _resolve_request_path(repo_path: str) -> str:
        quoted_repo = "/".join(quote(part, safe="") for part in GAIA_DATASET_REPO_ID.split("/"))
        quoted_revision = quote(GAIA_DATASET_REVISION, safe="")
        quoted_path = "/".join(quote(part, safe="") for part in repo_path.split("/"))
        return f"/datasets/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"

    @staticmethod
    def _raise_for_status(response: httpx.Response, repo_path: str) -> None:
        if response.status_code in {401, 403}:
            raise GatedDatasetAccessError(
                "HF_TOKEN is invalid or does not have accepted GAIA dataset access"
            )
        if response.status_code == 404:
            raise GatedDatasetAttachmentNotFoundError(
                f"official gated attachment was not found: {repo_path}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise GatedDatasetTransientError(
                f"temporary Hugging Face Hub HTTP {response.status_code}"
            )
        if response.is_error:
            raise GatedDatasetResponseError(
                f"permanent Hugging Face Hub HTTP {response.status_code}"
            )

    @staticmethod
    def _content_length(response: httpx.Response) -> int | None:
        raw_value = response.headers.get("content-length")
        if raw_value is None:
            return None
        try:
            value = int(raw_value)
        except ValueError:
            return None
        return value if value >= 0 else None
