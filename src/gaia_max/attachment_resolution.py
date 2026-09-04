"""Preferred official-endpoint attachment acquisition and fallback routing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_validation import (
    AttachmentRejectionReason,
    AttachmentValidationError,
    AttachmentValidator,
)
from gaia_max.clients import (
    GaiaApiClient,
    GaiaApiDownloadTooLargeError,
    GaiaApiNotFoundError,
    GaiaApiResponseError,
    GaiaApiTransientError,
)
from gaia_max.config import Settings
from gaia_max.domain import ArtifactRef, ArtifactSource, Question

AsyncSleep = Callable[[float], Awaitable[None]]


class OfficialAttachmentStatus(StrEnum):
    """Whether the preferred official endpoint resolved the attachment."""

    RESOLVED = "resolved"
    FALLBACK_REQUIRED = "fallback_required"


class OfficialFallbackReason(StrEnum):
    """Why orchestration should invoke the official gated-dataset fallback."""

    NOT_FOUND = "not_found"
    TRANSIENT_RETRIES_EXHAUSTED = "transient_retries_exhausted"
    RESPONSE_REJECTED = "response_rejected"
    VALIDATION_REJECTED = "validation_rejected"


class OfficialAttachmentOutcome(BaseModel):
    """Typed A09 result consumed by the future A10 fallback node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: OfficialAttachmentStatus
    task_id: str = Field(min_length=1)
    expected_file_name: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    artifact: ArtifactRef | None = None
    fallback_reason: OfficialFallbackReason | None = None
    fallback_validation_reason: AttachmentRejectionReason | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> OfficialAttachmentOutcome:
        if self.status is OfficialAttachmentStatus.RESOLVED:
            if self.artifact is None or self.fallback_reason is not None:
                raise ValueError("resolved outcome requires only an artifact")
            if self.fallback_validation_reason is not None:
                raise ValueError("resolved outcome cannot include a fallback validation reason")
        elif self.artifact is not None or self.fallback_reason is None:
            raise ValueError("fallback outcome requires a reason and no artifact")
        if (
            self.fallback_validation_reason is not None
            and self.fallback_reason is not OfficialFallbackReason.VALIDATION_REJECTED
        ):
            raise ValueError("validation reason requires a validation-rejected fallback")
        return self


class OfficialAttachmentRejectedError(RuntimeError):
    """The official response is unsafe or unsupported and must not be bypassed."""

    def __init__(
        self,
        task_id: str,
        reason: AttachmentRejectionReason,
        attempts: int,
    ) -> None:
        self.task_id = task_id
        self.reason = reason
        self.attempts = attempts
        super().__init__(
            f"official attachment for task {task_id} was rejected: {reason.value}"
        )


_FALLBACK_VALIDATION_REASONS = {
    AttachmentRejectionReason.EMPTY,
    AttachmentRejectionReason.MISSING_MEDIA_TYPE,
    AttachmentRejectionReason.MEDIA_TYPE_MISMATCH,
    AttachmentRejectionReason.HTML_ERROR_RESPONSE,
    AttachmentRejectionReason.CONTENT_MISMATCH,
    AttachmentRejectionReason.MALFORMED_CONTENT,
}


class OfficialAttachmentResolver:
    """Resolve through `/files` with bounded retries, validation, and immutable caching."""

    def __init__(
        self,
        *,
        client: GaiaApiClient,
        store: ArtifactStore,
        validator: AttachmentValidator,
        settings: Settings,
        sleep: AsyncSleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._store = store
        self._validator = validator
        self._settings = settings
        self._sleep = sleep

    async def resolve(self, question: Question) -> OfficialAttachmentOutcome:
        """Attempt the preferred route and return a deliberate A10 handoff when needed."""

        if question.file_name is None:
            raise ValueError(f"task {question.task_id} has no attachment filename")

        for attempt in range(1, self._settings.attachment_max_attempts + 1):
            try:
                raw = await self._client.get_attachment(
                    question.task_id,
                    max_bytes=self._validator.max_bytes,
                )
            except GaiaApiNotFoundError:
                return self._fallback(
                    question,
                    attempts=attempt,
                    reason=OfficialFallbackReason.NOT_FOUND,
                )
            except GaiaApiDownloadTooLargeError as exc:
                raise OfficialAttachmentRejectedError(
                    question.task_id,
                    AttachmentRejectionReason.TOO_LARGE,
                    attempt,
                ) from exc
            except GaiaApiTransientError:
                if attempt == self._settings.attachment_max_attempts:
                    return self._fallback(
                        question,
                        attempts=attempt,
                        reason=OfficialFallbackReason.TRANSIENT_RETRIES_EXHAUSTED,
                    )
                await self._sleep(self._retry_delay(attempt))
                continue
            except GaiaApiResponseError:
                return self._fallback(
                    question,
                    attempts=attempt,
                    reason=OfficialFallbackReason.RESPONSE_REJECTED,
                )

            try:
                validation = self._validator.validate(
                    raw.content,
                    original_name=question.file_name,
                    media_type=raw.content_type,
                )
            except AttachmentValidationError as exc:
                if exc.reason in _FALLBACK_VALIDATION_REASONS:
                    return self._fallback(
                        question,
                        attempts=attempt,
                        reason=OfficialFallbackReason.VALIDATION_REJECTED,
                        validation_reason=exc.reason,
                    )
                raise OfficialAttachmentRejectedError(
                    question.task_id,
                    exc.reason,
                    attempt,
                ) from exc

            artifact = self._store.put_bytes(
                raw.content,
                original_name=question.file_name,
                media_type=validation.canonical_media_type,
                source=ArtifactSource.OFFICIAL_API,
                task_id=question.task_id,
                source_locator=self._attachment_url(question.task_id),
                validation=validation,
            )
            return OfficialAttachmentOutcome(
                status=OfficialAttachmentStatus.RESOLVED,
                task_id=question.task_id,
                expected_file_name=question.file_name,
                attempts=attempt,
                artifact=artifact,
            )

        raise RuntimeError("unreachable attachment retry state")

    def _retry_delay(self, completed_attempt: int) -> float:
        return self._settings.attachment_retry_base_seconds * 2 ** (completed_attempt - 1)

    def _attachment_url(self, task_id: str) -> str:
        base_url = str(self._settings.gaia_api_url).rstrip("/")
        return f"{base_url}/files/{quote(task_id, safe='')}"

    @staticmethod
    def _fallback(
        question: Question,
        *,
        attempts: int,
        reason: OfficialFallbackReason,
        validation_reason: AttachmentRejectionReason | None = None,
    ) -> OfficialAttachmentOutcome:
        if question.file_name is None:
            raise ValueError("fallback requires an attachment filename")
        return OfficialAttachmentOutcome(
            status=OfficialAttachmentStatus.FALLBACK_REQUIRED,
            task_id=question.task_id,
            expected_file_name=question.file_name,
            attempts=attempts,
            fallback_reason=reason,
            fallback_validation_reason=validation_reason,
        )
