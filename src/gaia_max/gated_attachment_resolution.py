"""A10 resolution through exact files in the official gated GAIA repository."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from gaia_max.artifacts import ArtifactStore
from gaia_max.attachment_resolution import (
    OfficialAttachmentOutcome,
    OfficialAttachmentStatus,
    OfficialFallbackReason,
)
from gaia_max.attachment_validation import (
    AttachmentRejectionReason,
    AttachmentValidationError,
    AttachmentValidator,
)
from gaia_max.clients import (
    GaiaGatedDatasetClient,
    GatedDatasetDownloadTooLargeError,
    GatedDatasetTransientError,
)
from gaia_max.config import Settings
from gaia_max.domain import ArtifactRef, ArtifactSource, Question

AsyncSleep = Callable[[float], Awaitable[None]]


class GatedAttachmentResolution(BaseModel):
    """Successful immutable result from the official gated fallback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    expected_file_name: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    official_fallback_reason: OfficialFallbackReason
    repo_path: str = Field(min_length=1)
    resolved_revision: str = Field(min_length=1)
    artifact: ArtifactRef


class GatedAttachmentRetriesExhaustedError(RuntimeError):
    """All bounded attempts against the gated file route were transient failures."""

    def __init__(self, task_id: str, attempts: int) -> None:
        self.task_id = task_id
        self.attempts = attempts
        super().__init__(
            f"gated attachment retries exhausted for task {task_id} after {attempts} attempts"
        )


class GatedAttachmentRejectedError(RuntimeError):
    """The official gated bytes violated attachment validation policy."""

    def __init__(
        self,
        task_id: str,
        reason: AttachmentRejectionReason,
        attempts: int,
    ) -> None:
        self.task_id = task_id
        self.reason = reason
        self.attempts = attempts
        super().__init__(f"gated attachment for task {task_id} was rejected: {reason.value}")


class GatedAttachmentResolver:
    """Consume an A09 fallback handoff and resolve one answer-blind dataset file."""

    def __init__(
        self,
        *,
        client: GaiaGatedDatasetClient,
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

    async def resolve(
        self,
        question: Question,
        official_outcome: OfficialAttachmentOutcome,
    ) -> GatedAttachmentResolution:
        """Resolve only after a matching, explicit fallback result from A09."""

        file_name = self._validate_handoff(question, official_outcome)
        for attempt in range(1, self._settings.attachment_max_attempts + 1):
            try:
                raw = await self._client.download_attachment(
                    task_id=question.task_id,
                    expected_file_name=file_name,
                    max_bytes=self._validator.max_bytes,
                )
            except GatedDatasetDownloadTooLargeError as exc:
                raise GatedAttachmentRejectedError(
                    question.task_id,
                    AttachmentRejectionReason.TOO_LARGE,
                    attempt,
                ) from exc
            except GatedDatasetTransientError as exc:
                if attempt == self._settings.attachment_max_attempts:
                    raise GatedAttachmentRetriesExhaustedError(
                        question.task_id,
                        attempt,
                    ) from exc
                await self._sleep(self._retry_delay(attempt))
                continue

            try:
                validation = self._validator.validate(
                    raw.content,
                    original_name=file_name,
                    media_type=raw.content_type,
                )
            except AttachmentValidationError as exc:
                raise GatedAttachmentRejectedError(
                    question.task_id,
                    exc.reason,
                    attempt,
                ) from exc

            source_locator = (
                f"hf://datasets/{raw.repo_id}@{raw.resolved_revision}/{raw.repo_path}"
            )
            artifact = self._store.put_bytes(
                raw.content,
                original_name=file_name,
                media_type=validation.canonical_media_type,
                source=ArtifactSource.OFFICIAL_GATED_DATASET,
                task_id=question.task_id,
                source_locator=source_locator,
                validation=validation,
            )
            if official_outcome.fallback_reason is None:
                raise RuntimeError("validated fallback handoff lost its reason")
            return GatedAttachmentResolution(
                task_id=question.task_id,
                expected_file_name=file_name,
                attempts=attempt,
                official_fallback_reason=official_outcome.fallback_reason,
                repo_path=raw.repo_path,
                resolved_revision=raw.resolved_revision,
                artifact=artifact,
            )

        raise RuntimeError("unreachable gated attachment retry state")

    @staticmethod
    def _validate_handoff(
        question: Question,
        official_outcome: OfficialAttachmentOutcome,
    ) -> str:
        if question.file_name is None:
            raise ValueError(f"task {question.task_id} has no attachment filename")
        if official_outcome.status is not OfficialAttachmentStatus.FALLBACK_REQUIRED:
            raise ValueError("gated resolver requires an A09 fallback_required outcome")
        if official_outcome.task_id != question.task_id:
            raise ValueError("fallback task ID does not match the question")
        if official_outcome.expected_file_name != question.file_name:
            raise ValueError("fallback filename does not match the question")
        return question.file_name

    def _retry_delay(self, completed_attempt: int) -> float:
        return self._settings.attachment_retry_base_seconds * 2 ** (completed_attempt - 1)
