"""Deliberate, one-shot submission of an already frozen candidate manifest."""

from __future__ import annotations

from datetime import datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from gaia_max.config import Settings
from gaia_max.preflight import FrozenCandidateManifest

APPROVAL_PHRASE = "SUBMIT FROZEN GAIA CANDIDATE"


class SubmissionBlockedError(RuntimeError):
    """Raised before network access when a submission safety gate fails."""


class SubmissionTransportError(RuntimeError):
    """Answer-redacted failure for an attempted one-shot network operation."""


class SubmissionApproval(BaseModel):
    """Human approval bound to one immutable candidate hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator: str = Field(min_length=1)
    approved_at: datetime
    phrase: str

    @model_validator(mode="after")
    def require_explicit_phrase_and_timezone(self) -> SubmissionApproval:
        if self.phrase != APPROVAL_PHRASE:
            raise ValueError("submission approval phrase does not match")
        if self.approved_at.tzinfo is None or self.approved_at.utcoffset() is None:
            raise ValueError("submission approval timestamp must include a timezone")
        return self


class SubmissionReceipt(BaseModel):
    """Aggregate scorer response; intentionally contains no per-task answers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    username: str = Field(min_length=1)
    score: float
    correct_count: int = Field(ge=0)
    total_attempted: int = Field(ge=0)
    message: str
    timestamp: str

    @model_validator(mode="after")
    def validate_aggregate_counts(self) -> SubmissionReceipt:
        if self.correct_count > self.total_attempted:
            raise ValueError("correct count cannot exceed attempted count")
        if not 0 <= self.score <= 100:
            raise ValueError("score must be a percentage between zero and 100")
        return self


class _SubmittedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    submitted_answer: str = Field(min_length=1)


class _SubmissionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    username: str = Field(min_length=1)
    agent_code: HttpUrl
    answers: tuple[_SubmittedAnswer, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_owned_hugging_face_space(self) -> _SubmissionPayload:
        path_parts = (self.agent_code.path or "").strip("/").split("/")
        if (
            self.agent_code.host != "huggingface.co"
            or len(path_parts) < 3
            or path_parts[0] != "spaces"
            or path_parts[1] != self.username
        ):
            raise ValueError("agent code must be a Hugging Face Space owned by username")
        return self


class GaiaSubmissionClient:
    """A no-retry client whose only side effect is one approved submission."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=str(settings.gaia_api_url).rstrip("/"),
            timeout=settings.http_timeout_seconds,
        )

    async def __aenter__(self) -> GaiaSubmissionClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def submit_once(
        self,
        *,
        manifest: FrozenCandidateManifest,
        approval: SubmissionApproval,
        username: str,
        agent_code: str,
    ) -> SubmissionReceipt:
        """Submit exactly once after all local gates pass; never retry automatically."""

        self.validate_candidate(manifest)
        self._validate_approval(
            manifest=manifest,
            approval=approval,
            username=username,
        )
        try:
            payload = _SubmissionPayload(
                username=username,
                agent_code=HttpUrl(agent_code),
                answers=tuple(
                    _SubmittedAnswer(
                        task_id=entry.task_id,
                        submitted_answer=entry.answer,
                    )
                    for entry in manifest.answers
                ),
            )
        except ValueError as exc:
            raise SubmissionBlockedError("submission identity or Space URL is invalid") from exc

        try:
            response = await self._client.post(
                "/submit",
                json=payload.model_dump(mode="json"),
            )
        except httpx.HTTPError as exc:
            raise SubmissionTransportError(
                "submission outcome is unknown; do not retry until scorer state is checked"
            ) from exc

        if not 200 <= response.status_code < 300:
            raise SubmissionTransportError(
                f"scorer rejected the submission with HTTP {response.status_code}"
            )
        try:
            return SubmissionReceipt.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise SubmissionTransportError("scorer returned an invalid aggregate receipt") from exc

    def validate_candidate(self, manifest: FrozenCandidateManifest) -> None:
        """Check configuration and full inventory before requesting human approval."""

        if not self._settings.submission_enabled:
            raise SubmissionBlockedError("submission is disabled by configuration")
        if len(manifest.answers) != self._settings.expected_question_count:
            raise SubmissionBlockedError(
                "frozen candidate does not contain the expected task count"
            )

    @staticmethod
    def _validate_approval(
        *,
        manifest: FrozenCandidateManifest,
        approval: SubmissionApproval,
        username: str,
    ) -> None:
        if approval.candidate_sha256 != manifest.candidate_sha256:
            raise SubmissionBlockedError("approval is not bound to this frozen candidate")
        if approval.operator != username:
            raise SubmissionBlockedError("approval operator must match submission username")


__all__ = [
    "APPROVAL_PHRASE",
    "GaiaSubmissionClient",
    "SubmissionApproval",
    "SubmissionBlockedError",
    "SubmissionReceipt",
    "SubmissionTransportError",
]
