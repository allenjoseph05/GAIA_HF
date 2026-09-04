"""Bounded Wayback capture discovery and explicit temporal-validity policy."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import cast
from urllib.parse import urlsplit

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictStr,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from gaia_max.config import Settings
from gaia_max.domain import TemporalConstraint

AsyncSleep = Callable[[float], Awaitable[None]]

_HTTP_URL = TypeAdapter(HttpUrl)
_CDX_TIMESTAMP = re.compile(r"^\d{14}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CDX_FIELDS = ("timestamp", "original", "statuscode", "mimetype", "digest")


class ArchiveError(RuntimeError):
    """Sanitized archive-index failure."""

    def __init__(self, code: str) -> None:
        if _SAFE_CODE.fullmatch(code) is None:
            raise ValueError("archive error code must be a safe identifier")
        self.code = code
        super().__init__(f"archive request failed: {code}")


class ArchiveNotFoundError(ArchiveError):
    """No compatible capture exists for the requested URL and interval."""


class ArchiveCapture(BaseModel):
    """One successful archived capture with its content-state timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_url: HttpUrl
    captured_at: AwareDatetime
    cdx_timestamp: str = Field(pattern=_CDX_TIMESTAMP.pattern)
    status_code: int = Field(default=200, ge=200, le=200)
    media_type: StrictStr = Field(min_length=1, max_length=255)
    digest: StrictStr | None = Field(default=None, max_length=255)
    replay_url: HttpUrl

    @field_validator("captured_at")
    @classmethod
    def normalize_capture_time(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_serializer("original_url", "replay_url")
    def serialize_url(self, value: HttpUrl) -> str:
        return str(value)


def select_latest_capture(
    captures: Sequence[ArchiveCapture],
    cutoff: datetime,
) -> ArchiveCapture:
    """Select the newest capture inclusively at or before a cutoff."""

    cutoff_utc = _aware_utc(cutoff, "archive cutoff")
    timestamps = [capture.cdx_timestamp for capture in captures]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("archive captures contain duplicate timestamps")
    eligible = [capture for capture in captures if capture.captured_at <= cutoff_utc]
    if not eligible:
        raise ArchiveNotFoundError("no_capture_before_cutoff")
    return max(eligible, key=lambda capture: capture.captured_at)


class WaybackClient:
    """Read-only exact-URL client for the Wayback CDX JSON index."""

    def __init__(
        self,
        *,
        settings: Settings,
        cdx_url: str = "https://web.archive.org/cdx/search/cdx",
        replay_origin: str = "https://web.archive.org",
        http_client: httpx.AsyncClient | None = None,
        sleep: AsyncSleep = asyncio.sleep,
    ) -> None:
        self._cdx_url = self._https_url(cdx_url, "CDX URL")
        self._replay_origin = self._https_url(replay_origin, "replay origin").rstrip("/")
        self._settings = settings
        self._sleep = sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "gaia-max-score-agent/0.1 (historical research)"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def captures(
        self,
        original_url: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 50,
    ) -> tuple[ArchiveCapture, ...]:
        """Return bounded successful exact-URL captures in an inclusive interval."""

        requested = _HTTP_URL.validate_python(original_url)
        if requested.scheme not in {"http", "https"}:
            raise ValueError("archive target must use HTTP or HTTPS")
        start_utc = _aware_utc(start, "archive start") if start is not None else None
        end_utc = _aware_utc(end, "archive end") if end is not None else None
        if start_utc is not None and end_utc is not None and start_utc > end_utc:
            raise ValueError("archive start cannot follow end")
        if not 1 <= limit <= 100:
            raise ValueError("archive capture limit must be between 1 and 100")

        params = {
            "url": str(requested),
            "matchType": "exact",
            "output": "json",
            "fl": ",".join(_CDX_FIELDS),
            "filter": "statuscode:200",
            "collapse": "digest",
            "limit": str(limit),
        }
        if start_utc is not None:
            params["from"] = _cdx_timestamp(start_utc)
        if end_utc is not None:
            params["to"] = _cdx_timestamp(end_utc)
        payload = await self._request(params)
        if not payload:
            return ()
        header = payload[0]
        if not isinstance(header, list | tuple):
            raise ArchiveError("invalid_cdx_header")
        header_values = cast(Sequence[object], header)
        if tuple(header_values) != _CDX_FIELDS:
            raise ArchiveError("invalid_cdx_header")
        if len(payload) - 1 > limit:
            raise ArchiveError("capture_limit_exceeded")

        captures: list[ArchiveCapture] = []
        for raw_row in payload[1:]:
            if not isinstance(raw_row, list | tuple):
                raise ArchiveError("invalid_cdx_row")
            raw_values = cast(Sequence[object], raw_row)
            if len(raw_values) != len(_CDX_FIELDS):
                raise ArchiveError("invalid_cdx_row")
            row = tuple(raw_values)
            if any(not isinstance(value, str) for value in row):
                raise ArchiveError("invalid_cdx_cell")
            timestamp, observed_url, status, media_type, digest = cast(tuple[str, ...], row)
            if status != "200":
                raise ArchiveError("non_success_capture")
            if not _same_archived_resource(str(requested), observed_url):
                raise ArchiveError("capture_url_mismatch")
            captured_at = _parse_cdx_timestamp(timestamp)
            if start_utc is not None and captured_at < start_utc:
                raise ArchiveError("capture_before_requested_start")
            if end_utc is not None and captured_at > end_utc:
                raise ArchiveError("capture_after_requested_end")
            captures.append(
                ArchiveCapture.model_validate(
                    {
                        "original_url": observed_url,
                        "captured_at": captured_at,
                        "cdx_timestamp": timestamp,
                        "media_type": media_type,
                        "digest": digest or None,
                        "replay_url": (
                            f"{self._replay_origin}/web/{timestamp}id_/{observed_url}"
                        ),
                    }
                )
            )
        if len({capture.cdx_timestamp for capture in captures}) != len(captures):
            raise ArchiveError("duplicate_capture_timestamp")
        return tuple(sorted(captures, key=lambda capture: capture.captured_at))

    async def latest_at_or_before(
        self,
        original_url: str,
        cutoff: datetime,
    ) -> ArchiveCapture:
        """Discover and select the newest capture at or before cutoff."""

        captures = await self.captures(original_url, end=cutoff)
        return select_latest_capture(captures, cutoff)

    async def _request(self, params: Mapping[str, str]) -> Sequence[object]:
        for attempt in range(1, self._settings.page_max_attempts + 1):
            try:
                response = await self._client.get(self._cdx_url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self._settings.page_max_attempts:
                    raise ArchiveError("network_exhausted") from exc
                await self._sleep(self._retry_delay(attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self._settings.page_max_attempts:
                    raise ArchiveError("transient_http_exhausted")
                await self._sleep(self._retry_delay(attempt))
                continue
            if not 200 <= response.status_code < 300:
                raise ArchiveError("permanent_http_error")
            if len(response.content) > self._settings.max_page_bytes:
                raise ArchiveError("response_too_large")
            try:
                decoded = response.json()
            except ValueError as exc:
                raise ArchiveError("invalid_json") from exc
            if not isinstance(decoded, list | tuple):
                raise ArchiveError("invalid_cdx_payload")
            return cast(Sequence[object], decoded)
        raise RuntimeError("unreachable archive retry state")

    def _retry_delay(self, failed_attempt: int) -> float:
        return min(
            self._settings.page_retry_base_seconds * (2 ** (failed_attempt - 1)),
            self._settings.page_retry_max_seconds,
        )

    @staticmethod
    def _https_url(value: str, label: str) -> str:
        parsed = _HTTP_URL.validate_python(value)
        if parsed.scheme != "https":
            raise ValueError(f"archive {label} must use HTTPS")
        return str(parsed)


class TemporalEvidenceKind(StrEnum):
    """How a source establishes the historical state of its claim."""

    REVISION = "revision"
    ARCHIVE_SNAPSHOT = "archive_snapshot"
    CONTEMPORANEOUS_PUBLICATION = "contemporaneous_publication"
    RETROSPECTIVE = "retrospective"
    CURRENT_UNQUALIFIED = "current_unqualified"


class TemporalEvidence(BaseModel):
    """Temporal facts separated from retrieval time and claim text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: StrictStr = Field(min_length=1, max_length=200)
    kind: TemporalEvidenceKind
    retrieved_at: AwareDatetime
    content_state_at: AwareDatetime | None = None
    published_at: AwareDatetime | None = None
    documented_period_start: AwareDatetime | None = None
    documented_period_end: AwareDatetime | None = None

    @field_validator(
        "retrieved_at",
        "content_state_at",
        "published_at",
        "documented_period_start",
        "documented_period_end",
    )
    @classmethod
    def normalize_times(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> TemporalEvidence:
        if (
            self.documented_period_start is not None
            and self.documented_period_end is not None
            and self.documented_period_start > self.documented_period_end
        ):
            raise ValueError("documented period start cannot follow end")
        if self.kind in {
            TemporalEvidenceKind.REVISION,
            TemporalEvidenceKind.ARCHIVE_SNAPSHOT,
        } and self.content_state_at is None:
            raise ValueError("revision/archive evidence requires content_state_at")
        if (
            self.kind is TemporalEvidenceKind.CONTEMPORANEOUS_PUBLICATION
            and self.published_at is None
        ):
            raise ValueError("contemporaneous evidence requires published_at")
        if self.kind is TemporalEvidenceKind.RETROSPECTIVE and (
            self.documented_period_start is None or self.documented_period_end is None
        ):
            raise ValueError("retrospective evidence requires an explicit documented period")
        return self


class TemporalValidityCode(StrEnum):
    """Stable, answer-free temporal-policy outcome."""

    REVISION_COMPATIBLE = "revision_compatible"
    ARCHIVE_COMPATIBLE = "archive_compatible"
    PUBLICATION_COMPATIBLE = "publication_compatible"
    RETROSPECTIVE_COVERS_PERIOD = "retrospective_covers_period"
    CURRENT_STATE_UNQUALIFIED = "current_state_unqualified"
    CONTENT_STATE_TOO_NEW = "content_state_too_new"
    CONTENT_STATE_OUTSIDE_PERIOD = "content_state_outside_period"
    PUBLICATION_OUTSIDE_PERIOD = "publication_outside_period"
    RETROSPECTIVE_PERIOD_INCOMPLETE = "retrospective_period_incomplete"
    CONSTRAINT_INCOMPLETE = "constraint_incomplete"


class TemporalValidityDecision(BaseModel):
    """One deterministic temporal accept/reject result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: StrictStr = Field(min_length=1)
    valid: bool
    code: TemporalValidityCode


def validate_temporal_evidence(
    evidence: TemporalEvidence,
    constraint: TemporalConstraint,
) -> TemporalValidityDecision:
    """Validate content-state time; retrieval time intentionally has no role."""

    start = _aware_utc(constraint.start, "constraint start") if constraint.start else None
    end = _aware_utc(constraint.end, "constraint end") if constraint.end else None
    if start is None and end is None:
        return _decision(evidence, False, TemporalValidityCode.CONSTRAINT_INCOMPLETE)
    period_start = start or end
    period_end = end or start
    if period_start is None or period_end is None:
        return _decision(evidence, False, TemporalValidityCode.CONSTRAINT_INCOMPLETE)

    if evidence.kind is TemporalEvidenceKind.CURRENT_UNQUALIFIED:
        return _decision(evidence, False, TemporalValidityCode.CURRENT_STATE_UNQUALIFIED)
    if evidence.kind in {
        TemporalEvidenceKind.REVISION,
        TemporalEvidenceKind.ARCHIVE_SNAPSHOT,
    }:
        state_at = evidence.content_state_at
        if state_at is None:
            return _decision(evidence, False, TemporalValidityCode.CONSTRAINT_INCOMPLETE)
        if constraint.kind in {"revision_cutoff", "as_of"}:
            valid = state_at <= period_end
            good_code = (
                TemporalValidityCode.REVISION_COMPATIBLE
                if evidence.kind is TemporalEvidenceKind.REVISION
                else TemporalValidityCode.ARCHIVE_COMPATIBLE
            )
            return _decision(
                evidence,
                valid,
                good_code if valid else TemporalValidityCode.CONTENT_STATE_TOO_NEW,
            )
        valid = period_start <= state_at <= period_end
        good_code = (
            TemporalValidityCode.REVISION_COMPATIBLE
            if evidence.kind is TemporalEvidenceKind.REVISION
            else TemporalValidityCode.ARCHIVE_COMPATIBLE
        )
        return _decision(
            evidence,
            valid,
            good_code if valid else TemporalValidityCode.CONTENT_STATE_OUTSIDE_PERIOD,
        )
    if evidence.kind is TemporalEvidenceKind.CONTEMPORANEOUS_PUBLICATION:
        published_at = evidence.published_at
        if published_at is None:
            return _decision(evidence, False, TemporalValidityCode.CONSTRAINT_INCOMPLETE)
        if constraint.kind in {"revision_cutoff", "as_of"}:
            valid = published_at <= period_end
        else:
            valid = period_start <= published_at <= period_end
        return _decision(
            evidence,
            valid,
            (
                TemporalValidityCode.PUBLICATION_COMPATIBLE
                if valid
                else TemporalValidityCode.PUBLICATION_OUTSIDE_PERIOD
            ),
        )

    documented_start = evidence.documented_period_start
    documented_end = evidence.documented_period_end
    if documented_start is None or documented_end is None:
        return _decision(evidence, False, TemporalValidityCode.CONSTRAINT_INCOMPLETE)
    covers = documented_start <= period_start and documented_end >= period_end
    return _decision(
        evidence,
        covers,
        (
            TemporalValidityCode.RETROSPECTIVE_COVERS_PERIOD
            if covers
            else TemporalValidityCode.RETROSPECTIVE_PERIOD_INCOMPLETE
        ),
    )


def _decision(
    evidence: TemporalEvidence,
    valid: bool,
    code: TemporalValidityCode,
) -> TemporalValidityDecision:
    return TemporalValidityDecision(
        evidence_id=evidence.evidence_id,
        valid=valid,
        code=code,
    )


def _aware_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _cdx_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def _parse_cdx_timestamp(value: str) -> datetime:
    if _CDX_TIMESTAMP.fullmatch(value) is None:
        raise ArchiveError("invalid_capture_timestamp")
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ArchiveError("invalid_capture_timestamp") from exc


def _same_archived_resource(requested: str, observed: str) -> bool:
    try:
        left = urlsplit(requested)
        right = urlsplit(observed)
        left_port = left.port
        right_port = right.port
    except ValueError:
        return False
    if right.scheme not in {"http", "https"} or not right.hostname:
        return False
    return (
        left.hostname.casefold() if left.hostname else None,
        left_port,
        left.path or "/",
        left.query,
    ) == (
        right.hostname.casefold(),
        right_port,
        right.path or "/",
        right.query,
    )


__all__ = [
    "ArchiveCapture",
    "ArchiveError",
    "ArchiveNotFoundError",
    "TemporalEvidence",
    "TemporalEvidenceKind",
    "TemporalValidityCode",
    "TemporalValidityDecision",
    "WaybackClient",
    "select_latest_capture",
    "validate_temporal_evidence",
]
