from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from gaia_max.config import Settings
from gaia_max.domain import TemporalConstraint
from gaia_max.retrieval import (
    ArchiveError,
    ArchiveNotFoundError,
    TemporalEvidence,
    TemporalEvidenceKind,
    TemporalValidityCode,
    WaybackClient,
    validate_temporal_evidence,
)

NOW = datetime(2026, 8, 30, 14, 0, tzinfo=UTC)
START = datetime(2016, 11, 1, tzinfo=UTC)
END = datetime(2016, 11, 30, 23, 59, 59, tzinfo=UTC)


def settings(**updates: object) -> Settings:
    return Settings.model_validate(
        {
            "page_max_attempts": 2,
            "page_retry_base_seconds": 0,
            "page_retry_max_seconds": 0,
            "max_page_bytes": 100_000,
            **updates,
        }
    )


def constraint(kind: str = "event_period") -> TemporalConstraint:
    return TemporalConstraint.model_validate(
        {"kind": kind, "start": START, "end": END, "timezone": "UTC"}
    )


def evidence(kind: TemporalEvidenceKind, **updates: object) -> TemporalEvidence:
    return TemporalEvidence.model_validate(
        {
            "evidence_id": "synthetic-evidence",
            "kind": kind,
            "retrieved_at": NOW,
            **updates,
        }
    )


@pytest.mark.asyncio
async def test_wayback_query_is_bounded_inclusive_and_selects_latest_capture() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["matchType"] == "exact"
        assert request.url.params["filter"] == "statuscode:200"
        assert request.url.params["to"] == "20161130235959"
        return httpx.Response(
            200,
            request=request,
            json=[
                ["timestamp", "original", "statuscode", "mimetype", "digest"],
                [
                    "20161101000000",
                    "http://example.org/roster",
                    "200",
                    "text/html",
                    "DIGEST1",
                ],
                [
                    "20161130235959",
                    "https://example.org/roster",
                    "200",
                    "text/html",
                    "DIGEST2",
                ],
            ],
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = WaybackClient(settings=settings(), http_client=http_client)

    capture = await client.latest_at_or_before("https://example.org/roster", END)

    assert capture.captured_at == END
    assert capture.cdx_timestamp == "20161130235959"
    assert str(capture.replay_url) == (
        "https://web.archive.org/web/20161130235959id_/https://example.org/roster"
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_wayback_rejects_mismatched_resource_and_malformed_header() -> None:
    payloads = [
        [
            ["timestamp", "original", "statuscode", "mimetype", "digest"],
            ["20161101000000", "https://evil.example/", "200", "text/html", "D"],
        ],
        [["timestamp", "original"]],
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=payloads.pop(0))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = WaybackClient(settings=settings(), http_client=http_client)

    with pytest.raises(ArchiveError, match="capture_url_mismatch"):
        await client.captures("https://example.org/roster", end=END)
    with pytest.raises(ArchiveError, match="invalid_cdx_header"):
        await client.captures("https://example.org/roster", end=END)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_wayback_empty_result_and_transient_retry_are_explicit() -> None:
    attempts = 0
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            json=[["timestamp", "original", "statuscode", "mimetype", "digest"]],
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = WaybackClient(
        settings=settings(),
        http_client=http_client,
        sleep=sleep,
    )

    with pytest.raises(ArchiveNotFoundError, match="no_capture_before_cutoff"):
        await client.latest_at_or_before("https://example.org/roster", END)
    assert attempts == 2
    assert waits == [0]
    await http_client.aclose()


def test_revision_cutoff_is_inclusive_and_rejects_newer_content_state() -> None:
    cutoff = TemporalConstraint.model_validate(
        {"kind": "revision_cutoff", "end": END, "timezone": "UTC"}
    )
    exact = evidence(TemporalEvidenceKind.REVISION, content_state_at=END)
    newer = evidence(
        TemporalEvidenceKind.REVISION,
        content_state_at=datetime(2016, 12, 1, tzinfo=UTC),
    )

    accepted = validate_temporal_evidence(exact, cutoff)
    rejected = validate_temporal_evidence(newer, cutoff)

    assert accepted.valid is True
    assert accepted.code is TemporalValidityCode.REVISION_COMPATIBLE
    assert rejected.valid is False
    assert rejected.code is TemporalValidityCode.CONTENT_STATE_TOO_NEW


def test_archive_and_publication_must_fall_inside_event_period() -> None:
    archive = evidence(
        TemporalEvidenceKind.ARCHIVE_SNAPSHOT,
        content_state_at=datetime(2016, 11, 15, tzinfo=UTC),
    )
    late_publication = evidence(
        TemporalEvidenceKind.CONTEMPORANEOUS_PUBLICATION,
        published_at=datetime(2017, 1, 1, tzinfo=UTC),
    )

    archive_result = validate_temporal_evidence(archive, constraint())
    publication_result = validate_temporal_evidence(late_publication, constraint())

    assert archive_result.code is TemporalValidityCode.ARCHIVE_COMPATIBLE
    assert archive_result.valid is True
    assert publication_result.code is TemporalValidityCode.PUBLICATION_OUTSIDE_PERIOD
    assert publication_result.valid is False


def test_present_day_retrospective_source_passes_only_with_explicit_period_coverage() -> None:
    covering = evidence(
        TemporalEvidenceKind.RETROSPECTIVE,
        published_at=NOW,
        documented_period_start=datetime(2016, 1, 1, tzinfo=UTC),
        documented_period_end=datetime(2016, 12, 31, tzinfo=UTC),
    )
    partial = evidence(
        TemporalEvidenceKind.RETROSPECTIVE,
        published_at=NOW,
        documented_period_start=datetime(2016, 11, 10, tzinfo=UTC),
        documented_period_end=datetime(2016, 11, 20, tzinfo=UTC),
    )
    current = evidence(TemporalEvidenceKind.CURRENT_UNQUALIFIED)

    accepted = validate_temporal_evidence(covering, constraint())
    rejected = validate_temporal_evidence(partial, constraint())
    unqualified = validate_temporal_evidence(current, constraint())

    assert accepted.code is TemporalValidityCode.RETROSPECTIVE_COVERS_PERIOD
    assert accepted.valid is True
    assert rejected.code is TemporalValidityCode.RETROSPECTIVE_PERIOD_INCOMPLETE
    assert rejected.valid is False
    assert unqualified.code is TemporalValidityCode.CURRENT_STATE_UNQUALIFIED
    assert unqualified.valid is False


def test_retrieval_time_never_changes_temporal_validity() -> None:
    values = [
        evidence(
            TemporalEvidenceKind.ARCHIVE_SNAPSHOT,
            retrieved_at=datetime(2017, 1, 1, tzinfo=UTC),
            content_state_at=datetime(2016, 11, 15, tzinfo=UTC),
        ),
        evidence(
            TemporalEvidenceKind.ARCHIVE_SNAPSHOT,
            retrieved_at=NOW,
            content_state_at=datetime(2016, 11, 15, tzinfo=UTC),
        ),
    ]

    decisions = [validate_temporal_evidence(item, constraint()) for item in values]

    assert [(item.valid, item.code) for item in decisions] == [
        (True, TemporalValidityCode.ARCHIVE_COMPATIBLE),
        (True, TemporalValidityCode.ARCHIVE_COMPATIBLE),
    ]


def test_temporal_models_reject_missing_proof_fields_and_naive_constraints() -> None:
    with pytest.raises(ValidationError, match="content_state_at"):
        evidence(TemporalEvidenceKind.ARCHIVE_SNAPSHOT)
    naive = TemporalConstraint.model_validate(
        {"kind": "as_of", "end": datetime(2016, 11, 30), "timezone": "UTC"}
    )
    current = evidence(TemporalEvidenceKind.CURRENT_UNQUALIFIED)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_temporal_evidence(current, naive)
