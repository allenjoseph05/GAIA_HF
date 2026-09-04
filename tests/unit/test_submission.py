from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError

from gaia_max.config import Settings
from gaia_max.domain import IntegerAnswer, Question, TaskResult, TaskStatus
from gaia_max.preflight import CandidatePreflight, FrozenCandidateManifest
from gaia_max.snapshot import build_question_snapshot
from gaia_max.submission import (
    APPROVAL_PHRASE,
    GaiaSubmissionClient,
    SubmissionApproval,
    SubmissionBlockedError,
    SubmissionTransportError,
)


def manifest(count: int = 2) -> FrozenCandidateManifest:
    questions = [
        Question(task_id=f"task-{index}", question=f"Question {index}")
        for index in range(count)
    ]
    snapshot = build_question_snapshot(
        questions,
        source_url="https://course.test/questions",
        task_profiles_version="submission-v1",
        retrieved_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    results = [
        TaskResult(
            task_id=question.task_id,
            status=TaskStatus.READY,
            semantic_answer=IntegerAnswer(value=index),
            serialized_answer=str(index),
        )
        for index, question in enumerate(questions)
    ]
    report = CandidatePreflight().validate(
        run_id="frozen-run",
        snapshot=snapshot,
        results=results,
        snapshot_approved=True,
    )
    assert report.manifest is not None
    return report.manifest


def approval(candidate: FrozenCandidateManifest) -> SubmissionApproval:
    return SubmissionApproval(
        candidate_sha256=candidate.candidate_sha256,
        operator="learner",
        approved_at=datetime(2026, 8, 30, tzinfo=UTC),
        phrase=APPROVAL_PHRASE,
    )


@pytest.mark.asyncio
async def test_default_settings_block_before_network_access() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    candidate = manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(Settings(expected_question_count=2), http_client=http_client)
        with pytest.raises(SubmissionBlockedError, match="disabled"):
            await client.submit_once(
                manifest=candidate,
                approval=approval(candidate),
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent",
            )
    assert calls == 0


@pytest.mark.asyncio
async def test_mismatched_approval_hash_blocks_before_network_access() -> None:
    candidate = manifest()
    wrong = approval(candidate).model_copy(update={"candidate_sha256": "f" * 64})
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        with pytest.raises(SubmissionBlockedError, match="not bound"):
            await client.submit_once(
                manifest=candidate,
                approval=wrong,
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent",
            )


@pytest.mark.asyncio
async def test_approval_operator_must_match_submission_username() -> None:
    candidate = manifest()
    wrong_operator = approval(candidate).model_copy(update={"operator": "someone-else"})
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        with pytest.raises(SubmissionBlockedError, match="operator"):
            await client.submit_once(
                manifest=candidate,
                approval=wrong_operator,
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent",
            )


@pytest.mark.asyncio
async def test_agent_code_must_be_owned_hugging_face_space() -> None:
    candidate = manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        with pytest.raises(SubmissionBlockedError, match="Space URL"):
            await client.submit_once(
                manifest=candidate,
                approval=approval(candidate),
                username="learner",
                agent_code="https://example.com/not-a-space",
            )


def test_approval_requires_exact_phrase_and_timezone() -> None:
    candidate = manifest()
    with pytest.raises(ValidationError, match="phrase"):
        SubmissionApproval(
            candidate_sha256=candidate.candidate_sha256,
            operator="learner",
            approved_at=datetime(2026, 8, 30, tzinfo=UTC),
            phrase="submit",
        )
    with pytest.raises(ValidationError, match="timezone"):
        SubmissionApproval(
            candidate_sha256=candidate.candidate_sha256,
            operator="learner",
            approved_at=datetime(2026, 8, 30),
            phrase=APPROVAL_PHRASE,
        )


@pytest.mark.asyncio
async def test_incomplete_manifest_is_blocked() -> None:
    candidate = manifest()
    client = GaiaSubmissionClient(
        Settings(dry_run=False, allow_submit=True, expected_question_count=20)
    )
    try:
        with pytest.raises(SubmissionBlockedError, match="expected task count"):
            await client.submit_once(
                manifest=candidate,
                approval=approval(candidate),
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent",
            )
    finally:
        await client.__aexit__()


@pytest.mark.asyncio
async def test_valid_candidate_posts_once_and_parses_aggregate_receipt() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "username": "learner",
                "score": 100.0,
                "correct_count": 2,
                "total_attempted": 2,
                "message": "accepted",
                "timestamp": "2026-08-30T12:00:00Z",
            },
        )

    candidate = manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        receipt = await client.submit_once(
            manifest=candidate,
            approval=approval(candidate),
            username="learner",
            agent_code="https://huggingface.co/spaces/learner/agent",
        )

    assert len(calls) == 1
    assert calls[0].url.path == "/submit"
    payload = json.loads(calls[0].content)
    assert payload["answers"] == [
        {"task_id": "task-0", "submitted_answer": "0"},
        {"task_id": "task-1", "submitted_answer": "1"},
    ]
    assert receipt.correct_count == 2
    assert receipt.total_attempted == 2


@pytest.mark.asyncio
async def test_transport_error_is_redacted_and_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private-answer-was-in-request")

    candidate = manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        with pytest.raises(SubmissionTransportError) as caught:
            await client.submit_once(
                manifest=candidate,
                approval=approval(candidate),
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent",
            )

    assert calls == 1
    assert "private-answer" not in str(caught.value)
