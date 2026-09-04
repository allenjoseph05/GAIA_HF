from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from gaia_max.config import Settings
from gaia_max.domain import IntegerAnswer, Question, TaskResult, TaskStatus
from gaia_max.preflight import CandidatePreflight, FrozenCandidateManifest
from gaia_max.snapshot import build_question_snapshot
from gaia_max.submission import APPROVAL_PHRASE, GaiaSubmissionClient, SubmissionApproval
from gaia_max.submission_graph import (
    SubmissionWorkflowInput,
    SubmissionWorkflowStatus,
    build_submission_graph,
)

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false


def frozen_manifest() -> FrozenCandidateManifest:
    questions = [
        Question(task_id=f"task-{index}", question=f"Question {index}")
        for index in range(2)
    ]
    snapshot = build_question_snapshot(
        questions,
        source_url="https://course.test/questions",
        task_profiles_version="graph-v1",
        retrieved_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    report = CandidatePreflight().validate(
        run_id="submission-graph-run",
        snapshot=snapshot,
        results=[
            TaskResult(
                task_id=question.task_id,
                status=TaskStatus.READY,
                semantic_answer=IntegerAnswer(value=index),
                serialized_answer=f"private-{index}",
            )
            for index, question in enumerate(questions)
        ],
        snapshot_approved=True,
    )
    assert report.manifest is not None
    return report.manifest


@pytest.mark.asyncio
async def test_graph_interrupt_is_redacted_and_resume_submits_frozen_payload_once() -> None:
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

    candidate = frozen_manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=2),
            http_client=http_client,
        )
        graph = build_submission_graph(client, checkpointer=InMemorySaver())
        config: RunnableConfig = {
            "configurable": {"thread_id": "approved-submission"}
        }
        first = await graph.ainvoke(
            SubmissionWorkflowInput(
                manifest=candidate,
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent/tree/main",
            ).initial_state(),
            config=config,
        )

        raw_first = cast(dict[str, Any], first)
        interrupt_payload = raw_first["__interrupt__"][0].value
        encoded_interrupt = json.dumps(interrupt_payload)
        assert candidate.candidate_sha256 in encoded_interrupt
        assert "private-0" not in encoded_interrupt
        assert "private-1" not in encoded_interrupt
        assert calls == []

        approval = SubmissionApproval(
            candidate_sha256=candidate.candidate_sha256,
            operator="learner",
            approved_at=datetime(2026, 8, 30, tzinfo=UTC),
            phrase=APPROVAL_PHRASE,
        )
        resumed = await graph.ainvoke(
            Command(resume=approval.model_dump(mode="json")),
            config=config,
        )

    assert resumed["status"] is SubmissionWorkflowStatus.SUBMITTED
    assert resumed["receipt"] is not None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_incomplete_candidate_blocks_without_interrupt_or_http() -> None:
    candidate = frozen_manifest()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        base_url="https://course.test",
    )
    async with http_client:
        client = GaiaSubmissionClient(
            Settings(dry_run=False, allow_submit=True, expected_question_count=20),
            http_client=http_client,
        )
        graph = build_submission_graph(client, checkpointer=InMemorySaver())
        blocked_config: RunnableConfig = {
            "configurable": {"thread_id": "blocked-submission"}
        }
        result = await graph.ainvoke(
            SubmissionWorkflowInput(
                manifest=candidate,
                username="learner",
                agent_code="https://huggingface.co/spaces/learner/agent/tree/main",
            ).initial_state(),
            config=blocked_config,
        )

    assert result["status"] is SubmissionWorkflowStatus.BLOCKED
    assert result["error_code"] == "candidate_preflight_blocked"
