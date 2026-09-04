from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gaia_max.domain.models import (
    CandidateAnswer,
    DecimalAnswer,
    Evidence,
    EvidenceStatus,
    IntegerAnswer,
    Question,
    QuestionSnapshot,
    RiskAssessment,
    RiskLevel,
    TaskResult,
    TaskStatus,
)

NOW = datetime(2026, 8, 27, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def test_question_normalizes_empty_attachment_name() -> None:
    question = Question(task_id="task-1", question="A question", file_name="")

    assert question.file_name is None


def test_snapshot_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        QuestionSnapshot(
            retrieved_at=NOW,
            source_url="https://example.test/questions",
            count=2,
            task_ids=["task-1", "task-1"],
            task_hashes={"task-1": HASH_A},
            snapshot_sha256=HASH_B,
            task_profiles_version="test-v1",
        )


def test_snapshot_requires_hash_for_every_task() -> None:
    with pytest.raises(ValidationError, match="exactly the snapshot task IDs"):
        QuestionSnapshot(
            retrieved_at=NOW,
            source_url="https://example.test/questions",
            count=2,
            task_ids=["task-1", "task-2"],
            task_hashes={"task-1": HASH_A},
            snapshot_sha256=HASH_B,
            task_profiles_version="test-v1",
        )


def test_evidence_requires_source_or_artifact() -> None:
    with pytest.raises(ValidationError, match="source URL or artifact"):
        Evidence(
            evidence_id="evidence-1",
            claim="A supported claim",
            source_type="primary_paper",
            retrieved_at=NOW,
            is_primary_source=True,
            extraction_method="pdf_text",
            status=EvidenceStatus.PRIMARY,
        )


def test_decimal_candidate_round_trips_without_float_conversion() -> None:
    candidate = CandidateAnswer(
        candidate_id="candidate-1",
        semantic_value=DecimalAnswer(value=Decimal("1234.50")),
        solver="spreadsheet",
        deterministic_checks=["openpyxl_equals_pandas"],
        created_at=NOW,
    )

    restored = CandidateAnswer.model_validate_json(candidate.model_dump_json())

    assert restored.semantic_value.value == Decimal("1234.50")


def test_ready_task_requires_semantic_and_serialized_answers() -> None:
    with pytest.raises(ValidationError, match="requires semantic and serialized"):
        TaskResult(task_id="task-1", status=TaskStatus.READY)


def test_ready_task_rejects_hard_failures() -> None:
    with pytest.raises(ValidationError, match="cannot contain hard risk failures"):
        TaskResult(
            task_id="task-1",
            status=TaskStatus.READY,
            semantic_answer=IntegerAnswer(value=3),
            serialized_answer="3",
            risk=RiskAssessment(level=RiskLevel.BLOCKED, hard_failures=["missing evidence"]),
        )


def test_blocked_task_requires_reason() -> None:
    with pytest.raises(ValidationError, match="requires at least one error"):
        TaskResult(task_id="task-1", status=TaskStatus.BLOCKED)
