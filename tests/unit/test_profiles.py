from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaia_max.domain import OutputContract, Question, QuestionSnapshot
from gaia_max.domain.models import AnswerType, Modality, RiskLevel, TaskClass
from gaia_max.profiles import (
    ProfileRegistryError,
    TaskProfileRecord,
    TaskProfileRegistry,
    load_profile_registry,
    reconcile_profile_registry,
)
from gaia_max.snapshot import question_sha256

ROOT = Path(__file__).parents[2]
SHIPPED_REGISTRY = ROOT / "config" / "task_profiles.json"
NOW = datetime(2026, 8, 27, tzinfo=UTC)


def test_shipped_registry_is_valid_complete_and_answer_free() -> None:
    registry = load_profile_registry(SHIPPED_REGISTRY)
    raw = json.loads(SHIPPED_REGISTRY.read_text(encoding="utf-8"))

    assert registry.profile_count == 20
    assert len(registry.by_task_id()) == 20
    assert all("answer" not in profile for profile in raw["profiles"])
    assert all("final_answer" not in profile for profile in raw["profiles"])


def test_shipped_registry_matches_its_bound_snapshot_inventory() -> None:
    registry = load_profile_registry(SHIPPED_REGISTRY)
    task_hashes = {
        profile.task_id: profile.question_sha256 for profile in registry.profiles
    }
    snapshot = QuestionSnapshot(
        retrieved_at=NOW,
        source_url="https://course.test/questions",
        count=registry.profile_count,
        task_ids=sorted(task_hashes),
        task_hashes=task_hashes,
        snapshot_sha256=registry.question_snapshot_sha256,
        task_profiles_version=registry.profiles_version,
    )

    audit = reconcile_profile_registry(registry, snapshot)

    assert audit.submission_safe is True
    assert len(audit.matched_task_ids) == 20


def test_reconciliation_reports_stale_hash_without_hiding_other_matches() -> None:
    registry = load_profile_registry(SHIPPED_REGISTRY)
    task_hashes = {
        profile.task_id: profile.question_sha256 for profile in registry.profiles
    }
    stale_id = registry.profiles[0].task_id
    task_hashes[stale_id] = "c" * 64
    snapshot = QuestionSnapshot(
        retrieved_at=NOW,
        source_url="https://course.test/questions",
        count=registry.profile_count,
        task_ids=sorted(task_hashes),
        task_hashes=task_hashes,
        snapshot_sha256="d" * 64,
        task_profiles_version=registry.profiles_version,
    )

    audit = reconcile_profile_registry(registry, snapshot)

    assert audit.submission_safe is False
    assert audit.stale_profile_task_ids == [stale_id]
    assert len(audit.matched_task_ids) == 19


def test_materialize_combines_public_question_with_answer_free_metadata() -> None:
    question = Question(task_id="task-1", question="Reverse this text", level="1")
    record = TaskProfileRecord(
        task_id=question.task_id,
        question_sha256=question_sha256(question),
        modality=Modality.TEXT,
        task_class=TaskClass.DETERMINISTIC_TEXT,
        route="deterministic_text",
        requested_operation="reverse text",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="round_trip",
        risk_level=RiskLevel.LOW,
    )

    profile = record.materialize(question)

    assert profile.question == question.question
    assert profile.route == "deterministic_text"
    assert profile.risk_level is RiskLevel.LOW
    assert "answer" not in profile.model_dump()


def test_materialize_rejects_changed_question() -> None:
    original = Question(task_id="task-1", question="Original")
    record = TaskProfileRecord(
        task_id=original.task_id,
        question_sha256=question_sha256(original),
        modality=Modality.TEXT,
        task_class=TaskClass.DETERMINISTIC_TEXT,
        route="deterministic_text",
        requested_operation="transform text",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="round_trip",
        risk_level=RiskLevel.LOW,
    )

    with pytest.raises(ProfileRegistryError, match="hash does not match"):
        record.materialize(Question(task_id="task-1", question="Changed"))


def test_loader_rejects_forbidden_answer_key(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"final_answer": "forbidden"}), encoding="utf-8")

    with pytest.raises(ProfileRegistryError, match="forbidden answer-bearing key"):
        load_profile_registry(path)


def test_registry_rejects_duplicate_task_ids() -> None:
    question = Question(task_id="task-1", question="A question")
    record = TaskProfileRecord(
        task_id=question.task_id,
        question_sha256=question_sha256(question),
        modality=Modality.TEXT,
        task_class=TaskClass.DETERMINISTIC_TEXT,
        route="deterministic_text",
        requested_operation="transform",
        output_contract=OutputContract(answer_type=AnswerType.STRING),
        verification_policy="round_trip",
        risk_level=RiskLevel.LOW,
    )

    with pytest.raises(ValueError, match="must be unique"):
        TaskProfileRegistry(
            schema_version=1,
            profiles_version="test-v1",
            question_snapshot_sha256="a" * 64,
            profile_count=2,
            profiles=[record, record],
        )
