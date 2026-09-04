from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from gaia_max.domain import Question
from gaia_max.snapshot import (
    StoredQuestionSnapshot,
    build_question_snapshot,
    persist_question_snapshot,
    question_sha256,
)

NOW = datetime(2026, 8, 27, tzinfo=UTC)
SOURCE_URL = "https://course.test/questions"
PROFILE_VERSION = "test-v1"


def question(task_id: str, text: str, file_name: str | None = None) -> Question:
    return Question(task_id=task_id, question=text, level="1", file_name=file_name)


def build(questions: list[Question]):
    return build_question_snapshot(
        questions,
        source_url=SOURCE_URL,
        task_profiles_version=PROFILE_VERSION,
        retrieved_at=NOW,
    )


def test_snapshot_hash_is_independent_of_api_list_order() -> None:
    first = question("task-a", "First question")
    second = question("task-b", "Second question", "file.xlsx")

    forward = build([first, second])
    reversed_order = build([second, first])

    assert forward.snapshot_sha256 == reversed_order.snapshot_sha256
    assert forward.task_ids == ["task-a", "task-b"]
    assert forward.attachment_task_ids == ["task-b"]


def test_question_change_changes_question_and_snapshot_hashes() -> None:
    original = question("task-a", "Original wording")
    changed = question("task-a", "Changed wording")

    assert question_sha256(original) != question_sha256(changed)
    assert build([original]).snapshot_sha256 != build([changed]).snapshot_sha256


def test_empty_question_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty question set"):
        build([])


def test_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate task IDs"):
        build([question("same", "One"), question("same", "Two")])


def test_snapshot_is_persisted_with_questions_in_canonical_order(tmp_path) -> None:
    questions = [question("task-b", "Second"), question("task-a", "First")]
    snapshot = build(questions)

    path = persist_question_snapshot(snapshot, questions, snapshots_dir=tmp_path)
    stored = StoredQuestionSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    assert path.name == f"questions-{snapshot.snapshot_sha256}.json"
    assert [item.task_id for item in stored.questions] == ["task-a", "task-b"]
    assert stored.snapshot.snapshot_sha256 == snapshot.snapshot_sha256


def test_persist_is_idempotent_for_same_snapshot(tmp_path) -> None:
    questions = [question("task-a", "First")]
    snapshot = build(questions)

    first_path = persist_question_snapshot(snapshot, questions, snapshots_dir=tmp_path)
    first_payload = first_path.read_bytes()
    second_path = persist_question_snapshot(snapshot, questions, snapshots_dir=tmp_path)

    assert first_path == second_path
    assert second_path.read_bytes() == first_payload


def test_persist_rejects_question_inventory_mismatch(tmp_path) -> None:
    snapshot = build([question("task-a", "First")])

    with pytest.raises(ValueError, match="task inventory"):
        persist_question_snapshot(
            snapshot,
            [question("task-b", "Other")],
            snapshots_dir=tmp_path,
        )


def test_persisted_json_contains_no_answer_field(tmp_path) -> None:
    questions = [question("task-a", "First")]
    snapshot = build(questions)

    path = persist_question_snapshot(snapshot, questions, snapshots_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "answer" not in json.dumps(payload).lower()
