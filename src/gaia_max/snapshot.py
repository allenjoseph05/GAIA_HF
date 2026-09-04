"""Canonical hashing and immutable persistence for public question snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl

from gaia_max.domain import Question, QuestionSnapshot


class StoredQuestionSnapshot(BaseModel):
    """The snapshot metadata and public questions written to local runtime storage."""

    snapshot: QuestionSnapshot
    questions: list[Question]


def _canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically across input ordering and whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_question(question: Question) -> dict[str, str | None]:
    """Return the public fields that define a question's identity."""

    return {
        "file_name": question.file_name,
        "level": question.level,
        "question": question.question,
        "task_id": question.task_id,
    }


def question_sha256(question: Question) -> str:
    """Hash one normalized public question record."""

    return hashlib.sha256(_canonical_json(canonical_question(question))).hexdigest()


def build_question_snapshot(
    questions: Sequence[Question],
    *,
    source_url: str,
    task_profiles_version: str,
    retrieved_at: datetime | None = None,
) -> QuestionSnapshot:
    """Build an order-independent snapshot of the complete public question set."""

    if not questions:
        raise ValueError("cannot build a snapshot from an empty question set")

    task_ids = [question.task_id for question in questions]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("cannot snapshot duplicate task IDs")

    ordered_questions = sorted(questions, key=lambda question: question.task_id)
    canonical_records = [canonical_question(question) for question in ordered_questions]
    task_hashes = {
        question.task_id: question_sha256(question) for question in ordered_questions
    }
    snapshot_sha256 = hashlib.sha256(_canonical_json(canonical_records)).hexdigest()

    return QuestionSnapshot(
        retrieved_at=retrieved_at or datetime.now(UTC),
        source_url=HttpUrl(source_url),
        count=len(ordered_questions),
        task_ids=[question.task_id for question in ordered_questions],
        task_hashes=task_hashes,
        snapshot_sha256=snapshot_sha256,
        attachment_task_ids=[
            question.task_id for question in ordered_questions if question.file_name is not None
        ],
        task_profiles_version=task_profiles_version,
    )


def persist_question_snapshot(
    snapshot: QuestionSnapshot,
    questions: Sequence[Question],
    *,
    snapshots_dir: Path,
) -> Path:
    """Atomically persist an immutable snapshot envelope and return its path."""

    by_id = {question.task_id: question for question in questions}
    if set(by_id) != set(snapshot.task_ids) or len(by_id) != len(questions):
        raise ValueError("questions do not match the snapshot task inventory")

    ordered_questions = [by_id[task_id] for task_id in snapshot.task_ids]
    rebuilt = build_question_snapshot(
        ordered_questions,
        source_url=str(snapshot.source_url),
        task_profiles_version=snapshot.task_profiles_version,
        retrieved_at=snapshot.retrieved_at,
    )
    if rebuilt.snapshot_sha256 != snapshot.snapshot_sha256:
        raise ValueError("questions do not match the snapshot content hash")

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    target = snapshots_dir / f"questions-{snapshot.snapshot_sha256}.json"
    if target.exists():
        stored = StoredQuestionSnapshot.model_validate_json(target.read_text(encoding="utf-8"))
        if stored.snapshot.snapshot_sha256 != snapshot.snapshot_sha256:
            raise ValueError("existing snapshot file failed its content identity check")
        return target

    envelope = StoredQuestionSnapshot(snapshot=snapshot, questions=ordered_questions)
    payload = envelope.model_dump_json(indent=2).encode("utf-8")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=snapshots_dir,
            prefix=".questions-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return target
