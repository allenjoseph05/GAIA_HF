from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaia_max.domain import (
    IntegerAnswer,
    Question,
    QuestionSnapshot,
    TaskResult,
    TaskStatus,
)
from gaia_max.preflight import (
    CandidatePreflight,
    FrozenCandidateManifest,
    PreflightIssueCode,
    PreflightReport,
    build_dry_run_report,
    preflight_run,
)
from gaia_max.run_graph import (
    GaiaRunResult,
    RunStatus,
    TaskDispatchAction,
    TaskDispatchRecord,
)
from gaia_max.snapshot import build_question_snapshot

NOW = datetime(2026, 8, 28, tzinfo=UTC)


def snapshot(count: int = 20) -> QuestionSnapshot:
    questions = [
        Question(task_id=f"task-{index:02d}", question=f"Question {index}")
        for index in range(count)
    ]
    return build_question_snapshot(
        questions,
        source_url="https://course.test/questions",
        task_profiles_version="preflight-v1",
        retrieved_at=NOW,
    )


def ready(task_id: str, answer: str = "1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=TaskStatus.READY,
        semantic_answer=IntegerAnswer(value=1),
        serialized_answer=answer,
    )


def complete_run(
    count: int = 3,
    *,
    answer: str = "1",
) -> tuple[GaiaRunResult, QuestionSnapshot]:
    current_snapshot = snapshot(count)
    results = {
        task_id: ready(task_id, answer=answer) for task_id in current_snapshot.task_ids
    }
    records = {
        task_id: TaskDispatchRecord(
            task_id=task_id,
            action=TaskDispatchAction.START,
            terminal_status=TaskStatus.READY,
        )
        for task_id in current_snapshot.task_ids
    }
    return (
        GaiaRunResult(
            run_id="preflight-run",
            status=RunStatus.COMPLETE,
            selected_task_ids=tuple(current_snapshot.task_ids),
            task_results=results,
            dispatch_records=records,
        ),
        current_snapshot,
    )


def issue_codes(report: PreflightReport) -> set[PreflightIssueCode]:
    return {issue.code for issue in report.issues}


def test_complete_twenty_task_candidate_freezes_in_snapshot_order() -> None:
    current_snapshot = snapshot()
    results = [ready(task_id) for task_id in reversed(current_snapshot.task_ids)]

    report = CandidatePreflight().validate(
        run_id="full-run",
        snapshot=current_snapshot,
        results=results,
        snapshot_approved=True,
    )

    assert report.valid is True
    assert report.issues == ()
    assert report.manifest is not None
    assert [entry.task_id for entry in report.manifest.answers] == current_snapshot.task_ids
    assert len(report.manifest.candidate_sha256) == 64


def test_manifest_hash_detects_tampering() -> None:
    current_snapshot = snapshot(2)
    report = CandidatePreflight().validate(
        run_id="tamper-run",
        snapshot=current_snapshot,
        results=[ready(task_id) for task_id in current_snapshot.task_ids],
        snapshot_approved=True,
    )
    assert report.manifest is not None
    payload = report.manifest.model_dump()
    payload["run_id"] = "changed-run"

    with pytest.raises(ValidationError, match="hash does not match"):
        FrozenCandidateManifest.model_validate(payload)


def test_incomplete_results_fail_with_count_and_missing_codes() -> None:
    current_snapshot = snapshot(3)

    report = CandidatePreflight().validate(
        run_id="incomplete-run",
        snapshot=current_snapshot,
        results=[ready(current_snapshot.task_ids[0])],
        snapshot_approved=True,
    )

    assert report.valid is False
    assert issue_codes(report) == {
        PreflightIssueCode.RESULT_COUNT_MISMATCH,
        PreflightIssueCode.MISSING_TASK_ID,
    }
    assert report.manifest is None


def test_duplicate_task_id_fails_preflight() -> None:
    current_snapshot = snapshot(2)
    duplicate = ready(current_snapshot.task_ids[0])

    report = CandidatePreflight().validate(
        run_id="duplicate-run",
        snapshot=current_snapshot,
        results=[duplicate, duplicate, ready(current_snapshot.task_ids[1])],
        snapshot_approved=True,
    )

    assert PreflightIssueCode.DUPLICATE_TASK_ID in issue_codes(report)
    assert PreflightIssueCode.RESULT_COUNT_MISMATCH in issue_codes(report)


def test_unknown_id_also_reports_the_expected_missing_id() -> None:
    current_snapshot = snapshot(2)

    report = CandidatePreflight().validate(
        run_id="unknown-run",
        snapshot=current_snapshot,
        results=[ready(current_snapshot.task_ids[0]), ready("unknown-task")],
        snapshot_approved=True,
    )

    assert issue_codes(report) == {
        PreflightIssueCode.UNKNOWN_TASK_ID,
        PreflightIssueCode.MISSING_TASK_ID,
    }


def test_empty_or_whitespace_answer_fails_preflight() -> None:
    current_snapshot = snapshot(1)

    report = CandidatePreflight().validate(
        run_id="empty-run",
        snapshot=current_snapshot,
        results=[ready(current_snapshot.task_ids[0], answer="   ")],
        snapshot_approved=True,
    )

    assert issue_codes(report) == {PreflightIssueCode.EMPTY_ANSWER}


def test_blocked_task_and_unapproved_snapshot_fail_preflight() -> None:
    current_snapshot = snapshot(1)
    blocked = TaskResult(
        task_id=current_snapshot.task_ids[0],
        status=TaskStatus.BLOCKED,
        errors=["synthetic:block"],
    )

    report = CandidatePreflight().validate(
        run_id="blocked-run",
        snapshot=current_snapshot,
        results=[blocked],
        snapshot_approved=False,
    )

    assert issue_codes(report) == {
        PreflightIssueCode.SNAPSHOT_NOT_APPROVED,
        PreflightIssueCode.TASK_NOT_READY,
        PreflightIssueCode.EMPTY_ANSWER,
    }


def test_incomplete_run_status_is_an_independent_failure() -> None:
    current_snapshot = snapshot(1)

    report = CandidatePreflight().validate(
        run_id="unfinished-run",
        snapshot=current_snapshot,
        results=[ready(current_snapshot.task_ids[0])],
        snapshot_approved=True,
        run_complete=False,
    )

    assert issue_codes(report) == {PreflightIssueCode.RUN_NOT_COMPLETE}


def test_dry_run_report_contains_status_but_never_candidate_strings() -> None:
    run, current_snapshot = complete_run(answer="private-candidate-value")

    report = build_dry_run_report(
        run,
        current_snapshot,
        snapshot_approved=True,
    )
    encoded = report.model_dump_json()

    assert report.ready_count == 3
    assert report.blocked_count == 0
    assert report.preflight_valid is True
    assert report.submission_capability_present is False
    assert "private-candidate-value" not in encoded
    assert '"serialized_answer":' not in encoded.casefold()


def test_partial_run_result_fails_full_snapshot_preflight() -> None:
    run, current_snapshot = complete_run()
    selected_id = current_snapshot.task_ids[0]
    partial = run.model_copy(
        update={
            "selected_task_ids": (selected_id,),
            "task_results": {selected_id: run.task_results[selected_id]},
            "dispatch_records": {selected_id: run.dispatch_records[selected_id]},
        }
    )

    report = preflight_run(partial, current_snapshot, snapshot_approved=True)

    assert PreflightIssueCode.RESULT_COUNT_MISMATCH in issue_codes(report)
    assert PreflightIssueCode.MISSING_TASK_ID in issue_codes(report)
